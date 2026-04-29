"""
MedicalAI — api/v1/endpoints/chat.py
对话相关端点：/chat、/chat/stream（SSE 流式思考进度）、/clear、/new-chat。
"""

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_service

router = APIRouter(tags=["对话"])


def _get_session_id(request: Request) -> str:
    """从 X-Session-ID 请求头或 cookie 会话中获取或创建 session ID。"""
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    return request.session["session_id"]


# ── 节点名 → 用户可见的中文进度提示（检索迭代不展示给用户）──────────────────
_NODE_LABELS = {
    "memory":        "📋 加载记忆档案…",
    "query_rewriter":"✍️ 理解并优化问题…",
    "planner":       "🧭 规划检索策略…",
    "research":      "🔍 搜索医学知识库…",
    "critic":        "🔬 核查答案准确性…",
}
# research 节点内部的子动作提示（ResearchAgent 运行时写入 state["_stream_hint"]）
_RESEARCH_HINTS = {
    "expand_query": "🔍 扩展查询词，重新检索…",
    "tavily":       "🌐 知识库未命中，联网搜索…",
    "tool_query":   "🛠️ 查询专业工具数据…",
    "llm_direct":   "💡 直接生成回答…",
    "accept":       "✅ 文档质量达标，生成答案…",
}


def _make_sse(event: str, data: dict) -> str:
    """格式化为 SSE 数据帧。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat", response_model=ChatResponse, summary="发送消息并获取 AI 回答")
async def chat_endpoint(request: ChatRequest, req: Request):
    """通过多智能体工作流处理用户消息，返回 AI 回答。"""
    if not chat_service.workflow_app:
        raise HTTPException(status_code=503, detail="系统尚未初始化，请稍后重试")
    session_id = _get_session_id(req)
    return await chat_service.process_message(session_id, request.message)


@router.post("/chat/stream", summary="流式返回 AI 思考进度（SSE）")
async def chat_stream_endpoint(request: ChatRequest, req: Request):
    """
    SSE 流式端点，在工作流执行过程中实时推送节点进度。

    事件类型
    --------
    progress  : 节点开始/完成时推送用户可见的进度文字
                {"step": "✍️ 理解并优化问题…", "node": "query_rewriter"}
    thinking  : QueryRewriter 完成后推送意图分析结果（用于前端"查询优化"面板）
                {"thinking_steps": [...], "query_intent": "...", "original_question": "..."}
    done      : 工作流完成，推送完整结果（同 /chat 的 response body）
    error     : 发生异常时推送错误信息

    前端使用方式
    ------------
    const source = new EventSource('/api/v1/chat/stream', {...});
    source.addEventListener('progress', e => showProgress(JSON.parse(e.data)));
    source.addEventListener('thinking', e => updateThinkingPanel(JSON.parse(e.data)));
    source.addEventListener('done',     e => { renderAnswer(JSON.parse(e.data)); source.close(); });
    source.addEventListener('error',    e => { showError(JSON.parse(e.data)); source.close(); });

    注意：检索迭代（rag_think_log）不在流式阶段推送，仅在 done 事件里携带，
    由前端在答案渲染后展示在"查看 AI 思考过程"折叠面板里。
    """
    if not chat_service.workflow_app:
        raise HTTPException(status_code=503, detail="系统尚未初始化，请稍后重试")

    session_id = _get_session_id(req)
    message = request.message

    async def event_generator():
        # ── 用 astream_events 监听节点级事件 ──────────────────────────────
        # LangGraph 1.0.x 的 astream_events 在每个节点 start/end 时发出事件
        try:
            from app.core.state import initialize_conversation_state, reset_query_state
            from app.services.database_service import db_service as _db

            state = chat_service._get_or_create_session_state(session_id)
            state = reset_query_state(state)
            state["question"] = message
            state["session_id"] = session_id

            _db.save_message(session_id, "user", message)

            seen_nodes = set()
            final_result = None

            async for event in chat_service.workflow_app.astream_events(
                state,
                version="v2",          # LangGraph 1.x 使用 v2 格式
                config={"recursion_limit": 25},
            ):
                kind      = event.get("event", "")
                node_name = event.get("name", "")

                # ── 节点开始：推送进度提示 ───────────────────────────────
                if kind == "on_chain_start" and node_name in _NODE_LABELS:
                    if node_name not in seen_nodes:
                        seen_nodes.add(node_name)
                        yield _make_sse("progress", {
                            "step": _NODE_LABELS[node_name],
                            "node": node_name,
                        })

                # ── 节点结束：提取关键中间结果 ───────────────────────────
                elif kind == "on_chain_end" and node_name in _NODE_LABELS:
                    output = event.get("data", {}).get("output", {}) or {}

                    # QueryRewriter 完成后推送意图分析结果（用于前端思考面板）
                    if node_name == "query_rewriter" and isinstance(output, dict):
                        yield _make_sse("thinking", {
                            "thinking_steps":    output.get("thinking_steps", []),
                            "query_intent":      output.get("query_intent", ""),
                            "original_question": output.get("original_question", message),
                            "expanded_queries":  output.get("expanded_queries", []),
                        })

                    # Research 完成后：把每轮迭代动作都推送为进度步骤
                    # 这样用户在等待的 30-60s 里能实时看到检索进展
                    # rag_think_log 本身不传给前端（不显示检索迭代细节）
                    elif node_name == "research" and isinstance(output, dict):
                        rag_log = output.get("rag_think_log", [])
                        for log_item in rag_log:
                            action = log_item.get("action", "accept")
                            hint   = _RESEARCH_HINTS.get(action, "🔍 检索中…")
                            yield _make_sse("progress", {
                                "step": hint,
                                "node": f"research_{action}",
                            })

                    # 最后一个节点（critic）完成时，收集最终 state
                    elif node_name == "critic" and isinstance(output, dict):
                        final_result = output

            # ── 工作流结束：构造与 /chat 相同的 done 事件 ──────────────────
            if final_result is None:
                # astream_events 没有给出 critic output 时，回退到 ainvoke
                state2 = chat_service._get_or_create_session_state(session_id)
                state2 = reset_query_state(state2)
                state2["question"]    = message
                state2["session_id"]  = session_id
                final_result = await chat_service.workflow_app.ainvoke(state2)

            # 更新会话状态（与 process_message 保持一致）
            chat_service._touch_session(session_id, final_result)

            response_text = final_result.get("generation", "暂时无法生成回复。")
            source_label  = final_result.get("source", "未知来源")
            _db.save_message(session_id, "assistant", response_text, source_label)

            from datetime import datetime
            yield _make_sse("done", {
                "response":          response_text,
                "source":            source_label,
                "timestamp":         datetime.now().strftime("%H:%M"),
                "success":           bool(response_text),
                "session_id":        session_id,
                "thinking_steps":    final_result.get("thinking_steps", []),
                "original_question": final_result.get("original_question", message),
                "query_intent":      final_result.get("query_intent", ""),
                # rag_think_log 不传给前端：检索迭代细节不在用户界面显示，
                # 进度信息已通过 progress 事件逐步推送，done 里不再重复
                "rag_think_log":     [],
                "tool_trace":        final_result.get("tool_trace", []),
            })

        except Exception as exc:
            import traceback
            yield _make_sse("error", {"message": f"处理失败：{exc}", "detail": traceback.format_exc()[-300:]})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",    # 禁用 Nginx 缓冲，保证实时推送
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.post("/clear", summary="清空当前会话的对话状态")
async def clear_endpoint(req: Request):
    """清空当前会话的内存对话状态。"""
    chat_service.clear_conversation(_get_session_id(req))
    return {"message": "对话已清空", "success": True}


@router.post("/new-chat", summary="创建新对话会话")
async def new_chat_endpoint(req: Request):
    """创建新的对话会话并生成新的 session ID。"""
    new_id = str(uuid.uuid4())
    req.session["session_id"] = new_id
    return {"message": "新对话已创建", "session_id": new_id, "success": True}