"""
MedicalAI — agents/memory.py
MemoryAgent：整合短期记忆（滑动窗口压缩）与长期记忆（持久化画像注入）。

工作流位置：memory → planner → ...

每轮执行两个阶段：

  Phase 1 — 短期记忆维护
    · 对话历史超过阈值时，LLM 压缩早期内容为摘要
    · 保留最近 N 轮完整对话 + 1 条摘要系统消息

  Phase 2 — 长期记忆读取 & 注入
    · 从数据库加载该 session 的用户画像与医疗事实
    · 格式化后写入 state["long_term_context"]，供 Executor 注入提示词

  Phase 3 — 长期记忆更新（异步，上轮对话提取）
    · 对上一轮 Q&A 提取结构化医疗信息（非阻塞，失败不影响主流程）
"""

from time import perf_counter
from typing import Optional

from app.core.logging_config import logger
from app.core.state import AgentState, append_tool_trace, set_node_latency
from app.memory.short_term import compress_history, build_context_window
from app.memory.long_term import long_term_memory


def MemoryAgent(state: AgentState) -> AgentState:
    start_time = perf_counter()
    append_tool_trace(state, "memory")

    session_id: str = state.get("session_id", "")

    # ── Phase 1：短期记忆压缩 ──────────────────────────────────────────────
    history = state.get("conversation_history", [])
    compressed = compress_history(history)
    state["conversation_history"] = compressed

    # 构建当前轮次使用的上下文窗口（Token 预算内）
    state["context_window"] = build_context_window(compressed)

    logger.debug(
        "短期记忆：原始 %d 条 → 压缩后 %d 条，上下文窗口 %d 条",
        len(history), len(compressed), len(state["context_window"]),
    )

    # ── Phase 2：长期记忆读取并注入 ─────────────────────────────────────────
    if session_id:
        try:
            lt_context = long_term_memory.format_for_prompt(session_id)
            state["long_term_context"] = lt_context
            if lt_context:
                logger.info("长期记忆：已为会话 %s 加载用户档案", session_id[:8])
            else:
                logger.debug("长期记忆：会话 %s 暂无历史档案", session_id[:8])
        except Exception as exc:
            logger.warning("长期记忆加载失败（不影响主流程）：%s", exc)
            state["long_term_context"] = ""
    else:
        state["long_term_context"] = ""

    # ── Phase 3：从上一轮对话提取并更新长期记忆 ────────────────────────────
    # 取最近一组 user+assistant 对话（不含本轮问题，因本轮尚未回答）
    if session_id:
        _try_extract_last_turn(state, session_id, compressed)

    set_node_latency(state, "memory", (perf_counter() - start_time) * 1000)
    return state


def _try_extract_last_turn(state: AgentState, session_id: str, history) -> None:
    """
    从压缩后的历史中找到最近一轮完整 Q&A，触发长期记忆提取。
    失败时静默降级，不抛出异常。
    """
    # 找最近的 assistant 回答及其对应的 user 问题
    last_answer: Optional[str] = None
    last_question: Optional[str] = None
    turn_index = 0

    for i in range(len(history) - 1, -1, -1):
        turn = history[i]
        if turn.get("role") == "assistant" and last_answer is None:
            last_answer = turn.get("content", "")
            turn_index = i
        elif turn.get("role") == "user" and last_answer is not None:
            last_question = turn.get("content", "")
            break

    if not last_question or not last_answer:
        return  # 还没有完整的一轮对话，跳过

    try:
        long_term_memory.extract_and_save(
            session_id=session_id,
            question=last_question,
            answer=last_answer,
            turn_index=turn_index,
        )
    except Exception as exc:
        logger.warning("长期记忆提取失败（不影响主流程）：%s", exc)
