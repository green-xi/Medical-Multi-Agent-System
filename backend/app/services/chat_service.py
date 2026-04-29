"""
MedicalAI — services/chat_service.py
ChatService：工作流编排、会话生命周期管理与可观测性指标收集。
"""

from collections import OrderedDict
from datetime import datetime
from time import perf_counter, time
from typing import Any, Dict, TypedDict
import asyncio

from app.core.config import MAX_ACTIVE_SESSIONS, SESSION_TTL_SECONDS
from app.core.langgraph_workflow import create_workflow
from app.core.logging_config import logger
from app.core.state import AgentState, initialize_conversation_state, reset_query_state
from app.services.database_service import db_service


class SessionEntry(TypedDict):
    state: AgentState
    last_accessed: float


class ChatService:
    """为每条聊天消息编排多智能体工作流。"""

    def __init__(self):
        self.workflow_app = None
        self.conversation_states: "OrderedDict[str, SessionEntry]" = OrderedDict()
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.metrics_summary = {
            "requests_total": 0,
            "fallback_requests": 0,
            "rag_attempts": 0,
            "rag_hits": 0,
            "llm_attempts": 0,
            "llm_failures": 0,
            "explanation_runs": 0,
            "total_latency_ms": 0.0,
        }
        logger.info("ChatService 初始化完成")

    def initialize_workflow(self) -> None:
        if not self.workflow_app:
            logger.info("正在初始化 LangGraph 工作流…")
            self.workflow_app = create_workflow()
            logger.info("LangGraph 工作流初始化成功")

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self.session_locks:
            self.session_locks[session_id] = asyncio.Lock()
        return self.session_locks[session_id]

    def _touch_session(self, session_id: str, state: AgentState) -> None:
        self.conversation_states[session_id] = {
            "state": state,
            "last_accessed": time(),
        }
        self.conversation_states.move_to_end(session_id)

    def _evict_stale_sessions(self) -> None:
        now = time()
        expired = [
            sid for sid, entry in self.conversation_states.items()
            if now - entry["last_accessed"] > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self.conversation_states.pop(sid, None)
            self.session_locks.pop(sid, None)
            logger.info("已清理过期会话 %s", sid[:8])

        while len(self.conversation_states) > MAX_ACTIVE_SESSIONS:
            sid, _ = self.conversation_states.popitem(last=False)
            self.session_locks.pop(sid, None)
            logger.info("已驱逐 LRU 会话 %s", sid[:8])

    def _get_or_create_session_state(self, session_id: str) -> AgentState:
        self._evict_stale_sessions()
        if session_id in self.conversation_states:
            state = self.conversation_states[session_id]["state"]
            self._touch_session(session_id, state)
            return state

        # ── 内存中没有该 session（首次加载或后端重启后）──────────────────────
        # 从数据库恢复对话历史，确保重启后上下文不丢失
        state = initialize_conversation_state(session_id=session_id)
        try:
            db_rows = db_service.get_chat_history(session_id)
            if db_rows:
                restored = []
                for row in db_rows:
                    role = row.get("role", "")
                    content = row.get("content", "")
                    if role in ("user", "assistant") and content:
                        turn = {"role": role, "content": content}
                        if row.get("source"):
                            turn["source"] = row["source"]
                        restored.append(turn)
                state["conversation_history"] = restored
                logger.info(
                    "会话 %s 从数据库恢复 %d 条历史消息",
                    session_id[:8], len(restored),
                )
        except Exception as exc:
            logger.warning("恢复会话历史失败（不影响主流程）：%s", exc)

        self._touch_session(session_id, state)
        return state

    def _record_metrics(self, result: AgentState, latency_ms: float) -> None:
        self.metrics_summary["requests_total"] += 1
        self.metrics_summary["total_latency_ms"] += latency_ms
        if result.get("fallback_events"):
            self.metrics_summary["fallback_requests"] += 1
        if result.get("rag_attempted"):
            self.metrics_summary["rag_attempts"] += 1
        if result.get("rag_success"):
            self.metrics_summary["rag_hits"] += 1
        if result.get("llm_attempted"):
            self.metrics_summary["llm_attempts"] += 1
        if result.get("llm_attempted") and not result.get("llm_success"):
            self.metrics_summary["llm_failures"] += 1
        if result.get("metrics", {}).get("explanation_used"):
            self.metrics_summary["explanation_runs"] += 1

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        requests_total = self.metrics_summary["requests_total"]
        rag_attempts = self.metrics_summary["rag_attempts"]
        llm_attempts = self.metrics_summary["llm_attempts"]
        return {
            "requests_total": requests_total,
            "active_sessions": len(self.conversation_states),
            "session_ttl_seconds": SESSION_TTL_SECONDS,
            "max_active_sessions": MAX_ACTIVE_SESSIONS,
            "avg_latency_ms": round(
                self.metrics_summary["total_latency_ms"] / requests_total, 2
            ) if requests_total else 0.0,
            "fallback_rate": round(
                self.metrics_summary["fallback_requests"] / requests_total, 4
            ) if requests_total else 0.0,
            "rag_hit_rate": round(
                self.metrics_summary["rag_hits"] / rag_attempts, 4
            ) if rag_attempts else 0.0,
            "llm_failure_rate": round(
                self.metrics_summary["llm_failures"] / llm_attempts, 4
            ) if llm_attempts else 0.0,
            "explanation_runs": self.metrics_summary["explanation_runs"],
        }

    async def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        logger.info("处理会话 %s 的消息…", session_id[:8])

        if not self.workflow_app:
            raise ValueError("工作流未初始化")

        session_lock = self._get_session_lock(session_id)
        async with session_lock:
            request_start = perf_counter()

            db_service.save_message(session_id, "user", message)

            state = self._get_or_create_session_state(session_id)
            state = reset_query_state(state)
            state["question"] = message
            # 确保 session_id 在 state 中始终正确
            state["session_id"] = session_id

            try:
                result = await self.workflow_app.ainvoke(state)
            except AttributeError:
                logger.warning("降级为同步工作流调用")
                result = self.workflow_app.invoke(state)

            total_latency_ms = (perf_counter() - request_start) * 1000
            result["metrics"]["total_latency_ms"] = round(total_latency_ms, 2)

            self._touch_session(session_id, result)
            self._record_metrics(result, total_latency_ms)

            response_text = result.get("generation", "暂时无法生成回复。")
            source = result.get("source", "未知来源")

            db_service.save_message(session_id, "assistant", response_text, source)

            logger.info(
                "工作流完成 | 会话=%s 路径=%s 耗时=%.2fms rag命中=%s 回退次数=%d",
                session_id[:8],
                " -> ".join(result.get("tool_trace", [])),
                total_latency_ms,
                result.get("metrics", {}).get("rag_hit", False),
                len(result.get("fallback_events", [])),
            )

            return {
                "response": response_text,
                "source": source,
                "timestamp": datetime.now().strftime("%H:%M"),
                "success": bool(result.get("generation")),
                "session_id": session_id,
                "thinking_steps":   result.get("thinking_steps", []),
                "original_question": result.get("original_question", ""),
                "query_intent":      result.get("query_intent", ""),
                "rag_think_log":     result.get("rag_think_log", []),
                "tool_trace":        result.get("tool_trace", []),
            }

    def clear_conversation(self, session_id: str) -> None:
        """清空内存对话状态（保留长期记忆）。"""
        if session_id in self.conversation_states:
            self._touch_session(session_id, initialize_conversation_state(session_id=session_id))
            logger.info("已清空会话 %s 的对话历史（长期记忆保留）", session_id[:8])


chat_service = ChatService()