"""短期记忆压缩 + 长期记忆注入与更新。"""

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

    history = state.get("conversation_history", [])
    compressed = compress_history(history)
    state["conversation_history"] = compressed
    state["context_window"] = build_context_window(compressed)
    logger.debug(
        "短期记忆：原始 %d 条 → 压缩后 %d 条，上下文窗口 %d 条",
        len(history), len(compressed), len(state["context_window"]),
    )

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

    if session_id:
        _try_extract_last_turn(state, session_id, compressed)

    set_node_latency(state, "memory", (perf_counter() - start_time) * 1000)
    return state


def _try_extract_last_turn(state: AgentState, session_id: str, history) -> None:
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
        return

    try:
        long_term_memory.extract_and_save(
            session_id=session_id,
            question=last_question,
            answer=last_answer,
            turn_index=turn_index,
        )
    except Exception as exc:
        logger.warning("长期记忆提取失败（不影响主流程）：%s", exc)
