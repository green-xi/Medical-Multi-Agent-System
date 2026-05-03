"""短期记忆压缩 + 长期记忆注入与更新。"""
"""
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
    question: str = state.get("question", "").strip()

    #  短路检查：完全相同的问题在本 session 是否有缓存答案 
    # 典型场景：用户重复提问，或前端因网络问题重发同一请求
    # 检查 conversation_history 中最近一次 user+assistant 轮次
    if question:
        cached = _find_cached_answer(state.get("conversation_history", []), question)
        if cached:
            state["cache_hit"] = True
            state["generation"] = cached
            logger.info("MemoryAgent 命中缓存答案，跳过全链路（问题前20字：%s…）", question[:20])
            set_node_latency(state, "memory", (perf_counter() - start_time) * 1000)
            return state

    state["cache_hit"] = False
    #  Phase 1：短期记忆压缩
    history = state.get("conversation_history", [])
    compressed = compress_history(history)
    state["conversation_history"] = compressed

    # 构建当前轮次使用的上下文窗口（Token 预算内）
    state["context_window"] = build_context_window(compressed)

    logger.debug(
        "短期记忆：原始 %d 条 → 压缩后 %d 条，上下文窗口 %d 条",
        len(history), len(compressed), len(state["context_window"]),
    )

    #  Phase 2：长期记忆读取并注入 
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

    #  Phase 3：从上一轮对话提取并更新长期记忆 
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

def _find_cached_answer(history: list, question: str) -> Optional[str]:
    """
    在对话历史中查找与当前问题完全匹配的最近一次答案。

    规则：
    - 只匹配完全相同的问题（strip 后比较），不做模糊匹配，避免误命中
    - 只取最近一次匹配，返回对应的 assistant 回答
    - 空答案不算缓存命中

    局限性说明：
    本函数仅用于防重发（网络重试、用户刷新）场景。
    语义相似但文字不同的问题不会命中，这是有意为之——
    相似问题可能有不同的时效性需求，强行缓存会给出过期答案。
    """
    last_user_idx: Optional[int] = None
    # 从后往前扫描，找到第一个与当前问题匹配的 user 消息
    for i in range(len(history) - 1, -1, -1):
        turn = history[i]
        if turn.get("role") == "user" and turn.get("content", "").strip() == question:
            last_user_idx = i
            break

    if last_user_idx is None:
        return None

    # 找紧跟在该 user 消息后的 assistant 回答
    for i in range(last_user_idx + 1, len(history)):
        turn = history[i]
        if turn.get("role") == "assistant":
            answer = turn.get("content", "").strip()
            return answer if answer else None
        if turn.get("role") == "user":
            # 中间又插了一条 user 消息，说明没有对应的 assistant 回答
            break

    return None
