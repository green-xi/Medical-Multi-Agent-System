"""
MedicalAI — memory/short_term.py
短期记忆管理器：维护当前会话的滑动对话窗口。

设计原则：
  - 保留最近 N 轮完整对话（默认 10 轮 = 20 条消息）
  - 超出窗口的早期内容由 LLM 压缩为一条摘要注入系统消息
  - 摘要本身也纳入 Token 预算估算，防止上下文溢出
  - 无 LLM 可用时直接截断，不阻塞流程
"""

from __future__ import annotations

import re
from time import perf_counter
from typing import List, Optional

from app.core.logging_config import logger
from app.core.state import ChatTurn

# 短期窗口保留的最近轮数（一轮 = user + assistant 各一条）
SHORT_TERM_WINDOW = 10
# 触发压缩的历史条数阈值
COMPRESS_THRESHOLD = SHORT_TERM_WINDOW * 2  # 20 条


def _rough_token_count(text: str) -> int:
    """粗略估算 Token 数（中文约 1.5 字/token，英文约 4 字/token）。"""
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    rest = len(text) - chinese
    return int(chinese / 1.5 + rest / 4)


def compress_history(
    history: List[ChatTurn],
    llm=None,
) -> List[ChatTurn]:
    """
    压缩超出短期窗口的历史记录。

    流程：
      1. 取出窗口之外的早期对话 old_turns
      2. 若 LLM 可用，调用其生成 3 句摘要
      3. 将摘要作为 system 消息插到最前，保留最近 N 轮详细对话
      4. LLM 不可用时直接丢弃早期对话（硬截断）
    """
    if len(history) <= COMPRESS_THRESHOLD:
        return history

    old_turns = history[:-SHORT_TERM_WINDOW * 2]
    recent_turns = history[-SHORT_TERM_WINDOW * 2:]

    # 过滤掉旧的 system 摘要条目，避免摘要叠加
    old_turns = [t for t in old_turns if t.get("role") != "system"]

    if not old_turns:
        return recent_turns

    summary_text = _summarize_with_llm(old_turns, llm)

    if summary_text:
        system_turn: ChatTurn = {
            "role": "system",
            "content": f"【历史对话摘要（早期）】{summary_text}",
        }
        logger.info(
            "短期记忆压缩：%d 条早期对话 → 摘要（约 %d tokens）",
            len(old_turns),
            _rough_token_count(summary_text),
        )
        return [system_turn] + list(recent_turns)
    else:
        # 无 LLM：硬截断，只保留近期对话
        logger.warning("短期记忆：无 LLM 可用，直接截断 %d 条早期对话", len(old_turns))
        return list(recent_turns)


def _summarize_with_llm(turns: List[ChatTurn], llm=None) -> Optional[str]:
    """调用 LLM 将多轮对话压缩为 3 句中文摘要。"""
    if llm is None:
        try:
            from app.tools.llm_client import get_llm
            llm = get_llm()
        except Exception:
            return None

    if llm is None:
        return None

    dialogue = "\n".join(
        f"{'患者' if t['role'] == 'user' else '助手'}：{t.get('content', '')}"
        for t in turns
        if t.get("role") in ("user", "assistant")
    )

    prompt = (
        "请用 3 句简洁的中文概括以下医疗对话的核心内容，"
        "重点保留：患者主诉、主要症状、诊疗建议、用药情况。"
        "不要加任何前缀或解释，直接输出摘要。\n\n"
        f"对话内容：\n{dialogue}"
    )

    try:
        t0 = perf_counter()
        response = llm.invoke(prompt)
        summary = (
            response.content.strip()
            if hasattr(response, "content")
            else str(response).strip()
        )
        logger.debug("短期记忆摘要生成耗时 %.1fms", (perf_counter() - t0) * 1000)
        return summary if summary else None
    except Exception as exc:
        logger.warning("短期记忆摘要生成失败：%s", exc)
        return None


def build_context_window(history: List[ChatTurn], max_tokens: int = 3000) -> List[ChatTurn]:
    """
    在 Token 预算内从历史中选取最有价值的上下文。

    策略：
      - system 摘要始终保留（高价值）
      - 从最近一轮开始往前取，直到接近 Token 上限
    """
    system_turns = [t for t in history if t.get("role") == "system"]
    dialogue_turns = [t for t in history if t.get("role") != "system"]

    used_tokens = sum(_rough_token_count(t.get("content", "")) for t in system_turns)
    selected: List[ChatTurn] = []

    for turn in reversed(dialogue_turns):
        cost = _rough_token_count(turn.get("content", ""))
        if used_tokens + cost > max_tokens:
            break
        selected.insert(0, turn)
        used_tokens += cost

    return system_turns + selected
