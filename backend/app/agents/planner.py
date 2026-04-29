"""
MedicalAI — agents/planner.py
PlannerAgent：Plan-Replan 闭环规划器。

职责
----
  本节点在工作流中被调用两次：

  第一次调用（初始规划）：
    - 分析用户意图，制定执行计划，路由至 ResearchAgent
    - state["planner_eval"] 为 None，表示尚未收到执行结果

  第二次调用（结果评估 + 可能的 Replan）：
    - 接收 ResearchAgent 写入的 state["generation"] 和 state["documents"]
    - 通过 LLM 判断执行结果是否满足原始查询意图
    - 满足 → 放行，进入 CriticAgent
    - 不满足且重规划次数未超上限 → 写入 replan_instruction，触发重执行
    - 超出上限 → 强制放行，避免死循环

重规划上限：MAX_REPLAN = 1（最多重跑一次，平衡质量与响应速度）
"""

import json
import re
from time import perf_counter

from app.core.logging_config import logger
from app.core.state import (
    AgentState,
    PlannerEval,
    RouteDecision,
    append_tool_trace,
    record_fallback,
    set_node_latency,
)
from app.tools.llm_client import get_llm

MAX_REPLAN = 1  # 最多重规划次数

MEDICAL_KEYWORDS = [
    "发烧", "头痛", "咳嗽", "胸痛", "腹痛", "腹泻", "呕吐", "恶心",
    "高血压", "糖尿病", "炎症", "感染", "症状", "药", "治疗", "诊断", "检查",
    "fever", "headache", "cough", "blood pressure", "diabetes",
    "pain", "treatment", "medicine", "symptom",
]

TOOL_KEYWORDS = [
    "天气", "气温", "下雨", "下雪", "湿度", "花粉", "气压",
    "布洛芬", "阿司匹林", "对乙酰氨基酚", "二甲双胍", "奥美拉唑", "阿莫西林",
    "氯雷他定", "药品", "副作用", "剂量", "服用",
    "是什么意思", "化验", "报告", "指标", "偏高", "偏低", "正常值",
    "肌酐", "血糖", "糖化", "甘油三酯", "ldl", "hdl", "tsh", "转氨酶",
    "ct报告", "mri报告",
]


# ── 初始路由决策 ───────────────────────────────────────────────────────────────

def _keyword_route(question: str) -> RouteDecision:
    normalized = question.lower()
    has_tool_kw = any(kw in normalized for kw in TOOL_KEYWORDS)
    has_medical_kw = any(kw in normalized for kw in MEDICAL_KEYWORDS)

    # 同时命中工具和医学关键词时，医学优先
    # 例："今天的温度和湿度，在北京腿疼是因为风湿吗" → 湿度触发工具、腿疼/风湿触发医学
    if has_tool_kw and has_medical_kw:
        return {
            "is_medical": True,
            "tool": "retriever",
            "confidence": 0.72,
            "reason": "同时命中工具和医学关键词，医学意图优先，走知识库检索。",
            "strategy": "keyword_fallback",
        }
    if has_tool_kw:
        return {
            "is_medical": True,
            "tool": "tool_agent",
            "confidence": 0.75,
            "reason": "命中工具类关键词（天气/药品/检验指标），优先调用结构化工具。",
            "strategy": "keyword_fallback",
        }
    if any(kw in normalized for kw in MEDICAL_KEYWORDS):
        return {
            "is_medical": True,
            "tool": "retriever",
            "confidence": 0.72,
            "reason": "命中医学关键词，优先使用知识库检索增强回答可靠性。",
            "strategy": "keyword_fallback",
        }
    return {
        "is_medical": False,
        "tool": "llm_agent",
        "confidence": 0.62,
        "reason": "未命中明显医学检索意图，优先走通用中文问答。",
        "strategy": "keyword_fallback",
    }


def _llm_route(question: str, llm) -> RouteDecision | None:
    prompt = (
        "你是医疗问句路由器。请判断用户问题最适合走哪条处理路径。\n"
        "只返回 JSON，不要输出额外文本。\n\n"
        "可选路径说明：\n"
        '- "tool_agent"：问题纯粹关于天气查询、具体药品信息、化验报告指标解读等需要实时/结构化数据的查询\n'
        '- "retriever"：问题涉及疾病症状、治疗方案、医学诊断等，适合从知识库检索\n'
        '- "llm_agent"：问题是一般健康咨询或闲聊，直接用大模型回答即可\n\n'
        "⚠️ 重要：\n"
        "1. 如果问题同时提到天气和疾病症状（如'下雨天腿疼'、'天冷关节痛'），"
        "核心意图是疾病诊断而非天气查询，应走 retriever。\n"
        "2. 如果问题问的是具体药品信息或化验单指标解读，走 tool_agent。\n"
        "3. 如果不确定，宁可选 retriever 也不要选 tool_agent。\n\n"
        f"用户问题：{question}\n\n"
        '返回格式：{"is_medical": true/false, "tool": "tool_agent" 或 "retriever" 或 "llm_agent", '
        '"confidence": 0 到 1 之间的小数, "reason": "一句中文理由"}'
    )
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        payload = json.loads(match.group())
        tool = payload.get("tool")
        if tool not in {"retriever", "llm_agent", "tool_agent"}:
            return None
        return {
            "is_medical": bool(payload.get("is_medical", True)),
            "tool": tool,
            "confidence": max(0.0, min(1.0, float(payload.get("confidence", 0.5)))),
            "reason": str(payload.get("reason", "")),
            "strategy": "llm_classifier",
        }
    except Exception as exc:
        logger.warning("Planner LLM 路由失败：%s", exc)
        return None


# ── 执行结果评估（第二次调用时使用） ──────────────────────────────────────────

def _llm_evaluate(question: str, generation: str, llm) -> PlannerEval | None:
    """
    用 LLM 判断 ResearchAgent 的生成结果是否满足原始查询意图。
    使用独立的 LLM 调用，不依赖 ResearchAgent 的推理链。
    """
    if not generation or len(generation.strip()) < 20:
        return {
            "satisfied": False,
            "reason": "生成内容为空或过短，无法满足查询需求。",
            "replan_action": "请重新检索并生成完整的医疗回答。",
            "replan_count": 0,
        }

    prompt = (
        "你是一名医疗问答质量评估员。请判断下方「AI回答」是否充分满足了「用户问题」的需求。\n"
        "只返回 JSON，不要输出额外文本。\n\n"
        f"用户问题：{question}\n\n"
        f"AI回答：{generation[:600]}\n\n"
        "评估标准：\n"
        "1. 回答是否直接针对了用户的问题（不是答非所问）\n"
        "2. 回答是否包含实质性的医学信息（不是空泛的建议）\n"
        "3. 对于症状/疾病类问题，是否涵盖了可能原因或处理建议\n\n"
        '返回格式：{"satisfied": true/false, "reason": "一句评估理由", '
        '"replan_action": "如不满足，给出具体的改进指令；满足则留空字符串"}'
    )
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        payload = json.loads(match.group())
        return {
            "satisfied": bool(payload.get("satisfied", False)),
            "reason": str(payload.get("reason", "")),
            "replan_action": str(payload.get("replan_action", "")),
            "replan_count": 0,  # 由调用方填充
        }
    except Exception as exc:
        logger.warning("Planner 评估 LLM 失败：%s", exc)
        return None


def _heuristic_evaluate(generation: str) -> PlannerEval:
    """LLM 不可用时的启发式评估降级。"""
    if not generation or len(generation.strip()) < 50:
        return {
            "satisfied": False,
            "reason": "回答内容过短，启发式判断不满足。",
            "replan_action": "请重新检索并生成更完整的回答。",
            "replan_count": 0,
        }
    # 检测常见"无法回答"套话
    evasion_patterns = ["暂时无法", "无法给出", "无法分析", "请咨询医生", "建议就医"]
    if all(p in generation for p in evasion_patterns[:2]):
        return {
            "satisfied": False,
            "reason": "回答仅包含免责声明，缺乏实质医学内容。",
            "replan_action": "请提供更具体的医学信息，不能仅给出就医建议。",
            "replan_count": 0,
        }
    return {
        "satisfied": True,
        "reason": "回答长度和内容符合基本要求（启发式判断）。",
        "replan_action": "",
        "replan_count": 0,
    }


# ── 主函数 ─────────────────────────────────────────────────────────────────────

def PlannerAgent(state: AgentState) -> AgentState:
    """
    Plan-Replan 闭环规划器。

    第一次调用：state["planner_eval"] 为 None → 执行初始路由规划
    第二次调用：state["planner_eval"] 不为 None → 评估 ResearchAgent 结果，决定放行或重规划
    """
    start_time = perf_counter()
    append_tool_trace(state, "planner")
    llm = get_llm()

    # ── 判断是第一次调用还是回溯评估 ─────────────────────────────────────────
    is_evaluating = state.get("planner_eval") is not None

    if not is_evaluating:
        # ══════════════════════════════════
        # 阶段一：初始规划（制定执行路径）
        # ══════════════════════════════════
        question = state["question"].strip()

        decision: RouteDecision | None = None
        if llm:
            decision = _llm_route(question, llm)
            if decision is None:
                record_fallback(state, "planner_llm_route_failed")

        if decision is None:
            if not llm:
                record_fallback(state, "planner_no_llm")
            decision = _keyword_route(question)

        state["route_decision"] = decision
        state["current_tool"] = decision["tool"]
        state["confidence_score"] = decision["confidence"]

        # 初始化 planner_eval 为占位值，区别于 None（代表"已路由，待评估"）
        state["planner_eval"] = {
            "satisfied": False,
            "reason": "初始规划完成，等待执行结果。",
            "replan_action": "",
            "replan_count": 0,
        }
        state["replan_instruction"] = ""

        logger.info(
            "Planner [初始规划] → %s，策略：%s（置信度=%.2f）",
            decision["tool"], decision["strategy"], decision["confidence"],
        )

    else:
        # ══════════════════════════════════════════
        # 阶段二：执行结果评估 + 可能的 Replan
        # ══════════════════════════════════════════
        question = state.get("original_question") or state["question"]
        generation = state.get("generation", "")
        current_replan_count = state["planner_eval"].get("replan_count", 0)

        # 超出重规划上限，强制放行
        if current_replan_count >= MAX_REPLAN:
            state["planner_eval"] = {
                "satisfied": True,
                "reason": f"已达重规划上限（{MAX_REPLAN}次），强制放行以保证响应速度。",
                "replan_action": "",
                "replan_count": current_replan_count,
            }
            state["metrics"]["replan_count"] = current_replan_count
            logger.info("Planner [评估] 重规划次数已达上限，强制放行")
            set_node_latency(state, "planner_eval", (perf_counter() - start_time) * 1000)
            return state

        # LLM 评估
        eval_result: PlannerEval | None = None
        if llm:
            eval_result = _llm_evaluate(question, generation, llm)
        if eval_result is None:
            eval_result = _heuristic_evaluate(generation)

        eval_result["replan_count"] = current_replan_count
        state["planner_eval"] = eval_result

        if eval_result["satisfied"]:
            state["metrics"]["replan_count"] = current_replan_count
            logger.info(
                "Planner [评估] ✓ 结果满足意图：%s", eval_result["reason"]
            )
        else:
            # 触发重规划：写入补充指令，重置 ResearchAgent 相关状态
            new_replan_count = current_replan_count + 1
            state["planner_eval"]["replan_count"] = new_replan_count

            # ── 智能重规划指令：检测上一轮失败模式，注入不同策略 ────────────
            # 根因：LLM 生成的 replan_action 通常很笼统（"请重新检索"），
            # ResearchAgent 会重复相同的失败路径（expand_query→tool_query→失败→短回答）。
            # 这里检测失败模式，注入具体指令打破循环。
            tool_results = state.get("tool_results", {}) or {}
            prev_tool_result = tool_results.get("result", "")
            prev_tool_name = tool_results.get("tool", "")

            # 模式1：get_weather 工具失败（网络错误/超时）→ 跳过天气，直接联网检索医学知识
            if "天气查询失败" in prev_tool_result or "get_weather" in prev_tool_name:
                state["replan_instruction"] = (
                    "【重规划指令】上一轮 get_weather 工具因网络问题失败。"
                    "本轮请直接使用 tavily 联网检索，不要再次调用 get_weather。"
                    "搜索词建议包含：天气与关节疼痛的关系、风湿诱因。"
                )
                logger.info("Planner [评估] 检测到天气工具失败，注入 Tavily 优先策略")
            # 模式2：RAG 盲区（rerank 偏低）→ 跳过 expand_query，直接 tavily
            elif state.get("rag_blind_spot"):
                state["replan_instruction"] = (
                    "【重规划指令】上一轮 RAG 检索命中盲区（本地知识库无相关内容）。"
                    "本轮请直接使用 tavily 联网检索，不要执行 rag_search 或 expand_query。"
                )
                logger.info("Planner [评估] 检测到 RAG 盲区，注入 Tavily 优先策略")
            # 模式3：默认使用 LLM 评估生成的指令
            else:
                state["replan_instruction"] = eval_result["replan_action"]

            state["metrics"]["replan_count"] = new_replan_count

            # 重置执行状态，让 ResearchAgent 重新运行
            state["generation"] = ""
            state["documents"] = []
            state["rag_grader_passed"] = False
            state["rag_iterations"] = 0
            state["rag_think_log"] = []
            state["llm_success"] = False
            state["rag_success"] = False
            state["wiki_attempted"] = False
            state["wiki_success"] = False
            state["tavily_attempted"] = False
            state["tavily_success"] = False

            record_fallback(state, f"planner_replan_{new_replan_count}:{eval_result['reason']}")
            logger.info(
                "Planner [评估] ✗ 不满足意图，触发重规划 #%d：%s",
                new_replan_count, eval_result["reason"],
            )

    set_node_latency(
        state,
        "planner_eval" if is_evaluating else "planner",
        (perf_counter() - start_time) * 1000,
    )
    return state
