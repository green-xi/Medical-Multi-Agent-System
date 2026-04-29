"""CriticAgent：独立事实核查 + 幻觉检测 + 风格润色。"""

import json
import re
from time import perf_counter
from typing import List, Optional

from langchain_core.documents import Document

from app.core.logging_config import logger
from app.core.state import (
    AgentState,
    CriticResult,
    FactCheckItem,
    append_tool_trace,
    record_fallback,
    set_node_latency,
)
from app.tools.llm_client import get_llm
from app.tools.mcp_client import (
    MCP_AVAILABLE,
    MCP_SERVER_CONFIGS,
    mcp_pubmed_search,
)

MAX_CRITIC_ATTEMPTS = 2
MIN_ANSWER_LENGTH   = 20
PUBMED_MAX_RESULTS  = 3
MIN_PUBMED_DOCS     = 2


def _extract_pubmed_queries(question: str, answer: str, llm) -> List[str]:
    prompt = (
        "你是一名医学文献检索专家。请从以下医疗问答中提取 1-2 个核心医学断言，"
        "转换为适合 PubMed 数据库检索的英文关键词（每条不超过 8 个词）。\n"
        "只返回 JSON，格式：{\"queries\": [\"query1\", \"query2\"]}\n\n"
        f"用户问题：{question}\n"
        f"AI 回答（节选）：{answer[:300]}\n\n"
        "要求：\n"
        "- 聚焦于答案中最核心、最需要权威文献验证的医学事实断言\n"
        "- 使用 PubMed 常用医学术语（MeSH 词汇优先）\n"
        "- 英文输出，每条简洁精准\n"
        "示例：{\"queries\": [\"vitamin C COVID-19 prevention RCT\", "
        "\"ascorbic acid immune function systematic review\"]}"
    )
    try:
        response = llm.invoke(prompt, config={"timeout": 10})
        text = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            queries = json.loads(match.group()).get("queries", [])
            valid = [q.strip() for q in queries if q.strip()][:2]
            if valid:
                return valid
    except Exception as exc:
        logger.warning("PubMed 查询词提取失败，使用降级查询：%s", exc)
    return [question[:60]]


def _broaden_pubmed_query(query: str, llm) -> Optional[str]:
    prompt = (
        "请将以下 PubMed 检索词放宽为更通用的版本（去掉具体数值/修饰词，保留核心医学概念）。"
        "只返回一个英文检索词，不超过 6 个词，不加任何解释。\n\n"
        f"原检索词：{query}"
    )
    try:
        response = llm.invoke(prompt, config={"timeout": 8})
        text = response.content if hasattr(response, "content") else str(response)
        broad = text.strip().strip('"').strip("'")
        if broad and broad != query and len(broad) < len(query) + 10:
            return broad
    except Exception as exc:
        logger.debug("_broaden_pubmed_query 失败：%s", exc)
    return None


def _fetch_pubmed_context(question: str, answer: str, llm) -> Optional[str]:
    if not MCP_AVAILABLE or "pubmed" not in MCP_SERVER_CONFIGS:
        logger.debug("PubMed MCP 未启用（MCP_PUBMED_ENABLED=false），降级为 RAG 文档核查")
        return None

    queries = _extract_pubmed_queries(question, answer, llm) if llm else [question[:60]]
    if not queries:
        return None

    seen_titles: set = set()
    all_results: List[dict] = []
    for query in queries:
        try:
            results = mcp_pubmed_search(query=query, max_results=PUBMED_MAX_RESULTS)
            hit_count = len(results)

            if hit_count == 0 and llm:
                broad_query = _broaden_pubmed_query(query, llm)
                if broad_query and broad_query != query:
                    logger.info("CriticAgent PubMed [%s] 命中 0 篇，尝试放宽查询：[%s]", query, broad_query)
                    try:
                        results = mcp_pubmed_search(query=broad_query, max_results=PUBMED_MAX_RESULTS)
                        hit_count = len(results)
                        logger.info("CriticAgent PubMed 放宽查询 [%s]：%d 篇文献", broad_query, hit_count)
                    except Exception as exc2:
                        logger.warning("CriticAgent PubMed 放宽查询失败：%s", exc2)

            for r in results:
                title = r.get("title", "")
                if title not in seen_titles:
                    seen_titles.add(title)
                    all_results.append(r)
            logger.info("CriticAgent PubMed [%s]：%d 篇文献", query, hit_count)
        except Exception as exc:
            logger.warning("CriticAgent PubMed 检索失败 [%s]：%s", query, exc)

    if not all_results:
        logger.warning("CriticAgent PubMed 未返回任何文献，降级为 RAG 文档核查")
        return None

    if len(all_results) < MIN_PUBMED_DOCS:
        logger.warning(
            "CriticAgent PubMed 文献数 %d < MIN_PUBMED_DOCS=%d，降级为 RAG 文档核查",
            len(all_results), MIN_PUBMED_DOCS,
        )
        return None

    parts = []
    for i, r in enumerate(all_results[:5]):
        title   = r.get("title", f"PubMed 文献 {i+1}")
        content = r.get("content", "")[:500]
        url     = r.get("url", "")
        parts.append(
            f"[PubMed-{i+1}] 《{title}》\n{content}"
            + (f"\n来源：{url}" if url else "")
        )

    context = "\n\n".join(parts)
    logger.info(
        "CriticAgent PubMed 上下文构建完成：%d 篇，%d 字符",
        len(all_results[:5]), len(context),
    )
    return context


def _build_rag_doc_context(docs: List[Document]) -> str:
    if not docs:
        return "（无检索文档，答案来自模型内部知识）"
    parts = [f"[RAG-{i+1}] {doc.page_content[:400]}" for i, doc in enumerate(docs[:5])]
    return "\n\n".join(parts)


def _llm_fact_check(
    question: str,
    answer: str,
    ref_context: str,
    ref_source: str,
    llm,
) -> "CriticResult | None":
    has_tool_doc = "[工具:" in ref_context

    if ref_source == "pubmed":
        source_hint = (
            "【参考来源】PubMed 权威医学文献库（独立于本次对话的 RAG 检索）。\n"
            "请严格基于这些文献摘要判断各断言准确性。\n"
            "重要原则：\n"
            "  - 文献未提及的通用医学常识（如基本生理机制）判为 unverifiable，不算幻觉。\n"
            "  - 答案中的具体数值（如体温阈值、时间天数）若与文献不符但属于\n"
            "    临床常识范围（如 38.5°C、3 天），应判为 unverifiable 而非 contradicted。\n"
            "  - 只有当数值或结论与文献明确矛盾时，才判为 contradicted。\n"
            "  - 幻觉（hallucination）仅指答案中出现文献完全未涉及且明显错误的\n"
            "    具体药名/剂量/数据，不包括公认临床常识数值。\n\n"
        )
    elif has_tool_doc:
        source_hint = (
            "【参考来源】包含工具查询结果（如天气 API 的结构化数据）。\n"
            "答案引用工具数据不算幻觉；通用医学常识类断言判为 unverifiable。\n\n"
        )
    elif ref_context.startswith("（无检索文档"):
        source_hint = (
            "【参考来源】无检索文档，答案来自模型通用医学知识。\n"
            "核查标准（宽松模式）：\n"
            "  - 答案所有断言均判为 unverifiable（无文档参照，无法 verified 也无法 contradicted）。\n"
            "  - hallucination_detected=true 的唯一条件：答案包含明显错误的具体剂量/药名，\n"
            "    例如'布洛芬每次3000mg'这类明显超出常规范围的数字。\n"
            "  - 否定性循证结论（如'目前无证据表明X能预防Y'）不算幻觉，判 unverifiable。\n"
            "  - passed=true 的条件：无明显医学错误，答案内容合理。\n\n"
        )
    else:
        source_hint = (
            "【参考来源】本地医学知识库（RAG 检索结果）。\n"
            "重要区分——两类幻觉的判定标准不同：\n"
            "  ■ 第一类（必须判 contradicted）：答案断言与文档内容明确矛盾。\n"
            "    例：文档写'每日不超过5g盐'，答案写'每日10g盐以内'。\n"
            "  ■ 第二类（不算幻觉，判 unverifiable）：答案包含文档未覆盖的陈述，\n"
            "    但该陈述属于公认医学常识或循证医学结论。\n"
            "    例：'目前没有科学证据表明维生素C可以预防新冠病毒感染'——\n"
            "    即便文档未提及维生素C，这是公认的循证医学结论，判 unverifiable，不算幻觉。\n"
            "    例：'接种疫苗是预防传染病最有效的手段'——同上，判 unverifiable。\n"
            "只有当答案出现文档完全未涉及且医学上明显错误的具体数值/药名/剂量时，\n"
            "才将 hallucination_detected 设为 true。\n\n"
        )

    prompt = (
        "你是一名独立的医疗事实核查员，任务是评估 AI 医疗回答的准确性。\n"
        "核查依据仅限于下方【参考文档】，不得用你自己的医学知识「补充」事实。\n"
        f"{source_hint}"
        "只返回 JSON，不要输出额外文字。\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【待核查的 AI 回答】\n{answer}\n\n"
        f"【参考文档】\n{ref_context}\n\n"
        "任务：\n"
        "1. 从 AI 回答中提取 2-4 个关键医学事实断言（症状/诊断/治疗方案/剂量等）\n"
        "2. 对每个断言判断其状态：\n"
        '   - "verified"    ：参考文档有明确依据支持\n'
        '   - "unverifiable"：文档未提及，但属于合理通用医学常识（包括否定性循证结论）\n'
        '   - "contradicted"：与文档内容明确矛盾，或存在明显医学错误\n'
        "   ⚠️ 关键区分：'目前无科学证据表明X能预防/治疗Y'这类否定性循证陈述\n"
        "      即便文档未提及，也属于公认医学常识，状态为 unverifiable，不是 contradicted。\n"
        "3. 判断幻觉：hallucination_detected=true 当且仅当答案出现文档完全未涉及\n"
        "   且医学上明显错误的具体剂量/药名/数值（如'每日服用维生素C 10000mg可预防癌症'）。\n"
        "   否定性陈述、预防建议（接种疫苗/戴口罩）等公认常识不属于幻觉。\n"
        "4. 整体通过（无 contradicted 且幻觉=false）→ 对答案通俗化润色；\n"
        "   不通过 → 给出具体修订意见（指出哪里错、应该怎么改）\n\n"
        "返回格式（严格 JSON）：\n"
        '{"passed": true/false,\n'
        ' "hallucination_detected": true/false,\n'
        ' "fact_checks": [\n'
        '   {"claim": "断言内容", "status": "verified/unverifiable/contradicted", "note": "说明"}\n'
        ' ],\n'
        ' "revised_answer": "通过时：润色后完整回答；不通过时：空字符串",\n'
        ' "feedback": "不通过时：具体修订指令；通过时：空字符串"}'
    )

    try:
        response = llm.invoke(prompt, config={"timeout": 25})
        text = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        payload = json.loads(match.group())
        fact_checks: List[FactCheckItem] = []
        has_contradiction = False
        for item in payload.get("fact_checks", [])[:6]:
            status = item.get("status", "unverifiable")
            if status not in {"verified", "unverifiable", "contradicted"}:
                status = "unverifiable"
            if status == "contradicted":
                has_contradiction = True
            fact_checks.append({
                "claim":  str(item.get("claim", "")),
                "status": status,
                "note":   str(item.get("note", "")),
            })

        hallucination = bool(payload.get("hallucination_detected", False))
        passed = (
            bool(payload.get("passed", False))
            and not has_contradiction
            and not hallucination
        )
        revised  = str(payload.get("revised_answer", "")).strip()
        feedback = str(payload.get("feedback", "")).strip()
        if passed and not revised:
            revised = answer

        return {
            "passed":                 passed,
            "hallucination_detected": hallucination,
            "fact_checks":            fact_checks,
            "revised_answer":         revised if passed else "",
            "feedback":               feedback if not passed else "",
        }

    except Exception as exc:
        logger.warning("CriticAgent LLM 核查失败：%s", exc)
        return None


def _heuristic_check(answer: str, docs: List[Document]) -> CriticResult:
    if not answer or len(answer.strip()) < MIN_ANSWER_LENGTH:
        return {
            "passed": False, "hallucination_detected": False,
            "fact_checks": [], "revised_answer": "",
            "feedback": "回答内容过短或为空，需要重新生成。",
        }

    risk_patterns = {
        "具体剂量无来源": bool(re.search(r"\d+\s*mg|\d+\s*毫克", answer)) and not docs,
        "确诊性语言":     any(p in answer for p in ["确诊为", "你患有", "你得了"]),
    }
    issues = [desc for desc, triggered in risk_patterns.items() if triggered]

    if issues:
        return {
            "passed": False, "hallucination_detected": True,
            "fact_checks": [
                {"claim": issue, "status": "contradicted", "note": "启发式检测到风险模式"}
                for issue in issues
            ],
            "revised_answer": "",
            "feedback": f"检测到以下问题：{'；'.join(issues)}。请修改后重新生成。",
        }

    disclaimer = "\n\n⚠️ 以上信息仅供参考，不构成医疗诊断，如有疑虑请咨询专业医生。"
    revised = answer if answer.endswith("就医。") else answer + disclaimer
    return {
        "passed": True, "hallucination_detected": False,
        "fact_checks": [{"claim": "整体内容", "status": "unverifiable", "note": "LLM 不可用，启发式通过"}],
        "revised_answer": revised, "feedback": "",
    }


def _force_pass_with_disclaimer(answer: str) -> CriticResult:
    disclaimer = (
        "\n\n⚠️ 本回答基于医学知识库及通用医学资料生成，部分内容未能与权威来源逐一核对，"
        "请以专业医生的判断为准，切勿自行诊断或用药。"
    )
    return {
        "passed": True, "hallucination_detected": False,
        "fact_checks": [{"claim": "（超出核查次数上限，强制放行）", "status": "unverifiable", "note": "已添加免责声明"}],
        "revised_answer": answer.rstrip() + disclaimer, "feedback": "",
    }


def CriticAgent(state: AgentState) -> AgentState:
    start_time = perf_counter()
    append_tool_trace(state, "critic")

    answer   = state.get("generation", "")
    docs     = state.get("documents", [])
    question = state.get("original_question") or state.get("question", "")

    attempt_count = state.get("critic_attempt_count", 0) + 1
    state["critic_attempt_count"] = attempt_count

    if not answer or len(answer.strip()) < MIN_ANSWER_LENGTH:
        state["critic_attempt_count"] = attempt_count - 1
        result: CriticResult = {
            "passed": False, "hallucination_detected": False,
            "fact_checks": [], "revised_answer": "",
            "feedback": "答案为空或过短，需要重新检索生成。",
        }
        state["critic_result"] = result
        state["metrics"]["critic_pass"] = False
        set_node_latency(state, "critic", (perf_counter() - start_time) * 1000)
        logger.warning("CriticAgent：答案为空，标记不通过（不消耗 attempt）")
        return state

    if attempt_count > MAX_CRITIC_ATTEMPTS:
        result = _force_pass_with_disclaimer(answer)
        state["critic_result"] = result
        state["generation"]    = result["revised_answer"]
        state["metrics"]["critic_pass"]          = True
        state["metrics"]["critic_attempt_count"] = attempt_count
        set_node_latency(state, "critic", (perf_counter() - start_time) * 1000)
        logger.info(
            "CriticAgent：已执行 %d 次核查（上限 %d），强制通过并添加免责声明",
            attempt_count - 1, MAX_CRITIC_ATTEMPTS,
        )
        return state

    llm = get_llm()

    pubmed_context: Optional[str] = None
    if llm:
        pubmed_context = _fetch_pubmed_context(question, answer, llm)

    if pubmed_context:
        ref_context = pubmed_context
        ref_source  = "pubmed"
        logger.info("CriticAgent 第 %d 次核查：使用 PubMed 独立文献（%d 字符）", attempt_count, len(pubmed_context))
    else:
        ref_context = _build_rag_doc_context(docs)
        ref_source  = "rag" if docs else "empty"
        logger.info("CriticAgent 第 %d 次核查：PubMed 不可用，降级为 %s（文档数=%d）", attempt_count, ref_source, len(docs))

    state["critic_ref_source"] = ref_source

    result = None
    if llm:
        result = _llm_fact_check(question, answer, ref_context, ref_source, llm)
    if result is None:
        result = _heuristic_check(answer, docs)

    state["critic_result"] = result
    state["metrics"]["critic_pass"]          = result["passed"]
    state["metrics"]["critic_attempt_count"] = attempt_count
    state["metrics"]["critic_ref_source"]    = ref_source

    if result["passed"]:
        state["generation"] = result["revised_answer"]
        logger.info(
            "CriticAgent ✓ 第 %d 次核查通过（来源=%s，幻觉=%s，矛盾数=%d）",
            attempt_count, ref_source, result["hallucination_detected"],
            sum(1 for fc in result["fact_checks"] if fc["status"] == "contradicted"),
        )
    else:
        state["replan_instruction"] = (
            f"[CriticAgent 第{attempt_count}次核查反馈] {result['feedback']}\n"
            "请重新检索并修正上述问题后重新生成答案。"
        )
        state["planner_eval"] = {
            "satisfied": False, "reason": f"CriticAgent 核查不通过：{result['feedback'][:80]}",
            "replan_action": result["feedback"],
            "replan_count": state.get("planner_eval" or {}).get("replan_count", 0),
        }
        record_fallback(state, f"critic_failed_attempt{attempt_count}:{result['feedback'][:60]}")
        logger.warning(
            "CriticAgent ✗ 第 %d 次核查不通过（来源=%s，幻觉=%s），触发重检索",
            attempt_count, ref_source, result["hallucination_detected"],
        )

    set_node_latency(state, "critic", (perf_counter() - start_time) * 1000)
    return state
