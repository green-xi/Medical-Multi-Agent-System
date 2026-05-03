"""MCP 统一外部工具客户端（Tavily/Wikipedia/PubMed）。"""

import json
import re
from time import perf_counter
from typing import List

from app.core.logging_config import logger
from app.core.state import AgentState, append_tool_trace, set_node_latency
from app.tools.llm_client import get_llm


# 　　 Prompt 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

_REWRITE_PROMPT = """\
你是一名医疗查询理解专家。请分析用户的原始问题，结合对话历史，完成查询重写任务。

【对话历史（最近3轮）】
{history}

【用户原始问题】
{question}

【任务说明】
请完成以下分析并严格按 JSON 格式返回：

1. intent（意图分类）：从以下选一个
   - symptom_inquiry（症状咨询）
   - medication_inquiry（用药/药物询问）
   - report_interpretation（检验/影像报告解读）
   - disease_inquiry（疾病知识查询）
   - treatment_inquiry（治疗方案咨询）
   - general_health（一般健康咨询）
   - chitchat（闲聊/非医疗）

2. rewritten_question（重写后的问题）：
   - 补全省略的主语/指代词
   - 将口语化表达转为医学规范术语
   - 若原问题已足够清晰，小幅优化即可，不要过度改写
   - 保持中文，长度控制在原问题的 1-1.5 倍以内

3. expanded_queries（扩展查询词列表，2-3个）：
   - 用于向量检索召回更多相关文档
   - 应覆盖不同角度：病因、症状、治疗、预后等

4. thinking（思考过程，给用户展示的推理步骤列表，每步一句话）：
   - 步骤1：理解用户意图
   - 步骤2：识别关键医学概念
   - 步骤3：查询优化决策
   - 自然、专业、简洁，每步不超过 30 字

只返回 JSON，不要有多余文字：
{{
  "intent": "<意图>",
  "rewritten_question": "<重写后的问题>",
  "expanded_queries": ["<扩展词1>", "<扩展词2>"],
  "thinking": ["<步骤1>", "<步骤2>", "<步骤3>"]
}}
"""


# 　　 启发式降级 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

_SYMPTOM_WORDS   = [
    "痛", "疼", "发烧", "咳嗽", "头晕", "恶心", "呕吐", "腹泻", "出血", "肿", "痒",
    "麻", "乏力", "胸闷", "心悸", "气短", "呼吸困难", "心跳", "晕厥", "抽搐",
    "流鼻涕", "鼻塞", "打喷嚏", "眼干", "眼涩", "视力", "耳鸣", "耳痛",
    "腹胀", "便秘", "腹部", "皮疹", "红疹", "脱发", "口干", "口渴", "多汗",
    "突然", "剧烈",  # 急症触发词
]
_MED_WORDS       = [
    # 通用用药词
    "药", "吃药", "服用", "剂量", "副作用", "处方", "用药", "停药", "换药",
    "过量", "中毒", "相互作用", "药物", "药品", "配伍",
    # 常见具体药名
    "布洛芬", "阿司匹林", "抗生素", "止痛", "退烧药", "降压药", "降糖药",
    "二甲双胍", "他汀", "氨氯地平", "厄贝沙坦", "氯沙坦", "美托洛尔",
    "维生素", "钙片", "叶酸", "铁剂", "鱼油", "益生菌",
    "头孢", "青霉素", "阿莫西林", "左氧氟沙星", "甲硝唑",
    "地塞米松", "泼尼松", "氢化可的松",
    "滴眼液", "眼药水", "鼻喷剂", "气雾剂", "栓剂",
    # 「是否需要用药」边界意图（AC-001 根本原因）
    "需要吃", "需要服", "该吃", "该用", "要不要吃", "要不要用",
    "需要用药", "是否用药", "开始用药", "吃降压", "吃降糖", "吃血压",
    "需要降压", "需要降糖", "需要控制",
]
_REPORT_WORDS    = [
    "报告", "化验", "检查结果", "检验单", "体检",
    "ct", "mri", "b超", "超声", "x光", "x线",
    "血常规", "尿常规", "血脂", "血糖", "肝功能", "肾功能", "甲功",
    "偏高", "偏低", "正常值", "参考范围", "指标", "数值",
    "肌酐", "尿酸", "血红蛋白", "白细胞", "血小板",
    "胆固醇", "甘油三酯", "ldl", "hdl",
]
_DISEASE_WORDS   = [
    "是什么", "什么病", "什么原因", "什么情况", "是什么疾病",
    "高血压", "低血压", "糖尿病", "高血糖", "低血糖",
    "肝炎", "肾炎", "肺炎", "胃炎", "肠炎", "关节炎", "结膜炎", "咽炎", "支气管炎",
    "肿瘤", "癌", "良性", "恶性", "结节", "息肉", "囊肿",
    "心脏病", "冠心病", "心肌梗死", "心衰", "心律失常",
    "脑梗", "脑出血", "中风", "偏瘫",
    "骨质疏松", "骨折", "椎间盘",
    "过敏", "哮喘", "湿疹", "荨麻疹",
    "甲亢", "甲减", "甲状腺",
    "贫血", "白血病", "淋巴瘤",
    "近视", "青光眼", "白内障",
    "痛风", "风湿", "类风湿",
]
_TREATMENT_WORDS = [
    "怎么治", "如何治", "治疗", "手术", "方案", "预后", "康复",
    "怎么办", "如何处理", "如何缓解", "怎么缓解",
    "能治好吗", "能痊愈吗", "会好吗",
    "手术风险", "保守治疗", "物理治疗",
]


def _heuristic_rewrite(question: str, history_text: str) -> dict:
    """LLM 不可用时的关键词启发式降级处理。"""
    q_lower = question.lower()

    # 　　 扩充词典：chitchat / 急症 / 预防 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
    _CHITCHAT_WORDS = ["你好", "hello", "hi ", "谢谢", "再见", "帮我", "介绍一下你", "你是谁"]
    _EMERGENCY_WORDS = ["急救", "120", "胸痛", "胸部疼痛", "失去意识", "休克", "大出血", "呼吸停止"]
    _PREVENTION_WORDS = ["预防", "怎么预防", "如何避免", "降低风险", "减少风险"]

    # 意图推断（优先级从高到低）
    if any(w in q_lower for w in _CHITCHAT_WORDS) and len(question.strip()) < 15:
        # 短句 + 闲聊词 → chitchat（避免把含"你好"的医疗问题误判）
        intent = "chitchat"
    elif any(w in q_lower for w in _EMERGENCY_WORDS):
        # 急症词优先于症状词
        intent = "symptom_inquiry"
    elif any(w in q_lower for w in _REPORT_WORDS):
        intent = "report_interpretation"
    elif any(w in q_lower for w in _MED_WORDS):
        # 「需要吃/是否用药」等新增词组现在能正确触发此分支
        intent = "medication_inquiry"
    elif any(w in q_lower for w in _TREATMENT_WORDS):
        intent = "treatment_inquiry"
    elif any(w in q_lower for w in _DISEASE_WORDS):
        intent = "disease_inquiry"
    elif any(w in q_lower for w in _SYMPTOM_WORDS):
        intent = "symptom_inquiry"
    elif any(w in q_lower for w in _PREVENTION_WORDS):
        intent = "general_health"
    else:
        intent = "general_health"

    # 简单扩展：添加「如何处理」「原因」两个近邻查询
    expanded = [
        f"{question} 原因",
        f"{question} 处理方法",
    ]

    thinking = [
        f"理解用户意图：{_INTENT_CN.get(intent, '健康咨询')}",
        "提取问题中的核心医学关键词",
        "生成扩展查询以提升检索召回率",
    ]

    return {
        "intent": intent,
        "rewritten_question": question,   # 降级时不改写，保持原样
        "expanded_queries": expanded,
        "thinking": thinking,
    }


_INTENT_CN = {
    "symptom_inquiry":      "症状咨询",
    "medication_inquiry":   "用药询问",
    "report_interpretation":"报告解读",
    "disease_inquiry":      "疾病查询",
    "treatment_inquiry":    "治疗咨询",
    "general_health":       "一般健康咨询",
    "chitchat":             "日常交流",
}


# 　　 主函数 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

def QueryRewriterAgent(state: AgentState) -> AgentState:
    """
    查询重写节点。

    读取：
      state["question"]            用户原始问题
      state["conversation_history"] 对话历史

    写入：
      state["question"]            重写后的问题（替换原始问题，供后续节点使用）
      state["original_question"]   保存原始问题（用于前端展示和调试）
      state["query_intent"]        意图分类
      state["expanded_queries"]    扩展查询词列表（供 RAGGrader expand_query 使用）
      state["thinking_steps"]      结构化思考步骤列表（供前端展示）
    """
    start_time = perf_counter()
    append_tool_trace(state, "query_rewriter")

    original_question = state.get("question", "").strip()
    if not original_question:
        set_node_latency(state, "query_rewriter", 0.0)
        return state

    # 保存原始问题
    state["original_question"] = original_question

    # 构建历史摘要文本
    history = state.get("context_window") or state.get("conversation_history", [])
    history_parts = []
    for turn in history[-6:]:   # 最近3轮
        role = turn.get("role", "")
        content = turn.get("content", "")[:80]
        if role == "user":
            history_parts.append(f"患者：{content}")
        elif role == "assistant":
            history_parts.append(f"助手：{content}")
    history_text = "\n".join(history_parts) if history_parts else "（无历史对话）"

    # 　　 chitchat 短路：闲聊类问题直接走启发式，不调用 LLM 　　　　　　　　　　　　　　　　　　　　
    # 判据：启发式能识别为 chitchat，且问题长度 < 20 字（避免把含问候语的医疗问题短路）
    # 收益：对"你好"/"谢谢"等问题节省 5-12 秒 LLM 调用时间
    _heuristic_pre = _heuristic_rewrite(original_question, "")
    if _heuristic_pre["intent"] == "chitchat":
        result = _heuristic_pre
        # 跳过 LLM，直接写回 state
        state["question"]          = result["rewritten_question"]
        state["original_question"] = original_question
        state["query_intent"]      = result["intent"]
        state["expanded_queries"]  = result["expanded_queries"]
        state["thinking_steps"]    = result["thinking"]
        latency_ms = (perf_counter() - start_time) * 1000
        set_node_latency(state, "query_rewriter", latency_ms)
        logger.info(
            "QueryRewriter [chitchat短路] | 原始='%s' | 耗时=%.1fms",
            original_question[:30], latency_ms,
        )
        return state

    llm = get_llm()
    result: dict | None = None

    if llm:
        prompt = _REWRITE_PROMPT.format(
            history=history_text,
            question=original_question,
        )
        try:
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                payload = json.loads(match.group())
                # 验证必要字段
                rewritten = str(payload.get("rewritten_question", "")).strip()
                expanded  = payload.get("expanded_queries", [])
                thinking  = payload.get("thinking", [])
                intent    = str(payload.get("intent", "general_health"))

                if not rewritten:
                    rewritten = original_question

                result = {
                    "intent": intent,
                    "rewritten_question": rewritten,
                    "expanded_queries": [str(q) for q in expanded[:3] if str(q).strip()],
                    "thinking": [str(s) for s in thinking if str(s).strip()],
                }
        except Exception as exc:
            logger.warning("QueryRewriter LLM 解析失败，降级启发式：%s", exc)

    if result is None:
        result = _heuristic_rewrite(original_question, history_text)

    # 　　 写回 state 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
    rewritten_q = result["rewritten_question"]

    # 构造完整思考步骤（含重写对比）
    thinking_steps: List[str] = list(result["thinking"])

    # 如果问题发生了实质性变化，追加一条重写说明
    if rewritten_q != original_question:
        thinking_steps.append(f"查询优化：「{original_question}」→「{rewritten_q}」")

    state["question"]          = rewritten_q
    state["original_question"] = original_question
    state["query_intent"]      = result["intent"]
    state["expanded_queries"]  = result["expanded_queries"]
    state["thinking_steps"]    = thinking_steps

    latency_ms = (perf_counter() - start_time) * 1000
    set_node_latency(state, "query_rewriter", latency_ms)

    logger.info(
        "QueryRewriter | 意图=%s | 原始='%s' → 重写='%s' | 扩展=%s | 耗时=%.1fms",
        result["intent"],
        original_question[:30],
        rewritten_q[:30],
        result["expanded_queries"],
        latency_ms,
    )
    return state