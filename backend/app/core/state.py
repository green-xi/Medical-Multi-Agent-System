"""工作流状态类型定义与辅助函数。"""

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.documents import Document

ToolName = Literal["retriever", "llm_agent", "tool_agent"]
NodeName = Literal[
    "memory",
    "query_rewriter",
    "planner",
    "research",
    "critic",
]


class ChatTurn(TypedDict, total=False):
    role: Literal["user", "assistant", "system"]
    content: str
    source: str
    timestamp: str


class RouteDecision(TypedDict):
    is_medical: bool
    tool: ToolName
    confidence: float
    reason: str
    strategy: Literal["llm_classifier", "keyword_fallback", "safe_default"]


# ── Planner Replan 相关 ────────────────────────────────────────────────────────

class PlannerEval(TypedDict):
    """Planner 对 ResearchAgent 执行结果的评估。"""
    satisfied: bool           # 是否满足原始查询意图
    reason: str               # 评估理由
    replan_action: str        # 不满足时的重规划指令，满足时为 ""
    replan_count: int         # 已重规划次数


# ── Critic 评估结果 ────────────────────────────────────────────────────────────

class FactCheckItem(TypedDict):
    claim: str                # 答案中提取的医学事实断言
    status: Literal["verified", "unverifiable", "contradicted"]
    note: str                 # 核查说明


class CriticResult(TypedDict):
    passed: bool              # 整体是否通过
    hallucination_detected: bool
    fact_checks: List[FactCheckItem]
    revised_answer: str       # 通过时为润色后答案，失败时为空
    feedback: str             # 打回给 ResearchAgent 的修订意见


class WorkflowMetrics(TypedDict):
    total_latency_ms: float
    node_latencies_ms: Dict[str, float]
    rag_hit: bool
    llm_used: bool
    fallback_count: int
    rerank_used: bool
    rerank_latency_ms: float
    replan_count: int         # Planner 重规划次数
    critic_pass: bool         # Critic 是否一次通过
    critic_attempt_count: int  # Critic 已执行核查次数（进入时递增，MAX_CRITIC_ATTEMPTS=2）


class AgentState(TypedDict):
    # ── 核心问答字段 ───────────────────────────────────
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    source: str
    search_query: str

    # ── 查询重写字段 ───────────────────────────────────
    query_intent: str
    expanded_queries: List[str]
    thinking_steps: List[str]

    # ── 会话标识 ───────────────────────────────────────
    session_id: str

    # ── 短期记忆 ───────────────────────────────────────
    conversation_history: List[ChatTurn]
    context_window: List[ChatTurn]

    # ── 长期记忆 ───────────────────────────────────────
    long_term_context: str

    # ── 路由与工具状态 ─────────────────────────────────
    current_tool: ToolName
    confidence_score: float
    route_decision: RouteDecision

    # ── Planner Plan-Replan 闭环字段 ──────────────────
    planner_eval: Optional[PlannerEval]   # Planner 对本轮执行结果的评估
    replan_instruction: str               # 重规划时给 ResearchAgent 的补充指令

    # ── ResearchAgent（原 RAGGrader）字段 ─────────────
    rag_grader_passed: bool
    rag_iterations: int
    rag_think_log: List[Dict]
    research_strategy: str                # 本轮 ResearchAgent 选用的策略

    # ── Tool 状态（内化进 ResearchAgent） ─────────────
    tool_results: Dict[str, Any]
    tool_agent_success: bool

    # ── LLM 状态 ───────────────────────────────────────
    llm_attempted: bool
    llm_success: bool
    rag_attempted: bool
    rag_success: bool
    wiki_attempted: bool
    wiki_success: bool
    tavily_attempted: bool
    tavily_success: bool
    retry_count: int

    # ── CriticAgent 字段 ──────────────────────────────
    critic_result: Optional[CriticResult]  # Critic 评估结果
    critic_attempt_count: int   # 已执行核查次数，进入时递增（MAX_CRITIC_ATTEMPTS=2）

    # ── 可观测性 ───────────────────────────────────────
    tool_trace: List[str]
    fallback_events: List[str]
    metrics: WorkflowMetrics


# ── 工厂函数 ──────────────────────────────────────────────────────────────────

def default_route_decision() -> RouteDecision:
    return {
        "is_medical": True,
        "tool": "retriever",
        "confidence": 0.5,
        "reason": "默认优先使用医学知识库检索以保证回答稳健。",
        "strategy": "safe_default",
    }


def default_metrics() -> WorkflowMetrics:
    return {
        "total_latency_ms": 0.0,
        "node_latencies_ms": {},
        "rag_hit": False,
        "llm_used": False,
        "fallback_count": 0,
        "rerank_used": False,
        "rerank_latency_ms": 0.0,
        "replan_count": 0,
        "critic_pass": False,
        "critic_attempt_count": 0,
    }


def initialize_conversation_state(session_id: str = "") -> AgentState:
    return {
        "question": "",
        "original_question": "",
        "documents": [],
        "generation": "",
        "source": "",
        "search_query": "",
        "query_intent": "",
        "expanded_queries": [],
        "thinking_steps": [],
        "session_id": session_id,
        "conversation_history": [],
        "context_window": [],
        "long_term_context": "",
        "current_tool": "retriever",
        "confidence_score": 0.5,
        "route_decision": default_route_decision(),
        "planner_eval": None,
        "replan_instruction": "",
        "rag_grader_passed": False,
        "rag_iterations": 0,
        "rag_think_log": [],
        "research_strategy": "",
        "tool_results": {},
        "tool_agent_success": False,
        "llm_attempted": False,
        "llm_success": False,
        "rag_attempted": False,
        "rag_success": False,
        "wiki_attempted": False,
        "wiki_success": False,
        "tavily_attempted": False,
        "tavily_success": False,
        "retry_count": 0,
        "critic_result": None,
        "critic_attempt_count": 0,
        "tool_trace": [],
        "fallback_events": [],
        "metrics": default_metrics(),
    }


def reset_query_state(state: AgentState) -> AgentState:
    """重置单次查询相关的临时字段，保留对话历史与记忆字段。"""
    state.update(
        {
            "question": "",
            "original_question": "",
            "documents": [],
            "generation": "",
            "source": "",
            "search_query": "",
            "query_intent": "",
            "expanded_queries": [],
            "thinking_steps": [],
            "context_window": [],
            "current_tool": "retriever",
            "confidence_score": 0.5,
            "route_decision": default_route_decision(),
            "planner_eval": None,
            "replan_instruction": "",
            "rag_grader_passed": False,
            "rag_iterations": 0,
            "rag_think_log": [],
            "research_strategy": "",
            "tool_results": {},
            "tool_agent_success": False,
            "llm_attempted": False,
            "llm_success": False,
            "rag_attempted": False,
            "rag_success": False,
            "wiki_attempted": False,
            "wiki_success": False,
            "tavily_attempted": False,
            "tavily_success": False,
            "retry_count": 0,
            "critic_result": None,
            "critic_attempt_count": 0,
            "tool_trace": [],
            "fallback_events": [],
            "metrics": default_metrics(),
        }
    )
    return state


def append_tool_trace(state: AgentState, node: str) -> None:
    state["tool_trace"].append(str(node))


def record_fallback(state: AgentState, message: str) -> None:
    state["fallback_events"].append(message)
    state["metrics"]["fallback_count"] = len(state["fallback_events"])


def set_node_latency(state: AgentState, node: str, latency_ms: float) -> None:
    state["metrics"]["node_latencies_ms"][str(node)] = round(latency_ms, 2)