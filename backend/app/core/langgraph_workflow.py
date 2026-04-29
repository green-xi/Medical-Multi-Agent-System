"""LangGraph 状态图：五 Agent 编排工作流。"""

from langgraph.graph import END, StateGraph

from app.agents.memory import MemoryAgent
from app.agents.query_rewriter import QueryRewriterAgent
from app.agents.planner import PlannerAgent
from app.agents.research import ResearchAgent
from app.agents.critic import CriticAgent
from app.core.state import AgentState


def _route_after_planner(state: AgentState) -> str:
    eval_result = state.get("planner_eval")
    if eval_result is None:
        return "research"

    reason = eval_result.get("reason", "")
    if "初始规划" in reason or "等待执行" in reason:
        return "research"

    if eval_result.get("satisfied", False):
        return "critic"
    else:
        return "research"


def _route_after_critic(state: AgentState) -> str:
    result = state.get("critic_result")
    if result and result.get("passed"):
        return END
    return "research"


def create_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("memory",         MemoryAgent)
    workflow.add_node("query_rewriter", QueryRewriterAgent)
    workflow.add_node("planner",        PlannerAgent)
    workflow.add_node("research",       ResearchAgent)
    workflow.add_node("critic",         CriticAgent)

    workflow.set_entry_point("memory")
    workflow.add_edge("memory",         "query_rewriter")
    workflow.add_edge("query_rewriter", "planner")
    workflow.add_edge("research",       "planner")

    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"research": "research", "critic": "critic"},
    )

    workflow.add_conditional_edges(
        "critic",
        _route_after_critic,
        {END: END, "research": "research"},
    )

    return workflow.compile()
