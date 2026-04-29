"""
MedicalAI — core/langgraph_workflow.py
LangGraph 状态图定义：五 Agent 精简架构。

工作流架构：

  memory → query_rewriter → planner(初始规划)
                                  │
                                  ▼
                            research_agent          ← ReAct 内部循环
                                  │                   (RAG/工具/Wiki/Tavily)
                                  ▼
                            planner(结果评估)
                             ┌────┴────┐
                       不满足│         │满足
                             ▼         ▼
                         research   critic_agent     ← 独立事实核查
                         (重规划)    ┌──┴──┐
                                 不通过│     │通过
                                     ▼     ▼
                                 research  END
                                 (重检索)

路由函数说明：
  _route_after_planner_init   初始规划后直接进入 research
  _route_after_planner_eval   评估后：满足→critic，不满足→research
  _route_after_critic         通过→END，不通过→research（Critic 触发重检索）
"""

from langgraph.graph import END, StateGraph

from app.agents.memory import MemoryAgent
from app.agents.query_rewriter import QueryRewriterAgent
from app.agents.planner import PlannerAgent
from app.agents.research import ResearchAgent
from app.agents.critic import CriticAgent
from app.core.state import AgentState


# ── 路由函数 ────────────────────────────────────────────────────────────────────

def _route_after_planner(state: AgentState) -> str:
    """
    Planner 被调用后的路由：
    - planner_eval 的 reason 含"初始规划"，说明是第一次调用 → 去 research
    - satisfied=True  → 进入 critic 核查
    - satisfied=False → 重规划，回到 research
    """
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
    """
    Critic 核查后的路由：
    - 通过 → END
    - 不通过且未超限 → 回到 research 重检索
    """
    result = state.get("critic_result")
    if result and result.get("passed"):
        return END
    # 不通过：Critic 已写好 replan_instruction，回到 research
    return "research"


# ── 工作流工厂 ──────────────────────────────────────────────────────────────────

def create_workflow():
    """构建并编译五 Agent LangGraph 工作流。"""
    workflow = StateGraph(AgentState)

    # ── 注册节点 ──────────────────────────────────────────────────────────────
    workflow.add_node("memory",         MemoryAgent)
    workflow.add_node("query_rewriter", QueryRewriterAgent)
    workflow.add_node("planner",        PlannerAgent)
    workflow.add_node("research",       ResearchAgent)
    workflow.add_node("critic",         CriticAgent)

    # ── 入口 ──────────────────────────────────────────────────────────────────
    workflow.set_entry_point("memory")

    # ── 有向边（固定顺序） ────────────────────────────────────────────────────
    workflow.add_edge("memory",         "query_rewriter")
    workflow.add_edge("query_rewriter", "planner")

    # research 完成后，交回 Planner 评估
    workflow.add_edge("research", "planner")

    # ── 条件边 ────────────────────────────────────────────────────────────────

    # Planner：初始规划→research；评估满足→critic；不满足→research
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "research": "research",
            "critic":   "critic",
        },
    )

    # Critic：通过→END；不通过→research 重检索
    workflow.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            END:        END,
            "research": "research",
        },
    )

    return workflow.compile()
