"""
MedicalAI — core/langgraph_workflow.py
LangGraph 状态图定义：五 Agent 精简架构。

工作流架构：

  memory ──(有缓存答案)───► END
    │
    │(无缓存)
    ▼
  query_rewriter → planner(初始规划, phase="init")
                         │
                         ▼
                   research_agent          ← ReAct 内部循环
                         │                   (RAG/工具/Wiki/Tavily)
                         ▼
                   planner(结果评估, phase="eval")
                    ┌────┴────┐
              不满足│         │满足
                    ▼         ▼
                research   critic_agent     ← 独立事实核查
                (重规划)    ┌──┴──┐
                         不通过│     │通过
                               ▼     ▼
                           research  END
                           (重检索，不经 planner)

路由函数说明：
  _route_after_memory       有缓存→END，无缓存→query_rewriter
  _route_after_planner      phase="init"→research；phase="eval" satisfied→critic；否则→research
  _route_after_critic       通过→END；不通过→research（直接重检索，不经 Planner）

设计原则：
  - planner_eval.phase 显式标记调用阶段，不依赖 reason 字符串匹配（脆弱）
  - Critic 失败只写 replan_instruction，不污染 planner_eval（职责隔离）
  - Critic 重入路径绕过 Planner，replan_count 不被提前消耗
"""

from langgraph.graph import END, StateGraph

from app.agents.memory import MemoryAgent
from app.agents.query_rewriter import QueryRewriterAgent
from app.agents.planner import PlannerAgent
from app.agents.research import ResearchAgent
from app.agents.critic import CriticAgent
from app.core.state import AgentState


# 路由函数 

def _route_after_memory(state: AgentState) -> str:
    """
    Memory 短路路由：
    - cache_hit=True → 直接 END，跳过全链路（重复问题缓存命中）
    - 否则 → query_rewriter 继续正常流程
    """
    if state.get("cache_hit", False):
        return END
    return "query_rewriter"


def _route_after_planner(state: AgentState) -> str:
    """
    Planner 路由，基于显式 phase 字段判断，不依赖 reason 字符串匹配：
    - phase="init"  → 初始规划刚完成，进入 research 执行
    - phase="eval" + satisfied=True  → 结果满意，进入 critic 核查
    - phase="eval" + satisfied=False → 结果不满意，重规划，回到 research
    - planner_eval 为 None（防御性兜底）→ 进入 research
    """
    eval_result = state.get("planner_eval")
    if eval_result is None:
        return "research"

    phase = eval_result.get("phase", "init")

    if phase == "init":
        return "research"

    # phase == "eval"
    if eval_result.get("satisfied", False):
        return "critic"
    return "research"


def _route_after_critic(state: AgentState) -> str:
    """
    Critic 核查后路由：
    - 通过 → END
    - 不通过 → 直接回到 research 重检索（不经 Planner，避免消耗 replan_count）

    Critic 已将修订指令写入 state["replan_instruction"]，
    ResearchAgent 会读取并注入到下一轮 THINK 阶段。
    """
    result = state.get("critic_result")
    if result and result.get("passed"):
        return END
    return "research"


#  工作流工厂 

def create_workflow():
    """构建并编译五 Agent LangGraph 工作流。"""
    workflow = StateGraph(AgentState)

    #  注册节点 
    workflow.add_node("memory",         MemoryAgent)
    workflow.add_node("query_rewriter", QueryRewriterAgent)
    workflow.add_node("planner",        PlannerAgent)
    workflow.add_node("research",       ResearchAgent)
    workflow.add_node("critic",         CriticAgent)

    #  入口 
    workflow.set_entry_point("memory")

    #  Memory 短路条件边 
    # 命中缓存 → END；未命中 → query_rewriter
    workflow.add_conditional_edges(
        "memory",
        _route_after_memory,
        {
            END:              END,
            "query_rewriter": "query_rewriter",
        },
    )

    #  固定顺序边 
    workflow.add_edge("query_rewriter", "planner")

    # research 完成后，交回 Planner 评估
    # 注意：Critic 失败后重入的 research 也走这条边，
    # 但 Planner 会通过 phase 字段正确识别为"eval"阶段，
    # 且 critic_attempt_count 的保护保证 Critic 最多执行 MAX_CRITIC_ATTEMPTS 次。
    workflow.add_edge("research", "planner")

    #  Planner 条件边 
    workflow.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "research": "research",
            "critic":   "critic",
        },
    )

    #  Critic 条件边 
    # 不通过时直接到 research，绕过 Planner
    workflow.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            END:        END,
            "research": "research",
        },
    )

    return workflow.compile()

