"""
test_workflow.py — LangGraph 工作流路由函数单元测试

覆盖 langgraph_workflow.py 中所有条件路由函数：
  _route_after_planner  — 初始规划→research；评估满足→critic；不满足→research
  _route_after_critic   — 通过→END；不通过→research

五 Agent 架构：
  memory → query_rewriter → planner → research → planner(评估) → critic → END
"""
import os
import sys

from langgraph.graph import END

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.langgraph_workflow import (   # noqa: E402
    _route_after_critic,
    _route_after_planner,
    create_workflow,
)
from app.core.state import initialize_conversation_state  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# _route_after_planner
# ══════════════════════════════════════════════════════════════════════════════

class TestRouteAfterPlanner:

    def test_routes_to_research_when_planner_eval_is_none(self):
        """planner_eval=None（极端初始状态），应去 research。"""
        state = initialize_conversation_state()
        state["planner_eval"] = None
        assert _route_after_planner(state) == "research"

    def test_routes_to_research_after_initial_planning(self):
        """初始规划完成后（reason 含'初始规划'），应去 research 执行。"""
        state = initialize_conversation_state()
        state["planner_eval"] = {
            "satisfied": False,
            "reason": "初始规划完成，等待执行结果。",
            "replan_action": "",
            "replan_count": 0,
        }
        assert _route_after_planner(state) == "research"

    def test_routes_to_critic_when_satisfied(self):
        """评估满足时，应进入 critic 核查。"""
        state = initialize_conversation_state()
        state["planner_eval"] = {
            "satisfied": True,
            "reason": "回答直接针对问题，包含实质医学信息",
            "replan_action": "",
            "replan_count": 0,
        }
        assert _route_after_planner(state) == "critic"

    def test_routes_to_research_when_not_satisfied(self):
        """评估不满足时，触发重规划，应回到 research。"""
        state = initialize_conversation_state()
        state["planner_eval"] = {
            "satisfied": False,
            "reason": "回答缺乏实质医学内容",
            "replan_action": "请重新检索并生成更详细的回答",
            "replan_count": 1,
        }
        assert _route_after_planner(state) == "research"

    def test_routes_to_critic_when_force_passed(self):
        """达到重规划上限强制放行时，satisfied=True 且 reason 不含'初始规划'，应到 critic。"""
        state = initialize_conversation_state()
        state["planner_eval"] = {
            "satisfied": True,
            "reason": "已达重规划上限（1次），强制放行以保证响应速度。",
            "replan_action": "",
            "replan_count": 1,
        }
        assert _route_after_planner(state) == "critic"

    def test_routes_to_research_when_waiting_for_execution(self):
        """reason 含'等待执行'也是初始规划阶段的标志，应去 research。"""
        state = initialize_conversation_state()
        state["planner_eval"] = {
            "satisfied": False,
            "reason": "等待执行结果",
            "replan_action": "",
            "replan_count": 0,
        }
        assert _route_after_planner(state) == "research"


# ══════════════════════════════════════════════════════════════════════════════
# _route_after_critic
# ══════════════════════════════════════════════════════════════════════════════

class TestRouteAfterCritic:

    def test_routes_to_end_when_passed(self):
        state = initialize_conversation_state()
        state["critic_result"] = {
            "passed": True,
            "hallucination_detected": False,
            "fact_checks": [],
            "revised_answer": "润色后答案",
            "feedback": "",
        }
        assert _route_after_critic(state) == END

    def test_routes_to_research_when_failed(self):
        state = initialize_conversation_state()
        state["critic_result"] = {
            "passed": False,
            "hallucination_detected": True,
            "fact_checks": [{"claim": "x", "status": "contradicted", "note": ""}],
            "revised_answer": "",
            "feedback": "答案含幻觉，需重新检索",
        }
        assert _route_after_critic(state) == "research"

    def test_routes_to_research_when_critic_result_is_none(self):
        """critic_result 为 None（极端情况），应回到 research 而非崩溃。"""
        state = initialize_conversation_state()
        state["critic_result"] = None
        assert _route_after_critic(state) == "research"

    def test_routes_to_end_when_force_passed_with_disclaimer(self):
        """超出重试上限的强制通过，passed=True，应直接到 END。"""
        state = initialize_conversation_state()
        state["critic_result"] = {
            "passed": True,
            "hallucination_detected": False,
            "fact_checks": [
                {"claim": "（超出核查重试上限，强制放行）", "status": "unverifiable", "note": ""}
            ],
            "revised_answer": "答案内容⚠️ 本回答基于医学知识库生成，请以专业医生判断为准。",
            "feedback": "",
        }
        assert _route_after_critic(state) == END

    def test_routes_to_research_when_hallucination_without_pass(self):
        """存在幻觉且 passed=False，应回到 research 重检索。"""
        state = initialize_conversation_state()
        state["critic_result"] = {
            "passed": False,
            "hallucination_detected": True,
            "fact_checks": [
                {"claim": "某断言", "status": "contradicted", "note": "与文档矛盾"}
            ],
            "revised_answer": "",
            "feedback": "存在幻觉，请修正",
        }
        assert _route_after_critic(state) == "research"


# ══════════════════════════════════════════════════════════════════════════════
# create_workflow 完整性检查
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowCreation:

    def test_workflow_compiles_without_error(self):
        """工作流应能正常编译，无异常。"""
        workflow = create_workflow()
        assert workflow is not None

    def test_workflow_has_all_five_nodes(self):
        """编译后的工作流应包含五个 Agent 节点。"""
        workflow = create_workflow()
        node_names = set(workflow.nodes.keys())
        expected = {"memory", "query_rewriter", "planner", "research", "critic"}
        assert expected.issubset(node_names), (
            f"缺少节点: {expected - node_names}。当前节点: {node_names}"
        )

    def test_workflow_node_count(self):
        """工作流 Agent 节点数量应为 5（五 Agent），不含内置 __start__ 等节点。"""
        workflow = create_workflow()
        agent_nodes = {n for n in workflow.nodes.keys() if not n.startswith("__")}
        assert len(agent_nodes) == 5, (
            f"期望 5 个 Agent 节点，实际 {len(agent_nodes)} 个：{agent_nodes}"
        )
