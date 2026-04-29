"""
test_agents.py — 五节点工作流 Agent 单元测试

覆盖：
  - MemoryAgent（短期压缩 + 长期读取 + 提取触发）
  - QueryRewriterAgent（LLM 路径 + 降级路径 + 意图分类）
  - PlannerAgent（初始路由 + 结果评估 + replan 触发）
  - ResearchAgent（llm_direct 路径 + ReAct 循环 + early-exit + 工具调用 + replan注入）
  - CriticAgent（通过 + 幻觉检测 + contradicted覆盖 + 强制通过 + 降级）
"""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.critic import CriticAgent
from app.agents.research import ResearchAgent
from app.agents.memory import MemoryAgent
from app.agents.planner import PlannerAgent
from app.agents.query_rewriter import QueryRewriterAgent
from app.core.state import initialize_conversation_state


# ══════════════════════════════════════════════════════════════════════════════
# PlannerAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestPlannerAgentInitialRouting:
    """第一次调用（初始规划）的测试。"""

    def test_llm_routes_to_retriever_for_medical_question(self, base_state):
        base_state["question"] = "我最近发烧咳嗽，应该怎么办？"
        llm_response = json.dumps({
            "is_medical": True,
            "tool": "retriever",
            "confidence": 0.92,
            "reason": "症状类医疗问题，优先使用知识库检索",
        })
        with patch("app.agents.planner.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=llm_response))
            )
            result = PlannerAgent(base_state)

        assert result["current_tool"] == "retriever"
        assert result["route_decision"]["strategy"] == "llm_classifier"
        assert result["confidence_score"] == pytest.approx(0.92)
        # 初始规划后 planner_eval 被初始化
        assert result["planner_eval"] is not None
        assert "初始规划" in result["planner_eval"]["reason"]

    def test_llm_routes_to_tool_agent_for_drug_query(self, base_state):
        base_state["question"] = "布洛芬每次服用多少剂量合适？"
        llm_response = json.dumps({
            "is_medical": True,
            "tool": "tool_agent",
            "confidence": 0.88,
            "reason": "药品剂量查询，需要结构化工具数据",
        })
        with patch("app.agents.planner.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=llm_response))
            )
            result = PlannerAgent(base_state)

        assert result["current_tool"] == "tool_agent"

    def test_falls_back_to_keyword_when_llm_unavailable(self, base_state):
        base_state["question"] = "我有高血压，平时要注意什么？"
        with patch("app.agents.planner.get_llm", return_value=None):
            result = PlannerAgent(base_state)

        assert result["current_tool"] == "retriever"
        assert result["route_decision"]["strategy"] == "keyword_fallback"
        assert "planner_no_llm" in result["fallback_events"]

    def test_falls_back_to_keyword_when_llm_returns_bad_json(self, base_state):
        base_state["question"] = "头痛怎么办"
        with patch("app.agents.planner.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content="not-json-at-all"))
            )
            result = PlannerAgent(base_state)

        assert result["route_decision"]["strategy"] == "keyword_fallback"
        assert any(e.startswith("planner_llm_route_failed") for e in result["fallback_events"])

    def test_falls_back_to_keyword_when_llm_returns_invalid_tool(self, base_state):
        base_state["question"] = "感冒了"
        bad_response = json.dumps({
            "is_medical": True,
            "tool": "nonexistent_tool",  # 非法 tool 值
            "confidence": 0.5,
            "reason": "test",
        })
        with patch("app.agents.planner.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=bad_response))
            )
            result = PlannerAgent(base_state)

        assert result["route_decision"]["strategy"] == "keyword_fallback"

    def test_keyword_route_tool_keywords_take_priority(self, base_state):
        """工具类关键词（药品/检验）应路由到 tool_agent。"""
        base_state["question"] = "我的肌酐偏高是什么意思"
        with patch("app.agents.planner.get_llm", return_value=None):
            result = PlannerAgent(base_state)
        assert result["current_tool"] == "tool_agent"

    def test_keyword_route_chitchat_routes_to_llm_agent(self, base_state):
        base_state["question"] = "今天天气不错"
        with patch("app.agents.planner.get_llm", return_value=None):
            result = PlannerAgent(base_state)
        assert result["current_tool"] == "llm_agent"


class TestPlannerAgentEvaluation:
    """第二次调用（结果评估）的测试。"""

    def _make_eval_state(self, base_state, generation: str, replan_count: int = 0):
        """构造已进入评估阶段的 state。"""
        base_state["question"] = "维生素C能预防新冠吗"
        base_state["original_question"] = "维生素C能预防新冠吗"
        base_state["generation"] = generation
        base_state["planner_eval"] = {
            "satisfied": False,
            "reason": "初始规划完成，等待执行结果。",
            "replan_action": "",
            "replan_count": replan_count,
        }
        return base_state

    def test_satisfied_when_answer_is_good(self, base_state):
        state = self._make_eval_state(
            base_state,
            "目前没有科学证据表明维生素C可以预防新冠病毒感染，建议接种疫苗并保持良好卫生习惯。",
        )
        eval_response = json.dumps({
            "satisfied": True,
            "reason": "回答直接针对问题，包含实质医学信息",
            "replan_action": "",
        })
        with patch("app.agents.planner.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=eval_response))
            )
            result = PlannerAgent(state)

        assert result["planner_eval"]["satisfied"] is True

    def test_triggers_replan_when_answer_is_insufficient(self, base_state):
        state = self._make_eval_state(base_state, "请咨询医生。")
        eval_response = json.dumps({
            "satisfied": False,
            "reason": "回答内容过于简短，缺乏实质医学信息",
            "replan_action": "请提供关于维生素C与新冠预防的具体科学证据",
        })
        with patch("app.agents.planner.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=eval_response))
            )
            result = PlannerAgent(state)

        assert result["planner_eval"]["satisfied"] is False
        assert result["planner_eval"]["replan_count"] == 1
        assert result["replan_instruction"] != ""
        # replan 时 ResearchAgent 相关状态被重置
        assert result["generation"] == ""
        assert result["documents"] == []
        assert result["rag_think_log"] == []

    def test_force_pass_when_replan_limit_reached(self, base_state):
        """已达 MAX_REPLAN=1 次，强制放行。"""
        state = self._make_eval_state(base_state, "部分回答内容", replan_count=1)
        with patch("app.agents.planner.get_llm") as mock_get:
            mock_get.return_value = MagicMock()
            result = PlannerAgent(state)

        assert result["planner_eval"]["satisfied"] is True
        assert "上限" in result["planner_eval"]["reason"]

    def test_heuristic_eval_short_answer_fails(self, base_state):
        """LLM 不可用时，过短答案应启发式判为不满足。"""
        state = self._make_eval_state(base_state, "好的")
        with patch("app.agents.planner.get_llm", return_value=None):
            result = PlannerAgent(state)

        assert result["planner_eval"]["satisfied"] is False

    def test_heuristic_eval_evasion_answer_fails(self, base_state):
        """只含免责套话的答案应启发式判为不满足。"""
        state = self._make_eval_state(
            base_state, "暂时无法给出具体建议，无法分析，请咨询医生就医。"
        )
        with patch("app.agents.planner.get_llm", return_value=None):
            result = PlannerAgent(state)

        assert result["planner_eval"]["satisfied"] is False


# ══════════════════════════════════════════════════════════════════════════════
# QueryRewriterAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryRewriterAgent:

    def test_rewrites_ambiguous_question_with_context(self, base_state):
        """'可以预防吗' 应结合历史补全为完整问题。"""
        base_state["question"] = "可以预防吗"
        base_state["conversation_history"] = [
            {"role": "user", "content": "维生素C有什么用"},
            {"role": "assistant", "content": "维生素C有助于免疫功能维持"},
        ]
        llm_response = json.dumps({
            "intent": "general_health",
            "rewritten_question": "维生素C可以预防新冠病毒感染吗？",
            "expanded_queries": ["维生素C与新冠病毒预防", "维生素C对免疫系统的作用"],
            "thinking": ["理解用户意图：健康咨询", "识别关键概念：维生素C、预防"],
        })
        with patch("app.agents.query_rewriter.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=llm_response))
            )
            result = QueryRewriterAgent(base_state)

        assert result["query_intent"] == "general_health"
        assert "维生素C" in result["question"]
        assert len(result["expanded_queries"]) >= 2
        assert len(result["thinking_steps"]) >= 2
        assert result["original_question"] == "可以预防吗"

    def test_detects_symptom_inquiry_intent(self, base_state):
        base_state["question"] = "我发烧38.5度，还头疼"
        llm_response = json.dumps({
            "intent": "symptom_inquiry",
            "rewritten_question": "发烧38.5度伴头痛，可能是什么原因？",
            "expanded_queries": ["发烧头痛的常见原因", "高烧头痛应对措施"],
            "thinking": ["意图：症状咨询", "关键词：发烧、头痛"],
        })
        with patch("app.agents.query_rewriter.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=llm_response))
            )
            result = QueryRewriterAgent(base_state)

        assert result["query_intent"] == "symptom_inquiry"

    def test_falls_back_to_heuristic_when_llm_unavailable(self, base_state):
        base_state["question"] = "我头痛怎么办"
        with patch("app.agents.query_rewriter.get_llm", return_value=None):
            result = QueryRewriterAgent(base_state)

        # 降级时保持原始问题不变
        assert result["question"] == "我头痛怎么办"
        assert result["original_question"] == "我头痛怎么办"
        assert len(result["expanded_queries"]) == 2
        assert result["query_intent"] == "symptom_inquiry"

    def test_falls_back_to_heuristic_when_llm_returns_bad_json(self, base_state):
        base_state["question"] = "咳嗽很久了"
        with patch("app.agents.query_rewriter.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content="这不是JSON格式"))
            )
            result = QueryRewriterAgent(base_state)

        assert result["query_intent"] in {
            "symptom_inquiry", "medication_inquiry", "report_interpretation",
            "disease_inquiry", "treatment_inquiry", "general_health", "chitchat",
        }

    def test_heuristic_detects_medication_intent(self, base_state):
        base_state["question"] = "布洛芬可以空腹吃吗"
        with patch("app.agents.query_rewriter.get_llm", return_value=None):
            result = QueryRewriterAgent(base_state)
        assert result["query_intent"] == "medication_inquiry"

    def test_heuristic_detects_report_intent(self, base_state):
        base_state["question"] = "我的血常规报告偏高怎么回事"
        with patch("app.agents.query_rewriter.get_llm", return_value=None):
            result = QueryRewriterAgent(base_state)
        assert result["query_intent"] == "report_interpretation"

    def test_original_question_preserved(self, base_state):
        """original_question 始终保存原始输入，不被覆盖。"""
        original = "它可以治好吗"
        base_state["question"] = original
        llm_response = json.dumps({
            "intent": "treatment_inquiry",
            "rewritten_question": "糖尿病可以被根治吗？",
            "expanded_queries": ["糖尿病治疗方案"],
            "thinking": ["意图：治疗咨询"],
        })
        with patch("app.agents.query_rewriter.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=llm_response))
            )
            result = QueryRewriterAgent(base_state)

        assert result["original_question"] == original


# ══════════════════════════════════════════════════════════════════════════════
# MemoryAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryAgent:

    def test_loads_long_term_context_into_state(self, base_state):
        """有历史档案时，long_term_context 应被正确写入 state。"""
        base_state["session_id"] = "sess-memory-test"
        expected_context = "【患者档案】\n· 年龄：45岁\n· 过敏史：青霉素过敏"

        with patch("app.agents.memory.long_term_memory") as mock_lt:
            mock_lt.format_for_prompt.return_value = expected_context
            mock_lt.extract_and_save.return_value = {}
            result = MemoryAgent(base_state)

        assert result["long_term_context"] == expected_context

    def test_sets_empty_context_when_no_history(self, base_state):
        base_state["session_id"] = "sess-new-user"
        with patch("app.agents.memory.long_term_memory") as mock_lt:
            mock_lt.format_for_prompt.return_value = ""
            mock_lt.extract_and_save.return_value = {}
            result = MemoryAgent(base_state)

        assert result["long_term_context"] == ""

    def test_gracefully_handles_long_term_load_failure(self, base_state):
        """长期记忆加载失败时不应抛异常，应降级为空字符串。"""
        base_state["session_id"] = "sess-lt-fail"
        with patch("app.agents.memory.long_term_memory") as mock_lt:
            mock_lt.format_for_prompt.side_effect = RuntimeError("DB connection failed")
            mock_lt.extract_and_save.return_value = {}
            result = MemoryAgent(base_state)

        assert result["long_term_context"] == ""

    def test_triggers_extraction_from_last_qa_turn(self, base_state):
        """有上一轮完整 Q&A 时，应调用 extract_and_save。"""
        base_state["session_id"] = "sess-extract-test"
        base_state["conversation_history"] = [
            {"role": "user", "content": "我有高血压"},
            {"role": "assistant", "content": "高血压患者需要定期监测血压，注意低盐饮食。"},
        ]
        with patch("app.agents.memory.long_term_memory") as mock_lt:
            mock_lt.format_for_prompt.return_value = ""
            mock_lt.extract_and_save.return_value = {"conditions": "高血压"}
            result = MemoryAgent(base_state)

        mock_lt.extract_and_save.assert_called_once()
        call_kwargs = mock_lt.extract_and_save.call_args
        assert call_kwargs.kwargs.get("session_id") == "sess-extract-test" or \
               call_kwargs.args[0] == "sess-extract-test"

    def test_skips_extraction_when_no_complete_turn(self, base_state):
        """只有 user 消息（无 assistant 回复）时，不应触发提取。"""
        base_state["session_id"] = "sess-no-extract"
        base_state["conversation_history"] = [
            {"role": "user", "content": "你好"},
        ]
        with patch("app.agents.memory.long_term_memory") as mock_lt:
            mock_lt.format_for_prompt.return_value = ""
            mock_lt.extract_and_save.return_value = {}
            MemoryAgent(base_state)

        mock_lt.extract_and_save.assert_not_called()

    def test_short_term_compression_triggered_over_threshold(self, base_state):
        """历史超过 20 条时应触发短期记忆压缩。"""
        base_state["session_id"] = "sess-compress"
        # 构造 22 条历史（超过 COMPRESS_THRESHOLD=20）
        history = []
        for i in range(11):
            history.append({"role": "user", "content": f"问题{i}"})
            history.append({"role": "assistant", "content": f"回答{i}"})
        base_state["conversation_history"] = history

        with patch("app.agents.memory.long_term_memory") as mock_lt, \
             patch("app.agents.memory.compress_history") as mock_compress:
            mock_lt.format_for_prompt.return_value = ""
            mock_lt.extract_and_save.return_value = {}
            mock_compress.return_value = history[-20:]  # 压缩结果
            MemoryAgent(base_state)

        mock_compress.assert_called_once()

    def test_empty_session_id_skips_long_term(self, base_state):
        """session_id 为空时，跳过所有长期记忆操作。"""
        base_state["session_id"] = ""
        with patch("app.agents.memory.long_term_memory") as mock_lt:
            MemoryAgent(base_state)
        mock_lt.format_for_prompt.assert_not_called()
        mock_lt.extract_and_save.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# CriticAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestCriticAgent:

    def _good_critic_response(self) -> str:
        return json.dumps({
            "passed": True,
            "hallucination_detected": False,
            "fact_checks": [
                {"claim": "维生素C无法预防新冠", "status": "verified", "note": "文档支持"},
                {"claim": "疫苗接种是有效预防手段", "status": "verified", "note": "文档支持"},
            ],
            "revised_answer": "目前没有科学证据表明维生素C可以预防新冠，疫苗接种才是有效手段。",
            "feedback": "",
        })

    def _hallucination_response(self) -> str:
        return json.dumps({
            "passed": False,
            "hallucination_detected": True,
            "fact_checks": [
                {"claim": "维生素C每日服用1000mg可预防新冠", "status": "contradicted",
                 "note": "文档中未提及具体剂量，属于捏造"},
            ],
            "revised_answer": "",
            "feedback": "答案包含无来源的具体剂量信息，请删除该断言并重新生成。",
        })

    def test_passes_and_updates_generation_with_revised_answer(self, state_with_docs):
        with patch("app.agents.critic.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=self._good_critic_response()))
            )
            result = CriticAgent(state_with_docs)

        assert result["critic_result"]["passed"] is True
        assert result["critic_result"]["hallucination_detected"] is False
        # generation 被更新为润色后答案
        assert "疫苗接种" in result["generation"]
        assert result["metrics"]["critic_pass"] is True

    def test_fails_and_writes_replan_instruction_on_hallucination(self, state_with_docs):
        with patch("app.agents.critic.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=self._hallucination_response()))
            )
            result = CriticAgent(state_with_docs)

        assert result["critic_result"]["passed"] is False
        assert result["critic_result"]["hallucination_detected"] is True
        assert result["replan_instruction"] != ""
        assert "CriticAgent 反馈" in result["replan_instruction"]
        assert result["critic_retry_count"] == 1
        assert result["metrics"]["critic_pass"] is False

    def test_force_pass_when_retry_limit_reached(self, state_with_docs):
        """已达 CRITIC_MAX_RETRY=1 次，应强制通过并追加免责声明。"""
        state_with_docs["critic_retry_count"] = 1  # 已用完重试次数
        with patch("app.agents.critic.get_llm") as mock_get:
            mock_get.return_value = MagicMock()
            result = CriticAgent(state_with_docs)

        assert result["critic_result"]["passed"] is True
        assert "免责声明" in result["generation"] or "专业医生" in result["generation"]

    def test_fails_when_answer_is_empty(self, base_state):
        base_state["generation"] = ""
        base_state["original_question"] = "头痛怎么办"
        with patch("app.agents.critic.get_llm") as mock_get:
            mock_get.return_value = MagicMock()
            result = CriticAgent(base_state)

        assert result["critic_result"]["passed"] is False

    def test_fails_when_answer_too_short(self, base_state):
        base_state["generation"] = "好的。"  # 小于 MIN_ANSWER_LENGTH=20
        base_state["original_question"] = "头痛怎么办"
        with patch("app.agents.critic.get_llm") as mock_get:
            mock_get.return_value = MagicMock()
            result = CriticAgent(base_state)

        assert result["critic_result"]["passed"] is False

    def test_heuristic_fallback_when_llm_unavailable(self, state_with_docs):
        """LLM 不可用时应降级到启发式核查，不抛异常。"""
        with patch("app.agents.critic.get_llm", return_value=None):
            result = CriticAgent(state_with_docs)

        # 启发式核查：答案有实质内容且无风险模式，应通过
        assert result["critic_result"]["passed"] is True
        assert "专业医生" in result["generation"] or len(result["generation"]) > 20

    def test_heuristic_detects_specific_dosage_without_docs(self, base_state):
        """无文档来源时，答案中出现具体剂量应被启发式标记为风险。"""
        base_state["generation"] = (
            "维生素C每日服用1000mg可以显著提升免疫力，"
            "建议按时服用以预防感冒和病毒感染。"
        )
        base_state["original_question"] = "维生素C怎么吃"
        base_state["documents"] = []
        with patch("app.agents.critic.get_llm", return_value=None):
            result = CriticAgent(base_state)

        assert result["critic_result"]["passed"] is False
        assert result["critic_result"]["hallucination_detected"] is True

    def test_contradicted_fact_causes_failure_regardless_of_passed_field(self, state_with_docs):
        """即使 LLM 返回 passed=True，但 fact_checks 含 contradicted，应强制为 False。"""
        mixed_response = json.dumps({
            "passed": True,  # LLM 误判为通过
            "hallucination_detected": False,
            "fact_checks": [
                {"claim": "维生素C可以直接杀灭新冠病毒", "status": "contradicted",
                 "note": "与文档矛盾"},
            ],
            "revised_answer": "修正后的回答",
            "feedback": "",
        })
        with patch("app.agents.critic.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=mixed_response))
            )
            result = CriticAgent(state_with_docs)

        # contradicted 存在时，passed 应被强制覆盖为 False
        assert result["critic_result"]["passed"] is False

    def test_tool_trace_includes_critic(self, state_with_docs):
        with patch("app.agents.critic.get_llm") as mock_get:
            mock_get.return_value = MagicMock(
                invoke=MagicMock(return_value=MagicMock(content=self._good_critic_response()))
            )
            result = CriticAgent(state_with_docs)

        assert "critic" in result["tool_trace"]


# ══════════════════════════════════════════════════════════════════════════════
# ResearchAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestResearchAgentLlmDirectPath:
    """Planner 路由到 llm_agent 时，直接走 LLM 不做 RAG 的路径。"""

    def test_llm_direct_skips_rag_and_sets_generation(self, base_state):
        base_state["question"] = "你好，请问今天几号"
        base_state["current_tool"] = "llm_agent"

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research.rerank_documents", return_value=[]):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "今天是2026年4月23日，有什么可以帮您的吗？"
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        assert result["research_strategy"] == "llm_direct"
        assert result["llm_success"] is True
        assert len(result["generation"]) > 10
        assert result["rag_attempted"] is False

    def test_llm_direct_overridden_when_replan_instruction_present(self, base_state):
        """replan_instruction 非空时，即使路由是 llm_agent 也应走 ReAct 循环。"""
        base_state["question"] = "头痛怎么办"
        base_state["current_tool"] = "llm_agent"
        base_state["replan_instruction"] = "请重新检索并补充头痛病因的医学证据"

        accept_response = json.dumps({
            "relevance": 7.0, "coverage": 7.0, "medical_depth": 7.0,
            "action": "accept", "param": "", "reason": "直接接受",
        })
        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research._rag_search", return_value=[]), \
             patch("app.agents.research.rerank_documents", return_value=[]):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = accept_response
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        # 不应走 llm_direct 直接返回，应进入 ReAct 循环
        assert result["research_strategy"] != "llm_direct"


class TestResearchAgentReActLoop:
    """ReAct 循环的核心分支测试。"""

    def _make_think_response(self, action: str, param: str = "", reason: str = "test") -> str:
        return json.dumps({
            "relevance": 7.0, "coverage": 7.0, "medical_depth": 7.0,
            "action": action, "param": param, "reason": reason,
        })

    def test_rag_search_action_adds_docs_to_state(self, base_state):
        base_state["question"] = "高血压有什么症状"
        base_state["current_tool"] = "retriever"

        from langchain_core.documents import Document
        mock_docs = [Document(
            page_content="高血压常见症状包括头痛、头晕、耳鸣，严重时可导致心脑血管并发症。",
            metadata={"rerank_score": 0.75},
        )]

        # iter0: rag_search → iter1: accept
        responses = [
            self._make_think_response("rag_search", "高血压症状"),
            self._make_think_response("accept"),
        ]
        call_count = [0]

        def side_effect(prompt, **kwargs):
            r = MagicMock()
            r.content = responses[min(call_count[0], len(responses) - 1)]
            call_count[0] += 1
            return r

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research._rag_search", return_value=mock_docs), \
             patch("app.agents.research.rerank_documents", return_value=mock_docs):
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = side_effect
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        assert result["rag_attempted"] is True
        assert result["rag_success"] is True
        assert len(result["documents"]) > 0

    def test_early_accept_when_rerank_score_above_threshold(self, base_state):
        """rerank_score >= 0.80 时应直接 early-exit，跳过 LLM THINK 调用。"""
        base_state["question"] = "发烧38度怎么办"
        base_state["current_tool"] = "retriever"

        from langchain_core.documents import Document
        high_score_docs = [Document(
            page_content="发烧38度属于低烧，可多饮水休息，若持续或超过39度请就医。",
            metadata={"rerank_score": 0.92},  # 高于 EARLY_ACCEPT_THRESHOLD=0.80
        )]

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research._rag_search", return_value=high_score_docs), \
             patch("app.agents.research.rerank_documents", return_value=high_score_docs):
            mock_llm = MagicMock()
            mock_get.return_value = mock_llm
            # 预填文档，让第一次 _think 就触发 early-exit
            base_state["documents"] = high_score_docs
            result = ResearchAgent(base_state)

        # early-exit 时 LLM THINK 调用次数应极少（仅生成答案时调用）
        # 关键断言：think_log 第一条 action 是 accept（early-exit）
        if result["rag_think_log"]:
            first_entry = result["rag_think_log"][0]
            assert first_entry["action"] == "accept"

    def test_force_accept_after_two_consecutive_expand_queries(self, base_state):
        """连续两轮 expand_query 后，第三轮应强制 accept，不再继续扩展。"""
        base_state["question"] = "某种罕见病的治疗方案"
        base_state["current_tool"] = "retriever"

        from langchain_core.documents import Document
        some_docs = [Document(
            page_content="相关医学资料内容",
            metadata={"rerank_score": 0.3},
        )]

        responses = [
            self._make_think_response("expand_query", "罕见病治疗"),
            self._make_think_response("expand_query", "罕见病最新研究"),
            self._make_think_response("accept"),  # 实际不会被调用，应被强制覆盖
        ]
        call_count = [0]

        def side_effect(prompt, **kwargs):
            r = MagicMock()
            r.content = responses[min(call_count[0], len(responses) - 1)]
            call_count[0] += 1
            return r

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research._rag_search", return_value=some_docs), \
             patch("app.agents.research.rerank_documents", return_value=some_docs):
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = side_effect
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        # 连续 expand_query 后应强制 accept，不超过 MAX_ITER=3 轮
        assert result["rag_iterations"] <= 3
        actions = [e["action"] for e in result["rag_think_log"]]
        assert "accept" in actions

    def test_tool_query_action_calls_tool_and_adds_doc(self, base_state):
        base_state["question"] = "布洛芬的用法用量"
        base_state["current_tool"] = "tool_agent"

        responses = [
            self._make_think_response("tool_query", "search_drug|布洛芬"),
            self._make_think_response("accept"),
        ]
        call_count = [0]

        def side_effect(prompt, **kwargs):
            r = MagicMock()
            r.content = responses[min(call_count[0], len(responses) - 1)]
            call_count[0] += 1
            return r

        mock_tool_result = "布洛芬：每次200-400mg，每日3次，饭后服用，不超过每日1200mg。"

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research._run_tool", return_value=mock_tool_result), \
             patch("app.agents.research.rerank_documents", return_value=[]):
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = side_effect
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        assert result["tool_agent_success"] is True
        # 工具结果应被包装为 Document 写入 documents
        assert any("[工具:search_drug]" in d.page_content for d in result["documents"])

    def test_wikipedia_fallback_langchain_when_mcp_unavailable(self, base_state):
        base_state["question"] = "流感病毒的传播机制"
        base_state["current_tool"] = "retriever"

        responses = [
            self._make_think_response("wikipedia", "流感病毒传播"),
            self._make_think_response("accept"),
        ]
        call_count = [0]

        def side_effect(prompt, **kwargs):
            r = MagicMock()
            r.content = responses[min(call_count[0], len(responses) - 1)]
            call_count[0] += 1
            return r

        wiki_content = "流感病毒主要通过飞沫传播，也可通过接触传播。" * 10  # 足够长

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research.MCP_AVAILABLE", False), \
             patch("app.agents.research.rerank_documents", return_value=[]), \
             patch("app.tools.wikipedia_search.get_wikipedia_wrapper") as mock_wiki_factory:
            mock_wiki = MagicMock()
            mock_wiki.run.return_value = wiki_content
            mock_wiki_factory.return_value = mock_wiki
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = side_effect
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        assert result["wiki_attempted"] is True
        assert result["wiki_success"] is True

    def test_tavily_fallback_langchain_when_mcp_unavailable(self, base_state):
        base_state["question"] = "2026年最新新冠治疗方案"
        base_state["current_tool"] = "retriever"

        responses = [
            self._make_think_response("tavily", "新冠治疗方案2026"),
            self._make_think_response("accept"),
        ]
        call_count = [0]

        def side_effect(prompt, **kwargs):
            r = MagicMock()
            r.content = responses[min(call_count[0], len(responses) - 1)]
            call_count[0] += 1
            return r

        tavily_results = [
            {"content": "2026年新冠最新治疗方案包括抗病毒药物和支持治疗。" * 5, "url": "https://example.com", "title": "新冠治疗"},
        ]

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research.MCP_AVAILABLE", False), \
             patch("app.agents.research.rerank_documents", return_value=[]), \
             patch("app.tools.tavily_search.get_tavily_search") as mock_tavily_factory:
            mock_tavily = MagicMock()
            mock_tavily.invoke.return_value = tavily_results
            mock_tavily_factory.return_value = mock_tavily
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = side_effect
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        assert result["tavily_attempted"] is True
        assert result["tavily_success"] is True

    def test_rag_think_log_scores_populated_on_llm_decision(self, base_state):
        """LLM 正常决策时，rag_think_log 的每条记录必须含 scores.overall。"""
        base_state["question"] = "糖尿病饮食注意事项"
        base_state["current_tool"] = "retriever"

        from langchain_core.documents import Document
        docs = [Document(
            page_content="糖尿病患者应控制碳水化合物摄入，避免高糖食物。",
            metadata={"rerank_score": 0.55},
        )]
        llm_response = json.dumps({
            "relevance": 8.0, "coverage": 7.5, "medical_depth": 8.0,
            "action": "accept", "param": "", "reason": "文档质量足够",
        })

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research._rag_search", return_value=docs), \
             patch("app.agents.research.rerank_documents", return_value=docs):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = llm_response
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        for entry in result["rag_think_log"]:
            assert "scores" in entry, f"迭代记录缺少 scores 字段：{entry}"
            assert entry["scores"].get("overall") is not None

    def test_max_iter_not_exceeded(self, base_state):
        """无论 LLM 一直返回非 accept 动作，迭代次数不应超过 MAX_ITER=3。"""
        base_state["question"] = "无法找到答案的奇怪问题"
        base_state["current_tool"] = "retriever"

        # LLM 始终返回 expand_query，不 accept
        always_expand = json.dumps({
            "relevance": 3.0, "coverage": 3.0, "medical_depth": 3.0,
            "action": "expand_query", "param": "继续扩展", "reason": "持续扩展",
        })

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research._rag_search", return_value=[]), \
             patch("app.agents.research.rerank_documents", return_value=[]):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = always_expand
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        assert len(result["rag_think_log"]) <= 3

    def test_appends_to_conversation_history_after_generation(self, base_state):
        """ResearchAgent 完成后应把本轮 Q&A 写入 conversation_history。"""
        base_state["question"] = "多喝水有好处吗"
        base_state["current_tool"] = "retriever"
        base_state["conversation_history"] = []

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research._rag_search", return_value=[]), \
             patch("app.agents.research.rerank_documents", return_value=[]):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = json.dumps({
                "relevance": 6.0, "coverage": 6.0, "medical_depth": 6.0,
                "action": "accept", "param": "", "reason": "直接接受",
            })
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        roles = [t["role"] for t in result["conversation_history"]]
        assert "user" in roles
        assert "assistant" in roles

    def test_tool_trace_includes_research(self, base_state):
        base_state["question"] = "test"
        base_state["current_tool"] = "llm_agent"

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research.rerank_documents", return_value=[]):
            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "足够长的回答内容，超过十个字符的回复。"
            mock_get.return_value = mock_llm
            result = ResearchAgent(base_state)

        assert "research" in result["tool_trace"]


class TestResearchAgentWithReplanInstruction:

    def test_injects_replan_instruction_into_think_prompt(self, base_state):
        """replan_instruction 非空时，应被注入 THINK prompt，LLM 应感知到重规划指令。"""
        base_state["question"] = "维生素C能预防新冠吗"
        base_state["current_tool"] = "retriever"
        base_state["replan_instruction"] = "[CriticAgent 反馈] 答案含幻觉，请删除无来源剂量信息。"

        captured_prompts = []

        def capture_invoke(prompt, **kwargs):
            captured_prompts.append(str(prompt))
            r = MagicMock()
            r.content = json.dumps({
                "relevance": 7.0, "coverage": 7.0, "medical_depth": 7.0,
                "action": "accept", "param": "", "reason": "按指令修正后接受",
            })
            return r

        with patch("app.agents.research.get_llm") as mock_get, \
             patch("app.agents.research._rag_search", return_value=[]), \
             patch("app.agents.research.rerank_documents", return_value=[]):
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = capture_invoke
            mock_get.return_value = mock_llm
            ResearchAgent(base_state)

        # 至少有一次 LLM 调用包含了 replan_instruction 内容
        assert any("CriticAgent 反馈" in p or "Planner 重规划指令" in p
                   for p in captured_prompts)