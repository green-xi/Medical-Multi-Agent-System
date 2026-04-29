"""
test_services.py — ChatService 单元测试
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.chat_service import ChatService


def _make_workflow_result(generation="测试回复", source="测试来源", critic_pass=True):
    """构造一个符合当前架构的完整 workflow 返回值。"""
    return {
        "generation": generation,
        "source": source,
        "llm_attempted": True,
        "llm_success": True,
        "rag_attempted": True,
        "rag_success": True,
        "wiki_attempted": False,
        "wiki_success": False,
        "tavily_attempted": False,
        "tavily_success": False,
        "tool_trace": ["memory", "query_rewriter", "planner", "research", "planner", "critic"],
        "fallback_events": [],
        "metrics": {
            "total_latency_ms": 2000.0,
            "node_latencies_ms": {"memory": 10.0, "research": 1500.0, "critic": 400.0},
            "rag_hit": True,
            "llm_used": True,
            "fallback_count": 0,
            "rerank_used": True,
            "rerank_latency_ms": 300.0,
            "replan_count": 0,
            "critic_pass": critic_pass,
            "critic_retry_count": 0,
        },
        "conversation_history": [],
        "question": "测试问题",
        "original_question": "测试问题",
        "documents": [],
        "query_intent": "general_health",
        "thinking_steps": ["分析用户意图", "识别关键词", "优化查询"],
        "rag_think_log": [
            {
                "iteration": 0,
                "action": "expand_query",
                "scores": {"relevance": 6.0, "coverage": 6.0, "medical_depth": 6.0, "overall": 6.0},
                "reasoning": "当前无文档，需扩展查询",
            },
            {
                "iteration": 1,
                "action": "accept",
                "scores": {"relevance": 8.0, "coverage": 7.0, "medical_depth": 7.5, "overall": 7.5},
                "reasoning": "文档质量足够，可以生成答案",
            },
        ],
        "expanded_queries": ["维生素C与新冠", "免疫功能增强"],
        "current_tool": "retriever",
        "confidence_score": 0.91,
        "route_decision": {
            "is_medical": True,
            "tool": "retriever",
            "confidence": 0.91,
            "reason": "医疗问题",
            "strategy": "llm_classifier",
        },
        "planner_eval": {
            "satisfied": True,
            "reason": "回答满足意图",
            "replan_action": "",
            "replan_count": 0,
        },
        "critic_result": {
            "passed": critic_pass,
            "hallucination_detected": False,
            "fact_checks": [],
            "revised_answer": generation,
            "feedback": "",
        },
    }


class TestChatServiceInit:

    def test_initial_state(self):
        service = ChatService()
        assert service.workflow_app is None
        assert len(service.conversation_states) == 0

    def test_initialize_workflow(self):
        service = ChatService()
        with patch("app.services.chat_service.create_workflow") as mock_create:
            mock_create.return_value = MagicMock()
            service.initialize_workflow()
        assert service.workflow_app is not None

    def test_initialize_workflow_idempotent(self):
        """重复调用不应重新创建 workflow。"""
        service = ChatService()
        mock_wf = MagicMock()
        service.workflow_app = mock_wf
        with patch("app.services.chat_service.create_workflow") as mock_create:
            service.initialize_workflow()
        mock_create.assert_not_called()


class TestProcessMessage:

    @pytest.mark.asyncio
    async def test_success_returns_correct_structure(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(
            return_value=_make_workflow_result()
        )
        from app.services import db_service
        with patch.object(db_service, "save_message"):
            result = await service.process_message("test-sess", "维生素C能预防新冠吗")

        assert result["success"] is True
        assert result["response"] == "测试回复"
        assert result["source"] == "测试来源"
        assert "thinking" in result
        assert "rag_think_log" in result["thinking"]
        assert len(result["thinking"]["rag_think_log"]) == 2

    @pytest.mark.asyncio
    async def test_thinking_rag_log_scores_present(self):
        """rag_think_log 中每条迭代记录应包含 scores 字段。"""
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(
            return_value=_make_workflow_result()
        )
        from app.services import db_service
        with patch.object(db_service, "save_message"):
            result = await service.process_message("test-sess", "问题")

        for iter_entry in result["thinking"]["rag_think_log"]:
            assert "scores" in iter_entry
            assert "overall" in iter_entry["scores"]

    @pytest.mark.asyncio
    async def test_raises_error_when_workflow_not_initialized(self):
        service = ChatService()
        with pytest.raises(ValueError, match="Workflow not initialized"):
            await service.process_message("test-sess", "hello")

    @pytest.mark.asyncio
    async def test_metrics_recorded_after_successful_call(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(
            return_value=_make_workflow_result()
        )
        from app.services import db_service
        with patch.object(db_service, "save_message"):
            await service.process_message("sess-metrics", "问题")

        snapshot = service.get_metrics_snapshot()
        assert snapshot["requests_total"] == 1
        assert snapshot["rag_hit_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_fallback_to_sync_invoke_when_ainvoke_not_available(self):
        service = ChatService()
        service.workflow_app = MagicMock()
        service.workflow_app.ainvoke = AsyncMock(side_effect=AttributeError)
        service.workflow_app.invoke = MagicMock(
            return_value=_make_workflow_result(generation="同步回复")
        )
        from app.services import db_service
        with patch.object(db_service, "save_message"):
            result = await service.process_message("sess-sync", "问题")

        assert result["response"] == "同步回复"


class TestSessionManagement:

    def test_clear_conversation_resets_state(self):
        service = ChatService()
        state = service._get_or_create_session_state("sess-clear")
        state["question"] = "已提问的内容"
        service._touch_session("sess-clear", state)
        service.clear_conversation("sess-clear")
        assert service.conversation_states["sess-clear"]["state"]["question"] == ""

    def test_lru_eviction_when_limit_exceeded(self):
        service = ChatService()
        with patch("app.services.chat_service.MAX_ACTIVE_SESSIONS", 2):
            for i in range(3):
                s = service._get_or_create_session_state(f"sess-{i}")
                service._touch_session(f"sess-{i}", s)
            service._evict_stale_sessions()

        assert len(service.conversation_states) <= 2

    def test_get_metrics_snapshot_structure(self):
        service = ChatService()
        snapshot = service.get_metrics_snapshot()
        required_keys = {"requests_total", "fallback_rate", "rag_hit_rate", "llm_failure_rate"}
        assert required_keys.issubset(snapshot.keys())
