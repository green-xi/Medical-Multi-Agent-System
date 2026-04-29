"""
conftest.py — MedicalAI 测试基础配置（五 Agent 精简架构对齐版）

工作流：memory → query_rewriter → planner → research → planner(评估) → critic → END
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.core.state import initialize_conversation_state  # noqa: E402


# ── 通用 state fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    """干净的初始 AgentState，每个测试独立隔离。"""
    return initialize_conversation_state(session_id="test-session-001")


@pytest.fixture
def state_with_docs(base_state):
    """预填检索文档 + 生成答案的 state，用于 Critic 测试。"""
    from langchain_core.documents import Document
    base_state["question"] = "维生素C可以预防新冠吗？"
    base_state["original_question"] = "维生素C可以预防新冠吗？"
    base_state["generation"] = (
        "目前没有科学证据表明维生素C可以预防新冠病毒感染。"
        "虽然维生素C有助于维持正常免疫功能，但它不能阻断病毒通过ACE2受体进入人体细胞的机制。"
        "预防感染最有效的方式仍是接种疫苗、佩戴口罩、保持手卫生和社交距离。"
    )
    base_state["documents"] = [
        Document(
            page_content=(
                "维生素C是一种水溶性抗氧化剂，有助于维持正常免疫系统功能，"
                "但目前无证据表明其能预防新冠感染。"
            ),
            metadata={"source": "医学知识库", "rerank_score": 0.566},
        ),
        Document(
            page_content=(
                "新冠病毒通过与ACE2受体结合进入宿主细胞，"
                "疫苗接种是目前最有效的预防手段。"
            ),
            metadata={"source": "医学知识库", "rerank_score": 0.421},
        ),
    ]
    return base_state


# ── FastAPI TestClient fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="function")
def test_client():
    """FastAPI 测试客户端，外部依赖全部 mock，返回 (client, mock_workflow)。"""
    from app.main import app
    from app.services import chat_service, db_service

    mock_workflow = MagicMock()
    mock_workflow.ainvoke = AsyncMock(return_value={
        "generation": "Test AI response",
        "source": "Test Source",
        "llm_success": True,
        "rag_attempted": False,
        "rag_success": False,
        "llm_attempted": True,
        "wiki_attempted": False,
        "wiki_success": False,
        "tavily_attempted": False,
        "tavily_success": False,
        "tool_trace": ["memory", "query_rewriter", "planner", "research", "critic"],
        "fallback_events": [],
        "metrics": {
            "total_latency_ms": 1000.0,
            "node_latencies_ms": {},
            "rag_hit": False,
            "llm_used": True,
            "fallback_count": 0,
            "rerank_used": False,
            "rerank_latency_ms": 0.0,
            "replan_count": 0,
            "critic_pass": True,
            "critic_retry_count": 0,
        },
        "conversation_history": [],
        "question": "test",
        "original_question": "test",
        "documents": [],
        "query_intent": "general_health",
        "thinking_steps": [],
        "rag_think_log": [],
        "expanded_queries": [],
        "current_tool": "retriever",
        "confidence_score": 0.9,
        "route_decision": {
            "is_medical": True,
            "tool": "retriever",
            "confidence": 0.9,
            "reason": "test",
            "strategy": "llm_classifier",
        },
        "planner_eval": {
            "satisfied": True,
            "reason": "初始规划完成",
            "replan_action": "",
            "replan_count": 0,
        },
        "critic_result": {
            "passed": True,
            "hallucination_detected": False,
            "fact_checks": [],
            "revised_answer": "Test AI response",
            "feedback": "",
        },
    })

    with patch.object(db_service, "init_db"), \
         patch.object(chat_service, "initialize_workflow"), \
         patch.object(chat_service, "workflow_app", mock_workflow), \
         patch.object(db_service, "save_message"), \
         patch("app.main.process_pdf"), \
         patch("app.main.get_or_create_vectorstore", return_value=MagicMock()):
        with TestClient(app) as client:
            yield client, mock_workflow
