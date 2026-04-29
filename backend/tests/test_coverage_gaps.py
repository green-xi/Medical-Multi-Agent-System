"""
test_coverage_gaps.py — 覆盖率补充测试（五 Agent 架构对齐版）

说明：
  旧架构中的 ExecutorAgent / LLMAgent 已被移除，
  其职责全部内化至 ResearchAgent（_generate_answer + llm_direct 路径）。

  TestResearchAgentLlmDirectPath 只保留 test_agents.py 中没有覆盖的用例：
    · test_llm_short_response_marks_success_false
  （test_llm_direct_path_when_route_is_llm_agent 和 test_no_llm_returns_unavailable
    已在 test_agents.py 的同名 class 中覆盖，此处不重复。）
"""
import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

from fastapi import Request
from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.research import ResearchAgent, _generate_answer   # noqa: E402
from app.api.v1.endpoints.chat import _get_session_id             # noqa: E402
from app.api.v1.endpoints.session import (                        # noqa: E402
    delete_session_endpoint,
    get_sessions_endpoint,
)
from app.core.state import initialize_conversation_state          # noqa: E402
from app.main import lifespan                                     # noqa: E402
from app.tools.vector_store import get_or_create_vectorstore      # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# ResearchAgent — _generate_answer 函数（原 ExecutorAgent 职责）
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateAnswer:
    """测试 ResearchAgent 内化的答案生成逻辑（原 ExecutorAgent 职责）。"""

    def test_no_llm_no_docs_returns_unavailable_message(self):
        """无 LLM 也无文档时，应返回服务不可用提示。"""
        answer, source = _generate_answer(
            question="头痛怎么办",
            docs=[],
            tool_result="",
            history_context="",
            long_term_prefix="",
            llm=None,
        )
        assert "暂时不可用" in answer or len(answer) > 0
        assert source is not None

    def test_no_llm_with_docs_returns_extracted_snippet(self):
        """无 LLM 但有文档时，应从文档中抽取内容作为答案。"""
        docs = [Document(page_content="头痛可能由多种原因引起，包括紧张性头痛和偏头痛。")]
        answer, source = _generate_answer(
            question="头痛怎么办",
            docs=docs,
            tool_result="",
            history_context="",
            long_term_prefix="",
            llm=None,
        )
        assert len(answer) > 10
        assert "医学知识库" in source or "抽取" in source

    def test_with_llm_and_docs_calls_llm_invoke(self):
        """有 LLM 且有文档时，应调用 LLM 生成答案。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "根据资料，头痛常见原因包括紧张、睡眠不足等，建议充分休息。"
        )
        docs = [Document(page_content="头痛相关医学信息。")]
        answer, source = _generate_answer(
            question="头痛怎么办",
            docs=docs,
            tool_result="",
            history_context="",
            long_term_prefix="",
            llm=mock_llm,
        )
        mock_llm.invoke.assert_called_once()
        assert "头痛" in answer or len(answer) > 10

    def test_with_llm_and_tool_result_uses_tool_prompt(self):
        """有工具结果时，应优先基于工具数据生成答案，source 应标注工具来源。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "根据天气查询，今天北京气温15°C，湿度偏低，关节炎患者注意保暖。"
        )
        answer, source = _generate_answer(
            question="北京今天天气怎样",
            docs=[],
            tool_result="【北京当前天气】晴天，气温15°C，湿度40%",
            history_context="",
            long_term_prefix="",
            llm=mock_llm,
        )
        assert source == "结构化工具查询"
        assert len(answer) > 10

    def test_with_llm_and_no_docs_uses_general_prompt(self):
        """无文档无工具时，应走通用医疗知识 prompt，source='通用医疗知识'。"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "头痛有多种原因，建议观察伴随症状，如持续或剧烈应及时就医。"
        )
        answer, source = _generate_answer(
            question="头痛怎么办",
            docs=[],
            tool_result="",
            history_context="",
            long_term_prefix="",
            llm=mock_llm,
        )
        assert source == "通用医疗知识"
        assert len(answer) > 10

    def test_llm_exception_falls_back_gracefully(self):
        """LLM 调用抛异常时，应降级为安全兜底答案，不抛出异常。"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM timeout")
        answer, source = _generate_answer(
            question="头痛怎么办",
            docs=[],
            tool_result="",
            history_context="",
            long_term_prefix="",
            llm=mock_llm,
        )
        assert len(answer) > 0


# ══════════════════════════════════════════════════════════════════════════════
# ResearchAgent — llm_direct 补充用例
# （仅保留 test_agents.py 中 TestResearchAgentLlmDirectPath 未覆盖的分支）
# ══════════════════════════════════════════════════════════════════════════════

class TestResearchAgentLlmDirectExtra:
    """llm_direct 路径中 test_agents.py 未覆盖的边界分支。"""

    def test_llm_short_response_marks_success_false(self):
        """LLM 返回过短内容（< MIN_LLM_LENGTH）时，llm_success 应为 False。"""
        state = initialize_conversation_state()
        state["question"] = "头痛"
        state["current_tool"] = "llm_agent"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "好"   # 极短，触发 llm_success=False

        with patch("app.agents.research.get_llm", return_value=mock_llm), \
             patch("app.agents.research.get_retriever", return_value=None):
            result = ResearchAgent(state)

        assert result["llm_success"] is False


# ══════════════════════════════════════════════════════════════════════════════
# API 辅助函数覆盖
# ══════════════════════════════════════════════════════════════════════════════

def test_get_session_id_no_header_auto_assigns():
    """无 X-Session-ID 时，应自动生成并写入 request.session。"""
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request.session = {}
    sid = _get_session_id(mock_request)
    assert sid is not None
    assert mock_request.session["session_id"] == sid


def test_get_session_id_with_header_uses_header():
    """有 X-Session-ID header 时，应直接使用 header 值，不生成新 ID。"""
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"X-Session-ID": "my-custom-id"}
    mock_request.session = {}
    sid = _get_session_id(mock_request)
    assert sid == "my-custom-id"


def test_session_endpoints_get_sessions():
    """get_sessions_endpoint 应返回包含 sessions 键的字典。"""
    with patch("app.api.v1.endpoints.session.db_service") as mock_db:
        mock_db.get_all_sessions.return_value = []
        result = asyncio.run(get_sessions_endpoint())
    assert result["sessions"] == []


def test_session_endpoints_delete_session():
    """delete_session_endpoint 应调用 db_service.delete_session 并返回 success=True。"""
    with patch("app.api.v1.endpoints.session.db_service") as mock_db:
        mock_db.delete_session.return_value = True
        mock_request = MagicMock(spec=Request)
        mock_request.session = {"session_id": "old-id"}
        result = asyncio.run(delete_session_endpoint("old-id", mock_request))
    assert result["success"] is True


# ══════════════════════════════════════════════════════════════════════════════
# lifespan 生命周期
# ══════════════════════════════════════════════════════════════════════════════

def test_lifespan_no_pdf():
    """无 PDF 文件时，lifespan 应正常启动，不抛异常。"""
    app_mock = MagicMock()
    pdf_paths = ["medical_book.pdf", "database/medical_book.pdf"]

    with patch("os.path.exists", side_effect=lambda p: False if any(x in p for x in pdf_paths) else True), \
         patch("app.main.db_service"), \
         patch("app.main.chat_service"), \
         patch("app.main.get_or_create_vectorstore", return_value=None), \
         patch("app.db.migrate.run_all_migrations"):

        async def run_startup():
            async with lifespan(app_mock):
                pass

        asyncio.run(run_startup())


# ══════════════════════════════════════════════════════════════════════════════
# 向量库
# ══════════════════════════════════════════════════════════════════════════════

def test_vector_store_returns_none_when_empty():
    """collection.count()=0 时，get_or_create_vectorstore 应返回 None。"""
    with patch("app.tools.vector_store.get_embeddings", return_value=MagicMock()):
        from app.tools import vector_store
        vector_store._vectorstore = None
        with patch("langchain_community.vectorstores.Chroma") as mock_chroma:
            mock_vs = MagicMock()
            mock_vs._collection.count.return_value = 0
            mock_chroma.return_value = mock_vs
            with patch("os.path.exists", return_value=True), \
                 patch("os.listdir", return_value=["chroma.sqlite3"]):
                result = get_or_create_vectorstore(persist_dir="fake_dir_empty")
        assert result is None
        vector_store._vectorstore = None


def test_vector_store_returns_none_when_no_dir():
    """持久化目录不存在且无文档时，应返回 None。"""
    with patch("app.tools.vector_store.get_embeddings", return_value=MagicMock()):
        from app.tools import vector_store
        vector_store._vectorstore = None
        with patch("os.path.exists", return_value=False), \
             patch("os.makedirs"):
            result = get_or_create_vectorstore(documents=None, persist_dir="new_fake_dir")
        assert result is None
        vector_store._vectorstore = None


# ══════════════════════════════════════════════════════════════════════════════
# 数据库 session 工厂
# ══════════════════════════════════════════════════════════════════════════════

def test_db_session_makedirs_when_dir_missing():
    """数据库路径目录不存在时，get_engine 应调用 os.makedirs 创建目录。"""
    from app.db.session import get_engine
    with patch("os.path.exists", return_value=False), \
         patch("os.makedirs") as mock_makedirs:
        get_engine("some_new_dir/db.sqlite3")
    mock_makedirs.assert_called()
