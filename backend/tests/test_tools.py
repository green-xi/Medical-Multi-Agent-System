"""Tests for tool wrappers."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.tools.duckduckgo_search as ddg_module  # noqa: E402
import app.tools.llm_client as llm_module  # noqa: E402
import app.tools.tavily_search as tavily_module  # noqa: E402
import app.tools.vector_store as vs_module  # noqa: E402
import app.tools.wikipedia_search as wiki_module  # noqa: E402
from app.tools.duckduckgo_search import get_duckduckgo_search  # noqa: E402
from app.tools.llm_client import get_llm  # noqa: E402
from app.tools.pdf_loader import process_pdf, split_documents  # noqa: E402
from app.tools.tavily_search import get_tavily_search  # noqa: E402
from app.tools.vector_store import get_embeddings, get_or_create_vectorstore, get_retriever  # noqa: E402
from app.tools.wikipedia_search import get_wikipedia_wrapper  # noqa: E402


def test_get_llm_no_key():
    llm_module._llm_instance = None
    with patch("app.tools.llm_client.DASHSCOPE_API_KEY", None):
        result = get_llm()
        assert result is None


def test_get_llm_with_key():
    llm_module._llm_instance = None
    with patch("app.tools.llm_client.DASHSCOPE_API_KEY", "fake-key"):
        with patch("langchain_community.chat_models.tongyi.ChatTongyi") as mock_tongyi:
            mock_tongyi.return_value = MagicMock()
            result = get_llm()
            assert result is not None
    llm_module._llm_instance = None


def test_get_llm_handles_init_error():
    llm_module._llm_instance = None
    with patch("app.tools.llm_client.DASHSCOPE_API_KEY", "fake-key"):
        with patch("langchain_community.chat_models.tongyi.ChatTongyi", side_effect=RuntimeError("boom")):
            result = get_llm()
            assert result is None


def test_get_wikipedia():
    wiki_module._wiki_wrapper = None
    with patch("langchain_community.utilities.wikipedia.WikipediaAPIWrapper") as mock_wiki:
        mock_wiki.return_value = MagicMock()
        wrapper = get_wikipedia_wrapper()
        assert wrapper is not None
        assert get_wikipedia_wrapper() == wrapper
    wiki_module._wiki_wrapper = None


def test_get_tavily_no_key():
    tavily_module._tavily_search = None
    with patch("app.tools.tavily_search.TAVILY_API_KEY", None):
        result = get_tavily_search()
        assert result is None


def test_get_tavily_with_key():
    tavily_module._tavily_search = None
    with patch("app.tools.tavily_search.TAVILY_API_KEY", "fake-key"):
        with patch("langchain_community.tools.tavily_search.TavilySearchResults") as mock_tavily:
            mock_tavily.return_value = MagicMock()
            result = get_tavily_search()
            assert result is not None
    tavily_module._tavily_search = None


def test_pdf_loader():
    with patch("langchain_community.document_loaders.PyPDFLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_cls.return_value = mock_loader

        with patch("app.tools.pdf_loader.split_documents") as mock_split:
            mock_split.return_value = ["chunk1"]
            result = process_pdf("path.pdf")
            assert result == ["chunk1"]


def test_get_duckduckgo_no_import():
    ddg_module._ddg_search = None
    with patch("langchain_community.tools.DuckDuckGoSearchRun", side_effect=ImportError):
        with patch("app.tools.duckduckgo_search.logger") as mock_logger:
            result = get_duckduckgo_search()
            assert result is None
            mock_logger.warning.assert_called()


def test_get_duckduckgo_success():
    ddg_module._ddg_search = None
    with patch("langchain_community.tools.DuckDuckGoSearchRun") as mock_ddg:
        mock_ddg.return_value = MagicMock()
        result = get_duckduckgo_search()
        assert result is not None
    ddg_module._ddg_search = None


def test_vector_store_embeddings():
    vs_module._embeddings = None
    with patch("langchain_huggingface.embeddings.HuggingFaceEmbeddings") as mock_embeddings:
        mock_embeddings.return_value = MagicMock()
        result = get_embeddings()
        assert result is not None
    vs_module._embeddings = None


def test_vector_store_get_or_create():
    vs_module._vectorstore = None
    vs_module._embeddings = MagicMock()

    with patch("langchain_chroma.Chroma") as mock_chroma_cls:
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 5
        mock_chroma_cls.return_value = mock_vs
        mock_chroma_cls.from_documents.return_value = mock_vs

        with patch("os.path.exists", return_value=True):
            with patch("os.listdir", return_value=["chroma.sqlite3"]):
                result = get_or_create_vectorstore(persist_dir="fake")
                assert result is not None

        vs_module._vectorstore = None

        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                result = get_or_create_vectorstore(documents=[MagicMock()], persist_dir="new")
                assert result is not None

    vs_module._vectorstore = None


def test_get_retriever():
    vs_module._vectorstore = MagicMock()
    vs_module._vectorstore.as_retriever.return_value = MagicMock()
    result = get_retriever()
    assert result is not None

    vs_module._vectorstore = None
    with patch("app.tools.vector_store.get_or_create_vectorstore", return_value=None):
        assert get_retriever() is None


def test_split_documents():
    mock_doc = MagicMock()
    with patch("langchain_text_splitters.RecursiveCharacterTextSplitter") as mock_splitter_cls:
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [mock_doc]
        mock_splitter_cls.from_tiktoken_encoder.return_value = mock_splitter

        result = split_documents([mock_doc])
        assert len(result) == 1
