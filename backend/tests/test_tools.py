"""工具包装器测试。"""

import os
import sys
from unittest.mock import MagicMock, patch
import app.tools.duckduckgo_search as ddg_module  
import app.tools.llm_client as llm_module  
import app.tools.tavily_search as tavily_module  
import app.tools.vector_store as vs_module  
import app.tools.wikipedia_search as wiki_module  
from app.tools.duckduckgo_search import get_duckduckgo_search  
from app.tools.llm_client import get_llm  
from app.tools.pdf_loader import process_pdf, split_documents  
from app.tools.tavily_search import get_tavily_search  
from app.tools.vector_store import get_embeddings, get_or_create_vectorstore, get_retriever  
from app.tools.wikipedia_search import get_wikipedia_wrapper  


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_get_llm_no_key():
    # 无 API Key 时返回 None
    llm_module._llm_instance = None
    with patch("app.tools.llm_client.DASHSCOPE_API_KEY", None):
        result = get_llm()
        assert result is None


def test_get_llm_with_key():
    # 有 API Key 时可正常初始化 LLM
    llm_module._llm_instance = None
    with patch("app.tools.llm_client.DASHSCOPE_API_KEY", "fake-key"):
        with patch("langchain_community.chat_models.tongyi.ChatTongyi") as mock_tongyi:
            mock_tongyi.return_value = MagicMock()
            result = get_llm()
            assert result is not None
    # 清理模块级缓存
    llm_module._llm_instance = None


def test_get_llm_handles_init_error():
    # 初始化出错时返回 None
    llm_module._llm_instance = None
    with patch("app.tools.llm_client.DASHSCOPE_API_KEY", "fake-key"):
        with patch("langchain_community.chat_models.tongyi.ChatTongyi", side_effect=RuntimeError("boom")):
            result = get_llm()
            assert result is None


def test_get_wikipedia():
    # 验证 Wikipedia 包装器单例与成功初始化
    wiki_module._wiki_wrapper = None
    with patch("langchain_community.utilities.wikipedia.WikipediaAPIWrapper") as mock_wiki:
        mock_wiki.return_value = MagicMock()
        wrapper = get_wikipedia_wrapper()
        assert wrapper is not None
        # 第二次调用应返回同一实例
        assert get_wikipedia_wrapper() == wrapper
    wiki_module._wiki_wrapper = None


def test_get_tavily_no_key():
    # 无 API Key 时返回 None
    tavily_module._tavily_search = None
    with patch("app.tools.tavily_search.TAVILY_API_KEY", None):
        result = get_tavily_search()
        assert result is None


def test_get_tavily_with_key():
    # 有 API Key 时可正常初始化 Tavily 搜索
    tavily_module._tavily_search = None
    with patch("app.tools.tavily_search.TAVILY_API_KEY", "fake-key"):
        with patch("langchain_community.tools.tavily_search.TavilySearchResults") as mock_tavily:
            mock_tavily.return_value = MagicMock()
            result = get_tavily_search()
            assert result is not None
    tavily_module._tavily_search = None


def test_pdf_loader():
    # 验证 PDF 加载与文档分割流程
    with patch("langchain_community.document_loaders.PyPDFLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_cls.return_value = mock_loader

        with patch("app.tools.pdf_loader.split_documents") as mock_split:
            mock_split.return_value = ["chunk1"]
            result = process_pdf("path.pdf")
            assert result == ["chunk1"]


def test_get_duckduckgo_no_import():
    # 导入失败时应返回 None 并记录警告
    ddg_module._ddg_search = None
    with patch("langchain_community.tools.DuckDuckGoSearchRun", side_effect=ImportError):
        with patch("app.tools.duckduckgo_search.logger") as mock_logger:
            result = get_duckduckgo_search()
            assert result is None
            mock_logger.warning.assert_called()


def test_get_duckduckgo_success():
    # 正常导入时返回搜索实例
    ddg_module._ddg_search = None
    with patch("langchain_community.tools.DuckDuckGoSearchRun") as mock_ddg:
        mock_ddg.return_value = MagicMock()
        result = get_duckduckgo_search()
        assert result is not None
    ddg_module._ddg_search = None


def test_vector_store_embeddings():
    # 验证嵌入模型单例
    vs_module._embeddings = None
    with patch("langchain_huggingface.embeddings.HuggingFaceEmbeddings") as mock_embeddings:
        mock_embeddings.return_value = MagicMock()
        result = get_embeddings()
        assert result is not None
    vs_module._embeddings = None


def test_vector_store_get_or_create():
    # 创建或加载向量存储（已有数据 vs 新建）
    vs_module._vectorstore = None
    vs_module._embeddings = MagicMock()

    with patch("langchain_chroma.Chroma") as mock_chroma_cls:
        mock_vs = MagicMock()
        mock_vs._collection.count.return_value = 5
        mock_chroma_cls.return_value = mock_vs
        mock_chroma_cls.from_documents.return_value = mock_vs

        # 目录已存在且有数据的情况
        with patch("os.path.exists", return_value=True):
            with patch("os.listdir", return_value=["chroma.sqlite3"]):
                result = get_or_create_vectorstore(persist_dir="fake")
                assert result is not None

        vs_module._vectorstore = None

        # 目录不存在时的新建流程
        with patch("os.path.exists", return_value=False):
            with patch("os.makedirs"):
                result = get_or_create_vectorstore(documents=[MagicMock()], persist_dir="new")
                assert result is not None

    vs_module._vectorstore = None


def test_get_retriever():
    # 获取检索器（向量库存在时）
    vs_module._vectorstore = MagicMock()
    vs_module._vectorstore.as_retriever.return_value = MagicMock()
    result = get_retriever()
    assert result is not None

    # 向量库不存在时返回 None
    vs_module._vectorstore = None
    with patch("app.tools.vector_store.get_or_create_vectorstore", return_value=None):
        assert get_retriever() is None


def test_split_documents():
    # 文档分割功能测试
    mock_doc = MagicMock()
    # page_content 必须是字符串，否则 re.sub 会抛出 TypeError
    mock_doc.page_content = "患者症状：1.发热 2.咳嗽 3.头痛。建议就医检查。"
    mock_doc.metadata = {"source": "test"}
    with patch("langchain_text_splitters.RecursiveCharacterTextSplitter") as mock_splitter_cls:
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [mock_doc]
        mock_splitter_cls.return_value = mock_splitter

        result = split_documents([mock_doc])
        assert len(result) == 1
