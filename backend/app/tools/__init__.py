"""
MedicalAI — tools/__init__.py
统一导出所有工具模块的获取函数。
"""

from app.tools.duckduckgo_search import get_duckduckgo_search
from app.tools.llm_client import get_llm
from app.tools.pdf_loader import process_pdf
from app.tools.tavily_search import get_tavily_search
from app.tools.vector_store import get_or_create_vectorstore, get_retriever
from app.tools.wikipedia_search import get_wikipedia_wrapper

__all__ = [
    "get_llm",
    "get_retriever",
    "get_or_create_vectorstore",
    "get_tavily_search",
    "get_wikipedia_wrapper",
    "get_duckduckgo_search",
    "process_pdf",
]
