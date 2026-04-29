"""Tavily 联网搜索工具单例。"""

from app.core.config import TAVILY_API_KEY
from app.core.logging_config import logger

_tavily_search = None


def get_tavily_search():
    global _tavily_search
    if _tavily_search is None:
        if not TAVILY_API_KEY:
            logger.warning("未找到 TAVILY_API_KEY 环境变量，Tavily 搜索不可用")
            return None
        from langchain_community.tools.tavily_search import TavilySearchResults
        _tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)
        logger.info("Tavily 搜索工具初始化成功")
    return _tavily_search
