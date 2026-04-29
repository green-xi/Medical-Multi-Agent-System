"""DuckDuckGo 搜索（免 API Key 的 Tavily 备选，暂未启用）。"""

from app.core.logging_config import logger

_ddg_search = None


def get_duckduckgo_search(max_results: int = 3):
    """
    返回缓存的 DuckDuckGoSearchRun 实例。
    未安装依赖时返回 None 并打印警告。
    """
    global _ddg_search
    if _ddg_search is None:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun

            _ddg_search = DuckDuckGoSearchRun()
            logger.info("DuckDuckGo 搜索工具初始化成功")
        except ImportError:
            logger.warning(
                "DuckDuckGo 搜索不可用，请执行：pip install duckduckgo-search"
            )
            return None
    return _ddg_search
