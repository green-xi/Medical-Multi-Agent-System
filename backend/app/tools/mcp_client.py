"""MCP 统一外部工具客户端（Tavily/Wikipedia/PubMed）。"""

"""
通过 MCP（Model Context Protocol）协议统一接入：
  - Tavily 实时联网搜索
  - Wikipedia 医学百科
  - PubMed 医学文献库（可选，需安装 @cyanheads/pubmed-mcp-server）
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from typing import Any, Callable, Dict, List, Optional

from app.core.logging_config import logger
from app.core.config import TAVILY_API_KEY, _env

#    MCP 可用性探测                                                             

def _check_mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False

MCP_AVAILABLE = _check_mcp_available()

#    .env 开关                                                                  
MCP_WIKIPEDIA_ENABLED = _env("MCP_WIKIPEDIA_ENABLED", "true").lower() == "true"
MCP_PUBMED_ENABLED    = _env("MCP_PUBMED_ENABLED",    "false").lower() == "true"


 
# MCP 服务器配置表
 
def _build_server_configs() -> Dict[str, Dict[str, Any]]:
    """
    根据环境变量动态构建 MCP 服务器配置。
    只注册已启用且 API Key 已配置的服务器。
    """
    configs: Dict[str, Dict[str, Any]] = {}

    #    Tavily
    if TAVILY_API_KEY:
        configs["tavily"] = {
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {
                **os.environ,
                "TAVILY_API_KEY": TAVILY_API_KEY,
            },
            "default_tool": "tavily_search",
        }

    #    Wikipedia（mcp-server-fetch 通用 HTTP fetch，免 API Key）             
    if MCP_WIKIPEDIA_ENABLED:
        configs["wikipedia"] = {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
            "env": dict(os.environ),
            "default_tool": "fetch",
        }

    #    PubMed 医学文献库（可选）                                             
    if MCP_PUBMED_ENABLED:
        # @cyanheads/pubmed-mcp-server（npm，无需预安装，npx 按需加载）
        configs["pubmed"] = {
            "command": "npx",
            "args": ["-y", "@cyanheads/pubmed-mcp-server@latest"],
            "env": dict(os.environ),
            "default_tool": "pubmed_search_articles",
        }

    return configs


MCP_SERVER_CONFIGS = _build_server_configs()


#  MCP 工具参数构造（供 act_mcp_tool 统一调度使用） 
# 每个服务器的查询参数如何从用户输入中构造，
# 避免在 action handler 中写服务器特定的逻辑。
MCP_TOOL_PARAMS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "tavily": lambda q: {
        "query": q,
        "max_results": 3,
        "search_depth": "advanced",
    },
    "wikipedia": lambda q: {
        "url": f"https://zh.wikipedia.org/wiki/{q.replace(' ', '_')}",
        "max_length": 2000,
    },
}


#  动态工具发现 
# MCP 工具缓存：{server_name: [{name, description, input_schema}]}
_mcp_tools_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None


def _fetch_server_tools_sync(
    server_name: str, cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    启动单个 MCP 服务器，调用 list_tools() 获取其工具列表。
    失败时返回空列表（日志记录原因），不中断其他服务器的加载。
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=cfg["command"],
        args=cfg["args"],
        env=cfg.get("env"),
    )

    async def _async_list():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in result.tools
                ]

    def _run_in_new_loop():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_async_list())
        finally:
            loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_new_loop)
            return future.result(timeout=30)
    except Exception as exc:
        logger.warning("MCP 工具发现失败 [%s]：%s（该服务器将被跳过）", server_name, exc)
        return []


def load_mcp_tools_sync() -> Dict[str, List[Dict[str, Any]]]:
    """
    遍历已配置的 MCP 服务器，通过 list_tools() 发现每个服务器上的工具。

    返回值
    {"server_name": [{"name": str, "description": str, "input_schema": dict}, ...]}

    首次调用会实际启动 MCP 服务器进行发现，结果缓存到模块级变量。
    后续调用直接返回缓存。MCP 不可用或服务器失败时返回空字典。
    """
    global _mcp_tools_cache
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache

    _mcp_tools_cache = {}

    if not MCP_AVAILABLE:
        logger.info("MCP 不可用，跳过工具发现")
        return _mcp_tools_cache

    if not MCP_SERVER_CONFIGS:
        logger.info("未配置 MCP 服务器，跳过工具发现")
        return _mcp_tools_cache

    for server_name, cfg in MCP_SERVER_CONFIGS.items():
        tools = _fetch_server_tools_sync(server_name, cfg)
        if tools:
            _mcp_tools_cache[server_name] = tools
            names = ", ".join(t["name"] for t in tools)
            logger.info("MCP 工具发现：%s ∈ %s", names, server_name)

    total = sum(len(v) for v in _mcp_tools_cache.values())
    logger.info("MCP 工具发现完成：共 %d 个工具，来自 %d 个服务器", total, len(_mcp_tools_cache))
    return _mcp_tools_cache


def get_discovered_tools() -> Dict[str, List[Dict[str, Any]]]:
    """返回已缓存的 MCP 工具列表（不触发重新发现）。"""
    return _mcp_tools_cache or {}


#  统一 MCP 工具调用（动态分发） 

def call_mcp_tool_dynamic(
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: float = 20.0,
) -> str:
    """
    动态调用指定服务器的 MCP 工具，返回原始文本结果。
    不预设 default_tool，由调用方指定工具名。
    """
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP 库未安装")

    cfg = MCP_SERVER_CONFIGS.get(server_name)
    if not cfg:
        raise ValueError(f"MCP 服务器未配置：{server_name}")

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=cfg["command"],
        args=cfg["args"],
        env=cfg.get("env"),
    )

    async def _async_call():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                parts = []
                for block in result.content:
                    if hasattr(block, "text") and block.text:
                        parts.append(block.text)
                return "\n".join(parts)

    def _run_in_new_loop():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_async_call())
        finally:
            loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_new_loop)
            return future.result(timeout=timeout + 5)
    except Exception as exc:
        logger.warning("MCP 调用失败 [%s/%s]：%s", server_name, tool_name, exc)
        raise


__all__ = [
    "MCP_AVAILABLE", "MCP_SERVER_CONFIGS", "MCP_TOOL_PARAMS",
    "load_mcp_tools_sync", "get_discovered_tools", "call_mcp_tool_dynamic",
    "call_mcp_tool",
    "mcp_pubmed_search", "get_mcp_status",
]



# 核心调用函数

async def _call_mcp_tool_async(
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: float = 20.0,
) -> str:
    """
    异步调用指定 MCP 服务器的工具，返回文本结果。
    每次调用独立启动/关闭 stdio 子进程（无状态，安全可重入）。
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cfg = MCP_SERVER_CONFIGS.get(server_name)
    if not cfg:
        raise ValueError(f"MCP 服务器未配置：{server_name}")

    params = StdioServerParameters(
        command=cfg["command"],
        args=cfg["args"],
        env=cfg.get("env"),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=timeout,
            )
            parts = []
            for block in result.content:
                if hasattr(block, "text") and block.text:
                    parts.append(block.text)
            return "\n".join(parts)


def call_mcp_tool(
    server_name: str,
    tool_name: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    timeout: float = 20.0,
) -> str:
    """
    同步包装：供现有同步代码（ResearchAgent）直接调用。

    参数
    server_name : MCP_SERVER_CONFIGS 中的键（"tavily" / "wikipedia" / "pubmed"）
    tool_name   : MCP 工具名，省略时使用该服务器的 default_tool
    arguments   : 传给工具的参数字典
    timeout     : 单次调用超时秒数

    实现说明
    不使用 asyncio.run()，因为 FastAPI/uvicorn 运行时已有事件循环，
    asyncio.run() 会尝试在已有循环中嵌套新循环，导致 TaskGroup 异常。
    改用 concurrent.futures.ThreadPoolExecutor 在独立线程中运行新事件循环，
    与主线程的事件循环完全隔离，彻底规避嵌套问题。
    """
    if not MCP_AVAILABLE:
        raise RuntimeError(
            "MCP 库未安装，请执行：pip install mcp langchain-mcp-adapters"
        )

    cfg = MCP_SERVER_CONFIGS.get(server_name)
    if not cfg:
        raise ValueError(
            f"MCP 服务器 '{server_name}' 未配置。"
            f"已配置的服务器：{list(MCP_SERVER_CONFIGS.keys())}"
        )

    actual_tool = tool_name or cfg["default_tool"]
    actual_args = arguments or {}

    import concurrent.futures

    def _run_in_new_loop():
        """在全新的事件循环中运行异步 MCP 调用，与 uvicorn 主循环完全隔离。"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                _call_mcp_tool_async(server_name, actual_tool, actual_args, timeout)
            )
        finally:
            loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_in_new_loop)
            return future.result(timeout=timeout + 5)
    except Exception as exc:
        logger.warning("MCP 调用失败 [%s/%s]：%s", server_name, actual_tool, exc)
        raise


# 高层封装：Wikipedia 和 Tavily

def mcp_pubmed_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    通过 MCP PubMed 工具检索医学文献（需启用 MCP_PUBMED_ENABLED=true）。

    包名：@cyanheads/pubmed-mcp-server
    工具名：pubmed_search_articles（参数：query: str）

    返回
    ----
    List[{"title": str, "content": str, "url": str}]
    """
    try:
        raw = call_mcp_tool(
            "pubmed",
            tool_name="pubmed_search_articles",
            arguments={"query": query, "maxResults": max_results, "summaryCount": max_results},
        )
        results = []
        try:
            parsed = json.loads(raw)
            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items[:max_results]:
                content = item.get("abstract") or item.get("content") or str(item)
                if len(content.strip()) > 50:
                    results.append({
                        "title": item.get("title", "PubMed 文献"),
                        "content": content.strip(),
                        "url": item.get("url", ""),
                    })
        except (json.JSONDecodeError, AttributeError):
            if raw and len(raw.strip()) > 50:
                results.append({
                    "title": f"PubMed：{query}",
                    "content": raw.strip(),
                    "url": "",
                })
        logger.info("MCP PubMed 检索成功，%d 条文献", len(results))
        return results
    except Exception as exc:
        logger.warning("MCP PubMed 检索失败：%s", exc)
        raise

 
# 状态查询（供健康检查端点使用）

def get_mcp_status() -> Dict[str, Any]:
    """返回当前 MCP 配置状态。"""
    return {
        "mcp_available":        MCP_AVAILABLE,
        "configured_servers":   list(MCP_SERVER_CONFIGS.keys()),
        "tavily_configured":    "tavily"    in MCP_SERVER_CONFIGS,
        "wikipedia_configured": "wikipedia" in MCP_SERVER_CONFIGS,
        "pubmed_configured":    "pubmed"    in MCP_SERVER_CONFIGS,
    }
