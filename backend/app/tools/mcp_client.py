"""MCP 统一外部工具客户端（Tavily/Wikipedia/PubMed）。"""

"""
通过 MCP（Model Context Protocol）协议统一接入：
  - Tavily 实时联网搜索
  - Wikipedia 医学百科
  - PubMed 医学文献库（可选，需安装 @cyanheads/pubmed-mcp-server）


MCP 用 stdio 子进程方式启动工具服务器，每次调用均建立临时会话。
为避免每次调用都重新启动进程，使用模块级连接池（_server_sessions）缓存活跃会话。

依赖安装
  pip install mcp langchain-mcp-adapters
  npm install -g @tavily/mcp          # Tavily MCP 服务器
  pip install mcp-server-fetch        # Wikipedia（通用 fetch）
  # 可选：@cyanheads/pubmed-mcp-server（npx 按需加载，无需预安装）

.env 配置
  TAVILY_API_KEY=tvly-xxx
  MCP_WIKIPEDIA_ENABLED=true          # 默认 true
  MCP_PUBMED_ENABLED=true            # 默认 true
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

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
            "args": ["-y", "@tavily/mcp"],
            "env": {
                **os.environ,
                "TAVILY_API_KEY": TAVILY_API_KEY,
            },
            "default_tool": "tavily-search",
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
 

def mcp_wikipedia_search(query: str, max_chars: int = 2000) -> List[Dict[str, str]]:
    """
    通过 MCP fetch 工具获取 Wikipedia 页面内容。
    策略：先查中文 Wikipedia，若结果过短再查英文。

    返回
    ----
    List[{"title": str, "content": str, "url": str}]
    """
    results = []

    zh_url = f"https://zh.wikipedia.org/wiki/{query.replace(' ', '_')}"
    try:
        text = call_mcp_tool(
            "wikipedia",
            arguments={"url": zh_url, "max_length": max_chars},
        )
        if text and len(text.strip()) > 150:
            results.append({
                "title": f"Wikipedia（中文）：{query}",
                "content": text.strip()[:max_chars],
                "url": zh_url,
            })
            logger.info("MCP Wikipedia（中文）检索成功，%d 字符", len(text))
    except Exception as exc:
        logger.warning("MCP Wikipedia 中文检索失败：%s", exc)

    if not results:
        en_url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        try:
            text = call_mcp_tool(
                "wikipedia",
                arguments={"url": en_url, "max_length": max_chars},
            )
            if text and len(text.strip()) > 150:
                results.append({
                    "title": f"Wikipedia（英文）：{query}",
                    "content": text.strip()[:max_chars],
                    "url": en_url,
                })
                logger.info("MCP Wikipedia（英文）检索成功，%d 字符", len(text))
        except Exception as exc:
            logger.warning("MCP Wikipedia 英文检索失败：%s", exc)

    return results


def mcp_tavily_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    通过 MCP Tavily 工具进行实时联网搜索。

    返回
    ----
    List[{"title": str, "content": str, "url": str}]
    """
    try:
        raw = call_mcp_tool(
            "tavily",
            arguments={
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
            },
        )

        results = []
        try:
            parsed = json.loads(raw)
            items = parsed if isinstance(parsed, list) else parsed.get("results", [])
            for item in items[:max_results]:
                content = item.get("content") or item.get("text") or ""
                if len(content.strip()) > 50:
                    results.append({
                        "title": item.get("title", ""),
                        "content": content.strip(),
                        "url": item.get("url", ""),
                    })
        except (json.JSONDecodeError, AttributeError):
            if raw and len(raw.strip()) > 50:
                results.append({
                    "title": f"Tavily 搜索：{query}",
                    "content": raw.strip(),
                    "url": "",
                })

        logger.info("MCP Tavily 检索成功，%d 条结果", len(results))
        return results

    except Exception as exc:
        logger.warning("MCP Tavily 检索失败：%s", exc)
        raise


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
