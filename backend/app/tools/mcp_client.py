"""MCP 统一外部工具客户端（Tavily/Wikipedia/PubMed）。"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from app.core.logging_config import logger
from app.core.config import TAVILY_API_KEY


def _check_mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False

MCP_AVAILABLE = _check_mcp_available()

MCP_WIKIPEDIA_ENABLED = os.getenv("MCP_WIKIPEDIA_ENABLED", "true").lower() == "true"
MCP_PUBMED_ENABLED    = os.getenv("MCP_PUBMED_ENABLED",    "false").lower() == "true"


def _build_server_configs() -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    if TAVILY_API_KEY:
        configs["tavily"] = {
            "command": "npx",
            "args": ["-y", "@tavily/mcp"],
            "env": {**os.environ, "TAVILY_API_KEY": TAVILY_API_KEY},
            "default_tool": "tavily-search",
        }
    if MCP_WIKIPEDIA_ENABLED:
        configs["wikipedia"] = {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
            "env": dict(os.environ),
            "default_tool": "fetch",
        }
    if MCP_PUBMED_ENABLED:
        configs["pubmed"] = {
            "command": "uvx",
            "args": ["--from", "pubmed-mcp-server", "pubmed_mcp_server"],
            "env": dict(os.environ),
            "default_tool": "search_pubmed_articles",
        }
    return configs

MCP_SERVER_CONFIGS = _build_server_configs()


async def _call_mcp_tool_async(
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: float = 20.0,
) -> str:
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
    try:
        return asyncio.run(
            _call_mcp_tool_async(server_name, actual_tool, actual_args, timeout)
        )
    except Exception as exc:
        logger.warning("MCP 调用失败 [%s/%s]：%s", server_name, actual_tool, exc)
        raise


def mcp_wikipedia_search(query: str, max_chars: int = 2000) -> List[Dict[str, str]]:
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
    try:
        raw = call_mcp_tool(
            "pubmed",
            tool_name="search_pubmed_articles",
            arguments={"query": query},
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


def get_mcp_status() -> Dict[str, Any]:
    return {
        "mcp_available":        MCP_AVAILABLE,
        "configured_servers":   list(MCP_SERVER_CONFIGS.keys()),
        "tavily_configured":    "tavily"    in MCP_SERVER_CONFIGS,
        "wikipedia_configured": "wikipedia" in MCP_SERVER_CONFIGS,
        "pubmed_configured":    "pubmed"    in MCP_SERVER_CONFIGS,
    }
