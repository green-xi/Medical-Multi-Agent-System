"""
MedicalAI — agents/__init__.py
统一导出五个核心 Agent 节点函数。

当前架构（五 Agent）：
  memory → query_rewriter → planner → research → critic

已移除的旧节点（职责已内化进 ResearchAgent）：
  RetrieverAgent   → research.py 的 rag_search action
  RAGGraderAgent   → research.py 的 THINK-ACT 评分循环
  LLMAgent         → research.py 的 llm_direct action
  ExecutorAgent    → research.py 的 _generate_answer()
  ExplanationAgent → critic.py 的 revised_answer 润色
  TavilyAgent      → research.py 的 tavily action（MCP）
  WikipediaAgent   → research.py 的 wikipedia action（MCP）
  ToolAgent        → research.py 的 tool_query action
"""

from app.agents.memory import MemoryAgent
from app.agents.query_rewriter import QueryRewriterAgent
from app.agents.planner import PlannerAgent
from app.agents.research import ResearchAgent
from app.agents.critic import CriticAgent

__all__ = [
    "MemoryAgent",
    "QueryRewriterAgent",
    "PlannerAgent",
    "ResearchAgent",
    "CriticAgent",
]