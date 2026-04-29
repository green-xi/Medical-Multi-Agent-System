"""
MedicalAI — memory/__init__.py
记忆系统模块导出。

  short_term  短期记忆：当前会话滑动窗口 + 历史压缩
  long_term   长期记忆：跨会话持久化用户画像与医疗事实
"""

from app.memory.short_term import build_context_window, compress_history
from app.memory.long_term import LongTermMemoryService, long_term_memory

__all__ = [
    "compress_history",
    "build_context_window",
    "LongTermMemoryService",
    "long_term_memory",
]
