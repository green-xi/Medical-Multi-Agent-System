"""导出 ORM 模型，确保 Base.metadata 包含全部表定义。"""

from app.models.message import Base, Message
from app.models.user_memory import UserMemory  # noqa: F401 — 注册到同一 Base

__all__ = ["Base", "Message", "UserMemory"]
