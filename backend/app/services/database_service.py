"""
MedicalAI — services/database_service.py
DatabaseService：聊天历史与长期记忆的全部 CRUD 操作。
"""

from typing import Dict, List, Optional

from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from app.core.logging_config import logger
from app.db.session import SessionLocal, engine
from app.models.message import Message
from app.models.user_memory import UserMemory
from app.models.message import Base  # 统一 Base（UserMemory 已注册其中）


class DatabaseService:
    """聊天历史与长期记忆数据库 CRUD 操作封装。"""

    def __init__(self, session_local=None, engine_instance=None):
        self.SessionLocal = session_local or SessionLocal
        self.engine = engine_instance or engine
        logger.info("DatabaseService 初始化完成")

    def init_db(self) -> None:
        """若数据表不存在则自动创建（含 messages 和 user_memory 表）。"""
        logger.info("正在初始化数据库表结构…")
        # 确保 UserMemory 表也被注册
        import app.models  # noqa: F401
        Base.metadata.create_all(bind=self.engine)
        logger.info("数据库表结构初始化完成")

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ── 聊天消息 CRUD ──────────────────────────────────────────────────────────

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        source: Optional[str] = None,
    ) -> None:
        logger.debug("保存 %s 消息，会话 %s…", role, session_id[:8])
        with self.get_session() as session:
            session.add(
                Message(
                    session_id=session_id, role=role, content=content, source=source
                )
            )
            session.commit()

    def get_chat_history(self, session_id: str) -> List[Dict]:
        with self.get_session() as session:
            stmt = (
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp)
            )
            return [msg.to_dict() for msg in session.execute(stmt).scalars().all()]

    def get_all_sessions(self) -> List[Dict]:
        with self.get_session() as session:
            latest_sub = (
                select(
                    Message.session_id,
                    func.max(Message.timestamp).label("max_ts"),
                )
                .where(Message.role == "user")
                .group_by(Message.session_id)
                .subquery()
            )
            stmt = (
                select(Message.session_id, Message.content, Message.timestamp)
                .join(
                    latest_sub,
                    (Message.session_id == latest_sub.c.session_id)
                    & (Message.timestamp == latest_sub.c.max_ts),
                )
                .order_by(desc(Message.timestamp))
            )
            return [
                {
                    "session_id": row[0],
                    "preview": row[1][:50] + "..." if len(row[1]) > 50 else row[1],
                    "last_active": row[2].isoformat() if row[2] else None,
                }
                for row in session.execute(stmt).all()
            ]

    def delete_session(self, session_id: str) -> None:
        """删除会话的聊天记录（长期记忆默认保留，可选择同时清除）。"""
        logger.info("正在删除会话 %s 的聊天记录…", session_id[:8])
        with self.get_session() as session:
            session.execute(delete(Message).where(Message.session_id == session_id))
            session.commit()

    def delete_session_full(self, session_id: str) -> None:
        """删除会话的聊天记录 + 长期记忆（完全清除）。"""
        self.delete_session(session_id)
        from app.memory.long_term import long_term_memory
        long_term_memory.delete_all(session_id)
        logger.info("已完全清除会话 %s（聊天记录 + 长期记忆）", session_id[:8])


# 模块级单例
db_service = DatabaseService()
