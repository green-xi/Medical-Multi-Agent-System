"""聊天消息 ORM 模型。"""
from datetime import datetime, timezone, timedelta
from typing import Dict

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

_CST = timezone(timedelta(hours=8))

def _now_cst():
    """返回北京时间（UTC+8），存入数据库不带时区信息。"""
    return datetime.now(_CST).replace(tzinfo=None)


class Message(Base):
    """持久化聊天消息（用户轮次或助手轮次）。"""
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    role       = Column(String(50), nullable=False)
    content    = Column(Text, nullable=False)
    source     = Column(String(255), nullable=True)
    timestamp  = Column(DateTime, default=_now_cst)

    def to_dict(self) -> Dict:
        return {
            "id":         self.id,
            "session_id": self.session_id,
            "role":       self.role,
            "content":    self.content,
            "source":     self.source,
            "timestamp":  self.timestamp.isoformat() if self.timestamp else None,
        }
