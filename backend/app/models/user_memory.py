"""
MedicalAI — models/user_memory.py
SQLAlchemy ORM 模型：用户长期记忆持久化。

长期记忆存储两类信息：
  - user_profile  用户画像：年龄、既往病史、过敏史、用药情况等
  - medical_fact  医疗事实：LLM 从对话中自动提取的关键医学信息
"""

from datetime import datetime
from typing import Dict

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

# 共享同一个 Base，与 Message 模型一起建表
from app.models.message import Base


class UserMemory(Base):
    """用户长期记忆条目。"""

    __tablename__ = "user_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 用 session_id 关联用户（项目无独立用户表，以 session_id 作为用户标识符）
    session_id = Column(String(255), nullable=False, index=True)

    # 记忆类型：user_profile | medical_fact | preference | summary
    memory_type = Column(String(50), nullable=False, default="medical_fact")

    # 记忆键（如 "血压情况" "过敏史" "主诉"）
    key = Column(String(255), nullable=False)

    # 记忆值
    value = Column(Text, nullable=False)

    # 重要性评分 0~1（用于检索时排序）
    importance = Column(Integer, nullable=False, default=5)  # 1~10

    # 来源轮次（来自哪次对话，用于溯源）
    source_turn = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        # SQLite ON CONFLICT DO UPDATE 要求冲突列有 UNIQUE 约束，普通 Index 不够
        # 同一用户的同类型+同 key 唯一（用于 upsert）
        UniqueConstraint(
            "session_id", "memory_type", "key",
            name="uq_user_memory_session_type_key",
        ),
        # 额外建普通索引加速 session_id 查询
        Index("ix_user_memory_session_id", "session_id"),
    )

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "memory_type": self.memory_type,
            "key": self.key,
            "value": self.value,
            "importance": self.importance,
            "source_turn": self.source_turn,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
