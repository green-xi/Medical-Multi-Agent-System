"""SQLAlchemy 引擎与会话工厂。"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import CHAT_DB_PATH
from app.core.logging_config import logger


def get_engine(db_path: str = CHAT_DB_PATH):
    """为给定 SQLite 路径创建并返回 SQLAlchemy 引擎。"""
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    logger.debug("数据库引擎已创建：%s", db_path)
    return create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )


def get_session_factory(engine):
    """返回绑定到给定引擎的 sessionmaker。"""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# 模块级单例
engine = get_engine()
SessionLocal = get_session_factory(engine)
