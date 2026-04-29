"""
test_database.py — 数据库服务单元测试（真实 SQLite）
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import get_engine, get_session_factory
from app.services.database_service import DatabaseService

TEST_DB = "tests/test_database/test_chat.db"


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test_chat.db")
    engine = get_engine(db_path)
    session_factory = get_session_factory(engine)
    service = DatabaseService(session_local=session_factory, engine_instance=engine)
    service.init_db()
    yield service
    engine.dispose()


class TestMessagePersistence:

    def test_save_and_retrieve_user_message(self, db):
        db.save_message("sess-1", "user", "维生素C能预防新冠吗")
        history = db.get_chat_history("sess-1")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "维生素C能预防新冠吗"

    def test_save_assistant_message_with_source(self, db):
        db.save_message("sess-1", "assistant", "目前无科学证据支持", source="医学知识库")
        history = db.get_chat_history("sess-1")
        assert history[0]["source"] == "医学知识库"

    def test_multiple_messages_preserve_order(self, db):
        db.save_message("sess-2", "user", "第一条")
        db.save_message("sess-2", "assistant", "第二条")
        db.save_message("sess-2", "user", "第三条")
        history = db.get_chat_history("sess-2")
        assert len(history) == 3
        assert history[0]["content"] == "第一条"
        assert history[2]["content"] == "第三条"

    def test_different_sessions_are_isolated(self, db):
        db.save_message("sess-A", "user", "A的消息")
        db.save_message("sess-B", "user", "B的消息")
        assert len(db.get_chat_history("sess-A")) == 1
        assert len(db.get_chat_history("sess-B")) == 1
        assert db.get_chat_history("sess-A")[0]["content"] == "A的消息"

    def test_empty_history_for_nonexistent_session(self, db):
        history = db.get_chat_history("nonexistent-session")
        assert history == []


class TestSessionManagement:

    def test_get_all_sessions_returns_all(self, db):
        db.save_message("sess-1", "user", "msg1")
        db.save_message("sess-2", "user", "msg2")
        db.save_message("sess-3", "user", "msg3")
        sessions = db.get_all_sessions()
        session_ids = [s["session_id"] for s in sessions]
        assert "sess-1" in session_ids
        assert "sess-2" in session_ids
        assert "sess-3" in session_ids

    def test_delete_session_removes_all_messages(self, db):
        db.save_message("sess-del", "user", "消息1")
        db.save_message("sess-del", "assistant", "回复1")
        assert len(db.get_chat_history("sess-del")) == 2
        db.delete_session("sess-del")
        assert db.get_chat_history("sess-del") == []

    def test_delete_nonexistent_session_is_safe(self, db):
        """删除不存在的会话不应抛异常。"""
        db.delete_session("nonexistent-session-xyz")
