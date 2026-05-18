"""
test_api.py — FastAPI 接口集成测试

覆盖：
  - /api/v1/health
  - /api/v1/chat（正常流程 / 系统未初始化 / session 管理）
  - /api/v1/history / /api/v1/sessions / /api/v1/session
  - /api/v1/clear / /api/v1/new-chat
"""
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services import chat_service, db_service


class TestHealthEndpoint:
    def test_returns_healthy_status(self, test_client):
        client, _ = test_client
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestChatEndpoint:

    def test_chat_success_returns_response(self, test_client):
        client, mock_workflow = test_client
        response = client.post(
            "/api/v1/chat",
            json={"message": "维生素C可以预防新冠吗"},
            headers={"X-Session-ID": "test-sess-001"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["response"] == "Test AI response"
        assert data["source"] == "Test Source"

    def test_chat_returns_503_when_workflow_not_initialized(self, test_client):
        client, _ = test_client
        with patch.object(chat_service, "workflow_app", None):
            response = client.post(
                "/api/v1/chat",
                json={"message": "hello"},
            )
        assert response.status_code == 503

    def test_chat_uses_x_session_id_header(self, test_client):
        """X-Session-ID header 应被用作会话标识。"""
        client, _ = test_client
        response = client.post(
            "/api/v1/chat",
            json={"message": "测试"},
            headers={"X-Session-ID": "custom-session-abc"},
        )
        assert response.status_code == 200

    def test_thinking_data_included_in_response(self, test_client):
        """响应应包含 thinking 字段供前端展示 AI 思考过程。"""
        client, _ = test_client
        response = client.post(
            "/api/v1/chat",
            json={"message": "头痛怎么办"},
            headers={"X-Session-ID": "test-thinking"},
        )
        data = response.json()
        assert "thinking" in data
        thinking = data["thinking"]
        assert "rag_think_log" in thinking
        assert "thinking_steps" in thinking
        assert "original_question" in thinking


class TestNewChatEndpoint:

    def test_creates_new_session_with_unique_id(self, test_client):
        client, _ = test_client
        r1 = client.post("/api/v1/new-chat")
        r2 = client.post("/api/v1/new-chat")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["session_id"] != r2.json()["session_id"]
        assert r1.json()["success"] is True


class TestHistoryEndpoint:

    def test_returns_chat_history_for_session(self, test_client):
        client, _ = test_client
        with patch.object(db_service, "get_chat_history") as mock_hist:
            mock_hist.return_value = [
                {"role": "user", "content": "维生素C能预防新冠吗"},
                {"role": "assistant", "content": "目前没有科学证据支持这一说法"},
            ]
            response = client.get(
                "/api/v1/history",
                headers={"X-Session-ID": "hist-sess"},
            )
        assert response.status_code == 200
        messages = response.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"

    def test_returns_empty_list_for_new_session(self, test_client):
        client, _ = test_client
        with patch.object(db_service, "get_chat_history", return_value=[]):
            response = client.get(
                "/api/v1/history",
                headers={"X-Session-ID": "brand-new-session"},
            )
        assert response.status_code == 200
        assert response.json()["messages"] == []


class TestSessionEndpoints:

    def test_get_all_sessions(self, test_client):
        client, _ = test_client
        with patch.object(db_service, "get_all_sessions") as mock_sessions:
            mock_sessions.return_value = [
                {"session_id": "sess-1", "preview": "头痛怎么办", "last_active": "2026-04-23"},
                {"session_id": "sess-2", "preview": "维生素C", "last_active": "2026-04-22"},
            ]
            response = client.get("/api/v1/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["sessions"]) == 2

    def test_load_session_by_id(self, test_client):
        client, _ = test_client
        with patch.object(db_service, "get_chat_history", return_value=[]):
            response = client.get("/api/v1/session/target-session-id")
        assert response.status_code == 200
        assert response.json()["session_id"] == "target-session-id"

    def test_delete_session(self, test_client):
        client, _ = test_client
        with patch.object(db_service, "delete_session") as mock_del:
            response = client.delete("/api/v1/session/sess-to-delete")
        assert response.status_code == 200
        assert response.json()["success"] is True
        mock_del.assert_called_once_with("sess-to-delete")


class TestClearEndpoint:

    def test_clears_conversation_for_session(self, test_client):
        client, _ = test_client
        response = client.post(
            "/api/v1/clear",
            headers={"X-Session-ID": "sess-to-clear"},
        )
        assert response.status_code == 200
        assert "cleared" in response.json().get("message", "").lower() or \
               response.json().get("success") is True


class TestMetricsEndpoint:

    def test_metrics_endpoint_returns_expected_keys(self, test_client):
        client, _ = test_client
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "fallback_rate" in data
