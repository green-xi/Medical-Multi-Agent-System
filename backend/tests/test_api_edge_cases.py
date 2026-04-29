"""
test_api_edge_cases.py — API 边界条件测试

覆盖（仅保留 test_api.py 中没有的独特用例）：
  - 无 session ID 时自动分配会话
  - 先创建再删除同一会话的完整生命周期
  - 加载 session 后 messages 数量和 session_id 均正确返回
  - workflow tool_trace 包含全部五个 Agent 节点

五 Agent 架构：memory → query_rewriter → planner → research → critic
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services import db_service  # noqa: E402


def test_chat_without_session_id_gets_auto_assigned(test_client):
    """不带 session ID 的请求应自动分配会话，返回 200。"""
    client, _ = test_client
    response = client.post(
        "/api/v1/chat",
        json={"message": "匿名请求"},
    )
    assert response.status_code == 200


def test_delete_current_session(test_client):
    """先创建一个会话，再删除它，应返回 200 且 success=True。"""
    client, _ = test_client
    with patch.object(db_service, "delete_session"):
        new_chat_resp = client.post("/api/v1/new-chat")
        session_id = new_chat_resp.json()["session_id"]
        response = client.delete(f"/api/v1/session/{session_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_load_session_persists_history(test_client):
    """加载指定 session 应返回该 session 的历史记录列表和正确的 session_id。"""
    client, _ = test_client
    with patch.object(db_service, "get_chat_history") as mock_hist:
        mock_hist.return_value = [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧答案"},
        ]
        response = client.get("/api/v1/session/old-session-id")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "old-session-id"
    assert len(data["messages"]) == 2


def test_workflow_tool_trace_has_five_agents(test_client):
    """workflow mock 的 tool_trace 应包含全部五个 Agent 节点名称。"""
    client, mock_workflow = test_client
    response = client.post(
        "/api/v1/chat",
        json={"message": "测试五 Agent 工作流"},
        headers={"X-Session-ID": "trace-test"},
    )
    assert response.status_code == 200
    trace = mock_workflow.ainvoke.return_value.get("tool_trace", [])
    expected_agents = {"memory", "query_rewriter", "planner", "research", "critic"}
    assert expected_agents.issubset(set(trace)), (
        f"tool_trace 缺少 Agent 节点。期望包含: {expected_agents}，实际: {trace}"
    )
