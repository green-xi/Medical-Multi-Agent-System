"""会话管理端点。"""

import uuid

from fastapi import APIRouter, Request

from app.services.database_service import db_service

router = APIRouter(tags=["会话管理"])


def _get_session_id(request: Request) -> str:
    """从 X-Session-ID 请求头或 cookie 会话中获取或创建 session ID。"""
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    return request.session["session_id"]


@router.get("/history", summary="获取当前会话的对话历史")
async def get_history_endpoint(req: Request):
    """返回当前会话的全部聊天记录。"""
    return {
        "messages": db_service.get_chat_history(_get_session_id(req)),
        "success": True,
    }


@router.get("/sessions", summary="获取所有历史会话列表")
async def get_sessions_endpoint():
    """返回所有对话会话及其预览摘要。"""
    return {"sessions": db_service.get_all_sessions(), "success": True}


@router.get("/session/{session_id}", summary="加载指定会话")
async def load_session_endpoint(session_id: str, req: Request):
    """加载指定 session ID 的历史对话，并设为当前活跃会话。"""
    req.session["session_id"] = session_id
    return {
        "messages": db_service.get_chat_history(session_id),
        "session_id": session_id,
        "success": True,
    }


@router.delete("/session/{session_id}", summary="删除指定会话")
async def delete_session_endpoint(session_id: str, req: Request):
    """删除指定会话；若删除的是当前会话，则自动生成新 session ID。"""
    db_service.delete_session(session_id)
    if req.session.get("session_id") == session_id:
        req.session["session_id"] = str(uuid.uuid4())
    return {"message": "会话已删除", "success": True}
