"""长期记忆管理端点。"""

import uuid
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.memory.long_term import long_term_memory

router = APIRouter(tags=["记忆管理"])


def _get_session_id(request: Request) -> str:
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    return request.session["session_id"]


# ── 查询 ──────────────────────────────────────────────────────────────────────

@router.get("/memory", summary="查看当前会话的长期记忆")
async def get_memory(req: Request):
    """
    返回当前会话的全部长期记忆，包括：
    - **user_profile**：用户画像（年龄、过敏史、既往病史等）
    - **medical_fact**：本次就诊提取的医疗事实（主诉、症状、诊断建议等）
    """
    session_id = _get_session_id(req)
    all_memories = long_term_memory.load(session_id, min_importance=1)
    profile = [m for m in all_memories if m["memory_type"] == "user_profile"]
    facts   = [m for m in all_memories if m["memory_type"] == "medical_fact"]
    other   = [m for m in all_memories if m["memory_type"] not in ("user_profile", "medical_fact")]

    return {
        "session_id": session_id,
        "user_profile": profile,
        "medical_facts": facts,
        "other": other,
        "total": len(all_memories),
        "formatted_context": long_term_memory.format_for_prompt(session_id),
        "success": True,
    }


@router.get("/memory/profile", summary="仅查看用户画像")
async def get_profile(req: Request):
    """返回结构化的用户画像字典（key → value）。"""
    session_id = _get_session_id(req)
    profile = long_term_memory.load_profile(session_id)
    return {"session_id": session_id, "profile": profile, "success": True}


# ── 手动写入 ──────────────────────────────────────────────────────────────────

class MemoryUpsertRequest(BaseModel):
    memory_type: str = Field(
        default="user_profile",
        description="记忆类型：user_profile | medical_fact | preference",
    )
    key: str = Field(..., description="记忆键，如 allergies、age、chief_complaint")
    value: str = Field(..., description="记忆值")
    importance: int = Field(default=7, ge=1, le=10, description="重要性（1~10）")


@router.post("/memory", summary="手动写入或更新一条长期记忆")
async def upsert_memory(req: Request, body: MemoryUpsertRequest):
    """
    手动向长期记忆写入一条条目（相同 key 则覆盖）。

    适用场景：用户主动告知医疗信息（如"我对青霉素过敏"），
    前端可直接调用此接口写入，无需等待 LLM 自动提取。
    """
    session_id = _get_session_id(req)
    long_term_memory.upsert(
        session_id=session_id,
        memory_type=body.memory_type,
        key=body.key,
        value=body.value,
        importance=body.importance,
    )
    return {
        "session_id": session_id,
        "message": f"记忆条目已写入：{body.key} = {body.value}",
        "success": True,
    }


# ── 清除 ──────────────────────────────────────────────────────────────────────

@router.delete("/memory", summary="清除当前会话的全部长期记忆")
async def clear_memory(req: Request):
    """
    删除当前会话的全部长期记忆（用户画像 + 医疗事实）。
    聊天记录不受影响。
    """
    session_id = _get_session_id(req)
    long_term_memory.delete_all(session_id)
    return {
        "session_id": session_id,
        "message": "长期记忆已清除",
        "success": True,
    }


@router.delete("/memory/{session_id}/full", summary="完全清除指定会话（聊天记录 + 长期记忆）")
async def clear_session_full(session_id: str):
    """
    完全删除指定会话的所有数据：聊天记录 + 长期记忆。
    通常由管理端或用户"彻底删除"操作触发。
    """
    from app.services.database_service import db_service
    db_service.delete_session_full(session_id)
    return {
        "session_id": session_id,
        "message": "会话数据已完全清除（聊天记录 + 长期记忆）",
        "success": True,
    }
