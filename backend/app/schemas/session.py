"""会话与消息响应 schema。"""

from typing import Optional

from pydantic import BaseModel


class SessionResponse(BaseModel):
    session_id: str
    preview: str
    last_active: Optional[str] = None


class MessageResponse(BaseModel):
    role: str
    content: str
    source: Optional[str] = None
    timestamp: Optional[str] = None
