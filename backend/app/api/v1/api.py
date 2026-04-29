"""v1 路由聚合器。"""

from fastapi import APIRouter

from app.api.v1.endpoints import chat, health, memory, session

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(session.router)
api_router.include_router(memory.router)
