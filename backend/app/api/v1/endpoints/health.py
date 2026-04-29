"""健康检查端点。"""

from fastapi import APIRouter
from app.services.chat_service import chat_service

router = APIRouter(tags=["健康检查"])


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "MedicalAI 后端 v2"}


@router.get("/metrics")
async def metrics():
    return chat_service.get_metrics_snapshot()
