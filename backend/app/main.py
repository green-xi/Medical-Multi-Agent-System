"""
MedicalAI — main.py
FastAPI 应用入口：应用初始化、生命周期管理与路由注册。

模块结构：
  core/               — 配置、日志、状态、工作流
  agents/             — 8 个 LangGraph 智能体节点
  tools/              — LLM 客户端、向量库、PDF 加载器、搜索工具
  db/                 — SQLAlchemy 会话工厂
  models/             — ORM 模型
  schemas/            — Pydantic 请求/响应模式
  services/           — DatabaseService、ChatService
  api/v1/endpoints/   — health、chat、session 路由处理器
  api/v1/api.py       — 路由聚合器
  main.py             — FastAPI 应用 + 生命周期  ← 当前文件
"""

import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.api.v1.api import api_router
from app.core.config import CHAT_DB_PATH, PDF_PATH, VECTOR_STORE_DIR
from app.core.logging_config import logger
from app.services.chat_service import chat_service
from app.services.database_service import db_service
from app.tools.pdf_loader import process_pdf
from app.tools.vector_store import get_or_create_vectorstore


# ── 生命周期 ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动与关闭生命周期管理。"""
    logger.info("MedicalAI 系统启动中…")

    # 数据库迁移（幂等，可重复执行）
    from app.db.migrate import run_all_migrations
    run_all_migrations()

    db_service.init_db()
    logger.info("数据库已初始化：%s", CHAT_DB_PATH)

    if os.path.exists(PDF_PATH):
        logger.info("正在处理 PDF：%s", PDF_PATH)
        documents = process_pdf(PDF_PATH)
        get_or_create_vectorstore(documents)
        logger.info("向量库就绪：%s", VECTOR_STORE_DIR)
    else:
        logger.warning("未找到 PDF 文件：%s — 向量库初始化跳过", PDF_PATH)

    chat_service.initialize_workflow()

    # ── Reranker 预热（消灭首次调用的冷启动延迟） ────────────────────────────
    # 本地 bge-reranker 首次加载模型需要 20-25 秒，在启动阶段完成加载，
    # 确保用户第一条消息得到正常响应速度而非等待模型初始化。
    try:
        from app.tools.reranker import rerank_documents
        from langchain_core.documents import Document as _Doc
        _warmup_docs = [_Doc(page_content="预热文档，用于触发模型加载。")]
        rerank_documents(query="预热", documents=_warmup_docs, top_k=1)
        logger.info("Reranker 预热完成，首次检索将直接使用已加载的模型")
    except Exception as exc:
        logger.warning("Reranker 预热失败（不影响启动）：%s", exc)

    logger.info("MedicalAI 系统就绪！")

    yield

    logger.info("MedicalAI 系统关闭中…")


# ── FastAPI 应用 ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedicalAI API",
    description="AI 驱动的医疗问诊系统 — 深度模块化多智能体架构",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

# 注册所有 API 路由
app.include_router(api_router)


# ── 入口 ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
