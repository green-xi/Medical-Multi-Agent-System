"""FastAPI 应用入口。"""

import asyncio
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


def _vector_store_exists() -> bool:
    """检查向量库是否已有数据，避免重复处理 PDF。"""
    vs_dir = VECTOR_STORE_DIR
    if not os.path.exists(vs_dir):
        return False
    try:
        for f in os.listdir(vs_dir):
            if f.endswith(".sqlite3") or f == "chroma.sqlite3":
                return True
    except OSError:
        pass
    return False


async def _warmup_reranker():
    """
    后台预热 Reranker 模型，不阻塞启动。
    本地 bge-reranker 首次加载需要 20-25 秒，提前在后台加载
    确保用户第一条消息得到正常响应速度。
    使用 run_in_executor 避免阻塞事件循环影响用户请求。
    """
    try:
        from app.tools.reranker import rerank_documents
        from langchain_core.documents import Document as _Doc

        def _sync_warmup():
            _warmup_docs = [_Doc(page_content="预热文档，用于触发模型加载。")]
            rerank_documents(query="预热", documents=_warmup_docs, top_k=1)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_warmup)
        logger.info("Reranker 预热完成，首次检索将直接使用已加载的模型")
    except Exception as exc:
        logger.warning("Reranker 预热失败（不影响启动）：%s", exc)


async def _warmup_llm():
    """
    后台预热 LLM 客户端，避免第一条用户消息承担连接建立的冷启动延迟。
    只初始化客户端实例，不发起真实 API 调用（节省费用）。
    """
    try:
        def _sync_init():
            from app.tools.llm_client import get_llm
            get_llm()  # 触发单例初始化，建立 DashScope 连接配置

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_init)
        logger.info("LLM 客户端预热完成")
    except Exception as exc:
        logger.warning("LLM 客户端预热失败（不影响启动）：%s", exc)


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

    # ── 向量库初始化：已有数据则跳过 PDF 处理 ──────────────────────────────
    if _vector_store_exists():
        logger.info("向量库已存在，跳过 PDF 处理，直接加载…")
        get_or_create_vectorstore()
    elif os.path.exists(PDF_PATH):
        logger.info("正在处理 PDF：%s", PDF_PATH)
        documents = process_pdf(PDF_PATH)
        get_or_create_vectorstore(documents)
    else:
        logger.warning("未找到 PDF 文件：%s — 向量库初始化跳过", PDF_PATH)

    logger.info("向量库就绪：%s", VECTOR_STORE_DIR)

    chat_service.initialize_workflow()

    # ── 后台预热：Reranker + LLM 客户端（不阻塞端口监听） ──────────────────
    asyncio.create_task(_warmup_reranker())
    asyncio.create_task(_warmup_llm())

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
