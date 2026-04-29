"""FastAPI 应用入口。"""

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MedicalAI 系统启动中…")

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

app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
