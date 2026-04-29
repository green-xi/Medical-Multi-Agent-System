"""ChromaDB 向量数据库：嵌入模型加载、向量库创建/检索。"""

import os
from typing import List, Optional

from langchain_core.documents import Document

from app.core.config import EMBEDDING_MODEL, VECTOR_STORE_DIR
from app.core.logging_config import logger

_embeddings = None
_vectorstore = None

_HF_CANDIDATES = [
    "BAAI/bge-small-zh-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def _try_load_hf_embedding(model_name: str):
    try:
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        emb.embed_query("头痛发烧")
        logger.info("嵌入模型已加载：%s", model_name)
        return emb
    except Exception as exc:
        logger.warning("嵌入模型 %s 加载失败，尝试下一个：%s", model_name, exc)
        return None


def get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    candidates = list(_HF_CANDIDATES)
    if EMBEDDING_MODEL and EMBEDDING_MODEL not in candidates:
        candidates = [EMBEDDING_MODEL] + candidates
    elif EMBEDDING_MODEL and EMBEDDING_MODEL in candidates:
        candidates = [EMBEDDING_MODEL] + [c for c in candidates if c != EMBEDDING_MODEL]

    for model_name in candidates:
        _embeddings = _try_load_hf_embedding(model_name)
        if _embeddings:
            return _embeddings

    logger.error(
        "所有嵌入方案均失败！RAG 不可用。\n"
        "解决方法：\n"
        "  1. 在 .env 中设置 HF_ENDPOINT=https://hf-mirror.com 使用国内镜像\n"
        "  2. 手动下载模型到本地，设置 EMBEDDING_MODEL=本地路径"
    )
    return None


def get_or_create_vectorstore(
    documents: Optional[List[Document]] = None,
    persist_dir: str = VECTOR_STORE_DIR,
):
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    from langchain_chroma import Chroma

    embeddings = get_embeddings()

    if embeddings is None:
        logger.warning(
            "嵌入模型不可用，跳过向量库构建。"
            "系统将以纯 LLM 模式运行（无 RAG 检索增强）。"
        )
        return None

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    db_files_exist = any(
        f.endswith(".sqlite3") or f == "chroma.sqlite3" or f.startswith("index")
        for f in os.listdir(persist_dir)
    ) if os.path.exists(persist_dir) else False

    if db_files_exist:
        logger.info("从 %s 加载已有向量库", persist_dir)
        _vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        if _vectorstore._collection.count() == 0:
            logger.warning("向量库为空，将重新构建")
            _vectorstore = None
            if not documents:
                logger.warning("向量库为空但未提供文档，跳过构建")
                return None
            logger.info("正在基于 %d 篇文档新建向量库…", len(documents))
            _vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"},
            )
            logger.info("向量库构建完成，共 %d 篇文档", len(documents))
            return _vectorstore
        logger.info("已加载 %d 篇文档到向量库", _vectorstore._collection.count())

    elif documents:
        logger.info("正在基于 %d 篇文档新建向量库（首次构建较慢，请耐心等待）…", len(documents))
        _vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info("向量库构建完成，共 %d 篇文档", len(documents))
    else:
        logger.warning("未找到已有向量库，也未提供文档，向量库初始化跳过")
        return None

    return _vectorstore


def get_retriever(k: int = 8, fetch_k: int = 30):
    vs = get_or_create_vectorstore()
    if vs:
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.65}
        )
    return None


def check_coverage(query: str, threshold: float = 0.3) -> dict:
    from app.tools.reranker import rerank_documents
    vs = get_or_create_vectorstore()
    if not vs:
        return {"covered": False, "top_score": 0.0, "suggestion": "向量库不可用"}

    docs = vs.similarity_search(query, k=3)
    if not docs:
        return {"covered": False, "top_score": 0.0, "suggestion": "知识库为空"}

    reranked = rerank_documents(query, docs, top_k=1)
    top_score = reranked[0].metadata.get("rerank_score", 0.0) if reranked else 0.0
    covered = top_score >= threshold
    suggestion = (
        "知识库有相关内容，可继续 RAG 检索" if covered
        else f"知识库盲区（top_score={top_score:.3f}），建议直接路由到 Tavily/Wikipedia"
    )
    return {"covered": covered, "top_score": top_score, "suggestion": suggestion}
