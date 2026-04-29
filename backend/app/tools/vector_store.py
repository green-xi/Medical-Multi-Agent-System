"""
MedicalAI — tools/vector_store.py
ChromaDB 向量数据库：嵌入模型、向量库创建/加载、检索器工厂。

始终使用本地 HuggingFace 嵌入模型，不支持 DashScope 云端嵌入。

Chroma 内置 ONNX 模型超时问题：
- 根本原因：未传入 embedding_function 时 Chroma 会自动下载内置 ONNX 模型
- 本文件已修复：始终显式传入 embedding_function，永远不触发 Chroma 自动下载
"""

import os
from typing import List, Optional

from langchain_core.documents import Document

from app.core.config import EMBEDDING_MODEL, VECTOR_STORE_DIR
from app.core.logging_config import logger

_embeddings = None
_vectorstore = None

# 嵌入模型候选列表（HuggingFace，按质量排序）
# 设置 HF_ENDPOINT=https://hf-mirror.com 可加速国内下载
_HF_CANDIDATES = [
    "BAAI/bge-small-zh-v1.5",                    # 轻量通用中文，~100MB
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 多语言兜底
    "sentence-transformers/all-MiniLM-L6-v2",    # 纯英文最终兜底
]


def _try_load_hf_embedding(model_name: str):
    """尝试加载单个 HuggingFace 嵌入模型，成功返回实例，失败返回 None。"""
    try:
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        # 冒烟测试：确认模型实际可运行
        emb.embed_query("头痛发烧")
        logger.info("嵌入模型已加载：%s", model_name)
        return emb
    except Exception as exc:
        logger.warning("嵌入模型 %s 加载失败，尝试下一个：%s", model_name, exc)
        return None


def get_embeddings():
    """
    返回可用的嵌入模型实例。加载策略：

    1. 若 EMBEDDING_MODEL 指定了路径/名称 → 优先尝试该模型
    2. 逐一尝试 _HF_CANDIDATES 列表中的模型
    3. 全部失败 → 返回 None，RAG 降级为纯 LLM 回答

    设置镜像加速 HuggingFace 下载：
      HF_ENDPOINT=https://hf-mirror.com
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    # 优先使用配置中指定的模型
    candidates = list(_HF_CANDIDATES)
    if EMBEDDING_MODEL and EMBEDDING_MODEL not in candidates:
        candidates = [EMBEDDING_MODEL] + candidates
    elif EMBEDDING_MODEL and EMBEDDING_MODEL in candidates:
        candidates = [EMBEDDING_MODEL] + [c for c in candidates if c != EMBEDDING_MODEL]

    # 逐一尝试 HuggingFace 模型
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
    """
    加载已有的 ChromaDB 向量库，或基于文档新建一个。

    关键修复：始终显式传入 embedding_function，
    避免 Chroma 触发内置 ONNX 模型的自动下载（会超时）。

    当 embeddings 为 None 时（所有模型均不可用），跳过向量库构建，
    系统降级为纯 LLM 回答，不影响启动。
    """
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    from langchain_chroma import Chroma

    embeddings = get_embeddings()

    # ── 核心修复：嵌入模型不可用时直接跳过，不让 Chroma 自己去下载 ONNX ──
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
            embedding_function=embeddings,   # 显式传入，避免 Chroma 自动下载 ONNX
            collection_metadata={"hnsw:space": "cosine"},
        )
        if _vectorstore._collection.count() == 0:
            logger.warning("向量库为空，将重新构建")
            _vectorstore = None
            if not documents:
                logger.warning("向量库为空但未提供文档，跳过构建")
                return None
            # 直接在此处构建，不递归调用自身（避免无限递归）
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
            embedding=embeddings,            # 显式传入，避免 Chroma 自动下载 ONNX
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info("向量库构建完成，共 %d 篇文档", len(documents))
    else:
        logger.warning("未找到已有向量库，也未提供文档，向量库初始化跳过")
        return None

    return _vectorstore


def get_retriever(k: int = 8, fetch_k: int = 30):
    """
    返回向量库的 MMR 检索器；向量库不可用时返回 None（系统自动降级）。

    参数
    ----
    k       : 最终返回文档数（默认8），匹配 RERANKER_TOP_K=5，让 reranker 做 8→5 精排
    fetch_k : MMR 候选池大小（默认30）

    lambda_mult 说明
    ----------------
    0.65 表示 65% 相关性 + 35% 多样性。
    较默认的 0.7 略降以增加多样性，减少召回结果同质化。
    """
    vs = get_or_create_vectorstore()
    if vs:
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.65}
        )
    return None


def check_coverage(query: str, threshold: float = 0.3) -> dict:
    """
    检测查询是否在知识库覆盖范围内。
    rerank_score 极低（<threshold）说明这是知识库盲区，
    需要路由到外部搜索（Tavily/Wikipedia）而不是徒劳扩展查询。

    返回
    ----
    {"covered": bool, "top_score": float, "suggestion": str}
    """
    from app.tools.reranker import rerank_documents
    vs = get_or_create_vectorstore()
    if not vs:
        return {"covered": False, "top_score": 0.0, "suggestion": "向量库不可用"}

    # 快速召回 3 篇，不需要多
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