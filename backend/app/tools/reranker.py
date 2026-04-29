"""
MedicalAI — tools/reranker.py
Reranker：对向量检索召回的候选文档重新精排，提升最终送入 LLM 的文档质量。

仅使用本地 CrossEncoder 模型（BGE Reranker），不支持云端 Reranker API。

调用方式
--------
  from app.tools.reranker import rerank_documents
  top_docs = rerank_documents(query="头痛怎么办", documents=docs, top_k=3)

Reranker 返回的文档已按相关性降序排列，并在 metadata 里注入 rerank_score。
"""

from __future__ import annotations
from time import perf_counter
from typing import List, Optional

from langchain_core.documents import Document

from app.core.config import RERANKER_MODEL, RERANKER_TOP_K
from app.core.logging_config import logger



# ── 本地 Cross-Encoder Reranker ───────────────────────────────────────────────

_local_model = None

# 本地模型优先级（都支持中文）
_LOCAL_RERANKER_CANDIDATES = [
    "BAAI/bge-reranker-base",          # ~280MB，中文最佳，首选
    "BAAI/bge-reranker-large",         # ~560MB，效果更好但内存占用大
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # ~80MB，英文模型，兜底
]


def _is_local_path(name: str) -> bool:
    """判断是否为本地文件系统路径（绝对路径或 ./.. 相对路径）。"""
    import os
    if os.path.isabs(name):          # /abs/path 或 C:/... F:/...
        return True
    if name.startswith(("./", ".\\", "../", "..\\")):
        return True
    # Windows 盘符：单字母 + 冒号，如 F:
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        return True
    return False


def _get_local_model(model_name: Optional[str] = None):
    """加载本地 CrossEncoder 模型（单例，延迟初始化）。"""
    global _local_model
    if _local_model is not None:
        return _local_model

    import os
    from sentence_transformers import CrossEncoder
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

    candidates = ([model_name] if model_name else []) + _LOCAL_RERANKER_CANDIDATES
    # 去重
    seen = set()
    candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]

    for name in candidates:

        # ── 本地路径分支：直接检查目录，不经过 HuggingFace Hub ──────────────
        if _is_local_path(name):
            load_path = name
            # 如果指定目录下没有 config.json，向下递归一层寻找
            # （有些下载工具会在路径下再建 namespace/model_name/ 子目录）
            if not os.path.isfile(os.path.join(load_path, "config.json")):
                found = None
                try:
                    for entry in os.scandir(load_path):
                        if entry.is_dir():
                            for sub in os.scandir(entry.path):
                                if sub.is_dir() and os.path.isfile(
                                    os.path.join(sub.path, "config.json")
                                ):
                                    found = sub.path
                                    break
                            if found:
                                break
                            if os.path.isfile(os.path.join(entry.path, "config.json")):
                                found = entry.path
                                break
                except Exception:
                    pass
                if found:
                    logger.info("本地 Reranker 路径自动修正：%s → %s", load_path, found)
                    load_path = found
                else:
                    logger.warning(
                        "本地 Reranker 路径 %s 下未找到 config.json，跳过", name
                    )
                    continue

            try:
                _local_model = CrossEncoder(load_path, max_length=512)
                _local_model.predict([("头痛怎么办", "头痛可能由多种原因引起")])
                logger.info("本地 Reranker 已加载（本地路径）：%s", load_path)
                return _local_model
            except Exception as exc:
                logger.warning("本地 Reranker %s 加载失败：%s", load_path, exc)
                _local_model = None
            continue

        # ── HuggingFace Hub repo_id 分支：检查本地缓存，不触发下载 ──────────
        cached = try_to_load_from_cache(name, "config.json")
        if cached is None or cached is _CACHED_NO_EXIST:
            logger.info("本地 Reranker %s 未缓存，跳过（不触发下载）", name)
            continue

        try:
            _local_model = CrossEncoder(name, max_length=512)
            _local_model.predict([("头痛怎么办", "头痛可能由多种原因引起")])
            logger.info("本地 Reranker 已加载：%s", name)
            return _local_model
        except Exception as exc:
            logger.warning("本地 Reranker %s 加载失败：%s", name, exc)
            _local_model = None

    logger.warning("所有本地 Reranker 加载失败，将跳过重排")
    return None


def _local_rerank(
    query: str,
    documents: List[Document],
    top_k: int,
    model_name: Optional[str] = None,
) -> List[Document]:
    """使用本地 CrossEncoder 对文档重排。"""
    model = _get_local_model(model_name)
    if model is None:
        return documents[:top_k]

    pairs = [(query, doc.page_content[:512]) for doc in documents]
    scores = model.predict(pairs)

    scored = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    reranked: List[Document] = []
    for score, doc in scored[:top_k]:
        doc.metadata["rerank_score"] = round(float(score), 4)
        doc.metadata["rerank_model"] = "local_cross_encoder"
        reranked.append(doc)

    return reranked


# ── 统一入口 ──────────────────────────────────────────────────────────────────

def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
) -> List[Document]:
    """
    对候选文档重排，返回 top_k 个按相关性降序的文档。

    仅使用本地 CrossEncoder 模型（BGE Reranker），不支持云端 Reranker API。

    参数
    ----
    query     : 用户原始问题（不是 embedding 用的 combined_query）
    documents : 向量检索召回的候选文档列表
    top_k     : 重排后保留的文档数，默认读 config（RERANKER_TOP_K）

    返回
    ----
    List[Document]，metadata 里含 rerank_score 字段（用于调试和可观测性）
    """
    if not documents:
        return documents

    # 显式禁用时直接截断返回
    if RERANKER_MODEL.lower() in ("none", "disabled"):
        return documents[:top_k or RERANKER_TOP_K]

    k = top_k or RERANKER_TOP_K
    k = min(k, len(documents))   # 不超过候选数量

    t0 = perf_counter()

    # 本地 CrossEncoder
    local_model_name = RERANKER_MODEL or None
    try:
        result = _local_rerank(
            query=query,
            documents=documents,
            top_k=k,
            model_name=local_model_name,
        )
        if result and result[0].metadata.get("rerank_score") is not None:
            logger.info(
                "本地 Rerank 完成：%d→%d 篇，耗时 %.1fms，top分=%.3f",
                len(documents), len(result),
                (perf_counter() - t0) * 1000,
                result[0].metadata.get("rerank_score", 0),
            )
            return result
    except Exception as exc:
        logger.warning("本地 Rerank 失败，跳过重排：%s", exc)

    # 降级，原样截断
    logger.info("Rerank 跳过（无可用 Reranker），返回原始 Top-%d", k)
    return documents[:k]


def get_reranker_status() -> dict:
    """返回当前 Reranker 状态，供健康检查端点使用。"""
    has_local = _local_model is not None
    return {
        "local_loaded": has_local,
        "reranker_model": RERANKER_MODEL or "auto",
        "top_k": RERANKER_TOP_K,
        "mode": "local" if has_local else "disabled",
    }