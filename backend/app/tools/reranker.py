"""CrossEncoder Reranker：对检索候选文档重排，提升送入 LLM 的文档质量。"""

from __future__ import annotations
from time import perf_counter
from typing import List, Optional

from langchain_core.documents import Document

from app.core.config import RERANKER_MODEL, RERANKER_TOP_K
from app.core.logging_config import logger


_local_model = None

_LOCAL_RERANKER_CANDIDATES = [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
]


def _is_local_path(name: str) -> bool:
    import os
    if os.path.isabs(name):
        return True
    if name.startswith(("./", ".\\", "../", "..\\")):
        return True
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        return True
    return False


def _get_local_model(model_name: Optional[str] = None):
    global _local_model
    if _local_model is not None:
        return _local_model

    import os
    from sentence_transformers import CrossEncoder
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

    candidates = ([model_name] if model_name else []) + _LOCAL_RERANKER_CANDIDATES
    seen = set()
    candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]

    for name in candidates:
        if _is_local_path(name):
            load_path = name
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
                    logger.warning("本地 Reranker 路径 %s 下未找到 config.json，跳过", name)
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


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
) -> List[Document]:
    if not documents:
        return documents

    if RERANKER_MODEL.lower() in ("none", "disabled"):
        return documents[:top_k or RERANKER_TOP_K]

    k = top_k or RERANKER_TOP_K
    k = min(k, len(documents))
    t0 = perf_counter()

    local_model_name = RERANKER_MODEL or None
    try:
        result = _local_rerank(
            query=query, documents=documents, top_k=k, model_name=local_model_name,
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

    logger.info("Rerank 跳过（无可用 Reranker），返回原始 Top-%d", k)
    return documents[:k]


def get_reranker_status() -> dict:
    has_local = _local_model is not None
    return {
        "local_loaded": has_local,
        "reranker_model": RERANKER_MODEL or "auto",
        "top_k": RERANKER_TOP_K,
        "mode": "local" if has_local else "disabled",
    }
