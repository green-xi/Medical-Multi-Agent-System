"""通义千问 LLM 客户端单例（DashScope）。"""

from app.core.config import DASHSCOPE_API_KEY
from app.core.logging_config import logger

_llm_instance = None


def get_llm():
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    if not DASHSCOPE_API_KEY:
        logger.warning("未找到 DASHSCOPE_API_KEY 环境变量，LLM 客户端不可用")
        return None

    try:
        from langchain_community.chat_models.tongyi import ChatTongyi
    except Exception as exc:
        logger.error("导入 ChatTongyi 失败：%s", exc)
        return None

    try:
        _llm_instance = ChatTongyi(
            model="qwen3-max",
            dashscope_api_key=DASHSCOPE_API_KEY,
            temperature=0.3,
            max_tokens=2048,
            model_kwargs={"enable_thinking": False},
        )
    except Exception as exc:
        logger.error("初始化通义客户端失败：%s", exc)
        _llm_instance = None
        return None

    logger.info("LLM 客户端初始化成功（DashScope / qwen3-max）")
    return _llm_instance
