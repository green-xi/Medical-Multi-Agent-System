"""环境变量与路径配置。"""

import os

from dotenv import load_dotenv

load_dotenv(override=True)


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key, default)
    return value.strip().rstrip("\r") if isinstance(value, str) else default


def _env_int(key: str, default: int) -> int:
    raw_value = _env(key, str(default))
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

LOG_DIR = _env("LOG_DIR", os.path.join(_BACKEND_DIR, "logs"))
CHAT_DB_PATH = _env("CHAT_DB_PATH", os.path.join(_BACKEND_DIR, "storage", "chat_db", "medicalai.db"))
VECTOR_STORE_DIR = _env("VECTOR_STORE_DIR", os.path.join(_BACKEND_DIR, "storage", "vector_store"))
PDF_PATH = _env("PDF_PATH", os.path.join(_BACKEND_DIR, "data", "medical_book.pdf"))

DASHSCOPE_API_KEY = _env("DASHSCOPE_API_KEY")
TAVILY_API_KEY = _env("TAVILY_API_KEY")

MCP_ENABLED = _env("MCP_ENABLED", "true").lower() == "true"

EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "")
RERANKER_MODEL = _env("RERANKER_MODEL", "")
RERANKER_TOP_K = _env_int("RERANKER_TOP_K", 5)

SESSION_TTL_SECONDS = _env_int("SESSION_TTL_SECONDS", 3600)
MAX_ACTIVE_SESSIONS = _env_int("MAX_ACTIVE_SESSIONS", 200)
