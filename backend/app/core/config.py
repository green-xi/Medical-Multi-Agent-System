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
CHAT_DB_PATH = _env(
    "CHAT_DB_PATH",
    os.path.join(_BACKEND_DIR, "storage", "chat_db", "medicalai.db"),
)
VECTOR_STORE_DIR = _env(
    "VECTOR_STORE_DIR", os.path.join(_BACKEND_DIR, "storage", "vector_store")
)
PDF_PATH = _env("PDF_PATH", os.path.join(_BACKEND_DIR, "data", "medical_book.pdf"))

DASHSCOPE_API_KEY = _env("DASHSCOPE_API_KEY")
TAVILY_API_KEY = _env("TAVILY_API_KEY")


EMBEDDING_BACKEND = _env("EMBEDDING_BACKEND", "huggingface").lower()


EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "")

OPENWEATHERMAP_API_KEY = _env("OPENWEATHERMAP_API_KEY")
DRUG_API_KEY = _env("DRUG_API_KEY")
TOOL_AGENT_ENABLED = _env("TOOL_AGENT_ENABLED", "true").lower() == "true"


RERANKER_MODEL = _env("RERANKER_MODEL", "")
RERANKER_TOP_K = _env_int("RERANKER_TOP_K", 5)   # 精排后送入 LLM 的文档数（原3→5：增加上下文覆盖，提升 Faithfulness）


SESSION_TTL_SECONDS = _env_int("SESSION_TTL_SECONDS", 3600)
MAX_ACTIVE_SESSIONS = _env_int("MAX_ACTIVE_SESSIONS", 200)


MCP_ENABLED = _env("MCP_ENABLED", "false").lower() == "true"
MCP_SERVER_URL = _env("MCP_SERVER_URL", "http://localhost:8001/sse")

