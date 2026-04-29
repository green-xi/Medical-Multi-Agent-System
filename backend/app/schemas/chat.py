"""
MedicalAI — schemas/chat.py
Pydantic schemas for chat request and response.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    source: str
    timestamp: str
    success: bool
    session_id: str = ""
    thinking_steps: list = []        # 查询重写的思考步骤
    original_question: str = ""      # 用户原始问题
    query_intent: str = ""           # 意图分类
    rag_think_log: list = []         # RAG 迭代评估日志
    tool_trace: list = []            # 节点执行路径