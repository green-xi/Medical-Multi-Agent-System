"""
MedicalAI — evaluation 模块

子模块
------
eval_dataset  : 统一评估数据集（RAG_SAMPLES / AGENT_CASES / CRITIC_CASES）
ragas_eval    : RAG 检索质量评估（Faithfulness / Relevancy / Precision / Recall）
agent_eval    : Agent 编排行为评估 + CriticAgent 单元评估

快速使用
--------
  # RAG 质量评估
  python -m app.evaluation.ragas_eval --export

  # Agent 行为评估（mock 模式，无需 API）
  python -m app.evaluation.agent_eval --mode agent --mock

  # Agent 行为评估（真实模式）
  python -m backend.app.evaluation.agent_eval --mode agent --real

  # CriticAgent 单元评估
  python -m app.evaluation.agent_eval --mode critic

  # 全部评估并导出
  python -m app.evaluation.agent_eval --mode all --export
"""

# 延迟导入：避免从项目根目录运行时 sys.path 尚未包含 backend/，
# 导致 from app.xxx import 在模块级失败。
# 需要时请直接 from app.evaluation.eval_dataset import xxx

# from .eval_dataset import AGENT_CASES, CRITIC_CASES, RAG_SAMPLES
# #from app.evaluation.eval_dataset import AGENT_CASES, CRITIC_CASES, RAG_SAMPLES

# __all__ = ["RAG_SAMPLES", "AGENT_CASES", "CRITIC_CASES"]