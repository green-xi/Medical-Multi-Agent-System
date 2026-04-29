"""评估模块。直接使用子模块：from app.evaluation.ragas_eval import xxx"""

# 延迟导入：避免从项目根目录运行时 sys.path 尚未包含 backend/，
# 导致 from app.xxx import 在模块级失败。
# 需要时请直接 from app.evaluation.eval_dataset import xxx

# from .eval_dataset import AGENT_CASES, CRITIC_CASES, RAG_SAMPLES
# #from app.evaluation.eval_dataset import AGENT_CASES, CRITIC_CASES, RAG_SAMPLES

# __all__ = ["RAG_SAMPLES", "AGENT_CASES", "CRITIC_CASES"]