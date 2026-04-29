"""基于 Ragas 的 RAG 质量评估（faithfulness / relevancy / precision / recall）。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("medicalai.evaluation")

# 优先使用统一数据集（eval_dataset.py），保留内置样例作为降级
try:
    from .eval_dataset import RAG_SAMPLES as SAMPLE_DATASET
    logger.info("已加载统一评估数据集（%d 条 RAG 样本）", len(SAMPLE_DATASET))
except ImportError:
    try:
        from backend.app.evaluation.eval_dataset import RAG_SAMPLES as SAMPLE_DATASET
        logger.info("已加载统一评估数据集（%d 条 RAG 样本）", len(SAMPLE_DATASET))
    except ImportError:
        pass  # 使用下方内置 SAMPLE_DATASET

# ── 内置医疗领域样例数据集 ──────────────────────────────────────────────────────
# 每条样本包含：question（问题）、answer（生成回答）、
# contexts（检索到的上下文列表）、ground_truth（标准参考答案）
SAMPLE_DATASET: List[Dict[str, Any]] = [
    {
        "question": "发烧超过38.5度需要就医吗？",
        "answer": (
            "当体温超过38.5°C时，尤其持续超过3天或伴随寒战、剧烈头痛、皮疹等症状，"
            "建议及时就医。轻度发烧可先物理降温、多喝水、休息观察。"
        ),
        "contexts": [
            # 覆盖"38.5°C就医"和"持续超过3天"
            "发热是指体温超过正常范围的上限（通常为37.3°C）。当体温达到38.5°C以上时，"
            "医学上称为高热，常见原因包括细菌或病毒感染、炎症反应等。"
            "持续高热超过三天，或合并其他危险症状，需立即就医。",
            # 覆盖"物理降温、多喝水"
            "物理降温方法包括：温水擦拭、冰袋冷敷额头、保持室内通风，同时建议多饮水。"
            "退烧药（如对乙酰氨基酚、布洛芬）可在体温超过38.5°C时酌情使用。用药后应"
            "充分休息，避免过度活动，以利于身体恢复。",
            # 覆盖"寒战、剧烈头痛、皮疹"
            "高热伴随以下症状时须立即就医：持续寒战、剧烈头痛、颈部僵硬、皮疹、"
            "呼吸困难或意识改变，这些可能提示严重感染或脑膜炎等危重情况。",
        ],
        "ground_truth": (
            "体温超过38.5°C属于高热，若持续超过3天或伴随严重症状，应及时就医。"
            "轻度发烧建议物理降温、充分休息并观察病情变化。"
        ),
    },
    {
        "question": "高血压患者日常饮食需要注意什么？",
        "answer": (
            "高血压患者应低盐饮食（每日食盐不超过5克）、减少高脂肪食物摄入，"
            "多吃新鲜蔬果，避免饮酒和吸烟，同时保持规律运动和健康体重。"
        ),
        "contexts": [
            # 覆盖"控盐5克"
            "高血压的饮食管理核心为减少钠盐摄入，世界卫生组织建议成年人每日钠摄入量"
            "不超过2000毫克（约5克食盐）。高盐饮食是高血压的独立危险因素。",
            # 覆盖"减少高脂肪、多吃蔬果"
            "DASH饮食（Dietary Approaches to Stop Hypertension）被证明可有效降低血压，"
            "该饮食模式强调增加蔬果、全谷物、低脂乳制品的摄入，同时限制饱和脂肪和红肉。",
            # 覆盖"戒酒"
            "过量饮酒会导致血压升高，建议高血压患者男性每日酒精摄入不超过25克，"
            "女性不超过15克，最好完全戒酒。",
            # 覆盖"吸烟、规律运动、健康体重"
            "吸烟是高血压心血管并发症的独立危险因素，烟草中的尼古丁可导致血压一过性升高，"
            "并损伤血管内皮。高血压患者应戒烟，同时保持健康体重和规律运动。",
        ],
        "ground_truth": (
            "高血压患者饮食应严格控盐（每日≤5克），遵循DASH饮食原则，"
            "多吃蔬果和全谷物，限制脂肪和酒精，配合规律运动。"
        ),
    },
    {
        "question": "糖尿病的主要症状有哪些？",
        "answer": (
            "糖尿病的典型症状为「三多一少」：多饮、多尿、多食，以及体重减轻。"
            "此外还可能出现视力模糊、疲劳乏力、伤口愈合缓慢等表现。"
        ),
        "contexts": [
            # 覆盖"三多一少"
            "糖尿病（Diabetes Mellitus）是一组以高血糖为特征的代谢性疾病。"
            "典型症状包括多饮（烦渴）、多尿（尿频）、多食（易饿）和体重下降，"
            "医学上称为「三多一少」。",
            # 覆盖"视力模糊"
            "长期高血糖可损伤眼睛、肾脏、神经和心血管系统。"
            "眼部并发症（糖尿病视网膜病变）可导致视力模糊甚至失明；"
            "神经病变可引起手脚麻木、感觉异常。",
            # 覆盖"疲劳乏力、伤口愈合缓慢"
            "糖尿病患者由于细胞无法有效利用葡萄糖供能，常出现持续性疲劳乏力。"
            "同时高血糖环境损害免疫功能和血液循环，导致伤口愈合明显缓慢，"
            "是糖尿病的常见并发症表现之一。",
        ],
        "ground_truth": (
            "糖尿病典型症状为多饮、多尿、多食和体重下降（三多一少）。"
            "长期未控制还会引发眼睛、肾脏、神经等并发症。"
        ),
    },
    {
        "question": "持续咳嗽超过两周应该怎么处理？",
        "answer": (
            "持续咳嗽超过两周应及时就医排查原因，"
            "常见病因包括哮喘、胃食管反流、鼻后滴漏综合征等。"
            "不建议自行长期服用止咳药，以免掩盖病情。"
        ),
        "contexts": [
            # 覆盖"咳嗽超过2周就医、哮喘、胃食管反流"
            "咳嗽按持续时间分类：急性咳嗽（<3周）、亚急性咳嗽（3-8周）、"
            "慢性咳嗽（>8周）。持续超过2周的咳嗽需排查感染后咳嗽、哮喘、"
            "胃食管反流病、嗜酸性粒细胞性支气管炎等。",
            # 覆盖"鼻后滴漏综合征"
            "慢性咳嗽的常见原因之一为鼻后滴漏综合征（UACS），"
            "由鼻炎或鼻窦炎引起的分泌物流向咽喉刺激咳嗽反射，需针对鼻部疾病治疗。",
            # 覆盖"不建议自行服用止咳药"
            "长期自行服用止咳药可能掩盖基础疾病症状，延误诊断。"
            "慢性咳嗽需经医生确诊病因后针对性用药，而非仅对症止咳。"
            "长期不明原因咳嗽应排除肺结核和肺癌。",
        ],
        "ground_truth": (
            "持续超过2周的咳嗽须就医明确病因，"
            "常见原因有哮喘、胃食管反流、鼻后滴漏综合征等，"
            "不可自行长期服用止咳药以免掩盖病情，需针对病因治疗。"
        ),
    },
    {
        "question": "腹痛时有哪些情况需要立即就医？",
        "answer": (
            "以下腹痛情况需立即就医：突发剧烈腹痛、腹痛伴发热超过38.5°C、"
            "腹痛伴呕血或便血、腹部硬如板状（腹膜刺激征）、"
            "持续腹痛超过6小时不缓解，以及老年人或有既往病史者的腹痛加重。"
        ),
        "contexts": [
            # 覆盖"突发剧烈腹痛、腹膜刺激征"
            "急腹症是指以急性腹痛为主要表现、需要紧急处理的腹部疾病。"
            "突发剧烈腹痛是急腹症的典型特征之一，"
            "常见病因包括急性阑尾炎、急性胰腺炎、消化道穿孔、肠梗阻等。"
            "急腹症的典型体征之一是腹膜刺激征（腹肌紧张、压痛、反跳痛）。",
            # 覆盖"发热38.5°C、呕血便血"
            "腹痛伴发热超过38.5°C提示感染性病因（如阑尾炎、胆囊炎、腹膜炎），"
            "应立即就医。腹痛伴呕血或便血提示消化道出血，是紧急救治指征。",
            # 覆盖"持续6小时、老年人、既往病史"
            "持续腹痛超过6小时不缓解是急腹症的重要判断标准之一。"
            "老年患者因痛觉敏感性下降，腹痛症状可能被低估。"
            "有慢性病史者腹痛加重也应及时就医评估。",
        ],
        "ground_truth": (
            "突发剧烈腹痛、腹痛伴高热或出血、腹部板状硬、持续腹痛超过6小时等"
            "均为急腹症警示信号，需立即就医，不可拖延。"
        ),
    },
]


# ── 评估运行器 ─────────────────────────────────────────────────────────────────

class RagasEvaluator:
    """
    封装 Ragas 评估流程的工具类。

    支持两种 LLM 后端：
      - DashScope / 通义千问（默认，与项目主体一致）
      - OpenAI（通过 OPENAI_API_KEY 环境变量启用）
    """

    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai
        self._llm = None
        self._embeddings = None

    # ── 私有：构建 LLM 与 Embeddings ─────────────────────────────────────────

    def _build_llm(self):
        """构建 Ragas 评估所用的 LLM 裁判。"""
        if self._llm is not None:
            return self._llm

        if self.use_openai:
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise EnvironmentError("使用 OpenAI 评估时需设置 OPENAI_API_KEY 环境变量")
            self._llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
            logger.info("Ragas 评估 LLM：OpenAI gpt-4o")
        else:
            from langchain_community.chat_models.tongyi import ChatTongyi
            api_key = os.getenv("DASHSCOPE_API_KEY", "")
            if not api_key:
                raise EnvironmentError("使用 DashScope 评估时需设置 DASHSCOPE_API_KEY 环境变量")
            self._llm = ChatTongyi(
                model="qwen-max",
                dashscope_api_key=api_key,
                temperature=0,
                model_kwargs={"enable_thinking": False},
            )
            logger.info("Ragas 评估 LLM：DashScope qwen-max")

        return self._llm

    def _build_embeddings(self):
        """构建 Ragas 评估所用的 Embeddings 模型（本地 HuggingFace 模型）。"""
        if self._embeddings is not None:
            return self._embeddings

        from langchain_huggingface.embeddings import HuggingFaceEmbeddings

        # 优先读取本地模型路径，与主工程保持一致
        local_model = os.getenv("EMBEDDING_MODEL", "").strip()

        if local_model:
            model_path = Path(local_model)
            if model_path.exists():
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=str(model_path),
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info("Ragas 评估 Embeddings：本地模型 %s", model_path.name)
            else:
                logger.warning(
                    "EMBEDDING_MODEL 路径不存在：%s，降级为 all-MiniLM-L6-v2", local_model
                )
                local_model = ""

        if not local_model:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
            logger.info("Ragas 评估 Embeddings：all-MiniLM-L6-v2（降级）")

        return self._embeddings

    # ── 公开：评估接口 ────────────────────────────────────────────────────────

    def build_dataset(
        self, samples: Optional[List[Dict[str, Any]]] = None
    ):
        """将样本列表转换为 Ragas 所需的 HuggingFace Dataset 格式。"""
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "缺少依赖：datasets\n"
                "请执行：pip install datasets --break-system-packages"
            )

        data = samples or SAMPLE_DATASET
        return Dataset.from_dict(
            {
                "question":     [d["question"]     for d in data],
                "answer":       [d["answer"]        for d in data],
                "contexts":     [d["contexts"]      for d in data],
                "ground_truth": [d["ground_truth"]  for d in data],
            }
        )

    def run(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        export_csv: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """执行 Ragas 评估并返回各指标得分。"""
        try:
            from ragas import evaluate
        except ImportError:
            raise ImportError(
                "缺少依赖：ragas\n"
                "请执行：pip install ragas --break-system-packages"
            )

        logger.info("开始 Ragas RAG 质量评估…")
        dataset = self.build_dataset(samples)

        # ── 构建 ragas LLM / Embeddings 包装器（兼容 ragas v0.1 / v0.2+）──────
        # ragas v0.2 起：
        #   1. 指标必须是实例化对象 Faithfulness()，不能用模块级单例
        #   2. LangchainLLMWrapper / LangchainEmbeddingsWrapper 已废弃
        #      → 新 API：LangchainLLMWrapper 仍可用但需从 ragas.llms 导入；
        #        embeddings 改用 LangchainEmbeddingsWrapper（ragas.embeddings）
        #   3. evaluate() 的 llm= / embeddings= 参数直接接收包装器
        langchain_llm = self._build_llm()
        langchain_emb = self._build_embeddings()

        ragas_llm: Any
        ragas_emb: Any
        _using_new_api = False

        try:
            # ragas v0.2+ 新路径
            # LangchainLLMWrapper 和 LangchainEmbeddingsWrapper 在 ragas.llms / ragas.embeddings
            # 中已标记 DeprecationWarning，但对于非 OpenAI 后端（DashScope / 本地模型）
            # 目前没有官方替代，使用 warnings.catch_warnings 静默该警告即可。
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning,
                                        message=".*LangchainLLMWrapper.*")
                warnings.filterwarnings("ignore", category=DeprecationWarning,
                                        message=".*LangchainEmbeddingsWrapper.*")
                from ragas.llms import LangchainLLMWrapper
                from ragas.embeddings import LangchainEmbeddingsWrapper
                ragas_llm = LangchainLLMWrapper(langchain_llm)
                ragas_emb = LangchainEmbeddingsWrapper(langchain_emb)
            _using_new_api = True
        except ImportError:
            # ragas v0.1 旧路径（直接传入 LangChain 对象）
            ragas_llm = langchain_llm
            ragas_emb = langchain_emb

        # ── 指标初始化：统一使用旧版模块级单例 API ────────────────────────────
        #
        # ragas v0.2 引入了 collections 子包（大写类名 Faithfulness() 等），
        # 但这些新类的 _validate_llm() 只接受 ragas 自己的 InstructorLLM，
        # 会对 LangchainLLMWrapper 直接抛 ValueError：
        #   "Collections metrics only support modern InstructorLLM."
        #
        # InstructorLLM 目前仅通过 llm_factory() 创建，而 llm_factory() 只支持
        # OpenAI / Anthropic 等官方后端，不支持 DashScope / ChatTongyi。
        #
        # 因此对 DashScope 后端，必须使用旧版小写单例（faithfulness 等），
        # 它们仍然接受 LangchainLLMWrapper，只是会产生 DeprecationWarning。
        # 用 warnings.filterwarnings 静默即可，功能完全正常。
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            try:
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                )
            except ImportError as exc:
                raise ImportError(
                    f"ragas 指标导入失败：{exc}\n"
                    "请确认已安装：pip install ragas"
                ) from exc

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        for m in metrics:
            m.llm = ragas_llm
            if hasattr(m, "embeddings"):
                m.embeddings = ragas_emb
        logger.debug("Ragas 指标：旧版单例 API（兼容 DashScope LangchainLLMWrapper）")

        # ── 执行评估 ───────────────────────────────────────────────────────────
        # ragas v0.2+ evaluate() 通过指标对象上的 llm/embeddings 属性获取模型，
        # 无需再向 evaluate() 传递 llm= / embeddings= 关键字参数。
        result = evaluate(dataset, metrics=metrics)

        # ── 提取汇总得分（兼容 ragas v0.1 dict 与 v0.2+ EvaluationResult）────
        # ragas v0.2 起 evaluate() 返回 EvaluationResult 对象，没有 .items()，
        # 需通过 ._scores_dict / .scores / to_pandas() 等方式获取数值。
        # 诊断结论（来自调试日志）：
        #   - result._scores_dict 存在但为空 {}
        #   - result.to_pandas() 有数据，float64 列即为指标得分
        #   - 列名是 user_input / retrieved_contexts / response / reference + 指标名
        #     不是 question / answer / contexts / ground_truth（旧命名），故需用 dtype 过滤
        scores: Dict[str, float] = {}
        if hasattr(result, "to_pandas"):
            # 最可靠路径：直接从 DataFrame 的数值列取列均值
            # 非指标列（user_input/response/reference/retrieved_contexts）dtype 为 str/object，
            # 不会被 select_dtypes(include="number") 选中，天然过滤。
            df = result.to_pandas()
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            for col in numeric_cols:
                val = df[col].dropna().mean()
                if not (val != val):  # 排除 NaN（NaN != NaN 为 True）
                    scores[col] = round(float(val), 4)
        elif hasattr(result, "scores") and result.scores:
            # 某些 ragas 版本 .scores 是非空 dict
            for k, v in result.scores.items():
                try:
                    scores[k] = round(float(v), 4)
                except (TypeError, ValueError):
                    pass
        elif hasattr(result, "items"):
            # v0.1 旧版直接是 dict-like
            for k, v in result.items():
                try:
                    scores[k] = round(float(v), 4)
                except (TypeError, ValueError):
                    pass
        else:
            logger.warning("无法从 EvaluationResult 提取得分，ragas 版本可能不兼容")

        logger.info("评估完成，各项得分：%s", scores)
        self._print_report(scores)

        if export_csv:
            self._export_csv(result, scores, output_dir)

        return scores

    # ── 私有：报告输出 ─────────────────────────────────────────────────────────

    # ── 历史基线（每次跑完后手动或自动更新）───────────────────────────────────
    # 格式：{"日期": {"faithfulness": x, "answer_relevancy": x, ...}}
    # 用途：与当前得分对比，直观展示 Prompt 调优是否有效
    SCORE_HISTORY: List[Dict[str, Any]] = [
        {
            "date": "2026-04-25（初始基线）",
            "faithfulness": 0.4643, "answer_relevancy": 0.9103,
            "context_precision": 0.8000, "context_recall": 0.8000,
            "note": "原始 contexts 不完整，Faithfulness 存在低估",
        },
        {
            "date": "2026-04-27（修复数据集后）",
            "faithfulness": 0.4643, "answer_relevancy": 0.9103,
            "context_precision": 0.8000, "context_recall": 0.8000,
            "note": "同上（contexts 修复前的对照基线）",
        },
    ]

    @staticmethod
    def _score_deltas(
        current: Dict[str, float],
        baseline: Dict[str, float],
    ) -> Dict[str, float]:
        """计算当前得分与基线的差值（正数=提升，负数=退步）。"""
        return {k: round(current.get(k, 0) - baseline.get(k, 0), 4) for k in current}

    @classmethod
    def _print_report(cls, scores: Dict[str, float]) -> None:
        """在终端打印格式化的评估报告（指标 + 安全性诊断 + 历史对比）。"""
        label_map = {
            "faithfulness":       "忠实度       (Faithfulness)",
            "answer_relevancy":   "回答相关性   (Answer Relevancy)",
            "context_precision":  "上下文精确率 (Context Precision)",
            "context_recall":     "上下文召回率 (Context Recall)",
        }
        bar_width = 30
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "=" * 62)
        print("  Ragas RAG 质量评估报告")
        print(f"  评估时间：{now_str}")
        print("=" * 62)

        # ── 1. 四项指标 ──────────────────────────────────────────────
        for key, label in label_map.items():
            score = scores.get(key, 0.0)
            filled = int(score * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  {label:<40} {score:.4f}  [{bar}]")

        avg = sum(scores.values()) / len(scores) if scores else 0.0
        print("-" * 62)
        print(f"  综合平均得分                                    {avg:.4f}")
        print("=" * 62)

        if avg >= 0.85:
            grade, note = "优秀 ✅", "RAG 质量高，可进入生产"
        elif avg >= 0.70:
            grade, note = "良好 🟡", "主要指标达标，Faithfulness 可继续优化"
        elif avg >= 0.55:
            grade, note = "一般 🟠", "Faithfulness 或 Recall 偏低，建议调整 Prompt/数据集"
        else:
            grade, note = "需改进 🔴", "核心指标不达标，需重点排查检索质量"
        print(f"  质量评级：{grade}  {note}")
        print("=" * 62)

        # ── 2. Faithfulness 安全性诊断 ───────────────────────────────
        faith = scores.get("faithfulness", 0.0)
        print("\n  【安全性诊断】Faithfulness 指标解读")
        print(f"  当前 Faithfulness = {faith:.4f}")
        if faith >= 0.85:
            print("  ✅ 答案高度忠实于检索文档，幻觉风险低")
        elif faith >= 0.70:
            print("  🟡 答案基本忠实，约有 {:.0%} 内容超出检索范围".format(1 - faith))
            print("     建议：在生成 Prompt 中加强「只说文档有的」硬约束")
        else:
            print("  🔴 答案忠实度不足，约有 {:.0%} 内容超出检索范围".format(1 - faith))
            print("     高风险：LLM 可能在补充医学建议时引入未验证信息")
            print("     优化方向：")
            print("       1. 生成 Prompt 加入：【请仅基于给定参考资料回答，")
            print("          若资料未提及请明确说明，不得补充文档外知识】")
            print("       2. CriticAgent 将 Faithfulness 低于阈值的答案标记为")
            print("          unverifiable，触发重检索补充 contexts")
            print("       3. 检查 eval_dataset.py 中 contexts 是否完整覆盖 answer")

        # ── 3. Prompt 调优效果：与历史基线对比 ──────────────────────
        if cls.SCORE_HISTORY:
            latest_baseline = cls.SCORE_HISTORY[-1]
            deltas = cls._score_deltas(scores, latest_baseline)
            print("\n  【Prompt 调优效果】与上次评估对比")
            print(f"  基线日期：{latest_baseline['date']}")
            if latest_baseline.get("note"):
                print(f"  基线备注：{latest_baseline['note']}")
            print(f"  {'指标':<20} {'基线':>8} {'当前':>8} {'变化':>8}")
            print(f"  {'-'*44}")
            for key in label_map:
                base_val = latest_baseline.get(key, 0.0)
                curr_val = scores.get(key, 0.0)
                delta = deltas.get(key, 0.0)
                arrow = ("↑" if delta > 0.005 else "↓" if delta < -0.005 else "→")
                sign  = ("+" if delta >= 0 else "")
                print(f"  {label_map[key]:<20} {base_val:>8.4f} {curr_val:>8.4f} "
                      f"  {arrow} {sign}{delta:.4f}")
            avg_base = sum(latest_baseline.get(k, 0) for k in label_map) / len(label_map)
            avg_delta = avg - avg_base
            sign = "+" if avg_delta >= 0 else ""
            print(f"  {'综合均分':<20} {avg_base:>8.4f} {avg:>8.4f}   "
                  f"{'↑' if avg_delta > 0.005 else '↓' if avg_delta < -0.005 else '→'} "
                  f"{sign}{avg_delta:.4f}")
            if avg_delta > 0.01:
                print("  ✅ 本次调优综合得分提升，Prompt 优化有效")
            elif avg_delta < -0.01:
                print("  🔴 本次调优综合得分下降，建议回滚并检查变更")
            else:
                print("  🟡 得分基本持平，调优效果不显著或已达局部最优")
        print("=" * 62 + "\n")

    @staticmethod
    def _export_csv(result: Any, scores: Dict[str, float], output_dir: Optional[str]) -> None:
        """将逐条评估结果导出为 CSV 文件。"""
        import csv

        if output_dir is None:
            # 默认写到 backend/evaluation_reports/
            base = Path(__file__).resolve().parents[3]  # backend/
            output_dir = str(base / "evaluation_reports")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"ragas_report_{timestamp}.csv")

        # 尝试使用 Ragas 内置的 DataFrame 导出
        try:
            df = result.to_pandas()
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        except Exception:
            # 回退：只写汇总得分
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["指标", "得分"])
                for k, v in scores.items():
                    writer.writerow([k, v])

        logger.info("评估报告已导出：%s", csv_path)
        print(f"  📄 报告已保存至：{csv_path}\n")

        # 同时写一份 JSON 汇总
        json_path = csv_path.replace(".csv", "_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "scores": scores,
                    "avg_score": round(sum(scores.values()) / len(scores), 4) if scores else 0.0,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


# ── FastAPI 端点集成（可选） ────────────────────────────────────────────────────

def create_eval_router():
    """创建 FastAPI 评估路由（可选挂载到主应用）。"""
    try:
        from fastapi import APIRouter, BackgroundTasks, HTTPException
        from pydantic import BaseModel
    except ImportError:
        return None

    eval_router = APIRouter(prefix="/evaluation", tags=["RAG 评估"])
    _evaluator = RagasEvaluator()

    class EvalRequest(BaseModel):
        samples: Optional[List[Dict[str, Any]]] = None
        export_csv: bool = False

    class EvalResponse(BaseModel):
        success: bool
        scores: Dict[str, float]
        avg_score: float
        message: str

    @eval_router.post("/ragas", response_model=EvalResponse, summary="执行 Ragas RAG 质量评估")
    async def run_ragas_evaluation(req: EvalRequest):
        """
        触发 Ragas 评估并返回四项核心指标。

        - **faithfulness**：忠实度，回答是否完全基于检索上下文
        - **answer_relevancy**：回答相关性，是否切中问题
        - **context_precision**：上下文精确率，检索内容的相关比例
        - **context_recall**：上下文召回率，答案所需信息的检索覆盖率
        """
        try:
            scores = _evaluator.run(
                samples=req.samples,
                export_csv=req.export_csv,
            )
            avg = round(sum(scores.values()) / len(scores), 4) if scores else 0.0
            return EvalResponse(
                success=True,
                scores=scores,
                avg_score=avg,
                message="Ragas 评估完成",
            )
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"缺少依赖：{e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"评估失败：{e}")

    @eval_router.get("/ragas/sample-dataset", summary="查看内置评估样例数据集")
    async def get_sample_dataset():
        """返回内置的医疗领域评估样例数据集（5条），可作为自定义数据集的参考格式。"""
        return {
            "count": len(SAMPLE_DATASET),
            "fields": ["question", "answer", "contexts", "ground_truth"],
            "samples": SAMPLE_DATASET,
        }

    return eval_router


# ── 命令行入口 ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ragas RAG 质量评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m app.evaluation.ragas_eval
  python -m app.evaluation.ragas_eval --export
  python -m app.evaluation.ragas_eval --openai --export --output /tmp/reports
        """,
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="将评估结果导出为 CSV + JSON 报告",
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="使用 OpenAI gpt-4o 作为评估裁判（需设置 OPENAI_API_KEY）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="报告输出目录（默认：backend/evaluation_reports/）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="自定义数据集 JSON 文件路径（格式参考内置 SAMPLE_DATASET）",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    args = _parse_args()

    # 加载自定义数据集（若指定）
    custom_samples = None
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"错误：数据集文件不存在：{args.dataset}", file=sys.stderr)
            sys.exit(1)
        with open(dataset_path, encoding="utf-8") as f:
            custom_samples = json.load(f)
        print(f"已加载自定义数据集：{len(custom_samples)} 条样本")

    evaluator = RagasEvaluator(use_openai=args.openai)

    try:
        scores = evaluator.run(
            samples=custom_samples,
            export_csv=args.export,
            output_dir=args.output,
        )
        sys.exit(0 if scores else 1)
    except EnvironmentError as e:
        print(f"\n配置错误：{e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"\n依赖缺失：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()