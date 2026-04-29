"""Agent 编排行为评估 + CriticAgent 单元评估。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

logger = logging.getLogger("medicalai.evaluation.agent")


def _compat_import(module_path: str):
    """
    兼容两种运行方式的动态导入：
      - 从 backend/ 目录运行：python -m app.evaluation.agent_eval
      - 从项目根目录运行：python -m backend.app.evaluation.agent_eval

    根本问题：从项目根目录运行时，sys.path 只有项目根，
    没有 backend/ 目录，导致所有 app.xxx 导入失败。
    解决方案：导入前把 backend/ 的绝对路径注入 sys.path（幂等）。
    """
    import importlib
    import sys
    from pathlib import Path

    # __file__ = .../backend/app/evaluation/agent_eval.py
    # parents[2] = .../backend/
    backend_dir = str(Path(__file__).resolve().parents[2])
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    return importlib.import_module(module_path)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _contains_any(text: str, patterns: List[str]) -> bool:
    return any(p in text for p in patterns)


def _contains_all_required(text: str, keywords: List[str]) -> Tuple[bool, List[str]]:
    """返回 (是否全部命中, 未命中的关键词列表)。"""
    missing = [k for k in keywords if k not in text]
    return len(missing) == 0, missing


def _colorize(text: str, passed: bool) -> str:
    """终端着色（ANSI）。"""
    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
    return f"{GREEN if passed else RED}{text}{RESET}"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Agent 编排行为评估
# ══════════════════════════════════════════════════════════════════════════════

class AgentBehaviorEvaluator:
    """
    对 Agent 工作流进行端到端黑盒评估。

    两种模式
    --------
    mock=True  : 不调用真实 LLM/API，用预设状态数据测试评估逻辑本身
    mock=False : 调用真实工作流（需要配置 API Key）

    评估指标
    --------
    - intent_accuracy       意图识别准确率（QueryRewriter）
    - rag_hit_rate          RAG 命中率
    - tavily_trigger_rate   Tavily 触发率（联网检索）
    - safety_pass_rate      安全红线通过率（forbidden_patterns）
    - keyword_coverage      关键词覆盖率
    - avg_latency_s         平均响应时间（秒）
    - replan_rate           重规划触发率
    """

    def __init__(self, mock: bool = True):
        self.mock = mock
        self._service = None  # 跨用例复用，避免每次重新加载向量库和工作流

    def _run_workflow_mock(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock 运行：不调用真实 LLM，基于用例预期值构造一个合理的伪状态。
        用于测试评估脚本本身的逻辑，不代表真实系统表现。
        """
        question = case["question"]
        answer_mock = f"关于您的问题「{question}」，以下是医疗建议：" + \
                      "".join(case.get("required_keywords", ["就医", "医生"]))

        # mock 答案：模拟真实 LLM 的回复长度和关键词分布
        required_kw = case.get("required_keywords", [])
        forbidden = case.get("forbidden_patterns", [])
        # 构造一个包含所有 required_keywords、不含 forbidden_patterns 的合法答案
        kw_part = "、".join(required_kw) if required_kw else "请注意健康"
        answer_mock = (
            "根据您的描述，以下是相关医疗建议：\n"
            "建议您关注以下几点：" + kw_part + "。"
            "请结合自身情况，如症状持续或加重，应及时" + (kw_part[:4] if kw_part else "就医") + "。"
            "以上建议仅供参考，具体诊疗请遵医嘱，保持良好的生活习惯。"
        )

        return {
            "question": question,
            "generation": answer_mock,
            "query_intent": case["expected_intent"],
            "metrics": {
                "rag_hit": case["should_hit_rag"],
                "fallback_count": 1 if case["should_use_tavily"] else 0,
                "replan_count": 1 if case["should_replan"] else 0,
                "total_latency_ms": 7200.0,  # mock 固定值
                "critic_pass": True,
            },
            "tool_trace": (
                ["memory", "query_rewriter", "planner", "research", "planner", "critic"]
                if case["should_hit_rag"] else
                ["memory", "query_rewriter", "planner", "research", "planner", "critic"]
            ),
        }

    def _run_workflow_real(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        真实调用工作流（需要 API Key 和完整环境）。

        sys.path 修复
        -------------
        从项目根目录以 python -m backend.app.evaluation.agent_eval 运行时，
        sys.path 里只有项目根目录，没有 backend/ 目录，导致所有 app.xxx 导入失败。
        解决方案：在调用前动态把 backend/ 的绝对路径加入 sys.path。
        """
        import asyncio
        import sys
        from pathlib import Path

        # 把 backend/ 目录加入 sys.path（幂等：已存在则跳过）
        # __file__ = .../backend/app/evaluation/agent_eval.py
        # parents[2] = .../backend/
        backend_dir = str(Path(__file__).resolve().parents[2])
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
            logger.debug("已将 %s 加入 sys.path", backend_dir)

        try:
            from app.services.chat_service import ChatService

            # 单例复用：initialize_workflow 只在首次调用时执行（同步操作，在 asyncio.run 外）
            # 避免每个用例都重新加载向量库和 LangGraph 工作流
            if self._service is None:
                service = ChatService()
                if not service.workflow_app:
                    logger.info("初始化工作流（向量库加载中，首次约需 10-30s）…")
                    service.initialize_workflow()
                self._service = service
                logger.info("ChatService 已就绪，后续用例将复用此实例")
            service = self._service

            session_id = f"eval_{case['id']}_{int(time.time())}"

            # 兼容 "事件循环已运行" 场景（如在 Jupyter/某些测试框架中）
            coro = service.process_message(session_id=session_id, message=case["question"])
            try:
                api_result = asyncio.run(coro)
            except RuntimeError:
                # 当前线程已有运行中的事件循环（如 nest_asyncio 环境）
                loop = asyncio.get_event_loop()
                api_result = loop.run_until_complete(coro)

            # ── 映射 process_message 返回值 → 评估框架字段 ──────────────
            # process_message 返回：
            #   {response, query_intent, tool_trace, source, success, ...}
            tool_trace = api_result.get("tool_trace", [])

            # rag_hit：tool_trace 经过 research 节点且 source 不是纯联网来源
            source = api_result.get("source", "")
            rag_hit = "research" in tool_trace and source not in ("tavily", "wikipedia")

            # fallback_count：tool_trace 中含 tavily/wikipedia 说明触发了联网
            fallback_count = sum(
                1 for node in tool_trace
                if "tavily" in node.lower() or "wikipedia" in node.lower()
            )

            # replan_count：planner 出现次数 - 1（第一次是初始规划，之后是重规划）
            replan_count = max(0, tool_trace.count("planner") - 1)

            return {
                "question":     case["question"],
                "generation":   api_result.get("response", ""),
                "query_intent": api_result.get("query_intent", ""),
                "metrics": {
                    "rag_hit":        rag_hit,
                    "fallback_count": fallback_count,
                    "replan_count":   replan_count,
                    # process_message 不直接返回 latency，用外层 perf_counter 计时
                    "total_latency_ms": 0,
                    "critic_pass":    True,
                },
                "tool_trace": tool_trace,
            }

        except Exception as exc:
            logger.error("真实工作流调用失败 [%s]：%s", case.get("id"), exc)
            return {}

    def evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个用例，返回结构化结果。"""
        start = time.perf_counter()

        state = (
            self._run_workflow_mock(case)
            if self.mock
            else self._run_workflow_real(case)
        )

        latency_s = time.perf_counter() - start
        metrics = state.get("metrics", {})
        answer = state.get("generation", "")
        intent = state.get("query_intent", "")

        # ── 各维度判断 ────────────────────────────────────────────────────
        intent_ok = (intent == case["expected_intent"])

        rag_hit = metrics.get("rag_hit", False)
        rag_ok = (rag_hit == case["should_hit_rag"])

        # Tavily 触发判断：fallback_count > 0 表示触发了联网检索
        tavily_triggered = metrics.get("fallback_count", 0) > 0
        tavily_ok = (tavily_triggered == case["should_use_tavily"])

        # 安全红线：答案中不应出现的词
        forbidden = case.get("forbidden_patterns", [])
        safety_ok = not _contains_any(answer, forbidden)
        triggered_forbidden = [p for p in forbidden if p in answer]

        # 关键词覆盖
        required = case.get("required_keywords", [])
        keywords_ok, missing_kw = _contains_all_required(answer, required)

        # 答案最短长度
        length_ok = len(answer) >= case.get("min_answer_length", 0)

        # 重规划
        replan_count = metrics.get("replan_count", 0)
        replan_ok = (replan_count > 0) == case["should_replan"]

        # 综合通过：安全红线 + 关键词 + 长度必须全通过；其他维度计入分数
        case_pass = safety_ok and keywords_ok and length_ok

        result = {
            "id":          case["id"],
            "description": case.get("description", ""),
            "passed":      case_pass,
            "latency_s":   round(metrics.get("total_latency_ms", 7200) / 1000 if self.mock else latency_s, 2),
            "details": {
                "intent_accuracy":   {"ok": intent_ok,   "expected": case["expected_intent"], "actual": intent},
                "rag_hit":           {"ok": rag_ok,       "expected": case["should_hit_rag"], "actual": rag_hit},
                "tavily_triggered":  {"ok": tavily_ok,    "expected": case["should_use_tavily"], "actual": tavily_triggered},
                "safety":            {"ok": safety_ok,    "triggered": triggered_forbidden},
                "keywords":          {"ok": keywords_ok,  "missing": missing_kw},
                "answer_length":     {"ok": length_ok,    "actual": len(answer), "min": case.get("min_answer_length", 0)},
                "replan":            {"ok": replan_ok,    "count": replan_count},
            },
        }
        return result

    def run(self, cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """运行全部用例，汇总评估报告。"""
        AGENT_CASES = _compat_import('app.evaluation.eval_dataset').AGENT_CASES
        all_cases = cases or AGENT_CASES

        logger.info("开始 Agent 编排评估（mode=%s，用例数=%d）…", "mock" if self.mock else "real", len(all_cases))

        results = []
        for case in all_cases:
            r = self.evaluate_case(case)
            results.append(r)
            # logger 不支持 ANSI 转义，始终用纯文字；终端打印用着色
            log_status = "PASS" if r["passed"] else "FAIL"
            logger.info("[%s] %s — %s (%.1fs)", case["id"], log_status, case.get("description", ""), r["latency_s"])

        # 汇总
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        avg_latency = sum(r["latency_s"] for r in results) / total if total else 0
        intent_acc = sum(1 for r in results if r["details"]["intent_accuracy"]["ok"]) / total if total else 0
        rag_acc    = sum(1 for r in results if r["details"]["rag_hit"]["ok"]) / total if total else 0
        safety_rate = sum(1 for r in results if r["details"]["safety"]["ok"]) / total if total else 0
        kw_rate    = sum(1 for r in results if r["details"]["keywords"]["ok"]) / total if total else 0

        summary = {
            "total":          total,
            "passed":         passed,
            "pass_rate":      round(passed / total, 4) if total else 0,
            "intent_accuracy":  round(intent_acc, 4),
            "rag_accuracy":     round(rag_acc, 4),
            "safety_pass_rate": round(safety_rate, 4),
            "keyword_coverage": round(kw_rate, 4),
            "avg_latency_s":    round(avg_latency, 2),
            "results":          results,
        }
        self._print_report(summary)
        return summary

    @staticmethod
    def _print_report(summary: Dict[str, Any]) -> None:
        """打印 Agent 行为评估报告。"""
        print("\n" + "=" * 65)
        print("  Agent 编排行为评估报告")
        print(f"  评估时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 65)
        print(f"  用例总数：{summary['total']}　通过：{summary['passed']}　通过率：{summary['pass_rate']:.1%}")
        print("-" * 65)
        print(f"  意图识别准确率   : {summary['intent_accuracy']:.1%}")
        print(f"  RAG 路由准确率   : {summary['rag_accuracy']:.1%}")
        print(f"  安全红线通过率   : {summary['safety_pass_rate']:.1%}")
        print(f"  关键词覆盖率     : {summary['keyword_coverage']:.1%}")
        print(f"  平均响应时间     : {summary['avg_latency_s']:.1f}s")
        print("-" * 65)
        for r in summary["results"]:
            mark = "✓" if r["passed"] else "✗"
            print(f"  [{r['id']}] {mark} {r['description'][:45]:<45} {r['latency_s']:.1f}s")
            if not r["passed"]:
                d = r["details"]
                if not d["safety"]["ok"]:
                    print(f"        ⚠ 触发安全红线：{d['safety']['triggered']}")
                if not d["keywords"]["ok"]:
                    print(f"        ⚠ 缺少关键词：{d['keywords']['missing']}")
                if not d["answer_length"]["ok"]:
                    print(f"        ⚠ 答案过短：{d['answer_length']['actual']} < {d['answer_length']['min']}")
        print("=" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CriticAgent 单元评估
# ══════════════════════════════════════════════════════════════════════════════

class CriticUnitEvaluator:
    """
    对 CriticAgent 进行单元级评估。

    评估方式
    --------
    直接调用 CriticAgent 函数（传入构造好的 AgentState），
    对比预期通过/不通过与实际结果。

    核心指标
    --------
    - true_positive_rate  : 应通过的答案被正确放行的比例
    - true_negative_rate  : 应拦截的答案被正确拦截的比例
    - hallucination_accuracy : 幻觉检测准确率
    - ref_source_distribution : 参考来源分布（pubmed/rag/empty）
    - avg_latency_s       : CriticAgent 平均执行时间
    """

    def _build_mock_state(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        将评估用例转换为完整的 AgentState 格式。

        必须包含 state.py 里 record_fallback / CriticAgent 访问的所有字段，
        否则在 state["fallback_events"].append() 等处报 KeyError。
        """
        from langchain_core.documents import Document

        # 延迟导入，确保 backend/ 已在 sys.path（_compat_import 已注入）
        try:
            from app.core.state import initialize_conversation_state, default_route_decision
        except ImportError:
            from backend.app.core.state import initialize_conversation_state, default_route_decision

        docs = [
            Document(page_content=ctx, metadata={"source": f"eval_ctx_{i}"})
            for i, ctx in enumerate(case.get("contexts", []))
        ]

        # 从官方初始化函数获取完整骨架，避免手写遗漏字段
        state = initialize_conversation_state(session_id=f"eval_{case['id']}")

        # 覆盖评估相关字段
        state.update({
            "question":           case["question"],
            "original_question":  case["question"],
            "generation":         case["answer"],
            "documents":          docs,
            "critic_attempt_count": 0,
            "critic_ref_source":  "",
            # critic.py 第478行：state.get("planner_eval", {}).get("replan_count", 0)
            # 当 planner_eval 为 None 时，.get() 返回 None 而非 {}，导致 AttributeError。
            # 评估场景不需要真实的 Planner 状态，给一个合法空 dict 即可。
            "planner_eval": {
                "satisfied":     True,
                "reason":        "eval_bypass",
                "replan_action": "",
                "replan_count":  0,
            },
            # critic_result 同理，初始化为空 dict 避免后续 .get() 在 None 上调用
            "critic_result": None,
            "metrics": {
                "total_latency_ms":     0.0,
                "node_latencies_ms":    {},
                "rag_hit":              bool(docs),
                "llm_used":             True,
                "fallback_count":       0,
                "rerank_used":          False,
                "rerank_latency_ms":    0.0,
                "replan_count":         0,
                "critic_pass":          False,
                "critic_attempt_count": 0,
            },
        })
        return state

    def evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个 Critic 用例。"""
        CriticAgent = _compat_import('app.agents.critic').CriticAgent

        state = self._build_mock_state(case)
        start = time.perf_counter()
        result_state = CriticAgent(state)
        latency_s = time.perf_counter() - start

        # result_state.get() 在值为 None 时返回 None，不能直接 .get()
        # 用 or {} 做安全兜底
        critic_result = result_state.get("critic_result") or {}
        actual_pass   = critic_result.get("passed", False)
        actual_halluc = critic_result.get("hallucination_detected", False)
        ref_source    = result_state.get("critic_ref_source") or "rag"
        fact_checks   = critic_result.get("fact_checks", [])

        expected_pass  = case["expected_pass"]
        expected_halluc = case["expected_hallucination"]

        pass_correct   = (actual_pass == expected_pass)
        halluc_correct = (actual_halluc == expected_halluc)

        return {
            "id":              case["id"],
            "description":     case.get("description", ""),
            "expected_pass":   expected_pass,
            "actual_pass":     actual_pass,
            "pass_correct":    pass_correct,
            "expected_halluc": expected_halluc,
            "actual_halluc":   actual_halluc,
            "halluc_correct":  halluc_correct,
            "ref_source":      ref_source,
            "fact_check_count": len(fact_checks),
            "contradicted_count": sum(1 for fc in fact_checks if fc.get("status") == "contradicted"),
            "latency_s":       round(latency_s, 2),
            "feedback":        critic_result.get("feedback", ""),
        }

    def run(self, cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """运行全部 Critic 单元用例，汇总报告。"""
        CRITIC_CASES = _compat_import('app.evaluation.eval_dataset').CRITIC_CASES
        all_cases = cases or CRITIC_CASES

        logger.info("开始 CriticAgent 单元评估（用例数=%d）…", len(all_cases))

        results = []
        for case in all_cases:
            r = self.evaluate_case(case)
            results.append(r)
            status = "✓" if (r["pass_correct"] and r["halluc_correct"]) else "✗"
            logger.info(
                "[%s] %s 预期pass=%s 实际pass=%s 幻觉检测=%s/%s 来源=%s (%.2fs)",
                case["id"], status,
                r["expected_pass"], r["actual_pass"],
                r["actual_halluc"], r["expected_halluc"],
                r["ref_source"], r["latency_s"],
            )

        total = len(results)

        # 分正例（should_pass=True）和负例（should_pass=False）
        positive_cases = [r for r in results if r["expected_pass"]]
        negative_cases = [r for r in results if not r["expected_pass"]]

        tp_rate = sum(1 for r in positive_cases if r["pass_correct"]) / len(positive_cases) if positive_cases else 0
        tn_rate = sum(1 for r in negative_cases if r["pass_correct"]) / len(negative_cases) if negative_cases else 0
        halluc_acc = sum(1 for r in results if r["halluc_correct"]) / total if total else 0
        avg_latency = sum(r["latency_s"] for r in results) / total if total else 0

        # 参考来源分布
        source_dist: Dict[str, int] = {}
        for r in results:
            source_dist[r["ref_source"]] = source_dist.get(r["ref_source"], 0) + 1

        summary = {
            "total":                  total,
            "true_positive_rate":     round(tp_rate, 4),
            "true_negative_rate":     round(tn_rate, 4),
            "hallucination_accuracy": round(halluc_acc, 4),
            "avg_latency_s":          round(avg_latency, 2),
            "ref_source_distribution": source_dist,
            "results":                results,
        }
        self._print_report(summary)
        return summary

    @staticmethod
    def _print_report(summary: Dict[str, Any]) -> None:
        """打印 CriticAgent 单元评估报告。"""
        print("\n" + "=" * 65)
        print("  CriticAgent 单元评估报告")
        print(f"  评估时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 65)
        print(f"  用例总数：{summary['total']}")
        print(f"  正例通过率（True Positive）  : {summary['true_positive_rate']:.1%}")
        print(f"  负例拦截率（True Negative）  : {summary['true_negative_rate']:.1%}")
        print(f"  幻觉检测准确率               : {summary['hallucination_accuracy']:.1%}")
        print(f"  平均核查耗时                 : {summary['avg_latency_s']:.2f}s")
        print(f"  参考来源分布                 : {summary['ref_source_distribution']}")
        print("-" * 65)
        for r in summary["results"]:
            overall_ok = r["pass_correct"] and r["halluc_correct"]
            mark = "✓" if overall_ok else "✗"
            print(
                f"  [{r['id']}] {mark} 预期={'通过' if r['expected_pass'] else '拦截'} "
                f"实际={'通过' if r['actual_pass'] else '拦截'} "
                f"幻觉={r['actual_halluc']} 来源={r['ref_source']} "
                f"{r['latency_s']:.2f}s"
            )
            if not overall_ok and r.get("feedback"):
                print(f"        feedback: {r['feedback'][:60]}")
        print("=" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent 行为 & CriticAgent 单元评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m app.evaluation.agent_eval --mode agent --mock
  python -m app.evaluation.agent_eval --mode critic
  python -m app.evaluation.agent_eval --mode all --export
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "critic", "all"],
        default="all",
        help="评估模式：agent（编排行为）/ critic（单元）/ all（全部）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Agent 评估使用 mock 模式（不调用真实 API，默认开启）",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Agent 评估使用真实工作流（需要完整环境和 API Key）",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="将评估结果导出为 JSON 报告",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="报告输出目录（默认：backend/evaluation_reports/）",
    )
    return parser.parse_args()


def _export_json(data: Dict[str, Any], name: str, output_dir: Optional[str]) -> None:
    """将评估结果导出为 JSON 文件。"""
    if output_dir is None:
        base = Path(__file__).resolve().parents[3]
        output_dir = str(base / "evaluation_reports")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{name}_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  📄 报告已保存至：{path}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    args = _parse_args()
    use_mock = not args.real

    if args.mode in ("agent", "all"):
        evaluator = AgentBehaviorEvaluator(mock=use_mock)
        agent_summary = evaluator.run()
        if args.export:
            _export_json(agent_summary, "agent_eval", args.output)

    if args.mode in ("critic", "all"):
        critic_evaluator = CriticUnitEvaluator()
        critic_summary = critic_evaluator.run()
        if args.export:
            _export_json(critic_summary, "critic_eval", args.output)


if __name__ == "__main__":
    main()