"""
MedicalAI — agents/research.py
ResearchAgent：自适应检索 Agent，整合 RAG、工具查询、Wikipedia、Tavily 为内部工具集。

核心设计
--------
原架构中 WikipediaAgent / TavilyAgent / ToolAgent 是独立的"伪 Agent"（无推理决策）。
本模块将它们全部内化为 ResearchAgent 持有的工具，由 ResearchAgent 自主决定调用时机，
体现真正的 ReAct 自主决策能力：

  ┌──────────────────────────────────────────────────────┐
  │  THINK   分析当前信息缺口，决定下一步行动              │
  │  ACT     执行：RAG检索 / 工具查询 / Wikipedia / Tavily │
  │  OBSERVE 观察结果质量，更新内部状态                    │
  │  REPEAT  最多 MAX_ITER 轮，直到质量达标或耗尽策略       │
  └──────────────────────────────────────────────────────┘

支持的 ACT 动作
--------------
  rag_search       向量库检索（带 Reranker 精排）
  tool_query       结构化工具（天气/药品/术语）
  expand_query     扩展查询词后重检索
  decompose        拆解复杂问题为子查询分别检索
  wikipedia        Wikipedia 医学知识兜底
  tavily           Tavily 实时联网搜索（最终兜底）
  llm_direct       直接 LLM 推理（无需外部文档）
  accept           当前文档质量足够，退出循环生成答案

Replan 支持
-----------
当 state["replan_instruction"] 非空时，ResearchAgent 将其注入 THINK 阶段的 prompt，
确保重规划时的执行方向符合 Planner 的修订意见。
"""

import json
import re
from time import perf_counter
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.documents import Document

from app.core.logging_config import logger
from app.core.state import AgentState, append_tool_trace, record_fallback, set_node_latency
from app.tools.llm_client import get_llm
from app.tools.reranker import rerank_documents
from app.tools.vector_store import get_retriever
from app.tools.wikipedia_search import get_wikipedia_wrapper
from app.tools.tavily_search import get_tavily_search

MAX_ITER = 3  # ReAct 最大迭代轮数


# ══════════════════════════════════════════════════════════════════════════════
# 内化工具实现
# ══════════════════════════════════════════════════════════════════════════════

# ── 工具注册表（供 LLM THINK 阶段参考） ───────────────────────────────────────
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_weather": {
        "description": "根据城市名查询当前天气，用于判断气象性头痛、过敏、关节痛等是否与天气相关。",
        "keywords": ["天气", "气温", "下雨", "下雪", "湿度", "花粉", "气压", "晴", "阴"],
    },
    "search_drug": {
        "description": "查询药品基本信息，包括适应症、常见副作用、注意事项。",
        "keywords": ["药", "药品", "片", "胶囊", "注射", "服用", "剂量", "副作用",
                     "布洛芬", "阿司匹林", "对乙酰氨基酚", "抗生素", "消炎", "降压药"],
    },
    "explain_medical_term": {
        "description": "解释医学术语、检查报告中的专业词汇。",
        "keywords": ["是什么意思", "什么是", "解释", "报告", "化验", "指标",
                     "偏高", "偏低", "正常值", "ct", "mri", "b超", "血常规"],
    },
}

CITY_COORDS: Dict[str, Dict[str, float]] = {
    "北京": {"lat": 39.9042, "lon": 116.4074},
    "上海": {"lat": 31.2304, "lon": 121.4737},
    "广州": {"lat": 23.1291, "lon": 113.2644},
    "深圳": {"lat": 22.5431, "lon": 114.0579},
    "成都": {"lat": 30.5728, "lon": 104.0668},
    "杭州": {"lat": 30.2741, "lon": 120.1551},
    "武汉": {"lat": 30.5928, "lon": 114.3055},
    "西安": {"lat": 34.3416, "lon": 108.9398},
    "南京": {"lat": 32.0603, "lon": 118.7969},
    "重庆": {"lat": 29.5630, "lon": 106.5516},
    "东京": {"lat": 35.6762, "lon": 139.6503},
    "大阪": {"lat": 34.6937, "lon": 135.5023},
}

DRUG_DICT = {
    "布洛芬": {
        "name": "布洛芬（Ibuprofen）", "category": "非甾体抗炎药（NSAIDs）",
        "indications": "退烧、缓解头痛、牙痛、肌肉痛、关节痛、痛经",
        "dosage": "成人每次 200-400mg，每4-6小时一次，每日不超过 1200mg",
        "side_effects": "胃肠道刺激（恶心、胃痛）、肾功能影响（长期大量使用）",
        "warnings": "胃溃疡、肾病患者慎用；孕晚期禁用；不建议空腹服用",
    },
    "对乙酰氨基酚": {
        "name": "对乙酰氨基酚（Acetaminophen/泰诺）", "category": "解热镇痛药",
        "indications": "退烧、缓解轻中度头痛、感冒症状",
        "dosage": "成人每次 325-650mg，每4-6小时一次，每日不超过 3000mg",
        "side_effects": "大剂量或长期使用可能导致肝损伤",
        "warnings": "肝病患者慎用；避免与含酒精饮料同服",
    },
    "阿司匹林": {
        "name": "阿司匹林（Aspirin）", "category": "非甾体抗炎药 / 抗血小板药",
        "indications": "退烧镇痛（大剂量）、心脑血管疾病预防（小剂量）",
        "dosage": "镇痛退烧：成人每次 300-600mg；抗血栓：每日 75-100mg",
        "side_effects": "胃肠道刺激、出血风险增加",
        "warnings": "12岁以下儿童禁用；消化道溃疡患者慎用；手术前停药",
    },
    "氯雷他定": {
        "name": "氯雷他定（Loratadine）", "category": "第二代抗组胺药",
        "indications": "过敏性鼻炎、荨麻疹、皮肤过敏",
        "dosage": "成人每日 10mg，一次服用",
        "side_effects": "较少嗜睡，可能有头痛、口干",
        "warnings": "严重肝功能损害者慎用",
    },
    "二甲双胍": {
        "name": "二甲双胍（Metformin）", "category": "双胍类降糖药",
        "indications": "2型糖尿病血糖控制",
        "dosage": "初始每次 500mg，每日2次，随餐服用；最大每日 2000mg",
        "side_effects": "胃肠道反应（恶心、腹泻），长期使用可能影响维生素B12吸收",
        "warnings": "肾功能不全者禁用；造影检查前后需暂停",
    },
    "奥美拉唑": {
        "name": "奥美拉唑（Omeprazole）", "category": "质子泵抑制剂（PPI）",
        "indications": "胃溃疡、十二指肠溃疡、胃食管反流病",
        "dosage": "每次 20-40mg，每日一次，晨起空腹服用",
        "side_effects": "头痛、腹泻；长期使用可能影响钙、镁、B12吸收",
        "warnings": "不建议超说明书长期自行服用",
    },
    "阿莫西林": {
        "name": "阿莫西林（Amoxicillin）", "category": "青霉素类抗生素",
        "indications": "细菌感染：呼吸道、泌尿道、皮肤、牙科感染等",
        "dosage": "成人每次 250-500mg，每8小时一次；需完成疗程",
        "side_effects": "皮疹、胃肠道不适；罕见严重过敏反应",
        "warnings": "青霉素过敏者禁用；病毒感染无效",
    },
}

MEDICAL_TERMS = {
    "肌酐": "肌酐是肌肉代谢产物，通过肾脏过滤排出。血肌酐偏高提示肾脏过滤功能下降，需结合 eGFR 等指标综合判断。",
    "空腹血糖": "空腹血糖正常值为 3.9-6.1 mmol/L。6.1-7.0 为糖尿病前期，≥7.0 提示糖尿病（需复查确认）。",
    "糖化血红蛋白": "HbA1c 反映过去 2-3 个月平均血糖水平。< 5.7% 正常，5.7-6.4% 为糖尿病前期，≥ 6.5% 提示糖尿病。",
    "甘油三酯": "血脂指标之一，正常值 < 1.7 mmol/L。偏高与饮食、运动、遗传有关，增加心血管风险。",
    "ldl": "低密度脂蛋白（坏胆固醇），偏高增加动脉粥样硬化风险。建议 < 3.4 mmol/L。",
    "hdl": "高密度脂蛋白（好胆固醇），偏高有保护心血管作用。男性 > 1.0 mmol/L，女性 > 1.3 mmol/L 为正常。",
    "tsh": "促甲状腺激素，甲状腺功能评估的核心指标。偏高提示甲状腺功能减退，偏低提示甲亢。正常值约 0.4-4.0 mIU/L。",
    "alt": "丙氨酸转氨酶，最敏感的肝功能指标。偏高主要提示肝细胞损伤，正常上限约 40 U/L。",
    "白细胞": "血常规中白细胞计数反映免疫状态。偏高常见于细菌感染、炎症；偏低见于病毒感染、骨髓抑制等。",
    "血红蛋白": "反映是否贫血。男性正常 120-160 g/L，女性 110-150 g/L，低于正常值提示贫血。",
    "ct": "计算机断层扫描，通过X射线断层成像，适合骨骼、出血、肺部等检查。",
    "mri": "磁共振成像，无辐射，软组织分辨率高，适合脑部、脊髓、关节、腹部实质脏器检查。",
}


def _run_get_weather(param: str) -> str:
    """
    查询城市天气，支持当前天气和明天预报。

    两个 API 源（自动切换）：
      1. Open-Meteo（主，免费，全球）：国内网络可能被阻断
      2. wttr.in（备，免费，文本友好）：国内多数网络可访问，JSON 格式
    """
    import json as _json

    coords = None
    city = param
    for name, c in CITY_COORDS.items():
        if name in param or param in name:
            coords = c
            city = name
            break
    if not coords:
        try:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={param}&count=1&language=zh&format=json"
            resp = httpx.get(geo_url, timeout=5.0)
            data = resp.json()
            if data.get("results"):
                r = data["results"][0]
                coords = {"lat": r["latitude"], "lon": r["longitude"]}
                city = r.get("name", param)
        except Exception as e:
            pass  # 不返回错误，继续尝试 wttr.in 兜底

    # ── 源1：Open-Meteo API ────────────────────────────────────────────────
    if coords:
        try:
            forecast_days = 3  # 始终返回当前+明天预报，用户经常问"明天天气如何"
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={coords['lat']}&longitude={coords['lon']}"
                f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,apparent_temperature"
                f"&daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_mean"
                f"&timezone=Asia%2FShanghai&forecast_days={forecast_days}"
            )
            resp = httpx.get(url, timeout=8.0)
            data = resp.json()

            wmo = {0:"晴天",1:"基本晴朗",2:"部分多云",3:"阴天",61:"小雨",63:"中雨",65:"大雨",71:"小雪",95:"雷暴"}
            parts = []

            # 当前天气
            current = data.get("current", {})
            if current:
                temp = current.get("temperature_2m", "N/A")
                feels = current.get("apparent_temperature", "N/A")
                humidity = current.get("relative_humidity_2m", "N/A")
                wind = current.get("wind_speed_10m", "N/A")
                wcode = wmo.get(current.get("weather_code", 0), "未知")
                parts.append(f"当前：{wcode}，{temp}°C（体感{feels}°C），湿度{humidity}%，风速{wind}km/h")

            # 明天预报（forecast_days=3 时 data 含 daily 数据）
            daily = data.get("daily", {})
            if daily and daily.get("time") and len(daily["time"]) > 1:
                    tomorrow = daily["time"][1]
                    tmax = daily.get("temperature_2m_max", ["N/A"])[1]
                    tmin = daily.get("temperature_2m_min", ["N/A"])[1]
                    rain_pct = daily.get("precipitation_probability_mean", [None])[1]
                    twmo = wmo.get(daily.get("weather_code", [0])[1], "未知")
                    rain_str = f"，降水概率{rain_pct}%" if rain_pct is not None else ""
                    parts.append(f"明天({tomorrow})：{twmo}，最高{tmax}°C/最低{tmin}°C{rain_str}")

            if parts:
                result = f"【{city}天气】" + " | ".join(parts) + "。"
                tips = []
                humidity_val = current.get("relative_humidity_2m")
                temp_val = current.get("temperature_2m")
                if isinstance(humidity_val, (int, float)) and humidity_val > 80:
                    tips.append("湿度较高，关节炎患者可能感到不适")
                if isinstance(temp_val, (int, float)) and temp_val < 5:
                    tips.append("气温较低，心脑血管疾病患者注意保暖")
                if tips:
                    result += " 健康提示：" + "；".join(tips) + "。"
                return result
        except Exception as e:
            logger.debug("Open-Meteo 查询失败，尝试 wttr.in 兜底：%s", e)

    # ── 源2：wttr.in 兜底（国内多数网络可访问） ──────────────────────────
    try:
        city_name = city or param
        wttr_url = f"https://wttr.in/{city_name}?format=j1&lang=zh"
        resp = httpx.get(wttr_url, timeout=8.0)
        wdata = resp.json()

        cc = wdata.get("current_condition", [{}])[0]
        temp = cc.get("temp_C", "N/A")
        feels = cc.get("FeelsLikeC", "N/A")
        humidity = cc.get("humidity", "N/A")
        desc = cc.get("lang_zh", [{}])[0].get("value", "") if cc.get("lang_zh") else cc.get("weatherDesc", [{}])[0].get("value", "")
        wind = cc.get("windspeedKmph", "N/A")
        result = f"【{city_name}天气】当前：{desc}，{temp}°C（体感{feels}°C），湿度{humidity}%，风速{wind}km/h。"

        # 始终包含明天预报（用户常问"明天天气"）
        forecasts = wdata.get("weather", [])
        if len(forecasts) >= 2:
            tmrw = forecasts[1]
            date = tmrw.get("date", "")
            tmax = tmrw.get("maxtempC", "N/A")
            tmin = tmrw.get("mintempC", "N/A")
            desc_t = tmrw.get("hourly", [{}])[0].get("lang_zh", [{}])[0].get("value", "") if tmrw.get("hourly") and tmrw["hourly"][0].get("lang_zh") else ""
            result += f" 明天({date})：{desc_t}，最高{tmax}°C/最低{tmin}°C。"

        tips = []
        if isinstance(humidity, (int, float)) and humidity > 80:
            tips.append("湿度较高，关节炎患者可能感到不适")
        if isinstance(temp, (int, float)) and int(temp) < 5:
            tips.append("气温较低，心脑血管疾病患者注意保暖")
        if tips:
            result += " 健康提示：" + "；".join(tips) + "。"
        return result
    except Exception as e:
        # 网络完全不可用时，给出该城市季节性气候参考
        city_name = city or param
        season_hint = ""
        try:
            from datetime import datetime
            month = datetime.now().month
            if 3 <= month <= 5:
                season_hint = "春季（15-22°C，早晚温差较大，可能有降雨）"
            elif 6 <= month <= 8:
                season_hint = "夏季（25-35°C，高温高湿，可能有台风）"
            elif 9 <= month <= 11:
                season_hint = "秋季（15-25°C，凉爽干燥，气温适中）"
            else:
                season_hint = "冬季（0-10°C，寒冷干燥，注意保暖）"
        except Exception:
            pass
        return (
            f"【{city_name}天气】天气服务暂时不可用。{city_name}当前为{season_hint}。"
            " 健康提示：风湿病患者对天气变化敏感，建议做好保暖防潮准备。"
        )


def _run_search_drug(param: str) -> str:
    for key, info in DRUG_DICT.items():
        if key in param or param in key:
            return (
                f"【{info['name']}】\n分类：{info['category']}\n"
                f"适应症：{info['indications']}\n常规剂量：{info['dosage']}\n"
                f"常见副作用：{info['side_effects']}\n注意事项：{info['warnings']}\n"
                "⚠️ 以上为一般性参考信息，具体用药请遵医嘱或药师指导。"
            )
    return f"未在数据库中找到 {param} 的详细信息，建议查阅药品说明书或咨询药师。"


def _run_explain_term(param: str) -> str:
    for key, explanation in MEDICAL_TERMS.items():
        if key in param.lower() or param.lower() in key:
            return f"【{key}】{explanation}"
    return f"未找到 {param} 的本地解释，建议向医生或专业医疗网站查询。"


def _run_tool(tool_name: str, param: str) -> str:
    """
    工具调用入口：使用本地硬编码实现。
    """
    if tool_name == "get_weather":
        return _run_get_weather(param)
    elif tool_name == "search_drug":
        return _run_search_drug(param)
    elif tool_name == "explain_medical_term":
        return _run_explain_term(param)
    return f"工具 {tool_name} 未实现。"


# ══════════════════════════════════════════════════════════════════════════════
# RAG 检索（带 Reranker）
# ══════════════════════════════════════════════════════════════════════════════

def _rag_search(query: str, state: AgentState, skip_coverage_check: bool = False) -> List[Document]:
    """
    RAG 检索（带 Reranker + 知识库盲区检测）。

    盲区检测逻辑
    ------------
    首次检索（state 里还没有任何文档）时，先做 3 篇快速召回 + rerank 评分。
    如果 top_score < 0.15，说明知识库完全没有相关内容（盲区），
    直接返回空列表 + 在 state 里写入盲区标记，让 THINK 阶段感知并路由到 tavily。

    相比原来无脑 expand_query 三轮的好处
    -------------------------------------
    - 维生素C预防新冠这类问题：原来3轮都在 expand_query，耗时 ~90s 才触发 tavily
    - 改造后：第一轮 _rag_search 检测到盲区（score=0.098<0.15），
      立即标记 rag_blind_spot=True，THINK 下一轮直接选 tavily，节省 ~60s
    """
    retriever = get_retriever()
    if not retriever:
        return []

    # ── 知识库盲区快速检测 ────────────────────────────────────────────────
    # 仅在首次 RAG 检索时触发（state 里还没有文档）
    if not skip_coverage_check and not state.get("documents"):
        try:
            from app.tools.vector_store import check_coverage
            coverage = check_coverage(query, threshold=0.15)
            if not coverage["covered"]:
                logger.info(
                    "ResearchAgent 知识库盲区：%s（top_score=%.3f）",
                    coverage["suggestion"], coverage["top_score"],
                )
                # 写入盲区标记，THINK 阶段 prompt 里会感知并路由到 tavily
                state["rag_blind_spot"] = True
                state["rag_blind_score"] = coverage["top_score"]
                return []   # 提前返回空，不做无效的完整召回
        except Exception as exc:
            logger.debug("盲区检测失败（忽略）：%s", exc)
    # ─────────────────────────────────────────────────────────────────────

    try:
        docs = retriever.invoke(query)
    except Exception as exc:
        logger.error("RAG 检索异常：%s", exc)
        return []
    valid_docs = [d for d in (docs or []) if len(d.page_content.strip()) > 100]
    if not valid_docs:
        return []
    try:
        reranked = rerank_documents(query=query, documents=valid_docs)
        state["metrics"]["rerank_used"] = True
        return reranked
    except Exception as exc:
        logger.warning("Reranker 异常，使用原始结果：%s", exc)
        return valid_docs


# ══════════════════════════════════════════════════════════════════════════════
# THINK 阶段：LLM 分析当前信息质量，决定下一步 ACT
# ══════════════════════════════════════════════════════════════════════════════

_ACT_CHOICES = ["rag_search", "tool_query", "expand_query", "decompose",
                "wikipedia", "tavily", "llm_direct", "accept"]

_TOOLS_DESC = "\n".join(
    f'  - "{k}": {v["description"]}' for k, v in TOOL_REGISTRY.items()
)


def _think(
    question: str,
    docs: List[Document],
    iteration: int,
    used_actions: List[str],
    replan_instruction: str,
    llm,
    prev_scores: Optional[Dict] = None,
    rag_blind_spot: bool = False,
    rag_blind_score: float = 0.0,
) -> Dict:
    """
    THINK 阶段：LLM 以独立视角分析当前文档质量，决定下一步行动。
    prev_scores 传入上一轮的评分，帮助模型感知质量变化趋势，避免无效重复迭代。
    """
    doc_summary = ""
    if docs:
        snippets = [f"[Doc{i+1}] {d.page_content[:200]}" for i, d in enumerate(docs[:3])]
        doc_summary = "\n".join(snippets)
    else:
        doc_summary = "（暂无检索文档）"

    replan_hint = f"\n\n【Planner 重规划指令】：{replan_instruction}" if replan_instruction else ""

    # 盲区感知：告诉 THINK 知识库没有相关内容，避免再次 expand_query
    blind_spot_hint = ""
    if rag_blind_spot:
        blind_spot_hint = (
            f"\n\n【知识库盲区警告】本地向量库无相关内容（top_rerank={rag_blind_score:.3f}<0.15）。"
            "请直接选择 tavily 或 wikipedia，不要再 expand_query。"
        )
    used_str = "、".join(used_actions) if used_actions else "无"

    # 上轮评分提示：帮助模型判断是否已经足够好，避免重复无效扩展
    prev_score_hint = ""
    if prev_scores and iteration > 0:
        prev_score_hint = (
            f"\n\n【上一轮评分参考】relevance={prev_scores.get('relevance',0):.1f}，"
            f"coverage={prev_scores.get('coverage',0):.1f}，"
            f"medical_depth={prev_scores.get('medical_depth',0):.1f}。"
            "如果本轮文档在此基础上有改善且三项均≥6，请直接选择 accept，不要继续扩展。"
        )

    # ── 硬性 early-exit：rerank_score ≥ 0.80 直接 accept，跳过 THINK LLM 调用 ──
    # 这是最重要的优化：Reranker 打分是客观信号，比 LLM 自评更可靠且无额外延迟。
    # top 分 ≥ 0.80 时文档已高度相关，继续 THINK 只是浪费 3-4s 的 LLM 调用。
    top_rerank_score = 0.0
    # 当 docs 为空时，_think 应直接返回 action=expand_query，不打分
    if not docs:
        return {
            "scores": None,   # 明确为 None，前端可据此显示"暂无文档"
            "reasoning": "当前无文档，执行检索扩展",
            "action": "expand_query",
            "param": question,
            "reason": "当前无文档，需扩展查询",
        }
    
    if docs:
        scores = [d.metadata.get("rerank_score", 0.0) for d in docs]
        top_rerank_score = max(scores) if scores else 0.0

    # ── 动态 early-exit 阈值 ─────────────────────────────────────────────────
    # 0.80 是经验值，但 bge-reranker-base 的分数分布不是线性的：
    #   0.90+ : 极高相关（文档几乎就是问题答案）→ 无条件 accept
    #   0.80-0.90 : 高度相关，但需要确认不是第0轮（首轮只有expand_query，无基础文档）
    #   0.70-0.80 : 中等相关，仅在已有多个文档时才 accept（避免信息不足）
    # 首轮（iteration=0）：expand_query 后得到的是第一批文档，阈值适当提高
    # 非首轮：已经历一轮迭代，阈值可以低一点（避免无效重复）
    EARLY_ACCEPT_THRESHOLD = 0.85 if iteration == 0 else 0.80

    # 额外条件：急症类问题（含"急""立即""马上""怎么办"等）降低阈值
    # 理由：急症场景宁可用次优文档快速回答，也不要拖延检索
    _URGENT_KEYWORDS = ["急", "立即", "马上", "怎么办", "救", "危险", "紧急"]
    if any(kw in question for kw in _URGENT_KEYWORDS):
        EARLY_ACCEPT_THRESHOLD = min(EARLY_ACCEPT_THRESHOLD, 0.75)

    if docs and top_rerank_score >= EARLY_ACCEPT_THRESHOLD:
        result = {
            "relevance": 8.0, "coverage": 7.0, "medical_depth": 7.0,
            "action": "accept",
            "param": "",
            "reason": (
                f"Reranker 客观评分 {top_rerank_score:.3f} ≥ 动态阈值 {EARLY_ACCEPT_THRESHOLD}，"
                f"文档高度相关，直接 accept（iter={iteration}）。"
            ),
        }
        logger.info(
            "THINK [iter=%d] early-exit: rerank_score=%.3f ≥ %.2f，跳过 LLM 直接 accept",
            iteration, top_rerank_score, EARLY_ACCEPT_THRESHOLD,
        )
        # 写入 early-exit 日志供阈值验证分析使用
        logger.debug(
            "EARLY_EXIT_LOG | question=%s | iter=%d | rerank_score=%.3f | threshold=%.2f",
            question[:50], iteration, top_rerank_score, EARLY_ACCEPT_THRESHOLD,
        )
        return result

    rerank_hint = (
        f"\n\n【Reranker 客观评分】Top 文档分数={top_rerank_score:.3f}（满分1.0）。"
        "分数≥0.80 直接 accept；0.60-0.80 酌情；<0.60 继续检索。"
        if docs else ""
    )

    prompt = (
        "你是一名医疗 RAG 评估专家。请分析当前检索结果，决定下一步行动。\n"
        "只返回 JSON，不要输出额外文字。\n\n"
        f"用户问题：{question}\n"
        f"当前迭代：第 {iteration + 1} 轮（最多 {MAX_ITER} 轮）\n"
        f"已执行动作：{used_str}\n"
        f"当前文档摘要：\n{doc_summary}"
        f"{rerank_hint}"
        f"{prev_score_hint}"
        f"{replan_hint}"
        f"{blind_spot_hint}\n\n"
        "评分维度（0-10）：\n"
        "  relevance：文档与问题的语义相关程度\n"
        "  coverage：文档对问题各要点的覆盖完整度\n"
        "  medical_depth：医学专业深度（包含病因/治疗/预后等）\n\n"
        "【accept 触发规则（满足任一即可选择 accept）】\n"
        "  ① 三项评分均≥6 且已有文档\n"
        "  ② 已执行 expand_query 且本轮评分相比上轮提升 < 0.5（边际收益递减）\n"
        "  ③ 已是最后一轮且有文档\n\n"
        "可选动作：\n"
        "  - accept：满足上述规则时选择，退出循环生成答案\n"
        "  - expand_query：文档不足且未连续扩展两轮（param: 具体的扩展查询词）\n"
        "  - decompose：问题复杂，拆解为子问题（param: 子问题，逗号分隔）\n"
        "  - tool_query：需要结构化工具（param: tool_name|查询参数）\n"
        f"    可用工具：\n{_TOOLS_DESC}\n"
        "  - wikipedia：向量库无相关内容（param: 搜索词）\n"
        "  - tavily：以上均失败，联网搜索（param: 搜索词）\n"
        "  - llm_direct：通用健康咨询，无需外部文档\n\n"
        "⚠️ 已连续两轮 expand_query 时，必须选 accept 或其他动作，不得三连扩展。\n\n"
        '返回格式：{"relevance": 分数, "coverage": 分数, "medical_depth": 分数, '
        '"action": "动作名", "param": "动作参数", "reason": "一句决策理由"}'
    )

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            payload = json.loads(match.group())
            action = payload.get("action", "accept")
            if action not in _ACT_CHOICES:
                action = "accept"
            return {
                "relevance": float(payload.get("relevance", 5)),
                "coverage": float(payload.get("coverage", 5)),
                "medical_depth": float(payload.get("medical_depth", 5)),
                "action": action,
                "param": str(payload.get("param", question)),
                "reason": str(payload.get("reason", "")),
            }
    except Exception as exc:
        logger.warning("THINK 阶段 LLM 失败：%s", exc)

    # 降级：无文档则检索，有文档则接受
    return {
        "relevance": 5.0, "coverage": 5.0, "medical_depth": 5.0,
        "action": "rag_search" if not docs else "accept",
        "param": question,
        "reason": "LLM 不可用，使用启发式策略。",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 最终答案生成
# ══════════════════════════════════════════════════════════════════════════════

def _generate_answer(
    question: str,
    docs: List[Document],
    tool_result: str,
    history_context: str,
    long_term_prefix: str,
    llm,
) -> tuple[str, str]:
    """生成最终答案，返回 (answer, source)。"""
    if not llm:
        if docs:
            snippets = "；".join(d.page_content[:180] for d in docs[:2])
            return f"根据医学资料：{snippets}。如症状持续，请就医。", "医学知识库（抽取式）"
        return "当前服务暂时不可用，如有明显不适请线下就医。", "系统提示"

    if tool_result and docs:
        content = "\n\n".join(d.page_content[:1000] for d in docs[:5])
        prompt = (
            "你是一名专业中文医疗AI助手。请结合下方工具查询结果和医学资料回答患者问题。\n"
            "【硬约束】\n"
            "1. 资料中已有的信息必须准确引用，不得编造具体数据、剂量、研究结论。\n"
            "2. 如果资料缺少回答所需的关键信息，可以基于通用医学常识合理补充，"
            "但必须明确区分\"资料提到\"和\"常识建议\"。\n"
            "3. 允许将工具数据和医学资料中的相关信息组合成连贯的回答，"
            "不得说\"未提及\"来回避问题。\n"
            "4. 不得使用确诊性语言（如\"你患有\"），不得编造药名或剂量。\n"
            "用3-5句中文回答。语气要专业、温和、清晰。"
            "如果涉及天气等实时数据，结合该数据给出具体的健康建议。\n"
            "最后提醒用户必要时及时线下就医。\n\n"
            + long_term_prefix
            + f"对话历史：\n{history_context}\n"
            f"患者问题：{question}\n\n"
            f"工具查询结果：\n{tool_result}\n\n"
            f"医学资料：\n{content}\n"
        )
        source = "知识库 + 实时数据"
    elif tool_result:
        prompt = (
            "你是一名专业中文医疗AI助手。请基于下方工具查询结果回答患者问题。\n"
            "【硬约束】仅使用工具结果中提供的信息，不得额外补充任何医学知识、"
            "治疗建议或诊断结论。如果工具结果中没有提及，请直接说明\"工具结果未提及\"。\n"
            "用2-4句通俗、温和的中文回答。如涉及用药必须提醒遵医嘱。\n\n"
            + long_term_prefix
            + f"对话历史：\n{history_context}\n"
            f"患者问题：{question}\n\n"
            f"工具查询结果：\n{tool_result}\n"
        )
        source = "结构化工具查询"
    elif docs:
        content = "\n\n".join(d.page_content[:1000] for d in docs[:5])
        prompt = (
            "你是一名中文医疗问答助手。请基于下方医学资料回答患者问题。\n"
            "【硬约束】\n"
            "1. 资料中已有的信息必须准确引用，不得编造具体数据、剂量、研究结论。\n"
            "2. 如果资料缺少回答所需的关键信息，可以基于通用医学常识合理补充，"
            "但必须明确区分\"资料提到\"和\"常识建议\"。\n"
            "3. 允许将多条资料中的相关信息组合成连贯的回答，不得说\"资料未提及\"来回避问题。\n"
            "4. 不得使用确诊性语言（如\"你患有\"），不得编造药名或剂量。\n"
            "用3-5句中文回答。语气要专业、温和、清晰。"
            "如果问题涉及出行、环境、季节等具体场景，结合资料给出切实可行的建议。\n"
            "最后提醒用户必要时及时线下就医。\n\n"
            + long_term_prefix
            + f"对话历史：\n{history_context}\n"
            f"当前问题：\n{question}\n\n"
            f"医学资料：\n{content}\n"
        )
        source = "医学知识库"
    else:
        prompt = (
            "你是一名专业、稳健、表达清晰的中文医疗AI助手。\n"
            "当前没有检索到相关医学资料，请仅基于通用医学常识回答，"
            "并在回答末尾明确添加免责声明。\n"
            "【硬约束】\n"
            "1. 不要给出超出常识范围的具体剂量、药名、治疗方案。\n"
            "2. 不要使用确诊性语言（如\"你患有\"、\"你得了\"）。\n"
            "3. 强调信息有限，建议线下就医。\n"
            "回答要具体、有实质内容，包括可能的原因、建议的处理方式。\n"
            "如果问题存在风险或不确定性，要提醒用户线下就医，但不要制造恐慌。\n\n"
            + long_term_prefix
            + f"对话历史：\n{history_context}\n"
            f"当前问题：\n{question}\n"
        )
        source = "通用医疗知识"

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
        if answer and len(answer) > 10:
            return answer, source
    except Exception as exc:
        logger.error("答案生成 LLM 失败：%s", exc)

    if docs:
        snippets = "；".join(d.page_content[:180] for d in docs[:2])
        return f"根据医学资料：{snippets}。如症状持续，请就医。", "医学知识库（抽取式）"
    return "当前服务暂时不可用，如有明显不适请线下就医。", "系统提示"


# ══════════════════════════════════════════════════════════════════════════════
# ResearchAgent 主函数
# ══════════════════════════════════════════════════════════════════════════════

def ResearchAgent(state: AgentState) -> AgentState:
    """
    自适应检索 Agent。
    通过 THINK-ACT-OBSERVE 循环，自主决定使用哪些工具完成检索任务。
    """
    start_time = perf_counter()
    append_tool_trace(state, "research")

    llm = get_llm()
    question = state["question"].strip()
    replan_instruction = state.get("replan_instruction", "")
    route_tool = state.get("current_tool", "retriever")

    # 构建历史上下文
    turns = state.get("context_window") or state.get("conversation_history", [])
    history_context = ""
    for item in turns[-5:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "user":
            history_context += f"患者：{content}\n"
        elif role == "assistant":
            history_context += f"助手：{content}\n"

    long_term_prefix = ""
    lt = state.get("long_term_context", "").strip()
    if lt:
        long_term_prefix = f"以下是该患者的历史档案，请参考：\n{lt}\n\n"

    # ── 初始化本轮状态 ─────────────────────────────────────────────────────
    docs: List[Document] = list(state.get("documents") or [])
    tool_result: str = ""
    used_actions: List[str] = []
    think_log: List[Dict] = list(state.get("rag_think_log") or [])

    # ── 如果 Planner 路由到 llm_agent，直接跳过检索 ───────────────────────
    if route_tool == "llm_agent" and not replan_instruction:
        answer, source = _generate_answer(
            question, [], "", history_context, long_term_prefix, llm
        )
        state["generation"] = answer
        state["source"] = source
        state["llm_success"] = bool(answer and len(answer) > 10)
        state["llm_attempted"] = True
        state["research_strategy"] = "llm_direct"
        state["metrics"]["llm_used"] = True
        set_node_latency(state, "research", (perf_counter() - start_time) * 1000)
        logger.info("ResearchAgent [llm_direct] 路径完成")
        return state

    # ── ReAct 主循环 ───────────────────────────────────────────────────────
    prev_scores: Optional[Dict] = None  # 上一轮评分，传给 THINK 避免无效重复
    for iteration in range(MAX_ITER):
        # 最后一轮强制 accept（避免跑满后再发起无效检索）
        if iteration == MAX_ITER - 1 and docs:
            think_result = {
                "relevance": 7.0, "coverage": 7.0, "medical_depth": 7.0,
                "action": "accept", "param": "", "reason": "已达最大迭代轮数，强制退出生成答案。",
                "iteration": iteration,
            }
            think_log.append(think_result)
            logger.info("ResearchAgent THINK [iter=%d] 强制 accept（已达 MAX_ITER）", iteration)
            break

        # ── 盲区强制路由：不依赖 LLM 决策，直接跳到最优工具 ──────────────
        # 原设计靠 blind_spot_hint 提示 LLM 选 tavily，但 LLM 倾向于按
        # 既有模式（无文档→expand_query）走，soft prompt 无效。
        # 改为：检测盲区类型，智能选择工具：
        #   天气相关 → get_weather（拿到实时气象数据后结合医学知识回答）
        #   其他 → tavily（联网检索最新信息）
        if (state.get("rag_blind_spot")
                and "tavily" not in used_actions
                and "wikipedia" not in used_actions):
            _WEATHER_KW = ["天气", "气温", "下雨", "下雪", "刮风", "湿度",
                           "降温", "升温", "weather", "temperature", "rain"]
            # 如果 replan 指令明确让跳过天气工具，则尊重该指令
            _skip_weather = "get_weather" in replan_instruction
            if (any(kw in question for kw in _WEATHER_KW)
                    and "tool_query" not in used_actions
                    and not _skip_weather):
                think_result = {
                    "relevance": 1.0, "coverage": 1.0, "medical_depth": 1.0,
                    "action": "tool_query",
                    "param": f"get_weather|{question}",
                    "reason": (
                        f"知识库盲区（score={state.get('rag_blind_score',0):.3f}）"
                        "且问题涉及天气，先获取实时气象数据。"
                    ),
                    "iteration": iteration,
                }
                logger.info(
                    "ResearchAgent THINK [iter=%d] 天气盲区强制路由 → get_weather",
                    iteration,
                )
            else:
                think_result = {
                    "relevance": 1.0, "coverage": 1.0, "medical_depth": 1.0,
                    "action": "tavily",
                    "param": question,
                    "reason": f"知识库盲区（score={state.get('rag_blind_score',0):.3f}），强制路由到 tavily。",
                    "iteration": iteration,
                }
                logger.info(
                    "ResearchAgent THINK [iter=%d] 盲区强制路由 → tavily（跳过 LLM 决策）",
                    iteration,
                )
        else:
            think_result = _think(
                question, docs, iteration, used_actions,
                replan_instruction, llm, prev_scores,
                rag_blind_spot=state.get("rag_blind_spot", False),
                rag_blind_score=state.get("rag_blind_score", 0.0),
            )
        # ───────────────────────────────────────────────────────────────
        think_result["iteration"] = iteration
        think_log.append(think_result)
        # 记录本轮评分，供下一轮参考
        prev_scores = {
            "relevance": think_result.get("relevance", 5.0),
            "coverage": think_result.get("coverage", 5.0),
            "medical_depth": think_result.get("medical_depth", 5.0),
        }
        action = think_result["action"]
        param = think_result["param"]

        logger.info(
            "ResearchAgent THINK [iter=%d] action=%s param=%s reason=%s",
            iteration, action, param[:40], think_result.get("reason", ""),
        )

        used_actions.append(action)

        # ── ACT ──────────────────────────────────────────────────────────
        if action == "accept":
            break

        elif action == "rag_search" or action == "expand_query":
            new_docs = _rag_search(param, state)
            if new_docs:
                # 合并去重
                existing_contents = {d.page_content for d in docs}
                for d in new_docs:
                    if d.page_content not in existing_contents:
                        docs.append(d)
                        existing_contents.add(d.page_content)
                state["rag_attempted"] = True
                state["rag_success"] = True
                state["metrics"]["rag_hit"] = True
            else:
                record_fallback(state, f"research_rag_empty_iter{iteration}")

        elif action == "decompose":
            sub_queries = [q.strip() for q in param.split(",") if q.strip()]
            for sq in sub_queries[:3]:
                sub_docs = _rag_search(sq, state)
                existing_contents = {d.page_content for d in docs}
                for d in sub_docs:
                    if d.page_content not in existing_contents:
                        docs.append(d)
                        existing_contents.add(d.page_content)
            state["rag_attempted"] = True

        elif action == "tool_query":
            # param 格式：tool_name|查询参数
            parts = param.split("|", 1)
            tool_name = parts[0].strip()
            tool_param = parts[1].strip() if len(parts) > 1 else question
            if tool_name in TOOL_REGISTRY:
                tool_result = _run_tool(tool_name, tool_param)
                state["tool_results"] = {"tool": tool_name, "result": tool_result}
                state["tool_agent_success"] = True
                logger.info("ResearchAgent 工具调用：%s → %s", tool_name, tool_result[:60])

                # ── 工具失败自动兜底：若工具返回错误信息且无高质量文档，直接尝试 Tavily ──
                # 根因：get_weather 在国内网络下调用 Open-Meteo API 经常超时/断连，
                # 如果此时文档不足（rerank 偏低），当前迭代直接 fallthrough 到 Tavily，
                # 而不是等下一轮 THINK 决策（往往来不及，已经接近 MAX_ITER 末尾）。
                _FAIL_INDICATORS = ("失败", "Error", "error", "未能", "无法", "timeout")
                _has_good_docs = any(
                    d.metadata.get("rerank_score", 0) >= 0.5 for d in docs
                ) if docs else False
                if any(ind in tool_result for ind in _FAIL_INDICATORS) and not _has_good_docs:
                    if "tavily" not in used_actions:
                        logger.info("工具 %s 失败，自动兜底→Tavily", tool_name)
                        tavily = get_tavily_search()
                        if tavily:
                            try:
                                results = tavily.invoke(f"{tool_param}")
                                valid = [r for r in (results or []) if isinstance(r, dict) and len(r.get("content", "")) > 50]
                                if valid:
                                    for r in valid:
                                        docs.append(Document(
                                            page_content=r["content"],
                                            metadata={"url": r.get("url", ""), "title": r.get("title", "")},
                                        ))
                                    state["tavily_success"] = True
                                    state["source"] = "实时医学搜索"
                                    logger.info("工具失败→Tavily 兜底成功，%d 条", len(valid))
                                else:
                                    record_fallback(state, f"research_tool_fallback_tavily_empty:{tool_name}")
                            except Exception as exc:
                                logger.error("工具失败→Tavily 兜底异常：%s", exc)
                                record_fallback(state, f"research_tool_fallback_tavily_exception:{exc}")
                        state["tavily_attempted"] = True
            else:
                record_fallback(state, f"research_unknown_tool:{tool_name}")

        elif action == "wikipedia":
            wiki = get_wikipedia_wrapper()
            if wiki:
                try:
                    content = wiki.run(f"{param} 医学 症状 治疗")
                    if content and len(content.strip()) > 100:
                        docs.append(Document(page_content=content))
                        state["wiki_success"] = True
                        state["source"] = "Wikipedia 医学资料"
                        logger.info("ResearchAgent Wikipedia 检索成功")
                    else:
                        state["wiki_success"] = False
                        record_fallback(state, f"research_wiki_empty_iter{iteration}")
                except Exception as exc:
                    logger.error("Wikipedia 检索异常：%s", exc)
                    record_fallback(state, f"research_wiki_exception:{exc}")
            state["wiki_attempted"] = True

        elif action == "tavily":
            tavily = get_tavily_search()
            if tavily:
                try:
                    results = tavily.invoke(f"{param}")
                    valid = [r for r in (results or []) if isinstance(r, dict) and len(r.get("content", "")) > 50]
                    if valid:
                        for r in valid:
                            docs.append(Document(
                                page_content=r["content"],
                                metadata={"url": r.get("url", ""), "title": r.get("title", "")},
                            ))
                        state["tavily_success"] = True
                        state["source"] = "实时医学搜索"
                        logger.info("ResearchAgent Tavily 检索成功，%d 条结果", len(valid))
                    else:
                        state["tavily_success"] = False
                        record_fallback(state, f"research_tavily_empty_iter{iteration}")
                except Exception as exc:
                    logger.error("Tavily 检索异常：%s", exc)
                    record_fallback(state, f"research_tavily_exception:{exc}")
            state["tavily_attempted"] = True

        elif action == "llm_direct":
            # 直接 LLM 推理，无需外部文档
            answer, source = _generate_answer(
                question, [], "", history_context, long_term_prefix, llm
            )
            state["generation"] = answer
            state["source"] = source
            state["llm_success"] = bool(answer and len(answer) > 10)
            state["llm_attempted"] = True
            state["metrics"]["llm_used"] = True
            state["research_strategy"] = "llm_direct"
            state["documents"] = docs
            state["rag_think_log"] = think_log
            state["rag_iterations"] = iteration + 1
            set_node_latency(state, "research", (perf_counter() - start_time) * 1000)
            logger.info("ResearchAgent [llm_direct] 退出循环")
            return state

    # ── 退出循环：生成最终答案 ─────────────────────────────────────────────
    answer, source = _generate_answer(
        question, docs, tool_result, history_context, long_term_prefix, llm
    )

    # 写回 state
    state["documents"] = docs
    state["generation"] = answer
    state["source"] = source
    state["rag_grader_passed"] = bool(docs or tool_result)
    state["rag_iterations"] = len([t for t in think_log if t.get("action") != "accept"])
    state["rag_think_log"] = think_log
    state["research_strategy"] = ",".join(dict.fromkeys(used_actions))  # 去重保序
    state["llm_attempted"] = True
    state["llm_success"] = bool(answer and len(answer) > 10)
    state["metrics"]["llm_used"] = True

    # 写入对话历史
    state["conversation_history"].append({"role": "user", "content": question})
    state["conversation_history"].append({"role": "assistant", "content": answer, "source": source})

    set_node_latency(state, "research", (perf_counter() - start_time) * 1000)
    logger.info(
        "ResearchAgent 完成 | 策略=%s | docs=%d | answer_len=%d",
        state["research_strategy"], len(docs), len(answer),
    )
    return state