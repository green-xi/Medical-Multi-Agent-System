"""ResearchAgent：ReAct 循环检索 Agent，整合 RAG、工具查询、Wikipedia、Tavily。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from langchain_core.documents import Document

from app.core.logging_config import logger
from app.core.state import AgentState, append_tool_trace, record_fallback, set_node_latency
from app.core.config import MCP_ENABLED
from app.tools.llm_client import get_llm
from app.tools.mcp_client import MCP_AVAILABLE, mcp_tavily_search, mcp_wikipedia_search
from app.tools.reranker import rerank_documents
from app.tools.vector_store import get_retriever
from app.tools.wikipedia_search import get_wikipedia_wrapper
from app.tools.tavily_search import get_tavily_search

MAX_ITER = 3  # ReAct 最大迭代轮数


# 工具注册表（供 LLM THINK 阶段参考）

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
    # 　 新增工具示例（只需在此处添加描述，并在 ACTION_REGISTRY 注册实现） 　
    # "medical_image_analysis": {
    #     "description": "分析医学影像（CT/MRI/X-ray）报告文本，解读关键发现。",
    #     "keywords": ["CT报告", "MRI报告", "X光", "影像", "病灶", "结节"],
    # },
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
        "indications": "退烧、缓解头痛、牙痛、肌肉痛",
        "dosage": "成人每次 325-650mg，每4-6小时一次，每日不超过 3000mg",
        "side_effects": "过量使用损伤肝脏",
        "warnings": "肝病患者慎用；避免与含酒精饮料同服；不要与其他含对乙酰氨基酚药物同服",
    },
    "阿莫西林": {
        "name": "阿莫西林（Amoxicillin）", "category": "青霉素类抗生素",
        "indications": "细菌性感染：上呼吸道感染、肺炎、尿路感染等",
        "dosage": "成人每次 250-500mg，每8小时一次，疗程遵医嘱",
        "side_effects": "腹泻、皮疹、过敏反应（严重时过敏性休克）",
        "warnings": "青霉素过敏者禁用；使用前需皮试；完成整个疗程，不可擅自停药",
    },
    # 　 从原版迁移的额外药品 　　　　　　
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
}

# 　 医学术语词库（从原版迁移，替代 _explain_medical_term 内的硬编码 dict） 　
MEDICAL_TERMS: Dict[str, str] = {
    "肌酐": "肌酐是肌肉代谢产物，通过肾脏过滤排出。血肌酐偏高提示肾脏过滤功能下降，需结合 eGFR 等指标综合判断。",
    "空腹血糖": "空腹血糖正常值为 3.9-6.1 mmol/L。6.1-7.0 为糖尿病前期，≥7.0 提示糖尿病（需复查确认）。",
    "糖化血红蛋白": "HbA1c 反映过去 2-3 个月平均血糖水平。< 5.7% 正常，5.7-6.4% 为糖尿病前期，≥ 6.5% 提示糖尿病。",
    "甘油三酯": "血脂指标之一，正常值 < 1.7 mmol/L。偏高与饮食、运动、遗传有关，增加心血管风险。",
    "ldl": "低密度脂蛋白（坏胆固醇），偏高增加动脉粥样硬化风险。建议 < 3.4 mmol/L。",
    "hdl": "高密度脂蛋白（好胆固醇），偏高有保护心血管作用。男性 > 1.0 mmol/L，女性 > 1.3 mmol/L 为正常。",
    "tsh": "促甲状腺激素，甲状腺功能评估的核心指标。偏高提示甲状腺功能减退，偏低提示甲亢。正常值约 0.4-4.0 mIU/L。",
    "alt": "丙氨酸转氨酶，最敏感的肝功能指标。偏高主要提示肝细胞损伤，正常上限约 40 U/L。",
    "转氨酶": "转氨酶（ALT/AST）是肝细胞内的酶。偏高（ALT>40U/L）通常提示肝细胞损伤，见于肝炎、脂肪肝等。",
    "白细胞": "血常规中白细胞计数反映免疫状态。偏高常见于细菌感染、炎症；偏低见于病毒感染、骨髓抑制等。",
    "血红蛋白": "反映是否贫血。男性正常 120-160 g/L，女性 110-150 g/L，低于正常值提示贫血。",
    "血糖": "空腹血糖正常值3.9-6.1mmol/L。6.1-7.0mmol/L为糖耐量异常；≥7.0mmol/L可诊断糖尿病。",
    "ct": "计算机断层扫描，通过X射线断层成像，适合骨骼、出血、肺部等检查。",
    "mri": "磁共振成像，无辐射，软组织分辨率高，适合脑部、脊髓、关节、腹部实质脏器检查。",
}


　
# 策略模式核心：ActionContext + ActionResult + ACTION_REGISTRY
　

@dataclass
class ActionContext:
    """
    每次 ACT 调用传入的统一上下文。
    策略函数通过此对象访问所需资源，不持有全局状态。

    字段设计原则：只传"当前 action 可能用到的"，避免传整个 state。
    """
    param: str                      # THINK 阶段生成的 action 参数
    question: str                   # 当前用户问题
    state: AgentState               # 完整 state 引用（写 metrics / flags 用）
    docs: List[Document]            # 当前已收集的文档（只读，写操作通过 ActionResult 返回）
    iteration: int                  # 当前迭代轮次
    history_context: str            # 对话历史文本
    long_term_prefix: str           # 长期记忆前缀
    llm: Any                        # LLM 实例（llm_direct 需要）
    replan_instruction: str = ""    # Planner 重规划指令


@dataclass
class ActionResult:
    """
    策略函数的统一返回值。主循环通过 _apply_result 将其合并到全局状态。

    exit_loop=True 时，主循环立即退出并写回 state，不再执行后续迭代。
    """
    new_docs: List[Document] = field(default_factory=list)
    tool_result: str = ""           # 非文档类工具的文本结果（天气/药品查询等）
    success: bool = False
    fallback_key: str = ""          # 失败时记录到 fallback_events 的 key
    exit_loop: bool = False         # True = 立即退出 ReAct 循环
    answer: str = ""                # exit_loop=True 时的预生成答案
    source: str = ""                # 数据来源描述


# 策略注册表：action name → ActionFn
ActionFn = Callable[[ActionContext], ActionResult]
ACTION_REGISTRY: Dict[str, ActionFn] = {}


def register_action(name: str) -> Callable[[ActionFn], ActionFn]:
    """
    装饰器：将函数注册为指定 action 的策略实现。

    用法：
        @register_action("my_action")
        def act_my_action(ctx: ActionContext) -> ActionResult:
            ...
    """
    def decorator(fn: ActionFn) -> ActionFn:
        ACTION_REGISTRY[name] = fn
        return fn
    return decorator


# 内化工具实现（从 research.py 迁移，保持原有逻辑）

def _run_tool(tool_name: str, tool_param: str) -> str:
    """执行结构化工具，返回文本结果。"""
    try:
        if tool_name == "get_weather":
            return _get_weather(tool_param)
        elif tool_name == "search_drug":
            return _search_drug(tool_param)
        elif tool_name == "explain_medical_term":
            return _explain_medical_term(tool_param)
        else:
            return f"工具 {tool_name} 未实现"
    except Exception as exc:
        return f"工具调用失败：{exc}"


def _get_weather(query: str) -> str:
    """
    天气查询。三级降级：
      1. Open-Meteo（主，含 geocoding 动态解析非内置城市）
      2. wttr.in（国内网络更稳定，含明日预报）
      3. 季节性启发式（完全离线时兜底）
    """
    # 识别城市：先查内置坐标，再调 geocoding API
    coords = None
    city = query
    for name, c in CITY_COORDS.items():
        if name in query or query in name:
            coords = c
            city = name
            break

    if not coords:
        try:
            geo_resp = httpx.get(
                f"https://geocoding-api.open-meteo.com/v1/search"
                f"?name={query}&count=1&language=zh&format=json",
                timeout=5.0,
            )
            geo_data = geo_resp.json()
            if geo_data.get("results"):
                r = geo_data["results"][0]
                coords = {"lat": r["latitude"], "lon": r["longitude"]}
                city = r.get("name", query)
        except Exception:
            pass  # geocoding 失败，继续尝试 wttr.in

    # 1. Open-Meteo（主通道）
    if coords:
        try:
            resp = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": coords["lat"], "longitude": coords["lon"],
                    "current_weather": True,
                    "hourly": "relativehumidity_2m,precipitation_probability",
                    "timezone": "Asia/Shanghai", "forecast_days": 1,
                },
                timeout=8.0,
            )
            resp.raise_for_status()
            data = resp.json()
            cw = data.get("current_weather", {})
            temp = cw.get("temperature", "N/A")
            wind = cw.get("windspeed", "N/A")
            code = cw.get("weathercode", -1)
            desc = (
                "晴朗" if code == 0 else
                "多云" if code in (1, 2, 3) else
                "有雨" if code in range(51, 100) else "其他"
            )
            hourly = data.get("hourly", {})
            humidity = hourly.get("relativehumidity_2m", [None])[0]
            rain_prob = hourly.get("precipitation_probability", [None])[0]
            return (
                f"{city}当前天气：{desc}，气温 {temp}°C，风速 {wind} km/h，"
                f"相对湿度 {humidity}%，降水概率 {rain_prob}%。"
            )
        except Exception as exc:
            logger.debug("Open-Meteo 查询失败，降级到 wttr.in：%s", exc)

    # 2. wttr.in（二级兜底，国内网络更稳定）
    try:
        city_name = city or query
        resp = httpx.get(f"https://wttr.in/{city_name}?format=j1&lang=zh", timeout=8.0)
        wdata = resp.json()
        cc = wdata.get("current_condition", [{}])[0]
        temp = cc.get("temp_C", "N/A")
        feels = cc.get("FeelsLikeC", "N/A")
        humidity = cc.get("humidity", "N/A")
        wind = cc.get("windspeedKmph", "N/A")
        desc = (
            cc.get("lang_zh", [{}])[0].get("value", "")
            if cc.get("lang_zh")
            else cc.get("weatherDesc", [{}])[0].get("value", "")
        )
        result = (
            f"{city_name}当前天气：{desc}，{temp}°C（体感{feels}°C），"
            f"湿度{humidity}%，风速{wind}km/h。"
        )
        forecasts = wdata.get("weather", [])
        if len(forecasts) >= 2:
            tmrw = forecasts[1]
            tmax = tmrw.get("maxtempC", "N/A")
            tmin = tmrw.get("mintempC", "N/A")
            result += f" 明天：最高{tmax}°C/最低{tmin}°C。"
        return result
    except Exception as exc:
        logger.debug("wttr.in 查询失败，使用季节启发式：%s", exc)

    # 3. 离线季节性启发式（终极兜底）
    city_name = city or query
    try:
        from datetime import datetime
        month = datetime.now().month
        season = (
            "春季（15-22°C，早晚温差大）" if 3 <= month <= 5 else
            "夏季（25-35°C，高温高湿）"   if 6 <= month <= 8 else
            "秋季（15-25°C，凉爽干燥）"   if 9 <= month <= 11 else
            "冬季（0-10°C，注意保暖）"
        )
    except Exception:
        season = "当前季节"
    return (
        f"【{city_name}天气】天气服务暂时不可用，{city_name}当前为{season}。"
        "风湿病患者对天气变化敏感，建议做好保暖防潮准备。"
    )


def _search_drug(query: str) -> str:
    drug = next((d for d in DRUG_DICT if d in query), None)
    if drug:
        info = DRUG_DICT[drug]
        return (
            f"【{info['name']}】\n类别：{info['category']}\n适应症：{info['indications']}\n"
            f"常规剂量：{info['dosage']}\n常见副作用：{info['side_effects']}\n注意事项：{info['warnings']}\n"
            "⚠️ 具体用药请遵医嘱或咨询执业药师。"
        )
    # 本地词库未命中：MCP Tavily 联网兜底
    if MCP_ENABLED and MCP_AVAILABLE:
        try:
            results = mcp_tavily_search(
                query=f"{query} 药品 说明书 用法 副作用", max_results=2
            )
            if results:
                parts = [f"【{r['title']}】{r['content'][:500]}" for r in results]
                return (
                    "（本地数据库未收录，以下为联网检索结果）\n\n"
                    + "\n\n".join(parts)
                    + "\n⚠️ 具体用药请遵医嘱或咨询执业药师。"
                )
        except Exception as exc:
            logger.debug("MCP 药品兜底检索失败：%s", exc)
    return f'未找到"{query}"的药品信息，请尝试使用通用名称（如：布洛芬、对乙酰氨基酚）。'


def _explain_medical_term(query: str) -> str:
    # 使用模块级 MEDICAL_TERMS 词库（12 条），替代原来的 4 条硬编码
    for term, explanation in MEDICAL_TERMS.items():
        if term in query.lower():
            return explanation
    # 本地词库未命中：MCP Tavily 联网兜底
    if MCP_ENABLED and MCP_AVAILABLE:
        try:
            results = mcp_tavily_search(
                query=f"{query} 医学术语 解释 临床意义", max_results=2
            )
            if results:
                parts = [f"【{r['title']}】{r['content'][:500]}" for r in results]
                return "（本地词库未收录，以下为联网检索结果）\n\n" + "\n\n".join(parts)
        except Exception as exc:
            logger.debug("MCP 术语兜底检索失败：%s", exc)
    return f'"{query}"是一个医学专业术语，建议结合检查报告向主治医生咨询。'


# 策略实现（每种 action 一个独立函数）

def _rag_fetch(query: str, state: AgentState, skip_coverage_check: bool = False) -> List[Document]:
    """
    核心 RAG 检索逻辑（带知识库盲区检测 + Reranker）。
    抽离为独立函数，供 rag_search / expand_query / decompose 复用。
    """
    retriever = get_retriever()
    if not retriever:
        return []

    # 知识库盲区快速检测（仅首次检索触发）
    if not skip_coverage_check and not state.get("documents"):
        try:
            from app.tools.vector_store import check_coverage
            coverage = check_coverage(query, threshold=0.15)
            if not coverage["covered"]:
                logger.info(
                    "ResearchAgent 知识库盲区：%s（top_score=%.3f）",
                    coverage["suggestion"], coverage["top_score"],
                )
                state["rag_blind_spot"] = True
                state["rag_blind_score"] = coverage["top_score"]
                return []
        except Exception as exc:
            logger.debug("盲区检测失败（忽略）：%s", exc)

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


@register_action("rag_search")
def act_rag_search(ctx: ActionContext) -> ActionResult:
    """直接 RAG 检索（带盲区检测 + Reranker）。"""
    new_docs = _rag_fetch(ctx.param, ctx.state)
    if new_docs:
        ctx.state["rag_attempted"] = True
        ctx.state["rag_success"] = True
        ctx.state["metrics"]["rag_hit"] = True
        logger.info("act_rag_search: 检索到 %d 篇文档", len(new_docs))
        return ActionResult(new_docs=new_docs, success=True)

    record_fallback(ctx.state, f"rag_search_empty_iter{ctx.iteration}")
    return ActionResult(success=False, fallback_key=f"rag_search_empty_iter{ctx.iteration}")


@register_action("expand_query")
def act_expand_query(ctx: ActionContext) -> ActionResult:
    """扩展查询词后重检索（与 rag_search 共享实现，语义上区分意图）。"""
    # 注意：expand_query 跳过盲区检测（已知有文档的情况下扩展）
    new_docs = _rag_fetch(ctx.param, ctx.state, skip_coverage_check=bool(ctx.docs))
    if new_docs:
        ctx.state["rag_attempted"] = True
        ctx.state["rag_success"] = True
        ctx.state["metrics"]["rag_hit"] = True
        logger.info("act_expand_query: 扩展检索到 %d 篇文档", len(new_docs))
        return ActionResult(new_docs=new_docs, success=True)

    record_fallback(ctx.state, f"expand_query_empty_iter{ctx.iteration}")
    return ActionResult(success=False, fallback_key=f"expand_query_empty_iter{ctx.iteration}")


@register_action("decompose")
def act_decompose(ctx: ActionContext) -> ActionResult:
    """拆解复杂问题为子查询，分别检索后合并去重。"""
    sub_queries = [q.strip() for q in ctx.param.split(",") if q.strip()]
    all_new_docs: List[Document] = []
    existing_contents = {d.page_content for d in ctx.docs}

    for sq in sub_queries[:3]:
        sub_docs = _rag_fetch(sq, ctx.state, skip_coverage_check=True)
        for d in sub_docs:
            if d.page_content not in existing_contents:
                all_new_docs.append(d)
                existing_contents.add(d.page_content)

    ctx.state["rag_attempted"] = True
    if all_new_docs:
        ctx.state["rag_success"] = True
        logger.info("act_decompose: 子查询合并得到 %d 篇新文档", len(all_new_docs))
        return ActionResult(new_docs=all_new_docs, success=True)

    record_fallback(ctx.state, f"decompose_empty_iter{ctx.iteration}")
    return ActionResult(success=False, fallback_key=f"decompose_empty_iter{ctx.iteration}")


@register_action("tool_query")
def act_tool_query(ctx: ActionContext) -> ActionResult:
    """
    调用结构化工具。
    param 格式：tool_name|查询参数

    工具失败时自动兜底 Tavily（若尚未尝试），无需等待下一轮 THINK 决策。
    这是关键的性能优化：避免在 MAX_ITER 末尾才触发兜底。
    """
    parts = ctx.param.split("|", 1)
    tool_name = parts[0].strip()
    tool_param = parts[1].strip() if len(parts) > 1 else ctx.question

    if tool_name not in TOOL_REGISTRY:
        record_fallback(ctx.state, f"unknown_tool:{tool_name}")
        logger.warning("act_tool_query: 工具 %s 未注册，可用：%s", tool_name, list(TOOL_REGISTRY))
        return ActionResult(success=False, fallback_key=f"unknown_tool:{tool_name}")

    tool_result = _run_tool(tool_name, tool_param)
    ctx.state["tool_results"] = {"tool": tool_name, "result": tool_result}
    ctx.state["tool_agent_success"] = True
    logger.info("act_tool_query: %s → %s", tool_name, tool_result[:60])

    # 工具失败自动兜底 Tavily
    _FAIL_INDICATORS = ("失败", "Error", "error", "未能", "无法", "timeout")
    _has_good_docs = any(d.metadata.get("rerank_score", 0) >= 0.5 for d in ctx.docs)
    is_fail = any(ind in tool_result for ind in _FAIL_INDICATORS)

    if is_fail and not _has_good_docs:
        logger.info("act_tool_query: %s 失败，自动兜底→Tavily", tool_name)
        tavily_result = act_tavily(ActionContext(
            param=tool_param, question=ctx.question, state=ctx.state,
            docs=ctx.docs, iteration=ctx.iteration,
            history_context=ctx.history_context,
            long_term_prefix=ctx.long_term_prefix, llm=ctx.llm,
        ))
        if tavily_result.success:
            return ActionResult(
                new_docs=tavily_result.new_docs,
                tool_result=tool_result,  # 保留失败结果，答案生成时会忽略
                success=True,
                source="实时医学搜索（工具失败兜底）",
            )

    return ActionResult(
        tool_result=tool_result,
        success=not is_fail,
    )


@register_action("wikipedia")
def act_wikipedia(ctx: ActionContext) -> ActionResult:
    """Wikipedia 医学知识检索。MCP 优先 → LangChain wrapper 降级。"""
    ctx.state["wiki_attempted"] = True

    # 通道 1：MCP（低延迟，连接池复用）
    if MCP_ENABLED and MCP_AVAILABLE:
        try:
            mcp_results = mcp_wikipedia_search(ctx.param)
            if mcp_results:
                existing_titles = {d.metadata.get("title", "") for d in ctx.docs}
                new_docs = [
                    Document(
                        page_content=r["content"],
                        metadata={"url": r.get("url", ""), "title": r["title"]},
                    )
                    for r in mcp_results
                    if r["title"] not in existing_titles
                ]
                ctx.state["wiki_success"] = True
                ctx.state["source"] = "Wikipedia 医学资料"
                logger.info("act_wikipedia [MCP]: 成功 %d 条", len(mcp_results))
                return ActionResult(new_docs=new_docs, success=True, source="Wikipedia 医学资料")
        except Exception as exc:
            logger.debug("MCP Wikipedia 失败，降级到 LangChain wrapper：%s", exc)

    # 通道 2：LangChain wrapper
    wiki = get_wikipedia_wrapper()
    if not wiki:
        return ActionResult(success=False, fallback_key="wikipedia_not_available")

    try:
        content = wiki.run(f"{ctx.param} 医学 症状 治疗")
        if content and len(content.strip()) > 100:
            ctx.state["wiki_success"] = True
            ctx.state["source"] = "Wikipedia 医学资料"
            logger.info("act_wikipedia [LangChain]: 检索成功 %d 字符", len(content))
            return ActionResult(new_docs=[Document(page_content=content)], success=True, source="Wikipedia 医学资料")

        ctx.state["wiki_success"] = False
        record_fallback(ctx.state, f"wiki_empty_iter{ctx.iteration}")
        return ActionResult(success=False, fallback_key=f"wiki_empty_iter{ctx.iteration}")
    except Exception as exc:
        logger.error("act_wikipedia [LangChain]: 检索异常 %s", exc)
        record_fallback(ctx.state, f"wiki_exception:{exc}")
        return ActionResult(success=False, fallback_key=f"wiki_exception:{exc}")


@register_action("tavily")
def act_tavily(ctx: ActionContext) -> ActionResult:
    """Tavily 联网搜索（最终兜底）。MCP 优先 → LangChain wrapper 降级。"""
    ctx.state["tavily_attempted"] = True

    # 通道 1：MCP
    if MCP_ENABLED and MCP_AVAILABLE:
        try:
            mcp_results = mcp_tavily_search(ctx.param, max_results=3)
            if mcp_results:
                new_docs = [
                    Document(
                        page_content=r["content"],
                        metadata={"url": r.get("url", ""), "title": r.get("title", "")},
                    )
                    for r in mcp_results
                ]
                ctx.state["tavily_success"] = True
                ctx.state["source"] = "实时医学搜索"
                logger.info("act_tavily [MCP]: 成功 %d 条", len(mcp_results))
                return ActionResult(new_docs=new_docs, success=True, source="实时医学搜索")
        except Exception as exc:
            logger.debug("MCP Tavily 失败，降级到 LangChain wrapper：%s", exc)

    # 通道 2：LangChain wrapper
    tavily = get_tavily_search()
    if not tavily:
        return ActionResult(success=False, fallback_key="tavily_not_available")

    try:
        results = tavily.invoke(ctx.param)
        valid = [
            r for r in (results or [])
            if isinstance(r, dict) and len(r.get("content", "")) > 50
        ]
        if valid:
            new_docs = [
                Document(
                    page_content=r["content"],
                    metadata={"url": r.get("url", ""), "title": r.get("title", "")},
                )
                for r in valid
            ]
            ctx.state["tavily_success"] = True
            ctx.state["source"] = "实时医学搜索"
            logger.info("act_tavily: 检索成功 %d 条结果", len(valid))
            return ActionResult(new_docs=new_docs, success=True, source="实时医学搜索")

        ctx.state["tavily_success"] = False
        record_fallback(ctx.state, f"tavily_empty_iter{ctx.iteration}")
        return ActionResult(success=False, fallback_key=f"tavily_empty_iter{ctx.iteration}")
    except Exception as exc:
        logger.error("act_tavily: 检索异常 %s", exc)
        record_fallback(ctx.state, f"tavily_exception:{exc}")
        return ActionResult(success=False, fallback_key=f"tavily_exception:{exc}")


@register_action("llm_direct")
def act_llm_direct(ctx: ActionContext) -> ActionResult:
    """
    跳过检索，直接用 LLM 回答通用医疗问题。
    触发后 exit_loop=True，主循环立即退出。
    """
    answer, source = _generate_answer(
        question=ctx.question,
        docs=[],
        tool_result="",
        history_context=ctx.history_context,
        long_term_prefix=ctx.long_term_prefix,
        llm=ctx.llm,
    )
    ctx.state["metrics"]["llm_used"] = True
    ctx.state["llm_attempted"] = True
    ctx.state["llm_success"] = bool(answer and len(answer) > 10)
    logger.info("act_llm_direct: 完成 answer_len=%d", len(answer))
    return ActionResult(
        exit_loop=True,
        answer=answer,
        source=source,
        success=bool(answer and len(answer) > 10),
    )


# 　 新增工具示例（只需添加这个函数 + 注册一行，主循环不需要改任何代码） 　　　
# @register_action("medical_image_analysis")
# def act_medical_image_analysis(ctx: ActionContext) -> ActionResult:
#     """分析医学影像报告文本，提取关键发现。"""
#     # 调用影像分析服务...
#     result = image_analysis_service.analyze(ctx.param)
#     return ActionResult(tool_result=result, success=True)


# 主循环辅助：应用 ActionResult 到全局 docs 和 state

def _apply_result(
    result: ActionResult,
    docs: List[Document],
    state: AgentState,
    tool_result_holder: List[str],
) -> None:
    """
    将 ActionResult 合并到全局状态。

    之所以需要 tool_result_holder（list 包装）是因为 Python 对基本类型不支持
    引用传递，用 list 模拟可变引用。
    """
    # 合并新文档（去重）
    if result.new_docs:
        existing = {d.page_content for d in docs}
        for d in result.new_docs:
            if d.page_content not in existing:
                docs.append(d)
                existing.add(d.page_content)

    # 更新 tool_result（工具查询类 action）
    if result.tool_result:
        tool_result_holder[0] = result.tool_result

    # 更新来源
    if result.source:
        state["source"] = result.source

    # 记录失败事件
    if result.fallback_key:
        record_fallback(state, result.fallback_key)


# THINK 阶段：LLM 分析当前信息质量，决定下一步 ACT

_ACT_CHOICES = list(ACTION_REGISTRY.keys()) + ["accept"]
_TOOLS_DESC = "\n".join(f'  - "{k}": {v["description"]}' for k, v in TOOL_REGISTRY.items())


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

    blind_spot_hint = ""
    if rag_blind_spot:
        blind_spot_hint = (
            f"\n\n【知识库盲区警告】本地向量库无相关内容（top_rerank={rag_blind_score:.3f}<0.15）。"
            "请直接选择 tavily 或 wikipedia，不要再 expand_query。"
        )

    used_str = "、".join(used_actions) if used_actions else "无"

    prev_score_hint = ""
    if prev_scores and iteration > 0:
        prev_score_hint = (
            f"\n\n【上一轮评分参考】relevance={prev_scores.get('relevance',0):.1f}，"
            f"coverage={prev_scores.get('coverage',0):.1f}，"
            f"medical_depth={prev_scores.get('medical_depth',0):.1f}。"
            "如果本轮文档在此基础上有改善且三项均≥6，请直接选择 accept。"
        )

    # 　 无文档时直接 expand_query，不调 LLM 　　　　　
    if not docs:
        return {
            "scores": None,
            "action": "expand_query",
            "param": question,
            "reason": "当前无文档，需扩展查询。",
        }

    # 　 硬性 early-exit：rerank_score ≥ 阈值直接 accept，跳过 LLM 　　　
    scores = [d.metadata.get("rerank_score", 0.0) for d in docs]
    top_rerank_score = max(scores) if scores else 0.0

    EARLY_ACCEPT_THRESHOLD = 0.85 if iteration == 0 else 0.80
    _URGENT_KEYWORDS = ["急", "立即", "马上", "怎么办", "救", "危险", "紧急"]
    if any(kw in question for kw in _URGENT_KEYWORDS):
        EARLY_ACCEPT_THRESHOLD = min(EARLY_ACCEPT_THRESHOLD, 0.75)

    if top_rerank_score >= EARLY_ACCEPT_THRESHOLD:
        logger.info(
            "THINK [iter=%d] early-exit: rerank=%.3f ≥ %.2f，跳过 LLM",
            iteration, top_rerank_score, EARLY_ACCEPT_THRESHOLD,
        )
        logger.debug(
            "EARLY_EXIT_LOG | question=%s | iter=%d | rerank_score=%.3f | threshold=%.2f",
            question[:50], iteration, top_rerank_score, EARLY_ACCEPT_THRESHOLD,
        )
        return {
            "relevance": 8.0, "coverage": 7.0, "medical_depth": 7.0,
            "action": "accept",
            "param": "",
            "reason": (
                f"Reranker 客观评分 {top_rerank_score:.3f} ≥ 动态阈值 {EARLY_ACCEPT_THRESHOLD}，"
                "文档高度相关，直接 accept。"
            ),
        }

    rerank_hint = (
        f"\n\n【Reranker 客观评分】Top 文档分数={top_rerank_score:.3f}（满分1.0）。"
        "分数≥0.80 直接 accept；0.60-0.80 酌情；<0.60 继续检索。"
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
        "【accept 触发规则（满足任一即可）】\n"
        "  ① 三项评分均≥6 且已有文档\n"
        "  ② 已执行 expand_query 且本轮评分相比上轮提升 < 0.5（边际收益递减）\n"
        "  ③ 已是最后一轮且有文档\n\n"
        "可选动作：\n"
        "  - accept：满足上述规则，退出循环生成答案\n"
        "  - expand_query：文档不足且未连续扩展两轮（param: 扩展查询词）\n"
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

    # 降级：有文档则接受，无文档则检索
    return {
        "relevance": 5.0, "coverage": 5.0, "medical_depth": 5.0,
        "action": "rag_search" if not docs else "accept",
        "param": question,
        "reason": "LLM 不可用，使用启发式策略。",
    }


# 最终答案生成

def _generate_answer(
    question: str,
    docs: List[Document],
    tool_result: str,
    history_context: str,
    long_term_prefix: str,
    llm,
) -> Tuple[str, str]:
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
            "【硬约束】不得编造具体数据、剂量、研究结论。用3-5句中文回答。\n\n"
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
            "【硬约束】仅使用工具结果中提供的信息。用2-4句通俗、温和的中文回答。\n\n"
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
            "【硬约束】不得编造具体数据、剂量。不得使用确诊性语言。用3-5句中文回答。\n"
            "最后提醒用户必要时及时线下就医。\n\n"
            + long_term_prefix
            + f"对话历史：\n{history_context}\n"
            f"当前问题：\n{question}\n\n"
            f"医学资料：\n{content}\n"
        )
        source = "医学知识库"
    else:
        prompt = (
            "你是一名专业、稳健的中文医疗AI助手。\n"
            "当前没有检索到相关医学资料，请仅基于通用医学常识回答，并添加免责声明。\n"
            "【硬约束】不要给出超出常识范围的具体剂量、药名、治疗方案。\n\n"
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


# ResearchAgent 主函数（策略模式重构后，ACT 分发仅 5 行）

def ResearchAgent(state: AgentState) -> AgentState:
    """
    自适应检索 Agent。
    通过 THINK-ACT-OBSERVE 循环，自主决定使用哪些工具完成检索任务。

    核心变化
    --------
    原实现：ACT 分发是一个 120 行 if-elif 链，新增工具需在链中插入。
    重构后：ACT 分发 = 3 行（查注册表 → 构造 context → 调策略）。
             新增工具只需实现 ActionFn 并 @register_action 注册，主循环零改动。
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

    # 　 初始化本轮状态 　　　　　　　
    docs: List[Document] = list(state.get("documents") or [])
    tool_result_holder: List[str] = [""]   # 可变引用包装
    used_actions: List[str] = []
    think_log: List[Dict] = list(state.get("rag_think_log") or [])

    # 　 Planner 路由到 llm_agent，直接跳过检索 　　　　
    if route_tool == "llm_agent" and not replan_instruction:
        ctx = ActionContext(
            param=question, question=question, state=state, docs=docs,
            iteration=0, history_context=history_context,
            long_term_prefix=long_term_prefix, llm=llm,
        )
        result = act_llm_direct(ctx)
        state["generation"] = result.answer
        state["source"] = result.source
        state["research_strategy"] = "llm_direct"
        set_node_latency(state, "research", (perf_counter() - start_time) * 1000)
        logger.info("ResearchAgent [llm_direct] 路径完成")
        return state

    # 　 ReAct 主循环 　　　　　　
    prev_scores: Optional[Dict] = None
    for iteration in range(MAX_ITER):

        # 最后一轮强制 accept
        if iteration == MAX_ITER - 1 and docs:
            think_result = {
                "relevance": 7.0, "coverage": 7.0, "medical_depth": 7.0,
                "action": "accept", "param": "",
                "reason": "已达最大迭代轮数，强制退出生成答案。",
                "iteration": iteration,
            }
            think_log.append(think_result)
            logger.info("ResearchAgent THINK [iter=%d] 强制 accept（MAX_ITER）", iteration)
            break

        # 　 盲区强制路由（不依赖 LLM 决策，直接选最优工具） 　　　
        if (state.get("rag_blind_spot")
                and "tavily" not in used_actions
                and "wikipedia" not in used_actions):
            _WEATHER_KW = ["天气", "气温", "下雨", "下雪", "刮风", "湿度", "降温", "升温"]
            _skip_weather = "get_weather" in replan_instruction
            if (any(kw in question for kw in _WEATHER_KW)
                    and "tool_query" not in used_actions
                    and not _skip_weather):
                think_result = {
                    "action": "tool_query",
                    "param": f"get_weather|{question}",
                    "reason": f"知识库盲区（score={state.get('rag_blind_score',0):.3f}），天气相关→get_weather。",
                    "iteration": iteration,
                }
            else:
                think_result = {
                    "action": "tavily",
                    "param": question,
                    "reason": f"知识库盲区（score={state.get('rag_blind_score',0):.3f}），强制→tavily。",
                    "iteration": iteration,
                }
        else:
            think_result = _think(
                question, docs, iteration, used_actions,
                replan_instruction, llm, prev_scores,
                rag_blind_spot=state.get("rag_blind_spot", False),
                rag_blind_score=state.get("rag_blind_score", 0.0),
            )

        think_result["iteration"] = iteration
        think_log.append(think_result)
        prev_scores = {
            "relevance": think_result.get("relevance", 5.0),
            "coverage": think_result.get("coverage", 5.0),
            "medical_depth": think_result.get("medical_depth", 5.0),
        }

        action = think_result["action"]
        param = think_result.get("param", question)
        logger.info(
            "ResearchAgent THINK [iter=%d] action=%s param=%s reason=%s",
            iteration, action, param[:40], think_result.get("reason", ""),
        )
        used_actions.append(action)

        # 　 ACT：策略模式分发（原 120 行 if-elif → 现在 5 行） 　
        if action == "accept":
            break

        strategy = ACTION_REGISTRY.get(action)
        if strategy is None:
            # 未知 action（LLM 幻觉生成了不存在的 action name）
            logger.warning("ResearchAgent: 未知 action=%s，跳过", action)
            record_fallback(state, f"unknown_action:{action}")
            continue

        ctx = ActionContext(
            param=param,
            question=question,
            state=state,
            docs=list(docs),          # 快照：避免策略函数意外修改
            iteration=iteration,
            history_context=history_context,
            long_term_prefix=long_term_prefix,
            llm=llm,
            replan_instruction=replan_instruction,
        )
        result = strategy(ctx)        # 调用策略

        # 　 OBSERVE：处理 exit_loop（llm_direct 等直接退出的策略） 　
        if result.exit_loop:
            state["generation"] = result.answer
            state["source"] = result.source
            state["rag_grader_passed"] = bool(docs or tool_result_holder[0])
            state["rag_iterations"] = iteration + 1
            state["rag_think_log"] = think_log
            state["research_strategy"] = ",".join(dict.fromkeys(used_actions))
            state["documents"] = docs
            state["conversation_history"].append({"role": "user", "content": question})
            state["conversation_history"].append({
                "role": "assistant", "content": result.answer, "source": result.source
            })
            set_node_latency(state, "research", (perf_counter() - start_time) * 1000)
            return state

        _apply_result(result, docs, state, tool_result_holder)
        # 　　　　　　　

    # 　 退出循环：生成最终答案 　　　　　
    answer, source = _generate_answer(
        question, docs, tool_result_holder[0], history_context, long_term_prefix, llm
    )

    state["documents"] = docs
    state["generation"] = answer
    state["source"] = source
    state["rag_grader_passed"] = bool(docs or tool_result_holder[0])
    state["rag_iterations"] = len([t for t in think_log if t.get("action") != "accept"])
    state["rag_think_log"] = think_log
    state["research_strategy"] = ",".join(dict.fromkeys(used_actions))
    state["llm_attempted"] = True
    state["llm_success"] = bool(answer and len(answer) > 10)
    state["metrics"]["llm_used"] = True

    state["conversation_history"].append({"role": "user", "content": question})
    state["conversation_history"].append({"role": "assistant", "content": answer, "source": source})

    set_node_latency(state, "research", (perf_counter() - start_time) * 1000)
    logger.info(
        "ResearchAgent 完成 | 策略=%s | docs=%d | answer_len=%d",
        state["research_strategy"], len(docs), len(answer),
    )
    return state