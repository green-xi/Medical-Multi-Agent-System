"""跨会话持久化用户医疗画像与关键事实（user_profile / medical_fact）。"""

from __future__ import annotations

import json
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from app.core.logging_config import logger

# ── 提取配置 ───────────────────────────────────────────────────────────────────
# LLM 需要从对话中尝试提取的字段（key → 中文描述）
EXTRACT_FIELDS: Dict[str, str] = {
    # 用户画像
    "age":            "患者年龄（如：35岁；不确定则留空）",
    "gender":         "患者性别（男/女/未知）",
    "allergies":      "过敏史（药物或食物过敏，多个用顿号分隔；无则填'无'）",
    "conditions":     "既往病史或慢性病（如高血压、糖尿病；无则填'无'）",
    "medications":    "当前用药情况（药品名+剂量；无则填'无'）",
    # 本次就诊事实
    "chief_complaint": "本次主诉（患者最主要的不适或问题，一句话概括）",
    "symptoms":        "症状列表（逗号分隔，如：发烧、头痛、咳嗽）",
    "symptom_duration": "症状持续时长（如：3天、1周；不明确则留空）",
    "diagnosis":       "诊断或疑似诊断（若对话中有明确提及）",
    "advice":          "给出的主要医疗建议或用药指导（一句话概括）",
}

# 字段到记忆类型的映射
FIELD_TYPE_MAP: Dict[str, str] = {
    "age":             "user_profile",
    "gender":          "user_profile",
    "allergies":       "user_profile",
    "conditions":      "user_profile",
    "medications":     "user_profile",
    "chief_complaint": "medical_fact",
    "symptoms":        "medical_fact",
    "symptom_duration":"medical_fact",
    "diagnosis":       "medical_fact",
    "advice":          "medical_fact",
}

# 字段重要性分值（1~10）
FIELD_IMPORTANCE: Dict[str, int] = {
    "age": 6, "gender": 5, "allergies": 9, "conditions": 9,
    "medications": 9, "chief_complaint": 8, "symptoms": 8,
    "symptom_duration": 6, "diagnosis": 9, "advice": 8,
}


class LongTermMemoryService:
    """
    LongTermMemoryService（依赖 SQLAlchemy Session，通过 DatabaseService 连接池工作）。
    """

    def __init__(self, session_factory=None):
        self._session_factory = session_factory

    def _get_session(self):
        if self._session_factory:
            return self._session_factory()
        from app.db.session import SessionLocal
        return SessionLocal()

    # ── 写入 ──────────────────────────────────────────────────────────────────

    def upsert(
        self,
        session_id: str,
        memory_type: str,
        key: str,
        value: str,
        importance: int = 5,
        source_turn: Optional[int] = None,
    ) -> None:
        """写入或更新一条长期记忆（相同 session_id + memory_type + key 则覆盖）。"""
        if not value or value.strip() in ("", "无", "不确定", "未知", "N/A"):
            return  # 无效值不写入

        from app.models.user_memory import UserMemory

        with self._get_session() as db:
            # SQLite upsert：ON CONFLICT DO UPDATE
            stmt = (
                sqlite_upsert(UserMemory)
                .values(
                    session_id=session_id,
                    memory_type=memory_type,
                    key=key,
                    value=value.strip(),
                    importance=importance,
                    source_turn=source_turn,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                .on_conflict_do_update(
                    index_elements=["session_id", "memory_type", "key"],
                    set_={
                        "value": value.strip(),
                        "importance": importance,
                        "source_turn": source_turn,
                        "updated_at": datetime.utcnow(),
                    },
                )
            )
            db.execute(stmt)
            db.commit()

    def upsert_batch(self, session_id: str, entries: List[Dict[str, Any]]) -> None:
        """批量写入多条长期记忆。"""
        for entry in entries:
            self.upsert(
                session_id=session_id,
                memory_type=entry.get("memory_type", "medical_fact"),
                key=entry["key"],
                value=entry["value"],
                importance=entry.get("importance", 5),
                source_turn=entry.get("source_turn"),
            )

    # ── 读取 ──────────────────────────────────────────────────────────────────

    def load(
        self,
        session_id: str,
        memory_types: Optional[List[str]] = None,
        min_importance: int = 4,
    ) -> List[Dict]:
        """
        加载指定用户的长期记忆条目。
        """
        from app.models.user_memory import UserMemory

        with self._get_session() as db:
            stmt = (
                select(UserMemory)
                .where(UserMemory.session_id == session_id)
                .where(UserMemory.importance >= min_importance)
                .order_by(UserMemory.importance.desc(), UserMemory.updated_at.desc())
            )
            if memory_types:
                stmt = stmt.where(UserMemory.memory_type.in_(memory_types))

            rows = db.execute(stmt).scalars().all()
            return [r.to_dict() for r in rows]

    def load_profile(self, session_id: str) -> Dict[str, str]:
        """加载用户画像，返回 {key: value} 字典。"""
        rows = self.load(session_id, memory_types=["user_profile"])
        return {r["key"]: r["value"] for r in rows}

    def load_recent_facts(self, session_id: str, limit: int = 5) -> List[Dict]:
        """加载最近提取的医疗事实（按更新时间倒序）。"""
        from app.models.user_memory import UserMemory

        with self._get_session() as db:
            stmt = (
                select(UserMemory)
                .where(UserMemory.session_id == session_id)
                .where(UserMemory.memory_type == "medical_fact")
                .order_by(UserMemory.updated_at.desc())
                .limit(limit)
            )
            rows = db.execute(stmt).scalars().all()
            return [r.to_dict() for r in rows]

    def delete_all(self, session_id: str) -> None:
        """删除指定用户的全部长期记忆（通常在用户主动清除时调用）。"""
        from app.models.user_memory import UserMemory

        with self._get_session() as db:
            db.execute(
                delete(UserMemory).where(UserMemory.session_id == session_id)
            )
            db.commit()
        logger.info("已删除会话 %s 的全部长期记忆", session_id[:8])

    # ── LLM 提取 ──────────────────────────────────────────────────────────────

    def extract_and_save(
        self,
        session_id: str,
        question: str,
        answer: str,
        turn_index: int = 0,
        llm=None,
    ) -> Dict[str, str]:
        """
        从一轮 Q&A 中提取医疗信息并写入长期记忆。
        """
        extracted = self._extract_with_llm(question, answer, llm)
        if not extracted:
            return {}

        entries = []
        for key, value in extracted.items():
            if not value or value.strip() in ("", "无", "不确定", "未知"):
                continue
            entries.append({
                "memory_type": FIELD_TYPE_MAP.get(key, "medical_fact"),
                "key": key,
                "value": value,
                "importance": FIELD_IMPORTANCE.get(key, 5),
                "source_turn": turn_index,
            })

        if entries:
            self.upsert_batch(session_id, entries)
            logger.info(
                "长期记忆：会话 %s 提取 %d 个字段",
                session_id[:8], len(entries),
            )

        return extracted

    def _extract_with_llm(
        self,
        question: str,
        answer: str,
        llm=None,
    ) -> Dict[str, str]:
        """调用 LLM 从 Q&A 中提取结构化医疗信息，返回 JSON 字典。"""
        if llm is None:
            try:
                from app.tools.llm_client import get_llm
                llm = get_llm()
            except Exception:
                return {}

        if llm is None:
            return {}

        fields_desc = "\n".join(
            f'  "{k}": "{v}"' for k, v in EXTRACT_FIELDS.items()
        )

        prompt = (
            "你是一名医疗信息提取助手。请从以下一轮医疗问答中提取结构化信息。\n"
            "只提取对话中明确提及的信息，不要推断或捏造。\n"
            "对于未提及的字段，返回空字符串 \"\"。\n"
            "只返回 JSON 对象，不要任何解释或代码块标记。\n\n"
            f"需要提取的字段：\n{fields_desc}\n\n"
            f"患者问题：{question}\n"
            f"助手回答：{answer}\n\n"
            "返回格式（严格 JSON）："
        )

        try:
            t0 = perf_counter()
            response = llm.invoke(prompt)
            raw = (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )
            # 清理可能的 markdown 代码块
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

            result = json.loads(raw)
            logger.debug(
                "长期记忆提取耗时 %.1fms，字段数：%d",
                (perf_counter() - t0) * 1000,
                len([v for v in result.values() if v]),
            )
            return {k: str(v) for k, v in result.items() if v}
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("长期记忆提取失败（JSON 解析错误）：%s", exc)
            return {}

    # ── 格式化为 Prompt 文本 ─────────────────────────────────────────────────

    def format_for_prompt(self, session_id: str) -> str:
        """
        将长期记忆格式化为可直接注入 LLM 提示词的文本块。
        """
        all_memories = self.load(session_id)
        if not all_memories:
            return ""

        profile = {m["key"]: m["value"] for m in all_memories if m["memory_type"] == "user_profile"}
        facts   = {m["key"]: m["value"] for m in all_memories if m["memory_type"] == "medical_fact"}

        lines = []

        if profile:
            profile_parts = []
            label_map = {
                "age": "年龄", "gender": "性别", "allergies": "过敏史",
                "conditions": "既往病史", "medications": "当前用药",
            }
            for key, label in label_map.items():
                if key in profile:
                    profile_parts.append(f"· {label}：{profile[key]}")
            if profile_parts:
                lines.append("【患者档案】")
                lines.extend(profile_parts)

        if facts:
            fact_parts = []
            label_map = {
                "chief_complaint": "主诉", "symptoms": "主要症状",
                "symptom_duration": "持续时长", "diagnosis": "诊断建议",
                "advice": "医嘱",
            }
            for key, label in label_map.items():
                if key in facts:
                    fact_parts.append(f"· {label}：{facts[key]}")
            if fact_parts:
                lines.append("【近期就诊记录】")
                lines.extend(fact_parts)

        return "\n".join(lines) if lines else ""


# 模块级单例
long_term_memory = LongTermMemoryService()
