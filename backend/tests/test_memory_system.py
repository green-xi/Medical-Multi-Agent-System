"""
test_memory_system.py — 记忆系统测试

覆盖：
  - LongTermMemoryService（upsert / load / format_for_prompt / extract_and_save）
  - short_term compress_history / build_context_window
"""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import get_engine, get_session_factory
from app.memory.long_term import LongTermMemoryService
from app.memory.short_term import build_context_window, compress_history
from app.models.user_memory import UserMemory


# ══════════════════════════════════════════════════════════════════════════════
# 测试用数据库 fixture
# ══════════════════════════════════════════════════════════════════════════════

TEST_DB = "tests/test_memory/memory_test.db"


@pytest.fixture
def memory_service():
    """使用独立 SQLite 的 LongTermMemoryService 实例。"""
    os.makedirs(os.path.dirname(TEST_DB), exist_ok=True)
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass

    engine = get_engine(TEST_DB)
    session_factory = get_session_factory(engine)

    # 建表
    from sqlalchemy import inspect
    from app.models.user_memory import Base
    Base.metadata.create_all(engine)

    service = LongTermMemoryService(session_factory=session_factory)
    yield service

    engine.dispose()
    try:
        os.remove(TEST_DB)
    except (PermissionError, FileNotFoundError):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# LongTermMemoryService — upsert / load
# ══════════════════════════════════════════════════════════════════════════════

class TestLongTermMemoryUpsertLoad:

    def test_upsert_and_load_single_field(self, memory_service):
        memory_service.upsert("sess-001", "user_profile", "age", "45岁", importance=6)
        rows = memory_service.load("sess-001")
        assert len(rows) == 1
        assert rows[0]["key"] == "age"
        assert rows[0]["value"] == "45岁"

    def test_upsert_overwrites_existing_key(self, memory_service):
        memory_service.upsert("sess-001", "user_profile", "age", "45岁", importance=6)
        memory_service.upsert("sess-001", "user_profile", "age", "46岁", importance=6)
        rows = memory_service.load("sess-001")
        assert len(rows) == 1
        assert rows[0]["value"] == "46岁"

    def test_upsert_skips_empty_and_placeholder_values(self, memory_service):
        for val in ("", "无", "不确定", "未知", "N/A"):
            memory_service.upsert("sess-002", "user_profile", "allergies", val)
        rows = memory_service.load("sess-002")
        assert len(rows) == 0

    def test_load_filters_by_min_importance(self, memory_service):
        memory_service.upsert("sess-003", "user_profile", "age", "30岁", importance=3)
        memory_service.upsert("sess-003", "user_profile", "allergies", "花粉过敏", importance=9)
        rows = memory_service.load("sess-003", min_importance=4)
        keys = [r["key"] for r in rows]
        assert "allergies" in keys
        assert "age" not in keys

    def test_load_filters_by_memory_type(self, memory_service):
        memory_service.upsert("sess-004", "user_profile", "age", "50岁", importance=6)
        memory_service.upsert("sess-004", "medical_fact", "chief_complaint", "头痛", importance=8)
        profile = memory_service.load("sess-004", memory_types=["user_profile"])
        assert all(r["memory_type"] == "user_profile" for r in profile)

    def test_load_profile_returns_dict(self, memory_service):
        memory_service.upsert("sess-005", "user_profile", "age", "35岁", importance=6)
        memory_service.upsert("sess-005", "user_profile", "gender", "女", importance=5)
        profile = memory_service.load_profile("sess-005")
        assert profile["age"] == "35岁"
        assert profile["gender"] == "女"

    def test_delete_all_removes_all_records(self, memory_service):
        memory_service.upsert("sess-006", "user_profile", "age", "60岁", importance=6)
        memory_service.upsert("sess-006", "medical_fact", "symptoms", "咳嗽", importance=8)
        memory_service.delete_all("sess-006")
        assert memory_service.load("sess-006") == []


# ══════════════════════════════════════════════════════════════════════════════
# LongTermMemoryService — format_for_prompt
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatForPrompt:

    def test_formats_user_profile_section(self, memory_service):
        memory_service.upsert("sess-fmt", "user_profile", "age", "45岁", importance=6)
        memory_service.upsert("sess-fmt", "user_profile", "allergies", "青霉素过敏", importance=9)
        text = memory_service.format_for_prompt("sess-fmt")
        assert "【患者档案】" in text
        assert "45岁" in text
        assert "青霉素过敏" in text

    def test_formats_medical_fact_section(self, memory_service):
        memory_service.upsert("sess-fmt2", "medical_fact", "chief_complaint", "反复头痛3天", importance=8)
        memory_service.upsert("sess-fmt2", "medical_fact", "symptoms", "头痛、颈部僵硬", importance=8)
        text = memory_service.format_for_prompt("sess-fmt2")
        assert "【近期就诊记录】" in text
        assert "反复头痛3天" in text

    def test_returns_empty_string_when_no_records(self, memory_service):
        text = memory_service.format_for_prompt("sess-nonexistent")
        assert text == ""


# ══════════════════════════════════════════════════════════════════════════════
# LongTermMemoryService — extract_and_save（LLM mock）
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAndSave:

    def test_extracts_and_saves_structured_fields(self, memory_service):
        extracted_json = json.dumps({
            "age": "38岁",
            "gender": "男",
            "allergies": "",
            "conditions": "高血压",
            "chief_complaint": "头痛3天",
            "symptoms": "头痛、颈部不适",
            "symptom_duration": "3天",
            "diagnosis": "",
            "advice": "",
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = extracted_json

        result = memory_service.extract_and_save(
            session_id="sess-extract",
            question="我头痛3天了，有高血压史",
            answer="高血压患者头痛需警惕血压波动，建议监测血压。",
            turn_index=1,
            llm=mock_llm,
        )

        assert "age" in result
        rows = memory_service.load("sess-extract")
        keys = {r["key"] for r in rows}
        assert "age" in keys
        assert "conditions" in keys
        # 空值字段不应写入
        assert "allergies" not in keys

    def test_returns_empty_dict_when_llm_unavailable(self, memory_service):
        with patch("app.memory.long_term.LongTermMemoryService._extract_with_llm", return_value={}):
            result = memory_service.extract_and_save(
                session_id="sess-no-llm",
                question="头痛",
                answer="请就医",
                llm=None,
            )
        assert result == {}

    def test_handles_malformed_llm_json_gracefully(self, memory_service):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "这不是JSON"
        result = memory_service.extract_and_save(
            session_id="sess-bad-json",
            question="test",
            answer="test",
            llm=mock_llm,
        )
        assert result == {}


# ══════════════════════════════════════════════════════════════════════════════
# 短期记忆：compress_history / build_context_window
# ══════════════════════════════════════════════════════════════════════════════

class TestShortTermMemory:

    def _make_history(self, n_turns: int):
        """生成 n_turns 轮对话（每轮 user + assistant 各一条）。"""
        history = []
        for i in range(n_turns):
            history.append({"role": "user", "content": f"用户问题{i}" * 5})
            history.append({"role": "assistant", "content": f"助手回答{i}" * 5})
        return history

    def test_compress_history_no_op_under_threshold(self):
        """历史不超过 20 条时，不压缩，原样返回。"""
        history = self._make_history(9)  # 18 条，未触发
        result = compress_history(history)
        assert result == history

    def test_compress_history_triggers_above_threshold(self):
        """历史超过 20 条时，应触发压缩，结果条数 ≤ 21（1摘要 + 20条）。"""
        history = self._make_history(12)  # 24 条，触发压缩
        with patch("app.memory.short_term._summarize_with_llm", return_value="早期对话摘要"):
            result = compress_history(history)
        assert len(result) <= 21
        # 第一条应是摘要 system 消息
        assert result[0]["role"] == "system"
        assert "摘要" in result[0]["content"]

    def test_compress_history_hard_truncates_without_llm(self):
        """LLM 不可用时，直接截断保留最近 20 条，不抛异常。"""
        history = self._make_history(12)
        with patch("app.memory.short_term._summarize_with_llm", return_value=None):
            result = compress_history(history)
        assert len(result) == 20

    def test_build_context_window_respects_token_budget(self):
        """build_context_window 应在 token 预算内返回历史。"""
        history = self._make_history(20)  # 大量历史
        window = build_context_window(history, max_tokens=500)
        # 结果应少于原始历史
        assert len(window) < len(history)

    def test_build_context_window_always_includes_system_summary(self):
        """system 摘要消息应始终出现在 context_window 中（token 超出也不丢弃）。"""
        history = [
            {"role": "system", "content": "【历史对话摘要】早期对话记录"},
        ] + self._make_history(5)
        window = build_context_window(history, max_tokens=100)
        system_msgs = [t for t in window if t.get("role") == "system"]
        assert len(system_msgs) >= 1
