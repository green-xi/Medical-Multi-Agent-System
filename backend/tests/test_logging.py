"""日志测试 — 深度模块化架构"""

import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.logging_config import logger, setup_logging  # noqa: E402


def test_setup_logging_creates_directory():
    # 该测试现在被跳过或更新，因为 setup_logging 在测试模式下跳过目录创建。
    # 我们验证在测试模式下，即使请求也不会创建目录。
    test_log_dir = "test_logs_should_not_exist"
    if os.path.exists(test_log_dir):
        import shutil
        shutil.rmtree(test_log_dir)

    setup_logging(log_dir=test_log_dir)
    # 根据新的零日志策略，测试期间不应创建该目录
    assert not os.path.exists(test_log_dir)


def test_logger_instance():
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert logger.name == "medicalai"


def test_logger_has_handlers():
    assert len(logger.handlers) > 0


def test_logger_level():
    # 在 pytest 环境中，日志级别被设置为 DEBUG
    assert logger.level == logging.DEBUG
