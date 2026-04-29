"""
MedicalAI — core/logging_config.py
滚动文件 + 控制台日志配置（按本地时区午夜轮转）。
"""

import logging
import os
import sys
import time
from logging.handlers import TimedRotatingFileHandler

from app.core.config import LOG_DIR


def setup_logging(log_dir: str = LOG_DIR) -> logging.Logger:
    _logger = logging.getLogger("medicalai")

    if _logger.handlers:
        return _logger

    is_testing = "pytest" in sys.modules or os.getenv("TESTING") == "1"

    # 统一格式（本地时区）
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter.converter = time.localtime  # 强制本地时区，防止跨午夜写错日期文件

    if is_testing:
        _logger.setLevel(logging.DEBUG)
    else:
        _logger.setLevel(logging.INFO)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 活动日志文件：medicalai.log，午夜轮转为 medicalai_YYYYMMDD.log
        active_log_file = os.path.join(log_dir, "medicalai.log")
        file_handler = TimedRotatingFileHandler(
            active_log_file,
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.suffix = "%Y%m%d"
        file_handler.converter = time.localtime

        def _dated_log_namer(default_name: str) -> str:
            directory, filename = os.path.split(default_name)
            parts = filename.split(".")
            date_part = parts[-1] if parts else "unknown"
            return os.path.join(directory, f"medicalai_{date_part}.log")

        file_handler.namer = _dated_log_namer
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    return _logger


logger = setup_logging()
