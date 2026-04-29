"""轻量级数据库迁移（无需 Alembic），v1→v2：user_memory 添加 UNIQUE 约束。"""

import sqlite3
import os

from app.core.config import CHAT_DB_PATH
from app.core.logging_config import logger


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CHAT_DB_PATH), exist_ok=True)
    return sqlite3.connect(CHAT_DB_PATH)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def _unique_constraint_exists(conn: sqlite3.Connection) -> bool:
    """检查 user_memory 表是否已有目标 UNIQUE 约束。"""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='user_memory' AND name=?",
        ("uq_user_memory_session_type_key",),
    )
    return cur.fetchone() is not None


def migrate_v2_add_unique_constraint() -> None:
    """
    为 user_memory 添加 UNIQUE 约束。

    SQLite 不支持 ALTER TABLE ADD CONSTRAINT，需要重建表：
      1. 重命名旧表
      2. 建新表（含 UNIQUE 约束）
      3. 复制数据（去重，保留最新一条）
      4. 删旧表
    """
    conn = _get_conn()
    try:
        if not _table_exists(conn, "user_memory"):
            logger.info("migrate_v2: user_memory 表不存在，跳过迁移")
            return

        if _unique_constraint_exists(conn):
            logger.info("migrate_v2: UNIQUE 约束已存在，跳过迁移")
            return

        logger.info("migrate_v2: 开始为 user_memory 添加 UNIQUE 约束…")

        conn.execute("BEGIN TRANSACTION")

        # 1. 重命名旧表
        conn.execute("ALTER TABLE user_memory RENAME TO user_memory_old")

        # 2. 建新表（与 ORM 定义一致，含 UNIQUE 约束）
        conn.execute("""
            CREATE TABLE user_memory (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  VARCHAR(255) NOT NULL,
                memory_type VARCHAR(50)  NOT NULL DEFAULT 'medical_fact',
                key         VARCHAR(255) NOT NULL,
                value       TEXT         NOT NULL,
                importance  INTEGER      NOT NULL DEFAULT 5,
                source_turn INTEGER,
                created_at  DATETIME     NOT NULL,
                updated_at  DATETIME     NOT NULL,
                CONSTRAINT uq_user_memory_session_type_key
                    UNIQUE (session_id, memory_type, key)
            )
        """)

        # 3. 建普通索引（加速 session_id 查询）
        # 使用 IF NOT EXISTS 防止重复执行时因索引已存在而崩溃
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_user_memory_session_id ON user_memory (session_id)"
        )

        # 4. 复制数据（去重：相同 session_id+memory_type+key 保留 updated_at 最新的）
        conn.execute("""
            INSERT INTO user_memory
                (id, session_id, memory_type, key, value, importance, source_turn, created_at, updated_at)
            SELECT id, session_id, memory_type, key, value, importance, source_turn, created_at, updated_at
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY session_id, memory_type, key
                           ORDER BY updated_at DESC
                       ) AS rn
                FROM user_memory_old
            )
            WHERE rn = 1
        """)

        # 5. 删旧表
        conn.execute("DROP TABLE user_memory_old")

        conn.execute("COMMIT")
        logger.info("migrate_v2: 迁移完成，UNIQUE 约束已添加")

    except Exception as exc:
        conn.execute("ROLLBACK")
        # 如果新表已建但旧表还在，回滚重命名
        try:
            if _table_exists(conn, "user_memory_old") and not _table_exists(conn, "user_memory"):
                conn.execute("ALTER TABLE user_memory_old RENAME TO user_memory")
                conn.commit()
        except Exception:
            pass
        logger.error("migrate_v2: 迁移失败，已回滚：%s", exc)
        raise
    finally:
        conn.close()


def run_all_migrations() -> None:
    """按顺序执行所有迁移，幂等（可重复执行）。"""
    try:
        migrate_v2_add_unique_constraint()
    except Exception as exc:
        # 迁移失败不阻断启动，但记录错误
        logger.error("数据库迁移异常（不影响主流程）：%s", exc)


if __name__ == "__main__":
    run_all_migrations()
    print("迁移完成。")
