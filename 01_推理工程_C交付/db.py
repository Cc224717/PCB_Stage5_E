"""
MySQL 连接与写入封装（用户 / 会话 / 审计事件 / 检测任务 / 检测框）。
"""

from __future__ import annotations

import hashlib
import json
import secrets
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pymysql
import yaml
from pymysql.connections import Connection
from pymysql.cursors import DictCursor


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120000,
    ).hex()
    return digest, salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    digest, _ = hash_password(password, salt=salt)
    return secrets.compare_digest(digest, password_hash)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "db_config.yaml"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class Database:
    def __init__(self, config_path: Path | str | None = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG
        self.config = self._load_config()
        self.enabled = bool(self.config.get("enabled", True))
        self._init_error: str | None = None

        if self.enabled:
            try:
                with self.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
            except Exception as exc:
                self.enabled = False
                self._init_error = str(exc)

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.is_file():
            return {"enabled": False}
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        return data

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        conn = pymysql.connect(
            host=self.config.get("host", "127.0.0.1"),
            port=int(self.config.get("port", 3306)),
            user=self.config.get("user", "pcb_app"),
            password=str(self.config.get("password", "")),
            database=self.config.get("database", "pcb_defect"),
            charset=self.config.get("charset", "utf8mb4"),
            cursorclass=DictCursor,
            autocommit=True,
        )
        try:
            yield conn
        finally:
            conn.close()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "config_path": str(self.config_path),
            "database": self.config.get("database"),
            "host": self.config.get("host"),
            "init_error": self._init_error,
        }

    def upsert_session_open(
        self,
        *,
        session_id: str,
        streamlit_session_id: str = "",
        operator: str = "未登录",
        opened_at: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (
                        session_id, streamlit_session_id, operator, opened_at, status
                    ) VALUES (%s, %s, %s, %s, 'open')
                    ON DUPLICATE KEY UPDATE
                        streamlit_session_id = VALUES(streamlit_session_id),
                        operator = VALUES(operator),
                        status = 'open',
                        updated_at = CURRENT_TIMESTAMP(3)
                    """,
                    (
                        session_id,
                        streamlit_session_id or None,
                        operator,
                        _parse_dt(opened_at) or datetime.now(),
                    ),
                )

    def update_session_login(
        self,
        *,
        session_id: str,
        operator: str,
        purpose: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sessions
                    SET operator = %s,
                        purpose = COALESCE(%s, purpose),
                        updated_at = CURRENT_TIMESTAMP(3)
                    WHERE session_id = %s
                    """,
                    (operator, purpose, session_id),
                )

    def close_session(
        self,
        *,
        session_id: str,
        close_reason: str,
        closed_at: str | None = None,
        click_count: int = 0,
        file_count: int = 0,
    ) -> None:
        if not self.enabled:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sessions
                    SET status = 'closed',
                        close_reason = %s,
                        closed_at = %s,
                        click_count = %s,
                        file_count = %s,
                        updated_at = CURRENT_TIMESTAMP(3)
                    WHERE session_id = %s
                    """,
                    (
                        close_reason,
                        _parse_dt(closed_at) or datetime.now(),
                        click_count,
                        file_count,
                        session_id,
                    ),
                )

    def touch_session_counts(
        self,
        *,
        session_id: str,
        click_count: int,
        file_count: int,
        operator: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                if operator:
                    cur.execute(
                        """
                        UPDATE sessions
                        SET click_count = %s,
                            file_count = %s,
                            operator = %s,
                            updated_at = CURRENT_TIMESTAMP(3)
                        WHERE session_id = %s
                        """,
                        (click_count, file_count, operator, session_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE sessions
                        SET click_count = %s,
                            file_count = %s,
                            updated_at = CURRENT_TIMESTAMP(3)
                        WHERE session_id = %s
                        """,
                        (click_count, file_count, session_id),
                    )

    def insert_event(
        self,
        *,
        session_id: str,
        operator: str,
        event_type: str,
        message: str = "",
        action: str | None = None,
        level: str = "INFO",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_events (
                        session_id, operator, event_type, message, action, level, payload_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, CAST(%s AS JSON))
                    """,
                    (
                        session_id,
                        operator,
                        event_type,
                        message,
                        action,
                        level,
                        json.dumps(payload or {}, ensure_ascii=False),
                    ),
                )

    def insert_detect_result(
        self,
        *,
        session_id: str,
        operator: str,
        mode: str,
        file_name: str,
        status: str,
        file_path: str | None = None,
        source_type: str | None = None,
        model_name: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
        detections: list[dict[str, Any]] | None = None,
        class_counter: dict[str, int] | None = None,
        predict_call_ms: float | None = None,
        total_ms: float | None = None,
        annotated_image: str | None = None,
        result_json: str | None = None,
        output_dir: str | None = None,
        error_message: str | None = None,
    ) -> int | None:
        if not self.enabled:
            return None

        detections = detections or []
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO detect_jobs (
                        session_id, operator, mode, file_name, file_path, source_type,
                        model_name, conf, iou, imgsz, status, detection_count,
                        class_counter_json, predict_call_ms, total_ms, annotated_image,
                        result_json, output_dir, error_message, finished_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        CAST(%s AS JSON), %s, %s, %s,
                        %s, %s, %s, CURRENT_TIMESTAMP(3)
                    )
                    """,
                    (
                        session_id,
                        operator,
                        mode,
                        file_name,
                        file_path,
                        source_type,
                        model_name,
                        conf,
                        iou,
                        imgsz,
                        status,
                        len(detections),
                        json.dumps(class_counter or {}, ensure_ascii=False),
                        predict_call_ms,
                        total_ms,
                        annotated_image,
                        result_json,
                        output_dir,
                        error_message,
                    ),
                )
                job_id = int(cur.lastrowid)

                for item in detections:
                    xyxy = item.get("xyxy") or [0, 0, 0, 0]
                    cur.execute(
                        """
                        INSERT INTO detections (
                            job_id, session_id, file_name, class_id, class_name,
                            confidence, x1, y1, x2, y2
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            job_id,
                            session_id,
                            file_name,
                            item.get("class_id"),
                            item.get("class_name", "unknown"),
                            float(item.get("confidence", 0.0)),
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ),
                    )
                return job_id

    def ensure_users_table(self) -> None:
        if not self.enabled:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        username VARCHAR(64) NOT NULL UNIQUE,
                        password_hash VARCHAR(255) NOT NULL,
                        salt VARCHAR(64) NOT NULL,
                        display_name VARCHAR(128) NULL,
                        role VARCHAR(32) NOT NULL DEFAULT 'user',
                        created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
                        last_login_at DATETIME(3) NULL,
                        is_active TINYINT(1) NOT NULL DEFAULT 1,
                        INDEX idx_users_username (username)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cur.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'users'
                      AND COLUMN_NAME = 'role'
                    """
                )
                row = cur.fetchone() or {}
                if int(row.get("cnt") or 0) == 0:
                    cur.execute(
                        """
                        ALTER TABLE users
                        ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'user'
                        AFTER display_name
                        """
                    )
        self.ensure_default_admin()

    def ensure_default_admin(self) -> None:
        """若无管理员，则创建默认账号 admin / Admin123!"""
        if not self.enabled:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin'"
                )
                row = cur.fetchone() or {}
                if int(row.get("cnt") or 0) > 0:
                    return
        # 没有 admin 角色用户时再创建
        if self.get_user("admin") is None:
            self.create_user(
                username="admin",
                password="Admin123!",
                display_name="系统管理员",
                role="admin",
            )
        else:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET role = 'admin' WHERE username = 'admin'"
                    )

    def get_user(self, username: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, password_hash, salt, display_name, role,
                           created_at, last_login_at, is_active
                    FROM users
                    WHERE username = %s
                    LIMIT 1
                    """,
                    (username,),
                )
                return cur.fetchone()

    def create_user(
        self,
        *,
        username: str,
        password: str,
        display_name: str | None = None,
        role: str = "user",
    ) -> tuple[bool, str]:
        if not self.enabled:
            return False, "数据库未连接"
        username = username.strip()
        role = (role or "user").strip().lower()
        if role not in {"user", "admin"}:
            role = "user"
        if not username or not password:
            return False, "用户名和密码不能为空"
        if len(username) < 3:
            return False, "用户名至少 3 位"
        if len(password) < 6:
            return False, "密码至少 6 位"
        if self.get_user(username) is not None:
            return False, "用户名已存在"

        password_hash, salt = hash_password(password)
        try:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO users (
                            username, password_hash, salt, display_name, role
                        ) VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            username,
                            password_hash,
                            salt,
                            display_name or username,
                            role,
                        ),
                    )
            return True, "注册成功"
        except Exception as exc:
            return False, f"注册失败：{exc}"

    def change_password(
        self,
        *,
        username: str,
        old_password: str,
        new_password: str,
    ) -> tuple[bool, str]:
        if not self.enabled:
            return False, "数据库未连接"
        username = username.strip()
        if len(new_password) < 6:
            return False, "新密码至少 6 位"
        user = self.get_user(username)
        if user is None:
            return False, "用户不存在"
        if not verify_password(old_password, user["password_hash"], user["salt"]):
            return False, "原密码错误"
        password_hash, salt = hash_password(new_password)
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET password_hash = %s, salt = %s
                    WHERE id = %s
                    """,
                    (password_hash, salt, user["id"]),
                )
        self.clear_login_failures(username)
        return True, "密码修改成功"

    def list_users(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, display_name, role, is_active,
                           created_at, last_login_at
                    FROM users
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return list(cur.fetchall() or [])

    def unlock_user_login(self, username: str) -> tuple[bool, str]:
        if not self.enabled:
            return False, "数据库未连接"
        username = username.strip()
        if not username:
            return False, "用户名为空"
        self.clear_login_failures(username)
        return True, f"已解除 {username} 的登录锁定"

    def recent_events_for_operator(
        self,
        operator: str,
        *,
        limit: int = 20,
        also_match: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        names: list[str] = []
        for name in [operator, *(also_match or [])]:
            value = (name or "").strip()
            if value and value not in names:
                names.append(value)
        if not names:
            return []
        placeholders = ", ".join(["%s"] * len(names))
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT created_at, event_type, action, message, level
                    FROM audit_events
                    WHERE operator IN ({placeholders})
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (*names, limit),
                )
                return list(cur.fetchall() or [])

    def ensure_login_guards_table(self) -> None:
        if not self.enabled:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS login_guards (
                        username VARCHAR(64) PRIMARY KEY,
                        fail_count INT NOT NULL DEFAULT 0,
                        lock_until DATETIME(3) NULL,
                        updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                            ON UPDATE CURRENT_TIMESTAMP(3)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )

    def get_login_guard(self, username: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        username = username.strip()
        if not username:
            return None
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT username, fail_count, lock_until, updated_at
                    FROM login_guards
                    WHERE username = %s
                    LIMIT 1
                    """,
                    (username,),
                )
                return cur.fetchone()

    def check_login_allowed(
        self,
        username: str,
        *,
        max_fails: int = 5,
    ) -> tuple[bool, str]:
        """返回 (是否允许尝试登录, 提示文案)。"""
        if not self.enabled:
            return False, "数据库未连接"
        self.ensure_login_guards_table()
        username = username.strip()
        if not username:
            return False, "请输入用户名和密码"

        guard = self.get_login_guard(username)
        if not guard:
            return True, ""

        lock_until = guard.get("lock_until")
        if lock_until is not None:
            now = datetime.now()
            if isinstance(lock_until, datetime) and lock_until > now:
                seconds = int((lock_until - now).total_seconds())
                minutes = max(1, (seconds + 59) // 60)
                return False, f"登录失败次数过多，请 {minutes} 分钟后再试"

        return True, ""

    def record_login_failure(
        self,
        username: str,
        *,
        max_fails: int = 5,
        lock_minutes: int = 10,
    ) -> tuple[int, bool]:
        """记录一次失败。返回 (当前失败次数, 是否已锁定)。"""
        if not self.enabled:
            return 0, False
        self.ensure_login_guards_table()
        username = username.strip()
        if not username:
            return 0, False

        from datetime import timedelta

        guard = self.get_login_guard(username)
        now = datetime.now()
        if guard and guard.get("lock_until") and guard["lock_until"] > now:
            return int(guard.get("fail_count") or 0), True

        fail_count = int((guard or {}).get("fail_count") or 0) + 1
        lock_until = (
            now + timedelta(minutes=lock_minutes)
            if fail_count >= max_fails
            else None
        )

        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO login_guards (username, fail_count, lock_until)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        fail_count = VALUES(fail_count),
                        lock_until = VALUES(lock_until)
                    """,
                    (username, fail_count, lock_until),
                )
        return fail_count, lock_until is not None

    def clear_login_failures(self, username: str) -> None:
        if not self.enabled:
            return
        self.ensure_login_guards_table()
        username = username.strip()
        if not username:
            return
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM login_guards WHERE username = %s",
                    (username,),
                )

    def authenticate(self, username: str, password: str) -> tuple[bool, str, dict[str, Any] | None]:
        if not self.enabled:
            return False, "数据库未连接", None
        username = username.strip()
        if not username or not password:
            return False, "请输入用户名和密码", None

        allowed, guard_msg = self.check_login_allowed(username)
        if not allowed:
            return False, guard_msg, None

        user = self.get_user(username)
        if user is None:
            fail_count, locked = self.record_login_failure(username)
            if locked:
                return False, "登录失败次数过多，请 10 分钟后再试", None
            left = max(0, 5 - fail_count)
            return False, f"用户名或密码错误（还可尝试 {left} 次）", None
        if not user.get("is_active", 1):
            return False, "账号已停用", None
        if not verify_password(password, user["password_hash"], user["salt"]):
            fail_count, locked = self.record_login_failure(username)
            if locked:
                return False, "登录失败次数过多，请 10 分钟后再试", None
            left = max(0, 5 - fail_count)
            return False, f"用户名或密码错误（还可尝试 {left} 次）", None

        self.clear_login_failures(username)
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET last_login_at = CURRENT_TIMESTAMP(3)
                    WHERE id = %s
                    """,
                    (user["id"],),
                )
        return True, "登录成功", user
