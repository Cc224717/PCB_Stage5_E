"""
PCB UI 审计日志：记录会话开闭、登录、点击、处理文件与检测结果。

输出：
- MySQL pcb_defect（sessions / audit_events / detect_jobs / detections）
- logs/ui_audit.log      人类可读全文日志
- logs/ui_audit.jsonl    结构化事件（每行一条 JSON）
- logs/sessions/<id>.jsonl  单会话明细
- logs/sessions/active/  进行中会话心跳（用于判断关闭）
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from db import Database


LOGGER_NAME = "pcb_ui_audit"
_LOCK = threading.Lock()
_SETUP_DONE = False


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def setup_ui_audit_logger(log_dir: Path) -> logging.Logger:
    global _SETUP_DONE

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (log_dir / "sessions" / "active").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if _SETUP_DONE and logger.handlers:
        return logger

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = TimedRotatingFileHandler(
        filename=log_dir / "ui_audit.log",
        when="midnight",
        interval=1,
        backupCount=60,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    _SETUP_DONE = True
    return logger


class UIAuditLogger:
    """面向界面的全量审计记录器。"""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.logger = setup_ui_audit_logger(self.log_dir)
        self.jsonl_path = self.log_dir / "ui_audit.jsonl"
        self.sessions_dir = self.log_dir / "sessions"
        self.active_dir = self.sessions_dir / "active"
        self.db = Database()

        self.session_id = uuid.uuid4().hex[:12]
        self.streamlit_session_id = ""
        self.operator = "未登录"
        self.opened_at = _now_iso()
        self.closed = False
        self.click_count = 0
        self.file_count = 0
        self.session_file = self.sessions_dir / f"{self.session_id}.jsonl"
        self._pending_job_meta: dict[str, Any] = {}

        if not self.db.enabled:
            self.logger.warning(
                "MySQL 未启用或连接失败，仅写文件日志。详情=%s",
                self.db.status(),
            )
        else:
            self.logger.info("MySQL 已连接 database=%s", self.db.status().get("database"))

        self._close_stale_sessions()

    def bind_streamlit_session(self, streamlit_session_id: str) -> None:
        self.streamlit_session_id = streamlit_session_id or ""

    def _write_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with _LOCK:
            with path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _active_path(self) -> Path:
        return self.active_dir / f"{self.session_id}.json"

    def _touch_active(self, extra: dict[str, Any] | None = None) -> None:
        data = {
            "session_id": self.session_id,
            "streamlit_session_id": self.streamlit_session_id,
            "operator": self.operator,
            "opened_at": self.opened_at,
            "last_seen": _now_iso(),
            "click_count": self.click_count,
            "file_count": self.file_count,
            "closed": self.closed,
        }
        if extra:
            data.update(extra)
        self._active_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _close_stale_sessions(self, timeout_sec: int = 1800) -> None:
        """把长时间无心跳的 active 会话标记为超时关闭。"""
        now = time.time()
        for path in self.active_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("closed"):
                    path.unlink(missing_ok=True)
                    continue
                last_seen = data.get("last_seen") or data.get("opened_at")
                if not last_seen:
                    continue
                ts = datetime.fromisoformat(last_seen).timestamp()
                if now - ts < timeout_sec:
                    continue

                event = {
                    "time": _now_iso(),
                    "event": "SESSION_CLOSE",
                    "close_reason": "timeout_no_heartbeat",
                    "session_id": data.get("session_id"),
                    "streamlit_session_id": data.get("streamlit_session_id", ""),
                    "operator": data.get("operator", "未知"),
                    "opened_at": data.get("opened_at"),
                    "closed_at": _now_iso(),
                    "click_count": data.get("click_count", 0),
                    "file_count": data.get("file_count", 0),
                }
                self.logger.info(
                    "event=SESSION_CLOSE reason=timeout session=%s operator=%s "
                    "opened_at=%s click_count=%s file_count=%s",
                    event["session_id"],
                    event["operator"],
                    event["opened_at"],
                    event["click_count"],
                    event["file_count"],
                )
                self._write_jsonl(self.jsonl_path, event)
                sid = data.get("session_id")
                if sid:
                    self._write_jsonl(self.sessions_dir / f"{sid}.jsonl", event)
                path.unlink(missing_ok=True)
            except Exception as exc:
                self.logger.error("清理过期会话失败 file=%s error=%s", path.name, exc)

    def log_event(
        self,
        event: str,
        message: str = "",
        level: str = "INFO",
        **fields: Any,
    ) -> str:
        payload: dict[str, Any] = {
            "time": _now_iso(),
            "event": event,
            "session_id": self.session_id,
            "streamlit_session_id": self.streamlit_session_id,
            "operator": self.operator,
            "message": message,
        }
        payload.update(fields)

        line = (
            f"event={event} | session={self.session_id} | operator={self.operator}"
            f" | {message}"
        )
        if fields:
            detail = " | ".join(f"{k}={v}" for k, v in fields.items())
            line = f"{line} | {detail}"

        if level.upper() == "ERROR":
            self.logger.error(line)
        elif level.upper() == "WARNING":
            self.logger.warning(line)
        else:
            self.logger.info(line)

        self._write_jsonl(self.jsonl_path, payload)
        self._write_jsonl(self.session_file, payload)
        self._db_insert_event(event=event, message=message, level=level, fields=fields)
        if not self.closed:
            self._touch_active()
            try:
                self.db.touch_session_counts(
                    session_id=self.session_id,
                    click_count=self.click_count,
                    file_count=self.file_count,
                    operator=self.operator,
                )
            except Exception as exc:
                self.logger.error("MySQL 更新会话计数失败: %s", exc)
        return f"{_now()} | {level.upper()} | {line}"

    def _db_insert_event(
        self,
        *,
        event: str,
        message: str,
        level: str,
        fields: dict[str, Any],
    ) -> None:
        try:
            self.db.insert_event(
                session_id=self.session_id,
                operator=self.operator,
                event_type=event,
                message=message,
                action=fields.get("action"),
                level=level.upper(),
                payload=fields,
            )
        except Exception as exc:
            self.logger.error("MySQL 写入 audit_events 失败: %s", exc)

    def session_open(self, page: str = "ui_app", client_info: dict[str, Any] | None = None) -> str:
        self.opened_at = _now_iso()
        self.closed = False
        self._touch_active({"page": page})
        try:
            self.db.upsert_session_open(
                session_id=self.session_id,
                streamlit_session_id=self.streamlit_session_id,
                operator=self.operator,
                opened_at=self.opened_at,
            )
        except Exception as exc:
            self.logger.error("MySQL 写入 sessions(open) 失败: %s", exc)
        return self.log_event(
            "SESSION_OPEN",
            message="用户打开页面，会话开始",
            page=page,
            opened_at=self.opened_at,
            client_info=client_info or {},
            mysql=self.db.status(),
        )

    def login(self, operator: str, purpose: str | None = None) -> str:
        self.operator = (operator or "").strip() or "匿名用户"
        try:
            self.db.update_session_login(
                session_id=self.session_id,
                operator=self.operator,
                purpose=purpose,
            )
        except Exception as exc:
            self.logger.error("MySQL 更新登录信息失败: %s", exc)
        return self.log_event(
            "SESSION_LOGIN",
            message="用户登录系统",
            operator_name=self.operator,
            purpose=purpose,
            login_at=_now_iso(),
        )

    def logout(self, reason: str = "user_logout") -> str:
        return self.session_close(reason=reason)

    def session_close(self, reason: str = "user_close") -> str:
        if self.closed:
            return f"{_now()} | INFO | event=SESSION_CLOSE | already_closed"
        self.closed = True
        closed_at = _now_iso()
        try:
            self.db.close_session(
                session_id=self.session_id,
                close_reason=reason,
                closed_at=closed_at,
                click_count=self.click_count,
                file_count=self.file_count,
            )
        except Exception as exc:
            self.logger.error("MySQL 关闭会话失败: %s", exc)
        line = self.log_event(
            "SESSION_CLOSE",
            message="用户关闭/退出页面，会话结束",
            close_reason=reason,
            opened_at=self.opened_at,
            closed_at=closed_at,
            click_count=self.click_count,
            file_count=self.file_count,
        )
        active = self._active_path()
        if active.exists():
            active.unlink(missing_ok=True)
        return line

    def click(self, action: str, **fields: Any) -> str:
        self.click_count += 1
        return self.log_event(
            "UI_CLICK",
            message=f"用户点击: {action}",
            action=action,
            click_index=self.click_count,
            **fields,
        )

    def setting_change(self, name: str, old_value: Any, new_value: Any) -> str:
        return self.log_event(
            "UI_SETTING_CHANGE",
            message=f"设置变更: {name}",
            setting=name,
            old_value=old_value,
            new_value=new_value,
        )

    def file_process_start(
        self,
        *,
        mode: str,
        file_name: str,
        file_path: str,
        model_name: str,
        conf: float,
        iou: float,
        imgsz: int,
        source_type: str,
    ) -> str:
        self.file_count += 1
        self._pending_job_meta = {
            "mode": mode,
            "file_name": file_name,
            "file_path": file_path,
            "model_name": model_name,
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "source_type": source_type,
        }
        return self.log_event(
            "FILE_PROCESS_START",
            message=f"开始处理文件: {file_name}",
            mode=mode,
            file_name=file_name,
            file_path=file_path,
            model_name=model_name,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            source_type=source_type,
            file_index=self.file_count,
        )

    def file_process_result(
        self,
        *,
        mode: str,
        file_name: str,
        status: str,
        detections: list[dict[str, Any]] | None = None,
        predict_call_ms: float | None = None,
        total_ms: float | None = None,
        annotated_image: str | None = None,
        result_json: str | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        detections = detections or []
        class_counter: dict[str, int] = {}
        for item in detections:
            name = item.get("class_name", "unknown")
            class_counter[name] = class_counter.get(name, 0) + 1

        fields: dict[str, Any] = {
            "mode": mode,
            "file_name": file_name,
            "status": status,
            "detection_count": len(detections),
            "class_counter": class_counter,
            "detections": detections,
            "predict_call_ms": predict_call_ms,
            "total_ms": total_ms,
            "annotated_image": annotated_image,
            "result_json": result_json,
        }
        if error:
            fields["error"] = error
        if extra:
            fields.update(extra)

        meta = self._pending_job_meta if self._pending_job_meta.get("file_name") == file_name else {}
        try:
            job_id = self.db.insert_detect_result(
                session_id=self.session_id,
                operator=self.operator,
                mode=mode,
                file_name=file_name,
                status=status,
                file_path=meta.get("file_path") or fields.get("file_path"),
                source_type=meta.get("source_type") or fields.get("source_type"),
                model_name=meta.get("model_name") or fields.get("model_name"),
                conf=meta.get("conf"),
                iou=meta.get("iou"),
                imgsz=meta.get("imgsz"),
                detections=detections,
                class_counter=class_counter,
                predict_call_ms=predict_call_ms,
                total_ms=total_ms,
                annotated_image=annotated_image,
                result_json=result_json,
                output_dir=(extra or {}).get("output_dir"),
                error_message=error,
            )
            fields["mysql_job_id"] = job_id
        except Exception as exc:
            self.logger.error("MySQL 写入 detect_jobs/detections 失败: %s", exc)
            fields["mysql_error"] = str(exc)

        level = "ERROR" if status != "ok" else "INFO"
        return self.log_event(
            "FILE_PROCESS_RESULT",
            message=(
                f"文件处理完成: {file_name} status={status} "
                f"boxes={len(detections)} classes={class_counter}"
            ),
            level=level,
            **fields,
        )

    def batch_summary(
        self,
        *,
        success: int,
        failure: int,
        avg_ms: float,
        fps: float,
        output_dir: str,
        json_path: str,
        csv_path: str,
        files: list[str],
    ) -> str:
        return self.log_event(
            "BATCH_SUMMARY",
            message=(
                f"批量检测汇总 success={success} failure={failure} "
                f"avg_ms={avg_ms:.2f} fps={fps:.2f}"
            ),
            success=success,
            failure=failure,
            avg_ms=round(avg_ms, 2),
            fps=round(fps, 2),
            output_dir=output_dir,
            json_path=json_path,
            csv_path=csv_path,
            files=files,
            file_total=len(files),
        )

    def error(self, message: str, **fields: Any) -> str:
        return self.log_event(
            "ERROR",
            message=message,
            level="ERROR",
            **fields,
        )

    def heartbeat(self, page_state: dict[str, Any] | None = None) -> None:
        if self.closed:
            return
        self._touch_active({"page_state": page_state or {}})
