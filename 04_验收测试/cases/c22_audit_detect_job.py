from __future__ import annotations

import json
import sys

from _harness import Checker
from _paths import ENGINE_ROOT, ensure_out


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine
    from ui_audit import UIAuditLogger

    log_dir = ensure_out() / "logs_c22"
    log_dir.mkdir(parents=True, exist_ok=True)
    audit = UIAuditLogger(log_dir=log_dir)
    audit.bind_streamlit_session("acceptance_c22")
    audit.session_open(page="acceptance_c22")
    audit.login("acceptance_bot", purpose="c22")

    image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    save_path = ensure_out() / "c22_ann.jpg"
    engine = InferenceEngine(device="cpu", imgsz=1024, conf=0.25, iou=0.7)

    audit.file_process_start(
        mode="single",
        file_name=image.name,
        file_path=str(image),
        model_name="高精度 (high_accuracy)",
        conf=0.25,
        iou=0.7,
        imgsz=1024,
        source_type="sample",
    )
    result = engine.detect_one(image_path=image, save_path=save_path)
    json_path = ensure_out() / "c22_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    line = audit.file_process_result(
        mode="single",
        file_name=image.name,
        status=result.get("status", "ok"),
        detections=result.get("detections", []),
        predict_call_ms=result.get("predict_call_ms"),
        total_ms=result.get("total_ms"),
        annotated_image=str(save_path),
        result_json=str(json_path),
        extra={"output_dir": str(ensure_out())},
    )
    c.check(isinstance(line, str) and len(line) > 0, "file_process_result 返回异常")
    c.check(save_path.is_file(), "审计链路标注图未保存")
    c.check(json_path.is_file(), "审计链路 result.json 未保存")

    # 若 MySQL 可用，应能查到 job
    if audit.db.enabled:
        with audit.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, status, detection_count, file_name
                    FROM detect_jobs
                    WHERE session_id=%s AND file_name=%s
                    ORDER BY id DESC LIMIT 1
                    """,
                    (audit.session_id, image.name),
                )
                row = cur.fetchone()
        c.check(row is not None, "detect_jobs 未写入")
        if row:
            c.check(row.get("status") == "ok", f"detect_jobs.status 异常: {row}")
            c.check(int(row.get("detection_count") or 0) >= 0, "detection_count 非法")
            with audit.db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) AS cnt FROM detections WHERE job_id=%s",
                        (row["id"],),
                    )
                    cnt_row = cur.fetchone() or {}
            c.check(
                int(cnt_row.get("cnt") or 0) == int(row.get("detection_count") or 0),
                "detections 行数与 detection_count 不一致",
            )

    audit.session_close(reason="acceptance_c22")
