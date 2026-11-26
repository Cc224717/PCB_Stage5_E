from __future__ import annotations

import sys
import uuid

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"数据库不可用: {db.status().get('init_error')}")
        return

    sid = f"casc_{uuid.uuid4().hex[:10]}"
    db.upsert_session_open(session_id=sid, operator="cascade_tester")
    db.insert_event(
        session_id=sid,
        operator="cascade_tester",
        event_type="TEST_EVENT",
        message="cascade",
    )
    job_id = db.insert_detect_result(
        session_id=sid,
        operator="cascade_tester",
        mode="single",
        file_name="c.jpg",
        status="ok",
        detections=[
            {
                "class_id": 1,
                "class_name": "mouse_bite",
                "confidence": 0.5,
                "xyxy": [1, 2, 3, 4],
            }
        ],
    )
    c.check(isinstance(job_id, int) and job_id > 0, f"job 写入失败: {job_id}")

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE session_id=%s", (sid,))
            cur.execute("SELECT COUNT(*) AS cnt FROM audit_events WHERE session_id=%s", (sid,))
            ev_cnt = int((cur.fetchone() or {}).get("cnt") or 0)
            cur.execute("SELECT COUNT(*) AS cnt FROM detect_jobs WHERE session_id=%s", (sid,))
            job_cnt = int((cur.fetchone() or {}).get("cnt") or 0)
            if job_id:
                cur.execute("SELECT COUNT(*) AS cnt FROM detections WHERE job_id=%s", (job_id,))
                det_cnt = int((cur.fetchone() or {}).get("cnt") or 0)
            else:
                det_cnt = -1

    c.check(ev_cnt == 0, f"删除 session 后 audit_events 未级联清理: {ev_cnt}")
    c.check(job_cnt == 0, f"删除 session 后 detect_jobs 未级联清理: {job_cnt}")
    c.check(det_cnt == 0, f"删除 session 后 detections 未级联清理: {det_cnt}")
