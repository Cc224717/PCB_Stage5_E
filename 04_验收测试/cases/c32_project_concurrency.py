from __future__ import annotations

import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _harness import Checker
from _paths import DEMO_DIR, ENGINE_ROOT, ensure_out


ROSTER = ensure_out() / "load_users_roster.json"
PASSWORD_DEFAULT = "LoadTest123!"


def _load_roster() -> tuple[list[str], str]:
    if ROSTER.is_file():
        data = json.loads(ROSTER.read_text(encoding="utf-8"))
        return list(data.get("users") or []), str(data.get("password") or PASSWORD_DEFAULT)
    users = [f"load_u{i:03d}" for i in range(1, 101)]
    return users, PASSWORD_DEFAULT


def run(c: Checker) -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database
    from inference_engine import InferenceEngine

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"数据库不可用: {db.status().get('init_error')}")
        return

    users, password = _load_roster()
    c.check(len(users) >= 100, f"压测用户不足 100: {len(users)}")
    if len(users) < 100:
        return

    # -------- 1) 100 用户并发登录 --------
    login_ok = 0
    login_fail = []

    def _login(username: str) -> tuple[str, bool, str]:
        try:
            ok, msg, _ = db.authenticate(username, password)
            return username, ok, msg
        except Exception as exc:
            return username, False, f"{exc.__class__.__name__}: {exc}"

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=32) as pool:
        futs = [pool.submit(_login, u) for u in users[:100]]
        for fut in as_completed(futs):
            username, ok, msg = fut.result()
            if ok:
                login_ok += 1
            else:
                login_fail.append({"user": username, "error": msg})
    login_ms = round((time.perf_counter() - t0) * 1000, 2)

    c.check(login_ok == 100, f"并发登录成功数 {login_ok}/100，失败样例={login_fail[:5]}")

    # -------- 2) 项目推理并发（共享一个 engine，多线程 detect） --------
    image = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    if not image.is_file():
        demos = sorted(DEMO_DIR.glob("*.jpg"))
        image = demos[0] if demos else image
    c.check(image.is_file(), f"缺压测图片: {image}")

    engine = InferenceEngine(device="cpu", imgsz=1024, conf=0.25, iou=0.7)
    engine.warmup(image_path=image, runs=1)

    detect_workers = 8
    detect_jobs = 16
    detect_ok = 0
    detect_fail = []
    detect_ms_list = []

    def _detect(job_id: int) -> tuple[int, bool, str, float]:
        try:
            result = engine.detect_one(image_path=image, save_path=None)
            ok = result.get("status") == "ok"
            return job_id, ok, "" if ok else str(result), float(result.get("total_ms") or 0)
        except Exception as exc:
            return job_id, False, f"{exc.__class__.__name__}: {exc}", 0.0

    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=detect_workers) as pool:
        futs = [pool.submit(_detect, i) for i in range(detect_jobs)]
        for fut in as_completed(futs):
            job_id, ok, err, ms = fut.result()
            if ok:
                detect_ok += 1
                detect_ms_list.append(ms)
            else:
                detect_fail.append({"job": job_id, "error": err})
    detect_wall_ms = round((time.perf_counter() - t1) * 1000, 2)

    c.check(
        detect_ok == detect_jobs,
        f"并发推理成功数 {detect_ok}/{detect_jobs}，失败={detect_fail[:3]}",
    )

    # -------- 3) 并发写审计/检测任务（模拟多用户落库） --------
    from ui_audit import UIAuditLogger

    log_dir = ensure_out() / "logs_c32_conc"
    log_dir.mkdir(parents=True, exist_ok=True)
    write_ok = 0
    write_fail = []

    def _write_job(idx: int) -> tuple[int, bool, str]:
        try:
            audit = UIAuditLogger(log_dir=log_dir / f"w{idx}")
            uname = users[idx % 100]
            audit.session_open(page="concurrency")
            audit.login(uname, purpose="c32")
            audit.file_process_start(
                mode="single",
                file_name=image.name,
                file_path=str(image),
                model_name="高精度 (high_accuracy)",
                conf=0.25,
                iou=0.7,
                imgsz=1024,
                source_type="concurrency",
            )
            audit.file_process_result(
                mode="single",
                file_name=image.name,
                status="ok",
                detections=[
                    {
                        "class_id": 1,
                        "class_name": "mouse_bite",
                        "confidence": 0.9,
                        "xyxy": [1, 2, 3, 4],
                    }
                ],
                predict_call_ms=10,
                total_ms=12,
                extra={"output_dir": str(ensure_out())},
            )
            audit.session_close(reason="concurrency_done")
            return idx, True, ""
        except Exception as exc:
            return idx, False, f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc(limit=3)}"

    t2 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = [pool.submit(_write_job, i) for i in range(40)]
        for fut in as_completed(futs):
            idx, ok, err = fut.result()
            if ok:
                write_ok += 1
            else:
                write_fail.append({"idx": idx, "error": err})
    write_ms = round((time.perf_counter() - t2) * 1000, 2)

    c.check(write_ok == 40, f"并发审计落库成功数 {write_ok}/40，失败样例={write_fail[:2]}")

    report = {
        "login": {
            "users": 100,
            "ok": login_ok,
            "fail": login_fail[:10],
            "wall_ms": login_ms,
            "qps": round(100 / (login_ms / 1000.0), 2) if login_ms else None,
        },
        "detect": {
            "workers": detect_workers,
            "jobs": detect_jobs,
            "ok": detect_ok,
            "fail": detect_fail[:5],
            "wall_ms": detect_wall_ms,
            "avg_infer_ms": round(sum(detect_ms_list) / len(detect_ms_list), 2) if detect_ms_list else None,
        },
        "audit_db_write": {
            "jobs": 40,
            "ok": write_ok,
            "fail_count": len(write_fail),
            "wall_ms": write_ms,
        },
    }
    (ensure_out() / "c32_project_concurrency.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
