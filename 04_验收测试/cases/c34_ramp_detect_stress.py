from __future__ import annotations

import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass

from _harness import Checker
from _paths import DEMO_DIR, ENGINE_ROOT, ensure_out


@dataclass
class LevelResult:
    device: str
    workers: int
    jobs: int
    ok: int
    fail: int
    wall_ms: float
    avg_ms: float
    p95_ms: float
    qps: float
    error_samples: list[str]
    stopped_reason: str | None = None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, max(0, int(round((p / 100.0) * (len(arr) - 1)))))
    return float(arr[idx])


def _pick_image():
    sample = ENGINE_ROOT / "samples" / "images" / "mouse_bite_07_mouse_bite_08.jpg"
    if sample.is_file():
        return sample
    demos = sorted(DEMO_DIR.glob("*.jpg"))
    return demos[0] if demos else sample


def _mem_mb() -> float | None:
    try:
        import os

        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def _run_level(engine, image, workers: int, jobs: int, device: str) -> LevelResult:
    latencies: list[float] = []
    errors: list[str] = []
    ok = 0

    def _one(job_id: int) -> tuple[bool, float, str]:
        try:
            result = engine.detect_one(image_path=image, save_path=None)
            if result.get("status") != "ok":
                return False, 0.0, f"status={result.get('status')}"
            return True, float(result.get("total_ms") or 0.0), ""
        except Exception as exc:
            return False, 0.0, f"{exc.__class__.__name__}: {exc}"

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, i) for i in range(jobs)]
        for fut in as_completed(futs):
            success, ms, err = fut.result()
            if success:
                ok += 1
                latencies.append(ms)
            else:
                errors.append(err)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    fail = jobs - ok
    avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
    p95_ms = _percentile(latencies, 95)
    qps = (ok / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0
    return LevelResult(
        device=device,
        workers=workers,
        jobs=jobs,
        ok=ok,
        fail=fail,
        wall_ms=round(wall_ms, 2),
        avg_ms=round(avg_ms, 2),
        p95_ms=round(p95_ms, 2),
        qps=round(qps, 4),
        error_samples=errors[:5],
    )


def _ramp_one_device(c: Checker, device: str, worker_steps: list[int]) -> dict:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from inference_engine import InferenceEngine

    image = _pick_image()
    c.check(image.is_file(), f"{device} 压测缺图片: {image}")
    if not image.is_file():
        return {"device": device, "skipped": True, "reason": "no image"}

    try:
        engine = InferenceEngine(device=device, imgsz=1024, conf=0.25, iou=0.7)
        engine.warmup(image_path=image, runs=1)
    except Exception as exc:
        c.fail(f"{device} 引擎初始化失败: {exc}")
        return {"device": device, "skipped": True, "reason": str(exc)}

    levels: list[LevelResult] = []
    peak_qps = 0.0
    peak_workers = 1
    # 预热后再取低并发参考，避免冷启动污染基线
    _ = _run_level(engine, image, workers=1, jobs=1, device=device)
    ref = _run_level(engine, image, workers=2, jobs=4, device=device)
    levels.append(ref)
    peak_qps = ref.qps
    peak_workers = 2
    limit = {
        "max_stable_workers": 2,
        "reason": "仅完成参考档",
        "baseline_p95_ms": ref.p95_ms,
        "peak_qps": ref.qps,
        "sweet_spot_workers": 2,
    }

    for workers in worker_steps:
        if workers <= 2:
            continue
        jobs = max(workers + 2, 3)
        try:
            level = _run_level(engine, image, workers=workers, jobs=jobs, device=device)
        except Exception as exc:
            level = LevelResult(
                device=device,
                workers=workers,
                jobs=jobs,
                ok=0,
                fail=jobs,
                wall_ms=0,
                avg_ms=0,
                p95_ms=0,
                qps=0,
                error_samples=[f"{exc.__class__.__name__}: {exc}"],
                stopped_reason="level_exception",
            )
            levels.append(level)
            prev = [x for x in levels if x.workers < workers and x.fail == 0]
            limit = {
                "max_stable_workers": prev[-1].workers if prev else 1,
                "sweet_spot_workers": peak_workers,
                "reason": f"在 workers={workers} 抛异常",
                "fail_level": asdict(level),
                "peak_qps": peak_qps,
                "mem_mb": _mem_mb(),
            }
            if prev:
                limit["last_stable"] = asdict(prev[-1])
            break

        levels.append(level)
        if level.qps > peak_qps:
            peak_qps = level.qps
            peak_workers = workers

        err_rate = level.fail / max(level.jobs, 1)
        reason = None
        if err_rate > 0.05:
            reason = f"错误率过高 {err_rate:.1%} @ workers={workers}"
        elif ref.p95_ms > 0 and level.p95_ms > ref.p95_ms * 4:
            reason = (
                f"P95 相对低并发参考膨胀 "
                f"{level.p95_ms:.0f}ms / {ref.p95_ms:.0f}ms (>4x) @ workers={workers}"
            )
        elif peak_qps > 0 and level.qps < peak_qps * 0.85 and level.p95_ms > ref.p95_ms * 2.5:
            reason = (
                f"吞吐相对峰值回落且延迟升高：qps={level.qps:.3f} peak={peak_qps:.3f} "
                f"p95={level.p95_ms:.0f}ms @ workers={workers}"
            )
        elif level.qps > 0 and level.workers >= 8 and level.qps < ref.qps * 1.15 and level.p95_ms > ref.p95_ms * 3:
            # 并发加很多但吞吐几乎不涨、延迟显著变差 => CPU/锁饱和
            reason = (
                f"并发增益耗尽（近似饱和）：workers={workers} qps={level.qps:.3f} "
                f"仅比参考 {ref.qps:.3f} 高有限，p95={level.p95_ms:.0f}ms"
            )

        if reason:
            level.stopped_reason = reason
            prev = [x for x in levels if x.workers < workers and x.fail == 0]
            limit = {
                "max_stable_workers": prev[-1].workers if prev else 1,
                "sweet_spot_workers": peak_workers,
                "reason": reason,
                "fail_level": asdict(level),
                "peak_qps": peak_qps,
                "mem_mb": _mem_mb(),
            }
            if prev:
                limit["last_stable"] = asdict(prev[-1])
            break
        else:
            limit = {
                "max_stable_workers": workers,
                "sweet_spot_workers": peak_workers,
                "reason": "本级稳定",
                "peak_qps": peak_qps,
                "last_stable": asdict(level),
                "mem_mb": _mem_mb(),
            }

    return {
        "device": device,
        "image": str(image),
        "worker_steps": worker_steps,
        "levels": [asdict(x) for x in levels],
        "limit": limit,
        "mem_mb_end": _mem_mb(),
    }


def run(c: Checker) -> None:
    import torch

    has_cuda = bool(torch.cuda.is_available())
    # CPU：按本机线程数逐步加压；GPU：可更高并发探吞吐天花板
    # 阶梯增压：先稀后密，便于定位拐点
    cpu_steps = [1, 2, 4, 6, 8, 12, 16]
    gpu_steps = [1, 2, 4, 8, 12, 16, 24, 32]

    reports = {
        "torch_threads": torch.get_num_threads(),
        "cuda": has_cuda,
        "cuda_name": torch.cuda.get_device_name(0) if has_cuda else None,
        "devices": {},
    }

    cpu_report = _ramp_one_device(c, "cpu", cpu_steps)
    reports["devices"]["cpu"] = cpu_report

    if has_cuda:
        gpu_report = _ramp_one_device(c, "0", gpu_steps)
        reports["devices"]["gpu"] = gpu_report
    else:
        reports["devices"]["gpu"] = {
            "skipped": True,
            "reason": "当前环境无可用 CUDA GPU，未做 GPU 增压",
        }
        c.fail("当前无 CUDA GPU，无法验证 GPU 并发极限（仅完成 CPU 阶梯增压）")

    out = ensure_out() / "c34_ramp_detect_stress.json"
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    # 极限结论写入 problems，方便 run_all 只输出这一段
    cpu_limit = (cpu_report or {}).get("limit") or {}
    if cpu_limit.get("reason") and cpu_limit.get("reason") != "本级稳定":
        c.fail(
            f"CPU 并发极限: 稳定 workers<={cpu_limit.get('max_stable_workers')}；"
            f"触发={cpu_limit.get('reason')}；peak_qps={cpu_limit.get('peak_qps')}"
        )
    else:
        c.fail(
            f"CPU 在台阶 {cpu_steps} 内未压垮，当前稳定上限 workers="
            f"{cpu_limit.get('max_stable_workers')}，peak_qps={cpu_limit.get('peak_qps')}"
        )

    if has_cuda:
        gpu_limit = ((reports["devices"].get("gpu") or {}).get("limit") or {})
        if gpu_limit.get("reason") and gpu_limit.get("reason") != "本级稳定":
            c.fail(
                f"GPU 并发极限: 稳定 workers<={gpu_limit.get('max_stable_workers')}；"
                f"触发={gpu_limit.get('reason')}；peak_qps={gpu_limit.get('peak_qps')}"
            )
        else:
            c.fail(
                f"GPU 在台阶 {gpu_steps} 内未压垮，当前稳定上限 workers="
                f"{gpu_limit.get('max_stable_workers')}，peak_qps={gpu_limit.get('peak_qps')}"
            )
