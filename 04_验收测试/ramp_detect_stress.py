"""
项目内检测并发阶梯增压（独立运行，带进度输出）。

用法:
  01_推理工程_C交付\\.venv\\Scripts\\python 04_验收测试\\ramp_detect_stress.py
  01_推理工程_C交付\\.venv\\Scripts\\python 04_验收测试\\ramp_detect_stress.py --max-workers 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from _harness import Checker, run_case  # noqa: E402
from _paths import ensure_out  # noqa: E402
from cases import c34_ramp_detect_stress as case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="PCB 检测并发阶梯增压")
    parser.add_argument("--max-workers", type=int, default=16, help="CPU 最大并发台阶")
    args = parser.parse_args()

    # 动态覆盖台阶
    steps = []
    for n in [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
        if n <= args.max_workers:
            steps.append(n)
    if not steps or steps[-1] != args.max_workers:
        steps.append(args.max_workers)

    def patched(c: Checker) -> None:
        import torch

        # 临时替换台阶
        old_run = case._ramp_one_device

        def ramp_with_print(checker, device, worker_steps):
            print(f"[ramp] device={device} steps={worker_steps}", flush=True)
            report = old_run(checker, device, worker_steps)
            limit = (report or {}).get("limit") or {}
            print(
                f"[ramp] device={device} limit workers<={limit.get('max_stable_workers')} "
                f"reason={limit.get('reason')} peak_qps={limit.get('peak_qps')}",
                flush=True,
            )
            for level in (report or {}).get("levels") or []:
                print(
                    f"  workers={level['workers']:>3} jobs={level['jobs']:>3} "
                    f"ok={level['ok']}/{level['jobs']} qps={level['qps']:.3f} "
                    f"avg={level['avg_ms']:.0f}ms p95={level['p95_ms']:.0f}ms "
                    f"wall={level['wall_ms']:.0f}ms",
                    flush=True,
                )
            return report

        case._ramp_one_device = ramp_with_print  # type: ignore[assignment]
        try:
            # 复用 case.run，但先改其内部 steps：通过 monkeypatch 函数属性较难，
            # 直接调用精简流程。
            has_cuda = bool(torch.cuda.is_available())
            reports = {
                "torch_threads": torch.get_num_threads(),
                "cuda": has_cuda,
                "cuda_name": torch.cuda.get_device_name(0) if has_cuda else None,
                "devices": {},
            }
            reports["devices"]["cpu"] = ramp_with_print(c, "cpu", steps)
            if has_cuda:
                gpu_steps = [n for n in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64] if n <= max(args.max_workers, 32)]
                reports["devices"]["gpu"] = ramp_with_print(c, "0", gpu_steps)
            else:
                reports["devices"]["gpu"] = {"skipped": True, "reason": "无 CUDA GPU"}
                c.fail("当前无 CUDA GPU，无法验证 GPU 并发极限（仅完成 CPU 阶梯增压）")

            out = ensure_out() / "c34_ramp_detect_stress.json"
            out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ramp] report => {out}", flush=True)

            cpu_limit = reports["devices"]["cpu"].get("limit") or {}
            c.fail(
                f"CPU 结论: 稳定 workers<={cpu_limit.get('max_stable_workers')}；"
                f"{cpu_limit.get('reason')}；peak_qps={cpu_limit.get('peak_qps')}"
            )
            if has_cuda:
                gpu_limit = reports["devices"]["gpu"].get("limit") or {}
                c.fail(
                    f"GPU 结论: 稳定 workers<={gpu_limit.get('max_stable_workers')}；"
                    f"{gpu_limit.get('reason')}；peak_qps={gpu_limit.get('peak_qps')}"
                )
        finally:
            case._ramp_one_device = old_run  # type: ignore[assignment]

    result = run_case("ramp_detect_stress", patched)
    if result.problems:
        print("问题/结论：")
        for p in result.problems:
            print(f"- {p}")
        return 1
    print("无问题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
