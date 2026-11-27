"""
逐项验收测试入口。

用法（在交付包根目录或本目录均可）:
  01_推理工程_C交付\\.venv\\Scripts\\python 04_验收测试\\run_all.py
  01_推理工程_C交付\\.venv\\Scripts\\python 04_验收测试\\run_all.py --only c01,c05

约定：只打印失败项；全部通过时仅输出一行“无问题”。
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from _harness import CaseResult  # noqa: E402
from _paths import ENGINE_ROOT, ensure_out  # noqa: E402


def _quiet_console_logging() -> None:
    """静音控制台日志，保留文件 handler，最终只汇报问题。"""
    if getattr(logging.StreamHandler.emit, "_acceptance_muted", False):
        return

    original_emit = logging.StreamHandler.emit

    def muted_emit(self, record):  # type: ignore[no-untyped-def]
        # FileHandler / TimedRotatingFileHandler 是子类，继续写文件
        if type(self) is logging.StreamHandler:
            return
        return original_emit(self, record)

    muted_emit._acceptance_muted = True  # type: ignore[attr-defined]
    logging.StreamHandler.emit = muted_emit  # type: ignore[method-assign]


CASES = [
    ("c01", "资源完整性", "cases.c01_assets"),
    ("c02", "模块导入", "cases.c02_imports"),
    ("c03", "阈值配置", "cases.c03_threshold"),
    ("c04", "日志写入", "cases.c04_logging"),
    ("c05", "单图检测", "cases.c05_single_detect"),
    ("c06", "推理耗时", "cases.c06_timing"),
    ("c07", "模型切换", "cases.c07_model_switch"),
    ("c08", "阈值调节生效", "cases.c08_threshold_effect"),
    ("c09", "批量检测导出", "cases.c09_batch_detect"),
    ("c10", "数据库连通", "cases.c10_db"),
    ("c11", "审计日志", "cases.c11_ui_audit"),
    ("c12", "界面接线", "cases.c12_ui_wiring"),
    ("c13", "遗留冒烟脚本", "cases.c13_legacy_smoke"),
    ("c14", "批量CLI", "cases.c14_batch_cli"),
    ("c15", "演示样本精度", "cases.c15_demo_accuracy"),
    ("c16", "异常路径", "cases.c16_exception_paths"),
    ("c17", "数据库业务流", "cases.c17_db_auth_flow"),
    ("c18", "多后端格式", "cases.c18_backend_formats"),
    ("c19", "性能抽样", "cases.c19_performance"),
    ("c20", "环境兼容", "cases.c20_compat_env"),
    ("c21", "界面AppTest", "cases.c21_ui_apptest"),
    ("c22", "审计检测落库", "cases.c22_audit_detect_job"),
    ("c23", "类别一致性", "cases.c23_classes_and_names"),
    ("c24", "界面HTTP冒烟", "cases.c24_ui_http_smoke"),
    ("c25", "验收清单静态", "cases.c25_acceptance_checklist_static"),
    ("c26", "界面深测流程", "cases.c26_ui_deep_flow"),
    ("c27", "ONNX实测", "cases.c27_onnx_infer"),
    ("c28", "重复推理一致性", "cases.c28_repeatability"),
    ("c29", "界面注册流程", "cases.c29_ui_register"),
    ("c30", "PT与ONNX对比", "cases.c30_pt_onnx_compare"),
    ("c31", "创建100压测用户", "cases.c31_create_100_users"),
    ("c32", "项目业务并发", "cases.c32_project_concurrency"),
    ("c33", "wrk HTTP并发", "cases.c33_wrk_http"),
    ("c34", "检测并发阶梯增压", "cases.c34_ramp_detect_stress"),
    ("c35", "数据隔离与权限边界", "cases.c35_data_isolation"),
    ("c36", "登录Token安全", "cases.c36_auth_token"),
    ("c37", "停用账号与恢复", "cases.c37_inactive_user"),
    ("c38", "锁定解锁流程", "cases.c38_admin_unlock"),
    ("c39", "GPU/CPU设备切换", "cases.c39_device_switch"),
    ("c40", "界面辅助函数", "cases.c40_helpers_unit"),
    ("c41", "恶意输入与注入", "cases.c41_security_inputs"),
    ("c42", "跨产线模型", "cases.c42_cross_line_model"),
    ("c43", "数据库Schema完整性", "cases.c43_schema_integrity"),
    ("c44", "密钥与Git安全", "cases.c44_auth_secret_safety"),
    ("c45", "角色越权防护", "cases.c45_role_escalation"),
    ("c46", "改密边界", "cases.c46_password_edges"),
    ("c47", "硬编码GPU路径", "cases.c47_hardcoded_gpu_path"),
    ("c48", "审计operator一致性", "cases.c48_operator_consistency"),
    ("c49", "异常图片输入", "cases.c49_weird_images"),
    ("c50", "会话级联删除", "cases.c50_session_cascade"),
    ("c51", "注册面板崩溃回归", "cases.c51_register_panel_bug"),
    ("c52", "注册角色防护", "cases.c52_register_role_guard"),
    ("c53", "历史operator可见性", "cases.c53_chao_history_visibility"),
    ("c54", "阈值滑条边界", "cases.c54_threshold_ui_bounds"),
    ("c55", "依赖锁定完整性", "cases.c55_requirements_lock_gap"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCB Stage5 验收测试（只汇报问题）")
    parser.add_argument(
        "--only",
        default="",
        help="只跑指定用例，逗号分隔，如 c01,c05",
    )
    return parser.parse_args()


def main() -> int:
    ensure_out()
    _quiet_console_logging()
    args = parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    # 保证可导入推理工程模块
    engine_root = str(ENGINE_ROOT)
    if engine_root not in sys.path:
        sys.path.insert(0, engine_root)

    selected = [item for item in CASES if not only or item[0] in only]
    if only:
        unknown = sorted(only - {item[0] for item in CASES})
        if unknown:
            print("问题汇总：")
            for name in unknown:
                print(f"- 未知用例: {name}")
            return 2

    results: list[CaseResult] = []
    started = time.perf_counter()

    from _harness import run_case

    for case_id, title, module_name in selected:
        mod = importlib.import_module(module_name)
        case_name = f"{case_id} {title}"
        result = run_case(case_name, mod.run)
        _quiet_console_logging()
        results.append(result)

    failed = [r for r in results if not r.ok]
    elapsed = time.perf_counter() - started

    report_path = TEST_ROOT / "out" / "problems_report.txt"
    lines: list[str] = []
    if not failed:
        lines.append("无问题")
        lines.append(f"已跑 {len(results)} 项，耗时 {elapsed:.1f}s")
        text = "\n".join(lines) + "\n"
        report_path.write_text(text, encoding="utf-8")
        print("无问题")
        return 0

    lines.append("问题汇总：")
    for result in failed:
        lines.append(f"[{result.name}]")
        for problem in result.problems:
            for part in str(problem).splitlines() or [str(problem)]:
                lines.append(f"  - {part}")
    lines.append(f"失败 {len(failed)}/{len(results)} 项，耗时 {elapsed:.1f}s")
    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
