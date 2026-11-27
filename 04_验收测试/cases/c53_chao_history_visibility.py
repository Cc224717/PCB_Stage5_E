from __future__ import annotations

import sys

from _harness import Checker
from _paths import ENGINE_ROOT


def run(c: Checker) -> None:
    """
    摘要改为仅 username 后：
    - 不应串视（好）
    - 但历史若写成 display_name=潮，则欧阳潮本人可能看不到旧检测（功能回退）
    """
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    from db import Database

    db = Database()
    if not db.status().get("enabled"):
        c.fail(f"数据库不可用: {db.status().get('init_error')}")
        return

    user = db.get_user("欧阳潮")
    if user is None:
        c.fail("库中不存在用户 欧阳潮，无法验证历史可见性")
        return

    by_username = db.recent_events_for_operator("欧阳潮", limit=50)
    by_display = db.recent_events_for_operator("潮", limit=50)

    file_by_user = [
        e for e in by_username if "FILE_PROCESS" in str(e.get("event_type") or "")
    ]
    file_by_disp = [
        e for e in by_display if "FILE_PROCESS" in str(e.get("event_type") or "")
    ]

    # UI 现行只查 username
    if len(file_by_disp) > 0 and len(file_by_user) == 0:
        c.fail(
            "【Bug】历史检测写在 operator=潮，但摘要只查 username=欧阳潮，"
            f"导致本人看不到旧检测（display侧 {len(file_by_disp)} 条，username侧 0 条）"
        )
    elif len(file_by_disp) > 0 and len(file_by_user) > 0:
        # 两边都有：可能已双写/迁移，仍提示存在双身份
        if len(file_by_disp) > len(file_by_user):
            c.fail(
                "历史 operator 不一致：operator=潮 的检测多于 username=欧阳潮，"
                f"摘要可能漏展示（潮={len(file_by_disp)}, 欧阳潮={len(file_by_user)}）"
            )

    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT operator, COUNT(*) AS cnt
                FROM audit_events
                WHERE operator IN ('欧阳潮','潮') AND event_type LIKE 'FILE_PROCESS%%'
                GROUP BY operator
                """
            )
            rows = list(cur.fetchall() or [])
    # 仅当存在潮且不存在欧阳潮的 FILE 时判定问题（上面已覆盖）
    mapping = {r["operator"]: int(r["cnt"]) for r in rows}
    if mapping.get("潮", 0) > 0 and mapping.get("欧阳潮", 0) == 0:
        c.fail(f"【Bug】检测审计仅挂在 operator=潮: {mapping}")
