from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CaseResult:
    name: str
    ok: bool
    problems: list[str] = field(default_factory=list)


class Checker:
    """收集断言失败；通过的不输出。"""

    def __init__(self, case_name: str):
        self.case_name = case_name
        self.problems: list[str] = []

    def check(self, cond: bool, message: str) -> None:
        if not cond:
            self.problems.append(message)

    def fail(self, message: str) -> None:
        self.problems.append(message)

    def result(self) -> CaseResult:
        return CaseResult(
            name=self.case_name,
            ok=not self.problems,
            problems=list(self.problems),
        )


def run_case(name: str, fn: Callable[[Checker], None]) -> CaseResult:
    checker = Checker(name)
    try:
        fn(checker)
    except Exception as exc:
        checker.fail(f"未捕获异常: {exc.__class__.__name__}: {exc}")
        checker.fail(traceback.format_exc(limit=6).strip())
    return checker.result()
