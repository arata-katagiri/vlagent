"""Compile model-written skill code with a small, closed namespace.

No imports, no dunder access, no file or network builtins. The code gets
`math` and `np`, and everything it needs about the world comes through the
`arm` object it is handed at run time. This is a hackathon sandbox, not a
security boundary: it stops the honest mistakes.
"""

from __future__ import annotations

import math
import re

import numpy as np

FORBIDDEN = re.compile(
    r"(^|\n)\s*(import|from)\s|__\w+__|(?<![\.\w])(exec|eval|open|compile|globals|locals|getattr|setattr|delattr|input)\s*\("
)

SAFE_BUILTINS = {
    n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
    for n in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
        "list", "max", "min", "print", "range", "reversed", "round", "sorted",
        "str", "sum", "tuple", "zip", "isinstance", "ValueError", "RuntimeError",
        "Exception", "True", "False", "None",
    )
    if (n in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, n))
}


class SkillCodeError(ValueError):
    """The code cannot be loaded as a skill."""


def compile_skill(code: str) -> dict:
    """Return {"run": fn, "check": fn | None} or raise SkillCodeError."""
    if not isinstance(code, str) or not code.strip():
        raise SkillCodeError("empty code")
    m = FORBIDDEN.search(code)
    if m:
        raise SkillCodeError(f"forbidden in skill code: {m.group(0).strip()!r}")
    namespace: dict = {"__builtins__": SAFE_BUILTINS, "math": math, "np": np}
    try:
        exec(compile(code, "<skill>", "exec"), namespace)  # noqa: S102
    except SyntaxError as exc:
        raise SkillCodeError(f"syntax error at line {exc.lineno}: {exc.msg}") from exc
    except Exception as exc:  # noqa: BLE001
        raise SkillCodeError(f"{type(exc).__name__} while loading: {exc}") from exc
    run = namespace.get("run")
    if not callable(run):
        raise SkillCodeError("the code must define run(arm, **args)")
    check = namespace.get("check")
    return {"run": run, "check": check if callable(check) else None}


__all__ = ["compile_skill", "SkillCodeError"]
