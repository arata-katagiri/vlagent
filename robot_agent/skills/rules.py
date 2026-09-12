"""Learned safety rules: policy the model proposes, the user keeps, and code enforces.

A rule is a small function over (before, after, action, args) that returns a
reason to refuse, plus an optional level() that can raise a step to caution or
irreversible. Rules are monotonic: they can only add refusals or raise levels,
never remove a built-in check or lower a verdict. They key on object tags so a
rule about "corrosive" things carries across scenes.

Where they run:
  built-in actions   at plan time, on the symbolically predicted after-state
  learned skills     after rehearsal and after the real run, on the measured after-state
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..actions.schema import ActionCall, Safety
from .sandbox import FORBIDDEN, SAFE_BUILTINS, SkillCodeError

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "rules"
LEVEL_RANK = {"safe": 0, "caution": 1, "irreversible": 2}


@dataclass
class Rule:
    name: str
    doc: str
    code: str
    author: str = "model"
    created: str = ""
    dry_run: list[str] = field(default_factory=list)   # what it blocked in the scene when kept
    blocks: int = 0                                     # times it refused something since


def compile_rule(code: str) -> dict:
    """Return {"check": fn, "level": fn | None} or raise SkillCodeError."""
    if not isinstance(code, str) or not code.strip():
        raise SkillCodeError("empty rule code")
    m = FORBIDDEN.search(code)
    if m:
        raise SkillCodeError(f"forbidden in rule code: {m.group(0).strip()!r}")
    ns: dict = {"__builtins__": SAFE_BUILTINS, "math": math, "dist": math.dist}
    try:
        exec(compile(code, "<rule>", "exec"), ns)  # noqa: S102
    except SyntaxError as exc:
        raise SkillCodeError(f"syntax error at line {exc.lineno}: {exc.msg}") from exc
    except Exception as exc:  # noqa: BLE001
        raise SkillCodeError(f"{type(exc).__name__} while loading: {exc}") from exc
    check = ns.get("check")
    if not callable(check):
        raise SkillCodeError("the rule must define check(before, after, action, args)")
    level = ns.get("level")
    return {"check": check, "level": level if callable(level) else None}


class RuleRegistry:
    def __init__(self, directory: Path | str = DEFAULT_DIR):
        self.dir = Path(directory)
        self.rules: dict[str, Rule] = {}
        self.tags: dict[str, list[str]] = {}      # object name -> tags, persisted in _tags.json

    # -- persistence -----------------------------------------------------
    def load(self) -> int:
        self.rules.clear()
        self.tags.clear()
        if not self.dir.exists():
            return 0
        tag_file = self.dir / "_tags.json"
        if tag_file.exists():
            try:
                self.tags.update({k: list(v) for k, v in json.loads(tag_file.read_text()).items()})
            except (json.JSONDecodeError, AttributeError):
                pass
        for path in sorted(self.dir.glob("*.json")):
            if path.name.startswith("_"):
                continue
            try:
                self.rules[path.stem] = Rule(**json.loads(path.read_text()))
            except (json.JSONDecodeError, TypeError) as exc:
                print(f"skipping {path.name}: {exc}")
        return len(self.rules)

    def save(self, rule: Rule) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{rule.name}.json"
        path.write_text(json.dumps(asdict(rule), indent=2))
        return path

    def save_tags(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "_tags.json").write_text(json.dumps(self.tags, indent=2))

    def register(self, rule: Rule, persist: bool = True) -> Rule:
        if not rule.name.isidentifier():
            raise ValueError(f"'{rule.name}' is not a valid rule name")
        compile_rule(rule.code)
        if not rule.created:
            rule.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.rules[rule.name] = rule
        if persist:
            self.save(rule)
        return rule

    def forget(self, name: str) -> bool:
        if name not in self.rules:
            return False
        del self.rules[name]
        path = self.dir / f"{name}.json"
        if path.exists():
            path.unlink()
        return True

    def tag(self, obj: str, *tags: str, persist: bool = True) -> list[str]:
        cur = self.tags.setdefault(obj, [])
        for t in tags:
            t = re.sub(r"[^a-z0-9_]", "_", t.strip().lower())
            if t and t not in cur:
                cur.append(t)
        if persist:
            self.save_tags()
        return cur

    # -- lookup ----------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self.rules

    def names(self) -> list[str]:
        return list(self.rules)

    def augment(self, state: dict) -> dict:
        """A copy of the state with each object's tags attached."""
        s = copy.deepcopy(state)
        for name, obj in s.get("objects", {}).items():
            obj["tags"] = list(self.tags.get(name, []))
        return s

    def describe(self) -> str:
        lines = []
        if self.tags:
            lines.append("Object tags: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in self.tags.items() if v))
        if self.rules:
            lines.append("Learned safety rules (deterministic; you cannot argue past them):")
            for r in self.rules.values():
                lines.append(f"  {r.name}: {r.doc}")
        return "\n".join(lines)

    # -- judgement -------------------------------------------------------
    def check(self, before: dict, after: dict, action: str, args: dict) -> list[str]:
        """Every refusal every rule raises. Only ever adds."""
        b, a = self.augment(before), self.augment(after)
        problems = []
        for r in self.rules.values():
            try:
                why = compile_rule(r.code)["check"](b, a, action, dict(args))
            except Exception as exc:  # noqa: BLE001
                why = f"rule {r.name} crashed ({type(exc).__name__}: {exc}); refusing to be safe"
            if why:
                r.blocks += 1
                problems.append(f"rule {r.name}: {why}")
        return problems

    def level(self, before: dict, after: dict, action: str, args: dict) -> tuple[str, str] | None:
        """The strictest level any rule raises, or None. Never lowers."""
        b, a = self.augment(before), self.augment(after)
        best: tuple[str, str] | None = None
        for r in self.rules.values():
            fn = compile_rule(r.code)["level"]
            if fn is None:
                continue
            try:
                got = fn(b, a, action, dict(args))
            except Exception:  # noqa: BLE001
                continue
            if not got:
                continue
            lvl, why = (got if isinstance(got, tuple) else (str(got), r.doc))
            lvl = str(lvl).lower()
            if lvl in LEVEL_RANK and (best is None or LEVEL_RANK[lvl] > LEVEL_RANK[best[0]]):
                best = (lvl, f"rule {r.name}: {why}")
        return best


RULES = RuleRegistry()


# -- hooks into the built-in safety layer ------------------------------------
def _predict(call: ActionCall, state: dict) -> dict:
    from ..actions.executor import _apply_effects  # noqa: PLC0415

    try:
        return _apply_effects(call, state)
    except Exception:  # noqa: BLE001
        return state


def precondition_hook(call: ActionCall, state: dict) -> list[str]:
    if not RULES.rules:
        return []
    return RULES.check(state, _predict(call, state), call.name, call.args)


def classify_hook(call: ActionCall, state: dict):
    """Runs first; folds in every other verdict and returns the strictest.
    Learned rules can raise a level, never lower one."""
    from ..actions import safety  # noqa: PLC0415

    others = [h for h in safety.CLASSIFY_HOOKS if h is not classify_hook]
    saved = safety.CLASSIFY_HOOKS[:]
    safety.CLASSIFY_HOOKS[:] = others
    try:
        base = safety.classify(call, state)
    finally:
        safety.CLASSIFY_HOOKS[:] = saved
    if not RULES.rules:
        return base
    raised = RULES.level(state, _predict(call, state), call.name, call.args)
    if raised is None:
        return base
    lvl, why = raised
    if LEVEL_RANK[lvl] > LEVEL_RANK[base[0].value]:
        return Safety(lvl), why
    return base


def install_hooks() -> None:
    from ..actions import safety  # noqa: PLC0415

    if precondition_hook not in safety.PRECONDITION_HOOKS:
        safety.PRECONDITION_HOOKS.append(precondition_hook)
    if classify_hook not in safety.CLASSIFY_HOOKS:
        safety.CLASSIFY_HOOKS.insert(0, classify_hook)


# -- dry run: what would this rule refuse in the scene right now? ------------
def dry_run(rule: Rule, state: dict, limit: int = 12) -> list[str]:
    from ..actions.executor import _apply_effects  # noqa: PLC0415
    from ..actions.schema import DIRECTIONS  # noqa: PLC0415

    fns = compile_rule(rule.code)
    b = RULES.augment(state)
    objects = [n for n, o in state["objects"].items() if o.get("graspable", True) and o.get("on_table")]
    targets = [n for n in state["objects"] if n not in ("",)]
    candidates: list[tuple[str, dict, dict]] = []
    for o in objects:
        pick = ActionCall(name="pick", args={"object": o}, rationale="")
        held = _apply_effects(pick, state)
        candidates.append(("pick", pick.args, held))
        for t in targets:
            if t == o:
                continue
            c = ActionCall(name="place_on", args={"target": t}, rationale="")
            candidates.append(("place_on", {"object": o, **c.args}, _apply_effects(c, held)))
            tx, ty = state["objects"][t]["position_m"][:2]
            c = ActionCall(name="place_at", args={"x_m": round(tx + 0.05, 3), "y_m": round(ty, 3)}, rationale="")
            candidates.append(("place_at", {"object": o, **c.args}, _apply_effects(c, held)))
        for d in DIRECTIONS:
            c = ActionCall(name="push", args={"object": o, "direction": d, "distance_m": 0.10}, rationale="")
            candidates.append(("push", c.args, _apply_effects(c, state)))
    blocked = []
    for action, args, after in candidates:
        try:
            why = fns["check"](b, RULES.augment(after), action, dict(args))
        except Exception as exc:  # noqa: BLE001
            why = f"rule crashed: {type(exc).__name__}: {exc}"
        if why:
            desc = ", ".join(f"{k}={v}" for k, v in args.items())
            blocked.append(f"{action}({desc}): {why}")
            if len(blocked) >= limit:
                break
    return blocked


__all__ = ["Rule", "RuleRegistry", "RULES", "compile_rule", "dry_run", "install_hooks",
           "precondition_hook", "classify_hook"]
