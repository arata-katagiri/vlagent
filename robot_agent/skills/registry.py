"""Layer 2: the skill library.

A skill is task-level code written against the Arm API in body.py. Skills that
shipped with the robot and skills the model wrote a minute ago are the same
kind of thing: name, arguments, declared effect, code, provenance. They are
persisted one JSON file per skill so the library survives a restart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "skills"

# The only argument names a skill may declare. Shared with the built-in actions
# so the planner's flat step schema stays reliable (see schema.plan_tool_schema).
SKILL_ARG_KEYS = (
    "object", "target", "x_m", "y_m", "direction", "distance_m",
    "angle_deg", "height_m", "count", "speed",
)

# What a skill may claim to do, in terms the rehearsal can measure itself.
# "other" means the world cannot check it and only the skill's own check() speaks.
EFFECT_KINDS = (
    "leaves_table",     # the object's centre ends up beyond the table edge
    "moves",            # the object moved at least 1 cm
    "moves_at_least",   # the object moved at least effect_value metres
    "rotates_by",       # yaw changed by effect_value degrees (within 20)
    "tips_over",        # the object is no longer upright
    "touches",          # the hand or fingers made contact with the object
    "rests_on",         # the object ends up supported by the `target` argument
    "stays",            # the object did not move
    "other",            # not measurable here; self-reported by check()
)


@dataclass
class Skill:
    name: str
    doc: str
    effect: str
    args: list[str]
    code: str
    effect_kind: str = "other"
    effect_of: str = "object"
    effect_value: str = ""
    # graph edges
    calls: list[str] = field(default_factory=list)        # recorded during rehearsal
    signature_vec: list[float] = field(default_factory=list)  # measured motion signature
    derived_from: str = ""                                 # the model's opinion
    similar_to: list[str] = field(default_factory=list)    # the model's opinion
    stale: bool = False                                    # a skill it calls was revised
    trust: str = "world"                                   # "world" | "human": who confirmed the effect
    author: str = "model"
    created: str = ""
    rehearsals: list[dict] = field(default_factory=list)
    uses: int = 0

    @property
    def last_level(self) -> str:
        """Safety level measured in the most recent successful rehearsal."""
        for r in reversed(self.rehearsals):
            if r.get("ok"):
                return str(r.get("level", "caution"))
        return "caution"

    def signature(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    @property
    def world_verified(self) -> bool:
        return self.trust == "world" and self.effect_kind in EFFECT_KINDS and self.effect_kind != "other"

    def effect_label(self) -> str:
        if self.trust == "human":
            return "human-verified (rehearsal overridden)"
        if not self.world_verified:
            return "self-reported"
        v = f", {self.effect_value}" if self.effect_value else ""
        return f"{self.effect_kind}({self.effect_of}{v})"


class Registry:
    def __init__(self, directory: Path | str = DEFAULT_DIR):
        self.dir = Path(directory)
        self.skills: dict[str, Skill] = {}

    # -- persistence -----------------------------------------------------
    def load(self) -> int:
        self.skills.clear()
        if not self.dir.exists():
            return 0
        for path in sorted(self.dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text())
                skill = Skill(**raw)
            except (json.JSONDecodeError, TypeError) as exc:
                print(f"skipping {path.name}: {exc}")
                continue
            self.skills[skill.name] = skill
        return len(self.skills)

    def save(self, skill: Skill) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{skill.name}.json"
        path.write_text(json.dumps(asdict(skill), indent=2))
        return path

    def register(self, skill: Skill, persist: bool = True) -> Skill:
        if not skill.created:
            skill.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        bad = [a for a in skill.args if a not in SKILL_ARG_KEYS]
        if bad:
            raise ValueError(f"skill arguments must come from {SKILL_ARG_KEYS}; got {bad}")
        self.skills[skill.name] = skill
        if persist:
            self.save(skill)
        return skill

    # -- graph -----------------------------------------------------------
    def dependents(self, name: str) -> list[str]:
        return [s.name for s in self.skills.values() if name in s.calls]

    def mark_stale_dependents(self, name: str) -> list[str]:
        """A revised skill invalidates the rehearsal record of everything built on it."""
        changed = []
        for dep in self.dependents(name):
            s = self.skills[dep]
            if not s.stale:
                s.stale = True
                self.save(s)
                changed.append(dep)
            changed += self.mark_stale_dependents(dep)
        return changed

    def nearest(self, vec: list[float], k: int = 2, exclude=()) -> list[tuple[str, float]]:
        from .graph import nearest  # noqa: PLC0415

        return nearest(self.skills, vec, k=k, exclude=exclude)

    def graph_lines(self) -> list[str]:
        from .graph import graph_lines  # noqa: PLC0415

        return graph_lines(self.skills)

    def forget(self, name: str) -> bool:
        if name not in self.skills:
            return False
        del self.skills[name]
        path = self.dir / f"{name}.json"
        if path.exists():
            path.unlink()
        return True

    # -- lookup ----------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self.skills

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def names(self) -> list[str]:
        return list(self.skills)

    def describe(self) -> str:
        """Catalogue lines for the system prompt, same shape as the built-ins."""
        if not self.skills:
            return ""
        lines = ["Learned skills (call them like any action):"]
        for s in self.skills.values():
            lines.append(f"  {s.signature()}")
            lines.append(f"      {s.doc}")
            lines.append(f"      effect: {s.effect}  [{s.effect_label()}]")
            if s.calls:
                lines.append(f"      built from: {', '.join(s.calls)}")
            if s.stale:
                lines.append("      STALE: a skill it calls was revised; it must be rehearsed again before use")
            if s.args:
                lines.append(f"      required keys in args: {', '.join(s.args)}")
        return "\n".join(lines)


REGISTRY = Registry()

__all__ = ["Skill", "Registry", "REGISTRY", "SKILL_ARG_KEYS", "EFFECT_KINDS", "DEFAULT_DIR"]
