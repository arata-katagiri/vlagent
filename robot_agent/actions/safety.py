"""Deterministic preconditions and irreversibility prediction.

Nothing here consults the LLM. The planner cannot downgrade a classification; it
only ever sees the result. Operates on the plain state dict from world_state.
"""

from __future__ import annotations

from .schema import ActionCall, Safety


def check_preconditions(call: ActionCall, state: dict) -> list[str]:
    """Return human-readable reasons the call cannot run now. Empty means OK."""
    raise NotImplementedError("Phase 2")


def classify(call: ActionCall, state: dict) -> tuple[Safety, str]:
    """Classify a call and give the concrete reason shown to the user.

    The reason must be specific enough to justify a confirmation prompt, e.g.
    "predicted final position is 0.12 m past the table edge; the glass would fall".
    """
    raise NotImplementedError("Phase 2")
