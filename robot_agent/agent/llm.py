"""OpenAI-compatible LLM client plus an offline mock.

Provider comes from the environment (OPENAI_BASE_URL / OPENAI_API_KEY /
OPENAI_MODEL) so we can switch between the event's credits, OpenRouter, or a
local endpoint without touching code. Never hardcode or print a key.

The installed openai SDK is 3.x -- read its tool-calling signature before
writing this rather than assuming the 1.x shape.
"""

from __future__ import annotations

TIMEOUT_S = 30.0
MAX_RETRIES = 1


class LLMClient:
    """Thin wrapper over the OpenAI-compatible chat completions API."""

    def __init__(self) -> None:
        raise NotImplementedError("Phase 3")


class MockLLM:
    """Maps the scripted demo commands to fixed plans. Makes --mock-llm fully offline."""

    def __init__(self) -> None:
        raise NotImplementedError("Phase 3")
