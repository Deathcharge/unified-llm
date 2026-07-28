from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from unified_llm import Message, ToolDefinition, UnifiedLLMResponse


class ScriptedProvider:
    def __init__(self, name: str, outcomes: Sequence[UnifiedLLMResponse | BaseException]) -> None:
        self.name = name
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.active = 0
        self.max_active = 0
        self.gate: asyncio.Event | None = None

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> UnifiedLLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout,
                "tools": tools,
            }
        )
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.gate is not None:
                await self.gate.wait()
            if not self.outcomes:
                raise AssertionError("No scripted provider outcome remains")
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        self.closed = True


def response(content: str = "ok", *, model: str = "served-model") -> UnifiedLLMResponse:
    return UnifiedLLMResponse(
        content=content,
        model=model,
        provider="scripted",
        usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    )
