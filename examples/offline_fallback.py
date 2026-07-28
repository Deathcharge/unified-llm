"""Credential-free fallback demonstration."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from unified_llm import Message, ProviderError, Route, ToolDefinition, UnifiedLLM, UnifiedLLMResponse


class DemoProvider:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail

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
        del messages, max_tokens, temperature, timeout, tools
        if self.fail:
            raise ProviderError(self.name, "demo transient failure", retryable=True)
        return UnifiedLLMResponse("offline fallback works", model, self.name)


async def main() -> None:
    routes = [
        Route(DemoProvider("primary", fail=True), "demo-primary"),
        Route(DemoProvider("backup"), "demo-backup"),
    ]
    async with UnifiedLLM(
        routes,
        max_attempts_per_route=1,
        max_total_attempts=2,
        backoff_base=0,
    ) as llm:
        result = await llm.generate_with_metadata("Demonstrate fallback")
        attempted = ",".join(attempt.provider for attempt in result.attempts)
        print(f"provider={result.provider} content={result.content} attempts={attempted}")


if __name__ == "__main__":
    asyncio.run(main())
