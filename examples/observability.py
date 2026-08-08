"""Export sanitized attempt metrics without exposing application content."""

from __future__ import annotations

import asyncio
from collections import Counter

from unified_llm import Attempt, Route, UnifiedLLM, UnifiedLLMResponse


class DemoProvider:
    name = "demo"

    async def complete(self, **_kwargs: object) -> UnifiedLLMResponse:
        return UnifiedLLMResponse(content="ok", model="demo-model", provider=self.name)


async def main() -> None:
    counters: Counter[tuple[str, str]] = Counter()

    def observe(attempt: Attempt) -> None:
        outcome = attempt.error or "success"
        counters[(attempt.provider, outcome)] += 1

    client = UnifiedLLM([Route(DemoProvider(), "demo-model")], on_attempt=observe)
    async with client:
        await client.generate("This text is never passed to observe().")

    print(counters)
    print(client.get_provider_health())


if __name__ == "__main__":
    asyncio.run(main())
