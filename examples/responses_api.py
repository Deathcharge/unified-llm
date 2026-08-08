"""Use OpenAI's Responses API without enabling server-side response storage."""

from __future__ import annotations

import asyncio
import os

from unified_llm import OpenAIResponsesProvider, Route, UnifiedLLM


async def main() -> None:
    provider = OpenAIResponsesProvider(
        name="openai-responses",
        api_key=os.environ["OPENAI_API_KEY"],
        store=False,
    )
    async with UnifiedLLM([Route(provider, "gpt-5-mini")]) as client:
        response = await client.generate_with_metadata(
            "Give me three practical reasons to bound LLM responses.",
            max_tokens=300,
            temperature=0.2,
        )
        print(response.content)
        print(f"provider={response.provider} model={response.model} tokens={response.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
