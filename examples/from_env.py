"""Make one bounded request to an environment-configured endpoint."""

import asyncio

from unified_llm import UnifiedLLM, UnifiedLLMError


async def main() -> None:
    try:
        async with UnifiedLLM.from_env() as llm:
            result = await llm.generate_with_metadata(
                "Explain bounded retries in two sentences.",
                max_tokens=150,
                temperature=0.2,
            )
    except UnifiedLLMError as exc:
        raise SystemExit(f"Request failed safely: {exc}") from exc
    print(f"content={result.content!r}")
    print(f"provider={result.provider!r} model={result.model!r} total_tokens={result.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
