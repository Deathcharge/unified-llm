# API reference

The supported public names are exported from `unified_llm`. Type information is included through `py.typed`.

## Core construction

### `OpenAICompatibleProvider(...)`

```python
OpenAICompatibleProvider(
    *,
    name: str,
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    headers: Mapping[str, str] | None = None,
    allow_insecure_http: bool = False,
    max_response_bytes: int = 2_000_000,
    client: httpx.AsyncClient | None = None,
)
```

Creates an adapter for `<base_url>/chat/completions`. A supplied HTTP client remains caller-owned.

### `OpenAIResponsesProvider(...)`

```python
OpenAIResponsesProvider(
    *,
    name: str,
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    headers: Mapping[str, str] | None = None,
    allow_insecure_http: bool = False,
    max_response_bytes: int = 2_000_000,
    store: bool = False,
    client: httpx.AsyncClient | None = None,
)
```

Creates an adapter for `<base_url>/responses`. It defaults to stateless `store=False`, translates Chat-style function tools and tool-result messages, and normalizes text, refusals, and function calls into `UnifiedLLMResponse`. Set `store=True` only when the configured endpoint's retention behavior is intentional.

### `Route(provider, model)`

Pairs a provider with its exact default model ID. Provider names must be unique in one router.

### `UnifiedLLM(routes, ...)`

```python
UnifiedLLM(
    routes,
    *,
    request_timeout=30.0,
    max_attempts_per_route=2,
    max_total_attempts=4,
    max_concurrency=10,
    max_input_chars=200_000,
    max_request_bytes=1_000_000,
    max_response_bytes=2_000_000,
    max_response_chars=1_000_000,
    max_tool_calls=128,
    max_output_tokens=32_768,
    backoff_base=0.25,
    max_retry_delay=5.0,
    retry_jitter=0.1,
)
```

Routes are tried in order when neither `provider` nor `model` selects a single destination.

### `UnifiedLLM.from_env(prefix="UNIFIED_LLM", *, environ=None)`

Builds one route from the variables documented in the README. Passing an explicit mapping as `environ` makes configuration deterministic in tests.

## Request methods

All methods are async.

- `generate(prompt, *, model=None, provider=None, max_tokens=512, temperature=0.7, system=None) -> str`
- `generate_with_metadata(...) -> UnifiedLLMResponse`
- `chat(messages, *, model=None, provider=None, max_tokens=512, temperature=0.7) -> str`
- `chat_with_metadata(..., tools=None) -> UnifiedLLMResponse`
- `chat_with_tools(messages, *, tools, ...) -> UnifiedLLMResponse`

Messages are mappings with a supported `role` (`assistant`, `developer`, `system`, `tool`, or `user`) and non-empty string `content`. Additional JSON-serializable message fields are preserved for compatible endpoints.

Convenience methods raise the same typed errors as metadata methods; they never convert errors to empty strings.

## Response types

### `UnifiedLLMResponse`

- `content: str`
- `model: str`
- `provider: str`
- `usage: dict[str, int]`
- `finish_reason: str`
- `tool_calls: tuple[dict[str, Any], ...]`
- `attempts: tuple[Attempt, ...]`
- `total_tokens: int` property

### `Attempt`

Contains `provider`, `model`, one-based route-attempt `number`, `latency_ms`, sanitized `error`, and `retryable`. It intentionally excludes endpoints, headers, prompts, response bodies, and exception text from unexpected adapters.

## Errors

All package errors derive from `UnifiedLLMError`.

- `ConfigurationError`
- `RequestValidationError` (also a `ValueError`)
- `ProviderError`: includes `provider`, optional `status_code`, `retryable`, optional `retry_after`, and completed `attempts`.
- `FallbackExhausted`: includes completed `attempts`.

Cancellation is not wrapped.

## Custom provider protocol

```python
from collections.abc import Mapping, Sequence
from typing import Any

from unified_llm import Message, ToolDefinition, UnifiedLLMResponse


class MyProvider:
    name = "my-provider"

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> UnifiedLLMResponse: ...
```

Adapters must honor the timeout, propagate `asyncio.CancelledError`, sanitize `ProviderError`, avoid logging sensitive values, and mark only failures that are safe to repeat as retryable. An optional sync or async `aclose()` is called by the router lifecycle.

The router validates every adapter result before returning it. Responses must be `UnifiedLLMResponse` instances with serializable metadata, non-negative integer usage values, dictionary tool calls, and values within the configured response limits.
