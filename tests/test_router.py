from __future__ import annotations

import asyncio

import pytest

from unified_llm import (
    ConfigurationError,
    FallbackExhausted,
    ProviderError,
    RequestValidationError,
    Route,
    UnifiedLLM,
    UnifiedLLMResponse,
)

from .conftest import ScriptedProvider, response


async def test_generate_returns_content_and_attempt_metadata() -> None:
    provider = ScriptedProvider("primary", [response("hello")])
    client = UnifiedLLM([Route(provider, "model-a")])

    result = await client.generate_with_metadata("Hi", system="Be concise", max_tokens=20, temperature=0)

    assert result.content == "hello"
    assert result.provider == "primary"
    assert result.model == "served-model"
    assert result.total_tokens == 5
    assert len(result.attempts) == 1
    assert result.attempts[0].error is None
    assert provider.calls[0]["messages"] == (
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Hi"},
    )


async def test_generate_convenience_returns_text() -> None:
    provider = ScriptedProvider("primary", [response("hello")])
    assert await UnifiedLLM([Route(provider, "model-a")]).generate("Hi") == "hello"


async def test_transient_errors_retry_then_fall_back() -> None:
    primary = ScriptedProvider(
        "primary",
        [
            ProviderError("primary", "temporary", status_code=503, retryable=True),
            ProviderError("primary", "still temporary", status_code=503, retryable=True),
        ],
    )
    backup = ScriptedProvider("backup", [response("from backup", model="model-b")])
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    client = UnifiedLLM(
        [Route(primary, "model-a"), Route(backup, "model-b")],
        backoff_base=0.5,
        retry_jitter=0,
        _sleep=fake_sleep,
    )

    result = await client.generate_with_metadata("Hi")

    assert result.content == "from backup"
    assert [attempt.provider for attempt in result.attempts] == ["primary", "primary", "backup"]
    assert delays == [0.5]


async def test_retry_after_is_capped() -> None:
    provider = ScriptedProvider(
        "primary",
        [ProviderError("primary", "rate limited", retryable=True, retry_after=30), response()],
    )
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    client = UnifiedLLM(
        [Route(provider, "model-a")],
        max_retry_delay=2,
        retry_jitter=0,
        _sleep=fake_sleep,
    )
    await client.generate("Hi")
    assert delays == [2]


async def test_permanent_failure_does_not_retry_or_fall_back() -> None:
    primary = ScriptedProvider("primary", [ProviderError("primary", "bad request", status_code=400, retryable=False)])
    backup = ScriptedProvider("backup", [response()])
    client = UnifiedLLM([Route(primary, "model-a"), Route(backup, "model-b")])

    with pytest.raises(ProviderError) as caught:
        await client.generate("Hi")

    assert caught.value.status_code == 400
    assert len(caught.value.attempts) == 1
    assert len(primary.calls) == 1
    assert backup.calls == []


async def test_total_attempt_budget_is_enforced() -> None:
    failures = [ProviderError("a", "temporary", retryable=True) for _ in range(3)]
    primary = ScriptedProvider("a", failures)
    backup = ScriptedProvider("b", [ProviderError("b", "temporary", retryable=True)])
    client = UnifiedLLM(
        [Route(primary, "model-a"), Route(backup, "model-b")],
        max_attempts_per_route=3,
        max_total_attempts=2,
        backoff_base=0,
    )

    with pytest.raises(FallbackExhausted) as caught:
        await client.generate("Hi")

    assert len(caught.value.attempts) == 2
    assert backup.calls == []


async def test_explicit_provider_disables_fallback_and_allows_model_override() -> None:
    primary = ScriptedProvider("primary", [response()])
    backup = ScriptedProvider("backup", [response("chosen")])
    client = UnifiedLLM([Route(primary, "model-a"), Route(backup, "model-b")])

    result = await client.generate_with_metadata("Hi", provider="backup", model="override")

    assert result.content == "chosen"
    assert primary.calls == []
    assert backup.calls[0]["model"] == "override"


async def test_exact_model_selects_matching_route() -> None:
    primary = ScriptedProvider("primary", [response()])
    backup = ScriptedProvider("backup", [response("chosen")])
    client = UnifiedLLM([Route(primary, "model-a"), Route(backup, "model-b")])

    assert await client.generate("Hi", model="model-b") == "chosen"
    assert primary.calls == []


@pytest.mark.parametrize(
    ("messages", "kwargs", "match"),
    [
        ([], {}, "non-empty sequence"),
        ([{"role": "invalid", "content": "x"}], {}, "role"),
        ([{"role": "user", "content": ""}], {}, "content"),
        ([{"role": "user", "content": "x"}], {"max_tokens": 0}, "max_tokens"),
        ([{"role": "user", "content": "x"}], {"temperature": 3}, "temperature"),
    ],
)
async def test_request_validation_happens_before_provider_call(
    messages: list[dict[str, str]], kwargs: dict[str, float], match: str
) -> None:
    provider = ScriptedProvider("primary", [response()])
    client = UnifiedLLM([Route(provider, "model-a")])

    with pytest.raises(RequestValidationError, match=match):
        await client.chat(messages, **kwargs)  # type: ignore[arg-type]
    assert provider.calls == []


async def test_input_size_and_tools_are_validated() -> None:
    provider = ScriptedProvider("primary", [response()])
    client = UnifiedLLM([Route(provider, "model-a")], max_input_chars=3)
    with pytest.raises(RequestValidationError, match="character limit"):
        await client.generate("four")
    with pytest.raises(RequestValidationError, match="at least one"):
        await client.chat_with_tools([{"role": "user", "content": "ok"}], tools=[])


async def test_unknown_route_selection_is_explicit() -> None:
    one = ScriptedProvider("one", [response()])
    two = ScriptedProvider("two", [response()])
    client = UnifiedLLM([Route(one, "model-a"), Route(two, "model-b")])

    with pytest.raises(RequestValidationError, match="Unknown provider"):
        await client.generate("Hi", provider="missing")
    with pytest.raises(RequestValidationError, match="requires an explicit provider"):
        await client.generate("Hi", model="unknown")


async def test_concurrency_is_bounded() -> None:
    provider = ScriptedProvider("primary", [response("one"), response("two")])
    provider.gate = asyncio.Event()
    client = UnifiedLLM([Route(provider, "model-a")], max_concurrency=1)

    first = asyncio.create_task(client.generate("one"))
    second = asyncio.create_task(client.generate("two"))
    for _ in range(20):
        if provider.calls:
            break
        await asyncio.sleep(0)
    assert len(provider.calls) == 1
    provider.gate.set()
    assert list(await asyncio.gather(first, second)) == ["one", "two"]
    assert provider.max_active == 1


async def test_cancellation_is_not_retried() -> None:
    provider = ScriptedProvider("primary", [response()])
    provider.gate = asyncio.Event()
    client = UnifiedLLM([Route(provider, "model-a")])

    task = asyncio.create_task(client.generate("Hi"))
    for _ in range(20):
        if provider.calls:
            break
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(provider.calls) == 1


async def test_unexpected_adapter_exception_is_sanitized() -> None:
    provider = ScriptedProvider("primary", [RuntimeError("secret-token")])
    client = UnifiedLLM([Route(provider, "model-a")])

    with pytest.raises(ProviderError) as caught:
        await client.generate("Hi")
    assert "secret-token" not in str(caught.value)
    assert caught.value.attempts[0].error == "adapter_error"


async def test_context_manager_closes_provider() -> None:
    provider = ScriptedProvider("primary", [response()])
    async with UnifiedLLM([Route(provider, "model-a")]) as client:
        assert client.get_available_providers() == ["primary"]
        assert client.providers == {"primary": provider}
    assert provider.closed


def test_response_derives_total_tokens() -> None:
    result = UnifiedLLMResponse("ok", "m", "p", {"input_tokens": 4, "output_tokens": 6})
    assert result.total_tokens == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"routes": []},
        {"request_timeout": 0},
        {"max_attempts_per_route": 0},
        {"max_total_attempts": 0},
        {"max_concurrency": 0},
        {"max_input_chars": 0},
        {"max_request_bytes": 0},
        {"max_output_tokens": 0},
        {"backoff_base": -1},
        {"retry_jitter": 2},
    ],
)
def test_router_configuration_is_bounded(kwargs: dict[str, object]) -> None:
    provider = ScriptedProvider("primary", [response()])
    routes = kwargs.pop("routes", [Route(provider, "model-a")])
    with pytest.raises(ConfigurationError):
        UnifiedLLM(routes, **kwargs)  # type: ignore[arg-type]


def test_duplicate_provider_names_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unique"):
        UnifiedLLM(
            [
                Route(ScriptedProvider("same", [response()]), "a"),
                Route(ScriptedProvider("same", [response()]), "b"),
            ]
        )


def test_route_configuration_is_validated() -> None:
    with pytest.raises(ConfigurationError, match="model"):
        Route(ScriptedProvider("primary", [response()]), "")
    with pytest.raises(ConfigurationError, match="Provider protocol"):
        Route(object(), "model")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="provider name"):
        Route(ScriptedProvider("", [response()]), "model")


def test_route_count_is_bounded() -> None:
    routes = [Route(ScriptedProvider(str(index), [response()]), "model") for index in range(9)]
    with pytest.raises(ConfigurationError, match="eight"):
        UnifiedLLM(routes)


async def test_all_transient_routes_exhaust_cleanly() -> None:
    one = ScriptedProvider("one", [ProviderError("one", "temporary", retryable=True)])
    two = ScriptedProvider("two", [ProviderError("two", "temporary", retryable=True)])
    client = UnifiedLLM(
        [Route(one, "a"), Route(two, "b")],
        max_attempts_per_route=1,
        max_total_attempts=4,
    )
    with pytest.raises(FallbackExhausted) as caught:
        await client.generate("Hi")
    assert [attempt.provider for attempt in caught.value.attempts] == ["one", "two"]


async def test_router_timeout_is_retryable_and_bounded() -> None:
    provider = ScriptedProvider("slow", [response()])
    provider.gate = asyncio.Event()
    client = UnifiedLLM(
        [Route(provider, "model")],
        request_timeout=0.001,
        max_attempts_per_route=1,
    )
    with pytest.raises(FallbackExhausted) as caught:
        await client.generate("Hi")
    assert caught.value.attempts[0].retryable


async def test_jittered_retry_and_chat_tool_conveniences(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ScriptedProvider(
        "primary",
        [ProviderError("primary", "temporary", retryable=True), response("tool result")],
    )
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("unified_llm.unified_llm.random.uniform", lambda _low, _high: 1.1)
    client = UnifiedLLM([Route(provider, "model")], backoff_base=1, retry_jitter=0.2, _sleep=fake_sleep)
    result = await client.chat_with_tools(
        [{"role": "user", "content": "Hi"}],
        tools=[{"type": "function", "function": {"name": "demo"}}],
    )
    assert result.content == "tool result"
    assert delays == [1.1]
    assert provider.calls[-1]["tools"] is not None


async def test_chat_convenience_and_single_route_model_override() -> None:
    provider = ScriptedProvider("primary", [response("chat")])
    client = UnifiedLLM([Route(provider, "default")])
    assert await client.chat([{"role": "user", "content": "Hi"}], model="override") == "chat"
    assert provider.calls[0]["model"] == "override"


@pytest.mark.parametrize("prompt", ["", "   "])
async def test_prompt_must_be_non_empty(prompt: str) -> None:
    client = UnifiedLLM([Route(ScriptedProvider("primary", [response()]), "model")])
    with pytest.raises(RequestValidationError, match="prompt"):
        await client.generate(prompt)


async def test_optional_system_and_route_selector_values_are_validated() -> None:
    provider = ScriptedProvider("primary", [response()])
    client = UnifiedLLM([Route(provider, "model")])
    with pytest.raises(RequestValidationError, match="system"):
        await client.generate("Hi", system="")
    with pytest.raises(RequestValidationError, match="provider"):
        await client.generate("Hi", provider="")
    with pytest.raises(RequestValidationError, match="model"):
        await client.generate("Hi", model="")
    with pytest.raises(RequestValidationError, match="model"):
        await client.generate("Hi", provider="primary", model="")


async def test_non_mapping_messages_and_tools_are_rejected() -> None:
    provider = ScriptedProvider("primary", [response()])
    client = UnifiedLLM([Route(provider, "model")])
    with pytest.raises(RequestValidationError, match=r"messages\[0\]"):
        await client.chat(["not-a-message"])  # type: ignore[list-item]
    with pytest.raises(RequestValidationError, match="tool definition"):
        await client.chat_with_metadata(
            [{"role": "user", "content": "Hi"}],
            tools=["not-a-tool"],  # type: ignore[list-item]
        )


async def test_full_serialized_request_is_bounded_and_validated() -> None:
    provider = ScriptedProvider("primary", [response()])
    client = UnifiedLLM([Route(provider, "model")], max_request_bytes=100)
    with pytest.raises(RequestValidationError, match="byte request limit"):
        await client.chat_with_metadata(
            [{"role": "user", "content": "Hi", "metadata": "x" * 100}],
            tools=[{"type": "function", "function": {"name": "demo"}}],
        )
    with pytest.raises(RequestValidationError, match="JSON serializable"):
        await client.chat_with_metadata(
            [{"role": "user", "content": "Hi", "metadata": object()}],
        )
    assert provider.calls == []


async def test_provider_without_close_is_supported() -> None:
    class NoCloseProvider:
        name = "no-close"

        async def complete(self, **_kwargs: object) -> UnifiedLLMResponse:
            return response()

    client = UnifiedLLM([Route(NoCloseProvider(), "model")])
    await client.aclose()
