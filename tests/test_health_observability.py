from __future__ import annotations

import pytest

from unified_llm import Attempt, ProviderError, Route, UnifiedLLM

from .conftest import ScriptedProvider, response


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


async def test_cooling_provider_is_deprioritized_then_recovers() -> None:
    clock = FakeClock()
    primary = ScriptedProvider(
        "primary",
        [
            ProviderError("primary", "temporary", retryable=True),
            ProviderError("primary", "temporary", retryable=True),
            response("recovered"),
        ],
    )
    backup = ScriptedProvider("backup", [response("backup-1"), response("backup-2"), response("backup-3")])
    client = UnifiedLLM(
        [Route(primary, "model-a"), Route(backup, "model-b")],
        max_attempts_per_route=1,
        health_failure_threshold=2,
        health_cooldown=30,
        _clock=clock,
    )

    assert await client.generate("one") == "backup-1"
    assert await client.generate("two") == "backup-2"
    health = client.get_provider_health()
    assert health["primary"].consecutive_failures == 2
    assert health["primary"].cooling_down
    assert health["primary"].cooldown_remaining == 30

    assert await client.generate("three") == "backup-3"
    assert len(primary.calls) == 2

    clock.now += 31
    assert await client.generate("four") == "recovered"
    assert len(primary.calls) == 3
    assert client.get_provider_health()["primary"].consecutive_failures == 0
    assert not client.get_provider_health()["primary"].cooling_down


async def test_opening_cooldown_stops_same_route_retries() -> None:
    primary = ScriptedProvider(
        "primary",
        [ProviderError("primary", "temporary", retryable=True), response("should-not-run")],
    )
    backup = ScriptedProvider("backup", [response("backup")])
    client = UnifiedLLM(
        [Route(primary, "model-a"), Route(backup, "model-b")],
        max_attempts_per_route=2,
        health_failure_threshold=1,
        health_cooldown=30,
    )
    assert await client.generate("hello") == "backup"
    assert len(primary.calls) == 1


async def test_explicit_provider_selection_can_probe_cooling_route() -> None:
    primary = ScriptedProvider(
        "primary",
        [ProviderError("primary", "temporary", retryable=True), response("probe-ok")],
    )
    backup = ScriptedProvider("backup", [response("backup")])
    client = UnifiedLLM(
        [Route(primary, "model-a"), Route(backup, "model-b")],
        max_attempts_per_route=1,
        health_failure_threshold=1,
    )
    assert await client.generate("implicit") == "backup"
    assert client.get_provider_health()["primary"].cooling_down
    assert await client.generate("explicit", provider="primary") == "probe-ok"
    assert not client.get_provider_health()["primary"].cooling_down


async def test_attempt_hook_receives_only_sanitized_attempts() -> None:
    observed: list[Attempt] = []

    async def observe(attempt: Attempt) -> None:
        observed.append(attempt)

    primary = ScriptedProvider("primary", [ProviderError("primary", "secret", retryable=True)])
    backup = ScriptedProvider("backup", [response("safe")])
    client = UnifiedLLM(
        [Route(primary, "model-a"), Route(backup, "model-b")],
        max_attempts_per_route=1,
        on_attempt=observe,
    )
    assert await client.generate("secret prompt") == "safe"
    assert [item.provider for item in observed] == ["primary", "backup"]
    assert observed[0].error == "provider_error"
    assert observed[0].retryable
    assert observed[1].error is None
    assert "secret" not in repr(observed)


async def test_attempt_hook_failure_does_not_break_generation() -> None:
    def broken_observer(_attempt: Attempt) -> None:
        raise RuntimeError("observer secret")

    provider = ScriptedProvider("primary", [response("ok")])
    client = UnifiedLLM([Route(provider, "model")], on_attempt=broken_observer)
    assert await client.generate("hello") == "ok"


async def test_permanent_failure_does_not_open_health_cooldown() -> None:
    provider = ScriptedProvider("primary", [ProviderError("primary", "bad request", retryable=False)])
    client = UnifiedLLM([Route(provider, "model")], health_failure_threshold=1)
    with pytest.raises(ProviderError):
        await client.generate("hello")
    assert client.get_provider_health()["primary"].consecutive_failures == 0
    assert not client.get_provider_health()["primary"].cooling_down
