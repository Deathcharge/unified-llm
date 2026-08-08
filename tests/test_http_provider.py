from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any

import httpx
import pytest

from unified_llm import ConfigurationError, OpenAICompatibleProvider, ProviderError, RequestValidationError


async def test_openai_compatible_request_and_response_contract() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        seen["attribution"] = request.headers.get("X-App")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [
                    {
                        "message": {
                            "content": [{"type": "text", "text": "hello"}],
                            "tool_calls": [{"id": "call-1", "type": "function"}],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3, "ignored": "x"},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        name="mock",
        base_url="https://example.test/v1/",
        api_key="secret-key",
        headers={"X-App": "test"},
        client=http_client,
    )
    result = await provider.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="requested-model",
        max_tokens=10,
        temperature=0,
        timeout=5,
        tools=[{"type": "function", "function": {"name": "demo", "parameters": {}}}],
    )

    assert seen == {
        "url": "https://example.test/v1/chat/completions",
        "authorization": "Bearer secret-key",
        "attribution": "test",
        "payload": {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "requested-model",
            "max_tokens": 10,
            "temperature": 0,
            "tools": [{"type": "function", "function": {"name": "demo", "parameters": {}}}],
        },
    }
    assert result.content == "hello"
    assert result.model == "served-model"
    assert result.finish_reason == "tool_calls"
    assert result.total_tokens == 3
    assert result.tool_calls == ({"id": "call-1", "type": "function"},)
    await provider.aclose()
    assert not http_client.is_closed
    await http_client.aclose()


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (401, False), (408, True), (429, True), (500, True), (503, True)],
)
async def test_http_statuses_are_classified_without_body_leakage(status: int, retryable: bool) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers={"Retry-After": "3"}, text="secret-response-body")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(name="mock", client=http_client)
    with pytest.raises(ProviderError) as caught:
        await provider.complete(
            messages=[{"role": "user", "content": "secret-prompt"}],
            model="m",
            max_tokens=1,
            temperature=0,
            timeout=1,
        )
    assert caught.value.status_code == status
    assert caught.value.retryable is retryable
    assert caught.value.retry_after == (3 if retryable else None)
    assert "secret-response-body" not in str(caught.value)
    assert "secret-prompt" not in str(caught.value)
    await http_client.aclose()


async def test_network_error_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret-endpoint-detail", request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(name="mock", client=http_client)
    with pytest.raises(ProviderError, match="unreachable") as caught:
        await provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            max_tokens=1,
            temperature=0,
            timeout=1,
        )
    assert caught.value.retryable
    assert "secret-endpoint-detail" not in str(caught.value)
    await http_client.aclose()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ReadTimeout("timeout"), "timed out"),
        (httpx.ProtocolError("protocol"), "request failed"),
    ],
)
async def test_transport_failures_are_sanitized(error: httpx.HTTPError, message: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        error.request = request
        raise error

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(name="mock", client=http_client)
    with pytest.raises(ProviderError, match=message) as caught:
        await provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            max_tokens=1,
            temperature=0,
            timeout=1,
        )
    assert caught.value.retryable
    await http_client.aclose()


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        (None, None),
        ("not-a-date", None),
        (format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30)), "positive"),
    ],
)
async def test_retry_after_parsing(retry_after: str | None, expected: str | None) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        headers = {"Retry-After": retry_after} if retry_after is not None else {}
        return httpx.Response(429, headers=headers)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(name="mock", client=http_client)
    with pytest.raises(ProviderError) as caught:
        await provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            max_tokens=1,
            temperature=0,
            timeout=1,
        )
    if expected == "positive":
        assert caught.value.retry_after is not None and caught.value.retry_after > 0
    else:
        assert caught.value.retry_after is None
    await http_client.aclose()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
async def test_malformed_success_response_is_rejected(payload: dict[str, object]) -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload)))
    provider = OpenAICompatibleProvider(name="mock", client=http_client)
    with pytest.raises(ProviderError, match=r"malformed|neither content") as caught:
        await provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            max_tokens=1,
            temperature=0,
            timeout=1,
        )
    assert not caught.value.retryable
    await http_client.aclose()


@pytest.mark.parametrize("metadata", [object(), float("nan"), float("inf")])
async def test_non_serializable_or_non_finite_payload_is_rejected_before_network(metadata: object) -> None:
    provider = OpenAICompatibleProvider(name="mock")
    with pytest.raises(RequestValidationError, match="JSON serializable"):
        await provider.complete(
            messages=[{"role": "user", "content": "Hi", "metadata": metadata}],
            model="m",
            max_tokens=1,
            temperature=0,
            timeout=1,
        )
    await provider.aclose()


async def test_string_content_and_missing_usage_use_defaults() -> None:
    payload = {"choices": [{"message": {"content": "hello"}}]}
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload)))
    provider = OpenAICompatibleProvider(name="mock", client=http_client)
    result = await provider.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="requested",
        max_tokens=1,
        temperature=0,
        timeout=1,
    )
    assert result.content == "hello"
    assert result.model == "requested"
    assert result.usage == {}
    assert result.finish_reason == "stop"
    await http_client.aclose()


async def test_internally_created_client_is_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
            kwargs.pop("timeout", None)
            return httpx.Request(method, url, **kwargs)

        async def send(self, request: httpx.Request, **_kwargs: object) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=request)

        async def aclose(self) -> None:
            self.closed = True

    fake = FakeClient()
    monkeypatch.setattr("unified_llm.unified_llm.httpx.AsyncClient", lambda: fake)
    provider = OpenAICompatibleProvider(name="mock")
    await provider.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="m",
        max_tokens=1,
        temperature=0,
        timeout=1,
    )
    await provider.aclose()
    assert fake.closed


@pytest.mark.parametrize("use_content_length", [True, False])
async def test_response_body_limit_is_enforced(use_content_length: bool) -> None:
    body = json.dumps({"choices": [{"message": {"content": "x" * 200}}]}).encode()
    headers = {"Content-Length": str(len(body))} if use_content_length else {}
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body, headers=headers))
    )
    provider = OpenAICompatibleProvider(name="mock", client=http_client, max_response_bytes=100)
    with pytest.raises(ProviderError, match="byte limit") as caught:
        await provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            max_tokens=1,
            temperature=0,
            timeout=1,
        )
    assert not caught.value.retryable
    await http_client.aclose()


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.com/v1",
        "https://user:pass@example.com/v1",
        "https://example.com/v1?token=secret",
        "http://example.com/v1",
    ],
)
def test_unsafe_base_urls_are_rejected(base_url: str) -> None:
    with pytest.raises(ConfigurationError):
        OpenAICompatibleProvider(name="test", base_url=base_url)


def test_invalid_port_and_empty_provider_name_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="invalid"):
        OpenAICompatibleProvider(name="test", base_url="https://example.com:not-a-port/v1")
    with pytest.raises(ConfigurationError, match="name"):
        OpenAICompatibleProvider(name=" ")
    with pytest.raises(ConfigurationError, match="max_response_bytes"):
        OpenAICompatibleProvider(name="test", max_response_bytes=0)


def test_local_http_and_explicit_private_http_are_supported() -> None:
    local = OpenAICompatibleProvider(name="local", base_url="http://127.0.0.1:11434/v1")
    private = OpenAICompatibleProvider(name="private", base_url="http://llm.internal/v1", allow_insecure_http=True)
    assert local.base_url == "http://127.0.0.1:11434/v1"
    assert private.base_url == "http://llm.internal/v1"


def test_ipv6_loopback_is_normalized() -> None:
    provider = OpenAICompatibleProvider(name="local", base_url="http://[::1]:11434/v1")
    assert provider.base_url == "http://[::1]:11434/v1"


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "other"},
        {"Host": "other"},
        {"X-Test\nInjected": "x"},
        {"X-Test": "x\r\nInjected: y"},
    ],
)
def test_unsafe_custom_headers_are_rejected(headers: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        OpenAICompatibleProvider(name="test", headers=headers)


def test_non_string_headers_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="strings"):
        OpenAICompatibleProvider(name="test", headers={"X-Test": 1})  # type: ignore[dict-item]
