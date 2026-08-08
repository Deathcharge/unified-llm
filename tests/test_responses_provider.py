from __future__ import annotations

import json

import httpx
import pytest

from unified_llm import (
    ConfigurationError,
    OpenAIResponsesProvider,
    ProviderError,
    RequestValidationError,
    Route,
    UnifiedLLM,
)


async def test_responses_request_and_normalized_response_contract() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-served",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Weather: "}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "get_weather",
                        "arguments": '{"city":"Boston"}',
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIResponsesProvider(name="openai", api_key="secret", client=http_client)
    result = await provider.complete(
        messages=[{"role": "developer", "content": "Be concise."}, {"role": "user", "content": "Weather?"}],
        model="gpt-requested",
        max_tokens=100,
        temperature=0,
        timeout=5,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            }
        ],
    )

    assert seen == {
        "url": "https://api.openai.com/v1/responses",
        "payload": {
            "model": "gpt-requested",
            "input": [
                {"role": "developer", "content": "Be concise."},
                {"role": "user", "content": "Weather?"},
            ],
            "max_output_tokens": 100,
            "temperature": 0,
            "store": False,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
        },
    }
    assert result.content == "Weather: "
    assert result.model == "gpt-served"
    assert result.finish_reason == "tool_calls"
    assert result.total_tokens == 14
    assert result.tool_calls == (
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Boston"}'},
        },
    )
    await http_client.aclose()


async def test_responses_tool_output_and_store_opt_in_are_translated() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "72 F"}]}],
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIResponsesProvider(name="openai", store=True, client=http_client)
    result = await provider.complete(
        messages=[{"role": "tool", "tool_call_id": "call-1", "content": "72 F"}],
        model="gpt-test",
        max_tokens=20,
        temperature=0,
        timeout=5,
    )
    assert seen["store"] is True
    assert seen["input"] == [{"type": "function_call_output", "call_id": "call-1", "output": "72 F"}]
    assert result.content == "72 F"
    assert result.finish_reason == "stop"
    await http_client.aclose()


async def test_responses_refusal_is_returned_as_content() -> None:
    payload = {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "Cannot comply."}]}],
    }
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload)))
    provider = OpenAIResponsesProvider(name="openai", client=http_client)
    result = await provider.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-test",
        max_tokens=10,
        temperature=0,
        timeout=5,
    )
    assert result.content == "Cannot comply."
    assert result.finish_reason == "stop"
    await http_client.aclose()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"output": "bad"},
        {"output": []},
        {"output": [{"type": "message", "content": [{"type": "other"}]}]},
    ],
)
async def test_malformed_or_empty_responses_output_is_rejected(payload: dict[str, object]) -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload)))
    provider = OpenAIResponsesProvider(name="openai", client=http_client)
    with pytest.raises(ProviderError, match=r"malformed|neither"):
        await provider.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-test",
            max_tokens=10,
            temperature=0,
            timeout=5,
        )
    await http_client.aclose()


def test_responses_configuration_and_shapes_are_validated() -> None:
    with pytest.raises(ConfigurationError, match="store"):
        OpenAIResponsesProvider(name="openai", store=1)  # type: ignore[arg-type]
    provider = OpenAIResponsesProvider(name="openai")
    with pytest.raises(RequestValidationError, match="tool_call_id"):
        provider._build_payload(
            messages=[{"role": "tool", "content": "result"}],
            model="m",
            max_tokens=1,
            temperature=0,
            tools=None,
        )
    with pytest.raises(RequestValidationError, match="function-tool shape"):
        provider._build_payload(
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            max_tokens=1,
            temperature=0,
            tools=[{"type": "other"}],
        )


async def test_router_accounts_for_exact_responses_payload() -> None:
    provider = OpenAIResponsesProvider(name="openai")
    client = UnifiedLLM([Route(provider, "m")], max_request_bytes=100)
    with pytest.raises(RequestValidationError, match="provider request"):
        await client.chat([{"role": "user", "content": "x" * 40}], max_tokens=1, temperature=0)
    await client.aclose()
