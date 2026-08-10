from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from examples.support_triage import TRIAGE_TOOL, TriageDecision, triage_ticket
from unified_llm import OpenAIResponsesProvider, Route, UnifiedLLM, UnifiedLLMResponse

ROOT = Path(__file__).parents[1]


async def test_support_triage_public_api_contract() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gpt-5-mini-2026-06-01",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-triage",
                        "name": "route_support_ticket",
                        "arguments": json.dumps(
                            {"queue": "security", "urgency": "critical", "summary": "Possible account takeover"}
                        ),
                    }
                ],
                "usage": {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIResponsesProvider(name="openai", api_key="test-secret", client=http_client, store=False)
    client = UnifiedLLM(
        [Route(provider, "gpt-5-mini")],
        max_request_bytes=64_000,
        max_response_bytes=256_000,
        max_response_chars=8_000,
        max_tool_calls=4,
    )
    result = await triage_ticket(client, "Unknown login changed my recovery email and locked me out.")

    assert result == TriageDecision(
        queue="security",
        urgency="critical",
        summary="Possible account takeover",
        provider="openai",
        model="gpt-5-mini-2026-06-01",
        total_tokens=42,
    )
    assert seen["url"] == "https://api.openai.com/v1/responses"
    assert seen["authorization"] == "Bearer test-secret"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["store"] is False
    assert payload["max_output_tokens"] == 300
    assert payload["temperature"] == 0
    assert payload["tools"] == [
        {
            "type": "function",
            "name": TRIAGE_TOOL["function"]["name"],
            "description": TRIAGE_TOOL["function"]["description"],
            "parameters": TRIAGE_TOOL["function"]["parameters"],
            "strict": True,
        }
    ]
    assert "test-secret" not in json.dumps(payload)
    await client.aclose()
    await http_client.aclose()


class DecisionProvider:
    name = "decision"

    def __init__(self, *, name: str = "route_support_ticket", arguments: str, count: int = 1) -> None:
        self.function_name = name
        self.arguments = arguments
        self.count = count

    async def complete(self, **_kwargs: object) -> UnifiedLLMResponse:
        call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": self.function_name, "arguments": self.arguments},
        }
        return UnifiedLLMResponse(
            content="",
            model="contract-model",
            provider=self.name,
            tool_calls=tuple(dict(call) for _ in range(self.count)),
        )


@pytest.mark.parametrize(
    "provider",
    [
        DecisionProvider(arguments="not-json"),
        DecisionProvider(arguments='{"queue":"unknown","urgency":"normal","summary":"x"}'),
        DecisionProvider(arguments='{"queue":[],"urgency":"normal","summary":"x"}'),
        DecisionProvider(arguments='{"queue":"bug","urgency":{},"summary":"x"}'),
        DecisionProvider(arguments='{"queue":"bug","urgency":"normal","summary":"x","extra":true}'),
        DecisionProvider(name="wrong_tool", arguments='{"queue":"bug","urgency":"normal","summary":"x"}'),
        DecisionProvider(arguments='{"queue":"bug","urgency":"normal","summary":"x"}', count=2),
    ],
)
async def test_support_triage_rejects_malformed_model_decisions(provider: DecisionProvider) -> None:
    client = UnifiedLLM([Route(provider, "contract-model")])
    with pytest.raises(ValueError, match="provider"):
        await triage_ticket(client, "The app crashed")


async def test_support_triage_rejects_empty_ticket() -> None:
    client = UnifiedLLM([Route(DecisionProvider(arguments="{}"), "contract-model")])
    with pytest.raises(ValueError, match="ticket"):
        await triage_ticket(client, " ")


def test_reference_consumer_uses_only_public_package_imports() -> None:
    source = (ROOT / "examples" / "support_triage.py").read_text(encoding="utf-8")
    assert "unified_llm.unified_llm" not in source
    assert "from unified_llm import" in source
