"""Reference consumer: classify a support ticket through a bounded function call."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from unified_llm import OpenAIResponsesProvider, Route, UnifiedLLM

TRIAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "route_support_ticket",
        "description": "Return the queue, urgency, and safe one-line summary for a support ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "queue": {"type": "string", "enum": ["billing", "bug", "security", "general"]},
                "urgency": {"type": "string", "enum": ["low", "normal", "high", "critical"]},
                "summary": {"type": "string", "minLength": 1, "maxLength": 240},
            },
            "required": ["queue", "urgency", "summary"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

_QUEUES = frozenset({"billing", "bug", "security", "general"})
_URGENCIES = frozenset({"low", "normal", "high", "critical"})


@dataclass(frozen=True, slots=True)
class TriageDecision:
    queue: str
    urgency: str
    summary: str
    provider: str
    model: str
    total_tokens: int


def build_client(api_key: str) -> UnifiedLLM:
    """Build the reference consumer's privacy-first production client."""

    provider = OpenAIResponsesProvider(name="openai", api_key=api_key, store=False)
    return UnifiedLLM(
        [Route(provider, "gpt-5-mini")],
        request_timeout=20,
        max_attempts_per_route=2,
        max_total_attempts=2,
        max_request_bytes=64_000,
        max_response_bytes=256_000,
        max_response_chars=8_000,
        max_tool_calls=4,
        max_output_tokens=1_000,
    )


async def triage_ticket(client: UnifiedLLM, ticket: str) -> TriageDecision:
    """Classify one ticket and reject malformed model tool arguments."""

    if not isinstance(ticket, str) or not ticket.strip():
        raise ValueError("ticket must be a non-empty string")
    response = await client.chat_with_tools(
        [
            {
                "role": "developer",
                "content": (
                    "Classify support tickets. Never include credentials, tokens, payment details, "
                    "or other secrets in the summary. Always call route_support_ticket exactly once."
                ),
            },
            {"role": "user", "content": ticket},
        ],
        tools=[TRIAGE_TOOL],
        max_tokens=300,
        temperature=0,
    )
    if len(response.tool_calls) != 1:
        raise ValueError("provider must return exactly one support-routing tool call")
    function = response.tool_calls[0].get("function")
    if not isinstance(function, dict) or function.get("name") != "route_support_ticket":
        raise ValueError("provider returned the wrong support-routing tool")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("provider returned invalid support-routing arguments")
    try:
        decision = json.loads(arguments)
    except (TypeError, ValueError) as exc:
        raise ValueError("provider returned malformed support-routing JSON") from exc
    if not isinstance(decision, dict) or set(decision) != {"queue", "urgency", "summary"}:
        raise ValueError("provider returned an invalid support-routing object")
    queue = decision["queue"]
    urgency = decision["urgency"]
    summary = decision["summary"]
    if not isinstance(queue, str) or queue not in _QUEUES:
        raise ValueError("provider returned an unsupported support-routing value")
    if not isinstance(urgency, str) or urgency not in _URGENCIES:
        raise ValueError("provider returned an unsupported support-routing value")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 240:
        raise ValueError("provider returned an invalid support-routing summary")
    return TriageDecision(
        queue=queue,
        urgency=urgency,
        summary=summary,
        provider=response.provider,
        model=response.model,
        total_tokens=response.total_tokens,
    )


async def main() -> None:
    async with build_client(os.environ["OPENAI_API_KEY"]) as client:
        decision = await triage_ticket(client, input("Support ticket: "))
    print(decision)


if __name__ == "__main__":
    asyncio.run(main())
