# unified-llm

`unified-llm` is a small async Python SDK that routes text and chat completions across an ordered set of OpenAI-compatible endpoints or custom provider adapters.

It is for application developers who want an in-process reliability boundary—validated inputs, normalized responses, timeouts, bounded retries, fallback, cancellation, and concurrency control—without deploying a gateway. It is not a hosted service or a broad provider-translation framework.

Maintained by **Samsarix LLC**.

> **Maturity: alpha (`0.1.0`).** The core route and failure behavior is implemented and tested without paid APIs. Live-provider validation and public release are gated as described in [Release status](#release-status).

## What it does

- Routes async chat or prompt requests through explicit provider/model pairs.
- Retries only transient transport, timeout, rate-limit, and selected server failures.
- Falls back in declared order under a strict total-attempt budget.
- Bounds request time, concurrency, input characters, serialized request bytes, output tokens, retry count, and retry delay.
- Normalizes content, model, provider, usage, finish reason, tool calls, and sanitized attempt metadata.
- Includes an OpenAI Chat Completions-compatible HTTP adapter and a small protocol for custom adapters.
- Never logs or includes API keys, headers, prompt bodies, or response bodies in its own errors.

It deliberately does not provide a proxy server, model catalog, database, Redis cache, authentication, billing, telemetry, native provider-specific schemas, or automatic model-name inference.

## Fastest evaluation: no key and no network

Prerequisite: Python 3.10 or newer.

```bash
git clone https://github.com/Deathcharge/unified-llm.git
cd unified-llm
python -m pip install -e .
python examples/offline_fallback.py
```

Expected output:

```text
provider=backup content=offline fallback works attempts=primary,backup
```

The demo uses two deterministic in-process adapters. It cannot contact a provider or incur cost.

## Use an OpenAI-compatible endpoint

`from_env()` reads environment variables; it does not load `.env` files.

```bash
export UNIFIED_LLM_API_KEY="your-key"
export UNIFIED_LLM_MODEL="your-provider-model-id"
# Optional; defaults to https://api.openai.com/v1
export UNIFIED_LLM_BASE_URL="https://your-provider.example/v1"
```

PowerShell uses `$env:UNIFIED_LLM_API_KEY = "your-key"` and the equivalent assignments for the other variables. Then run:

```python
import asyncio

from unified_llm import UnifiedLLM


async def main() -> None:
    async with UnifiedLLM.from_env() as llm:
        response = await llm.generate_with_metadata(
            "Explain bounded retries in two sentences.",
            max_tokens=150,
            temperature=0.2,
        )
        print(response.content)
        print(response.provider, response.model, response.total_tokens)


asyncio.run(main())
```

The complete example is [examples/from_env.py](examples/from_env.py). The default OpenAI URL requires `UNIFIED_LLM_API_KEY`. Localhost HTTP endpoints may omit a key; remote plain-HTTP endpoints are rejected unless explicitly trusted in programmatic configuration.

## Configure ordered fallback

```python
import asyncio
import os

from unified_llm import OpenAICompatibleProvider, Route, UnifiedLLM


async def main() -> None:
    routes = [
        Route(
            OpenAICompatibleProvider(
                name="primary",
                base_url="https://primary.example/v1",
                api_key=os.environ["PRIMARY_LLM_API_KEY"],
            ),
            model="primary-model-id",
        ),
        Route(
            OpenAICompatibleProvider(
                name="backup",
                base_url="https://backup.example/v1",
                api_key=os.environ["BACKUP_LLM_API_KEY"],
            ),
            model="backup-model-id",
        ),
    ]

    async with UnifiedLLM(
        routes,
        request_timeout=30,
        max_attempts_per_route=2,
        max_total_attempts=3,
        max_concurrency=10,
    ) as llm:
        print(await llm.generate("Summarize this request."))


asyncio.run(main())
```

Automatic routing uses every route in order. Passing `provider="backup"` selects one route and disables automatic fallback. A model override with multiple routes also requires an explicit provider, preventing accidental cross-provider model dispatch.

## Failure contract

The SDK raises instead of returning an ambiguous empty string:

- `ConfigurationError`: invalid providers, routes, URLs, or environment settings.
- `RequestValidationError`: invalid messages or request bounds; no network call occurs.
- `ProviderError`: a permanent provider/adapter failure; automatic fallback stops.
- `FallbackExhausted`: every eligible transient route failed within the attempt budget.

`asyncio.CancelledError` is propagated immediately and is never retried. Each provider attempt is available as sanitized `Attempt` metadata on successful responses or failure exceptions.

## Configuration

`UnifiedLLM.from_env()` supports:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `UNIFIED_LLM_MODEL` | Yes | — | Exact provider model ID. |
| `UNIFIED_LLM_API_KEY` | For default OpenAI URL | — | Bearer credential. |
| `UNIFIED_LLM_BASE_URL` | No | `https://api.openai.com/v1` | API root; `/chat/completions` is appended. |
| `UNIFIED_LLM_PROVIDER` | No | `primary` | Stable local route name. |
| `UNIFIED_LLM_TIMEOUT` | No | `30` | Per-attempt seconds, maximum 600. |
| `UNIFIED_LLM_MAX_ATTEMPTS_PER_ROUTE` | No | `2` | Attempts per route, 1–5. |
| `UNIFIED_LLM_MAX_TOTAL_ATTEMPTS` | No | `4` | Total request attempts, 1–16. |
| `UNIFIED_LLM_MAX_CONCURRENCY` | No | `10` | In-flight requests, 1–1000. |
| `UNIFIED_LLM_ALLOW_INSECURE_HTTP` | No | `false` | Explicitly allow trusted remote plain HTTP. |

See [.env.example](.env.example). Programmatic construction additionally controls input/output limits, retry delay, and custom non-auth headers.

## Development and verification

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m mypy unified_llm tests examples
python -m pytest --cov=unified_llm --cov-report=term-missing
python -m build
python -m twine check dist/*
```

Tests use fakes and `httpx.MockTransport`; they do not need credentials and do not call external APIs. CI runs lint, type checking, tests, package build, metadata checks, and a wheel smoke test. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Architecture

The public flow is intentionally short:

```text
validated messages
      ↓
ordered Route(provider, model) values
      ↓
bounded retry/fallback policy
      ↓
Provider.complete protocol
      ↓
normalized UnifiedLLMResponse or typed exception
```

The built-in HTTP providers own API translation. `UnifiedLLM` owns policy and has no dependency on legacy private services. Custom providers implement one async `complete` method. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/API.md](docs/API.md).

For new OpenAI integrations, use `OpenAIResponsesProvider`. It targets `/responses`, keeps server-side storage disabled by default, and participates in the same retry, fallback, request-size, and response-size boundaries. See [examples/responses_api.py](examples/responses_api.py).

## Security, privacy, reliability, and cost

- Treat endpoint URLs, API keys, models, and custom provider adapters as trusted operator configuration.
- Prompt and response content crosses the configured provider boundary. Review that provider's retention and training policy before sending sensitive data.
- The core performs no caching, persistence, telemetry, or prompt/response logging.
- Automatic retry can duplicate provider work if a response is lost after generation. Keep attempt and token limits conservative for your workload.
- Exact outbound payloads and normalized inbound responses are byte-bounded. The built-in HTTP adapter also caps raw success bodies before JSON decoding.
- The SDK reports token usage when the endpoint supplies it but does not estimate money. Operating cost is the sum over attempts of provider-reported input/output tokens multiplied by that provider/model's current rates.
- Custom adapters must sanitize their own `ProviderError` values and honor the supplied timeout; their normalized results are validated and bounded by the router.

Report vulnerabilities using [SECURITY.md](SECURITY.md).

## Support and contact

- Usage and product support: `support@samsarix.com`
- General and commercial inquiries: `contact@samsarix.com`
- Bugs and feature requests that are safe to discuss publicly: the [repository issue tracker](https://github.com/Deathcharge/unified-llm/issues)
- Vulnerabilities: follow [SECURITY.md](SECURITY.md), not the public issue tracker

See [SUPPORT.md](SUPPORT.md) for the information to include and the prerelease support boundary. Never send API keys, authorization headers, prompts, provider responses, or customer data in an issue or initial email.

## Limitations

- The built-in adapter targets the conservative text/tool subset of `/chat/completions`; provider extensions and multimodal input are not normalized.
- Native Anthropic Messages, Google GenAI, embeddings, images, audio, stateful Responses conversations, built-in hosted tools, and true token streaming are not included in `0.1.0`.
- Fallback happens only for transient failures. Authentication, invalid request, and malformed response failures stop immediately.
- Endpoint conformance is verified with deterministic mocks. Live provider compatibility depends on credentials and is an external release gate.

## Release status

The repository is a coherent release candidate for local evaluation. Public package publication remains blocked on:

1. owner-authorized live endpoint smoke tests;
2. owner authorization and configuration for PyPI trusted publishing/signing.

The package name was not present on PyPI when checked on 2026-07-28; re-check immediately before publishing. No package has been published or production infrastructure changed by this work.

See [docs/PRODUCTIZATION.md](docs/PRODUCTIZATION.md) for the evidence, remaining work, and acceptance criteria.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes to retry classification, provider protocol fields, or public exceptions are API changes and require tests plus changelog entries.

## License

Copyright (c) 2024-2026 Samsarix LLC.

Licensed under the [Apache License 2.0](LICENSE). It permits commercial use, modification, and redistribution subject to its terms. The license does not grant rights to Samsarix LLC trademarks.
