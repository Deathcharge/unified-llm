# Architecture

## Product boundary

`unified-llm` is an in-process Python library. It contains no server, authentication layer, persistence, cache, telemetry exporter, model catalog, or provider account logic. The host application owns those concerns.

## Components

### `Route`

An immutable pair of one explicitly named `Provider` and one exact model ID. Route order defines fallback order. Provider names must be unique so selecting `provider="name"` is unambiguous.

### `Provider`

A runtime-checkable protocol with a `name` and one async `complete` method. The method receives validated messages, generation parameters, a timeout, and optional tool definitions. It returns `UnifiedLLMResponse` or raises sanitized `ProviderError`.

The host can implement provider-specific authentication, translation, or test fakes without changing the router.

### `OpenAICompatibleProvider`

The built-in adapter posts the conservative text/tool request subset to `<base_url>/chat/completions` and normalizes the first choice. It rejects oversized success bodies while reading the response stream, before JSON decoding. It classifies only HTTP 408, 429, 500, 502, 503, and 504 plus transport/timeouts as retryable. Remote error bodies are not copied into package exceptions.

HTTPS is required except for loopback hosts. Trusted private HTTP must be opted into explicitly. Credentials in URLs, query strings, fragments, authorization-header overrides, and header control characters are rejected.

### `UnifiedLLM`

The router validates the exact provider payload before network I/O, including the resolved model and generation fields, then holds one concurrency permit for the logical request. It validates and bounds normalized results from built-in and custom adapters before returning them. Each route receives at most `max_attempts_per_route`; the request receives at most `max_total_attempts`. Retry delay uses bounded exponential backoff, optional jitter, and a capped `Retry-After` value.

Permanent provider failures stop immediately. Transient failures can retry and move to the next route. Cancellation propagates immediately. Unexpected custom-adapter exceptions are wrapped without copying their text.

## Trust boundaries

```text
application data and configuration (trusted host)
                  │
                  ├── messages/tools → validation and limits
                  │
                  └── endpoint/key/model → provider configuration
                                      │
                                      ▼
                            third-party provider
                                      │
                                      ▼
                       normalized non-persistent response
```

- Endpoint URLs, keys, model IDs, provider adapters, and custom headers are operator-controlled configuration.
- Message/tool content may be untrusted end-user data, but it is serialized as JSON rather than evaluated or used to construct paths/commands.
- The configured endpoint receives prompts and may have its own retention, training, or logging behavior.
- The package retains no prompt, response, key, or usage database.

## Resource and cost bounds

Defaults are 30 seconds per attempt, two attempts per route, four attempts total, ten concurrent logical requests, 200,000 message-content characters, 1,000,000 exact serialized request bytes, 2,000,000 normalized response bytes, 1,000,000 response characters, 128 tool calls, 32,768 output tokens, and a five-second maximum retry delay. The built-in HTTP adapter also caps raw success bodies at 2,000,000 bytes. Constructor validation sets absolute safety ceilings.

Retries are not perfectly idempotent: a provider can finish generation while the response is lost. The bounded attempt count limits amplification; applications with stricter budgets should use one attempt per route and lower token limits.

## Lifecycle

Use `async with UnifiedLLM(...)` or call `aclose()`. The built-in adapter closes only HTTP clients that it created. A caller-supplied `httpx.AsyncClient` remains caller-owned, enabling connection sharing and deterministic mock transports.

## Deliberate exclusions

- Response caching: privacy, retention, identity, and cache-key policy belong to the application.
- Global singleton: import-time environment discovery obscures configuration and resource lifecycle.
- Model-name inference and bundled model lists: provider identifiers and availability change independently of package releases.
- True streaming: fallback after bytes have been yielded needs a separate event and partial-output contract.
- Pricing: provider rates are temporal data; applications can calculate cost from normalized usage and their own versioned rate table.
