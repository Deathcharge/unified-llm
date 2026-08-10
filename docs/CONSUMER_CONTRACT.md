# Support-triage consumer contract

## Purpose and status

`examples/support_triage.py` is the repository's canonical reference consumer. It demonstrates a production-shaped use case: classify an untrusted support ticket into a queue and urgency through one strict function call, then validate the model-supplied arguments before application use.

This is deterministic contract evidence, not a claim of live deployment or Samsarix Unified adoption. A live endpoint remains an external release gate because it requires operator-owned credentials and can incur provider charges.

## Supported public boundary

The consumer imports only names exported by `unified_llm`:

- `OpenAIResponsesProvider`
- `Route`
- `UnifiedLLM`

It does not import implementation modules, private symbols, legacy services, persistence, or account logic. The fixture targets the current `0.1.0` public contract. Until a `0.2.0` release establishes a wider compatibility promise, consumers should pin the exact prerelease version or artifact digest they verify.

## Contract

| Boundary | Consumer requirement | Evidence |
| --- | --- | --- |
| Authentication | The API key is supplied only to the provider adapter and becomes an authorization header, never JSON payload data. | `test_support_triage_public_api_contract` |
| Privacy | Responses requests set `store=false`; the router has no persistence or content logging; the observer receives only sanitized attempt metadata. Model-written summary text is validated and discarded, then replaced with a local queue/urgency template. | Exact outbound payload assertion, adversarial summary fixture, and core architecture tests |
| Request safety | Ticket and strict tool schema are included in the exact request-byte calculation. The reference client caps requests at 64,000 bytes and output at 1,000 tokens. | Router request-bound tests and reference client configuration |
| Response safety | Raw responses are capped at 256,000 bytes, normalized content at 8,000 characters, and tool calls at four. | Reference client configuration and provider-bound tests |
| Decision integrity | Exactly one named function call is required. JSON must contain only `queue`, `urgency`, and `summary`; enum values and summary length are checked after parsing. The returned summary is generated locally from validated enums. | Happy-path, adversarial-summary, and malformed-decision tests |
| Failure behavior | Configuration, validation, provider, and fallback failures remain typed SDK exceptions. Invalid business output becomes a consumer-owned `ValueError`; no empty success value is returned. | Consumer negative tests and SDK error contract |
| Metadata | The decision records normalized provider, served model, and reported token total for audit/cost attribution. | Happy-path fixture |
| Lifecycle | The reference uses `async with UnifiedLLM(...)`, closing SDK-owned HTTP resources deterministically. | `main()` and lifecycle tests |

## Verification

Run from a clean checkout:

```bash
python -m pytest tests/test_consumer_contract.py -q
python -m mypy unified_llm tests examples
python -m ruff check .
python -m ruff format --check .
```

The HTTP fixture uses `httpx.MockTransport`; it requires no credentials, sends no network traffic, and asserts the provider boundary directly.

## Live conformance gate

Before describing this consumer as live-compatible, a maintainer must run one capped request against the intended endpoint using a revocable, least-privilege key and record:

1. exact package artifact digest and Python version;
2. provider, endpoint class, and exact model identifier without recording credentials or ticket content;
3. request/response limits and whether provider-side storage was disabled;
4. returned tool-call shape, usage metadata, and typed failure behavior;
5. timestamp, maintainer, cost, and credential revocation or rotation result.

Live evidence must not contain prompts, responses, authorization headers, customer data, or secrets.

## Ownership, compatibility, and rollback

- Owner: Samsarix LLC
- Support: `support@samsarix.com`
- Commercial contact: `contact@samsarix.com`
- Compatibility: exact verified prerelease artifact until a published compatibility window exists
- Rollback: stop invoking `triage_ticket`, remove the reference-consumer integration, and pin the previously verified SDK artifact; the core router and provider adapters remain independent of this example
- Adoption signal: count successful consumer-owned contract runs and, only after live adoption, valid routed tickets versus rejected malformed decisions
