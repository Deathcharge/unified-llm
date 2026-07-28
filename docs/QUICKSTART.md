# Quick start

## Prerequisites

- Python 3.10 or newer
- Git for a source checkout

The project has not been published to PyPI. Install the checkout:

```bash
python -m pip install -e .
```

## Offline verification

```bash
python examples/offline_fallback.py
```

Expected output:

```text
provider=backup content=offline fallback works attempts=primary,backup
```

This uses deterministic adapters and cannot access a network.

## Live OpenAI-compatible endpoint

Set an exact model ID and credentials supplied by your provider:

```bash
export UNIFIED_LLM_API_KEY="your-key"
export UNIFIED_LLM_MODEL="your-model-id"
export UNIFIED_LLM_BASE_URL="https://your-provider.example/v1"
python examples/from_env.py
```

PowerShell equivalent:

```powershell
$env:UNIFIED_LLM_API_KEY = "your-key"
$env:UNIFIED_LLM_MODEL = "your-model-id"
$env:UNIFIED_LLM_BASE_URL = "https://your-provider.example/v1"
python examples/from_env.py
```

Omit `UNIFIED_LLM_BASE_URL` to use `https://api.openai.com/v1`. The example makes one bounded request with at most 150 output tokens. Live use may incur provider charges.

## Common failures

- `ConfigurationError`: check variable names, exact model ID, HTTPS URL, and numeric limits.
- `RequestValidationError`: fix input before retrying; no request was sent.
- `ProviderError` with 400/401/403: fix the request, model, permissions, or key. The router will not amplify the error across fallback routes.
- `FallbackExhausted`: every eligible transient route failed within the configured budget. Inspect sanitized `attempts`, then retry at the application level only if appropriate.

Continue with [API.md](API.md) for public types or [ARCHITECTURE.md](ARCHITECTURE.md) for custom adapters.
