# Security policy

## Supported versions

No public release is currently supported. Security fixes target the latest repository revision until the first package release.

## Reporting a vulnerability

Use the repository's private GitHub vulnerability-reporting flow when available. If it is unavailable, open a minimal issue asking the maintainer for a private reporting channel; do not include exploit details, credentials, prompt content, provider responses, or customer data in a public issue.

Include the affected version or commit, the smallest reproduction, impact, preconditions, and whether a provider account was charged. Use synthetic data and revoke any credential that may have been exposed.

## Security model

- Endpoint URLs, API keys, model IDs, custom headers, and custom provider adapters are trusted operator configuration.
- Message and tool content can be untrusted and crosses the configured provider boundary as JSON.
- The core stores no prompts, responses, keys, usage records, or telemetry.
- Remote provider retention, training, authorization, quotas, and billing remain operator responsibilities.
- Retries are bounded and transient-only, but a lost response can still duplicate provider work.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed trust boundaries and controls.
