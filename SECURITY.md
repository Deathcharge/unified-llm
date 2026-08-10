# Security policy

## Supported versions

No public release is currently supported. Security fixes target the latest repository revision until the first package release.

## Reporting a vulnerability

Use the repository's private GitHub vulnerability-reporting flow when available. If it is unavailable, email `support@samsarix.com` with `[SECURITY]` in the subject and a minimal, non-sensitive summary requesting a private reporting channel. Do not include exploit details, credentials, prompt content, provider responses, or customer data in a public issue or initial email.

Include the affected version or commit, the smallest reproduction, impact, preconditions, and whether a provider account was charged. Use synthetic data and revoke any credential that may have been exposed.

Samsarix LLC does not publish a response-time or remediation-time service-level agreement for this prerelease package. Receipt and next steps will be communicated through the private channel.

## Security model

- Endpoint URLs, API keys, model IDs, custom headers, and custom provider adapters are trusted operator configuration.
- Message and tool content can be untrusted and crosses the configured provider boundary as JSON.
- Normalized provider text and metadata remain untrusted output; terminal and log consumers must encode control characters.
- The core stores no prompts, responses, keys, usage records, or telemetry.
- Remote provider retention, training, authorization, quotas, and billing remain operator responsibilities.
- Retries are bounded and transient-only, but a lost response can still duplicate provider work.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed trust boundaries and controls.
