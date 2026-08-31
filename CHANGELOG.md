# Changelog

All notable changes to this project will be documented here. The project follows semantic versioning after its first public release.

## Unreleased

### Added

- Isolated installed-wheel verification of the canonical support-triage consumer, shared by local checks and the supported-Python CI matrix.
- Configurable raw and normalized response byte limits, response character limits, and tool-call count limits.
- Runtime validation for normalized responses returned by every provider adapter.
- A stateless-by-default OpenAI Responses API adapter with text, refusal, function-call, and tool-result normalization.
- Content-free sync/async attempt observation and inspectable provider health snapshots.
- Cross-request transient-failure cooldown that deprioritizes unhealthy routes without removing last-resort fallback.
- A canonical support-ticket triage reference consumer and deterministic public-API contract fixture.
- Cross-version installed-wheel smoke tests, supply-chain contract tests, and an attested PyPI Trusted Publishing workflow.
- Commit-pinned, Node 24-native GitHub Actions for checkout and supported-Python setup.

### Changed

- Request byte accounting now measures each exact outbound provider body, including its resolved model and generation fields.
- Request validation and one-at-a-time route serialization now run inside the configured concurrency boundary.
- The built-in HTTP adapter now streams successful response bodies into a bounded buffer before JSON decoding.
- Built-in provider-specific payload builders now let the router account for exact Chat Completions and Responses request bodies.
- Sanitized adapter failures no longer retain secret-bearing exception causes or contexts.
- Release artifacts use a hash-locked builder closure, a current-main tag gate, and an isolated attestation-only OIDC job.
- Runnable provider examples represent untrusted text safely before writing it to a terminal.

## 0.1.0 - 2026-07-28

### Added

- Standalone async routing core with explicit provider/model routes.
- OpenAI Chat Completions-compatible HTTP adapter.
- Normalized response, usage, tool-call, and attempt metadata.
- Bounded transient retries, ordered fallback, concurrency, timeouts, and request sizes.
- Full serialized-request limits covering tool schemas and extra message fields.
- Typed configuration, validation, provider, and exhaustion errors.
- Environment configuration, deterministic examples, tests, CI, and release documentation.

### Removed

- Legacy private-monorepo imports, Redis cache coupling, BYOT/account logic, hard-coded model catalogs, and private service URLs from the standalone package.

### Changed

- Reset the unpublished package maturity from unsupported `1.0.0` metadata to honest alpha `0.1.0`.
- Convenience methods now raise typed errors instead of returning empty strings.
- Updated ownership, maintainer, support, security-reporting, and licensing-contact identity to Samsarix LLC.
- Replaced contradictory incomplete license notices with a complete Apache License 2.0 grant.
