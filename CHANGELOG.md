# Changelog

All notable changes to this project will be documented here. The project follows semantic versioning after its first public release.

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
