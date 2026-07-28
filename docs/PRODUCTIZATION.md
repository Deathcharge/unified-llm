# Productization Record

Last updated: 2026-07-28

This is the living record for turning this repository into an independently useful product. It distinguishes observed behavior from intended behavior and is updated as implementation and verification progress.

## Current repository assessment

The local branch began as three commits that extracted one 1,395-line Python module from `helix-unified`, added a minimal `setup.py`, and added placeholder documentation. The worktree was clean at audit start. Local `main` was three commits ahead of and three commits behind `origin/main`, but the histories have no merge base. The remote branch is a separate 15,645-line monorepo extraction with no root packaging manifest, committed Python bytecode, broad unverified examples, and a README that claims production readiness. It is not a safe upstream to merge.

The local package can be installed in editable mode, but it cannot be imported through its advertised public API. Most provider initialization depends on private `apps.backend.*` modules from another repository, so the extracted package cannot make an LLM request on its own. Documentation, packaging metadata, tests, CI, dependency declarations, and release guidance are absent or misleading.

## Chosen product definition

`unified-llm` will be a small, typed, asynchronous Python SDK for routing text/chat requests across an ordered set of OpenAI-compatible endpoints or user-supplied provider adapters.

Its distinguishing value is a narrow embeddable reliability layer:

- one stable response and error contract;
- explicit provider/model routes instead of fragile model-name inference;
- bounded concurrency, timeouts, retries, and failover;
- retries only for transient failures;
- deterministic fakes and HTTP transports for tests;
- no dependency on Helix services, Redis, databases, authentication systems, or deployment infrastructure.

This is deliberately smaller than broad gateways such as LiteLLM, which supports 100+ providers, a proxy server, budgets, and observability. OpenRouter already provides a hosted OpenAI-compatible aggregation endpoint with its own provider fallback. This package is for applications that need a compact in-process policy they own.

## Target user and primary use case

The target user is a Python application developer who uses one or more OpenAI-compatible inference endpoints and wants predictable local routing without operating a gateway.

Primary journey:

1. Install the package.
2. Configure one endpoint from environment variables or construct explicit providers and routes.
3. Make an async generation/chat request.
4. Receive normalized text, model, provider, token usage, finish reason, and attempt metadata.
5. On a transient failure, retry within a strict budget and fall back in route order; on invalid input or permanent provider errors, fail clearly without amplifying spend.

## Key product and architecture decisions

- Preserve the current local history on `codex/productize-unified-llm`; do not merge the unrelated `origin/main` history.
- Keep the product a library. No frontend, hosted service, auth, database, telemetry, or cloud resources are justified.
- Provide a built-in OpenAI Chat Completions-compatible HTTP adapter because that protocol is also supported by endpoints such as OpenRouter. Keep provider-specific protocols out of the first release; custom adapters use a small public protocol.
- Retain familiar `generate`, `chat`, and metadata-returning methods, but replace empty-string failure behavior with typed exceptions.
- Require explicit route configuration. Do not infer providers from model-name substrings or ship quickly stale model catalogs.
- Do not cache prompts or responses in the core. Caching has privacy, retention, and tenant-isolation implications and belongs in the host application.
- Never log API keys, prompt bodies, response bodies, or custom headers.
- Use modern `pyproject.toml` packaging. A library does not need an application lockfile; supported dependency ranges and CI across supported Python versions are the distribution contract.
- Reset maturity to `0.1.0`. The previous `1.0.0` label was not supported by a working import or published release evidence.

## Assumptions

- The repository owner wants an independently installable package rather than another Helix monorepo snapshot.
- An endpoint base URL and API key are trusted operator configuration, not end-user input.
- The first credible release can focus on non-streaming text/chat completions. True token streaming and native provider-specific schemas remain separate additions.
- The package has not been published under the `unified-llm` name; the PyPI project URL returned 404 during the 2026-07-28 audit. Availability must be checked again immediately before any owner-authorized publication.

## Baseline command results

Executed from the clean local `main` tree at commit `3e434d1` with Python 3.11.9:

| Command | Result |
| --- | --- |
| `python --version` | Passed: `Python 3.11.9`. |
| `python -m pip install -e . --no-build-isolation` | Passed; built and installed `unified-llm==1.0.0`. Pip also reported an unrelated invalid local `~andit` distribution warning. |
| `python -c "import unified_llm; print(unified_llm.__version__)"` | Failed: `ImportError`; `unified_llm.__init__` imports nonexistent `UnifiedLLM`. |
| `python -m pytest -q` | Failed: no tests collected (exit 1). |
| `python -m ruff check .` | Passed under the ambient Ruff configuration; the repository had no Ruff configuration of its own. |
| `python -m mypy unified_llm` | Failed with 13 errors, including missing private `apps.backend.*` modules, the nonexistent public class, unsafe registry typing, and an invalid structured response assumption. |
| `python -m build` | Failed because the active environment does not provide the PyPA `build` module (`build.__main__` is missing). |

No lint, type-check, test, build, start, or CI scripts were defined by the repository itself.

## Findings

### P0

- [ ] The documented top-level import fails because `UnifiedLLM` does not exist.
- [ ] Standalone provider discovery never creates usable providers without private `apps.backend.*` modules from `helix-unified`.
- [ ] Runtime dependencies (`aiohttp`, provider SDKs, Pydantic, Redis integration) are undeclared and inconsistently optional.
- [ ] There are no tests for the public package or primary request journey.
- [ ] Packaging metadata is insufficient for an honest distributable package.

### P1

- [ ] Retries and fallback catch essentially every exception, including permanent request/authentication failures, which can duplicate cost and hide configuration errors.
- [ ] Retry, fallback, concurrency, input size, and total request budgets are not bounded coherently.
- [ ] A Redis response-cache key omits provider and user identity while BYOT provider selection is supported, creating privacy and isolation risk.
- [ ] Environment-provided Helix service URLs can receive a shared internal secret; this private-infrastructure coupling does not belong in a standalone package.
- [ ] Provider and model selection relies on stale hard-coded model catalogs and model-name substring inference.
- [ ] Errors are converted to empty strings in convenience methods, making operational recovery ambiguous.
- [ ] Documentation contains nonexistent classes, methods, examples, directories, performance claims, and installation assumptions.
- [ ] The code uses Python 3.10 union syntax while packaging claims Python 3.9 support.
- [ ] There is no CI, changelog, security guidance, or installed-wheel verification.

### P2

- [ ] Native support for non-OpenAI-compatible provider protocols.
- [ ] True streaming with a documented fallback boundary.
- [ ] Optional metrics hooks that exclude prompt/response content by default.
- [ ] Optional cost estimation supplied by the application from versioned provider pricing.

## Implementation checklist

- [ ] Replace private Helix imports with a standalone provider protocol and HTTP adapter.
- [ ] Implement validated routes, normalized responses, typed errors, bounded retry/fallback, timeouts, cancellation, and concurrency.
- [ ] Add environment onboarding without implicit `.env` loading.
- [ ] Add deterministic unit and HTTP-contract tests.
- [ ] Add package-import and built-distribution verification.
- [ ] Add modern packaging, lint, typing, test, build, and CI configuration.
- [ ] Add credential-free examples and a real endpoint example.
- [ ] Rewrite README and focused API, architecture, and quick-start docs.
- [ ] Add changelog, contribution, and security guidance.
- [ ] Perform a repository-wide security scan and adversarial final review.

## Release acceptance criteria

- A clean install exposes the documented public API.
- The offline example reproduces routing and fallback without credentials or network access.
- An OpenAI-compatible request can be reproduced with documented environment variables.
- Invalid input, missing configuration, transient exhaustion, permanent provider failure, cancellation, and concurrency limits have tested behavior.
- Tests never contact paid APIs and cannot consume tokens.
- Ruff, mypy, pytest, coverage, package build, metadata check, and installed-wheel smoke tests pass.
- CI runs the meaningful checks on supported Python versions.
- Documentation contains no unverified performance, provider-count, or production-readiness claims.
- No locally actionable P0 remains.

## Completed work

- Protected the clean initial work on a dedicated branch.
- Audited local and remote histories and rejected an unsafe unrelated-history merge.
- Recorded baseline command outcomes.
- Performed bounded ecosystem research using current official OpenAI API, LiteLLM, OpenRouter, and Python Packaging documentation.
- Chosen the standalone SDK product boundary and primary journey.

## Deferred work and rationale

- Provider-specific APIs, streaming, telemetry, and pricing are P2 extensions. Adding them now would broaden the public contract before the core route/failure semantics are proven.
- A hosted proxy, UI, authentication, persistence, and billing are out of scope because they duplicate mature products and are unsupported by repository evidence.

## Owner-, credential-, legal-, or production-blocked tasks

- The repository contains a one-line `LICENSE` claiming Apache 2.0 and a separate proprietary notice. The owner must choose and install a complete, legally reviewed license before public release. Engineering must not infer or replace it.
- Live provider validation requires an owner-supplied endpoint, model, API key, and authorization to incur usage. Automated tests will use deterministic mocks instead.
- Publishing to PyPI, creating signing identities, configuring trusted publishing, and creating a GitHub release require owner authorization.

## Known risks

- OpenAI-compatible endpoints differ at protocol edges. The built-in adapter will support and test a conservative Chat Completions subset and report malformed responses explicitly.
- Retrying a failed generation can still create duplicate provider usage when a server completes work but its response is lost. Strict attempt limits and transient-only retry reduce but cannot eliminate this distributed-systems ambiguity.
- Applications can pass sensitive prompts to configured third-party endpoints. The SDK cannot determine provider retention policy; documentation must make that trust boundary explicit.

## Distribution and sustainability model

The simplest distribution is a source and wheel package published to PyPI after the license gate, with Git tags and a changelog. The package should remain a focused open developer component if the owner selects an open-source license. If commercial sustainability is desired, support, integration work, or a separately operated gateway can be sold without weakening or bloating the core library. No paid tier or revenue claim is justified by current evidence.
