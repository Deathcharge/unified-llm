# Productization Record

Last updated: 2026-07-28

This is the living record for turning this repository into an independently useful product. It distinguishes the audited baseline from the verified release-candidate state.

## Current repository assessment (2026-08-31)

The public `main` branch is synchronized at `73f932e5c4499df5ba2e1f73175f2953910d00a6`, including merged PRs #6 through #9. The standalone SDK now includes Chat Completions and Responses adapters, content-free attempt hooks, provider-health cooldown, exact request accounting, bounded response reads, and a support-triage reference consumer. It remains an alpha library, not a deployed service or evidence of product-market fit.

The current local verification is 135 tests passing with 94.69% branch-aware coverage. The earlier assessment and verification below are historical baselines, not the present branch state. The August 10 release dry run built, validated, and attested artifacts without publishing; its result was rechecked on August 31. Live endpoint conformance, independent adoption, and owner-controlled publication setup remain unverified.

## Historical starting assessment (2026-07-28)

The local branch began as three commits that extracted one 1,395-line Python module from `helix-unified`, added a minimal `setup.py`, and added placeholder documentation. The worktree was clean at audit start. Local `main` was three commits ahead of and three commits behind `origin/main`, but the histories have no merge base. The remote branch is a separate 15,645-line monorepo extraction with no root packaging manifest, committed Python bytecode, broad unverified examples, and a README that claims production readiness. It is not a safe upstream to merge.

At baseline, the local package could be installed in editable mode but could not be imported through its advertised public API. Most provider initialization depended on private `apps.backend.*` modules from another repository, so the extracted package could not make an LLM request on its own. Documentation, packaging metadata, tests, CI, dependency declarations, and release guidance were absent or misleading.

The release-candidate branch is now a standalone Apache-2.0 package with a deliberately small public API, an OpenAI-compatible HTTP adapter, bounded routing behavior, deterministic tests, distributable artifacts, CI, and honest operational documentation. The remaining release gates require owner action: authorize a live endpoint check and the publication workflow.

Samsarix LLC is the confirmed owner and maintainer identity. General inquiries use `contact@samsarix.com`; product and security-channel requests use `support@samsarix.com`. The Git remote remains `https://github.com/Deathcharge/unified-llm.git`, so published repository and issue URLs continue to use that verified location until an actual hosting migration occurs.

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

- The initial unrelated histories were reconciled earlier with owner authorization. Subsequent improvements use focused branches and reviewed merges into public `main`; the original divergence is not an ongoing blocker.
- Keep the product a library. No frontend, hosted service, auth, database, telemetry, or cloud resources are justified.
- Provide built-in Chat Completions and Responses adapters sharing the bounded transport and router contracts. Other native provider protocols remain deferred; custom adapters use a small public protocol.
- Retain familiar `generate`, `chat`, and metadata-returning methods, but replace empty-string failure behavior with typed exceptions.
- Require explicit route configuration. Do not infer providers from model-name substrings or ship quickly stale model catalogs.
- Do not cache prompts or responses in the core. Caching has privacy, retention, and tenant-isolation implications and belongs in the host application.
- Never log API keys, prompt bodies, response bodies, or custom headers.
- Use modern `pyproject.toml` packaging and compatible runtime dependency ranges. Separately lock the artifact builder's exact versions and hashes in `requirements/release-build.txt`; release construction must not resolve a mutable backend independently.
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

- [x] The documented top-level import failed because `UnifiedLLM` did not exist. The public class and compatibility alias now import from both editable and built-wheel installs.
- [x] Standalone provider discovery depended on private `apps.backend.*` modules. Those dependencies were removed in favor of a public provider protocol and built-in HTTP adapter.
- [x] Runtime dependencies were undeclared and inconsistently optional. The only runtime dependency is now an explicit bounded `httpx` range.
- [x] There were no tests for the public package or primary request journey. The deterministic suite now covers configuration, routing, HTTP behavior, limits, lifecycle, and failure semantics without live provider calls.
- [x] Packaging metadata was insufficient for an honest distributable package. Modern metadata, typed-package markers, source/wheel builds, and metadata checks are now present.

### P1

- [x] Retries and fallback caught essentially every exception. Retries are now limited to documented transient network/status failures, while permanent errors stop immediately.
- [x] Retry, fallback, concurrency, input size, serialized request size, output tokens, timeout, and attempt budgets are now explicit and bounded.
- [x] The cross-provider Redis response cache was removed from the standalone core, eliminating its tenant/provider key ambiguity and implicit retention behavior.
- [x] Private Helix URLs and shared-secret forwarding were removed.
- [x] Stale model catalogs and substring inference were replaced by explicit ordered routes.
- [x] Empty-string failure results were replaced by typed exceptions with attempt metadata and secret-safe messages.
- [x] Documentation was rewritten against the implemented public API and makes compatibility and release limitations explicit.
- [x] Packaging now requires Python 3.10 or newer, matching the syntax and CI matrix.
- [x] CI, changelog, contribution and security guidance, package builds, metadata checks, and installed-wheel smoke testing were added.

### P2

- [ ] Native support for non-OpenAI-compatible provider protocols.
- [ ] True streaming with a documented fallback boundary.
- [x] Optional metrics hooks that exclude prompt/response content by default (`on_attempt`, plus provider-health snapshots).
- [ ] Optional cost estimation supplied by the application from versioned provider pricing.

## Implementation checklist

- [x] Replace private Helix imports with a standalone provider protocol and HTTP adapter.
- [x] Implement validated routes, normalized responses, typed errors, bounded retry/fallback, timeouts, cancellation, and concurrency.
- [x] Add environment onboarding without implicit `.env` loading.
- [x] Add deterministic unit and HTTP-contract tests.
- [x] Add package-import and built-distribution verification.
- [x] Add modern packaging, lint, typing, test, build, and CI configuration.
- [x] Add credential-free examples and a real endpoint example.
- [x] Rewrite README and focused API, architecture, and quick-start docs.
- [x] Add changelog, contribution, and security guidance.
- [x] Perform a repository-wide security scan and adversarial final review.

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
- Chose the standalone SDK product boundary and primary journey.
- Rebuilt the package around explicit routes, a public provider protocol, and a conservative OpenAI-compatible transport.
- Added validation before provider I/O, including full serialized-request byte limits, and bounded all retry, timeout, token, attempt, and concurrency behavior.
- Initially added 84 network-free tests with 98.16% coverage; see the dated current verification below for the expanded suite.
- Verified Ruff lint and formatting, strict mypy checks, source and wheel builds, Twine metadata, editable installation, built-wheel import, and the offline fallback example.
- Audited the fully resolved runtime dependency set with `pip-audit`; no known vulnerabilities were reported for that set. The unpublished local package itself was intentionally excluded from the dependency-only audit.
- The initial review reported no findings. A later August 10 standard scan identified six findings, followed by remediation changes in PR #9. That scan describes its original revision; neither a green suite nor the earlier review establishes that the current tree has no remaining vulnerabilities.
- Updated the package ownership, maintainer, support, vulnerability-reporting, and licensing-contact identity to Samsarix LLC while preserving the verified code-hosting URL.
- Selected Apache License 2.0 for the public SDK and replaced the contradictory incomplete license files with the complete standard license text and SPDX package metadata.

## Historical release-candidate verification

Executed with Python 3.11.9 on 2026-07-28:

| Command or check | Result |
| --- | --- |
| `python -m ruff check .` | Passed. |
| `python -m ruff format --check .` | Passed. |
| `python -m mypy unified_llm tests examples` | Passed. |
| `python -m pytest -q --cov=unified_llm --cov-report=term-missing` | Passed: 84 tests, 98.16% coverage. |
| `python examples/offline_fallback.py` | Passed: deterministic fallback from `primary` to `backup`. |
| `python -m build` | Passed: source distribution and wheel. |
| `python -m twine check dist/*` | Passed for both artifacts. |
| Clean virtual-environment wheel install and import | Passed: `0.1.0 UnifiedLLM`. |
| Resolved runtime dependency audit | Passed: no known vulnerabilities found. |

## Current verification and evidence gaps

Re-run on 2026-08-31 using Python 3.11.9 against `73f932e`:

| Command or evidence | Result |
| --- | --- |
| `python -m pytest -q --cov=unified_llm --cov-report=term-missing` | 135 passed, 94.69% coverage. |
| `python -m ruff check .` | Passed. |
| `python -m ruff format --check .` | Passed, 29 files. |
| `python -m mypy unified_llm tests examples` | Passed, 17 source files. |
| PR #9 merged revision | `73f932e5c4499df5ba2e1f73175f2953910d00a6`. |
| [Release dry run 31425066883](https://github.com/Deathcharge/unified-llm/actions/runs/31425066883) | Historical August 10 run rechecked: success on that revision; publication skipped. |

These results prove the tested local contracts, not real-provider compatibility or independent adoption. The manual run skipped tag-specific gates; its success does not exercise rejection of an off-main tag. The installed-wheel verification now uses `scripts/verify_wheel_consumer.py` to run the canonical consumer contract in a fresh environment outside the checkout, with SDK source excluded. The same verifier is wired into the Python 3.10–3.13 wheel matrix.

Next locally actionable work, ordered by release value:

1. Verify tag-gate rejection with a local Git fixture and refresh dependency/security evidence before publication.

Preprocessing now has direct regression coverage in `tests/test_router.py`: a waiting request cannot enter validation before admission, cancelling a waiter never validates or consumes capacity, and an oversized first route prevents construction of later payloads. Both cancellation and validation rejection are followed by successful requests through the same single-permit client. These checks establish those specific boundaries, not an aggregate memory ceiling for arbitrary nested input graphs.

The engineering disposition is **release candidate with named external gates and remaining verification work**, not production-ready.

## Deferred work and rationale

- Streaming and pricing remain P2 extensions. Content-free attempt observation is implemented; hosted telemetry collection remains outside the SDK. The Responses adapter reuses the router and transport contracts.
- A hosted proxy, UI, authentication, persistence, and billing are out of scope because they duplicate mature products and are unsupported by repository evidence.

## Owner-, credential-, legal-, or production-blocked tasks

- Live provider validation requires an owner-supplied endpoint, model, API key, and authorization to incur usage. Automated tests will use deterministic mocks instead.
- Publishing to PyPI, creating signing identities, configuring trusted publishing, and creating a GitHub release require owner authorization.
- Independent adoption is not established by the repository reference consumer. An adopter must choose an application and confirm its integration; do not modify the flagship repository to manufacture adoption evidence.

## Known risks

- OpenAI-compatible endpoints differ at protocol edges. The built-in adapter will support and test a conservative Chat Completions subset and report malformed responses explicitly.
- Retrying a failed generation can still create duplicate provider usage when a server completes work but its response is lost. Strict attempt limits and transient-only retry reduce but cannot eliminate this distributed-systems ambiguity.
- Applications can pass sensitive prompts to configured third-party endpoints. The SDK cannot determine provider retention policy; documentation must make that trust boundary explicit.

## Distribution and sustainability model

The simplest distribution is an Apache-2.0 source and wheel package published to PyPI after owner-authorized release setup, with Git tags and a changelog. The package should remain a focused open developer component. If commercial sustainability is desired, support, integration work, or a separately operated gateway can be sold without weakening or bloating the core library. No paid tier or revenue claim is justified by current evidence.
