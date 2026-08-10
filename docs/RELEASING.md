# Release process

No package has been published from this repository yet. This process makes releases deliberate, reviewable, and recoverable; merging to `main` never uploads a package.

## One-time owner setup

1. Confirm that the `unified-llm` PyPI project name is available or controlled by Samsarix LLC.
2. In PyPI, register a pending GitHub Trusted Publisher for owner `Deathcharge`, repository `unified-llm`, workflow `release.yml`, and environment `pypi`.
3. Create the GitHub `pypi` environment and require manual approval. Restrict deployment to protected version tags when repository policy permits it.
4. Protect `main` and release tags, require the CI checks, and review any change to `.github/workflows/release.yml` or `requirements/release-build.txt` as a credential-equivalent security change. Release builds intentionally do not restore dependency caches.
5. Ensure at least two Samsarix-controlled recovery methods exist for the PyPI and GitHub owner accounts.

Do not add a long-lived PyPI token to repository secrets. The workflow requests a short-lived OIDC credential only inside the `pypi` environment.

## Candidate gate

Before tagging:

1. update `pyproject.toml`, `unified_llm.__version__`, and `CHANGELOG.md` to the same version;
2. verify the exact commit with Ruff, mypy, pytest/coverage, build, Twine, and the installed-wheel smoke test;
3. record any live endpoint conformance evidence described in `docs/CONSUMER_CONTRACT.md`, or explicitly defer publication;
4. run the release workflow manually from the candidate commit and verify its build provenance and downloadable artifacts; manual runs never publish;
5. review the source distribution and wheel contents, dependency metadata, license, owner/support identity, and artifact digests.

## Publication

Create and push an annotated tag exactly matching `v` plus the package version, for example `v0.1.0`, on the current public `main` commit. The release workflow rejects a mismatched or off-main tag. It builds the source distribution and wheel with the hash-locked builder closure in `requirements/release-build.txt`, validates them in a separate unprivileged job, creates GitHub-hosted SLSA provenance in an attestation-only OIDC job, and pauses at the protected `pypi` environment. An owner must inspect the run and approve that deployment before PyPI receives anything.

The official PyPA publishing action uses Trusted Publishing and uploads PyPI attestations by default. After approval, verify the project page, both distributions, their hashes and attestations, and installation in a clean supported Python environment. Then create the GitHub release from the same tag and attach or link the verification record.

## Rollback and incident response

PyPI release files and versions are immutable. Do not overwrite or reuse a version. If a release is faulty:

1. yank the affected version on PyPI with a concise reason;
2. publish a corrected higher patch version through the same reviewed workflow;
3. mark the GitHub release and changelog clearly;
4. if provenance or publisher identity is suspect, disable the PyPI trusted publisher and GitHub environment before investigating;
5. follow `SECURITY.md` for any confidentiality or integrity issue.

Existing users can remain pinned to the last verified artifact while the correction is prepared.
