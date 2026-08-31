"""Execute the actual workflow's main-commit gate against disposable Git remotes."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def git(directory: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()


@pytest.mark.parametrize("candidate", ["current", "stale", "off-main", "missing-sha", "missing-remote"])
def test_release_main_gate(tmp_path: Path, candidate: str) -> None:
    remote = tmp_path / "origin.git"
    remote.mkdir()
    git(remote, "init", "--bare")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "-b", "main")
    git(checkout, "config", "user.name", "Release gate fixture")
    git(checkout, "config", "user.email", "fixture@example.invalid")
    git(checkout, "config", "commit.gpgsign", "false")
    git(checkout, "commit", "--allow-empty", "-m", "initial")
    initial = git(checkout, "rev-parse", "HEAD")
    git(checkout, "remote", "add", "origin", str(remote))
    git(checkout, "push", "origin", "main")
    # Populate FETCH_HEAD first: a later failed fetch must not reuse stale data.
    git(checkout, "fetch", "--no-tags", "origin", "main")
    sha = initial
    if candidate == "stale":
        git(checkout, "commit", "--allow-empty", "-m", "new approved main")
        git(checkout, "push", "origin", "main")
    elif candidate == "off-main":
        git(checkout, "switch", "-c", "candidate")
        git(checkout, "commit", "--allow-empty", "-m", "unreviewed candidate")
        sha = git(checkout, "rev-parse", "HEAD")
    elif candidate == "missing-sha":
        sha = ""
    elif candidate == "missing-remote":
        git(checkout, "remote", "set-url", "origin", str(tmp_path / "absent.git"))

    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    section = workflow.split("- name: Require the tag to identify the current public main commit\n", 1)[1]
    section = section.split("      - name:", 1)[0]
    assert "if: github.ref_type == 'tag'" in section
    script = "\n".join(line[10:] for line in section.split("run: |\n", 1)[1].splitlines() if line.strip())
    git_executable = shutil.which("git")
    assert git_executable is not None
    bash = str(Path(git_executable).resolve().parent.parent / "bin/bash.exe") if os.name == "nt" else "bash"
    environment = os.environ.copy()
    environment["GITHUB_SHA"] = sha
    # Match GitHub's fail-fast Bash execution, including pipeline failure.
    result = subprocess.run(
        [bash, "--noprofile", "--norc", "-eo", "pipefail", "-c", script],
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert (result.returncode == 0) == (candidate == "current"), result.stderr
