from __future__ import annotations

import re
from pathlib import Path

import unified_llm

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_package_version_is_consistent() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    assert match.group(1) == unified_llm.__version__
    assert f"## {unified_llm.__version__} -" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_all_third_party_workflow_actions_are_commit_pinned() -> None:
    for workflow in WORKFLOWS.glob("*.yml"):
        source = workflow.read_text(encoding="utf-8")
        actions = re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)", source, flags=re.MULTILINE)
        assert actions, f"{workflow.name} must contain at least one action"
        for action in actions:
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action), f"mutable action reference in {workflow.name}: {action}"


def test_release_workflow_uses_tokenless_approved_publication() -> None:
    source = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")

    assert "github.event_name == 'push' && github.ref_type == 'tag'" in source
    assert re.search(r"environment:\s+name: pypi", source)
    assert "id-token: write" in source
    assert "pypa/gh-action-pypi-publish@" in source
    assert "actions/attest@" in source
    assert "python -m pytest --cov=unified_llm" in source
    assert "password:" not in source
    assert "secrets." not in source
