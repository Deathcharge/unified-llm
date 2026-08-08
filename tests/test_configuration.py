from __future__ import annotations

import re
from pathlib import Path

import pytest

import unified_llm
from unified_llm import ConfigurationError, UnifiedLLM

ROOT = Path(__file__).parents[1]


def test_public_version_matches_package_metadata() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"$', metadata, re.MULTILINE)
    assert version is not None
    assert unified_llm.__version__ == version.group(1)
    assert unified_llm.UnifiedLLMService is unified_llm.UnifiedLLM


def test_from_env_builds_a_local_route_without_a_key() -> None:
    client = UnifiedLLM.from_env(
        environ={
            "UNIFIED_LLM_MODEL": "local-model",
            "UNIFIED_LLM_BASE_URL": "http://localhost:11434/v1/",
            "UNIFIED_LLM_PROVIDER": "local",
            "UNIFIED_LLM_TIMEOUT": "5",
            "UNIFIED_LLM_MAX_ATTEMPTS_PER_ROUTE": "1",
            "UNIFIED_LLM_MAX_TOTAL_ATTEMPTS": "1",
            "UNIFIED_LLM_MAX_CONCURRENCY": "2",
            "UNIFIED_LLM_HEALTH_FAILURE_THRESHOLD": "4",
            "UNIFIED_LLM_HEALTH_COOLDOWN": "12.5",
        }
    )
    assert client.get_available_providers() == ["local"]
    assert client.routes[0].model == "local-model"
    assert client.request_timeout == 5
    assert client.health_failure_threshold == 4
    assert client.health_cooldown == 12.5


def test_from_env_requires_model_and_default_endpoint_key() -> None:
    with pytest.raises(ConfigurationError, match="MODEL"):
        UnifiedLLM.from_env(environ={})
    with pytest.raises(ConfigurationError, match="API_KEY"):
        UnifiedLLM.from_env(environ={"UNIFIED_LLM_MODEL": "m"})


@pytest.mark.parametrize(
    ("environment", "match"),
    [
        ({"UNIFIED_LLM_MODEL": "m", "UNIFIED_LLM_API_KEY": "k", "UNIFIED_LLM_TIMEOUT": "x"}, "number"),
        (
            {"UNIFIED_LLM_MODEL": "m", "UNIFIED_LLM_API_KEY": "k", "UNIFIED_LLM_MAX_CONCURRENCY": "x"},
            "integer",
        ),
        (
            {"UNIFIED_LLM_MODEL": "m", "UNIFIED_LLM_API_KEY": "k", "UNIFIED_LLM_HEALTH_COOLDOWN": "x"},
            "number",
        ),
        (
            {"UNIFIED_LLM_MODEL": "m", "UNIFIED_LLM_API_KEY": "k", "UNIFIED_LLM_ALLOW_INSECURE_HTTP": "x"},
            "boolean",
        ),
    ],
)
def test_from_env_reports_setting_names_not_values(environment: dict[str, str], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match) as caught:
        UnifiedLLM.from_env(environ=environment)
    assert "API_KEY" not in str(caught.value)
    assert " k" not in str(caught.value)


def test_environment_prefix_is_restricted() -> None:
    with pytest.raises(ConfigurationError, match="prefix"):
        UnifiedLLM.from_env("bad-prefix", environ={})


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_environment_boolean_values(value: str, expected: bool) -> None:
    environment = {
        "UNIFIED_LLM_MODEL": "m",
        "UNIFIED_LLM_BASE_URL": "http://private.internal/v1",
        "UNIFIED_LLM_ALLOW_INSECURE_HTTP": value,
    }
    if expected:
        client = UnifiedLLM.from_env(environ=environment)
        assert client.routes[0].provider.name == "primary"
    else:
        with pytest.raises(ConfigurationError, match="Plain HTTP"):
            UnifiedLLM.from_env(environ=environment)
