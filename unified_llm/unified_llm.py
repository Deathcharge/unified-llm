"""A small, bounded router for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Protocol, TypeAlias, runtime_checkable
from urllib.parse import urlsplit

import httpx

Message: TypeAlias = Mapping[str, Any]
ToolDefinition: TypeAlias = Mapping[str, Any]

_ALLOWED_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_SAFE_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_ENV_PREFIX_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class UnifiedLLMError(Exception):
    """Base exception for all public package errors."""


class ConfigurationError(UnifiedLLMError):
    """Raised when providers, routes, or environment settings are invalid."""


class RequestValidationError(UnifiedLLMError, ValueError):
    """Raised before network I/O when a request is outside configured bounds."""


@dataclass(frozen=True, slots=True)
class Attempt:
    """Sanitized metadata for one provider attempt."""

    provider: str
    model: str
    number: int
    latency_ms: float
    error: str | None = None
    retryable: bool = False


class ProviderError(UnifiedLLMError):
    """A sanitized provider or transport failure.

    Prompt bodies, response bodies, API keys, custom headers, and endpoint URLs
    are intentionally excluded from the exception contract.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        attempts: Sequence[Attempt] = (),
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts = tuple(attempts)


class FallbackExhausted(UnifiedLLMError):
    """Raised after the configured transient-failure budget is exhausted."""

    def __init__(self, attempts: Sequence[Attempt]) -> None:
        self.attempts = tuple(attempts)
        providers = ", ".join(dict.fromkeys(item.provider for item in attempts))
        super().__init__(
            f"All eligible LLM routes failed after {len(attempts)} attempt(s)"
            + (f": {providers}." if providers else ".")
        )


@dataclass(frozen=True, slots=True)
class UnifiedLLMResponse:
    """Normalized response returned by every provider adapter."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    tool_calls: tuple[dict[str, Any], ...] = ()
    attempts: tuple[Attempt, ...] = ()

    @property
    def total_tokens(self) -> int:
        """Return reported total tokens, or derive them from input/output counts."""

        if "total_tokens" in self.usage:
            return self.usage["total_tokens"]
        input_tokens = self.usage.get("prompt_tokens", self.usage.get("input_tokens", 0))
        output_tokens = self.usage.get("completion_tokens", self.usage.get("output_tokens", 0))
        return input_tokens + output_tokens


@runtime_checkable
class Provider(Protocol):
    """Minimal protocol implemented by built-in and application providers."""

    name: str

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> UnifiedLLMResponse:
        """Return a normalized completion or raise :class:`ProviderError`."""


@dataclass(frozen=True, slots=True)
class Route:
    """One provider/model destination in fallback order."""

    provider: Provider
    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigurationError("A route model must be a non-empty string.")
        if not isinstance(self.provider, Provider):
            raise ConfigurationError("A route provider must implement the Provider protocol.")
        if not self.provider.name or not self.provider.name.strip():
            raise ConfigurationError("A provider name must be a non-empty string.")


def _validate_base_url(base_url: str, *, allow_insecure_http: bool) -> str:
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError("The provider base URL is invalid.") from exc

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigurationError("The provider base URL must use http or https and include a host.")
    if parsed.username or parsed.password:
        raise ConfigurationError("Credentials must not be embedded in the provider base URL.")
    if parsed.query or parsed.fragment:
        raise ConfigurationError("The provider base URL must not contain a query or fragment.")
    if parsed.scheme == "http" and parsed.hostname not in _SAFE_LOCAL_HOSTS and not allow_insecure_http:
        raise ConfigurationError(
            "Plain HTTP is allowed only for localhost. Pass allow_insecure_http=True "
            "only for a trusted private endpoint."
        )

    authority = parsed.hostname
    if ":" in authority and not authority.startswith("["):
        authority = f"[{authority}]"
    if port is not None:
        authority = f"{authority}:{port}"
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{authority}{path}"


def _validate_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in (headers or {}).items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ConfigurationError("Provider header names and values must be strings.")
        lowered = name.casefold()
        if lowered in {"authorization", "content-type", "host"}:
            raise ConfigurationError(f"The {name!r} header is managed by the provider adapter.")
        if not name or any(char in name for char in "\r\n:") or any(char in value for char in "\r\n"):
            raise ConfigurationError("Provider headers must not contain control characters.")
        result[name] = value
    return result


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


class OpenAICompatibleProvider:
    """HTTP adapter for the conservative Chat Completions API subset.

    The adapter does no retries. Retry and fallback policy is centralized in
    :class:`UnifiedLLM` so the total attempt budget stays observable and bounded.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        allow_insecure_http: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("Provider name must be a non-empty string.")
        self.name = name.strip()
        self.base_url = _validate_base_url(base_url, allow_insecure_http=allow_insecure_http)
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._headers = _validate_headers(headers)
        self._client = client
        self._owns_client = client is None

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> UnifiedLLMResponse:
        payload: dict[str, Any] = {
            "messages": list(messages),
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = list(tools)
        try:
            json.dumps(payload)
        except (TypeError, ValueError, RecursionError) as exc:
            raise RequestValidationError("Messages and tools must be JSON serializable.") from exc

        headers = {"Accept": "application/json", "Content-Type": "application/json", **self._headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if self._client is None:
            self._client = httpx.AsyncClient()

        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"Provider {self.name!r} timed out.", retryable=True) from exc
        except httpx.NetworkError as exc:
            raise ProviderError(self.name, f"Provider {self.name!r} was unreachable.", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"Provider {self.name!r} request failed.", retryable=True) from exc

        if not 200 <= response.status_code < 300:
            retryable = response.status_code in _RETRYABLE_STATUS_CODES
            raise ProviderError(
                self.name,
                f"Provider {self.name!r} returned HTTP {response.status_code}.",
                status_code=response.status_code,
                retryable=retryable,
                retry_after=_parse_retry_after(response.headers.get("Retry-After")) if retryable else None,
            )

        try:
            data = response.json()
            choices = data["choices"]
            choice = choices[0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                self.name,
                f"Provider {self.name!r} returned a malformed completion response.",
                retryable=False,
            ) from exc

        raw_content = message.get("content")
        content = self._normalize_content(raw_content)
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))
        if not content and not tool_calls:
            raise ProviderError(
                self.name,
                f"Provider {self.name!r} returned neither content nor tool calls.",
                retryable=False,
            )

        response_model = data.get("model")
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage")
        return UnifiedLLMResponse(
            content=content,
            model=response_model if isinstance(response_model, str) and response_model else model,
            provider=self.name,
            usage={
                str(key): value
                for key, value in usage.items()
                if isinstance(usage, Mapping) and isinstance(value, int) and not isinstance(value, bool)
            }
            if isinstance(usage, Mapping)
            else {},
            finish_reason=finish_reason if isinstance(finish_reason, str) and finish_reason else "stop",
            tool_calls=tool_calls,
        )

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "".join(parts)
        return ""

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> tuple[dict[str, Any], ...]:
        if not isinstance(tool_calls, list):
            return ()
        return tuple(dict(call) for call in tool_calls if isinstance(call, Mapping))

    async def aclose(self) -> None:
        """Close the internally-created HTTP client, if any."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


Sleep: TypeAlias = Callable[[float], Awaitable[None]]


class UnifiedLLM:
    """Route chat completions through explicit, bounded provider routes."""

    def __init__(
        self,
        routes: Sequence[Route],
        *,
        request_timeout: float = 30.0,
        max_attempts_per_route: int = 2,
        max_total_attempts: int = 4,
        max_concurrency: int = 10,
        max_input_chars: int = 200_000,
        max_request_bytes: int = 1_000_000,
        max_output_tokens: int = 32_768,
        backoff_base: float = 0.25,
        max_retry_delay: float = 5.0,
        retry_jitter: float = 0.1,
        _sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._routes = tuple(routes)
        if not self._routes:
            raise ConfigurationError("At least one LLM route is required.")
        if len(self._routes) > 8:
            raise ConfigurationError("At most eight routes may be configured.")
        names = [route.provider.name for route in self._routes]
        if len(names) != len(set(names)):
            raise ConfigurationError("Provider names must be unique across routes.")
        if not 0 < request_timeout <= 600:
            raise ConfigurationError("request_timeout must be greater than 0 and at most 600 seconds.")
        if not 1 <= max_attempts_per_route <= 5:
            raise ConfigurationError("max_attempts_per_route must be between 1 and 5.")
        if not 1 <= max_total_attempts <= 16:
            raise ConfigurationError("max_total_attempts must be between 1 and 16.")
        if not 1 <= max_concurrency <= 1_000:
            raise ConfigurationError("max_concurrency must be between 1 and 1000.")
        if not 1 <= max_input_chars <= 10_000_000:
            raise ConfigurationError("max_input_chars must be between 1 and 10000000.")
        if not 1 <= max_request_bytes <= 20_000_000:
            raise ConfigurationError("max_request_bytes must be between 1 and 20000000.")
        if not 1 <= max_output_tokens <= 1_000_000:
            raise ConfigurationError("max_output_tokens must be between 1 and 1000000.")
        if backoff_base < 0 or max_retry_delay < 0:
            raise ConfigurationError("Retry delays cannot be negative.")
        if not 0 <= retry_jitter <= 1:
            raise ConfigurationError("retry_jitter must be between 0 and 1.")

        self.request_timeout = float(request_timeout)
        self.max_attempts_per_route = max_attempts_per_route
        self.max_total_attempts = max_total_attempts
        self.max_input_chars = max_input_chars
        self.max_request_bytes = max_request_bytes
        self.max_output_tokens = max_output_tokens
        self.backoff_base = float(backoff_base)
        self.max_retry_delay = float(max_retry_delay)
        self.retry_jitter = float(retry_jitter)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._sleep = _sleep

    @classmethod
    def from_env(
        cls,
        prefix: str = "UNIFIED_LLM",
        *,
        environ: Mapping[str, str] | None = None,
    ) -> UnifiedLLM:
        """Build a one-route client from explicit environment variables.

        Required: ``<PREFIX>_MODEL``. ``<PREFIX>_API_KEY`` is also required for
        the default OpenAI URL. Local endpoints may omit it.
        """

        if not _ENV_PREFIX_RE.fullmatch(prefix):
            raise ConfigurationError("Environment prefix must contain uppercase letters, digits, and underscores.")
        values = os.environ if environ is None else environ

        def read(name: str) -> str | None:
            value = values.get(f"{prefix}_{name}")
            return value.strip() if value and value.strip() else None

        model = read("MODEL")
        if model is None:
            raise ConfigurationError(f"{prefix}_MODEL is required.")
        base_url = read("BASE_URL") or "https://api.openai.com/v1"
        api_key = read("API_KEY")
        if base_url.rstrip("/") == "https://api.openai.com/v1" and api_key is None:
            raise ConfigurationError(f"{prefix}_API_KEY is required for the default OpenAI endpoint.")

        provider = OpenAICompatibleProvider(
            name=read("PROVIDER") or "primary",
            base_url=base_url,
            api_key=api_key,
            allow_insecure_http=_parse_env_bool(read("ALLOW_INSECURE_HTTP"), f"{prefix}_ALLOW_INSECURE_HTTP"),
        )
        return cls(
            [Route(provider, model)],
            request_timeout=_parse_env_float(read("TIMEOUT"), f"{prefix}_TIMEOUT", 30.0),
            max_attempts_per_route=_parse_env_int(
                read("MAX_ATTEMPTS_PER_ROUTE"), f"{prefix}_MAX_ATTEMPTS_PER_ROUTE", 2
            ),
            max_total_attempts=_parse_env_int(read("MAX_TOTAL_ATTEMPTS"), f"{prefix}_MAX_TOTAL_ATTEMPTS", 4),
            max_concurrency=_parse_env_int(read("MAX_CONCURRENCY"), f"{prefix}_MAX_CONCURRENCY", 10),
        )

    @property
    def routes(self) -> tuple[Route, ...]:
        """Return the configured routes in fallback order."""

        return self._routes

    @property
    def providers(self) -> dict[str, Provider]:
        """Return configured providers keyed by their explicit route name."""

        return {route.provider.name: route.provider for route in self._routes}

    def get_available_providers(self) -> list[str]:
        """Return provider names in route order."""

        return [route.provider.name for route in self._routes]

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> str:
        """Generate text and return only content; failures raise typed errors."""

        response = await self.generate_with_metadata(
            prompt,
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )
        return response.content

    async def generate_with_metadata(
        self,
        prompt: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> UnifiedLLMResponse:
        """Generate text and return normalized response and attempt metadata."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise RequestValidationError("prompt must be a non-empty string.")
        messages: list[Message] = []
        if system is not None:
            if not isinstance(system, str) or not system.strip():
                raise RequestValidationError("system must be a non-empty string when provided.")
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat_with_metadata(
            messages,
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        provider: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a chat completion and return only its text content."""

        response = await self.chat_with_metadata(
            messages,
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content

    async def chat_with_tools(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolDefinition],
        model: str | None = None,
        provider: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> UnifiedLLMResponse:
        """Generate a completion with OpenAI-compatible function definitions."""

        if not tools:
            raise RequestValidationError("tools must contain at least one tool definition.")
        return await self.chat_with_metadata(
            messages,
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
        )

    async def chat_with_metadata(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        provider: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> UnifiedLLMResponse:
        """Run the request through the selected bounded route set."""

        normalized_messages = self._validate_request(messages, max_tokens, temperature, tools)
        selected = self._select_routes(provider=provider, model=model)
        attempts: list[Attempt] = []
        total_attempts = 0

        async with self._semaphore:
            for route, resolved_model in selected:
                for route_attempt in range(1, self.max_attempts_per_route + 1):
                    if total_attempts >= self.max_total_attempts:
                        raise FallbackExhausted(attempts)
                    total_attempts += 1
                    started = time.perf_counter()
                    try:
                        response = await asyncio.wait_for(
                            route.provider.complete(
                                messages=normalized_messages,
                                model=resolved_model,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                timeout=self.request_timeout,
                                tools=tools,
                            ),
                            timeout=self.request_timeout,
                        )
                    except asyncio.CancelledError:
                        raise
                    except asyncio.TimeoutError:
                        provider_error = ProviderError(
                            route.provider.name,
                            f"Provider {route.provider.name!r} timed out.",
                            retryable=True,
                        )
                        should_continue = await self._handle_provider_error(
                            provider_error,
                            route,
                            resolved_model,
                            route_attempt,
                            started,
                            attempts,
                        )
                        if should_continue:
                            continue
                        break
                    except ProviderError as exc:
                        should_continue = await self._handle_provider_error(
                            exc,
                            route,
                            resolved_model,
                            route_attempt,
                            started,
                            attempts,
                        )
                        if should_continue:
                            continue
                        break
                    except Exception as exc:
                        attempts.append(
                            Attempt(
                                route.provider.name,
                                resolved_model,
                                route_attempt,
                                _elapsed_ms(started),
                                "adapter_error",
                                False,
                            )
                        )
                        raise ProviderError(
                            route.provider.name,
                            f"Provider adapter {route.provider.name!r} failed unexpectedly.",
                            retryable=False,
                            attempts=attempts,
                        ) from exc
                    else:
                        attempts.append(
                            Attempt(
                                route.provider.name,
                                resolved_model,
                                route_attempt,
                                _elapsed_ms(started),
                            )
                        )
                        return replace(
                            response,
                            provider=route.provider.name,
                            model=response.model or resolved_model,
                            attempts=tuple(attempts),
                        )

        raise FallbackExhausted(attempts)

    async def _handle_provider_error(
        self,
        error: ProviderError,
        route: Route,
        model: str,
        route_attempt: int,
        started: float,
        attempts: list[Attempt],
    ) -> bool:
        attempts.append(
            Attempt(
                route.provider.name,
                model,
                route_attempt,
                _elapsed_ms(started),
                f"http_{error.status_code}" if error.status_code is not None else "provider_error",
                error.retryable,
            )
        )
        if not error.retryable:
            message = (
                f"Provider {route.provider.name!r} returned HTTP {error.status_code}."
                if error.status_code is not None
                else f"Provider {route.provider.name!r} failed."
            )
            raise ProviderError(
                route.provider.name,
                message,
                status_code=error.status_code,
                retryable=False,
                attempts=attempts,
            ) from error
        if route_attempt >= self.max_attempts_per_route:
            return False

        requested_delay = error.retry_after
        if requested_delay is None:
            requested_delay = self.backoff_base * (2 ** (route_attempt - 1))
            if requested_delay and self.retry_jitter:
                requested_delay *= random.uniform(1 - self.retry_jitter, 1 + self.retry_jitter)
        delay = min(self.max_retry_delay, max(0.0, requested_delay))
        if delay:
            await self._sleep(delay)
        return True

    def _validate_request(
        self,
        messages: Sequence[Message],
        max_tokens: int,
        temperature: float,
        tools: Sequence[ToolDefinition] | None,
    ) -> tuple[dict[str, Any], ...]:
        if not messages:
            raise RequestValidationError("messages must be a non-empty sequence of mappings.")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= self.max_output_tokens
        ):
            raise RequestValidationError(f"max_tokens must be an integer between 1 and {self.max_output_tokens}.")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise RequestValidationError("temperature must be a number between 0 and 2.")

        normalized: list[dict[str, Any]] = []
        total_chars = 0
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise RequestValidationError(f"messages[{index}] must be a mapping.")
            role = message.get("role")
            content = message.get("content")
            if role not in _ALLOWED_ROLES:
                raise RequestValidationError(
                    f"messages[{index}].role must be one of: {', '.join(sorted(_ALLOWED_ROLES))}."
                )
            if not isinstance(content, str) or not content:
                raise RequestValidationError(f"messages[{index}].content must be a non-empty string.")
            total_chars += len(content)
            normalized.append(dict(message))
        if total_chars > self.max_input_chars:
            raise RequestValidationError(
                f"Total message content exceeds the configured {self.max_input_chars}-character limit."
            )
        if tools is not None and not all(isinstance(tool, Mapping) for tool in tools):
            raise RequestValidationError("Every tool definition must be a mapping.")
        try:
            serialized = json.dumps(
                {"messages": normalized, "tools": list(tools) if tools is not None else None},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise RequestValidationError("Messages and tools must be JSON serializable.") from exc
        if len(serialized) > self.max_request_bytes:
            raise RequestValidationError(
                f"Serialized messages and tools exceed the configured {self.max_request_bytes}-byte request limit."
            )
        return tuple(normalized)

    def _select_routes(self, *, provider: str | None, model: str | None) -> tuple[tuple[Route, str], ...]:
        if provider is not None:
            if not isinstance(provider, str) or not provider.strip():
                raise RequestValidationError("provider must be a non-empty string when provided.")
            for route in self._routes:
                if route.provider.name == provider:
                    resolved_model = model if model is not None else route.model
                    if not isinstance(resolved_model, str) or not resolved_model.strip():
                        raise RequestValidationError("model must be a non-empty string when provided.")
                    return ((route, resolved_model),)
            raise RequestValidationError(f"Unknown provider {provider!r}.")

        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise RequestValidationError("model must be a non-empty string when provided.")
            matches = tuple((route, route.model) for route in self._routes if route.model == model)
            if matches:
                return matches
            if len(self._routes) == 1:
                return ((self._routes[0], model),)
            raise RequestValidationError("A model override with multiple routes also requires an explicit provider.")

        return tuple((route, route.model) for route in self._routes)

    async def aclose(self) -> None:
        """Close provider resources without closing user-supplied HTTP clients."""

        seen: set[int] = set()
        for route in self._routes:
            if id(route.provider) in seen:
                continue
            seen.add(id(route.provider))
            close = getattr(route.provider, "aclose", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result

    async def __aenter__(self) -> UnifiedLLM:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _parse_env_bool(value: str | None, name: str) -> bool:
    if value is None:
        return False
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean value.")


def _parse_env_int(value: str | None, name: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc


def _parse_env_float(value: str | None, name: str, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


# Compatibility name for the original extracted implementation.
UnifiedLLMService = UnifiedLLM


__all__ = [
    "Attempt",
    "ConfigurationError",
    "FallbackExhausted",
    "Message",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "RequestValidationError",
    "Route",
    "ToolDefinition",
    "UnifiedLLM",
    "UnifiedLLMError",
    "UnifiedLLMResponse",
    "UnifiedLLMService",
]
