"""Provider-agnostic LLM layer.

The rest of the code only talks to ``LLMProvider`` (``stream`` / ``complete``) and reads
an ``LLMResult``. Two implementations:

* ``OpenAICompatProvider`` — the ``openai`` SDK pointed at any OpenAI-compatible Chat
  Completions endpoint (Groq, OpenRouter, Mistral, Gemini's OpenAI endpoint, OpenAI).
  Streams with ``stream_options={"include_usage": True}``; if the endpoint rejects that
  parameter the call is retried without it, and if no usage arrives at all the token
  counts are estimated (≈4 characters per token) and flagged with ``usage_estimated``.
* ``AnthropicProvider`` — the ``anthropic`` SDK, ``client.messages.stream(...)`` with the
  system prompt in the ``system`` parameter; usage is read from the final message.

SDKs are imported lazily inside the constructors, so tests and the CI run without them.
API failures (auth, unknown model, rate limit, connection) surface as ``ProviderError`` with
a one-line readable message, so callers never have to know the SDK exception classes.

Token accounting: ``input_tokens`` is the *total* prompt size including cached tokens and
``cache_read_tokens`` the cached subset — the convention ``costs.cost_usd`` expects.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings, settings

Message = dict[str, str]  # {"role": "user" | "assistant", "content": text}

PROVIDER_NAMES = ("openai_compat", "anthropic")


class ProviderError(RuntimeError):
    """An API call failed (auth, unknown model, rate limit, connection…); ``str(exc)`` is user-readable."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int  # total prompt tokens, cached ones included
    output_tokens: int
    cache_read_tokens: int  # subset of input_tokens served from the provider's prompt cache
    model: str  # model id reported by the provider (OpenRouter resolves aliases), else the requested one
    elapsed_seconds: float
    usage_estimated: bool  # True when the provider sent no usage and the counts are a heuristic
    stop_reason: str | None = None  # provider-specific: "stop"/"length" (OpenAI), "end_turn"/"max_tokens" (Anthropic)


class LLMProvider(Protocol):
    name: str

    def stream(
        self, messages: Sequence[Message], system: str, model: str, max_tokens: int
    ) -> Generator[str, None, LLMResult]:
        """Yield text deltas as they arrive; the generator's return value is the LLMResult."""
        ...

    def complete(self, messages: Sequence[Message], system: str, model: str, max_tokens: int) -> LLMResult:
        """Non-streaming call with the same result (for the eval script)."""
        ...


def estimate_tokens(text: str) -> int:
    """Rough token count (≈4 characters per token) for providers that send no usage."""
    return (len(text) + 3) // 4


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or key lookup, so SDK objects, their extra fields (dicts) and test fakes all work."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _rejects_param(exc: BaseException, param: str) -> bool:
    """True if an API error (HTTP 400/422) complains about a request parameter."""
    return _get(exc, "status_code") in (400, 422) and param in str(exc)


def _sdk_errors(package: str) -> tuple[type[BaseException], ...]:
    """The SDK's base exception class (``APIError`` covers status + connection errors), or () if not installed."""
    try:
        module = importlib.import_module(package)
    except ImportError:
        return ()
    return (module.APIError,)


def describe_api_error(exc: BaseException) -> str:
    """One readable line for an SDK error, e.g. "HTTP 404 (model_not_found): The model `x` does not exist"."""
    status = _get(exc, "status_code")
    body = _get(exc, "body")
    error = body.get("error", body) if isinstance(body, Mapping) else None  # openai: inner dict; anthropic: nested
    detail = error.get("message") if isinstance(error, Mapping) else None
    code = (error.get("code") or error.get("type")) if isinstance(error, Mapping) else None
    head = f"HTTP {status}" if status else type(exc).__name__
    if code:
        head += f" ({code})"
    return f"{head}: {detail or _get(exc, 'message') or str(exc) or 'request failed'}"


class OpenAICompatProvider:
    """Chat Completions via the ``openai`` SDK against any compatible ``base_url``."""

    name = "openai_compat"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        client: Any = None,
        include_usage: bool = True,
        api_errors: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("LLM_API_KEY is not set (see .env.example)")
            from openai import OpenAI  # lazy: tests and CI run without the SDK

            client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._client = client
        self._include_usage = include_usage  # switched off for good if the endpoint rejects stream_options
        self._api_errors = api_errors if api_errors is not None else _sdk_errors("openai")

    @staticmethod
    def _chat_messages(system: str, messages: Sequence[Message]) -> list[Message]:
        return [{"role": "system", "content": system}, *messages]

    def _create(self, **kwargs: Any) -> Any:
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "stream_options" not in kwargs or not _rejects_param(exc, "stream_options"):
                raise
            # e.g. Mistral rejects unknown parameters (and sends usage in the last chunk anyway)
            self._include_usage = False
            kwargs.pop("stream_options")
            return self._client.chat.completions.create(**kwargs)

    def stream(
        self, messages: Sequence[Message], system: str, model: str, max_tokens: int
    ) -> Generator[str, None, LLMResult]:
        try:
            return (yield from self._stream(messages, system, model, max_tokens))
        except self._api_errors as exc:
            raise ProviderError(describe_api_error(exc)) from exc

    def complete(self, messages: Sequence[Message], system: str, model: str, max_tokens: int) -> LLMResult:
        try:
            return self._complete(messages, system, model, max_tokens)
        except self._api_errors as exc:
            raise ProviderError(describe_api_error(exc)) from exc

    def _stream(
        self, messages: Sequence[Message], system: str, model: str, max_tokens: int
    ) -> Generator[str, None, LLMResult]:
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = dict(
            model=model, messages=self._chat_messages(system, messages), max_tokens=max_tokens, stream=True
        )
        if self._include_usage:
            kwargs["stream_options"] = {"include_usage": True}
        response = self._create(**kwargs)

        parts: list[str] = []
        usage = seen_model = finish = None
        try:
            for chunk in response:
                # usage normally rides on a final chunk without choices; Groq puts it under x_groq
                usage = _get(chunk, "usage") or _get(_get(chunk, "x_groq"), "usage") or usage
                seen_model = _get(chunk, "model") or seen_model
                choices = _get(chunk, "choices") or []
                if not choices:
                    continue
                finish = _get(choices[0], "finish_reason") or finish
                delta = _get(_get(choices[0], "delta"), "content")
                if delta:
                    parts.append(delta)
                    yield delta
        finally:
            close = getattr(response, "close", None)  # release the connection, also when the caller stops early
            if callable(close):
                close()
        return self._result("".join(parts), usage, seen_model or model, finish, system, messages, t0)

    def _complete(self, messages: Sequence[Message], system: str, model: str, max_tokens: int) -> LLMResult:
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model, messages=self._chat_messages(system, messages), max_tokens=max_tokens
        )
        choices = _get(response, "choices") or []
        first = choices[0] if choices else None
        text = _get(_get(first, "message"), "content") or ""
        return self._result(
            text,
            _get(response, "usage"),
            _get(response, "model") or model,
            _get(first, "finish_reason"),
            system,
            messages,
            t0,
        )

    @staticmethod
    def _result(
        text: str, usage: Any, model: str, finish: str | None, system: str, messages: Sequence[Message], t0: float
    ) -> LLMResult:
        elapsed = time.perf_counter() - t0
        prompt_tokens = _get(usage, "prompt_tokens")
        if prompt_tokens is not None:
            details = _get(usage, "prompt_tokens_details")
            return LLMResult(
                text=text,
                input_tokens=int(prompt_tokens),
                output_tokens=int(_get(usage, "completion_tokens") or 0),
                cache_read_tokens=int(_get(details, "cached_tokens") or 0),
                model=model,
                elapsed_seconds=elapsed,
                usage_estimated=False,
                stop_reason=finish,
            )
        prompt_text = system + "".join(m["content"] for m in messages)
        return LLMResult(
            text=text,
            input_tokens=estimate_tokens(prompt_text),
            output_tokens=estimate_tokens(text),
            cache_read_tokens=0,
            model=model,
            elapsed_seconds=elapsed,
            usage_estimated=True,
            stop_reason=finish,
        )


class AnthropicProvider:
    """Messages API via the ``anthropic`` SDK; the system prompt goes in the ``system`` parameter."""

    name = "anthropic"

    def __init__(
        self, api_key: str = "", client: Any = None, api_errors: tuple[type[BaseException], ...] | None = None
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set (see .env.example)")
            import anthropic  # lazy: tests and CI run without the SDK

            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self._api_errors = api_errors if api_errors is not None else _sdk_errors("anthropic")

    def stream(
        self, messages: Sequence[Message], system: str, model: str, max_tokens: int
    ) -> Generator[str, None, LLMResult]:
        try:
            return (yield from self._stream(messages, system, model, max_tokens))
        except self._api_errors as exc:
            raise ProviderError(describe_api_error(exc)) from exc

    def complete(self, messages: Sequence[Message], system: str, model: str, max_tokens: int) -> LLMResult:
        try:
            return self._complete(messages, system, model, max_tokens)
        except self._api_errors as exc:
            raise ProviderError(describe_api_error(exc)) from exc

    def _stream(
        self, messages: Sequence[Message], system: str, model: str, max_tokens: int
    ) -> Generator[str, None, LLMResult]:
        t0 = time.perf_counter()
        parts: list[str] = []
        with self._client.messages.stream(
            model=model, max_tokens=max_tokens, system=system, messages=list(messages)
        ) as stream:
            for delta in stream.text_stream:
                parts.append(delta)
                yield delta
            final = stream.get_final_message()
        return self._result("".join(parts), final, model, t0)

    def _complete(self, messages: Sequence[Message], system: str, model: str, max_tokens: int) -> LLMResult:
        t0 = time.perf_counter()
        final = self._client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=list(messages)
        )
        text = "".join(_get(block, "text") or "" for block in final.content if _get(block, "type") == "text")
        return self._result(text, final, model, t0)

    @staticmethod
    def _result(text: str, final: Any, model: str, t0: float) -> LLMResult:
        usage = final.usage
        cache_read = int(_get(usage, "cache_read_input_tokens") or 0)
        cache_written = int(_get(usage, "cache_creation_input_tokens") or 0)
        # Anthropic reports uncached, cache-read and cache-written input separately; cost_usd wants the total
        return LLMResult(
            text=text,
            input_tokens=int(usage.input_tokens) + cache_read + cache_written,
            output_tokens=int(usage.output_tokens),
            cache_read_tokens=cache_read,
            model=_get(final, "model") or model,
            elapsed_seconds=time.perf_counter() - t0,
            usage_estimated=False,
            stop_reason=_get(final, "stop_reason"),
        )


def get_provider(cfg: Settings = settings) -> LLMProvider:
    """Build the provider selected by ``LLM_PROVIDER`` with the keys and base URL from the settings."""
    if cfg.llm_provider == "openai_compat":
        return OpenAICompatProvider(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
    if cfg.llm_provider == "anthropic":
        return AnthropicProvider(api_key=cfg.anthropic_api_key)
    raise ValueError(f"Unknown LLM_PROVIDER {cfg.llm_provider!r}; expected one of {PROVIDER_NAMES}")
