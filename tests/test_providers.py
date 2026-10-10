"""Provider layer tests with fake SDK clients: no network, and no SDK needs to be installed."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from docs_chat.config import Settings
from docs_chat.providers import (
    AnthropicProvider,
    OpenAICompatProvider,
    ProviderError,
    describe_api_error,
    estimate_tokens,
    get_provider,
)

SYSTEM = "You answer from the numbered passages only."
MESSAGES = [{"role": "user", "content": "Passages:\n\n[1] (manual.pdf, page 2)\nE3 = overheated.\n\nQuestion: E3?"}]


def drain(gen):
    """Run a provider stream to the end: returns (list of deltas, LLMResult)."""
    deltas = []
    try:
        while True:
            deltas.append(next(gen))
    except StopIteration as stop:
        return deltas, stop.value


# --- OpenAI-compatible fakes --------------------------------------------------------------


def chunk(text=None, finish=None, usage=None, model="llama-3.3-70b-versatile", **extra):
    """One streamed chunk; without text/finish it has no choices (like the final usage-only chunk)."""
    choices = [NS(delta=NS(content=text), finish_reason=finish)] if text is not None or finish else []
    return NS(choices=choices, usage=usage, model=model, **extra)


OPENAI_USAGE = NS(prompt_tokens=120, completion_tokens=8, prompt_tokens_details=NS(cached_tokens=20))


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class FakeOpenAI:
    """Mimics ``openai.OpenAI().chat.completions.create`` for streaming and non-streaming calls."""

    def __init__(self, stream=None, response=None, fail_first=None):
        self.stream, self.response, self.fail_first = stream, response, fail_first
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_first is not None:
            error, self.fail_first = self.fail_first, None
            raise error
        return self.stream if kwargs.get("stream") else self.response


def test_openai_compat_stream_yields_deltas_and_reads_usage():
    stream = FakeStream(
        [chunk("Error "), chunk("E3 means "), chunk(), chunk("overheating [1]."), chunk(finish="stop"), chunk(usage=OPENAI_USAGE)]
    )
    client = FakeOpenAI(stream=stream)
    deltas, result = drain(OpenAICompatProvider(client=client).stream(MESSAGES, SYSTEM, "llama-3.3-70b-versatile", 700))
    assert deltas == ["Error ", "E3 means ", "overheating [1]."]
    assert result.text == "Error E3 means overheating [1]."
    assert (result.input_tokens, result.output_tokens, result.cache_read_tokens) == (120, 8, 20)
    assert result.usage_estimated is False
    assert result.stop_reason == "stop"
    assert result.model == "llama-3.3-70b-versatile"
    assert result.elapsed_seconds >= 0
    assert stream.closed
    call = client.calls[0]
    assert call["stream"] is True and call["stream_options"] == {"include_usage": True}
    assert call["model"] == "llama-3.3-70b-versatile" and call["max_tokens"] == 700
    assert call["messages"][0] == {"role": "system", "content": SYSTEM}
    assert call["messages"][1:] == MESSAGES


def test_openai_compat_estimates_tokens_when_no_usage_arrives():
    client = FakeOpenAI(stream=FakeStream([chunk("twelve chars"), chunk(finish="stop")]))
    _, result = drain(OpenAICompatProvider(client=client).stream(MESSAGES, SYSTEM, "m", 700))
    assert result.text == "twelve chars"
    assert result.usage_estimated is True
    assert result.output_tokens == estimate_tokens("twelve chars") == 3
    assert result.input_tokens == estimate_tokens(SYSTEM + MESSAGES[0]["content"]) > 0
    assert result.cache_read_tokens == 0


def test_openai_compat_reads_groq_style_usage():
    groq_chunk = chunk(finish="stop", x_groq={"usage": {"prompt_tokens": 50, "completion_tokens": 5}})
    client = FakeOpenAI(stream=FakeStream([chunk("hi"), groq_chunk]))
    _, result = drain(OpenAICompatProvider(client=client).stream(MESSAGES, SYSTEM, "m", 700))
    assert (result.input_tokens, result.output_tokens, result.usage_estimated) == (50, 5, False)


def test_openai_compat_retries_without_stream_options_when_rejected():
    error = RuntimeError("422: Extra inputs are not permitted: stream_options")
    error.status_code = 422
    client = FakeOpenAI(stream=FakeStream([chunk("ok"), chunk(usage=OPENAI_USAGE)]), fail_first=error)
    provider = OpenAICompatProvider(client=client)
    _, result = drain(provider.stream(MESSAGES, SYSTEM, "m", 700))
    assert result.text == "ok" and result.usage_estimated is False
    assert len(client.calls) == 2
    assert "stream_options" in client.calls[0] and "stream_options" not in client.calls[1]
    # remembered: the next call does not send stream_options at all
    drain(provider.stream(MESSAGES, SYSTEM, "m", 700))
    assert len(client.calls) == 3 and "stream_options" not in client.calls[2]


def test_openai_compat_other_errors_propagate():
    error = RuntimeError("401: invalid api key")
    error.status_code = 401
    client = FakeOpenAI(stream=FakeStream([]), fail_first=error)
    with pytest.raises(RuntimeError, match="401"):
        drain(OpenAICompatProvider(client=client).stream(MESSAGES, SYSTEM, "m", 700))
    assert len(client.calls) == 1


def test_openai_compat_complete():
    response = NS(
        choices=[NS(message=NS(content="Full answer [1]."), finish_reason="stop")], usage=OPENAI_USAGE, model="m-resolved"
    )
    client = FakeOpenAI(response=response)
    result = OpenAICompatProvider(client=client).complete(MESSAGES, SYSTEM, "m", 700)
    assert result.text == "Full answer [1]."
    assert result.model == "m-resolved" and result.stop_reason == "stop"
    assert (result.input_tokens, result.output_tokens, result.cache_read_tokens) == (120, 8, 20)
    assert "stream" not in client.calls[0] and client.calls[0]["messages"][0]["role"] == "system"


# --- Anthropic fakes ------------------------------------------------------------------------


class FakeAnthropicStream:
    """Mimics the context manager returned by ``client.messages.stream(...)``."""

    def __init__(self, deltas, final):
        self.deltas, self.final, self.exited = deltas, final, False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    @property
    def text_stream(self):
        return iter(self.deltas)

    def get_final_message(self):
        return self.final


class FakeAnthropic:
    def __init__(self, deltas, final, error=None):
        self.calls = []
        self.stream = FakeAnthropicStream(deltas, final)
        self.final, self.error = final, error
        self.messages = NS(stream=self._stream, create=self._create)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.stream

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.final


def anthropic_message(blocks, input_tokens, output_tokens, cache_read=0, cache_written=0, stop_reason="end_turn"):
    return NS(
        content=blocks,
        usage=NS(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_written,
        ),
        model="claude-haiku-5-5",
        stop_reason=stop_reason,
    )


def test_anthropic_stream_yields_deltas_and_reads_final_usage():
    final = anthropic_message([NS(type="text", text="E3 means overheating [1].")], 100, 12, cache_read=30)
    client = FakeAnthropic(["E3 means ", "overheating [1]."], final)
    deltas, result = drain(AnthropicProvider(client=client).stream(MESSAGES, SYSTEM, "claude-haiku-5-5", 700))
    assert deltas == ["E3 means ", "overheating [1]."]
    assert result.text == "E3 means overheating [1]."
    # input_tokens is the total prompt (100 uncached + 30 read from cache), as cost_usd expects
    assert (result.input_tokens, result.output_tokens, result.cache_read_tokens) == (130, 12, 30)
    assert result.usage_estimated is False
    assert result.stop_reason == "end_turn" and result.model == "claude-haiku-5-5"
    assert client.stream.exited
    call = client.calls[0]
    assert call["system"] == SYSTEM and call["max_tokens"] == 700 and call["model"] == "claude-haiku-5-5"
    assert call["messages"] == MESSAGES and all(m["role"] != "system" for m in call["messages"])


def test_anthropic_complete_joins_text_blocks():
    blocks = [NS(type="text", text="Part one. "), NS(type="tool_use"), NS(type="text", text="Part two.")]
    client = FakeAnthropic([], anthropic_message(blocks, 10, 5, stop_reason="max_tokens"))
    result = AnthropicProvider(client=client).complete(MESSAGES, SYSTEM, "claude-haiku-5-5", 700)
    assert result.text == "Part one. Part two."
    assert (result.input_tokens, result.output_tokens, result.cache_read_tokens) == (10, 5, 0)
    assert result.stop_reason == "max_tokens"
    assert client.calls[0]["system"] == SYSTEM and client.calls[0]["messages"] == MESSAGES


# --- selection and configuration ------------------------------------------------------------


def test_get_provider_rejects_unknown_provider_and_missing_keys():
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        get_provider(Settings(llm_provider="bedrock"))
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        get_provider(Settings(llm_provider="openai_compat", llm_api_key=""))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_provider(Settings(llm_provider="anthropic", anthropic_api_key=""))


def test_default_model_only_for_the_anthropic_provider():
    assert Settings(llm_provider="anthropic", llm_model="").llm_model == "claude-haiku-5-5"
    assert Settings(llm_provider="anthropic", llm_model="claude-sonnet-5-5").llm_model == "claude-sonnet-5-5"
    assert Settings(llm_provider="openai_compat", llm_model="").llm_model == ""


# --- API errors -> ProviderError ---------------------------------------------------------------


class FakeAPIError(Exception):
    """Shaped like the SDKs' APIError: ``status_code`` and a parsed ``body`` (both optional)."""

    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.message, self.status_code, self.body = message, status_code, body


MODEL_NOT_FOUND = FakeAPIError(
    "Error code: 404 - {...}",
    status_code=404,
    body={"message": "The model `llama-x` does not exist or you do not have access to it.", "type": "invalid_request_error",
          "code": "model_not_found"},
)


def test_describe_api_error_formats_status_code_and_detail():
    assert describe_api_error(MODEL_NOT_FOUND) == (
        "HTTP 404 (model_not_found): The model `llama-x` does not exist or you do not have access to it."
    )
    nested = FakeAPIError("Error code: 401", status_code=401,
                          body={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
    assert describe_api_error(nested) == "HTTP 401 (authentication_error): invalid x-api-key"
    assert describe_api_error(FakeAPIError("Connection error.")) == "FakeAPIError: Connection error."


def test_openai_compat_wraps_sdk_errors_in_provider_error():
    client = FakeOpenAI(stream=FakeStream([]), fail_first=MODEL_NOT_FOUND)
    provider = OpenAICompatProvider(client=client, api_errors=(FakeAPIError,))
    with pytest.raises(ProviderError, match="HTTP 404 \\(model_not_found\\)") as info:
        drain(provider.stream(MESSAGES, SYSTEM, "llama-x", 700))
    assert info.value.__cause__ is MODEL_NOT_FOUND
    client = FakeOpenAI(response=None, fail_first=FakeAPIError("Connection error."))
    with pytest.raises(ProviderError, match="Connection error"):
        OpenAICompatProvider(client=client, api_errors=(FakeAPIError,)).complete(MESSAGES, SYSTEM, "m", 700)


def test_openai_compat_still_retries_stream_options_before_wrapping():
    rejected = FakeAPIError("Extra inputs are not permitted: stream_options", status_code=422)
    client = FakeOpenAI(stream=FakeStream([chunk("ok"), chunk(usage=OPENAI_USAGE)]), fail_first=rejected)
    _, result = drain(OpenAICompatProvider(client=client, api_errors=(FakeAPIError,)).stream(MESSAGES, SYSTEM, "m", 700))
    assert result.text == "ok" and len(client.calls) == 2


def test_anthropic_wraps_sdk_errors_in_provider_error():
    error = FakeAPIError("Error code: 429", status_code=429,
                         body={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    client = FakeAnthropic([], None, error=error)
    provider = AnthropicProvider(client=client, api_errors=(FakeAPIError,))
    with pytest.raises(ProviderError, match="HTTP 429 \\(rate_limit_error\\): slow down"):
        drain(provider.stream(MESSAGES, SYSTEM, "claude-haiku-5-5", 700))
    with pytest.raises(ProviderError, match="rate_limit_error"):
        provider.complete(MESSAGES, SYSTEM, "claude-haiku-5-5", 700)
