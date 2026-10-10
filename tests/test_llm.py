"""Tests for prompt building (``docs_chat.llm.build_messages``). No network, no API key."""

from __future__ import annotations

from docs_chat.chunking import Chunk
from docs_chat.index import Hit
import pytest

from docs_chat.llm import (
    HISTORY_TURNS,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    WEAK_RETRIEVAL_NOTE,
    answer_stream,
    build_messages,
    normalize_answer,
    parse_citations,
)
from docs_chat.providers import LLMResult

QUESTION = "How many vacation days do new employees get?"


def make_hit(doc: str, page: int, text: str, pseudo: bool = False) -> Hit:
    chunk = Chunk(
        chunk_id=f"{doc}:{page}:0",
        doc_name=doc,
        page_no=page,
        text=text,
        section_hint=text[:20],
        is_pseudo_page=pseudo,
    )
    return Hit(chunk=chunk, score=0.03, dense_score=0.8, bm25_score=5.0)


HITS = [
    make_hit("handbook.pdf", 3, "Full-time employees get 24 days of paid leave."),
    make_hit("manual.pdf", 7, "Error code E3 means the device has overheated."),
    make_hit("notes.md", 2, "Pseudo-page text from a markdown file.", pseudo=True),
]


def test_passages_are_numbered_in_hit_order():
    content = build_messages([], HITS, QUESTION)[-1]["content"]
    assert "[1] (handbook.pdf, page 3)\nFull-time employees get 24 days of paid leave." in content
    assert "[2] (manual.pdf, page 7)\nError code E3 means the device has overheated." in content
    assert "[3] (notes.md, section 2)\nPseudo-page text from a markdown file." in content
    assert content.index("[1]") < content.index("[2]") < content.index("[3]")
    assert "[4]" not in content


def test_history_is_truncated_to_the_last_six_turns():
    history = [(f"question {i}", f"answer {i}") for i in range(10)]
    messages = build_messages(history, HITS, QUESTION)
    assert HISTORY_TURNS == 6
    assert len(messages) == 2 * 6 + 1
    assert messages[0] == {"role": "user", "content": "question 4"}
    assert messages[1] == {"role": "assistant", "content": "answer 4"}
    assert messages[-2] == {"role": "assistant", "content": "answer 9"}
    assert not any("question 3" in m["content"] for m in messages)
    # history turns are plain text (no passages) and roles alternate, starting with the user
    assert all("[1]" not in m["content"] for m in messages[:-1])
    assert [m["role"] for m in messages] == ["user", "assistant"] * 6 + ["user"]


def test_empty_history_gives_a_single_user_message():
    messages = build_messages([], HITS, QUESTION)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "[1] (handbook.pdf, page 3)" in messages[0]["content"]
    assert QUESTION in messages[0]["content"]


def test_question_comes_last_after_the_passages():
    messages = build_messages([("earlier question", "earlier answer")], HITS, QUESTION)
    final = messages[-1]
    assert final["role"] == "user"
    assert final["content"].rstrip().endswith(QUESTION)
    assert final["content"].rindex(QUESTION) > final["content"].index("[3]")
    # the question is only in the final message, never in the history turns
    assert all(QUESTION not in m["content"] for m in messages[:-1])


def test_weak_retrieval_sentence_is_appended():
    weak = build_messages([], HITS, QUESTION, weak_retrieval=True)[-1]["content"]
    strong = build_messages([], HITS, QUESTION)[-1]["content"]
    assert WEAK_RETRIEVAL_NOTE not in strong
    assert weak.rstrip().endswith(WEAK_RETRIEVAL_NOTE)
    assert weak.index(QUESTION) < weak.index(WEAK_RETRIEVAL_NOTE)
    assert weak.replace(WEAK_RETRIEVAL_NOTE, "").rstrip() == strong.rstrip()


def test_messages_are_plain_role_content_dicts():
    messages = build_messages([("q", "a")], HITS, QUESTION)
    for m in messages:
        assert set(m) == {"role", "content"}
        assert m["role"] in {"user", "assistant"}
        assert isinstance(m["content"], str) and m["content"].strip()


def test_blank_history_turns_are_skipped():
    history = [("", ""), ("lost question", "   "), ("kept question", "kept answer")]
    messages = build_messages(history, HITS, QUESTION)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "kept question"


def test_no_hits_still_produces_a_valid_message():
    content = build_messages([], [], QUESTION, weak_retrieval=True)[-1]["content"]
    assert "[1]" not in content
    assert QUESTION in content and WEAK_RETRIEVAL_NOTE in content


class RecordingProvider:
    """Fake LLMProvider: yields fixed deltas and records what it was asked."""

    name = "fake"

    def __init__(self, deltas):
        self.deltas, self.calls = deltas, []

    def _result(self, model):
        return LLMResult(
            text="".join(self.deltas), input_tokens=50, output_tokens=7, cache_read_tokens=0,
            model=model, elapsed_seconds=0.01, usage_estimated=False, stop_reason="stop",
        )

    def stream(self, messages, system, model, max_tokens):
        self.calls.append((list(messages), system, model, max_tokens))
        yield from self.deltas
        return self._result(model)

    def complete(self, messages, system, model, max_tokens):
        self.calls.append((list(messages), system, model, max_tokens))
        return self._result(model)


def test_answer_stream_uses_the_system_prompt_and_returns_the_result():
    provider = RecordingProvider(["24 days ", "[1]."])
    messages = build_messages([], HITS, QUESTION)
    gen = answer_stream(provider, messages, "some-model")
    deltas = []
    try:
        while True:
            deltas.append(next(gen))
    except StopIteration as stop:
        result = stop.value
    assert deltas == ["24 days ", "[1]."]
    assert result.text == "24 days [1]." and result.output_tokens == 7 and result.model == "some-model"
    assert provider.calls == [(messages, SYSTEM_PROMPT, "some-model", MAX_TOKENS)]
    assert MAX_TOKENS == 700


def test_answer_stream_requires_a_model():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        next(answer_stream(RecordingProvider([]), build_messages([], HITS, QUESTION), ""))


def drain(gen):
    deltas = []
    try:
        while True:
            deltas.append(next(gen))
    except StopIteration as stop:
        return deltas, stop.value


def test_parse_citations_accepts_fullwidth_forms_and_normalises_the_text():
    raw = "Employees get 24 days【1†L1-L3】【2†L9-L12】; cool for 30\u202fminutes\u00a0[3]. See【 9 】."
    text, valid, invalid = parse_citations(raw, n_passages=6)
    assert text == "Employees get 24 days[1][2]; cool for 30 minutes [3]. See[9]."
    assert valid == [1, 2, 3] and invalid == 1  # [9] is dropped from valid but kept (as ASCII) in the text
    assert not any(ch in text for ch in "【】†\u202f\u00a0")
    assert normalize_answer(text) == text  # idempotent


def test_answer_stream_normalises_deltas_even_when_a_citation_is_split():
    provider = RecordingProvider(["24 days", "【1†L1", "-L3】 and 30\u202fmin", "【2】."])
    deltas, result = drain(answer_stream(provider, build_messages([], HITS, QUESTION), "m"))
    assert "".join(deltas) == "24 days[1] and 30 min[2]."
    assert all("【" not in d and "】" not in d for d in deltas)
    assert result.text == "24 days[1] and 30 min[2]."  # the result text matches what was streamed


def test_answer_stream_flushes_text_held_back_at_the_end():
    provider = RecordingProvider(["see 【1", " never closed"])
    deltas, result = drain(answer_stream(provider, build_messages([], HITS, QUESTION), "m"))
    assert "".join(deltas) == "see 【1 never closed" == result.text


def test_system_prompt_requires_ascii_citations_and_is_versioned():
    assert "plain square brackets" in SYSTEM_PROMPT and "never 【1】" in SYSTEM_PROMPT
    assert "do not point the user back to the uploaded documents" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT_VERSION >= "2026-10-10.1"
