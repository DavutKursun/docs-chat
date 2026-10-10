"""UI helpers (pure functions) — skipped where Gradio is not installed (CI)."""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

import app  # noqa: E402
from docs_chat import service  # noqa: E402
from docs_chat.config import Settings  # noqa: E402


def test_model_choices_follow_the_configuration():
    groq_only = Settings(llm_provider="openai_compat", llm_model="llama-x", anthropic_api_key="")
    both = Settings(llm_provider="openai_compat", llm_model="llama-x", anthropic_api_key="k")
    claude = Settings(llm_provider="anthropic", llm_model="", anthropic_api_key="k")
    nothing = Settings(llm_provider="openai_compat", llm_model="", anthropic_api_key="")
    assert app.model_choices(groq_only) == ["llama-x · openai_compat"]
    assert app.model_choices(both) == ["llama-x · openai_compat", "claude-haiku-5-5 · anthropic", "claude-sonnet-5-5 · anthropic"]
    assert app.model_choices(claude) == ["claude-haiku-5-5 · anthropic", "claude-sonnet-5-5 · anthropic"]
    assert app.model_choices(nothing) == []


def test_resolve_model_rejects_missing_choice():
    with pytest.raises(ValueError, match="Choose a model"):
        app.resolve_model(None)


def test_render_answer_has_sources_block_and_grey_line():
    source = service.Source(1, "handbook.pdf", 1, False, "Full-time <b>employees</b>\n\nget 24 days\nper year")
    record = service.AnswerRecord(
        text="24 days [1].", citations=[1], invalid_citations=1, sources=[source], cost="free tier", cost_usd=0.0,
        latency_seconds=1.23, weak_retrieval=True, best_dense_score=0.4, n_passages=6, model="m", truncated=False,
    )
    out = app.render_answer(record)
    assert out.startswith("24 days [1].")
    assert "<details><summary>Sources (1)</summary>" in out and "[1] handbook.pdf p.1" in out
    assert "&lt;b&gt;employees&lt;/b&gt; get 24 days per year" in out  # escaped, and newlines collapsed
    details = out[out.index("<details>") : out.index("</details>")]
    assert "\n" not in details  # a blank line inside the HTML block would break the collapsible box
    assert "cost free tier · 1.2 s · 6 passages · 1 invalid citation(s) dropped" in out
    assert "retrieval was weak" in out


def test_build_ui_smoke_without_launching():
    """The Blocks app builds and is wired as CLAUDE.md §6 describes; nothing is launched."""
    demo = app.build_ui()
    assert isinstance(demo, gr.Blocks)
    config = demo.get_config_file()
    components = sorted(c["type"] for c in config["components"] if c["type"] not in ("state", "row", "column", "form"))
    assert components == ["button", "button", "button", "chatbot", "dataframe", "dropdown", "file", "markdown", "textbox"]
    assert [d.get("api_name") for d in config["dependencies"]] == ["build_index", "build_sample", "ask", "clear"]
    assert config["title"] == app.TITLE
