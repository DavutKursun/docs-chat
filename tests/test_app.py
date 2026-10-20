"""UI helpers (pure functions) and a config-level smoke test — skipped where Gradio is not installed (CI)."""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

import app  # noqa: E402
from docs_chat import service  # noqa: E402
from docs_chat.config import Settings  # noqa: E402


def test_model_choices_follow_the_configuration():
    groq_only = Settings(llm_provider="openai_compat", llm_model="llama-x", llm_api_key="k", anthropic_api_key="")
    both = Settings(llm_provider="openai_compat", llm_model="llama-x", llm_api_key="k", anthropic_api_key="k")
    claude = Settings(llm_provider="anthropic", llm_model="", llm_api_key="", anthropic_api_key="k")
    no_key = Settings(llm_provider="openai_compat", llm_model="llama-x", llm_api_key="", anthropic_api_key="")
    fake = Settings(llm_provider="openai_compat", llm_model="", llm_api_key="", fake_provider=True)
    assert app.model_choices(groq_only) == ["llama-x · openai_compat"]
    assert app.model_choices(both) == ["llama-x · openai_compat", "claude-haiku-5-5 · anthropic", "claude-sonnet-5-5 · anthropic"]
    assert app.model_choices(claude) == ["claude-haiku-5-5 · anthropic", "claude-sonnet-5-5 · anthropic"]
    assert app.model_choices(no_key) == [] and not app.key_configured(no_key)
    assert app.model_choices(fake) == [app.FAKE_LABEL] and app.key_configured(fake)


def test_banner_explains_a_missing_key_or_the_fake_provider():
    assert "Demo key not configured" in app.banner_text(Settings(llm_provider="openai_compat", llm_model="m", llm_api_key=""))
    assert "Fake provider" in app.banner_text(Settings(fake_provider=True))
    assert app.banner_text(Settings(llm_provider="openai_compat", llm_model="m", llm_api_key="k")) is None


def test_resolve_model_rejects_missing_choice_and_serves_the_fake_provider():
    with pytest.raises(ValueError, match="Demo key not configured"):
        app.resolve_model(None)
    provider, model = app.resolve_model(app.FAKE_LABEL)
    assert model == "mock" and provider.name == "openai_compat"


def test_render_answer_has_chips_passages_warning_and_grey_line():
    source = service.Source(1, "handbook.pdf", 1, False, "Full-time <b>employees</b>\n\nget 24 days\nper year")
    record = service.AnswerRecord(
        text="24 days [1].", citations=[1], invalid_citations=1, sources=[source], cost="free tier", cost_usd=0.0,
        latency_seconds=1.23, weak_retrieval=True, best_dense_score=0.4, n_passages=6, model="m", truncated=False,
    )
    out = app.render_answer(record)
    assert out.startswith("24 days [1].")
    assert "<kbd>[1] handbook.pdf p.1</kbd>" in out  # citation chip
    assert "<details><summary>Show the cited passages (1)</summary>" in out
    assert "&lt;b&gt;employees&lt;/b&gt; get 24 days per year" in out  # escaped, newlines collapsed
    details = out[out.index("<details>") : out.index("</details>")]
    assert "\n" not in details
    assert "> ⚠️ **Weak retrieval:**" in out
    assert "hybrid · cost free tier · 1.2 s · 6 passages · 1 invalid citation(s) dropped" in out


def test_health_payload_shape():
    payload = app.health_payload()
    assert payload["status"] in ("ok", "degraded") and payload["retrieval"] == "hybrid"
    assert set(payload["limits"]) == {"files", "file_mb", "pages", "questions_per_session", "answers_per_day"}
    assert "answers_left_today" in payload and "sample_index" in payload


def test_build_ui_smoke_without_launching():
    """The Blocks app builds and is wired as described; nothing is launched."""
    demo = app.build_ui()
    assert isinstance(demo, gr.Blocks)
    config = demo.get_config_file()
    kinds = sorted(c["type"] for c in config["components"] if c["type"] not in ("state", "row", "column", "form"))
    assert kinds.count("button") == 3 and kinds.count("markdown") >= 3
    for kind in ("chatbot", "dataframe", "dropdown", "file", "textbox"):
        assert kind in kinds
    assert {d.get("api_name") for d in config["dependencies"]} >= {"load", "build_index", "use_samples", "ask", "clear"}
    assert config["title"] == app.TITLE
    fastapi_app = gr.routes.App.create_app(demo)  # what launch() builds; /health is added to it
    app.add_health_route(fastapi_app)
    assert any(getattr(r, "path", "") == "/health" for r in fastapi_app.routes)
