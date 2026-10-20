"""docs-chat — the Gradio UI. Entry point for Hugging Face Spaces: ``python app.py`` serves on port 7860.

Served the standard Gradio way (``demo.launch()`` on ``GRADIO_SERVER_NAME``/``GRADIO_SERVER_PORT``), which is
what the Spaces runtime expects; the ``/health`` route is added to Gradio's own FastAPI app after launch.
Server-side rendering is off: Spaces set ``GRADIO_SSR_MODE=true``, which puts a Node front proxy on the public
port that only forwards Gradio's own paths to Python and answers everything else (``/health`` included) with
the rendered page.

Layout (CLAUDE.md §6): left column — file upload, "Build index" with progress, the indexed
documents (name, pages), the model dropdown, "Clear and use samples", "Clear session"; right
column — a chat with streaming answers; under each answer the cited passages as "doc p.N" chips,
a collapsible block with the passage text, a weak-retrieval warning when the flag is set, and a
grey "retrieval · cost · latency · N passages" line.

Public-demo behaviour: the sample documents are indexed at start-up (cached under
``SAMPLE_INDEX_DIR``, /tmp on a Space) and every new session starts with them, so visitors can ask
immediately. On a Space (``SPACE_ID`` set) secrets come from the environment only, uploads are limited
to 3 files / 10 MB / 60 pages, each session to 20 questions, and the whole deployment to
``MAX_DAILY_ANSWERS`` answers per UTC day. Without an API key the app starts anyway and shows a
"demo key not configured" banner. ``/health`` reports the state for monitoring.

All document handling goes through ``docs_chat.service`` and ``docs_chat.session``; this module maps
UI events to those calls and turns errors into ``gr.Warning`` toasts. The session id lives in ``gr.State``.
"""

from __future__ import annotations

import html
import json
import logging
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import gradio as gr

from docs_chat import service
from docs_chat.config import Settings, settings
from docs_chat.loaders import SUPPORTED_SUFFIXES
from docs_chat.providers import LLMProvider, ProviderError, get_provider
from docs_chat.session import LimitExceeded, Session, SessionStore

log = logging.getLogger("docs_chat.app")

TITLE = "docs-chat"
TAGLINE = (
    "Upload your documents (PDF, DOCX, TXT, MD) and ask questions in plain language: every answer cites "
    "the document and page, and the app says so when the answer is not in your documents. "
    "Two sample documents, an employee handbook and a product manual, are loaded already."
)
PRIVACY = "Uploaded files and their index are deleted when you clear the session or after 2 hours; no document text is logged."
RETRIEVAL_MODE = "hybrid"
CLEANUP_INTERVAL_S = 600
THINKING = "…"
FAKE_LABEL = "mock · fake"

_store: SessionStore | None = None
_providers: dict[str, LLMProvider] = {}
_lock = threading.Lock()
_samples: dict = {"index": None, "built": None, "seconds": 0.0, "error": None}


def get_store() -> SessionStore:
    global _store
    with _lock:
        if _store is None:
            _store = SessionStore()  # root = SESSIONS_DIR
        return _store


# ------------------------------------------------------------------------- configuration
def key_configured(cfg: Settings = settings) -> bool:
    """Can this deployment answer at all? (fake provider counts; it is for smoke tests)"""
    if cfg.fake_provider:
        return True
    if cfg.llm_provider == "anthropic":
        return bool(cfg.anthropic_api_key and cfg.llm_model)
    return bool(cfg.llm_api_key and cfg.llm_model)


def model_choices(cfg: Settings = settings) -> list[str]:
    """Dropdown labels "model · provider": the configured provider's model first; Claude only with a key."""
    if cfg.fake_provider:
        return [FAKE_LABEL]
    choices: list[str] = []
    if cfg.llm_model and (cfg.llm_api_key or cfg.llm_provider == "anthropic"):
        choices.append(f"{cfg.llm_model} · {cfg.llm_provider}")
    if cfg.anthropic_api_key:
        for model in cfg.llm_models_available:
            label = f"{model} · anthropic"
            if label not in choices:
                choices.append(label)
    return choices


def resolve_model(label: str | None) -> tuple[LLMProvider, str]:
    """Map a dropdown label back to (provider, model id); each provider is created once."""
    if label == FAKE_LABEL:
        with _lock:
            if "fake" not in _providers:
                from docs_chat.evaluation import MockProvider  # canned answers from the eval set

                questions = [json.loads(line) for line in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                _providers["fake"] = MockProvider(questions)
            return _providers["fake"], "mock"
    if not label or " · " not in label:
        raise ValueError("Demo key not configured: set LLM_API_KEY and LLM_MODEL (or ANTHROPIC_API_KEY) as secrets.")
    model, provider_name = label.rsplit(" · ", 1)
    with _lock:
        if provider_name not in _providers:
            _providers[provider_name] = get_provider(replace(settings, llm_provider=provider_name))
        return _providers[provider_name], model


def banner_text(cfg: Settings = settings) -> str | None:
    if cfg.fake_provider:
        return "⚠️ **Fake provider active** (smoke-test mode): answers are canned, not from an LLM."
    if not key_configured(cfg):
        return (
            "⚠️ **Demo key not configured.** Set `LLM_API_KEY` and `LLM_MODEL` (or `ANTHROPIC_API_KEY`) as Space "
            "secrets to enable answers. Uploading documents and building the index still work."
        )
    return None


def answers_left_today() -> int | None:
    return service.budget_guard(settings).remaining_calls()


def status_text() -> str:
    model = settings.llm_model or "not configured"
    provider = "fake" if settings.fake_provider else settings.llm_provider
    left = answers_left_today()
    left_text = f"{left} of {settings.max_daily_answers}" if left is not None else "unlimited"
    return (
        f"Model `{model}` via {provider} · retrieval **{RETRIEVAL_MODE}** · cost per answer: free tier · "
        f"limits: {settings.max_files} files, {settings.max_file_mb} MB each, {settings.max_total_pages} pages, "
        f"{settings.max_questions_per_session} questions per session · answers left today: {left_text}"
    )


# -------------------------------------------------------------------------- sample index
def ensure_sample_index() -> object | None:
    """Build (or load from the cache) the sample documents' index once per process."""
    with _lock:
        if _samples["index"] is None and _samples["error"] is None:
            t0 = time.perf_counter()
            try:
                index, built = service.load_or_build_sample_index(settings)
            except Exception as exc:  # the app must start even if the samples are broken
                _samples["error"] = str(exc)
                log.exception("sample index unavailable")
                return None
            _samples.update(index=index, built=built, seconds=time.perf_counter() - t0)
            log.info("sample index %s in %.1f s (%d chunks)", "built" if built else "loaded from cache", _samples["seconds"], len(index))
        return _samples["index"]


def health_payload() -> dict:
    index = _samples["index"]
    return {
        "status": "ok" if _samples["error"] is None else "degraded",
        "model": settings.llm_model or None,
        "provider": "fake" if settings.fake_provider else settings.llm_provider,
        "model_label": (model_choices() or [None])[0],
        "retrieval": RETRIEVAL_MODE,
        "embedder": settings.embedder,
        "key_configured": key_configured(),
        "fake_provider": settings.fake_provider,
        "on_space": settings.on_space,
        "sample_index": {"chunks": len(index) if index is not None else 0, "built_at_start": _samples["built"],
                         "seconds": round(_samples["seconds"], 1), "error": _samples["error"]},
        "answers_left_today": answers_left_today(),
        "limits": {"files": settings.max_files, "file_mb": settings.max_file_mb, "pages": settings.max_total_pages,
                   "questions_per_session": settings.max_questions_per_session, "answers_per_day": settings.max_daily_answers},
        "live_sessions": len(get_store()),
    }


# ---------------------------------------------------------------------------- rendering
def documents_table(session: Session | None) -> list[list]:
    return [] if session is None else [[name, pages] for name, pages in session.documents.items()]


def render_answer(record: service.AnswerRecord) -> str:
    """Answer + weak-retrieval warning + citation chips + collapsible passages + grey status line (markdown/HTML)."""
    parts = [record.text]
    if record.weak_retrieval:
        parts.append("> ⚠️ **Weak retrieval:** the question shares little with the documents, so the answer may not be in them.")
    if record.sources:
        chips = " ".join(f"<kbd>[{s.n}] {html.escape(s.doc_name)} {s.where}</kbd>" for s in record.sources)
        parts.append(f"Sources: {chips}")
        # one line per passage: a blank line inside the HTML block would end it in Markdown and break the box
        items = "".join(
            f"<p><b>[{s.n}] {html.escape(s.doc_name)} {s.where}</b><br>{html.escape(' '.join(s.text.split()))}</p>"
            for s in record.sources
        )
        parts.append(f"<details><summary>Show the cited passages ({len(record.sources)})</summary>{items}</details>")
    grey = f"{record.retrieval} · cost {record.cost} · {record.latency_seconds:.1f} s · {record.n_passages} passages"
    if record.invalid_citations:
        grey += f" · {record.invalid_citations} invalid citation(s) dropped"
    if record.truncated:
        grey += " · cut off at the length limit"
    parts.append(f'<small style="color: gray">{grey}</small>')
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- handlers
def _session(session_id: str | None) -> Session:
    """The caller's session, or a new one that starts with the sample documents."""
    store = get_store()
    session = store.get(session_id)
    if session is None:
        session = store.create()
        index = ensure_sample_index()
        if index is not None:
            service.attach_samples(session, index)
    return session


def on_load(session_id):
    session = _session(session_id)
    return session.id, documents_table(session), status_text()


def build_index(files, session_id, progress=gr.Progress()):
    """Index the uploaded files (replacing the samples); a new index starts a fresh chat."""
    session = _session(session_id)
    try:
        report = service.build_index(session, files or [], lambda fraction, msg: progress(fraction, desc=msg))
    except ValueError as exc:  # LimitExceeded, unsupported type, no readable text
        gr.Warning(str(exc))
        return session.id, documents_table(session), gr.skip(), gr.skip(), status_text()
    for warning in report.warnings:
        gr.Warning(warning)
    gr.Info(f"Indexed {len(report.documents)} document(s), {report.n_pages} page(s), {report.n_chunks} chunk(s) in {report.seconds:.1f} s")
    return session.id, documents_table(session), [], [], status_text()


def use_samples(session_id):
    """Clear the session's uploads and chat and go back to the sample documents."""
    session = _session(session_id)
    index = ensure_sample_index()
    if index is None:
        gr.Warning(f"The sample documents are not available: {_samples['error']}")
        return session.id, documents_table(session), gr.skip(), gr.skip(), status_text()
    service.attach_samples(session, index)
    gr.Info("Sample documents loaded: an employee handbook and a product manual.")
    return session.id, documents_table(session), [], [], status_text()


def clear_session(session_id):
    get_store().clear(session_id)
    gr.Info("Session cleared: uploaded files and the index were deleted. Use the samples or upload again.")
    return None, [], [], [], None, status_text()  # session id, documents table, chatbot, turns, upload widget, status


def respond(message, history, turns, session_id, model_label):
    """Generator: streams the answer into the last chatbot message. Yields (chatbot, textbox, turns, session id, status)."""
    message = (message or "").strip()
    history = list(history or [])
    turns = list(turns or [])
    if not message:
        yield history, "", turns, session_id, status_text()
        return
    session = _session(session_id)
    if session.index is None:
        gr.Warning("No documents yet: upload some and click Build index, or use the sample documents.")
        yield history, message, turns, session.id, status_text()
        return
    if not key_configured():
        gr.Warning("Demo key not configured: answers are disabled. Retrieval and uploads still work.")
        yield history, message, turns, session.id, status_text()
        return

    history += [{"role": "user", "content": message}, {"role": "assistant", "content": THINKING}]
    yield history, "", turns, session.id, status_text()
    answer = ""
    try:
        provider, model = resolve_model(model_label)
        stream = service.ask(session, message, turns, model, provider, mode=RETRIEVAL_MODE)
        while True:
            try:
                delta = next(stream)  # errors surface here, on the first next()
            except StopIteration as stop:
                record = stop.value
                break
            answer += delta
            history[-1]["content"] = answer
            yield history, "", turns, session.id, gr.skip()
    except (LimitExceeded, service.BudgetExceeded, ProviderError, ValueError) as exc:
        gr.Warning(str(exc))
        note = f"<small>⚠️ {html.escape(str(exc))}</small>"
        history[-1]["content"] = f"{answer}\n\n{note}" if answer else note
        yield history, "", turns, session.id, status_text()
        return
    history[-1]["content"] = render_answer(record)
    turns.append((message, record.text))
    yield history, "", turns, session.id, status_text()


# ----------------------------------------------------------------------------------- UI
def build_ui() -> gr.Blocks:
    choices = model_choices()
    with gr.Blocks(title=TITLE) as demo:
        session_state = gr.State(None)
        turns_state = gr.State([])  # (question, answer) pairs for the prompt history, without sources

        gr.Markdown(f"# {TITLE}\n{TAGLINE}")
        banner = banner_text()
        if banner:
            gr.Markdown(banner)
        status_md = gr.Markdown(status_text())

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                upload = gr.File(
                    label=f"Your documents (up to {settings.max_files} files, {settings.max_file_mb} MB each, "
                    f"{settings.max_total_pages} pages in total)",
                    file_count="multiple",
                    file_types=sorted(SUPPORTED_SUFFIXES),
                    type="filepath",
                )
                build_btn = gr.Button("Build index", variant="primary")
                docs_table = gr.Dataframe(
                    headers=["Document", "Pages"],
                    datatype=["str", "number"],
                    value=[],
                    interactive=False,
                    label="Indexed documents",
                    wrap=True,
                )
                model_dd = gr.Dropdown(
                    choices=choices, value=choices[0] if choices else None, label="Model", interactive=True
                )
                samples_btn = gr.Button("Clear and use samples")
                clear_btn = gr.Button("Clear session")
                gr.Markdown(f"<small>{PRIVACY}</small>")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=540, sanitize_html=True, render_markdown=True)
                question = gr.Textbox(
                    label="Question", placeholder="Ask about the documents, e.g. “How many days of annual leave do employees get?”",
                    submit_btn=True,
                )

        index_outputs = [session_state, docs_table, chatbot, turns_state, status_md]
        demo.load(on_load, inputs=[session_state], outputs=[session_state, docs_table, status_md], api_name="load")
        build_btn.click(build_index, inputs=[upload, session_state], outputs=index_outputs, api_name="build_index")
        samples_btn.click(use_samples, inputs=[session_state], outputs=index_outputs, api_name="use_samples")
        question.submit(
            respond,
            inputs=[question, chatbot, turns_state, session_state, model_dd],
            outputs=[chatbot, question, turns_state, session_state, status_md],
            api_name="ask",
        )
        clear_btn.click(
            clear_session,
            inputs=[session_state],
            outputs=[session_state, docs_table, chatbot, turns_state, upload, status_md],
            api_name="clear",
        )
    return demo


def add_health_route(fastapi_app) -> None:
    """``GET /health`` on Gradio's FastAPI app (``demo.app`` once launched): JSON for monitoring."""
    fastapi_app.add_api_route("/health", lambda: health_payload(), methods=["GET"], include_in_schema=False)


def start_cleanup_timer(interval_s: float = CLEANUP_INTERVAL_S) -> threading.Thread:
    def loop() -> None:
        while True:
            time.sleep(interval_s)
            try:
                get_store().cleanup()
            except Exception:  # never let the timer die
                log.exception("session cleanup failed")

    thread = threading.Thread(target=loop, name="session-cleanup", daemon=True)
    thread.start()
    return thread


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info(
        "space=%s provider=%s model=%r embedder=%s key_configured=%s limits=%s/%s MB/%s pages, %s answers/day",
        settings.on_space, settings.llm_provider, settings.llm_model, settings.embedder, key_configured(),
        settings.max_files, settings.max_file_mb, settings.max_total_pages, settings.max_daily_answers,
    )
    t0 = time.perf_counter()
    service.shared_embedder()  # ~8 s for e5-small on CPU, paid once here instead of on the first question
    log.info("embedder ready in %.1f s", time.perf_counter() - t0)
    ensure_sample_index()
    get_store().cleanup()
    start_cleanup_timer()
    demo = build_ui()
    demo.queue(default_concurrency_limit=4)
    port_env = os.environ.get("GRADIO_SERVER_PORT")  # 7860 on a Space; locally Gradio picks the next free port
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(port_env) if port_env else None,
        prevent_thread_lock=True,
        show_error=False,
        quiet=True,
        ssr_mode=False,  # keep uvicorn on the public port so /health is reachable (see the module docstring)
    )
    add_health_route(demo.app)
    log.info("serving on %s (health: /health)", demo.local_url)
    demo.block_thread()


if __name__ == "__main__":
    main()
