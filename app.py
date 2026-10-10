"""docs-chat — the Gradio UI (Milestone 3). Entry point for Hugging Face Spaces: ``python app.py``.

Layout (CLAUDE.md §6): left column — file upload, "Build index" with progress, the indexed
documents (name, pages), the model dropdown and "Clear session"; right column — a chat with
streaming answers, a collapsible "Sources" block and a grey "cost · latency · N passages"
line under each answer. The header has a one-sentence description and a button that indexes
the sample documents so visitors can try the app without uploading anything.

All document handling goes through ``docs_chat.service`` and ``docs_chat.session``; this
module only maps UI events to those calls and turns errors into ``gr.Warning`` toasts. The
session id lives in ``gr.State``. Per-session uploads and indexes are deleted on
"Clear session" or after SESSION_TTL_HOURS (cleanup runs on every new session and on a timer).
"""

from __future__ import annotations

import html
import logging
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
    "the document and page, and the app says so when the answer is not in your documents."
)
CLEANUP_INTERVAL_S = 600
THINKING = "…"

_store: SessionStore | None = None
_providers: dict[str, LLMProvider] = {}
_lock = threading.Lock()


def get_store() -> SessionStore:
    global _store
    with _lock:
        if _store is None:
            _store = SessionStore()  # root = SESSIONS_DIR
        return _store


# ------------------------------------------------------------------------------ models
def model_choices(cfg: Settings = settings) -> list[str]:
    """Dropdown labels "model · provider": the configured provider's model first; Claude only with a key."""
    choices: list[str] = []
    if cfg.llm_model:
        choices.append(f"{cfg.llm_model} · {cfg.llm_provider}")
    if cfg.anthropic_api_key:
        for model in cfg.llm_models_available:
            label = f"{model} · anthropic"
            if label not in choices:
                choices.append(label)
    return choices


def resolve_model(label: str | None) -> tuple[LLMProvider, str]:
    """Map a dropdown label back to (provider, model id); each provider is created once."""
    if not label or " · " not in label:
        raise ValueError("Choose a model first (set LLM_MODEL in .env, or ANTHROPIC_API_KEY for Claude).")
    model, provider_name = label.rsplit(" · ", 1)
    with _lock:
        if provider_name not in _providers:
            _providers[provider_name] = get_provider(replace(settings, llm_provider=provider_name))
        return _providers[provider_name], model


# ---------------------------------------------------------------------------- rendering
def documents_table(session: Session | None) -> list[list]:
    return [] if session is None else [[name, pages] for name, pages in session.documents.items()]


def render_answer(record: service.AnswerRecord) -> str:
    """Answer + weak-retrieval note + collapsible Sources + grey cost line, as markdown with inline HTML."""
    parts = [record.text]
    if record.weak_retrieval:
        parts.append("<small>⚠️ retrieval was weak: the answer may not be in the documents</small>")
    if record.sources:
        # one line per passage: a blank line inside the HTML block would end it in Markdown and break the box
        items = "".join(
            f"<p><b>[{s.n}] {html.escape(s.doc_name)} {s.where}</b><br>{html.escape(' '.join(s.text.split()))}</p>"
            for s in record.sources
        )
        parts.append(f"<details><summary>Sources ({len(record.sources)})</summary>{items}</details>")
    grey = f"cost {record.cost} · {record.latency_seconds:.1f} s · {record.n_passages} passages"
    if record.invalid_citations:
        grey += f" · {record.invalid_citations} invalid citation(s) dropped"
    if record.truncated:
        grey += " · cut off at the length limit"
    parts.append(f'<small style="color: gray">{grey}</small>')
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- handlers
def _session(session_id: str | None) -> Session:
    """The caller's session, or a new one (creating one also sweeps expired sessions)."""
    store = get_store()
    session = store.get(session_id)
    return session if session is not None else store.create()


def build_index(files, session_id, progress=gr.Progress()):
    """Index the uploaded files; returns (session id, documents table, chatbot, turns) — a new index starts a fresh chat."""
    session = _session(session_id)
    try:
        report = service.build_index(session, files or [], lambda fraction, msg: progress(fraction, desc=msg))
    except ValueError as exc:  # LimitExceeded, unsupported type, no readable text
        gr.Warning(str(exc))
        return session.id, documents_table(session), gr.skip(), gr.skip()
    for warning in report.warnings:
        gr.Warning(warning)
    gr.Info(
        f"Indexed {len(report.documents)} document(s), {report.n_pages} page(s), "
        f"{report.n_chunks} chunk(s) in {report.seconds:.1f} s"
    )
    return session.id, documents_table(session), [], []


def build_sample_index(session_id, progress=gr.Progress()):
    folder = Path(settings.sample_docs_dir)
    files = sorted(str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    return build_index(files, session_id, progress)


def respond(message, history, turns, session_id, model_label):
    """Generator: streams the answer into the last chatbot message. Yields (chatbot, textbox, turns, session id)."""
    message = (message or "").strip()
    history = list(history or [])
    turns = list(turns or [])
    if not message:
        yield history, "", turns, session_id
        return
    session = get_store().get(session_id)
    if session is None or session.index is None:
        gr.Warning("No index yet: upload documents and click Build index, or try the sample documents.")
        yield history, message, turns, session_id  # keep the question in the box
        return

    history += [{"role": "user", "content": message}, {"role": "assistant", "content": THINKING}]
    yield history, "", turns, session.id  # the placeholder covers the model's reasoning time before the first token
    answer = ""
    try:
        provider, model = resolve_model(model_label)
        stream = service.ask(session, message, turns, model, provider)
        while True:
            try:
                delta = next(stream)  # errors surface here, on the first next()
            except StopIteration as stop:
                record = stop.value
                break
            answer += delta
            history[-1]["content"] = answer
            yield history, "", turns, session.id
    except (LimitExceeded, service.BudgetExceeded, ProviderError, ValueError) as exc:
        gr.Warning(str(exc))
        note = f"<small>⚠️ {html.escape(str(exc))}</small>"
        history[-1]["content"] = f"{answer}\n\n{note}" if answer else note
        yield history, "", turns, session.id
        return
    history[-1]["content"] = render_answer(record)
    turns.append((message, record.text))
    yield history, "", turns, session.id


def clear_session(session_id):
    get_store().clear(session_id)
    gr.Info("Session cleared: uploaded files and the index were deleted.")
    return None, [], [], [], None  # session id, documents table, chatbot, turns, upload widget


# ----------------------------------------------------------------------------------- UI
def build_ui() -> gr.Blocks:
    choices = model_choices()
    with gr.Blocks(title=TITLE) as demo:
        session_state = gr.State(None)
        turns_state = gr.State([])  # (question, answer) pairs for the prompt history, without sources

        with gr.Row():
            gr.Markdown(f"# {TITLE}\n{TAGLINE}")
            sample_btn = gr.Button("Try the sample documents", variant="secondary", scale=0, min_width=220)

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                upload = gr.File(
                    label=f"Documents (up to {settings.max_files} files, {settings.max_file_mb} MB each, "
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
                clear_btn = gr.Button("Clear session")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=540, sanitize_html=True, render_markdown=True)
                question = gr.Textbox(
                    label="Question", placeholder="Ask a question about your documents…", submit_btn=True
                )

        index_outputs = [session_state, docs_table, chatbot, turns_state]
        build_btn.click(build_index, inputs=[upload, session_state], outputs=index_outputs, api_name="build_index")
        sample_btn.click(build_sample_index, inputs=[session_state], outputs=index_outputs, api_name="build_sample")
        question.submit(
            respond,
            inputs=[question, chatbot, turns_state, session_state, model_dd],
            outputs=[chatbot, question, turns_state, session_state],
            api_name="ask",
        )
        clear_btn.click(
            clear_session,
            inputs=[session_state],
            outputs=[session_state, docs_table, chatbot, turns_state, upload],
            api_name="clear",
        )
    return demo


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
        "provider=%s model=%r embedder=%s dropdown=%s", settings.llm_provider, settings.llm_model, settings.embedder,
        model_choices(),
    )
    t0 = time.perf_counter()
    service.shared_embedder()  # ~8 s for e5-small on CPU, paid once here instead of on the first question
    log.info("embedder ready in %.1f s", time.perf_counter() - t0)
    get_store().cleanup()
    start_cleanup_timer()
    demo = build_ui()
    demo.queue(default_concurrency_limit=4).launch(show_error=False)


if __name__ == "__main__":
    main()
