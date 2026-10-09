"""docs-chat — Gradio app (Milestone 3).

This file is the Hugging Face Spaces entry point. Until Milestone 3 is implemented it
only shows a placeholder so the Space builds. See CLAUDE.md §6 for the UI spec:

* left column: upload (limits from config), "Build index" with progress, indexed
  documents list, model dropdown, "Clear session";
* right column: chat with streaming answers, "Sources" accordion per answer,
  grey cost/latency line;
* header: one-sentence description + "Try it: load sample documents" button.
"""

from __future__ import annotations

import gradio as gr

from docs_chat import __version__

with gr.Blocks(title="docs-chat") as demo:
    gr.Markdown(
        f"# docs-chat (v{__version__})\n\n"
        "Upload documents and ask questions — answers come with page citations.\n\n"
        "*The web UI arrives in Milestone 3. Until then use `scripts/ingest.py` and `scripts/ask.py`.*"
    )

if __name__ == "__main__":
    demo.launch()
