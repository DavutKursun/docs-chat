---
title: docs-chat
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Ask your documents; every answer cites document and page
---

# docs-chat

Upload PDF, DOCX, TXT or MD files and ask questions in plain language. Every answer is grounded in
the retrieved passages and cites the document and page; when the answer is not in the documents the
app says so instead of guessing. Two sample documents (a fictional employee handbook and a product
manual) are loaded when the page opens, so you can ask straight away, for example
*"How many days of annual leave do employees get?"* or *"What does error code E3 mean?"*

How it works: hybrid retrieval (multilingual-e5-small embeddings in FAISS + BM25, fused with
reciprocal rank fusion), a strict system prompt, citation checking, and an evaluation on 52
questions (hit@6 0.97, abstention 1.00). Source code, design notes and the full evaluation:
https://github.com/DavutKursun/docs-chat

## Running this Space

Free CPU hardware is enough. The embedding model (`intfloat/multilingual-e5-small`, about 470 MB)
is downloaded on the first start, so the first boot takes a couple of minutes; later boots take
about 15 seconds. The sample index is built at start-up and cached in `/tmp`.

Secrets (Settings → Variables and secrets):

| name | value |
|---|---|
| `LLM_API_KEY` | API key of an OpenAI-compatible provider (the demo uses Groq's free tier) |
| `LLM_MODEL` | model id, e.g. `openai/gpt-oss-120b` (Groq retires model names; check their list) |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` (variable, not secret) |
| `LLM_PROVIDER` | `openai_compat` (default) or `anthropic` with `ANTHROPIC_API_KEY` |
| `SAMPLE_DOCS_DIR` | `sample_docs` (variable): the two sample PDFs ship in that folder of the Space |
| `MAX_DAILY_ANSWERS` | `38` (variable): 150 answers would need ~385k tokens/day; 38 keeps the demo within half of Groq's 200k tokens/day for the model, leaving the rest for local runs |

Without a key the app starts and shows a "demo key not configured" banner; uploads and retrieval
still work, answers are disabled.

Limits on the Space (set automatically when `SPACE_ID` is present; override with variables):
3 files per session, 10 MB each, 60 pages in total, 20 questions per session, and
`MAX_DAILY_ANSWERS` answers per day for the whole Space (38 on this deployment, about 2,600 tokens
per answer against a 200k tokens/day quota shared with other uses of the key), resetting at 00:00 UTC.

Privacy: uploaded files and their index live in a per-session temporary directory and are deleted
when the session is cleared or after 2 hours; no document text is logged. `/health` reports the
state of the deployment (model, sample index, answers left today).

`requirements.txt` of this Space is the repository's `requirements-space.txt` (CPU-only torch).
