# docs-chat

**Upload your documents, ask questions in plain language, get answers with page citations — and "I couldn't find this" instead of made-up answers.**

<!-- M5: ![demo](assets/demo.gif) -->

🚀 **Live demo:** _coming in Milestone 5_ · 📄 **How it works:** [docs/DESIGN.md](docs/DESIGN.md)

## The problem

Companies keep their rules in PDFs nobody reads: employee handbooks, product manuals, policies. People ask the same questions in chat and email every day, and a generic chatbot answers confidently and wrongly. This app answers only from the documents you give it, shows where the answer comes from, and says when it isn't there.

## What this does

- Upload PDF, DOCX, TXT or MD files (English and Turkish supported)
- Ask questions; answers stream in with `[1]`, `[2]` citations that expand to the exact passage and page
- Hybrid search (dense + keyword) so error codes, part numbers and amounts are found exactly
- Shows cost and latency per answer; a daily budget guard protects the public demo

## Results

52 questions (38 answerable, 14 unanswerable, 6 in Turkish) over two fictional documents of 31 pages;
`openai/gpt-oss-120b` on Groq's free tier as answerer and judge; 350-word chunks, top-6, the same index
for both runs (`scripts/eval.py`, full tables and failure analysis in [docs/DESIGN.md](docs/DESIGN.md)):

| Metric | Hybrid (default) | Dense only |
|---|---|---|
| Judge: correct / correct + partial | 0.84 / 1.00 | 0.84 / 0.97 |
| Abstention accuracy (14 unanswerable) | 1.00 | 1.00 |
| Retrieval hit@6 | 0.97 | 0.95 |
| Answers citing the gold page | 0.92 | 0.89 |
| Latency p50 / p95 (paced under the free-tier limit) | 1.1 s / 3.1 s | 0.9 s / 1.3 s |
| Cost per answer | free tier | free tier |

Hand check of 10 judgments: 6 agree with the judge; the 4 disagreements are the judge being stricter than a
human reader (three "partial" verdicts for details the question did not ask for, one Turkish answer marked wrong
although it matches the reference). No wrong answer was graded correct.

What does not work yet: scanned PDFs (no OCR), very large corpora (> 300 pages), persistent storage across sessions.

## Run it yourself

```bash
git clone https://github.com/DavutKursun/docs-chat && cd docs-chat
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # Linux: add --extra-index-url https://download.pytorch.org/whl/cpu for the CPU-only torch wheel (skips the ~2 GB CUDA build)
cp .env.example .env
```

Pick the LLM in `.env`. Groq's free tier is the default; any other OpenAI-compatible endpoint (OpenRouter, Mistral, Gemini) works the same way:

```ini
LLM_PROVIDER=openai_compat
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
LLM_MODEL=openai/gpt-oss-120b    # Groq retires model names: check https://console.groq.com/docs/models
```

Alternative, Claude via the Anthropic API:

```ini
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-haiku-5-5       # or claude-sonnet-5-5
```

Then, from the command line:

```bash
python scripts/ingest.py eval/sample_docs --out index/     # downloads multilingual-e5-small (~470 MB) once; ~10 s afterwards
python scripts/ask.py --index index/ "What does error code E3 mean?"
```

Or the web UI:

```bash
python app.py                     # loads the embedder (~10 s), then serves http://127.0.0.1:7860
```

Click **Try the sample documents**, or upload your own PDF/DOCX/TXT/MD files and click **Build index**, then
ask. Each answer streams in with `[n]` citations, a collapsible **Sources** list (document, page, passage) and a
grey `cost · latency · passages` line. **Clear session** deletes your files and index; otherwise they are
deleted after 2 hours.

<!-- M5: ![docs-chat UI](assets/screenshot.png) -->

Offline smoke test without downloading any model: `EMBEDDER=hash python scripts/ingest.py eval/sample_docs --out index/` (not semantic — for tests only).

Tests: `pip install -r requirements-dev.txt && pytest` (no network, no API key, no SDK or torch needed).

## Stack

Python · sentence-transformers (multilingual-e5-small) · FAISS · BM25 · any OpenAI-compatible LLM API (Groq, OpenRouter, Mistral, Gemini) or the Claude API · Gradio · pytest · GitHub Actions

## License

MIT
