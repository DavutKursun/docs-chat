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

| Metric | Value | How measured |
|---|---|---|
| Retrieval hit rate@6 | _M4_ | 24 answerable questions over the sample docs, `scripts/eval.py` |
| Answer correctness (judge) | _M4_ | 3-level rubric, Sonnet as judge, spot-checked by hand |
| Abstention accuracy | _M4_ | 8 unanswerable questions |
| Cost per answer (Haiku 5.5) | _M4_ | measured from API usage |
| Latency | _M4_ | median, free CPU Space |

What does not work yet: scanned PDFs (no OCR), very large corpora (> 300 pages), persistent storage across sessions.

## Run it yourself

```bash
git clone https://github.com/DavutKursun/docs-chat && cd docs-chat
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your ANTHROPIC_API_KEY

python scripts/ingest.py eval/sample_docs --out index/     # downloads the embedding model on first run
python scripts/ask.py --index index/ "What does error code E3 mean?"
python app.py                                               # web UI (Milestone 3)
```

Offline smoke test without downloading any model: `EMBEDDER=hash python scripts/ingest.py eval/sample_docs --out index/` (not semantic — for tests only).

Tests: `pip install -r requirements-dev.txt && pytest` (no network, no API key).

## Stack

Python · sentence-transformers (multilingual-e5-small) · FAISS · BM25 · Claude API (Haiku 5.5 / Sonnet 5.5) · Gradio · pytest · GitHub Actions

## License

MIT
