#!/usr/bin/env python
"""Milestone 4 — draft evaluation questions from the sample documents.

Target usage:
    python scripts/make_eval_set.py eval/sample_docs --out eval/questions.draft.jsonl --n 40

For each document, send pages to claude-sonnet-5-5 (structured output) and ask for
answerable questions with the exact answer and the page, plus plausible questions the
document does NOT answer. Then a human (Davut) reviews the draft and copies the good
ones into eval/questions.jsonl. Never commit a draft as the final set without review.

The current eval/questions.jsonl (32 questions) was written by hand for the two
original sample documents; page numbers were verified programmatically.
"""

if __name__ == "__main__":
    raise SystemExit("TODO (M4): implement question drafting; see module docstring.")
