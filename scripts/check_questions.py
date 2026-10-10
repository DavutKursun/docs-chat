#!/usr/bin/env python
"""Check eval/questions.jsonl against the rendered sample PDFs.

For every answerable question the ``evidence`` string must occur on exactly one page of its
document (compared case-insensitively on whitespace-normalised page text, the same text the
loader gives the chunker), and ``page`` must point to that page. Unanswerable questions must
have ``page: null`` and ``evidence: null``. Ids must be unique.

    python scripts/check_questions.py          # report; exit 1 on any problem
    python scripts/check_questions.py --fix    # also rewrite page numbers that moved
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docs_chat.loaders import load_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "eval" / "questions.jsonl"
DOCS = ROOT / "eval" / "sample_docs"


def normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def page_texts(doc_name: str) -> dict[int, str]:
    return {page.page_no: normalise(page.text) for page in load_file(DOCS / doc_name)}


def check(questions: list[dict], fix: bool = False) -> list[str]:
    """Return the list of problems (empty = consistent); with ``fix`` update moved page numbers in place."""
    problems: list[str] = []
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids: " + ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    texts: dict[str, dict[int, str]] = {}
    for q in questions:
        if not q["answerable"]:
            if q.get("page") is not None or q.get("evidence"):
                problems.append(f"{q['id']}: unanswerable questions must have page=null and evidence=null")
            continue
        if not q.get("evidence"):
            problems.append(f"{q['id']}: answerable question without evidence")
            continue
        if q["doc"] not in texts:
            texts[q["doc"]] = page_texts(q["doc"])
        hits = [n for n, text in texts[q["doc"]].items() if normalise(q["evidence"]) in text]
        if len(hits) != 1:
            problems.append(f"{q['id']}: evidence {q['evidence']!r} found on pages {hits} of {q['doc']} (need exactly one)")
        elif q.get("page") != hits[0]:
            if fix:
                q["page"] = hits[0]
            else:
                problems.append(f"{q['id']}: page is {q.get('page')} but the evidence is on page {hits[0]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="rewrite page numbers that moved")
    args = parser.parse_args(argv)

    questions = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    problems = check(questions, fix=args.fix)
    if args.fix:
        QUESTIONS.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    answerable = sum(q["answerable"] for q in questions)
    print(f"{len(questions)} questions: {answerable} answerable, {len(questions) - answerable} unanswerable")
    for problem in problems:
        print("PROBLEM:", problem)
    print("OK: every evidence string is on exactly one page and page numbers match" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
