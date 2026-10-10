"""The evaluation set stays consistent with the rendered sample documents (runs the checker script)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_checker():
    spec = importlib.util.spec_from_file_location("check_questions", ROOT / "scripts" / "check_questions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_questions() -> list[dict]:
    lines = (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_every_evidence_string_is_on_exactly_its_gold_page():
    checker = load_checker()
    assert checker.check(load_questions()) == []


def test_question_set_shape():
    questions = load_questions()
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    assert len(questions) >= 50 and len(answerable) >= 36 and len(unanswerable) >= 12
    assert len({q["id"] for q in questions}) == len(questions)
    assert all(q["expected_answer"] for q in questions)
    turkish = [q for q in questions if any(ch in q["question"] for ch in "çğıöşüÇĞİÖŞÜ")]
    assert len(turkish) >= 6  # cross-lingual questions on English documents
    assert {q["doc"] for q in questions} == {"aria_x200_manual.pdf", "lumen_employee_handbook.pdf"}
