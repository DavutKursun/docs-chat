"""Evaluation metrics on synthetic data, the mock provider, the retry helper, and a dry run of scripts/eval.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from docs_chat import evaluation as ev
from docs_chat.llm import NOT_FOUND_SENTENCE, SYSTEM_PROMPT, build_messages
from docs_chat.providers import ProviderError

ROOT = Path(__file__).resolve().parents[1]


# --- metrics -------------------------------------------------------------------------------


def test_hit_at_k():
    passages = [("a.pdf", 1), ("b.pdf", 2), ("a.pdf", 3), ("a.pdf", 4)]
    assert ev.hit_at_k(passages, "a.pdf", 3, 3) is True
    assert ev.hit_at_k(passages, "a.pdf", 3, 2) is False
    assert ev.hit_at_k(passages, "a.pdf", 4, 6) is True
    assert ev.hit_at_k(passages, "b.pdf", 3, 6) is False  # same page number, other document
    assert ev.hit_at_k(passages, "a.pdf", None, 6) is False  # unanswerable: no gold page


def test_abstained_requires_the_exact_sentence():
    assert ev.abstained(NOT_FOUND_SENTENCE)
    assert ev.abstained(f"{NOT_FOUND_SENTENCE} The dress code is not described.")
    assert ev.abstained(NOT_FOUND_SENTENCE.replace("'", "’"))  # curly apostrophe from the model
    assert not ev.abstained("I could not find this in the documents.")
    assert not ev.abstained("The probation period is three months [1].")


def test_percentile_and_mean():
    assert ev.percentile([1, 2, 3, 4, 5], 50) == 3
    assert ev.percentile([1, 2, 3, 4, 5], 95) == 5
    assert ev.percentile([1, 2, 3, 4], 50) == 2
    assert ev.percentile([10], 95) == 10
    assert ev.percentile([], 50) is None
    assert ev.mean([1, 2, 3]) == 2 and ev.mean([]) is None and ev.rate([]) is None


def test_parse_judge_reads_json_and_falls_back_to_rubric_words():
    assert ev.parse_judge('{"verdict": "correct", "reason": "Same facts."}') == ("correct", "Same facts.")
    assert ev.parse_judge('Here is my grade:\n{"verdict":"Partial","reason":"Misses the \\"two blocks\\" detail"}') == (
        "partial", 'Misses the "two blocks" detail')
    assert ev.parse_judge("The system answer is wrong: it says 20 days instead of 24.")[0] == "wrong"
    assert ev.parse_judge("Partially right, so not fully correct.")[0] == "partial"  # earliest rubric word wins
    assert ev.parse_judge("The answer is incorrect.")[0] == "unknown"  # 'incorrect' is not a verdict word
    assert ev.parse_judge("")[0] == "unknown"


def make_row(**over):
    base = dict(id="q", answerable=True, hit3=True, hit6=True, verdict="correct", abstained=False,
                invalid_citations=0, weak=False, cost_usd=0.001, judge_cost_usd=0.0005, latency_s=1.0)
    return {**base, **over}


SYNTHETIC = [
    make_row(id="a1", hit3=True, hit6=True, verdict="correct", cost_usd=0.001, latency_s=1.0),
    make_row(id="a2", hit3=True, hit6=True, verdict="correct", cost_usd=0.002, latency_s=2.0, invalid_citations=1),
    make_row(id="a3", hit3=False, hit6=True, verdict="partial", cost_usd=0.003, latency_s=3.0),
    make_row(id="a4", hit3=False, hit6=False, verdict="wrong", abstained=True, cost_usd=0.004, latency_s=4.0, weak=True),
    make_row(id="u1", answerable=False, hit3=False, hit6=False, verdict=None, abstained=True, cost_usd=0.005, latency_s=5.0),
    make_row(id="u2", answerable=False, hit3=False, hit6=False, verdict=None, abstained=False, cost_usd=0.006, latency_s=6.0),
    {"id": "e1", "answerable": True, "error": "HTTP 429 (rate_limit): slow down"},
]


def test_summarize_synthetic_rows():
    s = ev.summarize(SYNTHETIC)
    assert (s["n"], s["n_ok"], s["n_errors"], s["n_answerable"], s["n_unanswerable"]) == (7, 6, 1, 4, 2)
    assert s["hit@3"] == 0.5 and s["hit@6"] == 0.75
    assert (s["correct"], s["partial"], s["wrong"], s["unknown_verdicts"]) == (2, 1, 1, 0)
    assert s["correct_rate"] == 0.5 and s["correct_or_partial_rate"] == 0.75
    assert s["abstention_accuracy"] == 0.5 and s["false_abstention_rate"] == 0.25
    assert s["invalid_citation_rate"] == pytest.approx(1 / 6) and s["invalid_citations_total"] == 1
    assert s["weak_retrieval_rate"] == pytest.approx(1 / 6)
    assert s["cost_mean_usd"] == pytest.approx(0.0035) and s["cost_p50_usd"] == 0.003
    assert s["cost_total_usd"] == pytest.approx(0.021) and s["judge_cost_total_usd"] == pytest.approx(0.003)
    assert s["latency_p50_s"] == 3.0 and s["latency_p95_s"] == 6.0 and s["latency_mean_s"] == 3.5


def test_summarize_handles_no_unanswerable_questions():
    s = ev.summarize([make_row()])
    assert s["abstention_accuracy"] is None and s["hit@6"] == 1.0 and s["n_errors"] == 0


def test_markdown_report_lists_metrics_and_questions():
    config = {"generated_at": "now", "retrieval": "hybrid", "model": "m", "provider": "mock", "judge_model": "m",
              "judge_provider": "mock", "embedder": "hash-256"}
    report = ev.markdown_report("test", config, ev.summarize(SYNTHETIC), SYNTHETIC)
    assert "| Retrieval hit@6 | 0.75 |" in report
    assert "| Judge: correct / partial / wrong | 2 / 1 / 1 |" in report
    assert "| Abstention accuracy (2 unanswerable) | 0.50 |" in report
    assert "| Latency p50 / p95 | 3.0 s / 6.0 s |" in report
    assert "| e1 | A | – | – | error |" in report and "| u1 | U |" in report


# --- mock provider and retry --------------------------------------------------------------


def test_mock_provider_answers_from_the_eval_set():
    questions = [
        {"id": "q1", "question": "How long is probation?", "answerable": True, "expected_answer": "Three months"},
        {"id": "q2", "question": "What is the dress code?", "answerable": False, "expected_answer": NOT_FOUND_SENTENCE},
    ]
    mock = ev.MockProvider(questions)
    answer = mock.complete(build_messages([], [], "How long is probation?", weak_retrieval=True), SYSTEM_PROMPT, "m", 700)
    assert answer.text == "Three months [1]" and answer.model == "m"
    assert ev.abstained(mock.complete(build_messages([], [], "What is the dress code?"), SYSTEM_PROMPT, "m", 700).text)
    assert ev.abstained(mock.complete(build_messages([], [], "Unknown question?"), SYSTEM_PROMPT, "m", 700).text)
    assert ev.parse_judge(mock.complete(ev.judge_messages("q", "ref", "ans"), ev.JUDGE_SYSTEM, "m", 400).text)[0] == "correct"
    assert mock.calls == 4


def test_call_with_retry_waits_only_for_rate_limits():
    slept: list[float] = []
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise ProviderError("HTTP 429 (rate_limit_exceeded): slow down", status_code=429)
        return "ok"

    assert ev.call_with_retry(flaky, retries=4, sleep=slept.append) == "ok"
    assert slept == [15, 30] and attempts["n"] == 3

    with pytest.raises(ProviderError, match="401"):
        ev.call_with_retry(lambda: (_ for _ in ()).throw(ProviderError("HTTP 401: bad key", status_code=401)), sleep=slept.append)
    assert slept == [15, 30]  # no waiting for a non-rate-limit error

    def always_limited():
        raise ProviderError("Rate limit reached")  # message-based detection, no status code

    with pytest.raises(ProviderError, match="Rate limit"):
        ev.call_with_retry(always_limited, retries=2, sleep=slept.append)
    assert slept == [15, 30, 15, 30]


# --- dry run of the script ------------------------------------------------------------------


def test_eval_script_dry_run_writes_results(tmp_path):
    spec = importlib.util.spec_from_file_location("eval_script", ROOT / "scripts" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = module.main(["--dry-run", "--limit", "32", "--embedder", "hash", "--out-dir", str(tmp_path), "--name", "test",
                        "--retrieval", "dense"])
    assert code == 0
    rows = [json.loads(l) for l in (tmp_path / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    result = json.loads((tmp_path / "test.json").read_text(encoding="utf-8"))
    report = (tmp_path / "test.md").read_text(encoding="utf-8")
    summary = result["summary"]
    assert len(rows) == 32 and summary["n"] == 32 and summary["n_answerable"] == 24 and summary["n_unanswerable"] == 8
    assert summary["correct_or_partial_rate"] == 1.0  # the mock answers with the reference answer
    assert summary["abstention_accuracy"] == 1.0 and summary["false_abstention_rate"] == 0.0
    assert summary["invalid_citations_total"] == 0 and 0.0 <= summary["hit@6"] <= 1.0
    assert summary["cost_total_usd"] == 0.0 and summary["latency_p95_s"] >= 0
    assert result["config"]["retrieval"] == "dense" and result["config"]["dry_run"] is True
    assert rows[0]["id"] == "q01" and rows[0]["verdict"] == "correct" and len(rows[0]["retrieved"]) == 6
    assert "# Evaluation: test" in report and "| Retrieval hit@6 |" in report and "| q32 | U |" in report
