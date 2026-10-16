# docs-chat — design notes

How the system works, the decisions behind it, and what the evaluation taught us.
Written for a technical reader; the README stays short.

## Pipeline

```
files ─► loaders (pages) ─► chunking (≈350 words, 50 overlap, never across pages)
      ─► embeddings (multilingual-e5-small, "passage: ") ─► FAISS (IndexFlatIP)
      ─► BM25Plus over the same chunks
query ─► "query: " embedding ─► dense top-20 ─┐
      ─► BM25 top-20 ───────────────────────────┴─► RRF (k=60) ─► top-6 ─► LLM (any OpenAI-compatible API or Claude)
      ─► answer with [n] citations ─► rendered passages, cost, latency
```

## Decisions

- **Hybrid retrieval (dense + BM25).** Product manuals and policies are full of exact
  tokens (error codes, part numbers, amounts) where keyword search wins; paraphrased
  questions need dense search. RRF needs no score calibration between the two.
- **Chunks never cross pages** so every citation can name a page.
- **350-word chunks** (≈500 tokens) with a 50-word overlap, chosen on the eval set
  (see the sweep below): hit@6 0.97 against 0.92 at 250 words, with the same hit@3.
- **multilingual-e5-small**: small enough for the free CPU Space, handles Turkish and
  English, needs the `query:`/`passage:` prefixes.
- **BM25Plus** rather than BM25Okapi: on tiny corpora Okapi's IDF can be zero or
  negative for terms that appear in half of the documents, which drops exact matches.
- **Weak-retrieval signal from BM25, not from the dense score.** The LLM is told that
  retrieval was weak when `0 < best BM25 score < 15`. A best score of exactly 0 means the
  question shares no word with the documents, which is the normal case for a Turkish
  question on English documents (only dense retrieval works across languages), so it is
  never flagged. The dense cosine threshold exists but is off (`WEAK_RETRIEVAL_THRESHOLD=0`).
- **Provider-agnostic LLM layer**: one protocol, two implementations (OpenAI-compatible
  endpoints such as Groq, and Anthropic), so the demo can run on a free tier and the
  evaluation can use a stronger model as judge.
- **No LangChain**: every step is a readable function with tests.
- **Page furniture** ("Page 3", "3 of 12") is stripped from PDF text so it does not
  pollute chunks and BM25 statistics.

## Evaluation

### Set-up

- Documents: two original, fictional PDFs, 31 pages in total (`lumen_employee_handbook.pdf` 16,
  `aria_x200_manual.pdf` 15). `eval/questions.jsonl`: 52 questions, 38 answerable with document, page
  and an evidence string that occurs on exactly that page (checked by `scripts/check_questions.py` and
  `tests/test_eval_set.py`), 14 unanswerable but plausible; 6 questions in Turkish on the English documents.
- Harness: `scripts/eval.py` runs every question through `service.answer()`, the same retrieval, prompt,
  parsing and pricing as the app, then grades answerable questions with an LLM judge (correct / partial /
  wrong plus a one-sentence reason) and unanswerable questions by the exact abstention sentence.
- Model: `openai/gpt-oss-120b` on Groq's free tier, as answerer **and** judge. Self-grading is a caveat:
  a model tends to accept its own paraphrases. It is mitigated by the strict rubric, by abstention being
  checked exactly rather than judged, and by the hand check below. A second judge (Claude Sonnet via
  `--judge-provider anthropic --judge-model claude-sonnet-5-5`) is the planned upgrade.
- Rate limits matter more than cost on the free tier: 8,000 tokens per minute and 200,000 tokens per
  day for this model. A full run costs about 140k tokens (2.2k per answer, 0.5k per judge call), so one
  configuration per day is possible, and `--delay 12` keeps the per-minute limit. Latency numbers from
  the first run are inflated by Groq queueing requests when the pace was too fast (see below).

### Results

Dense-only, 250-word chunks, top-6, 2026-10-10 (`eval/results/dense-openai-gpt-oss-120b.md`):

| Metric | Value |
|---|---|
| Retrieval hit@3 | 0.76 |
| Retrieval hit@6 | 0.92 |
| Judge: correct / partial / wrong | 31 / 4 / 3 |
| Correct + partial | 0.92 |
| Abstention accuracy (14 unanswerable) | 1.00 |
| False abstentions (answerable) | 0.00 |
| Answers with invalid citations | 0.02 (1 dropped) |
| Cost per answer | free tier (2,158 input + 158 output tokens on average) |
| Latency p50 / p95 | 14.2 s / 17.0 s — Groq queueing at a 2.5 s pace; isolated calls take 0.5–1.4 s |

Of the 7 dense answers judged partial or wrong, 5 are omissions of a secondary detail (two-blocks rule,
5 learning days, two-weeks notice, parking and tolls) and 2 are retrieval misses (q20, q41) where the
gold page was not among the six passages, so the model answered from a neighbouring section.

Hybrid, 350-word chunks (the tuned defaults), top-6, 2026-10-16, paced at 12 s between calls
(`eval/results/hybrid-openai-gpt-oss-120b.md`):

| Metric | Value |
|---|---|
| Retrieval hit@3 | 0.84 |
| Retrieval hit@6 | 0.97 (only q13 misses) |
| Judge: correct / partial / wrong | 32 / 6 / 0 |
| Correct + partial | 1.00 |
| Abstention accuracy (14 unanswerable) | 1.00 |
| False abstentions (answerable) | 0.00 |
| Answers citing the gold page | 35 of 38 (0.92); 0 invalid citations |
| Weak-retrieval flags | 9 of 52, all on unanswerable questions, none on answerable |
| Cost per answer | free tier (2,409 input + 157 output tokens on average; 133k tokens for the run) |
| Latency p50 / p95 | 1.1 s / 3.1 s |

All six partial verdicts are omissions of a secondary detail (q02, q06, q10, q13, q34, q37). The two runs
are not a clean A/B: the dense run used 250-word chunks, the hybrid run the tuned 350-word index; the dense
rerun with the tuned defaults is pending the daily token window.

### Retrieval sweep (hybrid, no LLM calls)

Hit rates over the 38 answerable questions; the fused ranking does not depend on `TOP_K` (it only
truncates), so hit@3 is constant per chunk size and hit@6 at `TOP_K=4` is hit@4.

| chunk words | chunks | hit@3 | hit@4 | hit@6 | hit@8 | misses at 6 |
|---|---|---|---|---|---|---|
| 150 | 107 | 0.87 | 0.87 | 0.92 | 0.92 | q13, q20, q46 |
| 250 | 65 | 0.84 | 0.87 | 0.92 | 0.92 | q13, q20, q41 |
| **350** | 60 | 0.84 | 0.89 | **0.97** | 0.97 | q13 |

Dense-only at 250/6 also reaches hit@6 0.92 but only hit@3 0.76, so hybrid mainly improves the rank of
the gold page, not whether it is in the top six. 350-word chunks fix the two misses that became wrong
answers in the dense run. q13 (the Turkish probation question) misses at every size: BM25 sees no Turkish
word and dense ranks the welcome paragraph above the probation one. Decision: `CHUNK_WORDS=350`,
`TOP_K=6` (top-8 would not add anything at 350).

### Weak-retrieval signal

Best dense cosine among the retrieved passages, 250/6 index:

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| answerable (38) | 0.787 | 0.829 | 0.848 | 0.863 | 0.893 |
| unanswerable (14) | 0.745 | 0.804 | 0.831 | 0.850 | 0.869 |

The distributions overlap almost completely: the best threshold (0.844) flags 10 of 14 unanswerable
questions but also 14 of 38 answerable ones (balanced accuracy 0.67). The dense score is therefore not
used (`WEAK_RETRIEVAL_THRESHOLD=0`). The BM25 top score separates better (medians 23.4 vs 14.2) but
is 0 for the Turkish questions; the rule `0 < best BM25 < 15` flags 8 of 14 unanswerable questions and
none of the 38 answerable ones, and never fires on cross-lingual questions. It catches half of the
unanswerable cases, which is a useful hint for the model rather than a detector.

### Judge spot check

Ten dense-run judgments were hand-checked by Davut Kursun (q01, q06, q10, q19, q20, q34, q37, q41, q44,
q49; `hand_check` columns in the dense results JSONL): 6 of 10 agree with the judge. The judge marked q06,
q34 and q37 partial for omitting details the question did not ask for (the two-blocks rule, the notice
period, parking and tolls), and mis-judged the Turkish q44 as wrong although the answer matches the
expected one. The two real failures, q20 and q41, are retrieval misses: the gold page was not among the
passages. q19 is right but cites a non-existent passage ([7]), which the citation filter dropped. So the
self-grading judge is strict rather than lenient on this set, and "correct + partial" is the fairer
headline than "correct" alone.

### Latency under free-tier limits

Two hybrid runs on the same day show what the latency numbers mean. Paced at 12 s between calls (under
Groq's 8,000 tokens per minute), the 52-question run had a median of 1.1 s and p95 of 3.1 s per answer. An
unpaced 10-question run (`eval/results/hybrid-timing-10q.*`) took 0.9–1.8 s for the first three answers
and then 7–22 s (median 7.0 s): once the per-minute token budget is spent, Groq queues requests instead
of rejecting them. The paced figure is the model's latency and what a single user of the demo sees; the
unpaced one is the free tier under burst load. The unpaced run also stopped at its ninth question on the
200,000 tokens-per-day quota, which the full run had nearly used up.

### How to reproduce

```bash
# index with the tuned defaults (CHUNK_WORDS=350, TOP_K=6)
EMBEDDER=e5 python scripts/ingest.py eval/sample_docs --out index/

# evidence strings still on exactly one page each
python scripts/check_questions.py

# retrieval-only check (no API calls): the dry run uses a mocked provider
python scripts/eval.py --dry-run --retrieval hybrid --embedder e5 --index index/

# full runs on Groq's free tier: ~140k tokens each against a 200k tokens/day quota,
# so one configuration per day; --delay 12 stays under 8k tokens/minute
python scripts/eval.py --retrieval hybrid --index index/ --delay 12
python scripts/eval.py --retrieval dense  --index index/ --delay 12   # rerun with the tuned defaults

# with an independent judge
python scripts/eval.py --retrieval hybrid --index index/ --delay 12 \
    --judge-provider anthropic --judge-model claude-sonnet-5-5
```

Results land in `eval/results/<retrieval>-<model>.{jsonl,json,md}`; rows are appended as they come, and a
run stops cleanly when the provider reports its daily quota.

### Open items

- Dense rerun with the tuned defaults (daily token window), for a clean dense-vs-hybrid comparison.
- Reranker configuration (`BAAI/bge-reranker-v2-m3`) and the Haiku vs Sonnet comparison from the spec
  need an Anthropic key and a free CPU budget; not run yet.
- The dense run's 14 s median latency is a queueing artefact of its 2.5 s pace (see the latency note).

## Known limitations

- No OCR: scanned PDFs are reported, not read.
- Pseudo-pages for DOCX/TXT/MD are ~400-word sections, not real pages.
- The weak-retrieval hint catches about half of the unanswerable questions and is silent for
  cross-lingual questions; the abstention itself comes from the LLM, which reached 1.00 on this set.
- Groq's free tier: 8k tokens/minute and 200k tokens/day; the evaluation is paced accordingly.
