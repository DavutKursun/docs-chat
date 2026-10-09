# docs-chat — design notes

How the system works, the decisions behind it, and what the evaluation taught us.
Written for a technical reader; the README stays short.

## Pipeline

```
files ─► loaders (pages) ─► chunking (≈250 words, 50 overlap, per page)
      ─► embeddings (multilingual-e5-small, "passage: ") ─► FAISS (IndexFlatIP)
      ─► BM25Plus over the same chunks
query ─► "query: " embedding ─► dense top-20 ─┐
      ─► BM25 top-20 ───────────────────────────┴─► RRF (k=60) ─► top-6 ─► Claude
      ─► answer with [n] citations ─► rendered passages, cost, latency
```

## Decisions

- **Hybrid retrieval (dense + BM25).** Product manuals and policies are full of exact
  tokens (error codes, part numbers, amounts) where keyword search wins; paraphrased
  questions need dense search. RRF needs no score calibration between the two.
- **Chunks never cross pages** so every citation can name a page.
- **multilingual-e5-small**: small enough for the free CPU Space, handles Turkish and
  English, needs the `query:`/`passage:` prefixes.
- **BM25Plus** rather than BM25Okapi: on tiny corpora Okapi's IDF can be zero or
  negative for terms that appear in half of the documents, which drops exact matches.
- **No LangChain**: every step is a readable function with tests.
- **Page furniture** ("Page 3", "3 of 12") is stripped from PDF text so it does not
  pollute chunks and BM25 statistics.

## Evaluation

_To be filled in Milestone 4: retrieval hit rate per configuration, judge results,
abstention accuracy, cost and latency; what was tried (chunk sizes, top-k, reranker)
and what changed._

## Known limitations

- No OCR: scanned PDFs are reported, not read.
- Pseudo-pages for DOCX/TXT/MD are ~400-word sections, not real pages.
- The weak-retrieval threshold (0.75) is a placeholder for e5 cosine scores and must
  be tuned on the eval set in Milestone 4.
