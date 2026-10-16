# Evaluation: hybrid-timing-10q

Generated 2026-10-16 18:05 UTC · retrieval **hybrid** · model `openai/gpt-oss-120b` via openai_compat · judge `openai/gpt-oss-120b` via openai_compat · embedder `intfloat/multilingual-e5-small` · 9 questions (8 answerable, 0 unanswerable, 1 failed)

| Metric | Value |
|---|---|
| Retrieval hit@3 | 1.00 |
| Retrieval hit@6 | 1.00 |
| Judge: correct / partial / wrong | 7 / 1 / 0 |
| Correct + partial | 1.00 |
| Abstention accuracy (0 unanswerable) | n/a |
| False abstentions (answerable) | 0.00 |
| Answers with invalid citations | 0.00 (0 dropped) |
| Weak-retrieval flag rate | 0.00 |
| Cost per answer, mean / p50 | free tier / free tier |
| Cost total (answers + judge) | free tier + free tier |
| Latency p50 / p95 | 7.0 s / 22.1 s |

## Per question

| id | type | hit@3 | hit@6 | verdict | abstained | invalid cit. | cost | latency | reason |
|---|---|---|---|---|---|---|---|---|---|
| q01 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.8 s | System answer provides the same 24 working days per year as the reference, matching the ke |
| q02 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.9 s | System correctly states the probation period is three months, matching the reference answe |
| q03 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.9 s | System answer accurately states both the remote work limit and the office day matching the |
| q04 | A | ✓ | ✓ | partial | no | 0 | free tier | 16.4 s | Provides the correct one-time 400 EUR amount but omits the detail that it is paid with the |
| q05 | A | ✓ | ✓ | correct | no | 0 | free tier | 19.9 s | The system answer matches the reference, stating a medical certificate is required from th |
| q06 | A | ✓ | ✓ | correct | no | 0 | free tier | 7.0 s | The system answer matches all reference facts: 16 weeks, fully paid, can be taken in up to |
| q07 | A | ✓ | ✓ | correct | no | 0 | free tier | 18.7 s | System answer correctly states the salaries are paid on the 25th, matching the reference's |
| q08 | A | ✓ | ✓ | correct | no | 0 | free tier | 22.1 s | The system answer correctly states that expenses exceeding 250 EUR require prior written m |
| q09 | A | – | – | error | – | – | – | – | HTTP 429 (rate_limit_exceeded): Rate limit reached for model `openai/gpt-oss-120 |
