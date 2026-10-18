# Evaluation: dense-openai-gpt-oss-120b

Generated 2026-10-18 22:43 UTC · retrieval **dense** · model `openai/gpt-oss-120b` via openai_compat · judge `openai/gpt-oss-120b` via openai_compat · embedder `intfloat/multilingual-e5-small` · 52 questions (38 answerable, 14 unanswerable)

| Metric | Value |
|---|---|
| Retrieval hit@3 | 0.84 |
| Retrieval hit@6 | 0.95 |
| Judge: correct / partial / wrong | 32 / 5 / 1 |
| Correct + partial | 0.97 |
| Abstention accuracy (14 unanswerable) | 1.00 |
| False abstentions (answerable) | 0.00 |
| Answers with invalid citations | 0.00 (0 dropped) |
| Weak-retrieval flag rate | 0.17 |
| Cost per answer, mean / p50 | free tier / free tier |
| Cost total (answers + judge) | free tier + free tier |
| Latency p50 / p95 | 0.9 s / 1.3 s |

## Per question

| id | type | hit@3 | hit@6 | verdict | abstained | invalid cit. | cost | latency | reason |
|---|---|---|---|---|---|---|---|---|---|
| q01 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.3 s | The system answer correctly states the 24 working days per calendar year, matching the ref |
| q02 | A | ✗ | ✓ | correct | no | 0 | free tier | 1.2 s | System answer correctly states the probation period is three months, matching the referenc |
| q03 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.9 s | The system answer correctly states both that up to three days per week can be remote and t |
| q04 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.8 s | The system answer repeats the exact allowance amount, one-time nature, and payment timing  |
| q05 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.6 s | System answer matches the reference fact that a medical certificate is required from the f |
| q06 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.6 s | System answer correctly states the length (16 weeks) and that it is fully paid, covering t |
| q07 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.7 s | The system answer matches the reference answer exactly, stating salaries are paid on the 2 |
| q08 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.7 s | The system answer correctly states that expenses above 250 EUR require prior written manag |
| q09 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.3 s | System answer correctly states the daily meal reimbursement limit as 45 EUR per day, match |
| q10 | A | ✓ | ✓ | partial | no | 0 | free tier | 1.1 s | It correctly gives the €1,200 amount but omits the additional detail of up to 5 paid learn |
| q11 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.0 s | System answer correctly states the two‑month notice period for managers after probation, m |
| q12 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.0 s | The system answer accurately repeats the reference facts about the 5-day carryover limit a |
| q13 | A | ✗ | ✗ | correct | no | 0 | free tier | 1.1 s | The system correctly states the probation period as 90 days, matching the reference's thre |
| q14 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.2 s | System answer includes the reference facts (≈6 months, replace sooner if indicator) and ad |
| q15 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.5 s | The system answer correctly states the warranty as 24 months from the date of purchase. |
| q16 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.8 s | System answer accurately repeats the reference facts about overheating and the cooling/res |
| q17 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.3 s | System answer matches the reference fact that the X200 covers rooms up to 45 square metres |
| q18 | A | ✗ | ✓ | correct | no | 0 | free tier | 1.1 s | System answer matches the reference steps and timing for resetting the filter indicator. |
| q19 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.9 s | System answer provides the same part number AF-200 as the reference. |
| q20 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.2 s | System answer correctly states the device's loudness in sleep mode as 22 dB, matching the  |
| q21 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.7 s | The system answer matches the reference, stating that a red light ring indicates poor air  |
| q22 | A | ✗ | ✓ | correct | no | 0 | free tier | 0.7 s | System answer correctly states that the HEPA filter must not be washed or get wet, matchin |
| q23 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.5 s | System correctly states that the warranty does not cover filters, matching the reference a |
| q24 | A | ✗ | ✓ | correct | no | 0 | free tier | 1.4 s | System answer includes the reference fact of replacing every 6 months or earlier if the in |
| q25 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 0.8 s |  |
| q26 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.0 s |  |
| q27 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 0.8 s |  |
| q28 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.0 s |  |
| q29 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.1 s |  |
| q30 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 0.8 s |  |
| q31 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 0.9 s |  |
| q32 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 0.6 s |  |
| q33 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.8 s | System answer accurately restates the reference details about time off in lieu and 125% pa |
| q34 | A | ✓ | ✓ | partial | no | 0 | free tier | 0.5 s | Provides the 20‑day limit but omits the required two‑week advance request detail. |
| q35 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.9 s | System answer correctly states the sabbatical eligibility after five years and the four-we |
| q36 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.9 s | System answer restates the same limits (160 EUR standard, 220 EUR in high‑cost cities) as  |
| q37 | A | ✓ | ✓ | partial | no | 0 | free tier | 0.5 s | Provides the correct mileage rate but omits the additional detail about parking and tolls  |
| q38 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.8 s | System answer restates the same fact that written warnings expire after 12 months and are  |
| q39 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.6 s | The system answer matches the reference, stating reviews occur twice a year in April and O |
| q40 | A | ✓ | ✓ | partial | no | 0 | free tier | 0.7 s | It correctly states there are two paid volunteering days per year but omits the detail tha |
| q41 | A | ✗ | ✗ | wrong | no | 0 | free tier | 0.7 s | The system answer describes using a remote control button, which contradicts the reference |
| q42 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.6 s | The system answer accurately restates the reference facts about the fan speed sensor and r |
| q43 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.8 s | System answer matches the reference facts about ~5 m range with clear line of sight. |
| q44 | A | ✓ | ✓ | partial | no | 0 | free tier | 1.2 s | System answer gives the correct reset procedure but omits the reference detail that the fi |
| q45 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.0 s | The system answer matches the reference facts about sealed original packaging, dry room-te |
| q46 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.9 s | The system answer correctly states the same temperature range (5 °C to 40 °C) as the refer |
| q47 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.2 s |  |
| q48 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.0 s |  |
| q49 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 0.9 s |  |
| q50 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 0.9 s |  |
| q51 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 0.8 s |  |
| q52 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.4 s |  |
