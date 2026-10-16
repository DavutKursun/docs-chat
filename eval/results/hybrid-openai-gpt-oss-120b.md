# Evaluation: hybrid-openai-gpt-oss-120b

Generated 2026-10-16 18:03 UTC · retrieval **hybrid** · model `openai/gpt-oss-120b` via openai_compat · judge `openai/gpt-oss-120b` via openai_compat · embedder `intfloat/multilingual-e5-small` · 52 questions (38 answerable, 14 unanswerable)

| Metric | Value |
|---|---|
| Retrieval hit@3 | 0.84 |
| Retrieval hit@6 | 0.97 |
| Judge: correct / partial / wrong | 32 / 6 / 0 |
| Correct + partial | 1.00 |
| Abstention accuracy (14 unanswerable) | 1.00 |
| False abstentions (answerable) | 0.00 |
| Answers with invalid citations | 0.00 (0 dropped) |
| Weak-retrieval flag rate | 0.17 |
| Cost per answer, mean / p50 | free tier / free tier |
| Cost total (answers + judge) | free tier + free tier |
| Latency p50 / p95 | 1.1 s / 3.1 s |

## Per question

| id | type | hit@3 | hit@6 | verdict | abstained | invalid cit. | cost | latency | reason |
|---|---|---|---|---|---|---|---|---|---|
| q01 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.8 s | The system answer correctly states the 24 working days per year, matching the reference. |
| q02 | A | ✓ | ✓ | partial | no | 0 | free tier | 0.9 s | The answer correctly gives the three‑month length but omits the detail about one‑week noti |
| q03 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.8 s | The answer correctly states up to three remote days per week and that Tuesday is the offic |
| q04 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.2 s | System answer accurately restates the one-time 400 EUR allowance paid with the first salar |
| q05 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.1 s | System answer correctly states that a medical certificate is required from the fourth cons |
| q06 | A | ✓ | ✓ | partial | no | 0 | free tier | 0.8 s | The answer correctly states the 16 weeks and that it is fully paid but omits the detail th |
| q07 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.6 s | The system answer states the same fact (salaries paid on the 25th of each month by bank tr |
| q08 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.0 s | System answer matches the reference stating expenses above 250 EUR require prior written m |
| q09 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.1 s | System answer matches the reference fact that the limit is 45 EUR per day. |
| q10 | A | ✓ | ✓ | partial | no | 0 | free tier | 1.0 s | The system correctly states the 1,200 EUR per year budget but omits the additional up to 5 |
| q11 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.7 s | System answer correctly states the two-month notice period for managers, matching the refe |
| q12 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.9 s | The system answer matches the reference facts exactly, stating up to 5 days and the 31 Mar |
| q13 | A | ✗ | ✗ | partial | no | 0 | free tier | 0.8 s | System gives the correct trial period length (90 days) but omits the reference detail that |
| q14 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.6 s | System answer includes the reference fact of 6‑month replacement under normal use and note |
| q15 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.8 s | System answer matches the reference warranty period of 24 months from purchase. |
| q16 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.7 s | System answer matches the reference facts exactly, stating the device overheated and to un |
| q17 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.6 s | System answer matches the reference stating rooms up to 45 square metres. |
| q18 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.8 s | The system answer matches the reference steps, stating to hold the fan button for roughly  |
| q19 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.1 s | The system answer provides the same part number AF-200 as the reference. |
| q20 | A | ✗ | ✓ | correct | no | 0 | free tier | 2.1 s | System answer provides the same 22 dB figure for sleep mode, matching the reference. |
| q21 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.8 s | System answer matches the reference, stating the same fact about poor air quality with PM2 |
| q22 | A | ✗ | ✓ | correct | no | 0 | free tier | 1.1 s | The system correctly states that the HEPA filter cannot be washed and that water damages i |
| q23 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.9 s | The system correctly states that the warranty does not cover filters, matching the referen |
| q24 | A | ✗ | ✓ | correct | no | 0 | free tier | 3.1 s | System answer includes the same core facts (replace every 6 months or sooner if indicator) |
| q25 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 0.9 s |  |
| q26 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 2.0 s |  |
| q27 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.2 s |  |
| q28 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 0.7 s |  |
| q29 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.0 s |  |
| q30 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 2.7 s |  |
| q31 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.7 s |  |
| q32 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 3.2 s |  |
| q33 | A | ✓ | ✓ | correct | no | 0 | free tier | 7.3 s | The system answer accurately restates the reference facts about time off in lieu and the 1 |
| q34 | A | ✓ | ✓ | partial | no | 0 | free tier | 1.9 s | The system answer includes the 20‑day limit but omits the requirement to request at least  |
| q35 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.4 s | System answer states the same eligibility period and duration as the reference. |
| q36 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.9 s | System answer accurately states the same reimbursement limits (160 EUR generally, 220 EUR  |
| q37 | A | ✓ | ✓ | partial | no | 0 | free tier | 0.6 s | Provides the correct mileage rate but omits the additional detail about parking and tolls  |
| q38 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.0 s | The system answer accurately states that written warnings stay on file for 12 months befor |
| q39 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.7 s | The system answer matches the reference by stating reviews occur twice a year in April and |
| q40 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.2 s | The system answer correctly states the two paid volunteering days per year as in the refer |
| q41 | A | ✗ | ✓ | correct | no | 0 | free tier | 0.6 s | The system answer correctly states that holding the timer button for 3 seconds activates t |
| q42 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.9 s | System answer restates the same facts about the fan speed sensor and the required action,  |
| q43 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.7 s | System answer matches reference facts about 5 metre range, clear line of sight, and infrar |
| q44 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.2 s | System answer includes all reference facts and adds extra correct details without contradi |
| q45 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.0 s | System answer restates the reference facts exactly, adding only consistent extra detail. |
| q46 | A | ✗ | ✓ | correct | no | 0 | free tier | 2.0 s | System answer correctly states the same temperature range (5 °C to 40 °C) as the reference |
| q47 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.0 s |  |
| q48 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 2.2 s |  |
| q49 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.4 s |  |
| q50 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.7 s |  |
| q51 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.5 s |  |
| q52 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 1.2 s |  |
