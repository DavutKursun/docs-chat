# Evaluation: dense-openai-gpt-oss-120b

Generated 2026-10-10 21:24 UTC · retrieval **dense** · model `openai/gpt-oss-120b` via openai_compat · judge `openai/gpt-oss-120b` via openai_compat · embedder `intfloat/multilingual-e5-small` · 52 questions (38 answerable, 14 unanswerable)

| Metric | Value |
|---|---|
| Retrieval hit@3 | 0.76 |
| Retrieval hit@6 | 0.92 |
| Judge: correct / partial / wrong | 31 / 4 / 3 |
| Correct + partial | 0.92 |
| Abstention accuracy (14 unanswerable) | 1.00 |
| False abstentions (answerable) | 0.00 |
| Answers with invalid citations | 0.02 (1 dropped) |
| Weak-retrieval flag rate | 0.02 |
| Cost per answer, mean / p50 | free tier / free tier |
| Cost total (answers + judge) | free tier + free tier |
| Latency p50 / p95 | 14.2 s / 17.0 s |

## Per question

| id | type | hit@3 | hit@6 | verdict | abstained | invalid cit. | cost | latency | reason |
|---|---|---|---|---|---|---|---|---|---|
| q01 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.6 s | The system correctly states the 24 working days per year, matching the reference answer. |
| q02 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.7 s | System correctly states the probation period is three months, matching the reference. |
| q03 | A | ✓ | ✓ | correct | no | 0 | free tier | 0.5 s | System answer matches the reference facts about up to three remote days and Tuesday as the |
| q04 | A | ✓ | ✓ | correct | no | 0 | free tier | 1.0 s | System answer accurately repeats the reference facts about the one-time 400 EUR allowance  |
| q05 | A | ✓ | ✓ | correct | no | 0 | free tier | 7.7 s | System answer matches the reference fact that a medical certificate is required from the f |
| q06 | A | ✓ | ✓ | partial | no | 0 | free tier | 14.9 s | Provides correct length and payment status but omits the detail about splitting into up to |
| q07 | A | ✓ | ✓ | correct | no | 0 | free tier | 14.4 s | System answer correctly states salaries are paid on the 25th, matching the reference. |
| q08 | A | ✗ | ✓ | correct | no | 0 | free tier | 12.3 s | System answer matches the reference fact that expenses above 250 EUR require prior written |
| q09 | A | ✓ | ✓ | correct | no | 0 | free tier | 13.0 s | The system answer correctly states the reimbursement limit of up to 45 EUR per day. |
| q10 | A | ✓ | ✓ | partial | no | 0 | free tier | 11.8 s | System answer includes the 1,200 EUR per year but omits the additional up to 5 paid learni |
| q11 | A | ✓ | ✓ | correct | no | 0 | free tier | 15.2 s | System correctly states the two‑month notice period for managers, matching the reference a |
| q12 | A | ✓ | ✓ | correct | no | 0 | free tier | 13.8 s | The system answer matches the reference, stating up to 5 days can be carried over and must |
| q13 | A | ✗ | ✗ | correct | no | 0 | free tier | 13.5 s | System answer correctly states the probation period as 90 days, matching the three‑month d |
| q14 | A | ✗ | ✓ | correct | no | 0 | free tier | 16.4 s | The answer includes the reference facts (≈6 months under normal use, replace sooner if ind |
| q15 | A | ✓ | ✓ | correct | no | 0 | free tier | 14.8 s | The system answer correctly states the warranty period as 24 months from the date of purch |
| q16 | A | ✓ | ✓ | correct | no | 0 | free tier | 13.9 s | The system answer matches the reference facts exactly, stating the device overheated and t |
| q17 | A | ✓ | ✓ | correct | no | 0 | free tier | 17.0 s | The system answer correctly states the same fact that the X200 covers rooms up to 45 squar |
| q18 | A | ✗ | ✓ | correct | no | 0 | free tier | 13.0 s | The system answer correctly states to press and hold the fan button for about 5 seconds un |
| q19 | A | ✗ | ✓ | correct | no | 1 | free tier | 17.6 s | The system answer correctly states the part number AF‑200, matching the reference. |
| q20 | A | ✗ | ✗ | wrong | no | 0 | free tier | 15.2 s | The system answer does not provide the required numeric loudness (22 dB) and gives only a  |
| q21 | A | ✓ | ✓ | correct | no | 0 | free tier | 15.2 s | System answer correctly states that a red light ring indicates poor air quality with PM2.5 |
| q22 | A | ✗ | ✓ | correct | no | 0 | free tier | 17.1 s | The system answer correctly states that the HEPA filter should not be washed because water |
| q23 | A | ✓ | ✓ | correct | no | 0 | free tier | 11.1 s | The system answer correctly states that the warranty does not cover filters, matching the  |
| q24 | A | ✗ | ✓ | correct | no | 0 | free tier | 16.3 s | System answer includes the reference fact about changing the filter every 6 months or earl |
| q25 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 16.3 s |  |
| q26 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 16.3 s |  |
| q27 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 12.2 s |  |
| q28 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 15.4 s |  |
| q29 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 16.3 s |  |
| q30 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 16.1 s |  |
| q31 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 15.0 s |  |
| q32 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 14.2 s |  |
| q33 | A | ✓ | ✓ | correct | no | 0 | free tier | 13.3 s | The system answer restates the reference facts accurately, covering both time off in lieu  |
| q34 | A | ✓ | ✓ | partial | no | 0 | free tier | 14.5 s | Provides the correct limit of 20 days but omits the required advance notice detail. |
| q35 | A | ✓ | ✓ | correct | no | 0 | free tier | 12.5 s | The answer correctly states the sabbatical eligibility after five years and the duration o |
| q36 | A | ✓ | ✓ | correct | no | 0 | free tier | 12.8 s | The answer repeats the reference limits (160 EUR standard, 220 EUR in high-cost cities) ac |
| q37 | A | ✓ | ✓ | partial | no | 0 | free tier | 9.2 s | Provides the correct mileage rate but omits the additional detail about parking and tolls  |
| q38 | A | ✓ | ✓ | correct | no | 0 | free tier | 13.7 s | The system answer correctly states that written warnings expire after 12 months and are th |
| q39 | A | ✓ | ✓ | correct | no | 0 | free tier | 11.7 s | The system answer matches the reference answer, stating reviews occur twice a year in Apri |
| q40 | A | ✓ | ✓ | correct | no | 0 | free tier | 13.0 s | System answer correctly states there are two paid volunteering days per year, matching the |
| q41 | A | ✗ | ✗ | wrong | no | 0 | free tier | 13.6 s | The system answer gives a different method (remote control button) and does not mention ho |
| q42 | A | ✓ | ✓ | correct | no | 0 | free tier | 8.5 s | System answer correctly restates the reference facts about the fan speed sensor and the re |
| q43 | A | ✓ | ✓ | correct | no | 0 | free tier | 14.4 s | System answer correctly states the ~5 metre range with clear line of sight, matching the r |
| q44 | A | ✓ | ✓ | wrong | no | 0 | free tier | 14.7 s | System answer claims all settings reset, contradicting reference that the filter indicator |
| q45 | A | ✓ | ✓ | correct | no | 0 | free tier | 16.6 s | System answer matches the reference facts about storage conditions and duration. |
| q46 | A | ✓ | ✓ | correct | no | 0 | free tier | 14.5 s | System answer provides the same temperature range (5 °C to 40 °C) as the reference. |
| q47 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 10.2 s |  |
| q48 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 15.7 s |  |
| q49 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 15.5 s |  |
| q50 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 15.6 s |  |
| q51 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 12.5 s |  |
| q52 | U | ✗ | ✗ | abstained | yes | 0 | free tier | 14.4 s |  |
