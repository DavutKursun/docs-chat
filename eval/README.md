# Evaluation set

Both documents were written for this project and describe a **fictional** company
(Lumen Analytics) and a **fictional** product (Aria X200 air purifier). They are
original work, released under the repository's MIT license, so the demo can ship
them and visitors can test the app without uploading anything.

| File | Pages | Used for |
|---|---|---|
| `lumen_employee_handbook.pdf` | 16 | HR-policy questions (leave, remote work, expenses, travel, conduct, reviews…) |
| `aria_x200_manual.pdf` | 15 | product-manual questions (filters, warranty, error codes, modes, remote control…) |

The Markdown sources are in `sample_docs/src/`; `python scripts/make_sample_docs.py` re-renders
the PDFs. The documents were extended on 2026-10-10 from 3 + 2 pages to 16 + 15 pages so that
retrieval has real distractor pages; every original paragraph was kept unchanged.

`questions.jsonl` holds 52 questions: 38 answerable (with `doc`, `page` and an `evidence` string
that occurs on exactly that page) and 14 unanswerable but plausible ones (expected answer: the
abstention sentence), including 6 in Turkish on the English documents.
`python scripts/check_questions.py` verifies the evidence strings against the rendered PDFs and
`--fix` updates page numbers after the documents change; `tests/test_eval_set.py` runs the same check.

Want a bigger, real-world test? Add any openly licensed PDF here (a government or
university handbook, a manufacturer manual that permits redistribution) and note
its source and license in this file.
