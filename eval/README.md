# Evaluation set

Both documents were written for this project and describe a **fictional** company
(Lumen Analytics) and a **fictional** product (Aria X200 air purifier). They are
original work, released under the repository's MIT license, so the demo can ship
them and visitors can test the app without uploading anything.

| File | Pages | Used for |
|---|---|---|
| `lumen_employee_handbook.pdf` | 3 | HR-policy questions (leave, remote work, expenses…) |
| `aria_x200_manual.pdf` | 2 | product-manual questions (filters, warranty, error codes…) |

The Markdown sources are in `sample_docs/src/`; `python scripts/make_sample_docs.py` re-renders
the PDFs. The evaluation questions in `questions.jsonl` reference these PDFs by
file name and page number (page numbers were verified against the extracted text).

Want a bigger, real-world test? Add any openly licensed PDF here (a government or
university handbook, a manufacturer manual that permits redistribution) and note
its source and license in this file.
