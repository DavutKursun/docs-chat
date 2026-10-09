.PHONY: setup setup-dev test ingest ingest-offline ask app sample-docs eval

PY ?= python

setup:            ## full install (real embeddings + LLM + UI)
	$(PY) -m pip install -r requirements.txt

setup-dev:        ## light install for tests (no torch)
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q

ingest:           ## index the sample docs with the real embedder
	$(PY) scripts/ingest.py eval/sample_docs --out index/

ingest-offline:   ## same, with the toy hash embedder (no model download)
	EMBEDDER=hash $(PY) scripts/ingest.py eval/sample_docs --out index/

ask:              ## make ask Q="your question"
	$(PY) scripts/ask.py --index index/ "$(Q)"

app:
	$(PY) app.py

sample-docs:      ## re-render eval/sample_docs/*.pdf from the markdown sources
	$(PY) scripts/make_sample_docs.py

eval:
	$(PY) scripts/eval.py
