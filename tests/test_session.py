"""Session store: limits, TTL cleanup with a fake clock, clear(), and count-only logging."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from docs_chat.config import Settings
from docs_chat.embeddings import HashEmbedder
from docs_chat.session import LimitExceeded, SessionStore

TTL_HOURS = 2.0


def make_settings(**overrides) -> Settings:
    base = dict(max_files=2, max_file_mb=1, max_total_pages=3, max_questions_per_session=2, session_ttl_hours=TTL_HOURS)
    return Settings(**{**base, **overrides})


class FakeClock:
    """Starts at the real time (so directory mtimes are comparable) and only moves when told."""

    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now

    def advance(self, hours: float) -> None:
        self.now += hours * 3600


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(tmp_path: Path, clock: FakeClock) -> SessionStore:
    return SessionStore(tmp_path / "sessions", settings=make_settings(), clock=clock)


def write_text_file(path: Path, words: int, word: str = "lorem") -> Path:
    path.write_text(" ".join([word] * words), encoding="utf-8")
    return path


# --- sessions and uploads -------------------------------------------------------------------


def test_create_makes_an_isolated_directory_per_session(store, tmp_path):
    a, b = store.create(), store.create()
    assert a.id != b.id and len(store) == 2
    for s in (a, b):
        assert s.path.parent == tmp_path / "sessions"
        assert s.uploads_dir.is_dir() and s.index_dir.is_dir()
        assert s.question_count == 0 and s.index is None and s.documents == {}
    assert store.get(a.id) is a and store.get("nope") is None and store.get(None) is None


def test_add_uploads_copies_files_and_keeps_names_unique(store, tmp_path):
    session = store.create()
    src = write_text_file(tmp_path / "notes.txt", 50)
    copied = session.add_uploads([src, str(src)])  # the same name twice, as str and Path
    assert [p.name for p in copied] == ["notes.txt", "notes-2.txt"]
    assert all(p.parent == session.uploads_dir and p.read_text() == src.read_text() for p in copied)
    assert len(session.uploaded_files()) == 2


def test_max_files_is_enforced_across_batches_and_nothing_is_copied_on_failure(store, tmp_path):
    session = store.create()
    files = [write_text_file(tmp_path / f"f{i}.txt", 10) for i in range(3)]
    with pytest.raises(LimitExceeded, match=r"Too many files: 3 .* limit is 2"):
        session.add_uploads(files)
    assert session.uploaded_files() == []  # all-or-nothing
    session.add_uploads(files[:1])
    with pytest.raises(LimitExceeded, match="Too many files"):
        session.add_uploads(files[1:])  # 1 existing + 2 new > 2
    assert len(session.uploaded_files()) == 1


def test_max_file_mb_is_enforced(store, tmp_path):
    session = store.create()
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * int(1.5 * 1024 * 1024))
    with pytest.raises(LimitExceeded, match=r"File too large: big.txt is 1.5 MB, the limit is 1 MB"):
        session.add_uploads([write_text_file(tmp_path / "small.txt", 5), big])
    assert session.uploaded_files() == []  # the small file was not copied either


def test_unsupported_file_types_are_rejected_before_copying(store, tmp_path):
    session = store.create()
    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"MZ")
    with pytest.raises(ValueError, match="Unsupported file type: tool.exe"):
        session.add_uploads([exe])
    assert session.uploaded_files() == []


# --- index and pages ------------------------------------------------------------------------


def test_build_index_enforces_max_total_pages(store, two_page_pdf):
    session = store.create()
    session.add_uploads([two_page_pdf, two_page_pdf])  # 4 pages > 3
    with pytest.raises(LimitExceeded, match=r"Too many pages: 4 .* limit is 3"):
        session.build_index(HashEmbedder(dim=64))
    assert session.index is None


def test_build_index_loads_chunks_saves_and_keeps_the_index(store, two_page_pdf):
    session = store.create()
    session.add_uploads([two_page_pdf])
    report = session.build_index(HashEmbedder(dim=64))
    assert report.documents == {"two_pages.pdf": 2} == session.documents
    assert report.n_pages == 2 and report.n_chunks >= 2 and report.seconds >= 0 and report.warnings == []
    assert session.index is not None and len(session.index) == report.n_chunks
    assert {p.name for p in session.index_dir.iterdir()} == {"faiss.index", "chunks.json", "meta.json"}


def test_build_index_reports_progress_stages(store, two_page_pdf):
    session = store.create()
    session.add_uploads([two_page_pdf])
    calls = []
    session.build_index(HashEmbedder(dim=64), progress_cb=lambda f, m: calls.append((f, m)))
    fractions = [f for f, _ in calls]
    assert fractions == sorted(fractions) and 0 < fractions[0] and fractions[-1] == 1.0
    assert [m.split()[0] for _, m in calls][:3] == ["Reading", "Chunking", "Embedding"]
    assert calls[-1][1].startswith("Indexed 1 document(s), 2 page(s)")


def test_clear_uploads_removes_files_and_index_but_keeps_the_session(store, two_page_pdf):
    session = store.create()
    session.add_uploads([two_page_pdf])
    session.build_index(HashEmbedder(dim=64))
    session.clear_uploads()
    assert session.uploaded_files() == [] and session.index is None and session.documents == {}
    assert list(session.index_dir.iterdir()) == [] and session.path.exists()
    assert store.get(session.id) is session


def test_build_index_without_readable_text_is_a_clear_error(store):
    session = store.create()
    with pytest.raises(ValueError, match="No readable text"):
        session.build_index(HashEmbedder(dim=64))


# --- questions ------------------------------------------------------------------------------


def test_question_limit(store):
    session = store.create()
    assert session.next_question() == 1 and session.next_question() == 2
    with pytest.raises(LimitExceeded, match="Question limit reached: 2 questions per session"):
        session.next_question()
    assert session.question_count == 2


# --- TTL cleanup and clear ------------------------------------------------------------------


def test_cleanup_deletes_sessions_older_than_the_ttl(store, clock):
    old = store.create()
    clock.advance(1.0)
    young = store.create()  # create() runs cleanup too: nothing has expired yet
    assert store.get(old.id) is old
    clock.advance(1.5)  # old is 2.5 h, young 1.5 h; TTL 2 h
    assert store.cleanup() == 1
    assert store.get(old.id) is None and not old.path.exists()
    assert store.get(young.id) is young and young.path.exists()
    clock.advance(1.0)
    newest = store.create()  # young is 2.5 h now: swept by create()
    assert store.get(young.id) is None and not young.path.exists() and store.get(newest.id) is newest
    clock.advance(0.01)
    assert store.cleanup(ttl_hours=0.0) == 1 and len(store) == 0  # explicit TTL override


def test_cleanup_sweeps_orphan_directories_from_a_previous_process(store, clock):
    orphan = store.root / "orphan-from-last-run"
    orphan.mkdir()
    stale = clock() - 3 * 3600
    os.utime(orphan, (stale, stale))
    fresh = store.root / "fresh-orphan"
    fresh.mkdir()  # mtime is the real now, which the fake clock started from
    assert store.cleanup() == 1
    assert not orphan.exists() and fresh.exists()


def test_clear_removes_memory_and_directory(store):
    session = store.create()
    path = session.path
    store.clear(session.id)
    assert store.get(session.id) is None and not path.exists() and len(store) == 0
    store.clear(session.id)  # idempotent
    store.clear("does-not-exist-anywhere")
    store.clear(None)


def test_clear_handles_ids_from_before_a_restart_and_rejects_bad_ids(store):
    forgotten = "abcDEF123456_-xy"  # a valid id the store no longer knows (process restarted)
    (store.root / forgotten).mkdir()
    store.clear(forgotten)
    assert not (store.root / forgotten).exists()
    (store.root / "keep").mkdir()
    store.clear("../keep")  # path traversal attempt: ignored
    store.clear("keep/../keep")
    assert (store.root / "keep").exists()


# --- logging --------------------------------------------------------------------------------


def test_logs_contain_counts_but_never_file_names_or_text(store, two_page_pdf, caplog, tmp_path):
    secret = "ALPHA_SECRET_TOKEN_7731"
    src = write_text_file(tmp_path / "confidential-plan.txt", 30, secret)
    with caplog.at_level(logging.INFO, logger="docs_chat.session"):
        session = store.create()
        session.add_uploads([src, two_page_pdf])
        session.build_index(HashEmbedder(dim=64))
        session.next_question()
        store.clear(session.id)
    text = caplog.text
    assert secret not in text and "confidential-plan" not in text and "two_pages" not in text
    assert "Alpha policy" not in text  # text from the PDF fixture
    assert "2 file(s) added" in text and "indexed 2 file(s), 3 page(s)" in text and "cleared" in text
