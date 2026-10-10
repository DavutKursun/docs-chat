"""Per-session temp directories, limits and cleanup for the Gradio app (Milestone 3).

Each visitor gets a ``Session`` with its own directory under the store's root::

    <root>/<session id>/uploads/   the uploaded files (copied here, deleted with the session)
    <root>/<session id>/index/     the FAISS + BM25 index built from them

``SessionStore`` keeps the live sessions in memory, creates and clears them, and
``cleanup()`` deletes everything older than the TTL (``SESSION_TTL_HOURS``): live sessions
by creation time, and orphan directories on disk (left by a previous process) by
modification time. The app calls it on a timer; it also runs on every ``create()``.

The limits from ``config.Settings`` are enforced here, not in the UI: ``max_files``,
``max_file_mb``, ``max_total_pages`` and ``max_questions_per_session`` raise
``LimitExceeded`` (a ``ValueError``) with a message that can be shown to the user.

Logging: only counts, sizes and timings — never file names or document text.
"""

from __future__ import annotations

import logging
import re
import secrets
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_pages
from .config import Settings
from .config import settings as default_settings
from .embeddings import Embedder
from .index import DocIndex
from .loaders import SUPPORTED_SUFFIXES, load_folder

log = logging.getLogger(__name__)

MB = 1024 * 1024
ProgressCallback = Callable[[float, str], None]  # (fraction 0..1, message) for the UI's progress bar
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")  # what Session.new() generates; also guards clear()


class LimitExceeded(ValueError):
    """A per-session limit from Settings was exceeded; the message is safe to show to the user."""


def report_progress(cb: ProgressCallback | None, fraction: float, message: str) -> None:
    if cb is not None:
        cb(fraction, message)


@dataclass
class BuildReport:
    """What ``Session.build_index`` produced, for the UI's document list."""

    documents: dict[str, int]  # doc name -> pages with text
    n_pages: int
    n_chunks: int
    warnings: list[str]  # loader warnings (scanned pages, ...) to show in the UI; never logged
    seconds: float


@dataclass
class Session:
    id: str
    path: Path
    created_at: float  # epoch seconds from the store's clock
    settings: Settings = field(repr=False)
    question_count: int = 0
    index: DocIndex | None = field(default=None, repr=False)

    @classmethod
    def new(cls, root: str | Path, settings: Settings, now: float) -> Session:
        session_id = secrets.token_urlsafe(12)  # 16 URL- and filesystem-safe characters
        path = Path(root) / session_id
        (path / "uploads").mkdir(parents=True)
        (path / "index").mkdir()
        return cls(id=session_id, path=path, created_at=now, settings=settings)

    @property
    def uploads_dir(self) -> Path:
        return self.path / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.path / "index"

    @property
    def documents(self) -> dict[str, int]:
        """doc name -> pages with text, for the UI; empty until an index is built."""
        return self.index.documents if self.index is not None else {}

    def uploaded_files(self) -> list[Path]:
        if not self.uploads_dir.exists():
            return []
        return sorted(p for p in self.uploads_dir.iterdir() if p.is_file())

    def age_hours(self, now: float) -> float:
        return (now - self.created_at) / 3600

    def clear_uploads(self) -> None:
        """Delete the uploaded files and the index built from them; the session itself stays."""
        for directory in (self.uploads_dir, self.index_dir):
            if directory.exists():
                for p in directory.iterdir():
                    if p.is_file():
                        p.unlink()
        self.index = None

    # ------------------------------------------------------------------ limits
    def add_uploads(self, paths: Iterable[str | Path]) -> list[Path]:
        """Copy files into uploads/ after checking type, count and size (all-or-nothing)."""
        new = [Path(p) for p in paths]
        limits = self.settings
        total = len(self.uploaded_files()) + len(new)
        if total > limits.max_files:
            raise LimitExceeded(f"Too many files: {total} in this session, the limit is {limits.max_files}.")
        for p in new:
            if p.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Unsupported file type: {p.name} (supported: PDF, DOCX, TXT, MD).")
            size = p.stat().st_size
            if size > limits.max_file_mb * MB:
                raise LimitExceeded(
                    f"File too large: {p.name} is {size / MB:.1f} MB, the limit is {limits.max_file_mb} MB per file."
                )
        copied: list[Path] = []
        for p in new:
            dest = self._free_name(p.name)
            shutil.copyfile(p, dest)
            copied.append(dest)
        added_mb = sum(p.stat().st_size for p in copied) / MB
        log.info("session %s: %d file(s) added (%.1f MB), %d in total", self.id, len(copied), added_mb, total)
        return copied

    def _free_name(self, name: str) -> Path:
        """uploads/<name>, or <stem>-2<suffix>, <stem>-3<suffix>, ... if that name is taken."""
        base = Path(name).name  # never trust client paths
        dest = self.uploads_dir / base
        n = 2
        while dest.exists():
            dest = self.uploads_dir / f"{Path(base).stem}-{n}{Path(base).suffix}"
            n += 1
        return dest

    def build_index(self, embedder: Embedder, progress_cb: ProgressCallback | None = None) -> BuildReport:
        """Load uploads/, enforce max_total_pages, chunk, embed, save to index/ and keep the index.

        ``progress_cb(fraction, message)`` is called at each stage and after every embedding batch.
        """
        limits = self.settings
        t0 = time.perf_counter()
        report_progress(progress_cb, 0.05, "Reading documents…")
        pages, warnings = load_folder(self.uploads_dir)
        if len(pages) > limits.max_total_pages:
            raise LimitExceeded(
                f"Too many pages: {len(pages)} in total, the limit is {limits.max_total_pages} per session."
            )
        if all(p.is_empty for p in pages):  # also true when nothing was uploaded
            raise ValueError(
                "No readable text found: upload at least one PDF, DOCX, TXT or MD file with text "
                "(scanned PDFs need OCR, which is not supported)."
            )
        report_progress(progress_cb, 0.25, f"Chunking {len(pages)} page(s)…")
        chunks = chunk_pages(pages, chunk_words=limits.chunk_words, overlap_words=limits.chunk_overlap_words)
        report_progress(progress_cb, 0.3, f"Embedding {len(chunks)} chunk(s)…")
        index = DocIndex.build(
            chunks,
            embedder,
            progress_cb=lambda done, total: report_progress(
                progress_cb, 0.3 + 0.6 * done / total, f"Embedding {done}/{total} chunk(s)…"
            ),
        )
        report_progress(progress_cb, 0.95, "Saving index…")
        index.save(self.index_dir)
        self.index = index
        seconds = time.perf_counter() - t0
        report_progress(
            progress_cb,
            1.0,
            f"Indexed {len(index.documents)} document(s), {len(pages)} page(s), {len(chunks)} chunk(s) in {seconds:.1f} s",
        )
        log.info(
            "session %s: indexed %d file(s), %d page(s), %d chunk(s) in %.1f s",
            self.id, len(index.documents), len(pages), len(chunks), seconds,
        )
        return BuildReport(
            documents=index.documents, n_pages=len(pages), n_chunks=len(chunks), warnings=warnings, seconds=seconds
        )

    def next_question(self) -> int:
        """Count one question; raises LimitExceeded once max_questions_per_session is reached."""
        limit = self.settings.max_questions_per_session
        if self.question_count >= limit:
            raise LimitExceeded(
                f"Question limit reached: {limit} questions per session. Clear the session to start again."
            )
        self.question_count += 1
        return self.question_count


def _delete_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


class SessionStore:
    """Live sessions in memory plus their directories under ``root``."""

    def __init__(
        self,
        root: str | Path | None = None,
        settings: Settings = default_settings,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.root = Path(root) if root is not None else Path(settings.sessions_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()  # Gradio runs event handlers on worker threads

    def __len__(self) -> int:
        return len(self._sessions)

    def create(self) -> Session:
        self.cleanup()  # every new session sweeps expired ones, so nothing outlives the TTL for long
        session = Session.new(self.root, self.settings, self._clock())
        with self._lock:
            self._sessions[session.id] = session
            live = len(self._sessions)
        log.info("session %s created (%d live)", session.id, live)
        return session

    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def clear(self, session_id: str | None) -> None:
        """Forget the session and delete its directory; a no-op for unknown ids."""
        if not session_id or not _SESSION_ID.match(session_id):
            return
        with self._lock:
            session = self._sessions.pop(session_id, None)
        path = session.path if session is not None else self.root / session_id  # id from before a restart
        if path.exists():
            _delete_dir(path)
        log.info("session %s cleared", session_id)

    def cleanup(self, ttl_hours: float | None = None) -> int:
        """Delete sessions older than the TTL (default ``SESSION_TTL_HOURS``); returns how many."""
        ttl = (self.settings.session_ttl_hours if ttl_hours is None else ttl_hours) * 3600
        now = self._clock()
        with self._lock:
            expired = [s for s in self._sessions.values() if now - s.created_at > ttl]
            for s in expired:
                del self._sessions[s.id]
            live_paths = {s.path for s in self._sessions.values()}
            live = len(self._sessions)
        removed = 0
        for s in expired:
            _delete_dir(s.path)
            removed += 1
        # orphans: directories nobody owns (e.g. left by a previous process), judged by mtime
        for path in self.root.iterdir():
            try:
                stale = path.is_dir() and path not in live_paths and now - path.stat().st_mtime > ttl
            except OSError:  # removed concurrently
                continue
            if stale:
                _delete_dir(path)
                removed += 1
        if removed:
            log.info("cleanup: removed %d expired session(s), %d live", removed, live)
        return removed
