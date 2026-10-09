"""Split pages into retrieval chunks.

Rules (see CLAUDE.md §5):
* chunks never cross a page boundary;
* a chunk holds roughly ``chunk_words`` words (word count is a cheap proxy for tokens;
  250 words ≈ 350 tokens for English);
* consecutive chunks of the same page overlap by ``overlap_words`` words so that a
  sentence cut at a boundary is still retrievable;
* paragraphs are kept together when they fit; an oversized paragraph is split at
  sentence boundaries, and a single oversized sentence is split by words.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from .loaders import Page

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[\"'(\[]?[A-ZÇĞİÖŞÜ0-9])")


class Chunk(BaseModel):
    chunk_id: str
    doc_name: str
    page_no: int
    text: str
    section_hint: str  # first line of the chunk, for display
    is_pseudo_page: bool = False

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _split_sentences(paragraph: str) -> list[str]:
    parts = _SENTENCE_END.split(paragraph.strip())
    return [p for p in parts if p]


def _split_words(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _units(page_text: str, max_words: int) -> list[str]:
    """Paragraphs, except oversized ones become sentences (or word slices)."""
    units: list[str] = []
    for para in page_text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) <= max_words:
            units.append(para)
            continue
        for sentence in _split_sentences(para):
            if len(sentence.split()) <= max_words:
                units.append(sentence)
            else:
                units.extend(_split_words(sentence, max_words))
    return units


def _tail_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[-n:]) if n > 0 else ""


def chunk_page(page: Page, chunk_words: int = 250, overlap_words: int = 50) -> list[Chunk]:
    """Chunk one page. Empty pages produce no chunks."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")
    if page.is_empty:
        return []

    units = _units(page.text, chunk_words)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    carry = ""  # overlap text carried from the previous chunk

    def flush() -> None:
        nonlocal current, current_words, carry
        if not current:
            return
        body = "\n\n".join(current)
        text = f"{carry} {body}".strip() if carry else body
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{page.doc_name}:{page.page_no}:{idx}",
                doc_name=page.doc_name,
                page_no=page.page_no,
                text=text,
                section_hint=body.splitlines()[0][:120],
                is_pseudo_page=page.is_pseudo_page,
            )
        )
        carry = _tail_words(body, overlap_words)
        current, current_words = [], 0

    for unit in units:
        n = len(unit.split())
        if current and current_words + n > chunk_words:
            flush()
        current.append(unit)
        current_words += n
    flush()
    return chunks


def chunk_pages(pages: list[Page], chunk_words: int = 250, overlap_words: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, chunk_words=chunk_words, overlap_words=overlap_words))
    return chunks
