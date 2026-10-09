"""Turn files into pages of text.

Supported: .pdf (real pages, via pypdf), .docx, .txt, .md.
DOCX/TXT/MD have no physical pages, so they are split into *pseudo-pages* of about
``PSEUDO_PAGE_WORDS`` words at paragraph boundaries. The UI shows them as
"section N" so citations still point somewhere concrete.

Scanned PDFs (pages with almost no extractable text) are returned as empty pages and
reported in ``warnings`` – OCR is out of scope (see README).
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
MIN_CHARS_PER_PAGE = 20  # below this a PDF page is treated as scanned/empty
PSEUDO_PAGE_WORDS = 400

# Lines that are only page furniture ("Page 3", "3 / 12", "12 of 40"): common PDF
# headers/footers that would otherwise pollute every chunk.
_PAGE_FURNITURE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*(/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)


def strip_page_furniture(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _PAGE_FURNITURE.match(line))


class Page(BaseModel):
    doc_name: str
    page_no: int  # 1-based; PDF page number or pseudo-page number
    text: str
    is_pseudo_page: bool = False

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) < MIN_CHARS_PER_PAGE


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, keep paragraph breaks as a single blank line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- PDF
def _load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - corrupt page, keep going
            text = ""
        pages.append(Page(doc_name=path.name, page_no=i, text=normalize_whitespace(strip_page_furniture(text))))
    return pages


# -------------------------------------------------------------------------- DOCX
def _load_docx_text(path: Path) -> str:
    import docx  # python-docx

    document = docx.Document(str(path))
    parts: list[str] = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n\n".join(p for p in parts if p and p.strip())


# ------------------------------------------------------------------ pseudo-pages
def split_into_pseudo_pages(text: str, doc_name: str, words_per_page: int = PSEUDO_PAGE_WORDS) -> list[Page]:
    """Split plain text into pseudo-pages at paragraph boundaries (~words_per_page each)."""
    text = normalize_whitespace(text)
    if not text:
        return [Page(doc_name=doc_name, page_no=1, text="", is_pseudo_page=True)]
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pages: list[Page] = []
    current: list[str] = []
    current_words = 0
    for para in paragraphs:
        n = len(para.split())
        if current and current_words + n > words_per_page:
            pages.append(Page(doc_name=doc_name, page_no=len(pages) + 1, text="\n\n".join(current), is_pseudo_page=True))
            current, current_words = [], 0
        current.append(para)
        current_words += n
    if current:
        pages.append(Page(doc_name=doc_name, page_no=len(pages) + 1, text="\n\n".join(current), is_pseudo_page=True))
    return pages


# ------------------------------------------------------------------------ public
def load_file(path: str | Path) -> list[Page]:
    """Load one file into pages. Raises ValueError for unsupported types."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.name} (supported: {sorted(SUPPORTED_SUFFIXES)})")
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return split_into_pseudo_pages(_load_docx_text(path), path.name)
    return split_into_pseudo_pages(path.read_text(encoding="utf-8", errors="replace"), path.name)


def load_folder(folder: str | Path) -> tuple[list[Page], list[str]]:
    """Load every supported file in a folder (sorted by name).

    Returns (pages, warnings). Empty/scanned pages are kept (so page numbers stay
    correct) but a warning is added for each document that has them.
    """
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    pages: list[Page] = []
    warnings: list[str] = []
    if not files:
        warnings.append(f"No supported files found in {folder} (supported: {sorted(SUPPORTED_SUFFIXES)})")
    for f in files:
        doc_pages = load_file(f)
        empty = [p.page_no for p in doc_pages if p.is_empty]
        if empty and len(empty) == len(doc_pages):
            warnings.append(f"{f.name}: no extractable text on any page (scanned PDF? OCR is not supported)")
        elif empty:
            warnings.append(f"{f.name}: {len(empty)} page(s) without text were skipped: {empty[:10]}")
        pages.extend(doc_pages)
    return pages, warnings
