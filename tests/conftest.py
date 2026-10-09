from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docs_chat.embeddings import HashEmbedder  # noqa: E402


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=128)


@pytest.fixture
def sample_docs_dir() -> Path:
    return ROOT / "eval" / "sample_docs"


@pytest.fixture
def two_page_pdf(tmp_path: Path) -> Path:
    """A small real PDF with known text on each of its two pages."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = tmp_path / "two_pages.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.drawString(72, 750, "Alpha policy: employees get twenty vacation days.")
    c.drawString(72, 730, "This sentence is also on page one.")
    c.showPage()
    c.drawString(72, 750, "Beta policy: the warranty lasts twelve months.")
    c.showPage()
    c.save()
    return path
