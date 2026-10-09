#!/usr/bin/env python
"""Render the original sample documents (eval/sample_docs/src/*.md) to PDF.

The sample documents are written for this project (fictional company and product),
so they can be redistributed freely. Re-run this script after editing the .md
sources; it rewrites eval/sample_docs/*.pdf deterministically.

    python scripts/make_sample_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "eval" / "sample_docs" / "src"
OUT = ROOT / "eval" / "sample_docs"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["Normal"], fontName="Helvetica", fontSize=11.5, leading=18, alignment=TA_JUSTIFY, spaceAfter=8)
    return {
        "h1": ParagraphStyle("h1", parent=base["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=17, spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=11, leading=15, spaceBefore=6, spaceAfter=3),
        "body": body,
        "meta": ParagraphStyle("meta", parent=body, textColor="#555555", fontSize=9.5),
    }


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_to_flowables(markdown: str, styles: dict[str, ParagraphStyle]) -> list:
    flow: list = []
    for block in re.split(r"\n\s*\n", markdown.strip()):
        block = block.strip()
        if block.startswith("# "):
            flow.append(Paragraph(_escape(block[2:]), styles["h1"]))
        elif block.startswith("## "):
            flow.append(Paragraph(_escape(block[3:]), styles["h2"]))
        elif block.startswith("### "):
            flow.append(Paragraph(_escape(block[4:]), styles["h3"]))
        elif " · " in block and len(block) < 80:
            flow.append(Paragraph(_escape(block), styles["meta"]))
        else:
            flow.append(Paragraph(_escape(" ".join(block.splitlines())), styles["body"]))
    return flow


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.drawRightString(A4[0] - 2 * cm, 1.3 * cm, f"Page {doc.page}")
    canvas.restoreState()


def render(md_path: Path, pdf_path: Path) -> int:
    styles = _styles()
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title=md_path.stem.replace("_", " ").title(), author="docs-chat sample", creator="docs-chat",
    )
    flow = md_to_flowables(md_path.read_text(encoding="utf-8"), styles)
    flow.append(Spacer(1, 6))
    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    from pypdf import PdfReader

    return len(PdfReader(str(pdf_path)).pages)


def main() -> int:
    sources = sorted(SRC.glob("*.md"))
    if not sources:
        print(f"No sources in {SRC}", file=sys.stderr)
        return 1
    for md in sources:
        pdf = OUT / f"{md.stem}.pdf"
        n = render(md, pdf)
        print(f"{pdf.relative_to(ROOT)}: {n} page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
