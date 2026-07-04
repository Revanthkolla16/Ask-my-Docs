"""
backend/app/ingestion/parser.py
--------------------------------
Document parsing layer.  Supports PDF (via PyMuPDF), Markdown, and plain text.
Each page / section is returned as a ``DocumentPage`` dataclass so downstream
chunkers can handle the text uniformly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DocumentPage:
    """A single logical 'page' or section extracted from a source document."""

    content: str
    metadata: dict = field(default_factory=dict)
    # convenience properties stored inside metadata:
    #   source      – file path / name
    #   page_num    – 1-based page index (PDF) or section index (Markdown/text)
    #   section     – heading / section name (Markdown) or None


# ── PDF parser ────────────────────────────────────────────────────────────────

def parse_pdf(path: str | Path) -> List[DocumentPage]:
    """Extract text page-by-page from a PDF using PyMuPDF (``fitz``)."""
    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyMuPDF is required for PDF parsing. "
            "Install it with: pip install pymupdf"
        ) from exc

    path = Path(path)
    pages: List[DocumentPage] = []

    with fitz.open(str(path)) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue  # skip blank pages

            pages.append(
                DocumentPage(
                    content=text,
                    metadata={
                        "source": path.name,
                        "page_num": page_index,
                        "section": None,
                    },
                )
            )

    return pages


# ── Markdown parser ────────────────────────────────────────────────────────────

def parse_markdown(path: str | Path) -> List[DocumentPage]:
    """
    Split a Markdown file on ``##`` (level-2) headings.

    Everything before the first ``##`` heading is gathered into a 'preamble'
    section.  Each subsequent ``##`` block becomes its own ``DocumentPage``,
    with the heading text stored as ``section`` in metadata.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    # Split on ## headings (keep the delimiter via a capture group)
    # We only split on ## (not ###, ####, …) to keep sections reasonably large.
    parts = re.split(r"(?m)^(## .+)$", raw)

    pages: List[DocumentPage] = []
    section_index = 0

    # parts[0] is content before the first ## heading (may be empty / front-matter)
    preamble = parts[0].strip()
    if preamble:
        section_index += 1
        pages.append(
            DocumentPage(
                content=preamble,
                metadata={
                    "source": path.name,
                    "page_num": section_index,
                    "section": "preamble",
                },
            )
        )

    # Remaining parts come in pairs: (heading, body)
    it = iter(parts[1:])
    for heading, body in zip(it, it):
        body = body.strip()
        if not body:
            continue
        section_index += 1
        # Strip the leading "## " from the heading to get a clean section name
        section_name = heading.lstrip("#").strip()
        pages.append(
            DocumentPage(
                content=f"{heading}\n\n{body}",
                metadata={
                    "source": path.name,
                    "page_num": section_index,
                    "section": section_name,
                },
            )
        )

    return pages


# ── Plain-text parser ──────────────────────────────────────────────────────────

def parse_text(path: str | Path, lines_per_page: int = 100) -> List[DocumentPage]:
    """
    Parse a plain-text file by grouping every *lines_per_page* non-blank lines
    into a single ``DocumentPage``.  This gives a stable page boundary for long
    text files without needing a heading structure.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    pages: List[DocumentPage] = []
    page_index = 0
    buffer: List[str] = []

    def _flush(buf: List[str], idx: int) -> Optional[DocumentPage]:
        text = "\n".join(buf).strip()
        if not text:
            return None
        return DocumentPage(
            content=text,
            metadata={
                "source": path.name,
                "page_num": idx,
                "section": None,
            },
        )

    for line in lines:
        buffer.append(line)
        if len(buffer) >= lines_per_page:
            page_index += 1
            doc_page = _flush(buffer, page_index)
            if doc_page:
                pages.append(doc_page)
            buffer = []

    # flush remainder
    if buffer:
        page_index += 1
        doc_page = _flush(buffer, page_index)
        if doc_page:
            pages.append(doc_page)

    return pages


# ── Auto-router ───────────────────────────────────────────────────────────────

_EXTENSION_MAP = {
    ".pdf": parse_pdf,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".txt": parse_text,
    ".text": parse_text,
}


def parse_document(path: str | Path) -> List[DocumentPage]:
    """
    Dispatch to the correct parser based on the file extension.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()
    parser_fn = _EXTENSION_MAP.get(ext)
    if parser_fn is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported extensions: {list(_EXTENSION_MAP.keys())}"
        )
    return parser_fn(path)
