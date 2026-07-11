"""Hybrid PDF -> markdown extraction: fitz for text, pdfplumber for tables."""
import re

import fitz
import pdfplumber


def parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    out = []
    with pdfplumber.open(path) as pl:
        for i in range(len(doc)):
            text = (doc[i].get_text("text") or "").strip()
            out.append(f"<!-- page {i + 1} -->")
            if _is_table_heavy(text):
                tables_md = _tables_to_markdown(pl.pages[i])
                if tables_md:
                    out.append(tables_md)
                    if text:
                        out.append(text)
                    continue
            out.append(text)
    doc.close()
    return "\n\n".join(out)


def _is_table_heavy(text: str) -> bool:
    if len(text) < 40:
        return True
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    spaced = sum(1 for l in lines if re.search(r"\s{2,}\S", l))
    return spaced / len(lines) > 0.4


def _tables_to_markdown(page) -> str:
    tables = page.extract_tables()
    if not tables:
        return ""
    return "\n\n".join(_table_to_md(t) for t in tables if t)


def _table_to_md(table) -> str:
    rows = [[(c or "").strip() for c in row] for row in table if row]
    if not rows:
        return ""
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows[1:]:
        row = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
