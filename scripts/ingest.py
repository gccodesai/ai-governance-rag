"""Turn the raw NIST documents into retrieval-ready chunks.

Run once (output is committed to the repo):

    python scripts/ingest.py

Reads   data/raw/NIST.AI.100-1.pdf   (AI RMF 1.0)
        data/raw/NIST.AI.600-1.pdf   (Generative AI Profile)
        data/raw/playbook.json       (AI RMF Playbook)
Writes  data/chunks.jsonl            (one JSON object per chunk)
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "chunks.jsonl"

TARGET_CHARS = 1100   # aim for chunks of roughly this size
OVERLAP_CHARS = 150   # carry the tail of one chunk into the next

PDFS = [
    {
        "file": "NIST.AI.100-1.pdf",
        "doc_id": "ai-rmf",
        "title": "NIST AI 100-1: AI Risk Management Framework (AI RMF 1.0)",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "short": "AI RMF 1.0",
        "skip_pages": {1, 2, 3},  # cover, title page, "this publication is available" page
        "toc_until": 6,            # table of contents: don't treat its lines as headings
    },
    {
        "file": "NIST.AI.600-1.pdf",
        "doc_id": "genai-profile",
        "title": "NIST AI 600-1: Generative AI Profile",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        "short": "GenAI Profile",
        "skip_pages": {1, 2},
        "toc_until": 4,
    },
]

PLAYBOOK_URL = "https://airc.nist.gov/airmf-resources/playbook/"


def clean(text: str) -> str:
    """Normalise PDF text: ligatures, hyphenation, running headers, whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")
    # Running headers / bare page numbers.
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or re.fullmatch(r"\d{1,3}", s):
            continue
        if s.startswith("NIST AI 100-1 AI RMF 1.0") or s == "NIST AI 600-1":
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # "mea-\nsure" -> "measure"
    text = re.sub(r"\n(?=[a-z(])", " ", text)          # join wrapped lines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_text(text: str, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split on sentence boundaries into ~target-sized pieces with a small overlap."""
    sentences = re.split(r"(?<=[.!?:])\s+(?=[A-Z0-9•*(])|\n", text)
    chunks, current = [], ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current and len(current) + len(sent) + 1 > target:
            chunks.append(current.strip())
            tail = current[-overlap:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = tail + " " + sent
        else:
            current = f"{current} {sent}" if current else sent
    if current.strip():
        chunks.append(current.strip())
    # Merge a tiny trailing piece into its neighbour.
    if len(chunks) > 1 and len(chunks[-1]) < 200:
        chunks[-2] = chunks[-2] + " " + chunks.pop()
    return chunks


# Section headings ("3.2 Safe", "2.7. Human-AI Configuration") and subcategory rows
# ("GOVERN 1.1: ...") let us prefix each chunk with where it sits in the document.
# This "contextual chunk header" helps retrieval: a definition buried in a list
# still carries its section title.
HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][A-Za-z,/&()’' –-]{2,70})$")
APPENDIX_RE = re.compile(r"^Appendix [A-D]:\s+[A-Z].{3,80}$")
SUBCAT_RE = re.compile(r"^((?:GOVERN|MAP|MEASURE|MANAGE) \d{1,2}\.\d{1,2}):")


def pdf_segments(spec: dict):
    """Yield (page_no, context, raw_text) runs that share a heading/subcategory."""
    reader = PdfReader(RAW / spec["file"])
    heading, subcat = "", ""
    for page_no, page in enumerate(reader.pages, start=1):
        if page_no in spec["skip_pages"]:
            continue
        buf: list[str] = []
        ctx = " › ".join(p for p in (heading, subcat) if p)
        for line in unicodedata.normalize("NFKC", page.extract_text() or "").splitlines():
            s = line.strip()
            is_heading = page_no > spec["toc_until"] and not s[-1:].isdigit() and \
                (HEADING_RE.match(s) or APPENDIX_RE.match(s))
            sub = SUBCAT_RE.match(s)
            if is_heading or sub:
                if buf:
                    yield page_no, ctx, "\n".join(buf)
                    buf = []
                if is_heading:
                    heading, subcat = s, ""
                else:
                    subcat = sub.group(1)
                ctx = " › ".join(p for p in (heading, subcat) if p)
            buf.append(line)
        if buf:
            yield page_no, ctx, "\n".join(buf)


def pdf_chunks(spec: dict) -> list[dict]:
    out, counters = [], {}
    carry, carry_ctx = "", ""  # tiny segments (e.g. one short table row) roll forward
    for page_no, ctx, raw in pdf_segments(spec):
        if carry:
            # Keep the label of the merged-in segment too, e.g. "GOVERN 1.1 + GOVERN 1.2".
            if carry_ctx and carry_ctx != ctx:
                ctx = f"{carry_ctx} + {ctx.split(' › ')[-1]}" if ctx else carry_ctx
            text = clean(carry + "\n" + raw)
        else:
            text = clean(raw)
        carry, carry_ctx = "", ""
        if len(text) < 200:
            carry, carry_ctx = raw, ctx
            continue
        for piece in split_text(text):
            i = counters.get(page_no, 0)
            counters[page_no] = i + 1
            header = f"[{spec['short']}" + (f" › {ctx}" if ctx else "") + "]"
            out.append({
                "id": f"{spec['doc_id']}-p{page_no}-{i}",
                "doc_id": spec["doc_id"],
                "doc_title": spec["title"],
                "location": f"PDF page {page_no}" + (f", {ctx}" if ctx else ""),
                "url": f"{spec['url']}#page={page_no}",
                "text": f"{header}\n{piece}",
            })
    return out


def playbook_chunks() -> list[dict]:
    items = json.loads((RAW / "playbook.json").read_text(encoding="utf-8"))
    out = []
    for item in items:
        sub = item["title"]                      # e.g. "GOVERN 1.1"
        slug = sub.lower().replace(" ", "-").replace(".", "-")
        header = f"Playbook {sub}: {item['description'].strip()}"
        sections = [
            ("about", "About", item.get("section_about", "")),
            ("actions", "Suggested actions", item.get("section_actions", "")),
            ("documentation", "Transparency and documentation", item.get("section_doc", "")),
        ]
        for key, label, body in sections:
            body = unicodedata.normalize("NFKC", body or "").strip()
            if len(body) < 40:
                continue
            for i, piece in enumerate(split_text(body)):
                out.append({
                    "id": f"playbook-{slug}-{key}-{i}",
                    "doc_id": "playbook",
                    "doc_title": "NIST AI RMF Playbook",
                    "location": f"{sub} ({label})",
                    "url": PLAYBOOK_URL,
                    "text": f"{header}\n{label}: {piece}",
                })
    return out


def main() -> int:
    chunks = []
    for spec in PDFS:
        chunks += pdf_chunks(spec)
    chunks += playbook_chunks()
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    with OUT.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c["doc_id"]] = by_doc.get(c["doc_id"], 0) + 1
    print(f"Wrote {len(chunks)} chunks to {OUT.relative_to(ROOT)}: {by_doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
