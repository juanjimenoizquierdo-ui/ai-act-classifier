"""
Builds the AI Act corpus from the official EUR-lex PDFs.

Chunking strategy:
  1. Use PyMuPDF structured extraction (get_text "dict") to detect REAL article
     headers — spans whose text is exactly "Artículo N" or "ANEXO X" — instead
     of regex over flat text (which picks up inline article references in recitals).
  2. Accumulate page content between consecutive real headers.
  3. For chunks over MAX_CHUNK_CHARS, split further by numbered paragraph.

Run: python corpus/build_corpus.py
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


CORPUS_DIR = Path(__file__).parent
RAW_DIR = CORPUS_DIR / "raw"
CHUNKS_DIR = CORPUS_DIR / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

MAX_CHUNK_CHARS = 3500   # Chunks larger than this get paragraph-split
MIN_CHUNK_CHARS = 200    # Ignore leftover micro-fragments

# Matches a span whose entire text is an article or annex header
HEADER_SPAN_RE = re.compile(
    r"^(Art[ií]culo\s+\d+|Article\s+\d+|ANEXO\s+[IVX]+|ANNEX\s+[IVX]+)\s*$",
    re.IGNORECASE,
)

# Numbered paragraph at line start: "1.  Text..." or "1. Text..."
PARA_SPLIT_RE = re.compile(r"(?<=\n)(?=\s*\d+\.\s{2,}[^\d])")


def is_header_span(span_text: str) -> bool:
    return bool(HEADER_SPAN_RE.match(span_text.strip()))


def extract_structured(pdf_path: Path) -> list[tuple[str, str]]:
    """
    Returns list of (header, content) tuples by walking the PDF block by block.
    Headers are identified by span content (exact match), not font.
    Content preserves line breaks so paragraph-number splitting works downstream.
    """
    doc = fitz.open(str(pdf_path))

    current_header: str | None = None
    current_lines: list[str] = []
    sections: list[tuple[str, str]] = []

    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                # Join all spans in a line into one line of text
                line_text = " ".join(
                    span["text"] for span in line["spans"]
                ).strip()
                if not line_text:
                    continue

                if is_header_span(line_text):
                    # Save previous section
                    if current_header is not None:
                        content = "\n".join(current_lines).strip()
                        if content:
                            sections.append((current_header, content))
                    current_header = HEADER_SPAN_RE.match(line_text).group(1)
                    current_lines = []
                elif current_header is not None:
                    current_lines.append(line_text)

    # Last section
    if current_header and current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))

    return sections


def split_by_paragraph(header: str, content: str) -> list[tuple[str, str, str]]:
    """
    Split content by numbered paragraph markers (lines starting with "1.", "2.", ...).
    Returns list of (header, section_label, text).
    """
    # Split on lines starting with "1." (paragraphs) or "1)" (definitions/lists)
    parts = re.split(r"\n(?=\d+[.)]\s)", content)

    result = []
    for part in parts:
        part = part.strip()
        if len(part) < MIN_CHUNK_CHARS:
            continue
        para_match = re.match(r"^(\d+)[.)]", part)
        label = f"paragraph {para_match.group(1)}" if para_match else ""
        result.append((header, label, f"[{header}]\n{part}"))

    return result if result else [(header, "", f"[{header}]\n{content}")]


def build_chunks(pdf_path: Path, language: str) -> list[dict]:
    print(f"Processing {pdf_path.name} ({language})...")
    sections = extract_structured(pdf_path)
    print(f"  -> {len(sections)} sections detected")

    chunks: list[dict] = []
    chunk_id = 0

    for header, content in sections:
        if len(content) <= MAX_CHUNK_CHARS:
            chunks.append({
                "id": f"{language}_{chunk_id:04d}",
                "article": header,
                "section": "",
                "language": language,
                "text": f"[{header}]\n{content}",
                "char_count": len(content),
            })
            chunk_id += 1
        else:
            # Split by paragraph
            sub = split_by_paragraph(header, content)
            for h, label, text in sub:
                chunks.append({
                    "id": f"{language}_{chunk_id:04d}",
                    "article": h,
                    "section": label,
                    "language": language,
                    "text": text,
                    "char_count": len(text),
                })
                chunk_id += 1

    return chunks


def report(chunks: list[dict], label: str) -> None:
    sizes = [c["char_count"] for c in chunks]
    avg = sum(sizes) // len(sizes) if sizes else 0
    annex3 = [c for c in chunks if "III" in c["article"].upper()]
    art5 = [c for c in chunks if re.search(r"5\b", c["article"])]
    print(f"  -> {len(chunks)} chunks | avg {avg} | min {min(sizes)} | max {max(sizes)}")
    print(f"  -> Annex/Anexo III chunks: {len(annex3)}")
    print(f"  -> Article/Artículo 5 chunks: {len(art5)}")


def main():
    pdf_files = {
        "es": RAW_DIR / "ai_act_es.pdf",
        "en": RAW_DIR / "ai_act_en.pdf",
    }
    found = {lang: path for lang, path in pdf_files.items() if path.exists()}

    if not found:
        print(
            "No PDFs found in corpus/raw/.\n"
            "Download from EUR-lex:\n"
            "  ES: https://eur-lex.europa.eu/legal-content/ES/TXT/PDF/?uri=CELEX:32024R1689\n"
            "  EN: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689"
        )
        sys.exit(1)

    all_chunks: list[dict] = []
    for lang, path in found.items():
        chunks = build_chunks(path, lang)
        report(chunks, lang)

        out = CHUNKS_DIR / f"ai_act_{lang}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"  -> Saved to {out}")

        all_chunks.extend(chunks)

    combined = CHUNKS_DIR / "ai_act_combined.json"
    with open(combined, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    total_chars = sum(c["char_count"] for c in all_chunks)
    print(f"\nDone. {len(all_chunks)} chunks | {total_chars // 1000}K chars total")
    print(f"Saved: {combined}")
    print("\nNext: python scripts/ingest_corpus.py")


if __name__ == "__main__":
    main()
