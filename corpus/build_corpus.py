"""
Builds the AI Act corpus from the official EUR-lex PDFs.

Downloads (optional) and processes:
- ES version: https://eur-lex.europa.eu/legal-content/ES/TXT/PDF/?uri=CELEX:32024R1689
- EN version: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689

Chunks by legal unit (article / annex point) and saves to corpus/chunks/.

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


# ── Chunking strategy ─────────────────────────────────────────────────────────
# We chunk by article / annex point, not by fixed token size.
# Legal reasoning requires the full context of a provision, not arbitrary splits.

ARTICLE_PATTERN = re.compile(
    r"(Artículo\s+\d+|Article\s+\d+|ANEXO\s+[IVX]+|ANNEX\s+[IVX]+)",
    re.IGNORECASE,
)


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)


def chunk_by_article(text: str, language: str) -> list[dict]:
    """
    Splits the full regulation text into chunks, one per article/annex point.
    Returns list of dicts with: id, article, text, language.
    """
    # Find all article/annex headers and their positions
    matches = list(ARTICLE_PATTERN.finditer(text))

    if not matches:
        print(f"WARNING: No article headers found in {language} text. Check PDF extraction.")
        return []

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        chunk_text = text[start:end].strip()
        article_header = match.group(0).strip()

        # Skip very short chunks (likely headers without content)
        if len(chunk_text) < 100:
            continue

        chunks.append(
            {
                "id": f"{language}_{article_header.replace(' ', '_')}_{i}",
                "article": article_header,
                "section": "",
                "language": language,
                "text": chunk_text,
                "char_count": len(chunk_text),
            }
        )

    return chunks


def process_pdf(pdf_path: Path, language: str) -> list[dict]:
    print(f"Processing {pdf_path.name} ({language})...")
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_by_article(text, language)
    print(f"  -> {len(chunks)} chunks extracted")
    return chunks


def save_chunks(chunks: list[dict], language: str) -> Path:
    output_path = CHUNKS_DIR / f"ai_act_{language}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved to {output_path}")
    return output_path


def main():
    pdf_files = {
        "es": RAW_DIR / "ai_act_es.pdf",
        "en": RAW_DIR / "ai_act_en.pdf",
    }

    found = {lang: path for lang, path in pdf_files.items() if path.exists()}

    if not found:
        print(
            "No PDFs found in corpus/raw/.\n"
            "Download them from EUR-lex:\n"
            "  ES: https://eur-lex.europa.eu/legal-content/ES/TXT/PDF/?uri=CELEX:32024R1689\n"
            "  EN: https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689\n"
            "Save as: corpus/raw/ai_act_es.pdf and corpus/raw/ai_act_en.pdf"
        )
        sys.exit(1)

    all_chunks = []
    for lang, path in found.items():
        chunks = process_pdf(path, lang)
        save_chunks(chunks, lang)
        all_chunks.extend(chunks)

    # Save combined corpus
    combined_path = CHUNKS_DIR / "ai_act_combined.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(all_chunks)} total chunks saved to {combined_path}")


if __name__ == "__main__":
    main()
