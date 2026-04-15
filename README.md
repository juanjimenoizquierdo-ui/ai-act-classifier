# AI Act Risk Classifier

Automated risk classification of AI systems under **Regulation (EU) 2024/1689** (the EU AI Act).

Given a free-text description of an AI system's use case, this tool classifies it as **prohibited**, **high-risk**, **limited-risk**, or **minimal-risk** — with article-by-article legal justification.

> **This tool does not provide legal advice.** See [docs/legal_disclaimer.md](docs/legal_disclaimer.md).

---

## How it works

The classifier combines three layers:

```
User input (free text)
      │
      ▼
┌─────────────────────┐
│  Rule pre-filter    │  Keyword → risk signal (Article 5, Annex III, Article 50)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  RAG retrieval      │  Semantic search over AI Act corpus (ChromaDB)
│  (ChromaDB)         │  Returns most relevant provisions for the use case
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LLM classification │  Chain-of-thought legal reasoning + article citations
│  (Claude / Ollama)  │  Outputs structured JSON
└────────┬────────────┘
         │
         ▼
  ClassificationResult
  • risk_level
  • primary_citations (article + summary)
  • reasoning
  • ambiguities
  • confidence
```

**Why this architecture?**

- Rules catch clear cases fast and cheaply (no LLM call needed for the retrieval query).
- RAG grounds the LLM in the actual regulation text — reduces hallucination of article numbers.
- No LangChain: direct API calls make the logic transparent and the code readable.

---

## Corpus

The classifier uses the official EUR-lex text of Regulation (EU) 2024/1689.
**Source**: [EUR-lex CELEX:32024R1689](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689)

Core provisions covered:
- **Article 5** — Prohibited AI practices
- **Article 6** — Classification rules for high-risk AI
- **Article 50** — Transparency obligations (limited risk)
- **Annex I** — AI definition (GPAI)
- **Annex II** — Union harmonisation legislation
- **Annex III** — High-risk AI systems (8 areas)

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/ai-act-classifier.git
cd ai-act-classifier
pip install -r requirements.txt
```

### 2. Configure your environment

```bash
cp .env.example .env
# Edit .env — choose LLM_PROVIDER=claude or LLM_PROVIDER=ollama
```

**Option A — Claude API (recommended)**
```bash
# Set ANTHROPIC_API_KEY in .env
# Get your key at https://console.anthropic.com/
```

**Option B — Ollama (free, local)**
```bash
ollama pull llama3.3
ollama serve
# Set LLM_PROVIDER=ollama in .env
```

### 3. Build the corpus

Download the AI Act PDFs from EUR-lex and place them in `corpus/raw/`:
- `corpus/raw/ai_act_en.pdf`
- `corpus/raw/ai_act_es.pdf` (optional, improves Spanish queries)

```bash
python corpus/build_corpus.py
python scripts/ingest_corpus.py
```

### 4. Classify a use case

```bash
# Single classification
python scripts/demo.py "A bank uses AI to decide mortgage loan approvals automatically"

# Interactive mode
python scripts/demo.py --interactive

# Run built-in examples
python scripts/demo.py --run-examples

# JSON output (for integration)
python scripts/demo.py --json "A CV screening tool used in recruitment"
```

---

## Example output

**Input**: *"A municipality deploys cameras with real-time facial recognition in a public square to identify suspects."*

```
Classification: PROHIBITED  (confidence: medium)

Legal Basis:
  Article 5(1)(d) — Real-time remote biometric identification systems in
                    publicly accessible spaces are prohibited as a general
                    rule. Law enforcement exceptions in Article 5(2) are
                    narrow and subject to prior authorisation.

Reasoning:
  The system described performs real-time biometric identification (facial
  recognition) in a publicly accessible space (public square). This falls
  squarely within Article 5(1)(d). The stated purpose (identifying suspects)
  may invoke the law enforcement exception, but this requires prior judicial
  or independent administrative authorisation and is limited to specific
  offences listed in Annex V...

Ambiguities:
  • Whether the law enforcement exception (Article 5(2)) applies depends on
    the specific offence type and whether prior authorisation was obtained.
  • "Publicly accessible space" definition requires case-by-case assessment.
```

---

## Project structure

```
ai-act-classifier/
├── classifier/
│   ├── rules.py          # Keyword pre-filter (Article 5, Annex III, Article 50)
│   ├── retriever.py      # ChromaDB RAG retriever
│   ├── classifier.py     # Main pipeline orchestration
│   └── prompts.py        # LLM prompt templates
├── models/
│   └── schemas.py        # Pydantic output schema
├── corpus/
│   ├── build_corpus.py   # PDF → chunks (by article)
│   └── chunks/           # Processed JSON chunks
├── scripts/
│   ├── ingest_corpus.py  # Chunks → ChromaDB
│   └── demo.py           # CLI interface
├── tests/
│   ├── test_rules.py     # Unit tests (no API needed)
│   └── test_cases.json   # Labelled examples for evaluation
└── docs/
    └── legal_disclaimer.md
```

---

## Running tests

```bash
# Rule-based tests (no API key needed)
python -m pytest tests/test_rules.py -v
```

---

## Limitations

1. **Not legal advice** — see [docs/legal_disclaimer.md](docs/legal_disclaimer.md).
2. Classification quality depends on corpus coverage; v1 covers risk classification articles only.
3. Ambiguous cases (e.g. GPAI models, safety components) require human legal review.
4. The AI Act is subject to ongoing guidance from the European AI Office.

---

## Tech stack

| Component | Technology |
|---|---|
| LLM (primary) | Claude API (`claude-sonnet-4-6`) |
| LLM (local/free) | Ollama + Llama 3.3 |
| Vector store | ChromaDB |
| Embeddings | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| PDF processing | PyMuPDF |
| Output schema | Pydantic v2 |
| CLI | Typer + Rich |

---

## Legal basis reference

| Risk level | Regulation reference |
|---|---|
| Prohibited | Article 5(1)(a)–(h) |
| High risk — Annex I | Article 6(1) + Annex I (harmonised legislation) |
| High risk — Annex III | Article 6(2) + Annex III (8 areas) |
| Limited risk | Article 50 (transparency obligations) |
| Minimal risk | Recitals 47–48 (no specific obligations) |

---

## Author

Built by a Spanish lawyer (Máster de Acceso a la Abogacía) with experience in compliance and legal tech automation, as a study project for the EU AI Act and a portfolio demonstration.

**Disclaimer**: This project is for educational and portfolio purposes. It does not constitute legal advice.

---

## License

MIT
