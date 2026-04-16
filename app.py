"""
Streamlit web interface for the AI Act Risk Classifier.

Run locally:  streamlit run app.py
Deploy free:  https://streamlit.io/cloud
"""

import base64
import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from classifier import classify
from models.schemas import RiskLevel


# ── Auto-ingest on cold start ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI Act corpus into vector store...")
def ensure_corpus_loaded():
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "ai_act_corpus")
    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    chunks_file = Path("corpus/chunks/ai_act_combined.json")

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        with open(chunks_file, encoding="utf-8") as f:
            chunks = json.load(f)
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "article": c.get("article", ""),
                        "section": c.get("section", ""),
                        "language": c.get("language", ""),
                    }
                    for c in batch
                ],
            )
    return collection.count()


ensure_corpus_loaded()

# ── Logo ──────────────────────────────────────────────────────────────────────

def _logo_b64() -> str:
    logo_path = Path(__file__).parent / "assets" / "logo.png"
    return base64.b64encode(logo_path.read_bytes()).decode()

LOGO_B64 = _logo_b64()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Act Risk Classifier",
    page_icon=str(Path(__file__).parent / "assets" / "logo.png"),
    layout="centered",
)

# ── Design tokens ─────────────────────────────────────────────────────────────

RISK_STYLES = {
    RiskLevel.PROHIBITED: {
        "grad": "linear-gradient(135deg, #3D0A0A 0%, #5C1A1A 100%)",
        "border": "#EF4444",
        "text": "#FEE2E2",
        "label": "PROHIBITED",
        "icon": "🚫",
        "description": "Article 5 — This practice is banned under the AI Act.",
    },
    RiskLevel.HIGH: {
        "grad": "linear-gradient(135deg, #3D1D00 0%, #5C3410 100%)",
        "border": "#F59E0B",
        "text": "#FEF3C7",
        "label": "HIGH RISK",
        "icon": "⚠️",
        "description": "Article 6 + Annex III — Strict obligations apply before deployment.",
    },
    RiskLevel.LIMITED: {
        "grad": "linear-gradient(135deg, #0A1D3D 0%, #1A3A6B 100%)",
        "border": "#4F6BFF",
        "text": "#DBEAFE",
        "label": "LIMITED RISK",
        "icon": "ℹ️",
        "description": "Article 50 — Transparency obligations apply.",
    },
    RiskLevel.MINIMAL: {
        "grad": "linear-gradient(135deg, #0A2D1A 0%, #1A4D2E 100%)",
        "border": "#10B981",
        "text": "#D1FAE5",
        "label": "MINIMAL RISK",
        "icon": "✅",
        "description": "No specific AI Act obligations.",
    },
    RiskLevel.UNCLEAR: {
        "grad": "linear-gradient(135deg, #1C2030 0%, #2A2F45 100%)",
        "border": "#6B7280",
        "text": "#E5E7EB",
        "label": "UNCLEAR",
        "icon": "❓",
        "description": "Insufficient information — human legal review required.",
    },
}

CONFIDENCE_CONFIG = {
    "high":   ("#10B981", "High confidence"),
    "medium": ("#F59E0B", "Medium confidence"),
    "low":    ("#EF4444", "Low confidence — human review strongly recommended"),
}

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem !important; }

    /* Hero header */
    .eu-header {
        background: linear-gradient(135deg, #0A0E27 0%, #1a1f4e 60%, #2d35a8 100%);
        color: #fff;
        padding: 2rem 2rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        border: 1px solid rgba(79,107,255,0.25);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    .eu-badge {
        display: inline-block;
        background: rgba(79,107,255,0.15);
        border: 1px solid rgba(79,107,255,0.4);
        color: #7B93FF;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        margin-bottom: 0.9rem;
        text-transform: uppercase;
    }
    .eu-stars { font-size: 0.8rem; letter-spacing: 0.25em; color: #4F6BFF; margin-bottom: 0.7rem; }
    .eu-header h1 { font-size: 1.85rem; font-weight: 800; margin: 0 0 0.4rem; letter-spacing: 0.01em; }
    .eu-header p  { font-size: 0.88rem; color: #9CA3C0; margin: 0; line-height: 1.55; }

    /* Risk banner */
    .risk-banner {
        padding: 1.4rem 1.6rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        border-left: 5px solid;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    }
    .risk-label { font-size: 1.5rem; font-weight: 800; letter-spacing: 0.06em; margin: 0; }
    .risk-desc  { font-size: 0.84rem; margin-top: 0.35rem; opacity: 0.8; }

    /* Section labels */
    .section-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        color: #4F6BFF;
        text-transform: uppercase;
        margin: 1.4rem 0 0.5rem;
        border-bottom: 1px solid rgba(79,107,255,0.2);
        padding-bottom: 0.25rem;
    }

    /* Citation cards */
    .citation-box {
        background: #12163A;
        border: 1px solid #1C2152;
        border-left: 4px solid #4F6BFF;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .citation-article { font-weight: 700; font-size: 0.88rem; color: #7B93FF; }
    .citation-summary { font-size: 0.83rem; color: #9CA3C0; margin-top: 0.25rem; line-height: 1.5; }

    /* Reasoning card */
    .reasoning-box {
        background: #12163A;
        border: 1px solid #1C2152;
        padding: 1.1rem 1.3rem;
        border-radius: 10px;
        font-size: 0.87rem;
        color: #E5E7EB;
        line-height: 1.75;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }

    /* Ambiguity cards */
    .ambiguity-box {
        background: rgba(245,158,11,0.07);
        border: 1px solid rgba(245,158,11,0.2);
        border-left: 4px solid #F59E0B;
        padding: 0.8rem 1.1rem;
        margin: 0.45rem 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.85rem;
        color: #FEF3C7;
        line-height: 1.55;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.75rem;
        color: #6B7280;
        font-style: italic;
        border-top: 1px solid rgba(79,107,255,0.12);
        padding-top: 1rem;
        margin-top: 2rem;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="eu-header">
    <div class="eu-stars">★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★</div>
    <div class="eu-badge">Regulation (EU) 2024/1689 · AI Act</div>
    <h1>
        <img src="data:image/png;base64,{LOGO_B64}"
             style="height:3.5rem; vertical-align:middle; margin-right:0.6rem;">
        AI Act Risk Classifier
    </h1>
    <p>Classify AI system use cases by risk level — with article-by-article legal
    justification powered by RAG retrieval + LLM reasoning.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔑 API Key")
    st.caption(
        "This app uses the [Anthropic Claude API](https://console.anthropic.com). "
        "Your key is used only for this session and never stored."
    )
    user_api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        label_visibility="collapsed",
        help=(
            "🔒 Your API key is never stored or logged.\n\n"
            "It is kept in memory only for the duration of your browser session "
            "and discarded when you close or refresh the page.\n\n"
            "No key is shared between users. Each session is independent.\n\n"
            "You can verify this in the open-source code on GitHub."
        ),
    )
    if user_api_key:
        os.environ["ANTHROPIC_API_KEY"] = user_api_key
        os.environ["LLM_PROVIDER"] = "claude"
        st.success("Key set — ready to classify.")
    else:
        st.info("Get a free key with $5 credit at [console.anthropic.com](https://console.anthropic.com).")

    st.divider()
    st.markdown("### Risk levels")
    st.markdown(
        "- 🚫 **Prohibited** — Article 5\n"
        "- ⚠️ **High Risk** — Article 6 + Annex III\n"
        "- ℹ️ **Limited Risk** — Article 50\n"
        "- ✅ **Minimal Risk** — No specific obligations"
    )
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. **Rule pre-filter** — keyword signals\n"
        "2. **RAG retrieval** — semantic search over AI Act corpus\n"
        "3. **LLM analysis** — chain-of-thought legal reasoning"
    )
    st.divider()
    st.markdown(
        "⚠️ **Not legal advice.** [Read disclaimer]"
        "(https://github.com/juanjimenoizquierdo-ui/ai-act-classifier/blob/master/docs/legal_disclaimer.md)"
    )
    st.markdown("🔗 [GitHub](https://github.com/juanjimenoizquierdo-ui/ai-act-classifier)")

# ── Input tabs ────────────────────────────────────────────────────────────────

EXAMPLES = {
    "Mortgage loan AI decision": "A bank uses an AI model to automatically decide whether to approve or reject mortgage loan applications based on applicant financial data and credit history.",
    "Real-time facial recognition (public space)": "A municipality deploys cameras with real-time facial recognition in a public square to identify suspects in an ongoing criminal investigation.",
    "CV screening tool": "An HR software company offers a tool that scores candidates' CVs and ranks them before a human recruiter reviews the shortlist.",
    "Customer service chatbot": "A telecom company deploys a chatbot to handle customer billing queries and troubleshoot connectivity issues.",
    "Product recommendation engine": "An e-commerce platform uses AI to recommend products to users based on their browsing history.",
    "Student assessment AI": "A school uses AI to generate personalised study plans and assess student performance to determine academic progression.",
}

tab_own, tab_example = st.tabs(["✏️  Classify your use case", "📋  Try an example"])

with tab_own:
    st.caption("Describe the AI system — the more detail, the better the analysis.")
    use_case_own = st.text_area(
        "Use case description",
        height=140,
        placeholder="e.g. A recruitment platform that automatically screens CVs and ranks candidates before human review...",
        label_visibility="collapsed",
    )
    btn_own = st.button("Classify", type="primary", use_container_width=True, key="btn_own")

with tab_example:
    st.caption("Select a pre-built case to see the classifier across different risk levels.")
    selected = st.selectbox(
        "Example cases",
        options=list(EXAMPLES.keys()),
        label_visibility="collapsed",
    )
    st.info(EXAMPLES[selected])
    btn_example = st.button("Classify this example", type="primary", use_container_width=True, key="btn_example")

# ── Resolve input ─────────────────────────────────────────────────────────────

use_case = ""
classify_btn = False
if btn_own and use_case_own.strip():
    use_case = use_case_own.strip()
    classify_btn = True
elif btn_own and not use_case_own.strip():
    st.warning("Please describe your AI system use case before classifying.")
elif btn_example:
    use_case = EXAMPLES[selected]
    classify_btn = True

# ── Classification ────────────────────────────────────────────────────────────

if classify_btn:
    if not user_api_key:
        st.warning("Enter your Anthropic API key in the sidebar to classify.")
    elif not use_case.strip():
        st.warning("Please enter a use case description.")
    else:
        try:
            with st.spinner("Running pipeline: rules → RAG → LLM..."):
                result = classify(use_case.strip())
        except TypeError as e:
            if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                st.error(
                    "**LLM API key not configured.**\n\n"
                    "Enter your Anthropic API key in the sidebar.\n\n"
                    "Get a free key with $5 credit at [console.anthropic.com](https://console.anthropic.com)."
                )
            else:
                st.error(f"Unexpected error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.stop()

        cfg = RISK_STYLES.get(result.risk_level, RISK_STYLES[RiskLevel.UNCLEAR])
        conf_color, conf_label = CONFIDENCE_CONFIG.get(result.confidence, ("#6B7280", "Unknown"))

        # Risk banner
        st.markdown(
            f"""
            <div class="risk-banner"
                 style="background:{cfg['grad']}; color:{cfg['text']}; border-left-color:{cfg['border']}">
                <div class="risk-label">{cfg['icon']} {cfg['label']}</div>
                <div class="risk-desc">{cfg['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence
        st.markdown(
            f'<span style="color:{conf_color}; font-size:0.85rem; font-weight:600;">'
            f'● {conf_label}</span>',
            unsafe_allow_html=True,
        )

        # Legal basis
        if result.primary_citations:
            st.markdown('<div class="section-label">Legal Basis</div>', unsafe_allow_html=True)
            for citation in result.primary_citations:
                st.markdown(
                    f"""<div class="citation-box">
                        <div class="citation-article">{citation.article}</div>
                        <div class="citation-summary">{citation.summary}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Reasoning
        st.markdown('<div class="section-label">Legal Reasoning</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="reasoning-box">{result.reasoning}</div>',
            unsafe_allow_html=True,
        )

        # Ambiguities
        if result.ambiguities:
            st.markdown('<div class="section-label">Requires Human Review</div>', unsafe_allow_html=True)
            for amb in result.ambiguities:
                st.markdown(
                    f'<div class="ambiguity-box">⚠ {amb}</div>',
                    unsafe_allow_html=True,
                )

        # JSON export
        with st.expander("Raw JSON output"):
            st.json(result.model_dump())

        # Disclaimer
        st.markdown(
            f'<div class="disclaimer">{result.disclaimer}</div>',
            unsafe_allow_html=True,
        )
