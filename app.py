"""
Streamlit web interface for the AI Act Risk Classifier.

Run locally:
  streamlit run app.py

Deploy free:
  https://streamlit.io/cloud → connect GitHub repo → done.
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from classifier import classify
from models.schemas import RiskLevel


# ── Auto-ingest on cold start ─────────────────────────────────────────────────
# On Streamlit Cloud, chroma_db/ is not persisted. This function rebuilds
# the vector store from the committed JSON chunks on first load.

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
                    {"article": c.get("article", ""), "section": c.get("section", ""), "language": c.get("language", "")}
                    for c in batch
                ],
            )

    return collection.count()


ensure_corpus_loaded()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Act Risk Classifier",
    page_icon="⚖️",
    layout="centered",
)

# ── Styles ────────────────────────────────────────────────────────────────────

RISK_STYLES = {
    RiskLevel.PROHIBITED: {
        "bg": "#8B1A1A",
        "text": "#ffffff",
        "label": "PROHIBITED",
        "icon": "🚫",
        "description": "Article 5 — This practice is banned under the AI Act.",
    },
    RiskLevel.HIGH: {
        "bg": "#92400E",
        "text": "#ffffff",
        "label": "HIGH RISK",
        "icon": "⚠️",
        "description": "Article 6 + Annex III — Strict obligations apply before deployment.",
    },
    RiskLevel.LIMITED: {
        "bg": "#1E5A8A",
        "text": "#ffffff",
        "label": "LIMITED RISK",
        "icon": "ℹ️",
        "description": "Article 50 — Transparency obligations apply.",
    },
    RiskLevel.MINIMAL: {
        "bg": "#1A6B4A",
        "text": "#ffffff",
        "label": "MINIMAL RISK",
        "icon": "✅",
        "description": "No specific AI Act obligations.",
    },
    RiskLevel.UNCLEAR: {
        "bg": "#374151",
        "text": "#ffffff",
        "label": "UNCLEAR",
        "icon": "❓",
        "description": "Insufficient information — human legal review required.",
    },
}

CONFIDENCE_CONFIG = {
    "high": ("🟢", "High confidence"),
    "medium": ("🟡", "Medium confidence"),
    "low": ("🔴", "Low confidence — human review strongly recommended"),
}

st.markdown("""
<style>
    /* ── EU colour palette ── */
    :root {
        --eu-navy:       #003399;
        --eu-navy-dark:  #001A4D;
        --eu-pastel:     #D4E4F7;
        --eu-pastel-mid: #A8C8F0;
        --eu-text:       #1A1A2E;
    }

    /* ── App header strip ── */
    .eu-header {
        background: linear-gradient(135deg, var(--eu-navy-dark) 0%, var(--eu-navy) 100%);
        color: #ffffff;
        padding: 1.4rem 1.8rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
    }
    .eu-header h1 {
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: 0.02em;
    }
    .eu-header p {
        font-size: 0.88rem;
        opacity: 0.85;
        margin: 0;
    }
    .eu-stars {
        font-size: 1rem;
        letter-spacing: 0.15em;
        margin-bottom: 0.6rem;
        opacity: 0.7;
    }

    /* ── Risk banner ── */
    .risk-banner {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 6px solid rgba(255,255,255,0.4);
    }
    .risk-label {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin: 0;
    }
    .risk-desc {
        font-size: 0.88rem;
        margin-top: 0.3rem;
        opacity: 0.88;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: var(--eu-navy);
        text-transform: uppercase;
        margin: 1.2rem 0 0.4rem 0;
        border-bottom: 2px solid var(--eu-pastel);
        padding-bottom: 0.2rem;
    }

    /* ── Citation boxes ── */
    .citation-box {
        background: var(--eu-pastel);
        border-left: 4px solid var(--eu-navy);
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .citation-article {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--eu-navy-dark);
    }
    .citation-summary {
        font-size: 0.85rem;
        color: #3a4a6b;
        margin-top: 0.25rem;
    }

    /* ── Ambiguity boxes ── */
    .ambiguity-box {
        background: #EFF6FF;
        border-left: 4px solid var(--eu-pastel-mid);
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #1e3a5f;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        font-size: 0.76rem;
        color: #6b7280;
        font-style: italic;
        border-top: 2px solid var(--eu-pastel);
        padding-top: 1rem;
        margin-top: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="eu-header">
    <div class="eu-stars">★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★</div>
    <h1>⚖️ AI Act Risk Classifier</h1>
    <p>Classify AI system use cases under <strong>Regulation (EU) 2024/1689</strong>
    — article-by-article legal justification powered by RAG + LLM.</p>
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
    st.markdown("### About")
    st.markdown(
        "Classifies AI system use cases under the EU AI Act:\n\n"
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
        "⚠️ **Not legal advice.** [Read disclaimer](https://github.com/juanjimenoizquierdo-ui/ai-act-classifier/blob/master/docs/legal_disclaimer.md)"
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
    st.caption("Describe the AI system you want to classify — the more detail, the better the analysis.")
    use_case_own = st.text_area(
        "Use case description",
        height=140,
        placeholder="e.g. A recruitment platform that automatically screens CVs and ranks candidates before human review...",
        label_visibility="collapsed",
    )
    btn_own = st.button("Classify", type="primary", use_container_width=True, key="btn_own")

with tab_example:
    st.caption("Select a pre-built case to see how the classifier works across different risk levels.")
    selected = st.selectbox(
        "Example cases",
        options=list(EXAMPLES.keys()),
        label_visibility="collapsed",
    )
    st.info(EXAMPLES[selected])
    btn_example = st.button("Classify this example", type="primary", use_container_width=True, key="btn_example")

# Resolve which use case to classify
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
                    "To enable classifications on this deployment, add your Anthropic API key "
                    "in **Settings → Secrets**:\n\n"
                    "```toml\nANTHROPIC_API_KEY = \"sk-ant-...\"\nLLM_PROVIDER = \"claude\"\n```\n\n"
                    "Get a free key with $5 credit at [console.anthropic.com](https://console.anthropic.com)."
                )
            else:
                st.error(f"Unexpected error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.stop()

        cfg = RISK_STYLES.get(result.risk_level, RISK_STYLES[RiskLevel.UNCLEAR])
        conf_icon, conf_label = CONFIDENCE_CONFIG.get(
            result.confidence, ("🔴", "Low confidence")
        )

        # Risk banner
        st.markdown(
            f"""
            <div class="risk-banner" style="background:{cfg['bg']}; color:{cfg['text']}">
                <div class="risk-label">{cfg['icon']} {cfg['label']}</div>
                <div class="risk-desc">{cfg['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence
        st.markdown(f"{conf_icon} {conf_label}")
        st.divider()

        # Legal basis
        if result.primary_citations:
            st.markdown('<div class="section-label">Legal Basis</div>', unsafe_allow_html=True)
            for citation in result.primary_citations:
                st.markdown(
                    f"""
                    <div class="citation-box">
                        <div class="citation-article">{citation.article}</div>
                        <div class="citation-summary">{citation.summary}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Reasoning
        st.markdown('<div class="section-label">Legal Reasoning</div>', unsafe_allow_html=True)
        st.markdown(result.reasoning)

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
