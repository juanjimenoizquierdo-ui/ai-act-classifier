"""
Rule-based pre-filter for AI Act risk classification.

Maps keyword signals to risk categories based on:
- Article 5     → Prohibited practices
- Annex III     → High-risk categories (8 areas)
- Article 50    → Limited risk (transparency obligations)

This is NOT the final classification — it is a signal to guide RAG retrieval
and LLM reasoning. Ambiguous cases are flagged for full analysis.
"""

from dataclasses import dataclass
from models.schemas import RiskLevel


@dataclass
class RuleSignal:
    risk_level: RiskLevel
    matched_keywords: list[str]
    legal_reference: str
    confidence: str  # "high" | "medium" | "low"


# ── Article 5 — Prohibited Practices ─────────────────────────────────────────
PROHIBITED_PATTERNS: list[tuple[list[str], str]] = [
    (
        ["subliminal", "subconscious", "unconscious manipulation", "without awareness"],
        "Article 5(1)(a) — Subliminal manipulation",
    ),
    (
        ["exploit", "vulnerability", "vulnerable groups", "disability", "age-based manipulation"],
        "Article 5(1)(b) — Exploitation of vulnerabilities",
    ),
    (
        ["social scoring", "social credit", "citizen score", "trustworthiness score", "general purpose scoring"],
        "Article 5(1)(c) — Social scoring by public authorities",
    ),
    (
        ["real-time biometric", "live facial recognition", "public space surveillance", "remote biometric identification"],
        "Article 5(1)(d) — Real-time remote biometric identification in public spaces",
    ),
    (
        ["emotion recognition", "workplace emotion", "educational emotion"],
        "Article 5(1)(f) — Emotion recognition (workplace/education)",
    ),
    (
        ["biometric categorisation", "political opinion", "religious belief", "sexual orientation", "race inference"],
        "Article 5(1)(g) — Biometric categorisation for sensitive attributes",
    ),
    (
        ["predictive policing", "crime prediction", "individual crime risk", "criminal profiling"],
        "Article 5(1)(e) — Predictive policing based solely on profiling",
    ),
]

# ── Annex III — High-Risk Areas ───────────────────────────────────────────────
HIGH_RISK_PATTERNS: list[tuple[list[str], str]] = [
    # Point 1 — Biometric identification (non-prohibited)
    (
        ["biometric", "facial recognition", "fingerprint", "voice recognition", "gait recognition", "identity verification"],
        "Annex III, point 1 — Biometric identification and categorisation",
    ),
    # Point 2 — Critical infrastructure
    (
        ["critical infrastructure", "electricity grid", "water supply", "gas network", "transport safety", "digital infrastructure"],
        "Annex III, point 2 — Critical infrastructure management",
    ),
    # Point 3 — Education
    (
        ["student assessment", "educational admission", "exam evaluation", "academic performance", "learning outcome", "student scoring"],
        "Annex III, point 3 — Education and vocational training",
    ),
    # Point 4 — Employment
    (
        ["recruitment", "cv screening", "hiring decision", "employee monitoring", "performance evaluation", "promotion decision", "termination"],
        "Annex III, point 4 — Employment and workers management",
    ),
    # Point 5 — Essential services
    (
        ["credit scoring", "loan decision", "insurance risk", "benefit eligibility", "social assistance", "health insurance", "creditworthiness"],
        "Annex III, point 5 — Access to essential private/public services",
    ),
    # Point 6 — Law enforcement
    (
        ["law enforcement", "police", "criminal investigation", "evidence analysis", "threat assessment", "lie detection", "forensic"],
        "Annex III, point 6 — Law enforcement",
    ),
    # Point 7 — Migration and border control
    (
        ["migration", "asylum", "border control", "visa assessment", "immigration risk", "refugee"],
        "Annex III, point 7 — Migration, asylum and border control",
    ),
    # Point 8 — Administration of justice
    (
        ["judicial decision", "court", "sentencing", "legal dispute resolution", "justice administration", "legal aid"],
        "Annex III, point 8 — Administration of justice and democratic processes",
    ),
]

# ── Article 50 — Limited Risk (transparency obligations) ──────────────────────
LIMITED_RISK_PATTERNS: list[tuple[list[str], str]] = [
    (
        ["chatbot", "conversational agent", "virtual assistant", "customer service bot"],
        "Article 50(1) — Chatbots interacting with natural persons",
    ),
    (
        ["deepfake", "synthetic media", "generated image", "ai-generated content", "synthetic video"],
        "Article 50(4) — Deepfake and AI-generated content",
    ),
    (
        ["emotion recognition system", "biometric categorisation system"],
        "Article 50(3) — Emotion recognition / biometric categorisation disclosure",
    ),
]


def _matches(text: str, keywords: list[str]) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower]


def apply_rules(use_case: str) -> RuleSignal | None:
    """
    Returns the strongest RuleSignal found, or None if no clear signal.
    Priority: prohibited > high_risk > limited_risk.
    """
    # Check prohibited first — highest priority
    for keywords, reference in PROHIBITED_PATTERNS:
        matched = _matches(use_case, keywords)
        if matched:
            return RuleSignal(
                risk_level=RiskLevel.PROHIBITED,
                matched_keywords=matched,
                legal_reference=reference,
                confidence="medium",  # always verify with LLM — Article 5 has exceptions
            )

    # Check high-risk
    for keywords, reference in HIGH_RISK_PATTERNS:
        matched = _matches(use_case, keywords)
        if matched:
            return RuleSignal(
                risk_level=RiskLevel.HIGH,
                matched_keywords=matched,
                legal_reference=reference,
                confidence="medium",
            )

    # Check limited risk
    for keywords, reference in LIMITED_RISK_PATTERNS:
        matched = _matches(use_case, keywords)
        if matched:
            return RuleSignal(
                risk_level=RiskLevel.LIMITED,
                matched_keywords=matched,
                legal_reference=reference,
                confidence="low",
            )

    return None  # No rule matched — full RAG + LLM analysis needed
