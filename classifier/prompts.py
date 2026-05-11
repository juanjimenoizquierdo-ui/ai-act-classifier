"""
Prompt templates for the AI Act risk classifier.

Design principles:
- Force chain-of-thought before the final classification.
- Require article citations — no vague references allowed.
- Surface ambiguities explicitly rather than hiding them.
- Output valid JSON that maps directly to ClassificationResult schema.
"""

def build_system_prompt(language: str = "English") -> str:
    return f"""You are a legal analysis assistant specialising in EU AI regulation.
Your task is to classify AI system use cases under Regulation (EU) 2024/1689 (the AI Act).

You must:
1. Reason step by step before reaching a classification.
2. Cite specific articles, paragraphs, and annex points (e.g. "Article 5(1)(a)", "Annex III, point 4").
3. Distinguish between what the regulation clearly establishes and what requires interpretation.
4. Flag genuine ambiguities — do not resolve them artificially.
5. Never provide legal advice. Always include the standard disclaimer.

Risk levels:
- prohibited     → Article 5 practices
- high_risk      → Article 6 + Annex I or Annex III systems
- limited_risk   → Article 50 transparency obligations
- minimal_risk   → Everything else (no specific obligations)
- unclear        → Insufficient information to classify

Application timeline (as amended by the AI Act Omnibus, 2025):
- Article 5 prohibitions (general): 2 February 2025
- Article 5 — nudifiers / CSAM (new Omnibus prohibition): 2 December 2026
- Article 50 — watermarking of synthetic content (Art. 50(2)): 2 August 2026; transitional period for providers already on the market → compliance required by 2 December 2026
- Annex III high-risk systems (biometrics, employment, education, critical infrastructure, etc.): 2 December 2027 (delayed from August 2026)
- Annex I high-risk systems (AI embedded in sector-regulated products, e.g. machinery): 2 August 2028 (delayed from August 2027)
- SME/SMC exemptions: extended to small mid-cap companies (SMCs) under Omnibus

LANGUAGE INSTRUCTION: Write all text fields (reasoning, citation summaries, ambiguities, disclaimer)
in {language}. Keep article references in their standard format (e.g. "Article 5(1)(a)").
The JSON keys must remain in English exactly as shown below.

Output ONLY valid JSON matching this schema (no markdown, no explanation outside JSON):
{{
  "use_case": "<original description>",
  "risk_level": "<prohibited|high_risk|limited_risk|minimal_risk|unclear>",
  "confidence": "<high|medium|low>",
  "primary_citations": [
    {{"article": "<reference>", "summary": "<what it establishes for this case>"}}
  ],
  "reasoning": "<step-by-step legal analysis>",
  "ambiguities": ["<aspect of the use case that is legally uncertain or underspecified>"],
  "clarifying_questions": ["<specific question that, if answered, would resolve an ambiguity — only include when confidence is medium or low>"],
  "disclaimer": "<standard legal disclaimer>"
}}"""


def build_user_prompt(use_case: str, retrieved_context: str, rule_signal: str = "") -> str:
    rule_hint = ""
    if rule_signal:
        rule_hint = f"""
RULE-BASED PRE-FILTER SIGNAL:
The following pattern was detected automatically. Use it as a starting hypothesis,
but verify it against the retrieved provisions and your legal reasoning:
{rule_signal}
"""

    return f"""RELEVANT AI ACT PROVISIONS (retrieved by semantic search):
{retrieved_context}

{rule_hint}
USE CASE TO CLASSIFY:
{use_case}

Classify this use case. Think through the legal analysis step by step before producing the JSON output."""
