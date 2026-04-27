"""
Main classifier — orchestrates rules → RAG → LLM pipeline.

Flow:
  1. Rule pre-filter (fast, keyword-based)
  2. RAG retrieval (semantic search over AI Act corpus)
  3. LLM classification with chain-of-thought + article citations
  4. Parse and validate output as ClassificationResult
"""

import json
import os
import re
from dotenv import load_dotenv

load_dotenv()  # local dev; Streamlit Cloud injects secrets as env vars automatically

from models.schemas import ClassificationResult, RiskLevel, ArticleCitation
from classifier.rules import apply_rules
from classifier.retriever import AIActRetriever
from classifier.prompts import build_system_prompt, build_user_prompt

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude").lower()


def _call_claude(system: str, user: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


def _call_ollama(system: str, user: str) -> str:
    import ollama
    model = os.getenv("OLLAMA_MODEL", "llama3.3")

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.1},  # low temp for legal reasoning
    )
    return response["message"]["content"]


def _call_llm(system: str, user: str) -> str:
    if LLM_PROVIDER == "ollama":
        return _call_ollama(system, user)
    return _call_claude(system, user)


def _parse_response(raw: str, original_use_case: str) -> ClassificationResult:
    """Parse LLM JSON output into ClassificationResult. Handles markdown code fences."""
    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        # Remove unescaped control characters that smaller LLMs sometimes emit
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Second attempt: replace literal newlines inside string values
        try:
            cleaned2 = re.sub(r'(?<!\\)\n', ' ', cleaned)
            data = json.loads(cleaned2)
        except json.JSONDecodeError as e:
            # Fallback: return unclear with the raw reasoning
            return ClassificationResult(
                use_case=original_use_case,
                risk_level=RiskLevel.UNCLEAR,
                confidence="low",
                primary_citations=[],
                reasoning=f"Failed to parse LLM output: {e}\n\nRaw output:\n{raw}",
                ambiguities=["LLM output could not be parsed — manual review required"],
            )

    # Normalise citations
    citations = [
        ArticleCitation(article=c["article"], summary=c["summary"])
        for c in data.get("primary_citations", [])
    ]

    return ClassificationResult(
        use_case=data.get("use_case", original_use_case),
        risk_level=RiskLevel(data.get("risk_level", "unclear")),
        confidence=data.get("confidence", "low"),
        primary_citations=citations,
        reasoning=data.get("reasoning", ""),
        ambiguities=data.get("ambiguities", []),
        disclaimer=data.get("disclaimer", ClassificationResult.model_fields["disclaimer"].default),
    )


def classify(use_case: str, n_retrieved: int = 8, language: str = "English") -> ClassificationResult:
    """
    Classify an AI system use case under the EU AI Act.

    Args:
        use_case: Free-text description of the AI system and its intended use.
        n_retrieved: Number of corpus chunks to retrieve for RAG context.
        language: Output language for reasoning and text fields (e.g. "Spanish", "French").

    Returns:
        ClassificationResult with risk level, citations, and reasoning.
    """
    # Step 1 — Rule pre-filter
    rule_signal = apply_rules(use_case)
    rule_hint = ""
    if rule_signal:
        rule_hint = (
            f"Risk level signal: {rule_signal.risk_level.value}\n"
            f"Legal reference: {rule_signal.legal_reference}\n"
            f"Matched keywords: {', '.join(rule_signal.matched_keywords)}"
        )

    # Step 2 — RAG retrieval
    retriever = AIActRetriever()
    chunks = retriever.retrieve(use_case, n_results=n_retrieved)
    context = retriever.format_for_prompt(chunks)

    # Step 3 — LLM classification
    system_prompt = build_system_prompt(language)
    user_prompt = build_user_prompt(use_case, context, rule_hint)
    raw_response = _call_llm(system_prompt, user_prompt)

    # Step 4 — Parse and return
    return _parse_response(raw_response, use_case)
