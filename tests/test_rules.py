"""
Unit tests for the rule-based pre-filter.
These run without any API calls or corpus — pure logic tests.

Run: python -m pytest tests/test_rules.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from classifier.rules import apply_rules
from models.schemas import RiskLevel


def test_prohibited_social_scoring():
    signal = apply_rules("A government system that assigns social credit scores to citizens based on their behaviour.")
    assert signal is not None
    assert signal.risk_level == RiskLevel.PROHIBITED


def test_prohibited_realtime_biometric():
    signal = apply_rules("Real-time facial recognition deployed in train stations to identify individuals.")
    assert signal is not None
    assert signal.risk_level == RiskLevel.PROHIBITED


def test_high_risk_recruitment():
    signal = apply_rules("An AI tool that screens CVs and ranks candidates for recruitment purposes.")
    assert signal is not None
    assert signal.risk_level == RiskLevel.HIGH


def test_high_risk_credit():
    signal = apply_rules("A model that evaluates creditworthiness for loan applications.")
    assert signal is not None
    assert signal.risk_level == RiskLevel.HIGH


def test_limited_risk_chatbot():
    signal = apply_rules("A customer service chatbot that handles user queries.")
    assert signal is not None
    assert signal.risk_level == RiskLevel.LIMITED


def test_no_signal_for_generic_recommendation():
    signal = apply_rules("An AI system that recommends movies based on viewing history.")
    assert signal is None  # No specific pattern — full LLM analysis needed


def test_high_risk_law_enforcement():
    signal = apply_rules("A predictive policing tool used by police to assess crime risk in neighbourhoods.")
    assert signal is not None
    # Could be prohibited (predictive policing) or high risk (law enforcement)
    assert signal.risk_level in (RiskLevel.PROHIBITED, RiskLevel.HIGH)
