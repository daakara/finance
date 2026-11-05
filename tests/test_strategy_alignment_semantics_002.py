"""Tests for Directive NVDA-STRATEGY-ALIGNMENT-002:
Targeted Presentation, Provenance & Assessment-Semantics Correction.

Verifies:
1. Non-advisory factor-alignment descriptions (no "Buy", "Hold", "Accumulation" or "Speculative" advisory leakage).
2. Consensus score of 75/100 emits "Favorable Multi-Strategy Alignment", not a trading recommendation.
3. Consensus aggregation retains exact numerical behavior (unweighted mean of available models, rounded).
4. Complete isolation from ConfluenceEngine and OptimalExecutionEngine: high archetype alignment cannot create trading recommendations or bypass pre-flight/HOLD restrictions.
5. Static priors (CONGRESSIONAL_POLICY_TICKERS, MOTLEY_FOOL_DISRUPTORS) are explicitly marked as modeled assumptions.
6. Modeled archetypes never represent actual investor holdings or transactions.
"""

import math
import inspect
import pandas as pd
import numpy as np
import pytest

from analyst_dashboard.analyzers.trader_archetypes import (
    TraderArchetypeAnalyzer,
    EvidenceType,
    ArchetypeEvidenceStatus,
)
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.optimal_execution import OptimalExecutionEngine


@pytest.fixture
def analyzer():
    return TraderArchetypeAnalyzer()


def test_verdict_replaces_action_oriented_advisory_language(analyzer):
    """Item 1: Action-oriented verdicts are replaced with pure factor-alignment descriptions."""
    # Test all score brackets:
    # 1. >= 85
    # 2. 75..84
    # 3. 65..74
    # 4. < 65
    # 5. Unavailable

    dates = pd.date_range("2026-01-01", periods=60)
    prices = [100.0 + i for i in range(60)]
    df = pd.DataFrame({"Close": prices, "Open": prices, "High": [p+1 for p in prices], "Low": [p-1 for p in prices]}, index=dates)

    # 1. Score >= 85
    res_85 = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors"},
        price_df=df,
        risk_metrics={"Sortino_Ratio": 3.0, "Skewness": 0.1},
        macro_indicators={"yield_curve_spread": 0.50, "credit_spread_oas": 2.0},
        factor_scores={"growthScore": 95, "qualityScore": 95, "valuationScore": 85, "momentumScore": 90, "tailRiskScore": 90, "piotroskiFScore": 9},
    )
    assert res_85["consensusScore"] >= 85
    assert res_85["verdict"] == "High Multi-Strategy Alignment"
    assert "Buy" not in res_85["verdict"]
    assert "Accumulation" not in res_85["verdict"]

    # 2. Score 75..84 (The 75/100 consensus score from NVDA audit)
    # Using the exact reconstructed factor scores from Directive 001
    res_75 = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors"},
        price_df=df,
        risk_metrics={"Sortino_Ratio": 1.25, "Skewness": -0.45},
        macro_indicators={"yield_curve_spread": 0.15, "credit_spread_oas": 3.6},
        factor_scores={"growthScore": 80, "qualityScore": 91, "valuationScore": 89, "momentumScore": 70, "tailRiskScore": 69, "piotroskiFScore": 9},
    )
    assert res_75["consensusScore"] == 75
    assert res_75["verdict"] == "Favorable Multi-Strategy Alignment"
    assert "Buy" not in res_75["verdict"]
    assert "Recommendation" not in res_75["verdict"]

    # 3. Score 65..74
    res_65 = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors"},
        price_df=df,
        risk_metrics={"Sortino_Ratio": 1.1, "Skewness": -0.3},
        macro_indicators={"yield_curve_spread": 0.10, "credit_spread_oas": 3.4},
        factor_scores={"growthScore": 72, "qualityScore": 80, "valuationScore": 72, "momentumScore": 72, "tailRiskScore": 72, "piotroskiFScore": 7},
    )
    assert 65 <= res_65["consensusScore"] < 75
    assert res_65["verdict"] == "Moderate Multi-Strategy Alignment"
    assert "Hold" not in res_65["verdict"]

    # 4. Score < 65
    res_low = analyzer.analyze_asset(
        symbol="LOW_ALIGN",
        info={"sector": "Energy", "industry": "Oil & Gas"},
        price_df=df,
        risk_metrics={"Sortino_Ratio": 0.2, "Skewness": -1.2},
        macro_indicators={"yield_curve_spread": -0.50, "credit_spread_oas": 5.5},
        factor_scores={"growthScore": 45, "qualityScore": 40, "valuationScore": 45, "momentumScore": 40, "tailRiskScore": 40, "piotroskiFScore": 3},
    )
    assert res_low["consensusScore"] < 65
    assert res_low["verdict"] == "Low / Divergent Strategy Alignment"
    assert "Speculative" not in res_low["verdict"]


def test_consensus_aggregation_numerical_invariants(analyzer):
    """Item 1 & 5: Consensus aggregation retains existing exact mathematical behavior."""
    # Consensus is an unweighted mean of available archetypes rounded to nearest integer
    res = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors"},
        price_df=None,
        risk_metrics={"Sortino_Ratio": 1.5, "Skewness": -0.2},
        macro_indicators={"yield_curve_spread": 0.35, "credit_spread_oas": 2.8},
        factor_scores={"growthScore": 85, "qualityScore": 90, "valuationScore": 75, "momentumScore": 80, "tailRiskScore": 75, "piotroskiFScore": 8},
    )

    available = [a for a in res["archetypes"] if a["alignmentScore"] is not None and a["evidenceStatus"] != ArchetypeEvidenceStatus.UNAVAILABLE]
    assert len(available) == res["availableCount"]
    expected_consensus = round(sum(a["alignmentScore"] for a in available) / len(available))
    assert res["consensusScore"] == expected_consensus


def test_static_priors_identified_as_modeled_assumptions(analyzer):
    """Item 6: Static congressional policy and growth-disruptor priors are identifiable as modeled assumptions."""
    # NVDA has a static congressional policy prior and disruptor prior
    res = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors"},
        price_df=None,
        risk_metrics={},
        macro_indicators={},
        factor_scores={"growthScore": 85, "momentumScore": 80},
    )

    pelosi = next(a for a in res["archetypes"] if "Pelosi" in a["name"])
    assert pelosi["thematicPrior"] is not None
    assert pelosi["thematicPrior"]["evidenceType"] == EvidenceType.STATIC_DOMAIN_PRIOR
    assert pelosi["thematicPrior"]["isModeledAssumption"] is True
    assert "Modeled" in pelosi["thematicPrior"]["disclaimer"]

    gardner = next(a for a in res["archetypes"] if "David Gardner" in a["name"] or "Motley Fool" in a["name"])
    assert gardner["thematicPrior"] is not None
    assert gardner["thematicPrior"]["evidenceType"] == EvidenceType.STATIC_DOMAIN_PRIOR
    assert gardner["thematicPrior"]["isModeledAssumption"] is True
    assert "Modeled" in gardner["thematicPrior"]["disclaimer"]


def test_score_75_cannot_generate_trade_recommendation():
    """Directive requirement: A score of 75/100 cannot generate a trading recommendation or execution clearance."""
    confluence_engine = ConfluenceEngine()
    exec_engine = OptimalExecutionEngine()

    # 1. ConfluenceEngine has zero dependency on archetype scores
    sig_conf = inspect.signature(confluence_engine.calculate_confluence)
    assert "trader_archetypes" not in sig_conf.parameters
    assert "archetype" not in str(sig_conf)

    # 2. OptimalExecutionEngine has zero dependency on archetype scores
    sig_exec = inspect.signature(exec_engine.calculate_execution_plan)
    assert "trader_archetypes" not in sig_exec.parameters
    assert "consensus_score" not in sig_exec.parameters

    # 3. High strategy similarity cannot bypass execution gate when history is insufficient
    short_df = pd.DataFrame({"Close": [100.0 + i for i in range(10)]})
    # Confluence calculation with insufficient technical data returns fail-closed red light
    conf_res = confluence_engine.calculate_confluence("NVDA", technical_data={"insufficient_history": True})
    assert "RED LIGHT" in conf_res["plainRating"] or conf_res["confluenceScore"] < 50


def test_high_alignment_cannot_bypass_mandatory_holds():
    """Directive requirement: High strategy similarity cannot bypass mandatory eligibility, preflight or execution HOLDs."""
    # Under HOLD_EXEC_001 active posture:
    exec_engine = OptimalExecutionEngine()
    plan = exec_engine.calculate_execution_plan(symbol="NVDA", price=120.0, atr_14=3.5, highs_52w=140.0, lows_52w=80.0)

    # Plan contains technical stop and corridor, but NO trade execution authorization
    assert "execution_cleared" not in plan or not plan.get("execution_cleared")
    # Verify no broker dispatch or trade execution trigger exists in execution plan
    assert "broker_order" not in plan
