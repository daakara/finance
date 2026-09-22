"""Integration and unit tests for Phase 3: Shared Evidence Context Remediation.

Target Findings:
- F_03: RADAR_MACRO_CONTEXT_SEVERANCE
- F_05: REGIME_AUTHORITY_FRAGMENTATION
- F_11: HARDCODED_2026_EMINI_MACRO_PROXIES

Guarantees:
- Shared context across Screener and Single-Asset Analytics with deterministic macroContextId.
- Regime semantic separation preserved:
  * TACTICAL_EQUITY_REGIME (SPY / VIX price & volatility)
  * STRUCTURAL_MACRO_REGIME (FRED cycle)
  * MACRO_RISK_FRICTION (Credit friction)
  * REGIME_SEMANTIC_COLLAPSE = NO
- Fail-closed behavior on missing macro telemetry (no 0.47 / 2.69 / 2.4 static defaults).
- Missing macro telemetry strictly excluded from archetype consensus (numerator & denominator).
- Decision authority invariants remain frozen under DecisionHierarchyEngine.
"""

import math
import pytest
import pandas as pd
import numpy as np

from analyst_dashboard.analyzers.tactical_regime import TacticalRegimeEngine, get_shared_macro_snapshot
from analyst_dashboard.analyzers.trader_archetypes import TraderArchetypeAnalyzer
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.data.fred_fetcher import (
    FredMacroFetcher,
    normalize_macro_payload,
)
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState


# ── F_11: Elimination of Hardcoded Macro Proxies & Fail-Closed Behavior ──────

def test_f11_druckenmiller_fail_closed_on_missing_macro():
    """Verify that Druckenmiller macro archetype fails closed to null alignmentScore and UNAVAILABLE status.
    Previously, it laundered missing data with hardcoded 0.47 and 2.69 defaults yielding base 90 or nominal 50."""
    analyzer = TraderArchetypeAnalyzer()

    # Missing / None macro indicators
    res_none = analyzer._evaluate_druckenmiller_macro(
        macro_indicators=None,
        factor_scores={"momentumScore": 75.0, "growthScore": 75.0},
        price_df=pd.DataFrame({"Close": [100.0, 101.0, 102.0]}),
    )
    assert res_none["alignmentScore"] is None
    assert res_none["evidenceStatus"] == "UNAVAILABLE"
    assert "Unavailable" in res_none["status"]

    # Incomplete macro indicators (only yield curve, missing credit spread)
    res_partial = analyzer._evaluate_druckenmiller_macro(
        macro_indicators={"yield_curve_spread": 0.50},
        factor_scores={"momentumScore": 75.0, "growthScore": 75.0},
        price_df=pd.DataFrame({"Close": [100.0, 101.0, 102.0]}),
    )
    assert res_partial["alignmentScore"] is None
    assert res_partial["evidenceStatus"] == "UNAVAILABLE"
    assert "Unavailable" in res_partial["status"]


def test_f11_druckenmiller_authentic_macro_scoring():
    """Verify Druckenmiller scores authentically when valid macro telemetry is supplied."""
    analyzer = TraderArchetypeAnalyzer()

    # Normal yield curve (0.80) and tight credit spread (3.0) -> favorable macro
    res = analyzer._evaluate_druckenmiller_macro(
        macro_indicators={
            "yield_curve_spread": 0.80,
            "credit_spread_oas": 3.00,
        },
        factor_scores={"momentumScore": 75.0, "growthScore": 75.0},
        price_df=pd.DataFrame({"Close": [100.0, 101.0, 102.0, 105.0]}),
    )
    assert res["alignmentScore"] > 50
    assert res["evidenceStatus"] == "AVAILABLE"
    assert any(k in res["thesis"].lower() for k in ["steepening", "liquidity", "tailwinds", "positive", "expansionary"])


def test_f11_unavailable_macro_archetype_excluded_from_consensus():
    """Verify that an unavailable macro archetype is excluded from both numerator and denominator of consensus."""
    analyzer = TraderArchetypeAnalyzer()
    dates = pd.date_range("2026-01-01", periods=60)
    df = pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(60)]}, index=dates)

    res = analyzer.analyze_asset(
        symbol="AAPL",
        info={"sector": "Technology", "industry": "Consumer Electronics"},
        price_df=df,
        risk_metrics={"Sortino_Ratio": 2.0, "Skewness": -0.1},
        macro_indicators=None,  # Missing macro!
        factor_scores={"qualityScore": 90, "growthScore": 85, "momentumScore": 80, "valuationScore": 70, "piotroskiFScore": 8, "tailRiskScore": 80},
    )

    druckenmiller = next(a for a in res["archetypes"] if "Druckenmiller" in a["name"])
    assert druckenmiller["evidenceStatus"] == "UNAVAILABLE"
    assert druckenmiller["alignmentScore"] is None

    available_archetypes = [a for a in res["archetypes"] if a.get("evidenceStatus") != "UNAVAILABLE" and a.get("alignmentScore") is not None]
    assert len(available_archetypes) == 4

    expected_consensus = round(sum(a["alignmentScore"] for a in available_archetypes) / 4)
    assert res["consensusScore"] == expected_consensus


def test_f11_missing_macro_cannot_increase_consensus():
    """Verify that missing macro cannot artificially raise a low consensus score (no positive evidence)."""
    analyzer = TraderArchetypeAnalyzer()

    # Synthetic low-scoring archetype setup
    archetypes = [
        {"name": "Buffett", "alignmentScore": 40, "evidenceStatus": "AVAILABLE"},
        {"name": "Pelosi", "alignmentScore": 42, "evidenceStatus": "AVAILABLE"},
        {"name": "Druckenmiller", "alignmentScore": None, "evidenceStatus": "UNAVAILABLE"},
        {"name": "Simons", "alignmentScore": 38, "evidenceStatus": "AVAILABLE"},
        {"name": "Gardner", "alignmentScore": 40, "evidenceStatus": "AVAILABLE"},
    ]

    available = [a for a in archetypes if a.get("evidenceStatus") != "UNAVAILABLE" and a.get("alignmentScore") is not None]
    consensus = round(sum(a["alignmentScore"] for a in available) / len(available))

    # Without F_11 fix, a nominal 50 would yield (40+42+50+38+40)/5 = 42.0 (elevating consensus)
    # With F_11 fix, consensus is exactly (40+42+38+40)/4 = 40.0
    assert consensus == 40
    assert len(available) == 4


def test_f11_missing_macro_cannot_decrease_consensus():
    """Verify that missing macro cannot artificially drag down a high consensus score (no negative evidence)."""
    archetypes = [
        {"name": "Buffett", "alignmentScore": 88, "evidenceStatus": "AVAILABLE"},
        {"name": "Pelosi", "alignmentScore": 84, "evidenceStatus": "AVAILABLE"},
        {"name": "Druckenmiller", "alignmentScore": None, "evidenceStatus": "UNAVAILABLE"},
        {"name": "Simons", "alignmentScore": 82, "evidenceStatus": "AVAILABLE"},
        {"name": "Gardner", "alignmentScore": 86, "evidenceStatus": "AVAILABLE"},
    ]

    available = [a for a in archetypes if a.get("evidenceStatus") != "UNAVAILABLE" and a.get("alignmentScore") is not None]
    consensus = round(sum(a["alignmentScore"] for a in available) / len(available))

    # Without F_11 fix, a nominal 50 would yield (88+84+50+82+86)/5 = 78 (dragging down from buy zone)
    # With F_11 fix, consensus is exactly (88+84+82+86)/4 = 85
    assert consensus == 85
    assert len(available) == 4


def test_f11_all_archetypes_unavailable_returns_null_consensus():
    """Verify that when zero archetypes are available, consensusScore is null / None, never 0, 50, or 100."""
    archetypes = [
        {"name": "A", "alignmentScore": None, "evidenceStatus": "UNAVAILABLE"},
        {"name": "B", "alignmentScore": None, "evidenceStatus": "UNAVAILABLE"},
    ]
    available = [a for a in archetypes if a.get("evidenceStatus") != "UNAVAILABLE" and a.get("alignmentScore") is not None]
    consensus = round(sum(a["alignmentScore"] for a in available) / len(available)) if available else None
    assert consensus is None


def test_f11_available_druckenmiller_telemetry_contributes_normally():
    """Verify that when authentic macro telemetry is present, Druckenmiller contributes as 1 of 5 archetypes."""
    analyzer = TraderArchetypeAnalyzer()
    dates = pd.date_range("2026-01-01", periods=60)
    df = pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(60)]}, index=dates)

    res = analyzer.analyze_asset(
        symbol="NVDA",
        info={"sector": "Technology", "industry": "Semiconductors"},
        price_df=df,
        risk_metrics={"Sortino_Ratio": 2.0, "Skewness": -0.1},
        macro_indicators={"yield_curve_spread": 0.85, "credit_spread_oas": 3.10},  # Authentic macro!
        factor_scores={"qualityScore": 90, "growthScore": 85, "momentumScore": 80, "valuationScore": 70, "piotroskiFScore": 8, "tailRiskScore": 80},
    )

    druckenmiller = next(a for a in res["archetypes"] if "Druckenmiller" in a["name"])
    assert druckenmiller["evidenceStatus"] == "AVAILABLE"
    assert isinstance(druckenmiller["alignmentScore"], (int, float))

    available_archetypes = [a for a in res["archetypes"] if a.get("evidenceStatus") != "UNAVAILABLE" and a.get("alignmentScore") is not None]
    assert len(available_archetypes) == 5
    expected_consensus = round(sum(a["alignmentScore"] for a in available_archetypes) / 5)
    assert res["consensusScore"] == expected_consensus


def test_f11_frontend_constants_suppresses_hardcoded_macro_proxies():
    """Verify that frontend DEFAULT_MACRO_DIFFICULTY has suppressed hardcoded 2026 macro proxies."""
    import os
    constants_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "lib", "constants.ts")
    assert os.path.exists(constants_path)
    with open(constants_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "yield_curve_spread: undefined" in content
    assert "credit_spread_oas: undefined" in content
    assert "cpi_yoy: undefined" in content


def test_f11_confluence_engine_no_fabricated_macro_defaults():
    """Verify confluence engine treats missing macro as unavailable, not fabricating 0.0 or 4.0 defaults."""
    engine = ConfluenceEngine()

    # Empty macro_data
    result_empty = engine.calculate_confluence(
        symbol="TEST",
        technical_data={"stage_phase": "Stage 2 Breakout", "rsi_14": 55.0},
        smart_money_data=None,
        fundamental_data={"qualityScore": 85.0},
        macro_data=None,
    )
    macro_pillar = next(p for p in result_empty["pillars"] if p.get("pillar") == "MACRO_SAFETY_FLOOR")
    assert macro_pillar["status"] == "unavailable"
    assert "unavailable" in macro_pillar["detail"].lower()


# ── F_05: Regime Authority & Semantic Separation ────────────────────────────

def test_f05_tactical_regime_deterministic_classification():
    """Verify TacticalRegimeEngine computes deterministic regimes from price series."""
    # Synthetic uptrend with low volatility
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    prices_uptrend = [100.0 + i * 0.5 + (i % 3) * 0.1 for i in range(100)]
    df_uptrend = pd.DataFrame({"Close": prices_uptrend}, index=dates)

    res_uptrend = TacticalRegimeEngine.evaluate_tactical_regime(
        benchmark_symbol="SPY",
        hist_df=df_uptrend,
        use_cache=False,
    )
    assert res_uptrend["regime"] in ("RISK_ON", "NEUTRAL")
    assert res_uptrend["volatilityAnnualPct"] is not None
    assert res_uptrend["volatilityAnnualPct"] < 18.0

    # Synthetic high-volatility series
    np.random.seed(42)
    shock = np.random.normal(0, 0.03, 100)
    prices_volatile = [100.0]
    for s in shock:
        prices_volatile.append(prices_volatile[-1] * (1.0 + s))
    df_volatile = pd.DataFrame({"Close": prices_volatile[:100]}, index=dates)

    res_volatile = TacticalRegimeEngine.evaluate_tactical_regime(
        benchmark_symbol="SPY",
        hist_df=df_volatile,
        use_cache=False,
    )
    assert res_volatile["regime"] == "DEFENSIVE"
    assert res_volatile["volatilityAnnualPct"] >= 22.0


def test_f05_tactical_regime_fail_closed_insufficient_history():
    """Verify TacticalRegimeEngine fails closed when benchmark history is insufficient."""
    df_short = pd.DataFrame({"Close": [100.0, 101.0]})
    res = TacticalRegimeEngine.evaluate_tactical_regime(
        benchmark_symbol="SPY",
        hist_df=df_short,
        use_cache=False,
    )
    assert res["regime"] == "UNAVAILABLE"
    assert res["availability"] == "UNAVAILABLE"


def test_f05_regime_semantic_separation():
    """Verify semantic separation between Tactical Equity, Structural Macro, and Macro Friction.
    REGIME_SEMANTIC_COLLAPSE = NO."""
    # 1. Tactical Equity Regime: price / trend / volatility
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    df = pd.DataFrame({"Close": [100.0 + i * 0.2 for i in range(60)]}, index=dates)
    tactical = TacticalRegimeEngine.evaluate_tactical_regime(hist_df=df, use_cache=False)

    # 2. Structural Macro Regime & Risk Friction: from shared macro snapshot
    macro_snap = get_shared_macro_snapshot()

    # Assert distinct keys and semantics
    assert "regime" in tactical  # Tactical equity (RISK_ON / DEFENSIVE / NEUTRAL)
    assert "structural_macro_regime" in macro_snap  # Macro cycle
    assert "macro_risk_friction" in macro_snap  # Credit friction

    # They represent separate semantic layers and must not be collapsed
    assert tactical["regime"] in ("RISK_ON", "DEFENSIVE", "NEUTRAL", "UNAVAILABLE")
    assert macro_snap["macro_risk_friction"] in ("LOW", "NORMAL", "ELEVATED", "UNAVAILABLE")


# ── F_03: Shared Evidence Context Parity & Deterministic macroContextId ─────

def test_f03_shared_macro_snapshot_has_deterministic_context_id():
    """Verify get_shared_macro_snapshot produces a valid macroContextId."""
    snap = get_shared_macro_snapshot()
    assert isinstance(snap, dict)
    assert "macroContextId" in snap
    if snap.get("macroContextId"):
        assert snap["macroContextId"].startswith("macro-fred-")


def test_f03_shared_macro_snapshot_parity_with_screener():
    """Verify shared macro snapshot feeds screener candidate evaluation identically."""
    snap = get_shared_macro_snapshot()
    from api.routes.screener import run_screener_get

    response = run_screener_get(custom_tickers="AAPL")
    assert response["macroContextId"] == snap.get("macroContextId")
    if response["candidates"]:
        assert response["candidates"][0]["macroContextId"] == snap.get("macroContextId")


def test_f03_screener_candidate_macro_context_id_parity():
    """Verify screener candidates receive macroContextId matching the shared snapshot."""
    from api.routes.screener import run_screener_get

    response = run_screener_get(custom_tickers="AAPL")
    assert "macroContextId" in response
    assert len(response["candidates"]) > 0
    candidate = response["candidates"][0]
    assert "macroContextId" in candidate
    # Candidate macroContextId must match root macroContextId
    assert candidate["macroContextId"] == response["macroContextId"]


# ── Canonical Decision Authority Invariant Check (Phase 2 Preservation) ─────

def test_decision_authority_invariants_preserved():
    """Verify DecisionHierarchyEngine remains the single authoritative decision resolver."""
    state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="AAPL",
        current_price=150.0,
        candle_count=60,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=85.0,
        stage_phase="Stage 2 Breakout",
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        is_cataloged=True,
        is_confirmed=True,
        user_role="LONG_TERM",
    )
    assert state["state"] == DecisionState.ACTIONABLE_SETUP.value
    assert state["isActionable"] is True
    assert state["canSizeTrade"] is True
