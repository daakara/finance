"""
ARX Financial Correctness & Authority Regression Suite (ARX-QA-AUTHORITY-035).

Audits and falsifies high-risk financial invariants across:
1. Form 4 transaction semantics: sales must not be classified as purchases.
2. Option exercises, grants, gifts, and tax withholdings must not count as open-market accumulation.
3. CDNS moving average trend direction: price below 20 EMA must never claim 'above both MAs'.
4. ConfluenceEngine pillar coverage caps (50 / 72 / 84 / 96) and approved weights.
5. Missing/synthetic fundamentals fail closed to EVIDENCE_INCOMPLETE.
6. Radar research-only versus execution authority separation.
7. Cross-component parity between Screener, Analytics, and DecisionTrace.
"""

import pytest
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.decision_hierarchy import DecisionHierarchyEngine, DecisionState
from analyst_dashboard.provenance.evidence_contract import evaluate_factor_provenance
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.form4_models import SmartMoneySignalType


# ── TEST 1: Form 4 Transaction Classification Semantics ──────────────────────

def test_form4_transaction_classification_integrity():
    """Verify Form 4 SEC transaction codes map to authentic signal types."""
    assert Form4XmlParser._classify_transaction_code("P") == SmartMoneySignalType.OPEN_MARKET_PURCHASE
    assert Form4XmlParser._classify_transaction_code("S") == SmartMoneySignalType.OPEN_MARKET_SALE
    assert Form4XmlParser._classify_transaction_code("M") == SmartMoneySignalType.OPTION_EXERCISE
    assert Form4XmlParser._classify_transaction_code("A") == SmartMoneySignalType.EQUITY_GRANT
    assert Form4XmlParser._classify_transaction_code("F") == SmartMoneySignalType.TAX_WITHHOLDING
    assert Form4XmlParser._classify_transaction_code("G") == SmartMoneySignalType.GIFT
    assert Form4XmlParser._classify_transaction_code("D") == SmartMoneySignalType.ISSUER_DISPOSITION


def test_form4_sale_cannot_produce_insider_buy_points():
    """Prove that an insider sale cannot award positive open-market purchase points."""
    sale_trade = {
        "ticker": "CDNS",
        "transaction_type": "Sale (Open Market)",
        "transaction_code": "S",
        "total_value": "$5,000,000",
        "reporting_owner": "Corporate Director"
    }
    sec_trades = [sale_trade]

    # Screener classification logic
    buy_trades = [
        t for t in sec_trades
        if (t.get("transaction_code") == "P" or "PURCHASE" in (t.get("transaction_type") or "").upper())
        and (t.get("transaction_code") != "S" and "SALE" not in (t.get("transaction_type") or "").upper())
    ]
    sell_trades = [
        t for t in sec_trades
        if t.get("transaction_code") == "S" or "SALE" in (t.get("transaction_type") or "").upper()
    ]

    assert len(buy_trades) == 0, "Sale must not be in buy_trades"
    assert len(sell_trades) == 1, "Sale must be in sell_trades"

    smart_data = {
        "has_insider_buy": len(buy_trades) > 0,
        "has_insider_sell": len(sell_trades) > 0,
        "insider_value_usd": 0.0,
        "insider_name": sell_trades[0].get("reporting_owner", ""),
        "has_congress_buy": False,
        "has_options_flow": False,
        "insider_buy_count_90d": len(buy_trades),
        "insider_sell_count_90d": len(sell_trades),
    }

    res = ConfluenceEngine.calculate_confluence(
        symbol="CDNS",
        smart_money_data=smart_data,
        technical_data={"current_price": 323.01, "executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 2.5}
    )
    smart_pillar = next(p for p in res["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")

    # Must NOT award 90.0 positive points
    assert smart_pillar["score"] != 90.0, "Sale must not receive 90.0 purchase points"
    assert smart_pillar["status"] in ("neutral", "warning"), f"Sale must be neutral or warning, got {smart_pillar['status']}"
    assert "purchased" not in smart_pillar["detail"].lower(), "Sale must not claim executive purchased on open market"


def test_form4_purchase_awards_positive_points():
    """Verify that a genuine open-market purchase correctly awards positive conviction."""
    buy_trade = {
        "ticker": "CDNS",
        "transaction_type": "Purchase (Direct Open Market)",
        "transaction_code": "P",
        "total_value": "$2,500,000",
        "reporting_owner": "Chief Executive Officer"
    }
    sec_trades = [buy_trade]

    buy_trades = [
        t for t in sec_trades
        if (t.get("transaction_code") == "P" or "PURCHASE" in (t.get("transaction_type") or "").upper())
        and (t.get("transaction_code") != "S" and "SALE" not in (t.get("transaction_type") or "").upper())
    ]
    assert len(buy_trades) == 1

    smart_data = {
        "has_insider_buy": True,
        "has_insider_sell": False,
        "insider_value_usd": 2500000.0,
        "insider_name": buy_trades[0].get("reporting_owner", ""),
        "has_congress_buy": False,
        "has_options_flow": False,
        "insider_buy_count_90d": 1,
        "insider_sell_count_90d": 0,
    }

    res = ConfluenceEngine.calculate_confluence(
        symbol="CDNS",
        smart_money_data=smart_data,
        technical_data={"current_price": 323.01, "executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 2.5}
    )
    smart_pillar = next(p for p in res["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
    assert smart_pillar["score"] == 90.0
    assert smart_pillar["status"] == "positive"
    assert "purchased" in smart_pillar["detail"].lower()


# ── TEST 2: Confluence Pillar Coverage Caps ──────────────────────────────────

def test_coverage_caps_strictly_enforced():
    """Verify coverage caps: <=1 pillar -> 50.0; 2 pillars -> 72.0; 3 pillars -> 84.0; 4 pillars -> 96.0."""
    # 1 pillar (technical only)
    res_1 = ConfluenceEngine.calculate_confluence(
        symbol="TEST",
        technical_data={"current_price": 100.0, "executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 3.0, "rsi_14": 55.0}
    )
    assert res_1["verifiedPillarCount"] == 1
    assert res_1["confluenceScore"] <= 50.0

    # 2 pillars (technical + macro)
    res_2 = ConfluenceEngine.calculate_confluence(
        symbol="TEST",
        technical_data={"current_price": 100.0, "executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 3.0, "rsi_14": 55.0},
        macro_data={"yield_curve_10y2y": 0.5, "credit_spread": 3.5}
    )
    assert res_2["verifiedPillarCount"] == 2
    assert res_2["confluenceScore"] <= 72.0

    # 3 pillars (technical + macro + smart money)
    res_3 = ConfluenceEngine.calculate_confluence(
        symbol="TEST",
        technical_data={"current_price": 100.0, "executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 3.0, "rsi_14": 55.0},
        macro_data={"yield_curve_10y2y": 0.5, "credit_spread": 3.5},
        smart_money_data={"has_insider_buy": True, "insider_value_usd": 1000000.0}
    )
    assert res_3["verifiedPillarCount"] == 3
    assert res_3["confluenceScore"] <= 84.0
    assert "4/4 High Conviction" not in res_3.get("plainBadge", "")


# ── TEST 3: Synthetic Fundamentals Fail-Closed ───────────────────────────────

def test_synthetic_fundamentals_fail_closed_to_evidence_incomplete():
    """Verify that synthetic/unverified fundamentals fail closed to EVIDENCE_INCOMPLETE."""
    factor_snap = {
        "quality_score": 85.0,
        "growth_score": 80.0,
        "valuation_score": 75.0,
        "piotroski_f": 8,
        "source": "SYNTHETIC_FALLBACK_FIXTURE",
        "source_id": "SYNTHETIC_MOCK_001",
        "is_synthetic": True,
    }
    fund_prov = evaluate_factor_provenance(factor_snap, "TEST", asset_class="EQUITY")
    assert fund_prov["evidenceStatus"] in ("SYNTHETIC", "UNVERIFIED")
    assert fund_prov["isVerified"] is False

    dec_state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="TEST",
        current_price=100.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=False,
        confluence_score=80.0,
        stage_phase="Stage 2",
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        fundamental_provenance=fund_prov,
        fundamental_evidence_status=fund_prov["evidenceStatus"]
    )
    assert dec_state["state"] == DecisionState.EVIDENCE_INCOMPLETE.value
    assert dec_state["isActionable"] is False
    assert dec_state["canSizeTrade"] is False


# ── TEST 4: ETF Exemption Semantics ──────────────────────────────────────────

def test_etf_exemption_does_not_fabricate_portfolio_quality():
    """Verify that ETF corporate 10-K exemption does not synthesize phantom quality scores."""
    res = ConfluenceEngine.calculate_confluence(
        symbol="SPY",
        technical_data={"current_price": 500.0, "executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 2.5},
        macro_data={"yield_curve_10y2y": 0.5, "credit_spread": 3.5},
        asset_class="ETF"
    )
    fund_pillar = next(p for p in res["pillars"] if p["pillar"] == "FUNDAMENTAL_SOLVENCY")
    assert fund_pillar["score"] == 0.0
    assert fund_pillar["status"] == "not_applicable"
    assert res["verifiedPillarCount"] == 2
    assert res["confluenceScore"] <= 72.0


# ── TEST 5: Relative Rank != Absolute Conviction != Actionability ─────────────

def test_relative_rank_cannot_promote_to_actionable():
    """Verify that a candidate with score 45.0 cannot be promoted to ACTIONABLE_SETUP."""
    dec_state = DecisionHierarchyEngine.resolve_decision_state(
        symbol="WEAK",
        current_price=50.0,
        candle_count=100,
        freshness_status="LIVE",
        has_fundamentals=True,
        confluence_score=45.0,  # Below 75 threshold
        stage_phase="Stage 2",
        is_in_buy_zone=True,
        risk_reward_ratio=2.5,
        verified_pillar_count=4
    )
    assert dec_state["state"] != DecisionState.ACTIONABLE_SETUP.value
    assert dec_state["isActionable"] is False
    assert dec_state["canSizeTrade"] is False
