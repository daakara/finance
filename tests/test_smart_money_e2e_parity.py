"""
ARX Terminal — Smart Money End-to-End Authority & Synchronization Test Suite.
Directive: ARX_SMART_MONEY_E2E_PARITY_001

14 Adversarial Parity Scenarios:
1. Live Feed Disconnected -> UNAVAILABLE / PIPELINE_HELD, Confluence reweights (no score leakage, no 55 neutral fallback).
2. Authoritative Completed Search with 0 Results -> VERIFIED_NO_ACTIVITY, Confluence scores 50.0 (neutral, 0 alpha).
3. Unverified Empty Array -> Must NOT convert into VERIFIED_NO_ACTIVITY; fails closed to UNAVAILABLE / PIPELINE_HELD.
4. Single Form 4 Insider Buy -> VERIFIED_SIGNAL (BULLISH), Confluence receives bullish contribution.
5. Single Form 4 Insider Sale -> VERIFIED_SIGNAL (BEARISH), Confluence receives warning contribution.
6. Form 4 Late Filer (>2 business days) -> Late filer count recorded, compliance lag identified.
7. Congressional Purchase within Statutory Window -> VERIFIED_SIGNAL (BULLISH), Confluence receives bullish contribution.
8. Congressional Sale -> VERIFIED_SIGNAL (BEARISH), Confluence receives warning contribution.
9. Congressional Late Filer (>45 days) -> Staleness decay penalty applied, compliance tier Non-Compliant.
10. Unusual Options Call Sweep on Ask Side -> VERIFIED_SIGNAL (BULLISH).
11. Unusual Options Put Sweep on Ask Side -> VERIFIED_SIGNAL (BEARISH).
12. Mixed Divergent Signals (Insider Buy + Options Put Sweep) -> Direction MIXED, Balanced neutral score.
13. Inapplicable Asset Class (ETF / Crypto) -> Form 4 marked INAPPLICABLE, non-blocking, clean reweight or exempt.
14. Curated Research Archive Query (include_curated=True) -> CURATED_RESEARCH with explicit research provenance tag.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst_dashboard.analyzers.smart_money import (
    SmartMoneyEngine,
    SmartMoneyDomainStatus,
    SmartMoneyDirection,
)
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine
from analyst_dashboard.analyzers.decision_trace import DecisionTraceEngine


@pytest.fixture
def confluence():
    return ConfluenceEngine()


# -----------------------------------------------------------------------------
# Case 1: Live Feed Disconnected
# -----------------------------------------------------------------------------
def test_case_01_live_feed_disconnected(confluence):
    """Case 1: Live feed disconnected -> Returns PIPELINE_HELD, Confluence reweights with no score leakage."""
    res = SmartMoneyEngine.assess_symbol("AAPL", include_curated=False)

    assert res["overall_status"] == SmartMoneyDomainStatus.PIPELINE_HELD.value
    assert res["pillar_eligibility"] is False
    assert res["pillar_status"] == "unavailable"
    assert res["pillar_score"] is None
    assert res["reweight_required"] is True
    assert "HOLD_SEC_001" in res["active_holds"]

    # Verify sub-domains
    assert res["subdomains"]["options_flow"]["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
    assert res["subdomains"]["sec_form_4"]["status"] == SmartMoneyDomainStatus.PIPELINE_HELD.value
    assert res["subdomains"]["congressional_stock_act"]["status"] == SmartMoneyDomainStatus.PIPELINE_HELD.value

    # Confluence Engine Synthesis: Must reweight across technicals, fundamentals, macro
    conf = confluence.calculate_confluence(
        symbol="AAPL",
        technical_data={"current_price": 225.0, "executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 2.5},
        smart_money_data=res,
        fundamental_data={"piotroski_f": 8, "quality_score": 85.0},
        macro_data={"yield_curve_10y2y": 0.25, "credit_spread": 3.5},
    )

    smart_pillar = next(p for p in conf["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
    assert smart_pillar["status"] == "unavailable"
    assert smart_pillar["score"] == 0.0

    # Weights must be redistributed: 0.35 tech, 0.35 fund, 0.30 macro
    tech_pillar = next(p for p in conf["pillars"] if p["pillar"] == "TECHNICAL_STRUCTURE")
    fund_pillar = next(p for p in conf["pillars"] if p["pillar"] == "FUNDAMENTAL_SOLVENCY")
    macro_pillar = next(p for p in conf["pillars"] if p["pillar"] == "MACRO_SAFETY_FLOOR")

    assert tech_pillar["weight"] == 0.35
    assert fund_pillar["weight"] == 0.35
    assert macro_pillar["weight"] == 0.30
    assert smart_pillar["weight"] == 0.0


# -----------------------------------------------------------------------------
# Case 2: Authoritative Completed Search with 0 Results
# -----------------------------------------------------------------------------
def test_case_02_verified_no_activity(confluence):
    """Case 2: Authoritative search confirms 0 results -> VERIFIED_NO_ACTIVITY, Confluence scores 50.0 neutral."""
    res = SmartMoneyEngine.assess_symbol(
        symbol="AAPL",
        options_verified_empty=True,
        sec_verified_empty=True,
        congress_verified_empty=True,
    )

    assert res["overall_status"] == SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value
    assert res["pillar_eligibility"] is True
    assert res["pillar_status"] == "neutral"
    assert res["pillar_score"] == 50.0
    assert res["reweight_required"] is False
    assert res["directional_bias"] == SmartMoneyDirection.NEUTRAL.value

    # Confluence Engine Synthesis: 50.0 score with active 25% weight, zero alpha leakage
    conf = confluence.calculate_confluence(
        symbol="AAPL",
        technical_data={"current_price": 225.0, "executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 2.5},
        smart_money_data=res,
        fundamental_data={"piotroski_f": 8, "quality_score": 85.0},
        macro_data={"yield_curve_10y2y": 0.25, "credit_spread": 3.5},
    )

    smart_pillar = next(p for p in conf["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
    assert smart_pillar["status"] == "neutral"
    assert smart_pillar["score"] == 50.0
    assert smart_pillar["weight"] == 0.25


# -----------------------------------------------------------------------------
# Case 3: Unverified Empty Array (Empty array != zero activity)
# -----------------------------------------------------------------------------
def test_case_03_unverified_empty_array_fails_closed():
    """Case 3: Unverified empty array without confirmed search -> Fails closed to UNAVAILABLE / PIPELINE_HELD."""
    res = SmartMoneyEngine.assess_symbol(
        symbol="AAPL",
        live_options=[],
        live_congress=[],
        options_verified_empty=False,
        congress_verified_empty=False,
    )

    # Empty array without verified flag must NOT be emitted as VERIFIED_NO_ACTIVITY
    assert res["overall_status"] != SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value
    assert res["subdomains"]["options_flow"]["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
    assert res["subdomains"]["congressional_stock_act"]["status"] == SmartMoneyDomainStatus.UNAVAILABLE.value
    assert res["pillar_eligibility"] is False


# -----------------------------------------------------------------------------
# Case 4: Single Form 4 Insider Buy
# -----------------------------------------------------------------------------
def test_case_04_single_insider_buy(confluence):
    """Case 4: Verified Form 4 purchase ($2.5M, CEO) -> VERIFIED_SIGNAL (BULLISH), Confluence receives bullish boost."""
    live_sec = {
        "status": "AVAILABLE_WITH_RESULTS",
        "items": [
            {
                "ticker": "AAPL",
                "insider_name": "Tim Cook",
                "title": "Chief Executive Officer",
                "transaction_type": "Purchase (Open Market)",
                "transaction_code": "P",
                "total_value": 2500000.0,
                "days_to_filing": 1,
            }
        ],
    }

    res = SmartMoneyEngine.assess_symbol("AAPL", live_sec=live_sec)

    assert res["overall_status"] == SmartMoneyDomainStatus.VERIFIED_SIGNAL.value or res["overall_status"] == SmartMoneyDomainStatus.PARTIAL_COVERAGE.value
    assert res["subdomains"]["sec_form_4"]["status"] == SmartMoneyDomainStatus.VERIFIED_SIGNAL.value
    assert res["subdomains"]["sec_form_4"]["directional_bias"] == SmartMoneyDirection.BULLISH.value
    assert res["pillar_eligibility"] is True
    assert res["pillar_status"] == "positive"
    assert res["pillar_score"] >= 70.0
    assert res["has_insider_buy"] is True
    assert res["insider_value_usd"] == 2500000.0

    conf = confluence.calculate_confluence(
        symbol="AAPL",
        technical_data={"current_price": 225.0, "executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 2.5},
        smart_money_data=res,
        fundamental_data={"piotroski_f": 8, "quality_score": 85.0},
        macro_data={"yield_curve_10y2y": 0.25, "credit_spread": 3.5},
    )

    smart_pillar = next(p for p in conf["pillars"] if p["pillar"] == "SMART_MONEY_FLOW")
    assert smart_pillar["status"] == "positive"
    assert smart_pillar["score"] >= 70.0


# -----------------------------------------------------------------------------
# Case 5: Single Form 4 Insider Sale
# -----------------------------------------------------------------------------
def test_case_05_single_insider_sale(confluence):
    """Case 5: Verified Form 4 open-market sale ($3.0M) -> VERIFIED_SIGNAL (BEARISH), Confluence receives warning."""
    live_sec = {
        "status": "AVAILABLE_WITH_RESULTS",
        "items": [
            {
                "ticker": "AAPL",
                "insider_name": "Executive VP",
                "transaction_type": "Sale (Open Market)",
                "transaction_code": "S",
                "total_value": 3000000.0,
                "days_to_filing": 2,
            }
        ],
    }

    res = SmartMoneyEngine.assess_symbol("AAPL", live_sec=live_sec)

    assert res["subdomains"]["sec_form_4"]["status"] == SmartMoneyDomainStatus.VERIFIED_SIGNAL.value
    assert res["subdomains"]["sec_form_4"]["directional_bias"] == SmartMoneyDirection.BEARISH.value
    assert res["pillar_eligibility"] is True
    assert res["pillar_status"] == "warning"
    assert res["pillar_score"] <= 40.0
    assert res["has_insider_sell"] is True


# -----------------------------------------------------------------------------
# Case 6: Form 4 Late Filer
# -----------------------------------------------------------------------------
def test_case_06_form_4_late_filer():
    """Case 6: Form 4 filing latency > 2 business days (>4 calendar days) -> Late filer count recorded."""
    live_sec = {
        "status": "AVAILABLE_WITH_RESULTS",
        "items": [
            {
                "ticker": "AAPL",
                "insider_name": "Director",
                "transaction_type": "Purchase (Open Market)",
                "transaction_code": "P",
                "total_value": 500000.0,
                "days_to_filing": 14,  # Violates 2-business-day rule
            }
        ],
    }

    res = SmartMoneyEngine.assess_symbol("AAPL", live_sec=live_sec)
    assert res["subdomains"]["sec_form_4"]["late_filers_count"] == 1


# -----------------------------------------------------------------------------
# Case 7: Congressional Stock Act Purchase
# -----------------------------------------------------------------------------
def test_case_07_congressional_purchase(confluence):
    """Case 7: Congressional purchase within 45d statutory window -> BULLISH, Confluence receives boost."""
    live_congress = [
        {
            "id": "cong-test-1",
            "politician": "Rep. Jane Doe (D-CA)",
            "chamber": "House",
            "ticker": "AAPL",
            "transaction_type": "Purchase (Common Stock)",
            "amount_range": "$100,000 - $250,000",
            "filing_date": "2026-08-20",
            "transaction_date": "2026-08-05",
            "days_to_filing": 15,
            "signal_strength": 90,
            "details": {
                "committee_assignments": ["Financial Services"],
                "historical_win_rate_pct": 72.0,
            },
        }
    ]

    res = SmartMoneyEngine.assess_symbol("AAPL", live_congress=live_congress)

    assert res["subdomains"]["congressional_stock_act"]["status"] == SmartMoneyDomainStatus.VERIFIED_SIGNAL.value
    assert res["subdomains"]["congressional_stock_act"]["directional_bias"] == SmartMoneyDirection.BULLISH.value
    assert res["pillar_score"] >= 65.0
    assert res["has_congress_buy"] is True


# -----------------------------------------------------------------------------
# Case 8: Congressional Stock Act Sale
# -----------------------------------------------------------------------------
def test_case_08_congressional_sale():
    """Case 8: Congressional sale -> BEARISH directional bias."""
    live_congress = [
        {
            "id": "cong-test-2",
            "politician": "Sen. John Smith (R-TX)",
            "chamber": "Senate",
            "ticker": "AAPL",
            "transaction_type": "Sale (Full)",
            "amount_range": "$250,000 - $500,000",
            "filing_date": "2026-08-20",
            "transaction_date": "2026-08-01",
            "days_to_filing": 19,
            "signal_strength": 85,
            "details": {
                "committee_assignments": ["Commerce, Science, and Transportation"],
            },
        }
    ]

    res = SmartMoneyEngine.assess_symbol("AAPL", live_congress=live_congress)

    assert res["subdomains"]["congressional_stock_act"]["status"] == SmartMoneyDomainStatus.VERIFIED_SIGNAL.value
    assert res["subdomains"]["congressional_stock_act"]["directional_bias"] == SmartMoneyDirection.BEARISH.value
    assert res["pillar_status"] == "warning"


# -----------------------------------------------------------------------------
# Case 9: Congressional Late Filer (>45 days)
# -----------------------------------------------------------------------------
def test_case_09_congressional_late_filer():
    """Case 9: Congressional trade filed >45 days late -> Staleness penalty applied."""
    live_congress = [
        {
            "id": "cong-test-late",
            "politician": "Sen. Tommy Tuberville (R-AL)",
            "chamber": "Senate",
            "ticker": "AAPL",
            "transaction_type": "Purchase (Common Stock)",
            "amount_range": "$250,000 - $500,000",
            "filing_date": "2026-08-26",
            "transaction_date": "2026-06-25",
            "days_to_filing": 62,
            "signal_strength": 92,
            "details": {
                "committee_assignments": ["Senate Armed Services"],
            },
        }
    ]

    res = SmartMoneyEngine.assess_symbol("AAPL", live_congress=live_congress)
    cong_sub = res["subdomains"]["congressional_stock_act"]

    assert cong_sub["late_filers_count"] == 1
    enriched_record = cong_sub["records"][0]
    assert enriched_record["staleness_status"] == "LATE_FILER"
    assert enriched_record["compliance_tier"] == "Non-Compliant (>45 Days Late)"
    assert enriched_record["staleness_penalty"] == 32


# -----------------------------------------------------------------------------
# Case 10: Unusual Options Call Sweep on Ask Side
# -----------------------------------------------------------------------------
def test_case_10_options_call_sweep_ask():
    """Case 10: Institutional Call Sweep bought at Ask -> BULLISH."""
    live_options = [
        {
            "ticker": "AAPL",
            "type": "CALL",
            "side": "ASK",
            "premium": 1500000.0,
            "strike": 240.0,
            "expiry": "2026-11-20",
        }
    ]

    res = SmartMoneyEngine.assess_symbol("AAPL", live_options=live_options)
    opt_sub = res["subdomains"]["options_flow"]

    assert opt_sub["status"] == SmartMoneyDomainStatus.VERIFIED_SIGNAL.value
    assert opt_sub["directional_bias"] == SmartMoneyDirection.BULLISH.value
    assert opt_sub["observation_count"] == 1
    assert res["has_options_flow"] is True


# -----------------------------------------------------------------------------
# Case 11: Unusual Options Put Sweep on Ask Side
# -----------------------------------------------------------------------------
def test_case_11_options_put_sweep_ask():
    """Case 11: Institutional Put Sweep bought at Ask -> BEARISH."""
    live_options = [
        {
            "ticker": "AAPL",
            "type": "PUT",
            "side": "ASK",
            "premium": 2000000.0,
            "strike": 210.0,
            "expiry": "2026-11-20",
        }
    ]

    res = SmartMoneyEngine.assess_symbol("AAPL", live_options=live_options)
    opt_sub = res["subdomains"]["options_flow"]

    assert opt_sub["status"] == SmartMoneyDomainStatus.VERIFIED_SIGNAL.value
    assert opt_sub["directional_bias"] == SmartMoneyDirection.BEARISH.value
    assert res["pillar_score"] <= 40.0


# -----------------------------------------------------------------------------
# Case 12: Mixed Divergent Signals
# -----------------------------------------------------------------------------
def test_case_12_mixed_divergent_signals():
    """Case 12: Insider Buy + Options Put Sweep -> Direction MIXED, balanced score near 50."""
    live_sec = {
        "status": "AVAILABLE_WITH_RESULTS",
        "items": [
            {
                "ticker": "AAPL",
                "insider_name": "CFO",
                "transaction_type": "Purchase",
                "transaction_code": "P",
                "total_value": 1500000.0,
                "days_to_filing": 1,
            }
        ],
    }
    live_options = [
        {
            "ticker": "AAPL",
            "type": "PUT",
            "side": "ASK",
            "premium": 1500000.0,
        }
    ]

    res = SmartMoneyEngine.assess_symbol("AAPL", live_sec=live_sec, live_options=live_options)

    assert res["subdomains"]["sec_form_4"]["directional_bias"] == SmartMoneyDirection.BULLISH.value
    assert res["subdomains"]["options_flow"]["directional_bias"] == SmartMoneyDirection.BEARISH.value
    assert res["pillar_status"] == "neutral"
    assert 45.0 <= res["pillar_score"] <= 65.0


# -----------------------------------------------------------------------------
# Case 13: Inapplicable Asset Class (ETF / Crypto)
# -----------------------------------------------------------------------------
def test_case_13_inapplicable_asset_class():
    """Case 13: ETF assets (SPY) or Digital Assets (BTC-USD) -> Form 4 marked INAPPLICABLE."""
    spy_res = SmartMoneyEngine.assess_symbol("SPY", asset_class="ETF", security_type="ETF")
    assert spy_res["subdomains"]["sec_form_4"]["status"] == SmartMoneyDomainStatus.INAPPLICABLE.value
    assert spy_res["subdomains"]["sec_form_4"]["available"] is False

    btc_res = SmartMoneyEngine.assess_symbol("BTC-USD", asset_class="CRYPTO", security_type="CRYPTO")
    assert btc_res["subdomains"]["sec_form_4"]["status"] == SmartMoneyDomainStatus.INAPPLICABLE.value


# -----------------------------------------------------------------------------
# Case 14: Curated Research Archive Query
# -----------------------------------------------------------------------------
def test_case_14_curated_research_archive():
    """Case 14: Curated research archive query (include_curated=True) -> CURATED_RESEARCH with explicit disclosure."""
    res = SmartMoneyEngine.assess_symbol("NVDA", include_curated=True)

    assert res["overall_status"] == SmartMoneyDomainStatus.CURATED_RESEARCH.value
    assert res["pillar_eligibility"] is True
    assert res["subdomains"]["congressional_stock_act"]["status"] == SmartMoneyDomainStatus.CURATED_RESEARCH.value
    assert "Curated historical" in res["subdomains"]["congressional_stock_act"]["disclosure"]
    assert "Not a real-time live trading feed" in res["subdomains"]["congressional_stock_act"]["disclosure"]


# -----------------------------------------------------------------------------
# Decision Trace Parity Check
# -----------------------------------------------------------------------------
def test_decision_trace_smart_money_synchronization(confluence):
    """Verify DecisionTrace records canonical smart money fields without loss."""
    res = SmartMoneyEngine.assess_symbol(
        symbol="AAPL",
        options_verified_empty=True,
        sec_verified_empty=True,
        congress_verified_empty=True,
    )

    conf = confluence.calculate_confluence(
        symbol="AAPL",
        technical_data={"current_price": 225.0, "executionStatus": "IN_BUY_ZONE", "riskRewardRatio": 2.5},
        smart_money_data=res,
        fundamental_data={"piotroski_f": 8, "quality_score": 85.0},
        macro_data={"yield_curve_10y2y": 0.25, "credit_spread": 3.5},
    )

    trace = DecisionTraceEngine.build_decision_trace(
        symbol="AAPL",
        current_price=225.0,
        candles=[{"time": "2026-10-31", "close": 225.0}] * 60,
        freshness={"status": "LIVE", "providerSource": "exchange", "candleCount": 60},
        technicals={},
        confluence=conf,
        factor_scores={"qualityScore": 85.0},
        optimal_execution={"execution_status": "IN_BUY_ZONE", "risk_reward_ratio": 2.5},
        smart_money=res,
    )

    sm_trace = trace["trace"]["smartMoney"]
    assert sm_trace["overallStatus"] == SmartMoneyDomainStatus.VERIFIED_NO_ACTIVITY.value
    assert sm_trace["score"] == 50.0
    assert sm_trace["status"] == "neutral"
    assert sm_trace["reweighted"] is False
    assert sm_trace["hasCongressBuy"] is False
    assert sm_trace["hasInsiderBuy"] is False
    assert "options_flow" in sm_trace["subdomains"]
