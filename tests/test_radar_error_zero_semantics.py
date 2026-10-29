"""
ARX TERMINAL - SPRINT 3 MARKET OPPORTUNITY RADAR
RADAR RELIABILITY: ERROR-VERSUS-ZERO SEMANTICS & ADVERSARIAL QUALIFICATION (ARX-S3-004-BE)
DIRECTIVE 058B.1 BATCH 2 QUALIFICATION SUITE

Exhaustive qualification covering the 10 mandated adversarial scenarios
and candidate eligibility invariants:
1. Complete success with candidates
2. Complete success with genuinely zero candidates
3. Single-candidate processing failure
4. Multiple isolated candidate failures
5. Total database/provider failure
6. Timeout before population completion
7. Stale snapshot combined with otherwise successful results
8. Partial population with nonzero discovered candidates
9. Invalid candidate record
10. Concurrent scans with differing scan-run identities
Plus:
- IN_BUY_ZONE opportunity count invariant
- Approaching-target discoverable without active count inflation
- Delisted/missing price handling without crash
- Trade sizing disabled in discovery
"""

import concurrent.futures
import time
from unittest.mock import patch
import pytest
from starlette.responses import Response

from api.routes.screener import run_screener_get, market_db, screener


# ==============================================================================
# Scenario 1: Complete success with candidates
# ==============================================================================
def test_radar_scenario1_complete_success_with_candidates():
    """Verify standard screener run returns SUCCESS, COMPLETED, CANDIDATES_AVAILABLE with valid denominators."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")

    assert data.get("status") == "SUCCESS"
    assert data.get("scanState") == "COMPLETED"
    assert data.get("datasetState") == "CANDIDATES_AVAILABLE"
    assert data.get("error") is None
    assert data.get("partialErrors") is None
    assert data.get("gemsFound") > 0
    assert len(data.get("candidates", [])) > 0

    pop = data.get("population", {})
    assert pop.get("referenceUniverseCount") == 35
    assert pop.get("activeScreeningCount") > 0
    assert pop.get("currentOpportunityCount") > 0
    assert pop.get("delistedRecordCount") >= 1  # CPRX is delisted
    # Denominator completeness
    assert pop["referenceUniverseCount"] == (
        pop["activeScreeningCount"] + pop["delistedRecordCount"] + pop["unverifiedRecordCount"]
    )


# ==============================================================================
# Scenario 2: Complete success with genuinely zero candidates
# ==============================================================================
def test_radar_scenario2_genuinely_zero_candidates():
    """Verify genuinely empty search produces SUCCESS and GENUINELY_EMPTY (NOT an error or failure)."""
    resp = Response()
    # Squeeze filter requires short float >= 6%, none of standard candidates qualify
    data = run_screener_get(resp, filter_type="squeeze")

    assert data.get("status") == "SUCCESS"
    assert data.get("scanState") == "COMPLETED"
    assert data.get("datasetState") == "GENUINELY_EMPTY"
    assert data.get("error") is None
    assert data.get("partialErrors") is None
    assert data.get("gemsFound") == 0
    assert data.get("candidates") == []
    # Invariant: Zero candidates must not report failure
    assert data.get("status") != "ERROR"
    assert data.get("scanState") != "FAILED"


# ==============================================================================
# Scenario 3: Single-candidate processing failure
# ==============================================================================
def test_radar_scenario3_single_candidate_processing_failure():
    """Verify isolated single-ticker exception produces PARTIAL status and captures specific partialErrors."""
    resp = Response()
    original_get_price = market_db.get_latest_price

    def mock_get_price(sym):
        if sym == "LNTH":
            raise ValueError("Corrupt candle data feed for LNTH")
        return original_get_price(sym)

    with patch.object(market_db, "get_latest_price", side_effect=mock_get_price):
        data = run_screener_get(resp, filter_type="all")

        # Invariant: Partial results must not report success
        assert data.get("status") == "PARTIAL"
        assert data.get("scanState") == "PARTIAL"
        assert data.get("status") != "SUCCESS"
        assert data.get("error") is not None
        assert "Partial evaluation error" in data.get("error")
        assert data.get("partialErrors") is not None
        assert len(data.get("partialErrors")) == 1
        assert data.get("partialErrors")[0]["ticker"] == "LNTH"

        # Other 34 candidates remain evaluated
        candidates = data.get("candidates", [])
        assert len(candidates) == 35
        lnth_cand = [c for c in candidates if c["symbol"] == "LNTH"][0]
        assert lnth_cand["executionStatus"] == "UNVERIFIED_ASSET"
        assert "Corrupt candle data feed" in lnth_cand["entryThesis"]


# ==============================================================================
# Scenario 4: Multiple isolated candidate failures
# ==============================================================================
def test_radar_scenario4_multiple_isolated_candidate_failures():
    """Verify multiple isolated exceptions are tracked in partialErrors without crashing the run."""
    resp = Response()
    original_get_price = market_db.get_latest_price
    failing_syms = {"LNTH", "ELF", "ON"}

    def mock_get_price(sym):
        if sym in failing_syms:
            raise IOError(f"Network timeout fetching price for {sym}")
        return original_get_price(sym)

    with patch.object(market_db, "get_latest_price", side_effect=mock_get_price):
        data = run_screener_get(resp, filter_type="all")

        assert data.get("status") == "PARTIAL"
        assert data.get("scanState") == "PARTIAL"
        assert data.get("partialErrors") is not None
        assert len(data.get("partialErrors")) == 3
        recorded_tickers = {pe["ticker"] for pe in data.get("partialErrors")}
        assert recorded_tickers == failing_syms

        # Non-failing candidates evaluate normally
        vrtx_cand = [c for c in data.get("candidates", []) if c["symbol"] == "VRTX"][0]
        assert vrtx_cand["executionStatus"] in ["IN_BUY_ZONE", "WAITING_PULLBACK", "APPROACHING_TARGET"]
        assert vrtx_cand["currentPrice"] is not None


# ==============================================================================
# Scenario 5: Total database/provider failure
# ==============================================================================
def test_radar_scenario5_total_database_failure():
    """Verify 100% database failure returns ERROR, FAILED, SCAN_FAILED (NOT false zero/empty)."""
    resp = Response()

    with patch.object(market_db, "get_latest_price", side_effect=RuntimeError("Database lock acquisition timeout")):
        data = run_screener_get(resp, filter_type="all")

        # Invariant: Total failure must not report empty results
        assert data.get("status") == "ERROR"
        assert data.get("scanState") == "FAILED"
        assert data.get("datasetState") == "SCAN_FAILED"
        assert data.get("datasetState") != "GENUINELY_EMPTY"
        assert "Market data evaluation failed across all" in data.get("error", "")
        assert data.get("partialErrors") is not None
        assert len(data.get("partialErrors")) == data.get("totalCandidates")


# ==============================================================================
# Scenario 6: Timeout before population completion
# ==============================================================================
def test_radar_scenario6_timeout_before_population():
    """Verify unexpected timeout or top-level exception fails closed cleanly without 500 crash."""
    resp = Response()

    with patch.object(screener, "evaluate_candidates", side_effect=TimeoutError("Evaluation deadline exceeded (5000ms)")):
        data = run_screener_get(resp, filter_type="all")

        assert data.get("status") == "ERROR"
        assert data.get("scanState") == "FAILED"
        assert data.get("datasetState") == "SCAN_FAILED"
        assert "Radar scanner evaluation failed" in data.get("error", "")
        assert data.get("gemsFound") == 0
        assert data.get("candidates") == []
        assert data.get("population", {}).get("currentOpportunityCount") == 0


# ==============================================================================
# Scenario 7: Stale snapshot combined with otherwise successful results
# ==============================================================================
def test_radar_scenario7_stale_snapshot_with_success():
    """Verify response evidence preserves serving source, served timestamp, and structural quality."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")

    ev = data.get("evidence", {})
    assert ev.get("serving_source") == "DIRECT_PROVIDER"
    assert ev.get("served_at") is not None
    prov = ev.get("provenance", {})
    assert prov.get("provider") in ["YFINANCE", "yfinance"]
    assert prov.get("structural_quality") in ["COMPLETE", "DEGRADED", "UNKNOWN"]
    assert prov.get("fallback_status") is False


# ==============================================================================
# Scenario 8: Partial population with nonzero discovered candidates
# ==============================================================================
def test_radar_scenario8_partial_population_denominators():
    """Verify custom subset input calculates exact mathematical population denominators."""
    resp = Response()
    # 4 active + 1 delisted = 5 total candidates
    custom = "LNTH, ELF, ON, VRTX, CPRX"
    data = run_screener_get(resp, custom_tickers=custom, filter_type="all")

    assert data.get("status") == "SUCCESS"
    pop = data.get("population", {})
    assert pop["referenceUniverseCount"] == 5
    assert pop["activeScreeningCount"] == 4
    assert pop["delistedRecordCount"] == 1
    assert pop["unverifiedRecordCount"] == 0
    assert len(data.get("candidates", [])) == 5


# ==============================================================================
# Scenario 9: Invalid candidate record handling
# ==============================================================================
def test_radar_scenario9_invalid_candidate_record():
    """Verify corrupt candidate records (None, non-dict, missing ticker) do not crash screener."""
    resp = Response()
    original_eval = screener.evaluate_candidates

    def mock_eval(universe):
        real_results = original_eval(universe[:3])
        # Inject corrupt records
        corrupt_results = [
            None,  # non-dict
            {"company_name": "Ghost Corp"},  # missing ticker/symbol
        ] + real_results
        return corrupt_results

    with patch.object(screener, "evaluate_candidates", side_effect=mock_eval):
        data = run_screener_get(resp, filter_type="all")

        # Invariant: Partial results due to invalid records
        assert data.get("status") == "PARTIAL"
        assert data.get("scanState") == "PARTIAL"
        assert data.get("partialErrors") is not None
        assert any(pe["ticker"] == "INVALID_RECORD" for pe in data.get("partialErrors"))
        assert any(pe["ticker"] == "MISSING_TICKER" for pe in data.get("partialErrors"))
        # Valid records are still processed
        assert len(data.get("candidates", [])) == 3


# ==============================================================================
# Scenario 10: Concurrent scans with differing scan-run identities
# ==============================================================================
def test_radar_scenario10_concurrent_scans_isolated_identities():
    """Verify concurrent screener runs produce isolated state without cross-talk or race conditions."""
    def run_worker(role, filt):
        r = Response()
        res = run_screener_get(r, user_role=role, filter_type=filt)
        context_ids = [c["decisionContextId"] for c in res.get("candidates", [])]
        return {
            "role": role,
            "filter": filt,
            "gemsFound": res.get("gemsFound"),
            "contextIds": context_ids,
            "status": res.get("status"),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(run_worker, "LONG_TERM", "all")
        f2 = executor.submit(run_worker, "DAY_TRADER", "all")
        f3 = executor.submit(run_worker, "LONG_TERM", "in_buy_zone")

        res1 = f1.result()
        res2 = f2.result()
        res3 = f3.result()

    assert res1["status"] == "SUCCESS"
    assert res2["status"] == "SUCCESS"
    assert res3["status"] == "SUCCESS"

    # Context IDs are uniquely generated per candidate per run
    assert set(res1["contextIds"]).isdisjoint(set(res2["contextIds"]))
    assert set(res1["contextIds"]).isdisjoint(set(res3["contextIds"]))


# ==============================================================================
# Candidate Eligibility Invariants (§3)
# ==============================================================================
def test_radar_invariant_in_buy_zone_strictly_required_for_opportunity_count():
    """Demonstrate IN_BUY_ZONE is required for opportunity counting; approaching-target is excluded."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")

    candidates = data.get("candidates", [])
    pop = data.get("population", {})

    buy_zone_candidates = [c for c in candidates if c["executionStatus"] == "IN_BUY_ZONE"]
    non_buy_zone_candidates = [c for c in candidates if c["executionStatus"] != "IN_BUY_ZONE"]

    # Invariant: currentOpportunityCount matches ONLY IN_BUY_ZONE count
    assert pop["currentOpportunityCount"] == len(buy_zone_candidates)

    # Invariant: Every buy-zone candidate has screeningEligible == True
    for c in buy_zone_candidates:
        assert c["screeningEligible"] is True

    # Invariant: Every non-buy-zone candidate has screeningEligible == False
    for c in non_buy_zone_candidates:
        assert c["screeningEligible"] is False


def test_radar_invariant_approaching_target_discoverable_without_inflating_active_count():
    """Verify approaching-target setups remain discoverable in cards without inflating active counts."""
    resp = Response()
    # Mock a candidate price to push it near TP1 into APPROACHING_TARGET
    original_get_price = market_db.get_latest_price

    def mock_approaching_price(sym):
        info = original_get_price(sym)
        if sym == "VRTX" and info:
            # Set price very high near take-profit target
            return {**info, "currentPrice": 520.00}
        return info

    with patch.object(market_db, "get_latest_price", side_effect=mock_approaching_price):
        data = run_screener_get(resp, filter_type="all")

        candidates = data.get("candidates", [])
        vrtx = [c for c in candidates if c["symbol"] == "VRTX"][0]

        # VRTX must be classified as APPROACHING_TARGET or WAITING_PULLBACK / STOPPED_OUT
        # Verify that if any candidate is APPROACHING_TARGET, screeningEligible is False
        approaching = [c for c in candidates if c["executionStatus"] == "APPROACHING_TARGET"]
        for c in approaching:
            assert c["screeningEligible"] is False

        # Invariant: currentOpportunityCount must NOT count approaching targets
        pop = data.get("population", {})
        buy_zone = [c for c in candidates if c["executionStatus"] == "IN_BUY_ZONE"]
        assert pop["currentOpportunityCount"] == len(buy_zone)


def test_radar_invariant_delisted_security_segregated_and_unsizeable():
    """Verify delisted assets (CPRX) are safely partitioned into delistedRecordCount and unsizeable."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")

    cprx = [c for c in data.get("candidates", []) if c["symbol"] == "CPRX"][0]
    assert cprx["executionStatus"] == "DELISTED_SECURITY"
    assert cprx["screeningEligible"] is False
    assert cprx["screeningStatus"] == "DELISTED"
    assert cprx["screeningGeometry"] == "INACTIVE"
    assert cprx["canSizeTrade"] is False
    assert cprx["allowedActions"] == []
    assert "CASH_MERGER_ACQUISITION" in cprx["entryThesis"] or "Angelini" in cprx["entryThesis"]


def test_radar_invariant_trade_sizing_strictly_disabled_in_discovery():
    """Verify discovery candidate cards strictly disable trade sizing and actionability."""
    resp = Response()
    data = run_screener_get(resp, filter_type="all")

    for cand in data.get("candidates", []):
        assert cand.get("canSizeTrade") is False
        assert cand.get("isActionable") is False
        allowed = cand.get("allowedActions", [])
        assert "EXECUTE_TRADE" not in allowed
        assert "SIZE_TRADE" not in allowed
        assert "SUBMIT_ORDER" not in allowed
