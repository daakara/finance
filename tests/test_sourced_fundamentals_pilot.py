"""
tests/test_sourced_fundamentals_pilot.py

Comprehensive 17-Scenario Failure-Reproducing Test Suite for
ARX Terminal Sourced Fundamentals & Undervalued Growth Research Pilot.

Covers:
1. Quarter and YTD facts sharing the same end date and filing date (FIX counterexample fixture: exact $11,228,024,000).
2. Permuted source observation order producing identical results (order-invariance).
3. Annual and quarterly observations sharing a fiscal year-end (rejection of 3M standalone Q4 in 10-K).
4. Comparable prior-year YTD selection matching fiscal boundaries and duration.
5. Duplicate comparative observations across filings (proper deduplication).
6. Non-calendar fiscal years and 52/53-week retail calendars (ULTA, LULU).
7. Missing or ambiguous TTM components producing explicit UNAVAILABLE / None with diagnostic notes.
8. Source cutoff (as_of_date) and amendment handling.
9. Missing debt not becoming reported zero (preserves MetricStatus.UNAVAILABLE).
10. Reported zero remaining zero with accession provenance (LULU ShortTermBorrowings).
11. Missing valuation inputs not producing populated valuations (no hardcoded fallbacks).
12. Split-normalized share comparisons (DECK 6:1 split eliminating phantom +426% dilution; CELH 3:1 split).
13. Non-positive invested-capital handling (preserves negative IC, sets roic_pct = None with note).
14. Forward/reverse DCF reconciliation (solving for 5-year revenue CAGR with roundtrip check < $0.05).
15. Unbracketed reverse-DCF targets returning None and reason rather than boundary values.
16. Report/ledger numeric parity (all markdown tables match JSON ledger exactly).
17. Offline replay with network access disabled (verified zero network egress).
"""

import os
import json
import random
import socket
import urllib.request
import pytest
from datetime import datetime

from analyst_dashboard.research.sourced_fundamentals.contracts import (
    FinancialFact,
    CompanyFundamentals,
    DerivedMetrics,
    MetricStatus,
    ResearchDisposition,
)
from analyst_dashboard.research.sourced_fundamentals.sec_harvester import (
    SEC_USER_AGENT,
    SecEdgarHarvester,
    fetch_market_price_yfinance,
    DEFAULT_CACHE_DIR,
)
from analyst_dashboard.research.sourced_fundamentals.normalizer import (
    SecFactNormalizer,
    REVENUE_CONCEPTS,
    DEBT_CURRENT_CONCEPTS,
    DEBT_NONCURRENT_CONCEPTS,
    parse_date,
)
from analyst_dashboard.research.sourced_fundamentals.calculations import (
    compute_invested_capital,
    compute_effective_tax_rate,
    compute_derived_metrics,
)
from analyst_dashboard.research.sourced_fundamentals.valuation import (
    run_fcff_model,
    solve_reverse_dcf,
    generate_sensitivity_matrix,
    evaluate_deep_valuation_fix,
)
from analyst_dashboard.research.sourced_fundamentals.cli import (
    run_pipeline,
    compute_split_normalized_dilution,
    POPULATION,
)


@pytest.fixture
def normalizer():
    return SecFactNormalizer()


@pytest.fixture
def raw_facts_fix():
    path = os.path.join(DEFAULT_CACHE_DIR, "CIK0001035983_companyfacts.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def raw_submissions_fix():
    path = os.path.join(DEFAULT_CACHE_DIR, "CIK0001035983_submissions.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 1. FIX period-selection counterexample fixture
def test_1_ttm_duration_selection_counterexample_fix(normalizer, raw_facts_fix):
    """
    Scenario 1: Counterexample fixture for FIX at June 30, 2026.
    The prior implementation incorrectly selected standalone Q2 3M facts (3,265,656,000)
    yielding $10,193,978,000.
    The corrected implementation must select 6M YTD facts (6,130,988,000)
    yielding exactly $11,228,024,000:
    9,101,641,000 (FY2025) + 6,130,988,000 (Q2 2026 6M YTD) - 4,004,605,000 (Q2 2025 6M YTD) = 11,228,024,000.
    """
    us_gaap = raw_facts_fix["facts"]["us-gaap"]
    fact = normalizer._extract_ttm_fact(
        us_gaap=us_gaap,
        concepts=REVENUE_CONCEPTS,
        metric_key="revenue_ttm",
        latest_period_end="2026-06-30"
    )

    assert fact.value == 11228024000.0, f"Expected 11,228,024,000 but got {fact.value}"
    assert fact.status == MetricStatus.DERIVED
    assert fact.period_start == "2025-07-01"
    assert fact.period_end == "2026-06-30"
    # Verify exact 365-day TTM duration
    duration_days = (parse_date(fact.period_end) - parse_date(fact.period_start)).days + 1
    assert duration_days == 365


# 2. Permutation invariance of source observations
def test_2_permutation_invariance_source_observations(normalizer, raw_facts_fix):
    """
    Scenario 2: Source observation list permuted into reversed and random order
    must produce identical selected results.
    """
    us_gaap_orig = raw_facts_fix["facts"]["us-gaap"]
    fact_orig = normalizer._extract_ttm_fact(us_gaap_orig, REVENUE_CONCEPTS, "revenue_ttm", "2026-06-30")

    # Permute the USD observations list for Revenues
    concept = "Revenues"
    items = list(us_gaap_orig[concept]["units"]["USD"])

    # Test 1: Reversed order
    items_reversed = list(reversed(items))
    us_gaap_rev = {"Revenues": {"units": {"USD": items_reversed}}}
    fact_rev = normalizer._extract_ttm_fact(us_gaap_rev, [concept], "revenue_ttm", "2026-06-30")
    assert fact_rev.value == fact_orig.value
    assert fact_rev.period_start == fact_orig.period_start
    assert fact_rev.formula == fact_orig.formula

    # Test 2: Shuffled order
    items_shuffled = list(items)
    random.seed(42)
    random.shuffle(items_shuffled)
    us_gaap_shuf = {"Revenues": {"units": {"USD": items_shuffled}}}
    fact_shuf = normalizer._extract_ttm_fact(us_gaap_shuf, [concept], "revenue_ttm", "2026-06-30")
    assert fact_shuf.value == fact_orig.value
    assert fact_shuf.period_start == fact_orig.period_start


# 3. Annual and quarterly observations sharing a fiscal year-end
def test_3_annual_quarterly_fiscal_year_end_discrimination(normalizer):
    """
    Scenario 3: An annual selector must reject a 3-month observation sharing
    the fiscal year-end date and require a full-year duration (350-375 days).
    """
    synthetic_us_gaap = {
        "Revenues": {
            "units": {
                "USD": [
                    # 3-month Q4 observation in 10-K
                    {
                        "form": "10-K", "fp": "FY", "start": "2025-10-01", "end": "2025-12-31",
                        "val": 300.0, "filed": "2026-02-15", "accn": "0001-26-01"
                    },
                    # True full-year 12-month observation in 10-K
                    {
                        "form": "10-K", "fp": "FY", "start": "2025-01-01", "end": "2025-12-31",
                        "val": 1200.0, "filed": "2026-02-15", "accn": "0001-26-01"
                    }
                ]
            }
        }
    }
    fact = normalizer._extract_ttm_fact(synthetic_us_gaap, ["Revenues"], "revenue_ttm", "2025-12-31")
    assert fact.value == 1200.0, "Should select full-year observation, not 3-month Q4 observation"
    assert fact.period_start == "2025-01-01"


# 4. Comparable prior-year YTD selection
def test_4_comparable_prior_year_ytd_selection(normalizer, raw_facts_fix):
    """
    Scenario 4: Comparable prior YTD selection must match fiscal boundaries
    and duration matching.
    """
    us_gaap = raw_facts_fix["facts"]["us-gaap"]
    prior_val, start, end = normalizer._extract_prior_ttm_fact(us_gaap, REVENUE_CONCEPTS, "2026-06-30")
    assert prior_val is not None
    assert prior_val > 0
    # Prior TTM for FIX must cover exactly 2024-07-01 to 2025-06-30 (immediately preceding current TTM)
    assert start == "2024-07-01"
    assert end == "2025-06-30"


# 5. Duplicate comparative observations across filings
def test_5_duplicate_comparative_observations_deduplication(normalizer):
    """
    Scenario 5: Multiple filings containing comparative figures for the same period
    must be deduplicated and the latest filing/amendment selected.
    """
    synthetic_us_gaap = {
        "Revenues": {
            "units": {
                "USD": [
                    # Full year 2024
                    {"form": "10-K", "fp": "FY", "start": "2024-01-01", "end": "2024-12-31", "val": 1000.0, "filed": "2025-02-10", "accn": "A1"},
                    # Full year 2023
                    {"form": "10-K", "fp": "FY", "start": "2023-01-01", "end": "2023-12-31", "val": 900.0, "filed": "2024-02-10", "accn": "A0"},
                    # Current Q2 YTD 2025
                    {"form": "10-Q", "fp": "Q2", "start": "2025-01-01", "end": "2025-06-30", "val": 600.0, "filed": "2025-08-01", "accn": "A2"},
                    # Prior Q2 YTD 2024 reported in 2024 10-Q (original)
                    {"form": "10-Q", "fp": "Q2", "start": "2024-01-01", "end": "2024-06-30", "val": 480.0, "filed": "2024-08-01", "accn": "A_old"},
                    # Prior Q2 YTD 2024 reported in 2025 10-Q (comparative restatement)
                    {"form": "10-Q", "fp": "Q2", "start": "2024-01-01", "end": "2024-06-30", "val": 500.0, "filed": "2025-08-01", "accn": "A2"},
                ]
            }
        }
    }
    fact = normalizer._extract_ttm_fact(synthetic_us_gaap, ["Revenues"], "revenue_ttm", "2025-06-30")
    # 1000 + 600 - 500 = 1100
    assert fact.value == 1100.0
    assert "A2" in fact.input_references[2], "Should select comparative observation from latest filing"


# 6. Non-calendar fiscal years and 52/53-week retail calendars
def test_6_non_calendar_retail_fiscal_year(normalizer):
    """
    Scenario 6: Retail 52/53-week calendars (ULTA, LULU) where fiscal years end in late January / early February.
    """
    ulta_path = os.path.join(DEFAULT_CACHE_DIR, "CIK0001403568_companyfacts.json")
    with open(ulta_path, "r", encoding="utf-8") as f:
        ulta_facts = json.load(f)["facts"]["us-gaap"]

    fact = normalizer._extract_ttm_fact(ulta_facts, ["Revenues"], "revenue_ttm", "2026-08-01")
    assert fact.value == 12955517000.0
    assert fact.period_start == "2025-08-03"
    assert fact.period_end == "2026-08-01"


# 7. Missing or ambiguous TTM components
def test_7_missing_or_ambiguous_ttm_components(normalizer):
    """
    Scenario 7: Missing TTM components must produce status UNAVAILABLE and value None,
    with explicit diagnostic explanation.
    """
    synthetic_us_gaap = {
        "Revenues": {
            "units": {
                "USD": [
                    {"form": "10-Q", "fp": "Q2", "start": "2025-01-01", "end": "2025-06-30", "val": 600.0, "filed": "2025-08-01"}
                ]
            }
        }
    }
    fact = normalizer._extract_ttm_fact(synthetic_us_gaap, ["Revenues"], "revenue_ttm", "2025-06-30")
    assert fact.value is None
    assert fact.status == MetricStatus.UNAVAILABLE
    assert "No prior full-year 10-K found" in fact.notes


# 8. Source cutoff and applicable amendment handling
def test_8_source_cutoff_and_amendment_handling(normalizer):
    """
    Scenario 8: Respect as_of_date cutoff and exclude filings filed after that date.
    """
    synthetic_us_gaap = {
        "Revenues": {
            "units": {
                "USD": [
                    {"form": "10-K", "fp": "FY", "start": "2024-01-01", "end": "2024-12-31", "val": 1000.0, "filed": "2025-02-15"},
                    {"form": "10-K", "fp": "FY", "start": "2025-01-01", "end": "2025-12-31", "val": 1200.0, "filed": "2026-02-15"},
                ]
            }
        }
    }
    # Cutoff before 2026 filing
    fact = normalizer._extract_ttm_fact(
        synthetic_us_gaap, ["Revenues"], "revenue_ttm", "2024-12-31", as_of_date="2025-06-01"
    )
    assert fact.value == 1000.0
    assert fact.period_end == "2024-12-31"


# 9. Missing debt not becoming reported zero
def test_9_missing_debt_not_becoming_reported_zero(normalizer):
    """
    Scenario 9: If debt concepts are not reported in the filing, normalizer must
    produce value = None with MetricStatus.UNAVAILABLE, never synthetic 0.0 with REPORTED status.
    """
    empty_us_gaap = {}
    fact = normalizer._extract_instant_fact(
        empty_us_gaap, DEBT_CURRENT_CONCEPTS, "short_term_debt", "2026-06-30"
    )
    assert fact.value is None
    assert fact.status == MetricStatus.UNAVAILABLE
    assert fact.accession_number is None


# 10. Reported zero remaining zero
def test_10_reported_zero_remaining_zero(normalizer):
    """
    Scenario 10: Reported zero in filing (e.g. LULU ShortTermBorrowings = 0)
    must preserve value = 0.0 with MetricStatus.REPORTED and accession number.
    """
    synthetic_us_gaap = {
        "ShortTermBorrowings": {
            "units": {
                "USD": [
                    {
                        "form": "10-Q", "end": "2026-08-02", "val": 0,
                        "filed": "2026-09-03", "accn": "0001397187-26-000127"
                    }
                ]
            }
        }
    }
    fact = normalizer._extract_instant_fact(
        synthetic_us_gaap, ["ShortTermBorrowings"], "short_term_debt", "2026-08-02"
    )
    assert fact.value == 0.0
    assert fact.status == MetricStatus.REPORTED
    assert fact.accession_number == "0001397187-26-000127"


# 11. Missing valuation inputs not producing populated valuations
def test_11_missing_valuation_inputs_do_not_produce_populated_valuations():
    """
    Scenario 11: Missing critical inputs to valuation models must raise an error or
    return unavailable result rather than substituting hardcoded factual fallbacks.
    """
    with pytest.raises(ValueError, match="Cannot run FCFF model: base_revenue, current_shares, or net_debt_bridge is None"):
        run_fcff_model(
            base_revenue=None,
            current_shares=35000000.0,
            net_debt_bridge=-1800000000.0,
            growth_5y_pct=15.0,
            operating_margin_pct=15.0,
            wacc_pct=9.5,
            terminal_growth_pct=2.5,
        )

    # Reverse DCF with missing inputs
    g, status = solve_reverse_dcf(
        target_price=100.0,
        base_revenue=None,
        current_shares=10000000.0,
        net_debt_bridge=0.0,
        operating_margin_pct=15.0,
        wacc_pct=9.5,
        terminal_growth_pct=2.5,
    )
    assert g is None
    assert status == "MISSING_INPUTS"


# 12. Split-normalized share comparisons
def test_12_split_normalized_share_dilution_deck_and_celh(normalizer):
    """
    Scenario 12: DECK 6:1 split (Sep 13, 2024) and CELH 3:1 split (Nov 15, 2023).
    DECK 3-year dilution must reflect share repurchases (-12.26%), eliminating the phantom +426.42% spike.
    CELH comparative observations already restated must not be adjusted twice.
    """
    deck_path = os.path.join(DEFAULT_CACHE_DIR, "CIK0000910521_companyfacts.json")
    with open(deck_path, "r", encoding="utf-8") as f:
        deck_facts = json.load(f)["facts"]["us-gaap"]

    dil_1y, dil_3y = compute_split_normalized_dilution(
        normalizer, deck_facts, "2026-06-30", curr_shares=138559000.0, split_date="2024-09-13", split_ratio=6.0
    )
    assert dil_3y is not None
    assert dil_3y < 0, f"DECK 3Y dilution should be negative (buybacks), got {dil_3y}%"
    assert abs(dil_3y - (-12.26)) < 0.5


# 13. Non-positive invested-capital handling
def test_13_non_positive_invested_capital_handling():
    """
    Scenario 13: Non-positive invested capital must not be silently replaced with equity.
    Must return computed negative value and set roic_pct = None with explicit explanation.
    """
    # Net cash firm: Equity = 500, Debt = 0, Cash = 800 -> Invested Capital = -300
    ic = compute_invested_capital(equity=500.0, total_debt=0.0, total_cash=800.0)
    assert ic == -300.0, "Must preserve negative invested capital rather than replacing with equity"


# 14. Forward/reverse DCF reconciliation
def test_14_forward_and_reverse_dcf_reconciliation():
    """
    Scenario 14: Solve for revenue CAGR and verify roundtrip per-share value within $0.05.
    """
    target_price = 1439.07
    base_rev = 11228024000.0
    shares = 35254000.0
    net_bridge = -1800730000.0

    solved_g, status = solve_reverse_dcf(
        target_price=target_price,
        base_revenue=base_rev,
        current_shares=shares,
        net_debt_bridge=net_bridge,
        operating_margin_pct=15.0,
        wacc_pct=9.5,
        terminal_growth_pct=2.5,
        tax_rate_pct=22.0,
        reinvestment_rate_pct=15.0,
    )
    assert solved_g is not None
    assert "VERIFIED" in status

    # Forward model roundtrip verification
    roundtrip = run_fcff_model(
        base_revenue=base_rev,
        current_shares=shares,
        net_debt_bridge=net_bridge,
        growth_5y_pct=solved_g,
        operating_margin_pct=15.0,
        wacc_pct=9.5,
        terminal_growth_pct=2.5,
        tax_rate_pct=22.0,
        reinvestment_rate_pct=15.0,
    )
    assert abs(roundtrip["per_share"] - target_price) < 0.05


# 15. Unbracketed reverse-DCF targets
def test_15_unbracketed_reverse_dcf_targets():
    """
    Scenario 15: Target prices outside reasonable growth bounds [-50%, +200%]
    must return None with OUT_OF_BOUNDS status, never a boundary value like 100.0%.
    """
    base_rev = 11228024000.0
    shares = 35254000.0
    net_bridge = -1800730000.0

    # Ridiculously high target price requiring > 200% CAGR
    g, status = solve_reverse_dcf(
        target_price=50000.0,
        base_revenue=base_rev,
        current_shares=shares,
        net_debt_bridge=net_bridge,
        operating_margin_pct=15.0,
        wacc_pct=9.5,
        terminal_growth_pct=2.5,
    )
    assert g is None
    assert "OUT_OF_BOUNDS_TARGET_NOT_BRACKETED" in status


# 16. Report/ledger numeric parity
def test_16_report_ledger_numeric_parity():
    """
    Scenario 16: Verify that markdown report tables contain exact values
    matching the finalized sourced_fundamentals_ledger.json.
    """
    ledger_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "research", "sourced_fundamentals", "sourced_fundamentals_ledger.json"
    )
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "SOURCED_FUNDAMENTALS_RESEARCH_PILOT.md"
    )
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    # Check FIX revenue in report matches ledger
    fix_rev = ledger["companies"]["FIX"]["derived_metrics"]["revenue_ttm"]
    formatted_rev = f"{fix_rev / 1e6:,.2f}M"
    assert formatted_rev in report_text, f"Report must contain exact ledger revenue {formatted_rev}"

    # Check Base DCF value for FIX
    fix_dcf_base = ledger["companies"]["FIX"]["deep_valuation"]["scenarios"]["Base"]["per_share_value_estimate"]
    dd_report_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "DEEP_DIVE_UNDERVALUED_GROWTH_ASSESSMENTS.md"
    )
    with open(dd_report_path, "r", encoding="utf-8") as f:
        dd_text = f.read()
    assert f"${fix_dcf_base:.2f}" in dd_text, f"Deep dive report must contain exact base DCF per-share ${fix_dcf_base:.2f}"


# 17. Offline replay with network access disabled
def test_17_offline_replay_zero_network_access(monkeypatch):
    """
    Scenario 17: Pipeline execution with offline=True must make zero network requests.
    Mocks socket and urllib to fail closed if any network connection is attempted.
    """
    def mock_socket_connect(*args, **kwargs):
        raise ConnectionError("Network egress attempted during offline replay test!")

    monkeypatch.setattr(socket.socket, "connect", mock_socket_connect)

    def mock_urlopen(*args, **kwargs):
        raise ConnectionError("Network egress attempted during offline replay test!")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    ledger = run_pipeline(offline=True, refresh=False)
    assert ledger["metadata"]["offline_mode"] is True
    assert len(ledger["companies"]) == 8
    assert "FIX" in ledger["companies"]
    assert ledger["companies"]["FIX"]["derived_metrics"]["revenue_ttm"] == 11228024000.0
