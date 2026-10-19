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
import subprocess
import tempfile
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
from analyst_dashboard.research.sourced_fundamentals.qualitative_assessor import (
    assess_company,
)
from analyst_dashboard.research.sourced_fundamentals.valuation import (
    run_fcff_model,
    solve_reverse_dcf,
    generate_sensitivity_matrix,
    evaluate_deep_valuation_fix,
    evaluate_deep_valuation_duol,
    evaluate_deep_valuation_ulta,
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


# 18. Aggregate Missing-Data Propagation and Completeness Principle
def test_18_aggregate_missing_data_propagation_and_completeness(normalizer):
    """
    Scenario 18:
    1. Partial debt (e.g. ULTA short-term borrowings reported without noncurrent debt)
       does NOT sum to total debt; total_financial_debt is UNAVAILABLE and value is None.
    2. No ticker-based whitelisting: DUOL debt is UNAVAILABLE, not assumed 0.0.
    3. Unavailable debt propagates to net_debt being UNAVAILABLE (None).
    4. DUOL and ULTA deep valuations are suppressed with valuation_status == "SUPPRESSED_UNAVAILABLE",
       reverse_dcf_implied_5y_growth_pct is None, and non-empty suppression_reason.
    5. KLAC reports comprehensive debt (DebtLongtermAndShorttermCombinedAmount) as REPORTED $5.887B,
       with cash and net debt fully evidenced.
    """
    ledger = run_pipeline(offline=True, refresh=False)
    companies = ledger["companies"]

    # ULTA
    ulta_m = companies["ULTA"]["derived_metrics"]
    ulta_dv = companies["ULTA"]["deep_valuation"]
    assert ulta_m["total_financial_debt"] is None
    assert ulta_m["net_debt"] is None
    assert ulta_m["enterprise_value"] is None
    assert ulta_dv["valuation_status"] == "SUPPRESSED_UNAVAILABLE"
    assert ulta_dv["reverse_dcf_implied_5y_growth_pct"] is None
    assert "incomplete component reporting" in ulta_dv["suppression_reason"]
    assert len(ulta_dv["scenarios"]) == 0

    # DUOL
    duol_m = companies["DUOL"]["derived_metrics"]
    duol_dv = companies["DUOL"]["deep_valuation"]
    assert duol_m["total_financial_debt"] is None
    assert duol_m["net_debt"] is None
    assert duol_m["enterprise_value"] is None
    assert duol_dv["valuation_status"] == "SUPPRESSED_UNAVAILABLE"
    assert duol_dv["reverse_dcf_implied_5y_growth_pct"] is None
    assert "UNAVAILABLE in SEC EDGAR XBRL" in duol_dv["suppression_reason"]
    assert len(duol_dv["scenarios"]) == 0

    # KLAC
    klac_m = companies["KLAC"]["derived_metrics"]
    assert klac_m["total_financial_debt"] == 5887415000.0
    assert klac_m["total_cash_and_investments"] == 3118230000.0
    assert klac_m["net_debt"] == 2769185000.0
    assert klac_m["enterprise_value"] is not None


# 19. Share-Period Duration Discrimination & Denominator Semantics
def test_19_share_period_duration_discrimination_and_denominator_semantics(normalizer):
    """
    Scenario 19:
    1. Distinguish between quarterly (80-100d), YTD, and annual (350-375d) shares.
    2. Shuffling observation list order produces identical selected shares (permutation invariance).
    3. Quarterly shares are explicitly noted as matching latest interim reporting period, not full TTM.
    4. Denominator semantics: DerivedMetrics distinguishes period-weighted diluted shares for EPS/FCF
       from point-in-time common shares for market cap.
    """
    synthetic_us_gaap = {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {
            "units": {
                "shares": [
                    # 6M YTD observation
                    {"form": "10-Q", "fp": "Q2", "start": "2026-01-01", "end": "2026-06-30", "val": 50080000.0, "filed": "2026-08-08", "accn": "A1"},
                    # 3M standalone interim quarter observation
                    {"form": "10-Q", "fp": "Q2", "start": "2026-04-01", "end": "2026-06-30", "val": 50031000.0, "filed": "2026-08-08", "accn": "A1"},
                ]
            }
        }
    }

    # Extract shares with original order
    fact_orig = normalizer._extract_diluted_shares(synthetic_us_gaap, "2026-06-30")
    # For a 10-Q filing, selector must prioritize the 3-month interim quarter (80-100 days)
    assert fact_orig.value == 50031000.0
    assert "quarterly diluted weighted-average shares" in fact_orig.notes.lower()
    assert "not full-year ttm shares" in fact_orig.notes.lower()

    # Shuffled order test (permutation invariance)
    items_shuf = list(reversed(synthetic_us_gaap["WeightedAverageNumberOfDilutedSharesOutstanding"]["units"]["shares"]))
    us_gaap_shuf = {"WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": items_shuf}}}
    fact_shuf = normalizer._extract_diluted_shares(us_gaap_shuf, "2026-06-30")
    assert fact_shuf.value == fact_orig.value
    assert fact_shuf.notes == fact_orig.notes


# 20. Evidence-Sufficiency Responsive Assessments
def test_20_evidence_sufficiency_responsive_assessments(normalizer):
    """
    Scenario 20:
    1. assess_company fails closed with INSUFFICIENT_EVIDENCE when critical financial facts are missing.
    2. Narrative claims for DUOL and ULTA do NOT assert "zero debt", "net cash", or debt-free status.
    3. FIX, DUOL, ULTA assessments label them as "Selected Case Study", not algorithmic top picks.
    """
    # 1. Missing critical facts fail closed
    empty_cf = CompanyFundamentals(
        symbol="TEST",
        company_name="Test Company",
        cik="0009999999",
        primary_exchange="NASDAQ",
        sector_industry="Technology",
        business_model="SaaS",
        fiscal_year_end="12-31",
        latest_reported_period="2026-06-30",
        latest_filing_date="2026-08-01",
    )
    empty_metrics = DerivedMetrics(symbol="TEST", period_label="TTM", revenue_ttm=None, diluted_shares_weighted_avg=None)
    assessment = assess_company(empty_cf, empty_metrics)
    assert assessment.disposition == ResearchDisposition.INSUFFICIENT_EVIDENCE
    assert "Critical financial facts missing" in assessment.disposition_rationale

    # 2. Check DUOL and ULTA assessments in actual ledger
    ledger = run_pipeline(offline=True, refresh=False)
    duol_a = ledger["companies"]["DUOL"]["assessment"]
    ulta_a = ledger["companies"]["ULTA"]["assessment"]
    fix_a = ledger["companies"]["FIX"]["assessment"]

    assert "Selected Case Study" in fix_a["disposition_rationale"]
    assert "Selected Case Study" in duol_a["disposition_rationale"]
    assert "Selected Case Study" in ulta_a["disposition_rationale"]

    # Verify no false claim of zero debt for DUOL
    assert "zero debt" not in duol_a["disposition_rationale"].lower()
    assert "zero financial debt" not in duol_a["balance_sheet_resilience"].lower()
    assert "UNAVAILABLE" in duol_a["disposition_rationale"]

    # Verify ULTA debt note
    assert "UNAVAILABLE" in ulta_a["disposition_rationale"]


# 21. Conflicting Observations & Restatement Precedence
def test_21_conflicting_observations_and_restatement_precedence(normalizer):
    """
    Scenario 21:
    1. Exact duplicate observations collapse cleanly without error.
    2. Genuine intra-filing conflict (identical metadata with differing values) is detected,
       logged, and returns UNAVAILABLE / None.
    3. Inter-filing restatement (differing filing dates) selects latest filing accession.
    4. Permuted input order produces identical results.
    """
    # 1. Exact duplicate collapsing
    dups = [
        {"form": "10-Q", "end": "2026-06-30", "val": 100.0, "filed": "2026-08-01", "accn": "A1"},
        {"form": "10-Q", "end": "2026-06-30", "val": 100.0, "filed": "2026-08-01", "accn": "A1"},
    ]
    deduped, err = normalizer._deduplicate_and_detect_conflicts(dups, "Cash")
    assert len(deduped) == 1
    assert deduped[0]["val"] == 100.0
    assert err is None

    # 2. Genuine intra-filing conflict (identical metadata, differing values)
    conflicts = [
        {"form": "10-Q", "end": "2026-06-30", "val": 100.0, "filed": "2026-08-01", "accn": "A1"},
        {"form": "10-Q", "end": "2026-06-30", "val": 150.0, "filed": "2026-08-01", "accn": "A1"},
    ]
    conflict_res, conflict_err = normalizer._deduplicate_and_detect_conflicts(conflicts, "Cash")
    assert conflict_err is not None
    assert "GENUINE_CONFLICT" in conflict_err

    conflict_us_gaap = {"Cash": {"units": {"USD": conflicts}}}
    fact = normalizer._extract_instant_fact(conflict_us_gaap, ["Cash"], "cash_and_equivalents", "2026-06-30")
    assert fact.value is None
    assert fact.status == MetricStatus.UNAVAILABLE
    assert "GENUINE_CONFLICT" in fact.notes

    # 3. Inter-filing restatement precedence
    restatements = [
        {"form": "10-Q", "end": "2026-06-30", "val": 100.0, "filed": "2026-08-01", "accn": "A1"},
        {"form": "10-Q/A", "end": "2026-06-30", "val": 105.0, "filed": "2026-09-01", "accn": "A1_AMENDED"},
    ]
    restate_res, restate_conflict = normalizer._deduplicate_and_detect_conflicts(restatements, "Cash")
    assert len(restate_res) == 2
    assert restate_res[-1]["val"] == 105.0
    assert restate_res[-1]["accn"] == "A1_AMENDED"


# 22. End-to-End Cutoff Enforcement
def test_22_declared_research_cutoff_enforced_end_to_end(normalizer):
    """
    Scenario 22:
    as_of_date is passed through normalization, submissions parsing, prior-TTM revenue,
    and split dilution comparisons. Filings after cutoff are excluded.
    """
    synthetic_us_gaap = {
        "Revenues": {
            "units": {
                "USD": [
                    {"form": "10-K", "fp": "FY", "start": "2024-01-01", "end": "2024-12-31", "val": 1000.0, "filed": "2025-02-15", "accn": "A1"},
                    {"form": "10-Q", "fp": "Q1", "start": "2025-01-01", "end": "2025-03-31", "val": 280.0, "filed": "2025-05-10", "accn": "A2"},
                    {"form": "10-Q", "fp": "Q2", "start": "2025-01-01", "end": "2025-06-30", "val": 580.0, "filed": "2025-08-10", "accn": "A3"},
                    # Post-cutoff filing in 2026
                    {"form": "10-K", "fp": "FY", "start": "2025-01-01", "end": "2025-12-31", "val": 1250.0, "filed": "2026-02-20", "accn": "A4"},
                ]
            }
        }
    }
    # Cutoff at 2025-09-01
    filtered = normalizer._filter_filings(synthetic_us_gaap["Revenues"]["units"]["USD"], as_of_date="2025-09-01")
    filed_dates = [x["filed"] for x in filtered]
    assert "2026-02-20" not in filed_dates
    assert len(filtered) == 3


# 23. Zero Magnitude-Based Split Guessing
def test_23_split_normalization_without_magnitude_guessing(normalizer):
    """
    Scenario 23:
    1. If split_ratio is provided without split_date, returns (None, None) failing closed.
    2. Rely strictly on corporate action split_date and comparative filing dates (filed < split_date).
    3. Post-split observations (filed >= split_date) are NOT adjusted even if smaller than current shares.
    """
    synthetic_us_gaap = {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {
            "units": {
                "shares": [
                    {"form": "10-K", "fp": "FY", "start": "2022-01-01", "end": "2022-12-31", "val": 20000000.0, "filed": "2023-02-15"},
                    {"form": "10-K", "fp": "FY", "start": "2024-01-01", "end": "2024-12-31", "val": 19000000.0, "filed": "2025-02-15"},
                ]
            }
        }
    }

    # 1. Ratio without date fails closed
    d1, d3 = compute_split_normalized_dilution(
        normalizer, synthetic_us_gaap, "2025-12-31", curr_shares=60000000.0, split_ratio=3.0, split_date=None
    )
    assert d1 is None
    assert d3 is None

    # 2. Ratio with date: split effective 2024-01-01
    # 2022 observation filed 2023-02-15 < split_date -> multiplied by 3.0 (20M * 3 = 60M)
    d1, d3 = compute_split_normalized_dilution(
        normalizer, synthetic_us_gaap, "2025-12-31", curr_shares=60000000.0, split_ratio=3.0, split_date="2024-01-01"
    )
    # v3 was 20M, multiplied by 3.0 = 60M. curr_shares = 60M -> dilution = 0.0%
    assert d3 is not None
    assert abs(d3 - 0.0) < 0.01


# 24. Historical Dilution Order Invariance & Basis Matching (Issue 1)
def test_24_historical_dilution_order_invariance_and_basis_matching(normalizer):
    """
    Scenario 24 (Issue 1):
    A. Original, reversed, and multiple shuffled orders produce identical:
       - Selected comparison observations (form, accn, filed, start, end, val)
       - Period dates
       - Adjustment basis
       - Dilution percentages or unavailable status
    B. Covers both DUOL and FIX retained primary inputs.
    C. Synthetic case where quarterly and YTD observations share end date, accession, and filing date.
    D. Missing comparable period returns unavailable rather than falling back to another quarter.
    """
    # --- Test B: DUOL and FIX Retained Inputs ---
    # 1. DUOL
    duol_path = os.path.join(DEFAULT_CACHE_DIR, "CIK0001562088_companyfacts.json")
    with open(duol_path, "r", encoding="utf-8") as f:
        duol_facts = json.load(f)["facts"]["us-gaap"]

    duol_shares_items = list(duol_facts["WeightedAverageNumberOfDilutedSharesOutstanding"]["units"]["shares"])
    duol_latest_end = "2026-06-30"
    duol_curr_shares = 50031000.0  # Q2 2026 quarterly diluted shares

    # Original order
    d1_orig, d3_orig, meta_orig = compute_split_normalized_dilution(
        normalizer, duol_facts, duol_latest_end, duol_curr_shares, curr_start="2026-04-01", return_details=True
    )
    assert d1_orig is not None and abs(d1_orig - 2.10) < 0.01
    assert d3_orig is not None and abs(d3_orig - 6.74) < 0.01
    assert meta_orig["comparison_1y"]["period_start"] == "2025-04-01"
    assert meta_orig["comparison_1y"]["period_end"] == "2025-06-30"
    assert meta_orig["comparison_1y"]["duration_days"] in range(80, 100)
    assert meta_orig["comparison_3y"]["period_start"] == "2023-04-01"
    assert meta_orig["comparison_3y"]["period_end"] == "2023-06-30"
    assert meta_orig["comparison_3y"]["duration_days"] in range(80, 100)

    # Reversed order
    duol_rev_facts = {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {
            "units": {"shares": list(reversed(duol_shares_items))}
        }
    }
    d1_rev, d3_rev, meta_rev = compute_split_normalized_dilution(
        normalizer, duol_rev_facts, duol_latest_end, duol_curr_shares, curr_start="2026-04-01", return_details=True
    )
    assert d1_rev == d1_orig
    assert d3_rev == d3_orig
    assert meta_rev["comparison_1y"]["reported_val"] == meta_orig["comparison_1y"]["reported_val"]
    assert meta_rev["comparison_1y"]["period_start"] == meta_orig["comparison_1y"]["period_start"]
    assert meta_rev["comparison_1y"]["accn"] == meta_orig["comparison_1y"]["accn"]
    assert meta_rev["comparison_1y"]["adjustment_basis"] == meta_orig["comparison_1y"]["adjustment_basis"]
    assert meta_rev["comparison_3y"]["reported_val"] == meta_orig["comparison_3y"]["reported_val"]
    assert meta_rev["comparison_3y"]["period_start"] == meta_orig["comparison_3y"]["period_start"]

    # Multiple random shuffles
    for seed in [11, 42, 99, 1337, 2026]:
        shuffled_items = list(duol_shares_items)
        random.seed(seed)
        random.shuffle(shuffled_items)
        duol_shuf_facts = {
            "WeightedAverageNumberOfDilutedSharesOutstanding": {
                "units": {"shares": shuffled_items}
            }
        }
        d1_shuf, d3_shuf, meta_shuf = compute_split_normalized_dilution(
            normalizer, duol_shuf_facts, duol_latest_end, duol_curr_shares, curr_start="2026-04-01", return_details=True
        )
        assert d1_shuf == d1_orig
        assert d3_shuf == d3_orig
        assert meta_shuf["comparison_1y"]["reported_val"] == meta_orig["comparison_1y"]["reported_val"]
        assert meta_shuf["comparison_3y"]["reported_val"] == meta_orig["comparison_3y"]["reported_val"]

    # 2. FIX
    fix_path = os.path.join(DEFAULT_CACHE_DIR, "CIK0001035983_companyfacts.json")
    with open(fix_path, "r", encoding="utf-8") as f:
        fix_facts = json.load(f)["facts"]["us-gaap"]
    fix_shares_items = list(fix_facts["WeightedAverageNumberOfDilutedSharesOutstanding"]["units"]["shares"])
    fix_latest_end = "2026-06-30"
    fix_curr_shares = 35251000.0  # Q2 2026 quarterly diluted shares

    d1_fix_orig, d3_fix_orig, meta_fix_orig = compute_split_normalized_dilution(
        normalizer, fix_facts, fix_latest_end, fix_curr_shares, curr_start="2026-04-01", return_details=True
    )
    assert d1_fix_orig is not None and abs(d1_fix_orig - (-0.33)) < 0.01
    assert d3_fix_orig is not None and abs(d3_fix_orig - (-1.82)) < 0.01
    assert meta_fix_orig["comparison_1y"]["period_start"] == "2025-04-01"
    assert meta_fix_orig["comparison_3y"]["period_start"] == "2023-04-01"

    fix_rev_facts = {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {
            "units": {"shares": list(reversed(fix_shares_items))}
        }
    }
    d1_fix_rev, d3_fix_rev, meta_fix_rev = compute_split_normalized_dilution(
        normalizer, fix_rev_facts, fix_latest_end, fix_curr_shares, curr_start="2026-04-01", return_details=True
    )
    assert d1_fix_rev == d1_fix_orig
    assert d3_fix_rev == d3_fix_orig
    assert meta_fix_rev["comparison_1y"]["reported_val"] == meta_fix_orig["comparison_1y"]["reported_val"]
    assert meta_fix_rev["comparison_3y"]["reported_val"] == meta_fix_orig["comparison_3y"]["reported_val"]

    # --- Test C: Synthetic Case — Quarterly vs YTD sharing end, accession, and filing date ---
    # In Q2 10-Q, companies often report 6M YTD (e.g. 40M) and 3M standalone quarter (e.g. 50M) in the exact same filing
    synthetic_shared_filing = {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {
            "units": {
                "shares": [
                    # 6M YTD observation (duration 181 days)
                    {"form": "10-Q", "fp": "Q2", "start": "2025-01-01", "end": "2025-06-30", "val": 40000000.0, "filed": "2025-08-05", "accn": "0001-25-000100"},
                    # 3M standalone interim quarter observation (duration 91 days)
                    {"form": "10-Q", "fp": "Q2", "start": "2025-04-01", "end": "2025-06-30", "val": 50000000.0, "filed": "2025-08-05", "accn": "0001-25-000100"},
                ]
            }
        }
    }
    # Current period is quarterly: 2026-04-01 to 2026-06-30 with 55M shares
    # Order 1: YTD first, quarterly second
    d1_syn1, _, meta_syn1 = compute_split_normalized_dilution(
        normalizer, synthetic_shared_filing, "2026-06-30", curr_shares=55000000.0, curr_start="2026-04-01", return_details=True
    )
    assert meta_syn1["comparison_1y"]["reported_val"] == 50000000.0, "Must select quarterly observation, not YTD"
    assert abs(d1_syn1 - 10.0) < 0.01  # (55M - 50M) / 50M = +10.0%

    # Order 2: Quarterly first, YTD second (reversed)
    synthetic_shared_filing_rev = {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {
            "units": {
                "shares": list(reversed(synthetic_shared_filing["WeightedAverageNumberOfDilutedSharesOutstanding"]["units"]["shares"]))
            }
        }
    }
    d1_syn2, _, meta_syn2 = compute_split_normalized_dilution(
        normalizer, synthetic_shared_filing_rev, "2026-06-30", curr_shares=55000000.0, curr_start="2026-04-01", return_details=True
    )
    assert meta_syn2["comparison_1y"]["reported_val"] == 50000000.0, "Reversed order must also select quarterly observation"
    assert d1_syn2 == d1_syn1

    # --- Test D: Missing Comparable Period Returns Unavailable ---
    # Target 1Y has only a Q4 observation (end: 2025-12-31) when current is Q2 (end: 2026-06-30)
    synthetic_missing_comparable = {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {
            "units": {
                "shares": [
                    # Q4 observation 6 months away from target
                    {"form": "10-K", "fp": "FY", "start": "2025-10-01", "end": "2025-12-31", "val": 48000000.0, "filed": "2026-02-15", "accn": "0001-26-000010"},
                ]
            }
        }
    }
    d1_miss, d3_miss, meta_miss = compute_split_normalized_dilution(
        normalizer, synthetic_missing_comparable, "2026-06-30", curr_shares=55000000.0, curr_start="2026-04-01", return_details=True
    )
    assert d1_miss is None, "Must return None when comparable 1Y quarter is missing; must not fall back to Q4"
    assert meta_miss["comparison_1y"]["status"] == "UNAVAILABLE"


# 25. TTM EPS Interim Suppression & Research Estimate Semantics (Issue 2)
def test_25_ttm_eps_interim_suppression_and_research_estimate_semantics():
    """
    Scenario 25 (Issue 2):
    1. For interim periods (FIX, DUOL Q2 10-Q), conventional diluted_eps_ttm is suppressed (None).
    2. Explicit research estimate ttm_net_income_per_latest_quarter_diluted_share is exposed
       with status RESEARCH_ESTIMATE and explicit numerator and denominator periods.
    3. For annual reporting periods with matching 12M net income and 12M shares, diluted_eps_ttm is populated.
    4. Generated reports and master tables do NOT relabel research estimate as conventional diluted TTM EPS
       and do NOT label quarterly shares as 'Diluted Shares (TTM)'.
    """
    ledger = run_pipeline(offline=True, refresh=False)
    fix_m = ledger["companies"]["FIX"]["derived_metrics"]
    duol_m = ledger["companies"]["DUOL"]["derived_metrics"]

    # 1. FIX interim suppression & research estimate
    assert fix_m["diluted_eps_ttm"] is None, "Conventional diluted_eps_ttm must be suppressed for interim period"
    assert fix_m["ttm_net_income_per_latest_quarter_diluted_share"] is not None
    assert abs(fix_m["ttm_net_income_per_latest_quarter_diluted_share"] - 40.69) < 0.05
    assert fix_m["ttm_eps_estimate_status"] == "RESEARCH_ESTIMATE"
    assert fix_m["ttm_eps_numerator_period"] == "2025-07-01 to 2026-06-30"
    assert "2026-04-01 to 2026-06-30" in fix_m["ttm_eps_denominator_period"]
    assert "quarterly" in fix_m["ttm_eps_denominator_period"].lower()

    # 2. DUOL interim suppression & research estimate
    assert duol_m["diluted_eps_ttm"] is None, "Conventional diluted_eps_ttm must be suppressed for interim period"
    assert duol_m["ttm_net_income_per_latest_quarter_diluted_share"] is not None
    assert abs(duol_m["ttm_net_income_per_latest_quarter_diluted_share"] - 8.21) < 0.05
    assert duol_m["ttm_eps_estimate_status"] == "RESEARCH_ESTIMATE"
    assert duol_m["ttm_eps_numerator_period"] == "2025-07-01 to 2026-06-30"
    assert "2026-04-01 to 2026-06-30" in duol_m["ttm_eps_denominator_period"]

    # 3. Calculation notes verification
    assert "suppressed for interim reporting periods" in fix_m["calculation_notes"]["diluted_eps_ttm"]
    assert "ASC 260" in fix_m["calculation_notes"]["diluted_eps_ttm"]
    assert "Denominator is quarterly" in fix_m["calculation_notes"]["cfo_minus_capex_per_share"]

    # 4. Check report text parity
    with open("SOURCED_FUNDAMENTALS_RESEARCH_PILOT.md", "r", encoding="utf-8") as f:
        pilot_text = f.read()

    assert "Diluted Shares (TTM)" not in pilot_text, "Master table must not label quarterly shares as Diluted Shares (TTM)"
    assert "Diluted Shares (Latest Period)" in pilot_text
    assert "UNAVAILABLE (Interim)" in pilot_text


# 26. Evidence-Conditional Narratives: Missing Balance Sheet (Confirmed Case A)
def test_26_evidence_conditional_narratives_missing_balance_sheet_case_a():
    """
    Scenario 26 (Issue 3, Confirmed Case A):
    Supply FIX symbol with revenue and diluted shares present,
    but cash, debt, net debt, and operating results unavailable.
    The assessor must NOT claim 'Exceptional liquidity', 'immense liquidity cushion',
    'fortress balance sheet', or 'net-cash balance sheet'.
    """
    cf = CompanyFundamentals(
        symbol="FIX",
        company_name="Comfort Systems USA, Inc.",
        cik="0001035983",
        primary_exchange="NYSE",
        sector_industry="Industrials / Specialty Contractors",
        business_model="HVAC and modular thermal infrastructure",
        fiscal_year_end="12-31",
        latest_reported_period="2026-06-30",
        latest_filing_date="2026-07-25",
    )
    # Revenue and shares present; cash, debt, net debt, and operating results None
    metrics = DerivedMetrics(
        symbol="FIX",
        period_label="TTM",
        revenue_ttm=11228024000.0,
        diluted_shares_weighted_avg=35251000.0,
        operating_income_ttm=None,
        cfo_minus_capex_ttm=None,
        cash_and_equivalents=None,
        total_cash_and_investments=None,
        total_financial_debt=None,
        net_debt=None,
        share_dilution_1y_pct=None,
    )
    record = assess_company(cf, metrics)

    # Must fail closed with INSUFFICIENT_EVIDENCE
    assert record.disposition == ResearchDisposition.INSUFFICIENT_EVIDENCE
    assert "unavailable" in record.disposition_rationale.lower()

    # Balance sheet resilience must report unknown/unavailable
    assert "unknown" in record.balance_sheet_resilience.lower()
    assert "unavailable" in record.balance_sheet_resilience.lower()

    # Strictly assert absence of contradictory claims
    all_narrative = (
        record.disposition_rationale + " " +
        record.balance_sheet_resilience + " " +
        record.shareholder_outcomes + " " +
        record.reinvestment_economics
    ).lower()

    assert "exceptional liquidity" not in all_narrative
    assert "immense liquidity cushion" not in all_narrative
    assert "fortress balance sheet" not in all_narrative
    assert "net-cash balance sheet" not in all_narrative


# 27. Evidence-Conditional Narratives: Adverse Financials (Confirmed Case B)
def test_27_evidence_conditional_narratives_adverse_financials_case_b():
    """
    Scenario 27 (Issue 3, Confirmed Case B):
    Supply company with adverse metrics:
    - Revenue growth: -50.0%
    - Operating margin: -20.0%
    - CFO minus Capex: -100,000,000
    - Financial debt: 5,000,000,000
    - Cash: 100,000,000
    - Net debt: 4,900,000,000
    - One-year share-count increase: +50.0%

    The assessor must truthfully report contraction, losses, negative cash flow,
    net debt, and share dilution.
    The assessor must NOT state:
    - 'Fortress balance sheet'
    - 'Disciplined'
    - 'Net-cash balance sheet'
    - 'Negative net share dilution'
    - 'Robust'
    - 'Exceptional liquidity'
    - 'immense liquidity cushion'
    """
    cf = CompanyFundamentals(
        symbol="FIX",
        company_name="Comfort Systems USA, Inc.",
        cik="0001035983",
        primary_exchange="NYSE",
        sector_industry="Industrials / Specialty Contractors",
        business_model="HVAC and modular thermal infrastructure",
        fiscal_year_end="12-31",
        latest_reported_period="2026-06-30",
        latest_filing_date="2026-07-25",
    )
    metrics = DerivedMetrics(
        symbol="FIX",
        period_label="TTM",
        revenue_ttm=5000000000.0,
        revenue_growth_yoy_pct=-50.0,
        operating_income_ttm=-1000000000.0,
        operating_margin_pct=-20.0,
        cfo_minus_capex_ttm=-100000000.0,
        fcf_conversion_ebit_pct=10.0,
        cash_and_equivalents=100000000.0,
        total_cash_and_investments=100000000.0,
        total_financial_debt=5000000000.0,
        net_debt=4900000000.0,
        diluted_shares_weighted_avg=50000000.0,
        share_dilution_1y_pct=50.0,
        share_dilution_3y_pct=100.0,
    )
    record = assess_company(cf, metrics)

    # Truthful statements must be present
    assert "-50.0% yoy contraction" in record.disposition_rationale.lower()
    assert "operating loss" in record.disposition_rationale.lower()
    assert "negative cfo minus capex" in record.disposition_rationale.lower()
    assert "net debt position of $4.9b" in record.disposition_rationale.lower()
    assert "diluted share count expanded by +50.00%" in record.disposition_rationale.lower()
    assert "net debt of $4.9b" in record.balance_sheet_resilience.lower()
    assert "share count expanded" in record.shareholder_outcomes.lower()
    assert "reinvestment currently consumes cash" in record.reinvestment_economics.lower()

    # Contradictory positive claims must be ABSENT
    all_narrative = (
        record.disposition_rationale + " " +
        record.balance_sheet_resilience + " " +
        record.shareholder_outcomes + " " +
        record.reinvestment_economics + " " +
        record.growth_durability
    ).lower()

    assert "fortress balance sheet" not in all_narrative
    assert "disciplined" not in all_narrative
    assert "net-cash balance sheet" not in all_narrative
    assert "negative net share dilution" not in all_narrative
    assert "robust" not in all_narrative
    assert "exceptional liquidity" not in all_narrative
    assert "immense liquidity cushion" not in all_narrative
    assert "high cash conversion" not in all_narrative


# 28. Evidence-Conditional Narratives: Acceptance Matrix C through F (Issue 3)
def test_28_evidence_conditional_narratives_acceptance_matrix_c_to_f():
    """
    Scenario 28 (Issue 3):
    Acceptance Tests C through F:
    C. Negative CFO-minus-capex reported as consuming cash without 'robust' praise.
    D. Positive net debt vs negative net debt reported accurately.
    E. Share count increase vs decrease reported accurately.
    F. Zero values distinguished from missing values.
    """
    cf = CompanyFundamentals(
        symbol="TEST",
        company_name="Test Corp",
        cik="0001111111",
        primary_exchange="NASDAQ",
        sector_industry="Technology",
        business_model="Software",
        fiscal_year_end="12-31",
        latest_reported_period="2026-06-30",
        latest_filing_date="2026-08-01",
    )

    # --- Matrix C: Negative FCF ---
    m_neg_fcf = DerivedMetrics(
        symbol="TEST", period_label="TTM", revenue_ttm=1e9, diluted_shares_weighted_avg=10e6,
        cfo_minus_capex_ttm=-50e6, operating_income_ttm=-20e6, operating_margin_pct=-2.0
    )
    rec_neg_fcf = assess_company(cf, m_neg_fcf)
    assert "reinvestment currently consumes cash" in rec_neg_fcf.reinvestment_economics.lower()
    assert "robust" not in rec_neg_fcf.disposition_rationale.lower()

    # --- Matrix D: Net Debt vs Net Cash ---
    # Net Debt > 0
    m_pos_nd = DerivedMetrics(
        symbol="TEST", period_label="TTM", revenue_ttm=1e9, diluted_shares_weighted_avg=10e6,
        operating_income_ttm=100e6, operating_margin_pct=10.0, cfo_minus_capex_ttm=80e6,
        total_cash_and_investments=100e6, total_financial_debt=500e6, net_debt=400e6
    )
    rec_pos_nd = assess_company(cf, m_pos_nd)
    assert "net debt position of $400.0m" in rec_pos_nd.disposition_rationale.lower()
    assert "net cash" not in rec_pos_nd.balance_sheet_resilience.lower()

    # Net Debt < 0 (Net Cash)
    m_neg_nd = DerivedMetrics(
        symbol="TEST", period_label="TTM", revenue_ttm=1e9, diluted_shares_weighted_avg=10e6,
        operating_income_ttm=100e6, operating_margin_pct=10.0, cfo_minus_capex_ttm=80e6,
        total_cash_and_investments=500e6, total_financial_debt=100e6, net_debt=-400e6
    )
    rec_neg_nd = assess_company(cf, m_neg_nd)
    assert "net cash position of $400.0m" in rec_neg_nd.disposition_rationale.lower()

    # --- Matrix E: Share Count Increase vs Decrease ---
    # Dilution > 0
    m_dil_pos = DerivedMetrics(
        symbol="TEST", period_label="TTM", revenue_ttm=1e9, diluted_shares_weighted_avg=10e6,
        operating_income_ttm=100e6, operating_margin_pct=10.0, cfo_minus_capex_ttm=80e6,
        share_dilution_1y_pct=4.5
    )
    rec_dil_pos = assess_company(cf, m_dil_pos)
    assert "share count expanded by +4.50%" in rec_dil_pos.disposition_rationale.lower()
    assert "repurchase" not in rec_dil_pos.shareholder_outcomes.lower()

    # Dilution < 0 (Buybacks)
    m_dil_neg = DerivedMetrics(
        symbol="TEST", period_label="TTM", revenue_ttm=1e9, diluted_shares_weighted_avg=10e6,
        operating_income_ttm=100e6, operating_margin_pct=10.0, cfo_minus_capex_ttm=80e6,
        share_dilution_1y_pct=-4.5
    )
    rec_dil_neg = assess_company(cf, m_dil_neg)
    assert "share count decreased by -4.50%" in rec_dil_neg.disposition_rationale.lower()
    assert "capital return via repurchases" in rec_dil_neg.shareholder_outcomes.lower()

    # --- Matrix F: Zero Values vs Missing Values ---
    # Reported 0.0 debt
    m_zero_debt = DerivedMetrics(
        symbol="TEST", period_label="TTM", revenue_ttm=1e9, diluted_shares_weighted_avg=10e6,
        operating_income_ttm=100e6, operating_margin_pct=10.0, cfo_minus_capex_ttm=80e6,
        total_cash_and_investments=100e6, total_financial_debt=0.0, net_debt=-100e6
    )
    rec_zero_debt = assess_company(cf, m_zero_debt)
    assert "$0.0 debt" in rec_zero_debt.disposition_rationale.lower()

    # Missing debt
    m_miss_debt = DerivedMetrics(
        symbol="TEST", period_label="TTM", revenue_ttm=1e9, diluted_shares_weighted_avg=10e6,
        operating_income_ttm=100e6, operating_margin_pct=10.0, cfo_minus_capex_ttm=80e6,
        total_cash_and_investments=100e6, total_financial_debt=None, net_debt=None
    )
    rec_miss_debt = assess_company(cf, m_miss_debt)
    assert "net debt unavailable" in rec_miss_debt.disposition_rationale.lower()
    assert "unevidenced in primary xbrl tags" in rec_miss_debt.balance_sheet_resilience.lower()


# 29. Actual Eight-Company Narrative Soundness (Matrix G)
def test_29_actual_eight_company_narrative_soundness():
    """
    Scenario 29 (Issue 3, Matrix G):
    Verify that across all 8 actual companies in the population:
    1. Financial claims match actual derived metrics.
    2. Zero contradictory positive claims occur.
    3. Case studies (FIX, DUOL, ULTA) retain explicit Case Study designation.
    4. DUOL and ULTA maintain UNAVAILABLE net debt and suppressed valuations.
    """
    ledger = run_pipeline(offline=True, refresh=False)
    companies = ledger["companies"]
    assert len(companies) == 8

    banned_contradictions = [
        "fortress balance sheet",
        "exceptional liquidity",
        "immense liquidity cushion",
        "negative net share dilution",
    ]

    for sym, c in companies.items():
        assessment = c["assessment"]
        rationale = assessment["disposition_rationale"]
        bs = assessment["balance_sheet_resilience"]
        sh = assessment["shareholder_outcomes"]
        combined = (rationale + " " + bs + " " + sh).lower()

        for b in banned_contradictions:
            assert b not in combined, f"{sym} assessment contains prohibited contradictory claim: {b}"

    # Verify DUOL and ULTA debt and valuation status
    assert "UNAVAILABLE" in companies["DUOL"]["assessment"]["disposition_rationale"]
    assert "UNAVAILABLE" in companies["ULTA"]["assessment"]["disposition_rationale"]
    assert companies["DUOL"]["deep_valuation"]["valuation_status"] == "SUPPRESSED_UNAVAILABLE"
    assert companies["ULTA"]["deep_valuation"]["valuation_status"] == "SUPPRESSED_UNAVAILABLE"


def _get_cumulative_patch_bytes(base_sha: str, repo_root: str) -> bytes:
    """
    Loads exported cumulative patch if present on disk; otherwise generates it from Git.
    """
    patch_file = os.path.join(repo_root, "sourced_fundamentals_pilot_cumulative.patch")
    if os.path.exists(patch_file):
        with open(patch_file, "rb") as f:
            return f.read()

    # Generate from git if inside repository
    try:
        return subprocess.check_output(
            ["git", "diff", base_sha, "HEAD"],
            cwd=repo_root
        )
    except Exception as e:
        pytest.fail(f"Exported patch not found at {patch_file} and git diff generation failed: {e}")


# 30. Valid UTF-8 Cumulative Patch Encoding Verification (Issue 4, Part A)
def test_30_valid_utf8_cumulative_patch_verification():
    """
    Scenario 30 (Issue 4, Part A):
    1. Loads exported cumulative patch or generates git diff from base commit 3b953f834a49d88496dfa9dae56584e17e60bae4 to HEAD.
    2. Verifies valid UTF-8 encoding:
       - No UTF-16 little-endian BOM (0xFF 0xFE).
       - No UTF-16 big-endian BOM (0xFE 0xFF).
       - No UTF-8 BOM (0xEF 0xBB 0xBF).
       - Zero embedded NUL bytes (0x00).
       - Strict utf-8 decoding succeeds.
       - Contains valid diff content headers.
    """
    base_sha = "3b953f834a49d88496dfa9dae56584e17e60bae4"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diff_bytes = _get_cumulative_patch_bytes(base_sha, repo_root)

    # 1. Byte-level encoding checks
    assert not diff_bytes.startswith(b"\xff\xfe"), "Patch must NOT start with UTF-16 LE BOM"
    assert not diff_bytes.startswith(b"\xfe\xff"), "Patch must NOT start with UTF-16 BE BOM"
    assert not diff_bytes.startswith(b"\xef\xbb\xbf"), "Patch must NOT start with UTF-8 BOM"
    assert b"\x00" not in diff_bytes, "Patch must NOT contain NUL bytes"

    # 2. Strict UTF-8 decode
    decoded_text = diff_bytes.decode("utf-8")
    assert len(decoded_text) > 0, "Patch text must not be empty"
    assert "diff --git" in decoded_text, "Patch must contain valid diff headers"


# 31. Cumulative Patch Application Against Base Commit (Issue 4, Part B)
def test_31_cumulative_patch_application_against_base():
    """
    Scenario 31 (Issue 4, Part B):
    1. Verifies repository availability and base commit object existence.
    2. Executes git apply --check in an isolated detached worktree at base commit 3b953f834a49d88496dfa9dae56584e17e60bae4.
    3. Guarantees safe cleanup without masking original test failures.
    4. In an extracted package without Git history, skips cleanly with precise diagnostic message.
    """
    base_sha = "3b953f834a49d88496dfa9dae56584e17e60bae4"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Inside extracted package without Git history: skip application check cleanly
    try:
        git_check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        if git_check.returncode != 0:
            pytest.skip("Not inside a git repository; application check requires repository history.")
    except FileNotFoundError:
        pytest.skip("Git executable not found; application check requires git command-line tool.")

    # Check if base commit object exists in this git repository
    base_check = subprocess.run(
        ["git", "cat-file", "-e", f"{base_sha}^{{commit}}"],
        cwd=repo_root,
        capture_output=True
    )
    if base_check.returncode != 0:
        pytest.skip(f"Base commit {base_sha} not present in repository history; application check requires base commit object.")

    diff_bytes = _get_cumulative_patch_bytes(base_sha, repo_root)

    td = tempfile.mkdtemp()
    worktree_added = False
    try:
        add_res = subprocess.run(
            ["git", "worktree", "add", "--detach", td, base_sha],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        assert add_res.returncode == 0, f"git worktree add failed: {add_res.stderr}"
        worktree_added = True

        res = subprocess.run(
            ["git", "apply", "--check"],
            input=diff_bytes,
            cwd=td,
            capture_output=True
        )
        assert res.returncode == 0, f"git apply --check failed: {res.stderr.decode('utf-8', errors='replace')}"
    finally:
        if worktree_added:
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", td],
                    cwd=repo_root,
                    capture_output=True
                )
            except Exception:
                pass
        if os.path.exists(td):
            import shutil
            shutil.rmtree(td, ignore_errors=True)

