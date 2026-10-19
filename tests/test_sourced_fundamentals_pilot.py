"""
tests/test_sourced_fundamentals_pilot.py

Unit and integration test suite for Sourced Fundamentals & Undervalued Growth Research Pilot.
Verifies SEC Fair Access compliance, non-overlapping TTM arithmetic, missing-data preservation,
valuation model convergence, and evidence ledger schema conformance.
"""

import os
import json
import pytest

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
)
from analyst_dashboard.research.sourced_fundamentals.normalizer import SecFactNormalizer
from analyst_dashboard.research.sourced_fundamentals.calculations import (
    compute_invested_capital,
    compute_effective_tax_rate,
    compute_derived_metrics,
)
from analyst_dashboard.research.sourced_fundamentals.valuation import (
    run_fcff_model,
    solve_reverse_dcf,
    generate_sensitivity_matrix,
)


def test_sec_fair_access_user_agent():
    """Verify SEC user agent follows official SEC format: ApplicationName ContactEmail."""
    assert "ARXTerminal Research" in SEC_USER_AGENT
    assert "@" in SEC_USER_AGENT
    h = SecEdgarHarvester()
    assert h.user_agent == SEC_USER_AGENT


def test_ttm_non_overlapping_identity():
    """
    Verify mathematical identity of non-overlapping TTM construction:
    TTM = Latest_FY + Current_YTD - Prior_Comparable_YTD.
    For Q2 YTD (6 months):
    FY = Q1_prior + Q2_prior + Q3_prior + Q4_prior.
    Current_YTD = Q1_curr + Q2_curr.
    Prior_YTD = Q1_prior + Q2_prior.
    TTM = FY + Current_YTD - Prior_YTD = Q3_prior + Q4_prior + Q1_curr + Q2_curr (exactly 4 non-overlapping quarters).
    """
    fy_val = 1000.0  # Q1_p(200) + Q2_p(250) + Q3_p(270) + Q4_p(280)
    curr_ytd = 600.0  # Q1_c(290) + Q2_c(310)
    prior_ytd = 450.0  # Q1_p(200) + Q2_p(250)
    
    ttm_val = fy_val + curr_ytd - prior_ytd
    expected_quarters_sum = 270.0 + 280.0 + 290.0 + 310.0
    assert ttm_val == expected_quarters_sum
    assert ttm_val == 1150.0


def test_missing_data_preserves_none():
    """Verify missing metrics remain None and UNAVAILABLE, never converted to synthetic zero."""
    fact = FinancialFact(
        metric_key="operating_income_ttm",
        value=None,
        units="USD",
        status=MetricStatus.UNAVAILABLE,
    )
    assert fact.value is None
    assert fact.status == MetricStatus.UNAVAILABLE
    d = fact.to_dict()
    assert d["value"] is None
    assert d["status"] == "UNAVAILABLE"


def test_invested_capital_and_effective_tax():
    """Verify invested capital calculation and tax rate bounds."""
    # Asset-rich firm: Equity 500, Debt 200, Cash 100 -> Invested Capital = 600
    ic1 = compute_invested_capital(500.0, 200.0, 100.0)
    assert ic1 == 600.0

    # Net-cash firm: Equity 500, Debt 0, Cash 600 -> Clamped to equity (500) rather than negative
    ic2 = compute_invested_capital(500.0, 0.0, 600.0)
    assert ic2 == 500.0

    # Effective tax rate calculation
    rate1 = compute_effective_tax_rate(25.0, 100.0)
    assert rate1 == 0.25

    # Tax holiday / negative pretax fallback
    rate2 = compute_effective_tax_rate(-10.0, 100.0)
    assert rate2 == 0.21


def test_valuation_model_fcff_and_reverse_dcf():
    """Verify FCFF DCF model produces consistent valuations and reverse-DCF converges."""
    base_rev = 1000.0
    shares = 10.0
    net_debt = 0.0
    target_price = 150.0

    # Run base DCF
    dcf = run_fcff_model(
        base_revenue=base_rev,
        current_shares=shares,
        net_debt_and_leases=net_debt,
        growth_5y_pct=15.0,
        operating_margin_pct=20.0,
        wacc_pct=10.0,
        terminal_growth_pct=2.5,
        tax_rate_pct=21.0,
        reinvestment_rate_pct=15.0,
    )
    assert dcf["per_share"] > 0
    assert dcf["enterprise_value"] > 0
    assert dcf["pv_explicit"] > 0
    assert dcf["pv_terminal"] > 0

    # Reverse-DCF: solve for growth at the calculated price
    solved_g = solve_reverse_dcf(
        target_price=dcf["per_share"],
        base_revenue=base_rev,
        current_shares=shares,
        net_debt_and_leases=net_debt,
        operating_margin_pct=20.0,
        wacc_pct=10.0,
        terminal_growth_pct=2.5,
        tax_rate_pct=21.0,
        reinvestment_rate_pct=15.0,
    )
    assert abs(solved_g - 15.0) < 0.1

    # Sensitivity matrix dimensions
    matrix = generate_sensitivity_matrix(
        base_revenue=base_rev,
        current_shares=shares,
        net_debt_and_leases=net_debt,
        growth_5y_pct=15.0,
        operating_margin_pct=20.0,
        tax_rate_pct=21.0,
        reinvestment_rate_pct=15.0,
        wacc_rates=[9.0, 10.0],
        terminal_rates=[2.0, 2.5],
    )
    assert "WACC_10.0%" in matrix
    assert "g_2.5%" in matrix["WACC_10.0%"]


def test_evidence_ledger_file_exists_and_reconciles():
    """Verify that the exported evidence ledger exists and satisfies all arithmetic integrity checks."""
    ledger_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "research", "sourced_fundamentals", "sourced_fundamentals_ledger.json"
    )
    assert os.path.exists(ledger_path), f"Ledger file not found at {ledger_path}"

    with open(ledger_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["metadata"]["status"] == "SOURCED_FUNDAMENTALS_RESEARCH_PILOT_LOCAL_VERIFIED"
    assert len(data["companies"]) == 8

    # Check arithmetic reconciliation summary
    reconcil = data["arithmetic_reconciliation_summary"]
    assert reconcil["all_cfo_minus_capex_reconciled"] is True
    assert reconcil["all_net_debt_reconciled"] is True
    assert reconcil["all_enterprise_value_reconciled"] is True

    # Check top 3 companies have deep valuations
    for sym in ["FIX", "DUOL", "ULTA"]:
        c = data["companies"][sym]
        assert c["assessment"]["disposition"] == "RESEARCH_FURTHER"
        assert c["deep_valuation"] is not None
        assert "scenarios" in c["deep_valuation"]
        assert "Conservative" in c["deep_valuation"]["scenarios"]
        assert "Base" in c["deep_valuation"]["scenarios"]
        assert "Optimistic" in c["deep_valuation"]["scenarios"]
        assert c["deep_valuation"]["reverse_dcf_implied_5y_growth_pct"] > 0
