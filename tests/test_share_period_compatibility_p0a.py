"""
tests/test_share_period_compatibility_p0a.py

Comprehensive Positive, Negative, and Adversarial Test Suite for P0-A:
Share-Denominator and Reporting-Period Integrity.

Verifies the 16 Required Gates specified by Section 10 of ARX P0-A Engineering Standard:
1. Valid point-in-time share count and market-capitalization compatibility.
2. Valid annual diluted EPS denominator.
3. Invalid interim-to-TTM denominator substitution (fails closed unless RESEARCH_ESTIMATE).
4. Valid historical annual share periods.
5. Present-value DCF denominator compatibility (rejects FORWARD_PROJECTED_TERMINAL).
6. Terminal valuation denominator compatibility (enforces forward projected terminal shares).
7. Missing ADR conversion ratio (fails closed with UNVERIFIED_ADR_RATIO, never defaults to 1.0).
8. Verified ADR conversion ratio (succeeds with verified source).
9. Incorrect security-class mapping (fails closed on non-voting/tracking shares).
10. Zero or negative denominator (fails closed with INVALID_SHARE_DENOMINATOR).
11. Missing provenance (fails closed with MISSING_PROVENANCE).
12. Share-unit mismatches (fails closed with SHARE_UNIT_MISMATCH).
13. Reporting-date mismatch (fails closed with PERIOD_ALIGNMENT_FAILED).
14. Existing FIX valuation regression (exact reproducibility).
15. Existing DUOL valuation regression (exact suppression reproducibility).
16. Existing ULTA valuation regression (exact suppression reproducibility).
"""

import json
import os
import pytest
from datetime import datetime

from analyst_dashboard.research.sourced_fundamentals.contracts import (
    FinancialFact,
    MetricStatus,
    CompanyFundamentals,
    DerivedMetrics,
    ShareBasis,
    DenominatorUseCase,
    DenominatorErrorCode,
    ShareDenominatorError,
    IncompatibleShareBasisError,
    PeriodAlignmentError,
    UnverifiedADRRatioError,
    IncompatibleShareClassError,
    ShareUnitMismatchError,
    MissingProvenanceError,
    InvalidShareDenominatorError,
    ADRConversionMetadata,
    ShareDenominatorMetadata,
    DenominatorValidationResult,
    validate_share_denominator_compatibility,
    DeepValuationAssessment,
)
from analyst_dashboard.research.sourced_fundamentals.valuation import (
    run_fcff_model,
    compute_terminal_exit_valuation,
    solve_reverse_dcf,
    generate_sensitivity_matrix,
    evaluate_deep_valuation_fix,
    evaluate_deep_valuation_duol,
    evaluate_deep_valuation_ulta,
)
from analyst_dashboard.research.sourced_fundamentals.identity import (
    verify_company_identity,
    verify_adr_identity_binding,
    ATTRIBUTABLE_SEC_IDENTITY_AUTHORITY,
)

LEDGER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "analyst_dashboard", "research", "sourced_fundamentals", "sourced_fundamentals_ledger.json"
)


# ---------------------------------------------------------------------------
# Test 1: Valid point-in-time share count and market-capitalization compatibility
# ---------------------------------------------------------------------------
def test_01_valid_point_in_time_market_cap_compatibility():
    meta = ShareDenominatorMetadata(
        share_count=35_200_000.0,
        basis=ShareBasis.POINT_IN_TIME_OUTSTANDING,
        use_case=DenominatorUseCase.MARKET_CAPITALIZATION,
        units="shares",
        as_of_date="2026-06-30",
        filing_date="2026-07-25",
        accession_number="0001035983-26-000020",
        form="10-Q",
        provenance_notes="Cover page common shares outstanding"
    )
    res = validate_share_denominator_compatibility(meta, raise_on_error=True)
    assert res.is_valid is True
    assert res.share_count == 35_200_000.0
    assert res.basis == ShareBasis.POINT_IN_TIME_OUTSTANDING

    # Independent calculation of market capitalization
    ref_price = 1439.07
    expected_market_cap = 35_200_000.0 * ref_price
    assert expected_market_cap == 50_655_264_000.0


# ---------------------------------------------------------------------------
# Test 2: Valid annual diluted EPS denominator
# ---------------------------------------------------------------------------
def test_02_valid_annual_diluted_eps_denominator():
    meta = ShareDenominatorMetadata(
        share_count=35_000_000.0,
        basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
        use_case=DenominatorUseCase.HISTORICAL_ANNUAL_EPS,
        units="shares",
        period_start="2025-01-01",
        period_end="2025-12-31",
        filing_date="2026-02-20",
        accession_number="0001035983-26-000010",
        form="10-K",
        provenance_notes="Full year weighted-average diluted shares"
    )
    res = validate_share_denominator_compatibility(meta, raise_on_error=True)
    assert res.is_valid is True
    assert res.effective_shares == 35_000_000.0

    # Independent annual diluted EPS calculation
    net_income = 350_000_000.0
    annual_eps = net_income / res.effective_shares
    assert annual_eps == 10.00


# ---------------------------------------------------------------------------
# Test 3: Invalid interim-to-TTM denominator substitution
# ---------------------------------------------------------------------------
def test_03_invalid_interim_to_ttm_denominator_substitution():
    num_fact = FinancialFact(
        metric_key="net_income_ttm",
        value=400_000_000.0,
        units="USD",
        period_start="2025-07-01",
        period_end="2026-06-30",  # 364 days duration
    )
    # Denominator is single interim quarter (90 days) without explicit RESEARCH_ESTIMATE status
    interim_meta = ShareDenominatorMetadata(
        share_count=50_000_000.0,
        basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
        use_case=DenominatorUseCase.TRAILING_TWELVE_MONTHS_EPS,
        units="shares",
        period_start="2026-04-01",
        period_end="2026-06-30",  # 90 days duration
        filing_date="2026-08-08",
        accession_number="0001562088-26-000015",
        form="10-Q",
        estimate_status=None,  # Unmarked substitution
    )

    # Must fail closed with PeriodAlignmentError
    with pytest.raises(PeriodAlignmentError) as exc_info:
        validate_share_denominator_compatibility(
            interim_meta,
            numerator_fact=num_fact,
            raise_on_error=True
        )
    assert exc_info.value.code == DenominatorErrorCode.PERIOD_ALIGNMENT_FAILED
    assert "RESEARCH_ESTIMATE" in str(exc_info.value)

    # When marked with estimate_status="RESEARCH_ESTIMATE", it succeeds as an explicit estimate
    interim_meta.estimate_status = "RESEARCH_ESTIMATE"
    res = validate_share_denominator_compatibility(
        interim_meta,
        numerator_fact=num_fact,
        raise_on_error=True
    )
    assert res.is_valid is True
    assert res.estimate_status == "RESEARCH_ESTIMATE"


# ---------------------------------------------------------------------------
# Test 4: Valid historical annual share periods
# ---------------------------------------------------------------------------
def test_04_valid_historical_annual_share_periods():
    # Case A: Calendar annual 365 days -> Valid
    meta_cal = ShareDenominatorMetadata(
        share_count=100_000_000.0,
        basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
        use_case=DenominatorUseCase.HISTORICAL_ANNUAL_EPS,
        period_start="2025-01-01",
        period_end="2025-12-31",
        accession_number="A1",
        form="10-K",
    )
    res_cal = validate_share_denominator_compatibility(meta_cal, raise_on_error=True)
    assert res_cal.is_valid is True

    # Case B: 52-week retail annual calendar (364/371 days) -> Valid
    meta_retail = ShareDenominatorMetadata(
        share_count=42_800_000.0,
        basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
        use_case=DenominatorUseCase.HISTORICAL_ANNUAL_EPS,
        period_start="2025-02-02",
        period_end="2026-01-31",  # 363 days
        accession_number="A2",
        form="10-K",
    )
    res_retail = validate_share_denominator_compatibility(meta_retail, raise_on_error=True)
    assert res_retail.is_valid is True

    # Case C: 90-day interim quarter supplied to HISTORICAL_ANNUAL_EPS -> Fails closed
    meta_short = ShareDenominatorMetadata(
        share_count=42_800_000.0,
        basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
        use_case=DenominatorUseCase.HISTORICAL_ANNUAL_EPS,
        period_start="2025-11-01",
        period_end="2026-01-31",  # 91 days
        accession_number="A3",
        form="10-Q",
    )
    with pytest.raises(PeriodAlignmentError) as exc_info:
        validate_share_denominator_compatibility(meta_short, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.PERIOD_ALIGNMENT_FAILED


# ---------------------------------------------------------------------------
# Test 5: Present-value DCF denominator compatibility (rejects terminal shares)
# ---------------------------------------------------------------------------
def test_05_pv_dcf_denominator_rejects_terminal_shares():
    # Terminal projected shares cannot be used for Present-Value DCF
    terminal_meta = ShareDenominatorMetadata(
        share_count=45_000_000.0,
        basis=ShareBasis.FORWARD_PROJECTED_TERMINAL,
        use_case=DenominatorUseCase.PROJECTED_TERMINAL_EXIT,
        as_of_date="2031-12-31",
        provenance_notes="Modeled terminal diluted shares after 5-year dilution and repurchases"
    )

    with pytest.raises(IncompatibleShareBasisError) as exc_info:
        run_fcff_model(
            base_revenue=5000.0,
            current_shares=45_000_000.0,
            net_debt_bridge=-500.0,
            growth_5y_pct=15.0,
            operating_margin_pct=14.0,
            wacc_pct=9.5,
            terminal_growth_pct=2.5,
            share_metadata=terminal_meta,
        )
    assert exc_info.value.code == DenominatorErrorCode.INCOMPATIBLE_SHARE_BASIS
    assert "FORWARD_PROJECTED_TERMINAL" in str(exc_info.value)

    # Valid CURRENT_VALUATION_DILUTED basis runs successfully
    valid_pv_meta = ShareDenominatorMetadata(
        share_count=35_254_000.0,
        basis=ShareBasis.CURRENT_VALUATION_DILUTED,
        use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        as_of_date="2026-06-30",
        form="10-Q",
        accession_number="A1",
    )
    res = run_fcff_model(
        base_revenue=5000.0,
        current_shares=35_254_000.0,
        net_debt_bridge=-500.0,
        growth_5y_pct=15.0,
        operating_margin_pct=14.0,
        wacc_pct=9.5,
        terminal_growth_pct=2.5,
        share_metadata=valid_pv_meta,
    )
    assert res["per_share"] > 0


# ---------------------------------------------------------------------------
# Test 6: Terminal valuation denominator compatibility
# ---------------------------------------------------------------------------
def test_06_terminal_valuation_denominator_compatibility():
    term_meta = ShareDenominatorMetadata(
        share_count=50_000_000.0,
        basis=ShareBasis.FORWARD_PROJECTED_TERMINAL,
        use_case=DenominatorUseCase.PROJECTED_TERMINAL_EXIT,
        period_end="2031-12-31",
        provenance_notes="Forecast horizon exit diluted shares"
    )
    # Terminal FCF = $600M, WACC = 9.5%, g = 2.5%, terminal shares = 50M
    # Terminal EV = 600 * 1.025 / (0.095 - 0.025) = 615 / 0.070 = 8785.7142857M
    # Terminal Equity = 8785.7142857 - 0 = 8785.7142857M
    # Per share exit = 8785.7142857 / 50 = $175.7142857
    res = compute_terminal_exit_valuation(
        terminal_fcf=600.0,
        wacc_pct=9.5,
        terminal_growth_pct=2.5,
        terminal_shares=50.0,  # in millions
        terminal_net_debt=0.0,
        share_metadata=term_meta,
    )
    expected_exit = (600.0 * 1.025 / 0.070) / 50.0
    assert abs(res["terminal_per_share_exit"] - expected_exit) < 1e-4
    assert abs(res["terminal_per_share_exit"] - 175.714) < 0.01


# ---------------------------------------------------------------------------
# Test 7: Missing ADR conversion ratio fails closed
# ---------------------------------------------------------------------------
def test_07_missing_adr_conversion_ratio():
    # Case A: ADR metadata with ratio = None (never default to 1.0)
    meta_none_ratio = ShareDenominatorMetadata(
        share_count=498_000_000.0,
        basis=ShareBasis.SECURITY_CLASS_ADR_ADJUSTED,
        use_case=DenominatorUseCase.ADR_CROSS_BORDER_CLAIM,
        as_of_date="2026-06-30",
        form="20-F",
        accession_number="B1",
        adr_metadata=ADRConversionMetadata(
            is_adr=True,
            adr_ratio=None,  # Missing ratio!
            is_verified=False,
            ratio_source=None,
        )
    )
    with pytest.raises(UnverifiedADRRatioError) as exc_info:
        validate_share_denominator_compatibility(meta_none_ratio, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.UNVERIFIED_ADR_RATIO

    # Case B: Unverified ADR ratio fails closed
    meta_unverified = ShareDenominatorMetadata(
        share_count=498_000_000.0,
        basis=ShareBasis.SECURITY_CLASS_ADR_ADJUSTED,
        use_case=DenominatorUseCase.ADR_CROSS_BORDER_CLAIM,
        as_of_date="2026-06-30",
        form="20-F",
        accession_number="B1",
        adr_metadata=ADRConversionMetadata(
            is_adr=True,
            adr_ratio=1.0,
            is_verified=False,  # Not verified!
            ratio_source="Internet assumption (unverified)",
        )
    )
    with pytest.raises(UnverifiedADRRatioError) as exc_info:
        validate_share_denominator_compatibility(meta_unverified, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.UNVERIFIED_ADR_RATIO


# ---------------------------------------------------------------------------
# Test 8: Verified ADR conversion ratio
# ---------------------------------------------------------------------------
def test_08_verified_adr_conversion_ratio():
    # Case A: 1:1 verified ADR ratio (e.g. Alcon NYSE: ALC)
    meta_alc = ShareDenominatorMetadata(
        share_count=498_600_000.0,
        basis=ShareBasis.SECURITY_CLASS_ADR_ADJUSTED,
        use_case=DenominatorUseCase.ADR_CROSS_BORDER_CLAIM,
        as_of_date="2026-06-30",
        form="20-F",
        accession_number="0001758509-26-000008",
        adr_metadata=ADRConversionMetadata(
            is_adr=True,
            adr_ratio=1.0,
            ratio_source="SEC Form 20-F Item 9.A / JPMorgan Depositary Agreement",
            depositary_bank="JPMorgan Chase Bank, N.A.",
            is_verified=True,
        )
    )
    res_alc = validate_share_denominator_compatibility(meta_alc, raise_on_error=True)
    assert res_alc.is_valid is True
    assert res_alc.effective_shares == 498_600_000.0

    # Case B: Fractional ADR ratio (e.g. 1 ADR = 5 ordinary shares -> ratio = 0.20 ADR per ordinary)
    meta_foreign = ShareDenominatorMetadata(
        share_count=100_000_000.0,
        basis=ShareBasis.SECURITY_CLASS_ADR_ADJUSTED,
        use_case=DenominatorUseCase.ADR_CROSS_BORDER_CLAIM,
        as_of_date="2026-06-30",
        form="20-F",
        accession_number="0000999999-26-000001",
        economic_claim_ratio=0.20,
        adr_metadata=ADRConversionMetadata(
            is_adr=True,
            adr_ratio=0.20,
            ratio_source="SEC Form F-6 Registration Statement No. 333-123456",
            depositary_bank="Citibank, N.A.",
            is_verified=True,
        )
    )
    res_foreign = validate_share_denominator_compatibility(meta_foreign, raise_on_error=True)
    assert res_foreign.is_valid is True
    assert res_foreign.effective_shares == 20_000_000.0

    # Identity authority check
    id_res = verify_adr_identity_binding("ALC", "0001758509", 1.0, "SEC Form 20-F")
    assert id_res.is_valid is True


# ---------------------------------------------------------------------------
# Test 9: Incorrect security-class mapping
# ---------------------------------------------------------------------------
def test_09_incorrect_security_class_mapping():
    # Non-voting preferred share class cannot serve as common equity denominator
    meta_pref = ShareDenominatorMetadata(
        share_count=10_000_000.0,
        basis=ShareBasis.POINT_IN_TIME_OUTSTANDING,
        use_case=DenominatorUseCase.MARKET_CAPITALIZATION,
        share_class="Non-Voting Preferred",
        is_multi_class=True,
        as_of_date="2026-06-30",
        form="10-Q",
        accession_number="P1",
    )
    with pytest.raises(IncompatibleShareClassError) as exc_info:
        validate_share_denominator_compatibility(meta_pref, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.INCOMPATIBLE_SHARE_CLASS

    # Non-positive economic claim ratio fails closed
    meta_zero_claim = ShareDenominatorMetadata(
        share_count=10_000_000.0,
        basis=ShareBasis.POINT_IN_TIME_OUTSTANDING,
        use_case=DenominatorUseCase.MARKET_CAPITALIZATION,
        share_class="Class B",
        is_multi_class=True,
        economic_claim_ratio=0.0,
        as_of_date="2026-06-30",
        form="10-Q",
        accession_number="P2",
    )
    with pytest.raises(IncompatibleShareClassError) as exc_info:
        validate_share_denominator_compatibility(meta_zero_claim, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.INCOMPATIBLE_SHARE_CLASS


# ---------------------------------------------------------------------------
# Test 10: Zero or negative denominator
# ---------------------------------------------------------------------------
def test_10_zero_or_negative_denominator():
    # Zero share count
    meta_zero = ShareDenominatorMetadata(
        share_count=0.0,
        basis=ShareBasis.POINT_IN_TIME_OUTSTANDING,
        use_case=DenominatorUseCase.MARKET_CAPITALIZATION,
        as_of_date="2026-06-30",
        form="10-Q",
        accession_number="Z1",
    )
    with pytest.raises(InvalidShareDenominatorError) as exc_info:
        validate_share_denominator_compatibility(meta_zero, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.INVALID_SHARE_DENOMINATOR

    # Negative share count
    meta_neg = ShareDenominatorMetadata(
        share_count=-100_000.0,
        basis=ShareBasis.POINT_IN_TIME_OUTSTANDING,
        use_case=DenominatorUseCase.MARKET_CAPITALIZATION,
        as_of_date="2026-06-30",
        form="10-Q",
        accession_number="Z2",
    )
    with pytest.raises(InvalidShareDenominatorError) as exc_info:
        validate_share_denominator_compatibility(meta_neg, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.INVALID_SHARE_DENOMINATOR

    # Valuation function raises InvalidShareDenominatorError (inherits from ValueError)
    with pytest.raises(InvalidShareDenominatorError):
        run_fcff_model(5000.0, 0.0, 0.0, 10.0, 15.0, 9.0, 2.0)


# ---------------------------------------------------------------------------
# Test 11: Missing provenance
# ---------------------------------------------------------------------------
def test_11_missing_provenance():
    # Missing date AND missing accession/form/notes
    meta_no_prov = ShareDenominatorMetadata(
        share_count=50_000_000.0,
        basis=ShareBasis.POINT_IN_TIME_OUTSTANDING,
        use_case=DenominatorUseCase.MARKET_CAPITALIZATION,
        as_of_date=None,
        period_end=None,
        filing_date=None,
        accession_number=None,
        form=None,
        provenance_notes=None,
    )
    with pytest.raises(MissingProvenanceError) as exc_info:
        validate_share_denominator_compatibility(meta_no_prov, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.MISSING_PROVENANCE


# ---------------------------------------------------------------------------
# Test 12: Share-unit mismatches
# ---------------------------------------------------------------------------
def test_12_share_unit_mismatches():
    # Denominator with units="USD" (e.g. dollar value passed instead of shares)
    meta_usd = ShareDenominatorMetadata(
        share_count=50_000_000.0,
        basis=ShareBasis.POINT_IN_TIME_OUTSTANDING,
        use_case=DenominatorUseCase.MARKET_CAPITALIZATION,
        units="USD",  # Mismatch!
        as_of_date="2026-06-30",
        form="10-Q",
        accession_number="U1",
    )
    with pytest.raises(ShareUnitMismatchError) as exc_info:
        validate_share_denominator_compatibility(meta_usd, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.SHARE_UNIT_MISMATCH

    # Denominator with units="percent"
    meta_pct = ShareDenominatorMetadata(
        share_count=15.5,
        basis=ShareBasis.POINT_IN_TIME_OUTSTANDING,
        use_case=DenominatorUseCase.MARKET_CAPITALIZATION,
        units="percent",
        as_of_date="2026-06-30",
        form="10-Q",
        accession_number="U2",
    )
    with pytest.raises(ShareUnitMismatchError) as exc_info:
        validate_share_denominator_compatibility(meta_pct, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.SHARE_UNIT_MISMATCH


# ---------------------------------------------------------------------------
# Test 13: Reporting-date mismatch
# ---------------------------------------------------------------------------
def test_13_reporting_date_mismatch():
    num_fact = FinancialFact(
        metric_key="operating_income_ttm",
        value=500_000_000.0,
        units="USD",
        period_end="2026-06-30",
    )
    # Stale denominator from 1 year earlier (2025-06-30, diff = 365 days > 7 days)
    stale_den = ShareDenominatorMetadata(
        share_count=35_000_000.0,
        basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
        use_case=DenominatorUseCase.TRAILING_TWELVE_MONTHS_EPS,
        period_start="2024-07-01",
        period_end="2025-06-30",
        form="10-K",
        accession_number="S1",
        estimate_status="RESEARCH_ESTIMATE",
    )
    with pytest.raises(PeriodAlignmentError) as exc_info:
        validate_share_denominator_compatibility(stale_den, numerator_fact=num_fact, raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.PERIOD_ALIGNMENT_FAILED
    assert "delta=365 days" in str(exc_info.value)

    # Lookahead date: share measurement date after valuation date
    meta_lookahead = ShareDenominatorMetadata(
        share_count=35_000_000.0,
        basis=ShareBasis.POINT_IN_TIME_OUTSTANDING,
        use_case=DenominatorUseCase.MARKET_CAPITALIZATION,
        as_of_date="2026-09-30",
        form="10-Q",
        accession_number="S2",
    )
    with pytest.raises(PeriodAlignmentError) as exc_info:
        validate_share_denominator_compatibility(meta_lookahead, valuation_date="2026-06-30", raise_on_error=True)
    assert exc_info.value.code == DenominatorErrorCode.PERIOD_ALIGNMENT_FAILED
    assert "lookahead bias" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 14: Existing FIX valuation regression
# ---------------------------------------------------------------------------
def test_14_existing_fix_valuation_regression():
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    fix_comp = ledger["companies"]["FIX"]
    cf_data = fix_comp["fundamentals"]
    m_data = fix_comp["derived_metrics"]

    # Reconstruct CompanyFundamentals and DerivedMetrics
    facts = {}
    for k, v in cf_data["facts"].items():
        facts[k] = FinancialFact(
            metric_key=v["metric_key"],
            value=v["value"],
            units=v["units"],
            concept=v.get("concept"),
            period_start=v.get("period_start"),
            period_end=v.get("period_end"),
            filing_date=v.get("filing_date"),
            accession_number=v.get("accession_number"),
            form=v.get("form"),
            status=MetricStatus(v.get("status", "REPORTED")),
            notes=v.get("notes"),
        )
    cf = CompanyFundamentals(
        symbol=cf_data["symbol"],
        company_name=cf_data["company_name"],
        cik=cf_data["cik"],
        primary_exchange=cf_data["primary_exchange"],
        sector_industry=cf_data["sector_industry"],
        business_model=cf_data["business_model"],
        fiscal_year_end=cf_data["fiscal_year_end"],
        latest_reported_period=cf_data["latest_reported_period"],
        latest_filing_date=cf_data["latest_filing_date"],
        facts=facts,
    )
    metrics = DerivedMetrics(
        symbol=m_data["symbol"],
        period_label=m_data["period_label"],
        revenue_ttm=m_data["revenue_ttm"],
        net_income_ttm=m_data["net_income_ttm"],
        diluted_shares_weighted_avg=m_data["diluted_shares_weighted_avg"],
        common_shares_outstanding=m_data["common_shares_outstanding"],
        net_debt=m_data["net_debt"],
        market_price=m_data["market_price"],
        market_price_date=m_data["market_price_date"],
        market_cap=m_data["market_cap"],
    )

    assessment = evaluate_deep_valuation_fix(cf, metrics)
    assert assessment is not None
    assert assessment.valuation_status == "AVAILABLE"
    assert assessment.is_certified is True
    assert "Conservative" in assessment.scenarios
    assert "Base" in assessment.scenarios
    assert "Optimistic" in assessment.scenarios

    # Exact regression parity bounds from authoritative valuation engine
    assert abs(assessment.scenarios["Conservative"].per_share_value_estimate - 527.35) < 0.10
    assert abs(assessment.scenarios["Base"].per_share_value_estimate - 1068.94) < 0.10
    assert abs(assessment.scenarios["Optimistic"].per_share_value_estimate - 1971.75) < 0.10
    assert abs(assessment.reverse_dcf_implied_5y_growth_pct - 30.43) < 0.10


# ---------------------------------------------------------------------------
# Test 15: Existing DUOL valuation regression
# ---------------------------------------------------------------------------
def test_15_existing_duol_valuation_regression():
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    duol_comp = ledger["companies"]["DUOL"]
    cf_data = duol_comp["fundamentals"]
    m_data = duol_comp["derived_metrics"]

    facts = {}
    for k, v in cf_data["facts"].items():
        facts[k] = FinancialFact(
            metric_key=v["metric_key"],
            value=v["value"],
            units=v["units"],
            status=MetricStatus(v.get("status", "REPORTED")),
        )
    cf = CompanyFundamentals(
        symbol=cf_data["symbol"],
        company_name=cf_data["company_name"],
        cik=cf_data["cik"],
        primary_exchange=cf_data["primary_exchange"],
        sector_industry=cf_data["sector_industry"],
        business_model=cf_data["business_model"],
        fiscal_year_end=cf_data["fiscal_year_end"],
        latest_reported_period=cf_data["latest_reported_period"],
        latest_filing_date=cf_data["latest_filing_date"],
        facts=facts,
    )
    metrics = DerivedMetrics(
        symbol=m_data["symbol"],
        period_label=m_data["period_label"],
        revenue_ttm=m_data["revenue_ttm"],
        diluted_shares_weighted_avg=m_data["diluted_shares_weighted_avg"],
        net_debt=m_data["net_debt"],  # None for DUOL
        market_price=m_data["market_price"],
    )

    assessment = evaluate_deep_valuation_duol(cf, metrics)
    assert assessment is not None
    assert assessment.valuation_status == "SUPPRESSED_UNAVAILABLE"
    assert assessment.is_certified is False
    assert "net debt bridge cannot be established" in assessment.suppression_reason
    assert assessment.reverse_dcf_fixed_assumptions["solution_status"] == "SUPPRESSED_NET_DEBT_UNAVAILABLE"
    assert assessment.reverse_dcf_implied_5y_growth_pct is None


# ---------------------------------------------------------------------------
# Test 16: Existing ULTA valuation regression
# ---------------------------------------------------------------------------
def test_16_existing_ulta_valuation_regression():
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)

    ulta_comp = ledger["companies"]["ULTA"]
    cf_data = ulta_comp["fundamentals"]
    m_data = ulta_comp["derived_metrics"]

    facts = {}
    for k, v in cf_data["facts"].items():
        facts[k] = FinancialFact(
            metric_key=v["metric_key"],
            value=v["value"],
            units=v["units"],
            status=MetricStatus(v.get("status", "REPORTED")),
        )
    cf = CompanyFundamentals(
        symbol=cf_data["symbol"],
        company_name=cf_data["company_name"],
        cik=cf_data["cik"],
        primary_exchange=cf_data["primary_exchange"],
        sector_industry=cf_data["sector_industry"],
        business_model=cf_data["business_model"],
        fiscal_year_end=cf_data["fiscal_year_end"],
        latest_reported_period=cf_data["latest_reported_period"],
        latest_filing_date=cf_data["latest_filing_date"],
        facts=facts,
    )
    metrics = DerivedMetrics(
        symbol=m_data["symbol"],
        period_label=m_data["period_label"],
        revenue_ttm=m_data["revenue_ttm"],
        diluted_shares_weighted_avg=m_data["diluted_shares_weighted_avg"],
        net_debt=m_data["net_debt"],  # None for ULTA
        market_price=m_data["market_price"],
    )

    assessment = evaluate_deep_valuation_ulta(cf, metrics)
    assert assessment is not None
    assert assessment.valuation_status == "SUPPRESSED_UNAVAILABLE"
    assert assessment.is_certified is False
    assert "net debt bridge cannot be established" in assessment.suppression_reason
    assert assessment.reverse_dcf_fixed_assumptions["solution_status"] == "SUPPRESSED_NET_DEBT_UNAVAILABLE"
    assert assessment.reverse_dcf_implied_5y_growth_pct is None


# ---------------------------------------------------------------------------
# Adversarial Suite: P0-A Targeted Remediation Verification (12 Scenarios)
# ---------------------------------------------------------------------------

def _load_fix_fundamentals_and_metrics():
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    fix_comp = ledger["companies"]["FIX"]
    cf_data = fix_comp["fundamentals"]
    m_data = fix_comp["derived_metrics"]
    facts = {
        k: FinancialFact(
            metric_key=v["metric_key"],
            value=v["value"],
            units=v["units"],
            concept=v.get("concept"),
            period_start=v.get("period_start"),
            period_end=v.get("period_end"),
            filing_date=v.get("filing_date"),
            accession_number=v.get("accession_number"),
            form=v.get("form"),
            status=MetricStatus(v.get("status", "REPORTED")),
            notes=v.get("notes"),
        )
        for k, v in cf_data["facts"].items()
    }
    cf = CompanyFundamentals(
        symbol=cf_data["symbol"],
        company_name=cf_data["company_name"],
        cik=cf_data["cik"],
        primary_exchange=cf_data["primary_exchange"],
        sector_industry=cf_data["sector_industry"],
        business_model=cf_data["business_model"],
        fiscal_year_end=cf_data["fiscal_year_end"],
        latest_reported_period=cf_data["latest_reported_period"],
        latest_filing_date=cf_data["latest_filing_date"],
        facts=facts,
    )
    metrics = DerivedMetrics(
        symbol=m_data["symbol"],
        period_label=m_data["period_label"],
        revenue_ttm=m_data["revenue_ttm"],
        net_income_ttm=m_data["net_income_ttm"],
        diluted_shares_weighted_avg=m_data["diluted_shares_weighted_avg"],
        common_shares_outstanding=m_data["common_shares_outstanding"],
        net_debt=m_data["net_debt"],
        market_price=m_data["market_price"],
        market_price_date=m_data["market_price_date"],
        market_cap=m_data["market_cap"],
    )
    return cf, metrics


# 1. FIX valuation with forward terminal shares incorrectly supplied for present-value DCF
def test_adversarial_01_fix_valuation_fails_closed_on_terminal_shares():
    cf, metrics = _load_fix_fundamentals_and_metrics()
    bad_meta = ShareDenominatorMetadata(
        share_count=35_254_000.0,
        basis=ShareBasis.FORWARD_PROJECTED_TERMINAL,
        use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        filing_date="2026-07-24",
        form="10-Q",
        accession_number="0001035983-26-000045",
    )
    metrics.share_denominator_metadata = {"diluted_shares": bad_meta.to_dict()}
    assessment = evaluate_deep_valuation_fix(cf, metrics)
    assert assessment.valuation_status == "SUPPRESSED_DENOMINATOR_INCOMPATIBLE"
    assert assessment.is_certified is False
    assert "INCOMPATIBLE_SHARE_BASIS" in assessment.suppression_reason
    assert "never projected terminal shares" in assessment.suppression_reason


# 2. FIX valuation with incompatible historical weighted-average share periods
def test_adversarial_02_fix_valuation_fails_closed_on_incompatible_period():
    cf, metrics = _load_fix_fundamentals_and_metrics()
    bad_meta = ShareDenominatorMetadata(
        share_count=35_254_000.0,
        basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
        use_case=DenominatorUseCase.HISTORICAL_ANNUAL_EPS,
        period_start="2026-04-01",
        period_end="2026-06-30",  # 90 days, invalid for annual (~365 days)
        filing_date="2026-07-24",
        form="10-Q",
        accession_number="0001035983-26-000045",
    )
    metrics.share_denominator_metadata = {"diluted_shares": bad_meta.to_dict()}
    assessment = evaluate_deep_valuation_fix(cf, metrics)
    assert assessment.valuation_status == "SUPPRESSED_DENOMINATOR_INCOMPATIBLE"
    assert assessment.is_certified is False
    assert "PERIOD_ALIGNMENT_FAILED" in assessment.suppression_reason


# 3. Missing valuation-date denominator provenance
def test_adversarial_03_fix_valuation_fails_closed_on_missing_provenance():
    cf, metrics = _load_fix_fundamentals_and_metrics()
    bad_meta = ShareDenominatorMetadata(
        share_count=35_254_000.0,
        basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
        use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        accession_number=None,
        form=None,
        filing_date=None,
        estimate_status=None,
    )
    metrics.share_denominator_metadata = {"diluted_shares": bad_meta.to_dict()}
    assessment = evaluate_deep_valuation_fix(cf, metrics)
    assert assessment.valuation_status in ["SUPPRESSED_DENOMINATOR_INCOMPATIBLE", "SUPPRESSED_DENOMINATOR_UNAVAILABLE"]
    assert assessment.is_certified is False
    assert "MISSING_PROVENANCE" in assessment.suppression_reason


# 4. Unverified ADR conversion
def test_adversarial_04_fix_valuation_fails_closed_on_unverified_adr():
    cf, metrics = _load_fix_fundamentals_and_metrics()
    bad_meta = ShareDenominatorMetadata(
        share_count=35_254_000.0,
        basis=ShareBasis.SECURITY_CLASS_ADR_ADJUSTED,
        use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        filing_date="2026-07-24",
        form="10-Q",
        accession_number="0001035983-26-000045",
        adr_metadata=ADRConversionMetadata(is_adr=True, is_verified=False, adr_ratio=None),
    )
    metrics.share_denominator_metadata = {"diluted_shares": bad_meta.to_dict()}
    assessment = evaluate_deep_valuation_fix(cf, metrics)
    assert assessment.valuation_status == "SUPPRESSED_DENOMINATOR_INCOMPATIBLE"
    assert assessment.is_certified is False
    assert "UNVERIFIED_ADR_RATIO" in assessment.suppression_reason


# 5. Valid compatible denominator metadata produces certified valuation
def test_adversarial_05_valid_compatible_metadata_produces_certified_valuation():
    cf, metrics = _load_fix_fundamentals_and_metrics()
    good_meta = ShareDenominatorMetadata(
        share_count=35_254_000.0,
        basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
        use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        period_start="2026-04-01",
        period_end="2026-06-30",
        filing_date="2026-07-24",
        form="10-Q",
        accession_number="0001035983-26-000045",
    )
    metrics.share_denominator_metadata = {"diluted_shares": good_meta.to_dict()}
    assessment = evaluate_deep_valuation_fix(cf, metrics)
    assert assessment.valuation_status == "AVAILABLE"
    assert assessment.is_certified is True
    assert assessment.share_basis == "WEIGHTED_AVERAGE_DILUTED"
    assert assessment.share_denominator_metadata is not None
    assert abs(assessment.scenarios["Base"].per_share_value_estimate - 1068.94) < 0.10


# 6. Reverse-DCF propagation fails closed
def test_adversarial_06_reverse_dcf_propagation_fails_closed():
    bad_meta = ShareDenominatorMetadata(
        share_count=35_254_000.0,
        basis=ShareBasis.FORWARD_PROJECTED_TERMINAL,
        use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        filing_date="2026-07-24",
        form="10-Q",
        accession_number="0001035983-26-000045",
    )
    implied_g, note = solve_reverse_dcf(
        target_price=1439.07,
        base_revenue=6160.71,
        current_shares=35.79,
        net_debt_bridge=-1800.73,
        operating_margin_pct=15.0,
        wacc_pct=9.5,
        terminal_growth_pct=2.5,
        share_metadata=bad_meta,
    )
    assert implied_g is None
    assert "SUPPRESSED_DENOMINATOR_INCOMPATIBLE" in note
    assert "INCOMPATIBLE_SHARE_BASIS" in note


# 7. Sensitivity-matrix propagation fails closed
def test_adversarial_07_sensitivity_matrix_propagation_fails_closed():
    bad_meta = ShareDenominatorMetadata(
        share_count=35_254_000.0,
        basis=ShareBasis.FORWARD_PROJECTED_TERMINAL,
        use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        filing_date="2026-07-24",
        form="10-Q",
        accession_number="0001035983-26-000045",
    )
    matrix = generate_sensitivity_matrix(
        base_revenue=6160.71,
        current_shares=35.79,
        net_debt_bridge=-1800.73,
        growth_5y_pct=22.0,
        operating_margin_pct=15.0,
        tax_rate_pct=22.0,
        reinvestment_rate_pct=15.0,
        share_metadata=bad_meta,
    )
    for w_key, row in matrix.items():
        for g_key, val in row.items():
            assert val is None, f"Expected None in cell {w_key} {g_key}, got {val}"


# 8. Unsupported share class fails closed
def test_adversarial_08_unsupported_share_class_fails_closed():
    cf, metrics = _load_fix_fundamentals_and_metrics()
    bad_meta = ShareDenominatorMetadata(
        share_count=35_254_000.0,
        basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
        use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        share_class="Non-Voting Preferred",
        is_multi_class=True,
        filing_date="2026-07-24",
        form="10-Q",
        accession_number="0001035983-26-000045",
    )
    metrics.share_denominator_metadata = {"diluted_shares": bad_meta.to_dict()}
    assessment = evaluate_deep_valuation_fix(cf, metrics)
    assert assessment.valuation_status == "SUPPRESSED_DENOMINATOR_INCOMPATIBLE"
    assert assessment.is_certified is False
    assert "INCOMPATIBLE_SHARE_CLASS" in assessment.suppression_reason


# 9. Legacy caller without metadata handled honestly
def test_adversarial_09_legacy_caller_without_metadata_handled_honestly():
    # Empty cf with no share facts
    cf = CompanyFundamentals(
        symbol="FIX",
        company_name="Comfort Systems USA",
        cik="0001035983",
        primary_exchange="NYSE",
        sector_industry="Industrials",
        business_model="HVAC",
        fiscal_year_end="12-31",
        latest_reported_period="2026-06-30",
        latest_filing_date="2026-07-24",
        facts={},  # No diluted_shares_weighted_avg fact
    )
    metrics = DerivedMetrics(
        symbol="FIX",
        period_label="TTM",
        revenue_ttm=6160.71,
        net_income_ttm=400.0,
        diluted_shares_weighted_avg=35.254,
        net_debt=-1800.73,
        market_price=1439.07,
        share_denominator_metadata=None,
    )
    # When certified metadata required: suppresses
    assessment_strict = evaluate_deep_valuation_fix(cf, metrics, require_certified_metadata=True)
    assert assessment_strict.valuation_status == "SUPPRESSED_DENOMINATOR_UNAVAILABLE"
    assert assessment_strict.is_certified is False

    # When uncertified evaluation allowed: runs but marked is_certified=False
    assessment_legacy = evaluate_deep_valuation_fix(cf, metrics, require_certified_metadata=False)
    assert assessment_legacy.valuation_status == "AVAILABLE"
    assert assessment_legacy.is_certified is False


# 10. Existing DUOL and ULTA suppression behavior preserved
def test_adversarial_10_existing_duol_and_ulta_suppression_behavior():
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    for sym in ["DUOL", "ULTA"]:
        comp = ledger["companies"][sym]
        cf_data = comp["fundamentals"]
        m_data = comp["derived_metrics"]
        facts = {
            k: FinancialFact(
                metric_key=v["metric_key"],
                value=v["value"],
                units=v["units"],
                status=MetricStatus(v.get("status", "REPORTED")),
            )
            for k, v in cf_data["facts"].items()
        }
        cf = CompanyFundamentals(
            symbol=cf_data["symbol"],
            company_name=cf_data["company_name"],
            cik=cf_data["cik"],
            primary_exchange=cf_data["primary_exchange"],
            sector_industry=cf_data["sector_industry"],
            business_model=cf_data["business_model"],
            fiscal_year_end=cf_data["fiscal_year_end"],
            latest_reported_period=cf_data["latest_reported_period"],
            latest_filing_date=cf_data["latest_filing_date"],
            facts=facts,
        )
        metrics = DerivedMetrics(
            symbol=m_data["symbol"],
            period_label=m_data["period_label"],
            revenue_ttm=m_data["revenue_ttm"],
            diluted_shares_weighted_avg=m_data["diluted_shares_weighted_avg"],
            net_debt=m_data["net_debt"],
            market_price=m_data["market_price"],
        )
        func = evaluate_deep_valuation_duol if sym == "DUOL" else evaluate_deep_valuation_ulta
        assessment = func(cf, metrics)
        assert assessment.valuation_status == "SUPPRESSED_UNAVAILABLE"
        assert assessment.is_certified is False
        assert "Without verified total financial debt, net debt bridge cannot be established; deep DCF valuation is suppressed." in assessment.suppression_reason


# 11. No silent certification of legacy records
def test_adversarial_11_no_silent_certification_of_legacy_records():
    # Direct instantiation without metadata validation must not default to is_certified=True
    raw_assessment = DeepValuationAssessment(
        symbol="FIX",
        framework_used="FCFF",
        current_price=1439.07,
        current_price_date="2026-10-16",
        current_market_cap=50660.0,
        diluted_shares=35.254,
        scenarios={},
        reverse_dcf_implied_5y_growth_pct=None,
        reverse_dcf_fixed_assumptions={},
        sensitivity_matrix={},
        thesis_invalidation_thresholds=[],
        next_quarter_evidence_milestones=[],
    )
    assert raw_assessment.is_certified is False


# 12. Numerical parity for compatible existing inputs
def test_adversarial_12_numerical_parity_for_compatible_existing_inputs():
    cf, metrics = _load_fix_fundamentals_and_metrics()
    assessment = evaluate_deep_valuation_fix(cf, metrics)
    assert assessment.valuation_status == "AVAILABLE"
    assert assessment.is_certified is True

    # Check exact outputs
    c_val = assessment.scenarios["Conservative"].per_share_value_estimate
    b_val = assessment.scenarios["Base"].per_share_value_estimate
    o_val = assessment.scenarios["Optimistic"].per_share_value_estimate
    rev_g = assessment.reverse_dcf_implied_5y_growth_pct

    assert abs(c_val - 527.35) < 0.10, f"Conservative {c_val} != 527.35"
    assert abs(b_val - 1068.94) < 0.10, f"Base {b_val} != 1068.94"
    assert abs(o_val - 1971.75) < 0.10, f"Optimistic {o_val} != 1971.75"
    assert abs(rev_g - 30.43) < 0.10, f"Reverse DCF {rev_g} != 30.43"
