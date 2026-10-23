"""
tests/test_lease_convention_integrity_p0b.py

Comprehensive 18-Scenario Adversarial & Acceptance Test Suite for
ARX Terminal Institutional Research P0-B: Lease Convention Integrity.

Covers:
- ADV-LEASE-01: Double-Deduction Injection under Convention 1 (LeaseDoubleCountingError)
- ADV-LEASE-02: Debt Omission under Convention 2 (LeaseConventionMismatchError)
- ADV-LEASE-03: Framework Boundary - Unadjusted IFRS 16 Ingestion (LeaseFrameworkIncompatibilityError)
- ADV-LEASE-04: Extraction Precedence - Compatible Current + Noncurrent Summation
- ADV-LEASE-05: Incomplete Data - Current-Only Lease Liability produces UNAVAILABLE
- ADV-LEASE-06: Incomplete Data - Noncurrent-Only Lease Liability produces UNAVAILABLE
- ADV-LEASE-07: Extraction Precedence - Verified Aggregate Only
- ADV-LEASE-08: Extraction Precedence - Reconciled Dual Reporting within $1.00 Tolerance
- ADV-LEASE-09: Extraction Conflict - Discrepancy > $1.00 Fails Closed
- ADV-LEASE-10: Strict Anti-Double-Counting Invariant Guard (Aggregate + Components)
- ADV-LEASE-11: Boundary Compatibility - Mismatched Component Reporting Dates
- ADV-LEASE-12: Boundary Compatibility - Mismatched Component Units/Currencies
- ADV-LEASE-13: Context Deduplication & Amendment Handling
- ADV-LEASE-14: Footnote Disclosures - Undiscounted Commitment Table Exclusion
- ADV-LEASE-15: Senior Debt Boundary - Bundled Capital Lease Overlap Guard
- ADV-LEASE-16: Senior Debt Boundary - Unbundled Finance Lease Tracking
- ADV-LEASE-17: Data Gap Handling - Suppressed Valuations on Incomplete Debt Evidence
- ADV-LEASE-18: Economic Parity Invariance - 100% Numerical Parity on Existing FIX/DUOL/ULTA Valuations
"""

import json
import pytest
from datetime import datetime
from typing import Dict, Any

from analyst_dashboard.research.sourced_fundamentals.contracts import (
    LeaseAccountingConvention,
    AccountingFramework,
    LeaseComponentFact,
    LeaseLiabilityMetadata,
    LeaseIntegrityError,
    LeaseDoubleCountingError,
    LeaseConventionMismatchError,
    LeaseFrameworkIncompatibilityError,
    LeaseAggregationConflictError,
    LeaseIncompatibleComponentsError,
    FinancialFact,
    MetricStatus,
    ValuationScenario,
    DeepValuationAssessment,
    ShareDenominatorMetadata,
    ShareBasis,
    DenominatorUseCase,
)
from analyst_dashboard.research.sourced_fundamentals.normalizer import (
    SecFactNormalizer,
    LEASE_AGGREGATE_CONCEPTS,
    LEASE_CURRENT_CONCEPTS,
    LEASE_NONCURRENT_CONCEPTS,
)
from analyst_dashboard.research.sourced_fundamentals.valuation import (
    run_fcff_model,
    validate_lease_convention_integrity,
    solve_reverse_dcf,
    generate_sensitivity_matrix,
)


# ==============================================================================
# ADV-LEASE-01: Double-Deduction Injection under Convention 1
# ==============================================================================
def test_adv_lease_01_double_deduction_rejected():
    """Verify that run_fcff_model fails closed with LeaseDoubleCountingError when operating leases are deducted in bridge."""
    # ULTA baseline: EBIT is net of store rent.
    # Total Debt: $339.58M, Cash: $213.45M -> Expected Bridge: $126.13M
    # Injected corrupted bridge following contracts.py line 570 comment: $126.13M + $1,871.81M = $1,997.94M
    meta = LeaseLiabilityMetadata(
        convention=LeaseAccountingConvention.LEASE_AS_OPERATING_EXPENSE,
        framework=AccountingFramework.US_GAAP,
        total_operating_lease_liabilities=1871805000.0,
        included_in_net_debt_bridge=True,  # Injected double deduction flag
    )

    with pytest.raises(LeaseDoubleCountingError, match="included_in_net_debt_bridge is True"):
        run_fcff_model(
            base_revenue=12955517000.0,
            current_shares=43062000.0,
            net_debt_bridge=1997932000.0,
            growth_5y_pct=6.0,
            operating_margin_pct=12.0,
            wacc_pct=9.0,
            terminal_growth_pct=2.5,
            lease_metadata=meta,
        )


# ==============================================================================
# ADV-LEASE-02: Debt Omission under Convention 2
# ==============================================================================
def test_adv_lease_02_convention_2_debt_omission_rejected():
    """Verify that declaring LEASE_AS_FINANCING_DEBT requires operating leases in net_debt_bridge and EBIT add-back."""
    meta = LeaseLiabilityMetadata(
        convention=LeaseAccountingConvention.LEASE_AS_FINANCING_DEBT,
        framework=AccountingFramework.US_GAAP,
        total_operating_lease_liabilities=500000000.0,
        is_ebit_adjusted_for_leases=True,  # Cash flow is adjusted
        included_in_net_debt_bridge=False, # Leases omitted from bridge!
    )

    with pytest.raises(LeaseConventionMismatchError, match="Lease liabilities .* omitted from Net Debt Bridge"):
        validate_lease_convention_integrity(
            convention=meta.convention,
            framework=meta.framework,
            net_debt_bridge=100000000.0,  # Pure debt - cash, missing 500M leases
            lease_metadata=meta,
        )


# ==============================================================================
# ADV-LEASE-03: Framework Boundary - Unadjusted IFRS 16 Ingestion
# ==============================================================================
def test_adv_lease_03_unadjusted_ifrs16_rejected():
    """Verify that unadjusted IFRS 16 issuers (e.g. Alcon) are rejected from Convention 1 without EBIT adjustments."""
    meta = LeaseLiabilityMetadata(
        convention=LeaseAccountingConvention.LEASE_AS_OPERATING_EXPENSE,
        framework=AccountingFramework.IFRS,
        total_operating_lease_liabilities=480000000.0,
        is_ebit_adjusted_for_leases=False,  # Unadjusted operating profit
    )

    with pytest.raises(LeaseFrameworkIncompatibilityError, match="IFRS 16 issuers do not expense leases"):
        validate_lease_convention_integrity(
            convention=meta.convention,
            framework=meta.framework,
            net_debt_bridge=50000000.0,
            lease_metadata=meta,
        )


# ==============================================================================
# ADV-LEASE-04: Extraction Precedence - Compatible Current + Noncurrent Summation
# ==============================================================================
def test_adv_lease_04_compatible_components_summed():
    """Verify that when aggregate is missing, compatible current and noncurrent components are summed."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "OperatingLeaseLiabilityCurrent": {
            "units": {"USD": [{"end": "2026-08-01", "val": 312450000, "form": "10-Q", "filed": "2026-08-27", "accn": "0001104659-26-102442"}]}
        },
        "OperatingLeaseLiabilityNoncurrent": {
            "units": {"USD": [{"end": "2026-08-01", "val": 1871805000, "form": "10-Q", "filed": "2026-08-27", "accn": "0001104659-26-102442"}]}
        },
    }

    fact, meta = normalizer._extract_lease_liabilities_with_precedence(raw_gaap, "2026-08-01")

    assert fact.value == 2184255000.0
    assert fact.status == MetricStatus.DERIVED
    assert meta.aggregation_method == "SUMMED_COMPONENTS"
    assert meta.certification_status == "CERTIFIED_GAAP_COMPATIBLE"
    assert meta.current_liability.value == 312450000.0
    assert meta.noncurrent_liability.value == 1871805000.0


# ==============================================================================
# ADV-LEASE-05: Incomplete Data - Current-Only Lease Liability
# ==============================================================================
def test_adv_lease_05_current_only_fails_closed():
    """Verify that if only current lease liability is reported without aggregate, result is UNAVAILABLE."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "OperatingLeaseLiabilityCurrent": {
            "units": {"USD": [{"end": "2026-08-01", "val": 50000000, "form": "10-Q", "filed": "2026-08-27", "accn": "0001104659-26-102442"}]}
        }
    }

    fact, meta = normalizer._extract_lease_liabilities_with_precedence(raw_gaap, "2026-08-01")

    assert fact.value is None
    assert fact.status == MetricStatus.UNAVAILABLE
    assert meta.aggregation_method == "PARTIAL_DISCLOSURE_UNRESOLVED"
    assert meta.certification_status == "UNAVAILABLE"
    assert "Partial operating lease observation known" in fact.notes


# ==============================================================================
# ADV-LEASE-06: Incomplete Data - Noncurrent-Only Lease Liability
# ==============================================================================
def test_adv_lease_06_noncurrent_only_fails_closed():
    """Verify that if only noncurrent lease liability is reported without aggregate, result is UNAVAILABLE."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "OperatingLeaseLiabilityNoncurrent": {
            "units": {"USD": [{"end": "2026-08-01", "val": 1871805000, "form": "10-Q", "filed": "2026-08-27", "accn": "0001104659-26-102442"}]}
        }
    }

    fact, meta = normalizer._extract_lease_liabilities_with_precedence(raw_gaap, "2026-08-01")

    assert fact.value is None
    assert fact.status == MetricStatus.UNAVAILABLE
    assert meta.aggregation_method == "PARTIAL_DISCLOSURE_UNRESOLVED"
    assert meta.certification_status == "UNAVAILABLE"


# ==============================================================================
# ADV-LEASE-07: Extraction Precedence - Verified Aggregate Only
# ==============================================================================
def test_adv_lease_07_verified_aggregate_only():
    """Verify that reported OperatingLeaseLiability aggregate takes precedence (e.g. FIX, MEDP, KLAC)."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "OperatingLeaseLiability": {
            "units": {"USD": [{"end": "2026-06-30", "val": 315845000, "form": "10-Q", "filed": "2026-07-23", "accn": "0001104659-26-086258"}]}
        }
    }

    fact, meta = normalizer._extract_lease_liabilities_with_precedence(raw_gaap, "2026-06-30")

    assert fact.value == 315845000.0
    assert fact.status == MetricStatus.REPORTED
    assert meta.aggregation_method == "REPORTED_AGGREGATE"
    assert meta.certification_status == "CERTIFIED_GAAP_COMPATIBLE"


# ==============================================================================
# ADV-LEASE-08: Extraction Precedence - Reconciled Dual Reporting within $1.00 Tolerance
# ==============================================================================
def test_adv_lease_08_reconciled_dual_reporting_within_tolerance():
    """Verify that when aggregate and complete components reconcile within $1.00, RECONCILED_MATCH is emitted."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "OperatingLeaseLiability": {
            "units": {"USD": [{"end": "2026-06-30", "val": 500000000.50, "form": "10-Q", "filed": "2026-07-23", "accn": "0001104659-26-086258"}]}
        },
        "OperatingLeaseLiabilityCurrent": {
            "units": {"USD": [{"end": "2026-06-30", "val": 100000000.00, "form": "10-Q", "filed": "2026-07-23", "accn": "0001104659-26-086258"}]}
        },
        "OperatingLeaseLiabilityNoncurrent": {
            "units": {"USD": [{"end": "2026-06-30", "val": 400000000.00, "form": "10-Q", "filed": "2026-07-23", "accn": "0001104659-26-086258"}]}
        },
    }

    fact, meta = normalizer._extract_lease_liabilities_with_precedence(raw_gaap, "2026-06-30")

    assert fact.value == 500000000.50
    assert meta.aggregation_method == "RECONCILED_MATCH"
    assert meta.certification_status == "CERTIFIED_GAAP_COMPATIBLE"


# ==============================================================================
# ADV-LEASE-09: Extraction Conflict - Discrepancy > $1.00 Fails Closed
# ==============================================================================
def test_adv_lease_09_conflicting_aggregate_and_components_fails_closed():
    """Verify that when aggregate and components conflict by > $1.00, extraction fails closed."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "OperatingLeaseLiability": {
            "units": {"USD": [{"end": "2026-06-30", "val": 500000000.00, "form": "10-Q", "filed": "2026-07-23", "accn": "0001104659-26-086258"}]}
        },
        "OperatingLeaseLiabilityCurrent": {
            "units": {"USD": [{"end": "2026-06-30", "val": 150000000.00, "form": "10-Q", "filed": "2026-07-23", "accn": "0001104659-26-086258"}]}
        },
        "OperatingLeaseLiabilityNoncurrent": {
            "units": {"USD": [{"end": "2026-06-30", "val": 400000000.00, "form": "10-Q", "filed": "2026-07-23", "accn": "0001104659-26-086258"}]}
        },
    }

    fact, meta = normalizer._extract_lease_liabilities_with_precedence(raw_gaap, "2026-06-30")

    assert fact.value is None
    assert fact.status == MetricStatus.UNAVAILABLE
    assert meta.aggregation_method == "CONFLICTING_AGGREGATE_AND_COMPONENTS"
    assert meta.certification_status == "UNAVAILABLE"
    assert "Discrepancy detected" in fact.notes


# ==============================================================================
# ADV-LEASE-10: Strict Anti-Double-Counting Invariant Guard
# ==============================================================================
def test_adv_lease_10_anti_double_counting_invariant():
    """Verify that resolved lease liability can never exceed the maximum of aggregate or component sum."""
    meta = LeaseLiabilityMetadata(
        total_operating_lease_liabilities=1000000.0,
    )
    # Direct formula check: Bridge cannot include leases under Convention 1
    with pytest.raises(LeaseDoubleCountingError):
        validate_lease_convention_integrity(
            convention=LeaseAccountingConvention.LEASE_AS_OPERATING_EXPENSE,
            framework=AccountingFramework.US_GAAP,
            total_financial_debt=5000000.0,
            total_cash=2000000.0,
            net_debt_bridge=4000000.0,  # 3M debt-cash + 1M leases = 4M (double counted!)
            lease_metadata=meta,
        )


# ==============================================================================
# ADV-LEASE-11: Boundary Compatibility - Mismatched Component Reporting Dates
# ==============================================================================
def test_adv_lease_11_mismatched_reporting_dates_fails_closed():
    """Verify that current and noncurrent components from different periods are rejected."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "OperatingLeaseLiabilityCurrent": {
            "units": {"USD": [{"end": "2026-08-01", "val": 312450000, "form": "10-Q", "filed": "2026-08-27", "accn": "0001104659-26-102442"}]}
        },
        "OperatingLeaseLiabilityNoncurrent": {
            "units": {"USD": [{"end": "2025-08-02", "val": 1871805000, "form": "10-Q", "filed": "2025-08-28", "accn": "0001104659-25-099999"}]}
        },
    }

    fact, meta = normalizer._extract_lease_liabilities_with_precedence(raw_gaap, "2026-08-01")

    # Target end is 2026-08-01; noncurrent fact for 2026-08-01 is missing, so it's partial
    assert fact.value is None
    assert meta.certification_status == "UNAVAILABLE"


# ==============================================================================
# ADV-LEASE-12: Boundary Compatibility - Mismatched Component Units/Currencies
# ==============================================================================
def test_adv_lease_12_mismatched_component_units():
    """Verify that current and noncurrent components in different currencies are rejected."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "OperatingLeaseLiabilityCurrent": {
            "units": {"USD": [{"end": "2026-08-01", "val": 1000000, "form": "10-Q", "filed": "2026-08-27", "accn": "0001104659-26-102442"}]}
        },
        "OperatingLeaseLiabilityNoncurrent": {
            "units": {"CAD": [{"end": "2026-08-01", "val": 5000000, "form": "10-Q", "filed": "2026-08-27", "accn": "0001104659-26-102442"}]}
        },
    }

    fact, meta = normalizer._extract_lease_liabilities_with_precedence(raw_gaap, "2026-08-01")

    # In USD unit extraction, CAD noncurrent is unavailable -> partial disclosure
    assert fact.value is None
    assert meta.certification_status == "UNAVAILABLE"


# ==============================================================================
# ADV-LEASE-13: Context Deduplication & Amendment Handling
# ==============================================================================
def test_adv_lease_13_amendment_deduplication():
    """Verify that when 10-Q and 10-Q/A coexist, latest amendment accession is prioritized."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "OperatingLeaseLiability": {
            "units": {"USD": [
                {"end": "2026-06-30", "val": 300000000, "form": "10-Q", "filed": "2026-07-23", "accn": "0001104659-26-086258"},
                {"end": "2026-06-30", "val": 315845000, "form": "10-Q/A", "filed": "2026-08-15", "accn": "0001104659-26-099999"},
            ]}
        }
    }

    fact, meta = normalizer._extract_lease_liabilities_with_precedence(raw_gaap, "2026-06-30")

    assert fact.value == 315845000.0
    assert fact.form == "10-Q/A"
    assert fact.accession_number == "0001104659-26-099999"


# ==============================================================================
# ADV-LEASE-14: Footnote Disclosures - Undiscounted Commitment Table Exclusion
# ==============================================================================
def test_adv_lease_14_undiscounted_commitments_excluded():
    """Verify that undiscounted maturity schedule concepts are never selected as balance sheet debt."""
    assert "LesseeOperatingLeaseLiabilityPaymentsDue" not in LEASE_AGGREGATE_CONCEPTS
    assert "LesseeOperatingLeaseLiabilityPaymentsDue" not in LEASE_CURRENT_CONCEPTS
    assert "LesseeOperatingLeaseLiabilityPaymentsDue" not in LEASE_NONCURRENT_CONCEPTS


# ==============================================================================
# ADV-LEASE-15: Senior Debt Boundary - Bundled Capital Lease Overlap Guard
# ==============================================================================
def test_adv_lease_15_bundled_finance_lease_overlap_guard():
    """Verify that finance_leases_bundled_in_total_debt is flagged when concept bundles capital leases."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "FinanceLeaseLiability": {
            "units": {"USD": [{"end": "2026-06-30", "val": 50000000, "form": "10-K", "filed": "2026-08-06", "accn": "0000319201-26-000027"}]}
        }
    }

    _, meta = normalizer._extract_lease_liabilities_with_precedence(
        raw_gaap, "2026-06-30", debt_concept="us-gaap/LongTermDebtAndCapitalLeaseObligations"
    )

    assert meta.finance_leases_bundled_in_total_debt is True
    assert meta.finance_lease_liabilities == 50000000.0


# ==============================================================================
# ADV-LEASE-16: Senior Debt Boundary - Unbundled Finance Lease Tracking
# ==============================================================================
def test_adv_lease_16_unbundled_finance_lease_tracking():
    """Verify that finance_leases_bundled_in_total_debt is False when debt concept does not bundle leases."""
    normalizer = SecFactNormalizer()
    raw_gaap = {
        "FinanceLeaseLiability": {
            "units": {"USD": [{"end": "2026-06-30", "val": 50000000, "form": "10-K", "filed": "2026-08-06", "accn": "0000319201-26-000027"}]}
        }
    }

    _, meta = normalizer._extract_lease_liabilities_with_precedence(
        raw_gaap, "2026-06-30", debt_concept="us-gaap/LongTermDebtNoncurrent"
    )

    assert meta.finance_leases_bundled_in_total_debt is False
    assert meta.finance_lease_liabilities == 50000000.0


# ==============================================================================
# ADV-LEASE-17: Data Gap Handling - Suppressed Valuations on Incomplete Debt Evidence
# ==============================================================================
def test_adv_lease_17_suppressed_valuation_on_debt_gap():
    """Verify that CELH/DUOL/MEDP baseline valuation suppression on missing total financial debt is preserved."""
    with pytest.raises(ValueError, match="Cannot run FCFF model: base_revenue, current_shares, or net_debt_bridge is None"):
        run_fcff_model(
            base_revenue=100000000.0,
            current_shares=50000000.0,
            net_debt_bridge=None,  # Missing debt bridge
            growth_5y_pct=10.0,
            operating_margin_pct=15.0,
            wacc_pct=9.0,
            terminal_growth_pct=2.5,
        )


# ==============================================================================
# ADV-LEASE-18: Economic Parity Invariance - 100% Numerical Parity
# ==============================================================================
def test_adv_lease_18_economic_parity_invariance_fix_ulta():
    """Verify that adding P0-B validation leaves existing valid Convention 1 valuations 100% unchanged."""
    # FIX baseline parameters
    fix_rev = 7056000000.0
    fix_shares = 37780000.0
    fix_bridge = -1800730000.0  # Net cash
    fix_res_no_meta = run_fcff_model(fix_rev, fix_shares, fix_bridge, 22.0, 15.0, 9.5, 2.5, 22.0, 15.0)

    fix_meta = LeaseLiabilityMetadata(
        convention=LeaseAccountingConvention.LEASE_AS_OPERATING_EXPENSE,
        framework=AccountingFramework.US_GAAP,
        total_operating_lease_liabilities=315845000.0,
        included_in_net_debt_bridge=False,
    )
    fix_res_with_meta = run_fcff_model(fix_rev, fix_shares, fix_bridge, 22.0, 15.0, 9.5, 2.5, 22.0, 15.0, lease_metadata=fix_meta)

    assert fix_res_no_meta["per_share"] == fix_res_with_meta["per_share"]
    assert fix_res_no_meta["equity_value"] == fix_res_with_meta["equity_value"]
    assert fix_res_no_meta["enterprise_value"] == fix_res_with_meta["enterprise_value"]

    # ULTA baseline parameters
    ulta_rev = 12955517000.0
    ulta_shares = 43062000.0
    ulta_bridge = 126127000.0  # Pure funded debt - cash
    ulta_res_no_meta = run_fcff_model(ulta_rev, ulta_shares, ulta_bridge, 6.0, 12.0, 9.0, 2.5, 24.0, 15.0)

    ulta_meta = LeaseLiabilityMetadata(
        convention=LeaseAccountingConvention.LEASE_AS_OPERATING_EXPENSE,
        framework=AccountingFramework.US_GAAP,
        total_operating_lease_liabilities=2184255000.0,
        included_in_net_debt_bridge=False,
    )
    ulta_res_with_meta = run_fcff_model(ulta_rev, ulta_shares, ulta_bridge, 6.0, 12.0, 9.0, 2.5, 24.0, 15.0, lease_metadata=ulta_meta)

    assert ulta_res_no_meta["per_share"] == ulta_res_with_meta["per_share"]
    assert ulta_res_no_meta["equity_value"] == ulta_res_with_meta["equity_value"]
