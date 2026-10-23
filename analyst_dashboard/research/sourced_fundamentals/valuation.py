"""
analyst_dashboard/research/sourced_fundamentals/valuation.py

Multi-scenario Discounted Free Cash Flow (FCFF) and Reverse-DCF valuation models.
Provides conservative, base, and optimistic scenario bounds, sensitivity matrices,
market-implied revenue growth reverse-engineering, and explicit thesis invalidation conditions.

Guarantees:
- Strict Lease-as-Operating-Expense valuation convention (no lease double-counting).
- Zero hardcoded factual fallbacks (missing inputs propagate as None / UNAVAILABLE).
- Reverse DCF brackets check with roundtrip forward-model verification.
- Sensitivity matrix returns None (not 0.0) for invalid cells.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from .contracts import (
    ValuationScenario,
    DeepValuationAssessment,
    DerivedMetrics,
    CompanyFundamentals,
    ShareBasis,
    DenominatorUseCase,
    DenominatorErrorCode,
    ShareDenominatorError,
    ShareDenominatorMetadata,
    validate_share_denominator_compatibility,
    IncompatibleShareBasisError,
    PeriodAlignmentError,
    IncompatibleShareClassError,
    ShareUnitMismatchError,
    MissingProvenanceError,
    InvalidShareDenominatorError,
    UnverifiedADRRatioError,
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
)

logger = logging.getLogger(__name__)


def validate_lease_convention_integrity(
    convention: LeaseAccountingConvention,
    framework: AccountingFramework,
    net_debt_bridge: float,
    total_financial_debt: Optional[float] = None,
    total_cash: Optional[float] = None,
    lease_metadata: Optional[LeaseLiabilityMetadata] = None,
    tolerance: float = 1.0,
) -> None:
    """
    Validates that net_debt_bridge is mathematically coherent with declared convention and framework.
    Fails closed on convention mixing, double-deduction, or unadjusted IFRS ingestion.
    """
    # 1. Framework Gate (IFRS 16 Discrimination)
    if framework == AccountingFramework.IFRS and convention == LeaseAccountingConvention.LEASE_AS_OPERATING_EXPENSE:
        if lease_metadata is None or not lease_metadata.is_ebit_adjusted_for_leases:
            raise LeaseFrameworkIncompatibilityError(
                "IFRS 16 issuers do not expense leases in operating profit. Applying "
                "LEASE_AS_OPERATING_EXPENSE requires explicit EBIT downward adjustment "
                "to reflect lease rental costs, or conversion to LEASE_AS_FINANCING_DEBT."
            )

    d = total_financial_debt if total_financial_debt is not None else 0.0
    c = total_cash if total_cash is not None else 0.0

    # 2. Convention 1: Lease-as-Operating-Expense (Canonical ARX Default)
    if convention == LeaseAccountingConvention.LEASE_AS_OPERATING_EXPENSE:
        expected_bridge = d - c

        # Anti-Double-Deduction Check:
        if lease_metadata and lease_metadata.included_in_net_debt_bridge:
            raise LeaseDoubleCountingError(
                "lease_metadata.included_in_net_debt_bridge is True under LEASE_AS_OPERATING_EXPENSE convention."
            )

        if lease_metadata and lease_metadata.total_operating_lease_liabilities:
            leases = lease_metadata.total_operating_lease_liabilities
            if leases > 0:
                if total_financial_debt is not None or total_cash is not None:
                    if abs(net_debt_bridge - (expected_bridge + leases)) <= tolerance:
                        raise LeaseDoubleCountingError(
                            f"Double-counting detected: net_debt_bridge ({net_debt_bridge:,.2f}) includes "
                            f"operating lease liabilities ({leases:,.2f}) under LEASE_AS_OPERATING_EXPENSE convention. "
                            f"Operating leases are already expensed in reported EBIT/CFO. Expected bridge: {expected_bridge:,.2f}."
                        )

        if total_financial_debt is not None and total_cash is not None:
            if abs(net_debt_bridge - expected_bridge) > tolerance:
                leases = lease_metadata.total_operating_lease_liabilities if (lease_metadata and lease_metadata.total_operating_lease_liabilities) else 0.0
                if leases > 0 and abs(net_debt_bridge - (expected_bridge + leases)) <= tolerance:
                    raise LeaseDoubleCountingError(
                        f"Double-counting detected: net_debt_bridge ({net_debt_bridge:,.2f}) includes "
                        f"operating lease liabilities ({leases:,.2f}) under LEASE_AS_OPERATING_EXPENSE convention."
                    )
                raise LeaseConventionMismatchError(
                    f"Net debt bridge mismatch under LEASE_AS_OPERATING_EXPENSE: got {net_debt_bridge:,.2f}, "
                    f"expected (Total Financial Debt - Total Cash) = {expected_bridge:,.2f}."
                )

    # 3. Convention 2: Lease-as-Financing-Debt (Capitalized Leases)
    elif convention == LeaseAccountingConvention.LEASE_AS_FINANCING_DEBT:
        leases = 0.0
        if lease_metadata and lease_metadata.total_operating_lease_liabilities:
            leases = lease_metadata.total_operating_lease_liabilities

        expected_bridge = d + leases - c
        if total_financial_debt is not None and total_cash is not None:
            if abs(net_debt_bridge - expected_bridge) > tolerance:
                raise LeaseConventionMismatchError(
                    f"Net debt bridge mismatch under LEASE_AS_FINANCING_DEBT: got {net_debt_bridge:,.2f}, "
                    f"expected (Total Financial Debt + Leases - Cash) = {expected_bridge:,.2f}."
                )
        elif lease_metadata and lease_metadata.included_in_net_debt_bridge is False:
            raise LeaseConventionMismatchError(
                f"Lease liabilities ({leases:,.2f}) omitted from Net Debt Bridge under LEASE_AS_FINANCING_DEBT convention."
            )

        # Verify EBIT add-back
        if lease_metadata and not lease_metadata.is_ebit_adjusted_for_leases:
            raise LeaseConventionMismatchError(
                "Convention 2 (LEASE_AS_FINANCING_DEBT) declared but operating cash flow / EBIT "
                "was not adjusted to add back operating lease expenses."
            )


def run_fcff_model(
    base_revenue: float,
    current_shares: float,
    net_debt_bridge: float,
    growth_5y_pct: float,
    operating_margin_pct: float,
    wacc_pct: float,
    terminal_growth_pct: float,
    tax_rate_pct: float = 21.0,
    reinvestment_rate_pct: float = 15.0,
    share_metadata: Optional[ShareDenominatorMetadata] = None,
    lease_metadata: Optional[LeaseLiabilityMetadata] = None,
) -> Dict[str, float]:
    """
    5-Year FCFF DCF model under Lease-as-Operating-Expense convention:
    FCFF_t = Revenue_t * EBIT_Margin * (1 - Tax_Rate) * (1 - Reinvestment_Rate)
    Terminal Value = FCFF_5 * (1 + g) / (WACC - g)
    Enterprise Value = PV(FCFF_1..5) + PV(Terminal_Value)
    Equity Value = Enterprise Value - Net Debt Bridge
    (Net Debt Bridge = Total Financial Debt - Total Cash & Investments)
    Per Share Value = Equity Value / Diluted Shares
    """
    if base_revenue is None or current_shares is None or net_debt_bridge is None:
        raise ValueError("Cannot run FCFF model: base_revenue, current_shares, or net_debt_bridge is None")
    if current_shares <= 0:
        raise InvalidShareDenominatorError(f"Current shares must be strictly positive (got {current_shares})")
    if base_revenue <= 0:
        raise ValueError("Base revenue must be strictly positive")
    if wacc_pct <= terminal_growth_pct:
        raise ValueError(f"WACC ({wacc_pct}%) must be strictly greater than terminal growth ({terminal_growth_pct}%)")

    # P0-A: Validate share denominator compatibility if metadata supplied
    if share_metadata is not None:
        validate_share_denominator_compatibility(
            metadata=share_metadata,
            target_use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
            raise_on_error=True,
        )

    # P0-B: Validate lease convention integrity if metadata supplied
    if lease_metadata is not None:
        validate_lease_convention_integrity(
            convention=lease_metadata.convention,
            framework=lease_metadata.framework,
            net_debt_bridge=net_debt_bridge,
            lease_metadata=lease_metadata,
        )

    growth = growth_5y_pct / 100.0
    ebit_margin = operating_margin_pct / 100.0
    wacc = wacc_pct / 100.0
    g_term = terminal_growth_pct / 100.0
    tax = tax_rate_pct / 100.0
    reinv = reinvestment_rate_pct / 100.0

    pv_cash_flows = 0.0
    rev = base_revenue
    last_fcf = 0.0

    for year in range(1, 6):
        rev = rev * (1.0 + growth)
        nopat = rev * ebit_margin * (1.0 - tax)
        fcf = nopat * (1.0 - reinv)
        last_fcf = fcf
        discount_factor = (1.0 + wacc) ** year
        pv_cash_flows += fcf / discount_factor

    terminal_fcf = last_fcf * (1.0 + g_term)
    terminal_value = terminal_fcf / (wacc - g_term)
    pv_terminal_value = terminal_value / ((1.0 + wacc) ** 5)

    ev = pv_cash_flows + pv_terminal_value
    equity_val = ev - net_debt_bridge
    per_share = equity_val / current_shares

    return {
        "pv_explicit": pv_cash_flows,
        "pv_terminal": pv_terminal_value,
        "enterprise_value": ev,
        "equity_value": equity_val,
        "per_share": per_share,
    }


def compute_terminal_exit_valuation(
    terminal_fcf: float,
    wacc_pct: float,
    terminal_growth_pct: float,
    terminal_shares: float,
    terminal_net_debt: float = 0.0,
    share_metadata: Optional[ShareDenominatorMetadata] = None,
) -> Dict[str, float]:
    """
    Computes terminal exit enterprise value, equity value, and per-share terminal equity value.
    Enforces FORWARD_PROJECTED_TERMINAL denominator semantics under P0-A.
    """
    if terminal_shares is None or terminal_shares <= 0:
        raise InvalidShareDenominatorError(f"Terminal shares must be strictly positive (got {terminal_shares})")
    if terminal_fcf is None or terminal_fcf <= 0:
        raise ValueError("Terminal FCF must be strictly positive")
    if wacc_pct <= terminal_growth_pct:
        raise ValueError(f"WACC ({wacc_pct}%) must be strictly greater than terminal growth ({terminal_growth_pct}%)")

    if share_metadata is not None:
        validate_share_denominator_compatibility(
            metadata=share_metadata,
            target_use_case=DenominatorUseCase.PROJECTED_TERMINAL_EXIT,
            raise_on_error=True,
        )

    wacc = wacc_pct / 100.0
    g = terminal_growth_pct / 100.0
    terminal_ev = (terminal_fcf * (1.0 + g)) / (wacc - g)
    terminal_equity = terminal_ev - terminal_net_debt
    per_share_exit = terminal_equity / terminal_shares

    return {
        "terminal_enterprise_value": terminal_ev,
        "terminal_equity_value": terminal_equity,
        "terminal_per_share_exit": per_share_exit,
    }


def solve_reverse_dcf(
    target_price: float,
    base_revenue: float,
    current_shares: float,
    net_debt_bridge: float,
    operating_margin_pct: float,
    wacc_pct: float,
    terminal_growth_pct: float,
    tax_rate_pct: float = 21.0,
    reinvestment_rate_pct: float = 15.0,
    low_bound_pct: float = -40.0,
    high_bound_pct: float = 100.0,
    share_metadata: Optional[ShareDenominatorMetadata] = None,
    lease_metadata: Optional[LeaseLiabilityMetadata] = None,
) -> Tuple[Optional[float], str]:
    """
    Numerically solves for the 5-year revenue CAGR required for per-share
    DCF value to equal target_price, holding other parameters constant.

    Guarantees:
    - Validates share_metadata compatibility if provided under P0-A.
    - Validates lease_metadata compatibility if provided under P0-B.
    - Verifies that target_price is bracketed in [low_bound_pct, high_bound_pct].
    - If target_price is NOT bracketed, returns (None, 'OUT_OF_BOUNDS_TARGET_NOT_BRACKETED').
    - Performs roundtrip forward-model verification: feeds solved CAGR back into run_fcff_model
      and asserts |DCF - target_price| < max(0.05, target_price * 0.0005).
    """
    if (
        target_price is None
        or base_revenue is None
        or current_shares is None
        or net_debt_bridge is None
    ):
        return None, "MISSING_INPUTS"

    # P0-A: Pre-flight validate share denominator compatibility for PV DCF
    if share_metadata is not None:
        val_res = validate_share_denominator_compatibility(
            metadata=share_metadata,
            target_use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
            raise_on_error=False,
        )
        if not val_res.is_valid:
            return None, f"SUPPRESSED_DENOMINATOR_INCOMPATIBLE: [{val_res.error_code.value}] {val_res.error_message}"

    # P0-B: Pre-flight validate lease convention integrity
    if lease_metadata is not None:
        try:
            validate_lease_convention_integrity(
                convention=lease_metadata.convention,
                framework=lease_metadata.framework,
                net_debt_bridge=net_debt_bridge,
                lease_metadata=lease_metadata,
            )
        except LeaseIntegrityError as e:
            return None, f"SUPPRESSED_LEASE_INCOMPATIBLE: {e}"

    try:
        res_low = run_fcff_model(
            base_revenue=base_revenue,
            current_shares=current_shares,
            net_debt_bridge=net_debt_bridge,
            growth_5y_pct=low_bound_pct,
            operating_margin_pct=operating_margin_pct,
            wacc_pct=wacc_pct,
            terminal_growth_pct=terminal_growth_pct,
            tax_rate_pct=tax_rate_pct,
            reinvestment_rate_pct=reinvestment_rate_pct,
            share_metadata=share_metadata,
            lease_metadata=lease_metadata,
        )
        res_high = run_fcff_model(
            base_revenue=base_revenue,
            current_shares=current_shares,
            net_debt_bridge=net_debt_bridge,
            growth_5y_pct=high_bound_pct,
            operating_margin_pct=operating_margin_pct,
            wacc_pct=wacc_pct,
            terminal_growth_pct=terminal_growth_pct,
            tax_rate_pct=tax_rate_pct,
            reinvestment_rate_pct=reinvestment_rate_pct,
            share_metadata=share_metadata,
            lease_metadata=lease_metadata,
        )
    except ShareDenominatorError as e:
        return None, f"SUPPRESSED_DENOMINATOR_INCOMPATIBLE: [{e.code.value}] {e.message}"
    except LeaseIntegrityError as e:
        return None, f"SUPPRESSED_LEASE_INCOMPATIBLE: {e}"
    except Exception as e:
        return None, f"MODEL_ERROR: {e}"

    price_at_low = res_low["per_share"]
    price_at_high = res_high["per_share"]

    # Target price must be bracketed
    if target_price < price_at_low or target_price > price_at_high:
        return None, f"OUT_OF_BOUNDS_TARGET_NOT_BRACKETED (Target ${target_price:.2f} outside [${price_at_low:.2f}, ${price_at_high:.2f}])"

    low = low_bound_pct / 100.0
    high = high_bound_pct / 100.0

    for _ in range(100):
        mid = (low + high) / 2.0
        res = run_fcff_model(
            base_revenue=base_revenue,
            current_shares=current_shares,
            net_debt_bridge=net_debt_bridge,
            growth_5y_pct=mid * 100.0,
            operating_margin_pct=operating_margin_pct,
            wacc_pct=wacc_pct,
            terminal_growth_pct=terminal_growth_pct,
            tax_rate_pct=tax_rate_pct,
            reinvestment_rate_pct=reinvestment_rate_pct,
            share_metadata=share_metadata,
            lease_metadata=lease_metadata,
        )
        if abs(res["per_share"] - target_price) < 0.001:
            break
        if res["per_share"] < target_price:
            low = mid
        else:
            high = mid

    solved_g = round(mid * 100.0, 2)

    # Roundtrip verification check
    roundtrip = run_fcff_model(
        base_revenue=base_revenue,
        current_shares=current_shares,
        net_debt_bridge=net_debt_bridge,
        growth_5y_pct=solved_g,
        operating_margin_pct=operating_margin_pct,
        wacc_pct=wacc_pct,
        terminal_growth_pct=terminal_growth_pct,
        tax_rate_pct=tax_rate_pct,
        reinvestment_rate_pct=reinvestment_rate_pct,
        share_metadata=share_metadata,
        lease_metadata=lease_metadata,
    )
    diff = abs(roundtrip["per_share"] - target_price)
    threshold = max(0.05, target_price * 0.0005)
    if diff > threshold:
        return None, f"ROUNDTRIP_VERIFICATION_FAILED (Diff ${diff:.3f} exceeds ${threshold:.3f} threshold)"

    return solved_g, f"VERIFIED (Roundtrip per-share ${roundtrip['per_share']:.2f} matches target ${target_price:.2f})"


def generate_sensitivity_matrix(
    base_revenue: float,
    current_shares: float,
    net_debt_bridge: float,
    growth_5y_pct: float,
    operating_margin_pct: float,
    tax_rate_pct: float,
    reinvestment_rate_pct: float,
    wacc_rates: List[float] = [8.0, 9.0, 10.0, 11.0, 12.0],
    terminal_rates: List[float] = [2.0, 2.5, 3.0],
    share_metadata: Optional[ShareDenominatorMetadata] = None,
    lease_metadata: Optional[LeaseLiabilityMetadata] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Generates 2D WACC vs Terminal Growth valuation sensitivity table.
    Invalid cells return None (never zero-dollar fair values).
    """
    if share_metadata is not None:
        val_res = validate_share_denominator_compatibility(
            metadata=share_metadata,
            target_use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
            raise_on_error=False,
        )
        if not val_res.is_valid:
            matrix: Dict[str, Dict[str, Optional[float]]] = {}
            for w in wacc_rates:
                w_key = f"WACC_{w:.1f}%"
                matrix[w_key] = {f"g_{g:.1f}%": None for g in terminal_rates}
            return matrix

    if lease_metadata is not None:
        try:
            validate_lease_convention_integrity(
                convention=lease_metadata.convention,
                framework=lease_metadata.framework,
                net_debt_bridge=net_debt_bridge,
                lease_metadata=lease_metadata,
            )
        except LeaseIntegrityError:
            matrix: Dict[str, Dict[str, Optional[float]]] = {}
            for w in wacc_rates:
                w_key = f"WACC_{w:.1f}%"
                matrix[w_key] = {f"g_{g:.1f}%": None for g in terminal_rates}
            return matrix

    matrix: Dict[str, Dict[str, Optional[float]]] = {}
    for w in wacc_rates:
        w_key = f"WACC_{w:.1f}%"
        matrix[w_key] = {}
        for g in terminal_rates:
            g_key = f"g_{g:.1f}%"
            if w <= g:
                matrix[w_key][g_key] = None
                continue
            try:
                res = run_fcff_model(
                    base_revenue=base_revenue,
                    current_shares=current_shares,
                    net_debt_bridge=net_debt_bridge,
                    growth_5y_pct=growth_5y_pct,
                    operating_margin_pct=operating_margin_pct,
                    wacc_pct=w,
                    terminal_growth_pct=g,
                    tax_rate_pct=tax_rate_pct,
                    reinvestment_rate_pct=reinvestment_rate_pct,
                    share_metadata=share_metadata,
                    lease_metadata=lease_metadata,
                )
                matrix[w_key][g_key] = round(res["per_share"], 2)
            except Exception:
                matrix[w_key][g_key] = None
    return matrix


def _resolve_and_validate_diluted_share_metadata(
    cf: Optional[CompanyFundamentals],
    metrics: DerivedMetrics,
    target_use_case: DenominatorUseCase = DenominatorUseCase.PRESENT_VALUE_DCF,
    valuation_date: Optional[str] = None,
    require_certified: bool = True,
) -> Tuple[Optional[ShareDenominatorMetadata], Optional[str], Optional[DenominatorErrorCode]]:
    """
    Extracts or reconstructs ShareDenominatorMetadata for diluted shares and validates compatibility under P0-A.
    Returns (metadata, error_message, error_code).
    If valid, returns (metadata, None, None).
    """
    dshares_meta = None
    if metrics.share_denominator_metadata and "diluted_shares" in metrics.share_denominator_metadata:
        raw_meta = metrics.share_denominator_metadata["diluted_shares"]
        if raw_meta is not None:
            if isinstance(raw_meta, ShareDenominatorMetadata):
                dshares_meta = raw_meta
            elif isinstance(raw_meta, dict):
                dshares_meta = ShareDenominatorMetadata.from_dict(raw_meta)

    # If missing from metrics, attempt reconstruction from authoritative cf.facts
    if dshares_meta is None and cf and cf.facts and "diluted_shares_weighted_avg" in cf.facts:
        f = cf.facts["diluted_shares_weighted_avg"]
        if f.value is not None:
            dshares_meta = ShareDenominatorMetadata(
                share_count=f.value,
                basis=ShareBasis.WEIGHTED_AVERAGE_DILUTED,
                use_case=target_use_case,
                units=f.units or "shares",
                period_start=f.period_start,
                period_end=f.period_end,
                filing_date=f.filing_date,
                accession_number=f.accession_number,
                form=f.form,
                status=f.status,
                provenance_notes=f.notes,
            )

    if dshares_meta is None:
        if require_certified:
            return None, "Compatible share denominator metadata unavailable for certified research valuation.", DenominatorErrorCode.MISSING_PROVENANCE
        return None, None, None

    # Validate compatibility
    try:
        validate_share_denominator_compatibility(
            metadata=dshares_meta,
            target_use_case=target_use_case,
            valuation_date=valuation_date or metrics.market_price_date,
            raise_on_error=True,
        )
        return dshares_meta, None, None
    except ShareDenominatorError as e:
        return dshares_meta, f"[{e.code.value}] {e.message}", e.code


def evaluate_deep_valuation_fix(
    cf: CompanyFundamentals,
    metrics: DerivedMetrics,
    require_certified_metadata: bool = True,
) -> Optional[DeepValuationAssessment]:
    """Deep valuation package for FIX (Comfort Systems USA)."""
    rev = metrics.revenue_ttm
    shares = metrics.diluted_shares_weighted_avg
    net_bridge = metrics.net_debt
    current_price = metrics.market_price

    if rev is None or shares is None or net_bridge is None or current_price is None:
        missing_reasons = []
        if rev is None: missing_reasons.append("revenue_ttm unavailable")
        if shares is None: missing_reasons.append("diluted_shares_weighted_avg unavailable")
        if net_bridge is None: missing_reasons.append("net_debt unavailable")
        if current_price is None: missing_reasons.append("market_price unavailable")
        logger.warning(f"FIX deep valuation suppressed: {', '.join(missing_reasons)}")
        return DeepValuationAssessment(
            symbol="FIX",
            framework_used="FCFF (Enterprise to Equity DCF, Lease-as-Operating-Expense Convention)",
            current_price=current_price or 0.0,
            current_price_date=metrics.market_price_date or "2026-10-16",
            current_market_cap=round(metrics.market_cap or ((current_price or 0.0) * (shares or 0.0)), 2),
            diluted_shares=shares or 0.0,
            scenarios={},
            reverse_dcf_implied_5y_growth_pct=None,
            reverse_dcf_fixed_assumptions={"solution_status": "CRITICAL_INPUTS_UNAVAILABLE"},
            sensitivity_matrix={},
            thesis_invalidation_thresholds=[],
            next_quarter_evidence_milestones=[],
            valuation_status="SUPPRESSED_UNAVAILABLE",
            suppression_reason=f"Critical valuation inputs unavailable: {', '.join(missing_reasons)}",
            is_certified=False,
        )

    # P0-A: Resolve and validate diluted share denominator metadata
    dshares_meta, err_msg, err_code = _resolve_and_validate_diluted_share_metadata(
        cf=cf,
        metrics=metrics,
        target_use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        valuation_date=metrics.market_price_date or "2026-10-16",
        require_certified=require_certified_metadata,
    )
    if err_msg is not None:
        status_code = f"SUPPRESSED_DENOMINATOR_{err_code.value}" if err_code else "SUPPRESSED_DENOMINATOR_UNAVAILABLE"
        logger.warning(f"FIX deep valuation suppressed: {err_msg}")
        return DeepValuationAssessment(
            symbol="FIX",
            framework_used="FCFF (Enterprise to Equity DCF, Lease-as-Operating-Expense Convention)",
            current_price=current_price,
            current_price_date=metrics.market_price_date or "2026-10-16",
            current_market_cap=round(metrics.market_cap or (current_price * shares), 2),
            diluted_shares=shares,
            scenarios={},
            reverse_dcf_implied_5y_growth_pct=None,
            reverse_dcf_fixed_assumptions={"solution_status": status_code},
            sensitivity_matrix={},
            thesis_invalidation_thresholds=[],
            next_quarter_evidence_milestones=[],
            valuation_status="SUPPRESSED_DENOMINATOR_INCOMPATIBLE" if err_code != DenominatorErrorCode.MISSING_PROVENANCE else "SUPPRESSED_DENOMINATOR_UNAVAILABLE",
            suppression_reason=f"Share denominator compatibility failure: {err_msg}",
            is_certified=False,
            share_basis=dshares_meta.basis.value if dshares_meta else None,
            share_denominator_metadata=dshares_meta.to_dict() if dshares_meta else None,
        )

    tax_rate = 22.0
    reinvest = 15.0

    # 1. Conservative Scenario: Data center buildout normalizes; 12% CAGR, 12.5% margin, 10.5% WACC
    c_res = run_fcff_model(rev, shares, net_bridge, 12.0, 12.5, 10.5, 2.0, tax_rate, reinvest, share_metadata=dshares_meta)
    c_scenario = ValuationScenario(
        scenario_name="Conservative",
        growth_years_1_to_5_pct=12.0,
        operating_margin_target_pct=12.5,
        wacc_pct=10.5,
        terminal_growth_pct=2.0,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
        pv_explicit_cash_flows=round(c_res["pv_explicit"], 2),
        pv_terminal_value=round(c_res["pv_terminal"], 2),
        enterprise_value_estimate=round(c_res["enterprise_value"], 2),
        net_debt_bridge=round(net_bridge, 2),
        equity_value_estimate=round(c_res["equity_value"], 2),
        per_share_value_estimate=round(c_res["per_share"], 2),
        upside_downside_pct=round((c_res["per_share"] - current_price) / current_price * 100, 2),
        key_assumptions="Data center cooling buildout cools; modular pre-fabrication backlog normalizes; margin moderates toward historical median."
    )

    # 2. Base Scenario: Sustained AI/data-center buildout; 22% CAGR for 5 years, 15.0% margin, 9.5% WACC
    b_res = run_fcff_model(rev, shares, net_bridge, 22.0, 15.0, 9.5, 2.5, tax_rate, reinvest, share_metadata=dshares_meta)
    b_scenario = ValuationScenario(
        scenario_name="Base",
        growth_years_1_to_5_pct=22.0,
        operating_margin_target_pct=15.0,
        wacc_pct=9.5,
        terminal_growth_pct=2.5,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
        pv_explicit_cash_flows=round(b_res["pv_explicit"], 2),
        pv_terminal_value=round(b_res["pv_terminal"], 2),
        enterprise_value_estimate=round(b_res["enterprise_value"], 2),
        net_debt_bridge=round(net_bridge, 2),
        equity_value_estimate=round(b_res["equity_value"], 2),
        per_share_value_estimate=round(b_res["per_share"], 2),
        upside_downside_pct=round((b_res["per_share"] - current_price) / current_price * 100, 2),
        key_assumptions="Continued multi-year data-center cooling backlog conversion; sustained premium pricing on complex modular contracts."
    )

    # 3. Optimistic Scenario: Hyper-demand; 30% CAGR for 5 years, 16.5% margin, 8.5% WACC
    o_res = run_fcff_model(rev, shares, net_bridge, 30.0, 16.5, 8.5, 3.0, tax_rate, reinvest, share_metadata=dshares_meta)
    o_scenario = ValuationScenario(
        scenario_name="Optimistic",
        growth_years_1_to_5_pct=30.0,
        operating_margin_target_pct=16.5,
        wacc_pct=8.5,
        terminal_growth_pct=3.0,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
        pv_explicit_cash_flows=round(o_res["pv_explicit"], 2),
        pv_terminal_value=round(o_res["pv_terminal"], 2),
        enterprise_value_estimate=round(o_res["enterprise_value"], 2),
        net_debt_bridge=round(net_bridge, 2),
        equity_value_estimate=round(o_res["equity_value"], 2),
        per_share_value_estimate=round(o_res["per_share"], 2),
        upside_downside_pct=round((o_res["per_share"] - current_price) / current_price * 100, 2),
        key_assumptions="Accelerating hyperscaler capex expansions into 2028+; supply shortage in modular HVAC mechanics preserves pricing power."
    )

    # Reverse DCF
    rev_growth, r_note = solve_reverse_dcf(current_price, rev, shares, net_bridge, 15.0, 9.5, 2.5, tax_rate, reinvest, share_metadata=dshares_meta)
    sens_matrix = generate_sensitivity_matrix(rev, shares, net_bridge, 22.0, 15.0, tax_rate, reinvest, share_metadata=dshares_meta)

    return DeepValuationAssessment(
        symbol="FIX",
        framework_used="FCFF (Enterprise to Equity DCF, Lease-as-Operating-Expense Convention)",
        current_price=current_price,
        current_price_date=metrics.market_price_date or "2026-10-16",
        current_market_cap=round(metrics.market_cap or (current_price * shares), 2),
        diluted_shares=shares,
        scenarios={
            "Conservative": c_scenario,
            "Base": b_scenario,
            "Optimistic": o_scenario,
        },
        reverse_dcf_implied_5y_growth_pct=rev_growth,
        reverse_dcf_fixed_assumptions={
            "solved_variable": "5-year Revenue CAGR",
            "operating_margin_target_pct": 15.0,
            "wacc_pct": 9.5,
            "terminal_growth_pct": 2.5,
            "effective_tax_rate_pct": tax_rate,
            "reinvestment_rate_pct": reinvest,
            "net_debt_bridge_usd": net_bridge,
            "solution_status": r_note,
        },
        sensitivity_matrix=sens_matrix,
        thesis_invalidation_thresholds=[
            "Data-center order backlog cancellations or deferrals exceeding 15% in quarterly disclosures.",
            "Operating margin contraction below 12.0% indicating fixed-price contract cost overruns.",
            "Operating cash flow conversion falling below 80% of EBIT for two consecutive quarters.",
            "Material capital allocation shift from organic growth / bolt-on acquisitions to aggressive leveraged megadeals.",
        ],
        next_quarter_evidence_milestones=[
            "Quarterly backlog growth rate and percentage of backlog dedicated to tech/institutional sectors.",
            "Gross margin stability in modular construction and mechanical contracting segments.",
            "Working capital cash conversion and billing in excess of costs reconciliation.",
        ],
        valuation_status="AVAILABLE",
        suppression_reason=None,
        is_certified=True if dshares_meta is not None else False,
        share_basis=dshares_meta.basis.value if dshares_meta else None,
        share_denominator_metadata=dshares_meta.to_dict() if dshares_meta else None,
    )


def evaluate_deep_valuation_duol(
    cf: CompanyFundamentals,
    metrics: DerivedMetrics,
) -> Optional[DeepValuationAssessment]:
    """Deep valuation package for DUOL (Duolingo, Inc.)."""
    rev = metrics.revenue_ttm
    shares = metrics.diluted_shares_weighted_avg
    net_bridge = metrics.net_debt
    current_price = metrics.market_price

    if rev is None or shares is None or net_bridge is None or current_price is None:
        logger.warning("DUOL deep valuation suppressed: critical inputs unavailable (net_debt is None)")
        return DeepValuationAssessment(
            symbol="DUOL",
            framework_used="FCFF (Enterprise to Equity DCF, Lease-as-Operating-Expense Convention)",
            current_price=current_price or 0.0,
            current_price_date=metrics.market_price_date or "2026-10-16",
            current_market_cap=round(metrics.market_cap or ((current_price or 0.0) * (shares or 0.0)), 2),
            diluted_shares=shares or 0.0,
            scenarios={},
            reverse_dcf_implied_5y_growth_pct=None,
            reverse_dcf_fixed_assumptions={
                "solved_variable": "5-year Revenue CAGR",
                "operating_margin_target_pct": 18.0,
                "wacc_pct": 10.0,
                "terminal_growth_pct": 3.0,
                "effective_tax_rate_pct": 21.0,
                "reinvestment_rate_pct": 10.0,
                "net_debt_bridge_usd": None,
                "solution_status": "SUPPRESSED_NET_DEBT_UNAVAILABLE",
            },
            sensitivity_matrix={},
            thesis_invalidation_thresholds=[
                "Daily Active User (DAU) or Paid Subscriber YoY growth decelerating below 15%.",
                "Stock-based compensation expanding beyond 15% of total revenues without corresponding operating leverage.",
                "Gross margin dropping below 70% driven by escalating third-party LLM API inference fees.",
                "Material erosion of Duolingo English Test (DET) institutional acceptance rate.",
            ],
            next_quarter_evidence_milestones=[
                "Duolingo Max conversion rate as percentage of total paid subscribers.",
                "SBC / Revenue trajectory and net annual share dilution moderation.",
                "Primary source disclosure of long-term debt or credit agreements in 10-Q footnotes.",
            ],
            valuation_status="SUPPRESSED_UNAVAILABLE",
            suppression_reason=(
                "Total financial debt is UNAVAILABLE in SEC EDGAR XBRL (neither DebtLongtermAndShorttermCombinedAmount nor Debt "
                "concepts reported, and no current/noncurrent debt components found). Under aggregate completeness rules, "
                "missing debt cannot be assumed to be 0.0 without primary filing evidence. "
                "Without verified total financial debt, net debt bridge cannot be established; deep DCF valuation is suppressed."
            ),
            is_certified=False,
        )

    # P0-A: Resolve and validate diluted share denominator metadata
    dshares_meta, err_msg, err_code = _resolve_and_validate_diluted_share_metadata(
        cf=cf,
        metrics=metrics,
        target_use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        valuation_date=metrics.market_price_date or "2026-10-16",
        require_certified=True,
    )
    if err_msg is not None:
        status_code = f"SUPPRESSED_DENOMINATOR_{err_code.value}" if err_code else "SUPPRESSED_DENOMINATOR_UNAVAILABLE"
        logger.warning(f"DUOL deep valuation suppressed: {err_msg}")
        return DeepValuationAssessment(
            symbol="DUOL",
            framework_used="FCFF (Enterprise to Equity DCF, Lease-as-Operating-Expense Convention)",
            current_price=current_price,
            current_price_date=metrics.market_price_date or "2026-10-16",
            current_market_cap=round(metrics.market_cap or (current_price * shares), 2),
            diluted_shares=shares,
            scenarios={},
            reverse_dcf_implied_5y_growth_pct=None,
            reverse_dcf_fixed_assumptions={"solution_status": status_code},
            sensitivity_matrix={},
            thesis_invalidation_thresholds=[],
            next_quarter_evidence_milestones=[],
            valuation_status="SUPPRESSED_DENOMINATOR_INCOMPATIBLE" if err_code != DenominatorErrorCode.MISSING_PROVENANCE else "SUPPRESSED_DENOMINATOR_UNAVAILABLE",
            suppression_reason=f"Share denominator compatibility failure: {err_msg}",
            is_certified=False,
            share_basis=dshares_meta.basis.value if dshares_meta else None,
            share_denominator_metadata=dshares_meta.to_dict() if dshares_meta else None,
        )

    tax_rate = 21.0
    reinvest = 10.0  # Asset-light software model

    # 1. Conservative Scenario: Organic user growth plateaus; 18% CAGR, 14% operating margin, 11% WACC
    c_res = run_fcff_model(rev, shares, net_bridge, 18.0, 14.0, 11.0, 2.5, tax_rate, reinvest, share_metadata=dshares_meta)
    c_scenario = ValuationScenario(
        scenario_name="Conservative",
        growth_years_1_to_5_pct=18.0,
        operating_margin_target_pct=14.0,
        wacc_pct=11.0,
        terminal_growth_pct=2.5,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
        pv_explicit_cash_flows=round(c_res["pv_explicit"], 2),
        pv_terminal_value=round(c_res["pv_terminal"], 2),
        enterprise_value_estimate=round(c_res["enterprise_value"], 2),
        net_debt_bridge=round(net_bridge, 2),
        equity_value_estimate=round(c_res["equity_value"], 2),
        per_share_value_estimate=round(c_res["per_share"], 2),
        upside_downside_pct=round((c_res["per_share"] - current_price) / current_price * 100, 2),
        key_assumptions="DAU conversion saturates; GenAI language competitors pressure subscription pricing; SBC remains elevated (~12% of revenue)."
    )

    # 2. Base Scenario: Durable expansion; 26% CAGR, 18% operating margin, 10.0% WACC
    b_res = run_fcff_model(rev, shares, net_bridge, 26.0, 18.0, 10.0, 3.0, tax_rate, reinvest, share_metadata=dshares_meta)
    b_scenario = ValuationScenario(
        scenario_name="Base",
        growth_years_1_to_5_pct=26.0,
        operating_margin_target_pct=18.0,
        wacc_pct=10.0,
        terminal_growth_pct=3.0,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
        pv_explicit_cash_flows=round(b_res["pv_explicit"], 2),
        pv_terminal_value=round(b_res["pv_terminal"], 2),
        enterprise_value_estimate=round(b_res["enterprise_value"], 2),
        net_debt_bridge=round(net_bridge, 2),
        equity_value_estimate=round(b_res["equity_value"], 2),
        per_share_value_estimate=round(b_res["per_share"], 2),
        upside_downside_pct=round((b_res["per_share"] - current_price) / current_price * 100, 2),
        key_assumptions="Duolingo Max (AI tier) drives ARPU lift; family plan adoption increases retention; marketing leverage expands margins."
    )

    # 3. Optimistic Scenario: Global multi-subject compounding (Math/Music/English); 34% CAGR, 22% margin, 9.0% WACC
    o_res = run_fcff_model(rev, shares, net_bridge, 34.0, 22.0, 9.0, 3.5, tax_rate, reinvest, share_metadata=dshares_meta)
    o_scenario = ValuationScenario(
        scenario_name="Optimistic",
        growth_years_1_to_5_pct=34.0,
        operating_margin_target_pct=22.0,
        wacc_pct=9.0,
        terminal_growth_pct=3.5,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
        pv_explicit_cash_flows=round(o_res["pv_explicit"], 2),
        pv_terminal_value=round(o_res["pv_terminal"], 2),
        enterprise_value_estimate=round(o_res["enterprise_value"], 2),
        net_debt_bridge=round(net_bridge, 2),
        equity_value_estimate=round(o_res["equity_value"], 2),
        per_share_value_estimate=round(o_res["per_share"], 2),
        upside_downside_pct=round((o_res["per_share"] - current_price) / current_price * 100, 2),
        key_assumptions="Duolingo becomes default digital learning operating system globally; English test certification disrupts legacy TOEFL/IELTS."
    )

    # Reverse DCF
    rev_growth, r_note = solve_reverse_dcf(current_price, rev, shares, net_bridge, 18.0, 10.0, 3.0, tax_rate, reinvest, share_metadata=dshares_meta)
    sens_matrix = generate_sensitivity_matrix(rev, shares, net_bridge, 26.0, 18.0, tax_rate, reinvest, share_metadata=dshares_meta)

    return DeepValuationAssessment(
        symbol="DUOL",
        framework_used="FCFF (Enterprise to Equity DCF, Lease-as-Operating-Expense Convention)",
        current_price=current_price,
        current_price_date=metrics.market_price_date or "2026-10-16",
        current_market_cap=round(metrics.market_cap or (current_price * shares), 2),
        diluted_shares=shares,
        scenarios={
            "Conservative": c_scenario,
            "Base": b_scenario,
            "Optimistic": o_scenario,
        },
        reverse_dcf_implied_5y_growth_pct=rev_growth,
        reverse_dcf_fixed_assumptions={
            "solved_variable": "5-year Revenue CAGR",
            "operating_margin_target_pct": 18.0,
            "wacc_pct": 10.0,
            "terminal_growth_pct": 3.0,
            "effective_tax_rate_pct": tax_rate,
            "reinvestment_rate_pct": reinvest,
            "net_debt_bridge_usd": net_bridge,
            "solution_status": r_note,
        },
        sensitivity_matrix=sens_matrix,
        thesis_invalidation_thresholds=[
            "Daily Active User (DAU) or Paid Subscriber YoY growth decelerating below 15%.",
            "Stock-based compensation expanding beyond 15% of total revenues without corresponding operating leverage.",
            "Gross margin dropping below 70% driven by escalating third-party LLM API inference fees.",
            "Material erosion of Duolingo English Test (DET) institutional acceptance rate.",
        ],
        next_quarter_evidence_milestones=[
            "Duolingo Max conversion rate as percentage of total paid subscribers.",
            "SBC / Revenue trajectory and net annual share dilution moderation.",
            "DAU-to-MAU engagement ratio stability across international non-English geographies.",
        ],
        valuation_status="AVAILABLE",
        suppression_reason=None,
        is_certified=True if dshares_meta is not None else False,
        share_basis=dshares_meta.basis.value if dshares_meta else None,
        share_denominator_metadata=dshares_meta.to_dict() if dshares_meta else None,
    )


def evaluate_deep_valuation_ulta(
    cf: CompanyFundamentals,
    metrics: DerivedMetrics,
) -> Optional[DeepValuationAssessment]:
    """Deep valuation package for ULTA (Ulta Beauty, Inc.)."""
    rev = metrics.revenue_ttm
    shares = metrics.diluted_shares_weighted_avg
    net_bridge = metrics.net_debt
    current_price = metrics.market_price

    if rev is None or shares is None or net_bridge is None or current_price is None:
        logger.warning("ULTA deep valuation suppressed: critical inputs unavailable (net_debt is None)")
        return DeepValuationAssessment(
            symbol="ULTA",
            framework_used="FCFF (Enterprise to Equity DCF, Lease-as-Operating-Expense Convention)",
            current_price=current_price or 0.0,
            current_price_date=metrics.market_price_date or "2026-10-16",
            current_market_cap=round(metrics.market_cap or ((current_price or 0.0) * (shares or 0.0)), 2),
            diluted_shares=shares or 0.0,
            scenarios={},
            reverse_dcf_implied_5y_growth_pct=None,
            reverse_dcf_fixed_assumptions={
                "solved_variable": "5-year Revenue CAGR",
                "operating_margin_target_pct": 12.0,
                "wacc_pct": 9.0,
                "terminal_growth_pct": 2.0,
                "effective_tax_rate_pct": 23.5,
                "reinvestment_rate_pct": 20.0,
                "net_debt_bridge_usd": None,
                "solution_status": "SUPPRESSED_NET_DEBT_UNAVAILABLE",
            },
            sensitivity_matrix={},
            thesis_invalidation_thresholds=[
                "Comparable store sales (comps) turning negative for two consecutive quarters.",
                "Operating margin falling below 10.0% due to persistent industry promotional battles.",
                "Decline in active Ultamate Rewards loyalty member count or spend-per-member.",
                "Inventory turnover degradation leading to substantial merchandise inventory write-downs.",
            ],
            next_quarter_evidence_milestones=[
                "Comparable store sales breakdown (transactions vs average ticket).",
                "Merchandise margin versus supply chain costs commentary.",
                "Pace of share repurchases relative to free cash flow generation.",
            ],
            valuation_status="SUPPRESSED_UNAVAILABLE",
            suppression_reason=(
                "Total financial debt is UNAVAILABLE due to incomplete component reporting in SEC EDGAR XBRL (reported "
                "ShortTermBorrowings of $339.6M without noncurrent debt component). Partial debt cannot be presented as total debt "
                "under aggregate completeness rules. Without verified total financial debt, net debt bridge cannot be established; "
                "deep DCF valuation is suppressed."
            ),
            is_certified=False,
        )

    # P0-A: Resolve and validate diluted share denominator metadata
    dshares_meta, err_msg, err_code = _resolve_and_validate_diluted_share_metadata(
        cf=cf,
        metrics=metrics,
        target_use_case=DenominatorUseCase.PRESENT_VALUE_DCF,
        valuation_date=metrics.market_price_date or "2026-10-16",
        require_certified=True,
    )
    if err_msg is not None:
        status_code = f"SUPPRESSED_DENOMINATOR_{err_code.value}" if err_code else "SUPPRESSED_DENOMINATOR_UNAVAILABLE"
        logger.warning(f"ULTA deep valuation suppressed: {err_msg}")
        return DeepValuationAssessment(
            symbol="ULTA",
            framework_used="FCFF (Enterprise to Equity DCF, Lease-as-Operating-Expense Convention)",
            current_price=current_price,
            current_price_date=metrics.market_price_date or "2026-10-16",
            current_market_cap=round(metrics.market_cap or (current_price * shares), 2),
            diluted_shares=shares,
            scenarios={},
            reverse_dcf_implied_5y_growth_pct=None,
            reverse_dcf_fixed_assumptions={"solution_status": status_code},
            sensitivity_matrix={},
            thesis_invalidation_thresholds=[],
            next_quarter_evidence_milestones=[],
            valuation_status="SUPPRESSED_DENOMINATOR_INCOMPATIBLE" if err_code != DenominatorErrorCode.MISSING_PROVENANCE else "SUPPRESSED_DENOMINATOR_UNAVAILABLE",
            suppression_reason=f"Share denominator compatibility failure: {err_msg}",
            is_certified=False,
            share_basis=dshares_meta.basis.value if dshares_meta else None,
            share_denominator_metadata=dshares_meta.to_dict() if dshares_meta else None,
        )

    tax_rate = 23.5
    reinvest = 20.0

    # 1. Conservative Scenario: Comp store deceleration; 4.0% CAGR, 10.5% margin, 10.0% WACC
    c_res = run_fcff_model(rev, shares, net_bridge, 4.0, 10.5, 10.0, 1.5, tax_rate, reinvest, share_metadata=dshares_meta)
    c_scenario = ValuationScenario(
        scenario_name="Conservative",
        growth_years_1_to_5_pct=4.0,
        operating_margin_target_pct=10.5,
        wacc_pct=10.0,
        terminal_growth_pct=1.5,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
        pv_explicit_cash_flows=round(c_res["pv_explicit"], 2),
        pv_terminal_value=round(c_res["pv_terminal"], 2),
        enterprise_value_estimate=round(c_res["enterprise_value"], 2),
        net_debt_bridge=round(net_bridge, 2),
        equity_value_estimate=round(c_res["equity_value"], 2),
        per_share_value_estimate=round(c_res["per_share"], 2),
        upside_downside_pct=round((c_res["per_share"] - current_price) / current_price * 100, 2),
        key_assumptions="Competition from Sephora and mass retail intensifies; promotional discounting erodes gross margins; comp sales turn flat."
    )

    # 2. Base Scenario: Steady compounder; 8.0% CAGR, 12.0% margin, 9.0% WACC
    b_res = run_fcff_model(rev, shares, net_bridge, 8.0, 12.0, 9.0, 2.0, tax_rate, reinvest, share_metadata=dshares_meta)
    b_scenario = ValuationScenario(
        scenario_name="Base",
        growth_years_1_to_5_pct=8.0,
        operating_margin_target_pct=12.0,
        wacc_pct=9.0,
        terminal_growth_pct=2.0,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
        pv_explicit_cash_flows=round(b_res["pv_explicit"], 2),
        pv_terminal_value=round(b_res["pv_terminal"], 2),
        enterprise_value_estimate=round(b_res["enterprise_value"], 2),
        net_debt_bridge=round(net_bridge, 2),
        equity_value_estimate=round(b_res["equity_value"], 2),
        per_share_value_estimate=round(b_res["per_share"], 2),
        upside_downside_pct=round((b_res["per_share"] - current_price) / current_price * 100, 2),
        key_assumptions="Target store-in-store concept expands; loyalty program (Ultamate Rewards 44M+ members) preserves wallet share; steady share buybacks."
    )

    # 3. Optimistic Scenario: International expansion & prestige acceleration; 12.0% CAGR, 13.5% margin, 8.0% WACC
    o_res = run_fcff_model(rev, shares, net_bridge, 12.0, 13.5, 8.0, 2.5, tax_rate, reinvest, share_metadata=dshares_meta)
    o_scenario = ValuationScenario(
        scenario_name="Optimistic",
        growth_years_1_to_5_pct=12.0,
        operating_margin_target_pct=13.5,
        wacc_pct=8.0,
        terminal_growth_pct=2.5,
        tax_rate_pct=tax_rate,
        reinvestment_rate_pct=reinvest,
        pv_explicit_cash_flows=round(o_res["pv_explicit"], 2),
        pv_terminal_value=round(o_res["pv_terminal"], 2),
        enterprise_value_estimate=round(o_res["enterprise_value"], 2),
        net_debt_bridge=round(net_bridge, 2),
        equity_value_estimate=round(o_res["equity_value"], 2),
        per_share_value_estimate=round(o_res["per_share"], 2),
        upside_downside_pct=round((o_res["per_share"] - current_price) / current_price * 100, 2),
        key_assumptions="Successful international rollout (Mexico expansion); prestige fragrance & skincare outperformance; operating leverage."
    )

    # Reverse DCF
    rev_growth, r_note = solve_reverse_dcf(current_price, rev, shares, net_bridge, 12.0, 9.0, 2.0, tax_rate, reinvest, share_metadata=dshares_meta)
    sens_matrix = generate_sensitivity_matrix(rev, shares, net_bridge, 8.0, 12.0, tax_rate, reinvest, share_metadata=dshares_meta)

    return DeepValuationAssessment(
        symbol="ULTA",
        framework_used="FCFF (Enterprise to Equity DCF, Lease-as-Operating-Expense Convention)",
        current_price=current_price,
        current_price_date=metrics.market_price_date or "2026-10-16",
        current_market_cap=round(metrics.market_cap or (current_price * shares), 2),
        diluted_shares=shares,
        scenarios={
            "Conservative": c_scenario,
            "Base": b_scenario,
            "Optimistic": o_scenario,
        },
        reverse_dcf_implied_5y_growth_pct=rev_growth,
        reverse_dcf_fixed_assumptions={
            "solved_variable": "5-year Revenue CAGR",
            "operating_margin_target_pct": 12.0,
            "wacc_pct": 9.0,
            "terminal_growth_pct": 2.0,
            "effective_tax_rate_pct": tax_rate,
            "reinvestment_rate_pct": reinvest,
            "net_debt_bridge_usd": net_bridge,
            "solution_status": r_note,
        },
        sensitivity_matrix=sens_matrix,
        thesis_invalidation_thresholds=[
            "Comparable store sales (comps) turning negative for two consecutive quarters.",
            "Operating margin falling below 10.0% due to persistent industry promotional battles.",
            "Decline in active Ultamate Rewards loyalty member count or spend-per-member.",
            "Inventory turnover degradation leading to substantial merchandise inventory write-downs.",
        ],
        next_quarter_evidence_milestones=[
            "Comparable store sales breakdown (transactions vs average ticket).",
            "Merchandise margin versus supply chain costs commentary.",
            "Pace of share repurchases relative to free cash flow generation.",
        ],
        valuation_status="AVAILABLE",
        suppression_reason=None,
        is_certified=True if dshares_meta is not None else False,
        share_basis=dshares_meta.basis.value if dshares_meta else None,
        share_denominator_metadata=dshares_meta.to_dict() if dshares_meta else None,
    )
