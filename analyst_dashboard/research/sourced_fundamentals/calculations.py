"""
analyst_dashboard/research/sourced_fundamentals/calculations.py

Reproducible, auditable financial calculations for Sourced Fundamentals research.
Computes growth, margins, non-GAAP CFO-minus-Capex, ROIC, balance sheet ratios,
valuation multiples, and executes rigorous arithmetic integrity reconciliations.
Enforces explicit None-checks, Lease-as-Operating-Expense valuation convention,
split-normalized share comparisons, and honest non-positive invested capital reporting.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from .contracts import CompanyFundamentals, DerivedMetrics
from .normalizer import parse_date

logger = logging.getLogger(__name__)


def compute_invested_capital(
    equity: Optional[float],
    total_debt: Optional[float],
    total_cash: Optional[float]
) -> Optional[float]:
    """
    Invested Capital = Total Equity + Total Debt - Cash & Short-Term Investments.
    Reflects net capital deployed in core operations.
    Returns calculated value as-is (preserves negative/zero values without silent substitution).
    """
    if equity is None or total_debt is None or total_cash is None:
        return None
    return equity + total_debt - total_cash


def compute_effective_tax_rate(
    tax_expense: Optional[float],
    pretax_income: Optional[float],
    statutory_fallback: float = 0.21
) -> float:
    """
    Effective tax rate = Tax Expense / Pretax Income.
    Clamped to reasonable bounds [0.05, 0.40] to handle tax holidays,
    discrete valuation allowance releases, or losses.
    """
    if tax_expense is not None and pretax_income is not None and pretax_income > 0:
        rate = tax_expense / pretax_income
        if 0.05 <= rate <= 0.40:
            return round(rate, 4)
    return statutory_fallback


def compute_derived_metrics(
    cf: CompanyFundamentals,
    market_data: Dict[str, Any],
    revenue_prior_ttm: Optional[float] = None,
    dilution_1y_pct: Optional[float] = None,
    dilution_3y_pct: Optional[float] = None,
    split_notes: Optional[str] = None,
    dilution_details: Optional[Dict[str, Any]] = None,
) -> DerivedMetrics:
    """
    Compute derived financial metrics, valuation multiples, and verification checks.
    Propagates unavailable critical inputs into dependent calculations.
    """
    sym = cf.symbol
    facts = cf.facts

    rev_ttm = facts["revenue_ttm"].value
    ebit_ttm = facts["operating_income_ttm"].value
    ni_ttm = facts["net_income_ttm"].value
    cfo_ttm = facts["cfo_ttm"].value
    capex_ttm = facts["capex_ttm"].value
    fcf_ttm = facts["cfo_minus_capex_ttm"].value
    sbc_ttm = facts["sbc_ttm"].value if "sbc_ttm" in facts else None

    cash = facts["cash_and_equivalents"].value
    st_inv = facts["short_term_investments"].value
    tot_cash = facts["total_cash_and_investments"].value

    s_debt = facts["short_term_debt"].value
    l_debt = facts["long_term_debt"].value
    tot_debt = facts["total_financial_debt"].value
    net_debt = facts["net_debt"].value
    leases = facts["operating_lease_liabilities"].value
    equity = facts["stockholders_equity"].value if "stockholders_equity" in facts else None

    dshares = facts["diluted_shares_weighted_avg"].value
    cshares = facts["common_shares_outstanding"].value

    calculation_notes: Dict[str, str] = {}
    arithmetic_checks: Dict[str, bool] = {}

    # 1. Growth & Margins
    rev_growth_yoy = None
    if rev_ttm is not None and revenue_prior_ttm is not None and revenue_prior_ttm > 0:
        rev_growth_yoy = ((rev_ttm - revenue_prior_ttm) / revenue_prior_ttm) * 100.0

    ebit_margin = None
    if ebit_ttm is not None and rev_ttm is not None and rev_ttm > 0:
        ebit_margin = (ebit_ttm / rev_ttm) * 100.0

    ni_margin = None
    if ni_ttm is not None and rev_ttm is not None and rev_ttm > 0:
        ni_margin = (ni_ttm / rev_ttm) * 100.0

    # 2. Cash Flow & Conversion
    fcf_conv_ebit = None
    if fcf_ttm is not None and ebit_ttm is not None and ebit_ttm > 0:
        fcf_conv_ebit = (fcf_ttm / ebit_ttm) * 100.0

    fcf_conv_ni = None
    if fcf_ttm is not None and ni_ttm is not None and ni_ttm > 0:
        fcf_conv_ni = (fcf_ttm / ni_ttm) * 100.0

    sbc_pct_rev = None
    if sbc_ttm is not None and rev_ttm is not None and rev_ttm > 0:
        sbc_pct_rev = (sbc_ttm / rev_ttm) * 100.0

    # 3. Capital Intensity & ROIC
    tax_exp = facts["tax_expense_ttm"].value if "tax_expense_ttm" in facts else None
    pretax = facts["pretax_income_ttm"].value if "pretax_income_ttm" in facts else None
    eff_tax_rate = compute_effective_tax_rate(tax_exp, pretax)
    nopat = (ebit_ttm * (1.0 - eff_tax_rate)) if ebit_ttm is not None else None

    effective_cash_for_ic = tot_cash if tot_cash is not None else cash
    inv_cap = compute_invested_capital(equity, tot_debt, effective_cash_for_ic)
    roic = None
    if nopat is not None and inv_cap is not None:
        if inv_cap > 0:
            roic = (nopat / inv_cap) * 100.0
        else:
            roic = None
            calculation_notes["roic"] = (
                f"Invested capital is non-positive ({inv_cap:,.0f} USD due to net cash exceeding equity). "
                "Conventional ROIC is not economically meaningful."
            )

    # 4. Per-Share Metrics & Denominator Semantics
    # Distinguish period-weighted diluted shares from point-in-time common shares
    dshares_fact = facts.get("diluted_shares_weighted_avg")
    ni_fact = facts.get("net_income_ttm")

    dshares_duration = (parse_date(dshares_fact.period_end) - parse_date(dshares_fact.period_start)).days if (dshares_fact and dshares_fact.period_start and dshares_fact.period_end) else 0
    ni_duration = (parse_date(ni_fact.period_end) - parse_date(ni_fact.period_start)).days if (ni_fact and ni_fact.period_start and ni_fact.period_end) else 0

    eps_ttm = None
    est_eps = None
    est_status = None
    num_period = f"{ni_fact.period_start} to {ni_fact.period_end}" if (ni_fact and ni_fact.period_start and ni_fact.period_end) else None
    den_period = f"{dshares_fact.period_start} to {dshares_fact.period_end}" if (dshares_fact and dshares_fact.period_start and dshares_fact.period_end) else None

    if ni_ttm is not None and dshares is not None and dshares > 0:
        if 340 <= dshares_duration <= 385 and 340 <= ni_duration <= 385:
            # Full fiscal year / annual match: conventional diluted EPS is supported
            eps_ttm = ni_ttm / dshares
            est_status = "REPORTED_ANNUAL_COMPATIBLE"
            den_period = f"{den_period} (annual weighted-average shares)"
            calculation_notes["diluted_eps_ttm"] = (
                f"Annual Diluted EPS: Net Income ({ni_ttm:,.0f}) / annual diluted weighted-average shares ({dshares:,.0f}). "
                f"Compatible 12-month periods ({num_period})."
            )
        else:
            # Interim period (e.g. 10-Q): conventional diluted_eps_ttm is suppressed
            eps_ttm = None
            est_eps = ni_ttm / dshares
            est_status = "RESEARCH_ESTIMATE"
            den_period = f"{den_period} (quarterly weighted-average shares)"
            calculation_notes["diluted_eps_ttm"] = (
                f"Conventional diluted_eps_ttm is suppressed for interim reporting periods because US GAAP ASC 260 "
                "diluted weighted-average shares cannot be constructed by aggregating quarterly share counts. "
                f"Exposing separately named research estimate: TTM net income per latest-quarter diluted share (${est_eps:.2f})."
            )
            calculation_notes["ttm_net_income_per_latest_quarter_diluted_share"] = (
                f"Research estimate: TTM Net Income ({ni_ttm:,.0f}) / latest-quarter diluted weighted-average shares ({dshares:,.0f}). "
                f"Numerator period: {num_period}; Denominator period: {den_period}; Status: {est_status}."
            )

    fcf_per_share = None
    if fcf_ttm is not None and dshares is not None and dshares > 0:
        fcf_per_share = fcf_ttm / dshares
        if dshares_duration < 300:
            calculation_notes["cfo_minus_capex_per_share"] = (
                f"TTM CFO-minus-Capex per latest-quarter diluted share: CFO minus Capex ({fcf_ttm:,.0f}) / "
                f"latest reported diluted weighted-average shares ({dshares:,.0f}). "
                f"Denominator is quarterly ({den_period}); not a full-year weighted average."
            )
        else:
            calculation_notes["cfo_minus_capex_per_share"] = (
                f"TTM CFO-minus-Capex per annual diluted share: CFO minus Capex ({fcf_ttm:,.0f}) / "
                f"annual reported diluted weighted-average shares ({dshares:,.0f})."
            )

    # Dilution notes with comparison metadata
    if dilution_details:
        comp_1y = dilution_details.get("comparison_1y", {})
        comp_3y = dilution_details.get("comparison_3y", {})
        basis = dilution_details.get("measurement_basis", "period-weighted")
        if dilution_1y_pct is not None and comp_1y.get("status") == "AVAILABLE":
            calculation_notes["share_dilution_1y_pct"] = (
                f"1-year change in {str(basis).lower()} diluted weighted-average shares: "
                f"{comp_1y.get('period_start')}..{comp_1y.get('period_end')} ({comp_1y.get('reported_val', 0):,.0f} shares, "
                f"form {comp_1y.get('form')} filed {comp_1y.get('filed')}, accn {comp_1y.get('accn')}) to current period "
                f"({dshares:,.0f} shares). Adjustment basis: {comp_1y.get('adjustment_basis')}. "
                "Represents period-weighted diluted share count change, distinct from point-in-time common shares outstanding."
            )
        if dilution_3y_pct is not None and comp_3y.get("status") == "AVAILABLE":
            calculation_notes["share_dilution_3y_pct"] = (
                f"3-year change in {str(basis).lower()} diluted weighted-average shares: "
                f"{comp_3y.get('period_start')}..{comp_3y.get('period_end')} ({comp_3y.get('reported_val', 0):,.0f} shares, "
                f"form {comp_3y.get('form')} filed {comp_3y.get('filed')}, accn {comp_3y.get('accn')}) to current period "
                f"({dshares:,.0f} shares). Adjustment basis: {comp_3y.get('adjustment_basis')}. "
                "Represents period-weighted diluted share count change, distinct from point-in-time common shares outstanding."
            )

    # 5. Market Reference & Market Capitalization
    price = market_data.get("market_price")
    price_date = market_data.get("price_date")

    market_cap = None
    if price is not None and cshares is not None and cshares > 0:
        market_cap = price * cshares
        calculation_notes["market_cap"] = (
            f"Calculated from point-in-time common shares outstanding ({cshares:,.0f} shares) "
            f"multiplied by reference price (${price:.2f})"
        )
    elif price is not None and dshares is not None and dshares > 0:
        market_cap = price * dshares
        calculation_notes["market_cap"] = (
            f"Estimated proxy: reference price (${price:.2f}) multiplied by period-weighted diluted shares ({dshares:,.0f}). "
            "Point-in-time common shares outstanding was unavailable; not an authoritative point-in-time market capitalization."
        )

    # Enterprise Value: Lease-as-Operating-Expense Convention
    # EV = Market Cap + Net Debt
    # Operating leases are already expensed in EBIT/CFO, so lease liabilities are NOT added into EV
    ev = None
    if market_cap is not None and net_debt is not None:
        ev = market_cap + net_debt

    pe_ratio = None
    if market_cap is not None and ni_ttm is not None and ni_ttm > 0:
        pe_ratio = market_cap / ni_ttm

    ev_to_ebit = None
    if ev is not None and ebit_ttm is not None and ebit_ttm > 0:
        ev_to_ebit = ev / ebit_ttm

    p_to_fcf = None
    if market_cap is not None and fcf_ttm is not None and fcf_ttm > 0:
        p_to_fcf = market_cap / fcf_ttm

    ev_to_sales = None
    if ev is not None and rev_ttm is not None and rev_ttm > 0:
        ev_to_sales = ev / rev_ttm

    # 6. Arithmetic Reconciliations & Verification
    # CFO minus Capex
    if cfo_ttm is not None and capex_ttm is not None and fcf_ttm is not None:
        reconciled_fcf = abs((cfo_ttm - capex_ttm) - fcf_ttm) < 1.0
        arithmetic_checks["cfo_minus_capex_reconciled"] = reconciled_fcf
        calculation_notes["cfo_minus_capex"] = f"CFO ({cfo_ttm:,.0f}) - Capex ({capex_ttm:,.0f}) = {fcf_ttm:,.0f}"

    # Net Debt
    effective_cash_reconcile = tot_cash if tot_cash is not None else cash
    if tot_debt is not None and effective_cash_reconcile is not None and net_debt is not None:
        reconciled_debt = abs((tot_debt - effective_cash_reconcile) - net_debt) < 1.0
        arithmetic_checks["net_debt_reconciled"] = reconciled_debt
        cash_label = "Total Cash & Investments" if tot_cash is not None else "Cash & Equivalents"
        calculation_notes["net_debt"] = f"Total Debt ({tot_debt:,.0f}) - {cash_label} ({effective_cash_reconcile:,.0f}) = {net_debt:,.0f}"

    # Enterprise Value
    if market_cap is not None and net_debt is not None and ev is not None:
        expected_ev = market_cap + net_debt
        reconciled_ev = abs(expected_ev - ev) < 1.0
        arithmetic_checks["enterprise_value_reconciled"] = reconciled_ev
        calculation_notes["enterprise_value"] = (
            f"Market Cap ({market_cap:,.0f}) + Net Debt ({net_debt:,.0f}) = {ev:,.0f} "
            "(Lease-as-operating-expense convention: operating lease liabilities excluded from bridge)"
        )

    # Dilution Notes
    if split_notes:
        calculation_notes["dilution"] = (
            f"{split_notes} | 1Y Dilution: {dilution_1y_pct:.2f}%, 3Y Dilution: {dilution_3y_pct:.2f}%"
            if (dilution_1y_pct is not None and dilution_3y_pct is not None)
            else split_notes
        )
    else:
        calculation_notes["dilution"] = (
            f"1Y Dilution: {dilution_1y_pct:.2f}%, 3Y Dilution: {dilution_3y_pct:.2f}%"
            if (dilution_1y_pct is not None and dilution_3y_pct is not None)
            else "Dilution data unavailable"
        )

    return DerivedMetrics(
        symbol=sym,
        period_label=f"TTM ended {cf.latest_reported_period}",
        revenue_ttm=round(rev_ttm, 2) if rev_ttm is not None else None,
        revenue_prior_ttm=round(revenue_prior_ttm, 2) if revenue_prior_ttm is not None else None,
        revenue_growth_yoy_pct=round(rev_growth_yoy, 2) if rev_growth_yoy is not None else None,
        operating_income_ttm=round(ebit_ttm, 2) if ebit_ttm is not None else None,
        operating_margin_pct=round(ebit_margin, 2) if ebit_margin is not None else None,
        net_income_ttm=round(ni_ttm, 2) if ni_ttm is not None else None,
        net_margin_pct=round(ni_margin, 2) if ni_margin is not None else None,
        cfo_ttm=round(cfo_ttm, 2) if cfo_ttm is not None else None,
        capex_ttm=round(capex_ttm, 2) if capex_ttm is not None else None,
        cfo_minus_capex_ttm=round(fcf_ttm, 2) if fcf_ttm is not None else None,
        fcf_conversion_ebit_pct=round(fcf_conv_ebit, 2) if fcf_conv_ebit is not None else None,
        fcf_conversion_net_income_pct=round(fcf_conv_ni, 2) if fcf_conv_ni is not None else None,
        sbc_ttm=round(sbc_ttm, 2) if sbc_ttm is not None else None,
        sbc_to_revenue_pct=round(sbc_pct_rev, 2) if sbc_pct_rev is not None else None,
        invested_capital=round(inv_cap, 2) if inv_cap is not None else None,
        effective_tax_rate_pct=round(eff_tax_rate * 100.0, 2),
        nopat_ttm=round(nopat, 2) if nopat is not None else None,
        roic_pct=round(roic, 2) if roic is not None else None,
        cash_and_equivalents=round(cash, 2) if cash is not None else None,
        short_term_investments=round(st_inv, 2) if st_inv is not None else None,
        total_cash_and_investments=round(tot_cash, 2) if tot_cash is not None else None,
        short_term_debt=round(s_debt, 2) if s_debt is not None else None,
        long_term_debt=round(l_debt, 2) if l_debt is not None else None,
        total_financial_debt=round(tot_debt, 2) if tot_debt is not None else None,
        net_debt=round(net_debt, 2) if net_debt is not None else None,
        operating_lease_liabilities=round(leases, 2) if leases is not None else None,
        diluted_shares_weighted_avg=dshares,
        common_shares_outstanding=cshares,
        diluted_eps_ttm=round(eps_ttm, 2) if eps_ttm is not None else None,
        ttm_net_income_per_latest_quarter_diluted_share=round(est_eps, 2) if est_eps is not None else None,
        ttm_eps_estimate_status=est_status,
        ttm_eps_numerator_period=num_period,
        ttm_eps_denominator_period=den_period,
        cfo_minus_capex_per_share=round(fcf_per_share, 2) if fcf_per_share is not None else None,
        share_dilution_1y_pct=round(dilution_1y_pct, 2) if dilution_1y_pct is not None else None,
        share_dilution_3y_pct=round(dilution_3y_pct, 2) if dilution_3y_pct is not None else None,
        market_price=price,
        market_price_date=price_date,
        market_cap=round(market_cap, 2) if market_cap is not None else None,
        enterprise_value=round(ev, 2) if ev is not None else None,
        pe_ratio=round(pe_ratio, 2) if pe_ratio is not None else None,
        ev_to_ebit=round(ev_to_ebit, 2) if ev_to_ebit is not None else None,
        p_to_cfo_minus_capex=round(p_to_fcf, 2) if p_to_fcf is not None else None,
        ev_to_sales=round(ev_to_sales, 2) if ev_to_sales is not None else None,
        calculation_notes=calculation_notes,
        arithmetic_checks=arithmetic_checks,
    )
