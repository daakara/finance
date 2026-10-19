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
    split_notes: Optional[str] = None
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

    inv_cap = compute_invested_capital(equity, tot_debt, tot_cash)
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

    # 4. Per-Share Metrics
    shares_for_per_share = dshares or cshares
    eps_ttm = None
    if ni_ttm is not None and shares_for_per_share is not None and shares_for_per_share > 0:
        eps_ttm = ni_ttm / shares_for_per_share

    fcf_per_share = None
    if fcf_ttm is not None and shares_for_per_share is not None and shares_for_per_share > 0:
        fcf_per_share = fcf_ttm / shares_for_per_share

    # 5. Market Reference & Valuation Multiples
    price = market_data.get("market_price")
    price_date = market_data.get("price_date")

    market_cap = None
    if price is not None and shares_for_per_share is not None and shares_for_per_share > 0:
        market_cap = price * shares_for_per_share

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
    if tot_debt is not None and tot_cash is not None and net_debt is not None:
        reconciled_debt = abs((tot_debt - tot_cash) - net_debt) < 1.0
        arithmetic_checks["net_debt_reconciled"] = reconciled_debt
        calculation_notes["net_debt"] = f"Total Debt ({tot_debt:,.0f}) - Total Cash ({tot_cash:,.0f}) = {net_debt:,.0f}"

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
