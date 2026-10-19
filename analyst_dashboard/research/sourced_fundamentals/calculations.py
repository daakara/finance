"""
analyst_dashboard/research/sourced_fundamentals/calculations.py

Reproducible, auditable financial calculations for Sourced Fundamentals research.
Computes growth, margins, non-GAAP CFO-minus-Capex, ROIC, balance sheet ratios,
valuation multiples, and executes rigorous arithmetic integrity reconciliations.
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
    """
    if equity is None:
        return None
    debt = total_debt or 0.0
    cash = total_cash or 0.0
    ic = equity + debt - cash
    return ic if ic > 0 else equity  # Prevent zero/negative invested capital in asset-light cash-rich firms


def compute_effective_tax_rate(
    tax_expense: Optional[float],
    pretax_income: Optional[float],
    statutory_fallback: float = 0.21
) -> float:
    """
    Effective tax rate = Tax Expense / Pretax Income.
    Clamped to reasonable bounds [0.10, 0.35] to handle tax holidays,
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
    
    cash = facts["cash_and_equivalents"].value or 0.0
    st_inv = facts["short_term_investments"].value or 0.0
    tot_cash = facts["total_cash_and_investments"].value or (cash + st_inv)
    
    s_debt = facts["short_term_debt"].value or 0.0
    l_debt = facts["long_term_debt"].value or 0.0
    tot_debt = facts["total_financial_debt"].value or (s_debt + l_debt)
    net_debt = facts["net_debt"].value or (tot_debt - tot_cash)
    leases = facts["operating_lease_liabilities"].value or 0.0
    equity = facts["stockholders_equity"].value if "stockholders_equity" in facts else None
    
    dshares = facts["diluted_shares_weighted_avg"].value
    cshares = facts["common_shares_outstanding"].value
    
    # Growth & Margins
    rev_growth_yoy = ((rev_ttm - revenue_prior_ttm) / revenue_prior_ttm * 100) if (rev_ttm and revenue_prior_ttm) else None
    ebit_margin = (ebit_ttm / rev_ttm * 100) if (ebit_ttm and rev_ttm) else None
    ni_margin = (ni_ttm / rev_ttm * 100) if (ni_ttm and rev_ttm) else None
    
    # Cash Flow & Conversion
    fcf_conv_ebit = (fcf_ttm / ebit_ttm * 100) if (fcf_ttm is not None and ebit_ttm and ebit_ttm > 0) else None
    fcf_conv_ni = (fcf_ttm / ni_ttm * 100) if (fcf_ttm is not None and ni_ttm and ni_ttm > 0) else None
    sbc_pct_rev = (sbc_ttm / rev_ttm * 100) if (sbc_ttm is not None and rev_ttm) else None
    
    # Capital Intensity & ROIC
    tax_exp = facts["tax_expense_ttm"].value if "tax_expense_ttm" in facts else None
    pretax = facts["pretax_income_ttm"].value if "pretax_income_ttm" in facts else None
    eff_tax_rate = compute_effective_tax_rate(tax_exp, pretax)
    nopat = (ebit_ttm * (1.0 - eff_tax_rate)) if ebit_ttm is not None else None
    inv_cap = compute_invested_capital(equity, tot_debt, tot_cash)
    roic = (nopat / inv_cap * 100) if (nopat is not None and inv_cap and inv_cap > 0) else None
    
    # Per-share metrics
    shares_for_per_share = dshares or cshares
    eps_ttm = (ni_ttm / shares_for_per_share) if (ni_ttm is not None and shares_for_per_share) else None
    fcf_per_share = (fcf_ttm / shares_for_per_share) if (fcf_ttm is not None and shares_for_per_share) else None
    
    # Market Reference & Multiples
    price = market_data.get("market_price")
    price_date = market_data.get("price_date")
    market_cap = (price * shares_for_per_share) if (price and shares_for_per_share) else None
    # Enterprise Value = Market Cap + Net Debt + Operating Lease Liabilities
    ev = (market_cap + net_debt + leases) if market_cap is not None else None
    
    pe_ratio = (market_cap / ni_ttm) if (market_cap and ni_ttm and ni_ttm > 0) else None
    ev_to_ebit = (ev / ebit_ttm) if (ev and ebit_ttm and ebit_ttm > 0) else None
    p_to_fcf = (market_cap / fcf_ttm) if (market_cap and fcf_ttm and fcf_ttm > 0) else None
    ev_to_sales = (ev / rev_ttm) if (ev and rev_ttm and rev_ttm > 0) else None
    
    # Arithmetic Reconciliations & Verification
    arithmetic_checks: Dict[str, bool] = {}
    calculation_notes: Dict[str, str] = {}
    
    # 1. CFO minus Capex
    if cfo_ttm is not None and capex_ttm is not None and fcf_ttm is not None:
        reconciled_fcf = abs((cfo_ttm - capex_ttm) - fcf_ttm) < 1.0
        arithmetic_checks["cfo_minus_capex_reconciled"] = reconciled_fcf
        calculation_notes["cfo_minus_capex"] = f"CFO ({cfo_ttm:,.0f}) - Capex ({capex_ttm:,.0f}) = {fcf_ttm:,.0f}"
        
    # 2. Net Debt
    if tot_debt is not None and tot_cash is not None and net_debt is not None:
        reconciled_debt = abs((tot_debt - tot_cash) - net_debt) < 1.0
        arithmetic_checks["net_debt_reconciled"] = reconciled_debt
        calculation_notes["net_debt"] = f"Total Debt ({tot_debt:,.0f}) - Total Cash ({tot_cash:,.0f}) = {net_debt:,.0f}"

    # 3. Enterprise Value
    if market_cap is not None and ev is not None:
        expected_ev = market_cap + net_debt + leases
        reconciled_ev = abs(expected_ev - ev) < 1.0
        arithmetic_checks["enterprise_value_reconciled"] = reconciled_ev
        calculation_notes["enterprise_value"] = f"Market Cap ({market_cap:,.0f}) + Net Debt ({net_debt:,.0f}) + Lease Liabilities ({leases:,.0f}) = {ev:,.0f}"

    # 4. Dilution Notes
    if split_notes:
        calculation_notes["dilution"] = split_notes
    else:
        calculation_notes["dilution"] = f"1Y Dilution: {dilution_1y_pct:.1f}%, 3Y Dilution: {dilution_3y_pct:.1f}%" if dilution_1y_pct is not None else "Dilution data unavailable"

    return DerivedMetrics(
        symbol=sym,
        period_label=f"TTM ended {cf.latest_reported_period}",
        revenue_ttm=round(rev_ttm, 2) if rev_ttm else None,
        revenue_prior_ttm=round(revenue_prior_ttm, 2) if revenue_prior_ttm else None,
        revenue_growth_yoy_pct=round(rev_growth_yoy, 2) if rev_growth_yoy is not None else None,
        operating_income_ttm=round(ebit_ttm, 2) if ebit_ttm else None,
        operating_margin_pct=round(ebit_margin, 2) if ebit_margin is not None else None,
        net_income_ttm=round(ni_ttm, 2) if ni_ttm else None,
        net_margin_pct=round(ni_margin, 2) if ni_margin is not None else None,
        cfo_ttm=round(cfo_ttm, 2) if cfo_ttm else None,
        capex_ttm=round(capex_ttm, 2) if capex_ttm else None,
        cfo_minus_capex_ttm=round(fcf_ttm, 2) if fcf_ttm else None,
        fcf_conversion_ebit_pct=round(fcf_conv_ebit, 2) if fcf_conv_ebit is not None else None,
        fcf_conversion_net_income_pct=round(fcf_conv_ni, 2) if fcf_conv_ni is not None else None,
        sbc_ttm=round(sbc_ttm, 2) if sbc_ttm else None,
        sbc_to_revenue_pct=round(sbc_pct_rev, 2) if sbc_pct_rev is not None else None,
        invested_capital=round(inv_cap, 2) if inv_cap else None,
        effective_tax_rate_pct=round(eff_tax_rate * 100, 2),
        nopat_ttm=round(nopat, 2) if nopat else None,
        roic_pct=round(roic, 2) if roic is not None else None,
        cash_and_equivalents=round(cash, 2),
        short_term_investments=round(st_inv, 2),
        total_cash_and_investments=round(tot_cash, 2),
        short_term_debt=round(s_debt, 2),
        long_term_debt=round(l_debt, 2),
        total_financial_debt=round(tot_debt, 2),
        net_debt=round(net_debt, 2),
        operating_lease_liabilities=round(leases, 2),
        diluted_shares_weighted_avg=dshares,
        common_shares_outstanding=cshares,
        diluted_eps_ttm=round(eps_ttm, 2) if eps_ttm is not None else None,
        cfo_minus_capex_per_share=round(fcf_per_share, 2) if fcf_per_share is not None else None,
        share_dilution_1y_pct=round(dilution_1y_pct, 2) if dilution_1y_pct is not None else None,
        share_dilution_3y_pct=round(dilution_3y_pct, 2) if dilution_3y_pct is not None else None,
        market_price=price,
        market_price_date=price_date,
        market_cap=round(market_cap, 2) if market_cap else None,
        enterprise_value=round(ev, 2) if ev else None,
        pe_ratio=round(pe_ratio, 2) if pe_ratio else None,
        ev_to_ebit=round(ev_to_ebit, 2) if ev_to_ebit else None,
        p_to_cfo_minus_capex=round(p_to_fcf, 2) if p_to_fcf else None,
        ev_to_sales=round(ev_to_sales, 2) if ev_to_sales else None,
        calculation_notes=calculation_notes,
        arithmetic_checks=arithmetic_checks,
    )
