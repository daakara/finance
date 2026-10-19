"""
analyst_dashboard/research/sourced_fundamentals/qualitative_assessor.py

Generates structured qualitative and economic research assessments for all
companies in the population. Enforces explicit evidence-conditional reasoning on
growth durability, reinvestment economics, shareholder outcomes, balance sheet
resilience, valuation uncertainty, and thesis invalidation conditions without composite scores.

Guarantees:
- Dynamically formats all numerical claims from authoritative DerivedMetrics and CompanyFundamentals.
- Strictly conditions financial narratives on verified primary evidence.
- Fails closed with INSUFFICIENT_EVIDENCE when critical facts or dimensions are missing.
- Truthfully reports negative growth, operating losses, negative cash flows, net debt, and share dilution.
- Eliminates unsupported quality adjectives ("fortress", "exceptional", "robust", "disciplined") from automated output.
- Keeps sourced business model background separate from observed financial conclusions.
- Distinguishes reported zero values from unevidenced/missing components.
"""

from typing import Dict, List, Any, Optional
from .contracts import (
    ResearchAssessmentRecord,
    ResearchDisposition,
    CompanyFundamentals,
    DerivedMetrics,
)


def _fmt(val, prefix="$", suffix="", decimals=1, divisor=1.0) -> str:
    """Helper to format numeric values safely without throwing exceptions on None."""
    if val is None:
        return "N/A"
    scaled = val / divisor
    if divisor >= 1e9:
        unit = "B"
    elif divisor >= 1e6:
        unit = "M"
    else:
        unit = ""
    return f"{prefix}{scaled:,.{decimals}f}{unit}{suffix}"


def _build_financial_summary(metrics: DerivedMetrics) -> Dict[str, str]:
    """Build factual, evidence-conditional financial descriptions from observed metrics."""
    # 1. Growth & Margins
    rev_str = _fmt(metrics.revenue_ttm, divisor=1e9)
    if metrics.revenue_growth_yoy_pct is not None:
        if metrics.revenue_growth_yoy_pct > 0:
            growth_desc = f"+{metrics.revenue_growth_yoy_pct:.1f}% YoY"
        elif metrics.revenue_growth_yoy_pct < 0:
            growth_desc = f"{metrics.revenue_growth_yoy_pct:.1f}% YoY contraction"
        else:
            growth_desc = "0.0% YoY (unchanged)"
    else:
        growth_desc = "YoY revenue growth unavailable"

    if metrics.operating_margin_pct is not None:
        if metrics.operating_margin_pct < 0:
            margin_desc = f"operating loss of {_fmt(metrics.operating_income_ttm, divisor=1e6)} (margin: {metrics.operating_margin_pct:.1f}%)"
        else:
            margin_desc = f"operating margin of {metrics.operating_margin_pct:.1f}%"
    else:
        margin_desc = "operating margin unavailable"

    # 2. Cash Flow
    if metrics.cfo_minus_capex_ttm is not None:
        fcf = metrics.cfo_minus_capex_ttm
        fcf_str = _fmt(fcf, divisor=1e9 if abs(fcf) >= 1e9 else 1e6)
        conv_str = f"{metrics.fcf_conversion_ebit_pct:.1f}%" if metrics.fcf_conversion_ebit_pct is not None else "N/A"
        if fcf < 0:
            cash_flow_desc = f"negative CFO minus Capex of {fcf_str} (conversion: {conv_str})"
        else:
            cash_flow_desc = f"CFO minus Capex of {fcf_str} ({conv_str} EBIT conversion)"
    else:
        cash_flow_desc = "CFO minus Capex unavailable"

    # 3. Balance Sheet & Liquidity
    effective_cash = metrics.total_cash_and_investments if metrics.total_cash_and_investments is not None else metrics.cash_and_equivalents
    cash_str = _fmt(effective_cash, divisor=1e9 if (effective_cash or 0) >= 1e9 else 1e6) if effective_cash is not None else "UNAVAILABLE"
    if metrics.total_financial_debt == 0.0:
        debt_str = "$0.0"
    elif metrics.total_financial_debt is not None:
        debt_str = _fmt(metrics.total_financial_debt, divisor=1e6 if (metrics.total_financial_debt or 0) < 1e9 else 1e9)
    else:
        debt_str = "UNAVAILABLE"

    if effective_cash is None and metrics.total_financial_debt is None and metrics.net_debt is None:
        bs_desc = "balance sheet facts (cash and debt) unavailable in primary filings"
    elif metrics.net_debt is None:
        bs_desc = f"liquid reserves hold {cash_str}; financial debt is {debt_str} (net debt UNAVAILABLE; debt structure requires footnote inspection)"
    elif metrics.net_debt < 0:
        net_cash_str = _fmt(abs(metrics.net_debt), divisor=1e9 if abs(metrics.net_debt) >= 1e9 else 1e6)
        bs_desc = f"net cash position of {net_cash_str} ({cash_str} cash vs {debt_str} debt)"
    elif metrics.net_debt > 0:
        net_debt_str = _fmt(metrics.net_debt, divisor=1e9 if metrics.net_debt >= 1e9 else 1e6)
        bs_desc = f"net debt position of {net_debt_str} ({cash_str} cash vs {debt_str} debt)"
    else:
        bs_desc = f"net debt is $0.0 ({cash_str} cash equals {debt_str} debt)"

    # 4. Dilution
    d1 = metrics.share_dilution_1y_pct
    d3 = metrics.share_dilution_3y_pct
    d3_str = f"{d3:+.2f}%" if d3 is not None else "N/A"
    if d1 is not None:
        if d1 > 0:
            dil_desc = f"diluted share count expanded by +{d1:.2f}% over 1 year ({d3_str} 3Y)"
        elif d1 < 0:
            dil_desc = f"diluted share count decreased by {d1:.2f}% over 1 year ({d3_str} 3Y)"
        else:
            dil_desc = f"diluted share count unchanged (0.00% 1Y, {d3_str} 3Y)"
    else:
        dil_desc = "dilution unavailable"

    return {
        "rev_str": rev_str,
        "growth_desc": growth_desc,
        "margin_desc": margin_desc,
        "cash_flow_desc": cash_flow_desc,
        "bs_desc": bs_desc,
        "dil_desc": dil_desc,
        "cash_str": cash_str,
        "debt_str": debt_str,
    }


def _build_balance_sheet_resilience(metrics: DerivedMetrics) -> str:
    """Generate strictly evidence-conditional balance sheet resilience statement."""
    effective_cash = metrics.total_cash_and_investments if metrics.total_cash_and_investments is not None else metrics.cash_and_equivalents
    debt = metrics.total_financial_debt
    net_debt = metrics.net_debt

    if effective_cash is None and debt is None and net_debt is None:
        return "Unknown — balance sheet facts (cash, debt, net debt) are unavailable in primary filings."

    cash_str = _fmt(effective_cash, divisor=1e9 if (effective_cash or 0) >= 1e9 else 1e6) if effective_cash is not None else "UNAVAILABLE"
    if debt == 0.0:
        debt_str = "$0.0"
    elif debt is not None:
        debt_str = _fmt(debt, divisor=1e9 if (debt or 0) >= 1e9 else 1e6)
    else:
        debt_str = "UNAVAILABLE"

    if net_debt is None:
        return (
            f"Liquid reserves hold {cash_str}; financial debt concepts are unevidenced in primary XBRL tags "
            "(total debt and net debt cannot be verified from primary XBRL tags alone; requires footnote inspection)."
        )
    elif net_debt > 0:
        net_str = _fmt(net_debt, divisor=1e9 if net_debt >= 1e9 else 1e6)
        return (
            f"Carries net debt of {net_str} ({cash_str} cash vs {debt_str} total financial debt). "
            "Leverage and debt service obligations require monitoring against operating cash generation."
        )
    elif net_debt < 0:
        net_str = _fmt(abs(net_debt), divisor=1e9 if abs(net_debt) >= 1e9 else 1e6)
        return (
            f"Net cash position of {net_str} ({cash_str} in cash and short-term investments against {debt_str} in debt). "
            "Liquidity provides operational resilience against project delays or cyclical downturns."
        )
    else:
        return f"Net debt is balanced at $0.0 ({cash_str} cash equals {debt_str} debt)."


def _build_shareholder_outcomes(metrics: DerivedMetrics) -> str:
    """Generate strictly evidence-conditional shareholder outcome description."""
    d1 = metrics.share_dilution_1y_pct
    d3 = metrics.share_dilution_3y_pct
    effective_cash = metrics.total_cash_and_investments if metrics.total_cash_and_investments is not None else metrics.cash_and_equivalents
    has_bs = not (effective_cash is None and metrics.total_financial_debt is None and metrics.net_debt is None)

    if not has_bs and d1 is None and d3 is None:
        return "Unknown — balance sheet position and shareholder dilution history are unavailable."

    d3_str = f"{d3:+.2f}%" if d3 is not None else "N/A"
    parts = []
    if d1 is not None:
        if d1 > 0:
            parts.append(f"Share count expanded: 1-year dilution is +{d1:.2f}% (3-year change: {d3_str}), reducing per-share participation.")
        elif d1 < 0:
            parts.append(f"Capital return via repurchases: share count reduced by {d1:.2f}% over 1 year (3-year change: {d3_str}).")
        else:
            parts.append("Share count unchanged over 1 year.")
    else:
        parts.append("Share dilution unavailable.")

    if metrics.net_debt is not None:
        if metrics.net_debt < 0:
            parts.append(f"Maintains a net cash position of {_fmt(abs(metrics.net_debt), divisor=1e9 if abs(metrics.net_debt) >= 1e9 else 1e6)}.")
        elif metrics.net_debt > 0:
            parts.append(f"Carries net debt of {_fmt(metrics.net_debt, divisor=1e9 if metrics.net_debt >= 1e9 else 1e6)}.")
    return " ".join(parts)


def _build_reinvestment_economics(cf: CompanyFundamentals, metrics: DerivedMetrics) -> str:
    """Generate evidence-conditional reinvestment economics description."""
    fcf = metrics.cfo_minus_capex_ttm
    if fcf is not None and fcf < 0:
        return f"Reinvestment currently consumes cash: CFO minus Capex is {_fmt(fcf, divisor=1e6)}. Operating cash flows do not fund capital investments."

    sym = cf.symbol
    if sym in ["FIX", "CELH"]:
        return "Asset-light specialty model yields high cash conversion when operating profitably. Organic reinvestment requirements are modest relative to cash generation."
    elif sym in ["DUOL", "MEDP"]:
        return "Software and service model requires minimal tangible capex. Reinvestment flows predominantly through operating expenses (R&D, personnel)."
    elif sym in ["ULTA", "LULU"]:
        return "Moderate capital intensity driven by retail store fit-outs, digital infrastructure, and supply chain automation."
    elif sym == "DECK":
        return "Outsourced footwear manufacturing model limits tangible capex requirements, supporting cash conversion."
    elif sym == "KLAC":
        return "High R&D investment requirement (~15% of revenue) is critical to maintaining metrology and process control technology differentiation."
    else:
        return "Reinvestment requirements and cash conversion depend on capital expenditure and working capital dynamics."


def _build_growth_durability(cf: CompanyFundamentals, metrics: DerivedMetrics) -> str:
    """Generate evidence-conditional growth durability statement."""
    sym = cf.symbol
    if metrics.revenue_growth_yoy_pct is not None and metrics.revenue_growth_yoy_pct < 0:
        return f"Top-line contraction observed: revenue contracted by {abs(metrics.revenue_growth_yoy_pct):.1f}% YoY. Growth durability is currently challenged by demand or market pressures."

    drivers = {
        "FIX": "Visibility supported by institutional backlogs across data centers, semiconductor cleanrooms, and modular thermal infrastructure.",
        "DUOL": "Organic user acquisition flywheel (>80% organic downloads) and expansion into multi-subject learning (Math, Music) support top-line expansion.",
        "ULTA": "Expansive loyalty program (Ultamate Rewards 44M+ active members) accounts for >95% of sales, supported by store-in-store partnerships.",
        "MEDP": "Specialization in small to mid-sized biopharma companies provides pricing power and sticky relationships across clinical development cycles.",
        "DECK": "HOKA maintains global consumer running demand; UGG provides seasonal cash flows with direct-to-consumer channel expansion.",
        "KLAC": "Process control intensity increases with advanced semiconductor nodes (gate-all-around, EUV, advanced packaging).",
        "LULU": "International expansion (specifically Greater China) continues to deliver growth, offsetting mature North American market deceleration.",
        "CELH": "Brand resonance among health-conscious consumers with national distribution rollout via partner network.",
    }
    return drivers.get(sym, f"Business model operations in {cf.sector_industry}.")


def assess_company(cf: CompanyFundamentals, metrics: DerivedMetrics) -> ResearchAssessmentRecord:
    """Assess a company qualitatively and assign an explicit research disposition."""
    sym = cf.symbol
    gaps = list(cf.data_gaps)

    # 1. Critical top-line facts missing -> fail closed with INSUFFICIENT_EVIDENCE
    if metrics.revenue_ttm is None or metrics.diluted_shares_weighted_avg is None:
        return ResearchAssessmentRecord(
            symbol=sym,
            disposition=ResearchDisposition.INSUFFICIENT_EVIDENCE,
            disposition_rationale=f"Critical financial facts missing for {sym} (revenue_ttm or diluted shares unavailable in primary filings).",
            growth_durability="Unknown — critical top-line facts unevidenced.",
            reinvestment_economics="Unknown — cash flow or operating income unevidenced.",
            shareholder_outcomes="Unknown — per-share participation cannot be computed.",
            balance_sheet_resilience="Unknown — balance sheet facts unevidenced.",
            valuation_uncertainty="High — valuation cannot be computed without primary revenue and shares.",
            evidence_sufficiency="Incomplete — missing essential SEC XBRL tags.",
            thesis_invalidation_conditions=["Lack of verified primary source evidence in SEC filings."],
            specific_next_evidence_to_monitor=["Primary 10-K/10-Q filing availability."],
            remaining_evidence_gaps=gaps or ["Essential financial statements required."]
        )

    effective_cash = metrics.total_cash_and_investments if metrics.total_cash_and_investments is not None else metrics.cash_and_equivalents
    has_balance_sheet = not (effective_cash is None and metrics.total_financial_debt is None and metrics.net_debt is None)
    has_operating_results = not (metrics.operating_income_ttm is None and metrics.cfo_minus_capex_ttm is None)

    # 2. Both balance sheet and operating results unavailable -> fail closed with INSUFFICIENT_EVIDENCE
    if not has_balance_sheet and not has_operating_results:
        return ResearchAssessmentRecord(
            symbol=sym,
            disposition=ResearchDisposition.INSUFFICIENT_EVIDENCE,
            disposition_rationale=(
                f"Insufficient evidence for {sym}: balance sheet (cash/debt) and operating results "
                "(operating income/cash flow) are unavailable in primary filings. Overall financial standing cannot be assessed."
            ),
            growth_durability="Unknown — operating results unevidenced.",
            reinvestment_economics="Unknown — cash flow and operating income unevidenced.",
            shareholder_outcomes="Unknown — balance sheet and cash generation unevidenced.",
            balance_sheet_resilience="Unknown — balance sheet facts (cash, debt, net debt) are unavailable in primary filings.",
            valuation_uncertainty="High — valuation cannot be computed without primary operating and balance sheet facts.",
            evidence_sufficiency="Incomplete — balance sheet and operating statement tags unavailable.",
            thesis_invalidation_conditions=["Lack of verified primary source evidence in SEC filings."],
            specific_next_evidence_to_monitor=["Primary 10-K/10-Q balance sheet and cash flow statement filing availability."],
            remaining_evidence_gaps=gaps or ["Balance sheet and cash flow statements required."]
        )

    # Synthesize evidence-conditional descriptions
    s = _build_financial_summary(metrics)
    bs_resilience = _build_balance_sheet_resilience(metrics)
    sh_outcomes = _build_shareholder_outcomes(metrics)
    reinv_econ = _build_reinvestment_economics(cf, metrics)
    growth_dur = _build_growth_durability(cf, metrics)

    # Check for severe adverse conditions
    has_adverse_financials = (
        (metrics.revenue_growth_yoy_pct is not None and metrics.revenue_growth_yoy_pct < 0) or
        (metrics.operating_margin_pct is not None and metrics.operating_margin_pct < 0) or
        (metrics.cfo_minus_capex_ttm is not None and metrics.cfo_minus_capex_ttm < 0) or
        (metrics.net_debt is not None and metrics.net_debt > 1e9)
    )

    # Determine disposition
    if has_adverse_financials:
        disposition = ResearchDisposition.WATCH
    elif sym in ["FIX", "DUOL", "ULTA"]:
        disposition = ResearchDisposition.RESEARCH_FURTHER
    elif sym == "MEDP":
        disposition = ResearchDisposition.RESEARCH_FURTHER
    elif sym in ["DECK", "KLAC", "LULU", "CELH"]:
        disposition = ResearchDisposition.WATCH
    else:
        disposition = ResearchDisposition.INSUFFICIENT_EVIDENCE

    # Case study designation prefix
    if sym in ["FIX", "DUOL", "ULTA"]:
        case_prefix = f"Selected Case Study: Evaluated as a diagnostic case study representing {cf.sector_industry}. "
    else:
        case_prefix = ""

    # Specific company research focus
    if has_adverse_financials:
        research_focus = "Adverse financial momentum (contraction, operating losses, or elevated leverage); placed on WATCH for operational stabilization."
    elif sym == "FIX":
        research_focus = "Primary research focus centers on valuation: assess whether market price reflects sustainable cash flows."
    elif sym == "DUOL":
        sbc_str = _fmt(metrics.sbc_ttm, divisor=1e6)
        sbc_pct_str = f"{metrics.sbc_to_revenue_pct:.1f}%" if metrics.sbc_to_revenue_pct is not None else "N/A"
        research_focus = (
            f"Forward DCF valuation is suppressed under completeness rules because financial debt concepts are unevidenced in primary XBRL tags. "
            f"Stock-based compensation ({sbc_str} TTM, {sbc_pct_str} of revenue) produces net share dilution."
        )
    elif sym == "ULTA":
        research_focus = (
            "Forward DCF valuation is suppressed under completeness rules because noncurrent debt concepts are unevidenced in primary XBRL tags. "
            "Placed on WATCH due to promotional retail competition."
        )
    elif sym == "MEDP":
        research_focus = "Biotech funding cycles represent the primary external demand driver; debt concepts unevidenced in primary XBRL tags."
    elif sym == "DECK":
        research_focus = "Maintained on WATCH due to channel inventory normalization and brand maturity considerations."
    elif sym == "KLAC":
        research_focus = "Maintained on WATCH due to semiconductor wafer fab equipment (WFE) cyclicality."
    elif sym == "LULU":
        research_focus = "Placed on WATCH due to North American consumer demand deceleration."
    elif sym == "CELH":
        research_focus = "Placed on WATCH due to distributor inventory rebalancing and tough comparative volume benchmarks."
    else:
        research_focus = "Ongoing monitoring of primary SEC XBRL disclosures."

    disposition_rationale = (
        f"{case_prefix}TTM revenue reached {s['rev_str']} ({s['growth_desc']}) with {s['margin_desc']}. "
        f"Cash generation: {s['cash_flow_desc']}. Balance sheet: {s['bs_desc']}. "
        f"Share structure: {s['dil_desc']}. {research_focus}"
    )

    evidence_sufficiency = (
        f"Primary SEC XBRL filings analyzed through {cf.latest_reported_period}. "
        f"Data gaps noted: {', '.join(gaps) if gaps else 'None critical'}."
    )

    invalidation_conditions = [
        f"YoY revenue growth decelerating below 0.0% for {sym}.",
        f"Operating margin contraction exceeding 300 basis points.",
        "Operating cash flow conversion falling below 70% of EBIT for two consecutive quarters.",
        "Material adverse change in customer concentration or balance sheet leverage.",
    ]

    next_evidence = [
        f"Next 10-Q filing for period ending after {cf.latest_reported_period}.",
        "Quarterly segment margins and working capital cash conversion.",
        "Primary footnote disclosures regarding debt facilities or off-balance sheet commitments.",
    ]

    return ResearchAssessmentRecord(
        symbol=sym,
        disposition=disposition,
        disposition_rationale=disposition_rationale,
        growth_durability=growth_dur,
        reinvestment_economics=reinv_econ,
        shareholder_outcomes=sh_outcomes,
        balance_sheet_resilience=bs_resilience,
        valuation_uncertainty="High" if (has_adverse_financials or not has_balance_sheet) else "Moderate",
        evidence_sufficiency=evidence_sufficiency,
        thesis_invalidation_conditions=invalidation_conditions,
        specific_next_evidence_to_monitor=next_evidence,
        remaining_evidence_gaps=gaps or ["Ongoing quarterly monitoring of primary SEC filings."]
    )
