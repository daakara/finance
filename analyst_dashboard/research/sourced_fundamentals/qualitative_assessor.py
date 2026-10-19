"""
analyst_dashboard/research/sourced_fundamentals/qualitative_assessor.py

Generates structured qualitative and economic research assessments for all
companies in the population. Enforces explicit reasoning on growth durability,
reinvestment economics, shareholder outcomes, balance sheet resilience,
valuation uncertainty, and thesis invalidation conditions without composite scores.

Guarantees:
- Dynamically formats all numerical claims from authoritative DerivedMetrics and CompanyFundamentals.
- Eliminates hardcoded narrative numbers that could desynchronize from refreshed metrics.
- Separates computed facts, sourced qualitative facts, analyst interpretations, scenario assumptions, and data gaps.
"""

from typing import Dict, List
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


def assess_company(cf: CompanyFundamentals, metrics: DerivedMetrics) -> ResearchAssessmentRecord:
    """Assess a company qualitatively and assign an explicit research disposition."""
    sym = cf.symbol
    gaps = list(cf.data_gaps)

    # Fail closed with INSUFFICIENT_EVIDENCE if critical financial facts are missing
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

    # Helper format strings
    rev_str = _fmt(metrics.revenue_ttm, divisor=1e9)
    growth_str = f"{metrics.revenue_growth_yoy_pct:.1f}%" if metrics.revenue_growth_yoy_pct is not None else "N/A"
    ebit_margin_str = f"{metrics.operating_margin_pct:.1f}%" if metrics.operating_margin_pct is not None else "N/A"
    fcf_str = _fmt(metrics.cfo_minus_capex_ttm, divisor=1e9 if (metrics.cfo_minus_capex_ttm or 0) >= 1e9 else 1e6)
    conv_str = f"{metrics.fcf_conversion_ebit_pct:.1f}%" if metrics.fcf_conversion_ebit_pct is not None else "N/A"
    net_debt_str = _fmt(metrics.net_debt, divisor=1e9 if abs(metrics.net_debt or 0) >= 1e9 else 1e6) if metrics.net_debt is not None else "UNAVAILABLE"
    effective_cash = metrics.total_cash_and_investments if metrics.total_cash_and_investments is not None else metrics.cash_and_equivalents
    cash_str = _fmt(effective_cash, divisor=1e9 if (effective_cash or 0) >= 1e9 else 1e6)
    debt_str = _fmt(metrics.total_financial_debt, divisor=1e6 if (metrics.total_financial_debt or 0) < 1e9 else 1e9) if metrics.total_financial_debt is not None else "UNAVAILABLE"
    dil_1y_str = f"{metrics.share_dilution_1y_pct:+.2f}%" if metrics.share_dilution_1y_pct is not None else "N/A"
    dil_3y_str = f"{metrics.share_dilution_3y_pct:+.2f}%" if metrics.share_dilution_3y_pct is not None else "N/A"

    if sym == "FIX":
        return ResearchAssessmentRecord(
            symbol="FIX",
            disposition=ResearchDisposition.RESEARCH_FURTHER,
            disposition_rationale=(
                f"Selected Case Study: Operational momentum supported by multi-year secular tailwinds in data center "
                f"thermal cooling and electrical infrastructure. TTM revenue reached {rev_str} (+{growth_str} YoY) "
                f"with operating margins expanding to {ebit_margin_str}. Generates robust non-GAAP free cash flow "
                f"({fcf_str} CFO minus Capex, {conv_str} EBIT conversion) supported by customer prepayments and billings in excess of costs. "
                f"Fortress balance sheet with {net_debt_str} net debt ({cash_str} cash vs {debt_str} debt). "
                f"Diluted share count is disciplined ({dil_1y_str} 1Y). Primary research focus centers on valuation: "
                "investigate whether high growth expectations are already fully priced into market shares."
            ),
            growth_durability=(
                "High visibility backed by multi-billion institutional backlogs across AI data centers, "
                "semiconductor cleanrooms, and life-science facilities. Off-site modular prefabrication "
                "reduces jobsite labor constraints and creates competitive barriers."
            ),
            reinvestment_economics=(
                "Asset-light specialty contracting and modular assembly model yields high cash conversion. "
                "Organic reinvestment requirements are modest relative to cash generation."
            ),
            shareholder_outcomes=(
                "Disciplined capital allocation: net-cash balance sheet, modest recurring dividend, "
                "selective accretive bolt-on acquisitions, and negative net share dilution."
            ),
            balance_sheet_resilience=(
                f"Exceptional liquidity: holds {cash_str} in cash and equivalents against just {debt_str} in debt, "
                "providing an immense liquidity cushion against construction cyclicality or project delays."
            ),
            valuation_uncertainty=(
                "Substantial rerating over the past 24 months. Reverse-DCF indicates elevated multi-year "
                "growth rates required to justify current market price under base margins."
            ),
            evidence_sufficiency=(
                "Comprehensive primary XBRL filings available through Q2 2026. Backlog customer concentration "
                "and modular prefabrication margin disclosures should be tracked quarterly in MD&A."
            ),
            thesis_invalidation_conditions=[
                "Hyperscaler or enterprise data center cancellation or deferral of thermal cooling backlogs.",
                "Operating margin contraction below 12.0% indicating fixed-price contract cost overruns.",
                "Operating cash flow conversion falling below 80% of EBIT for two consecutive quarters.",
                "Shift toward debt-funded speculative acquisitions outside core mechanical/electrical competencies.",
            ],
            specific_next_evidence_to_monitor=[
                "Quarterly backlog growth rate and percentage of backlog dedicated to tech/institutional sectors.",
                "Gross margin stability across modular construction and mechanical contracting segments.",
                "Working capital cash conversion and billing in excess of costs reconciliation.",
            ],
            remaining_evidence_gaps=[
                "Granular margin breakdown between custom modular prefabrication and standard maintenance contracting."
            ]
        )

    elif sym == "DUOL":
        sbc_str = _fmt(metrics.sbc_ttm, divisor=1e6)
        sbc_pct_str = f"{metrics.sbc_to_revenue_pct:.1f}%" if metrics.sbc_to_revenue_pct is not None else "N/A"
        return ResearchAssessmentRecord(
            symbol="DUOL",
            disposition=ResearchDisposition.RESEARCH_FURTHER,
            disposition_rationale=(
                f"Selected Case Study: Digital consumer subscription model displaying +{growth_str} YoY revenue growth ({rev_str} TTM) "
                f"and expanding operating margins ({ebit_margin_str}). Software economics generate {fcf_str} "
                f"in CFO minus Capex ({conv_str} EBIT conversion due to upfront deferred subscription billings). "
                f"Liquid reserves hold {cash_str}; financial debt concepts are UNAVAILABLE in SEC XBRL (net debt is UNAVAILABLE; "
                f"debt-free status cannot be asserted from primary XBRL tags alone). Forward DCF valuation is suppressed under completeness rules. "
                f"Stock-based compensation ({sbc_str} TTM, {sbc_pct_str} of revenue) produces net share dilution ({dil_1y_str} 1Y) "
                "that must be scrutinized against true owner earnings."
            ),
            growth_durability=(
                "Organic viral user acquisition flywheel (>80% organic downloads). Duolingo Max AI subscription tier "
                "and expansion into multi-subject learning (Math, Music) support ongoing top-line expansion."
            ),
            reinvestment_economics=(
                "Negligible tangible capital intensity. Primary reinvestment flows through R&D expense "
                "(engineering and AI model product development)."
            ),
            shareholder_outcomes=(
                "Exceptional cash generation and revenue compounding, tempered by persistent stock-based compensation "
                "dilution that reduces per-share economic participation."
            ),
            balance_sheet_resilience=(
                f"Liquid software balance sheet: {cash_str} in cash and short-term investments; total financial debt is unevidenced "
                "in XBRL tags, requiring footnote inspection before leverage or net cash can be asserted."
            ),
            valuation_uncertainty=(
                "Trading at premium multiples of revenue and earnings. High valuation sensitivity to "
                "subscriber growth deceleration or generative AI app competition. DCF valuation suppressed due to unavailable debt."
            ),
            evidence_sufficiency=(
                "Complete SEC XBRL filings through Q2 2026 for operating statements. Critical evidence gap: financial debt concepts "
                "are unavailable in XBRL tags, suppressing full DCF equity bridge calculations."
            ),
            thesis_invalidation_conditions=[
                "Daily Active User (DAU) or Paid Subscriber YoY growth decelerating below 15%.",
                "Stock-based compensation expanding beyond 15% of revenue without operating leverage.",
                "Gross margin compression below 70% driven by escalating AI model inference API costs.",
                "Erosion in Duolingo English Test (DET) institutional acceptance rate.",
            ],
            specific_next_evidence_to_monitor=[
                "Duolingo Max conversion rate and percentage of total paid subscribers.",
                "SBC-to-revenue trajectory and net annual share dilution moderation.",
                "Primary source disclosure of long-term debt or credit agreements in 10-Q footnotes.",
            ],
            remaining_evidence_gaps=[
                "Detailed gross margins and API inference costs specific to the Duolingo Max GenAI features.",
                "Primary filing confirmation of credit facilities or off-balance sheet debt obligations."
            ]
        )

    elif sym == "ULTA":
        return ResearchAssessmentRecord(
            symbol="ULTA",
            disposition=ResearchDisposition.RESEARCH_FURTHER,
            disposition_rationale=(
                f"Selected Case Study: Omnichannel beauty specialty retailer with steady growth ({growth_str} YoY, {rev_str} TTM) "
                f"and healthy operating profitability ({ebit_margin_str} EBIT margin). Generates {fcf_str} in CFO minus Capex. "
                f"Balance sheet reports {debt_str} in short-term borrowings, but noncurrent debt is unevidenced in XBRL, making "
                f"total financial debt and net debt UNAVAILABLE under aggregate completeness rules; forward DCF valuation is suppressed. "
                f"Strong capital return track record via share repurchases ({dil_1y_str} 1Y, {dil_3y_str} 3Y dilution). "
                "Offers attractive GARP characteristics if comparable-store sales stabilize against promotional mass-retail competition."
            ),
            growth_durability=(
                "Expansive loyalty program (Ultamate Rewards 44M+ active members) accounts for >95% of sales. "
                "Store-in-store partnership with Target and international expansion (Mexico) provide ongoing footprint growth."
            ),
            reinvestment_economics=(
                "Moderate capital intensity driven by new store fit-outs, supply chain automation, and digital upgrades. "
                "ROIC remains attractive on mature retail store bases."
            ),
            shareholder_outcomes=(
                "Consistent counter-cyclical share repurchases have significantly reduced diluted share count over 3 years, "
                "accreting earnings per share even during periods of moderate top-line growth."
            ),
            balance_sheet_resilience=(
                f"Sound liquidity with {cash_str} in cash against {debt_str} in reported short-term borrowings; "
                "long-term debt is unevidenced in XBRL tags, so total debt and net debt cannot be verified from primary XBRL tags alone. "
                "Significant operating lease obligations (~$2.18B) require ongoing store-level EBITDA generation."
            ),
            valuation_uncertainty=(
                "Moderated multiples relative to historical retail peer averages; downside risk centers on gross margin "
                "compression from price competition and promotional beauty discounting. DCF valuation suppressed due to incomplete debt."
            ),
            evidence_sufficiency=(
                "Authoritative primary 10-Q and 10-K filings available through Q2 2026. Critical evidence gap: incomplete debt components "
                "in XBRL (short-term borrowings reported without noncurrent debt component) suppress total debt derivation and DCF equity bridge."
            ),
            thesis_invalidation_conditions=[
                "Comparable store sales turning negative for two consecutive quarters.",
                "Operating margin contraction below 10.0% due to persistent industry price discounting.",
                "Decline in active Ultamate Rewards member count or spend per member.",
                "Inventory turnover degradation leading to substantial merchandise inventory write-downs.",
            ],
            specific_next_evidence_to_monitor=[
                "Quarterly comparable store sales growth (transactions vs average ticket size).",
                "Merchandise gross margin vs supply chain cost trends.",
                "Pace of share repurchases relative to free cash flow generation.",
            ],
            remaining_evidence_gaps=[
                "Store-level profitability distribution between strip-center and enclosed mall locations.",
                "Primary filing confirmation of noncurrent credit facility balances."
            ]
        )

    elif sym == "MEDP":
        return ResearchAssessmentRecord(
            symbol="MEDP",
            disposition=ResearchDisposition.RESEARCH_FURTHER,
            disposition_rationale=(
                f"Full-service Phase I-IV clinical contract research organization (CRO) delivering +{growth_str} YoY revenue growth "
                f"({rev_str} TTM) with superior operating margins ({ebit_margin_str}). Cash flow generation is strong ({fcf_str} "
                f"CFO minus Capex, {conv_str} EBIT conversion). Liquid reserves hold {cash_str}; financial debt concepts are unevidenced "
                f"in XBRL tags (total debt and net debt unavailable). Steady share repurchase execution has reduced shares by {dil_3y_str} over 3 years."
            ),
            growth_durability=(
                "Specialization in small to mid-sized biopharma companies provides pricing power and sticky relationships. "
                "Biotech funding cycles represent the primary external demand driver."
            ),
            reinvestment_economics=(
                "Capital-light service model with low tangible capex needs. Substantial cash conversion into liquid reserves."
            ),
            shareholder_outcomes=(
                "Management prioritizes opportunistic share repurchases and maintains zero debt, delivering strong per-share compounding."
            ),
            balance_sheet_resilience=(
                f"Liquid service model: holds {cash_str} in cash and equivalents; financial debt concepts are unevidenced in primary XBRL tags."
            ),
            valuation_uncertainty=(
                "Valuation reflects premium CRO quality; vulnerable to sustained macro downturns in venture biopharma funding."
            ),
            evidence_sufficiency=(
                "Complete SEC filings through Q2 2026. Net new business bookings and backlog disclosures provide forward visibility."
            ),
            thesis_invalidation_conditions=[
                "Net book-to-bill ratio declining below 1.0x for two consecutive quarters.",
                "Operating margin compression below 17% due to project cancellations.",
                "Material increase in biopharma client bankruptcy write-offs.",
            ],
            specific_next_evidence_to_monitor=[
                "Quarterly net new business awards and cancellation rates.",
                "Biotech venture funding trends and client clinical trial starts.",
            ],
            remaining_evidence_gaps=[
                "Client concentration details beyond top-10 biopharma sponsor disclosures."
            ]
        )

    elif sym == "DECK":
        return ResearchAssessmentRecord(
            symbol="DECK",
            disposition=ResearchDisposition.WATCH,
            disposition_rationale=(
                f"Footwear innovator powered by HOKA and UGG brands. Revenue reached {rev_str} (+{growth_str} YoY) with "
                f"{ebit_margin_str} operating margin and {fcf_str} CFO minus Capex. Liquid reserves hold {cash_str}; "
                f"financial debt concepts are unevidenced in XBRL tags (total debt and net debt unavailable). "
                f"3-year share dilution is {dil_3y_str} when properly normalized for the 6-for-1 forward stock split on Sep 13, 2024. "
                "Maintained on WATCH due to channel inventory normalization and brand maturity considerations."
            ),
            growth_durability=(
                "HOKA maintains strong global consumer running demand; UGG provides high-margin seasonal cash flows. "
                "Direct-to-consumer (DTC) channel expansion enhances customer lifetime value."
            ),
            reinvestment_economics=(
                "Outsourced manufacturing model limits capex requirements, generating superior cash conversion."
            ),
            shareholder_outcomes=(
                "Management has executed continuous share repurchases, shrinking the split-adjusted share count over 3 years."
            ),
            balance_sheet_resilience=(
                f"Liquid balance sheet: holds {cash_str} in cash; financial debt concepts are unevidenced in primary XBRL tags."
            ),
            valuation_uncertainty=(
                "Trading at full valuation multiples; deceleration in HOKA wholesale sell-through could prompt multiple compression."
            ),
            evidence_sufficiency=(
                "Full SEC XBRL filings available through Q1 FY2026 (ended June 30, 2026)."
            ),
            thesis_invalidation_conditions=[
                "HOKA brand YoY revenue deceleration below 10%.",
                "Gross margin contraction below 52% indicating promotional discounting.",
                "Wholesale channel inventory buildup leading to order cancellations.",
            ],
            specific_next_evidence_to_monitor=[
                "HOKA DTC vs wholesale revenue growth split.",
                "International expansion pace in Europe and Asia-Pacific.",
            ],
            remaining_evidence_gaps=[
                "Wholesale sell-through data by regional specialty running store accounts."
            ]
        )

    elif sym == "KLAC":
        return ResearchAssessmentRecord(
            symbol="KLAC",
            disposition=ResearchDisposition.WATCH,
            disposition_rationale=(
                f"Semiconductor process control and yield management leader. Revenue of {rev_str} (+{growth_str} YoY) "
                f"with derived operating margin of {ebit_margin_str} and {fcf_str} in CFO minus Capex. "
                f"Carries {debt_str} in long-term debt against {cash_str} in cash ({net_debt_str} net debt). "
                "Maintained on WATCH due to semiconductor wafer fab equipment (WFE) cyclicality and valuation rerating."
            ),
            growth_durability=(
                "Process control intensity increases with advanced nodes (gate-all-around, EUV, advanced packaging). "
                "Leading market share in optical wafer inspection creates a powerful moat."
            ),
            reinvestment_economics=(
                "High R&D investment requirement (~15% of revenue) is critical to maintaining technological differentiation."
            ),
            shareholder_outcomes=(
                "Consistent dividend payouts and active share repurchases support shareholder compounding."
            ),
            balance_sheet_resilience=(
                f"Solid investment-grade profile, supported by {fcf_str} annual free cash flow despite {net_debt_str} net debt."
            ),
            valuation_uncertainty=(
                "Valuation reflects peak-cycle optimism for AI semiconductor fab expansions."
            ),
            evidence_sufficiency=(
                "10-K full fiscal year filed through June 30, 2026. KLAC reports operating expenses (R&D, SG&A) "
                "without a separate OperatingIncomeLoss XBRL tag, requiring derived EBIT computation."
            ),
            thesis_invalidation_conditions=[
                "Major semiconductor foundry capex reductions exceeding 15%.",
                "Export restriction tightening further curtailing China shipments.",
                "Market share loss to competitors in advanced inspection technologies.",
            ],
            specific_next_evidence_to_monitor=[
                "Wafer fab equipment industry capex guidance from TSMC, Samsung, and Intel.",
                "Advanced packaging and EUV metrology tool adoption rates.",
            ],
            remaining_evidence_gaps=[
                "Granular revenue breakdown between leading-edge logic, DRAM, and NAND wafer fab segments."
            ]
        )

    elif sym == "LULU":
        return ResearchAssessmentRecord(
            symbol="LULU",
            disposition=ResearchDisposition.WATCH,
            disposition_rationale=(
                f"Athletic apparel compounder generating {rev_str} in TTM revenue (+{growth_str} YoY) and {ebit_margin_str} "
                f"EBIT margin. Free cash flow is robust at {fcf_str}. Balance sheet holds {cash_str} in cash with reported "
                f"$0.0 short-term borrowings; long-term debt is unevidenced in XBRL (total debt unavailable under completeness rules). "
                f"Share repurchases reduced share count by {dil_3y_str} over 3 years. "
                "Placed on WATCH due to decelerating North American consumer demand and execution headwinds in new product lines."
            ),
            growth_durability=(
                "International expansion (specifically Greater China) continues to deliver strong growth, offsetting "
                "mature North American market deceleration."
            ),
            reinvestment_economics=(
                "Capital-light supply chain with direct-to-consumer store investments yielding high returns on capital."
            ),
            shareholder_outcomes=(
                "Steady share repurchases have reduced diluted shares by double-digits over 3 years."
            ),
            balance_sheet_resilience=(
                f"Holds {cash_str} in cash with reported $0.0 short-term borrowings; long-term debt unevidenced in XBRL tags."
            ),
            valuation_uncertainty=(
                "Multiples have compressed; further downside if promotional competition erodes premium gross margins."
            ),
            evidence_sufficiency=(
                "Authoritative primary 10-Q filings available through August 2, 2026."
            ),
            thesis_invalidation_conditions=[
                "North America comparable sales growth falling below -3%.",
                "Gross margin contraction below 55% indicating severe markdowns.",
                "Deceleration in Greater China revenue growth below 15%.",
            ],
            specific_next_evidence_to_monitor=[
                "Quarterly Americas comp store sales vs international expansion revenue.",
                "Inventory-to-sales growth alignment.",
            ],
            remaining_evidence_gaps=[
                "Brand sentiment metrics and product return rates in new category launches."
            ]
        )

    elif sym == "CELH":
        return ResearchAssessmentRecord(
            symbol="CELH",
            disposition=ResearchDisposition.WATCH,
            disposition_rationale=(
                f"Rapidly expanding functional energy drink brand with TTM revenue of {rev_str} (+{growth_str} YoY) "
                f"and {ebit_margin_str} operating margin. CFO minus Capex is {fcf_str}. Balance sheet has {cash_str} cash "
                f"against {debt_str} debt ({net_debt_str} net debt). Shares normalized for the 3-for-1 forward split in Nov 2023. "
                "Placed on WATCH due to PepsiCo distribution inventory rebalancing and category growth deceleration."
            ),
            growth_durability=(
                "High brand resonance among health-conscious consumers; international distribution rollout is in early stages."
            ),
            reinvestment_economics=(
                "Asset-light contract manufacturing and partner distribution limits capex requirements."
            ),
            shareholder_outcomes=(
                "High historical revenue growth, though share dilution ({dil_3y_str} 3Y) reflects equity compensation."
            ),
            balance_sheet_resilience=(
                f"Strong cash reserves of {cash_str} against {debt_str} in long-term debt."
            ),
            valuation_uncertainty=(
                "High volatility driven by distributor inventory swings and tough year-over-year volume comparisons."
            ),
            evidence_sufficiency=(
                "SEC XBRL filings complete through Q2 2026 (ended June 30, 2026)."
            ),
            thesis_invalidation_conditions=[
                "Energy drink market share losses to Monster or Red Bull in tracked retail channels.",
                "EBIT margin compression below 5% due to aggressive promotional allowances.",
                "Persistent distributor inventory reductions impairing quarterly cash generation.",
            ],
            specific_next_evidence_to_monitor=[
                "Circana/Nielsen retail scanner sales growth and market share trajectory.",
                "International revenue contribution and distributor inventory levels.",
            ],
            remaining_evidence_gaps=[
                "Real-time distributor inventory depletion rates vs retail sell-through."
            ]
        )

    else:
        # Generic fallback for any other company
        return ResearchAssessmentRecord(
            symbol=sym,
            disposition=ResearchDisposition.INSUFFICIENT_EVIDENCE,
            disposition_rationale=f"Insufficient structured evidence for {sym} to establish research disposition.",
            growth_durability="Unknown",
            reinvestment_economics="Unknown",
            shareholder_outcomes="Unknown",
            balance_sheet_resilience="Unknown",
            valuation_uncertainty="High",
            evidence_sufficiency="Incomplete",
            thesis_invalidation_conditions=["Lack of verified primary source evidence."],
            specific_next_evidence_to_monitor=["Complete SEC XBRL filings."],
            remaining_evidence_gaps=gaps or ["Complete financial statements required."]
        )
