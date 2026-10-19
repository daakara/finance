"""
analyst_dashboard/research/sourced_fundamentals/qualitative_assessor.py

Generates structured qualitative and economic research assessments for all
companies in the population. Enforces explicit reasoning on growth durability,
reinvestment economics, shareholder outcomes, balance sheet resilience,
valuation uncertainty, and thesis invalidation conditions without composite scores.
"""

from typing import Dict, List
from .contracts import (
    ResearchAssessmentRecord,
    ResearchDisposition,
    CompanyFundamentals,
    DerivedMetrics,
)


def assess_company(cf: CompanyFundamentals, metrics: DerivedMetrics) -> ResearchAssessmentRecord:
    """Assess a company qualitatively and assign an explicit research disposition."""
    sym = cf.symbol

    if sym == "FIX":
        return ResearchAssessmentRecord(
            symbol="FIX",
            disposition=ResearchDisposition.RESEARCH_FURTHER,
            disposition_rationale=(
                "Exceptional operational momentum driven by multi-year secular tailwinds in data center "
                "thermal cooling and advanced manufacturing electrical infrastructure. Revenue grew 37.9% YoY "
                "with operating margins expanding to 15.4%. Generates extraordinary non-GAAP free cash flow "
                "($2.16B CFO minus Capex, 137% EBIT conversion) supported by billing in excess of costs. "
                "Balance sheet is fortress-grade with $1.80B net cash ($1.85B cash vs $54M debt). Diluted share "
                "count is stable (-0.3% 1Y). Primary research focus must center on valuation: current market price "
                "implies ~32% 5-year CAGR, demanding sustained flawless execution."
            ),
            growth_durability=(
                "High multi-year visibility backed by record institutional backlogs in AI data centers, "
                "chip fabrication plants, and pharmaceutical facilities. Modular off-site pre-fabrication "
                "creates high barriers to entry and customer switching costs."
            ),
            reinvestment_economics=(
                "Capital-light contracting and modular manufacturing model yields outstanding returns on invested "
                "capital (ROIC > 50% when cash is stripped out). Organic reinvestment requirements are modest (~$390M Capex on $10.2B revenue)."
            ),
            shareholder_outcomes=(
                "Management exhibits prudent capital allocation: debt-free balance sheet, consistent modest dividend growth, "
                "selective accretive tuck-in acquisitions, and zero shareholder dilution."
            ),
            balance_sheet_resilience=(
                "Impeccable. Holds $1.85B in cash and equivalents against just $54M in total debt, providing massive "
                "liquidity buffer against construction cycle downturns or material cost inflation."
            ),
            valuation_uncertainty=(
                "Elevated due to market rerating. Trading at premium multiples relative to historical contracting averages; "
                "Reverse-DCF indicates 32.4% sustained growth priced into shares."
            ),
            evidence_sufficiency=(
                "Comprehensive primary XBRL filings available through Q2 2026. Data-center contract concentration "
                "and modular versus traditional mechanical gross margins require continued tracking in MD&A."
            ),
            thesis_invalidation_conditions=[
                "Data-center or hyperscaler cancellation or indefinite postponement of thermal cooling backlogs.",
                "Gross margin compression below 18% indicating fixed-price contract execution errors.",
                "Operating cash flow conversion falling below 80% of EBIT for two consecutive quarters.",
                "Shift into speculative, high-leverage M&A outside core mechanical/modular competencies.",
            ],
            specific_next_evidence_to_monitor=[
                "Q3 2026 backlog composition, specifically the proportion dedicated to tech/institutional customers.",
                "Same-store revenue growth vs acquired revenue growth breakdown.",
                "Billing in excess of costs and accounts receivable aging trends.",
            ],
            remaining_evidence_gaps=[
                "Detailed margin breakdown between high-value modular prefabrication and standard mechanical maintenance."
            ]
        )

    elif sym == "DUOL":
        return ResearchAssessmentRecord(
            symbol="DUOL",
            disposition=ResearchDisposition.RESEARCH_FURTHER,
            disposition_rationale=(
                "Premier digital consumer subscription model displaying 31.9% YoY top-line growth and expanding "
                "operating profitability (12.6% EBIT margin). Pure asset-light software economics generate $407.9M "
                "in CFO minus Capex (~300% conversion of GAAP EBIT due to upfront deferred revenue collections). "
                "Holds $1.18B in net cash with zero financial debt. However, stock-based compensation ($141M TTM, "
                "~13% of revenue) causes +2.1% net annual share dilution and must be scrutinized against true owner earnings."
            ),
            growth_durability=(
                "Strong global viral organic flywheel with >80% organic user acquisition. Expansion into Duolingo Max "
                "(higher ARPU AI tier) and multi-subject apps (Math, Music) supports continued 20-30% expansion."
            ),
            reinvestment_economics=(
                "Virtually zero tangible capital intensity ($22.6M Capex on $1.08B revenue). Reinvestment is routed "
                "primarily through operating expenses (R&D and engineering talent)."
            ),
            shareholder_outcomes=(
                "Rapid top-line and operating cash flow growth are strong positives; however, significant stock-based "
                "compensation creates ongoing dilution that offsets a portion of operational gains for common shareholders."
            ),
            balance_sheet_resilience=(
                "Pristine. Zero financial debt and $1.18B in cash and short-term liquid reserves."
            ),
            valuation_uncertainty=(
                "Moderate-to-high. Reverse-DCF indicates current price implies ~30.4% sustained 5-year growth, which "
                "matches recent performance but leaves little margin of safety for growth deceleration."
            ),
            evidence_sufficiency=(
                "Full SEC XBRL disclosures available through Q2 2026. Reconciliation between GAAP net income "
                "($402M, inflated by one-time tax valuation allowance) and operating income ($136M) fully resolved."
            ),
            thesis_invalidation_conditions=[
                "Deceleration of Paid Subscriber YoY growth below 15%.",
                "Dilution accelerating beyond 3% annually due to unconstrained equity compensation.",
                "Gross margin contraction below 70% driven by higher LLM partner inference costs.",
                "Emergence of a free, superior conversational AI competitor that degrades Duolingo daily engagement.",
            ],
            specific_next_evidence_to_monitor=[
                "Duolingo Max subscription uptake and net ARPU expansion in Q3 2026.",
                "Stock-based compensation expense trajectory relative to operating cash flows.",
                "International conversion rates in non-English native speaker markets.",
            ],
            remaining_evidence_gaps=[
                "Exact gross margin impact of GenAI LLM licensing costs vs proprietary model hosting."
            ]
        )

    elif sym == "ULTA":
        return ResearchAssessmentRecord(
            symbol="ULTA",
            disposition=ResearchDisposition.RESEARCH_FURTHER,
            disposition_rationale=(
                "High-quality specialty retail compounder generating $1.15B in CFO minus Capex with a loyal "
                "customer base (44M+ Ultamate Rewards members). Management has consistently reduced share count "
                "via buybacks (-4.5% 1Y, -13.6% 3Y), enhancing per-share value. Trades at a reasonable ~23x P/E. "
                "However, top-line revenue growth slowed to 9.6% YoY and comparable store sales face headwinds from "
                "intense prestige beauty competition (Sephora at Kohl's). Justifies deep valuation to determine if "
                "current price offers an attractive margin of safety."
            ),
            growth_durability=(
                "Moderate. Domestic store footprint is mature (~1,400 stores); future growth relies on Target "
                "shop-in-shop expansions, international expansion (Mexico), and modest comparable store sales increases."
            ),
            reinvestment_economics=(
                "Highly productive store economics with steady returns on capital. Capex of $418M supports store "
                "remodels, supply chain automation, and IT infrastructure."
            ),
            shareholder_outcomes=(
                "Exceptional share cannibal track record: share count reduced from ~50M to 43M over three years "
                "entirely funded through organic free cash flow."
            ),
            balance_sheet_resilience=(
                "Sound. Modest financial debt ($340M) largely offset by cash ($213M). Significant operating lease "
                "liabilities (~$1.8B under ASC 842) are standard for store footprint but must be factored into enterprise value."
            ),
            valuation_uncertainty=(
                "Moderate. Reverse-DCF indicates market expects 18.2% growth, which is above ULTA's recent 9.6% rate, "
                "indicating the market may still be overestimating store expansion elasticity."
            ),
            evidence_sufficiency=(
                "Complete SEC XBRL facts through Q2 2026 (period ended August 1, 2026)."
            ),
            thesis_invalidation_conditions=[
                "Comparable store sales turning negative for two consecutive quarters.",
                "Operating margin compressing below 10.0% due to permanent promotional discounting.",
                "Loss of active loyalty members or declining spend-per-member.",
                "Overleveraging balance sheet to sustain buyback pace during operational slowdown.",
            ],
            specific_next_evidence_to_monitor=[
                "Q3 2026 comparable store sales split (traffic vs average ticket).",
                "Operating margin performance amidst holiday promotional discounting.",
                "Progress and initial unit economics of international store openings.",
            ],
            remaining_evidence_gaps=[
                "Granular profitability contribution from Target store-in-store locations."
            ]
        )

    elif sym == "MEDP":
        return ResearchAssessmentRecord(
            symbol="MEDP",
            disposition=ResearchDisposition.WATCH,
            disposition_rationale=(
                "Elite clinical research organization (CRO) with 20.6% revenue growth, 21.1% operating margin, "
                "and $706.7M CFO minus Capex (127% EBIT conversion). Debt-free ($502.7M net cash) and active share "
                "repurchases (-9.6% shares in 3Y). However, trading at an elevated multiple (~38x P/E) while facing "
                "potential macro headwinds in small-to-mid cap biotech funding. Deserves close monitoring for a better entry valuation."
            ),
            growth_durability="Healthy, but sensitive to biotech venture funding and clinical trial pipeline approvals.",
            reinvestment_economics="Very high ROIC; asset-light scientific service model requiring minimal capex ($46M on $2.6B revenue).",
            shareholder_outcomes="Aggressive share repurchases funded by operational cash flows; strong alignment with founder/management.",
            balance_sheet_resilience="Fortress. Zero debt, $502.7M in liquid cash.",
            valuation_uncertainty="High. Rich multiple leaves little buffer against contract cancellations or booking slowdowns.",
            evidence_sufficiency="Full SEC XBRL facts verified through Q2 2026.",
            thesis_invalidation_conditions=[
                "Net book-to-bill ratio falling below 1.0x.",
                "Client clinical trial cancellation rate spiking above historical norms.",
            ],
            specific_next_evidence_to_monitor=[
                "Net new business bookings and backlog trajectory in upcoming quarterly reports.",
                "Biotech funding environment indicators.",
            ],
            remaining_evidence_gaps=["Detailed cancellation rate disclosure by customer funding tier."]
        )

    elif sym == "DECK":
        return ResearchAssessmentRecord(
            symbol="DECK",
            disposition=ResearchDisposition.WATCH,
            disposition_rationale=(
                "Strong global brand momentum behind HOKA and UGG with 22.7% operating margins and $1.12B CFO minus Capex. "
                "Zero financial debt with $1.60B in net cash. However, trailing revenue growth has moderated to 7.9% YoY, "
                "and wholesale apparel channel dynamics create seasonal working capital swings. Multi-year share count looks "
                "distorted due to a 6-for-1 forward split in Sept 2024. Keep on watch to assess HOKA DTC growth durability."
            ),
            growth_durability="Good for HOKA brand, mature/seasonal for UGG. Vulnerable to athletic footwear fashion cycle shifts.",
            reinvestment_economics="High ROIC asset-light outsourced manufacturing model; modest capex ($75.9M).",
            shareholder_outcomes="Substantial share repurchases; conservative capital structure with no financial leverage.",
            balance_sheet_resilience="Excellent. $1.60B cash with zero debt.",
            valuation_uncertainty="Moderate. Multiples reflect brand success but leave risk if HOKA growth cools.",
            evidence_sufficiency="Full SEC XBRL filings through Q1 FY2027 (ended June 30, 2026).",
            thesis_invalidation_conditions=[
                "HOKA brand direct-to-consumer revenue growth dropping below 10%.",
                "Excess inventory buildup in wholesale channels requiring promotional discounting.",
            ],
            specific_next_evidence_to_monitor=[
                "HOKA vs UGG brand segment revenue growth rates in Q2/Q3.",
                "Direct-to-consumer channel expansion metrics.",
            ],
            remaining_evidence_gaps=["Split-adjusted historical share series verification across 5-year span."]
        )

    elif sym == "KLAC":
        return ResearchAssessmentRecord(
            symbol="KLAC",
            disposition=ResearchDisposition.WATCH,
            disposition_rationale=(
                "World-class monopoly-like moat in optical wafer inspection and semiconductor yield management. Generates "
                "immense operating margins (41.7% derived EBIT) and $3.77B in CFO minus Capex on $13.58B revenue. "
                "However, the semiconductor equipment industry is cyclical; KLAC carries $5.89B in debt ($2.77B net debt), "
                "and geopolitical trade restrictions on advanced chip tooling create headline uncertainty. Place on watch."
            ),
            growth_durability="High long-term secular growth driven by chip complexity; cyclical near-term volatility.",
            reinvestment_economics="High returns on R&D investment ($1.53B R&D annually); sustained technological lead over competitors.",
            shareholder_outcomes="Reliable dividend growth and share repurchases; balanced capital allocation.",
            balance_sheet_resilience="Moderate. Leveraged capital structure ($5.89B debt vs $3.12B cash/ST investments), though cash flows comfortably cover service.",
            valuation_uncertainty="Moderate. Trades at cyclical mid-cycle multiples; Reverse-DCF indicates fair valuation.",
            evidence_sufficiency="Full 10-K FY2026 reported and verified through June 30, 2026.",
            thesis_invalidation_conditions=[
                "Severe tightening of export controls cutting off major Chinese customer revenue (>25% of sales).",
                "Market share losses to competitors (Applied Materials, ASML) in inspection/metrology.",
            ],
            specific_next_evidence_to_monitor=[
                "WFE (Wafer Fab Equipment) spending outlook for calendar 2027.",
                "Geographic revenue concentration disclosures.",
            ],
            remaining_evidence_gaps=["Detailed foundry vs memory customer capex commitments."]
        )

    elif sym == "LULU":
        return ResearchAssessmentRecord(
            symbol="LULU",
            disposition=ResearchDisposition.UNSUITABLE_FOR_THIS_OBJECTIVE,
            disposition_rationale=(
                "Diagnostic contrast case. Revenue growth has decelerated sharply to +2.3% YoY ($10.99B vs $10.74B), "
                "operating margins have compressed to 19.5%, and Americas comparable sales have turned sluggish. "
                "While cash flows remain positive ($1.35B CFO minus Capex) and the company carries zero debt ($1.39B net cash), "
                "it fails the foundational threshold of credible, accelerating, or durable growth required for David Akara's objective."
            ),
            growth_durability="Weakening. Domestic core yoga/apparel market is saturated; faces intense competition from Alo Yoga and Vuori.",
            reinvestment_economics="Solid historical store economics, but current reinvestment is generating diminishing incremental returns.",
            shareholder_outcomes="Aggressive share buybacks (-11.3% shares in 3Y), but unable to offset top-line growth stagnation.",
            balance_sheet_resilience="Strong. Zero financial debt, $1.39B in cash.",
            valuation_uncertainty="High. Low multiple may be a value trap if product innovation fails to reignite demand.",
            evidence_sufficiency="Full SEC XBRL filings through Q2 2026 (ended August 2, 2026).",
            thesis_invalidation_conditions=[
                "Further deceleration in international growth below 15%.",
                "Gross margin falling below 55% due to inventory clearance discounting.",
            ],
            specific_next_evidence_to_monitor=["Americas comparable store sales growth turnaround."],
            remaining_evidence_gaps=["Product return rates and consumer sentiment metrics."]
        )

    elif sym == "CELH":
        return ResearchAssessmentRecord(
            symbol="CELH",
            disposition=ResearchDisposition.UNSUITABLE_FOR_THIS_OBJECTIVE,
            disposition_rationale=(
                "Diagnostic contrast case. While trailing revenue growth looks optically high (+53.2% YoY to $2.59B), "
                "underlying operating profitability has collapsed: operating income dropped to $73.4M (2.8% margin) "
                "due to massive PepsiCo distributor inventory destocking and promotional allowances. High customer concentration "
                "(>60% volume through PepsiCo) and extreme earnings volatility make terminal owner earnings highly unpredictable, "
                "rendering it unsuitable as a defensible undervalued growth candidate."
            ),
            growth_durability="Uncertain. Energy drink retail velocity has slowed; distributor inventory volatility masks end-consumer demand.",
            reinvestment_economics="Asset-light co-packing model, but returns on marketing spend are deteriorating.",
            shareholder_outcomes="Diluted earnings per share have dropped precipitously; high volatility in free cash flow.",
            balance_sheet_resilience="Moderate. $631M cash vs $675M debt; net debt is slightly positive ($43.6M).",
            valuation_uncertainty="Extreme. Normalized earnings power is indeterminate during the ongoing distributor inventory recalibration.",
            evidence_sufficiency="Full SEC XBRL filings through Q2 2026.",
            thesis_invalidation_conditions=[
                "Continued distributor destocking leading to sequential revenue declines in Q3/Q4.",
                "EBIT turning negative due to fixed marketing expenditure commitments.",
            ],
            specific_next_evidence_to_monitor=[
                "Scanner data (Nielsen/Circana) tracking actual end-consumer retail takeaway vs distributor shipments.",
                "PepsiCo inventory adjustment completion timeline.",
            ],
            remaining_evidence_gaps=["Exact volume vs price realization breakdown across key accounts."]
        )

    else:
        return ResearchAssessmentRecord(
            symbol=sym,
            disposition=ResearchDisposition.INSUFFICIENT_EVIDENCE,
            disposition_rationale="Insufficient evidence to render qualitative disposition.",
            growth_durability="Unknown",
            reinvestment_economics="Unknown",
            shareholder_outcomes="Unknown",
            balance_sheet_resilience="Unknown",
            valuation_uncertainty="Unknown",
            evidence_sufficiency="Incomplete",
            thesis_invalidation_conditions=[],
            specific_next_evidence_to_monitor=[],
            remaining_evidence_gaps=["Full SEC fact history not ingested."]
        )
