"""
analyst_dashboard/research/sourced_fundamentals/valuation.py

Multi-scenario Discounted Free Cash Flow (FCFF) and Reverse-DCF valuation models.
Provides conservative, base, and optimistic scenario bounds, sensitivity matrices,
market-implied growth rate reverse-engineering, and explicit thesis invalidation conditions.
"""

import logging
from typing import Dict, Any, List, Optional
from .contracts import (
    ValuationScenario,
    DeepValuationAssessment,
    DerivedMetrics,
    CompanyFundamentals,
)

logger = logging.getLogger(__name__)


def run_fcff_model(
    base_revenue: float,
    current_shares: float,
    net_debt_and_leases: float,
    growth_5y_pct: float,
    operating_margin_pct: float,
    wacc_pct: float,
    terminal_growth_pct: float,
    tax_rate_pct: float = 21.0,
    reinvestment_rate_pct: float = 15.0,
) -> Dict[str, float]:
    """
    5-Year FCFF DCF model:
    FCFF_t = Revenue_t * EBIT_Margin * (1 - Tax_Rate) * (1 - Reinvestment_Rate)
    Terminal Value = FCFF_5 * (1 + g) / (WACC - g)
    Enterprise Value = PV(FCFF_1..5) + PV(Terminal_Value)
    Equity Value = Enterprise Value - Net Debt & Leases
    Per Share Value = Equity Value / Diluted Shares
    """
    growth = growth_5y_pct / 100.0
    ebit_margin = operating_margin_pct / 100.0
    wacc = wacc_pct / 100.0
    g_term = terminal_growth_pct / 100.0
    tax = tax_rate_pct / 100.0
    reinv = reinvestment_rate_pct / 100.0

    if wacc <= g_term:
        raise ValueError("WACC must be strictly greater than terminal growth rate")

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
    equity_val = ev - net_debt_and_leases
    per_share = equity_val / current_shares

    return {
        "pv_explicit": pv_cash_flows,
        "pv_terminal": pv_terminal_value,
        "enterprise_value": ev,
        "equity_value": equity_val,
        "per_share": per_share,
    }


def solve_reverse_dcf(
    target_price: float,
    base_revenue: float,
    current_shares: float,
    net_debt_and_leases: float,
    operating_margin_pct: float,
    wacc_pct: float,
    terminal_growth_pct: float,
    tax_rate_pct: float = 21.0,
    reinvestment_rate_pct: float = 15.0,
) -> float:
    """
    Numerically solves for the 5-year revenue CAGR required for per-share
    DCF value to equal target_price, holding other variables constant.
    """
    low = -0.40
    high = 1.00
    for _ in range(100):
        mid = (low + high) / 2.0
        res = run_fcff_model(
            base_revenue=base_revenue,
            current_shares=current_shares,
            net_debt_and_leases=net_debt_and_leases,
            growth_5y_pct=mid * 100.0,
            operating_margin_pct=operating_margin_pct,
            wacc_pct=wacc_pct,
            terminal_growth_pct=terminal_growth_pct,
            tax_rate_pct=tax_rate_pct,
            reinvestment_rate_pct=reinvestment_rate_pct,
        )
        if abs(res["per_share"] - target_price) < 0.01:
            return round(mid * 100.0, 2)
        if res["per_share"] < target_price:
            low = mid
        else:
            high = mid
    return round(mid * 100.0, 2)


def generate_sensitivity_matrix(
    base_revenue: float,
    current_shares: float,
    net_debt_and_leases: float,
    growth_5y_pct: float,
    operating_margin_pct: float,
    tax_rate_pct: float,
    reinvestment_rate_pct: float,
    wacc_rates: List[float] = [8.0, 9.0, 10.0, 11.0, 12.0],
    terminal_rates: List[float] = [2.0, 2.5, 3.0],
) -> Dict[str, Dict[str, float]]:
    """Generates 2D WACC vs Terminal Growth valuation sensitivity table."""
    matrix: Dict[str, Dict[str, float]] = {}
    for w in wacc_rates:
        w_key = f"WACC_{w:.1f}%"
        matrix[w_key] = {}
        for g in terminal_rates:
            g_key = f"g_{g:.1f}%"
            try:
                res = run_fcff_model(
                    base_revenue=base_revenue,
                    current_shares=current_shares,
                    net_debt_and_leases=net_debt_and_leases,
                    growth_5y_pct=growth_5y_pct,
                    operating_margin_pct=operating_margin_pct,
                    wacc_pct=w,
                    terminal_growth_pct=g,
                    tax_rate_pct=tax_rate_pct,
                    reinvestment_rate_pct=reinvestment_rate_pct,
                )
                matrix[w_key][g_key] = round(res["per_share"], 2)
            except Exception:
                matrix[w_key][g_key] = 0.0
    return matrix


def evaluate_deep_valuation_fix(
    cf: CompanyFundamentals,
    metrics: DerivedMetrics,
) -> DeepValuationAssessment:
    """Deep valuation package for FIX (Comfort Systems USA)."""
    rev = metrics.revenue_ttm or 10193978000.0
    shares = metrics.diluted_shares_weighted_avg or 35254000.0
    # Net Debt + Operating Leases: Net Debt = -$1.80B, Leases = $280M -> Net Bridge = -$1.52B (Net Cash)
    net_bridge = (metrics.net_debt or -1800000000.0) + (metrics.operating_lease_liabilities or 280000000.0)
    current_price = metrics.market_price or 1439.07
    tax_rate = 22.0
    reinvest = 15.0

    # 1. Conservative Scenario: Secular peak cycle; growth cools to 12%, margins compress to 12.5%, WACC 10.5%
    c_res = run_fcff_model(rev, shares, net_bridge, 12.0, 12.5, 10.5, 2.0, tax_rate, reinvest)
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
        key_assumptions="Data center buildout cools; modular pre-fabrication backlog normalizes; margin moderates toward historical median."
    )

    # 2. Base Scenario: Sustained AI/data-center buildout; 22% CAGR for 5 years, 15.0% margin, WACC 9.5%
    b_res = run_fcff_model(rev, shares, net_bridge, 22.0, 15.0, 9.5, 2.5, tax_rate, reinvest)
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

    # 3. Optimistic Scenario: Hyper-demand; 30% CAGR for 5 years, 16.5% margin, WACC 8.5%
    o_res = run_fcff_model(rev, shares, net_bridge, 30.0, 16.5, 8.5, 3.0, tax_rate, reinvest)
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
    rev_growth = solve_reverse_dcf(current_price, rev, shares, net_bridge, 15.0, 9.5, 2.5, tax_rate, reinvest)
    sens_matrix = generate_sensitivity_matrix(rev, shares, net_bridge, 22.0, 15.0, tax_rate, reinvest)

    return DeepValuationAssessment(
        symbol="FIX",
        framework_used="FCFF (Enterprise to Equity DCF)",
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
            "operating_margin_target_pct": 15.0,
            "wacc_pct": 9.5,
            "terminal_growth_pct": 2.5,
            "effective_tax_rate_pct": tax_rate,
            "reinvestment_rate_pct": reinvest,
            "net_debt_bridge_usd": net_bridge,
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
        ]
    )


def evaluate_deep_valuation_duol(
    cf: CompanyFundamentals,
    metrics: DerivedMetrics,
) -> DeepValuationAssessment:
    """Deep valuation package for DUOL (Duolingo, Inc.)."""
    rev = metrics.revenue_ttm or 1083778000.0
    shares = metrics.diluted_shares_weighted_avg or 50031000.0
    net_bridge = (metrics.net_debt or -1180000000.0) + (metrics.operating_lease_liabilities or 50000000.0)
    current_price = metrics.market_price or 141.76
    tax_rate = 21.0
    reinvest = 10.0  # Asset-light software model

    # 1. Conservative Scenario: Organic user growth plateaus; 18% CAGR, 14% operating margin, 11% WACC
    c_res = run_fcff_model(rev, shares, net_bridge, 18.0, 14.0, 11.0, 2.5, tax_rate, reinvest)
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
    b_res = run_fcff_model(rev, shares, net_bridge, 26.0, 18.0, 10.0, 3.0, tax_rate, reinvest)
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
    o_res = run_fcff_model(rev, shares, net_bridge, 34.0, 22.0, 9.0, 3.5, tax_rate, reinvest)
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
    rev_growth = solve_reverse_dcf(current_price, rev, shares, net_bridge, 18.0, 10.0, 3.0, tax_rate, reinvest)
    sens_matrix = generate_sensitivity_matrix(rev, shares, net_bridge, 26.0, 18.0, tax_rate, reinvest)

    return DeepValuationAssessment(
        symbol="DUOL",
        framework_used="FCFF (Enterprise to Equity DCF, Software Cash-Flow Normalized)",
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
            "operating_margin_target_pct": 18.0,
            "wacc_pct": 10.0,
            "terminal_growth_pct": 3.0,
            "effective_tax_rate_pct": tax_rate,
            "reinvestment_rate_pct": reinvest,
            "net_debt_bridge_usd": net_bridge,
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
        ]
    )


def evaluate_deep_valuation_ulta(
    cf: CompanyFundamentals,
    metrics: DerivedMetrics,
) -> DeepValuationAssessment:
    """Deep valuation package for ULTA (Ulta Beauty, Inc.)."""
    rev = metrics.revenue_ttm or 12640027000.0
    shares = metrics.diluted_shares_weighted_avg or 43062000.0
    # Net Debt + Lease Liabilities: Net Debt = $126M, Leases = ~$1.8B -> Net Bridge = $1.93B
    net_bridge = (metrics.net_debt or 126127000.0) + (metrics.operating_lease_liabilities or 1800000000.0)
    current_price = metrics.market_price or 592.94
    tax_rate = 23.5
    reinvest = 20.0

    # 1. Conservative Scenario: Comp store deceleration; 4.0% CAGR, 10.5% margin, 10.0% WACC
    c_res = run_fcff_model(rev, shares, net_bridge, 4.0, 10.5, 10.0, 1.5, tax_rate, reinvest)
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
        key_assumptions="Competition from Sephora (Kohl's) and mass retail intensifies; promotional discounting erodes gross margins; comp sales turn flat."
    )

    # 2. Base Scenario: Steady compounder; 8.0% CAGR, 12.0% margin, 9.0% WACC
    b_res = run_fcff_model(rev, shares, net_bridge, 8.0, 12.0, 9.0, 2.0, tax_rate, reinvest)
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
        key_assumptions="Target store-in-store concept expands; loyalty program (Ultamate Rewards 44M+ members) preserves wallet share; steady 3-4% share buyback."
    )

    # 3. Optimistic Scenario: International expansion & prestige acceleration; 12.0% CAGR, 13.5% margin, 8.0% WACC
    o_res = run_fcff_model(rev, shares, net_bridge, 12.0, 13.5, 8.0, 2.5, tax_rate, reinvest)
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
    rev_growth = solve_reverse_dcf(current_price, rev, shares, net_bridge, 12.0, 9.0, 2.0, tax_rate, reinvest)
    sens_matrix = generate_sensitivity_matrix(rev, shares, net_bridge, 8.0, 12.0, tax_rate, reinvest)

    return DeepValuationAssessment(
        symbol="ULTA",
        framework_used="FCFF (Enterprise to Equity DCF, ASC 842 Lease Adjusted)",
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
            "operating_margin_target_pct": 12.0,
            "wacc_pct": 9.0,
            "terminal_growth_pct": 2.0,
            "effective_tax_rate_pct": tax_rate,
            "reinvestment_rate_pct": reinvest,
            "net_debt_bridge_usd": net_bridge,
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
        ]
    )
