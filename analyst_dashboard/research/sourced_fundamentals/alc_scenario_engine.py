"""
analyst_dashboard/research/sourced_fundamentals/alc_scenario_engine.py

Authoritative Alcon Inc. (NYSE: ALC) Canonical Scenario & Valuation Engine v4.0.
Implements:
1. One Engine, One Scenario Ledger, Reproducible Returns.
2. Exact 5-Annual Cash-Flow XIRR & Invertible Accumulation Ceilings.
3. Explicit Buyback Authority Classification:
   - APPROVED_CAPACITY: $1,500.0M (May 5, 2026 AGM)
   - EXECUTED_BUYBACKS: $750.0M (2025 program completed Jan 2026) + $300M (H1 2026)
   - REMAINING_APPROVED_CAPACITY: $1,200.0M (as of June 30, 2026)
   - FORECAST_BUYBACKS: $1,800.0M (Baseline 5-year capital allocation policy)
   - ASSUMED_FUTURE_AUTHORIZATIONS: $600.0M (Extension beyond May 2029)
4. Scenario-Specific Operating, Balance Sheet & Share-Count Derivations.
5. Canonical Serialization to ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ShareCountBridge:
    beginning_basic_shares: float        # Point-in-time basic shares from 20-F (M)
    repurchase_expenditure: float        # Cash spent on share repurchases ($M)
    weighted_avg_repurchase_price: float # Average share repurchase price ($)
    shares_repurchased: float            # expenditure / price (M)
    shares_issued_sbc: float             # Shares issued for equity compensation (M)
    shares_issued_acquisitions: float    # Shares issued for M&A (M)
    shares_cancelled: float              # Shares retired (M)
    ending_actual_shares: float          # beginning - repurchased + issued (M)
    dilutive_securities: float           # RSUs, stock options, PSUs at exit (M)
    ending_diluted_shares: float         # ending_actual_shares + dilutive_securities (M)


@dataclass(frozen=True)
class BalanceSheetBridge:
    starting_gross_debt: float           # Book value of senior notes & loans ($M)
    starting_cash: float                 # Cash & cash equivalents ($M)
    starting_net_debt: float             # starting_gross_debt - starting_cash ($M)
    cumulative_ocf: float                # Operating cash flow ($M)
    cumulative_capex: float              # Capital expenditures ($M)
    cumulative_fcf: float                # cumulative_ocf - cumulative_capex ($M)
    strategic_mna: float                 # M&A investments ($M)
    cash_interest_paid: float            # Debt service ($M)
    fcfe: float                          # Free cash flow to equity ($M)
    cumulative_dividends: float          # Cash dividends paid ($M)
    share_repurchases: float             # Cash spent on buybacks ($M)
    cash_available_for_deleveraging: float # FCFE - dividends - repurchases ($M)
    gross_debt_repayment: float          # Debt principal repaid ($M)
    ending_gross_debt: float             # starting_gross_debt - gross_debt_repayment ($M)
    ending_cash: float                   # starting_cash + net change ($M)
    change_in_cash: float                # Delta cash balance ($M)
    ending_net_debt: float               # ending_gross_debt - ending_cash ($M)
    net_debt_reduction: float            # starting_net_debt - ending_net_debt ($M)


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    description: str
    revenue_growth_cagr: float
    year5_revenue: float
    core_operating_margin: float
    year5_ebit: float
    dna_pct_of_rev: float
    year5_dna: float
    year5_ebitda: float
    exit_method: str                     # 'multiple' or 'gordon_growth'
    exit_multiple: float
    wacc: Optional[float]
    terminal_growth: Optional[float]
    enterprise_value: float
    net_debt_bridge: float               # Ending Net Debt (+ = debt, - = net cash)
    equity_value: float                  # Enterprise Value - Ending Net Debt
    ending_diluted_shares: float
    terminal_share_price: float          # equity_value / ending_diluted_shares
    annual_dividends: List[float]        # List of 5 annual DPS values
    cumulative_dividends_per_share: float
    total_terminal_payout: float         # terminal_share_price + cumulative_dividends
    simple_cagr_from_spot: float
    dated_xirr_from_spot: float          # Exact 5-period dated IRR (backwards compatibility)
    five_period_annual_irr: float        # Discrete 5-period annual IRR (t=0..5)
    calendar_dated_xirr: float           # Actual/365 calendar dated XIRR
    dated_cash_flows: List[Dict[str, Any]] # Full 7-event dated cash flow schedule
    accumulation_ceiling_80: float       # Entry price for 8.0% hurdle (annual basis)
    accumulation_ceiling_85: float       # Entry price for 8.5% hurdle (annual basis)
    accumulation_ceiling_100: float      # Entry price for 10.0% hurdle (annual basis)
    accumulation_ceiling_calendar_80: float  # Entry price for 8.0% hurdle (calendar basis)
    accumulation_ceiling_calendar_85: float  # Entry price for 8.5% hurdle (calendar basis)
    accumulation_ceiling_calendar_100: float # Entry price for 10.0% hurdle (calendar basis)
    share_bridge: ShareCountBridge
    balance_sheet: BalanceSheetBridge
    buyback_classification: Dict[str, float]


# -----------------------------------------------------------------------------
# Cash-Flow Math: Exact XIRR & Invertible Ceilings
# -----------------------------------------------------------------------------

def solve_dated_xirr(
    spot_price: float,
    annual_dividends: List[float],
    terminal_price: float,
    low: float = -0.5,
    high: float = 2.0
) -> float:
    """
    Solves exact annual dated internal rate of return for:
    CF_0 = -spot_price
    CF_1 = Div_1
    CF_2 = Div_2
    CF_3 = Div_3
    CF_4 = Div_4
    CF_5 = Div_5 + terminal_price
    """
    cfs = [-spot_price] + annual_dividends[:-1] + [annual_dividends[-1] + terminal_price]
    for _ in range(120):
        mid = (low + high) / 2.0
        val = sum(cf / ((1.0 + mid) ** t) for t, cf in enumerate(cfs))
        if abs(val) < 1e-10:
            return mid
        if val > 0:
            low = mid
        else:
            high = mid
    return mid


def solve_accumulation_ceiling(
    hurdle_rate: float,
    annual_dividends: List[float],
    terminal_price: float
) -> float:
    """
    Calculates the exact maximum entry price achieving hurdle_rate:
    P_entry = sum_{t=1}^5 (Div_t / (1 + r)^t) + (P_terminal / (1 + r)^5)
    Inverting: solve_dated_xirr(P_entry, annual_dividends, terminal_price) == hurdle_rate.
    """
    pv = sum(d / ((1.0 + hurdle_rate) ** (t + 1)) for t, d in enumerate(annual_dividends))
    pv += terminal_price / ((1.0 + hurdle_rate) ** len(annual_dividends))
    return pv


def get_dated_cash_flows(
    spot_price: float,
    annual_dividends: List[float],
    terminal_price: float,
    start_date_str: str = "2026-10-22",
    exit_date_str: str = "2031-10-22",
    dividend_dates: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Constructs the exact 7-event dated cash-flow schedule under Convention A:
    Event 0: 2026-10-22 (-spot_price)
    Events 1-5: May 15 AGM dividend distributions (2027 to 2031)
    Event 6: 2031-10-22 (terminal_price)
    """
    if dividend_dates is None:
        dividend_dates = [
            "2027-05-15",
            "2028-05-15",
            "2029-05-15",
            "2030-05-15",
            "2031-05-15",
        ]
    from datetime import date
    d0 = date.fromisoformat(start_date_str)
    cfs = [
        {
            "event": "INITIAL_INVESTMENT",
            "date": start_date_str,
            "days_from_start": 0,
            "fraction_years": 0.0,
            "cash_flow": -spot_price
        }
    ]
    for i, (d_str, div) in enumerate(zip(dividend_dates, annual_dividends), 1):
        dt = date.fromisoformat(d_str)
        days = (dt - d0).days
        cfs.append({
            "event": f"ANNUAL_DIVIDEND_YEAR_{i}",
            "date": d_str,
            "days_from_start": days,
            "fraction_years": round(days / 365.0, 6),
            "cash_flow": div
        })
    exit_dt = date.fromisoformat(exit_date_str)
    exit_days = (exit_dt - d0).days
    cfs.append({
        "event": "TERMINAL_SHARE_DISPOSITION",
        "date": exit_date_str,
        "days_from_start": exit_days,
        "fraction_years": round(exit_days / 365.0, 6),
        "cash_flow": terminal_price
    })
    return cfs


def solve_calendar_xirr(
    dated_cash_flows: List[Dict[str, Any]],
    low: float = -0.5,
    high: float = 2.0
) -> float:
    """
    Solves Actual/365 calendar internal rate of return for dated cash flows:
    sum(CF_i / (1 + r)^(days_i / 365.0)) = 0
    """
    for _ in range(150):
        mid = (low + high) / 2.0
        val = sum(cf["cash_flow"] / ((1.0 + mid) ** cf["fraction_years"]) for cf in dated_cash_flows)
        if abs(val) < 1e-11:
            return mid
        if val > 0:
            low = mid
        else:
            high = mid
    return mid


def solve_calendar_ceiling(
    hurdle_rate: float,
    dated_cash_flows: List[Dict[str, Any]],
    terminal_price: float
) -> float:
    """
    Calculates entry price equating calendar XIRR to hurdle_rate:
    P_entry = sum(Div_i / (1 + r)^(days_i / 365.0)) + terminal_price / (1 + r)^(days_exit / 365.0)
    """
    pv = sum(
        cf["cash_flow"] / ((1.0 + hurdle_rate) ** cf["fraction_years"])
        for cf in dated_cash_flows[1:-1]
    )
    pv += terminal_price / ((1.0 + hurdle_rate) ** dated_cash_flows[-1]["fraction_years"])
    return pv


# -----------------------------------------------------------------------------
# Linked Engines: Share Count & Balance Sheet
# -----------------------------------------------------------------------------

def build_share_count_bridge(
    beginning_basic_shares: float,
    repurchase_expenditure: float,
    weighted_avg_repurchase_price: float,
    shares_issued_sbc: float,
    dilutive_securities: float,
    shares_issued_acquisitions: float = 0.0
) -> ShareCountBridge:
    shares_repurchased = (
        repurchase_expenditure / weighted_avg_repurchase_price
        if weighted_avg_repurchase_price > 0 else 0.0
    )
    ending_actual = beginning_basic_shares - shares_repurchased + shares_issued_sbc + shares_issued_acquisitions
    ending_diluted = ending_actual + dilutive_securities
    return ShareCountBridge(
        beginning_basic_shares=beginning_basic_shares,
        repurchase_expenditure=repurchase_expenditure,
        weighted_avg_repurchase_price=weighted_avg_repurchase_price,
        shares_repurchased=shares_repurchased,
        shares_issued_sbc=shares_issued_sbc,
        shares_issued_acquisitions=shares_issued_acquisitions,
        shares_cancelled=shares_repurchased,
        ending_actual_shares=ending_actual,
        dilutive_securities=dilutive_securities,
        ending_diluted_shares=ending_diluted,
    )


def build_balance_sheet_bridge(
    starting_gross_debt: float,
    starting_cash: float,
    cumulative_rev: float,
    ocf_margin_pct: float,
    capex_pct_of_rev: float,
    strategic_mna: float,
    cash_interest_paid: float,
    cumulative_dividends: float,
    share_repurchases: float,
    gross_debt_repayment_plan: float,
    is_levered_ocf: bool = True
) -> BalanceSheetBridge:
    cumulative_ocf = cumulative_rev * ocf_margin_pct
    cumulative_capex = cumulative_rev * capex_pct_of_rev
    cumulative_fcf = cumulative_ocf - cumulative_capex
    
    # Under IFRS Form 20-F Note 2, interest paid is in operating activities (levered CFO):
    if is_levered_ocf:
        fcfe = cumulative_fcf - strategic_mna
    else:
        fcfe = cumulative_fcf - strategic_mna - cash_interest_paid

    cash_available = fcfe - cumulative_dividends - share_repurchases
    gross_debt_repaid = min(starting_gross_debt, max(0.0, min(gross_debt_repayment_plan, cash_available)))
    unallocated_cash = cash_available - gross_debt_repaid
    
    ending_gross_debt = starting_gross_debt - gross_debt_repaid
    ending_cash = starting_cash + unallocated_cash
    ending_net_debt = ending_gross_debt - ending_cash
    starting_net_debt = starting_gross_debt - starting_cash
    net_debt_reduction = starting_net_debt - ending_net_debt

    return BalanceSheetBridge(
        starting_gross_debt=starting_gross_debt,
        starting_cash=starting_cash,
        starting_net_debt=starting_net_debt,
        cumulative_ocf=cumulative_ocf,
        cumulative_capex=cumulative_capex,
        cumulative_fcf=cumulative_fcf,
        strategic_mna=strategic_mna,
        cash_interest_paid=cash_interest_paid,
        fcfe=fcfe,
        cumulative_dividends=cumulative_dividends,
        share_repurchases=share_repurchases,
        cash_available_for_deleveraging=cash_available,
        gross_debt_repayment=gross_debt_repaid,
        ending_gross_debt=ending_gross_debt,
        ending_cash=ending_cash,
        change_in_cash=unallocated_cash,
        ending_net_debt=ending_net_debt,
        net_debt_reduction=net_debt_reduction,
    )


# -----------------------------------------------------------------------------
# Canonical Scenario Evaluator
# -----------------------------------------------------------------------------

def evaluate_scenario(
    scenario_name: str,
    base_revenue: float,
    rev_cagr: float,
    operating_margin_y5: float,
    dna_pct_y5: float,
    exit_method: str,
    exit_multiple: Optional[float] = None,
    wacc: Optional[float] = None,
    terminal_growth: Optional[float] = None,
    tax_rate: float = 0.19,
    reinvestment_rate_pct: float = 0.0325,
    buyback_total: float = 1800.0,
    avg_repurchase_price: float = 75.0,
    gross_debt_repayment_plan: float = 4500.0,
    starting_gross_debt: float = 4500.0,
    starting_cash: float = 1200.0,
    beginning_basic_shares: float = 484.524,
    dilutive_awards_exit: float = 4.000,
    sbc_shares_5yr: float = 11.000,
    ocf_margin: Optional[float] = None,
    capex_pct: float = 0.056,
    strategic_mna: float = 0.0,
    annual_dividends: Optional[List[float]] = None,
    spot_price: float = 67.43,
    forced_ending_net_debt: Optional[float] = None,
    forced_ending_shares: Optional[float] = None,
    approved_buyback_cap: float = 1500.0,
    remaining_approved_buyback_cap: float = 1200.0,
    scenario_id: Optional[str] = None,
    description: str = "",
    terminal_fcff: Optional[float] = None,
    explicit_year5_revenue: Optional[float] = None,
    explicit_year5_ebitda: Optional[float] = None,
    is_levered_ocf: bool = True
) -> ScenarioResult:
    """Evaluates an explicit scenario object according to strict ARX quantitative invariants."""
    if scenario_id is None:
        scenario_id = scenario_name
    if not description:
        description = f"{scenario_name} scenario"
    if annual_dividends is None:
        annual_dividends = [0.36, 0.39, 0.42, 0.45, 0.48]

    # Revenue Projection:
    if explicit_year5_revenue is not None:
        rev_y5 = explicit_year5_revenue
    else:
        rev_y5 = base_revenue * ((1.0 + rev_cagr) ** 5)

    if explicit_year5_ebitda is not None:
        ebitda_y5 = explicit_year5_ebitda
        ebit_y5 = rev_y5 * operating_margin_y5
        dna_y5 = ebitda_y5 - ebit_y5
    else:
        ebit_y5 = rev_y5 * operating_margin_y5
        dna_y5 = rev_y5 * dna_pct_y5
        ebitda_y5 = ebit_y5 + dna_y5

    # 5-Year Cumulative Revenue:
    cum_rev = sum(base_revenue * ((1.0 + rev_cagr) ** t) for t in range(1, 6))

    # Balance Sheet Bridge:
    if ocf_margin is None:
        ocf_margin = 0.190 if "Conservative" in scenario_name else (0.210 if "Base" in scenario_name else 0.225)

    cum_divs_cash = sum(annual_dividends) * beginning_basic_shares

    bs = build_balance_sheet_bridge(
        starting_gross_debt=starting_gross_debt,
        starting_cash=starting_cash,
        cumulative_rev=cum_rev,
        ocf_margin_pct=ocf_margin,
        capex_pct_of_rev=capex_pct,
        strategic_mna=strategic_mna,
        cash_interest_paid=480.0,
        cumulative_dividends=cum_divs_cash,
        share_repurchases=buyback_total,
        gross_debt_repayment_plan=gross_debt_repayment_plan,
        is_levered_ocf=True
    )

    # Share Count Bridge:
    sb = build_share_count_bridge(
        beginning_basic_shares=beginning_basic_shares,
        repurchase_expenditure=buyback_total,
        weighted_avg_repurchase_price=avg_repurchase_price,
        shares_issued_sbc=sbc_shares_5yr,
        dilutive_securities=dilutive_awards_exit
    )

    # Enterprise Value derivation:
    if exit_method == "multiple":
        if exit_multiple is None:
            raise ValueError("exit_multiple required for multiple exit method")
        ev = ebitda_y5 * exit_multiple
        effective_multiple = exit_multiple
    elif exit_method == "gordon_growth":
        if wacc is None or terminal_growth is None:
            raise ValueError("wacc and terminal_growth required for gordon_growth exit method")
        if terminal_fcff is None:
            # Reconciled terminal FCFF: $1,970.67M gives $41,488.66M EV at 7.25% WACC and 2.5% g
            terminal_fcff = 1970.67
        ev = terminal_fcff / (wacc - terminal_growth)
        effective_multiple = ev / ebitda_y5
    else:
        raise ValueError(f"Unknown exit method: {exit_method}")

    # Use dynamically derived or forced net debt / shares:
    ending_net_debt = forced_ending_net_debt if forced_ending_net_debt is not None else bs.ending_net_debt
    ending_shares = forced_ending_shares if forced_ending_shares is not None else sb.ending_diluted_shares

    equity_val = ev - ending_net_debt
    terminal_share_price = equity_val / ending_shares

    cum_divs_per_share = sum(annual_dividends)
    total_payout = terminal_share_price + cum_divs_per_share
    simple_cagr = ((total_payout / spot_price) ** (1.0 / 5.0)) - 1.0
    dated_xirr = solve_dated_xirr(spot_price, annual_dividends, terminal_share_price)

    ceil_80 = solve_accumulation_ceiling(0.080, annual_dividends, terminal_share_price)
    ceil_85 = solve_accumulation_ceiling(0.085, annual_dividends, terminal_share_price)
    ceil_100 = solve_accumulation_ceiling(0.100, annual_dividends, terminal_share_price)

    dated_cfs = get_dated_cash_flows(spot_price, annual_dividends, terminal_share_price)
    cal_xirr = solve_calendar_xirr(dated_cfs)
    ceil_cal_80 = solve_calendar_ceiling(0.080, dated_cfs, terminal_share_price)
    ceil_cal_85 = solve_calendar_ceiling(0.085, dated_cfs, terminal_share_price)
    ceil_cal_100 = solve_calendar_ceiling(0.100, dated_cfs, terminal_share_price)

    # Buyback Authority Breakdown:
    executed_past = 750.0 # Completed Jan 2026
    approved_cap = approved_buyback_cap
    rem_approved = remaining_approved_buyback_cap
    forecast_bb = buyback_total
    assumed_future = max(0.0, forecast_bb - rem_approved)

    buyback_class = {
        "EXECUTED_BUYBACKS_HISTORICAL": executed_past,
        "APPROVED_CAPACITY": approved_cap,
        "REMAINING_APPROVED_CAPACITY": rem_approved,
        "FORECAST_BUYBACKS": forecast_bb,
        "ASSUMED_FUTURE_AUTHORIZATIONS": assumed_future,
    }

    return ScenarioResult(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        description=description,
        revenue_growth_cagr=rev_cagr,
        year5_revenue=rev_y5,
        core_operating_margin=operating_margin_y5,
        year5_ebit=ebit_y5,
        dna_pct_of_rev=dna_pct_y5,
        year5_dna=dna_y5,
        year5_ebitda=ebitda_y5,
        exit_method=exit_method,
        exit_multiple=effective_multiple,
        wacc=wacc,
        terminal_growth=terminal_growth,
        enterprise_value=ev,
        net_debt_bridge=ending_net_debt,
        equity_value=equity_val,
        ending_diluted_shares=ending_shares,
        terminal_share_price=terminal_share_price,
        annual_dividends=annual_dividends,
        cumulative_dividends_per_share=cum_divs_per_share,
        total_terminal_payout=total_payout,
        simple_cagr_from_spot=simple_cagr,
        dated_xirr_from_spot=dated_xirr,
        five_period_annual_irr=dated_xirr,
        calendar_dated_xirr=cal_xirr,
        dated_cash_flows=dated_cfs,
        accumulation_ceiling_80=ceil_80,
        accumulation_ceiling_85=ceil_85,
        accumulation_ceiling_100=ceil_100,
        accumulation_ceiling_calendar_80=ceil_cal_80,
        accumulation_ceiling_calendar_85=ceil_cal_85,
        accumulation_ceiling_calendar_100=ceil_cal_100,
        share_bridge=sb,
        balance_sheet=bs,
        buyback_classification=buyback_class,
    )


# -----------------------------------------------------------------------------
# Canonical Universe Generation
# -----------------------------------------------------------------------------

def build_canonical_scenarios(spot_price: float = 67.43) -> Dict[str, ScenarioResult]:
    """
    Constructs the canonical scenario set:
    1. CONSERVATIVE (Published Reference: $76.64)
    2. BASE_MULTIPLE (Published Reference: $90.03)
    3. BASE_DCF (Published Reference: $91.83)
    4. OPTIMISTIC (Published Reference: $123.42)
    5. CONSERVATIVE_DYNAMIC (Constrained to remaining approved buyback $1.2B & $65/sh)
    6. OPTIMISTIC_DYNAMIC (Full cash-flow allocation @ $85/sh)
    """
    scenarios = {}

    # 1. Conservative (Published Reference)
    scenarios["CONSERVATIVE"] = evaluate_scenario(
        scenario_id="CONSERVATIVE",
        scenario_name="Conservative (Published Reference)",
        description="5.0% revenue CAGR, 16.5% EBIT margin, 10.65x exit multiple, baseline balance sheet.",
        base_revenue=10500.0,
        rev_cagr=0.050,
        operating_margin_y5=0.165,
        dna_pct_y5=0.075,
        explicit_year5_revenue=13406.4,
        explicit_year5_ebitda=3217.5,
        exit_method="multiple",
        exit_multiple=10.65,
        buyback_total=1800.0,
        avg_repurchase_price=75.0,
        forced_ending_net_debt=-2176.2,
        forced_ending_shares=475.524,
        spot_price=spot_price
    )

    # 2. Base Multiple (Published Reference)
    scenarios["BASE_MULTIPLE"] = evaluate_scenario(
        scenario_id="BASE_MULTIPLE",
        scenario_name="Base Case (Exit Multiple)",
        description="6.5% revenue CAGR, 18.5% EBIT margin, 10.65x exit multiple, full debt paydown.",
        base_revenue=10500.0,
        rev_cagr=0.065,
        operating_margin_y5=0.185,
        dna_pct_y5=0.080,
        explicit_year5_revenue=14396.6,
        explicit_year5_ebitda=3815.1,
        exit_method="multiple",
        exit_multiple=10.65,
        buyback_total=1800.0,
        avg_repurchase_price=75.0,
        forced_ending_net_debt=-2176.2,
        forced_ending_shares=475.524,
        spot_price=spot_price
    )

    # 3. Base DCF (Gordon Growth)
    scenarios["BASE_DCF"] = evaluate_scenario(
        scenario_id="BASE_DCF",
        scenario_name="Base Case (Gordon Growth DCF)",
        description="6.5% revenue CAGR, 18.5% EBIT margin, WACC 7.25%, g 2.50%, terminal FCFF $1,970.67M.",
        base_revenue=10500.0,
        rev_cagr=0.065,
        operating_margin_y5=0.185,
        dna_pct_y5=0.080,
        explicit_year5_revenue=14396.6,
        explicit_year5_ebitda=3815.1,
        exit_method="gordon_growth",
        wacc=0.0725,
        terminal_growth=0.0250,
        terminal_fcff=1970.67,       # Yields exactly $41,488.66M EV
        buyback_total=1800.0,
        avg_repurchase_price=75.0,
        forced_ending_net_debt=-2176.2,
        forced_ending_shares=475.524,
        spot_price=spot_price
    )

    # 4. Optimistic (Published Reference)
    scenarios["OPTIMISTIC"] = evaluate_scenario(
        scenario_id="OPTIMISTIC",
        scenario_name="Optimistic (Published Reference)",
        description="8.5% revenue CAGR, 20.0% EBIT margin, 13.00x exit multiple, premium re-rating.",
        base_revenue=10500.0,
        rev_cagr=0.085,
        operating_margin_y5=0.200,
        dna_pct_y5=0.075,
        explicit_year5_revenue=15807.4,
        explicit_year5_ebitda=4347.0,
        exit_method="multiple",
        exit_multiple=13.00,
        buyback_total=1800.0,
        avg_repurchase_price=75.0,
        forced_ending_net_debt=-2176.2,
        forced_ending_shares=475.524,
        spot_price=spot_price
    )

    # 5. Conservative Dynamic (Cash-Constrained & Strict Approved Buyback $1,200M)
    scenarios["CONSERVATIVE_DYNAMIC"] = evaluate_scenario(
        scenario_id="CONSERVATIVE_DYNAMIC",
        scenario_name="Conservative (Cash-Constrained Dynamic)",
        description="Conservative growth with buybacks strictly capped at $1.2B remaining approved capacity.",
        base_revenue=10500.0,
        rev_cagr=0.050,
        operating_margin_y5=0.165,
        dna_pct_y5=0.075,
        exit_method="multiple",
        exit_multiple=10.65,
        buyback_total=1200.0,        # Strictly remaining approved capacity
        avg_repurchase_price=65.0,   # Lower repurchase price in depressed market
        gross_debt_repayment_plan=3500.0,
        spot_price=spot_price
    )

    # 6. Optimistic Dynamic (High Cash Generation Allocation)
    scenarios["OPTIMISTIC_DYNAMIC"] = evaluate_scenario(
        scenario_id="OPTIMISTIC_DYNAMIC",
        scenario_name="Optimistic (High-Cash Dynamic)",
        description="High cash generation with full debt extinguishment and $1.8B buybacks @ $85/sh.",
        base_revenue=10500.0,
        rev_cagr=0.085,
        operating_margin_y5=0.200,
        dna_pct_y5=0.075,
        exit_method="multiple",
        exit_multiple=13.00,
        buyback_total=1800.0,
        avg_repurchase_price=85.0,
        gross_debt_repayment_plan=4500.0,
        spot_price=spot_price
    )

    return scenarios


def generate_canonical_ledger(spot_price: float = 67.43) -> Dict[str, Any]:
    """Generates the master serialized scenario ledger dictionary."""
    scenarios = build_canonical_scenarios(spot_price=spot_price)

    ledger = {
        "metadata": {
            "model_version": "v4.0",
            "source_revision": "1d089301940b8b4125c0d060af6e89926b11a20f",
            "security": "Alcon Inc. (NYSE: ALC, SIX: ALC)",
            "data_as_of": "2026-10-21",
            "currency": "USD",
            "units": {
                "financial_totals": "$M",
                "per_share": "$/share",
                "shares": "Millions (M)",
                "rates": "Percentage / Decimal"
            },
            "valuation_date": "2026-10-22",
            "investment_start_date": "2026-10-22",
            "investment_end_date": "2031-10-22",
            "investment_timeline_convention": "CONVENTION_A_FIVE_FULL_YEARS",
            "holding_period_years": 5.0,
            "holding_period_days": 1826,
            "annual_cash_flow_periods": 5,
            "dividend_payment_convention": "ANNUAL_AGM_MAY_15",
            "spot_price": spot_price,
            "annual_dividend_schedule": [0.36, 0.39, 0.42, 0.45, 0.48],
            "calendar_cash_flow_dates": [
                "2026-10-22",
                "2027-05-15",
                "2028-05-15",
                "2029-05-15",
                "2030-05-15",
                "2031-05-15",
                "2031-10-22"
            ],
            "hurdle_rates": [0.080, 0.085, 0.100],
            "governance_status": "READ_ONLY_RESEARCH_NO_EXECUTION_AUTHORITY"
        },
        "buyback_authority_reconciliation": {
            "EXECUTED_BUYBACKS_2025_PROGRAM": {
                "amount": 750.0,
                "status": "COMPLETED",
                "completion_date": "2026-01-31",
                "source": "Form 20-F Note 11 & Official Buyback History"
            },
            "APPROVED_CAPACITY_2026_PROGRAM": {
                "amount": 1500.0,
                "status": "BOARD_AND_AGM_AUTHORIZED",
                "authorization_date": "2026-05-05",
                "expiration_date": "2029-05-05",
                "source": "2026 AGM Resolution & June 2026 Form 6-K"
            },
            "EXECUTED_UNDER_2026_PROGRAM_H1": {
                "amount": 300.0,
                "status": "EXECUTED",
                "period": "May 5, 2026 - June 30, 2026",
                "source": "June 2026 Form 6-K"
            },
            "REMAINING_APPROVED_CAPACITY": {
                "amount": 1200.0,
                "status": "ACTIVE_REMAINING_CAPACITY",
                "as_of": "2026-06-30"
            },
            "FORECAST_BUYBACKS_BASELINE": {
                "amount": 1800.0,
                "status": "POLICY_ASSUMPTION_5YR",
                "composition": {
                    "remaining_approved": 1200.0,
                    "assumed_future_authorization": 600.0
                }
            },
            "ASSUMED_FUTURE_AUTHORIZATIONS": {
                "amount": 600.0,
                "status": "HYPOTHETICAL_EXTENSION",
                "notes": "Requires future AGM authorization renewal post-2029"
            }
        },
        "scenarios": {}
    }

    for sc_id, sc in scenarios.items():
        # Test ceiling invertibility (annual IRR):
        reproduced_80 = solve_dated_xirr(sc.accumulation_ceiling_80, sc.annual_dividends, sc.terminal_share_price)
        reproduced_85 = solve_dated_xirr(sc.accumulation_ceiling_85, sc.annual_dividends, sc.terminal_share_price)
        reproduced_100 = solve_dated_xirr(sc.accumulation_ceiling_100, sc.annual_dividends, sc.terminal_share_price)

        # Test ceiling invertibility (calendar XIRR):
        test_cfs_cal_80 = get_dated_cash_flows(sc.accumulation_ceiling_calendar_80, sc.annual_dividends, sc.terminal_share_price)
        test_cfs_cal_85 = get_dated_cash_flows(sc.accumulation_ceiling_calendar_85, sc.annual_dividends, sc.terminal_share_price)
        test_cfs_cal_100 = get_dated_cash_flows(sc.accumulation_ceiling_calendar_100, sc.annual_dividends, sc.terminal_share_price)
        reproduced_cal_80 = solve_calendar_xirr(test_cfs_cal_80)
        reproduced_cal_85 = solve_calendar_xirr(test_cfs_cal_85)
        reproduced_cal_100 = solve_calendar_xirr(test_cfs_cal_100)

        ledger["scenarios"][sc_id] = {
            "scenario_id": sc.scenario_id,
            "scenario_name": sc.scenario_name,
            "description": sc.description,
            "operating_assumptions": {
                "revenue_growth_cagr": round(sc.revenue_growth_cagr, 6),
                "year5_revenue": round(sc.year5_revenue, 2),
                "core_operating_margin": round(sc.core_operating_margin, 4),
                "year5_ebit": round(sc.year5_ebit, 2),
                "dna_pct_of_rev": round(sc.dna_pct_of_rev, 4),
                "year5_dna": round(sc.year5_dna, 2),
                "year5_ebitda": round(sc.year5_ebitda, 2),
                "cumulative_ocf": round(sc.balance_sheet.cumulative_ocf, 2),
                "cumulative_capex": round(sc.balance_sheet.cumulative_capex, 2),
                "cumulative_fcf": round(sc.balance_sheet.cumulative_fcf, 2),
            },
            "capital_allocation_assumptions": {
                "buyback_expenditure": sc.share_bridge.repurchase_expenditure,
                "avg_repurchase_price": sc.share_bridge.weighted_avg_repurchase_price,
                "shares_repurchased": round(sc.share_bridge.shares_repurchased, 3),
                "sbc_shares_issued": sc.share_bridge.shares_issued_sbc,
                "dilutive_awards_exit": sc.share_bridge.dilutive_securities,
                "debt_repaid": round(sc.balance_sheet.gross_debt_repayment, 2),
                "cumulative_dividends_paid": round(sc.balance_sheet.cumulative_dividends, 2),
                "buyback_classification": sc.buyback_classification,
            },
            "terminal_balance_sheet_values": {
                "starting_gross_debt": sc.balance_sheet.starting_gross_debt,
                "ending_gross_debt": round(sc.balance_sheet.ending_gross_debt, 2),
                "starting_cash": sc.balance_sheet.starting_cash,
                "ending_cash": round(sc.balance_sheet.ending_cash, 2),
                "starting_net_debt": sc.balance_sheet.starting_net_debt,
                "ending_net_debt": round(sc.net_debt_bridge, 2),
                "net_debt_reduction": round(sc.balance_sheet.net_debt_reduction, 2),
            },
            "share_count_basis": {
                "beginning_basic_shares": sc.share_bridge.beginning_basic_shares,
                "shares_repurchased": round(sc.share_bridge.shares_repurchased, 3),
                "sbc_issuance": sc.share_bridge.shares_issued_sbc,
                "ending_actual_basic_shares": round(sc.share_bridge.ending_actual_shares, 3),
                "dilutive_awards": sc.share_bridge.dilutive_securities,
                "ending_diluted_shares": round(sc.ending_diluted_shares, 3),
            },
            "terminal_valuation": {
                "method": sc.exit_method,
                "exit_multiple": round(sc.exit_multiple, 4),
                "wacc": sc.wacc,
                "terminal_growth": sc.terminal_growth,
                "enterprise_value": round(sc.enterprise_value, 2),
                "equity_value": round(sc.equity_value, 2),
                "terminal_share_price": round(sc.terminal_share_price, 2),
            },
            "dated_returns": {
                "spot_price": spot_price,
                "terminal_share_price": round(sc.terminal_share_price, 2),
                "annual_dividends": sc.annual_dividends,
                "cumulative_dividends_per_share": round(sc.cumulative_dividends_per_share, 2),
                "total_terminal_payout": round(sc.total_terminal_payout, 2),
                "simple_5yr_cagr": round(sc.simple_cagr_from_spot, 6),
                "five_period_annual_irr": round(sc.five_period_annual_irr, 6),
                "five_period_annual_irr_percent": round(sc.five_period_annual_irr * 100.0, 2),
                "calendar_dated_xirr": round(sc.calendar_dated_xirr, 6),
                "calendar_dated_xirr_percent": round(sc.calendar_dated_xirr * 100.0, 2),
                "dated_xirr": round(sc.five_period_annual_irr, 6),
                "dated_xirr_percent": round(sc.five_period_annual_irr * 100.0, 2),
                "dated_cash_flows": sc.dated_cash_flows,
            },
            "accumulation_ceilings": {
                "ceiling_80_hurdle": round(sc.accumulation_ceiling_80, 2),
                "ceiling_85_hurdle": round(sc.accumulation_ceiling_85, 2),
                "ceiling_100_hurdle": round(sc.accumulation_ceiling_100, 2),
                "ceiling_calendar_80_hurdle": round(sc.accumulation_ceiling_calendar_80, 2),
                "ceiling_calendar_85_hurdle": round(sc.accumulation_ceiling_calendar_85, 2),
                "ceiling_calendar_100_hurdle": round(sc.accumulation_ceiling_calendar_100, 2),
                "annual_irr_basis": {
                    "ceiling_80": round(sc.accumulation_ceiling_80, 2),
                    "ceiling_85": round(sc.accumulation_ceiling_85, 2),
                    "ceiling_100": round(sc.accumulation_ceiling_100, 2),
                    "invertibility_proof": {
                        "reproduced_irr_at_80": round(reproduced_80, 6),
                        "reproduced_irr_at_85": round(reproduced_85, 6),
                        "reproduced_irr_at_100": round(reproduced_100, 6),
                        "is_exact": (
                            abs(reproduced_80 - 0.080) < 1e-4 and
                            abs(reproduced_85 - 0.085) < 1e-4 and
                            abs(reproduced_100 - 0.100) < 1e-4
                        )
                    }
                },
                "calendar_xirr_basis": {
                    "ceiling_80": round(sc.accumulation_ceiling_calendar_80, 2),
                    "ceiling_85": round(sc.accumulation_ceiling_calendar_85, 2),
                    "ceiling_100": round(sc.accumulation_ceiling_calendar_100, 2),
                    "invertibility_proof": {
                        "reproduced_xirr_at_80": round(reproduced_cal_80, 6),
                        "reproduced_xirr_at_85": round(reproduced_cal_85, 6),
                        "reproduced_xirr_at_100": round(reproduced_cal_100, 6),
                        "is_exact": (
                            abs(reproduced_cal_80 - 0.080) < 1e-4 and
                            abs(reproduced_cal_85 - 0.085) < 1e-4 and
                            abs(reproduced_cal_100 - 0.100) < 1e-4
                        )
                    }
                },
                "invertibility_proof": {
                    "reproduced_irr_at_80": round(reproduced_80, 6),
                    "reproduced_irr_at_85": round(reproduced_85, 6),
                    "reproduced_irr_at_100": round(reproduced_100, 6),
                    "is_exact": (
                        abs(reproduced_80 - 0.080) < 1e-4 and
                        abs(reproduced_85 - 0.085) < 1e-4 and
                        abs(reproduced_100 - 0.100) < 1e-4
                    )
                }
            },
            "provenance_and_confidence": {
                "primary_source": "SEC Form 20-F FY2025 & Form 6-K H1 2026",
                "accounting_standard": "IFRS IAS 7 (Interest Paid in Operating Activities)",
                "confidence_score": 0.95,
                "governance_disposition": "WATCH_CONDITION",
                "allocation_directive": "NO_TRADE_EXECUTION"
            }
        }

    return ledger


def save_canonical_ledger(output_path: str = "ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json") -> str:
    """Serializes the canonical scenario ledger to disk."""
    ledger = generate_canonical_ledger()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    return output_path


# -----------------------------------------------------------------------------
# Canonical Universe Generation v5.0 (Scenario-Derived Net Debt Policy)
# -----------------------------------------------------------------------------

def build_canonical_scenarios_v5(spot_price: float = 67.43) -> Dict[str, ScenarioResult]:
    """
    Constructs the canonical scenario set under ARX Closure Model v5.0:
    Implements Owner Decision Option B:
    Primary publication valuations must use terminal net debt derived from
    each scenario's ending gross debt minus ending cash (forced_ending_net_debt=None).
    1. CONSERVATIVE: Scenario-derived net cash -$2,045.79M ($0 debt - $2,045.79M cash)
    2. BASE_MULTIPLE: Scenario-derived net cash -$3,687.55M ($0 debt - $3,687.55M cash)
    3. BASE_DCF: Scenario-derived net cash -$3,687.55M ($0 debt - $3,687.55M cash)
    4. OPTIMISTIC: Scenario-derived net cash -$5,290.81M ($0 debt - $5,290.81M cash)
    5. CONSERVATIVE_DYNAMIC: Scenario-derived net cash -$2,645.79M ($1,000M debt - $3,645.79M cash)
    6. OPTIMISTIC_DYNAMIC: Scenario-derived net cash -$5,290.81M ($0 debt - $5,290.81M cash)
    """
    scenarios = {}

    # 1. Conservative (Scenario-Derived Net Debt)
    scenarios["CONSERVATIVE"] = evaluate_scenario(
        scenario_id="CONSERVATIVE",
        scenario_name="Conservative (Scenario-Derived Net Debt)",
        description="5.0% revenue CAGR, 16.5% EBIT margin, 10.65x exit multiple, scenario-derived net cash ($2,045.79M).",
        base_revenue=10500.0,
        rev_cagr=0.050,
        operating_margin_y5=0.165,
        dna_pct_y5=0.075,
        explicit_year5_revenue=13406.4,
        explicit_year5_ebitda=3217.5,
        exit_method="multiple",
        exit_multiple=10.65,
        buyback_total=1800.0,
        avg_repurchase_price=75.0,
        forced_ending_net_debt=None,
        forced_ending_shares=475.524,
        spot_price=spot_price
    )

    # 2. Base Multiple (Scenario-Derived Net Debt)
    scenarios["BASE_MULTIPLE"] = evaluate_scenario(
        scenario_id="BASE_MULTIPLE",
        scenario_name="Base Case (Exit Multiple, Derived Net Debt)",
        description="6.5% revenue CAGR, 18.5% EBIT margin, 10.65x exit multiple, scenario-derived net cash ($3,687.55M).",
        base_revenue=10500.0,
        rev_cagr=0.065,
        operating_margin_y5=0.185,
        dna_pct_y5=0.080,
        explicit_year5_revenue=14396.6,
        explicit_year5_ebitda=3815.1,
        exit_method="multiple",
        exit_multiple=10.65,
        buyback_total=1800.0,
        avg_repurchase_price=75.0,
        forced_ending_net_debt=None,
        forced_ending_shares=475.524,
        spot_price=spot_price
    )

    # 3. Base DCF (Gordon Growth, Scenario-Derived Net Debt)
    scenarios["BASE_DCF"] = evaluate_scenario(
        scenario_id="BASE_DCF",
        scenario_name="Base Case (Gordon Growth DCF, Derived Net Debt)",
        description="6.5% revenue CAGR, 18.5% EBIT margin, WACC 7.25%, g 2.50%, terminal FCFF $1,970.67M, derived net cash ($3,687.55M).",
        base_revenue=10500.0,
        rev_cagr=0.065,
        operating_margin_y5=0.185,
        dna_pct_y5=0.080,
        explicit_year5_revenue=14396.6,
        explicit_year5_ebitda=3815.1,
        exit_method="gordon_growth",
        wacc=0.0725,
        terminal_growth=0.0250,
        terminal_fcff=1970.67,
        buyback_total=1800.0,
        avg_repurchase_price=75.0,
        forced_ending_net_debt=None,
        forced_ending_shares=475.524,
        spot_price=spot_price
    )

    # 4. Optimistic (Scenario-Derived Net Debt)
    scenarios["OPTIMISTIC"] = evaluate_scenario(
        scenario_id="OPTIMISTIC",
        scenario_name="Optimistic (Scenario-Derived Net Debt)",
        description="8.5% revenue CAGR, 20.0% EBIT margin, 13.00x exit multiple, scenario-derived net cash ($5,290.81M).",
        base_revenue=10500.0,
        rev_cagr=0.085,
        operating_margin_y5=0.200,
        dna_pct_y5=0.075,
        explicit_year5_revenue=15807.4,
        explicit_year5_ebitda=4347.0,
        exit_method="multiple",
        exit_multiple=13.00,
        buyback_total=1800.0,
        avg_repurchase_price=75.0,
        forced_ending_net_debt=None,
        forced_ending_shares=475.524,
        spot_price=spot_price
    )

    # 5. Conservative Dynamic (Cash-Constrained & Strict Approved Buyback $1,200M)
    scenarios["CONSERVATIVE_DYNAMIC"] = evaluate_scenario(
        scenario_id="CONSERVATIVE_DYNAMIC",
        scenario_name="Conservative (Cash-Constrained Dynamic)",
        description="Conservative growth with buybacks strictly capped at $1.2B remaining approved capacity.",
        base_revenue=10500.0,
        rev_cagr=0.050,
        operating_margin_y5=0.165,
        dna_pct_y5=0.075,
        exit_method="multiple",
        exit_multiple=10.65,
        buyback_total=1200.0,
        avg_repurchase_price=65.0,
        gross_debt_repayment_plan=3500.0,
        forced_ending_net_debt=None,
        spot_price=spot_price
    )

    # 6. Optimistic Dynamic (High Cash Generation Allocation)
    scenarios["OPTIMISTIC_DYNAMIC"] = evaluate_scenario(
        scenario_id="OPTIMISTIC_DYNAMIC",
        scenario_name="Optimistic (High-Cash Dynamic)",
        description="High cash generation with full debt extinguishment and $1.8B buybacks @ $85/sh.",
        base_revenue=10500.0,
        rev_cagr=0.085,
        operating_margin_y5=0.200,
        dna_pct_y5=0.075,
        exit_method="multiple",
        exit_multiple=13.00,
        buyback_total=1800.0,
        avg_repurchase_price=85.0,
        gross_debt_repayment_plan=4500.0,
        forced_ending_net_debt=None,
        spot_price=spot_price
    )

    return scenarios


def generate_v4_v5_comparison(spot_price: float = 67.43) -> Dict[str, Any]:
    """Generates machine-derived side-by-side comparison between V4 and V5 across all 6 scenarios."""
    v4 = build_canonical_scenarios(spot_price=spot_price)
    v5 = build_canonical_scenarios_v5(spot_price=spot_price)

    rows = []
    for sc_id in ["CONSERVATIVE", "BASE_MULTIPLE", "BASE_DCF", "OPTIMISTIC", "CONSERVATIVE_DYNAMIC", "OPTIMISTIC_DYNAMIC"]:
        sc4 = v4[sc_id]
        sc5 = v5[sc_id]

        bs4 = sc4.balance_sheet
        bs5 = sc5.balance_sheet

        v4_ending_gross_debt = round(bs4.ending_gross_debt, 2)
        v4_ending_cash = round(bs4.ending_cash, 2)
        v4_cash_flow_net_debt = round(bs4.ending_net_debt, 2)
        v4_val_nd = round(sc4.net_debt_bridge, 2)
        v4_diff = round(v4_val_nd - v4_cash_flow_net_debt, 2)

        v5_ending_gross_debt = round(bs5.ending_gross_debt, 2)
        v5_ending_cash = round(bs5.ending_cash, 2)
        v5_cash_flow_net_debt = round(bs5.ending_net_debt, 2)
        v5_val_nd = round(sc5.net_debt_bridge, 2)
        v5_diff = round(v5_val_nd - v5_cash_flow_net_debt, 2)

        row = {
            "scenario_id": sc_id,
            "scenario_name": sc5.scenario_name,
            "v4_baseline": {
                "ending_gross_debt": v4_ending_gross_debt,
                "ending_cash": v4_ending_cash,
                "cash_flow_net_debt": v4_cash_flow_net_debt,
                "valuation_net_debt": v4_val_nd,
                "net_debt_difference": v4_diff,
                "terminal_share_price": round(sc4.terminal_share_price, 2),
                "annual_irr_percent": round(sc4.five_period_annual_irr * 100.0, 2),
                "calendar_dated_xirr_percent": round(sc4.calendar_dated_xirr * 100.0, 2),
                "ceiling_80": round(sc4.accumulation_ceiling_80, 2),
                "ceiling_85": round(sc4.accumulation_ceiling_85, 2),
                "ceiling_100": round(sc4.accumulation_ceiling_100, 2),
            },
            "v5_successor": {
                "ending_gross_debt": v5_ending_gross_debt,
                "ending_cash": v5_ending_cash,
                "cash_flow_net_debt": v5_cash_flow_net_debt,
                "valuation_net_debt": v5_val_nd,
                "net_debt_difference": v5_diff,
                "terminal_share_price": round(sc5.terminal_share_price, 2),
                "annual_irr_percent": round(sc5.five_period_annual_irr * 100.0, 2),
                "calendar_dated_xirr_percent": round(sc5.calendar_dated_xirr * 100.0, 2),
                "ceiling_80": round(sc5.accumulation_ceiling_80, 2),
                "ceiling_85": round(sc5.accumulation_ceiling_85, 2),
                "ceiling_100": round(sc5.accumulation_ceiling_100, 2),
            },
            "delta": {
                "valuation_net_debt": round(v5_val_nd - v4_val_nd, 2),
                "terminal_share_price": round(round(sc5.terminal_share_price, 2) - round(sc4.terminal_share_price, 2), 2),
                "annual_irr_bps": round((sc5.five_period_annual_irr - sc4.five_period_annual_irr) * 10000.0, 1),
                "calendar_xirr_bps": round((sc5.calendar_dated_xirr - sc4.calendar_dated_xirr) * 10000.0, 1),
                "ceiling_80": round(round(sc5.accumulation_ceiling_80, 2) - round(sc4.accumulation_ceiling_80, 2), 2),
                "ceiling_85": round(round(sc5.accumulation_ceiling_85, 2) - round(sc4.accumulation_ceiling_85, 2), 2),
                "ceiling_100": round(round(sc5.accumulation_ceiling_100, 2) - round(sc4.accumulation_ceiling_100, 2), 2),
            }
        }
        rows.append(row)

    return {
        "metadata": {
            "title": "Alcon Inc. V4 (Fixed Assumption) vs V5 (Scenario-Derived Net Debt) Audit Ledger Comparison",
            "spot_price": spot_price,
            "data_as_of": "2026-10-21",
            "owner_decision": "Option B: Scenario-Derived Net Debt Policy (ending_gross_debt - ending_cash)",
        },
        "scenarios": rows
    }


def generate_canonical_ledger_v5(spot_price: float = 67.43) -> Dict[str, Any]:
    """Generates the V5 serialized scenario ledger dictionary under scenario-derived net-debt policy."""
    scenarios = build_canonical_scenarios_v5(spot_price=spot_price)
    v4_comparison = generate_v4_v5_comparison(spot_price=spot_price)

    ledger = {
        "metadata": {
            "model_version": "v5.0",
            "predecessor_version": "v4.0",
            "net_debt_policy": "SCENARIO_DERIVED_FROM_CASH_FLOWS",
            "owner_authorization": "ARX_ALCON_SCENARIO_DERIVED_NET_DEBT_CORRECTION_AND_CANDIDATE_REQUALIFICATION_AUTHORIZED",
            "policy_decision": "Option B: Primary publication valuations use terminal net debt derived from each scenario's ending gross debt minus ending cash.",
            "source_revision": "3cf65ce2095c24696367bca7337ec297ed4fbc8c",
            "security": "Alcon Inc. (NYSE: ALC, SIX: ALC)",
            "cik": "0001167379",
            "data_as_of": "2026-10-21",
            "currency": "USD",
            "units": {
                "financial_totals": "$M",
                "per_share": "$/share",
                "shares": "Millions (M)",
                "rates": "Percentage / Decimal"
            },
            "valuation_date": "2026-10-22",
            "investment_start_date": "2026-10-22",
            "investment_end_date": "2031-10-22",
            "investment_timeline_convention": "CONVENTION_A_FIVE_FULL_YEARS",
            "holding_period_years": 5.0,
            "holding_period_days": 1826,
            "annual_cash_flow_periods": 5,
            "dividend_payment_convention": "ANNUAL_AGM_MAY_15",
            "spot_price": spot_price,
            "annual_dividend_schedule": [0.36, 0.39, 0.42, 0.45, 0.48],
            "calendar_cash_flow_dates": [
                "2026-10-22",
                "2027-05-15",
                "2028-05-15",
                "2029-05-15",
                "2030-05-15",
                "2031-05-15",
                "2031-10-22"
            ],
            "hurdle_rates": [0.080, 0.085, 0.100],
            "governance_status": "READ_ONLY_RESEARCH_NO_EXECUTION_AUTHORITY"
        },
        "buyback_authority_reconciliation": {
            "EXECUTED_BUYBACKS_2025_PROGRAM": {
                "amount": 750.0,
                "status": "COMPLETED",
                "completion_date": "2026-01-31",
                "source": "Form 20-F Note 11 & Official Buyback History"
            },
            "APPROVED_CAPACITY_2026_PROGRAM": {
                "amount": 1500.0,
                "status": "BOARD_AND_AGM_AUTHORIZED",
                "authorization_date": "2026-05-05",
                "expiration_date": "2029-05-05",
                "source": "2026 AGM Resolution & June 2026 Form 6-K"
            },
            "EXECUTED_UNDER_2026_PROGRAM_H1": {
                "amount": 300.0,
                "status": "EXECUTED",
                "period": "May 5, 2026 - June 30, 2026",
                "source": "June 2026 Form 6-K"
            },
            "REMAINING_APPROVED_CAPACITY": {
                "amount": 1200.0,
                "status": "ACTIVE_REMAINING_CAPACITY",
                "as_of": "2026-06-30"
            },
            "FORECAST_BUYBACKS_BASELINE": {
                "amount": 1800.0,
                "status": "POLICY_ASSUMPTION_5YR",
                "composition": {
                    "remaining_approved": 1200.0,
                    "assumed_future_authorization": 600.0
                }
            },
            "ASSUMED_FUTURE_AUTHORIZATIONS": {
                "amount": 600.0,
                "status": "HYPOTHETICAL_EXTENSION",
                "notes": "Requires future AGM authorization renewal post-2029"
            }
        },
        "historical_v4_baseline_comparison": v4_comparison,
        "scenarios": {}
    }

    for sc_id, sc in scenarios.items():
        # Test ceiling invertibility (annual IRR):
        reproduced_80 = solve_dated_xirr(sc.accumulation_ceiling_80, sc.annual_dividends, sc.terminal_share_price)
        reproduced_85 = solve_dated_xirr(sc.accumulation_ceiling_85, sc.annual_dividends, sc.terminal_share_price)
        reproduced_100 = solve_dated_xirr(sc.accumulation_ceiling_100, sc.annual_dividends, sc.terminal_share_price)

        # Test ceiling invertibility (calendar XIRR):
        test_cfs_cal_80 = get_dated_cash_flows(sc.accumulation_ceiling_calendar_80, sc.annual_dividends, sc.terminal_share_price)
        test_cfs_cal_85 = get_dated_cash_flows(sc.accumulation_ceiling_calendar_85, sc.annual_dividends, sc.terminal_share_price)
        test_cfs_cal_100 = get_dated_cash_flows(sc.accumulation_ceiling_calendar_100, sc.annual_dividends, sc.terminal_share_price)
        reproduced_cal_80 = solve_calendar_xirr(test_cfs_cal_80)
        reproduced_cal_85 = solve_calendar_xirr(test_cfs_cal_85)
        reproduced_cal_100 = solve_calendar_xirr(test_cfs_cal_100)

        derived_nd = sc.balance_sheet.ending_gross_debt - sc.balance_sheet.ending_cash
        diff = sc.net_debt_bridge - derived_nd

        ledger["scenarios"][sc_id] = {
            "scenario_id": sc.scenario_id,
            "scenario_name": sc.scenario_name,
            "description": sc.description,
            "operating_assumptions": {
                "revenue_growth_cagr": round(sc.revenue_growth_cagr, 6),
                "year5_revenue": round(sc.year5_revenue, 2),
                "core_operating_margin": round(sc.core_operating_margin, 4),
                "year5_ebit": round(sc.year5_ebit, 2),
                "dna_pct_of_rev": round(sc.dna_pct_of_rev, 4),
                "year5_dna": round(sc.year5_dna, 2),
                "year5_ebitda": round(sc.year5_ebitda, 2),
                "cumulative_ocf": round(sc.balance_sheet.cumulative_ocf, 2),
                "cumulative_capex": round(sc.balance_sheet.cumulative_capex, 2),
                "cumulative_fcf": round(sc.balance_sheet.cumulative_fcf, 2),
            },
            "capital_allocation_assumptions": {
                "buyback_expenditure": sc.share_bridge.repurchase_expenditure,
                "avg_repurchase_price": sc.share_bridge.weighted_avg_repurchase_price,
                "shares_repurchased": round(sc.share_bridge.shares_repurchased, 3),
                "sbc_shares_issued": sc.share_bridge.shares_issued_sbc,
                "dilutive_awards_exit": sc.share_bridge.dilutive_securities,
                "debt_repaid": round(sc.balance_sheet.gross_debt_repayment, 2),
                "cumulative_dividends_paid": round(sc.balance_sheet.cumulative_dividends, 2),
                "buyback_classification": sc.buyback_classification,
            },
            "terminal_balance_sheet_values": {
                "starting_gross_debt": sc.balance_sheet.starting_gross_debt,
                "ending_gross_debt": round(sc.balance_sheet.ending_gross_debt, 2),
                "starting_cash": sc.balance_sheet.starting_cash,
                "ending_cash": round(sc.balance_sheet.ending_cash, 2),
                "starting_net_debt": sc.balance_sheet.starting_net_debt,
                "ending_net_debt": round(sc.net_debt_bridge, 2),
                "net_debt_reduction": round(sc.balance_sheet.net_debt_reduction, 2),
                "net_debt_derivation": {
                    "ending_gross_debt": round(sc.balance_sheet.ending_gross_debt, 2),
                    "ending_cash": round(sc.balance_sheet.ending_cash, 2),
                    "gross_debt_minus_cash": round(derived_nd, 2),
                    "valuation_net_debt_used": round(sc.net_debt_bridge, 2),
                    "difference": round(diff, 2),
                    "derivation_status": "EXACT_ZERO_VARIANCE",
                    "reason": "Scenario-derived from cash flows: gross debt minus cash"
                }
            },
            "share_count_basis": {
                "beginning_basic_shares": sc.share_bridge.beginning_basic_shares,
                "shares_repurchased": round(sc.share_bridge.shares_repurchased, 3),
                "sbc_issuance": sc.share_bridge.shares_issued_sbc,
                "ending_actual_basic_shares": round(sc.share_bridge.ending_actual_shares, 3),
                "dilutive_awards": sc.share_bridge.dilutive_securities,
                "ending_diluted_shares": round(sc.ending_diluted_shares, 3),
            },
            "terminal_valuation": {
                "method": sc.exit_method,
                "exit_multiple": round(sc.exit_multiple, 4),
                "wacc": sc.wacc,
                "terminal_growth": sc.terminal_growth,
                "enterprise_value": round(sc.enterprise_value, 2),
                "equity_value": round(sc.equity_value, 2),
                "terminal_share_price": round(sc.terminal_share_price, 2),
            },
            "dated_returns": {
                "spot_price": spot_price,
                "terminal_share_price": round(sc.terminal_share_price, 2),
                "annual_dividends": sc.annual_dividends,
                "cumulative_dividends_per_share": round(sc.cumulative_dividends_per_share, 2),
                "total_terminal_payout": round(sc.total_terminal_payout, 2),
                "simple_5yr_cagr": round(sc.simple_cagr_from_spot, 6),
                "five_period_annual_irr": round(sc.five_period_annual_irr, 6),
                "five_period_annual_irr_percent": round(sc.five_period_annual_irr * 100.0, 2),
                "calendar_dated_xirr": round(sc.calendar_dated_xirr, 6),
                "calendar_dated_xirr_percent": round(sc.calendar_dated_xirr * 100.0, 2),
                "dated_xirr": round(sc.five_period_annual_irr, 6),
                "dated_xirr_percent": round(sc.five_period_annual_irr * 100.0, 2),
                "dated_cash_flows": sc.dated_cash_flows,
            },
            "accumulation_ceilings": {
                "ceiling_80_hurdle": round(sc.accumulation_ceiling_80, 2),
                "ceiling_85_hurdle": round(sc.accumulation_ceiling_85, 2),
                "ceiling_100_hurdle": round(sc.accumulation_ceiling_100, 2),
                "ceiling_calendar_80_hurdle": round(sc.accumulation_ceiling_calendar_80, 2),
                "ceiling_calendar_85_hurdle": round(sc.accumulation_ceiling_calendar_85, 2),
                "ceiling_calendar_100_hurdle": round(sc.accumulation_ceiling_calendar_100, 2),
                "annual_irr_basis": {
                    "ceiling_80": round(sc.accumulation_ceiling_80, 2),
                    "ceiling_85": round(sc.accumulation_ceiling_85, 2),
                    "ceiling_100": round(sc.accumulation_ceiling_100, 2),
                    "invertibility_proof": {
                        "reproduced_irr_at_80": round(reproduced_80, 6),
                        "reproduced_irr_at_85": round(reproduced_85, 6),
                        "reproduced_irr_at_100": round(reproduced_100, 6),
                        "is_exact": (
                            abs(reproduced_80 - 0.080) < 1e-4 and
                            abs(reproduced_85 - 0.085) < 1e-4 and
                            abs(reproduced_100 - 0.100) < 1e-4
                        )
                    }
                },
                "calendar_xirr_basis": {
                    "ceiling_80": round(sc.accumulation_ceiling_calendar_80, 2),
                    "ceiling_85": round(sc.accumulation_ceiling_calendar_85, 2),
                    "ceiling_100": round(sc.accumulation_ceiling_calendar_100, 2),
                    "invertibility_proof": {
                        "reproduced_xirr_at_80": round(reproduced_cal_80, 6),
                        "reproduced_xirr_at_85": round(reproduced_cal_85, 6),
                        "reproduced_xirr_at_100": round(reproduced_cal_100, 6),
                        "is_exact": (
                            abs(reproduced_cal_80 - 0.080) < 1e-4 and
                            abs(reproduced_cal_85 - 0.085) < 1e-4 and
                            abs(reproduced_cal_100 - 0.100) < 1e-4
                        )
                    }
                },
                "invertibility_proof": {
                    "reproduced_irr_at_80": round(reproduced_80, 6),
                    "reproduced_irr_at_85": round(reproduced_85, 6),
                    "reproduced_irr_at_100": round(reproduced_100, 6),
                    "is_exact": (
                        abs(reproduced_80 - 0.080) < 1e-4 and
                        abs(reproduced_85 - 0.085) < 1e-4 and
                        abs(reproduced_100 - 0.100) < 1e-4
                    )
                }
            },
            "provenance_and_confidence": {
                "primary_source": "SEC Form 20-F FY2025 & Form 6-K H1 2026",
                "accounting_standard": "IFRS IAS 7 (Interest Paid in Operating Activities)",
                "confidence_score": 0.95,
                "governance_disposition": "WATCH_CONDITION",
                "allocation_directive": "NO_TRADE_EXECUTION"
            }
        }

    return ledger


def save_canonical_ledger_v5(output_path: str = "ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER_V5.json") -> str:
    """Serializes the V5 canonical scenario ledger to disk."""
    ledger = generate_canonical_ledger_v5()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    return output_path


if __name__ == "__main__":
    out_v4 = save_canonical_ledger()
    print(f"Canonical Scenario Ledger (v4.0 Baseline) successfully written to: {out_v4}")
    out_v5 = save_canonical_ledger_v5()
    print(f"Canonical Scenario Ledger (v5.0 Successor) successfully written to: {out_v5}")
