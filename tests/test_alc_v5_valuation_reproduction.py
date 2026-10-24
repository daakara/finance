"""
tests/test_alc_v5_valuation_reproduction.py

Deterministic reproduction and validation test suite for Alcon Inc. (NYSE: ALC)
Model v5.0 Successor with Scenario-Derived Net Debt Policy (Option B).

Authorized under:
ARX_ALCON_SCENARIO_DERIVED_NET_DEBT_CORRECTION_AND_CANDIDATE_REQUALIFICATION_AUTHORIZED

Invariants Tested:
1. Scenario-Derived Net Debt Policy: ending_gross_debt - ending_cash = valuation_net_debt.
   Zero net debt variance (difference = 0.00) across all six scenarios.
2. Distinct Scenarios: Conservative, Base Multiple, Base DCF, Optimistic, Conservative Dynamic, Optimistic Dynamic.
3. Ceiling Invertibility: Exact inversion of accumulation ceilings to hurdle rates (8.0%, 8.5%, 10.0%)
   under both 5-period discrete annual IRR and Actual/365 calendar dated XIRR.
4. Historical V4 Preservation: V4 ledger ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json remains bitwise
   immutable with SHA-256 d28270a18984de2b851ab5ce24dfcbe822d6bc55479024b708e418e21acdced4.
5. V4-vs-V5 Machine-Derived Comparison: Quantified deltas across all six scenarios.
6. Identity Invariant: CIK 0001167379, NYSE: ALC, spot price $67.43 (October 21, 2026).
"""

import hashlib
import json
import math
import os
from pathlib import Path
import pytest

pytestmark = pytest.mark.tier1

from analyst_dashboard.research.sourced_fundamentals.alc_scenario_engine import (
    build_canonical_scenarios,
    build_canonical_scenarios_v5,
    generate_canonical_ledger,
    generate_canonical_ledger_v5,
    generate_v4_v5_comparison,
    solve_dated_xirr,
    solve_calendar_xirr,
    get_dated_cash_flows,
)


EXPECTED_V4_LEDGER_SHA256 = "d28270a18984de2b851ab5ce24dfcbe822d6bc55479024b708e418e21acdced4"


class TestAlconV5ValuationReproduction:
    SPOT_PRICE = 67.43

    # Expected V5 targets and metrics under scenario-derived net debt
    EXPECTED_V5 = {
        "CONSERVATIVE": {
            "gross_debt": 0.0,
            "cash": 2045.79,
            "net_debt": -2045.79,
            "shares": 475.524,
            "target_price": 76.36,
            "annual_irr_pct": 3.11,
            "calendar_xirr_pct": 3.12,
            "ceiling_80": 53.63,
            "ceiling_85": 52.42,
            "ceiling_100": 48.99,
        },
        "BASE_MULTIPLE": {
            "gross_debt": 0.0,
            "cash": 3687.55,
            "net_debt": -3687.55,
            "shares": 475.524,
            "target_price": 93.20,
            "annual_irr_pct": 7.23,
            "calendar_xirr_pct": 7.24,
            "ceiling_80": 65.09,
            "ceiling_85": 63.62,
            "ceiling_100": 59.44,
        },
        "BASE_DCF": {
            "gross_debt": 0.0,
            "cash": 3687.55,
            "net_debt": -3687.55,
            "shares": 475.524,
            "target_price": 95.00,
            "annual_irr_pct": 7.64,
            "calendar_xirr_pct": 7.65,
            "ceiling_80": 66.31,
            "ceiling_85": 64.82,
            "ceiling_100": 60.56,
        },
        "OPTIMISTIC": {
            "gross_debt": 0.0,
            "cash": 5290.81,
            "net_debt": -5290.81,
            "shares": 475.524,
            "target_price": 129.97,
            "annual_irr_pct": 14.50,
            "calendar_xirr_pct": 14.52,
            "ceiling_80": 90.11,
            "ceiling_85": 88.07,
            "ceiling_100": 82.27,
        },
        "CONSERVATIVE_DYNAMIC": {
            "gross_debt": 1000.0,
            "cash": 3645.79,
            "net_debt": -2645.79,
            "shares": 481.062,
            "target_price": 76.70,
            "annual_irr_pct": 3.20,
            "calendar_xirr_pct": 3.21,
            "ceiling_80": 53.86,
            "ceiling_85": 52.65,
            "ceiling_100": 49.20,
        },
        "OPTIMISTIC_DYNAMIC": {
            "gross_debt": 0.0,
            "cash": 5290.81,
            "net_debt": -5290.81,
            "shares": 478.348,
            "target_price": 129.06,
            "annual_irr_pct": 14.34,
            "calendar_xirr_pct": 14.36,
            "ceiling_80": 89.49,
            "ceiling_85": 87.46,
            "ceiling_100": 81.71,
        },
    }

    def test_01_canonical_v5_engine_evaluation(self):
        """Verifies that build_canonical_scenarios_v5 evaluates all 6 scenarios."""
        scenarios = build_canonical_scenarios_v5(spot_price=self.SPOT_PRICE)
        assert len(scenarios) == 6
        for sc_id in self.EXPECTED_V5.keys():
            assert sc_id in scenarios, f"Scenario {sc_id} missing from V5 engine output"

    def test_02_scenario_derived_net_debt_zero_variance(self):
        """
        Core Invariant:
        In V5 (Option B), every scenario's terminal net debt MUST be derived from
        ending gross debt minus ending cash, with zero variance.
        """
        scenarios = build_canonical_scenarios_v5(spot_price=self.SPOT_PRICE)
        for sc_id, sc in scenarios.items():
            bs = sc.balance_sheet
            exp = self.EXPECTED_V5[sc_id]

            # 1. Gross debt and cash balance verification
            assert math.isclose(bs.ending_gross_debt, exp["gross_debt"], abs_tol=0.01)
            assert math.isclose(bs.ending_cash, exp["cash"], abs_tol=0.01)

            # 2. Derived net debt = gross debt - cash
            derived_nd = bs.ending_gross_debt - bs.ending_cash
            assert math.isclose(derived_nd, exp["net_debt"], abs_tol=0.01)

            # 3. Valuation net debt used MUST exactly equal derived net debt (zero variance)
            assert math.isclose(sc.net_debt_bridge, derived_nd, abs_tol=1e-6), (
                f"{sc_id}: Valuation net debt ({sc.net_debt_bridge}) differs from cash-flow-derived net debt ({derived_nd})"
            )

            # 4. EV to Equity Value bridge
            assert math.isclose(sc.equity_value, sc.enterprise_value - sc.net_debt_bridge, abs_tol=0.01)

            # 5. Per-share price = equity value / diluted shares
            assert math.isclose(sc.terminal_share_price, sc.equity_value / sc.ending_diluted_shares, abs_tol=0.01)

    def test_03_v5_target_prices_and_returns(self):
        """Validates all per-share targets, annual IRRs, and calendar XIRRs in V5."""
        scenarios = build_canonical_scenarios_v5(spot_price=self.SPOT_PRICE)
        for sc_id, exp in self.EXPECTED_V5.items():
            sc = scenarios[sc_id]

            # Target price
            assert math.isclose(sc.terminal_share_price, exp["target_price"], abs_tol=0.02), (
                f"{sc_id} target price mismatch: got {sc.terminal_share_price}, expected {exp['target_price']}"
            )

            # Annual IRR %
            annual_irr_pct = round(sc.five_period_annual_irr * 100.0, 2)
            assert math.isclose(annual_irr_pct, exp["annual_irr_pct"], abs_tol=0.02), (
                f"{sc_id} annual IRR mismatch: got {annual_irr_pct}%, expected {exp['annual_irr_pct']}%"
            )

            # Calendar XIRR %
            cal_xirr_pct = round(sc.calendar_dated_xirr * 100.0, 2)
            assert math.isclose(cal_xirr_pct, exp["calendar_xirr_pct"], abs_tol=0.02), (
                f"{sc_id} calendar XIRR mismatch: got {cal_xirr_pct}%, expected {exp['calendar_xirr_pct']}%"
            )

            # Accumulation ceilings
            assert math.isclose(sc.accumulation_ceiling_80, exp["ceiling_80"], abs_tol=0.02)
            assert math.isclose(sc.accumulation_ceiling_85, exp["ceiling_85"], abs_tol=0.02)
            assert math.isclose(sc.accumulation_ceiling_100, exp["ceiling_100"], abs_tol=0.02)

    def test_04_accumulation_ceiling_exact_inversion(self):
        """Verifies exact invertibility of accumulation ceilings to hurdle rates in V5."""
        scenarios = build_canonical_scenarios_v5(spot_price=self.SPOT_PRICE)
        for sc_id, sc in scenarios.items():
            divs = sc.annual_dividends
            p = sc.terminal_share_price

            # Annual IRR basis:
            irr_80 = solve_dated_xirr(sc.accumulation_ceiling_80, divs, p)
            irr_85 = solve_dated_xirr(sc.accumulation_ceiling_85, divs, p)
            irr_100 = solve_dated_xirr(sc.accumulation_ceiling_100, divs, p)

            assert math.isclose(irr_80, 0.080, abs_tol=1e-4)
            assert math.isclose(irr_85, 0.085, abs_tol=1e-4)
            assert math.isclose(irr_100, 0.100, abs_tol=1e-4)

            # Calendar XIRR basis:
            cfs_80 = get_dated_cash_flows(sc.accumulation_ceiling_calendar_80, divs, p)
            cfs_85 = get_dated_cash_flows(sc.accumulation_ceiling_calendar_85, divs, p)
            cfs_100 = get_dated_cash_flows(sc.accumulation_ceiling_calendar_100, divs, p)

            cal_80 = solve_calendar_xirr(cfs_80)
            cal_85 = solve_calendar_xirr(cfs_85)
            cal_100 = solve_calendar_xirr(cfs_100)

            assert math.isclose(cal_80, 0.080, abs_tol=1e-4)
            assert math.isclose(cal_85, 0.085, abs_tol=1e-4)
            assert math.isclose(cal_100, 0.100, abs_tol=1e-4)

    def test_05_v4_baseline_preservation_and_comparison(self):
        """
        Preservation Invariant:
        Historical V4 canonical ledger ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json
        MUST remain bitwise immutable and match SHA-256 d28270a18984de2b851ab5ce24dfcbe822d6bc55479024b708e418e21acdced4.
        """
        v4_path = Path("ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json")
        assert v4_path.exists(), "V4 ledger must exist"

        h = hashlib.sha256()
        with open(v4_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_sha = h.hexdigest()
        assert actual_sha == EXPECTED_V4_LEDGER_SHA256, (
            f"V4 ledger was modified! Expected {EXPECTED_V4_LEDGER_SHA256}, got {actual_sha}"
        )

        # Machine-derived comparison:
        comp = generate_v4_v5_comparison(spot_price=self.SPOT_PRICE)
        assert len(comp["scenarios"]) == 6

        # Check Base DCF delta in comparison:
        base_dcf_comp = next(r for r in comp["scenarios"] if r["scenario_id"] == "BASE_DCF")
        assert base_dcf_comp["v4_baseline"]["terminal_share_price"] == 91.82
        assert base_dcf_comp["v5_successor"]["terminal_share_price"] == 95.00
        assert base_dcf_comp["delta"]["terminal_share_price"] == 3.18
        assert base_dcf_comp["delta"]["valuation_net_debt"] == -1511.35
        assert base_dcf_comp["delta"]["annual_irr_bps"] == 71.9

    def test_06_v5_ledger_disk_serialization_integrity(self):
        """Validates that ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER_V5.json on disk matches live V5 generation."""
        v5_path = Path("ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER_V5.json")
        assert v5_path.exists(), "V5 ledger must exist on disk"

        with open(v5_path, "r", encoding="utf-8") as f:
            disk_ledger = json.load(f)

        meta = disk_ledger.get("metadata", {})
        assert meta["model_version"] == "v5.0"
        assert meta["predecessor_version"] == "v4.0"
        assert meta["net_debt_policy"] == "SCENARIO_DERIVED_FROM_CASH_FLOWS"
        assert meta["cik"] == "0001167379"

        # Check every scenario in disk ledger:
        for sc_id, exp in self.EXPECTED_V5.items():
            sc_data = disk_ledger["scenarios"][sc_id]
            tv = sc_data["terminal_valuation"]
            bs = sc_data["terminal_balance_sheet_values"]
            dr = sc_data["dated_returns"]

            assert tv["terminal_share_price"] == exp["target_price"]
            assert dr["five_period_annual_irr_percent"] == exp["annual_irr_pct"]
            assert dr["calendar_dated_xirr_percent"] == exp["calendar_xirr_pct"]
            assert bs["net_debt_derivation"]["difference"] == 0.0
            assert bs["net_debt_derivation"]["derivation_status"] == "EXACT_ZERO_VARIANCE"
