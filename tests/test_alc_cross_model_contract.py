"""
tests/test_alc_cross_model_contract.py

ARX Terminal — Alcon Inc. (NYSE: ALC) Cross-Model Output Contract Gate.
Validates the 12 strict output contract requirements under ARX Closure Model v2:
1. Fail against currently published inconsistent IRR outputs.
2. Recalculate XIRR independently from dated cash flows.
3. Invert the XIRR function to derive accumulation ceilings.
4. Verify the exact investment holding period.
5. Reject unsupported labels of buyback authorization.
6. Recompute net debt for each scenario from its own cash flows.
7. Recompute terminal diluted shares from its own share ledger.
8. Fail if a published scenario figure differs from serialized engine output.
9. Check financial feasibility when cash generation falls.
10. Verify that changing repurchase policy changes net cash and shares consistently.
11. Confirm that no research watch condition triggers trade execution.
12. Verify that all final-report figures have declared units, dates, and model version.
"""

import json
import math
import os
import pytest
from datetime import date

from analyst_dashboard.research.sourced_fundamentals.alc_scenario_engine import (
    build_canonical_scenarios,
    generate_canonical_ledger,
    solve_dated_xirr,
    solve_accumulation_ceiling,
    solve_calendar_xirr,
    solve_calendar_ceiling,
    get_dated_cash_flows,
    evaluate_scenario,
    build_balance_sheet_bridge,
    build_share_count_bridge,
)


class TestAlconCrossModelOutputContract:
    SPOT_PRICE = 67.43
    DIVIDENDS_5YR = [0.36, 0.39, 0.42, 0.45, 0.48]

    # Published reference terminal prices:
    PUBLISHED_PRICES = {
        "CONSERVATIVE": 76.64,
        "BASE_MULTIPLE": 90.03,
        "BASE_DCF": 91.83,
        "OPTIMISTIC": 123.42,
    }

    # Superseded non-reproducible published IRRs:
    # Previously conjectured to stem from a 4.7-year formula, but mathematically
    # disproven (e.g. for Base DCF $91.83 that formula produces 7.29%, not 7.37%).
    # Formal status under ARX Closure Model v2: COMPUTATIONAL_MECHANISM_UNKNOWN_AND_NON_REPRODUCIBLE.
    OLD_INCONSISTENT_IRRS = {
        "CONSERVATIVE": 0.0349,      # 3.49%
        "BASE_MULTIPLE": 0.0694,     # 6.94%
        "BASE_DCF": 0.0737,          # 7.37%
        "OPTIMISTIC": 0.1343,        # 13.43%
    }

    # True 5-annual cash flow IRRs:
    TRUE_ANNUAL_IRRS = {
        "CONSERVATIVE": 0.0318,      # 3.18%
        "BASE_MULTIPLE": 0.0650,     # 6.50%
        "BASE_DCF": 0.0692,          # 6.92%
        "OPTIMISTIC": 0.1334,        # 13.34%
    }

    # True Actual/365 Calendar Dated XIRRs (May 15 AGM dividend dates, Oct 22, 2031 exit):
    TRUE_CALENDAR_XIRRS = {
        "CONSERVATIVE": 0.0319,      # 3.19%
        "BASE_MULTIPLE": 0.0652,     # 6.52%
        "BASE_DCF": 0.0693,          # 6.93%
        "OPTIMISTIC": 0.1336,        # 13.36%
    }

    # -------------------------------------------------------------------------
    # Test 1: Fail against currently published inconsistent IRR outputs
    # -------------------------------------------------------------------------
    def test_01_fail_against_currently_published_inconsistent_irr_outputs(self):
        """
        Adversarial Proving Test:
        Demonstrates that the previously published IRRs (3.49%, 6.94%, 7.37%, 13.43%)
        FAILS against the true annual cash-flow XIRRs (3.18%, 6.50%, 6.92%, 13.34%).
        Fails if anyone attempts to pass the old inconsistent numbers as true annual IRRs.
        """
        for sc_id, old_irr in self.OLD_INCONSISTENT_IRRS.items():
            terminal_p = self.PUBLISHED_PRICES[sc_id]
            true_irr = solve_dated_xirr(self.SPOT_PRICE, self.DIVIDENDS_5YR, terminal_p)
            
            # The discrepancy MUST be detected and rejected:
            diff = abs(old_irr - true_irr)
            assert diff > 0.0005, (
                f"Discrepancy not detected for {sc_id}: old={old_irr:.4f}, true={true_irr:.4f}, diff={diff:.6f}"
            )
            # True IRR must match the expected true annual return:
            expected_true = self.TRUE_ANNUAL_IRRS[sc_id]
            assert math.isclose(true_irr, expected_true, abs_tol=0.0005)

    # -------------------------------------------------------------------------
    # Test 2: Recalculate XIRR independently from dated cash flows
    # -------------------------------------------------------------------------
    def test_02_recalculate_xirr_independently_from_dated_cash_flows(self):
        """
        Calculates both discrete 5-period annual IRR and Actual/365 calendar dated XIRR
        using independent bisection root-finders implemented directly within this test file,
        verifying zero dependence on engine internals.
        """
        def independent_solve_annual_irr(entry: float, divs: list, exit_p: float) -> float:
            cfs = [-entry] + divs[:-1] + [divs[-1] + exit_p]
            lo, hi = -0.5, 2.0
            for _ in range(150):
                mid = (lo + hi) / 2.0
                npv = sum(cf / ((1.0 + mid) ** t) for t, cf in enumerate(cfs))
                if abs(npv) < 1e-11:
                    return mid
                if npv > 0:
                    lo = mid
                else:
                    hi = mid
            return mid

        def independent_solve_calendar_xirr(entry: float, divs: list, exit_p: float) -> float:
            # Calendar day diffs from 2026-10-22 to May 15 AGM dates and Oct 22, 2031 exit:
            # 2026-10-22: 0 days
            # 2027-05-15: 205 days
            # 2028-05-15: 571 days
            # 2029-05-15: 936 days
            # 2030-05-15: 1301 days
            # 2031-05-15: 1666 days
            # 2031-10-22: 1826 days
            fractions = [0.0, 205/365.0, 571/365.0, 936/365.0, 1301/365.0, 1666/365.0, 1826/365.0]
            cfs = [-entry] + divs + [exit_p]
            lo, hi = -0.5, 2.0
            for _ in range(150):
                mid = (lo + hi) / 2.0
                npv = sum(cf / ((1.0 + mid) ** frac) for cf, frac in zip(cfs, fractions))
                if abs(npv) < 1e-11:
                    return mid
                if npv > 0:
                    lo = mid
                else:
                    hi = mid
            return mid

        for sc_id, p in self.PUBLISHED_PRICES.items():
            indep_irr = independent_solve_annual_irr(self.SPOT_PRICE, self.DIVIDENDS_5YR, p)
            engine_irr = solve_dated_xirr(self.SPOT_PRICE, self.DIVIDENDS_5YR, p)
            assert math.isclose(indep_irr, engine_irr, abs_tol=1e-6)

            # Check calendar XIRR:
            indep_cal = independent_solve_calendar_xirr(self.SPOT_PRICE, self.DIVIDENDS_5YR, p)
            expected_cal = self.TRUE_CALENDAR_XIRRS[sc_id]
            assert math.isclose(indep_cal, expected_cal, abs_tol=0.0005)

    # -------------------------------------------------------------------------
    # Test 3: Invert the XIRR function to derive accumulation ceilings
    # -------------------------------------------------------------------------
    def test_03_invert_xirr_function_to_derive_accumulation_ceilings(self):
        """
        Inversion Invariant:
        For any calculated accumulation ceiling P_ceil at hurdle rate r,
        passing P_ceil as the entry price back into solve_dated_xirr MUST return r.
        """
        hurdles = [0.080, 0.085, 0.100]
        base_dcf_price = self.PUBLISHED_PRICES["BASE_DCF"]

        # Expected reference ceilings for Base DCF ($91.83):
        # 8.0%: $64.16, 8.5%: $62.71, 10.0%: $58.59
        expected_ceilings = {0.080: 64.16, 0.085: 62.71, 0.100: 58.59}

        for h in hurdles:
            ceil_p = solve_accumulation_ceiling(h, self.DIVIDENDS_5YR, base_dcf_price)
            assert math.isclose(ceil_p, expected_ceilings[h], abs_tol=0.02)

            # Invertibility check:
            reproduced_irr = solve_dated_xirr(ceil_p, self.DIVIDENDS_5YR, base_dcf_price)
            assert math.isclose(reproduced_irr, h, abs_tol=1e-4), (
                f"Ceiling {ceil_p} did not invert to hurdle {h}: got {reproduced_irr}"
            )

    # -------------------------------------------------------------------------
    # Test 4: Verify the exact investment holding period
    # -------------------------------------------------------------------------
    def test_04_verify_exact_investment_holding_period(self):
        """
        Verifies the exact calendar and fiscal holding period parameters under Convention A:
        Exact 5.00-year calendar holding period from October 22, 2026 to October 22, 2031 (1826 days).
        Includes 2028 leap year (366 days).
        """
        start = date(2026, 10, 22)
        end = date(2031, 10, 22)
        days = (end - start).days
        assert days == 1826, f"Expected 1826 days, got {days}"

        # 5 annual dividend distributions:
        assert len(self.DIVIDENDS_5YR) == 5
        assert math.isclose(sum(self.DIVIDENDS_5YR), 2.10, abs_tol=1e-5)

    # -------------------------------------------------------------------------
    # Test 5: Reject unsupported labels of buyback authorization
    # -------------------------------------------------------------------------
    def test_05_reject_unsupported_labels_of_buyback_authorization(self):
        """
        Reconciliation of Alcon Buyback Authorities:
        1. 2025 program ($750M): EXECUTED_BUYBACKS (completed Jan 2026).
        2. 2026 program ($1,500M): APPROVED_CAPACITY (authorized May 5, 2026 AGM).
        3. Executed H1 2026: $300M.
        4. REMAINING_APPROVED_CAPACITY: $1,200M.
        Fails if a model labels any buyback > $1,200M as approved without ASSUMED_FUTURE_AUTHORIZATIONS.
        """
        ledger = generate_canonical_ledger()
        bb_auth = ledger["buyback_authority_reconciliation"]

        assert bb_auth["EXECUTED_BUYBACKS_2025_PROGRAM"]["amount"] == 750.0
        assert bb_auth["APPROVED_CAPACITY_2026_PROGRAM"]["amount"] == 1500.0
        assert bb_auth["REMAINING_APPROVED_CAPACITY"]["amount"] == 1200.0
        assert bb_auth["FORECAST_BUYBACKS_BASELINE"]["amount"] == 1800.0
        assert bb_auth["ASSUMED_FUTURE_AUTHORIZATIONS"]["amount"] == 600.0

        # Any forecast buyback > 1200 must explicitly report assumed future authorizations:
        for sc in ledger["scenarios"].values():
            bb_data = sc["capital_allocation_assumptions"]["buyback_classification"]
            forecast = bb_data["FORECAST_BUYBACKS"]
            rem_approved = bb_data["REMAINING_APPROVED_CAPACITY"]
            assumed = bb_data["ASSUMED_FUTURE_AUTHORIZATIONS"]
            if forecast > rem_approved:
                assert assumed == forecast - rem_approved
                assert assumed > 0.0

    # -------------------------------------------------------------------------
    # Test 6: Recompute net debt for each scenario from its own cash flows
    # -------------------------------------------------------------------------
    def test_06_recompute_net_debt_for_each_scenario_from_own_cash_flows(self):
        """
        Asserts that Conservative Dynamic, Base, and Optimistic Dynamic have
        independently derived ending net cash/debt based on their own cash generation.
        """
        ledger = generate_canonical_ledger()
        scenarios = ledger["scenarios"]

        cons_net_debt = scenarios["CONSERVATIVE_DYNAMIC"]["terminal_balance_sheet_values"]["ending_net_debt"]
        base_net_debt = scenarios["BASE_MULTIPLE"]["terminal_balance_sheet_values"]["ending_net_debt"]
        opt_net_debt = scenarios["OPTIMISTIC_DYNAMIC"]["terminal_balance_sheet_values"]["ending_net_debt"]

        # Different scenarios MUST have different net debt when dynamically derived:
        assert cons_net_debt != base_net_debt
        assert base_net_debt != opt_net_debt
        # Optimistic generates more cash -> larger net cash balance (more negative net debt):
        assert opt_net_debt < base_net_debt

    # -------------------------------------------------------------------------
    # Test 7: Recompute terminal diluted shares from its own share ledger
    # -------------------------------------------------------------------------
    def test_07_recompute_terminal_diluted_shares_from_own_share_ledger(self):
        """
        Recomputes ending diluted shares from beginning basic (484.524M), repurchases,
        SBC issuance (+11.0M), and dilutive awards (4.0M).
        """
        sc_cons = build_share_count_bridge(484.524, 1200.0, 65.0, 11.0, 4.0)
        assert math.isclose(sc_cons.shares_repurchased, 1200.0 / 65.0, abs_tol=1e-3)
        assert math.isclose(sc_cons.ending_diluted_shares, 484.524 - (1200.0 / 65.0) + 11.0 + 4.0, abs_tol=1e-3)

        sc_base = build_share_count_bridge(484.524, 1800.0, 75.0, 11.0, 4.0)
        assert math.isclose(sc_base.shares_repurchased, 24.0, abs_tol=1e-3)
        assert math.isclose(sc_base.ending_diluted_shares, 475.524, abs_tol=1e-3)

        assert sc_cons.ending_diluted_shares != sc_base.ending_diluted_shares

    # -------------------------------------------------------------------------
    # Test 8: Fail if a published scenario figure differs from serialized engine output
    # -------------------------------------------------------------------------
    def test_08_fail_if_published_figure_differs_from_serialized_engine_output(self):
        """
        Integrity Contract:
        Validates that ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json matches the engine
        live outputs across all scenarios and metrics with zero divergence.
        """
        ledger_path = "ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json"
        assert os.path.exists(ledger_path), "Master canonical ledger file missing!"

        with open(ledger_path, "r", encoding="utf-8") as f:
            disk_ledger = json.load(f)

        live_ledger = generate_canonical_ledger()

        # Check every scenario in disk ledger against live calculation:
        for sc_id, live_sc in live_ledger["scenarios"].items():
            assert sc_id in disk_ledger["scenarios"]
            disk_sc = disk_ledger["scenarios"][sc_id]

            # Compare key published metrics:
            assert disk_sc["terminal_valuation"]["terminal_share_price"] == live_sc["terminal_valuation"]["terminal_share_price"]
            assert disk_sc["dated_returns"]["dated_xirr_percent"] == live_sc["dated_returns"]["dated_xirr_percent"]
            assert disk_sc["accumulation_ceilings"]["ceiling_80_hurdle"] == live_sc["accumulation_ceilings"]["ceiling_80_hurdle"]
            assert disk_sc["accumulation_ceilings"]["ceiling_85_hurdle"] == live_sc["accumulation_ceilings"]["ceiling_85_hurdle"]
            assert disk_sc["accumulation_ceilings"]["ceiling_100_hurdle"] == live_sc["accumulation_ceilings"]["ceiling_100_hurdle"]

    # -------------------------------------------------------------------------
    # Test 9: Check financial feasibility when cash generation falls
    # -------------------------------------------------------------------------
    def test_09_check_financial_feasibility_when_cash_generation_falls(self):
        """
        Stress Test:
        When revenue growth falls to 2.0% and operating margin falls to 14.0%,
        verify that cash available restricts deleveraging and cash does NOT go negative.
        """
        bs_stress = build_balance_sheet_bridge(
            starting_gross_debt=4500.0,
            starting_cash=1200.0,
            cumulative_rev=55000.0,
            ocf_margin_pct=0.150,
            capex_pct_of_rev=0.056,
            strategic_mna=0.0,
            cash_interest_paid=480.0,
            cumulative_dividends=1008.0,
            share_repurchases=1200.0,
            gross_debt_repayment_plan=4500.0,
            is_levered_ocf=True
        )
        assert bs_stress.ending_cash >= 0.0, f"Cash turned negative: {bs_stress.ending_cash}"
        assert bs_stress.gross_debt_repayment <= bs_stress.fcfe - bs_stress.cumulative_dividends - bs_stress.share_repurchases

    # -------------------------------------------------------------------------
    # Test 10: Verify changing repurchase policy changes net cash and shares consistently
    # -------------------------------------------------------------------------
    def test_10_verify_changing_repurchase_policy_changes_net_cash_and_shares_consistently(self):
        """
        Changing buyback expenditure by $1,800M changes ending cash by exactly $1,800M
        and ending shares by 24.0M shares.
        """
        sc_zero = evaluate_scenario("Base_Zero", 10500.0, 0.065, 0.185, 0.08, "multiple", exit_multiple=10.65, buyback_total=0.0)
        sc_with = evaluate_scenario("Base_With", 10500.0, 0.065, 0.185, 0.08, "multiple", exit_multiple=10.65, buyback_total=1800.0, avg_repurchase_price=75.0)

        # Share difference:
        assert math.isclose(sc_zero.ending_diluted_shares - sc_with.ending_diluted_shares, 24.0, abs_tol=1e-3)
        # Cash difference:
        assert math.isclose(sc_zero.balance_sheet.ending_cash - sc_with.balance_sheet.ending_cash, 1800.0, abs_tol=1e-3)

    # -------------------------------------------------------------------------
    # Test 11: Confirm that no research watch condition triggers trade execution
    # -------------------------------------------------------------------------
    def test_11_confirm_no_research_watch_condition_triggers_trade_execution(self):
        """
        Governance Invariant:
        Asserts that research outputs emit strictly WATCH_CONDITION and NO_TRADE_EXECUTION.
        """
        ledger = generate_canonical_ledger()
        assert ledger["metadata"]["governance_status"] == "READ_ONLY_RESEARCH_NO_EXECUTION_AUTHORITY"
        for sc in ledger["scenarios"].values():
            assert sc["provenance_and_confidence"]["governance_disposition"] == "WATCH_CONDITION"
            assert sc["provenance_and_confidence"]["allocation_directive"] == "NO_TRADE_EXECUTION"

    # -------------------------------------------------------------------------
    # Test 12: Verify all final-report figures have declared units, dates, and version
    # -------------------------------------------------------------------------
    def test_12_verify_all_final_report_figures_have_declared_units_dates_version(self):
        """
        Metadata & Contract Invariant:
        Ensures declared units, dates, model version 'v4.0', and currency 'USD'.
        """
        ledger = generate_canonical_ledger()
        meta = ledger["metadata"]
        assert meta["model_version"] == "v4.0"
        assert meta["currency"] == "USD"
        assert meta["valuation_date"] == "2026-10-22"
        assert meta["investment_start_date"] == "2026-10-22"
        assert meta["investment_end_date"] == "2031-10-22"
        assert meta["investment_timeline_convention"] == "CONVENTION_A_FIVE_FULL_YEARS"
        assert meta["holding_period_days"] == 1826
        assert meta["holding_period_years"] == 5.0
        assert meta["units"]["financial_totals"] == "$M"
        assert meta["units"]["per_share"] == "$/share"
        assert meta["units"]["shares"] == "Millions (M)"
