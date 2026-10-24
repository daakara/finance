"""
tests/test_alc_cross_model_integrity.py

Comprehensive Cross-Model Integration Test Suite for Alcon Inc. (NYSE: ALC).
Enforces the 14 mandatory integration invariants specified in Section E of the
ARX Terminal ALC Cross-Model Certification Recovery Gate.

Tests:
1. Conservative EBITDA multiplied by conservative multiple equals conservative EV.
2. Optimistic EBITDA multiplied by optimistic multiple equals optimistic EV.
3. Altering scenario revenue changes its own EBITDA and terminal EV.
4. Altering scenario exit multiple changes only the appropriate valuation-dependent outputs.
5. Repurchase expenditure and average price determine shares repurchased.
6. Issuance minus repurchases determines ending actual shares.
7. Actual shares and dilutive securities determine forecast diluted shares.
8. Repurchase cash expenditure reduces cash available for debt paydown.
9. Removing repurchases changes terminal shares and balance-sheet projections consistently.
10. Interest expense is deducted once under the selected accounting convention.
11. Closing debt and cash reconcile to opening balances and modeled financing transactions.
12. Every scenario's dated return uses its own terminal price and dividend schedule.
13. Accumulation ceilings reproduce each scenario's dated return hurdle.
14. No research valuation output activates an execution or capital-allocation command.
"""

import math
import pytest

pytestmark = pytest.mark.tier1

from analyst_dashboard.research.sourced_fundamentals.alc_scenario_engine import (
    build_share_count_bridge,
    build_balance_sheet_bridge,
    evaluate_scenario,
    solve_dated_xirr,
    solve_accumulation_ceiling
)


class TestAlconCrossModelIntegrity:
    BASE_REVENUE_TTM = 10758.0
    SPOT_PRICE = 67.43

    # --------------------------------------------------------------------------
    # Test 1: Conservative EBITDA * multiple = Conservative EV
    # --------------------------------------------------------------------------
    def test_01_conservative_ebitda_multiple_equals_conservative_ev(self):
        """
        Invariant 1: Conservative EBITDA multiplied by conservative multiple equals conservative EV.
        Exposes prior defect: Prior report stated Conservative EBITDA = $3,217.5M and multiple = 10.65x,
        but published EV was $40,631.9M (which came from Base EBITDA $3,815.1M * 10.65x).
        Asserts true Conservative EV is $3,217.5M * 10.65 = $34,266.4M (~$34.27B).
        """
        ebitda_cons = 3217.5
        multiple_cons = 10.65

        # Failing-before condition: prior published EV was $40,631.9M
        prior_published_ev = 40631.9
        assert not math.isclose(prior_published_ev, ebitda_cons * multiple_cons, rel_tol=0.05)

        # Passing-after condition: true conservative EV strictly derived from its own inputs
        scenario = evaluate_scenario(
            scenario_name="Conservative",
            base_revenue=self.BASE_REVENUE_TTM,
            rev_cagr=0.045,
            operating_margin_y5=0.185,
            dna_pct_y5=0.055,
            exit_method="multiple",
            exit_multiple=multiple_cons
        )
        assert math.isclose(scenario.year5_ebitda, ebitda_cons, abs_tol=1.0)
        assert math.isclose(scenario.enterprise_value, 34266.4, abs_tol=5.0)
        assert math.isclose(scenario.enterprise_value, scenario.year5_ebitda * multiple_cons, abs_tol=1e-4)

    # --------------------------------------------------------------------------
    # Test 2: Optimistic EBITDA * multiple = Optimistic EV
    # --------------------------------------------------------------------------
    def test_02_optimistic_ebitda_multiple_equals_optimistic_ev(self):
        """
        Invariant 2: Optimistic EBITDA multiplied by optimistic multiple equals optimistic EV.
        Exposes prior defect: Prior report stated Optimistic EBITDA = $4,347.0M and multiple = 13.00x,
        but published EV was $49,597.6M (which came from Base EBITDA $3,815.1M * 13.00x).
        Asserts true Optimistic EV is $4,347.0M * 13.00 = $56,511.0M (~$56.51B).
        """
        ebitda_opt = 4347.0
        multiple_opt = 13.00

        # Failing-before condition: prior published EV was $49,597.6M
        prior_published_ev = 49597.6
        assert not math.isclose(prior_published_ev, ebitda_opt * multiple_opt, rel_tol=0.05)

        # Passing-after condition: true optimistic EV strictly derived from its own inputs
        scenario = evaluate_scenario(
            scenario_name="Optimistic",
            base_revenue=self.BASE_REVENUE_TTM,
            rev_cagr=0.080,
            operating_margin_y5=0.225,
            dna_pct_y5=0.050,
            exit_method="multiple",
            exit_multiple=multiple_opt
        )
        assert math.isclose(scenario.year5_ebitda, ebitda_opt, abs_tol=1.0)
        assert math.isclose(scenario.enterprise_value, 56511.0, abs_tol=5.0)
        assert math.isclose(scenario.enterprise_value, scenario.year5_ebitda * multiple_opt, abs_tol=1e-4)

    # --------------------------------------------------------------------------
    # Test 3: Altering scenario revenue changes its own EBITDA and terminal EV
    # --------------------------------------------------------------------------
    def test_03_altering_revenue_changes_ebitda_and_ev(self):
        """
        Invariant 3: Altering scenario revenue changes its own EBITDA and terminal EV.
        """
        scen_6pct = evaluate_scenario("Base_6", self.BASE_REVENUE_TTM, 0.06, 0.215, 0.05, "multiple", exit_multiple=10.65)
        scen_7pct = evaluate_scenario("Base_7", self.BASE_REVENUE_TTM, 0.07, 0.215, 0.05, "multiple", exit_multiple=10.65)

        assert scen_7pct.year5_revenue > scen_6pct.year5_revenue
        assert scen_7pct.year5_ebitda > scen_6pct.year5_ebitda
        assert scen_7pct.enterprise_value > scen_6pct.enterprise_value
        # Exact mathematical relationship:
        ratio = scen_7pct.year5_revenue / scen_6pct.year5_revenue
        assert math.isclose(scen_7pct.year5_ebitda / scen_6pct.year5_ebitda, ratio, rel_tol=1e-4)
        assert math.isclose(scen_7pct.enterprise_value / scen_6pct.enterprise_value, ratio, rel_tol=1e-4)

    # --------------------------------------------------------------------------
    # Test 4: Altering scenario exit multiple changes only valuation-dependent outputs
    # --------------------------------------------------------------------------
    def test_04_altering_multiple_changes_only_valuation_outputs(self):
        """
        Invariant 4: Altering scenario exit multiple changes only valuation-dependent outputs,
        leaving operating EBITDA, revenue, CapEx, cash, and debt unchanged.
        """
        scen_10x = evaluate_scenario("Base_10x", self.BASE_REVENUE_TTM, 0.06, 0.215, 0.05, "multiple", exit_multiple=10.0)
        scen_12x = evaluate_scenario("Base_12x", self.BASE_REVENUE_TTM, 0.06, 0.215, 0.05, "multiple", exit_multiple=12.0)

        # Operating and balance-sheet metrics MUST be identical:
        assert math.isclose(scen_10x.year5_ebitda, scen_12x.year5_ebitda, abs_tol=1e-5)
        assert math.isclose(scen_10x.balance_sheet.ending_net_debt, scen_12x.balance_sheet.ending_net_debt, abs_tol=1e-5)
        assert math.isclose(scen_10x.ending_diluted_shares, scen_12x.ending_diluted_shares, abs_tol=1e-5)

        # Valuation outputs MUST scale:
        assert scen_12x.enterprise_value > scen_10x.enterprise_value
        assert scen_12x.terminal_share_price > scen_10x.terminal_share_price
        assert math.isclose(scen_12x.enterprise_value / scen_10x.enterprise_value, 12.0 / 10.0, rel_tol=1e-4)

    # --------------------------------------------------------------------------
    # Test 5: Repurchase expenditure and average price determine shares repurchased
    # --------------------------------------------------------------------------
    def test_05_repurchase_expenditure_and_price_determine_shares(self):
        """
        Invariant 5: Shares repurchased = expenditure / average purchase price.
        Exposes prior defect: Spending $5,476.2M buybacks but retiring only 3.7M shares implies $1,480/share!
        """
        # Failing-before condition:
        flawed_spend = 5476.2
        flawed_net_shares_retired = 3.7
        flawed_implied_price = flawed_spend / flawed_net_shares_retired
        assert flawed_implied_price > 1400.0, "Prior model had $1,480/share implied price!"

        # Passing-after condition:
        sb = build_share_count_bridge(
            beginning_basic_shares=484.524,
            repurchase_expenditure=1800.0,
            weighted_avg_repurchase_price=75.0,
            shares_issued_sbc=11.0,
            dilutive_securities=4.0
        )
        assert math.isclose(sb.shares_repurchased, 1800.0 / 75.0, abs_tol=1e-5)
        assert math.isclose(sb.shares_repurchased, 24.0, abs_tol=1e-5)

    # --------------------------------------------------------------------------
    # Test 6: Issuance minus repurchases determines ending actual shares
    # --------------------------------------------------------------------------
    def test_06_issuance_minus_repurchases_determines_ending_actual_shares(self):
        """
        Invariant 6: Ending actual shares = Beginning actual - repurchased + issued_sbc + issued_mna.
        """
        beginning = 484.524
        repurchased = 24.0
        issued_sbc = 11.0
        issued_mna = 0.0

        sb = build_share_count_bridge(beginning, 1800.0, 75.0, issued_sbc, 4.0, issued_mna)
        expected_actual = beginning - repurchased + issued_sbc + issued_mna
        assert math.isclose(sb.ending_actual_shares, expected_actual, abs_tol=1e-5)
        assert math.isclose(sb.ending_actual_shares, 471.524, abs_tol=1e-3)

    # --------------------------------------------------------------------------
    # Test 7: Actual shares and dilutive securities determine forecast diluted shares
    # --------------------------------------------------------------------------
    def test_07_actual_shares_and_dilutive_securities_determine_diluted_shares(self):
        """
        Invariant 7: Ending diluted shares = Ending actual shares + dilutive securities.
        """
        sb = build_share_count_bridge(484.524, 1800.0, 75.0, 11.0, dilutive_securities=4.0)
        assert math.isclose(sb.ending_diluted_shares, sb.ending_actual_shares + 4.0, abs_tol=1e-5)
        assert math.isclose(sb.ending_diluted_shares, 475.524, abs_tol=1e-3)

    # --------------------------------------------------------------------------
    # Test 8: Repurchase cash expenditure reduces cash available for debt paydown
    # --------------------------------------------------------------------------
    def test_08_repurchase_expenditure_reduces_cash_for_debt_paydown(self):
        """
        Invariant 8: Cash available for deleveraging = FCFE - dividends - repurchases.
        Increasing repurchases dollar-for-dollar reduces cash available for debt paydown.
        """
        bs_low_buyback = build_balance_sheet_bridge(
            starting_gross_debt=4710.0, starting_cash=1462.0, cumulative_rev=64282.5,
            ocf_margin_pct=0.21, capex_pct_of_rev=0.056, strategic_mna=1000.0,
            cash_interest_paid=900.0, cumulative_dividends=1026.0, share_repurchases=1000.0,
            gross_debt_repayment_plan=2150.0
        )
        bs_high_buyback = build_balance_sheet_bridge(
            starting_gross_debt=4710.0, starting_cash=1462.0, cumulative_rev=64282.5,
            ocf_margin_pct=0.21, capex_pct_of_rev=0.056, strategic_mna=1000.0,
            cash_interest_paid=900.0, cumulative_dividends=1026.0, share_repurchases=2000.0,
            gross_debt_repayment_plan=2150.0
        )
        delta_buybacks = 2000.0 - 1000.0
        delta_available = bs_low_buyback.cash_available_for_deleveraging - bs_high_buyback.cash_available_for_deleveraging
        assert math.isclose(delta_available, delta_buybacks, abs_tol=1e-4)

    # --------------------------------------------------------------------------
    # Test 9: Removing repurchases changes terminal shares and balance sheet consistently
    # --------------------------------------------------------------------------
    def test_09_removing_repurchases_changes_shares_and_balance_sheet_consistently(self):
        """
        Invariant 9: Removing buybacks increases terminal shares and increases ending cash / debt reduction.
        """
        scen_with_buyback = evaluate_scenario("WithBuyback", self.BASE_REVENUE_TTM, 0.06, 0.215, 0.05, "multiple", exit_multiple=10.65, buyback_total=1800.0)
        scen_zero_buyback = evaluate_scenario("ZeroBuyback", self.BASE_REVENUE_TTM, 0.06, 0.215, 0.05, "multiple", exit_multiple=10.65, buyback_total=0.0)

        # Without buybacks, shares are higher:
        assert scen_zero_buyback.ending_diluted_shares > scen_with_buyback.ending_diluted_shares
        assert math.isclose(scen_zero_buyback.ending_diluted_shares - scen_with_buyback.ending_diluted_shares, 1800.0 / 75.0, abs_tol=1e-3)

        # Without buybacks, cash balance is higher (or net debt is lower):
        assert scen_zero_buyback.balance_sheet.ending_cash > scen_with_buyback.balance_sheet.ending_cash
        assert math.isclose(scen_zero_buyback.balance_sheet.ending_cash - scen_with_buyback.balance_sheet.ending_cash, 1800.0, abs_tol=1e-3)

    # --------------------------------------------------------------------------
    # Test 10: Interest expense is deducted once under selected accounting convention
    # --------------------------------------------------------------------------
    def test_10_interest_expense_deducted_once_under_accounting_convention(self):
        """
        Invariant 10: Under IFRS Form 20-F, interest paid is in operating activities (levered OCF).
        If unlevered OCF is used, interest is deducted once in financing.
        Fails if interest is deducted twice or omitted.
        """
        cum_rev = 64282.5
        interest = 900.0

        # Unlevered convention: OCF is pre-interest; interest deducted in FCFE
        bs_unlevered = build_balance_sheet_bridge(
            4710.0, 1462.0, cum_rev, ocf_margin_pct=0.21, capex_pct_of_rev=0.056,
            strategic_mna=1000.0, cash_interest_paid=interest, cumulative_dividends=1026.0,
            share_repurchases=1800.0, gross_debt_repayment_plan=2150.0, is_levered_ocf=False
        )

        # Levered convention (IFRS Form 20-F): OCF margin is net of interest (0.21 - 900/cum_rev)
        levered_ocf_margin = 0.21 - (interest / cum_rev)
        bs_levered = build_balance_sheet_bridge(
            4710.0, 1462.0, cum_rev, ocf_margin_pct=levered_ocf_margin, capex_pct_of_rev=0.056,
            strategic_mna=1000.0, cash_interest_paid=interest, cumulative_dividends=1026.0,
            share_repurchases=1800.0, gross_debt_repayment_plan=2150.0, is_levered_ocf=True
        )

        # Both conventions MUST yield the exact same FCFE and ending cash:
        assert math.isclose(bs_unlevered.fcfe, bs_levered.fcfe, abs_tol=1e-3)
        assert math.isclose(bs_unlevered.ending_net_debt, bs_levered.ending_net_debt, abs_tol=1e-3)

    # --------------------------------------------------------------------------
    # Test 11: Closing debt and cash reconcile to opening balances & transactions
    # --------------------------------------------------------------------------
    def test_11_closing_debt_and_cash_reconcile_to_opening_balances(self):
        """
        Invariant 11:
        Ending Gross Debt = Starting Gross Debt - Repayments.
        Ending Cash = Starting Cash + Net Cash Flow.
        Ending Net Debt = Ending Gross Debt - Ending Cash.
        """
        bs = build_balance_sheet_bridge(
            starting_gross_debt=4710.0, starting_cash=1462.0, cumulative_rev=64282.5,
            ocf_margin_pct=0.21, capex_pct_of_rev=0.056, strategic_mna=1000.0,
            cash_interest_paid=900.0, cumulative_dividends=1026.0, share_repurchases=1800.0,
            gross_debt_repayment_plan=2150.0
        )
        assert math.isclose(bs.ending_gross_debt, 4710.0 - bs.gross_debt_repayment, abs_tol=1e-5)
        assert math.isclose(bs.ending_cash, 1462.0 + bs.change_in_cash, abs_tol=1e-5)
        assert math.isclose(bs.ending_net_debt, bs.ending_gross_debt - bs.ending_cash, abs_tol=1e-5)
        assert math.isclose(bs.net_debt_reduction, bs.starting_net_debt - bs.ending_net_debt, abs_tol=1e-5)

    # --------------------------------------------------------------------------
    # Test 12: Every scenario's dated return uses its own terminal price & dividends
    # --------------------------------------------------------------------------
    def test_12_scenario_dated_return_uses_own_terminal_price(self):
        """
        Invariant 12: Conservative, Base, and Optimistic dated returns MUST be calculated
        strictly from their own terminal prices and dividend schedule.
        """
        scen_cons = evaluate_scenario("Conservative", self.BASE_REVENUE_TTM, 0.045, 0.185, 0.055, "multiple", exit_multiple=10.65, avg_repurchase_price=65.0)
        scen_base = evaluate_scenario("Base", self.BASE_REVENUE_TTM, 0.060, 0.215, 0.050, "multiple", exit_multiple=10.65, avg_repurchase_price=75.0)
        scen_opt = evaluate_scenario("Optimistic", self.BASE_REVENUE_TTM, 0.080, 0.225, 0.050, "multiple", exit_multiple=13.00, avg_repurchase_price=85.0)

        # Terminal prices must be strictly ascending:
        assert scen_cons.terminal_share_price < scen_base.terminal_share_price < scen_opt.terminal_share_price

        # Dated returns must strictly correspond to those terminal prices:
        assert scen_cons.dated_xirr_from_spot < scen_base.dated_xirr_from_spot < scen_opt.dated_xirr_from_spot
        divs = [0.36, 0.39, 0.42, 0.45, 0.48]
        assert math.isclose(scen_cons.dated_xirr_from_spot, solve_dated_xirr(self.SPOT_PRICE, divs, scen_cons.terminal_share_price), rel_tol=1e-5)
        assert math.isclose(scen_base.dated_xirr_from_spot, solve_dated_xirr(self.SPOT_PRICE, divs, scen_base.terminal_share_price), rel_tol=1e-5)
        assert math.isclose(scen_opt.dated_xirr_from_spot, solve_dated_xirr(self.SPOT_PRICE, divs, scen_opt.terminal_share_price), rel_tol=1e-5)

    # --------------------------------------------------------------------------
    # Test 13: Accumulation ceilings reproduce each scenario's dated return hurdle
    # --------------------------------------------------------------------------
    def test_13_accumulation_ceilings_reproduce_dated_return_hurdles(self):
        """
        Invariant 13: Entry at accumulation_ceiling_85 MUST yield exactly 8.50% Dated XIRR.
        Entry at accumulation_ceiling_100 MUST yield exactly 10.00% Dated XIRR.
        """
        scen = evaluate_scenario("Base", self.BASE_REVENUE_TTM, 0.060, 0.215, 0.050, "multiple", exit_multiple=10.65)
        divs = [0.36, 0.39, 0.42, 0.45, 0.48]

        # Test 8.5% hurdle:
        p_entry_85 = scen.accumulation_ceiling_85
        irr_85 = solve_dated_xirr(p_entry_85, divs, scen.terminal_share_price)
        assert math.isclose(irr_85, 0.085, abs_tol=1e-5)

        # Test 10.0% hurdle:
        p_entry_100 = scen.accumulation_ceiling_100
        irr_100 = solve_dated_xirr(p_entry_100, divs, scen.terminal_share_price)
        assert math.isclose(irr_100, 0.100, abs_tol=1e-5)

    # --------------------------------------------------------------------------
    # Test 14: No research valuation output activates execution or capital allocation
    # --------------------------------------------------------------------------
    def test_14_no_research_output_activates_execution_or_capital_allocation(self):
        """
        Invariant 14: Research outputs produce WATCH_CONDITION and VALUATION_REVIEW_TRIGGER only.
        Never AUTOMATIC_BUY, AUTOMATIC_ALLOCATION, or EXECUTION_AUTHORIZATION.
        """
        forbidden_tokens = ["AUTOMATIC_BUY", "AUTOMATIC_ALLOCATION", "EXECUTION_AUTHORIZATION", "TRIGGER_BUY", "TRIGGER_PURCHASE"]

        with open("ARX_ALC_CORRECTED_INVESTMENT_DECISION_RECORD.md", "r", encoding="utf-8") as f:
            content = f.read()

        for token in forbidden_tokens:
            assert token not in content, f"Forbidden execution trigger '{token}' found!"

        assert "WATCH_CONDITION" in content
        assert "VALUATION_REVIEW_TRIGGER" in content
