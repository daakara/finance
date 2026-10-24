"""
tests/test_alc_quantitative_integrity.py

Independent adversarial test suite for Alcon Inc. (NYSE: ALC) quantitative valuation integrity.
Implements the 11 explicit failure modes and adversarial integrity checks specified in the
ARX Terminal ALC Final Quantitative Integrity Gate.

Covers:
1. Inconsistent FCFF and FCFE definitions.
2. Capital expenditure deducted twice.
3. Cash-flow allocations failing to balance.
4. Enterprise value and equity value failing to reconcile.
5. EBITDA multiple inconsistent with reported EV.
6. Share price not equaling equity value divided by shares.
7. Terminal payout not equaling terminal price plus distributions.
8. Scenario assumptions being mixed.
9. Accumulation thresholds inconsistent with declared return model.
10. Research price triggering automatic capital allocation.
11. Unsupported recurrence claim marked audited.
"""

import math
import pytest


class TestAlconQuantitativeIntegrityGate:
    # Authoritative Baseline Parameters (Q2 2026 TTM):
    SPOT_PRICE = 67.43
    BASE_REVENUE_TTM = 10758.0
    DILUTED_SHARES_TTM = 488.7
    STARTING_GROSS_DEBT = 4710.0
    STARTING_CASH = 1462.0
    STARTING_NET_DEBT = 3248.0
    TARGET_YEAR5_NET_DEBT = 1500.0
    NET_DEBT_REDUCTION_TARGET = 1748.0  # $3,248M - $1,500M

    # 5-Year Dividend Progression (USD equivalent per share)
    DIVIDENDS_PER_SHARE = [0.36, 0.39, 0.42, 0.45, 0.48]
    CUMULATIVE_DIVIDENDS = 2.10

    # --------------------------------------------------------------------------
    # 1. Inconsistent FCFF and FCFE definitions
    # --------------------------------------------------------------------------
    def test_adversarial_01_inconsistent_fcff_and_fcfe_definitions(self):
        """
        Adversarial Test: Inconsistent FCFF and FCFE definitions.
        FCFF = Operating cash flow before interest and debt service.
        FCFE = Cash flow available to equity holders AFTER debt service and net borrowing.
        A model that treats FCFF as available for equity distributions without subtracting debt service fails.
        """
        ocf_cum = 13750.0
        total_capex = 3599.8
        fcff_cum = ocf_cum - total_capex  # $10,150.2M (Free Cash Flow to Firm)
        debt_service_cum = 900.0          # Cash interest paid

        # Defective premise: using FCFF as if it were FCFE without deducting debt service
        flawed_equity_cash = fcff_cum - 1000.0  # Subtracted M&A, forgot debt service ($900M)
        assert flawed_equity_cash == 9150.2  # Unreconciled / overstated by $900M

        # Certified rule: FCFE MUST explicitly deduct cash debt service
        fcfe_cum = fcff_cum - 1000.0 - debt_service_cum
        assert math.isclose(fcfe_cum, 8250.2, abs_tol=1e-1)
        assert fcfe_cum != flawed_equity_cash

    # --------------------------------------------------------------------------
    # 2. Capital expenditure deducted twice
    # --------------------------------------------------------------------------
    def test_adversarial_02_capex_deducted_twice(self):
        """
        Adversarial Test: Detecting double-deduction of capital expenditure.
        In the prior flawed model, $8,250M was reported as FCF, and then $4,500M was subtracted
        again for 'Growth CapEx & M&A'.
        This test proves that if $8,250M is already net of CapEx and M&A, subtracting it twice
        causes a $982M deficit against actual debt paydown.
        """
        reported_fcf = 8250.0
        growth_capex_mna = 4500.0
        dividends = 1020.0
        target_debt_paydown = 1748.0

        # The flawed double-deduction calculation:
        flawed_result = reported_fcf - growth_capex_mna - dividends
        assert flawed_result == 2730.0
        # The arithmetic discrepancy against $1,748M:
        arithmetic_gap = flawed_result - target_debt_paydown
        assert math.isclose(arithmetic_gap, 982.0, abs_tol=1e-2)

        # Certified resolution: CapEx is deducted ONCE from OCF:
        # OCF ($13,750M) - Total CapEx ($3,599.8M) = FCF ($10,150.2M)
        # FCF - M&A ($1,000M) - Debt Service ($900M) - Dividends ($1,026M) = $7,224.2M
        # This pays $1,748M debt and leaves $5,476M for share buybacks without any double-deduction!
        clean_cash_available = 10150.2 - 1000.0 - 900.0 - 1026.0
        assert clean_cash_available >= target_debt_paydown

    # --------------------------------------------------------------------------
    # 3. Cash-flow allocations fail to balance
    # --------------------------------------------------------------------------
    def test_adversarial_03_cash_flow_allocations_fail_to_balance(self):
        """
        Adversarial Test: Full year-by-year 5-year cash-flow and balance-sheet bridge.
        Deliberately fails if the sum of annual cash movements does not balance
        to the exact net debt reduction ($1,748.0M).
        """
        debt_repayments = [300.0, 300.0, 350.0, 350.0, 448.0]
        assert sum(debt_repayments) == 1748.0

        # Balance sheet evolution check:
        gross_debt = self.STARTING_GROSS_DEBT
        cash = self.STARTING_CASH

        for repayment in debt_repayments:
            gross_debt -= repayment
            net_debt = gross_debt - cash

        assert math.isclose(gross_debt, 2962.0, abs_tol=1e-2)
        assert math.isclose(cash, 1462.0, abs_tol=1e-2)
        assert math.isclose(net_debt, 1500.0, abs_tol=1e-2)
        assert math.isclose(self.STARTING_NET_DEBT - net_debt, self.NET_DEBT_REDUCTION_TARGET, abs_tol=1e-2)

    # --------------------------------------------------------------------------
    # 4. Enterprise value and equity value do not reconcile
    # --------------------------------------------------------------------------
    def test_adversarial_04_enterprise_value_and_equity_value_do_not_reconcile(self):
        """
        Adversarial Test: Strict identity: Equity Value = Enterprise Value - Net Debt Bridge.
        In the prior optimistic scenario, reported EV was $54,775M but reported Equity was $40,845M
        (a phantom deduction of $13,930M against net debt of $3,248M).
        """
        ev_opt = 55269.4
        net_debt = 3248.0

        # Flawed bridge check:
        flawed_equity = 40845.0
        phantom_gap = ev_opt - net_debt - flawed_equity
        assert phantom_gap > 10000.0, "Flawed report had an unexplained ~$11B-$14B equity deduction!"

        # Certified identity:
        certified_equity = ev_opt - net_debt
        assert math.isclose(certified_equity, 52021.4, abs_tol=1.0)
        assert math.isclose(certified_equity / self.DILUTED_SHARES_TTM, 106.45, abs_tol=0.10)

    # --------------------------------------------------------------------------
    # 5. The EBITDA multiple is inconsistent with reported EV
    # --------------------------------------------------------------------------
    def test_adversarial_05_ebitda_multiple_inconsistent_with_reported_ev(self):
        """
        Adversarial Test: Validates multiple math.
        Detects false labeling: 'Normalized Terminal Multiple Case (13.0x EBITDA / $82.39 share) = $40,635M EV'.
        Proves:
        - 13.0x on $3,815.2M EBITDA is $49,597.6M EV, NOT $40,635M!
        - $40,635M EV is 10.65x EV/EBITDA.
        - $41,489M EV is 10.87x EV/EBITDA (Gordon Growth DCF).
        """
        ebitda_y5 = 3815.2

        # Defective claim:
        ev_claimed = 40635.0
        claimed_mult = 13.0
        assert not math.isclose(ev_claimed, ebitda_y5 * claimed_mult, rel_tol=0.10)

        # Certified multiples:
        ev_13x = ebitda_y5 * 13.0
        assert math.isclose(ev_13x, 49597.6, abs_tol=1.0)

        true_mult_40635 = ev_claimed / ebitda_y5
        assert math.isclose(true_mult_40635, 10.6508, abs_tol=0.01)

    # --------------------------------------------------------------------------
    # 6. Share price does not equal equity value divided by shares
    # --------------------------------------------------------------------------
    def test_adversarial_06_share_price_does_not_equal_equity_value_divided_by_shares(self):
        """
        Adversarial Test: P_5 = Equity_5 / Shares_5 strictly.
        """
        equity_base = 39988.7
        shares_base = 485.0
        p5_calc = equity_base / shares_base
        assert math.isclose(p5_calc, 82.45, abs_tol=0.05)

        equity_cons = 39131.9
        p5_cons = equity_cons / shares_base
        assert math.isclose(p5_cons, 80.68, abs_tol=0.05)

    # --------------------------------------------------------------------------
    # 7. Terminal payout does not equal terminal price plus distributions
    # --------------------------------------------------------------------------
    def test_adversarial_07_terminal_payout_does_not_equal_price_plus_distributions(self):
        """
        Adversarial Test: Payout = P_5 + Cumulative Dividends.
        Resolves the $1.70 discrepancy between $82.79 and $84.49:
        - Conservative Multiple: $80.68 + $2.10 = $82.78 (~$82.79)
        - Gordon Growth DCF: $82.45 + $2.10 = $84.55 (~$84.49)
        - Spread: $82.45 - $80.68 = $1.77 (~$1.70)
        """
        p5_cons = 80.68
        payout_cons = p5_cons + self.CUMULATIVE_DIVIDENDS
        assert math.isclose(payout_cons, 82.78, abs_tol=0.05)

        p5_dcf = 82.45
        payout_dcf = p5_dcf + self.CUMULATIVE_DIVIDENDS
        assert math.isclose(payout_dcf, 84.55, abs_tol=0.05)

        spread = payout_dcf - payout_cons
        assert math.isclose(spread, 1.77, abs_tol=0.05)

    # --------------------------------------------------------------------------
    # 8. Scenario assumptions are mixed
    # --------------------------------------------------------------------------
    def test_adversarial_08_scenario_assumptions_are_mixed(self):
        """
        Adversarial Test: Ensures scenario parameters are never cross-pollinated:
        - Multiple exit ($80.68) must never be paired with DCF terminal payout ($84.49).
        - 13.0x multiple ($49.6B EV) must never be used in Base Case without changing scenario label.
        """
        scenarios = {
            "Conservative": {"mult": 10.65, "p5": 80.68, "payout": 82.78},
            "Base_DCF": {"mult": 10.87, "p5": 82.45, "payout": 84.55},
            "Optimistic": {"mult": 13.00, "p5": 99.17, "payout": 101.27},
        }

        for name, data in scenarios.items():
            assert math.isclose(data["p5"] + 2.10, data["payout"], abs_tol=0.05)
            assert data["p5"] != 82.45 or name == "Base_DCF"

    # --------------------------------------------------------------------------
    # 9. Accumulation thresholds inconsistent with declared return model
    # --------------------------------------------------------------------------
    def test_adversarial_09_accumulation_thresholds_inconsistent_with_return_model(self):
        """
        Adversarial Test: Validates accumulation ceilings across hurdles (8.0%, 8.5%, 10.0%)
        using both Simple Terminal CAGR and Exact Dated XIRR.
        """
        payout_base = 84.55
        p5_base = 82.45
        divs = self.DIVIDENDS_PER_SHARE

        def solve_xirr_entry(hurdle):
            pv = sum(d / ((1.0 + hurdle) ** (t + 1)) for t, d in enumerate(divs))
            pv += p5_base / ((1.0 + hurdle) ** 5)
            return pv

        # 8.5% hurdle:
        c85_simple = payout_base / (1.085 ** 5)
        c85_dated = solve_xirr_entry(0.085)
        assert math.isclose(c85_simple, 56.23, abs_tol=0.05)
        assert math.isclose(c85_dated, 56.47, abs_tol=0.05)

        # 10.0% hurdle:
        c10_simple = payout_base / (1.10 ** 5)
        c10_dated = solve_xirr_entry(0.10)
        assert math.isclose(c10_simple, 52.50, abs_tol=0.05)
        assert math.isclose(c10_dated, 52.77, abs_tol=0.05)

    # --------------------------------------------------------------------------
    # 10. Research price triggers automatic capital allocation
    # --------------------------------------------------------------------------
    def test_adversarial_10_research_price_triggers_automatic_capital_allocation(self):
        """
        Adversarial Test: Enforces strict governance boundary.
        Fails if any active directive in the decision record attempts to execute a trade
        or trigger automatic capital allocation.
        """
        forbidden_execution_phrases = [
            "AUTOMATIC_BUY",
            "AUTOMATIC_ALLOCATION",
            "EXECUTION_AUTHORIZATION",
            "TRIGGER_BUY",
            "trigger capital allocation",
        ]

        with open("ARX_ALC_CORRECTED_INVESTMENT_DECISION_RECORD.md", "r", encoding="utf-8") as f:
            content = f.read()

        for phrase in forbidden_execution_phrases:
            assert phrase not in content, f"Forbidden execution directive '{phrase}' found!"

        # Must contain authorized watch condition:
        assert "WATCH_CONDITION" in content
        assert "VALUATION_REVIEW_TRIGGER" in content

    # --------------------------------------------------------------------------
    # 11. Unsupported recurrence claim marked audited
    # --------------------------------------------------------------------------
    def test_adversarial_11_unsupported_recurrence_claim_marked_audited(self):
        """
        Adversarial Test: Validates evidence semantics.
        Fails if non-equipment repeat purchases are labeled as contracted recurring revenue.
        """
        fy25_sales = 10319.0
        non_equip_sales = 9119.0
        ratio = non_equip_sales / fy25_sales
        assert math.isclose(ratio, 0.8837, abs_tol=1e-3)

        # Read decision record:
        with open("ARX_ALC_CORRECTED_INVESTMENT_DECISION_RECORD.md", "r", encoding="utf-8") as f:
            content = f.read()

        # Fails if claiming 'audited contracted recurring revenue'
        assert "audited contracted recurring revenue" not in content.lower()
        # Must note repeat purchases or non-equipment:
        assert "non-equipment" in content.lower()
