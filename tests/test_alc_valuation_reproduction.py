"""
tests/test_alc_valuation_reproduction.py

Deterministic reproduction and audit test harness for Alcon Inc. (ALC) institutional valuation.
Reconciles arithmetic against canonical ARX valuation functions:
- run_fcff_model()
- solve_reverse_dcf()
from research_canonical_valuation.py (derived directly from origin/main:analyst_dashboard/research/sourced_fundamentals/valuation.py).
"""

import pytest
import math
import research_canonical_valuation as rcv


class TestAlconValuationReproductionHarness:
    """Deterministic unit tests validating valuation formulas, reverse solver, and shareholder returns."""

    # Reconciled Authoritative Inputs (as of Q2 2026, June 30, 2026 / Oct 21, 2026 market close)
    BASE_REVENUE_TTM = 10758.0  # FY2025 ($10,319M) - H1 2025 ($5,028M) + H1 2026 ($5,467M)
    DILUTED_SHARES = 488.7      # Period weighted-average diluted shares (Form 6-K Q2 2026 Note 8)
    NET_DEBT_BRIDGE = 3248.0    # Total Financial Debt ($4,719M) - Total Cash & Liquidity ($1,471M)
    SPOT_PRICE = 67.43          # NYSE close October 21, 2026
    EFFECTIVE_TAX_RATE = 19.0   # Statutory corporate rate

    def test_01_canonical_engine_import(self):
        """Verifies that actual canonical ARX valuation functions are invoked."""
        assert callable(rcv.run_fcff_model)
        assert callable(rcv.solve_reverse_dcf)

    def test_02_reproduce_audit_discrepancy_5_reinvestment_convention(self):
        """
        Diagnoses Discrepancy 5:
        Previous report claimed EV=$35,844M at 7.14% CAGR, 20.5% margin, 19% tax, 3.25% reinvestment.
        Reproduces that canonical engine gives EV=$37,262M (not $35,844M) at 7.14% CAGR,
        and proves that the true growth solution at $35,844M is 6.24%.
        """
        # Net Reinvestment of 3.25% of Revenue corresponds to:
        # 3.25% / (20.5% * (1 - 0.19)) = 3.25 / 16.605 = 19.572418% of NOPAT
        reinv_nopat_pct = 3.25 / (0.205 * (1.0 - 0.19))

        # At 7.14% CAGR:
        res_714 = rcv.run_fcff_model(
            base_revenue=10773.0,  # Previous dossier revenue
            current_shares=483.4,  # Previous dossier shares
            net_debt_bridge=3248.0,
            growth_5y_pct=7.14,
            operating_margin_pct=20.5,
            wacc_pct=7.35,
            terminal_growth_pct=2.50,
            tax_rate_pct=19.0,
            reinvestment_rate_pct=reinv_nopat_pct,
        )
        assert math.isclose(res_714["enterprise_value"], 37261.56, rel_tol=1e-4)
        assert not math.isclose(res_714["enterprise_value"], 35844.0, rel_tol=1e-2)

        # Reverse solve for target EV=$35,844M:
        target_price = (35844.0 - 3248.0) / 483.4
        solved_cagr, status = rcv.solve_reverse_dcf(
            target_price=target_price,
            base_revenue=10773.0,
            current_shares=483.4,
            net_debt_bridge=3248.0,
            operating_margin_pct=20.5,
            wacc_pct=7.35,
            terminal_growth_pct=2.50,
            tax_rate_pct=19.0,
            reinvestment_rate_pct=reinv_nopat_pct,
        )
        assert status.startswith("VERIFIED")
        assert math.isclose(solved_cagr, 6.24, abs_tol=0.02)

    def test_03_authoritative_reconciled_reverse_dcf(self):
        """
        Solves market-implied growth using authoritative reconciled inputs:
        Base Revenue = $10,758M, Diluted Shares = 488.7M, Spot Price = $67.43, Net Debt = $3,248M.
        """
        reinv_nopat_pct = 3.25 / (0.205 * (1.0 - 0.19))
        solved_cagr, status = rcv.solve_reverse_dcf(
            target_price=self.SPOT_PRICE,
            base_revenue=self.BASE_REVENUE_TTM,
            current_shares=self.DILUTED_SHARES,
            net_debt_bridge=self.NET_DEBT_BRIDGE,
            operating_margin_pct=20.5,
            wacc_pct=7.35,
            terminal_growth_pct=2.50,
            tax_rate_pct=self.EFFECTIVE_TAX_RATE,
            reinvestment_rate_pct=reinv_nopat_pct,
        )
        assert status.startswith("VERIFIED")
        # At reconciled revenue $10,758M, market-implied CAGR is 6.51%
        assert math.isclose(solved_cagr, 6.51, abs_tol=0.05)

        # Roundtrip verification
        fwd_res = rcv.run_fcff_model(
            base_revenue=self.BASE_REVENUE_TTM,
            current_shares=self.DILUTED_SHARES,
            net_debt_bridge=self.NET_DEBT_BRIDGE,
            growth_5y_pct=solved_cagr,
            operating_margin_pct=20.5,
            wacc_pct=7.35,
            terminal_growth_pct=2.50,
            tax_rate_pct=self.EFFECTIVE_TAX_RATE,
            reinvestment_rate_pct=reinv_nopat_pct,
        )
        assert math.isclose(fwd_res["per_share"], self.SPOT_PRICE, rel_tol=1e-3)

    def test_04_equity_bridge_arithmetic_consistency(self):
        """
        Verifies that Equity Value = Enterprise Value - Net Debt Bridge strictly holds,
        eliminating arbitrary 'cash drag' or 'cash generation credit' additions/subtractions.
        """
        for name, g, m, w, tg, r_rev in [
            ("Conservative", 4.5, 18.5, 7.80, 2.0, 3.25),
            ("Base", 6.0, 20.5, 7.25, 2.5, 3.25),
            ("Optimistic", 8.0, 22.5, 6.75, 3.0, 3.25),
        ]:
            r_nopat = r_rev / ((m / 100.0) * (1.0 - (self.EFFECTIVE_TAX_RATE / 100.0)))
            res = rcv.run_fcff_model(
                base_revenue=self.BASE_REVENUE_TTM,
                current_shares=self.DILUTED_SHARES,
                net_debt_bridge=self.NET_DEBT_BRIDGE,
                growth_5y_pct=g,
                operating_margin_pct=m,
                wacc_pct=w,
                terminal_growth_pct=tg,
                tax_rate_pct=self.EFFECTIVE_TAX_RATE,
                reinvestment_rate_pct=r_nopat,
            )
            # Mathematical invariant: EV - NetDebt == Equity Value
            assert math.isclose(res["enterprise_value"] - self.NET_DEBT_BRIDGE, res["equity_value"], rel_tol=1e-7)
            # Mathematical invariant: Equity Value / Shares == Per Share Value
            assert math.isclose(res["equity_value"] / self.DILUTED_SHARES, res["per_share"], rel_tol=1e-7)

    def test_05_reconciled_scenario_present_values(self):
        """Calculates exact, certified scenario Present Values under authoritative reconciled inputs."""
        # 1. Conservative (4.5% CAGR, 18.5% margin, 7.80% WACC, 2.0% g)
        r_cons = 3.25 / (0.185 * 0.81)
        cons = rcv.run_fcff_model(
            self.BASE_REVENUE_TTM, self.DILUTED_SHARES, self.NET_DEBT_BRIDGE,
            4.5, 18.5, 7.80, 2.0, self.EFFECTIVE_TAX_RATE, r_cons
        )
        assert math.isclose(cons["per_share"], 44.05, abs_tol=0.10)
        assert math.isclose(cons["enterprise_value"], 24771.55, rel_tol=1e-3)

        # 2. Base (6.0% CAGR, 20.5% margin, 7.25% WACC, 2.5% g)
        r_base = 3.25 / (0.205 * 0.81)
        base = rcv.run_fcff_model(
            self.BASE_REVENUE_TTM, self.DILUTED_SHARES, self.NET_DEBT_BRIDGE,
            6.0, 20.5, 7.25, 2.5, self.EFFECTIVE_TAX_RATE, r_base
        )
        assert math.isclose(base["per_share"], 67.43, abs_tol=0.20)  # Almost exactly spot price!
        assert math.isclose(base["enterprise_value"], 36190.0, rel_tol=1e-3)

        # 3. Optimistic (8.0% CAGR, 22.5% margin, 6.75% WACC, 3.0% g)
        r_opt = 3.25 / (0.225 * 0.81)
        opt = rcv.run_fcff_model(
            self.BASE_REVENUE_TTM, self.DILUTED_SHARES, self.NET_DEBT_BRIDGE,
            8.0, 22.5, 6.75, 3.0, self.EFFECTIVE_TAX_RATE, r_opt
        )
        assert math.isclose(opt["per_share"], 106.49, abs_tol=0.20)
        assert math.isclose(opt["enterprise_value"], 55269.0, rel_tol=1e-3)

    def test_06_forward_five_year_total_return_and_irr(self):
        """
        Separates today's intrinsic value (PV DCF) from 5-year future shareholder returns (IRR).
        Models terminal equity value at Year 5, cumulative dividends, and calculates exact IRR.
        """
        # Base Case forward cash generation over 5 years:
        g = 0.06
        m = 0.205
        t = 0.19
        r = 0.0325
        rev = self.BASE_REVENUE_TTM
        cum_fcf = 0.0
        last_fcf = 0.0

        for year in range(1, 6):
            rev = rev * (1.0 + g)
            nopat = rev * m * (1.0 - t)
            fcf = nopat - (rev * r)
            cum_fcf += fcf
            last_fcf = fcf

        # Terminal Value at Year 5 (undiscounted at t=5):
        wacc = 0.0725
        tg = 0.025
        terminal_fcf = last_fcf * (1.0 + tg)
        ev_5 = terminal_fcf / (wacc - tg)

        # Reconcile Year 5 Net Debt:
        # Debt service: after-tax interest = 3.65% on $4,719M debt = $172.2M/yr * 5 = $861M
        fcfe = cum_fcf - 861.0
        # Dividend distributions: $2.10/share * 488.7M = $1,026M
        # Buybacks to neutralize SBC dilution: $1,500M
        # Remaining cash retained: $7,725M - $1,026M - $1,500M = $5,199M
        # Net debt paydown: Net debt drops from $3,248M to $0 (net cash of ~$1,950M).
        # To be conservative, assume $1,500M net debt remains (due to M&A / working capital):
        net_debt_5 = 1500.0
        equity_5 = ev_5 - net_debt_5
        shares_5 = 485.0  # Modest net reduction from $1.5B buyback
        p_5 = equity_5 / shares_5

        # Dividends received per share over 5 years
        cum_div_per_share = 2.10

        # Annualized IRR from purchase at spot ($67.43):
        irr_from_spot = ((p_5 + cum_div_per_share) / self.SPOT_PRICE) ** (1.0 / 5.0) - 1.0

        # At $67.43 spot, Base Case IRR is ~4.5% to 4.8%, far below 10% hurdle!
        assert 0.040 <= irr_from_spot <= 0.055

    def test_07_accumulation_ceiling_thresholds(self):
        """
        Calculates exact accumulation ceilings (NOT 'entry price floors')
        for 8.0%, 8.5%, and 10.0% annualized 5-year shareholder returns.
        """
        # From simplified terminal payout P_5 + D_cum = $70.00:
        c80_simple = 70.00 / ((1.0 + 0.08) ** 5)
        c85_simple = 70.00 / ((1.0 + 0.085) ** 5)
        c10_simple = 70.00 / ((1.0 + 0.10) ** 5)

        assert math.isclose(c80_simple, 47.64, abs_tol=0.02)
        assert math.isclose(c85_simple, 46.55, abs_tol=0.02)
        assert math.isclose(c10_simple, 43.46, abs_tol=0.02)

        # From forward modeled payoff P_5 + D_cum = $84.50:
        c80_fwd = 84.50 / ((1.0 + 0.08) ** 5)
        c85_fwd = 84.50 / ((1.0 + 0.085) ** 5)
        c10_fwd = 84.50 / ((1.0 + 0.10) ** 5)

        assert math.isclose(c80_fwd, 57.51, abs_tol=0.05)
        assert math.isclose(c85_fwd, 56.19, abs_tol=0.05)
        assert math.isclose(c10_fwd, 52.47, abs_tol=0.05)

        # In all forward models, an accumulation ceiling of <= $54.00 unlocks >= 9.3% IRR
        irr_at_54 = (84.50 / 54.0) ** (1.0 / 5.0) - 1.0
        assert irr_at_54 >= 0.093
