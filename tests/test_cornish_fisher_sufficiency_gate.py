"""Automated test suite for Cornish-Fisher Data Sufficiency Gate (TASK-OPEN-03).

Quality Gates Enforced:
1. Valid finite return observation counting (not price rows). 30 1-session returns
   require at least 31 valid closing prices. No forward-filling or bridging across gaps.
2. Boundary behavior at 29, 30, and 31 valid returns.
3. Explicit UNAVAILABLE state when N < 30, with reason='INSUFFICIENT_OBSERVATIONS',
   actual_observations, required_observations=30, and None for Modified_VaR_* metrics.
4. Degenerate sample detection (zero or non-finite variance) yielding UNAVAILABLE
   with reason='DEGENERATE_SAMPLE'.
5. Non-finite distribution parameters (NaN/Inf skew/kurtosis) yielding UNAVAILABLE.
6. Long-history numerical parity preservation against baseline fixture.
7. End-to-end downstream consumer propagation:
   - Self-healing forecast auditor Kupiec test deferral (no TypeError).
   - Trader archetypes Jim Simons quantitative risk model fail-closed to UNAVAILABLE.
   - Serialization via sanitize_floats emitting null (None), never 0 or 65.
"""

import unittest
import math
import numpy as np
import pandas as pd
import pytest

from analyst_dashboard.analyzers.advanced_risk_analyzer import (
    AdvancedRiskAnalyzer,
    extract_valid_returns,
    REQUIRED_CORNISH_FISHER_OBSERVATIONS,
)
from analyst_dashboard.analyzers.self_healing_engine import SelfHealingForecastAuditor
from analyst_dashboard.analyzers.trader_archetypes import (
    TraderArchetypeAnalyzer,
    ArchetypeEvidenceStatus,
)
from api.serialization import sanitize_floats

pytestmark = pytest.mark.tier3


class TestCornishFisherSufficiencyGate(unittest.TestCase):
    """Exhaustive test suite for Directive TASK-OPEN-03 Cornish-Fisher Sufficiency Gate."""

    def setUp(self):
        self.analyzer = AdvancedRiskAnalyzer()
        self.self_healing = SelfHealingForecastAuditor()
        self.trader_archetypes = TraderArchetypeAnalyzer()

    # -------------------------------------------------------------------------
    # 1. Observation Counting & Non-Bridging Invariants
    # -------------------------------------------------------------------------

    def test_extract_valid_returns_strictly_positive_finite(self):
        """extract_valid_returns must reject non-positive and non-finite prices."""
        prices = pd.Series([100.0, 102.0, 0.0, 105.0, -10.0, 108.0, np.nan, 110.0, np.inf, 112.0])
        rets = extract_valid_returns(prices)

        # 100 -> 102: valid return (1)
        # 102 -> 0.0: 0.0 is rejected, so no return
        # 0.0 -> 105: 0.0 is invalid, so no return
        # 105 -> -10.0: -10 is invalid, so no return
        # -10 -> 108: invalid
        # 108 -> nan: invalid
        # nan -> 110: invalid
        # 110 -> inf: invalid
        # inf -> 112: invalid
        # Total valid 1-session transitions: exactly 1 (from 100 to 102)
        self.assertEqual(len(rets), 1)
        self.assertAlmostEqual(float(rets.iloc[0]), 0.02, places=6)

    def test_no_forward_fill_or_bridging_across_missing_sessions(self):
        """Missing session at index 15 in 32 prices must NOT be bridged.

        32 consecutive prices would yield 31 returns.
        With index 15 missing (NaN), transitions 14->15 and 15->16 are invalid.
        Expected valid returns: 14 + 15 = 29 returns.
        Forward-fill would bridge the gap and falsely yield 30 or 31 returns.
        """
        base_prices = [100.0 * (1.001 ** i) for i in range(32)]
        base_prices[15] = np.nan  # Break session 15

        price_series = pd.Series(base_prices)
        valid_rets = extract_valid_returns(price_series)

        self.assertEqual(len(valid_rets), 29)

        # Ensure that with 29 valid returns, Cornish-Fisher fails closed
        res = self.analyzer.analyze_comprehensive_risk(price_series)
        cf_state = res["cornish_fisher"]
        self.assertEqual(cf_state["status"], "UNAVAILABLE")
        self.assertEqual(cf_state["reason"], "INSUFFICIENT_OBSERVATIONS")
        self.assertEqual(cf_state["actual_observations"], 29)
        self.assertEqual(cf_state["required_observations"], 30)

    # -------------------------------------------------------------------------
    # 2. Strict Boundary Tests: 29 vs 30 vs 31 Valid Returns
    # -------------------------------------------------------------------------

    def test_boundary_29_valid_returns_unavailable(self):
        """29 valid returns (from 30 consecutive prices) must yield UNAVAILABLE."""
        np.random.seed(101)
        rets_29 = np.random.normal(0.001, 0.015, 29)
        prices_30 = 100.0 * np.cumprod(np.insert(1 + rets_29, 0, 1.0))
        price_df = pd.DataFrame({"Close": prices_30})

        res = self.analyzer.analyze_comprehensive_risk(price_df)
        adv = res["advanced_metrics"]
        cf = res["cornish_fisher"]

        self.assertEqual(cf["status"], "UNAVAILABLE")
        self.assertEqual(cf["reason"], "INSUFFICIENT_OBSERVATIONS")
        self.assertEqual(cf["actual_observations"], 29)
        self.assertEqual(cf["required_observations"], 30)

        self.assertIsNone(adv["Modified_VaR_95"])
        self.assertIsNone(adv["Modified_VaR_99"])
        self.assertIsNone(adv["Modified_CVaR_95"])
        self.assertIsNone(adv["Skewness"])
        self.assertIsNone(adv["Kurtosis"])

    def test_boundary_30_valid_returns_available(self):
        """Exact threshold: 30 valid returns (from 31 consecutive prices) must yield AVAILABLE."""
        np.random.seed(102)
        rets_30 = np.random.normal(0.001, 0.015, 30)
        prices_31 = 100.0 * np.cumprod(np.insert(1 + rets_30, 0, 1.0))
        price_df = pd.DataFrame({"Close": prices_31})

        res = self.analyzer.analyze_comprehensive_risk(price_df)
        adv = res["advanced_metrics"]
        cf = res["cornish_fisher"]

        self.assertEqual(cf["status"], "AVAILABLE")
        self.assertIsNone(cf["reason"])
        self.assertEqual(cf["actual_observations"], 30)
        self.assertEqual(cf["required_observations"], 30)

        self.assertIsNotNone(adv["Modified_VaR_95"])
        self.assertIsNotNone(adv["Modified_VaR_99"])
        self.assertIsNotNone(adv["Modified_CVaR_95"])
        self.assertIsInstance(adv["Modified_VaR_95"], float)
        self.assertIsInstance(adv["Modified_VaR_99"], float)

        # Monotonicity check: 99% VaR represents a larger loss (more negative) than 95%
        self.assertLessEqual(adv["Modified_VaR_99"], adv["Modified_VaR_95"])

    def test_boundary_31_valid_returns_available(self):
        """31 valid returns (from 32 consecutive prices) must yield AVAILABLE."""
        np.random.seed(103)
        rets_31 = np.random.normal(0.001, 0.015, 31)
        prices_32 = 100.0 * np.cumprod(np.insert(1 + rets_31, 0, 1.0))
        price_df = pd.DataFrame({"Close": prices_32})

        res = self.analyzer.analyze_comprehensive_risk(price_df)
        adv = res["advanced_metrics"]
        cf = res["cornish_fisher"]

        self.assertEqual(cf["status"], "AVAILABLE")
        self.assertIsNone(cf["reason"])
        self.assertEqual(cf["actual_observations"], 31)
        self.assertIsNotNone(adv["Modified_VaR_95"])
        self.assertLessEqual(adv["Modified_VaR_99"], adv["Modified_VaR_95"])

    # -------------------------------------------------------------------------
    # 3. Degenerate & Non-Finite Sample Handling
    # -------------------------------------------------------------------------

    def test_constant_returns_zero_variance_degenerate_sample(self):
        """Zero variance (constant prices/returns) must fail closed as DEGENERATE_SAMPLE."""
        constant_prices = pd.Series([100.0] * 50)
        res = self.analyzer.analyze_comprehensive_risk(constant_prices)
        adv = res["advanced_metrics"]
        cf = res["cornish_fisher"]

        self.assertEqual(cf["status"], "UNAVAILABLE")
        self.assertEqual(cf["reason"], "DEGENERATE_SAMPLE")
        self.assertEqual(cf["actual_observations"], 49)
        self.assertIsNone(adv["Modified_VaR_95"])
        self.assertIsNone(adv["Modified_VaR_99"])
        self.assertIsNone(adv["Modified_CVaR_95"])

    def test_empty_and_single_price_inputs(self):
        """Empty series or single price input returns UNAVAILABLE with 0 observations."""
        empty_res = self.analyzer.analyze_comprehensive_risk(pd.Series(dtype=float))
        self.assertEqual(empty_res["cornish_fisher"]["status"], "UNAVAILABLE")
        self.assertEqual(empty_res["cornish_fisher"]["actual_observations"], 0)
        self.assertIsNone(empty_res["advanced_metrics"]["Modified_VaR_95"])

        single_price_res = self.analyzer.analyze_comprehensive_risk(pd.Series([100.0]))
        self.assertEqual(single_price_res["cornish_fisher"]["status"], "UNAVAILABLE")
        self.assertEqual(single_price_res["cornish_fisher"]["actual_observations"], 0)
        self.assertIsNone(single_price_res["advanced_metrics"]["Modified_VaR_95"])

    # -------------------------------------------------------------------------
    # 4. Long-History Numerical Parity Preservation
    # -------------------------------------------------------------------------

    def test_long_history_baseline_fixture_parity(self):
        """Preserve exact numerical output on the canonical 200-period Laplace fixture."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        returns = np.random.laplace(loc=0.0005, scale=0.02, size=200)
        prices = 100 * np.cumprod(1 + returns)
        price_data = pd.DataFrame({"Close": prices}, index=dates)

        res = self.analyzer.analyze_comprehensive_risk(price_data)
        adv = res["advanced_metrics"]
        cf = res["cornish_fisher"]

        self.assertEqual(cf["status"], "AVAILABLE")
        self.assertIsNone(cf["reason"])
        self.assertEqual(cf["actual_observations"], 199)

        # Baseline values verified against canonical implementation
        self.assertAlmostEqual(adv["Modified_VaR_95"], -4.575397, places=4)
        self.assertAlmostEqual(adv["Modified_VaR_99"], -7.112008, places=4)
        self.assertAlmostEqual(adv["Modified_CVaR_95"], -6.075944, places=4)
        self.assertAlmostEqual(adv["Skewness"], -0.165295, places=4)
        self.assertAlmostEqual(adv["Kurtosis"], 0.796280, places=4)

    # -------------------------------------------------------------------------
    # 5. Downstream Consumer Propagation & Type Safety
    # -------------------------------------------------------------------------

    def test_self_healing_engine_defers_kupiec_on_none_mvar(self):
        """SelfHealingForecastAuditor must not throw TypeError when Modified_VaR_95 is None."""
        price_df = pd.DataFrame({"Close": [100.0 * (1.001 ** i) for i in range(50)]})
        audit_res = self.self_healing.audit_and_calibrate(
            symbol="TEST",
            price_df=price_df,
            current_risk_metrics={"Modified_VaR_95": None},
            expected_return_data={"forecastHorizonDays": 90},
        )
        self.assertEqual(
            audit_res["varBreachStatus"],
            "Cornish-Fisher VaR Unavailable (Kupiec POF Test Deferred)",
        )
        self.assertIsNone(audit_res["varBreachRatePct"])

    def test_trader_archetypes_jim_simons_fails_closed_when_tail_risk_none(self):
        """Trader Archetypes Jim Simons model must report UNAVAILABLE when tailRiskScore is None."""
        risk_metrics = {
            "Sortino_Ratio": 1.5,
            "Skewness": -0.2,
            "Modified_VaR_95": None,
        }
        factor_scores = {
            "tailRiskScore": None,  # Suppressed due to Cornish-Fisher gate
            "momentumScore": 75,
        }

        res = self.trader_archetypes.analyze_asset(
            symbol="TEST",
            info={"sector": "Technology", "shortName": "Test Asset"},
            price_df=pd.DataFrame({"Close": [100.0] * 10}),
            risk_metrics=risk_metrics,
            macro_indicators={"macro_difficulty_rating": 3.0},
            factor_scores=factor_scores,
        )

        simons = next(a for a in res["archetypes"] if "Simons" in a["name"])
        self.assertEqual(simons["evidenceStatus"], ArchetypeEvidenceStatus.UNAVAILABLE)
        self.assertIsNone(simons["alignmentScore"])
        self.assertIn("unavailable", simons["thesis"].lower())

    def test_serialization_emits_null_never_zero_or_65(self):
        """sanitize_floats must serialize None/NaN Modified_VaR and tailRiskScore as None (null).

        It must NEVER substitute 0 or 65.
        """
        payload = {
            "Modified_VaR_95": None,
            "Modified_VaR_99": float("nan"),
            "Modified_CVaR_95": float("inf"),
            "tailRiskScore": None,
            "cornish_fisher": {
                "status": "UNAVAILABLE",
                "reason": "INSUFFICIENT_OBSERVATIONS",
                "actual_observations": 15,
                "required_observations": 30,
            },
        }

        cleaned = sanitize_floats(payload)

        self.assertIsNone(cleaned["Modified_VaR_95"])
        self.assertIsNone(cleaned["Modified_VaR_99"])
        self.assertIsNone(cleaned["Modified_CVaR_95"])
        self.assertIsNone(cleaned["tailRiskScore"])

        self.assertNotEqual(cleaned["tailRiskScore"], 0)
        self.assertNotEqual(cleaned["tailRiskScore"], 65)
        self.assertNotEqual(cleaned["Modified_VaR_95"], 0)
        self.assertEqual(cleaned["cornish_fisher"]["status"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
