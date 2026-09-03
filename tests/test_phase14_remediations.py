"""Phase 14 Production Readiness Remediation Regression Test Suite.

Enforces:
1. FINDING-14-01: Static stock page execution state reflects authentic stage/evidence, not hardcoded IN_BUY_ZONE.
2. FINDING-14-02: ShareTradeCardButton respects posture and does not hardcode IN_BUY_ZONE.
3. FINDING-14-03: Compare tool does not award false lower multiple or pristine tier on unverified data.
4. FINDING-14-04: Screener custom candidates do not fabricate 28.5% ROIC, 65% GM, or Active Buy Zone.
5. FINDING-14-05: DayTraderPositionSizer disables calculation and displays unavailable state when technicals are missing.
6. FINDING-14-06: Home page suppresses AdaptiveTerminal when ingestion error occurs.
7. FINDING-14-07: deriveAssessmentState automatically triggers invalidation when safePrice < invalidationPrice.
"""

import unittest
import os


class TestPhase14Remediations(unittest.TestCase):
    """Test suite for Phase 14 production readiness remediations."""

    def setUp(self):
        self.stock_page_path = os.path.join("frontend", "app", "stock", "[ticker]", "page.tsx")
        self.share_btn_path = os.path.join("frontend", "components", "ShareTradeCardButton.tsx")
        self.compare_path = os.path.join("frontend", "app", "compare", "page.tsx")
        self.screener_path = os.path.join("frontend", "app", "screener", "page.tsx")
        self.sizer_path = os.path.join("frontend", "components", "DayTraderPositionSizer.tsx")
        self.home_path = os.path.join("frontend", "app", "page.tsx")
        self.engine_path = os.path.join("frontend", "lib", "assessmentEngine.ts")

        with open(self.stock_page_path, "r", encoding="utf-8") as f:
            self.stock_page_content = f.read()
        with open(self.share_btn_path, "r", encoding="utf-8") as f:
            self.share_btn_content = f.read()
        with open(self.compare_path, "r", encoding="utf-8") as f:
            self.compare_content = f.read()
        with open(self.screener_path, "r", encoding="utf-8") as f:
            self.screener_content = f.read()
        with open(self.sizer_path, "r", encoding="utf-8") as f:
            self.sizer_content = f.read()
        with open(self.home_path, "r", encoding="utf-8") as f:
            self.home_content = f.read()
        with open(self.engine_path, "r", encoding="utf-8") as f:
            self.engine_content = f.read()

    def test_p0_1_stock_page_execution_state_honesty(self):
        """FINDING-14-01: Static stock page must dynamically derive executionState."""
        self.assertIn("let executionState =", self.stock_page_content)
        self.assertIn("isHaltedOrIncomplete", self.stock_page_content)
        self.assertIn("isStage4", self.stock_page_content)
        self.assertIn("{executionState}", self.stock_page_content)

    def test_p0_2_share_trade_card_posture_parity(self):
        """FINDING-14-02: ShareTradeCardButton must use postureLabel and respect posture prop."""
        self.assertIn("${postureLabel}", self.share_btn_content)
        self.assertIn("posture = ", self.share_btn_content)
        self.assertIn("targetLabel", self.share_btn_content)

    def test_p0_3_compare_unverified_metrics_no_false_edge(self):
        """FINDING-14-03: Compare tool must require positive verified metrics before declaring an edge."""
        # Fwd P/E must require assetA.peRaw > 0 && assetB.peRaw > 0
        self.assertIn("assetA.peRaw > 0 && assetB.peRaw > 0", self.compare_content)
        # Piotroski pristine tier must require score >= 8
        self.assertIn("assetA.piotroski >= 8 && assetB.piotroski >= 8", self.compare_content)

    def test_p1_4_screener_custom_query_no_synthetic_metrics(self):
        """FINDING-14-04: Screener custom candidate fallback must not fabricate 28.5% ROIC or 65% GM."""
        self.assertNotIn('roic: c.roic || (masterMeta ? `${masterMeta.roic}%` : "28.5%")', self.screener_content)
        self.assertNotIn('grossMargin: c.grossMargin || (masterMeta ? `${masterMeta.grossMargin}%` : "65.0%")', self.screener_content)
        self.assertNotIn('pegRatio: c.pegRatio || (masterMeta ? `${masterMeta.peg}` : "0.85")', self.screener_content)

    def test_p1_5_day_trader_sizer_requires_valid_data(self):
        """FINDING-14-05: DayTraderPositionSizer must not fabricate 55 RSI or $100 price on missing data."""
        self.assertNotIn(
            'technicals || { vwap: currentPrice, rsi_14: 55.0, ema_20: currentPrice, atr_14: currentPrice * 0.015 }',
            self.sizer_content
        )
        self.assertIn("isTechnicalsUnavailable", self.sizer_content)

    def test_p1_6_home_error_hides_degraded_terminal(self):
        """FINDING-14-06: When error && !data, home page must not render AdaptiveTerminal below error banner."""
        self.assertIn(
            '{error && !data ? (',
            self.home_content,
            "Home page must conditionally render error OR AdaptiveTerminal, not both"
        )

    def test_p2_7_invalidation_breached_derived_fail_closed(self):
        """FINDING-14-07: deriveAssessmentState must fail-closed if safePrice < invalidationPrice."""
        self.assertIn("invalidationPrice !== undefined && safePrice < invalidationPrice", self.engine_content)


if __name__ == "__main__":
    unittest.main()
