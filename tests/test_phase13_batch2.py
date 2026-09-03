"""Phase 13.1 Product & UX Remediation - Batch 2 Regression Test Suite.

Enforces:
1. FINDING-13-10: Portfolio ownership preservation via `&ownership=OWNED` on position links.
2. FINDING-13-13: Context-sensitive primary CTA based on posture in terminal views.
3. FINDING-13-05: Target/R:R gating when trend is incomplete (target1/target2 undefined when !isTrendAvailable).
4. FINDING-13-01: Screener unification with MASTER_ASSET_CATALOG, eliminating BASE_PRICES (FIX at $1,560.13, not $346.20).
5. FINDING-13-12: Strategy catalog dynamic price hydration from MASTER_ASSET_CATALOG / SpotPriceRegistry.
6. FINDING-13-06: Ingestion error handling with explicit retry UI in frontend/app/page.tsx.
"""

import unittest
import os


class TestPhase13Batch2Remediations(unittest.TestCase):
    """Test suite for Phase 13.1 Batch 2 (P1 Journey & Context)."""

    def setUp(self):
        self.portfolio_path = os.path.join("frontend", "app", "portfolio", "page.tsx")
        self.standard_view_path = os.path.join("frontend", "components", "terminal", "StandardTerminalView.tsx")
        self.guided_view_path = os.path.join("frontend", "components", "terminal", "GuidedTerminalView.tsx")
        self.insight_path = os.path.join("frontend", "lib", "insightGenerator.ts")
        self.screener_path = os.path.join("frontend", "app", "screener", "page.tsx")
        self.strategy_path = os.path.join("frontend", "app", "strategy", "[type]", "page.tsx")
        self.home_path = os.path.join("frontend", "app", "page.tsx")

        with open(self.portfolio_path, "r", encoding="utf-8") as f:
            self.portfolio_content = f.read()
        with open(self.standard_view_path, "r", encoding="utf-8") as f:
            self.standard_view_content = f.read()
        with open(self.guided_view_path, "r", encoding="utf-8") as f:
            self.guided_view_content = f.read()
        with open(self.insight_path, "r", encoding="utf-8") as f:
            self.insight_content = f.read()
        with open(self.screener_path, "r", encoding="utf-8") as f:
            self.screener_content = f.read()
        with open(self.strategy_path, "r", encoding="utf-8") as f:
            self.strategy_content = f.read()
        with open(self.home_path, "r", encoding="utf-8") as f:
            self.home_content = f.read()

    def test_p1_1_portfolio_ownership_preservation(self):
        """FINDING-13-10: Clicking portfolio holding must pass ownership=OWNED to Terminal."""
        self.assertIn("ownership=OWNED", self.portfolio_content)
        self.assertIn("/?symbol=${pos.symbol}&ownership=OWNED", self.portfolio_content)

    def test_p1_2_context_sensitive_cta(self):
        """FINDING-13-13: Primary CTA must adapt to posture rather than always sizing positions."""
        # StandardTerminalView must inspect posture
        self.assertIn("insight.terminalState.posture", self.standard_view_content)
        # GuidedTerminalView must inspect posture
        self.assertIn("insight.terminalState.posture", self.guided_view_content)

    def test_p1_3_target_rr_gating_when_trend_unavailable(self):
        """FINDING-13-05: Targets and R:R must be gated when trend is unavailable."""
        self.assertIn("target1: isTrendAvailable ? target1 : undefined", self.insight_content)
        self.assertIn("target2: isTrendAvailable ? target2 : undefined", self.insight_content)
        self.assertIn("profitRiskRatio: isTrendAvailable ? profitRisk : undefined", self.insight_content)

    def test_p1_4_screener_catalog_unification(self):
        """FINDING-13-01: Screener must eliminate BASE_PRICES and unify with MASTER_ASSET_CATALOG."""
        self.assertNotIn("FIX: { price: 346.20", self.screener_content)
        self.assertNotIn("const BASE_PRICES: Record<string", self.screener_content)
        self.assertIn("MASTER_ASSET_CATALOG[sym]", self.screener_content)

    def test_p1_5_strategy_catalog_unification(self):
        """FINDING-13-12: Strategy page must hydrate candidate prices from MASTER_ASSET_CATALOG."""
        self.assertIn("MASTER_ASSET_CATALOG", self.strategy_content)
        self.assertIn("CATALOG_BASELINE_PRICES", self.strategy_content)

    def test_p1_6_home_error_retry_state(self):
        """FINDING-13-06: Ingestion errors must set error state and render explicit retry UI."""
        self.assertIn("setError", self.home_content)
        self.assertIn("Retry Analysis", self.home_content)


if __name__ == "__main__":
    unittest.main()
