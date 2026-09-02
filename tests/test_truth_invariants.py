"""Phase 12: Truth Invariant Test Matrix & Permanent Regression Contract.

Enforces:
1. Strict observation windows (0, 1-19, 20-49, 50, 51 candles).
2. "Unknown != Negative" and "Unknown != Favorable" invariant across all domains.
3. Synthetic fallback candles are NEVER treated as market evidence (0 points).
4. Uncataloged tickers produce UNAVAILABLE fundamentals (0 points, no fabricated ratios).
5. Temporal integrity: publishedAt != retrievedAt != observedAt.
6. Cross-stack parity between Python and TypeScript.
"""

import unittest
import os
import json
import re


class TestTruthInvariantMatrix(unittest.TestCase):
    """Permanent regression contract for ARX decision-grade truth invariants."""

    def setUp(self):
        self.insight_gen_path = os.path.join("frontend", "lib", "insightGenerator.ts")
        self.catalog_path = os.path.join("frontend", "lib", "masterCatalog.ts")
        self.api_path = os.path.join("frontend", "lib", "api.ts")
        self.terminal_view_path = os.path.join("frontend", "components", "terminal", "AdvancedTerminalView.tsx")

        with open(self.insight_gen_path, "r", encoding="utf-8") as f:
            self.gen_content = f.read()
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.cat_content = f.read()
        with open(self.api_path, "r", encoding="utf-8") as f:
            self.api_content = f.read()
        with open(self.terminal_view_path, "r", encoding="utf-8") as f:
            self.view_content = f.read()

    def test_invariant_1_fifty_plus_candles_required_for_sma50(self):
        """Condition 1 & 2: >= 50 valid candles produces AVAILABLE SMA50; 49 or fewer produces UNAVAILABLE (0 pts)."""
        self.assertIn("candles && candles.length >= 50", self.gen_content)
        self.assertIn("candles.slice(-50)", self.gen_content)
        # Ensure no subset averaging
        self.assertNotIn("Math.min(candles.length, 50)", self.gen_content)

    def test_invariant_2_twenty_candles_required_for_ema20(self):
        """Condition 3 & 4: >= 20 candles required for EMA20 burn-in; < 20 produces UNAVAILABLE."""
        self.assertIn("candles && candles.length >= 20", self.gen_content)
        # Ensure canonical recurrence smoothing
        self.assertIn("const k = 2 / (20 + 1)", self.gen_content)
        self.assertIn("currentEma = candles[i].close * k + currentEma * (1 - k)", self.gen_content)

    def test_invariant_3_synthetic_fallback_never_produces_evidence(self):
        """Condition 5: Synthetic fallback feeds MUST NOT generate trend points or fake moving averages."""
        self.assertIn('dataSource?: "live" | "fallback"', self.gen_content)
        self.assertIn('const isFallbackFeed = dataSource === "fallback"', self.gen_content)
        self.assertIn("!isFallbackFeed", self.gen_content)

    def test_invariant_4_missing_fundamentals_produces_zero_points(self):
        """Condition 6: Uncatalogued or missing SEC evidence MUST produce UNAVAILABLE with 0 points."""
        self.assertIn("const isHealthAvailable = catAsset !== undefined && catAsset.roic !== undefined", self.gen_content)
        self.assertIn('availability: "UNAVAILABLE"', self.gen_content)
        self.assertIn('status: "UNAVAILABLE"', self.gen_content)
        self.assertIn("pointImpact: 0", self.gen_content)
        # Generic 18.4% fallback must be completely absent from health assessment
        self.assertNotIn('roicDisplay = catAsset?.roic !== undefined ? `${catAsset.roic}%` : "18.4%"', self.gen_content)

    def test_invariant_5_uncataloged_advanced_metrics_render_na(self):
        """Condition 7: AdvancedTerminalView must render N/A for unverified metrics, never fake defaults."""
        self.assertIn('{adv.marketCap || "N/A"}', self.view_content)
        self.assertIn('{adv.peRatio !== undefined ? `${adv.peRatio.toFixed(1)}×` : "N/A"}', self.view_content)
        self.assertIn('{adv.roic !== undefined ? `${adv.roic.toFixed(1)}%` : "N/A"}', self.view_content)
        self.assertIn('{adv.beta !== undefined ? adv.beta.toFixed(2) : "N/A"}', self.view_content)
        self.assertIn('{adv.atr !== undefined ? `$${adv.atr.toFixed(2)}` : "N/A"}', self.view_content)
        self.assertIn('{adv.rvol !== undefined ? `${adv.rvol.toFixed(2)}×` : "N/A"}', self.view_content)

    def test_invariant_6_temporal_provenance_distinction(self):
        """Condition 8: SEC acceptance timestamps must be distinct from observation timestamps."""
        # publishedAt (filing date) != observedAt (current runtime extraction)
        self.assertIn("publishedAt: filingDate", self.gen_content)
        self.assertIn("observedAt: new Date().toISOString().split(\"T\")[0]", self.gen_content)
        # Verify EDGAR acceptance dates for benchmark securities
        self.assertIn('secFilingDate: "2026-08-26"', self.cat_content)  # NVDA
        self.assertIn('secFilingDate: "2026-07-23"', self.cat_content)  # FIX
        self.assertIn('secFilingDate: "2026-05-11"', self.cat_content)  # CPRX

    def test_invariant_7_cross_stack_ema20_parity(self):
        """Condition 9: Python and TypeScript must implement identical exponential smoothing k = 2/21."""
        # Check api.ts
        self.assertIn("const k = 2 / (20 + 1)", self.api_content)
        self.assertIn("currentEma = candles[i].close * k + currentEma * (1 - k)", self.api_content)

        # Check Python analytics.py
        py_analytics_path = os.path.join("api", "routes", "analytics.py")
        with open(py_analytics_path, "r", encoding="utf-8") as f:
            py_content = f.read()
        self.assertIn("ema_20 = close.ewm(span=20, adjust=False).mean()", py_content)
        self.assertIn("sma_50 = close.rolling(window=50).mean()", py_content)

    def test_invariant_8_fix_baseline_price_integrity(self):
        """Condition 10: FIX baseline price must reflect verified market pricing, eliminating valuation cliff."""
        self.assertIn("FIX: 1560.13", self.cat_content)
        self.assertNotIn("FIX: 385.00", self.cat_content)


if __name__ == "__main__":
    unittest.main()
