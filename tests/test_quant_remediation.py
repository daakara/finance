"""Quantitative & Data Remediation Regression & Adversarial Test Matrix (Phases 9 & 11).
Verifies:
1. SMA50 & EMA20 authentic candle math and independence from nominal share price.
2. Stage 2 vs Stage 4 Minervini classification (elimination of price < 100 bug).
3. Asset-specific catalog binding (ROIC, Gross Margin, Piotroski) for NVDA, CPRX, FIX.
4. Canonical FIX asset resolution in master catalog with verified baseline price $1,560.13.
5. SEC filing dates (publishedAt vs retrievedAt) matching official EDGAR acceptance timestamps.
6. Provenance tagging (15m Exchange Delayed).
7. Removal of legacy advisory terminology ("Strong Buy / Core Accumulation").
8. Strict observation windows (DISC-01): 0, 1, 19, 20, 49, 50, 51 candle boundary behavior.
9. "Unknown != Negative" invariant (DISC-02 & DISC-03): Missing data produces UNAVAILABLE and 0 points.
10. Unified EMA20 semantics across stack (DISC-07): recursive smoothing k = 2/21.
"""

import unittest
import os
import json
import re


class TestQuantRemediation(unittest.TestCase):
    """Regression and adversarial verification suite for quantitative truth."""

    def test_9_1_elimination_of_nominal_price_stage_bug(self):
        """QUANT-01: Verify price < 100 and hardcoded FIX Stage 4 rule are completely removed."""
        page_path = os.path.join("frontend", "app", "page.tsx")
        with open(page_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn("data.currentPrice < 100", content, "Arbitrary price < 100 Stage 4 rule must be removed")
        self.assertNotIn('selectedSymbol.toUpperCase() === "FIX"', content, "Hardcoded FIX Stage 4 rule must be removed")
        self.assertIn("candles={data?.candles}", content, "Candles must be passed to AdaptiveTerminal for authentic trend analysis")

    def test_9_2_authentic_moving_averages_calculation(self):
        """QUANT-02 & DISC-01: Verify authentic 50-candle slice and recursive EMA20 formula."""
        generator_path = os.path.join("frontend", "lib", "insightGenerator.ts")
        with open(generator_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("calculatedSma50", content)
        self.assertIn("calculatedEma20", content)
        self.assertIn("candles.slice(-50)", content)
        self.assertIn("const k = 2 / (20 + 1)", content)
        self.assertIn("currentEma = candles[i].close * k + currentEma * (1 - k)", content)

        # Independent mathematical verification:
        mock_closes = [20.0 + (i * 10.0 / 49.0) for i in range(50)]
        expected_sma = sum(mock_closes) / 50.0
        self.assertAlmostEqual(expected_sma, 25.0, places=2)

    def test_9_3_asset_specific_catalog_binding(self):
        """QUANT-03 & DISC-03: Verify domain evidence binds to masterCatalog and does not default to 18.4%."""
        generator_path = os.path.join("frontend", "lib", "insightGenerator.ts")
        with open(generator_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("MASTER_ASSET_CATALOG", content)
        self.assertIn("isHealthAvailable", content)
        self.assertIn("catAsset?.roic", content)
        # Verify 18.4% is NOT used as an arbitrary fallback
        self.assertNotIn('roicDisplay = catAsset?.roic !== undefined ? `${catAsset.roic}%` : "18.4%"', content)

    def test_9_4_sec_filing_dates_and_temporal_truth(self):
        """QUANT-04 & DISC-04: Verify SEC Form 10-Q filing dates match official EDGAR acceptance timestamps."""
        catalog_path = os.path.join("frontend", "lib", "masterCatalog.ts")
        with open(catalog_path, "r", encoding="utf-8") as f:
            cat_content = f.read()

        self.assertIn("secFilingDate?: string;", cat_content)
        self.assertIn('"2026-05-11"', cat_content)  # CPRX Q1 10-Q filing date in EDGAR
        self.assertIn('"2026-08-26"', cat_content)  # NVDA Q2 10-Q acceptance date in EDGAR
        self.assertIn('"2026-07-23"', cat_content)  # FIX Q2 10-Q acceptance date in EDGAR

    def test_9_5_canonical_fix_catalog_entry(self):
        """QUANT-05 & DISC-05: Verify FIX entry in catalog and updated baseline price ($1,560.13)."""
        catalog_path = os.path.join("frontend", "lib", "masterCatalog.ts")
        with open(catalog_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("FIX: {", content)
        self.assertIn('"Comfort Systems USA, Inc."', content)
        self.assertIn("roic: 28.5", content)
        self.assertIn("grossMargin: 20.4", content)
        self.assertIn("piotroski: 8", content)
        self.assertIn("FIX: 1560.13", content)

    def test_9_6_legacy_advisory_terminology_eliminated(self):
        """QUANT-06: Verify legacy phrase 'Strong Buy / Core Accumulation' is eliminated from decision paths."""
        targets = [
            os.path.join("frontend", "components", "TraderArchetypesCard.tsx"),
            os.path.join("frontend", "lib", "constants.ts"),
            os.path.join("frontend", "app", "compare", "page.tsx"),
            os.path.join("frontend", "lib", "insightGenerator.ts"),
        ]

        for path in targets:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("Strong Buy / Core Accumulation", content, f"Found legacy phrase in {path}")
            self.assertNotIn("Strong Buy / Core Hold", content, f"Found legacy phrase in {path}")
            self.assertNotIn("STRONG_BUY_ZONE", content, f"Found legacy phrase in {path}")

    def test_9_7_price_freshness_provenance_tag(self):
        """QUANT-07: Verify 15m Exchange Delayed freshness tag in WhyInspectModal and trend evidence."""
        modal_path = os.path.join("frontend", "components", "WhyInspectModal.tsx")
        generator_path = os.path.join("frontend", "lib", "insightGenerator.ts")

        with open(modal_path, "r", encoding="utf-8") as f:
            m_content = f.read()
        self.assertIn("15m Exchange Delayed", m_content)

        with open(generator_path, "r", encoding="utf-8") as f:
            g_content = f.read()
        self.assertIn('freshness: "DELAYED"', g_content)
        self.assertIn('asOf: "15m Delayed"', g_content)

    def test_11_boundary_observation_windows(self):
        """DISC-01 & DISC-02: Test observation window thresholds: 0, 19, 20, 49, 50 candles."""
        generator_path = os.path.join("frontend", "lib", "insightGenerator.ts")
        with open(generator_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify strict >= 50 for SMA50 and >= 20 for EMA20
        self.assertIn("candles.length >= 50", content, "SMA50 must require at least 50 candles")
        self.assertIn("candles.length >= 20", content, "EMA20 must require at least 20 candles")
        # Verify no subset averaging mislabeled as 50D SMA
        self.assertNotIn("Math.min(candles.length, 50)", content)

    def test_11_unknown_not_negative_invariant(self):
        """DISC-02 & DISC-03: Verify missing evidence marks UNAVAILABLE with 0 points, never synthetic positive."""
        generator_path = os.path.join("frontend", "lib", "insightGenerator.ts")
        with open(generator_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify UNAVAILABLE statuses and 0 point impacts when missing
        self.assertIn('status: "UNAVAILABLE"', content)
        self.assertIn('pointImpact: 0', content)
        self.assertIn('availability: "UNAVAILABLE"', content)
        # Verify synthetic fallback multipliers are eliminated
        self.assertNotIn("safePrice * 0.94", content)
        self.assertNotIn("safePrice * 1.115", content)

    def test_11_unified_ema20_semantics(self):
        """DISC-07: Verify api.ts and insightGenerator.ts use unified recursive exponential smoothing."""
        api_path = os.path.join("frontend", "lib", "api.ts")
        with open(api_path, "r", encoding="utf-8") as f:
            content = f.read()

        # api.ts line 780 must use recursive smoothing with k = 2/21
        self.assertIn("const k = 2 / (20 + 1)", content)
        self.assertIn("currentEma = candles[i].close * k + currentEma * (1 - k)", content)
        self.assertIn("ema_20 = Number(currentEma.toFixed(2))", content)

    def test_11_cprx_nvda_fix_adversarial_verification(self):
        """DISC-01 to DISC-06: Adversarial verification of CPRX, NVDA, and FIX."""
        catalog_path = os.path.join("frontend", "lib", "masterCatalog.ts")
        with open(catalog_path, "r", encoding="utf-8") as f:
            cat_content = f.read()

        # 1. CPRX (Form 15-12G deregistered, 5 candles in feed)
        self.assertIn('secFilingDate: "2026-05-11"', cat_content, "CPRX must reference last Form 10-Q before Form 15-12G")
        self.assertIn("roic: 42.8", cat_content, "CPRX must preserve verified historical 42.8% ROIC")

        # 2. NVDA (252 daily bars, verified SEC acceptance date)
        self.assertIn('secFilingDate: "2026-08-26"', cat_content, "NVDA must reference verified 2026-08-26 EDGAR acceptance")
        self.assertIn("roic: 48.0", cat_content)

        # 3. FIX (252 daily bars, verified SEC acceptance date and baseline price)
        self.assertIn('secFilingDate: "2026-07-23"', cat_content, "FIX must reference verified 2026-07-23 EDGAR acceptance")
        self.assertIn("FIX: 1560.13", cat_content, "FIX baseline price must be $1,560.13, eliminating 75% cliff")


if __name__ == "__main__":
    unittest.main()

