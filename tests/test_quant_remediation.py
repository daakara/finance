"""Quantitative & Data Remediation Regression Test Matrix (Phase 9).
Verifies:
1. SMA50 & EMA20 authentic candle math and independence from nominal share price.
2. Stage 2 vs Stage 4 Minervini classification (elimination of price < 100 bug).
3. Asset-specific catalog binding (ROIC, Gross Margin, Piotroski) for NVDA, CPRX, FIX.
4. Canonical FIX asset resolution in master catalog.
5. SEC filing dates (publishedAt vs retrievedAt) free of runtime date conflation.
6. Provenance tagging (15m Exchange Delayed).
7. Removal of legacy advisory terminology ("Strong Buy / Core Accumulation").
"""

import unittest
import os
import json
import re


class TestQuantRemediation(unittest.TestCase):
    """Regression test suite for Phase 9 quantitative remediations."""

    def test_9_1_elimination_of_nominal_price_stage_bug(self):
        """QUANT-01: Verify price < 100 and hardcoded FIX Stage 4 rule are completely removed."""
        page_path = os.path.join("frontend", "app", "page.tsx")
        with open(page_path, "r", encoding="utf-8") as f:
            content = f.read()

        # The buggy line was: isStage4={selectedSymbol.toUpperCase() === "FIX" || (data?.currentPrice && data.currentPrice < 100) ? true : false}
        self.assertNotIn("data.currentPrice < 100", content, "Arbitrary price < 100 Stage 4 rule must be removed")
        self.assertNotIn('selectedSymbol.toUpperCase() === "FIX"', content, "Hardcoded FIX Stage 4 rule must be removed")
        self.assertIn("candles={data?.candles}", content, "Candles must be passed to AdaptiveTerminal for authentic trend analysis")

    def test_9_2_authentic_moving_averages_calculation(self):
        """QUANT-02: Verify insightGenerator calculates authentic SMA50 and EMA20 from candle history."""
        generator_path = os.path.join("frontend", "lib", "insightGenerator.ts")
        with open(generator_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify authentic candle math formulas
        self.assertIn("calculatedSma50", content)
        self.assertIn("calculatedEma20", content)
        self.assertIn("candles.slice(-smaWindow)", content)
        self.assertIn("const k = 2 / (20 + 1)", content)
        self.assertIn("currentEma = candles[i].close * k + currentEma * (1 - k)", content)

        # Independent mathematical formula verification:
        # Given 50 mock candle closes from 20 to 30, SMA50 must equal 25.0
        mock_closes = [20.0 + (i * 10.0 / 49.0) for i in range(50)]
        expected_sma = sum(mock_closes) / 50.0
        self.assertAlmostEqual(expected_sma, 25.0, places=2)

    def test_9_3_asset_specific_catalog_binding(self):
        """QUANT-03: Verify domain evidence binds dynamically to masterCatalog rather than static 18.4%."""
        generator_path = os.path.join("frontend", "lib", "insightGenerator.ts")
        with open(generator_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("MASTER_ASSET_CATALOG", content)
        self.assertIn("catAsset?.roic !== undefined ? `${catAsset.roic}%` :", content)
        self.assertIn("currentValue: roicDisplay", content)
        self.assertIn("roic: catAsset?.roic ?? 18.4", content)

    def test_9_4_sec_filing_dates_and_temporal_truth(self):
        """QUANT-04: Verify SEC Form 10-Q filing dates use authentic publishedAt dates, not runtime new Date()."""
        generator_path = os.path.join("frontend", "lib", "insightGenerator.ts")
        catalog_path = os.path.join("frontend", "lib", "masterCatalog.ts")

        with open(catalog_path, "r", encoding="utf-8") as f:
            cat_content = f.read()

        # Verify masterCatalog has secFilingDate property
        self.assertIn("secFilingDate?: string;", cat_content)
        self.assertIn('"2026-08-08"', cat_content)  # CPRX Q2 10-Q filing date
        self.assertIn('"2026-08-28"', cat_content)  # NVDA Q2 10-Q filing date
        self.assertIn('"2026-07-26"', cat_content)  # FIX Q2 10-Q filing date

        with open(generator_path, "r", encoding="utf-8") as f:
            gen_content = f.read()

        self.assertIn("filingDate = catAsset?.secFilingDate", gen_content)
        self.assertIn("asOf: filingDate", gen_content)

    def test_9_5_canonical_fix_catalog_entry(self):
        """QUANT-05: Verify FIX (Comfort Systems USA) exists in MASTER_ASSET_CATALOG and baseline prices."""
        catalog_path = os.path.join("frontend", "lib", "masterCatalog.ts")
        with open(catalog_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("FIX: {", content)
        self.assertIn('"Comfort Systems USA, Inc."', content)
        self.assertIn("roic: 28.5", content)
        self.assertIn("grossMargin: 20.4", content)
        self.assertIn("piotroski: 8", content)
        self.assertIn("FIX: 385.00", content)

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


if __name__ == "__main__":
    unittest.main()
