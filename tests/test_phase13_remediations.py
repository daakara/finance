"""Phase 13.1 Product & UX Remediation Regression Test Suite.

Enforces:
1. Position Sizer Safety: entry <= stopLoss strictly returns 0 shares and marks setup invalid.
2. Posture Evidence-Gating: Missing core domains (trend/health) force posture="RESEARCH".
   ACQUIRE and WATCH are strictly prohibited when core evidence is UNAVAILABLE.
3. Guided View Lens Parity: whyPills derive from domain status, eliminating hardcoded "Healthy".
4. Financial Risk Disclaimers: Non-fiduciary safe harbor and risk copy present globally.
5. Comparison Data Integrity: Uncataloged tickers render "N/A", eliminating synthetic fundamentals.
"""

import unittest
import os
import re


class TestPhase13Batch1Remediations(unittest.TestCase):
    """Test suite for Phase 13.1 Batch 1 (P0 Safety & Integrity)."""

    def setUp(self):
        self.sizer_path = os.path.join("frontend", "components", "PositionSizerModal.tsx")
        self.engine_path = os.path.join("frontend", "lib", "assessmentEngine.ts")
        self.insight_path = os.path.join("frontend", "lib", "insightGenerator.ts")
        self.compare_path = os.path.join("frontend", "app", "compare", "page.tsx")
        self.layout_path = os.path.join("frontend", "app", "layout.tsx")
        self.disclaimer_path = os.path.join("frontend", "components", "FinancialDisclaimer.tsx")

        with open(self.sizer_path, "r", encoding="utf-8") as f:
            self.sizer_content = f.read()
        with open(self.engine_path, "r", encoding="utf-8") as f:
            self.engine_content = f.read()
        with open(self.insight_path, "r", encoding="utf-8") as f:
            self.insight_content = f.read()
        with open(self.compare_path, "r", encoding="utf-8") as f:
            self.compare_content = f.read()
        with open(self.layout_path, "r", encoding="utf-8") as f:
            self.layout_content = f.read()

    def test_p0_1_position_sizer_invalidated_setup_zero_shares(self):
        """FINDING-13-14: If entryPrice <= stopLoss, sizing is disabled and shares must be 0."""
        # Check that safeEntry <= safeStop check exists
        self.assertIn("safeEntry <= safeStop", self.sizer_content)
        # Check that invalid setup disables calculation and forces shares = 0
        self.assertIn("isSetupInvalid", self.sizer_content)
        # The old clamping bug Math.max(0.01, safeEntry - safeStop) must not execute for invalid setup
        self.assertIn("shares = isSetupInvalid ? 0", self.sizer_content)
        self.assertIn("Setup Invalidated", self.sizer_content)

    def test_p0_2_posture_evidence_gating_research_hierarchy(self):
        """FINDING-13-03: Missing core evidence must force RESEARCH, preventing false ACQUIRE/WATCH."""
        # Invalidation takes first precedence
        self.assertIn("isInvalidationBreached", self.engine_content)
        # Core domain missing check
        self.assertIn('isTrendAvailable', self.engine_content)
        self.assertIn('!isTrendAvailable', self.engine_content)
        self.assertIn('posture = "RESEARCH"', self.engine_content)
        # Ensure ACQUIRE requires ELIGIBLE status
        self.assertIn('overallEligibility === "ELIGIBLE"', self.engine_content)
        # Fabricated 50D SMA target string must be guarded or removed from default WATCH
        self.assertNotIn("Price remains below 50-day average ($${reclaimTarget.toFixed(2)})", self.engine_content)

    def test_p0_3_guided_view_lens_parity_dynamic_why_pills(self):
        """FINDING-13-04: whyPills must dynamically derive from domain status, eliminating hardcoded Healthy."""
        # Must check domain status / availability
        self.assertIn("!isHealthAvailable", self.insight_content)
        self.assertIn("!isTrendAvailable", self.insight_content)
        # Old hardcoded Health pill assertion must be eliminated
        self.assertNotIn(
            'status: "Healthy",\n          description: "Stable financials and strong profitability across core metrics.",\n          sentiment: "positive",',
            self.insight_content
        )

    def test_p0_4_financial_disclaimer_presence(self):
        """FINDING-13-08: Global financial disclaimer must be present in layout and component tree."""
        self.assertTrue(os.path.exists(self.disclaimer_path), "FinancialDisclaimer.tsx must exist")
        self.assertIn("FinancialDisclaimer", self.layout_content)
        with open(self.disclaimer_path, "r", encoding="utf-8") as f:
            disc_content = f.read()
        self.assertIn("not a registered investment adviser", disc_content.lower())
        self.assertIn("no personalized financial advice", disc_content.lower())
        self.assertIn("risk of loss", disc_content.lower())

    def test_p0_5_compare_no_synthetic_fundamentals_for_uncataloged(self):
        """FINDING-13-11: Uncataloged tickers in /compare must render N/A, not synthetic estimates."""
        # Ensure default synthetic values are replaced with N/A / uncataloged handling
        self.assertNotIn('registered?.roic ?? 24.0', self.compare_content)
        self.assertNotIn('registered?.fwdPe ?? 25.0', self.compare_content)
        self.assertNotIn('registered?.peg ?? 1.15', self.compare_content)
        self.assertNotIn('registered?.marketCap || `$${(price * 0.45).toFixed(1)}B Est`', self.compare_content)


if __name__ == "__main__":
    unittest.main()
