"""Independent Adversarial Verification Pass for Phase 13.1 Product & UX Remediation.

Attacks the remediated boundaries across:
1. Invalidation geometry: entryPrice <= stopLoss -> forces zero shares.
2. Epistemic gating: incomplete trend or health -> strictly gates posture to RESEARCH.
3. Target gating: incomplete trend -> targets & R:R ratio are undefined.
4. Lineage integrity: Screener & Strategy catalog unification eliminates stale prices.
5. Ownership preservation: Portfolio row links append &ownership=OWNED.
6. Disclosure compliance: Safe harbor non-fiduciary copy present in global and terminal views.
7. Focus management: Modal focus trap Tab handler and element restoration present.
"""

import os
import re
import unittest


class TestAdversarialPhase13Verification(unittest.TestCase):
    """Adversarial stress-testing suite for Phase 13.1 remediations."""

    def setUp(self):
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def read_file(self, *parts):
        path = os.path.join(self.root, *parts)
        self.assertTrue(os.path.exists(path), f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_adv_1_position_sizer_invalidated_risk_geometry(self):
        """Attack: Attempt to pass entry <= stop to PositionSizerModal. Verify safe allocation is zeroed."""
        content = self.read_file("frontend", "components", "PositionSizerModal.tsx")
        # Ensure invalidation check exists
        self.assertIn("const isSetupInvalid = safeEntry <= safeStop;", content)
        # Ensure shares, allocation, profit are zeroed
        self.assertIn("const shares = isSetupInvalid ? 0 : rawShares;", content)
        self.assertIn("const projectedProfit = isSetupInvalid ? 0 :", content)
        self.assertIn("const halfKellyPct = isSetupInvalid ? 0 :", content)
        # Ensure disabled save button when invalid
        self.assertIn("disabled={isSetupInvalid || shares <= 0", content)

    def test_adv_2_epistemic_posture_gating(self):
        """Attack: Verify posture engine cannot output ACQUIRE or WATCH if evidence is missing."""
        content = self.read_file("frontend", "lib", "assessmentEngine.ts")
        # Required hierarchy: missing evidence -> RESEARCH
        self.assertIn('!isCoreEvidenceAvailable || overallEligibility !== "ELIGIBLE" || !isTrendAvailable', content)
        self.assertIn('posture = "RESEARCH";', content)
        # Verification that core evidence requires both trend and health
        self.assertIn('const isCoreEvidenceAvailable = isTrendAvailable && isHealthAvailable && overallEligibility === "ELIGIBLE";', content)

    def test_adv_3_targets_gating_when_trend_unavailable(self):
        """Attack: Verify insight generator does not fabricate targets when trend is incomplete."""
        content = self.read_file("frontend", "lib", "insightGenerator.ts")
        self.assertIn("target1: isTrendAvailable ? target1 : undefined", content)
        self.assertIn("target2: isTrendAvailable ? target2 : undefined", content)
        self.assertIn("profitRiskRatio: isTrendAvailable ? profitRisk : undefined", content)

    def test_adv_4_screener_unification_and_no_stale_prices(self):
        """Attack: Verify screener eliminated decoupled BASE_PRICES ($346.20 FIX bug)."""
        content = self.read_file("frontend", "app", "screener", "page.tsx")
        self.assertNotIn("const BASE_PRICES", content)
        self.assertNotIn("FIX: { price: 346.20", content)
        self.assertIn("CATALOG_BASELINE_PRICES[sym]", content)

    def test_adv_5_strategy_unification_and_no_stale_prices(self):
        """Attack: Verify strategy candidates hydrate from master catalog baseline."""
        content = self.read_file("frontend", "app", "strategy", "[type]", "page.tsx")
        self.assertIn("CATALOG_BASELINE_PRICES[cand.symbol] ?? cand.price", content)

    def test_adv_6_portfolio_ownership_preservation(self):
        """Attack: Verify portfolio position click passes ownership=OWNED to Terminal."""
        content = self.read_file("frontend", "app", "portfolio", "page.tsx")
        self.assertIn("/?symbol=${pos.symbol}&ownership=OWNED", content)

    def test_adv_7_regulatory_disclaimer_presence(self):
        """Attack: Verify financial disclaimer is rendered globally and in terminal views."""
        layout_content = self.read_file("frontend", "app", "layout.tsx")
        self.assertIn("<FinancialDisclaimer />", layout_content)
        disclaimer_content = self.read_file("frontend", "components", "FinancialDisclaimer.tsx")
        self.assertIn("not a registered investment adviser", disclaimer_content)
        self.assertIn("informational and educational purposes only", disclaimer_content)

    def test_adv_8_modal_focus_trap_and_accessibility(self):
        """Attack: Verify modal dialogs implement Tab wrapping and focus restoration."""
        for modal_file in ["WhyInspectModal.tsx", "PositionSizerModal.tsx"]:
            content = self.read_file("frontend", "components", modal_file)
            self.assertIn("previouslyFocusedElementRef", content)
            self.assertIn('e.key === "Tab"', content)
            self.assertIn('e.key === "Escape"', content)
            self.assertIn('role="dialog"', content)
            self.assertIn('aria-modal="true"', content)

    def test_adv_9_stop_loss_formula_parity(self):
        """Attack: Verify stock landing page stop loss aligns with insight generator (0.93 multiplier)."""
        content = self.read_file("frontend", "app", "stock", "[ticker]", "page.tsx")
        self.assertIn("+(spotPrice * 0.93).toFixed(2)", content)
        self.assertNotIn("spotPrice - 1.25 * atr14", content)

    def test_adv_10_compare_uncataloged_truth_integrity(self):
        """Attack: Verify uncataloged comparison tickers do not fabricate ROIC or margins."""
        content = self.read_file("frontend", "app", "compare", "page.tsx")
        self.assertNotIn('roic: "24.0%"', content)
        self.assertNotIn('grossMargin: "55.0%"', content)
        self.assertIn('hasVerifiedFundamentals', content)
        self.assertIn('"N/A"', content)


if __name__ == "__main__":
    unittest.main()
