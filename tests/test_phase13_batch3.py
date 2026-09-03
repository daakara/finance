"""Phase 13.1 Product & UX Remediation - Batch 3 Regression Test Suite.

Enforces:
1. FINDING-13-09: Stop-loss formula parity between stock/[ticker]/page.tsx and insightGenerator.ts.
2. FINDING-13-07: Modal focus trapping and focus restoration in WhyInspectModal.tsx.
3. FINDING-13-07: Modal focus trapping and focus restoration in PositionSizerModal.tsx.
4. FINDING-13-02: Screener breadcrumb and goal preservation in card header link.
"""

import unittest
import os


class TestPhase13Batch3Remediations(unittest.TestCase):
    """Test suite for Phase 13.1 Batch 3 (P2 Polish & Accessibility)."""

    def setUp(self):
        self.stock_page_path = os.path.join("frontend", "app", "stock", "[ticker]", "page.tsx")
        self.why_modal_path = os.path.join("frontend", "components", "WhyInspectModal.tsx")
        self.sizer_modal_path = os.path.join("frontend", "components", "PositionSizerModal.tsx")
        self.screener_path = os.path.join("frontend", "app", "screener", "page.tsx")

        with open(self.stock_page_path, "r", encoding="utf-8") as f:
            self.stock_page_content = f.read()
        with open(self.why_modal_path, "r", encoding="utf-8") as f:
            self.why_modal_content = f.read()
        with open(self.sizer_modal_path, "r", encoding="utf-8") as f:
            self.sizer_modal_content = f.read()
        with open(self.screener_path, "r", encoding="utf-8") as f:
            self.screener_content = f.read()

    def test_p2_1_stop_loss_formula_parity(self):
        """FINDING-13-09: Stock landing page stop loss must use canonical 0.93 multiplier."""
        self.assertNotIn("spotPrice - 1.25 * atr14", self.stock_page_content)
        self.assertIn("spotPrice * 0.93", self.stock_page_content)

    def test_p2_2_why_modal_focus_management(self):
        """FINDING-13-07: WhyInspectModal must trap focus and restore focus on close."""
        self.assertIn("previouslyFocusedElement", self.why_modal_content)
        self.assertIn('e.key === "Tab"', self.why_modal_content)

    def test_p2_3_sizer_modal_focus_management(self):
        """FINDING-13-07: PositionSizerModal must trap focus and restore focus on close."""
        self.assertIn("previouslyFocusedElement", self.sizer_modal_content)
        self.assertIn('e.key === "Tab"', self.sizer_modal_content)

    def test_p2_4_screener_card_header_context_preservation(self):
        """FINDING-13-02: Screener card header link must preserve fromGoal and fromCount."""
        self.assertIn("fromGoal=${selectedFilter}", self.screener_content)
        self.assertIn("fromCount=${displayGems.length}", self.screener_content)


if __name__ == "__main__":
    unittest.main()
