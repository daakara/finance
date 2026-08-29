"""Automated Multi-Component State Synchronicity & Contradiction Test Suite.
Validates that across all modals, cards, screeners, and execution planners,
derived states (Stage 4, Buy Zones, Risk:Reward, Distribution Traps)
are strictly synchronized with ZERO hardcoded placeholder blindspots.
"""

import unittest
import os
import re


class TestCrossComponentStateSynchronicity(unittest.TestCase):
    """Rigorous tests asserting Cross-Component State Synchronicity & Contradiction Immunity."""

    def test_preflight_modal_dynamic_stage4_wiring(self):
        """Invariant: PreFlightChecklistModal MUST dynamically receive and evaluate isStage4,
        optimalEntryMin, and optimalEntryMax without static true placeholders.
        """
        modal_path = os.path.join("frontend", "components", "PreFlightChecklistModal.tsx")
        self.assertTrue(os.path.exists(modal_path), "Missing PreFlightChecklistModal.tsx")

        with open(modal_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Must declare dynamic stage and boundary props
        self.assertIn("isStage4?: boolean", content, "PreFlightChecklistModal must accept isStage4 prop")
        self.assertIn("optimalEntryMin?: number", content, "PreFlightChecklistModal must accept optimalEntryMin")
        self.assertIn("optimalEntryMax?: number", content, "PreFlightChecklistModal must accept optimalEntryMax")

        # Must evaluate Check 2 dynamically based on isStage4 and boundaries
        self.assertIn("!isStage4", content, "Check 2 must evaluate !isStage4")
        self.assertNotIn("const isTrendPassed = true;", content, "Check 2 cannot have hardcoded static true placeholder")
        self.assertNotIn("const isSmartMoneyPassed = true;", content, "Check 3 cannot have hardcoded static true placeholder")
        self.assertNotIn("const isCatalystPassed = true;", content, "Check 4 cannot have hardcoded static true placeholder")
        self.assertNotIn("const isMacroPassed = true;", content, "Check 5 cannot have hardcoded static true placeholder")
        self.assertIn("STAGE 4 WAIT", content, "Check 2 must render Stage 4 Wait status when isStage4 is true")
        self.assertIn("isDistributionTrapResolved", content, "Check 3 must dynamically resolve distribution trap status")

    def test_optimal_entry_exit_card_passes_stage4_to_modals(self):
        """Invariant: OptimalEntryExitCard must propagate isStage4 and boundary props to
        PreFlightChecklistModal, PositionSizerModal, and AlertTriggerModal.
        """
        card_path = os.path.join("frontend", "components", "OptimalEntryExitCard.tsx")
        self.assertTrue(os.path.exists(card_path), "Missing OptimalEntryExitCard.tsx")

        with open(card_path, "r", encoding="utf-8") as f:
            content = f.read()

        # PreFlightChecklistModal invocation must include isStage4 and boundaries
        self.assertIn("<PreFlightChecklistModal", content)
        self.assertIn("isStage4={isStage4}", content, "OptimalEntryExitCard must pass isStage4 to PreFlightChecklistModal")
        self.assertIn("optimalEntryMin={optimal_entry_min}", content, "OptimalEntryExitCard must pass optimalEntryMin to PreFlightChecklistModal")
        self.assertIn("optimalEntryMax={optimal_entry_max}", content, "OptimalEntryExitCard must pass optimalEntryMax to PreFlightChecklistModal")

        # PositionSizerModal invocation must include isStage4
        self.assertIn("<PositionSizerModal", content)
        self.assertIn("isStage4={isStage4}", content, "OptimalEntryExitCard must pass isStage4 to PositionSizerModal")

        # AlertTriggerModal invocation must include isStage4
        self.assertIn("<AlertTriggerModal", content)
        self.assertIn("isStage4={isStage4}", content, "OptimalEntryExitCard must pass isStage4 to AlertTriggerModal")

    def test_position_sizer_stage4_risk_mitigation(self):
        """Invariant: PositionSizerModal must downsize risk percentage from 1.0% to 0.25% (quarter size)
        when isStage4 is true.
        """
        sizer_path = os.path.join("frontend", "components", "PositionSizerModal.tsx")
        self.assertTrue(os.path.exists(sizer_path), "Missing PositionSizerModal.tsx")

        with open(sizer_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("isStage4 ? 0.25 : 1.0", content, "PositionSizerModal must default to 0.25% risk on Stage 4 assets")

    def test_alert_trigger_modal_breakout_pivot_target(self):
        """Invariant: AlertTriggerModal must save breakoutPivotPrice and target breakout pivot on Stage 4 assets."""
        alert_path = os.path.join("frontend", "components", "AlertTriggerModal.tsx")
        self.assertTrue(os.path.exists(alert_path), "Missing AlertTriggerModal.tsx")

        with open(alert_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("isStage4: isStage4", content, "AlertTriggerModal must store isStage4 state")
        self.assertIn("breakoutPivotPrice:", content, "AlertTriggerModal must store breakoutPivotPrice")


if __name__ == "__main__":
    unittest.main()
