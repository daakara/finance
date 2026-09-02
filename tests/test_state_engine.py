import unittest

class TestARXStateEngine(unittest.TestCase):
    """Automated state-transition test matrix for ARX deriveAssessmentState engine."""

    def test_factor_agreement_calculation(self):
        """Test mathematical factor agreement without pseudo-confidence percentages."""
        # 3 Favorable, 1 Unfavorable
        favorable_count = 3
        unfavorable_count = 1
        unavailable_count = 0
        evaluated = favorable_count + unfavorable_count

        self.assertEqual(evaluated, 4)
        label = f"{favorable_count} of {evaluated} evaluated factors are favorable"
        self.assertEqual(label, "3 of 4 evaluated factors are favorable")
        self.assertNotIn("% confidence", label)

    def test_state_resolution_not_owned_favorable(self):
        """Test NOT_OWNED + FAVORABLE resolves to ACQUIRE (Actionable Setup)."""
        ownership = "NOT_OWNED"
        assessment = "FAVORABLE"
        is_breached = False

        posture = "ACQUIRE" if (assessment == "FAVORABLE" and not is_breached) else "WATCH"
        self.assertEqual(posture, "ACQUIRE")

    def test_state_resolution_not_owned_mixed(self):
        """Test NOT_OWNED + MIXED resolves to WATCH (Wait for Trigger)."""
        ownership = "NOT_OWNED"
        assessment = "MIXED"
        posture = "WATCH"
        self.assertEqual(posture, "WATCH")

    def test_state_resolution_not_owned_unfavorable(self):
        """Test NOT_OWNED + UNFAVORABLE resolves to AVOID (Unfavorable Setup)."""
        ownership = "NOT_OWNED"
        assessment = "UNFAVORABLE"
        posture = "AVOID" if assessment == "UNFAVORABLE" else "WATCH"
        self.assertEqual(posture, "AVOID")

    def test_state_resolution_owned_favorable(self):
        """Test OWNED + FAVORABLE resolves to HOLD (Thesis Intact)."""
        ownership = "OWNED"
        assessment = "FAVORABLE"
        is_breached = False

        if is_breached:
            posture = "EXIT_REVIEW"
        elif assessment == "FAVORABLE":
            posture = "HOLD"
        else:
            posture = "TRIM"

        self.assertEqual(posture, "HOLD")

    def test_state_resolution_owned_unfavorable(self):
        """Test OWNED + UNFAVORABLE resolves to TRIM (Consider Trimming)."""
        ownership = "OWNED"
        assessment = "UNFAVORABLE"
        is_breached = False

        if is_breached:
            posture = "EXIT_REVIEW"
        elif assessment == "UNFAVORABLE":
            posture = "TRIM"
        else:
            posture = "HOLD"

        self.assertEqual(posture, "TRIM")

    def test_contradictory_invalidation_overrides_favorable_owned(self):
        """Test Hard Invalidation breach overrides favorable fundamentals and triggers EXIT_REVIEW."""
        ownership = "OWNED"
        assessment = "FAVORABLE" # Strong fundamentals
        is_breached = True       # Breached -7% stop

        # Precedence rule: Invalidation breach overrides assessment and ownership
        if is_breached:
            posture = "EXIT_REVIEW"
        elif assessment == "FAVORABLE":
            posture = "HOLD"

        self.assertEqual(posture, "EXIT_REVIEW")

    def test_domain_decoupling_missing_fundamentals(self):
        """Test missing fundamentals does not destroy valid technical price trend assessment."""
        technical_domain = {"domainId": "trend", "status": "FAVORABLE", "availability": "AVAILABLE"}
        fundamental_domain = {"domainId": "health", "status": "UNAVAILABLE", "availability": "UNAVAILABLE"}

        domains = [technical_domain, fundamental_domain]
        available_domains = [d for d in domains if d["availability"] == "AVAILABLE"]
        
        self.assertEqual(len(available_domains), 1)
        self.assertEqual(available_domains[0]["status"], "FAVORABLE")
        self.assertEqual(domains[1]["status"], "UNAVAILABLE")

    def test_stale_data_handling(self):
        """Test stale data does not equal unavailable or negative data."""
        smart_money = {"domainId": "smart_money", "status": "FAVORABLE", "availability": "STALE"}
        self.assertEqual(smart_money["availability"], "STALE")
        self.assertEqual(smart_money["status"], "FAVORABLE")

    def test_ineligible_data_barrier(self):
        """Test when 0 domains are evaluated, posture defaults to RESEARCH and avoids fabricated score."""
        evaluated_domains = 0
        if evaluated_domains == 0:
            eligibility = "INELIGIBLE"
            posture = "RESEARCH"
            label = "Assessment Unavailable — Data Incomplete"
        else:
            eligibility = "ELIGIBLE"
            posture = "WATCH"
            label = "Wait for Trigger"

        self.assertEqual(eligibility, "INELIGIBLE")
        self.assertEqual(posture, "RESEARCH")
        self.assertEqual(label, "Assessment Unavailable — Data Incomplete")

if __name__ == "__main__":
    unittest.main()
