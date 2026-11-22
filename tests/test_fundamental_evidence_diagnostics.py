"""Offline diagnostics: source metadata never represents filing authentication."""
import unittest
from datetime import datetime, timedelta, timezone

from analyst_dashboard.provenance.evidence_contract import evaluate_factor_provenance
from analyst_dashboard.analyzers.confluence_engine import ConfluenceEngine


class FundamentalEvidenceDiagnosticsTests(unittest.TestCase):
    def test_runtime_scores_remain_unverified_and_freshness_unknown(self):
        row = dict(source="LIVE_RUNTIME", quality_score=99, piotroski_f=9,
                   fetched_at=datetime.now(timezone.utc).isoformat())
        result = evaluate_factor_provenance(row, "NVDA")
        self.assertEqual(result["evidenceStatus"], "UNVERIFIED")
        self.assertFalse(result["isVerified"])
        self.assertEqual(result["freshnessStatus"], "UNKNOWN")
        self.assertIn("Financial evidence not integrated", result["disqualificationReason"])
        self.assertNotIn("regulator", result["disqualificationReason"])
        self.assertIn("filing_date", result["missingEvidenceFields"])
        output = ConfluenceEngine.calculate_confluence("NVDA", fundamental_data=row,
                                                     fundamental_provenance=result)
        pillar = next(p for p in output["pillars"] if p["pillar"] == "FUNDAMENTAL_SOLVENCY")
        self.assertEqual(pillar["status"], "unverified")
        self.assertEqual(pillar["score"], 0)

    def test_allowlisted_metadata_is_not_document_authentication(self):
        result = evaluate_factor_provenance(dict(source="sec_edgar", filing_date="2026-11-20"), "NVDA")
        self.assertEqual(result["verificationBasis"], "SOURCE_AND_DATE_METADATA")
        self.assertEqual(result["filingAuthenticationStatus"], "NOT_CHECKED")

    def test_missing_row_is_not_an_adverse_financial_finding(self):
        result = evaluate_factor_provenance(None, "NVDA")
        self.assertEqual(result["evidenceStatus"], "UNAVAILABLE")
        self.assertEqual(result["freshnessStatus"], "UNKNOWN")

    def test_bad_and_future_dates_are_explicit(self):
        for date, expected in [("bad", "INVALID_DATE"),
                               ((datetime.now(timezone.utc) + timedelta(days=2)).date().isoformat(), "INVALID_FUTURE_DATE")]:
            with self.subTest(date=date):
                result = evaluate_factor_provenance(dict(source="sec_edgar", filing_date=date), "NVDA")
                self.assertFalse(result["isVerified"])
                self.assertEqual(result["freshnessStatus"], expected)

    def test_stale_and_exempt_states_remain_distinct(self):
        stale = evaluate_factor_provenance(dict(source="sec_edgar", filing_date="2020-01-01"), "NVDA")
        self.assertEqual(stale["freshnessStatus"], "STALE")
        exempt = evaluate_factor_provenance(None, "SPY", asset_class="ETF")
        self.assertEqual(exempt["freshnessStatus"], "NOT_APPLICABLE")
        self.assertEqual(exempt["filingAuthenticationStatus"], "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main()
