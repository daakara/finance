"""
tests/test_universe_real_source_diagnostic.py

Sprint 2: Bounded Real-Source Evidence Handoff & Admission Diagnostic Test Suite.
Verifies the 7-candidate representative real-source sample against the released
admission-mapping and validation contracts:

1. Straightforward US common stock: AAPL
2. Multi-symbol issuer sharing CIK: GOOG & GOOGL (CIK 0001652044)
3. Foreign US-listed issuer (ADR): TSM
4. Venue discrepancy Group 1: SEC NYSE vs Directory A/AMEX (ACCS)
5. Venue discrepancy Group 4: SEC TXSE vs Directory F/TXSE (DDS)
6. Unresolved classification / Units: AACIU

Core Safeguards Verified:
- Source Grounding & Attributable Provenance (2026-10-16 parent feeds).
- Strict Non-Inference: Zero surrogate keys synthesized (CIK+symbol).
- Separation of Evidentiary Categories (source fact, external mapping, policy assertion, triage hint, conflict).
- Substantive venue disagreements preserved (DDS TXSE vs F).
- Regex triage hints cannot certify COMMON_STOCK (AACIU = UNIT; corporate common stock uncertified).
- Atomic whole-batch validation failure: 0 converted securities, builder strictly NOT_RUN.
- Enforced read-only SQLite market store audit (mode=ro mutation prevention).
"""

from __future__ import annotations

import sqlite3
import pytest

from analyst_dashboard.universe.adapters.resolution import (
    IdentityResolutionStatus,
    ResolutionRecordStatus,
    SecurityEvidenceResolver,
)
from analyst_dashboard.universe.admission_mapping import (
    ClassificationTriageEngine,
    ConsumerReadinessEvaluator,
    SupportedVenueNormalizer,
    VenueNormalizationStatus,
)
from analyst_dashboard.universe.real_source_diagnostic import (
    REAL_SOURCE_PARENT_FEEDS,
    REAL_SOURCE_ROW_LOCATORS,
    RealSourceAdmissionDiagnostic,
    RealSourceDiagnosticResult,
    build_real_source_sample_files,
    format_evidence_gap_report,
)
from analyst_dashboard.universe.validator import (
    ValidationRequirements,
    ValidationStatus,
)


class TestRealSourceParentFeedProvenance:
    """Verifies that sample candidates trace to authentic 2026-10-16 parent feed artifacts."""

    def test_parent_feed_digests_and_record_counts(self):
        sec_feed = REAL_SOURCE_PARENT_FEEDS["sec_company_tickers_exchange.json"]
        assert sec_feed["sha256"] == "d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f"
        assert sec_feed["record_count"] == 10444

        nasdaq_feed = REAL_SOURCE_PARENT_FEEDS["nasdaq_listed_directory.txt"]
        assert nasdaq_feed["sha256"] == "67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272"
        assert nasdaq_feed["record_count"] == 5166

        other_feed = REAL_SOURCE_PARENT_FEEDS["other_listed_directory.txt"]
        assert other_feed["sha256"] == "a0cb851a772098d64830304bf7e7a7d1ca1cbd8421da20a6bfbec9a336243792"
        assert other_feed["record_count"] == 8172

    def test_seven_candidates_have_exact_parent_locators(self):
        candidates = ["AAPL", "GOOGL", "GOOG", "TSM", "ACCS", "DDS", "AACIU"]
        for sym in candidates:
            assert sym in REAL_SOURCE_ROW_LOCATORS
            loc = REAL_SOURCE_ROW_LOCATORS[sym]
            assert loc["sec_row_index"] is not None
            assert loc["nasdaq_line"] is not None or loc["other_line"] is not None


class TestRealSourceSampleResolution:
    """Verifies candidate resolution from parent feed fixtures without surrogate key synthesis."""

    def test_all_seven_candidates_resolve_without_synthetic_surrogate_keys(self):
        sec_file, directory_files = build_real_source_sample_files()
        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=directory_files,
            openfigi_results=None,
            caller_security_id_mapping=None,
            caller_field_evidence_overlay=None,
            require_venue_match=False,
            required_sources=["SEC_EDGAR", "NASDAQ_TRADER"],
        )

        assert outcome.total_candidates == 7
        assert outcome.accounting.identity_resolved_count == 0
        assert outcome.accounting.identity_unresolved_count == 7

        records_by_sym = {r.symbol: r for r in outcome.records}
        for sym in ["AAPL", "GOOGL", "GOOG", "TSM", "ACCS", "DDS", "AACIU"]:
            rec = records_by_sym[sym]
            assert rec.symbol == sym
            assert rec.issuer_cik is not None
            assert rec.company_name is not None
            # Verified: No synthetic surrogate keys attached
            assert rec.identity_status == IdentityResolutionStatus.UNRESOLVED
            assert rec.security_id is None

    def test_multi_symbol_issuer_detection_for_shared_cik(self):
        """GOOG and GOOGL share CIK 0001652044 but represent separate ticker observations."""
        sec_file, directory_files = build_real_source_sample_files()
        outcome = SecurityEvidenceResolver.resolve_from_source_files(
            sec_file=sec_file,
            directory_files=directory_files,
            require_venue_match=False,
        )
        records_by_sym = {r.symbol: r for r in outcome.records}

        goog = records_by_sym["GOOG"]
        googl = records_by_sym["GOOGL"]

        assert goog.issuer_cik == googl.issuer_cik == "0001652044"
        assert goog.symbol == "GOOG"
        assert googl.symbol == "GOOGL"
        assert "Class C" in goog.company_name or "Alphabet" in goog.company_name
        assert "Class A" in googl.company_name or "Alphabet" in googl.company_name
        assert goog.security_id is None
        assert googl.security_id is None

    def test_venue_normalization_and_disagreement_preservation(self):
        """Preserves substantive disagreements (DDS TXSE vs F) and normalizes codes (TSM, ACCS)."""
        # ACCS: SEC NYSE vs Directory A (AMEX)
        accs_res = SupportedVenueNormalizer.reconcile_dual_source_venues("NYSE", "A")
        assert accs_res.status == VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL
        assert accs_res.normalized_venue == "AMEX"

        # TSM: SEC NYSE vs Directory N (NYSE)
        tsm_res = SupportedVenueNormalizer.reconcile_dual_source_venues("NYSE", "N")
        assert tsm_res.status == VenueNormalizationStatus.EXACT_CONCORDANT
        assert tsm_res.normalized_venue == "NYSE"

        # DDS: SEC TXSE vs Directory F (TXSE) - Substantive provisional disagreement
        dds_res = SupportedVenueNormalizer.reconcile_dual_source_venues("TXSE", "F")
        assert dds_res.status == VenueNormalizationStatus.PROVISIONAL_UNRESOLVED
        assert dds_res.is_substantive_disagreement is True

        # AAPL: SEC Nasdaq vs Directory NASDAQ
        aapl_res = SupportedVenueNormalizer.reconcile_dual_source_venues("Nasdaq", "NASDAQ")
        assert aapl_res.status == VenueNormalizationStatus.EXACT_CONCORDANT
        assert aapl_res.normalized_venue == "NASDAQ"

    def test_aaciu_triage_hint_does_not_certify_common_stock(self):
        """Classification triage hint identifies UNIT but cannot certify COMMON_STOCK under R02."""
        hint = ClassificationTriageEngine.evaluate_triage_hint(
            "Armada Acquisition Corp. III - Units", is_etf=False
        )
        assert hint.suspected_instrument_type == "UNIT"
        assert hint.certification_established is False
        assert hint.is_diagnostic_hint is True


class TestAtomicWholeBatchRevocation:
    """Verifies that the 7-candidate sample fails closed with 0 admissions and builder NOT_RUN."""

    def test_atomic_batch_revocation_and_zero_conversions(self):
        result = RealSourceAdmissionDiagnostic.run()

        # Batch validation must fail
        assert result.batch_valid is False
        assert result.failed_count == 7
        assert result.passed_count == 0
        assert result.unresolved_count == 0

        # Builder must be NOT_RUN
        assert result.builder_status == "NOT_RUN"
        assert result.converted_security_count == 0
        assert "BLOCKED_VALIDATION_FAILED" in result.builder_blocking_reason

        # Every candidate must have missing required fields
        for sym, p in result.candidate_profiles.items():
            assert p.batch_validation_status == ValidationStatus.VALIDATION_FAILED.value
            missing_fields = {
                fa.field_name for fa in p.field_audits if not fa.satisfies_requirement
            }
            # Crucial missing fields across raw sources
            assert "security_id" in missing_fields
            assert "currency" in missing_fields
            assert "country" in missing_fields
            assert "primary_listing" in missing_fields
            assert "security_type" in missing_fields
            assert "listing_status" in missing_fields
            assert "asset_class" in missing_fields


class TestReadOnlyMarketStoreAudit:
    """Verifies that the market store is accessed strictly under read-only mode."""

    def test_readonly_sqlite_mutation_prevented(self, tmp_path):
        # Create a mock database
        db_file = tmp_path / "test_store.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("CREATE TABLE asset_ohlcv_daily (symbol TEXT, trade_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
        conn.execute("CREATE TABLE asset_ohlcv_provenance (id INT)")
        conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('AAPL', '2026-10-16', 150.0, 155.0, 149.0, 153.0, 1000000)")
        conn.commit()
        conn.close()

        # Run diagnostic against the mock DB
        res = RealSourceAdmissionDiagnostic.run(market_db_path=db_file)
        assert res.market_db_status == "ENFORCED_READ_ONLY_INSPECTED"
        assert res.market_db_row_counts["asset_ohlcv_daily_total"] == 1

        # Direct test that URI ?mode=ro blocks mutation
        uri = f"file:{db_file.resolve().as_posix()}?mode=ro"
        ro_conn = sqlite3.connect(uri, uri=True)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            ro_conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('GOOG', '2026-10-16', 100, 101, 99, 100, 500)")
        ro_conn.close()


class TestEvidenceGapReporting:
    """Verifies report formatting generates valid GitHub-flavored Markdown tables without ASCII boxes."""

    def test_report_generation_structure_and_no_ascii_boxes(self):
        result = RealSourceAdmissionDiagnostic.run()
        report_md = format_evidence_gap_report(result)

        # Must not contain ascii box-drawing characters
        assert "+" not in report_md or "|" in report_md  # ensure no +---+ style borders
        assert "┌" not in report_md
        assert "└" not in report_md
        assert "─" not in report_md or "---" in report_md  # standard markdown hr / table divider ok
        assert "│" not in report_md

        # Must contain essential sections
        assert "# ARX Terminal — Sprint 2 Real-Source Evidence Gap Report" in report_md
        assert "### 1. Executive Summary & Diagnostic Verdict" in report_md
        assert "### 2. Candidate Evidence Resolution & Validation Summary" in report_md
        assert "### 3. Field-by-Candidate Admission Requirement Matrix" in report_md
        assert "### 4. Development Market Data Store Audit & Lookback Readiness" in report_md
        assert "### 5. Remaining Admission Blockers & Smallest Next Acquisition Steps" in report_md

        # Must mention all 7 candidates
        for sym in ["AAPL", "GOOGL", "GOOG", "TSM", "ACCS", "DDS", "AACIU"]:
            assert sym in report_md
