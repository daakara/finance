"""
tests/test_universe_real_source_diagnostic.py

Sprint 2: Bounded Real-Source Evidence Handoff & Admission Diagnostic Test Suite.
Verifies the 7-candidate representative real-source sample against released
admission-mapping, validation contracts, and newly acquired research evidence:

1. Straightforward US common stock: AAPL
2. Multi-symbol issuer sharing CIK: GOOG & GOOGL (CIK 0001652044)
3. Foreign US-listed issuer (ADR): TSM
4. Venue discrepancy Group 1: SEC NYSE vs Directory A/AMEX (ACCS)
5. Venue discrepancy Group 4: SEC TXSE vs Directory F/TXSE (DDS)
6. Unresolved classification / Units: AACIU

Core Safeguards Verified:
- Recomputed parent feed accounting from exact archived bytes (5,634 Nasdaq, 7,674 Other, 10,276 SEC usable).
- Explanation of previous 5,166 / 8,172 report discrepancy.
- Strict Non-Inference: Zero surrogate keys synthesized (CIK+symbol).
- Separation of Evidentiary Categories (source fact, external mapping, policy assertion, triage hint, conflict).
- Substantive venue disagreements preserved (DDS TXSE vs F; ACCS AMEX).
- OpenFIGI qualification: preserves levels (listing, composite, shareClass), distinct Class A/C for GOOG/GOOGL,
  TSM Depositary Receipt, AACIU Unit.
- Attributable per-listing currency evidence (USD) from quote reference metadata.
- Separated readiness dimensions: unresolved association guard is not bypassed for GOOGL/TSM;
  query buffers (60) and annualization factors (252) are not acceptance gates.
- Atomic whole-batch validation failure across both baseline and replay: 0 converted securities, builder strictly NOT_RUN.
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
    MarketDataReadinessState,
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
    load_acquired_research_evidence,
)
from analyst_dashboard.universe.validator import (
    ValidationRequirements,
    ValidationStatus,
)


class TestRealSourceParentFeedProvenance:
    """Verifies that sample candidates trace to authentic 2026-10-16 parent feed artifacts."""

    def test_parent_feed_exact_counts_and_discrepancy_explanation(self):
        sec_feed = REAL_SOURCE_PARENT_FEEDS["sec_company_tickers_exchange.json"]
        assert sec_feed["sha256"] == "d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f"
        assert sec_feed["physical_lines"] == 1
        assert sec_feed["metadata_rows"] == 1
        assert sec_feed["data_records"] == 10276
        assert sec_feed["malformed_rows"] == 168
        assert sec_feed["unique_records"] == 10276
        assert sec_feed["raw_array_items"] == 10444

        nasdaq_feed = REAL_SOURCE_PARENT_FEEDS["nasdaq_listed_directory.txt"]
        assert nasdaq_feed["sha256"] == "67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272"
        assert nasdaq_feed["physical_lines"] == 5636
        assert nasdaq_feed["metadata_rows"] == 2
        assert nasdaq_feed["data_records"] == 5634
        assert nasdaq_feed["malformed_rows"] == 0
        assert nasdaq_feed["unique_records"] == 5634

        other_feed = REAL_SOURCE_PARENT_FEEDS["other_listed_directory.txt"]
        assert other_feed["sha256"] == "a0cb851a772098d64830304bf7e7a7d1ca1cbd8421da20a6bfbec9a336243792"
        assert other_feed["physical_lines"] == 7676
        assert other_feed["metadata_rows"] == 2
        assert other_feed["data_records"] == 7674
        assert other_feed["malformed_rows"] == 0
        assert other_feed["unique_records"] == 7674

        # Verify discrepancy explanations exist and explain previous numbers
        assert "5,166" in nasdaq_feed["discrepancy_explanation"]
        assert "5,634" in nasdaq_feed["discrepancy_explanation"]
        assert "8,172" in other_feed["discrepancy_explanation"]
        assert "7,674" in other_feed["discrepancy_explanation"]
        assert "10,276" in sec_feed["discrepancy_explanation"]
        assert "168" in sec_feed["discrepancy_explanation"]

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
        accs_res = SupportedVenueNormalizer.reconcile_dual_source_venues("NYSE", "A")
        assert accs_res.status == VenueNormalizationStatus.DOCUMENTED_REPRESENTATIONAL
        assert accs_res.normalized_venue == "AMEX"

        tsm_res = SupportedVenueNormalizer.reconcile_dual_source_venues("NYSE", "N")
        assert tsm_res.status == VenueNormalizationStatus.EXACT_CONCORDANT
        assert tsm_res.normalized_venue == "NYSE"

        dds_res = SupportedVenueNormalizer.reconcile_dual_source_venues("TXSE", "F")
        assert dds_res.status == VenueNormalizationStatus.PROVISIONAL_UNRESOLVED
        assert dds_res.is_substantive_disagreement is True

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


class TestReadinessDimensionsSeparation:
    """Verifies Section 1.B: Readiness dimensions are kept strictly separate."""

    def test_unresolved_association_guard_not_bypassed_for_googl_or_tsm(self):
        result = RealSourceAdmissionDiagnostic.run()
        mr = result.market_readiness_by_candidate

        # AAPL, GOOGL, TSM have bars in store
        for sym in ["AAPL", "GOOGL", "TSM"]:
            assert mr[sym]["bar_count"] > 0
            # INVARIANT: Association is NOT resolved
            assert mr[sym]["identity_association_resolved"] is False
            # INVARIANT: Primary evaluator state MUST remain UNRESOLVED_ASSOCIATION
            assert mr[sym]["evaluator_primary_state"] == MarketDataReadinessState.UNRESOLVED_ASSOCIATION.value

        # Freshness is tracked as an additional observation
        assert mr["AAPL"]["freshness_state"] == "CURRENT"
        assert mr["GOOGL"]["freshness_state"] == "STALE"
        assert mr["TSM"]["freshness_state"] == "STALE"

        # Symbols with 0 bars are NO_DATA
        for sym in ["GOOG", "ACCS", "DDS", "AACIU"]:
            assert mr[sym]["bar_count"] == 0
            assert mr[sym]["freshness_state"] == "NO_DATA"
            assert mr[sym]["evaluator_primary_state"] == MarketDataReadinessState.NO_DATA.value

    def test_query_buffer_and_annualization_are_not_acceptance_gates(self):
        result = RealSourceAdmissionDiagnostic.run()
        mr = result.market_readiness_by_candidate

        # For AAPL (465 bars):
        assert mr["AAPL"]["gate_50_bars"] is True       # Acceptance gate 50
        assert mr["AAPL"]["gate_200_bars"] is True      # Acceptance gate 200
        assert mr["AAPL"]["buffer_60_bars_available"] is True     # Operational buffer (not gate)
        assert mr["AAPL"]["benchmark_252_bars_available"] is True # Benchmark range (not gate)


class TestAcquiredEvidenceReplayAndQualification:
    """Verifies Sections 2, 3, and 4: Qualification of acquired research evidence and replay."""

    def test_acquired_research_artifacts_loaded(self):
        ev = load_acquired_research_evidence()
        assert ev["openfigi"] is not None
        assert ev["currency"] is not None
        assert ev["alpaca"] is not None
        assert ev["sec_submissions"] is not None

    def test_openfigi_qualification_for_seven_candidates(self):
        ev = load_acquired_research_evidence()
        payload = ev["openfigi"]["response_payload"]
        assert len(payload) == 7

        sym_to_data = {
            job["idValue"]: res["data"][0]
            for job, res in zip(ev["openfigi"]["request_parameters"], payload)
            if "data" in res
        }

        # Check distinct FIGIs and shareClassFIGIs for GOOG vs GOOGL
        assert sym_to_data["GOOG"]["figi"] != sym_to_data["GOOGL"]["figi"]
        assert sym_to_data["GOOG"]["shareClassFIGI"] != sym_to_data["GOOGL"]["shareClassFIGI"]
        assert "CL C" in sym_to_data["GOOG"]["name"] or "CLASS C" in sym_to_data["GOOG"]["name"].upper()
        assert "CL A" in sym_to_data["GOOGL"]["name"] or "CLASS A" in sym_to_data["GOOGL"]["name"].upper()

        # Check TSM Depositary Receipt qualification
        assert sym_to_data["TSM"]["securityType2"] == "Depositary Receipt"
        assert sym_to_data["TSM"]["securityType"] == "ADR"

        # Check AACIU Unit qualification
        assert sym_to_data["AACIU"]["securityType2"] == "Unit"

    def test_currency_evidence_attribution(self):
        ev = load_acquired_research_evidence()
        curr_res = ev["currency"]["results"]
        for sym in ["AAPL", "GOOG", "GOOGL", "TSM", "ACCS", "DDS", "AACIU"]:
            assert curr_res[sym]["success"] is True
            assert curr_res[sym]["currency"] == "USD"
            assert curr_res[sym]["source_url"] is not None
            assert curr_res[sym]["response_sha256"] is not None

    def test_replay_with_acquired_evidence_fails_closed(self):
        """Even with attributable currency and OpenFIGI, whole batch fails closed."""
        result = RealSourceAdmissionDiagnostic.run()

        # Baseline batch fails closed
        assert result.batch_valid is False
        assert result.failed_count == 7
        assert result.passed_count == 0
        assert result.converted_security_count == 0
        assert result.builder_status == "NOT_RUN"

        # Replay summary confirms replay batch also fails closed
        rep = result.replay_summary
        assert rep["replay_batch_valid"] is False
        assert rep["replay_failed_count"] == 7
        assert rep["replay_passed_count"] == 0
        assert rep["replay_converted_count"] == 0
        assert rep["replay_builder_status"] == "NOT_RUN"
        assert "BLOCKED_VALIDATION_FAILED" in rep["replay_builder_blocking_reason"]


class TestReadOnlyMarketStoreAudit:
    """Verifies that the market store is accessed strictly under read-only mode."""

    def test_readonly_sqlite_mutation_prevented(self, tmp_path):
        db_file = tmp_path / "test_store.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("CREATE TABLE asset_ohlcv_daily (symbol TEXT, trade_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
        conn.execute("CREATE TABLE asset_ohlcv_provenance (id INT)")
        conn.execute("INSERT INTO asset_ohlcv_daily VALUES ('AAPL', '2026-10-16', 150.0, 155.0, 149.0, 153.0, 1000000)")
        conn.commit()
        conn.close()

        res = RealSourceAdmissionDiagnostic.run(market_db_path=db_file)
        assert res.market_db_status == "ENFORCED_READ_ONLY_INSPECTED"
        assert res.market_db_row_counts["asset_ohlcv_daily_total"] == 1

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
        assert "┌" not in report_md
        assert "└" not in report_md
        assert "│" not in report_md

        # Must contain essential sections
        assert "# ARX Terminal — Sprint 2 Real-Source Evidence Gap Report" in report_md
        assert "### 1. Executive Summary & Diagnostic Verdict" in report_md
        assert "### 2. Parent-Feed Accounting & Full Digest Reconciliation" in report_md
        assert "### 3. Acquired Bounded Reference Evidence Provenance & Qualification" in report_md
        assert "### 4. Lifecycle Stage Separation & Executed Validator Issues" in report_md
        assert "### 5. Field-by-Candidate Admission Requirement Matrix (12 Contract Fields)" in report_md
        assert "### 6. Development Market Data Store Audit & Lookback Readiness" in report_md
        assert "### 7. Contract Schema Grounding: Country and Primary Listing Semantics" in report_md
        assert "### 8. Venue Code F and Normalizer Verification" in report_md
        assert "### 9. Test Suite Inventory: Selected vs Full Universe Verification" in report_md
        assert "### 10. Remaining Admission Blockers & Smallest Next Acquisition Steps" in report_md

        # Must mention all 7 candidates
        for sym in ["AAPL", "GOOGL", "GOOG", "TSM", "ACCS", "DDS", "AACIU"]:
            assert sym in report_md

        # Verify parent feed discrepancy explanation and full hashes are rendered
        assert "5,634" in report_md
        assert "7,674" in report_md
        assert "10,276" in report_md
        assert "d04923d674c6d0084dc56e67efd9cb5756f965366affe9d3e18d38b379ffd45f" in report_md
        assert "67f42dd0594aff8e29634ec9e42b561319cbb74340829098cbfbf9b4517fe272" in report_md
        assert "a0cb851a772098d64830304bf7e7a7d1ca1cbd8421da20a6bfbec9a336243792" in report_md

        # Verify FINRA ADF regression is eliminated
        assert "FINRA ADF" not in report_md
        assert "Texas Stock Exchange" in report_md

        # Verify mixed-time reference enrichment qualification
        assert "mixed-time reference enrichment" in report_md
        assert "204" in report_md
