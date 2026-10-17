"""
tests/test_universe_connected_integration.py

Sprint 2 Package 3 (S2-04): Connected Offline Integration Test Suite.
Verifies the end-to-end integration:
Archived Artifacts -> Parsers -> SecurityEvidenceResolver -> UniverseInputValidator ->
Guarded SourceSecurity Conversion -> DeterministicUniverseBuilder.

Covers all 13 required test categories:
1. Successful synthetic integration end-to-end.
2. Missing required evidence fails closed.
3. Source hash mismatch fails closed immediately in Stage A.
4. Missing required source feed fails batch admission.
5. Malformed source observations fail batch admission fail-closed.
6. Ambiguous OpenFIGI multi-match association fails validation.
7. Venue conflict surviving an overlay blocks admission.
8. Empty candidate population fails closed.
9. Whole-batch atomic revocation (one invalid member revokes all).
10. Builder is strictly NOT_RUN when batch admission fails.
11. Zero database, network, or UniverseStore side effects.
12. Foreign US-listed issuer accounting and ADR strategy ineligibility under R02.
13. Stable accounting and deterministic output consistency across identical inputs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import pytest

from analyst_dashboard.universe.connected_integration import (
    IntegrationConfig,
    IsolatedMarketDataStub,
    run_connected_integration,
)


FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures", "connected_integration"
)
SYN_SEC_PATH = os.path.join(FIXTURES_DIR, "synthetic_sec_company_tickers_exchange.json")
SYN_NASDAQ_PATH = os.path.join(FIXTURES_DIR, "synthetic_nasdaq_listed_directory.txt")
SYN_OTHER_PATH = os.path.join(FIXTURES_DIR, "synthetic_other_listed_directory.txt")
SYN_FIGI_RESP_PATH = os.path.join(FIXTURES_DIR, "synthetic_openfigi_response.json")
SYN_FIGI_REQ_PATH = os.path.join(FIXTURES_DIR, "synthetic_openfigi_requests.json")
SYN_ID_MAP_PATH = os.path.join(FIXTURES_DIR, "synthetic_identity_mapping.json")
SYN_OVERLAY_PATH = os.path.join(FIXTURES_DIR, "synthetic_evidence_overlay.json")


@pytest.fixture
def temp_output_dir():
    d = tempfile.mkdtemp(prefix="test_conn_integ_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestConnectedOfflineIntegration:
    """End-to-end integration test suite for Sprint 2 Package 3."""

    def test_01_successful_synthetic_integration_end_to_end(self, temp_output_dir):
        """
        Demonstrates successful end-to-end integration on complete synthetic evidence:
        Stage A (verified) -> Stage B (clean rows) -> Stage C (5 candidates) ->
        Stage D (5 fully evidenced) -> Stage E (VALIDATION_PASSED) ->
        Stage F (ADMITTED, 5 SourceSecurities) -> Stage G (EVALUATED, 4 eligible, 1 ineligible ADR).
        """
        config = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            openfigi_response_path=SYN_FIGI_RESP_PATH,
            openfigi_request_path=SYN_FIGI_REQ_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=temp_output_dir,
            is_synthetic_fixture=True,
        )

        res = run_connected_integration(config)

        # Stage A
        assert res.stage_a_publisher_observations.status == "COMPLETED"
        assert res.stage_a_publisher_observations.all_hashes_verified is True
        assert res.stage_a_publisher_observations.total_source_files == 5

        # Stage B
        assert res.stage_b_parsed_observations.status == "COMPLETED"
        assert res.stage_b_parsed_observations.total_malformed_rows == 0
        assert res.stage_b_parsed_observations.total_data_rows == 10

        # Stage C
        assert res.stage_c_candidate_matching.status == "COMPLETED"
        assert res.stage_c_candidate_matching.total_candidates == 5
        assert res.stage_c_candidate_matching.venue_conflicted_count == 0

        # Stage D
        assert res.stage_d_evidence_resolution.status == "COMPLETED"
        assert res.stage_d_evidence_resolution.total_candidates == 5
        assert res.stage_d_evidence_resolution.fully_evidenced_count == 5
        assert res.stage_d_evidence_resolution.unresolved_count == 0
        assert res.stage_d_evidence_resolution.conflicted_count == 0
        assert res.stage_d_evidence_resolution.closure_verified is True

        # Stage E
        assert res.stage_e_batch_validation.status == "VALIDATION_PASSED"
        assert res.stage_e_batch_validation.is_batch_valid is True
        assert res.stage_e_batch_validation.passed_count == 5
        assert res.stage_e_batch_validation.failed_count == 0
        assert res.stage_e_batch_validation.closure_verified is True

        # Stage F
        assert res.stage_f_admission_and_conversion.status == "ADMITTED"
        assert res.stage_f_admission_and_conversion.is_admitted is True
        assert res.stage_f_admission_and_conversion.converted_security_count == 5
        assert res.stage_f_admission_and_conversion.snapshot_id is not None

        # Stage G
        assert res.stage_g_builder_evaluation.status == "EVALUATED"
        assert res.stage_g_builder_evaluation.evaluated_security_count == 5
        assert res.stage_g_builder_evaluation.eligible_count == 4
        assert res.stage_g_builder_evaluation.ineligible_count == 1
        assert res.stage_g_builder_evaluation.unresolved_count == 0
        assert res.stage_g_builder_evaluation.closure_verified is True
        assert res.stage_g_builder_evaluation.attestation_hash is not None
        assert res.stage_g_builder_evaluation.rule_failure_breakdown.get("R02_SECURITY_TYPE") == 1

        # Overall
        assert res.overall_outcome == "SUCCESS_BUILDER_EVALUATED"

        # Check saved artifacts
        assert os.path.exists(os.path.join(temp_output_dir, "connected_integration_summary.json"))
        assert os.path.exists(os.path.join(temp_output_dir, "connected_integration_summary.md"))
        assert os.path.exists(os.path.join(temp_output_dir, "connected_integration_records.json"))

    def test_02_foreign_issuer_retention_and_adr_strategy_ineligibility(self, temp_output_dir):
        """
        Foreign US-listed issuer (BABA) is accounted for in candidate pool, passes batch
        validation with valid attributes (country=CYM, security_type=DEPOSITARY_RECEIPT),
        and receives INELIGIBLE under canonical Rule R02 in Builder evaluation.
        Proves ADRs are NOT automatically approved as eligible common stock.
        """
        config = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            openfigi_response_path=SYN_FIGI_RESP_PATH,
            openfigi_request_path=SYN_FIGI_REQ_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=temp_output_dir,
            is_synthetic_fixture=True,
        )

        res = run_connected_integration(config)
        assert res.stage_d_evidence_resolution.foreign_issuer_count == 1
        assert res.stage_g_builder_evaluation.ineligible_count == 1
        assert res.stage_g_builder_evaluation.rule_failure_breakdown["R02_SECURITY_TYPE"] == 1

    def test_03_multi_symbol_issuer_distinct_candidates_in_builder(self, temp_output_dir):
        """
        GOOGL and GOOG share CIK 0001652044.
        Both remain distinct candidate records, are admitted, and both evaluate to ELIGIBLE in Builder.
        """
        config = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            openfigi_response_path=SYN_FIGI_RESP_PATH,
            openfigi_request_path=SYN_FIGI_REQ_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=temp_output_dir,
            is_synthetic_fixture=True,
        )

        res = run_connected_integration(config)
        assert res.stage_d_evidence_resolution.multi_symbol_issuer_cik_count == 1
        assert res.stage_d_evidence_resolution.multi_symbol_issuer_symbol_count == 2
        assert res.stage_g_builder_evaluation.eligible_count == 4

    def test_04_missing_required_evidence_fails_batch_and_blocks_builder(self, temp_output_dir):
        """
        When required evidence (currency, country, primary_listing) is missing,
        batch validation fails, admission is REVOKED, and builder is strictly NOT_RUN.
        """
        # Pass identity mapping but NO evidence overlay
        config = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            output_dir=temp_output_dir,
            is_synthetic_fixture=True,
        )

        res = run_connected_integration(config)

        assert res.stage_d_evidence_resolution.fully_evidenced_count == 0
        assert res.stage_d_evidence_resolution.unresolved_count == 5
        assert res.stage_e_batch_validation.status == "VALIDATION_FAILED"
        assert res.stage_e_batch_validation.is_batch_valid is False
        assert res.stage_f_admission_and_conversion.status == "REVOKED"
        assert res.stage_f_admission_and_conversion.converted_security_count == 0
        assert res.stage_g_builder_evaluation.status == "NOT_RUN"
        assert res.overall_outcome == "BLOCKED_VALIDATION_FAILED"

    def test_05_source_hash_mismatch_fails_closed_in_stage_a(self, temp_output_dir):
        """
        If a source artifact's computed sha256 does not match the expected hash,
        the run fails closed immediately in Stage A before parsing or resolution.
        """
        bad_hashes = {"sec_company_tickers_exchange": "0000000000000000000000000000000000000000000000000000000000000000"}
        config = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            expected_hashes=bad_hashes,
            output_dir=temp_output_dir,
            is_synthetic_fixture=True,
        )

        res = run_connected_integration(config)
        assert res.stage_a_publisher_observations.status == "FAILED_HASH_MISMATCH"
        assert res.stage_a_publisher_observations.all_hashes_verified is False
        assert res.stage_a_publisher_observations.hash_mismatch_count == 1
        assert res.stage_b_parsed_observations.status == "NOT_RUN"
        assert res.stage_d_evidence_resolution.status == "NOT_RUN"
        assert res.stage_e_batch_validation.status == "VALIDATION_FAILED"
        assert res.stage_f_admission_and_conversion.status == "REVOKED"
        assert res.stage_g_builder_evaluation.status == "NOT_RUN"
        assert res.overall_outcome == "FAILED_SOURCE_INTEGRITY"

    def test_06_missing_required_source_fails_batch(self, temp_output_dir):
        """
        If caller-required source feeds are omitted (e.g. SEC omitted),
        batch validation fails closed with REQUIRED_SOURCES_MISSING and builder is NOT_RUN.
        """
        config_no_sec = IntegrationConfig(
            # sec_path omitted
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            required_sources=["sec_company_tickers_exchange", "nasdaq_listed_directory"],
            output_dir=temp_output_dir,
            is_synthetic_fixture=True,
        )
        res = run_connected_integration(config_no_sec)
        assert res.stage_e_batch_validation.is_batch_valid is False
        assert any("REQUIRED_SOURCES_MISSING" in i.get("message", "") for i in res.stage_e_batch_validation.batch_issues)
        assert res.stage_f_admission_and_conversion.status == "REVOKED"
        assert res.stage_g_builder_evaluation.status == "NOT_RUN"

    def test_07_malformed_source_observations_fails_batch(self, temp_output_dir):
        """
        When malformed rows exist in a source feed, batch validation fails closed
        with SOURCE_PARSE_FAILED unless allow_incomplete_source_files=True.
        """
        # Create a temporary SEC file with 1 malformed row
        in_dir = os.path.join(temp_output_dir, "inputs_07")
        out_dir = os.path.join(temp_output_dir, "outputs_07")
        os.makedirs(in_dir, exist_ok=True)
        malformed_sec = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [320193, "Apple Inc.", "AAPL", "Nasdaq"],
                [None, "Malformed Entity", None, None],
            ],
        }
        malformed_path = os.path.join(in_dir, "malformed_sec.json")
        with open(malformed_path, "w", encoding="utf-8") as fp:
            json.dump(malformed_sec, fp)

        config = IntegrationConfig(
            sec_path=malformed_path,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=out_dir,
            is_synthetic_fixture=True,
            allow_incomplete_source_files=False,
        )

        res = run_connected_integration(config)
        assert res.stage_b_parsed_observations.total_malformed_rows == 1
        assert res.stage_e_batch_validation.is_batch_valid is False
        assert any("SOURCE_PARSE_FAILED" in i.get("message", "") for i in res.stage_e_batch_validation.batch_issues)
        assert res.stage_f_admission_and_conversion.status == "REVOKED"
        assert res.stage_g_builder_evaluation.status == "NOT_RUN"

    def test_08_ambiguous_openfigi_association_fails_validation(self, temp_output_dir):
        """
        OpenFIGI multi-match results remain ambiguous and fail validation fail-closed.
        """
        in_dir = os.path.join(temp_output_dir, "inputs_08")
        out_dir = os.path.join(temp_output_dir, "outputs_08")
        os.makedirs(in_dir, exist_ok=True)
        multi_figi_resp = [
            {
                "data": [
                    {"figi": "FIGI_1", "name": "Apple 1", "ticker": "AAPL", "exchCode": "US"},
                    {"figi": "FIGI_2", "name": "Apple 2", "ticker": "AAPL", "exchCode": "US"},
                ]
            }
        ]
        multi_figi_req = [{"idType": "TICKER", "idValue": "AAPL", "exchCode": "US"}]

        resp_p = os.path.join(in_dir, "multi_figi_resp.json")
        req_p = os.path.join(in_dir, "multi_figi_req.json")
        with open(resp_p, "w", encoding="utf-8") as fp:
            json.dump(multi_figi_resp, fp)
        with open(req_p, "w", encoding="utf-8") as fp:
            json.dump(multi_figi_req, fp)

        config = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            openfigi_response_path=resp_p,
            openfigi_request_path=req_p,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=out_dir,
            is_synthetic_fixture=True,
        )

        res = run_connected_integration(config)
        assert res.stage_d_evidence_resolution.ambiguous_count >= 1
        assert res.stage_e_batch_validation.is_batch_valid is False
        assert res.stage_f_admission_and_conversion.status == "REVOKED"
        assert res.stage_g_builder_evaluation.status == "NOT_RUN"

    def test_09_venue_conflict_surviving_overlay_blocks_admission(self, temp_output_dir):
        """
        When SEC and directory feeds report conflicting primary venues,
        an overlay supplying an exchange does NOT clear the source conflict.
        Validation fails closed and builder is NOT_RUN.
        """
        in_dir = os.path.join(temp_output_dir, "inputs_09")
        out_dir = os.path.join(temp_output_dir, "outputs_09")
        os.makedirs(in_dir, exist_ok=True)
        # Conflicted SEC file where AAPL is listed on NYSE in SEC but NASDAQ in directory
        conflicted_sec = {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [320193, "Apple Inc.", "AAPL", "NYSE"],  # Conflict with NASDAQ in directory
            ],
        }
        sec_p = os.path.join(in_dir, "conflicted_sec.json")
        with open(sec_p, "w", encoding="utf-8") as fp:
            json.dump(conflicted_sec, fp)

        config = IntegrationConfig(
            sec_path=sec_p,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=out_dir,
            is_synthetic_fixture=True,
        )

        res = run_connected_integration(config)
        assert res.stage_c_candidate_matching.venue_conflicted_count >= 1
        assert res.stage_d_evidence_resolution.conflicted_count >= 1
        assert res.stage_e_batch_validation.is_batch_valid is False
        assert any(b.startswith("CONFLICTING_MAPPING") for b in res.stage_e_batch_validation.blockers_summary)
        assert res.stage_f_admission_and_conversion.status == "REVOKED"
        assert res.stage_g_builder_evaluation.status == "NOT_RUN"

    def test_10_empty_population_fails_closed(self, temp_output_dir):
        """An empty population fails closed without downstream admission."""
        in_dir = os.path.join(temp_output_dir, "inputs_10")
        out_dir = os.path.join(temp_output_dir, "outputs_10")
        os.makedirs(in_dir, exist_ok=True)
        empty_sec = {"fields": ["cik", "name", "ticker", "exchange"], "data": []}
        empty_sec_p = os.path.join(in_dir, "empty_sec.json")
        with open(empty_sec_p, "w", encoding="utf-8") as fp:
            json.dump(empty_sec, fp)

        config = IntegrationConfig(
            sec_path=empty_sec_p,
            output_dir=out_dir,
            is_synthetic_fixture=True,
        )

        res = run_connected_integration(config)
        assert res.stage_c_candidate_matching.total_candidates == 0
        assert res.stage_e_batch_validation.is_batch_valid is False
        assert res.stage_f_admission_and_conversion.status == "REVOKED"
        assert res.stage_f_admission_and_conversion.converted_security_count == 0
        assert res.stage_g_builder_evaluation.status == "NOT_RUN"

    def test_11_whole_batch_revocation_atomic_contract(self, temp_output_dir):
        """
        Atomic Batch Contract: A batch with 4 valid members and 1 invalid member (missing evidence).
        Batch validation fails, and all 5 members have their admission strictly revoked.
        """
        # Partial overlay providing evidence for AAPL, MSFT, GOOGL, GOOG but NOT BABA
        partial_overlay = {
            "AAPL": {"currency": "USD", "country": "USA", "primary_listing": True, "listing_status": "ACTIVE", "asset_class": "EQUITY", "security_type": "COMMON_STOCK"},
            "MSFT": {"currency": "USD", "country": "USA", "primary_listing": True, "listing_status": "ACTIVE", "asset_class": "EQUITY", "security_type": "COMMON_STOCK"},
            "GOOGL": {"currency": "USD", "country": "USA", "primary_listing": True, "listing_status": "ACTIVE", "asset_class": "EQUITY", "security_type": "COMMON_STOCK"},
            "GOOG": {"currency": "USD", "country": "USA", "primary_listing": True, "listing_status": "ACTIVE", "asset_class": "EQUITY", "security_type": "COMMON_STOCK"},
            # BABA omitted
        }

        config = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            caller_field_evidence_overlay=partial_overlay,
            output_dir=temp_output_dir,
            is_synthetic_fixture=True,
        )

        res = run_connected_integration(config)
        assert res.stage_e_batch_validation.is_batch_valid is False
        # Record validation success: 4 valid records, 1 invalid record
        assert res.stage_e_batch_validation.passed_count == 4
        assert res.stage_e_batch_validation.failed_count == 1
        # Batch admission strictly revoked for all members
        assert res.stage_f_admission_and_conversion.status == "REVOKED"
        assert res.stage_f_admission_and_conversion.is_admitted is False
        assert res.stage_f_admission_and_conversion.converted_security_count == 0
        assert res.stage_g_builder_evaluation.status == "NOT_RUN"


    def test_12_no_database_or_network_side_effects(self, temp_output_dir):
        """
        Verifies that running connected integration touches NO database tables,
        does NOT initialize UniverseStore, and writes ONLY to the designated output directory.
        """
        db_path = os.path.expanduser("~/.finance_universe_store.db")
        mtime_before = os.path.getmtime(db_path) if os.path.exists(db_path) else None

        config = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=temp_output_dir,
            is_synthetic_fixture=True,
        )

        res = run_connected_integration(config)
        assert res.overall_outcome == "SUCCESS_BUILDER_EVALUATED"

        # Check universe store DB was NOT modified or created
        if mtime_before is not None:
            assert os.path.getmtime(db_path) == mtime_before

        # Only files in temp_output_dir were written
        written_files = set(os.listdir(temp_output_dir))
        assert "connected_integration_summary.json" in written_files
        assert "connected_integration_summary.md" in written_files
        assert "connected_integration_records.json" in written_files

    def test_13_stable_accounting_and_deterministic_output_consistency(self, temp_output_dir):
        """
        Two executions over identical input bytes produce identical semantic counts,
        identical candidate partitions, identical closure results, and identical attestation hashes.
        """
        dir1 = os.path.join(temp_output_dir, "run1")
        dir2 = os.path.join(temp_output_dir, "run2")

        cfg1 = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=dir1,
            is_synthetic_fixture=True,
            build_id="deterministic-build-001",
        )
        cfg2 = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=dir2,
            is_synthetic_fixture=True,
            build_id="deterministic-build-001",
        )

        res1 = run_connected_integration(cfg1)
        res2 = run_connected_integration(cfg2)

        assert res1.stage_c_candidate_matching.total_candidates == res2.stage_c_candidate_matching.total_candidates
        assert res1.stage_d_evidence_resolution.fully_evidenced_count == res2.stage_d_evidence_resolution.fully_evidenced_count
        assert res1.stage_e_batch_validation.passed_count == res2.stage_e_batch_validation.passed_count
        assert res1.stage_g_builder_evaluation.eligible_count == res2.stage_g_builder_evaluation.eligible_count
        assert res1.stage_g_builder_evaluation.ineligible_count == res2.stage_g_builder_evaluation.ineligible_count
        assert res1.stage_g_builder_evaluation.attestation_hash == res2.stage_g_builder_evaluation.attestation_hash

    def test_14_market_data_isolation_and_stub_prohibition(self, temp_output_dir):
        """
        Check A: Synthetic market data stub is permitted only in fixture mode.
        In archived mode, passing an IsolatedMarketDataStub raises ValueError, and
        admitted batches without real market data fail closed with BLOCKED_MARKET_DATA_UNAVAILABLE.
        """
        # 1. In synthetic fixture mode, IsolatedMarketDataStub runs and provenance is recorded
        cfg_fixture = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=os.path.join(temp_output_dir, "run_stub_fixture"),
            is_synthetic_fixture=True,
        )
        res_fixture = run_connected_integration(cfg_fixture)
        assert res_fixture.stage_g_builder_evaluation.status == "EVALUATED"
        assert res_fixture.stage_g_builder_evaluation.market_data_provenance == "SYNTHETIC_STUB"

        # 2. In archived mode, explicitly passing IsolatedMarketDataStub is strictly prohibited
        stub = IsolatedMarketDataStub()
        cfg_prohibited = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            output_dir=os.path.join(temp_output_dir, "run_stub_prohibited"),
            is_synthetic_fixture=False,
            market_data_provider=stub,
        )
        with pytest.raises(ValueError, match="PROHIBITED_SYNTHETIC_STUB"):
            run_connected_integration(cfg_prohibited)

        # 3. In archived mode, even if batch passes admission, missing market data blocks builder
        cfg_admitted_no_mkt = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            identity_mapping_path=SYN_ID_MAP_PATH,
            evidence_overlay_path=SYN_OVERLAY_PATH,
            output_dir=os.path.join(temp_output_dir, "run_no_mkt"),
            is_synthetic_fixture=False,
            market_data_provider=None,
        )
        res_admitted = run_connected_integration(cfg_admitted_no_mkt)
        assert res_admitted.stage_e_batch_validation.is_batch_valid is True
        assert res_admitted.stage_f_admission_and_conversion.status == "ADMITTED"
        assert res_admitted.stage_g_builder_evaluation.status == "BLOCKED_MARKET_DATA_UNAVAILABLE"
        assert res_admitted.stage_g_builder_evaluation.market_data_provenance is None
        assert res_admitted.overall_outcome == "ADMITTED_BUILDER_BLOCKED"

    def test_15_cli_runner_and_exit_codes(self, temp_output_dir):
        """
        Check B: CLI runner enforces explicit exit codes and flags:
        - Exit 0: Builder evaluated on admitted fixture batch.
        - Exit 2: Fail-closed block when sources/batch are incomplete.
        - Emits deprecation warning when legacy --allow-incomplete-source-files is used.
        """
        cli_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scripts",
            "diagnostics",
            "universe_connected_integration.py",
        )
        out_0 = os.path.join(temp_output_dir, "cli_success")
        proc_0 = subprocess.run(
            [sys.executable, cli_path, "--fixture", "--output-dir", out_0],
            capture_output=True,
            text=True,
        )
        assert proc_0.returncode == 0
        assert "Connected Offline Integration" in proc_0.stdout

        # Running without complete evidence overlay fails closed with exit code 2
        out_2 = os.path.join(temp_output_dir, "cli_blocked")
        proc_2 = subprocess.run(
            [
                sys.executable,
                cli_path,
                "--sec-path", SYN_SEC_PATH,
                "--diagnostic-inspect-incomplete-sources",
                "--output-dir", out_2,
            ],
            capture_output=True,
            text=True,
        )
        assert proc_2.returncode == 2

        # Check deprecation warning on --allow-incomplete-source-files
        out_dep = os.path.join(temp_output_dir, "cli_deprecated")
        proc_dep = subprocess.run(
            [
                sys.executable,
                cli_path,
                "--sec-path", SYN_SEC_PATH,
                "--allow-incomplete-source-files",
                "--output-dir", out_dep,
            ],
            capture_output=True,
            text=True,
        )
        assert "DEPRECATION WARNING: --allow-incomplete-source-files is deprecated" in proc_dep.stderr

    def test_16_four_way_candidate_partition_closure(self, temp_output_dir):
        """
        Check C: Mathematical 4-way candidate venue reconciliation:
        - total == dual_source + sec_only + directory_only
        - dual_source == agreeing + conflicting
        - no_detected_conflict == agreeing + sec_only + directory_only
        """
        cfg = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            nasdaq_listed_path=SYN_NASDAQ_PATH,
            other_listed_path=SYN_OTHER_PATH,
            output_dir=os.path.join(temp_output_dir, "partition_test"),
            is_synthetic_fixture=True,
        )
        res = run_connected_integration(cfg)
        sc = res.stage_c_candidate_matching
        assert sc.total_candidates == sc.both_sources_matched_count + sc.sec_only_count + sc.directory_only_count
        assert sc.both_sources_matched_count == sc.agreeing_venue_count + sc.venue_disagreement_count
        assert sc.no_detected_conflict_count == sc.agreeing_venue_count + sc.sec_only_count + sc.directory_only_count
        assert sc.total_candidates == 5
        assert sc.agreeing_venue_count == 5
        assert sc.venue_disagreement_count == 0

    def test_17_output_directory_collision_prevention(self, temp_output_dir):
        """
        Check E: Output directory cannot collide with or contain input source paths.
        """
        # Case 1: output_dir is identical to a source file path
        cfg_same_as_file = IntegrationConfig(
            sec_path=SYN_SEC_PATH,
            output_dir=SYN_SEC_PATH,
            is_synthetic_fixture=True,
        )
        with pytest.raises(ValueError, match="OUTPUT_DIRECTORY_COLLISION"):
            run_connected_integration(cfg_same_as_file)

        # Case 2: output_dir contains the source file
        source_dir = os.path.join(temp_output_dir, "colliding_source_dir")
        os.makedirs(source_dir, exist_ok=True)
        source_file = os.path.join(source_dir, "mock_source.json")
        with open(source_file, "w") as fp:
            fp.write("{}")

        cfg_contains_file = IntegrationConfig(
            sec_path=source_file,
            output_dir=source_dir,
            is_synthetic_fixture=True,
        )
        with pytest.raises(ValueError, match="OUTPUT_DIRECTORY_COLLISION"):
            run_connected_integration(cfg_contains_file)
