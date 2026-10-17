"""
tests/test_universe_offline_build_coverage.py

Sprint 2: Bounded Offline Universe Build & Coverage Integration Test Suite.

Verifies the complete offline path:
attributable candidate records
→ identity/listing association checks (admission_mapping.py)
→ input validation and atomic conversion (validator.py)
→ Universe Builder eligibility evaluation (builder.py, canonical Rules R01–R07)
→ market-data readiness diagnostics (coverage_diagnostics.py)
→ structured outcome report

Covers all required test categories:
1. Full pipeline execution through to diagnostics (Scenario A).
2. Whole-batch atomic conversion failure & zero partial handoff (Scenario B).
3. Strict no-network enforcement (network access blocked/mocked).
4. Real-source retained evidence replay & provenance preservation (Scenario C).
5. Diagnostic reporting accuracy & disjoint outcome precedence.
6. Cross-stage mathematical arithmetic closure.
7. Read-safety & zero database mutation (URI ?mode=ro).
8. CLI runner execution and artifact generation.
"""

from __future__ import annotations

from datetime import date, timedelta
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from typing import Generator
import pytest

from analyst_dashboard.universe.offline_build_coverage import (
    OfflineBuildCoverageConfig,
    OfflineBuildCoverageResult,
    OfflineCandidateRecord,
    build_scenario_a_candidates,
    build_scenario_b_candidates,
    build_scenario_b2_conversion_failure_candidates,
    build_scenario_d_identity_conflict_candidates,
    build_scenario_c_candidates,
    format_offline_build_coverage_markdown,
    run_offline_build_and_coverage,
    save_offline_build_coverage_artifacts,
)
from analyst_dashboard.universe.coverage_diagnostics import (
    PrimaryDiagnosticOutcome,
    ReadOnlyMarketStore,
)
from analyst_dashboard.universe.validator import ValidationRequirements

pytestmark = pytest.mark.tier2c


@pytest.fixture
def test_market_db(tmp_path: Path) -> Generator[str, None, None]:
    """Creates a temporary SQLite database seeded with controlled test histories."""
    db_file = tmp_path / "test_market_data.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE asset_ohlcv_daily (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            PRIMARY KEY (symbol, trade_date)
        )
    """)

    base_date = date(2025, 1, 1)
    target_date = "2026-10-16"

    # 1. AAPL: 252 bars, fresh session date (2026-10-16) -> DATA_READY
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else target_date
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("AAPL", d, 150.0, 155.0, 149.0, 152.0, 1000000),
        )

    # 2. MSFT: 200 bars, stale session date (2026-10-12) -> STALE_QUOTE
    for i in range(200):
        d = (base_date + timedelta(days=i)).isoformat() if i < 199 else "2026-10-12"
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("MSFT", d, 300.0, 305.0, 298.0, 302.0, 500000),
        )

    # 3. GLX: 47 bars (< 50 bars) -> INSUFFICIENT_HISTORY
    for i in range(47):
        d = (base_date + timedelta(days=i)).isoformat()
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("GLX", d, 10.0, 11.0, 9.5, 10.5, 50000),
        )

    # 4. SPY: 252 bars, fresh session date (2026-10-16) -> DATA_READY in market, ETF in builder
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else target_date
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("SPY", d, 500.0, 505.0, 498.0, 502.0, 5000000),
        )

    # 5. SEC1: 252 bars, fresh session date (2026-10-16) -> DATA_READY in market, secondary in builder
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else target_date
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("SEC1", d, 75.0, 77.0, 74.0, 76.0, 300000),
        )

    # 6. DISP1: 252 bars, fresh session date (2026-10-16)
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else target_date
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("DISP1", d, 25.0, 26.0, 24.5, 25.5, 100000),
        )

    # 7. INV_OHLCV: 55 bars, but last bar has negative price -> INVALID_OHLCV
    for i in range(54):
        d = (base_date + timedelta(days=i)).isoformat()
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("INV_OHLCV", d, 10.0, 11.0, 9.5, 10.0, 10000),
        )
    cur.execute(
        "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("INV_OHLCV", target_date, 10.0, 11.0, 9.5, -5.0, 10000),  # Negative close
    )

    # 8. UNSUP1: 252 bars, fresh session date (2026-10-16) -> DATA_READY
    for i in range(252):
        d = (base_date + timedelta(days=i)).isoformat() if i < 251 else target_date
        cur.execute(
            "INSERT INTO asset_ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("UNSUP1", d, 45.0, 46.0, 44.0, 45.5, 200000),
        )

    # Note: NOMKT is omitted to test MISSING_MARKET_DATA

    conn.commit()
    conn.close()

    yield str(db_file)


class TestUniverseOfflineBuildCoverage:
    """Test suite for bounded offline universe build & coverage integration."""

    def test_01_full_pipeline_scenario_a_convertible_batch_end_to_end(
        self, test_market_db: str, tmp_path: Path
    ) -> None:
        """
        Scenario A: Fully convertible batch reaching builder and readiness.
        Verifies:
        - 7 candidates pass Stage 1 (identity association).
        - 7 candidates pass Stage 2 (batch validation & atomic conversion).
        - UniverseBuilder evaluates all 7 (Stage 3): 4 eligible, 3 ineligible.
        - CoverageDiagnostics evaluates all 7 distinct symbols (Stage 4).
        - Exact arithmetic closure across all stages.
        """
        candidates = build_scenario_a_candidates()
        out_dir = str(tmp_path / "scenario_a_out")

        with ReadOnlyMarketStore(test_market_db) as store:
            config = OfflineBuildCoverageConfig(
                candidate_records=candidates,
                as_of_date="2026-10-16",
                market_store=store,
                scenario_label="SCENARIO_A_CONVERTIBLE_BATCH",
                is_synthetic=True,
                output_dir=out_dir,
            )
            result = run_offline_build_and_coverage(config)

        # Stage 1: Identity Association
        assert result.stage_1_identity_association.status == "COMPLETED"
        assert result.stage_1_identity_association.total_candidates == 7
        assert result.stage_1_identity_association.distinct_symbols == 7
        assert result.stage_1_identity_association.resolved_identities_count == 7
        assert result.stage_1_identity_association.blocked_identities_count == 0
        assert result.stage_1_identity_association.unavailable_identities_count == 0
        assert result.stage_1_identity_association.closure_verified is True

        # Stage 2: Batch Validation & Conversion
        assert result.stage_2_validation_and_conversion.status == "ADMITTED"
        assert result.stage_2_validation_and_conversion.component_validation_status == "ADMITTED"
        assert result.stage_2_validation_and_conversion.conversion_status == "SUCCEEDED"
        assert result.stage_2_validation_and_conversion.is_batch_valid is True
        assert result.stage_2_validation_and_conversion.passed_count == 7
        assert result.stage_2_validation_and_conversion.failed_count == 0
        assert result.stage_2_validation_and_conversion.converted_security_count == 7
        assert result.stage_2_validation_and_conversion.closure_verified is True

        # Stage 3: Universe Builder Evaluation (Rules R01–R07)
        assert result.stage_3_builder_evaluation.status == "EVALUATED"
        assert result.stage_3_builder_evaluation.evaluated_security_count == 7
        assert result.stage_3_builder_evaluation.eligible_count == 4
        assert result.stage_3_builder_evaluation.ineligible_count == 3
        assert result.stage_3_builder_evaluation.unresolved_count == 0
        assert set(result.stage_3_builder_evaluation.eligible_symbols) == {"AAPL", "MSFT", "GLX", "NOMKT"}
        assert set(result.stage_3_builder_evaluation.ineligible_symbols) == {"SPY", "SEC1", "UNSUP1"}
        assert result.stage_3_builder_evaluation.rule_failure_breakdown.get("R01_ASSET_CLASS") == 1  # SPY
        assert result.stage_3_builder_evaluation.rule_failure_breakdown.get("R06_PRIMARY_LISTING") == 1  # SEC1
        assert result.stage_3_builder_evaluation.rule_failure_breakdown.get("R04_PRIMARY_EXCHANGE") == 1  # UNSUP1
        assert result.stage_3_builder_evaluation.production_adopted is False
        assert result.stage_3_builder_evaluation.production_published is False
        assert result.stage_3_builder_evaluation.closure_verified is True

        # Precise failed rule predicate checks:
        pred_spy = result.stage_3_builder_evaluation.rule_predicate_details["SPY"]
        assert pred_spy["reason_code"] == "INELIGIBLE_ASSET_CLASS"
        failed_rules_spy = [p["rule"] for p in pred_spy["failed_predicates"]]
        assert "R01_ASSET_CLASS" in failed_rules_spy
        assert "R02_SECURITY_TYPE" in failed_rules_spy

        pred_sec1 = result.stage_3_builder_evaluation.rule_predicate_details["SEC1"]
        assert pred_sec1["reason_code"] == "INELIGIBLE_SECONDARY_LISTING"
        assert pred_sec1["failed_predicates"][0]["rule"] == "R06_PRIMARY_LISTING"

        pred_unsup1 = result.stage_3_builder_evaluation.rule_predicate_details["UNSUP1"]
        assert pred_unsup1["reason_code"] == "INELIGIBLE_EXCHANGE"
        assert pred_unsup1["failed_predicates"][0]["rule"] == "R04_PRIMARY_EXCHANGE"
        assert pred_unsup1["failed_predicates"][0]["actual"] == "AMEX"
        assert result.stage_1_identity_association.associated_records[6]["canonical_exchange"] == "AMEX"

        # Stage 4: Market-Data Readiness Diagnostics
        assert result.stage_4_coverage_diagnostics.status == "COMPLETED"
        assert result.stage_4_coverage_diagnostics.is_diagnostic_only is True
        assert result.stage_4_coverage_diagnostics.distinct_lookup_keys == 7
        assert result.stage_4_coverage_diagnostics.data_ready_count == 4  # AAPL, SPY, SEC1, UNSUP1
        assert result.stage_4_coverage_diagnostics.stale_quote_count == 1  # MSFT
        assert result.stage_4_coverage_diagnostics.insufficient_history_count == 1  # GLX
        assert result.stage_4_coverage_diagnostics.missing_market_data_count == 1  # NOMKT
        assert result.stage_4_coverage_diagnostics.closure_verified is True

        # Overall Outcome
        assert result.overall_outcome == "SUCCESS_BUILD_AND_COVERAGE_VERIFIED"
        assert result.reconciliation_closure["all_stages_closed"] is True

        # Verify output files
        assert os.path.exists(os.path.join(out_dir, "offline_build_coverage_scenario_a_convertible_batch.json"))
        assert os.path.exists(os.path.join(out_dir, "offline_build_coverage_scenario_a_convertible_batch.md"))

    def test_02_atomic_batch_conversion_failure_zero_partial_handoff(
        self, test_market_db: str, tmp_path: Path
    ) -> None:
        """
        Scenario B: Atomic fail-closed batch.
        Verifies:
        - Batch contains 2 valid records (AAPL, MSFT) and 1 invalid record (missing currency).
        - Whole batch fails closed: admission is REVOKED.
        - Converted security count is strictly 0.
        - UniverseBuilder is strictly NOT_RUN (evaluated count = 0).
        - Zero partial conversion and zero partial builder handoff verified.
        """
        candidates = build_scenario_b_candidates()
        assert len(candidates) == 3

        with ReadOnlyMarketStore(test_market_db) as store:
            config = OfflineBuildCoverageConfig(
                candidate_records=candidates,
                as_of_date="2026-10-16",
                market_store=store,
                scenario_label="SCENARIO_B_ATOMIC_FAIL_CLOSED",
                is_synthetic=True,
            )
            result = run_offline_build_and_coverage(config)

        # Stage 2: Whole batch failed closed
        assert result.stage_2_validation_and_conversion.status == "REVOKED_VALIDATION_FAILED"
        assert result.stage_2_validation_and_conversion.component_validation_status == "REVOKED"
        assert result.stage_2_validation_and_conversion.conversion_status == "NOT_RUN"
        assert result.stage_2_validation_and_conversion.is_batch_valid is False
        assert result.stage_2_validation_and_conversion.converted_security_count == 0
        assert "Atomic batch validation failed" in result.stage_2_validation_and_conversion.revocation_reason

        # Stage 3: Universe Builder is strictly NOT_RUN
        assert result.stage_3_builder_evaluation.status == "NOT_RUN"
        assert result.stage_3_builder_evaluation.evaluated_security_count == 0
        assert result.stage_3_builder_evaluation.eligible_count == 0
        assert result.stage_3_builder_evaluation.ineligible_count == 0
        assert result.stage_3_builder_evaluation.production_adopted is False
        assert result.stage_3_builder_evaluation.production_published is False
        assert result.stage_3_builder_evaluation.closure_verified is True

        # Overall Outcome
        assert result.overall_outcome == "BLOCKED_ATOMIC_VALIDATION_FAILED"
        assert result.reconciliation_closure["zero_partial_handoff_verified"] is True

    def test_02b_conversion_boundary_failure_staging_profile_pass_converter_fail(
        self, test_market_db: str, tmp_path: Path
    ) -> None:
        """
        Scenario B2: Missing conversion-boundary failure on unknown primary listing.
        Verifies:
        - Batch satisfies staging validation profile (ValidationRequirements.for_explicit_listing_profile where primary_listing is permitted null).
        - Validator component reports is_batch_valid=True, component_validation_status=ADMITTED.
        - SourceSecurity conversion fails closed because converter requires primary_listing is not None.
        - Converted security count is strictly 0.
        - Stage 2 status is REVOKED_CONVERSION_FAILED with conversion_status=BLOCKED.
        - UniverseBuilder is strictly NOT_RUN (evaluated count = 0).
        - Zero partial conversion and zero partial builder handoff verified.
        """
        candidates = build_scenario_b2_conversion_failure_candidates()
        assert len(candidates) == 3

        with ReadOnlyMarketStore(test_market_db) as store:
            config = OfflineBuildCoverageConfig(
                candidate_records=candidates,
                as_of_date="2026-10-16",
                market_store=store,
                scenario_label="SCENARIO_B2_CONVERSION_FAILURE",
                is_synthetic=True,
            )
            result = run_offline_build_and_coverage(config)

        # Stage 1: Identity resolved
        assert result.stage_1_identity_association.status == "COMPLETED"
        assert result.stage_1_identity_association.total_candidates == 3
        assert result.stage_1_identity_association.resolved_identities_count == 3
        assert result.stage_1_identity_association.blocked_identities_count == 0

        # Stage 2: Staging validation passes, but atomic conversion fails closed
        assert result.stage_2_validation_and_conversion.is_batch_valid is True
        assert result.stage_2_validation_and_conversion.component_validation_status == "ADMITTED"
        assert result.stage_2_validation_and_conversion.conversion_status == "BLOCKED"
        assert result.stage_2_validation_and_conversion.status == "REVOKED_CONVERSION_FAILED"
        assert result.stage_2_validation_and_conversion.converted_security_count == 0
        assert "Atomic conversion boundary failed closed" in result.stage_2_validation_and_conversion.revocation_reason

        # Stage 3: Universe Builder is strictly NOT_RUN
        assert result.stage_3_builder_evaluation.status == "NOT_RUN"
        assert result.stage_3_builder_evaluation.evaluated_security_count == 0
        assert result.stage_3_builder_evaluation.closure_verified is True

        # Overall Outcome
        assert result.overall_outcome == "BLOCKED_CONVERSION_FAILED"
        assert result.reconciliation_closure["zero_partial_handoff_verified"] is True

    def test_02c_unresolved_identity_conflict_blocks_before_conversion(
        self, test_market_db: str, tmp_path: Path
    ) -> None:
        """
        Scenario D: Unresolved identity conflict batch.
        Verifies:
        - Candidate with disputed/unresolved listing association (DISP1) is withheld at Stage 1.
        - Stage 1: 1 resolved identity (AAPL), 1 blocked identity (DISP1).
        - Stage 2: Batch validation fails closed under whole-batch protections.
        - Converted security count is strictly 0.
        - UniverseBuilder is strictly NOT_RUN.
        - Zero partial conversion and zero partial builder handoff verified.
        """
        candidates = build_scenario_d_identity_conflict_candidates()
        assert len(candidates) == 2

        with ReadOnlyMarketStore(test_market_db) as store:
            config = OfflineBuildCoverageConfig(
                candidate_records=candidates,
                as_of_date="2026-10-16",
                market_store=store,
                scenario_label="SCENARIO_D_IDENTITY_CONFLICT",
                is_synthetic=True,
            )
            result = run_offline_build_and_coverage(config)

        # Stage 1: Identity gate blocks unresolved conflict
        assert result.stage_1_identity_association.status == "COMPLETED"
        assert result.stage_1_identity_association.total_candidates == 2
        assert result.stage_1_identity_association.resolved_identities_count == 1
        assert result.stage_1_identity_association.blocked_identities_count == 1
        assert result.stage_1_identity_association.resolved_symbols == ["AAPL"]
        assert result.stage_1_identity_association.blocked_symbols == ["DISP1"]
        assert result.stage_1_identity_association.disputed_association_count == 1
        assert result.stage_1_identity_association.closure_verified is True

        # Stage 2: Batch validation fails closed
        assert result.stage_2_validation_and_conversion.status == "REVOKED_VALIDATION_FAILED"
        assert result.stage_2_validation_and_conversion.component_validation_status == "REVOKED"
        assert result.stage_2_validation_and_conversion.conversion_status == "NOT_RUN"
        assert result.stage_2_validation_and_conversion.is_batch_valid is False
        assert result.stage_2_validation_and_conversion.converted_security_count == 0

        # Stage 3: Universe Builder is strictly NOT_RUN
        assert result.stage_3_builder_evaluation.status == "NOT_RUN"
        assert result.stage_3_builder_evaluation.evaluated_security_count == 0
        assert result.stage_3_builder_evaluation.closure_verified is True

        # Overall Outcome
        assert result.overall_outcome == "BLOCKED_IDENTITY_CONFLICT"
        assert result.reconciliation_closure["zero_partial_handoff_verified"] is True

    def test_03_no_network_enforcement_network_blocked(
        self, test_market_db: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Verifies that the entire offline pipeline executes with zero network calls.
        Monkeypatches socket.socket to raise an exception if any network connection is attempted.
        """
        def blocked_socket(*args, **kwargs):
            raise RuntimeError("NETWORK_ACCESS_PROHIBITED: Offline universe pipeline must not make network calls.")

        monkeypatch.setattr(socket, "socket", blocked_socket)

        candidates = build_scenario_a_candidates()
        with ReadOnlyMarketStore(test_market_db) as store:
            config = OfflineBuildCoverageConfig(
                candidate_records=candidates,
                as_of_date="2026-10-16",
                market_store=store,
                scenario_label="SCENARIO_A_NETWORK_GUARD",
                is_synthetic=True,
            )
            # Must succeed without triggering socket.socket
            result = run_offline_build_and_coverage(config)

        assert result.overall_outcome == "SUCCESS_BUILD_AND_COVERAGE_VERIFIED"

    def test_04_scenario_c_real_source_retained_evidence_replay(
        self, test_market_db: str
    ) -> None:
        """
        Scenario C: Real-source retained evidence replay.
        Verifies:
        - 7 representative real candidates (AAPL, GOOGL, GOOG, TSM, ACCS, DDS, AACIU).
        - Currency and primary_listing remain unevidenced (None) in raw directory feeds.
        - Without synthetic fabrication, batch fails closed truthfully (0 admitted).
        - UniverseBuilder is NOT_RUN (0 evaluated).
        - Venue disagreements (DDS: TXSE vs F, ACCS: NYSE vs AMEX) preserved.
        - Multi-symbol issuer (GOOG/GOOGL under CIK 0001652044) tracked in Stage 1.
        """
        candidates = build_scenario_c_candidates()
        assert len(candidates) == 7

        with ReadOnlyMarketStore(test_market_db) as store:
            config = OfflineBuildCoverageConfig(
                candidate_records=candidates,
                as_of_date="2026-10-16",
                market_store=store,
                scenario_label="SCENARIO_C_REAL_SOURCE_REPLAY",
                is_synthetic=False,
            )
            result = run_offline_build_and_coverage(config)

        # Stage 1: Identity Association
        assert result.stage_1_identity_association.status == "COMPLETED"
        assert result.stage_1_identity_association.total_candidates == 7
        assert result.stage_1_identity_association.distinct_symbols == 7
        assert result.stage_1_identity_association.resolved_identities_count == 0  # 0 resolved security identities
        assert result.stage_1_identity_association.blocked_identities_count == 1   # 1 blocked (ACCS venue disagreement)
        assert result.stage_1_identity_association.unavailable_identities_count == 6  # 6 unavailable identities
        assert result.stage_1_identity_association.issuer_associated_count == 7   # 7 issuer associated via CIK
        assert result.stage_1_identity_association.provisional_mapping_count == 7  # 7 provisional mapping available
        assert result.stage_1_identity_association.resolved_symbols == []
        assert set(result.stage_1_identity_association.blocked_symbols) == {"ACCS"}
        assert set(result.stage_1_identity_association.unavailable_symbols) == {"AAPL", "GOOGL", "GOOG", "TSM", "DDS", "AACIU"}
        assert result.stage_1_identity_association.multi_symbol_cik_count >= 1  # CIK 0001652044 (GOOG, GOOGL)
        assert result.stage_1_identity_association.venue_normalized_count >= 1  # DDS (code F -> TXSE concordant mapping)
        assert result.stage_1_identity_association.disputed_association_count == 1  # ACCS (genuine disagreement)
        assert result.stage_1_identity_association.closure_verified is True  # 7 == 0 + 1 + 6

        # Stage 2: Batch Validation fails closed without fabricated currency/identity
        assert result.stage_2_validation_and_conversion.status == "REVOKED_VALIDATION_FAILED"
        assert result.stage_2_validation_and_conversion.component_validation_status == "REVOKED"
        assert result.stage_2_validation_and_conversion.conversion_status == "NOT_RUN"
        assert result.stage_2_validation_and_conversion.is_batch_valid is False
        assert result.stage_2_validation_and_conversion.converted_security_count == 0

        # Stage 3: Builder is NOT_RUN
        assert result.stage_3_builder_evaluation.status == "NOT_RUN"
        assert result.stage_3_builder_evaluation.evaluated_security_count == 0
        assert result.stage_3_builder_evaluation.closure_verified is True

        # Overall Outcome
        assert result.overall_outcome == "BLOCKED_ATOMIC_VALIDATION_FAILED"
        assert result.reconciliation_closure["zero_partial_handoff_verified"] is True

    def test_05_diagnostic_reporting_accuracy_and_reconciliation(
        self, test_market_db: str
    ) -> None:
        """
        Verifies diagnostic precedence order in CoverageDiagnosticEngine:
        MISSING_MARKET_DATA > INSUFFICIENT_HISTORY > INVALID_OHLCV > STALE_QUOTE > DATA_READY.
        """
        test_records = [
            OfflineCandidateRecord(
                record_locator="T#1", symbol="AAPL", exchange="NASDAQ", is_synthetic=True
            ),
            OfflineCandidateRecord(
                record_locator="T#2", symbol="MSFT", exchange="NASDAQ", is_synthetic=True
            ),
            OfflineCandidateRecord(
                record_locator="T#3", symbol="GLX", exchange="NASDAQ", is_synthetic=True
            ),
            OfflineCandidateRecord(
                record_locator="T#4", symbol="NOMKT", exchange="NASDAQ", is_synthetic=True
            ),
            OfflineCandidateRecord(
                record_locator="T#5", symbol="INV_OHLCV", exchange="NASDAQ", is_synthetic=True
            ),
        ]

        with ReadOnlyMarketStore(test_market_db) as store:
            config = OfflineBuildCoverageConfig(
                candidate_records=test_records,
                as_of_date="2026-10-16",
                market_store=store,
                scenario_label="SCENARIO_PRECEDENCE_CHECK",
                is_synthetic=True,
            )
            result = run_offline_build_and_coverage(config)

        diag = result.stage_4_coverage_diagnostics
        assert diag.status == "COMPLETED"
        assert diag.distinct_lookup_keys == 5
        assert diag.data_ready_count == 1  # AAPL
        assert diag.stale_quote_count == 1  # MSFT
        assert diag.insufficient_history_count == 1  # GLX
        assert diag.missing_market_data_count == 1  # NOMKT
        assert diag.invalid_ohlcv_count == 1  # INV_OHLCV
        assert diag.closure_verified is True

    def test_06_arithmetic_closure_cross_stage_reconciliation(
        self, test_market_db: str
    ) -> None:
        """
        Verifies exact arithmetic closure across all pipeline stages:
        - Candidate records count == Stage 1 total candidates.
        - Stage 2 total == passed + failed + unresolved.
        - Stage 3 evaluated == eligible + ineligible + unresolved.
        - Stage 4 distinct keys == sum of disjoint primary outcomes.
        """
        candidates = build_scenario_a_candidates()
        with ReadOnlyMarketStore(test_market_db) as store:
            config = OfflineBuildCoverageConfig(
                candidate_records=candidates,
                as_of_date="2026-10-16",
                market_store=store,
                scenario_label="SCENARIO_CLOSURE_TEST",
                is_synthetic=True,
            )
            result = run_offline_build_and_coverage(config)

        rec = result.reconciliation_closure
        assert rec["candidate_input_count"] == 7
        assert rec["distinct_candidate_symbols"] == 7
        assert rec["stage_1_closure_verified"] is True
        assert rec["stage_2_closure_verified"] is True
        assert rec["stage_3_closure_verified"] is True
        assert rec["stage_4_closure_verified"] is True
        assert rec["all_stages_closed"] is True

    def test_07_read_safety_zero_database_mutation(
        self, test_market_db: str
    ) -> None:
        """
        Verifies strict read-safety: ReadOnlyMarketStore cannot write to or mutate SQLite file.
        Checks pre- and post-run file size and mtime.
        """
        stat_before = os.stat(test_market_db)
        size_before = stat_before.st_size
        mtime_before = stat_before.st_mtime

        candidates = build_scenario_a_candidates()
        with ReadOnlyMarketStore(test_market_db) as store:
            config = OfflineBuildCoverageConfig(
                candidate_records=candidates,
                as_of_date="2026-10-16",
                market_store=store,
                scenario_label="SCENARIO_READ_SAFETY",
                is_synthetic=True,
            )
            _ = run_offline_build_and_coverage(config)

        stat_after = os.stat(test_market_db)
        assert stat_after.st_size == size_before, "Database file size changed during read-only operation!"
        assert stat_after.st_mtime == mtime_before, "Database mtime changed during read-only operation!"

    def test_08_cli_runner_and_artifact_generation(
        self, test_market_db: str, tmp_path: Path
    ) -> None:
        """
        Verifies that scripts/diagnostics/universe_offline_build_coverage.py CLI runner
        executes successfully and writes structured JSON and Markdown artifacts.
        """
        cli_script = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scripts",
            "diagnostics",
            "universe_offline_build_coverage.py",
        )
        out_dir = str(tmp_path / "cli_out")

        cmd = [
            sys.executable,
            cli_script,
            "--scenario", "all",
            "--market-db", test_market_db,
            "--output-dir", out_dir,
            "--as-of", "2026-10-16",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, f"CLI runner failed: {proc.stderr}\nStdout: {proc.stdout}"

        # Verify output artifacts for all five scenarios
        expected_scenarios = [
            "scenario_a_convertible_batch",
            "scenario_b_atomic_fail_closed",
            "scenario_b2_conversion_failure",
            "scenario_d_identity_conflict",
            "scenario_c_real_source_replay",
        ]
        for scen in expected_scenarios:
            json_file = os.path.join(out_dir, f"offline_build_coverage_{scen}.json")
            md_file = os.path.join(out_dir, f"offline_build_coverage_{scen}.md")
            assert os.path.exists(json_file), f"Missing JSON artifact: {json_file}"
            assert os.path.exists(md_file), f"Missing Markdown artifact: {md_file}"

        # Verify Markdown content contains stop point and invariants
        with open(os.path.join(out_dir, "offline_build_coverage_scenario_a_convertible_batch.md"), "r", encoding="utf-8") as fp:
            md_content = fp.read()
        assert "OFFLINE_BUILD_COVERAGE_INTEGRATION_LOCAL_VERIFIED" in md_content
        assert "ZERO_DATABASE_WRITES = ENFORCED" in md_content

        # Verify dynamic denominator in B2 / B / C / D when Stage 3 is NOT_RUN
        with open(os.path.join(out_dir, "offline_build_coverage_scenario_b2_conversion_failure.md"), "r", encoding="utf-8") as fp:
            b2_md = fp.read()
        assert "0 == 0 + 0 + 0 (NOT_RUN)" in b2_md

    def test_09_z_directory_code_normalizes_to_bats_and_passes_r04(
        self, test_market_db: str
    ) -> None:
        """
        Regression test: Raw directory exchange code 'Z' normalizes via SupportedVenueNormalizer
        to canonical exchange 'BATS' (operating MIC 'BATS').
        Verifies:
        - Stage 1: Venue normalized count incremented, raw exchange 'Z' and canonical 'BATS' preserved.
        - Stage 2: Validated record and SourceSecurity receive canonical exchange 'BATS'.
        - Stage 3: DeterministicUniverseBuilder evaluates Rule R04 (PRIMARY_EXCHANGE) against 'BATS'.
        - Candidate passes R04 and is ELIGIBLE; does not fail R04 solely because source used directory code 'Z'.
        """
        z_candidate = OfflineCandidateRecord(
            record_locator="REG_Z#1_BATS_NORMALIZED",
            symbol="ZBATS",
            exchange="Z",  # Raw directory code for BATS
            security_type="COMMON_STOCK",
            listing_status="ACTIVE",
            currency="USD",
            country="USA",
            primary_listing=True,
            asset_class="EQUITY",
            security_id="SEC_ZBATS_001",
            issuer_cik="0001999999",
            is_synthetic=True,
            notes="Regression test for directory exchange code 'Z' normalisation to BATS.",
        )

        with ReadOnlyMarketStore(test_market_db) as store:
            config = OfflineBuildCoverageConfig(
                candidate_records=[z_candidate],
                as_of_date="2026-10-16",
                market_store=store,
                scenario_label="SCENARIO_REGRESSION_Z_TO_BATS",
                is_synthetic=True,
            )
            result = run_offline_build_and_coverage(config)

        # Stage 1: Normalization verified
        s1 = result.stage_1_identity_association
        assert s1.status == "COMPLETED"
        assert s1.venue_normalized_count == 1
        assoc = s1.associated_records[0]
        assert assoc["raw_exchange"] == "Z"
        assert assoc["canonical_exchange"] == "BATS"
        assert assoc["operating_mic"] == "BATS"
        assert "Exchange code 'Z' = BATS" in assoc["normalization_provenance"]

        # Stage 2: Validator and SourceSecurity received canonical exchange
        s2 = result.stage_2_validation_and_conversion
        assert s2.is_batch_valid is True
        assert s2.conversion_status == "SUCCEEDED"
        assert s2.converted_security_count == 1

        # Stage 3: Builder evaluated and R04 passed
        s3 = result.stage_3_builder_evaluation
        assert s3.status == "EVALUATED"
        assert s3.evaluated_security_count == 1
        assert s3.eligible_count == 1
        assert s3.ineligible_count == 0
        assert s3.eligible_symbols == ["ZBATS"]
        assert "R04_PRIMARY_EXCHANGE" not in s3.rule_failure_breakdown
