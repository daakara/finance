"""Unit and integration test suite for Directive ARX-LEARNING-004 corrections.

Verifies:
1. Historical census reports UNAVAILABLE on database failure with error and source identity;
   never returns hardcoded fallback constants (16,093 / 2,727).
2. Market data authority: Empty market data bars classify observations as unresolved and
   report INELIGIBLE_FOR_REAL_SETTLEMENT; authoritative price bars qualify evaluations.
3. API authorization: /api/v1/governance/prospective-learning/census requires valid
   X-Evaluation-Key, X-Admin-Key, or Bearer credential and returns HTTP 401 otherwise.
4. Outcome immutability: Idempotent identical replays succeed; unresolvable mutations of
   terminal settled outcomes are blocked; authorized corrections append to audit trail.
"""

import os
import json
import sqlite3
import tempfile
import pytest
import pandas as pd
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from api.main import app
from analyst_dashboard.governance.prospective_capture import (
    ProspectiveDecisionCaptureEngine,
    UNRESOLVED_OUTCOME_STATUSES,
    TERMINAL_SETTLED_STATUSES,
)
from analyst_dashboard.governance.shadow_replay_runner import ShadowReplayRunner
from scripts.replay_prospective_settlement import load_authoritative_market_data


@pytest.fixture
def capture_env(tmp_path):
    db_file = str(tmp_path / "gov_test.db")
    stream_file = str(tmp_path / "stream_test.jsonl")
    engine = ProspectiveDecisionCaptureEngine(
        db_path=db_file,
        stream_path=stream_file,
        fail_open_client=False,
    )
    return engine, db_file, stream_file


class TestProspectiveLearningCorrections:
    """Test suite strictly enforcing Directive ARX-LEARNING-004."""

    def test_correction_1_historical_census_fails_closed_without_constants(self, capture_env, tmp_path):
        """Historical census reports UNAVAILABLE on database failure and never returns 16,093 or 2,727."""
        engine, _, _ = capture_env

        # 1. Simulate corrupted / unreadable historical database
        corrupt_db = str(tmp_path / "corrupt_history.db")
        with open(corrupt_db, "w") as f:
            f.write("CORRUPTED_NON_SQLITE_PAYLOAD")

        census = engine.get_prospective_learning_census(history_db_path=corrupt_db)

        # Must report UNAVAILABLE with explicit error and source identity
        assert census["status"] == "UNAVAILABLE"
        assert "error" in census
        assert census["sourceIdentity"] == corrupt_db

        recon = census["historicalCensusReconciliation"]
        assert recon["status"] == "UNAVAILABLE"
        assert recon["globalDatabaseCensus"]["totalLoggedSetups"] is None
        assert recon["globalDatabaseCensus"]["resolvedCount"] is None
        assert recon["symbolSpecificCensus"]["AAPL"]["totalLoggedSetups"] is None

        # Critical: Hardcoded fallback constants must NEVER be returned
        assert recon["globalDatabaseCensus"]["totalLoggedSetups"] != 16093
        assert recon["symbolSpecificCensus"]["AAPL"]["totalLoggedSetups"] != 2727

    def test_correction_1_historical_census_reconciles_real_database(self, capture_env, tmp_path):
        """Historical census successfully reconciles real records when database is available."""
        engine, _, _ = capture_env
        valid_db = str(tmp_path / "authoritative_history.db")

        from analyst_dashboard.data.db_engine import HistoryDatabaseEngine
        hdb = HistoryDatabaseEngine(db_path=valid_db)
        for _ in range(7):
            hdb.log_trade_recommendation("AAPL", {"current_price": 220.0})
        for _ in range(4):
            hdb.log_trade_recommendation("MSFT", {"current_price": 410.0})

        conn = sqlite3.connect(valid_db)
        with conn:
            # Mark 2 AAPL as resolved (1 SUCCESS, 1 STOPPED_OUT)
            conn.execute("UPDATE trade_recommendation_history SET outcome_status = 'SUCCESS' WHERE id = 1")
            conn.execute("UPDATE trade_recommendation_history SET outcome_status = 'STOPPED_OUT' WHERE id = 2")
        conn.close()

        census = engine.get_prospective_learning_census(history_db_path=valid_db)
        assert census["status"] == "SUCCESS"
        recon = census["historicalCensusReconciliation"]
        assert recon["status"] == "AVAILABLE"
        assert recon["globalDatabaseCensus"]["totalLoggedSetups"] == 11
        assert recon["globalDatabaseCensus"]["resolvedCount"] == 2
        assert recon["globalDatabaseCensus"]["pendingCount"] == 9
        assert recon["symbolSpecificCensus"]["AAPL"]["totalLoggedSetups"] == 7
        assert recon["symbolSpecificCensus"]["AAPL"]["resolvedCount"] == 2

    def test_correction_2_market_data_authority_empty_bars_unresolved(self, capture_env):
        """Replay runner with empty market data bars classifies decisions as unresolved and ineligible."""
        engine, _, _ = capture_env
        cycle = engine.create_or_get_evaluation_cycle("EXPLICIT_USER_REQUEST", "2026.1")

        dec = engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="AAPL",
            instrument_id="AAPL",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=85.0,
            rejection_reason={"isRejected": False},
            trade_plan_state={
                "corridorMin": 220.0,
                "corridorMax": 222.0,
                "stopLoss": 215.0,
                "takeProfit1": 235.0,
                "confluenceScore": 85.0,
            },
            evaluation_started_at_utc="2026-11-01T14:30:00Z",
        )

        runner = ShadowReplayRunner(capture_engine=engine)
        # Empty market map
        batch_result = runner.replay_captured_decisions_batch(
            decisions=[{
                "decision_id": dec["decision_id"],
                "symbol": "AAPL",
                "episode_id": dec["episode_id"],
                "trade_plan_json": {
                    "corridorMin": 220.0,
                    "corridorMax": 222.0,
                    "stopLoss": 215.0,
                    "takeProfit1": 235.0,
                },
                "evaluation_started_at_utc": "2026-11-01T14:30:00Z",
            }],
            market_data_map={},
            persist=True,
        )

        assert batch_result["total_evaluated"] == 1
        assert batch_result["status_counts"]["OUTCOME_DATA_MISSING"] == 1
        assert batch_result["resolved_count"] == 0
        assert batch_result["unresolved_count"] == 1
        assert "UNRESOLVED_ONLY" in batch_result["market_data_authority"]
        assert batch_result["settlement_eligibility"] == "INELIGIBLE_FOR_REAL_SETTLEMENT"
        assert batch_result["settlement_qualification"] == "UNQUALIFIED"

    def test_correction_2_market_data_authority_authoritative_bars_qualified(self, capture_env, tmp_path):
        """Authoritative market data file correctly parsed and qualifies evaluation."""
        engine, _, _ = capture_env

        # Create valid market data JSON file
        bars_data = {
            "AAPL": [
                {"date": "2026-11-02", "open": 221.0, "high": 223.0, "low": 220.5, "close": 222.0, "volume": 1000000},
                {"date": "2026-11-03", "open": 222.5, "high": 236.0, "low": 222.0, "close": 235.5, "volume": 1200000},
            ]
        }
        data_file = str(tmp_path / "aapl_bars.json")
        with open(data_file, "w") as f:
            json.dump(bars_data, f)

        market_map, file_hash = load_authoritative_market_data(data_file)
        assert "AAPL" in market_map
        assert len(file_hash) == 64

        cycle = engine.create_or_get_evaluation_cycle("EXPLICIT_USER_REQUEST", "2026.1")
        dec = engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="AAPL",
            instrument_id="AAPL",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=90.0,
            rejection_reason={"isRejected": False},
            trade_plan_state={
                "corridorMin": 220.0,
                "corridorMax": 222.0,
                "stopLoss": 215.0,
                "takeProfit1": 235.0,
            },
            evaluation_started_at_utc="2026-11-01T14:30:00Z",
        )

        runner = ShadowReplayRunner(capture_engine=engine)
        batch_result = runner.replay_captured_decisions_batch(
            decisions=[{
                "decision_id": dec["decision_id"],
                "symbol": "AAPL",
                "episode_id": dec["episode_id"],
                "trade_plan_json": {
                    "corridorMin": 220.0,
                    "corridorMax": 222.0,
                    "stopLoss": 215.0,
                    "takeProfit1": 235.0,
                },
                "evaluation_started_at_utc": "2026-11-01T14:30:00Z",
            }],
            market_data_map=market_map,
            persist=True,
        )

        assert batch_result["total_evaluated"] == 1
        assert batch_result["status_counts"]["SUCCESS"] == 1
        assert batch_result["resolved_count"] == 1
        assert batch_result["market_data_authority"] == "AUTHORITATIVE_BARS_SUPPLIED"
        assert batch_result["settlement_eligibility"] == "QUALIFIED_FOR_EVALUATION"
        assert batch_result["settlement_qualification"] == "OUTCOME_SETTLEMENT_QUALIFIED"

    def test_correction_3_api_authorization_prospective_census(self, monkeypatch):
        """Endpoint /api/v1/governance/prospective-learning/census rejects unauthenticated calls and accepts valid credentials."""
        client = TestClient(app)
        monkeypatch.setenv("ARX_EVALUATION_KEY", "test-eval-secret-key-123")
        monkeypatch.setenv("ARX_ADMIN_KEY", "test-admin-secret-key-456")

        # 1. Unauthenticated request -> HTTP 401
        res_unauth = client.get("/api/v1/governance/prospective-learning/census")
        assert res_unauth.status_code == 401
        assert "Unauthorized" in res_unauth.json()["detail"]

        # 2. Invalid credential -> HTTP 401
        res_bad = client.get(
            "/api/v1/governance/prospective-learning/census",
            headers={"X-Evaluation-Key": "wrong-key"},
        )
        assert res_bad.status_code == 401

        # 3. Valid X-Evaluation-Key -> HTTP 200
        res_eval = client.get(
            "/api/v1/governance/prospective-learning/census",
            headers={"X-Evaluation-Key": "test-eval-secret-key-123"},
        )
        assert res_eval.status_code == 200
        assert "prospectiveCapture" in res_eval.json()

        # 4. Valid X-Admin-Key -> HTTP 200
        res_admin = client.get(
            "/api/v1/governance/prospective-learning/census",
            headers={"X-Admin-Key": "test-admin-secret-key-456"},
        )
        assert res_admin.status_code == 200
        assert "prospectiveCapture" in res_admin.json()

        # 5. Valid Authorization Bearer -> HTTP 200
        res_bearer = client.get(
            "/api/v1/governance/prospective-learning/census",
            headers={"Authorization": "Bearer test-eval-secret-key-123"},
        )
        assert res_bearer.status_code == 200

    def test_correction_4_outcome_immutability_and_corrections_audit(self, capture_env):
        """Outcome links enforce idempotent replay, block silent overwrite of settled states, and log authorized corrections."""
        engine, db_file, stream_file = capture_env
        cycle = engine.create_or_get_evaluation_cycle("EXPLICIT_USER_REQUEST", "2026.1")

        dec = engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="MSFT",
            instrument_id="MSFT",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=87.0,
            rejection_reason={"isRejected": False},
        )
        dec_id = dec["decision_id"]
        ep_id = dec["episode_id"]

        # Step A: Record initial unresolved state (PENDING_UNMATURED)
        link1 = engine.record_outcome_link(
            decision_id=dec_id,
            episode_id=ep_id,
            outcome_status="PENDING_UNMATURED",
        )
        assert link1["outcome_status"] == "PENDING_UNMATURED"

        # Step B: Valid forward progression: PENDING_UNMATURED -> SUCCESS
        link2 = engine.record_outcome_link(
            decision_id=dec_id,
            episode_id=ep_id,
            outcome_status="SUCCESS",
            realized_gross_return_pct=6.5,
            realized_net_simulated_return_pct=6.25,
            realized_r_multiple=2.0,
        )
        assert link2["outcome_status"] == "SUCCESS"
        assert link2["realized_r_multiple"] == 2.0

        # Step C: Idempotent replay: calling with identical status and metrics succeeds without mutation
        link_idemp = engine.record_outcome_link(
            decision_id=dec_id,
            episode_id=ep_id,
            outcome_status="SUCCESS",
            realized_gross_return_pct=6.5,
            realized_net_simulated_return_pct=6.25,
            realized_r_multiple=2.0,
        )
        assert link_idemp["idempotent_replay"] is True
        assert link_idemp["outcome_status"] == "SUCCESS"

        # Step D: Block arbitrary silent overwrite of settled outcome without authorization
        with pytest.raises(ValueError) as excinfo:
            engine.record_outcome_link(
                decision_id=dec_id,
                episode_id=ep_id,
                outcome_status="FAILURE",
                realized_gross_return_pct=-3.0,
                realized_r_multiple=-1.0,
            )
        assert "MUTATION_OF_SETTLED_OUTCOME_PROHIBITED" in str(excinfo.value)

        # Step E: Authorized correction with explicit reason and author
        link_corr = engine.record_outcome_link(
            decision_id=dec_id,
            episode_id=ep_id,
            outcome_status="FAILURE",
            realized_gross_return_pct=-3.0,
            realized_net_simulated_return_pct=-3.25,
            realized_r_multiple=-1.0,
            correction_reason="Post-trade reconciliation identified earlier exit execution",
            authorized_by="HEAD_OF_GOVERNANCE_CHIEF_RISK_OFFICER",
        )
        assert link_corr["outcome_status"] == "FAILURE"
        assert "correction_event_id" in link_corr
        assert link_corr["correction_reason"] == "Post-trade reconciliation identified earlier exit execution"

        # Verify entry in prospective_decision_corrections table
        conn = engine.db.get_connection()
        try:
            cur = conn.execute(
                "SELECT * FROM prospective_decision_corrections WHERE original_decision_id = ?",
                (dec_id,),
            )
            corr_row = cur.fetchone()
            assert corr_row is not None
            assert corr_row["corrected_field_name"] == "outcome_status"
            assert corr_row["authorized_by"] == "HEAD_OF_GOVERNANCE_CHIEF_RISK_OFFICER"
            orig_vals = json.loads(corr_row["original_value_json"])
            assert orig_vals["outcome_status"] == "SUCCESS"
            corr_vals = json.loads(corr_row["corrected_value_json"])
            assert corr_vals["outcome_status"] == "FAILURE"
        finally:
            conn.close()

        # Verify recovery stream recorded both transition and correction events
        with open(stream_file, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        event_types = [l["record_type"] for l in lines]
        assert "OUTCOME_LINK_RECORDED" in event_types
        assert "OUTCOME_LINK_TRANSITIONED" in event_types
        assert "OUTCOME_CORRECTION_RECORDED" in event_types

    def test_correction_5_cli_safe_replay_defaults_and_persistence_qualification(self, capture_env, tmp_path, monkeypatch):
        """Settlement CLI defaults to non-persisting dry-run and rejects persistence without authoritative market data."""
        engine, db_file, _ = capture_env
        cycle = engine.create_or_get_evaluation_cycle("EXPLICIT_USER_REQUEST", "2026.1")
        dec = engine.record_decision_event(
            evaluation_cycle_id=cycle["evaluation_cycle_id"],
            symbol="NVDA",
            instrument_id="NVDA",
            cycle_type="EXPLICIT_USER_REQUEST",
            population_sampling_mode="USER_SELECTED",
            evidence_origin="NATURAL_PRODUCTION",
            scope_status="IN_SCOPE",
            decision_state="ACTIONABLE_SETUP",
            actionability_state="ACTIONABLE",
            confluence_score=88.0,
            rejection_reason={"isRejected": False},
            trade_plan_state={
                "corridorMin": 130.0,
                "corridorMax": 132.0,
                "stopLoss": 125.0,
                "takeProfit1": 145.0,
            },
            evaluation_started_at_utc="2026-11-01T14:30:00Z",
        )

        import io
        import sys
        from contextlib import redirect_stdout
        from scripts.replay_prospective_settlement import main as cli_main

        # 1. Default CLI execution (no flags passed) -> MUST be dry-run, zero persisted links
        monkeypatch.setattr(sys, "argv", ["replay_prospective_settlement.py", "--db-path", db_file])
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_main()
        out1 = json.loads(buf.getvalue().strip())
        assert out1["dry_run"] is True
        assert out1["persistence_committed"] is False
        assert out1["settlement_qualification"] == "UNQUALIFIED"

        conn = engine.db.get_connection()
        try:
            cnt1 = conn.execute("SELECT COUNT(*) as c FROM prospective_outcome_links").fetchone()["c"]
            assert cnt1 == 0, f"Expected 0 persisted links under default CLI dry-run, found {cnt1}"
        finally:
            conn.close()

        # 2. Persistence requested without authoritative market data -> MUST revert to dry-run
        monkeypatch.setattr(sys, "argv", ["replay_prospective_settlement.py", "--db-path", db_file, "--persist"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_main()
        out2 = json.loads(buf.getvalue().strip())
        assert out2["dry_run"] is True
        assert out2["persistence_committed"] is False
        assert out2["settlement_qualification"] == "UNQUALIFIED"

        conn = engine.db.get_connection()
        try:
            cnt2 = conn.execute("SELECT COUNT(*) as c FROM prospective_outcome_links").fetchone()["c"]
            assert cnt2 == 0, f"Expected 0 persisted links when persistence requested without data, found {cnt2}"
        finally:
            conn.close()

        # 3. Persistence requested WITH authoritative market data -> MUST persist
        bars_data = {
            "NVDA": [
                {"date": "2026-11-02", "open": 131.0, "high": 133.0, "low": 130.5, "close": 132.0, "volume": 1000000},
                {"date": "2026-11-03", "open": 132.5, "high": 146.0, "low": 132.0, "close": 145.5, "volume": 1200000},
            ]
        }
        data_file = str(tmp_path / "nvda_bars.json")
        with open(data_file, "w") as f:
            json.dump(bars_data, f)

        monkeypatch.setattr(
            sys, "argv",
            ["replay_prospective_settlement.py", "--db-path", db_file, "--market-data-file", data_file, "--persist"]
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_main()
        out3 = json.loads(buf.getvalue().strip())
        assert out3["dry_run"] is False
        assert out3["persistence_committed"] is True
        assert out3["settlement_qualification"] == "OUTCOME_SETTLEMENT_QUALIFIED"
        assert out3["status_counts"]["SUCCESS"] == 1

        conn = engine.db.get_connection()
        try:
            cnt3 = conn.execute("SELECT COUNT(*) as c FROM prospective_outcome_links").fetchone()["c"]
            assert cnt3 == 1, f"Expected 1 persisted link when persistence authorized and data supplied, found {cnt3}"
        finally:
            conn.close()

    def test_correction_6_api_census_internal_error_sanitization(self, monkeypatch):
        """Census API route sanitizes internal errors and never leaks raw exception details."""
        client = TestClient(app)
        monkeypatch.setenv("ARX_EVALUATION_KEY", "test-eval-secret-key-123")

        # Monkeypatch ProspectiveDecisionCaptureEngine to raise a simulated sensitive database exception
        def _failing_census(self):
            raise sqlite3.OperationalError("database is locked: /private/internal/db/credentials_secret.db")

        from analyst_dashboard.governance.prospective_capture import ProspectiveDecisionCaptureEngine
        monkeypatch.setattr(ProspectiveDecisionCaptureEngine, "get_prospective_learning_census", _failing_census)

        res = client.get(
            "/api/v1/governance/prospective-learning/census",
            headers={"X-Evaluation-Key": "test-eval-secret-key-123"},
        )
        assert res.status_code == 200
        payload = res.json()
        assert payload["status"] == "ERROR"
        assert "CENSUS_GENERATION_FAILED" in payload["error"]
        # Raw internal trace / credentials path must NOT be leaked
        assert "/private/internal/db/credentials_secret.db" not in payload["error"]
        assert "database is locked" not in payload["error"]
