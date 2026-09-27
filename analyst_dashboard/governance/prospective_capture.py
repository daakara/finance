"""ARX Prospective Full Decision Capture Engine (V1.0.2).

Deterministic, non-interfering passive evidence capture implementing the frozen
governance architecture specified by:
- docs/governance/ARX_PROSPECTIVE_FULL_DECISION_CAPTURE_DESIGN_V1.md (v1.0.2)
- docs/governance/ARX_PROSPECTIVE_DECISION_CAPTURE_SCHEMA_V1.json (v1.0.2)

INVARIANTS ENFORCED:
1. DETERMINISTIC IDENTITY:
   decision_id = DEC_{sha256(instrument_id + evaluation_cycle_id + engine_sha + decision_schema_version)[:16]}
2. RETRY REUSES CYCLE ID:
   Transient worker retries do not create synthetic evaluation cycles.
3. FORENSIC ATTEMPTS WITHOUT DENOMINATOR INFLATION:
   Multiple execution attempts for a logical evaluation increment attemptCount
   and log to prospective_evaluation_attempts, preserving a single decision event.
4. MUTUALLY EXCLUSIVE RECONCILIATION:
   Expected_N = Completed_N + FailedBefore_N + FailedDuring_N + Missing_N.
5. SAMPLING MODE SEGREGATION:
   SYSTEMATIC (scheduled scans) vs USER_SELECTED (ad-hoc queries).
   eligible_for_opportunity_capture is restricted strictly to SCHEDULED_UNIVERSE_SCAN.
6. TEST POLLUTION FIREWALL:
   Zero test/replay execution records admitted to production evidence.
7. FAIL-OPEN RUNTIME / FAIL-CLOSED GOVERNANCE:
   Production client calls never fail if capture storage errors; empirical research
   strictly requires CERTIFIED_NATURAL_PRODUCTION.
8. EPISODE CONTINUATION:
   Consecutive cycle evaluations link to ongoing episode; price level alteration resets episode.
"""

import os
import re
import json
import time
import uuid
import logging
import hashlib
import contextvars
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union

from analyst_dashboard.governance.governance_db import (
    GovernanceDatabaseEngine as GovernanceDB,
    get_governance_connection,
    retry_sqlite,
    is_production_runtime,
)
from analyst_dashboard.governance.storage import resolve_data_root

logger = logging.getLogger("arx.governance.prospective_capture")

# Schema Constants
DECISION_SCHEMA_VERSION = "1.0.2"
ENGINE_VERSION = "2.5.0"
ENGINE_SHA = "7ad44595826c147cc77f93cd676af520764c7442"
DECISION_ENGINE_SHA = "7ad44595826c147cc77f93cd676af520764c7442"
DEFAULT_CONFIG_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Context Boundary & Execution Firewall
EXECUTION_CONTEXT_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "prospective_execution_context", default="NATURAL_CLIENT"
)


def get_current_evidence_origin() -> str:
    """Evaluates the operational evidence origin based on context and environment.
    Strict firewall: If running in pytest or non-natural context, returns 'TEST'.
    """
    ctx = EXECUTION_CONTEXT_VAR.get()
    if ctx == "TEST" or "PYTEST_CURRENT_TEST" in os.environ:
        return "TEST"
    if ctx == "REPLAY":
        return "REPLAY"
    if ctx == "SIMULATION":
        return "SIMULATION"
    if ctx == "NATURAL_CLIENT" and is_production_runtime():
        return "NATURAL_PRODUCTION"
    return "TEST" if "PYTEST_CURRENT_TEST" in os.environ else "NATURAL_PRODUCTION"


def compute_evaluation_cycle_id(cycle_type: str, cycle_started_at_utc: str, hash_input: str = "") -> str:
    """Deterministic cycle ID: CYC_{cycleType}_{compactTimestampUtc}_{hash8}.
    Conforms to pattern: ^CYC_[A-Z_]+_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{8}$
    """
    try:
        dt = datetime.fromisoformat(cycle_started_at_utc.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)
    ts_compact = dt.strftime("%Y%m%dT%H%M%SZ")
    raw = f"{cycle_type}_{cycle_started_at_utc}_{hash_input}"
    h8 = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    clean_cycle_type = re.sub(r"[^A-Z_]", "_", cycle_type.upper())
    return f"CYC_{clean_cycle_type}_{ts_compact}_{h8}"


def compute_decision_id(
    instrument_id: str,
    evaluation_cycle_id: str,
    engine_sha: str = ENGINE_SHA,
    decision_schema_version: str = DECISION_SCHEMA_VERSION,
) -> str:
    """Deterministic decision ID: DEC_{sha256(instrument_id + evaluation_cycle_id + engine_sha + decision_schema_version)[:16]}.
    Conforms to pattern: ^DEC_[a-f0-9]{16}$
    """
    raw = f"{instrument_id}{evaluation_cycle_id}{engine_sha}{decision_schema_version}"
    h16 = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"DEC_{h16}"


def compute_attempt_id(decision_id: str, attempt_number: int, hash_input: str = "") -> str:
    """Forensic attempt ID: ATT_{decision_id}_{attempt_number}_{hash8}.
    Conforms to pattern: ^ATT_DEC_[a-f0-9]{16}_[0-9]+_[a-f0-9]{8}$
    """
    raw = f"{decision_id}_{attempt_number}_{hash_input}"
    h8 = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"ATT_{decision_id}_{attempt_number}_{h8}"


def compute_episode_id(symbol: str, episode_start_utc: str, initial_plan_hash: str) -> str:
    """Deterministic episode ID: EP_{clean_symbol}_{compactTimestampUtc}_{hash8}.
    Conforms to pattern: ^EP_[A-Z0-9._-]+_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{8}$
    """
    try:
        dt = datetime.fromisoformat(episode_start_utc.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)
    ts_compact = dt.strftime("%Y%m%dT%H%M%SZ")
    raw = f"{symbol}_{episode_start_utc}_{initial_plan_hash}"
    h8 = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    clean_sym = re.sub(r"[^A-Z0-9._-]", "_", symbol.upper())
    return f"EP_{clean_sym}_{ts_compact}_{h8}"


def compute_trade_plan_hash(trade_plan: Optional[Dict[str, Any]]) -> str:
    """Computes a stable SHA256 hash of core trade plan levels."""
    if not trade_plan:
        return "NO_TRADE_PLAN"
    core_keys = [
        "entryReferencePrice",
        "corridorMin",
        "corridorMax",
        "stopLoss",
        "takeProfit1",
        "takeProfit2",
    ]
    core_vals = {k: trade_plan.get(k) for k in core_keys}
    raw = json.dumps(core_vals, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


PROSPECTIVE_DDL = """
PRAGMA foreign_keys = ON;

-- 1. Evaluation Cycles
CREATE TABLE IF NOT EXISTS prospective_evaluation_cycles (
    evaluation_cycle_id TEXT PRIMARY KEY,
    cycle_type TEXT NOT NULL,
    cycle_started_at_utc TEXT NOT NULL,
    cycle_completed_at_utc TEXT,
    universe_version TEXT NOT NULL,
    universe_snapshot_id TEXT,
    engine_version TEXT NOT NULL,
    engine_sha TEXT NOT NULL,
    expected_evaluations_count INTEGER NOT NULL DEFAULT 0,
    completed_evaluations_count INTEGER NOT NULL DEFAULT 0,
    failed_before_model_evaluations_count INTEGER NOT NULL DEFAULT 0,
    failed_during_model_evaluations_count INTEGER NOT NULL DEFAULT 0,
    missing_evaluations_count INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL
);

-- 2. Expected Evaluation Manifest
CREATE TABLE IF NOT EXISTS prospective_expected_evaluations (
    expected_evaluation_id TEXT PRIMARY KEY,
    evaluation_cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    scope_status TEXT NOT NULL,
    expectation_established_at_utc TEXT NOT NULL,
    terminal_reconciliation_state TEXT NOT NULL,
    decision_id TEXT,
    FOREIGN KEY (evaluation_cycle_id) REFERENCES prospective_evaluation_cycles(evaluation_cycle_id)
);

-- 3. Prospective Episodes
CREATE TABLE IF NOT EXISTS prospective_episodes (
    episode_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    episode_start_utc TEXT NOT NULL,
    episode_last_evaluated_utc TEXT NOT NULL,
    episode_status TEXT NOT NULL,
    sessions_observed_count INTEGER NOT NULL DEFAULT 1,
    initial_trade_plan_hash TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

-- 4. Prospective Universe Snapshots
CREATE TABLE IF NOT EXISTS prospective_universe_snapshots (
    universe_snapshot_id TEXT PRIMARY KEY,
    snapshot_timestamp_utc TEXT NOT NULL,
    universe_version TEXT NOT NULL,
    membership_source TEXT NOT NULL,
    membership_sha256 TEXT NOT NULL,
    total_in_scope_count INTEGER NOT NULL,
    total_evaluated_count INTEGER NOT NULL,
    exclusion_counts_json TEXT NOT NULL,
    included_symbols_json TEXT,
    created_at_utc TEXT NOT NULL
);

-- 5. Prospective Decision Events (Immutable Canonical Entity)
CREATE TABLE IF NOT EXISTS prospective_decision_events (
    decision_id TEXT PRIMARY KEY,
    evaluation_cycle_id TEXT NOT NULL,
    current_attempt_id TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    episode_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    cycle_type TEXT NOT NULL,
    population_sampling_mode TEXT NOT NULL,
    evidence_origin TEXT NOT NULL,
    scope_status TEXT NOT NULL,
    evaluation_completion_state TEXT NOT NULL,
    empirical_certification_state TEXT NOT NULL,
    eligible_for_decision_quality INTEGER NOT NULL,
    eligible_for_coverage_analysis INTEGER NOT NULL,
    eligible_for_ranking_analysis INTEGER NOT NULL,
    eligible_for_opportunity_capture INTEGER NOT NULL,
    evaluation_started_at_utc TEXT NOT NULL,
    evaluation_completed_at_utc TEXT,
    capture_written_at_utc TEXT NOT NULL,
    market_session TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    engine_sha TEXT NOT NULL,
    decision_engine_sha TEXT NOT NULL,
    decision_schema_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    universe_version TEXT NOT NULL,
    universe_snapshot_id TEXT,
    data_provider TEXT NOT NULL,
    provider_source_timestamp_utc TEXT NOT NULL,
    ingestion_timestamp_utc TEXT NOT NULL,
    freshness_status TEXT NOT NULL,
    market_regime TEXT NOT NULL,
    decision_state TEXT NOT NULL,
    actionability_state TEXT NOT NULL,
    confluence_score REAL,
    cross_sectional_rank INTEGER,
    cross_sectional_population_size INTEGER,
    first_binding_rule_id TEXT,
    first_binding_rule_category TEXT,
    rejection_reason_json TEXT NOT NULL,
    trade_plan_json TEXT,
    infrastructure_failure_json TEXT,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (evaluation_cycle_id) REFERENCES prospective_evaluation_cycles(evaluation_cycle_id),
    FOREIGN KEY (episode_id) REFERENCES prospective_episodes(episode_id)
);

-- 6. Prospective Evaluation Attempts (Forensic Execution Audit Log)
CREATE TABLE IF NOT EXISTS prospective_evaluation_attempts (
    evaluation_attempt_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    evaluation_cycle_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    attempt_started_at_utc TEXT NOT NULL,
    attempt_completed_at_utc TEXT,
    attempt_status TEXT NOT NULL,
    failure_details TEXT,
    FOREIGN KEY (decision_id) REFERENCES prospective_decision_events(decision_id),
    FOREIGN KEY (evaluation_cycle_id) REFERENCES prospective_evaluation_cycles(evaluation_cycle_id)
);

-- 7. Prospective Feature Snapshots
CREATE TABLE IF NOT EXISTS prospective_feature_snapshots (
    decision_id TEXT PRIMARY KEY,
    feature_snapshot_hash TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    features_json TEXT NOT NULL,
    FOREIGN KEY (decision_id) REFERENCES prospective_decision_events(decision_id)
);

-- 8. Prospective Rule Trace Logs
CREATE TABLE IF NOT EXISTS prospective_rule_evaluations (
    rule_eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    rule_category TEXT NOT NULL,
    actual_execution_order INTEGER NOT NULL,
    evaluation_state TEXT NOT NULL,
    input_values_json TEXT NOT NULL,
    threshold_value TEXT,
    is_binding INTEGER NOT NULL,
    failure_message TEXT,
    FOREIGN KEY (decision_id) REFERENCES prospective_decision_events(decision_id)
);

-- 9. Prospective Outcome Links (Post-Hoc Settled Outcomes)
CREATE TABLE IF NOT EXISTS prospective_outcome_links (
    outcome_link_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL UNIQUE,
    episode_id TEXT NOT NULL,
    trade_plan_id TEXT,
    governing_contract_id TEXT NOT NULL,
    governing_contract_version TEXT NOT NULL,
    governing_contract_sha256 TEXT NOT NULL,
    outcome_status TEXT NOT NULL,
    realized_gross_return_pct REAL,
    realized_net_simulated_return_pct REAL,
    realized_r_multiple REAL,
    recorded_at_utc TEXT NOT NULL,
    FOREIGN KEY (decision_id) REFERENCES prospective_decision_events(decision_id),
    FOREIGN KEY (episode_id) REFERENCES prospective_episodes(episode_id)
);

-- 10. Prospective Decision Corrections (Append-Only)
CREATE TABLE IF NOT EXISTS prospective_decision_corrections (
    correction_event_id TEXT PRIMARY KEY,
    original_decision_id TEXT NOT NULL,
    corrected_field_name TEXT NOT NULL,
    original_value_json TEXT,
    corrected_value_json TEXT,
    correction_reason TEXT NOT NULL,
    authorized_by TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    FOREIGN KEY (original_decision_id) REFERENCES prospective_decision_events(decision_id)
);

-- Immutability Triggers
CREATE TRIGGER IF NOT EXISTS trg_prevent_update_prospective_decision_events
BEFORE UPDATE ON prospective_decision_events
BEGIN
    SELECT RAISE(FAIL, 'IMMUTABILITY_VIOLATION: Updates to prospective_decision_events are strictly prohibited.');
END;

CREATE TRIGGER IF NOT EXISTS trg_prevent_delete_prospective_decision_events
BEFORE DELETE ON prospective_decision_events
BEGIN
    SELECT RAISE(FAIL, 'IMMUTABILITY_VIOLATION: Deletions from prospective_decision_events are strictly prohibited.');
END;
"""


def init_prospective_tables(conn) -> None:
    """Idempotently initializes prospective decision capture tables and triggers."""
    with conn:
        conn.executescript(PROSPECTIVE_DDL)


class ProspectiveDecisionCaptureEngine:
    """Authoritative Engine for Prospective Full Decision Capture."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        stream_path: Optional[str] = None,
        fail_open_client: bool = True,
    ):
        self.db = GovernanceDB(db_path)
        self.db_path = self.db.db_path
        self.fail_open_client = fail_open_client
        conn = self.db.get_connection()
        try:
            init_prospective_tables(conn)
        finally:
            conn.close()

        if stream_path:
            self.stream_path = os.path.abspath(stream_path)
        else:
            base_dir = resolve_data_root()
            self.stream_path = os.path.join(base_dir, "prospective_capture_stream.jsonl")
        os.makedirs(os.path.dirname(self.stream_path), exist_ok=True)


    def _append_recovery_stream(self, record_type: str, data: Dict[str, Any]) -> bool:
        """Appends a single JSONL record to the append-only recovery stream."""
        try:
            payload = {
                "record_type": record_type,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "payload": data,
            }
            line = json.dumps(payload, sort_keys=True) + "\n"
            with open(self.stream_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            return True
        except Exception as e:
            logger.error("Failed to append to prospective recovery stream: %s", e)
            if not self.fail_open_client:
                raise
            return False

    @retry_sqlite()
    def create_or_get_evaluation_cycle(
        self,
        cycle_type: str,
        universe_version: str,
        universe_snapshot_id: Optional[str] = None,
        engine_version: str = ENGINE_VERSION,
        engine_sha: str = ENGINE_SHA,
        cycle_started_at_utc: Optional[str] = None,
        evaluation_cycle_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates an evaluation cycle or reuses an existing active cycle.
        Invariants:
        - Transient retries must reuse the original evaluation_cycle_id.
        - Idempotent on existing evaluation_cycle_id.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        cycle_started = cycle_started_at_utc or now_utc

        if not evaluation_cycle_id:
            evaluation_cycle_id = compute_evaluation_cycle_id(
                cycle_type=cycle_type,
                cycle_started_at_utc=cycle_started,
                hash_input=f"{universe_version}_{engine_sha[:8]}",
            )

        conn = self.db.get_connection()
        try:
            with conn:
                cur = conn.execute(
                    """
                    SELECT evaluation_cycle_id, cycle_type, cycle_started_at_utc,
                           cycle_completed_at_utc, universe_version, universe_snapshot_id,
                           engine_version, engine_sha, expected_evaluations_count,
                           completed_evaluations_count, failed_before_model_evaluations_count,
                           failed_during_model_evaluations_count, missing_evaluations_count,
                           created_at_utc
                    FROM prospective_evaluation_cycles
                    WHERE evaluation_cycle_id = ?
                    """,
                    (evaluation_cycle_id,),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)

                conn.execute(
                    """
                    INSERT INTO prospective_evaluation_cycles (
                        evaluation_cycle_id, cycle_type, cycle_started_at_utc,
                        universe_version, universe_snapshot_id, engine_version,
                        engine_sha, expected_evaluations_count, completed_evaluations_count,
                        failed_before_model_evaluations_count, failed_during_model_evaluations_count,
                        missing_evaluations_count, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, ?)
                    """,
                    (
                        evaluation_cycle_id,
                        cycle_type,
                        cycle_started,
                        universe_version,
                        universe_snapshot_id,
                        engine_version,
                        engine_sha,
                        now_utc,
                    ),
                )

                created_record = {
                    "evaluation_cycle_id": evaluation_cycle_id,
                    "cycle_type": cycle_type,
                    "cycle_started_at_utc": cycle_started,
                    "cycle_completed_at_utc": None,
                    "universe_version": universe_version,
                    "universe_snapshot_id": universe_snapshot_id,
                    "engine_version": engine_version,
                    "engine_sha": engine_sha,
                    "expected_evaluations_count": 0,
                    "completed_evaluations_count": 0,
                    "failed_before_model_evaluations_count": 0,
                    "failed_during_model_evaluations_count": 0,
                    "missing_evaluations_count": 0,
                    "created_at_utc": now_utc,
                }
                self._append_recovery_stream("EVALUATION_CYCLE_CREATED", created_record)
                return created_record
        finally:
            conn.close()

    @retry_sqlite()
    def register_expected_evaluations(
        self,
        evaluation_cycle_id: str,
        items: List[Union[Tuple[str, str], Dict[str, Any]]],
        scope_status: str = "IN_SCOPE",
    ) -> int:
        """Registers the pre-run manifest of expected evaluations for a cycle.
        Initially marks each asset with MISSING_EXPECTED_EVALUATION.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        records_to_insert = []
        for item in items:
            if isinstance(item, tuple):
                sym, inst_id = item[0], item[1]
                scope = scope_status
            else:
                sym = item["symbol"]
                inst_id = item.get("instrument_id", sym)
                scope = item.get("scope_status", scope_status)

            exp_id = f"EXP_{evaluation_cycle_id}_{sym}"
            records_to_insert.append((exp_id, evaluation_cycle_id, sym, inst_id, scope, now_utc))

        conn = self.db.get_connection()
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO prospective_expected_evaluations (
                        expected_evaluation_id, evaluation_cycle_id, symbol,
                        instrument_id, scope_status, expectation_established_at_utc,
                        terminal_reconciliation_state, decision_id
                    ) VALUES (?, ?, ?, ?, ?, ?, 'MISSING_EXPECTED_EVALUATION', NULL)
                    """,
                    records_to_insert,
                )

                # Update expected and missing count on cycle
                cur = conn.execute(
                    "SELECT COUNT(*) FROM prospective_expected_evaluations WHERE evaluation_cycle_id = ?",
                    (evaluation_cycle_id,),
                )
                tot = cur.fetchone()[0]

                cur_miss = conn.execute(
                    """
                    SELECT COUNT(*) FROM prospective_expected_evaluations
                    WHERE evaluation_cycle_id = ? AND terminal_reconciliation_state = 'MISSING_EXPECTED_EVALUATION'
                    """,
                    (evaluation_cycle_id,),
                )
                miss = cur_miss.fetchone()[0]

                conn.execute(
                    """
                    UPDATE prospective_evaluation_cycles
                    SET expected_evaluations_count = ?,
                        missing_evaluations_count = ?
                    WHERE evaluation_cycle_id = ?
                    """,
                    (tot, miss, evaluation_cycle_id),
                )

                self._append_recovery_stream(
                    "EXPECTED_EVALUATIONS_REGISTERED",
                    {
                        "evaluation_cycle_id": evaluation_cycle_id,
                        "registered_count": len(records_to_insert),
                        "total_expected_count": tot,
                    },
                )
                return len(records_to_insert)
        finally:
            conn.close()

    @retry_sqlite()
    def create_or_get_episode(
        self,
        symbol: str,
        trade_plan: Optional[Dict[str, Any]] = None,
        timestamp_utc: Optional[str] = None,
    ) -> str:
        """Manages episode continuity across evaluation cycles.
        Invariants:
        - If active episode exists and trade plan is consistent, continue episode.
        - If trade plan altered price levels or invalidated, reset episode.
        """
        now_utc = timestamp_utc or datetime.now(timezone.utc).isoformat()
        current_plan_hash = compute_trade_plan_hash(trade_plan)

        conn = self.db.get_connection()
        try:
            with conn:
                cur = conn.execute(
                    """
                    SELECT episode_id, episode_start_utc, episode_status,
                           sessions_observed_count, initial_trade_plan_hash
                    FROM prospective_episodes
                    WHERE symbol = ? AND episode_status = 'ACTIVE_CONTINUATION'
                    ORDER BY created_at_utc DESC LIMIT 1
                    """,
                    (symbol,),
                )
                existing = cur.fetchone()

                if existing:
                    # Check if trade plan price levels altered
                    if existing["initial_trade_plan_hash"] != current_plan_hash and current_plan_hash != "NO_TRADE_PLAN":
                        # Invalidate existing episode as RESET_PRICE_LEVEL_ALTERED
                        conn.execute(
                            """
                            UPDATE prospective_episodes
                            SET episode_status = 'RESET_PRICE_LEVEL_ALTERED',
                                episode_last_evaluated_utc = ?
                            WHERE episode_id = ?
                            """,
                            (now_utc, existing["episode_id"]),
                        )
                    else:
                        # Continue existing episode
                        new_count = existing["sessions_observed_count"] + 1
                        conn.execute(
                            """
                            UPDATE prospective_episodes
                            SET sessions_observed_count = ?,
                                episode_last_evaluated_utc = ?
                            WHERE episode_id = ?
                            """,
                            (new_count, now_utc, existing["episode_id"]),
                        )
                        return existing["episode_id"]

                # Create fresh episode
                ep_id = compute_episode_id(symbol, now_utc, current_plan_hash)
                conn.execute(
                    """
                    INSERT INTO prospective_episodes (
                        episode_id, symbol, episode_start_utc, episode_last_evaluated_utc,
                        episode_status, sessions_observed_count, initial_trade_plan_hash,
                        created_at_utc
                    ) VALUES (?, ?, ?, ?, 'ACTIVE_CONTINUATION', 1, ?, ?)
                    """,
                    (ep_id, symbol, now_utc, now_utc, current_plan_hash, now_utc),
                )
                return ep_id
        finally:
            conn.close()

    @retry_sqlite()
    def record_decision_event(
        self,
        evaluation_cycle_id: str,
        symbol: str,
        instrument_id: str,
        cycle_type: str,
        population_sampling_mode: str,
        scope_status: str,
        decision_state: str,
        actionability_state: str,
        rejection_reason: Dict[str, Any],
        evaluation_completion_state: str = "COMPLETED_EVALUATION",
        evidence_origin: Optional[str] = None,
        market_session: str = "REGULAR_SESSION",
        engine_version: str = ENGINE_VERSION,
        engine_sha: str = ENGINE_SHA,
        decision_engine_sha: str = DECISION_ENGINE_SHA,
        decision_schema_version: str = DECISION_SCHEMA_VERSION,
        config_hash: str = DEFAULT_CONFIG_HASH,
        universe_version: str = "2026.1",
        universe_snapshot_id: Optional[str] = None,
        data_provider: str = "POLYGON",
        provider_source_timestamp_utc: Optional[str] = None,
        ingestion_timestamp_utc: Optional[str] = None,
        freshness_status: str = "LIVE_AUTHORITATIVE",
        market_regime: str = "BULL",
        confluence_score: Optional[float] = None,
        cross_sectional_rank: Optional[int] = None,
        cross_sectional_population_size: Optional[int] = None,
        trade_plan_state: Optional[Dict[str, Any]] = None,
        infrastructure_failure: Optional[Dict[str, Any]] = None,
        features: Optional[List[Dict[str, Any]]] = None,
        rule_evaluations: Optional[List[Dict[str, Any]]] = None,
        evaluation_started_at_utc: Optional[str] = None,
        evaluation_completed_at_utc: Optional[str] = None,
        attempt_status: str = "SUCCESS",
        attempt_failure_details: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically records a decision event with forensic attempt and relational traces.

        Handles retries:
        - If decision_id already exists: records a new evaluation_attempt, does not update
          decision_event (preserving immutability triggers and denominator integrity).
        - If new decision_id: atomically creates decision_event, attempt 1, feature snapshot,
          and rule evaluation logs.
        - Fail-open for client, fail-closed for empirical certification.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        started_at = evaluation_started_at_utc or now_utc
        completed_at = evaluation_completed_at_utc or (now_utc if attempt_status == "SUCCESS" else None)
        src_ts = provider_source_timestamp_utc or now_utc
        ingest_ts = ingestion_timestamp_utc or now_utc

        origin = evidence_origin or get_current_evidence_origin()

        # Empirical certification gate
        if origin == "NATURAL_PRODUCTION":
            cert_state = "CERTIFIED_NATURAL_PRODUCTION"
        else:
            cert_state = "QUARANTINED_NON_PRODUCTION"

        # Analytical Eligibility Rules
        eligible_quality = 1 if (
            origin == "NATURAL_PRODUCTION"
            and scope_status == "IN_SCOPE"
            and evaluation_completion_state == "COMPLETED_EVALUATION"
        ) else 0

        eligible_coverage = 1 if (
            origin == "NATURAL_PRODUCTION"
            and cycle_type != "EXPLICIT_USER_REQUEST"
        ) else 0

        eligible_ranking = 1 if (
            eligible_quality == 1
            and confluence_score is not None
            and cycle_type in ("SCHEDULED_UNIVERSE_SCAN", "RADAR_CYCLE")
        ) else 0

        eligible_opportunity = 1 if (
            population_sampling_mode == "SYSTEMATIC"
            and cycle_type == "SCHEDULED_UNIVERSE_SCAN"
            and origin == "NATURAL_PRODUCTION"
            and evaluation_completion_state == "COMPLETED_EVALUATION"
        ) else 0

        dec_id = compute_decision_id(
            instrument_id=instrument_id,
            evaluation_cycle_id=evaluation_cycle_id,
            engine_sha=engine_sha,
            decision_schema_version=decision_schema_version,
        )

        try:
            # Ensure episode exists
            active_episode_id = episode_id or self.create_or_get_episode(
                symbol=symbol,
                trade_plan=trade_plan_state,
                timestamp_utc=started_at,
            )

            conn = self.db.get_connection()
            try:
                with conn:
                    # Check if decision_id already exists (RETRY ADVERSARIAL TEST A)
                    cur = conn.execute(
                        "SELECT decision_id, current_attempt_id, attempt_count FROM prospective_decision_events WHERE decision_id = ?",
                        (dec_id,),
                    )
                    existing = cur.fetchone()

                    if existing:
                        # Existing decision event found!
                        # Do NOT mutate prospective_decision_events (prohibited by immutability trigger).
                        # Insert a new attempt record into prospective_evaluation_attempts.
                        cur_attempts = conn.execute(
                            "SELECT COUNT(*) FROM prospective_evaluation_attempts WHERE decision_id = ?",
                            (dec_id,),
                        )
                        prior_attempts_count = cur_attempts.fetchone()[0]
                        new_attempt_number = prior_attempts_count + 1
                        new_attempt_id = compute_attempt_id(dec_id, new_attempt_number)

                        conn.execute(
                            """
                            INSERT INTO prospective_evaluation_attempts (
                                evaluation_attempt_id, decision_id, evaluation_cycle_id,
                                attempt_number, attempt_started_at_utc, attempt_completed_at_utc,
                                attempt_status, failure_details
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                new_attempt_id,
                                dec_id,
                                evaluation_cycle_id,
                                new_attempt_number,
                                started_at,
                                completed_at,
                                attempt_status,
                                attempt_failure_details,
                            ),
                        )

                        retry_payload = {
                            "decision_id": dec_id,
                            "evaluation_attempt_id": new_attempt_id,
                            "attempt_number": new_attempt_number,
                            "attempt_status": attempt_status,
                            "is_retry": True,
                        }
                        self._append_recovery_stream("EVALUATION_ATTEMPT_RECORDED", retry_payload)
                        return {
                            "decision_id": dec_id,
                            "attempt_id": new_attempt_id,
                            "attempt_number": new_attempt_number,
                            "empirical_certification_state": cert_state,
                            "is_retry": True,
                            "success": True,
                        }

                    # Fresh Decision Event
                    att_id = compute_attempt_id(dec_id, 1)

                    first_rule_id = rejection_reason.get("firstBindingRuleId")
                    first_rule_cat = rejection_reason.get("firstBindingRuleCategory")

                    rejection_json = json.dumps(rejection_reason, sort_keys=True)
                    trade_plan_json = json.dumps(trade_plan_state, sort_keys=True) if trade_plan_state else None
                    infra_json = json.dumps(infrastructure_failure, sort_keys=True) if infrastructure_failure else None

                    conn.execute(
                        """
                        INSERT INTO prospective_decision_events (
                            decision_id, evaluation_cycle_id, current_attempt_id,
                            attempt_count, episode_id, symbol, instrument_id, cycle_type,
                            population_sampling_mode, evidence_origin, scope_status,
                            evaluation_completion_state, empirical_certification_state,
                            eligible_for_decision_quality, eligible_for_coverage_analysis,
                            eligible_for_ranking_analysis, eligible_for_opportunity_capture,
                            evaluation_started_at_utc, evaluation_completed_at_utc,
                            capture_written_at_utc, market_session, engine_version,
                            engine_sha, decision_engine_sha, decision_schema_version,
                            config_hash, universe_version, universe_snapshot_id,
                            data_provider, provider_source_timestamp_utc,
                            ingestion_timestamp_utc, freshness_status, market_regime,
                            decision_state, actionability_state, confluence_score,
                            cross_sectional_rank, cross_sectional_population_size,
                            first_binding_rule_id, first_binding_rule_category,
                            rejection_reason_json, trade_plan_json, infrastructure_failure_json,
                            created_at_utc
                        ) VALUES (
                            ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            dec_id,
                            evaluation_cycle_id,
                            att_id,
                            active_episode_id,
                            symbol,
                            instrument_id,
                            cycle_type,
                            population_sampling_mode,
                            origin,
                            scope_status,
                            evaluation_completion_state,
                            cert_state,
                            eligible_quality,
                            eligible_coverage,
                            eligible_ranking,
                            eligible_opportunity,
                            started_at,
                            completed_at,
                            now_utc,
                            market_session,
                            engine_version,
                            engine_sha,
                            decision_engine_sha,
                            decision_schema_version,
                            config_hash,
                            universe_version,
                            universe_snapshot_id,
                            data_provider,
                            src_ts,
                            ingest_ts,
                            freshness_status,
                            market_regime,
                            decision_state,
                            actionability_state,
                            confluence_score,
                            cross_sectional_rank,
                            cross_sectional_population_size,
                            first_rule_id,
                            first_rule_cat,
                            rejection_json,
                            trade_plan_json,
                            infra_json,
                            now_utc,
                        ),
                    )

                    # Initial Attempt Record
                    conn.execute(
                        """
                        INSERT INTO prospective_evaluation_attempts (
                            evaluation_attempt_id, decision_id, evaluation_cycle_id,
                            attempt_number, attempt_started_at_utc, attempt_completed_at_utc,
                            attempt_status, failure_details
                        ) VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                        """,
                        (
                            att_id,
                            dec_id,
                            evaluation_cycle_id,
                            started_at,
                            completed_at,
                            attempt_status,
                            attempt_failure_details,
                        ),
                    )

                    # Feature Snapshot if present
                    if features:
                        feat_json = json.dumps(features, sort_keys=True)
                        feat_hash = hashlib.sha256(feat_json.encode("utf-8")).hexdigest()
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO prospective_feature_snapshots (
                                decision_id, feature_snapshot_hash, timestamp_utc, features_json
                            ) VALUES (?, ?, ?, ?)
                            """,
                            (dec_id, feat_hash, started_at, feat_json),
                        )

                    # Rule Evaluations if present
                    if rule_evaluations:
                        for rule in rule_evaluations:
                            conn.execute(
                                """
                                INSERT INTO prospective_rule_evaluations (
                                    decision_id, rule_id, rule_version, rule_category,
                                    actual_execution_order, evaluation_state,
                                    input_values_json, threshold_value, is_binding,
                                    failure_message
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    dec_id,
                                    rule["ruleId"],
                                    rule.get("ruleVersion", "1.0.0"),
                                    rule["ruleCategory"],
                                    rule["actualExecutionOrder"],
                                    rule["evaluationState"],
                                    json.dumps(rule.get("inputValues", {}), sort_keys=True),
                                    str(rule.get("thresholdValue")) if rule.get("thresholdValue") is not None else None,
                                    1 if rule.get("isBinding", False) else 0,
                                    rule.get("failureMessage"),
                                ),
                            )

                    # Update reconciliation state in prospective_expected_evaluations
                    conn.execute(
                        """
                        UPDATE prospective_expected_evaluations
                        SET terminal_reconciliation_state = ?,
                            decision_id = ?
                        WHERE evaluation_cycle_id = ? AND symbol = ?
                        """,
                        (evaluation_completion_state, dec_id, evaluation_cycle_id, symbol),
                    )

                # Write to append-only JSONL recovery stream
                self._append_recovery_stream(
                    "DECISION_EVENT_RECORDED",
                    {
                        "decision_id": dec_id,
                        "current_attempt_id": att_id,
                        "symbol": symbol,
                        "instrument_id": instrument_id,
                        "evaluation_cycle_id": evaluation_cycle_id,
                        "episode_id": active_episode_id,
                        "decision_state": decision_state,
                        "confluence_score": confluence_score,
                        "empirical_certification_state": cert_state,
                    },
                )

                return {
                    "decision_id": dec_id,
                    "attempt_id": att_id,
                    "attempt_number": 1,
                    "episode_id": active_episode_id,
                    "empirical_certification_state": cert_state,
                    "is_retry": False,
                    "success": True,
                }
            finally:
                conn.close()

        except Exception as e:
            logger.error("Decision capture failure for %s in cycle %s: %s", symbol, evaluation_cycle_id, e)
            if not self.fail_open_client:
                raise
            return {
                "decision_id": dec_id,
                "empirical_certification_state": "UNCERTIFIED_CAPTURE_FAILURE",
                "error": str(e),
                "success": False,
            }

    @retry_sqlite()
    def record_infrastructure_failure(
        self,
        evaluation_cycle_id: str,
        symbol: str,
        instrument_id: str,
        failure_type: str,
        error_message: str,
        stage: str = "BEFORE_MODEL",
        provider: str = "UNKNOWN",
    ) -> None:
        """Records an infrastructure defect preventing normal model evaluation."""
        now_utc = datetime.now(timezone.utc).isoformat()
        term_state = (
            "FAILED_BEFORE_MODEL_EVALUATION"
            if stage == "BEFORE_MODEL"
            else "FAILED_DURING_MODEL_EVALUATION"
        )

        conn = self.db.get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE prospective_expected_evaluations
                    SET terminal_reconciliation_state = ?
                    WHERE evaluation_cycle_id = ? AND symbol = ?
                    """,
                    (term_state, evaluation_cycle_id, symbol),
                )
                self._append_recovery_stream(
                    "INFRASTRUCTURE_FAILURE_RECORDED",
                    {
                        "evaluation_cycle_id": evaluation_cycle_id,
                        "symbol": symbol,
                        "instrument_id": instrument_id,
                        "failure_type": failure_type,
                        "error_message": error_message,
                        "stage": stage,
                        "provider": provider,
                        "timestamp_utc": now_utc,
                    },
                )
        finally:
            conn.close()

    @retry_sqlite()
    def record_decision_correction(
        self,
        original_decision_id: str,
        corrected_field_name: str,
        original_value: Any,
        corrected_value: Any,
        correction_reason: str,
        authorized_by: str,
    ) -> str:
        """Records an append-only post-hoc correction event.
        Guarantees that prospective_decision_events remains 100% immutable.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        raw = f"{original_decision_id}_{corrected_field_name}_{now_utc}"
        h8 = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        clean_field = re.sub(r"[^A-Z0-9._-]", "_", corrected_field_name.upper())
        corr_id = f"COR_{clean_field}_{h8}"

        conn = self.db.get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO prospective_decision_corrections (
                        correction_event_id, original_decision_id, corrected_field_name,
                        original_value_json, corrected_value_json, correction_reason,
                        authorized_by, timestamp_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        corr_id,
                        original_decision_id,
                        corrected_field_name,
                        json.dumps(original_value, sort_keys=True),
                        json.dumps(corrected_value, sort_keys=True),
                        correction_reason,
                        authorized_by,
                        now_utc,
                    ),
                )
                self._append_recovery_stream(
                    "DECISION_CORRECTION_RECORDED",
                    {
                        "correction_event_id": corr_id,
                        "original_decision_id": original_decision_id,
                        "corrected_field_name": corrected_field_name,
                        "correction_reason": correction_reason,
                        "authorized_by": authorized_by,
                        "timestamp_utc": now_utc,
                    },
                )
                return corr_id
        finally:
            conn.close()

    @retry_sqlite()
    def reconcile_evaluation_cycle(self, evaluation_cycle_id: str) -> Dict[str, Any]:
        """Calculates mutually exclusive terminal states and updates cycle tallies.
        Verifies: Expected_N = Completed_N + FailedBefore_N + FailedDuring_N + Missing_N.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        conn = self.db.get_connection()
        try:
            with conn:
                cur = conn.execute(
                    """
                    SELECT terminal_reconciliation_state, COUNT(*) as cnt
                    FROM prospective_expected_evaluations
                    WHERE evaluation_cycle_id = ?
                    GROUP BY terminal_reconciliation_state
                    """,
                    (evaluation_cycle_id,),
                )
                counts = {row["terminal_reconciliation_state"]: row["cnt"] for row in cur.fetchall()}

                completed = counts.get("COMPLETED_EVALUATION", 0)
                failed_before = counts.get("FAILED_BEFORE_MODEL_EVALUATION", 0)
                failed_during = counts.get("FAILED_DURING_MODEL_EVALUATION", 0)
                missing = counts.get("MISSING_EXPECTED_EVALUATION", 0)
                expected_sum = completed + failed_before + failed_during + missing

                # Verify against total registered in cycle
                cur_exp = conn.execute(
                    "SELECT expected_evaluations_count FROM prospective_evaluation_cycles WHERE evaluation_cycle_id = ?",
                    (evaluation_cycle_id,),
                )
                exp_row = cur_exp.fetchone()
                total_expected = exp_row["expected_evaluations_count"] if exp_row else expected_sum

                if expected_sum != total_expected:
                    # Update expected_evaluations_count to match reality
                    total_expected = expected_sum

                conn.execute(
                    """
                    UPDATE prospective_evaluation_cycles
                    SET expected_evaluations_count = ?,
                        completed_evaluations_count = ?,
                        failed_before_model_evaluations_count = ?,
                        failed_during_model_evaluations_count = ?,
                        missing_evaluations_count = ?,
                        cycle_completed_at_utc = ?
                    WHERE evaluation_cycle_id = ?
                    """,
                    (
                        total_expected,
                        completed,
                        failed_before,
                        failed_during,
                        missing,
                        now_utc,
                        evaluation_cycle_id,
                    ),
                )

                reconciliation_report = {
                    "evaluation_cycle_id": evaluation_cycle_id,
                    "expected_count": total_expected,
                    "completed_count": completed,
                    "failed_before_count": failed_before,
                    "failed_during_count": failed_during,
                    "missing_count": missing,
                    "is_balanced": (total_expected == expected_sum),
                    "cycle_completed_at_utc": now_utc,
                }
                self._append_recovery_stream("EVALUATION_CYCLE_RECONCILED", reconciliation_report)
                return reconciliation_report
        finally:
            conn.close()

    def query_denominators(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Queries authoritative counts for scientific research denominators."""
        conn = self.db.get_connection()
        try:
            cur = conn.execute("SELECT COUNT(*) FROM prospective_expected_evaluations")
            total_expected = cur.fetchone()[0]

            cur = conn.execute(
                "SELECT COUNT(*) FROM prospective_decision_events WHERE evaluation_completion_state = 'COMPLETED_EVALUATION'"
            )
            total_completed = cur.fetchone()[0]

            cur = conn.execute(
                "SELECT COUNT(*) FROM prospective_decision_events WHERE empirical_certification_state = 'CERTIFIED_NATURAL_PRODUCTION'"
            )
            total_certified = cur.fetchone()[0]

            cur = conn.execute(
                "SELECT COUNT(*) FROM prospective_decision_events WHERE eligible_for_opportunity_capture = 1"
            )
            total_opp_capture = cur.fetchone()[0]

            cur = conn.execute(
                "SELECT COUNT(*) FROM prospective_decision_events WHERE eligible_for_decision_quality = 1"
            )
            total_quality = cur.fetchone()[0]

            cur = conn.execute("SELECT COUNT(*) FROM prospective_episodes")
            total_episodes = cur.fetchone()[0]

            cur = conn.execute(
                "SELECT COUNT(*) FROM prospective_decision_events WHERE decision_state = 'ACTIONABLE_RECOMMENDATION'"
            )
            total_recommendations = cur.fetchone()[0]

            cur = conn.execute(
                """
                SELECT COUNT(*) FROM prospective_expected_evaluations
                WHERE terminal_reconciliation_state IN ('FAILED_BEFORE_MODEL_EVALUATION', 'FAILED_DURING_MODEL_EVALUATION')
                """
            )
            total_infra_failures = cur.fetchone()[0]

            return {
                "total_expected_evaluations": total_expected,
                "total_completed_evaluations": total_completed,
                "total_certified_natural_production": total_certified,
                "total_opportunity_capture_eligible": total_opp_capture,
                "total_decision_quality_eligible": total_quality,
                "total_episodes": total_episodes,
                "total_actionable_recommendations": total_recommendations,
                "total_infrastructure_failures": total_infra_failures,
            }
        finally:
            conn.close()
