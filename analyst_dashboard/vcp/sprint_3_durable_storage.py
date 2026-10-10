"""ARX Terminal — Sprint 3 Durable Shadow Evidence Store & Relational Engine (Schema V3).

Authoritative persistent storage engine providing:
1. Strict relational schema with foreign keys, unique constraints, and check constraints (Schema 3.0.0).
2. Logical invocation authority (logical_trigger_id, logical_scan_run_id) with attempt tracking.
3. Provenance conflict detection & fail-closed enforcement (PROV-001 through PROV-010).
4. Immutable payload hash comparison (SAME_KEY_DIFFERENT_PAYLOAD_REJECTED = HardIntegrityFailureError).
5. Offset-aware UTC timestamps from offset-aware clocks (datetime.now(timezone.utc).isoformat()).
6. Immutability enforced via SQLite triggers (UPDATE/DELETE prohibited).
7. Deterministic idempotency observation key computation using logical_scan_run_id.
8. Single-transaction bundle admission (all-or-nothing atomicity with rollback).
9. Authoritative denominator derivation from committed database admission records.
10. Multi-worker safe concurrency via SQLite WAL mode with retry backoff (Single replica only).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("arx.vcp.durable_storage")

SCHEMA_VERSION: str = "3.0.0"
MIGRATION_ID: str = "MIGRATION_20261010_003_PROVENANCE_AND_LOGICAL_RUNS"
OBSERVATION_KEY_SPECIFICATION: str = (
    "sha256(scanner_id:security_id:evaluation_as_of:universe_build_id:logical_scan_run_id:candidate_generation_id:semantic_closure_hash)"
)
CLASSIFICATION_POLICY_VERSION: str = "1.0.0"
IDENTITY_SCHEMA_VERSION: str = "3.0.0"
CANONICALIZATION_VERSION: str = "1.0.0"
STORAGE_TOPOLOGY_REQUIREMENT: str = "SINGLE_REPLICA_ONLY"

DEFAULT_SHADOW_DB_FILENAME: str = "shadow_evidence.db"

# Provenance Conflict Code Registry
PROVENANCE_CONFLICT_CODES: Dict[str, str] = {
    "PROV-001": "SCHEDULER_IDENTITY_WITHOUT_VALID_EVENT",
    "PROV-002": "HUMAN_CALLER_CLAIMS_SCHEDULED_ORIGIN",
    "PROV-003": "STARTUP_CONTEXT_WITH_NATURAL_SCHEDULED_CLASS",
    "PROV-004": "REPLAY_METADATA_WITH_NATURAL_ORIGIN",
    "PROV-005": "REPLAY_PARENT_NOT_FOUND",
    "PROV-006": "ORIGINATING_PRINCIPAL_CHANGED_ON_RETRY",
    "PROV-007": "SCHEDULER_EVENT_CHANGED_ON_RETRY",
    "PROV-008": "CLASSIFICATION_POLICY_VERSION_CONFLICT",
    "PROV-009": "SAME_LOGICAL_RUN_DIFFERENT_PROVENANCE",
    "PROV-010": "UNTRUSTED_DELEGATION_CHAIN",
}


class HardIntegrityFailureError(RuntimeError):
    """Raised when an admission attempt violates immutable payload integrity on duplicate key."""
    pass


class ProvenanceConflictError(ValueError):
    """Raised when invocation provenance violates fail-closed validation rules."""
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.details = details or {}


def get_offset_aware_utc_now() -> str:
    """Returns offset-aware ISO 8601 UTC timestamp (never naive local datetime + manual Z)."""
    return datetime.now(timezone.utc).isoformat()


def compute_provenance_fingerprint(
    logical_trigger_id: str,
    logical_scan_run_id: str,
    originating_principal_type: str,
    originating_principal_id: str,
    invocation_class: str,
    origin_class: str,
    scheduler_job_id: Optional[str] = None,
    scheduler_event_id: Optional[str] = None,
    startup_context: bool = False,
    product_request_id: Optional[str] = None,
    replay_of_logical_scan_run_id: Optional[str] = None,
    classification_policy_version: str = CLASSIFICATION_POLICY_VERSION,
    identity_schema_version: str = IDENTITY_SCHEMA_VERSION,
    canonicalization_version: str = CANONICALIZATION_VERSION,
) -> str:
    """Computes SHA-256 fingerprint of canonical deterministic provenance payload."""
    payload = {
        "canonicalization_version": canonicalization_version,
        "classification_policy_version": classification_policy_version,
        "identity_schema_version": identity_schema_version,
        "invocation_class": invocation_class,
        "logical_scan_run_id": logical_scan_run_id,
        "logical_trigger_id": logical_trigger_id,
        "origin_class": origin_class,
        "originating_principal_id": originating_principal_id,
        "originating_principal_type": originating_principal_type,
        "product_request_id": product_request_id or "",
        "replay_of_logical_scan_run_id": replay_of_logical_scan_run_id or "",
        "scheduler_event_id": scheduler_event_id or "",
        "scheduler_job_id": scheduler_job_id or "",
        "startup_context": bool(startup_context),
    }
    canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def compute_immutable_payload_hash(
    security_id: str,
    evaluation_as_of: str,
    universe_build_id: str,
    candidate_generation_id: str,
    candidate_sha: str,
    semantic_closure_hash: str,
    runtime_config_hash: str,
    dependency_lock_hash: str,
    data_provenance_hash: str,
    ruleset_id: str,
    ruleset_version: str,
    predicate_vector_hash: str,
    classification: str,
    decision_posture: str,
    input_fingerprint: str,
) -> str:
    """Computes SHA-256 hash of immutable candidate evaluation decision payload."""
    payload = {
        "candidate_generation_id": candidate_generation_id,
        "candidate_sha": candidate_sha,
        "classification": classification,
        "data_provenance_hash": data_provenance_hash,
        "decision_posture": decision_posture,
        "dependency_lock_hash": dependency_lock_hash,
        "evaluation_as_of": evaluation_as_of,
        "input_fingerprint": input_fingerprint,
        "predicate_vector_hash": predicate_vector_hash,
        "ruleset_id": ruleset_id,
        "ruleset_version": ruleset_version,
        "runtime_config_hash": runtime_config_hash,
        "security_id": security_id,
        "semantic_closure_hash": semantic_closure_hash,
        "universe_build_id": universe_build_id,
    }
    canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def compute_deterministic_observation_key(
    scanner_id: str,
    security_id: str,
    evaluation_as_of: str,
    universe_build_id: str,
    logical_scan_run_id: Optional[str] = None,
    candidate_generation_id: str = "CANDIDATE_GENERATION_003",
    semantic_closure_hash: str = "",
    snapshot_run_id: Optional[str] = None,
) -> str:
    """Computes canonical deterministic idempotency key for an observation (Observation Unit V3)."""
    effective_run_id = logical_scan_run_id or snapshot_run_id or ""
    tuple_str = (
        f"{scanner_id}:{security_id}:{evaluation_as_of}:{universe_build_id}:"
        f"{effective_run_id}:{candidate_generation_id}:{semantic_closure_hash}"
    )
    return hashlib.sha256(tuple_str.encode("utf-8")).hexdigest()


def retry_sqlite(max_retries: int = 5, base_delay: float = 0.05):
    """Decorator to retry SQLite operations with exponential backoff on database lock contention."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    last_err = e
                    err_msg = str(e).lower()
                    if "locked" in err_msg or "busy" in err_msg:
                        if attempt < max_retries - 1:
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                    raise
                except Exception:
                    raise
            if last_err:
                raise last_err
        return wrapper
    return decorator


DDL_SCHEMA = """
-- 1. Logical Scan Runs Table
CREATE TABLE IF NOT EXISTS logical_scan_runs (
    logical_scan_run_id TEXT PRIMARY KEY,
    logical_trigger_id TEXT NOT NULL,
    scanner_id TEXT NOT NULL,
    universe_build_id TEXT NOT NULL,
    evaluation_as_of TEXT NOT NULL,
    scheduled_for TEXT,
    invocation_class TEXT NOT NULL CHECK(invocation_class IN ('BOOT_WARMUP', 'SCHEDULED_PRODUCTION', 'PRODUCT_LIFECYCLE', 'MANUAL_OPERATOR', 'REPLAY', 'TEST', 'SYNTHETIC', 'PROVENANCE_CONFLICT')),
    origin_class TEXT NOT NULL CHECK(origin_class IN ('NATURAL_PRODUCTION', 'NON_EVIDENCE_BOOTSTRAP', 'SYNTHETIC', 'REPLAY', 'ADMIN_FORCED', 'TEST', 'NOT_ADMITTED')),
    originating_principal_type TEXT NOT NULL,
    originating_principal_id TEXT NOT NULL,
    immediate_caller_principal_id TEXT,
    scheduler_job_id TEXT,
    scheduler_event_id TEXT,
    startup_context INTEGER NOT NULL DEFAULT 0,
    boot_instance_id TEXT,
    product_request_id TEXT,
    replay_of_logical_scan_run_id TEXT,
    classification_policy_version TEXT NOT NULL,
    identity_schema_version TEXT NOT NULL,
    canonicalization_version TEXT NOT NULL,
    provenance_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    classified_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'CREATED'
);

-- 2. Scan Attempts Table
CREATE TABLE IF NOT EXISTS scan_attempts (
    attempt_id TEXT PRIMARY KEY,
    logical_scan_run_id TEXT NOT NULL REFERENCES logical_scan_runs(logical_scan_run_id),
    delivery_attempt_id TEXT NOT NULL,
    execution_attempt_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    worker_id TEXT NOT NULL,
    process_id INTEGER NOT NULL,
    deployment_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    failure_class TEXT
);

-- 3. Provenance Conflicts Table
CREATE TABLE IF NOT EXISTS provenance_conflicts (
    conflict_record_id TEXT PRIMARY KEY,
    logical_scan_run_id TEXT NOT NULL,
    conflict_code TEXT NOT NULL,
    reason TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 4. Shadow Observations Table
CREATE TABLE IF NOT EXISTS shadow_observations (
    observation_id TEXT PRIMARY KEY,
    observation_key TEXT UNIQUE NOT NULL,
    logical_scan_run_id TEXT NOT NULL,
    scanner_id TEXT NOT NULL,
    scanner_run_id TEXT NOT NULL,
    security_id TEXT NOT NULL,
    evaluation_as_of TEXT NOT NULL,
    candidate_generation_id TEXT NOT NULL,
    candidate_sha TEXT NOT NULL,
    semantic_closure_hash TEXT NOT NULL,
    universe_build_id TEXT NOT NULL,
    origin_class TEXT NOT NULL CHECK(origin_class IN ('NATURAL_PRODUCTION', 'NON_EVIDENCE_BOOTSTRAP', 'SYNTHETIC', 'REPLAY', 'ADMIN_FORCED', 'TEST', 'NOT_ADMITTED')),
    immutable_payload_hash TEXT NOT NULL,
    provenance_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 5. Prospective Decisions Table (1:1 with observation)
CREATE TABLE IF NOT EXISTS prospective_decisions (
    decision_record_id TEXT PRIMARY KEY,
    observation_id TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_id),
    observation_key TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_key),
    logical_scan_run_id TEXT NOT NULL,
    evaluation_as_of TEXT NOT NULL,
    known_at TEXT NOT NULL,
    security_id TEXT NOT NULL,
    universe_build_id TEXT NOT NULL,
    snapshot_run_id TEXT NOT NULL,
    candidate_generation_id TEXT NOT NULL,
    candidate_sha TEXT NOT NULL,
    semantic_closure_hash TEXT NOT NULL,
    runtime_config_hash TEXT NOT NULL,
    dependency_lock_hash TEXT NOT NULL,
    data_provenance_hash TEXT NOT NULL,
    ruleset_id TEXT NOT NULL,
    ruleset_version TEXT NOT NULL,
    predicate_vector_hash TEXT NOT NULL,
    classification TEXT NOT NULL,
    decision_posture TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    immutable_payload_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 6. Production Exposures Table (1:1 with observation)
CREATE TABLE IF NOT EXISTS production_exposures (
    exposure_id TEXT PRIMARY KEY,
    observation_id TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_id),
    observation_key TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_key),
    logical_scan_run_id TEXT NOT NULL,
    security_id TEXT NOT NULL,
    evaluation_as_of TEXT NOT NULL,
    group_or_episode_id TEXT NOT NULL,
    universe_build_id TEXT NOT NULL,
    snapshot_run_id TEXT NOT NULL,
    candidate_generation_id TEXT NOT NULL,
    candidate_sha TEXT NOT NULL,
    semantic_closure_hash TEXT NOT NULL,
    runtime_config_hash TEXT NOT NULL,
    data_provenance_hash TEXT NOT NULL,
    developer_visible INTEGER NOT NULL DEFAULT 1,
    user_visible INTEGER NOT NULL DEFAULT 0,
    exposure_type TEXT NOT NULL,
    exposed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 7. Holdout Exclusions Table (1:N with observation)
CREATE TABLE IF NOT EXISTS holdout_exclusions (
    exclusion_id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL REFERENCES shadow_observations(observation_id),
    observation_key TEXT NOT NULL,
    exclusion_type TEXT NOT NULL CHECK(exclusion_type IN ('CASE', 'EPISODE_GROUP', 'SECURITY_TIME_WINDOW')),
    exclusion_hash TEXT NOT NULL,
    security_id TEXT NOT NULL,
    evaluation_as_of TEXT NOT NULL,
    exclusion_reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(exclusion_type, exclusion_hash)
);

-- 8. Shadow Evidence Admissions Table (1:1 with observation, Authoritative Denominator Source)
CREATE TABLE IF NOT EXISTS shadow_evidence_admissions (
    admission_id TEXT PRIMARY KEY,
    observation_id TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_id),
    observation_key TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_key),
    logical_scan_run_id TEXT NOT NULL,
    origin_class TEXT NOT NULL CHECK(origin_class IN ('NATURAL_PRODUCTION', 'NON_EVIDENCE_BOOTSTRAP', 'SYNTHETIC', 'REPLAY', 'ADMIN_FORCED', 'TEST', 'NOT_ADMITTED')),
    candidate_generation_id TEXT NOT NULL,
    candidate_sha TEXT NOT NULL,
    semantic_closure_hash TEXT NOT NULL,
    receipt_hash TEXT NOT NULL,
    admitted_at TEXT NOT NULL
);

-- 9. Shadow Outbox Table (Transactional outbox for decoupled dispatch)
CREATE TABLE IF NOT EXISTS shadow_outbox (
    outbox_id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL REFERENCES shadow_observations(observation_id),
    observation_key TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    created_at TEXT NOT NULL
);

-- Triggers for Immutability Enforcement
CREATE TRIGGER IF NOT EXISTS prevent_logical_scan_runs_update
BEFORE UPDATE ON logical_scan_runs
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_LOGICAL_SCAN_RUNS_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_logical_scan_runs_delete
BEFORE DELETE ON logical_scan_runs
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_LOGICAL_SCAN_RUNS_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_scan_attempts_update
BEFORE UPDATE ON scan_attempts
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_SCAN_ATTEMPTS_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_scan_attempts_delete
BEFORE DELETE ON scan_attempts
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_SCAN_ATTEMPTS_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_provenance_conflicts_update
BEFORE UPDATE ON provenance_conflicts
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_PROVENANCE_CONFLICTS_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_provenance_conflicts_delete
BEFORE DELETE ON provenance_conflicts
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_PROVENANCE_CONFLICTS_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_prospective_decision_update
BEFORE UPDATE ON prospective_decisions
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_PROSPECTIVE_DECISION_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_prospective_decision_delete
BEFORE DELETE ON prospective_decisions
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_PROSPECTIVE_DECISION_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_admission_update
BEFORE UPDATE ON shadow_evidence_admissions
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_SHADOW_ADMISSION_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_admission_delete
BEFORE DELETE ON shadow_evidence_admissions
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_SHADOW_ADMISSION_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_observation_update
BEFORE UPDATE ON shadow_observations
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_SHADOW_OBSERVATION_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_observation_delete
BEFORE DELETE ON shadow_observations
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_SHADOW_OBSERVATION_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_exposure_update
BEFORE UPDATE ON production_exposures
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_PRODUCTION_EXPOSURE_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_exposure_delete
BEFORE DELETE ON production_exposures
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_PRODUCTION_EXPOSURE_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_exclusion_update
BEFORE UPDATE ON holdout_exclusions
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_HOLDOUT_EXCLUSION_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_exclusion_delete
BEFORE DELETE ON holdout_exclusions
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_HOLDOUT_EXCLUSION_PROHIBITED');
END;
"""

CANONICAL_DDL_HASH: str = hashlib.sha256(DDL_SCHEMA.strip().encode("utf-8")).hexdigest()


def resolve_shadow_db_path(custom_path: Optional[str] = None) -> str:
    """Resolves authoritative path to persistent SQLite database."""
    if custom_path:
        return custom_path

    env_path = os.getenv("ARX_SHADOW_DB_PATH")
    if env_path:
        return env_path

    # Check for production persistent volume mount
    try:
        from analyst_dashboard.governance.storage import (
            is_production_runtime,
            resolve_persistent_volume_root,
        )
        if is_production_runtime():
            vol_root = resolve_persistent_volume_root()
            data_dir = os.path.join(vol_root, "analyst_dashboard", "data")
            os.makedirs(data_dir, exist_ok=True)
            return os.path.join(data_dir, DEFAULT_SHADOW_DB_FILENAME)
    except Exception:
        pass

    # Local default
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(repo_root, "analyst_dashboard", "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, DEFAULT_SHADOW_DB_FILENAME)


class Sprint3DurableEvidenceStore:
    """Authoritative durable storage engine for Sprint 3 production shadow evidence (Schema V3)."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = resolve_shadow_db_path(db_path)
        self._is_memory = (self.db_path == ":memory:")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # Explicit transaction control
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if not self._is_memory:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 30000;")
        return conn

    @retry_sqlite()
    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            # Check for existing V2 database and run migration if needed
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_observations';")
            has_obs = cur.fetchone() is not None

            if has_obs:
                cur.execute("PRAGMA table_info(shadow_observations);")
                columns = {r["name"] for r in cur.fetchall()}
                if "logical_scan_run_id" not in columns:
                    # Run Schema V3 migration
                    self._apply_migration_v3(conn)
                    return

            conn.executescript(DDL_SCHEMA)
        finally:
            conn.close()

    def _apply_migration_v3(self, conn: sqlite3.Connection) -> None:
        """Applies Schema V3 migration to existing Schema V2 database."""
        conn.execute("BEGIN IMMEDIATE;")
        try:
            # 1. Create new tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logical_scan_runs (
                    logical_scan_run_id TEXT PRIMARY KEY,
                    logical_trigger_id TEXT NOT NULL,
                    scanner_id TEXT NOT NULL,
                    universe_build_id TEXT NOT NULL,
                    evaluation_as_of TEXT NOT NULL,
                    scheduled_for TEXT,
                    invocation_class TEXT NOT NULL CHECK(invocation_class IN ('BOOT_WARMUP', 'SCHEDULED_PRODUCTION', 'PRODUCT_LIFECYCLE', 'MANUAL_OPERATOR', 'REPLAY', 'TEST', 'SYNTHETIC', 'PROVENANCE_CONFLICT')),
                    origin_class TEXT NOT NULL CHECK(origin_class IN ('NATURAL_PRODUCTION', 'NON_EVIDENCE_BOOTSTRAP', 'SYNTHETIC', 'REPLAY', 'ADMIN_FORCED', 'TEST', 'NOT_ADMITTED')),
                    originating_principal_type TEXT NOT NULL,
                    originating_principal_id TEXT NOT NULL,
                    immediate_caller_principal_id TEXT,
                    scheduler_job_id TEXT,
                    scheduler_event_id TEXT,
                    startup_context INTEGER NOT NULL DEFAULT 0,
                    boot_instance_id TEXT,
                    product_request_id TEXT,
                    replay_of_logical_scan_run_id TEXT,
                    classification_policy_version TEXT NOT NULL,
                    identity_schema_version TEXT NOT NULL,
                    canonicalization_version TEXT NOT NULL,
                    provenance_fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    classified_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'CREATED'
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    logical_scan_run_id TEXT NOT NULL REFERENCES logical_scan_runs(logical_scan_run_id),
                    delivery_attempt_id TEXT NOT NULL,
                    execution_attempt_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    worker_id TEXT NOT NULL,
                    process_id INTEGER NOT NULL,
                    deployment_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    failure_class TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provenance_conflicts (
                    conflict_record_id TEXT PRIMARY KEY,
                    logical_scan_run_id TEXT NOT NULL,
                    conflict_code TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)

            # 2. Add columns to existing tables
            conn.execute("ALTER TABLE shadow_observations ADD COLUMN logical_scan_run_id TEXT DEFAULT 'LEGACY_MIGRATION_RUN';")
            conn.execute("ALTER TABLE shadow_observations ADD COLUMN immutable_payload_hash TEXT DEFAULT 'LEGACY_PAYLOAD_HASH';")
            conn.execute("ALTER TABLE shadow_observations ADD COLUMN provenance_fingerprint TEXT DEFAULT 'LEGACY_PROVENANCE_FINGERPRINT';")

            conn.execute("ALTER TABLE prospective_decisions ADD COLUMN logical_scan_run_id TEXT DEFAULT 'LEGACY_MIGRATION_RUN';")
            conn.execute("ALTER TABLE prospective_decisions ADD COLUMN immutable_payload_hash TEXT DEFAULT 'LEGACY_PAYLOAD_HASH';")

            conn.execute("ALTER TABLE production_exposures ADD COLUMN logical_scan_run_id TEXT DEFAULT 'LEGACY_MIGRATION_RUN';")
            conn.execute("ALTER TABLE shadow_evidence_admissions ADD COLUMN logical_scan_run_id TEXT DEFAULT 'LEGACY_MIGRATION_RUN';")

            # 3. Create new triggers
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_logical_scan_runs_update
                BEFORE UPDATE ON logical_scan_runs
                BEGIN
                    SELECT RAISE(ABORT, 'MUTATION_OF_LOGICAL_SCAN_RUNS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_logical_scan_runs_delete
                BEFORE DELETE ON logical_scan_runs
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE_OF_LOGICAL_SCAN_RUNS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_scan_attempts_update
                BEFORE UPDATE ON scan_attempts
                BEGIN
                    SELECT RAISE(ABORT, 'MUTATION_OF_SCAN_ATTEMPTS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_scan_attempts_delete
                BEFORE DELETE ON scan_attempts
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE_OF_SCAN_ATTEMPTS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_provenance_conflicts_update
                BEFORE UPDATE ON provenance_conflicts
                BEGIN
                    SELECT RAISE(ABORT, 'MUTATION_OF_PROVENANCE_CONFLICTS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_provenance_conflicts_delete
                BEFORE DELETE ON provenance_conflicts
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE_OF_PROVENANCE_CONFLICTS_PROHIBITED');
                END;
            """)
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    def validate_provenance(
        self,
        logical_trigger_id: str,
        logical_scan_run_id: str,
        originating_principal_type: str,
        originating_principal_id: str,
        invocation_class: str,
        origin_class: str,
        scheduler_job_id: Optional[str] = None,
        scheduler_event_id: Optional[str] = None,
        startup_context: bool = False,
        product_request_id: Optional[str] = None,
        replay_of_logical_scan_run_id: Optional[str] = None,
        classification_policy_version: str = CLASSIFICATION_POLICY_VERSION,
        identity_schema_version: str = IDENTITY_SCHEMA_VERSION,
        caller_delegation_valid: bool = True,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Validates invocation provenance fail-closed against PROV-001 through PROV-010."""
        # PROV-010: UNTRUSTED_DELEGATION_CHAIN
        if not caller_delegation_valid:
            raise ProvenanceConflictError(
                "PROV-010",
                "Untrusted delegation chain detected in invocation headers",
                {"principal": originating_principal_id},
            )

        # PROV-002: HUMAN_CALLER_CLAIMS_SCHEDULED_ORIGIN
        if originating_principal_type in ("HUMAN_OPERATOR", "OPERATOR"):
            if invocation_class == "SCHEDULED_PRODUCTION" or origin_class == "NATURAL_PRODUCTION":
                raise ProvenanceConflictError(
                    "PROV-002",
                    "Human operator principal cannot claim scheduled production origin",
                    {"principal": originating_principal_id, "invocation_class": invocation_class},
                )

        # PROV-001: SCHEDULER_IDENTITY_WITHOUT_VALID_EVENT
        if originating_principal_type == "SCHEDULER" or invocation_class == "SCHEDULED_PRODUCTION":
            if not scheduler_job_id or not scheduler_event_id:
                raise ProvenanceConflictError(
                    "PROV-001",
                    "Scheduler invocation missing valid scheduler_job_id or scheduler_event_id",
                    {"scheduler_job_id": scheduler_job_id, "scheduler_event_id": scheduler_event_id},
                )

        # PROV-003: STARTUP_CONTEXT_WITH_NATURAL_SCHEDULED_CLASS
        if startup_context:
            if invocation_class == "SCHEDULED_PRODUCTION" or origin_class == "NATURAL_PRODUCTION":
                raise ProvenanceConflictError(
                    "PROV-003",
                    "Startup / boot context cannot claim scheduled production or natural evidence admission",
                    {"startup_context": startup_context, "origin_class": origin_class},
                )

        # PROV-004: REPLAY_METADATA_WITH_NATURAL_ORIGIN
        if replay_of_logical_scan_run_id:
            if invocation_class == "SCHEDULED_PRODUCTION" or origin_class == "NATURAL_PRODUCTION":
                raise ProvenanceConflictError(
                    "PROV-004",
                    "Replay run cannot claim natural production origin class",
                    {"replay_of": replay_of_logical_scan_run_id, "origin_class": origin_class},
                )

        # PROV-005: REPLAY_PARENT_NOT_FOUND
        if invocation_class == "REPLAY":
            if not replay_of_logical_scan_run_id:
                raise ProvenanceConflictError(
                    "PROV-005",
                    "Replay invocation missing parent logical scan run id",
                    {"invocation_class": invocation_class},
                )
            if conn is not None and not replay_of_logical_scan_run_id.startswith("parent-legacy-"):
                cur = conn.cursor()
                cur.execute(
                    "SELECT 1 FROM logical_scan_runs WHERE logical_scan_run_id = ?",
                    (replay_of_logical_scan_run_id,),
                )
                if cur.fetchone() is None:
                    raise ProvenanceConflictError(
                        "PROV-005",
                        f"Replay parent logical scan run '{replay_of_logical_scan_run_id}' not found in database",
                        {"replay_of": replay_of_logical_scan_run_id},
                    )

        # PROV-008: CLASSIFICATION_POLICY_VERSION_CONFLICT
        if classification_policy_version != CLASSIFICATION_POLICY_VERSION:
            raise ProvenanceConflictError(
                "PROV-008",
                f"Classification policy version conflict: expected '{CLASSIFICATION_POLICY_VERSION}', got '{classification_policy_version}'",
                {"expected": CLASSIFICATION_POLICY_VERSION, "actual": classification_policy_version},
            )


        # Existing run checks (PROV-006, PROV-007, PROV-009)
        if conn is not None:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT originating_principal_type, originating_principal_id,
                       scheduler_job_id, scheduler_event_id, provenance_fingerprint
                FROM logical_scan_runs WHERE logical_scan_run_id = ?
                """,
                (logical_scan_run_id,),
            )
            existing = cur.fetchone()
            if existing:
                # PROV-006: ORIGINATING_PRINCIPAL_CHANGED_ON_RETRY
                if (
                    (existing["originating_principal_id"] or "") != (originating_principal_id or "") or
                    (existing["originating_principal_type"] or "") != (originating_principal_type or "")
                ):
                    raise ProvenanceConflictError(
                        "PROV-006",
                        "Originating principal changed on logical run retry",
                        {"existing": existing["originating_principal_id"], "new": originating_principal_id},
                    )

                # PROV-007: SCHEDULER_EVENT_CHANGED_ON_RETRY
                if originating_principal_type == "SCHEDULER" or invocation_class == "SCHEDULED_PRODUCTION":
                    if (
                        (existing["scheduler_event_id"] or "") != (scheduler_event_id or "") or
                        (existing["scheduler_job_id"] or "") != (scheduler_job_id or "")
                    ):
                        raise ProvenanceConflictError(
                            "PROV-007",
                            "Scheduler event or job identity changed on logical run retry",
                            {"existing_event": existing["scheduler_event_id"], "new_event": scheduler_event_id},
                        )

                # PROV-009: SAME_LOGICAL_RUN_DIFFERENT_PROVENANCE
                computed_fp = compute_provenance_fingerprint(
                    logical_trigger_id=logical_trigger_id,
                    logical_scan_run_id=logical_scan_run_id,
                    originating_principal_type=originating_principal_type,
                    originating_principal_id=originating_principal_id,
                    invocation_class=invocation_class,
                    origin_class=origin_class,
                    scheduler_job_id=scheduler_job_id,
                    scheduler_event_id=scheduler_event_id,
                    startup_context=startup_context,
                    product_request_id=product_request_id,
                    replay_of_logical_scan_run_id=replay_of_logical_scan_run_id,
                    classification_policy_version=classification_policy_version,
                    identity_schema_version=identity_schema_version,
                )
                if existing["provenance_fingerprint"] != computed_fp:
                    raise ProvenanceConflictError(
                        "PROV-009",
                        "Same logical run submitted with conflicting provenance fingerprint",
                        {"existing_fp": existing["provenance_fingerprint"], "new_fp": computed_fp},
                    )

    @retry_sqlite()
    def record_provenance_conflict(
        self,
        logical_scan_run_id: str,
        conflict_code: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Records a provenance conflict fail-closed in the provenance_conflicts table."""
        conflict_record_id = f"conf-{conflict_code}-{time.time_ns()}-{os.urandom(4).hex()}"
        now_utc = get_offset_aware_utc_now()
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO provenance_conflicts (
                    conflict_record_id, logical_scan_run_id, conflict_code,
                    reason, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conflict_record_id,
                    logical_scan_run_id,
                    conflict_code,
                    reason,
                    json.dumps(details or {}),
                    now_utc,
                ),
            )
            conn.execute("COMMIT;")
            return conflict_record_id
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    @retry_sqlite()
    def ensure_logical_scan_run(
        self,
        logical_scan_run_id: str,
        logical_trigger_id: str,
        scanner_id: str,
        universe_build_id: str,
        evaluation_as_of: str,
        invocation_class: str,
        origin_class: str,
        originating_principal_type: str,
        originating_principal_id: str,
        immediate_caller_principal_id: Optional[str] = None,
        scheduler_job_id: Optional[str] = None,
        scheduler_event_id: Optional[str] = None,
        scheduled_for: Optional[str] = None,
        startup_context: bool = False,
        boot_instance_id: Optional[str] = None,
        product_request_id: Optional[str] = None,
        replay_of_logical_scan_run_id: Optional[str] = None,
        classification_policy_version: str = CLASSIFICATION_POLICY_VERSION,
        identity_schema_version: str = IDENTITY_SCHEMA_VERSION,
        canonicalization_version: str = CANONICALIZATION_VERSION,
    ) -> Dict[str, Any]:
        """Ensures logical scan run exists or validates retry consistency."""
        now_utc = get_offset_aware_utc_now()
        fp = compute_provenance_fingerprint(
            logical_trigger_id=logical_trigger_id,
            logical_scan_run_id=logical_scan_run_id,
            originating_principal_type=originating_principal_type,
            originating_principal_id=originating_principal_id,
            invocation_class=invocation_class,
            origin_class=origin_class,
            scheduler_job_id=scheduler_job_id,
            scheduler_event_id=scheduler_event_id,
            startup_context=startup_context,
            product_request_id=product_request_id,
            replay_of_logical_scan_run_id=replay_of_logical_scan_run_id,
            classification_policy_version=classification_policy_version,
            identity_schema_version=identity_schema_version,
            canonicalization_version=canonicalization_version,
        )

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()

            # Validate provenance rules
            self.validate_provenance(
                logical_trigger_id=logical_trigger_id,
                logical_scan_run_id=logical_scan_run_id,
                originating_principal_type=originating_principal_type,
                originating_principal_id=originating_principal_id,
                invocation_class=invocation_class,
                origin_class=origin_class,
                scheduler_job_id=scheduler_job_id,
                scheduler_event_id=scheduler_event_id,
                startup_context=startup_context,
                product_request_id=product_request_id,
                replay_of_logical_scan_run_id=replay_of_logical_scan_run_id,
                classification_policy_version=classification_policy_version,
                identity_schema_version=identity_schema_version,
                conn=conn,
            )

            cur.execute(
                "SELECT logical_scan_run_id, provenance_fingerprint FROM logical_scan_runs WHERE logical_scan_run_id = ?",
                (logical_scan_run_id,),
            )
            existing = cur.fetchone()
            if existing:
                conn.execute("COMMIT;")
                return {
                    "logical_scan_run_id": existing["logical_scan_run_id"],
                    "provenance_fingerprint": existing["provenance_fingerprint"],
                    "status": "EXISTING_LOGICAL_RUN",
                }

            cur.execute(
                """
                INSERT INTO logical_scan_runs (
                    logical_scan_run_id, logical_trigger_id, scanner_id, universe_build_id,
                    evaluation_as_of, scheduled_for, invocation_class, origin_class,
                    originating_principal_type, originating_principal_id, immediate_caller_principal_id,
                    scheduler_job_id, scheduler_event_id, startup_context, boot_instance_id,
                    product_request_id, replay_of_logical_scan_run_id, classification_policy_version,
                    identity_schema_version, canonicalization_version, provenance_fingerprint,
                    created_at, classified_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    logical_scan_run_id, logical_trigger_id, scanner_id, universe_build_id,
                    evaluation_as_of, scheduled_for, invocation_class, origin_class,
                    originating_principal_type, originating_principal_id, immediate_caller_principal_id,
                    scheduler_job_id, scheduler_event_id, 1 if startup_context else 0, boot_instance_id,
                    product_request_id, replay_of_logical_scan_run_id, classification_policy_version,
                    identity_schema_version, canonicalization_version, fp,
                    now_utc, now_utc, "CREATED"
                ),
            )
            conn.execute("COMMIT;")
            return {
                "logical_scan_run_id": logical_scan_run_id,
                "provenance_fingerprint": fp,
                "status": "CREATED",
            }
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    @retry_sqlite()
    def record_scan_attempt(
        self,
        logical_scan_run_id: str,
        delivery_attempt_id: str,
        execution_attempt_id: str,
        attempt_number: int,
        worker_id: str,
        process_id: int,
        deployment_id: str,
        failure_class: Optional[str] = None,
    ) -> str:
        """Records a physical delivery / execution attempt without altering logical run or denominator."""
        attempt_id = f"att-{delivery_attempt_id[:16]}-{attempt_number}"
        now_utc = get_offset_aware_utc_now()
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO scan_attempts (
                    attempt_id, logical_scan_run_id, delivery_attempt_id, execution_attempt_id,
                    attempt_number, worker_id, process_id, deployment_id, started_at,
                    completed_at, failure_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id, logical_scan_run_id, delivery_attempt_id, execution_attempt_id,
                    attempt_number, worker_id, process_id, deployment_id, now_utc,
                    now_utc if failure_class is None else None, failure_class
                ),
            )
            conn.execute("COMMIT;")
            return attempt_id
        except Exception:
            conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    @retry_sqlite()
    def admit_observation_bundle(
        self,
        security_id: str,
        evaluation_as_of: str,
        universe_build_id: str,
        snapshot_run_id: str,
        candidate_generation_id: str,
        candidate_sha: str,
        semantic_closure_hash: str,
        runtime_config_hash: str,
        dependency_lock_hash: str,
        data_provenance_hash: str,
        ruleset_id: str,
        ruleset_version: str,
        predicate_vector_hash: str,
        classification: str,
        decision_posture: str,
        input_fingerprint: str,
        group_or_episode_id: str,
        origin_class: Optional[str] = None,
        invocation_class: Optional[str] = None,
        logical_scan_run_id: Optional[str] = None,
        logical_trigger_id: Optional[str] = None,
        originating_principal_type: Optional[str] = None,
        originating_principal_id: Optional[str] = None,
        immediate_caller_principal_id: Optional[str] = None,
        scheduler_job_id: Optional[str] = None,
        scheduler_event_id: Optional[str] = None,
        scheduled_for: Optional[str] = None,
        startup_context: bool = False,
        boot_instance_id: Optional[str] = None,
        product_request_id: Optional[str] = None,
        replay_of_logical_scan_run_id: Optional[str] = None,
        delivery_attempt_id: Optional[str] = None,
        execution_attempt_id: Optional[str] = None,
        caller_delegation_valid: bool = True,
        developer_visible: bool = True,
        user_visible: bool = False,
        exposure_type: str = "PRODUCTION_SHADOW_OBSERVATION",
        scanner_id: str = "MINERVINI_VCP",
        failure_injection_point: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically admits a complete evidence bundle in one single database transaction (Schema V3).

        Enforces:
        - Logical invocation authority & attempt tracking.
        - Provenance conflict fail-closed validation (PROV-001 through PROV-010).
        - Payload content immutability (SAME_KEY_DIFFERENT_PAYLOAD_REJECTED = HardIntegrityFailureError).
        - Single-transaction atomicity with rollback on failure injection points A-G.
        """
        # Resolve logical run id & trigger id
        eff_logical_run_id = logical_scan_run_id or snapshot_run_id or f"log-run-{int(time.time()*1000)}"
        eff_logical_trigger_id = logical_trigger_id or f"trig-{eff_logical_run_id}"

        # Resolve invocation & origin classes
        if invocation_class is not None:
            eff_invocation_class = invocation_class
            if eff_invocation_class == "BOOT_WARMUP":
                eff_origin_class = "NON_EVIDENCE_BOOTSTRAP"
            elif eff_invocation_class == "SCHEDULED_PRODUCTION":
                eff_origin_class = "NATURAL_PRODUCTION"
            elif eff_invocation_class == "PRODUCT_LIFECYCLE":
                eff_origin_class = "NATURAL_PRODUCTION" if caller_delegation_valid else "ADMIN_FORCED"
            elif eff_invocation_class == "MANUAL_OPERATOR":
                eff_origin_class = "ADMIN_FORCED"
            elif eff_invocation_class == "REPLAY":
                eff_origin_class = "REPLAY"
            elif eff_invocation_class == "TEST":
                eff_origin_class = "TEST"
            elif eff_invocation_class == "SYNTHETIC":
                eff_origin_class = "SYNTHETIC"
            elif eff_invocation_class == "PROVENANCE_CONFLICT":
                eff_origin_class = "NOT_ADMITTED"
            else:
                eff_origin_class = origin_class or "TEST"
        elif origin_class is not None:
            eff_origin_class = origin_class
            if eff_origin_class == "NATURAL_PRODUCTION":
                eff_invocation_class = "SCHEDULED_PRODUCTION"
            elif eff_origin_class == "NON_EVIDENCE_BOOTSTRAP":
                eff_invocation_class = "BOOT_WARMUP"
            elif eff_origin_class == "ADMIN_FORCED":
                eff_invocation_class = "MANUAL_OPERATOR"
            elif eff_origin_class == "REPLAY":
                eff_invocation_class = "REPLAY"
            elif eff_origin_class == "SYNTHETIC":
                eff_invocation_class = "SYNTHETIC"
            elif eff_origin_class == "TEST":
                eff_invocation_class = "TEST"
            else:
                eff_invocation_class = "TEST"
        else:
            eff_invocation_class = "SCHEDULED_PRODUCTION"
            eff_origin_class = "NATURAL_PRODUCTION"

        # Resolve principal defaults if not provided
        if originating_principal_type is None:
            if eff_invocation_class == "BOOT_WARMUP":
                eff_principal_type = "BOOTSTRAP"
                eff_principal_id = originating_principal_id or "bootstrap:container-warmup"
            elif eff_invocation_class == "SCHEDULED_PRODUCTION":
                eff_principal_type = "SCHEDULER"
                eff_principal_id = originating_principal_id or "scheduler:arx-daily-cadence"
            elif eff_invocation_class == "MANUAL_OPERATOR":
                eff_principal_type = "HUMAN_OPERATOR"
                eff_principal_id = originating_principal_id or "operator:manual"
            elif eff_invocation_class == "REPLAY":
                eff_principal_type = "REPLAY_CONTROLLER"
                eff_principal_id = originating_principal_id or "replay:controller"
            else:
                eff_principal_type = "TEST_RUNNER"
                eff_principal_id = originating_principal_id or "test:runner"
        else:
            eff_principal_type = originating_principal_type
            eff_principal_id = originating_principal_id or f"principal:{eff_principal_type.lower()}"

        # Compute deterministic observation key (V3 using eff_logical_run_id)
        observation_key = compute_deterministic_observation_key(
            scanner_id=scanner_id,
            security_id=security_id,
            evaluation_as_of=evaluation_as_of,
            universe_build_id=universe_build_id,
            logical_scan_run_id=eff_logical_run_id,
            candidate_generation_id=candidate_generation_id,
            semantic_closure_hash=semantic_closure_hash,
        )

        # Compute immutable payload hash
        immutable_payload_hash = compute_immutable_payload_hash(
            security_id=security_id,
            evaluation_as_of=evaluation_as_of,
            universe_build_id=universe_build_id,
            candidate_generation_id=candidate_generation_id,
            candidate_sha=candidate_sha,
            semantic_closure_hash=semantic_closure_hash,
            runtime_config_hash=runtime_config_hash,
            dependency_lock_hash=dependency_lock_hash,
            data_provenance_hash=data_provenance_hash,
            ruleset_id=ruleset_id,
            ruleset_version=ruleset_version,
            predicate_vector_hash=predicate_vector_hash,
            classification=classification,
            decision_posture=decision_posture,
            input_fingerprint=input_fingerprint,
        )

        # Compute provenance fingerprint
        provenance_fingerprint = compute_provenance_fingerprint(
            logical_trigger_id=eff_logical_trigger_id,
            logical_scan_run_id=eff_logical_run_id,
            originating_principal_type=eff_principal_type,
            originating_principal_id=eff_principal_id,
            invocation_class=eff_invocation_class,
            origin_class=eff_origin_class,
            scheduler_job_id=scheduler_job_id,
            scheduler_event_id=scheduler_event_id,
            startup_context=startup_context,
            product_request_id=product_request_id,
            replay_of_logical_scan_run_id=replay_of_logical_scan_run_id,
        )

        observation_id = f"obs-{observation_key[:20]}"
        decision_record_id = f"dec-{observation_key[:20]}"
        exposure_id = f"exp-{observation_key[:20]}"
        admission_id = f"adm-{observation_key[:20]}"
        now_utc = get_offset_aware_utc_now()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()

            # 1. Check Provenance & Fail Closed on Conflict
            try:
                self.validate_provenance(
                    logical_trigger_id=eff_logical_trigger_id,
                    logical_scan_run_id=eff_logical_run_id,
                    originating_principal_type=eff_principal_type,
                    originating_principal_id=eff_principal_id,
                    invocation_class=eff_invocation_class,
                    origin_class=eff_origin_class,
                    scheduler_job_id=scheduler_job_id,
                    scheduler_event_id=scheduler_event_id,
                    startup_context=startup_context,
                    product_request_id=product_request_id,
                    replay_of_logical_scan_run_id=replay_of_logical_scan_run_id,
                    caller_delegation_valid=caller_delegation_valid,
                    conn=conn,
                )
            except ProvenanceConflictError as pce:
                # Record conflict in table
                cur.execute(
                    """
                    INSERT INTO provenance_conflicts (
                        conflict_record_id, logical_scan_run_id, conflict_code,
                        reason, details_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"conf-{pce.code}-{time.time_ns()}-{os.urandom(4).hex()}",
                        eff_logical_run_id,
                        pce.code,
                        str(pce),
                        json.dumps(pce.details),
                        now_utc,
                    ),
                )
                conn.execute("COMMIT;")
                raise

            # 2. Idempotency & Payload Immutability Check
            cur.execute(
                """
                SELECT a.admission_id, a.origin_class, a.receipt_hash, a.admitted_at,
                       o.immutable_payload_hash
                FROM shadow_evidence_admissions a
                JOIN shadow_observations o ON a.observation_key = o.observation_key
                WHERE a.observation_key = ?
                """,
                (observation_key,),
            )
            existing = cur.fetchone()
            if existing:
                existing_hash = existing["immutable_payload_hash"]
                if existing_hash != immutable_payload_hash:
                    conn.execute("ROLLBACK;")
                    raise HardIntegrityFailureError(
                        f"SAME_KEY_DIFFERENT_PAYLOAD_REJECTED: Existing observation key '{observation_key}' "
                        f"has immutable payload hash '{existing_hash}', which conflicts with incoming payload hash '{immutable_payload_hash}'."
                    )
                conn.execute("COMMIT;")
                return {
                    "status": "ALREADY_ADMITTED",
                    "observation_key": observation_key,
                    "observation_id": observation_id,
                    "logical_scan_run_id": eff_logical_run_id,
                    "admission_id": existing["admission_id"],
                    "origin_class": existing["origin_class"],
                    "receipt_hash": existing["receipt_hash"],
                    "admitted_at": existing["admitted_at"],
                    "immutable_payload_hash": existing_hash,
                    "provenance_fingerprint": provenance_fingerprint,
                    "new_admission": False,
                }

            # 3. Ensure Logical Scan Run exists
            cur.execute(
                "SELECT logical_scan_run_id FROM logical_scan_runs WHERE logical_scan_run_id = ?",
                (eff_logical_run_id,),
            )
            if cur.fetchone() is None:
                cur.execute(
                    """
                    INSERT INTO logical_scan_runs (
                        logical_scan_run_id, logical_trigger_id, scanner_id, universe_build_id,
                        evaluation_as_of, scheduled_for, invocation_class, origin_class,
                        originating_principal_type, originating_principal_id, immediate_caller_principal_id,
                        scheduler_job_id, scheduler_event_id, startup_context, boot_instance_id,
                        product_request_id, replay_of_logical_scan_run_id, classification_policy_version,
                        identity_schema_version, canonicalization_version, provenance_fingerprint,
                        created_at, classified_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eff_logical_run_id, eff_logical_trigger_id, scanner_id, universe_build_id,
                        evaluation_as_of, scheduled_for, eff_invocation_class, eff_origin_class,
                        eff_principal_type, eff_principal_id, immediate_caller_principal_id,
                        scheduler_job_id, scheduler_event_id, 1 if startup_context else 0, boot_instance_id,
                        product_request_id, replay_of_logical_scan_run_id, CLASSIFICATION_POLICY_VERSION,
                        IDENTITY_SCHEMA_VERSION, CANONICALIZATION_VERSION, provenance_fingerprint,
                        now_utc, now_utc, "CREATED"
                    ),
                )

            # Record attempt if attempt IDs provided
            if delivery_attempt_id and execution_attempt_id:
                attempt_id = f"att-{delivery_attempt_id[:16]}-1"
                cur.execute(
                    """
                    INSERT OR IGNORE INTO scan_attempts (
                        attempt_id, logical_scan_run_id, delivery_attempt_id, execution_attempt_id,
                        attempt_number, worker_id, process_id, deployment_id, started_at, completed_at, failure_class
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt_id, eff_logical_run_id, delivery_attempt_id, execution_attempt_id,
                        1, "worker-1", os.getpid(), "deployment-default", now_utc, now_utc, None
                    ),
                )

            # 4. Insert shadow_observations
            cur.execute(
                """
                INSERT INTO shadow_observations (
                    observation_id, observation_key, logical_scan_run_id, scanner_id, scanner_run_id,
                    security_id, evaluation_as_of, candidate_generation_id, candidate_sha,
                    semantic_closure_hash, universe_build_id, origin_class,
                    immutable_payload_hash, provenance_fingerprint, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id, observation_key, eff_logical_run_id, scanner_id, snapshot_run_id,
                    security_id, evaluation_as_of, candidate_generation_id, candidate_sha,
                    semantic_closure_hash, universe_build_id, eff_origin_class,
                    immutable_payload_hash, provenance_fingerprint, now_utc
                ),
            )
            if failure_injection_point == "A":
                raise RuntimeError("FAILURE_INJECTION_A: Simulated failure after shadow_observation insert")

            # 5. Insert prospective_decisions
            cur.execute(
                """
                INSERT INTO prospective_decisions (
                    decision_record_id, observation_id, observation_key, logical_scan_run_id,
                    evaluation_as_of, known_at, security_id, universe_build_id, snapshot_run_id,
                    candidate_generation_id, candidate_sha, semantic_closure_hash,
                    runtime_config_hash, dependency_lock_hash, data_provenance_hash,
                    ruleset_id, ruleset_version, predicate_vector_hash, classification,
                    decision_posture, input_fingerprint, immutable_payload_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_record_id, observation_id, observation_key, eff_logical_run_id,
                    evaluation_as_of, now_utc, security_id, universe_build_id, snapshot_run_id,
                    candidate_generation_id, candidate_sha, semantic_closure_hash,
                    runtime_config_hash, dependency_lock_hash, data_provenance_hash,
                    ruleset_id, ruleset_version, predicate_vector_hash, classification,
                    decision_posture, input_fingerprint, immutable_payload_hash, now_utc
                ),
            )
            if failure_injection_point == "B":
                raise RuntimeError("FAILURE_INJECTION_B: Simulated failure after prospective_decision insert")

            # 6. Insert production_exposures
            cur.execute(
                """
                INSERT INTO production_exposures (
                    exposure_id, observation_id, observation_key, logical_scan_run_id,
                    security_id, evaluation_as_of, group_or_episode_id, universe_build_id,
                    snapshot_run_id, candidate_generation_id, candidate_sha,
                    semantic_closure_hash, runtime_config_hash, data_provenance_hash,
                    developer_visible, user_visible, exposure_type, exposed_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exposure_id, observation_id, observation_key, eff_logical_run_id,
                    security_id, evaluation_as_of, group_or_episode_id, universe_build_id,
                    snapshot_run_id, candidate_generation_id, candidate_sha,
                    semantic_closure_hash, runtime_config_hash, data_provenance_hash,
                    1 if developer_visible else 0, 1 if user_visible else 0,
                    exposure_type, now_utc, now_utc
                ),
            )
            if failure_injection_point == "C":
                raise RuntimeError("FAILURE_INJECTION_C: Simulated failure after production_exposure insert")

            # 7. Insert holdout exclusions
            case_hash = hashlib.sha256(f"CASE:{security_id}:{evaluation_as_of}".encode("utf-8")).hexdigest()
            cur.execute(
                """
                INSERT OR IGNORE INTO holdout_exclusions (
                    exclusion_id, observation_id, observation_key, exclusion_type,
                    exclusion_hash, security_id, evaluation_as_of, exclusion_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"excl-case-{case_hash[:16]}", observation_id, observation_key, "CASE",
                    case_hash, security_id, evaluation_as_of, "PRODUCTION_SHADOW_EXPOSURE", now_utc
                ),
            )
            if failure_injection_point == "D":
                raise RuntimeError("FAILURE_INJECTION_D: Simulated failure after case exclusion insert")

            group_hash = hashlib.sha256(f"GROUP:{group_or_episode_id}".encode("utf-8")).hexdigest()
            cur.execute(
                """
                INSERT OR IGNORE INTO holdout_exclusions (
                    exclusion_id, observation_id, observation_key, exclusion_type,
                    exclusion_hash, security_id, evaluation_as_of, exclusion_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"excl-grp-{group_hash[:16]}", observation_id, observation_key, "EPISODE_GROUP",
                    group_hash, security_id, evaluation_as_of, "PRODUCTION_SHADOW_EXPOSURE", now_utc
                ),
            )
            if failure_injection_point == "E":
                raise RuntimeError("FAILURE_INJECTION_E: Simulated failure after group exclusion insert")

            # 8. Insert shadow_evidence_admissions
            receipt_data = f"{admission_id}:{observation_key}:{eff_origin_class}:{candidate_sha}:{now_utc}"
            receipt_hash = hashlib.sha256(receipt_data.encode("utf-8")).hexdigest()

            if failure_injection_point == "F":
                raise RuntimeError("FAILURE_INJECTION_F: Simulated failure before admission insert")

            cur.execute(
                """
                INSERT INTO shadow_evidence_admissions (
                    admission_id, observation_id, observation_key, logical_scan_run_id,
                    origin_class, candidate_generation_id, candidate_sha,
                    semantic_closure_hash, receipt_hash, admitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    admission_id, observation_id, observation_key, eff_logical_run_id,
                    eff_origin_class, candidate_generation_id, candidate_sha,
                    semantic_closure_hash, receipt_hash, now_utc
                ),
            )

            # 9. Insert shadow_outbox
            cur.execute(
                """
                INSERT INTO shadow_outbox (
                    outbox_id, observation_id, observation_key, event_type,
                    payload_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"out-{observation_key[:20]}", observation_id, observation_key,
                    "SHADOW_OBSERVATION_ADMITTED",
                    json.dumps({
                        "security_id": security_id,
                        "origin_class": eff_origin_class,
                        "logical_scan_run_id": eff_logical_run_id,
                        "candidate_generation_id": candidate_generation_id,
                        "evaluation_as_of": evaluation_as_of,
                    }),
                    "PENDING",
                    now_utc
                ),
            )

            if failure_injection_point == "G":
                raise RuntimeError("FAILURE_INJECTION_G: Simulated failure immediately before COMMIT")

            conn.execute("COMMIT;")
            return {
                "status": "ADMITTED",
                "observation_key": observation_key,
                "observation_id": observation_id,
                "logical_scan_run_id": eff_logical_run_id,
                "decision_record_id": decision_record_id,
                "exposure_id": exposure_id,
                "admission_id": admission_id,
                "origin_class": eff_origin_class,
                "receipt_hash": receipt_hash,
                "admitted_at": now_utc,
                "immutable_payload_hash": immutable_payload_hash,
                "provenance_fingerprint": provenance_fingerprint,
                "new_admission": True,
            }
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        finally:
            conn.close()

    @retry_sqlite()
    def get_authoritative_denominator_counts(self) -> Dict[str, int]:
        """Derives authoritative denominator metrics strictly from committed database admissions."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT origin_class, COUNT(*) as cnt
                FROM shadow_evidence_admissions
                GROUP BY origin_class
                """
            )
            rows = cur.fetchall()
            counts = {
                "NATURAL_PRODUCTION": 0,
                "NON_EVIDENCE_BOOTSTRAP": 0,
                "SYNTHETIC": 0,
                "REPLAY": 0,
                "ADMIN_FORCED": 0,
                "TEST": 0,
                "NOT_ADMITTED": 0,
            }
            total = 0
            for r in rows:
                cls = r["origin_class"]
                c = r["cnt"]
                if cls in counts:
                    counts[cls] = c
                total += c
            return {
                "shadow_record_count": total,
                "natural_production_shadow_record_count": counts["NATURAL_PRODUCTION"],
                "non_evidence_bootstrap_record_count": counts["NON_EVIDENCE_BOOTSTRAP"],
                "synthetic_shadow_record_count": counts["SYNTHETIC"],
                "replay_shadow_record_count": counts["REPLAY"],
                "admin_forced_shadow_record_count": counts["ADMIN_FORCED"],
                "test_shadow_record_count": counts["TEST"],
            }
        finally:
            conn.close()

    @retry_sqlite()
    def audit_cross_ledger_integrity(self) -> Dict[str, Any]:
        """Executes canonical reconciliation queries ensuring zero partial bundles or orphaned rows."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()

            # 1. Orphaned prospective decisions (not in admissions)
            cur.execute(
                """
                SELECT COUNT(*) as cnt FROM prospective_decisions p
                LEFT JOIN shadow_evidence_admissions a ON p.observation_id = a.observation_id
                WHERE a.admission_id IS NULL
                """
            )
            orphaned_prospective = cur.fetchone()["cnt"]

            # 2. Orphaned exposures
            cur.execute(
                """
                SELECT COUNT(*) as cnt FROM production_exposures e
                LEFT JOIN shadow_evidence_admissions a ON e.observation_id = a.observation_id
                WHERE a.admission_id IS NULL
                """
            )
            orphaned_exposures = cur.fetchone()["cnt"]

            # 3. Admitted exposures missing required holdout exclusions
            cur.execute(
                """
                SELECT COUNT(*) as cnt FROM shadow_evidence_admissions a
                LEFT JOIN holdout_exclusions h ON a.observation_id = h.observation_id
                WHERE h.exclusion_id IS NULL
                """
            )
            orphaned_required_exclusions = cur.fetchone()["cnt"]

            # 4. Duplicate admissions by observation_key
            cur.execute(
                """
                SELECT COUNT(*) as cnt FROM (
                    SELECT observation_key, COUNT(*) as k_cnt
                    FROM shadow_evidence_admissions
                    GROUP BY observation_key
                    HAVING COUNT(*) > 1
                )
                """
            )
            duplicate_admissions = cur.fetchone()["cnt"]

            # 5. Row count parity
            cur.execute("SELECT COUNT(*) as cnt FROM shadow_observations")
            obs_cnt = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM prospective_decisions")
            dec_cnt = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM production_exposures")
            exp_cnt = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM shadow_evidence_admissions")
            adm_cnt = cur.fetchone()["cnt"]

            denominator_mismatch = 1 if not (obs_cnt == dec_cnt == exp_cnt == adm_cnt) else 0

            return {
                "orphaned_prospective_records": orphaned_prospective,
                "orphaned_exposure_records": orphaned_exposures,
                "orphaned_required_exclusions": orphaned_required_exclusions,
                "duplicate_admissions": duplicate_admissions,
                "denominator_mismatch": denominator_mismatch,
                "row_counts": {
                    "shadow_observations": obs_cnt,
                    "prospective_decisions": dec_cnt,
                    "production_exposures": exp_cnt,
                    "shadow_evidence_admissions": adm_cnt,
                },
                "integrity_status": "PASS" if (
                    orphaned_prospective == 0
                    and orphaned_exposures == 0
                    and orphaned_required_exclusions == 0
                    and duplicate_admissions == 0
                    and denominator_mismatch == 0
                ) else "FAIL",
            }
        finally:
            conn.close()

    @retry_sqlite()
    def get_prospective_decision(self, observation_key: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM prospective_decisions WHERE observation_key = ?", (observation_key,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    @retry_sqlite()
    def compute_prospective_content_hash(self, observation_key: str) -> Optional[str]:
        data = self.get_prospective_decision(observation_key)
        if not data:
            return None
        canonical_str = json.dumps({
            "decision_record_id": data["decision_record_id"],
            "observation_id": data["observation_id"],
            "observation_key": data["observation_key"],
            "evaluation_as_of": data["evaluation_as_of"],
            "security_id": data["security_id"],
            "universe_build_id": data["universe_build_id"],
            "snapshot_run_id": data["snapshot_run_id"],
            "candidate_generation_id": data["candidate_generation_id"],
            "candidate_sha": data["candidate_sha"],
            "semantic_closure_hash": data["semantic_closure_hash"],
            "runtime_config_hash": data["runtime_config_hash"],
            "dependency_lock_hash": data["dependency_lock_hash"],
            "data_provenance_hash": data["data_provenance_hash"],
            "ruleset_id": data["ruleset_id"],
            "ruleset_version": data["ruleset_version"],
            "predicate_vector_hash": data["predicate_vector_hash"],
            "classification": data["classification"],
            "decision_posture": data["decision_posture"],
            "input_fingerprint": data["input_fingerprint"],
        }, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
