"""ARX Terminal — Sprint 3 Durable Shadow Evidence Store & Relational Engine (Schema V4).

Authoritative persistent storage engine providing:
1. Strict relational schema with foreign keys, unique constraints, and check constraints (Schema 4.0.0).
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

SCHEMA_VERSION: str = "4.0.0"
MIGRATION_ID: str = "MIGRATION_20261010_004_NATURAL_EVIDENCE_EPOCH_AND_HISTORICAL_ISOLATION"
OBSERVATION_KEY_SPECIFICATION: str = (
    "sha256(scanner_id:security_id:evaluation_as_of:universe_build_id:logical_scan_run_id:candidate_generation_id:semantic_closure_hash)"
)
CLASSIFICATION_POLICY_VERSION: str = "1.0.0"
IDENTITY_SCHEMA_VERSION: str = "4.0.0"
CANONICALIZATION_VERSION: str = "1.0.0"
STORAGE_TOPOLOGY_REQUIREMENT: str = "SINGLE_REPLICA_ONLY"

NATURAL_EVIDENCE_EPOCH_MODEL_VERSION: str = "1.0.0"
NATURAL_EVIDENCE_EPOCH_ID: str = "SPRINT3_CANDIDATE004_EPOCH_001"
LOCAL_FREEZE_EPOCH_STATUS: str = "PRE_ACTIVATION"
DENOMINATOR_POLICY_VERSION: str = "1.0.0"
MIGRATION_MANIFEST_CANONICALIZATION_VERSION: str = "1.0.0"
PROVENANCE_CONTRACT_VERSION: str = "1.0.0"
HISTORICAL_MIGRATION_UNIT: str = "SHADOW_OBSERVATION_BUNDLE"

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
    candidate_generation_id: str = "CANDIDATE_GENERATION_004",
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


DDL_SCHEMA_V3 = """
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


DDL_SCHEMA_V4_ADDITIONS = """
-- 10. Natural Evidence Epochs Table
CREATE TABLE IF NOT EXISTS natural_evidence_epochs (
    epoch_id TEXT PRIMARY KEY,
    evidence_stream_id TEXT NOT NULL,
    candidate_generation_id TEXT NOT NULL,
    candidate_functional_sha TEXT NOT NULL,
    semantic_closure_hash TEXT NOT NULL,
    provenance_contract_version TEXT NOT NULL,
    identity_schema_version TEXT NOT NULL,
    denominator_policy_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('DEFINED', 'PRE_ACTIVATION', 'ACTIVE', 'CLOSED')),
    activation_receipt_id TEXT,
    activation_sequence INTEGER,
    activated_at TEXT,
    closed_receipt_id TEXT,
    closed_sequence INTEGER,
    closed_at TEXT,
    created_at TEXT NOT NULL
);

-- 11. Epoch Activation Receipts Table
CREATE TABLE IF NOT EXISTS epoch_activation_receipts (
    activation_receipt_id TEXT PRIMARY KEY,
    epoch_id TEXT NOT NULL REFERENCES natural_evidence_epochs(epoch_id),
    candidate_generation_id TEXT NOT NULL,
    candidate_functional_sha TEXT NOT NULL,
    semantic_closure_hash TEXT NOT NULL,
    database_schema_version TEXT NOT NULL,
    provenance_policy_version TEXT NOT NULL,
    identity_schema_version TEXT NOT NULL,
    denominator_policy_version TEXT NOT NULL,
    scheduler_contract_identity TEXT NOT NULL,
    activation_sequence INTEGER NOT NULL,
    activated_at TEXT NOT NULL,
    receipt_content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 12. Evidence Epoch Memberships Table
CREATE TABLE IF NOT EXISTS evidence_epoch_memberships (
    logical_scan_run_id TEXT PRIMARY KEY REFERENCES logical_scan_runs(logical_scan_run_id),
    epoch_id TEXT NOT NULL REFERENCES natural_evidence_epochs(epoch_id),
    membership_class TEXT NOT NULL CHECK(membership_class IN ('CURRENT_PROSPECTIVE_EPOCH', 'PRE_EPOCH_LEGACY', 'PRE_ACTIVATION_DELAYED_EVENT', 'NON_NATURAL', 'MIGRATION_CONFLICT')),
    prospective_disposition TEXT NOT NULL CHECK(prospective_disposition IN ('PROSPECTIVE_CANDIDATE', 'PRE_EPOCH_INELIGIBLE', 'NON_NATURAL_INELIGIBLE', 'MIGRATION_CONFLICT_INELIGIBLE')),
    activation_receipt_id TEXT,
    assignment_sequence INTEGER NOT NULL,
    assigned_at TEXT NOT NULL,
    membership_fingerprint TEXT NOT NULL
);

-- 13. Migration Source Manifests Table
CREATE TABLE IF NOT EXISTS migration_source_manifests (
    manifest_id TEXT PRIMARY KEY,
    canonicalization_version TEXT NOT NULL,
    source_unit_count INTEGER NOT NULL,
    population_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 14. Migration Unit Dispositions Table
CREATE TABLE IF NOT EXISTS migration_unit_dispositions (
    source_unit_id TEXT PRIMARY KEY,
    source_schema_version TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    migration_disposition TEXT NOT NULL CHECK(migration_disposition IN ('LEGACY_NO_CURRENT_PROVENANCE', 'MIGRATION_CONFLICT', 'PREEXISTING_EQUIVALENT_PROVENANCE')),
    legacy_recorded_origin_class TEXT NOT NULL,
    epoch_membership_class TEXT NOT NULL,
    prospective_disposition TEXT NOT NULL,
    migration_id TEXT NOT NULL,
    migration_run_id TEXT NOT NULL,
    migrated_at TEXT NOT NULL,
    disposition_hash TEXT NOT NULL
);

-- 15. Historical Reconciliation Records Table
CREATE TABLE IF NOT EXISTS historical_reconciliation_records (
    reconciliation_id TEXT PRIMARY KEY,
    source_unit_id TEXT NOT NULL,
    reconciled_principal_type TEXT,
    reconciled_principal_id TEXT,
    reconciled_invocation_class TEXT,
    reconciled_origin_class TEXT,
    justification TEXT NOT NULL,
    reconciled_at TEXT NOT NULL,
    reconciled_by TEXT NOT NULL
);

-- Triggers for Natural Evidence Epochs
CREATE TRIGGER IF NOT EXISTS prevent_natural_evidence_epochs_delete
BEFORE DELETE ON natural_evidence_epochs
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_NATURAL_EVIDENCE_EPOCHS_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS validate_natural_evidence_epochs_update
BEFORE UPDATE ON natural_evidence_epochs
BEGIN
    SELECT CASE
        WHEN OLD.status = 'CLOSED' THEN RAISE(ABORT, 'CLOSED_EPOCH_CANNOT_BE_MODIFIED')
        WHEN OLD.epoch_id != NEW.epoch_id THEN RAISE(ABORT, 'EPOCH_ID_UPDATE_REJECTED')
        WHEN OLD.candidate_generation_id != NEW.candidate_generation_id THEN RAISE(ABORT, 'EPOCH_CANDIDATE_GENERATION_UPDATE_REJECTED')
        WHEN OLD.candidate_functional_sha != NEW.candidate_functional_sha THEN RAISE(ABORT, 'EPOCH_FUNCTIONAL_SHA_UPDATE_REJECTED')
        WHEN OLD.semantic_closure_hash != NEW.semantic_closure_hash THEN RAISE(ABORT, 'EPOCH_SEMANTIC_CLOSURE_UPDATE_REJECTED')
        ELSE 1
    END;
END;

-- Triggers for Epoch Activation Receipts
CREATE TRIGGER IF NOT EXISTS prevent_epoch_activation_receipts_update
BEFORE UPDATE ON epoch_activation_receipts
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_EPOCH_ACTIVATION_RECEIPTS_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_epoch_activation_receipts_delete
BEFORE DELETE ON epoch_activation_receipts
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_EPOCH_ACTIVATION_RECEIPTS_PROHIBITED');
END;

-- Triggers for Evidence Epoch Memberships
CREATE TRIGGER IF NOT EXISTS prevent_evidence_epoch_memberships_update
BEFORE UPDATE ON evidence_epoch_memberships
BEGIN
    SELECT CASE
        WHEN OLD.epoch_id != NEW.epoch_id THEN RAISE(ABORT, 'EPOCH_ID_UPDATE_REJECTED')
        WHEN OLD.membership_class != NEW.membership_class THEN RAISE(ABORT, 'MEMBERSHIP_CLASS_UPDATE_REJECTED')
        WHEN OLD.prospective_disposition != NEW.prospective_disposition THEN RAISE(ABORT, 'PROSPECTIVE_DISPOSITION_UPDATE_REJECTED')
        ELSE RAISE(ABORT, 'MUTATION_OF_EVIDENCE_EPOCH_MEMBERSHIPS_PROHIBITED')
    END;
END;

CREATE TRIGGER IF NOT EXISTS prevent_evidence_epoch_memberships_delete
BEFORE DELETE ON evidence_epoch_memberships
BEGIN
    SELECT RAISE(ABORT, 'MEMBERSHIP_DELETE_REJECTED');
END;

-- Triggers for Migration Source Manifests
CREATE TRIGGER IF NOT EXISTS prevent_migration_source_manifests_update
BEFORE UPDATE ON migration_source_manifests
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_MIGRATION_SOURCE_MANIFESTS_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_migration_source_manifests_delete
BEFORE DELETE ON migration_source_manifests
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_MIGRATION_SOURCE_MANIFESTS_PROHIBITED');
END;

-- Triggers for Migration Unit Dispositions
CREATE TRIGGER IF NOT EXISTS prevent_migration_unit_dispositions_update
BEFORE UPDATE ON migration_unit_dispositions
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_MIGRATION_UNIT_DISPOSITIONS_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_migration_unit_dispositions_delete
BEFORE DELETE ON migration_unit_dispositions
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_MIGRATION_UNIT_DISPOSITIONS_PROHIBITED');
END;

-- Triggers for Historical Reconciliation Records
CREATE TRIGGER IF NOT EXISTS prevent_historical_reconciliation_records_update
BEFORE UPDATE ON historical_reconciliation_records
BEGIN
    SELECT RAISE(ABORT, 'MUTATION_OF_HISTORICAL_RECONCILIATION_PROHIBITED');
END;

CREATE TRIGGER IF NOT EXISTS prevent_historical_reconciliation_records_delete
BEFORE DELETE ON historical_reconciliation_records
BEGIN
    SELECT RAISE(ABORT, 'DELETE_OF_HISTORICAL_RECONCILIATION_PROHIBITED');
END;
"""

DDL_SCHEMA = DDL_SCHEMA_V3 + DDL_SCHEMA_V4_ADDITIONS
CANONICAL_DDL_HASH: str = hashlib.sha256(DDL_SCHEMA.strip().encode("utf-8")).hexdigest()
CANONICAL_DDL_HASH_V3: str = hashlib.sha256(DDL_SCHEMA_V3.strip().encode("utf-8")).hexdigest()



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

    def __init__(self, db_path: Optional[str] = None, schema_version: str = SCHEMA_VERSION) -> None:
        self.db_path = resolve_shadow_db_path(db_path)
        self.schema_version = schema_version
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
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_observations';")
            has_obs = cur.fetchone() is not None

            if self.schema_version == "3.0.0":
                if has_obs:
                    cur.execute("PRAGMA table_info(shadow_observations);")
                    columns = {r["name"] for r in cur.fetchall()}
                    if "logical_scan_run_id" not in columns:
                        self._apply_migration_v3(conn)
                        return
                conn.executescript(DDL_SCHEMA_V3)
                return

            # Default Schema 4.0.0
            if has_obs:
                cur.execute("PRAGMA table_info(shadow_observations);")
                columns = {r["name"] for r in cur.fetchall()}
                if "logical_scan_run_id" not in columns:
                    self._apply_migration_v3(conn)
                # Check for natural_evidence_epochs
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='natural_evidence_epochs';")
                has_epochs = cur.fetchone() is not None
                if not has_epochs:
                    self._apply_migration_v4(conn)
                    return

            conn.executescript(DDL_SCHEMA)
            # Ensure default candidate 004 epoch exists in PRE_ACTIVATION state
            cur.execute("SELECT epoch_id FROM natural_evidence_epochs WHERE epoch_id = ?", (NATURAL_EVIDENCE_EPOCH_ID,))
            if not cur.fetchone():
                now_utc = get_offset_aware_utc_now()
                conn.execute(
                    """
                    INSERT OR IGNORE INTO natural_evidence_epochs (
                        epoch_id, evidence_stream_id, candidate_generation_id,
                        candidate_functional_sha, semantic_closure_hash,
                        provenance_contract_version, identity_schema_version,
                        denominator_policy_version, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        NATURAL_EVIDENCE_EPOCH_ID,
                        "ARX_RADAR_SPRINT_3_SHADOW_STREAM",
                        "CANDIDATE_GENERATION_004",
                        "07b8b40cdb82328087b0f12adb08928a53e0234b",
                        "53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6",
                        PROVENANCE_CONTRACT_VERSION,
                        IDENTITY_SCHEMA_VERSION,
                        DENOMINATOR_POLICY_VERSION,
                        LOCAL_FREEZE_EPOCH_STATUS,
                        now_utc,
                    ),
                )
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
    def _apply_migration_v4(self, conn: sqlite3.Connection, migration_run_id: Optional[str] = None) -> Dict[str, Any]:
        """Applies Schema V4 migration with deterministic source manifest and historical unit dispositions."""
        now_utc = get_offset_aware_utc_now()
        run_id = migration_run_id or f"MIGRUN_{int(time.time())}_{os.urandom(4).hex()}"
        conn.execute("BEGIN IMMEDIATE;")
        try:
            # 1. Create Schema V4 tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS natural_evidence_epochs (
                    epoch_id TEXT PRIMARY KEY,
                    evidence_stream_id TEXT NOT NULL,
                    candidate_generation_id TEXT NOT NULL,
                    candidate_functional_sha TEXT NOT NULL,
                    semantic_closure_hash TEXT NOT NULL,
                    provenance_contract_version TEXT NOT NULL,
                    identity_schema_version TEXT NOT NULL,
                    denominator_policy_version TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('DEFINED', 'PRE_ACTIVATION', 'ACTIVE', 'CLOSED')),
                    activation_receipt_id TEXT,
                    activation_sequence INTEGER,
                    activated_at TEXT,
                    closed_receipt_id TEXT,
                    closed_sequence INTEGER,
                    closed_at TEXT,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS epoch_activation_receipts (
                    activation_receipt_id TEXT PRIMARY KEY,
                    epoch_id TEXT NOT NULL REFERENCES natural_evidence_epochs(epoch_id),
                    candidate_generation_id TEXT NOT NULL,
                    candidate_functional_sha TEXT NOT NULL,
                    semantic_closure_hash TEXT NOT NULL,
                    database_schema_version TEXT NOT NULL,
                    provenance_policy_version TEXT NOT NULL,
                    identity_schema_version TEXT NOT NULL,
                    denominator_policy_version TEXT NOT NULL,
                    scheduler_contract_identity TEXT NOT NULL,
                    activation_sequence INTEGER NOT NULL,
                    activated_at TEXT NOT NULL,
                    receipt_content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_epoch_memberships (
                    logical_scan_run_id TEXT PRIMARY KEY REFERENCES logical_scan_runs(logical_scan_run_id),
                    epoch_id TEXT NOT NULL REFERENCES natural_evidence_epochs(epoch_id),
                    membership_class TEXT NOT NULL CHECK(membership_class IN ('CURRENT_PROSPECTIVE_EPOCH', 'PRE_EPOCH_LEGACY', 'PRE_ACTIVATION_DELAYED_EVENT', 'NON_NATURAL', 'MIGRATION_CONFLICT')),
                    prospective_disposition TEXT NOT NULL CHECK(prospective_disposition IN ('PROSPECTIVE_CANDIDATE', 'PRE_EPOCH_INELIGIBLE', 'NON_NATURAL_INELIGIBLE', 'MIGRATION_CONFLICT_INELIGIBLE')),
                    activation_receipt_id TEXT,
                    assignment_sequence INTEGER NOT NULL,
                    assigned_at TEXT NOT NULL,
                    membership_fingerprint TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migration_source_manifests (
                    manifest_id TEXT PRIMARY KEY,
                    canonicalization_version TEXT NOT NULL,
                    source_unit_count INTEGER NOT NULL,
                    population_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migration_unit_dispositions (
                    source_unit_id TEXT PRIMARY KEY,
                    source_schema_version TEXT NOT NULL,
                    source_content_hash TEXT NOT NULL,
                    migration_disposition TEXT NOT NULL CHECK(migration_disposition IN ('LEGACY_NO_CURRENT_PROVENANCE', 'MIGRATION_CONFLICT', 'PREEXISTING_EQUIVALENT_PROVENANCE')),
                    legacy_recorded_origin_class TEXT NOT NULL,
                    epoch_membership_class TEXT NOT NULL,
                    prospective_disposition TEXT NOT NULL,
                    migration_id TEXT NOT NULL,
                    migration_run_id TEXT NOT NULL,
                    migrated_at TEXT NOT NULL,
                    disposition_hash TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_reconciliation_records (
                    reconciliation_id TEXT PRIMARY KEY,
                    source_unit_id TEXT NOT NULL,
                    reconciled_principal_type TEXT,
                    reconciled_principal_id TEXT,
                    reconciled_invocation_class TEXT,
                    reconciled_origin_class TEXT,
                    justification TEXT NOT NULL,
                    reconciled_at TEXT NOT NULL,
                    reconciled_by TEXT NOT NULL
                );
            """)

            # 2. Insert closed legacy epoch and default Candidate 004 pre-activation epoch
            conn.execute(
                """
                INSERT OR IGNORE INTO natural_evidence_epochs (
                    epoch_id, evidence_stream_id, candidate_generation_id, candidate_functional_sha,
                    semantic_closure_hash, provenance_contract_version, identity_schema_version,
                    denominator_policy_version, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "LEGACY_EPOCH_PRE_C004", "ARX_RADAR_SPRINT_3_SHADOW_STREAM",
                    "CANDIDATE_GENERATION_003_OR_EARLIER", "07b8b40cdb82328087b0f12adb08928a53e0234b",
                    "53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6",
                    "1.0.0", "3.0.0", "1.0.0", "CLOSED", "2026-10-10T00:00:00+00:00"
                )
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO natural_evidence_epochs (
                    epoch_id, evidence_stream_id, candidate_generation_id, candidate_functional_sha,
                    semantic_closure_hash, provenance_contract_version, identity_schema_version,
                    denominator_policy_version, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    NATURAL_EVIDENCE_EPOCH_ID, "ARX_RADAR_SPRINT_3_SHADOW_STREAM",
                    "CANDIDATE_GENERATION_004", "07b8b40cdb82328087b0f12adb08928a53e0234b",
                    "53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6",
                    PROVENANCE_CONTRACT_VERSION, IDENTITY_SCHEMA_VERSION,
                    DENOMINATOR_POLICY_VERSION, LOCAL_FREEZE_EPOCH_STATUS, now_utc
                )
            )

            # 3. Collect historical units (SHADOW_OBSERVATION_BUNDLE)
            cur = conn.cursor()
            cur.execute("""
                SELECT a.admission_id, a.observation_key, a.logical_scan_run_id, a.origin_class,
                       a.candidate_sha, a.admitted_at
                FROM shadow_evidence_admissions a
                ORDER BY a.admission_id;
            """)
            admissions = cur.fetchall()

            unit_hashes = []
            source_units = []
            for adm in admissions:
                unit_id = adm["admission_id"]
                content_payload = f"{adm['admission_id']}:{adm['observation_key']}:{adm['logical_scan_run_id']}:{adm['origin_class']}:{adm['candidate_sha']}"
                chash = hashlib.sha256(content_payload.encode("utf-8")).hexdigest()
                unit_hashes.append(chash)
                source_units.append({
                    "unit_id": unit_id,
                    "content_hash": chash,
                    "logical_scan_run_id": adm["logical_scan_run_id"],
                    "origin_class": adm["origin_class"],
                })

            sorted_hashes = sorted(unit_hashes)
            population_hash = hashlib.sha256("".join(sorted_hashes).encode("utf-8")).hexdigest()
            manifest_id = f"MANIFEST_{MIGRATION_ID}_{len(source_units)}"

            conn.execute(
                """
                INSERT OR IGNORE INTO migration_source_manifests (
                    manifest_id, canonicalization_version, source_unit_count, population_hash, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (manifest_id, MIGRATION_MANIFEST_CANONICALIZATION_VERSION, len(source_units), population_hash, now_utc)
            )

            # 4. Record dispositions and epoch memberships
            assigned_runs = set()
            cur.execute("SELECT logical_scan_run_id FROM evidence_epoch_memberships;")
            for r in cur.fetchall():
                assigned_runs.add(r[0])

            cur.execute("SELECT COALESCE(MAX(assignment_sequence), 0) FROM evidence_epoch_memberships;")
            cur_seq = cur.fetchone()[0]

            for unit in source_units:
                uid = unit["unit_id"]
                chash = unit["content_hash"]
                orig_cls = unit["origin_class"]
                lrun_id = unit["logical_scan_run_id"]

                if not orig_cls or orig_cls not in ('NATURAL_PRODUCTION', 'NON_EVIDENCE_BOOTSTRAP', 'SYNTHETIC', 'REPLAY', 'ADMIN_FORCED', 'TEST', 'NOT_ADMITTED'):
                    disp = "MIGRATION_CONFLICT"
                    mem_cls = "MIGRATION_CONFLICT"
                    prosp_disp = "MIGRATION_CONFLICT_INELIGIBLE"
                else:
                    disp = "LEGACY_NO_CURRENT_PROVENANCE"
                    mem_cls = "PRE_EPOCH_LEGACY"
                    prosp_disp = "PRE_EPOCH_INELIGIBLE"

                disp_payload = f"{uid}:{chash}:{disp}:{orig_cls}:{mem_cls}:{prosp_disp}:{MIGRATION_ID}"
                disp_hash = hashlib.sha256(disp_payload.encode("utf-8")).hexdigest()

                conn.execute(
                    """
                    INSERT OR IGNORE INTO migration_unit_dispositions (
                        source_unit_id, source_schema_version, source_content_hash, migration_disposition,
                        legacy_recorded_origin_class, epoch_membership_class, prospective_disposition,
                        migration_id, migration_run_id, migrated_at, disposition_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (uid, "3.0.0", chash, disp, orig_cls, mem_cls, prosp_disp, MIGRATION_ID, run_id, now_utc, disp_hash)
                )

                if lrun_id and lrun_id not in assigned_runs:
                    cur.execute("SELECT 1 FROM logical_scan_runs WHERE logical_scan_run_id = ?", (lrun_id,))
                    if cur.fetchone() is None:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO logical_scan_runs (
                                logical_scan_run_id, logical_trigger_id, scanner_id, universe_build_id, evaluation_as_of,
                                invocation_class, origin_class, originating_principal_type, originating_principal_id,
                                classification_policy_version, identity_schema_version, canonicalization_version,
                                provenance_fingerprint, created_at, classified_at, status
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                lrun_id, f"TRIG_{lrun_id}", "RADAR_VCP_SCANNER", "UNIVERSE_LEGACY", "2026-10-10",
                                "MANUAL_OPERATOR", "ADMIN_FORCED", "LEGACY_MIGRATION", "migration_system",
                                CLASSIFICATION_POLICY_VERSION, IDENTITY_SCHEMA_VERSION, CANONICALIZATION_VERSION,
                                hashlib.sha256(lrun_id.encode("utf-8")).hexdigest(), now_utc, now_utc, "CLASSIFIED"
                            )
                        )
                    cur_seq += 1
                    mem_fp_str = f"{lrun_id}:LEGACY_EPOCH_PRE_C004:{mem_cls}:{prosp_disp}:{cur_seq}"
                    mem_fp = hashlib.sha256(mem_fp_str.encode("utf-8")).hexdigest()

                    conn.execute(
                        """
                        INSERT OR IGNORE INTO evidence_epoch_memberships (
                            logical_scan_run_id, epoch_id, membership_class, prospective_disposition,
                            activation_receipt_id, assignment_sequence, assigned_at, membership_fingerprint
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (lrun_id, "LEGACY_EPOCH_PRE_C004", mem_cls, prosp_disp, None, cur_seq, now_utc, mem_fp)
                    )
                    assigned_runs.add(lrun_id)

            # 5. Create triggers
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_natural_evidence_epochs_delete
                BEFORE DELETE ON natural_evidence_epochs
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE_OF_NATURAL_EVIDENCE_EPOCHS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS validate_natural_evidence_epochs_update
                BEFORE UPDATE ON natural_evidence_epochs
                BEGIN
                    SELECT CASE
                        WHEN OLD.status = 'CLOSED' THEN RAISE(ABORT, 'CLOSED_EPOCH_CANNOT_BE_MODIFIED')
                        WHEN OLD.epoch_id != NEW.epoch_id THEN RAISE(ABORT, 'EPOCH_ID_UPDATE_REJECTED')
                        WHEN OLD.candidate_generation_id != NEW.candidate_generation_id THEN RAISE(ABORT, 'EPOCH_CANDIDATE_GENERATION_UPDATE_REJECTED')
                        WHEN OLD.candidate_functional_sha != NEW.candidate_functional_sha THEN RAISE(ABORT, 'EPOCH_FUNCTIONAL_SHA_UPDATE_REJECTED')
                        WHEN OLD.semantic_closure_hash != NEW.semantic_closure_hash THEN RAISE(ABORT, 'EPOCH_SEMANTIC_CLOSURE_UPDATE_REJECTED')
                        ELSE 1
                    END;
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_epoch_activation_receipts_update
                BEFORE UPDATE ON epoch_activation_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'MUTATION_OF_EPOCH_ACTIVATION_RECEIPTS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_epoch_activation_receipts_delete
                BEFORE DELETE ON epoch_activation_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE_OF_EPOCH_ACTIVATION_RECEIPTS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_evidence_epoch_memberships_update
                BEFORE UPDATE ON evidence_epoch_memberships
                BEGIN
                    SELECT CASE
                        WHEN OLD.epoch_id != NEW.epoch_id THEN RAISE(ABORT, 'EPOCH_ID_UPDATE_REJECTED')
                        WHEN OLD.membership_class != NEW.membership_class THEN RAISE(ABORT, 'MEMBERSHIP_CLASS_UPDATE_REJECTED')
                        WHEN OLD.prospective_disposition != NEW.prospective_disposition THEN RAISE(ABORT, 'PROSPECTIVE_DISPOSITION_UPDATE_REJECTED')
                        ELSE RAISE(ABORT, 'MUTATION_OF_EVIDENCE_EPOCH_MEMBERSHIPS_PROHIBITED')
                    END;
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_evidence_epoch_memberships_delete
                BEFORE DELETE ON evidence_epoch_memberships
                BEGIN
                    SELECT RAISE(ABORT, 'MEMBERSHIP_DELETE_REJECTED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_migration_source_manifests_update
                BEFORE UPDATE ON migration_source_manifests
                BEGIN
                    SELECT RAISE(ABORT, 'MUTATION_OF_MIGRATION_SOURCE_MANIFESTS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_migration_source_manifests_delete
                BEFORE DELETE ON migration_source_manifests
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE_OF_MIGRATION_SOURCE_MANIFESTS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_migration_unit_dispositions_update
                BEFORE UPDATE ON migration_unit_dispositions
                BEGIN
                    SELECT RAISE(ABORT, 'MUTATION_OF_MIGRATION_UNIT_DISPOSITIONS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_migration_unit_dispositions_delete
                BEFORE DELETE ON migration_unit_dispositions
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE_OF_MIGRATION_UNIT_DISPOSITIONS_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_historical_reconciliation_records_update
                BEFORE UPDATE ON historical_reconciliation_records
                BEGIN
                    SELECT RAISE(ABORT, 'MUTATION_OF_HISTORICAL_RECONCILIATION_PROHIBITED');
                END;
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS prevent_historical_reconciliation_records_delete
                BEFORE DELETE ON historical_reconciliation_records
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE_OF_HISTORICAL_RECONCILIATION_PROHIBITED');
                END;
            """)

            conn.execute("COMMIT;")
            return {
                "manifest_id": manifest_id,
                "source_count": len(source_units),
                "population_hash": population_hash,
                "status": "MIGRATION_COMPLETE",
            }
        except Exception:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise

    @retry_sqlite()
    def define_natural_evidence_epoch(
        self,
        epoch_id: str,
        evidence_stream_id: str,
        candidate_generation_id: str,
        candidate_functional_sha: str,
        semantic_closure_hash: str,
        provenance_contract_version: str = PROVENANCE_CONTRACT_VERSION,
        identity_schema_version: str = IDENTITY_SCHEMA_VERSION,
        denominator_policy_version: str = DENOMINATOR_POLICY_VERSION,
        status: str = "PRE_ACTIVATION",
    ) -> Dict[str, Any]:
        """Explicitly defines a natural evidence epoch in the repository."""
        now_utc = get_offset_aware_utc_now()
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO natural_evidence_epochs (
                    epoch_id, evidence_stream_id, candidate_generation_id,
                    candidate_functional_sha, semantic_closure_hash,
                    provenance_contract_version, identity_schema_version,
                    denominator_policy_version, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    epoch_id, evidence_stream_id, candidate_generation_id,
                    candidate_functional_sha, semantic_closure_hash,
                    provenance_contract_version, identity_schema_version,
                    denominator_policy_version, status, now_utc
                ),
            )
            conn.execute("COMMIT;")
            return {
                "epoch_id": epoch_id,
                "status": status,
                "candidate_generation_id": candidate_generation_id,
                "created_at": now_utc,
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
    def activate_natural_evidence_epoch(
        self,
        epoch_id: str,
        candidate_generation_id: str,
        candidate_functional_sha: str,
        semantic_closure_hash: str,
        scheduler_contract_identity: str,
        activation_receipt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically commits an immutable activation receipt and promotes epoch to ACTIVE status."""
        now_utc = get_offset_aware_utc_now()
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()

            cur.execute("SELECT * FROM natural_evidence_epochs WHERE epoch_id = ?", (epoch_id,))
            epoch_row = cur.fetchone()
            if not epoch_row:
                raise ValueError(f"Epoch '{epoch_id}' not found.")
            if epoch_row["status"] == "ACTIVE":
                raise RuntimeError(f"Epoch '{epoch_id}' is already ACTIVE.")
            if epoch_row["status"] == "CLOSED":
                raise RuntimeError(f"Epoch '{epoch_id}' is CLOSED and cannot be activated.")

            cur.execute("SELECT epoch_id FROM natural_evidence_epochs WHERE status = 'ACTIVE';")
            active_existing = cur.fetchone()
            if active_existing:
                raise RuntimeError(
                    f"ACTIVE_EPOCH_EXISTS: Cannot activate '{epoch_id}' because '{active_existing[0]}' is already ACTIVE."
                )

            cur.execute("SELECT COALESCE(MAX(activation_sequence), 0) + 1 FROM epoch_activation_receipts;")
            act_seq = cur.fetchone()[0]

            receipt_id = activation_receipt_id or f"rcpt-act-{epoch_id}-{act_seq}"

            receipt_payload = {
                "activation_receipt_id": receipt_id,
                "epoch_id": epoch_id,
                "candidate_generation_id": candidate_generation_id,
                "candidate_functional_sha": candidate_functional_sha,
                "semantic_closure_hash": semantic_closure_hash,
                "database_schema_version": SCHEMA_VERSION,
                "provenance_policy_version": PROVENANCE_CONTRACT_VERSION,
                "identity_schema_version": IDENTITY_SCHEMA_VERSION,
                "denominator_policy_version": DENOMINATOR_POLICY_VERSION,
                "scheduler_contract_identity": scheduler_contract_identity,
                "activation_sequence": act_seq,
                "activated_at": now_utc,
            }
            receipt_hash = hashlib.sha256(
                json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

            cur.execute(
                """
                INSERT INTO epoch_activation_receipts (
                    activation_receipt_id, epoch_id, candidate_generation_id,
                    candidate_functional_sha, semantic_closure_hash, database_schema_version,
                    provenance_policy_version, identity_schema_version, denominator_policy_version,
                    scheduler_contract_identity, activation_sequence, activated_at,
                    receipt_content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id, epoch_id, candidate_generation_id,
                    candidate_functional_sha, semantic_closure_hash, SCHEMA_VERSION,
                    PROVENANCE_CONTRACT_VERSION, IDENTITY_SCHEMA_VERSION, DENOMINATOR_POLICY_VERSION,
                    scheduler_contract_identity, act_seq, now_utc,
                    receipt_hash, now_utc
                ),
            )

            cur.execute(
                """
                UPDATE natural_evidence_epochs
                SET status = 'ACTIVE',
                    activation_receipt_id = ?,
                    activation_sequence = ?,
                    activated_at = ?
                WHERE epoch_id = ?
                """,
                (receipt_id, act_seq, now_utc, epoch_id),
            )

            conn.execute("COMMIT;")
            return {
                "status": "ACTIVE",
                "epoch_id": epoch_id,
                "activation_receipt_id": receipt_id,
                "activation_sequence": act_seq,
                "activated_at": now_utc,
                "receipt_content_hash": receipt_hash,
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
    def close_natural_evidence_epoch(
        self,
        epoch_id: str,
        closed_receipt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically closes an active epoch."""
        now_utc = get_offset_aware_utc_now()
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()

            cur.execute("SELECT * FROM natural_evidence_epochs WHERE epoch_id = ?", (epoch_id,))
            epoch_row = cur.fetchone()
            if not epoch_row:
                raise ValueError(f"Epoch '{epoch_id}' not found.")
            if epoch_row["status"] != "ACTIVE":
                raise RuntimeError(f"Cannot close epoch '{epoch_id}' with status '{epoch_row['status']}'. Must be ACTIVE.")

            cur.execute("SELECT COALESCE(MAX(closed_sequence), 0) + 1 FROM natural_evidence_epochs;")
            closed_seq = cur.fetchone()[0]
            rcpt_id = closed_receipt_id or f"rcpt-cls-{epoch_id}-{closed_seq}"

            cur.execute(
                """
                UPDATE natural_evidence_epochs
                SET status = 'CLOSED',
                    closed_receipt_id = ?,
                    closed_sequence = ?,
                    closed_at = ?
                WHERE epoch_id = ?
                """,
                (rcpt_id, closed_seq, now_utc, epoch_id),
            )
            conn.execute("COMMIT;")
            return {
                "status": "CLOSED",
                "epoch_id": epoch_id,
                "closed_receipt_id": rcpt_id,
                "closed_sequence": closed_seq,
                "closed_at": now_utc,
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
    def get_active_natural_evidence_epoch(self, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
        """Returns currently active natural evidence epoch, if any."""
        close_conn = False
        if conn is None:
            conn = self._get_connection()
            close_conn = True
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='natural_evidence_epochs';")
            if not cur.fetchone():
                return None
            cur.execute("SELECT * FROM natural_evidence_epochs WHERE status = 'ACTIVE' ORDER BY activation_sequence DESC LIMIT 1;")
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            if close_conn:
                conn.close()

    @retry_sqlite()
    def get_natural_evidence_epoch(self, epoch_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves epoch record by ID."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='natural_evidence_epochs';")
            if not cur.fetchone():
                return None
            cur.execute("SELECT * FROM natural_evidence_epochs WHERE epoch_id = ?", (epoch_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    @retry_sqlite()
    def get_epoch_membership(self, logical_scan_run_id: str, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
        """Retrieves immutable epoch membership for a logical scan run."""
        close_conn = False
        if conn is None:
            conn = self._get_connection()
            close_conn = True
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evidence_epoch_memberships';")
            if not cur.fetchone():
                return None
            cur.execute("SELECT * FROM evidence_epoch_memberships WHERE logical_scan_run_id = ?", (logical_scan_run_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            if close_conn:
                conn.close()

    @retry_sqlite()
    def reconcile_historical_provenance(
        self,
        source_unit_id: str,
        reconciled_principal_type: str,
        reconciled_principal_id: str,
        reconciled_invocation_class: str,
        reconciled_origin_class: str,
        justification: str,
        reconciled_by: str,
    ) -> Dict[str, Any]:
        """Appends historical provenance reconciliation without changing epoch membership or derived denominator."""
        now_utc = get_offset_aware_utc_now()
        reconciliation_id = f"recon-{int(time.time_ns())}-{os.urandom(4).hex()}"
        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO historical_reconciliation_records (
                    reconciliation_id, source_unit_id, reconciled_principal_type,
                    reconciled_principal_id, reconciled_invocation_class, reconciled_origin_class,
                    justification, reconciled_at, reconciled_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reconciliation_id, source_unit_id, reconciled_principal_type,
                    reconciled_principal_id, reconciled_invocation_class, reconciled_origin_class,
                    justification, now_utc, reconciled_by
                ),
            )
            conn.execute("COMMIT;")
            return {
                "reconciliation_id": reconciliation_id,
                "source_unit_id": source_unit_id,
                "reconciled_origin_class": reconciled_origin_class,
                "reconciled_at": now_utc,
                "epoch_mutations": 0,
                "prospective_mutations": 0,
                "denominator_delta": 0,
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
    def build_migration_manifest(self) -> Dict[str, Any]:
        """Builds canonical source manifest from current database admissions."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT a.admission_id, a.observation_key, a.logical_scan_run_id, a.origin_class, a.candidate_sha
                FROM shadow_evidence_admissions a
                ORDER BY a.admission_id;
            """)
            admissions = cur.fetchall()
            unit_hashes = []
            for adm in admissions:
                content_payload = f"{adm['admission_id']}:{adm['observation_key']}:{adm['logical_scan_run_id']}:{adm['origin_class']}:{adm['candidate_sha']}"
                chash = hashlib.sha256(content_payload.encode("utf-8")).hexdigest()
                unit_hashes.append(chash)
            sorted_hashes = sorted(unit_hashes)
            population_hash = hashlib.sha256("".join(sorted_hashes).encode("utf-8")).hexdigest()
            manifest_id = f"MANIFEST_{MIGRATION_ID}_{len(unit_hashes)}"
            return {
                "manifest_id": manifest_id,
                "canonicalization_version": MIGRATION_MANIFEST_CANONICALIZATION_VERSION,
                "source_unit_count": len(unit_hashes),
                "population_hash": population_hash,
                "unit_hashes": unit_hashes,
            }
        finally:
            conn.close()


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

            # Bind epoch membership if evidence_epoch_memberships exists (Schema V4)
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evidence_epoch_memberships';")
            if cur.fetchone() is not None:
                cur.execute("SELECT * FROM natural_evidence_epochs WHERE status = 'ACTIVE' ORDER BY activation_sequence DESC LIMIT 1;")
                active_row = cur.fetchone()
                if not active_row:
                    cur.execute("SELECT * FROM natural_evidence_epochs ORDER BY created_at DESC LIMIT 1;")
                    active_row = cur.fetchone()

                active_epoch_dict = dict(active_row) if active_row else None
                eff_epoch_id = active_epoch_dict["epoch_id"] if active_epoch_dict else "UNKNOWN_EPOCH"
                rcpt_id = active_epoch_dict.get("activation_receipt_id") if active_epoch_dict else None

                if startup_context or invocation_class == "BOOT_WARMUP":
                    mem_class = "NON_NATURAL"
                    prosp_disp = "NON_NATURAL_INELIGIBLE"
                elif invocation_class == "MANUAL_OPERATOR" or origin_class == "ADMIN_FORCED":
                    mem_class = "NON_NATURAL"
                    prosp_disp = "NON_NATURAL_INELIGIBLE"
                elif invocation_class == "REPLAY" or origin_class == "REPLAY" or replay_of_logical_scan_run_id:
                    mem_class = "NON_NATURAL"
                    prosp_disp = "NON_NATURAL_INELIGIBLE"
                elif invocation_class in ("SYNTHETIC", "TEST") or origin_class in ("SYNTHETIC", "TEST"):
                    mem_class = "NON_NATURAL"
                    prosp_disp = "NON_NATURAL_INELIGIBLE"
                elif invocation_class == "PROVENANCE_CONFLICT":
                    mem_class = "MIGRATION_CONFLICT"
                    prosp_disp = "MIGRATION_CONFLICT_INELIGIBLE"
                elif invocation_class == "SCHEDULED_PRODUCTION" and origin_class == "NATURAL_PRODUCTION":
                    if not active_epoch_dict or active_epoch_dict.get("status") != "ACTIVE":
                        mem_class = "PRE_ACTIVATION_DELAYED_EVENT"
                        prosp_disp = "PRE_EPOCH_INELIGIBLE"
                    elif scheduled_for and scheduled_for < active_epoch_dict.get("activated_at", ""):
                        mem_class = "PRE_ACTIVATION_DELAYED_EVENT"
                        prosp_disp = "PRE_EPOCH_INELIGIBLE"
                    else:
                        mem_class = "CURRENT_PROSPECTIVE_EPOCH"
                        prosp_disp = "PROSPECTIVE_CANDIDATE"
                else:
                    mem_class = "NON_NATURAL"
                    prosp_disp = "NON_NATURAL_INELIGIBLE"

                cur.execute("SELECT COALESCE(MAX(assignment_sequence), 0) + 1 FROM evidence_epoch_memberships;")
                assign_seq = cur.fetchone()[0]

                mem_fp_str = f"{logical_scan_run_id}:{eff_epoch_id}:{mem_class}:{prosp_disp}:{assign_seq}"
                mem_fp = hashlib.sha256(mem_fp_str.encode("utf-8")).hexdigest()

                cur.execute(
                    """
                    INSERT INTO evidence_epoch_memberships (
                        logical_scan_run_id, epoch_id, membership_class, prospective_disposition,
                        activation_receipt_id, assignment_sequence, assigned_at, membership_fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (logical_scan_run_id, eff_epoch_id, mem_class, prosp_disp, rcpt_id, assign_seq, now_utc, mem_fp),
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

                # Bind epoch membership if evidence_epoch_memberships exists (Schema V4)
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evidence_epoch_memberships';")
                if cur.fetchone() is not None:
                    cur.execute("SELECT * FROM natural_evidence_epochs WHERE status = 'ACTIVE' ORDER BY activation_sequence DESC LIMIT 1;")
                    active_row = cur.fetchone()
                    if not active_row:
                        cur.execute("SELECT * FROM natural_evidence_epochs ORDER BY created_at DESC LIMIT 1;")
                        active_row = cur.fetchone()

                    active_epoch_dict = dict(active_row) if active_row else None
                    eff_epoch_id = active_epoch_dict["epoch_id"] if active_epoch_dict else "UNKNOWN_EPOCH"
                    rcpt_id = active_epoch_dict.get("activation_receipt_id") if active_epoch_dict else None

                    if startup_context or eff_invocation_class == "BOOT_WARMUP":
                        mem_class = "NON_NATURAL"
                        prosp_disp = "NON_NATURAL_INELIGIBLE"
                    elif eff_invocation_class == "MANUAL_OPERATOR" or eff_origin_class == "ADMIN_FORCED":
                        mem_class = "NON_NATURAL"
                        prosp_disp = "NON_NATURAL_INELIGIBLE"
                    elif eff_invocation_class == "REPLAY" or eff_origin_class == "REPLAY" or replay_of_logical_scan_run_id:
                        mem_class = "NON_NATURAL"
                        prosp_disp = "NON_NATURAL_INELIGIBLE"
                    elif eff_invocation_class in ("SYNTHETIC", "TEST") or eff_origin_class in ("SYNTHETIC", "TEST"):
                        mem_class = "NON_NATURAL"
                        prosp_disp = "NON_NATURAL_INELIGIBLE"
                    elif eff_invocation_class == "PROVENANCE_CONFLICT":
                        mem_class = "MIGRATION_CONFLICT"
                        prosp_disp = "MIGRATION_CONFLICT_INELIGIBLE"
                    elif eff_invocation_class == "SCHEDULED_PRODUCTION" and eff_origin_class == "NATURAL_PRODUCTION":
                        if not active_epoch_dict or active_epoch_dict.get("status") != "ACTIVE":
                            mem_class = "PRE_ACTIVATION_DELAYED_EVENT"
                            prosp_disp = "PRE_EPOCH_INELIGIBLE"
                        elif scheduled_for and scheduled_for < active_epoch_dict.get("activated_at", ""):
                            mem_class = "PRE_ACTIVATION_DELAYED_EVENT"
                            prosp_disp = "PRE_EPOCH_INELIGIBLE"
                        else:
                            mem_class = "CURRENT_PROSPECTIVE_EPOCH"
                            prosp_disp = "PROSPECTIVE_CANDIDATE"
                    else:
                        mem_class = "NON_NATURAL"
                        prosp_disp = "NON_NATURAL_INELIGIBLE"

                    cur.execute("SELECT COALESCE(MAX(assignment_sequence), 0) + 1 FROM evidence_epoch_memberships;")
                    assign_seq = cur.fetchone()[0]

                    mem_fp_str = f"{eff_logical_run_id}:{eff_epoch_id}:{mem_class}:{prosp_disp}:{assign_seq}"
                    mem_fp = hashlib.sha256(mem_fp_str.encode("utf-8")).hexdigest()

                    cur.execute(
                        """
                        INSERT INTO evidence_epoch_memberships (
                            logical_scan_run_id, epoch_id, membership_class, prospective_disposition,
                            activation_receipt_id, assignment_sequence, assigned_at, membership_fingerprint
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (eff_logical_run_id, eff_epoch_id, mem_class, prosp_disp, rcpt_id, assign_seq, now_utc, mem_fp),
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
        """Derives authoritative denominator metrics strictly from committed database admissions joined with active epoch."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as cnt FROM shadow_evidence_admissions;")
            total = cur.fetchone()["cnt"]

            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evidence_epoch_memberships';")
            has_epoch_tables = cur.fetchone() is not None

            if has_epoch_tables:
                cur.execute(
                    """
                    SELECT COUNT(a.admission_id) as cnt
                    FROM shadow_evidence_admissions a
                    JOIN logical_scan_runs l ON a.logical_scan_run_id = l.logical_scan_run_id
                    JOIN evidence_epoch_memberships m ON l.logical_scan_run_id = m.logical_scan_run_id
                    JOIN natural_evidence_epochs e ON m.epoch_id = e.epoch_id
                    WHERE e.status = 'ACTIVE'
                      AND m.membership_class = 'CURRENT_PROSPECTIVE_EPOCH'
                      AND m.prospective_disposition = 'PROSPECTIVE_CANDIDATE'
                      AND a.origin_class = 'NATURAL_PRODUCTION'
                      AND l.origin_class = 'NATURAL_PRODUCTION'
                      AND l.invocation_class = 'SCHEDULED_PRODUCTION';
                    """
                )
                natural_admitted = cur.fetchone()["cnt"]
            else:
                cur.execute("SELECT COUNT(*) as cnt FROM shadow_evidence_admissions WHERE origin_class = 'NATURAL_PRODUCTION';")
                natural_admitted = cur.fetchone()["cnt"]

            cur.execute(
                """
                SELECT origin_class, COUNT(*) as cnt
                FROM shadow_evidence_admissions
                GROUP BY origin_class;
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
            for r in rows:
                cls = r["origin_class"]
                c = r["cnt"]
                if cls in counts:
                    counts[cls] = c

            return {
                "shadow_record_count": total,
                "natural_production_shadow_record_count": natural_admitted,
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
                JOIN shadow_observations o ON a.observation_id = o.observation_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM holdout_exclusions h
                    WHERE h.observation_id = a.observation_id
                       OR (h.exclusion_type = 'CASE' AND h.security_id = o.security_id AND h.evaluation_as_of = o.evaluation_as_of)
                )
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

            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evidence_epoch_memberships';")
            has_epoch_tables = cur.fetchone() is not None
            orphaned_memberships = 0
            duplicate_memberships = 0
            if has_epoch_tables:
                cur.execute(
                    """
                    SELECT COUNT(*) as cnt FROM logical_scan_runs l
                    LEFT JOIN evidence_epoch_memberships m ON l.logical_scan_run_id = m.logical_scan_run_id
                    WHERE m.logical_scan_run_id IS NULL;
                    """
                )
                orphaned_memberships = cur.fetchone()["cnt"]

                cur.execute(
                    """
                    SELECT COUNT(*) as cnt FROM (
                        SELECT logical_scan_run_id, COUNT(*) as m_cnt
                        FROM evidence_epoch_memberships
                        GROUP BY logical_scan_run_id
                        HAVING COUNT(*) > 1
                    );
                    """
                )
                duplicate_memberships = cur.fetchone()["cnt"]

            return {
                "orphaned_prospective_records": orphaned_prospective,
                "orphaned_exposure_records": orphaned_exposures,
                "orphaned_required_exclusions": orphaned_required_exclusions,
                "duplicate_admissions": duplicate_admissions,
                "orphaned_run_memberships": orphaned_memberships,
                "duplicate_run_memberships": duplicate_memberships,
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
                    and orphaned_memberships == 0
                    and duplicate_memberships == 0
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
