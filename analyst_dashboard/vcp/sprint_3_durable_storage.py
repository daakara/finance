"""ARX Terminal — Sprint 3 Durable Shadow Evidence Store & Relational Engine.

Authoritative persistent storage engine providing:
1. Strict relational schema with foreign keys, unique constraints, and check constraints.
2. Immutability enforced via SQLite triggers (UPDATE/DELETE prohibited).
3. Deterministic idempotency observation key computation.
4. Single-transaction bundle admission (all-or-nothing atomicity with rollback).
5. Authoritative denominator derivation from committed database admission records.
6. Multi-worker safe concurrency via SQLite WAL mode with retry backoff.
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

SCHEMA_VERSION: str = "2.0.0"
MIGRATION_ID: str = "MIGRATION_20261010_002_DURABLE_SHADOW_EVIDENCE"
OBSERVATION_KEY_SPECIFICATION: str = (
    "sha256(scanner_id:security_id:evaluation_as_of:universe_build_id:snapshot_run_id:candidate_generation_id:semantic_closure_hash)"
)

DEFAULT_SHADOW_DB_FILENAME: str = "shadow_evidence.db"


def compute_deterministic_observation_key(
    scanner_id: str,
    security_id: str,
    evaluation_as_of: str,
    universe_build_id: str,
    snapshot_run_id: str,
    candidate_generation_id: str,
    semantic_closure_hash: str,
) -> str:
    """Computes canonical deterministic idempotency key for an observation."""
    tuple_str = (
        f"{scanner_id}:{security_id}:{evaluation_as_of}:{universe_build_id}:"
        f"{snapshot_run_id}:{candidate_generation_id}:{semantic_closure_hash}"
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
-- 1. Shadow Observations Table
CREATE TABLE IF NOT EXISTS shadow_observations (
    observation_id TEXT PRIMARY KEY,
    observation_key TEXT UNIQUE NOT NULL,
    scanner_id TEXT NOT NULL,
    scanner_run_id TEXT NOT NULL,
    security_id TEXT NOT NULL,
    evaluation_as_of TEXT NOT NULL,
    candidate_generation_id TEXT NOT NULL,
    candidate_sha TEXT NOT NULL,
    semantic_closure_hash TEXT NOT NULL,
    universe_build_id TEXT NOT NULL,
    origin_class TEXT NOT NULL CHECK(origin_class IN ('NATURAL_PRODUCTION', 'SYNTHETIC', 'REPLAY', 'ADMIN_FORCED', 'TEST')),
    created_at TEXT NOT NULL
);

-- 2. Prospective Decisions Table (1:1 with observation)
CREATE TABLE IF NOT EXISTS prospective_decisions (
    decision_record_id TEXT PRIMARY KEY,
    observation_id TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_id),
    observation_key TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_key),
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
    created_at TEXT NOT NULL
);

-- 3. Production Exposures Table (1:1 with observation)
CREATE TABLE IF NOT EXISTS production_exposures (
    exposure_id TEXT PRIMARY KEY,
    observation_id TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_id),
    observation_key TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_key),
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

-- 4. Holdout Exclusions Table (1:N with observation)
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

-- 5. Shadow Evidence Admissions Table (1:1 with observation, Authoritative Denominator Source)
CREATE TABLE IF NOT EXISTS shadow_evidence_admissions (
    admission_id TEXT PRIMARY KEY,
    observation_id TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_id),
    observation_key TEXT UNIQUE NOT NULL REFERENCES shadow_observations(observation_key),
    origin_class TEXT NOT NULL CHECK(origin_class IN ('NATURAL_PRODUCTION', 'SYNTHETIC', 'REPLAY', 'ADMIN_FORCED', 'TEST')),
    candidate_generation_id TEXT NOT NULL,
    candidate_sha TEXT NOT NULL,
    semantic_closure_hash TEXT NOT NULL,
    receipt_hash TEXT NOT NULL,
    admitted_at TEXT NOT NULL
);

-- 6. Shadow Outbox Table (Transactional outbox for decoupled dispatch)
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
    """Authoritative durable storage engine for Sprint 3 production shadow evidence."""

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
            conn.executescript(DDL_SCHEMA)
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
        origin_class: str = "NATURAL_PRODUCTION",
        developer_visible: bool = True,
        user_visible: bool = False,
        exposure_type: str = "PRODUCTION_SHADOW_OBSERVATION",
        scanner_id: str = "MINERVINI_VCP",
        failure_injection_point: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically admits a complete evidence bundle in one single database transaction.

        Supports failure injection at points A through G for transaction rollback verification.
        """
        valid_origins = {"NATURAL_PRODUCTION", "SYNTHETIC", "REPLAY", "ADMIN_FORCED", "TEST"}
        if origin_class not in valid_origins:
            raise ValueError(f"Invalid origin_class: {origin_class}. Must be one of {valid_origins}")

        # Compute deterministic observation key
        observation_key = compute_deterministic_observation_key(
            scanner_id=scanner_id,
            security_id=security_id,
            evaluation_as_of=evaluation_as_of,
            universe_build_id=universe_build_id,
            snapshot_run_id=snapshot_run_id,
            candidate_generation_id=candidate_generation_id,
            semantic_closure_hash=semantic_closure_hash,
        )

        observation_id = f"obs-{observation_key[:20]}"
        decision_record_id = f"dec-{observation_key[:20]}"
        exposure_id = f"exp-{observation_key[:20]}"
        admission_id = f"adm-{observation_key[:20]}"
        now_utc = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")

            # 1. Idempotency Check: if already admitted, return existing admission safely
            cur = conn.cursor()
            cur.execute(
                "SELECT admission_id, origin_class, receipt_hash, admitted_at FROM shadow_evidence_admissions WHERE observation_key = ?",
                (observation_key,),
            )
            existing = cur.fetchone()
            if existing:
                conn.execute("COMMIT;")
                return {
                    "status": "ALREADY_ADMITTED",
                    "observation_key": observation_key,
                    "observation_id": observation_id,
                    "admission_id": existing["admission_id"],
                    "origin_class": existing["origin_class"],
                    "receipt_hash": existing["receipt_hash"],
                    "admitted_at": existing["admitted_at"],
                    "new_admission": False,
                }

            # 2. Insert shadow_observations
            cur.execute(
                """
                INSERT INTO shadow_observations (
                    observation_id, observation_key, scanner_id, scanner_run_id,
                    security_id, evaluation_as_of, candidate_generation_id, candidate_sha,
                    semantic_closure_hash, universe_build_id, origin_class, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id, observation_key, scanner_id, snapshot_run_id,
                    security_id, evaluation_as_of, candidate_generation_id, candidate_sha,
                    semantic_closure_hash, universe_build_id, origin_class, now_utc
                ),
            )
            if failure_injection_point == "A":
                raise RuntimeError("FAILURE_INJECTION_A: Simulated failure after shadow_observation insert")

            # 3. Insert prospective_decisions
            cur.execute(
                """
                INSERT INTO prospective_decisions (
                    decision_record_id, observation_id, observation_key, evaluation_as_of,
                    known_at, security_id, universe_build_id, snapshot_run_id,
                    candidate_generation_id, candidate_sha, semantic_closure_hash,
                    runtime_config_hash, dependency_lock_hash, data_provenance_hash,
                    ruleset_id, ruleset_version, predicate_vector_hash, classification,
                    decision_posture, input_fingerprint, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_record_id, observation_id, observation_key, evaluation_as_of,
                    now_utc, security_id, universe_build_id, snapshot_run_id,
                    candidate_generation_id, candidate_sha, semantic_closure_hash,
                    runtime_config_hash, dependency_lock_hash, data_provenance_hash,
                    ruleset_id, ruleset_version, predicate_vector_hash, classification,
                    decision_posture, input_fingerprint, now_utc
                ),
            )
            if failure_injection_point == "B":
                raise RuntimeError("FAILURE_INJECTION_B: Simulated failure after prospective_decision insert")

            # 4. Insert production_exposures
            cur.execute(
                """
                INSERT INTO production_exposures (
                    exposure_id, observation_id, observation_key, security_id,
                    evaluation_as_of, group_or_episode_id, universe_build_id,
                    snapshot_run_id, candidate_generation_id, candidate_sha,
                    semantic_closure_hash, runtime_config_hash, data_provenance_hash,
                    developer_visible, user_visible, exposure_type, exposed_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exposure_id, observation_id, observation_key, security_id,
                    evaluation_as_of, group_or_episode_id, universe_build_id,
                    snapshot_run_id, candidate_generation_id, candidate_sha,
                    semantic_closure_hash, runtime_config_hash, data_provenance_hash,
                    1 if developer_visible else 0, 1 if user_visible else 0,
                    exposure_type, now_utc, now_utc
                ),
            )
            if failure_injection_point == "C":
                raise RuntimeError("FAILURE_INJECTION_C: Simulated failure after production_exposure insert")

            # 5. Insert holdout exclusions (Case hash, Episode Group hash, Window hash)
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

            # 6. Insert shadow_evidence_admissions
            receipt_data = f"{admission_id}:{observation_key}:{origin_class}:{candidate_sha}:{now_utc}"
            receipt_hash = hashlib.sha256(receipt_data.encode("utf-8")).hexdigest()

            if failure_injection_point == "F":
                raise RuntimeError("FAILURE_INJECTION_F: Simulated failure before admission insert")

            cur.execute(
                """
                INSERT INTO shadow_evidence_admissions (
                    admission_id, observation_id, observation_key, origin_class,
                    candidate_generation_id, candidate_sha, semantic_closure_hash,
                    receipt_hash, admitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    admission_id, observation_id, observation_key, origin_class,
                    candidate_generation_id, candidate_sha, semantic_closure_hash,
                    receipt_hash, now_utc
                ),
            )

            # 7. Insert shadow_outbox
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
                        "origin_class": origin_class,
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
                "decision_record_id": decision_record_id,
                "exposure_id": exposure_id,
                "admission_id": admission_id,
                "origin_class": origin_class,
                "receipt_hash": receipt_hash,
                "admitted_at": now_utc,
                "new_admission": True,
            }
        except Exception:
            conn.execute("ROLLBACK;")
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
                "SYNTHETIC": 0,
                "REPLAY": 0,
                "ADMIN_FORCED": 0,
                "TEST": 0,
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
        # Canonical representation of original immutable fields
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
