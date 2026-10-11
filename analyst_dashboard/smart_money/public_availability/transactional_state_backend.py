"""
ARX Terminal — Smart Money Public Availability
TRANSACTIONAL STATE AUTHORITY SUBSTRATE (Theorem C)

Implements:
1. Monotonic epoch generation, latest-head tracking, and anti-rollback verification
2. Distributed fencing tokens preventing stale runner writes
3. Real transactional Compare-And-Swap (CAS) for single-winner Gate-3 claims
4. Crash-atomic accession admission and denominator incrementation
5. Response-loss idempotent replay without duplicate increments
6. Anti-second-genesis enforcement when local/cache state is lost
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Set, Tuple


class StaleWriterFencingError(RuntimeError):
    """Raised when a stale runner attempts a write with superseded generation or fencing token."""
    pass


class SecondGenesisAttemptError(RuntimeError):
    """Raised when an empty runner attempts genesis while an authoritative epoch already exists."""
    pass


class DuplicateGate3ClaimError(RuntimeError):
    """Raised when a Gate-3 authorization has already been claimed or consumed."""
    pass


class GlobalRollbackError(RuntimeError):
    """Raised when a checkpoint sequence regresses against the authoritative latest head."""
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonicalize(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class TransactionalStateBackend:
    """Authoritative transactional state coordinator satisfying Theorem C."""

    def __init__(self, db_path: str, epoch_id: str = "smart-money-durable-state-v2"):
        self.db_path = os.path.abspath(db_path)
        self.epoch_id = epoch_id
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS authoritative_epoch (
                    epoch_id TEXT PRIMARY KEY,
                    epoch_initialized INTEGER NOT NULL,
                    generation INTEGER NOT NULL,
                    fencing_token TEXT NOT NULL,
                    latest_checkpoint_sequence INTEGER NOT NULL,
                    latest_checkpoint_hash TEXT NOT NULL,
                    latest_authority_event_sequence INTEGER NOT NULL,
                    governance_state TEXT NOT NULL,
                    natural_denominator INTEGER NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consumed_gate3_claims (
                    authorization_id TEXT PRIMARY KEY,
                    claiming_run_id TEXT NOT NULL,
                    claimed_at_generation INTEGER NOT NULL,
                    claimed_at_utc TEXT NOT NULL,
                    receipt_sha256 TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_accessions (
                    accession TEXT PRIMARY KEY,
                    authority_event_id TEXT NOT NULL,
                    admitted_at_generation INTEGER NOT NULL,
                    admitted_at_utc TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS idempotent_executions (
                    idempotency_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    committed_at_generation INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
            """)

    def initialize_epoch(self, initial_commit_sha: str) -> Tuple[int, str]:
        """Initializes pristine epoch if not already initialized."""
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()
            cur.execute("SELECT epoch_initialized, generation, fencing_token FROM authoritative_epoch WHERE epoch_id = ?", (self.epoch_id,))
            row = cur.fetchone()
            if row:
                conn.execute("COMMIT;")
                return row[1], row[2]

            initial_token = sha256_hex(f"{self.epoch_id}:1:{initial_commit_sha}".encode("utf-8"))
            now_utc = datetime.now(timezone.utc).isoformat()
            cur.execute("""
                INSERT INTO authoritative_epoch (
                    epoch_id, epoch_initialized, generation, fencing_token,
                    latest_checkpoint_sequence, latest_checkpoint_hash,
                    latest_authority_event_sequence, governance_state,
                    natural_denominator, updated_at_utc
                ) VALUES (?, 1, 1, ?, 0, '', 0, 'PRE_OBSERVATION_READY', 0, ?)
            """, (self.epoch_id, initial_token, now_utc))
            conn.execute("COMMIT;")
            return 1, initial_token

    def verify_genesis_safety(self, local_state_exists: bool) -> None:
        """Enforces Genesis Theorem: prevents clean runner from starting a second genesis if epoch exists."""
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT epoch_initialized, generation FROM authoritative_epoch WHERE epoch_id = ?", (self.epoch_id,))
            row = cur.fetchone()
            if row and row[0] == 1 and not local_state_exists:
                raise SecondGenesisAttemptError(
                    f"CRITICAL GENESIS VIOLATION: Authoritative epoch {self.epoch_id} is active at generation {row[1]}, "
                    "but local state is missing. A second genesis is impossible. Fails closed to TERMINAL_HOLD."
                )

    def get_authoritative_state(self) -> Dict[str, Any]:
        """Retrieves current authoritative state snapshot."""
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT epoch_id, epoch_initialized, generation, fencing_token, latest_checkpoint_sequence, latest_checkpoint_hash, latest_authority_event_sequence, governance_state, natural_denominator, updated_at_utc FROM authoritative_epoch WHERE epoch_id = ?", (self.epoch_id,))
            row = cur.fetchone()
            if not row:
                return {"epoch_initialized": False, "generation": 0, "natural_denominator": 0}
            return {
                "epoch_id": row[0],
                "epoch_initialized": bool(row[1]),
                "generation": row[2],
                "fencing_token": row[3],
                "latest_checkpoint_sequence": row[4],
                "latest_checkpoint_hash": row[5],
                "latest_authority_event_sequence": row[6],
                "governance_state": row[7],
                "natural_denominator": row[8],
                "updated_at_utc": row[9],
            }

    def claim_authorization_cas(
        self,
        authorization_id: str,
        run_id: str,
        expected_generation: int,
        fencing_token: str,
    ) -> Tuple[bool, int, str, Dict[str, Any]]:
        """Atomic Compare-And-Swap (CAS) claiming Gate-3 authorization and advancing generation."""
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()

            # 1. Check if already claimed / consumed
            cur.execute("SELECT claiming_run_id, claimed_at_generation, receipt_sha256 FROM consumed_gate3_claims WHERE authorization_id = ?", (authorization_id,))
            existing = cur.fetchone()
            if existing:
                conn.execute("ROLLBACK;")
                raise DuplicateGate3ClaimError(
                    f"Gate-3 authorization {authorization_id} already claimed by run {existing[0]} at generation {existing[1]}."
                )

            # 2. Verify generation and fencing token
            cur.execute("SELECT generation, fencing_token FROM authoritative_epoch WHERE epoch_id = ?", (self.epoch_id,))
            epoch_row = cur.fetchone()
            if not epoch_row:
                conn.execute("ROLLBACK;")
                raise StaleWriterFencingError("Epoch uninitialized in authoritative backend")

            current_gen, current_token = epoch_row[0], epoch_row[1]
            if current_gen != expected_generation or current_token != fencing_token:
                conn.execute("ROLLBACK;")
                raise StaleWriterFencingError(
                    f"Fencing conflict: Expected generation {expected_generation} (token {fencing_token[:8]}...), "
                    f"observed {current_gen} (token {current_token[:8]}...)."
                )

            # 3. Advance generation and mint new fencing token
            new_generation = current_gen + 1
            now_utc = datetime.now(timezone.utc).isoformat()
            new_token = sha256_hex(f"{self.epoch_id}:{new_generation}:{run_id}:{authorization_id}".encode("utf-8"))

            receipt = {
                "authorization_id": authorization_id,
                "claiming_run_id": run_id,
                "claimed_at_generation": new_generation,
                "claimed_at_utc": now_utc,
            }
            receipt_sha = sha256_hex(canonicalize(receipt).encode("utf-8"))

            cur.execute("""
                INSERT INTO consumed_gate3_claims (authorization_id, claiming_run_id, claimed_at_generation, claimed_at_utc, receipt_sha256)
                VALUES (?, ?, ?, ?, ?)
            """, (authorization_id, run_id, new_generation, now_utc, receipt_sha))

            cur.execute("""
                UPDATE authoritative_epoch
                SET generation = ?, fencing_token = ?, governance_state = 'AUTHORIZATION_CLAIMED', updated_at_utc = ?
                WHERE epoch_id = ?
            """, (new_generation, new_token, now_utc, self.epoch_id))

            conn.execute("COMMIT;")
            return True, new_generation, new_token, receipt

    def record_accession_and_denominator(
        self,
        accession: str,
        authority_event_id: str,
        idempotency_key: str,
        generation: int,
        fencing_token: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Atomically admits natural accession and increments natural denominator in a single transaction."""
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()

            # 1. Idempotency Check (Response-loss protection)
            cur.execute("SELECT response_json FROM idempotent_executions WHERE idempotency_key = ?", (idempotency_key,))
            cached = cur.fetchone()
            if cached:
                conn.execute("ROLLBACK;")
                cached_res = json.loads(cached[0])
                cached_res["status"] = "ALREADY_COMMITTED_IDEMPOTENT_SUCCESS"
                return True, cached_res

            # 2. Fencing validation
            cur.execute("SELECT generation, fencing_token, natural_denominator, latest_authority_event_sequence FROM authoritative_epoch WHERE epoch_id = ?", (self.epoch_id,))
            row = cur.fetchone()
            if not row or row[0] != generation or row[1] != fencing_token:
                conn.execute("ROLLBACK;")
                raise StaleWriterFencingError("Stale runner rejected during admission transaction")

            current_denom = row[2]
            current_event_seq = row[3]

            # 3. Check duplicate accession
            cur.execute("SELECT admitted_at_generation FROM processed_accessions WHERE accession = ?", (accession,))
            if cur.fetchone():
                conn.execute("ROLLBACK;")
                return False, {"error": f"Accession {accession} already processed"}

            # 4. Atomic mutations
            new_denom = current_denom + 1
            new_event_seq = current_event_seq + 1
            now_utc = datetime.now(timezone.utc).isoformat()

            cur.execute("""
                INSERT INTO processed_accessions (accession, authority_event_id, admitted_at_generation, admitted_at_utc)
                VALUES (?, ?, ?, ?)
            """, (accession, authority_event_id, generation, now_utc))

            cur.execute("""
                UPDATE authoritative_epoch
                SET natural_denominator = ?, latest_authority_event_sequence = ?, governance_state = 'DENOMINATOR_INCREMENTED', updated_at_utc = ?
                WHERE epoch_id = ?
            """, (new_denom, new_event_seq, now_utc, self.epoch_id))

            result_payload = {
                "status": "ADMISSION_COMMITTED",
                "accession": accession,
                "authority_event_id": authority_event_id,
                "natural_denominator": new_denom,
                "authority_event_sequence": new_event_seq,
                "committed_at_utc": now_utc,
            }

            cur.execute("""
                INSERT INTO idempotent_executions (idempotency_key, response_json, committed_at_generation, created_at_utc)
                VALUES (?, ?, ?, ?)
            """, (idempotency_key, json.dumps(result_payload), generation, now_utc))

            conn.execute("COMMIT;")
            return True, result_payload

    def verify_and_commit_checkpoint(
        self,
        checkpoint_sequence: int,
        checkpoint_hash: str,
        generation: int,
        fencing_token: str,
    ) -> Tuple[bool, str]:
        """Verifies checkpoint monotonicity and updates authoritative latest head."""
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()

            cur.execute("SELECT generation, fencing_token, latest_checkpoint_sequence FROM authoritative_epoch WHERE epoch_id = ?", (self.epoch_id,))
            row = cur.fetchone()
            if not row or row[0] != generation or row[1] != fencing_token:
                conn.execute("ROLLBACK;")
                raise StaleWriterFencingError("Stale runner rejected during checkpoint commit")

            latest_seq = row[2]
            if checkpoint_sequence <= latest_seq:
                conn.execute("ROLLBACK;")
                raise GlobalRollbackError(
                    f"Rollback rejected: Attempted checkpoint sequence {checkpoint_sequence} <= authoritative latest sequence {latest_seq}."
                )

            now_utc = datetime.now(timezone.utc).isoformat()
            cur.execute("""
                UPDATE authoritative_epoch
                SET latest_checkpoint_sequence = ?, latest_checkpoint_hash = ?, updated_at_utc = ?
                WHERE epoch_id = ?
            """, (checkpoint_sequence, checkpoint_hash, now_utc, self.epoch_id))

            conn.execute("COMMIT;")
            return True, "CHECKPOINT_HEAD_UPDATED"
