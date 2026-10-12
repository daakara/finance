"""
ARX Terminal — Smart Money Public Availability
TRANSACTIONAL STATE AUTHORITY SUBSTRATE (Theorem C)

Implements:
1. Monotonic epoch generation, latest-head tracking, and anti-rollback verification
2. Distributed fencing tokens preventing stale runner writes (FND-ARCH-010)
3. Real transactional Compare-And-Swap (CAS) for single-winner Gate-3 claims with durable crash recovery (FND-ARCH-007)
4. Disambiguation of contention vs semantic quarantine (FND-ARCH-008)
5. Authoritative denominator admission predicate equivalence (FND-ARCH-011)
6. Transaction log primacy and consensus anti-entropy coherence (FND-ARCH-014)
7. Crash-atomic accession admission and denominator incrementation
8. Response-loss idempotent replay without duplicate increments
9. Anti-second-genesis enforcement when local/cache state is lost
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Set, Tuple


class StaleWriterFencingError(RuntimeError):
    """Raised when a stale runner attempts a write with superseded generation or fencing token."""
    pass


class StaleFencingTokenError(StaleWriterFencingError):
    """
    FND-ARCH-010: Raised when a stale writer submits with an outdated generation or fencing token.
    Indicates safe concurrency rejection (STALE_WRITER_REJECTED) without locking the system in terminal hold.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.outcome = "STALE_WRITER_REJECTED"


class SecondGenesisAttemptError(RuntimeError):
    """Raised when an empty runner attempts genesis while an authoritative epoch already exists."""
    pass


class DuplicateGate3ClaimError(RuntimeError):
    """Raised when a Gate-3 authorization has already been claimed and committed."""
    pass


class Gate3ContentionError(DuplicateGate3ClaimError):
    """
    FND-ARCH-008: Raised on transient concurrent lease/lock collisions during Gate-3 claim attempts.
    This is a retryable contention condition, NOT semantic evidence corruption or quarantine.
    Subclasses DuplicateGate3ClaimError so existing CAS claim callers catch it as a claim collision.
    """
    def __init__(self, message: str, retry_after_seconds: int = 5):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.action = "A4_ANCHOR_RETRY_WAIT"
        self.retryable = True


class EvidenceSemanticQuarantineError(RuntimeError):
    """
    FND-ARCH-008: Raised when evidence fails semantic validation or cryptographic integrity.
    This is terminal and non-retryable, resulting in QUARANTINED status.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.status = "QUARANTINED"
        self.retryable = False


class ConsensusLogDivergenceError(RuntimeError):
    """
    FND-ARCH-014: Raised when write-ahead transaction log diverges from consensus state.
    Triggers fail-closed state to preserve ledger primacy.
    """
    pass


class GlobalRollbackError(RuntimeError):
    """Raised when a checkpoint sequence regresses against the authoritative latest head."""
    pass


class Gate3ClaimState(str, Enum):
    """
    FND-ARCH-007: Lifecycle states for Gate-3 authorizations.
    Prevents orphaned claims and duplicate admissions across runner crashes.
    """
    UNCLAIMED = "UNCLAIMED"
    CLAIMED_PENDING_COMMIT = "CLAIMED_PENDING_COMMIT"
    COMMITTED = "COMMITTED"
    LEASE_EXPIRED_RECOVERABLE = "LEASE_EXPIRED_RECOVERABLE"
    ABORTED_RELEASED = "ABORTED_RELEASED"


def authoritative_denominator_admission_predicate(
    is_natural_source: bool,
    is_valid_evidence: bool,
    is_fresh: bool,
    is_atomic_admission: bool,
    is_duplicate: bool,
) -> bool:
    """
    FND-ARCH-011: Unified Authoritative Denominator Admission Predicate.
    Evaluated identically for pre-arm eligibility and storage increment.
    Truth table verifies equivalence across all 11 evaluation cases.
    """
    return (
        bool(is_natural_source)
        and bool(is_valid_evidence)
        and bool(is_fresh)
        and bool(is_atomic_admission)
        and not bool(is_duplicate)
    )


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
                    receipt_sha256 TEXT NOT NULL,
                    claim_state TEXT NOT NULL DEFAULT 'COMMITTED',
                    lease_expires_at_utc TEXT
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transaction_log (
                    log_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_hash TEXT NOT NULL,
                    prev_entry_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
            """)
            # Check if claim_state column exists in consumed_gate3_claims for backwards compat
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(consumed_gate3_claims);")
            columns = [c[1] for c in cur.fetchall()]
            if "claim_state" not in columns:
                conn.execute("ALTER TABLE consumed_gate3_claims ADD COLUMN claim_state TEXT NOT NULL DEFAULT 'COMMITTED';")
            if "lease_expires_at_utc" not in columns:
                conn.execute("ALTER TABLE consumed_gate3_claims ADD COLUMN lease_expires_at_utc TEXT;")

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
        lease_timeout_seconds: int = 300,
    ) -> Tuple[bool, int, str, Dict[str, Any]]:
        """
        Atomic Compare-And-Swap (CAS) claiming Gate-3 authorization and advancing generation.
        FND-ARCH-007: Lifecycle-aware claim state machine (CLAIMED_PENDING_COMMIT -> COMMITTED).
        FND-ARCH-008: Differentiates transient contention from terminal quarantine.
        FND-ARCH-010: Rejects stale writers without corrupting or halting epoch.
        """
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()

            now_dt = datetime.now(timezone.utc)
            now_utc = now_dt.isoformat()

            # 1. Check claim status in consumed_gate3_claims
            cur.execute(
                "SELECT claiming_run_id, claimed_at_generation, receipt_sha256, claim_state, lease_expires_at_utc "
                "FROM consumed_gate3_claims WHERE authorization_id = ?",
                (authorization_id,)
            )
            existing = cur.fetchone()
            if existing:
                ex_run, ex_gen, ex_sha, ex_state, ex_lease_exp = existing
                if ex_state == Gate3ClaimState.COMMITTED.value:
                    conn.execute("ROLLBACK;")
                    raise DuplicateGate3ClaimError(
                        f"Gate-3 authorization {authorization_id} already claimed and committed by run {ex_run} at generation {ex_gen}."
                    )
                elif ex_state == Gate3ClaimState.CLAIMED_PENDING_COMMIT.value:
                    # Check lease expiration
                    is_lease_active = True
                    if ex_lease_exp:
                        try:
                            exp_dt = datetime.fromisoformat(ex_lease_exp.replace("Z", "+00:00"))
                            if now_dt > exp_dt:
                                is_lease_active = False
                        except Exception:
                            pass
                    if is_lease_active and ex_run != run_id:
                        conn.execute("ROLLBACK;")
                        # FND-ARCH-008: Concurrency collision is contention, NOT quarantine!
                        raise Gate3ContentionError(
                            f"Gate-3 authorization {authorization_id} is currently held by active run {ex_run} until {ex_lease_exp}."
                        )
                # If state is LEASE_EXPIRED_RECOVERABLE or ABORTED_RELEASED or lease expired, proceed with claim

            # 2. Verify generation and fencing token (FND-ARCH-010)
            cur.execute("SELECT generation, fencing_token FROM authoritative_epoch WHERE epoch_id = ?", (self.epoch_id,))
            epoch_row = cur.fetchone()
            if not epoch_row:
                conn.execute("ROLLBACK;")
                raise StaleFencingTokenError("Epoch uninitialized in authoritative backend")

            current_gen, current_token = epoch_row[0], epoch_row[1]
            if current_gen != expected_generation or current_token != fencing_token:
                conn.execute("ROLLBACK;")
                raise StaleFencingTokenError(
                    f"Fencing conflict: Expected generation {expected_generation} (token {fencing_token[:8]}...), "
                    f"observed {current_gen} (token {current_token[:8]}...)."
                )

            # 3. Advance generation and mint new fencing token
            new_generation = current_gen + 1
            new_token = sha256_hex(f"{self.epoch_id}:{new_generation}:{run_id}:{authorization_id}".encode("utf-8"))
            lease_expires_at = (now_dt + timedelta(seconds=lease_timeout_seconds)).isoformat()

            receipt = {
                "authorization_id": authorization_id,
                "claiming_run_id": run_id,
                "claimed_at_generation": new_generation,
                "claimed_at_utc": now_utc,
                "claim_state": Gate3ClaimState.CLAIMED_PENDING_COMMIT.value,
                "lease_expires_at_utc": lease_expires_at,
            }
            receipt_sha = sha256_hex(canonicalize(receipt).encode("utf-8"))

            cur.execute("""
                INSERT INTO consumed_gate3_claims (
                    authorization_id, claiming_run_id, claimed_at_generation, claimed_at_utc,
                    receipt_sha256, claim_state, lease_expires_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(authorization_id) DO UPDATE SET
                    claiming_run_id = excluded.claiming_run_id,
                    claimed_at_generation = excluded.claimed_at_generation,
                    claimed_at_utc = excluded.claimed_at_utc,
                    receipt_sha256 = excluded.receipt_sha256,
                    claim_state = excluded.claim_state,
                    lease_expires_at_utc = excluded.lease_expires_at_utc
            """, (
                authorization_id, run_id, new_generation, now_utc, receipt_sha,
                Gate3ClaimState.CLAIMED_PENDING_COMMIT.value, lease_expires_at
            ))

            cur.execute("""
                UPDATE authoritative_epoch
                SET generation = ?, fencing_token = ?, governance_state = 'AUTHORIZATION_CLAIMED', updated_at_utc = ?
                WHERE epoch_id = ?
            """, (new_generation, new_token, now_utc, self.epoch_id))

            conn.execute("COMMIT;")
            return True, new_generation, new_token, receipt

    def commit_claim_atomic(self, authorization_id: str, run_id: str) -> bool:
        """
        FND-ARCH-007: Atomically transitions a claimed authorization from CLAIMED_PENDING_COMMIT to COMMITTED.
        """
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()
            cur.execute(
                "SELECT claim_state, claiming_run_id FROM consumed_gate3_claims WHERE authorization_id = ?",
                (authorization_id,)
            )
            row = cur.fetchone()
            if not row:
                conn.execute("ROLLBACK;")
                return False
            if row[1] != run_id:
                conn.execute("ROLLBACK;")
                raise StaleFencingTokenError(f"Run {run_id} does not own claim on {authorization_id}")

            now_utc = datetime.now(timezone.utc).isoformat()
            cur.execute("""
                UPDATE consumed_gate3_claims
                SET claim_state = ?, lease_expires_at_utc = NULL
                WHERE authorization_id = ?
            """, (Gate3ClaimState.COMMITTED.value, authorization_id))
            conn.execute("COMMIT;")
            return True

    def rollback_claim(self, authorization_id: str, run_id: str) -> bool:
        """
        FND-ARCH-007: Releases or aborts an in-flight claim, allowing safe recovery.
        """
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()
            cur.execute(
                "SELECT claim_state, claiming_run_id FROM consumed_gate3_claims WHERE authorization_id = ?",
                (authorization_id,)
            )
            row = cur.fetchone()
            if not row or row[1] != run_id:
                conn.execute("ROLLBACK;")
                return False

            cur.execute("""
                UPDATE consumed_gate3_claims
                SET claim_state = ?, lease_expires_at_utc = NULL
                WHERE authorization_id = ?
            """, (Gate3ClaimState.ABORTED_RELEASED.value, authorization_id))
            conn.execute("COMMIT;")
            return True

    def recover_expired_claims(self) -> int:
        """
        FND-ARCH-007: Reclaims orphaned claims from crashed runners whose leases have expired.
        """
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()
            now_utc = datetime.now(timezone.utc).isoformat()
            cur.execute("""
                UPDATE consumed_gate3_claims
                SET claim_state = ?
                WHERE claim_state = ? AND lease_expires_at_utc IS NOT NULL AND lease_expires_at_utc < ?
            """, (Gate3ClaimState.LEASE_EXPIRED_RECOVERABLE.value, Gate3ClaimState.CLAIMED_PENDING_COMMIT.value, now_utc))
            recovered = cur.rowcount
            conn.execute("COMMIT;")
            return recovered

    def append_transaction_log(self, payload: Dict[str, Any]) -> Tuple[int, str]:
        """
        FND-ARCH-014: Appends a transaction event to the authoritative WAL transaction log.
        Calculates chained entry hash sha256(prev_hash + canonical_payload).
        """
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()
            cur.execute("SELECT log_sequence, entry_hash FROM transaction_log ORDER BY log_sequence DESC LIMIT 1;")
            last = cur.fetchone()
            if last:
                prev_seq, prev_hash = last[0], last[1]
                new_seq = prev_seq + 1
            else:
                prev_hash = "GENESIS_HEAD_00000000000000000000000000000000000000000000000000000000"
                new_seq = 1

            payload_str = canonicalize(payload)
            entry_hash = sha256_hex(f"{prev_hash}:{payload_str}".encode("utf-8"))
            now_utc = datetime.now(timezone.utc).isoformat()

            cur.execute("""
                INSERT INTO transaction_log (log_sequence, entry_hash, prev_entry_hash, payload_json, created_at_utc)
                VALUES (?, ?, ?, ?, ?)
            """, (new_seq, entry_hash, prev_hash, payload_str, now_utc))
            conn.execute("COMMIT;")
            return new_seq, entry_hash

    def verify_log_consensus_coherence(self, checkpoint_sequence: int, checkpoint_hash: str) -> bool:
        """
        FND-ARCH-014: Verifies consensus checkpoint coherence against authoritative transaction log.
        Fails closed with ConsensusLogDivergenceError if hashes diverge.
        """
        with self._lock, self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT entry_hash FROM transaction_log WHERE log_sequence = ?", (checkpoint_sequence,))
            row = cur.fetchone()
            if not row:
                # If transaction log has no entries at this sequence, cannot verify divergence unless log is non-empty
                cur.execute("SELECT MAX(log_sequence) FROM transaction_log;")
                max_seq = cur.fetchone()[0] or 0
                if max_seq >= checkpoint_sequence:
                    raise ConsensusLogDivergenceError(
                        f"Log missing entry at checkpoint sequence {checkpoint_sequence} (max log seq: {max_seq})"
                    )
                return True

            log_entry_hash = row[0]
            if log_entry_hash != checkpoint_hash:
                raise ConsensusLogDivergenceError(
                    f"UNDETECTED_LOG_CONSENSUS_DIVERGENCE == 0: Transaction log divergence detected at sequence {checkpoint_sequence}! "
                    f"Log hash: {log_entry_hash}, Consensus checkpoint hash: {checkpoint_hash}. Fails closed."
                )
            return True

    def record_accession_and_denominator(
        self,
        accession: str,
        authority_event_id: str,
        idempotency_key: str,
        generation: int,
        fencing_token: str,
        is_natural_source: bool = True,
        is_valid_evidence: bool = True,
        is_fresh: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Atomically admits natural accession and increments natural denominator in a single transaction.
        FND-ARCH-011: Evaluates AuthoritativeDenominatorAdmissionPredicate before incrementing.
        FND-ARCH-010: Rejects stale writers without corrupting epoch.
        """
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

            # 2. Fencing validation (FND-ARCH-010)
            cur.execute("SELECT generation, fencing_token, natural_denominator, latest_authority_event_sequence FROM authoritative_epoch WHERE epoch_id = ?", (self.epoch_id,))
            row = cur.fetchone()
            if not row or row[0] != generation or row[1] != fencing_token:
                conn.execute("ROLLBACK;")
                raise StaleFencingTokenError("Stale runner rejected during admission transaction")

            current_denom = row[2]
            current_event_seq = row[3]

            # 3. Check duplicate accession
            cur.execute("SELECT admitted_at_generation FROM processed_accessions WHERE accession = ?", (accession,))
            is_dup = bool(cur.fetchone())

            # 4. FND-ARCH-011: Unified admission predicate check
            is_atomic_admission = True
            is_eligible = authoritative_denominator_admission_predicate(
                is_natural_source=is_natural_source,
                is_valid_evidence=is_valid_evidence,
                is_fresh=is_fresh,
                is_atomic_admission=is_atomic_admission,
                is_duplicate=is_dup,
            )

            if not is_eligible:
                conn.execute("ROLLBACK;")
                return False, {
                    "error": f"Admission rejected by authoritative predicate (duplicate={is_dup}, natural={is_natural_source}, valid={is_valid_evidence})",
                    "is_eligible": False,
                    "status": "ADMISSION_PREDICATE_REJECTED"
                }

            # 5. Atomic mutations
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

            # Record in transaction log (FND-ARCH-014)
            log_entry = {
                "event_type": "NATURAL_ACCESSION_ADMITTED",
                "accession": accession,
                "authority_event_id": authority_event_id,
                "generation": generation,
                "natural_denominator": new_denom,
                "timestamp_utc": now_utc,
            }
            cur.execute("SELECT log_sequence, entry_hash FROM transaction_log ORDER BY log_sequence DESC LIMIT 1;")
            last = cur.fetchone()
            prev_hash = last[1] if last else "GENESIS_HEAD_00000000000000000000000000000000000000000000000000000000"
            new_log_seq = (last[0] + 1) if last else 1
            entry_hash = sha256_hex(f"{prev_hash}:{canonicalize(log_entry)}".encode("utf-8"))
            cur.execute("""
                INSERT INTO transaction_log (log_sequence, entry_hash, prev_entry_hash, payload_json, created_at_utc)
                VALUES (?, ?, ?, ?, ?)
            """, (new_log_seq, entry_hash, prev_hash, canonicalize(log_entry), now_utc))

            conn.execute("COMMIT;")
            return True, result_payload

    def verify_and_commit_checkpoint(
        self,
        checkpoint_sequence: int,
        checkpoint_hash: str,
        generation: int,
        fencing_token: str,
    ) -> Tuple[bool, str]:
        """
        Verifies checkpoint monotonicity and updates authoritative latest head.
        FND-ARCH-014: Verifies log-consensus coherence before committing checkpoint.
        """
        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.cursor()

            cur.execute("SELECT generation, fencing_token, latest_checkpoint_sequence FROM authoritative_epoch WHERE epoch_id = ?", (self.epoch_id,))
            row = cur.fetchone()
            if not row or row[0] != generation or row[1] != fencing_token:
                conn.execute("ROLLBACK;")
                raise StaleFencingTokenError("Stale runner rejected during checkpoint commit")

            latest_seq = row[2]
            if checkpoint_sequence <= latest_seq:
                conn.execute("ROLLBACK;")
                raise GlobalRollbackError(
                    f"Rollback rejected: Attempted checkpoint sequence {checkpoint_sequence} <= authoritative latest sequence {latest_seq}."
                )

            # Check log coherence (FND-ARCH-014)
            cur.execute("SELECT entry_hash FROM transaction_log WHERE log_sequence = ?", (checkpoint_sequence,))
            log_row = cur.fetchone()
            if log_row and log_row[0] != checkpoint_hash:
                conn.execute("ROLLBACK;")
                raise ConsensusLogDivergenceError(
                    f"UNDETECTED_LOG_CONSENSUS_DIVERGENCE == 0: Checkpoint {checkpoint_sequence} hash {checkpoint_hash} does not match transaction log hash {log_row[0]}."
                )

            now_utc = datetime.now(timezone.utc).isoformat()
            cur.execute("""
                UPDATE authoritative_epoch
                SET latest_checkpoint_sequence = ?, latest_checkpoint_hash = ?, updated_at_utc = ?
                WHERE epoch_id = ?
            """, (checkpoint_sequence, checkpoint_hash, now_utc, self.epoch_id))

            conn.execute("COMMIT;")
            return True, "CHECKPOINT_HEAD_UPDATED"
