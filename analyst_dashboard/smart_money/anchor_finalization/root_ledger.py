"""
Finalization Root Ledger & Attempt Store Architecture (Sections 19, 20).
Implements append-only storage for external root payloads, signed receipts,
and root submission attempt logs with ACID guarantees and ambiguous commit recovery.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from analyst_dashboard.smart_money.canonicalization import canonicalize_json
from analyst_dashboard.smart_money.anchor_finalization.models import CriticalIntegrityError
from analyst_dashboard.smart_money.anchor_finalization.root_models import (
    FinalizationRootStatus,
    RootLookupResult,
    RootErrorClass,
    FinalizationRootPayload,
    FinalizationRootReceipt,
    FinalizationRootAttemptRecord,
)
from analyst_dashboard.smart_money.anchor_finalization.root_serializer import (
    compute_finalization_root_hash,
    derive_finalization_root_id,
    derive_finalization_root_idempotency_key,
)


class FinalizationRootLedgerStore:
    """
    ACID append-only store for Finalization Roots and Root Submission Attempts.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:")
            self._mem_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            self._mem_conn = None
        self._init_db()

    @contextmanager
    def _get_connection(self):
        if self._mem_conn is not None:
            yield self._mem_conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            try:
                yield conn
            finally:
                conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS finalization_roots (
                    root_id TEXT PRIMARY KEY,
                    parent_run_id TEXT NOT NULL UNIQUE,
                    parent_run_final_hash TEXT NOT NULL,
                    finalization_ledger_root_hash TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    root_payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    receipt_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS finalization_root_attempts (
                    root_attempt_id TEXT PRIMARY KEY,
                    root_id TEXT NOT NULL,
                    parent_run_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_payload_sha256 TEXT NOT NULL,
                    request_started_at TEXT NOT NULL,
                    request_completed_at TEXT NOT NULL,
                    provider_response_hash TEXT NOT NULL,
                    result TEXT NOT NULL,
                    error_class TEXT NOT NULL,
                    lookup_required INTEGER NOT NULL,
                    lookup_result TEXT,
                    FOREIGN KEY (root_id) REFERENCES finalization_roots(root_id)
                );
            """)
            conn.commit()

    def record_root_payload(
        self,
        payload: FinalizationRootPayload,
        status: FinalizationRootStatus = FinalizationRootStatus.PAYLOAD_SEALED,
    ) -> str:
        """
        Records canonical root payload with strict idempotency and conflict detection (Section 9).
        """
        root_id = derive_finalization_root_id(payload.parent_run_id, payload.parent_run_final_hash)
        root_hash = compute_finalization_root_hash(payload)
        idempotency_key = derive_finalization_root_idempotency_key(
            payload.parent_run_id, payload.parent_run_final_hash, root_hash
        )
        payload_json = json.dumps(payload.to_dict())
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        with self._get_connection() as conn:
            # Check for existing run root
            cursor = conn.execute(
                "SELECT root_id, finalization_ledger_root_hash, root_payload_json FROM finalization_roots WHERE parent_run_id = ?",
                (payload.parent_run_id,),
            )
            existing = cursor.fetchone()
            if existing:
                ex_id, ex_hash, ex_json = existing
                if ex_hash != root_hash:
                    raise CriticalIntegrityError(
                        f"CRITICAL_INTEGRITY_CONFLICT: Same parent_run_id '{payload.parent_run_id}' "
                        f"with conflicting root hash '{root_hash}' vs existing '{ex_hash}'."
                    )
                if ex_json != payload_json:
                    raise CriticalIntegrityError(
                        f"CRITICAL_INTEGRITY_FAILURE: Same idempotency key with different payload for run '{payload.parent_run_id}'."
                    )
                # Duplicate identical submission is idempotent
                return ex_id

            conn.execute(
                """
                INSERT INTO finalization_roots (
                    root_id, parent_run_id, parent_run_final_hash,
                    finalization_ledger_root_hash, idempotency_key,
                    root_payload_json, status, receipt_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    root_id,
                    payload.parent_run_id,
                    payload.parent_run_final_hash,
                    root_hash,
                    idempotency_key,
                    payload_json,
                    status.value,
                    now,
                    now,
                ),
            )
            conn.commit()
            return root_id

    def record_root_attempt(self, attempt: FinalizationRootAttemptRecord) -> None:
        """Appends an attempt record to the attempt ledger (Section 19)."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO finalization_root_attempts (
                    root_attempt_id, root_id, parent_run_id, attempt_number,
                    idempotency_key, request_payload_sha256, request_started_at,
                    request_completed_at, provider_response_hash, result,
                    error_class, lookup_required, lookup_result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.root_attempt_id,
                    attempt.root_id,
                    attempt.parent_run_id,
                    attempt.attempt_number,
                    attempt.idempotency_key,
                    attempt.request_payload_sha256,
                    attempt.request_started_at,
                    attempt.request_completed_at,
                    attempt.provider_response_hash,
                    attempt.result,
                    attempt.error_class.value,
                    1 if attempt.lookup_required else 0,
                    attempt.lookup_result.value if attempt.lookup_result else None,
                ),
            )
            conn.commit()

    def lookup_existing_root(
        self,
        parent_run_id: str,
        expected_root_hash: str,
    ) -> Tuple[RootLookupResult, Optional[FinalizationRootPayload]]:
        """
        Ambiguous commit recovery lookup (Section 20).
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT finalization_ledger_root_hash, root_payload_json FROM finalization_roots WHERE parent_run_id = ?",
                (parent_run_id,),
            )
            row = cursor.fetchone()
            if not row:
                return RootLookupResult.NOT_FOUND, None

            found_hash, payload_json = row
            payload_dict = json.loads(payload_json)
            payload = FinalizationRootPayload(
                root_schema_version=payload_dict["root_schema_version"],
                root_policy_id=payload_dict["root_policy_id"],
                root_policy_version=payload_dict["root_policy_version"],
                parent_run_id=payload_dict["parent_run_id"],
                parent_run_final_hash=payload_dict["parent_run_final_hash"],
                finalization_event_count=payload_dict["finalization_event_count"],
                first_finalization_sequence=payload_dict["first_finalization_sequence"],
                last_finalization_sequence=payload_dict["last_finalization_sequence"],
                finalization_event_hashes=payload_dict["finalization_event_hashes"],
                authority_eligibility=payload_dict["authority_eligibility"],
                authority_effective_status=payload_dict["authority_effective_status"],
                authority_became_effective_at=payload_dict["authority_became_effective_at"],
                anchor_request_ids=payload_dict["anchor_request_ids"],
                anchor_receipt_hashes=payload_dict["anchor_receipt_hashes"],
                policy_set_hash=payload_dict["policy_set_hash"],
                created_at_utc=payload_dict["created_at_utc"],
            )

            if found_hash == expected_root_hash:
                return RootLookupResult.FOUND_MATCHING, payload
            return RootLookupResult.FOUND_CONFLICTING, payload

    def confirm_root(self, root_id: str, receipt: FinalizationRootReceipt) -> None:
        """Transitions root to CONFIRMED and attaches receipt (Section 21)."""
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        receipt_json = json.dumps(receipt.to_dict())
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE finalization_roots SET status = ?, receipt_json = ?, updated_at = ? WHERE root_id = ?",
                (FinalizationRootStatus.CONFIRMED.value, receipt_json, now, root_id),
            )
            conn.commit()

    def update_root_status(self, root_id: str, new_status: FinalizationRootStatus) -> None:
        """Updates lifecycle status for a root record."""
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE finalization_roots SET status = ?, updated_at = ? WHERE root_id = ?",
                (new_status.value, now, root_id),
            )
            conn.commit()

    def get_root(self, root_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves root record by root_id."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT root_id, parent_run_id, parent_run_final_hash, finalization_ledger_root_hash, "
                "idempotency_key, root_payload_json, status, receipt_json, created_at, updated_at "
                "FROM finalization_roots WHERE root_id = ?",
                (root_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "root_id": row[0],
                "parent_run_id": row[1],
                "parent_run_final_hash": row[2],
                "finalization_ledger_root_hash": row[3],
                "idempotency_key": row[4],
                "root_payload": json.loads(row[5]),
                "status": row[6],
                "receipt": json.loads(row[7]) if row[7] else None,
                "created_at": row[8],
                "updated_at": row[9],
            }

    def get_attempts(self, root_id: str) -> List[FinalizationRootAttemptRecord]:
        """Retrieves all attempts for a given root in order."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT root_attempt_id, root_id, parent_run_id, attempt_number, idempotency_key, "
                "request_payload_sha256, request_started_at, request_completed_at, provider_response_hash, "
                "result, error_class, lookup_required, lookup_result "
                "FROM finalization_root_attempts WHERE root_id = ? ORDER BY attempt_number ASC",
                (root_id,),
            )
            rows = cursor.fetchall()
            attempts = []
            for r in rows:
                attempts.append(
                    FinalizationRootAttemptRecord(
                        root_attempt_id=r[0],
                        root_id=r[1],
                        parent_run_id=r[2],
                        attempt_number=r[3],
                        idempotency_key=r[4],
                        request_payload_sha256=r[5],
                        request_started_at=r[6],
                        request_completed_at=r[7],
                        provider_response_hash=r[8],
                        result=r[9],
                        error_class=RootErrorClass(r[10]),
                        lookup_required=bool(r[11]),
                        lookup_result=RootLookupResult(r[12]) if r[12] else None,
                    )
                )
            return attempts
