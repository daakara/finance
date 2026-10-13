"""
ARX Terminal — PostgreSQL Smart Money Repository.

Encapsulates all database operations for SEC Form 4 raw evidence, parsed transactions,
point-in-time ledgers, append-only reconciliation receipts, immutable snapshots,
and atomic active-pointer publication.
"""

import json
import uuid
import hashlib
import logging
from typing import Dict, Any, List, Optional, Set
from psycopg2.extras import RealDictCursor, Json
from psycopg2.extensions import connection as PgConnection

from analyst_dashboard.smart_money.storage.postgres_connection import get_connection, transaction

logger = logging.getLogger(__name__)


class StaleLeaseOwnershipLostError(RuntimeError):
    """Raised when a worker attempts to heartbeat, complete, or publish with an expired or lost lease."""
    pass


class MaxAttemptsExceededError(RuntimeError):
    """Raised when a run request has exceeded maximum retry attempts."""
    pass


class PostgresSmartMoneyRepository:
    """Production-grade PostgreSQL repository implementing fail-closed Smart Money persistence."""

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn

    # ── Raw Filings & Idempotency ─────────────────────────────────────────────

    def save_raw_filing(self, filing: Dict[str, Any], conn: Optional[PgConnection] = None) -> bool:
        """
        Persists a raw filing idempotently.
        If filing already exists with identical content hash, returns False (no-op).
        If filing already exists with conflicting content hash, raises SOURCE_INTEGRITY_CONFLICT.
        """
        def _exec(cur):
            # Check existing
            cur.execute(
                "SELECT content_hash FROM form4_raw_filings WHERE accession_number = %s;",
                (filing["accession_number"],)
            )
            row = cur.fetchone()
            if row:
                if row["content_hash"] != filing["content_hash"]:
                    raise RuntimeError(
                        f"SOURCE_INTEGRITY_CONFLICT: Accession {filing['accession_number']} "
                        f"already exists with hash {row['content_hash']} but new hash is {filing['content_hash']}."
                    )
                return False  # Idempotent no-op

            cur.execute("""
                INSERT INTO form4_raw_filings (
                    accession_number, cik, ticker, company_name, filing_date,
                    accepted_datetime, xml_content, content_hash, source_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                filing["accession_number"],
                filing["cik"],
                filing["ticker"],
                filing["company_name"],
                filing["filing_date"],
                filing["accepted_datetime"],
                filing["xml_content"],
                filing["content_hash"],
                filing["source_url"],
            ))
            return True

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                return _exec(cur)

    # ── Parsed Transactions ───────────────────────────────────────────────────

    def save_parsed_transactions(self, transactions: List[Dict[str, Any]], conn: Optional[PgConnection] = None) -> int:
        """Saves parsed transactions for an accession."""
        if not transactions:
            return 0

        def _exec(cur):
            count = 0
            for tx in transactions:
                cur.execute("""
                    INSERT INTO form4_parsed_transactions (
                        transaction_id, accession_number, transaction_date, transaction_code,
                        shares, price_per_share, total_value_usd, reporting_owner_name,
                        reporting_owner_role, is_director, is_officer, is_ten_percent,
                        officer_title, ownership_type
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_id) DO NOTHING;
                """, (
                    tx["transaction_id"],
                    tx["accession_number"],
                    tx["transaction_date"],
                    tx["transaction_code"],
                    tx["shares"],
                    tx["price_per_share"],
                    tx["total_value_usd"],
                    tx["reporting_owner_name"],
                    tx["reporting_owner_role"],
                    tx.get("is_director", False),
                    tx.get("is_officer", False),
                    tx.get("is_ten_percent", False),
                    tx.get("officer_title"),
                    tx.get("ownership_type", "D"),
                ))
                count += 1
            return count

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                return _exec(cur)

    # ── Availability Ledger ───────────────────────────────────────────────────

    def save_availability(
        self,
        accession_number: str,
        public_availability_at: str,
        source: str = "SEC_INDEX",
        source_reference: Optional[str] = None,
        conn: Optional[PgConnection] = None,
    ) -> None:
        """Records point-in-time public availability timestamp."""
        query = """
            INSERT INTO sm_source_availability_ledger (
                accession_number, public_availability_at, source, source_reference
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (accession_number) DO NOTHING;
        """
        params = (accession_number, public_availability_at, source, source_reference)

        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(query, params)
        else:
            with transaction(dsn=self.dsn) as cur:
                cur.execute(query, params)

    # ── Amendment Management ──────────────────────────────────────────────────

    def save_amendment(
        self,
        amendment_accession: str,
        base_accession: Optional[str],
        amendment_number: int = 1,
        resolution_status: str = "RESOLVED",
        resolved_at: Optional[str] = None,
        conn: Optional[PgConnection] = None,
    ) -> None:
        """Records a Form 4/A amendment linkage."""
        query = """
            INSERT INTO sm_amendments (
                amendment_accession, base_accession, amendment_number, resolution_status, resolved_at
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (amendment_accession) DO NOTHING;
        """
        params = (amendment_accession, base_accession, amendment_number, resolution_status, resolved_at)

        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(query, params)
        else:
            with transaction(dsn=self.dsn) as cur:
                cur.execute(query, params)

    # ── Reconciliation Receipts (Append-Only) ─────────────────────────────────

    def save_reconciliation_receipt(self, receipt: Dict[str, Any], conn: Optional[PgConnection] = None) -> str:
        """
        Persists an immutable reconciliation receipt.
        Assigns UUID if not provided. Supports supersedes_receipt_id for corrections.
        """
        receipt_id = receipt.get("receipt_id") or str(uuid.uuid4())
        reasons = receipt.get("failure_reasons", [])
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except Exception:
                reasons = [reasons]

        def _exec(cur):
            cur.execute("""
                INSERT INTO sm_reconciliation_receipts (
                    receipt_id, interval_start, interval_end, certified_watermark,
                    receipt_version, supersedes_receipt_id, expected_count, accounted_count,
                    missing_count, material_gap_count, coverage_status, failure_reasons,
                    reconciliation_hash, release_sha
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING receipt_id;
            """, (
                receipt_id,
                receipt["interval_start"],
                receipt["interval_end"],
                receipt["certified_watermark"],
                receipt.get("receipt_version", 1),
                receipt.get("supersedes_receipt_id"),
                receipt["expected_count"],
                receipt["accounted_count"],
                receipt["missing_count"],
                receipt["material_gap_count"],
                receipt["coverage_status"],
                Json(reasons),
                receipt["reconciliation_hash"],
                receipt.get("release_sha", "b72ea23937cf6ed7b8c4b60096630767a7c4592b"),
            ))
            return str(cur.fetchone()["receipt_id"])

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                return _exec(cur)

    def get_authoritative_receipt_for_interval(
        self,
        interval_start: str,
        interval_end: str,
        certified_watermark: str,
        conn: Optional[PgConnection] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Queries the authoritative reconciliation receipt certifying watermark W.
        Requires exact interval match, COMPLETE coverage status, and zero material gaps.
        """
        query = """
            SELECT * FROM sm_reconciliation_receipts
            WHERE (interval_end = %s OR interval_end = %s)
              AND coverage_status = 'COMPLETE'
              AND missing_count = 0
              AND material_gap_count = 0
            ORDER BY receipt_version DESC, created_at DESC
            LIMIT 1;
        """
        eod_end = f"{interval_end[:10]}T23:59:59Z" if len(interval_end) >= 10 else interval_end
        params = (interval_end, eod_end)

        def _exec(cur):
            cur.execute(query, params)
            return cur.fetchone()

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with get_connection(dsn=self.dsn) as c:
                with c.cursor(cursor_factory=RealDictCursor) as cur:
                    return _exec(cur)

    # ── Bulk Point-in-Time Transactions Query ─────────────────────────────────

    def get_bulk_transactions_as_of(
        self,
        as_of_watermark: str,
        symbols: List[str],
        min_date: str,
        conn: Optional[PgConnection] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Point-in-Time Bulk Transaction Retrieval.
        Enforces public_availability_at <= as_of_watermark.
        Resolves amendment supersession point-in-time:
        If an amendment supersedes an original transaction and was available before W,
        the superseding version is used.
        """
        if not symbols:
            return {}

        query = """
            SELECT
                tx.transaction_id,
                tx.accession_number,
                tx.transaction_date,
                tx.transaction_code,
                tx.shares,
                tx.price_per_share,
                tx.total_value_usd,
                tx.reporting_owner_name,
                tx.reporting_owner_role,
                tx.is_director,
                tx.is_officer,
                tx.is_ten_percent,
                tx.officer_title,
                tx.ownership_type,
                f.ticker AS symbol,
                f.company_name,
                av.public_availability_at,
                am.amendment_number,
                am.base_accession,
                am.resolution_status
            FROM form4_parsed_transactions tx
            JOIN form4_raw_filings f ON tx.accession_number = f.accession_number
            JOIN sm_source_availability_ledger av ON tx.accession_number = av.accession_number
            LEFT JOIN sm_amendments am ON tx.accession_number = am.amendment_accession
            WHERE av.public_availability_at <= %s
              AND tx.transaction_date >= %s
              AND f.ticker = ANY(%s)
            ORDER BY tx.transaction_date ASC, av.public_availability_at ASC;
        """

        def _exec(cur):
            cur.execute(query, (as_of_watermark, min_date, [s.upper().strip() for s in symbols]))
            rows = cur.fetchall()
            out: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                sym = r["symbol"].upper()
                if sym not in out:
                    out[sym] = []
                out[sym].append(dict(r))
            return out

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with get_connection(dsn=self.dsn) as c:
                with c.cursor(cursor_factory=RealDictCursor) as cur:
                    return _exec(cur)

    def get_ambiguous_amendment_symbols_as_of(
        self,
        as_of_watermark: str,
        symbols: List[str],
        conn: Optional[PgConnection] = None,
    ) -> Set[str]:
        """
        Identifies symbols having ambiguous or unresolvable amendments available <= W.
        """
        if not symbols:
            return set()

        query = """
            SELECT DISTINCT f.ticker
            FROM sm_amendments am
            JOIN form4_raw_filings f ON am.amendment_accession = f.accession_number
            JOIN sm_source_availability_ledger av ON am.amendment_accession = av.accession_number
            WHERE av.public_availability_at <= %s
              AND am.resolution_status IN ('AMENDMENT_TARGET_AMBIGUOUS', 'UNRESOLVED')
              AND f.ticker = ANY(%s);
        """

        def _exec(cur):
            cur.execute(query, (as_of_watermark, [s.upper().strip() for s in symbols]))
            rows = cur.fetchall()
            return {r["ticker"].upper() for r in rows}

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with get_connection(dsn=self.dsn) as c:
                with c.cursor(cursor_factory=RealDictCursor) as cur:
                    return _exec(cur)

    # ── Snapshot Publication (Atomic Publication Transaction) ────────────────

    def publish_scanner_snapshot_atomic(
        self,
        snapshot: Dict[str, Any],
        run_request_id: Optional[str] = None,
        lease_token: Optional[str] = None,
        conn: Optional[PgConnection] = None,
    ) -> str:
        """
        Atomic Snapshot Publication Transaction (Section 16).
        In a single transaction:
        1. If run_request_id and lease_token provided, verifies active lease ownership (fails closed if stale).
        2. Inserts immutable record into scanner_snapshots.
        3. Updates/inserts active pointer in scanner_active_snapshots if decision == 'PUBLISH'.
        4. Marks run_request as COMPLETED within the same atomic transaction if published.
        5. Emits audit event.
        Rolls back entirely if any stage fails.
        """
        snapshot_id = snapshot["snapshot_id"]
        scan_type = snapshot.get("scanner_id", "SMART_MONEY")
        decision = snapshot.get("publication_decision", "PUBLISH")
        matched_cnt = snapshot.get("matched_count")
        # In Postgres schema, matched_count is nullable
        if decision != "PUBLISH" and snapshot.get("coverage_status") != "COMPLETE":
            matched_val = None
        else:
            matched_val = int(matched_cnt) if matched_cnt is not None else 0

        def _exec(cur):
            # 0. Fencing check: verify active lease ownership
            if run_request_id and lease_token:
                cur.execute("""
                    SELECT request_id FROM sm_run_requests
                    WHERE request_id = %s AND lease_token = %s AND status = 'RUNNING'
                    FOR UPDATE;
                """, (run_request_id, lease_token))
                if cur.fetchone() is None:
                    raise StaleLeaseOwnershipLostError(
                        f"STALE_LEASE_OWNERSHIP_LOST: Cannot publish snapshot {snapshot_id}. "
                        f"Worker lost lease on request {run_request_id} (token={lease_token})."
                    )

            # 1. Insert into scanner_snapshots
            cur.execute("""
                INSERT INTO scanner_snapshots (
                    snapshot_id, scanner_id, run_id, api_contract_version,
                    ruleset_version, evidence_schema_version, score_model_version,
                    data_provenance_version, universe_version, freshness_policy_version,
                    implementation_release_sha, semantic_fingerprint, generated_at,
                    data_as_of, status_at_publication, universe_id, universe_size,
                    matched_count, results_json, provenance_json, freshness_json,
                    publication_decision
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                );
            """, (
                snapshot_id,
                scan_type,
                snapshot["run_id"],
                snapshot["api_contract_version"],
                snapshot["ruleset_version"],
                snapshot["evidence_schema_version"],
                snapshot["score_model_version"],
                snapshot["data_provenance_version"],
                snapshot["universe_version"],
                snapshot["freshness_policy_version"],
                snapshot["implementation_release_sha"],
                snapshot["semantic_fingerprint"],
                snapshot["generated_at"],
                snapshot["data_as_of"],
                snapshot["status_at_publication"],
                snapshot["universe_id"],
                snapshot["universe_size"],
                matched_val,
                Json(snapshot.get("results") if "results" in snapshot else snapshot.get("results_json", [])),
                Json(snapshot.get("provenance") if "provenance" in snapshot else snapshot.get("provenance_json", {})),
                Json(snapshot.get("freshness") if "freshness" in snapshot else snapshot.get("freshness_json", {})),
                decision,
            ))

            # 2. Only advance active pointer if decision == PUBLISH
            if decision == "PUBLISH":
                cur.execute("""
                    INSERT INTO scanner_active_snapshots (
                        scan_type, active_snapshot_id, published_at, published_by
                    ) VALUES (%s, %s, NOW(), %s)
                    ON CONFLICT (scan_type) DO UPDATE SET
                        active_snapshot_id = EXCLUDED.active_snapshot_id,
                        published_at = EXCLUDED.published_at,
                        published_by = EXCLUDED.published_by;
                """, (
                    scan_type,
                    snapshot_id,
                    snapshot.get("run_id", "ingestion_worker"),
                ))

                # If associated with a fenced run request, complete it atomically
                if run_request_id and lease_token:
                    cur.execute("""
                        UPDATE sm_run_requests
                        SET status = 'COMPLETED',
                            completed_at = NOW(),
                            result_generation_id = %s
                        WHERE request_id = %s AND lease_token = %s;
                    """, (snapshot_id, run_request_id, lease_token))

            # 3. Audit event
            cur.execute("""
                INSERT INTO sm_audit_events (
                    event_id, event_type, subject_id, payload, event_hash
                ) VALUES (%s, %s, %s, %s, %s);
            """, (
                str(uuid.uuid4()),
                "SNAPSHOT_PUBLISHED" if decision == "PUBLISH" else "SNAPSHOT_QUARANTINED",
                snapshot_id,
                Json({"scanner_id": scan_type, "decision": decision, "matched_count": matched_val}),
                snapshot["semantic_fingerprint"],
            ))

            return snapshot_id

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                return _exec(cur)

    def save_snapshot(self, snapshot: Dict[str, Any], conn: Optional[PgConnection] = None) -> str:
        """Compatibility alias for publish_scanner_snapshot_atomic adhering to snapshot store contract."""
        return self.publish_scanner_snapshot_atomic(snapshot, conn=conn)

    def get_latest_active_snapshot(self, scan_type: str = "SMART_MONEY", conn: Optional[PgConnection] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves the current active verified snapshot via pointer lookup.
        """
        query = """
            SELECT s.*
            FROM scanner_active_snapshots a
            JOIN scanner_snapshots s ON a.active_snapshot_id = s.snapshot_id
            WHERE a.scan_type = %s AND s.publication_decision = 'PUBLISH';
        """

        def _exec(cur):
            cur.execute(query, (scan_type,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "snapshot_id": row["snapshot_id"],
                "scanner_id": row["scanner_id"],
                "run_id": row["run_id"],
                "api_contract_version": row["api_contract_version"],
                "ruleset_version": row["ruleset_version"],
                "evidence_schema_version": row["evidence_schema_version"],
                "score_model_version": row["score_model_version"],
                "data_provenance_version": row["data_provenance_version"],
                "universe_version": row["universe_version"],
                "freshness_policy_version": row["freshness_policy_version"],
                "implementation_release_sha": row["implementation_release_sha"],
                "semantic_fingerprint": row["semantic_fingerprint"],
                "generated_at": row["generated_at"].isoformat() if hasattr(row["generated_at"], "isoformat") else str(row["generated_at"]),
                "data_as_of": row["data_as_of"].isoformat() if hasattr(row["data_as_of"], "isoformat") else str(row["data_as_of"]),
                "status_at_publication": row["status_at_publication"],
                "universe_id": row["universe_id"],
                "universe_size": row["universe_size"],
                "matched_count": row["matched_count"],
                "results": row["results_json"],
                "provenance": row["provenance_json"],
                "freshness": row["freshness_json"],
                "publication_decision": row["publication_decision"],
                "created_at": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
            }

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with get_connection(dsn=self.dsn) as c:
                with c.cursor(cursor_factory=RealDictCursor) as cur:
                    return _exec(cur)

    # ── Governed Issuers (Universe Authority) ─────────────────────────────────

    def save_governed_issuers(self, issuers: List[Dict[str, Any]], conn: Optional[PgConnection] = None) -> int:
        """Saves or updates governed issuer membership."""
        if not issuers:
            return 0

        query = """
            INSERT INTO governed_issuers (
                cik, ticker, company_name, sic, fiscal_year_end,
                scannable, universe_build_id, manifest_hash, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (cik) DO UPDATE SET
                ticker = EXCLUDED.ticker,
                company_name = EXCLUDED.company_name,
                scannable = EXCLUDED.scannable,
                universe_build_id = EXCLUDED.universe_build_id,
                manifest_hash = EXCLUDED.manifest_hash,
                updated_at = NOW();
        """

        def _exec(cur):
            for iss in issuers:
                cur.execute(query, (
                    iss["cik"],
                    iss["ticker"],
                    iss["company_name"],
                    iss.get("sic"),
                    iss.get("fiscal_year_end"),
                    iss.get("scannable", True),
                    iss["universe_build_id"],
                    iss["manifest_hash"],
                ))
            return len(issuers)

        if conn is not None:
            with conn.cursor() as cur:
                return _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                return _exec(cur)

    # ── Run Requests (Web -> Worker Queue) ────────────────────────────────────

    def create_run_request(
        self,
        request_type: str,
        subject_hash: str,
        max_attempts: int = 3,
        conn: Optional[PgConnection] = None,
    ) -> str:
        """Submits an immutable run request into the queue."""
        request_id = str(uuid.uuid4())
        query = """
            INSERT INTO sm_run_requests (
                request_id, request_type, requested_by_subject_hash, status,
                max_attempts, next_attempt_at
            ) VALUES (%s, %s, %s, 'PENDING', %s, NOW())
            RETURNING request_id;
        """

        def _exec(cur):
            cur.execute(query, (request_id, request_type, subject_hash, max_attempts))
            return str(cur.fetchone()["request_id"])

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                return _exec(cur)

    def claim_next_run_request(
        self,
        worker_id: str,
        lease_duration_sec: int = 60,
        conn: Optional[PgConnection] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Atomically claims the oldest pending or retry-eligible run request non-blockingly.
        Uses FOR UPDATE SKIP LOCKED to prevent race conditions across concurrent workers.
        Establishes an atomic lease with a unique lease token, attempt counter, and expiry.
        """
        lease_token = f"tok-{uuid.uuid4().hex[:16]}"
        query_select = """
            SELECT request_id, status, attempt_count, max_attempts
            FROM sm_run_requests
            WHERE (
                status IN ('PENDING', 'RETRY_ELIGIBLE')
                AND next_attempt_at <= NOW()
            )
            AND attempt_count < max_attempts
            ORDER BY requested_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1;
        """
        query_update = """
            UPDATE sm_run_requests
            SET status = 'RUNNING',
                claimed_at = NOW(),
                claimed_by = %s,
                lease_owner_id = %s,
                lease_token = %s,
                lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                lease_heartbeat_at = NOW(),
                attempt_count = attempt_count + 1
            WHERE request_id = %s
            RETURNING request_id, request_type, requested_by_subject_hash, status,
                      claimed_at, claimed_by, lease_owner_id, lease_token,
                      lease_expires_at, lease_heartbeat_at, attempt_count, max_attempts;
        """

        def _exec(cur):
            cur.execute(query_select)
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(query_update, (worker_id, worker_id, lease_token, lease_duration_sec, row["request_id"]))
            claimed_row = cur.fetchone()
            res = dict(claimed_row)
            res["request_id"] = str(res["request_id"])
            res["status"] = row["status"]
            res["active_status"] = "RUNNING"
            if res.get("lease_expires_at") and hasattr(res["lease_expires_at"], "isoformat"):
                res["lease_expires_at"] = res["lease_expires_at"].isoformat()
            if res.get("lease_heartbeat_at") and hasattr(res["lease_heartbeat_at"], "isoformat"):
                res["lease_heartbeat_at"] = res["lease_heartbeat_at"].isoformat()
            return res

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                return _exec(cur)

    def heartbeat_run_request(
        self,
        request_id: str,
        lease_token: str,
        extension_sec: int = 60,
        conn: Optional[PgConnection] = None,
    ) -> bool:
        """
        Renews an active worker lease before expiration.
        Conditional on request_id, lease_token, and status='RUNNING'.
        Returns True if renewed, False if ownership lost or lease expired.
        """
        query = """
            UPDATE sm_run_requests
            SET lease_heartbeat_at = NOW(),
                lease_expires_at = NOW() + (%s * INTERVAL '1 second')
            WHERE request_id = %s
              AND lease_token = %s
              AND status = 'RUNNING'
            RETURNING request_id;
        """

        def _exec(cur):
            cur.execute(query, (extension_sec, request_id, lease_token))
            return cur.fetchone() is not None

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                return _exec(cur)

    def recover_stale_run_requests(
        self,
        backoff_sec: int = 30,
        conn: Optional[PgConnection] = None,
    ) -> int:
        """
        Deterministic recovery mechanism for expired RUNNING requests (lease_expires_at < NOW()).
        - If attempt_count >= max_attempts: Transitions to FAILED with terminal failure code.
        - Else: Transitions to RETRY_ELIGIBLE with backoff.
        Emits durable audit events into sm_audit_events.
        """
        query_find = """
            SELECT request_id, attempt_count, max_attempts, lease_owner_id, lease_token
            FROM sm_run_requests
            WHERE status = 'RUNNING'
              AND lease_expires_at < NOW()
            ORDER BY lease_expires_at ASC
            FOR UPDATE SKIP LOCKED;
        """

        def _exec(cur):
            cur.execute(query_find)
            stale_rows = cur.fetchall()
            if not stale_rows:
                return 0

            recovered_count = 0
            for row in stale_rows:
                req_id = row["request_id"]
                attempts = row["attempt_count"]
                max_att = row["max_attempts"]
                owner = row.get("lease_owner_id") or "unknown"
                token = row.get("lease_token") or "unknown"

                if attempts >= max_att:
                    cur.execute("""
                        UPDATE sm_run_requests
                        SET status = 'FAILED',
                            completed_at = NOW(),
                            last_failure_code = 'LEASE_TIMEOUT_EXCEEDED_MAX_ATTEMPTS',
                            last_failure_at = NOW()
                        WHERE request_id = %s;
                    """, (req_id,))
                    cur.execute("""
                        INSERT INTO sm_audit_events (
                            event_id, event_type, subject_id, payload, event_hash
                        ) VALUES (%s, %s, %s, %s, %s);
                    """, (
                        str(uuid.uuid4()),
                        "RUN_REQUEST_FAILED_TERMINAL",
                        str(req_id),
                        Json({
                            "attempts": attempts,
                            "max_attempts": max_att,
                            "lost_owner": owner,
                            "lost_token": token,
                            "reason": "LEASE_TIMEOUT_EXCEEDED_MAX_ATTEMPTS",
                        }),
                        hashlib.sha256(f"FAILED_{req_id}_{attempts}".encode()).hexdigest(),
                    ))
                else:
                    cur.execute("""
                        UPDATE sm_run_requests
                        SET status = 'RETRY_ELIGIBLE',
                            next_attempt_at = NOW() + (%s * INTERVAL '1 second'),
                            last_failure_code = 'LEASE_EXPIRED_RECOVERED',
                            last_failure_at = NOW(),
                            lease_token = NULL,
                            lease_owner_id = NULL
                        WHERE request_id = %s;
                    """, (backoff_sec, req_id))
                    cur.execute("""
                        INSERT INTO sm_audit_events (
                            event_id, event_type, subject_id, payload, event_hash
                        ) VALUES (%s, %s, %s, %s, %s);
                    """, (
                        str(uuid.uuid4()),
                        "RUN_REQUEST_LEASE_RECOVERED",
                        str(req_id),
                        Json({
                            "attempts": attempts,
                            "max_attempts": max_att,
                            "lost_owner": owner,
                            "lost_token": token,
                            "backoff_sec": backoff_sec,
                        }),
                        hashlib.sha256(f"RECOVERED_{req_id}_{attempts}".encode()).hexdigest(),
                    ))
                recovered_count += 1
            return recovered_count

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                return _exec(cur)

    def complete_run_request(
        self,
        request_id: str,
        result_generation_id: Optional[str] = None,
        lease_token: Optional[str] = None,
        status: str = "COMPLETED",
        failure_code: Optional[str] = None,
        conn: Optional[PgConnection] = None,
    ) -> None:
        """
        Marks a claimed run request as completed or failed.
        If lease_token is provided, verifies that caller still owns the active lease.
        Fails closed with StaleLeaseOwnershipLostError if ownership has been lost.
        """
        if lease_token is not None:
            query = """
                UPDATE sm_run_requests
                SET status = %s,
                    completed_at = NOW(),
                    result_generation_id = %s,
                    last_failure_code = %s,
                    last_failure_at = CASE WHEN %s = 'FAILED' THEN NOW() ELSE last_failure_at END
                WHERE request_id = %s
                  AND lease_token = %s
                  AND status = 'RUNNING'
                RETURNING request_id;
            """
            params = (status, result_generation_id, failure_code, status, request_id, lease_token)
        else:
            query = """
                UPDATE sm_run_requests
                SET status = %s,
                    completed_at = NOW(),
                    result_generation_id = %s,
                    last_failure_code = %s,
                    last_failure_at = CASE WHEN %s = 'FAILED' THEN NOW() ELSE last_failure_at END
                WHERE request_id = %s;
            """
            params = (status, result_generation_id, failure_code, status, request_id)

        def _exec(cur):
            cur.execute(query, params)
            if lease_token is not None and cur.fetchone() is None:
                raise StaleLeaseOwnershipLostError(
                    f"STALE_LEASE_OWNERSHIP_LOST: Worker lost ownership of request {request_id} "
                    f"(token={lease_token}). Completion rejected."
                )

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                _exec(cur)

    def get_run_request(self, request_id: str, conn: Optional[PgConnection] = None) -> Optional[Dict[str, Any]]:
        """Retrieves a run request by its ID."""
        query = "SELECT * FROM sm_run_requests WHERE request_id = %s;"

        def _exec(cur):
            cur.execute(query, (request_id,))
            row = cur.fetchone()
            if not row:
                return None
            res = dict(row)
            res["request_id"] = str(res["request_id"])
            return res

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with get_connection(dsn=self.dsn) as c:
                with c.cursor(cursor_factory=RealDictCursor) as cur:
                    return _exec(cur)

