"""
ARX Terminal — PostgreSQL Smart Money Repository.

Encapsulates all database operations for SEC Form 4 raw evidence, parsed transactions,
point-in-time ledgers, append-only reconciliation receipts, immutable snapshots,
and atomic active-pointer publication.
"""

import os
import json
import uuid
import hashlib
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    from psycopg2.extensions import connection as PgConnection
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    Json = None
    PgConnection = Any
    PSYCOPG2_AVAILABLE = False

from analyst_dashboard.smart_money.storage.postgres_connection import get_connection, transaction

logger = logging.getLogger(__name__)


def _to_iso(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        try:
            val = datetime.fromisoformat(val.replace("Z", "+00:00"))
        except Exception:
            return val
    if hasattr(val, "astimezone"):
        return val.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%dT%H:%M:%SZ")
    s = str(val)
    if s.endswith("+00:00"):
        s = s[:-6] + "Z"
    return s


def compute_canonical_receipt_hash(
    schema_version: int,
    ticker: str,
    issuer_cik: str,
    interval_start: Any,
    interval_end: Any,
    certified_watermark: Any,
    receipt_version: int,
    supersedes_receipt_id: Optional[str],
    accession_inventory: List[str],
    source_hashes: Dict[str, str],
    material_gaps: List[Dict[str, Any]],
    zero_activity_proof: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Computes a deterministic cryptographic hash across canonical structured receipt fields.
    Includes complete material gap records, source inventory, and catalog proofs.
    """
    sorted_gaps = []
    for g in sorted(
        material_gaps or [],
        key=lambda x: (
            str(x.get("accession_number", "")),
            str(x.get("gap_type", "")),
            str(x.get("description", "")),
        ) if isinstance(x, dict) else ("", "", str(x)),
    ):
        sorted_gaps.append({k: g[k] for k in sorted(g.keys())} if isinstance(g, dict) else {"description": str(g)})

    payload: Dict[str, Any] = {
        "schema_version": schema_version,
        "ticker": str(ticker).upper().strip(),
        "issuer_cik": str(issuer_cik).strip().zfill(10),
        "interval_start": _to_iso(interval_start),
        "interval_end": _to_iso(interval_end),
        "certified_watermark": _to_iso(certified_watermark),
        "receipt_version": receipt_version,
        "supersedes_receipt_id": supersedes_receipt_id,
        "accession_inventory": sorted(list(set(accession_inventory or []))),
        "source_hashes": {k: source_hashes[k] for k in sorted((source_hashes or {}).keys())},
        "material_gaps": sorted_gaps,
    }
    if zero_activity_proof:
        payload["zero_activity_proof"] = {k: zero_activity_proof[k] for k in sorted(zero_activity_proof.keys())}

    raw_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw_bytes).hexdigest()


class StaleLeaseOwnershipLostError(RuntimeError):
    """Raised when a worker attempts to heartbeat, complete, or publish with an expired or lost lease."""
    pass


class MaxAttemptsExceededError(RuntimeError):
    """Raised when a run request has exceeded maximum retry attempts."""
    pass


class PostgresSmartMoneyRepository:
    """Production-grade PostgreSQL repository implementing fail-closed Smart Money persistence."""

    def __init__(self, dsn: Optional[str] = None, containment_gate: Optional[Any] = None):
        self.dsn = dsn
        self.containment_gate = containment_gate

    @contextmanager
    def ingestion_transaction(self):
        """Commit complete filing evidence together; never retain partial ingestion."""
        with get_connection(dsn=self.dsn) as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def resolve_amendment_base(self, filing, original_date, owner_ciks, conn):
        """Resolve only a unique original Form 4 with the same issuer/owner set.

        Accession prefixes are submitting agents, not reporting-owner identities.
        An ambiguous or absent match remains unresolved and cannot imply conviction.
        """
        from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
        if not original_date or not owner_ciks:
            return None
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT accession_number, xml_content FROM form4_raw_filings
                   WHERE cik = %s AND ticker = %s AND filing_date = %s
                     AND accession_number <> %s AND accepted_datetime <= %s;""",
                (filing["cik"], filing["ticker"], original_date,
                 filing["accession_number"], filing["accepted_datetime"]),
            )
            candidates = cur.fetchall()
        matches = []
        for candidate in candidates:
            parsed = Form4XmlParser.parse_filing(
                candidate["xml_content"], accession_number=candidate["accession_number"]
            )
            if (parsed and parsed.disposition != "PARSE_FAILED"
                    and parsed.issuer_cik == str(filing["cik"]).zfill(10)
                    and parsed.document_type == "4"
                    and set(parsed.reporting_owner_ciks) == set(owner_ciks)):
                matches.append(candidate["accession_number"])
        return matches[0] if len(matches) == 1 else None

    # ── Raw Filings & Idempotency ─────────────────────────────────────────────

    def get_existing_accessions_for_issuer(self, ticker: str) -> Set[str]:
        """Read existing Form 4 accession IDs for one governed issuer.

        This lookup is read-only and must fail closed on database failure.
        Do not infer that an inaccessible or empty repository has zero filings.
        """
        if not self.dsn:
            raise RuntimeError("SEC_FILING_WATERMARK_DB_UNAVAILABLE")
        with transaction(dsn=self.dsn) as cur:
            cur.execute(
                """SELECT f.accession_number FROM form4_raw_filings f
                   JOIN sm_source_availability_ledger av USING (accession_number)
                   WHERE f.ticker = %s AND NOT EXISTS (
                     SELECT 1 FROM sm_amendments a
                     WHERE a.amendment_accession = f.accession_number
                       AND a.resolution_status <> 'RESOLVED'
                       AND a.resolution_status NOT IN ('FULL_REPLACEMENT', 'PARTIAL_CORRECTION', 'ADDITIONAL_TRANSACTION')
                   );""",
                (ticker,),
            )
            return {row["accession_number"] for row in cur.fetchall()}

    def save_raw_filing(
        self,
        filing: Dict[str, Any],
        conn: Optional[PgConnection] = None,
        containment_gate: Optional[Any] = None,
    ) -> bool:
        """
        Persists a raw filing idempotently.
        If filing already exists with identical content hash, returns False (no-op).
        If filing already exists with conflicting content hash, raises SOURCE_INTEGRITY_CONFLICT.
        Fails closed if backup age containment is breached.
        """
        gate = containment_gate or self.containment_gate
        if gate is not None:
            gate.enforce_ingestion_gate()
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

    def get_raw_filing(
        self,
        accession_number: str,
        conn: Optional[PgConnection] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves raw filing metadata and XML content by accession number."""
        query = "SELECT * FROM form4_raw_filings WHERE accession_number = %s;"
        def _exec(cur):
            cur.execute(query, (accession_number,))
            row = cur.fetchone()
            return dict(row) if row else None

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with get_connection(dsn=self.dsn) as c:
                with c.cursor(cursor_factory=RealDictCursor) as cur:
                    return _exec(cur)

    # ── Parsed Transactions ───────────────────────────────────────────────────

    def save_parsed_transactions(
        self,
        transactions: List[Dict[str, Any]],
        conn: Optional[PgConnection] = None,
        containment_gate: Optional[Any] = None,
    ) -> int:
        """Saves parsed transactions for an accession. Fails closed if containment is breached."""
        if not transactions:
            return 0

        gate = containment_gate or self.containment_gate
        if gate is not None:
            gate.enforce_ingestion_gate()

        def _exec(cur):
            count = 0
            for tx in transactions:
                cur.execute("""
                    INSERT INTO form4_parsed_transactions (
                        transaction_id, accession_number, transaction_date, transaction_code,
                        shares, price_per_share, total_value_usd, reporting_owner_name,
                        reporting_owner_role, is_director, is_officer, is_ten_percent,
                        officer_title, ownership_type,
                        is_derivative, signal_type, acquired_disposed,
                        is_positive_buying_conviction, is_open_market_purchase_evidence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_id) DO UPDATE SET
                        shares = EXCLUDED.shares,
                        price_per_share = EXCLUDED.price_per_share,
                        total_value_usd = EXCLUDED.total_value_usd,
                        is_derivative = EXCLUDED.is_derivative,
                        signal_type = EXCLUDED.signal_type,
                        acquired_disposed = EXCLUDED.acquired_disposed,
                        is_positive_buying_conviction = EXCLUDED.is_positive_buying_conviction,
                        is_open_market_purchase_evidence = EXCLUDED.is_open_market_purchase_evidence;
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
                    tx.get("is_derivative", False),
                    tx.get("signal_type"),
                    tx.get("acquired_disposed"),
                    tx.get("is_positive_buying_conviction", False),
                    tx.get("is_open_market_purchase_evidence", False),
                ))
                count += 1
            return count

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                return _exec(cur)

    def get_parsed_transactions_for_accession(
        self,
        accession_number: str,
        conn: Optional[PgConnection] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves parsed transactions for a specific accession number."""
        query = "SELECT * FROM form4_parsed_transactions WHERE accession_number = %s ORDER BY transaction_date ASC;"
        def _exec(cur):
            cur.execute(query, (accession_number,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with get_connection(dsn=self.dsn) as c:
                with c.cursor(cursor_factory=RealDictCursor) as cur:
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
        """
        Records or updates a Form 4/A amendment linkage and governance classification.
        Supports controlled state transitions:
        - UNRESOLVED -> RESOLVED, FULL_REPLACEMENT, PARTIAL_CORRECTION, ADDITIONAL_TRANSACTION, AMENDMENT_SCOPE_AMBIGUOUS
        - RESOLVED -> FULL_REPLACEMENT, PARTIAL_CORRECTION, ADDITIONAL_TRANSACTION, AMENDMENT_SCOPE_AMBIGUOUS
        - AMENDMENT_SCOPE_AMBIGUOUS -> FULL_REPLACEMENT, PARTIAL_CORRECTION, ADDITIONAL_TRANSACTION
        - Idempotent re-execution of the same status
        Fails closed with AMENDMENT_TRANSITION_CONFLICT on unauthorized transitions or base accession mutations.
        """
        query = """
            INSERT INTO sm_amendments (
                amendment_accession, base_accession, amendment_number, resolution_status, resolved_at
            ) VALUES (%s, %s, %s, %s, COALESCE(%s::timestamptz, NOW()))
            ON CONFLICT (amendment_accession) DO UPDATE SET
                base_accession = COALESCE(EXCLUDED.base_accession, sm_amendments.base_accession),
                amendment_number = EXCLUDED.amendment_number,
                resolution_status = EXCLUDED.resolution_status,
                resolved_at = COALESCE(EXCLUDED.resolved_at, sm_amendments.resolved_at, NOW())
            WHERE (
                -- 1. Idempotent re-execution
                (sm_amendments.resolution_status = EXCLUDED.resolution_status AND (sm_amendments.base_accession IS NOT DISTINCT FROM EXCLUDED.base_accession))
                -- 2. Transitions from unclassified, lineage-only, or ambiguous to classified scope
                OR (
                    sm_amendments.resolution_status IN ('UNRESOLVED', 'AMENDMENT_TARGET_AMBIGUOUS', 'RESOLVED', 'AMENDMENT_SCOPE_AMBIGUOUS')
                    AND EXCLUDED.resolution_status IN ('RESOLVED', 'FULL_REPLACEMENT', 'PARTIAL_CORRECTION', 'ADDITIONAL_TRANSACTION', 'AMENDMENT_SCOPE_AMBIGUOUS')
                    AND (sm_amendments.base_accession IS NULL OR sm_amendments.base_accession = EXCLUDED.base_accession OR EXCLUDED.base_accession IS NULL)
                )
            )
            RETURNING sm_amendments.resolution_status, sm_amendments.base_accession;
        """
        params = (amendment_accession, base_accession, amendment_number, resolution_status, resolved_at)

        def _exec(cur):
            cur.execute(query, params)
            res = cur.fetchone()
            if res is None:
                cur.execute(
                    "SELECT resolution_status, base_accession FROM sm_amendments WHERE amendment_accession = %s;",
                    (amendment_accession,),
                )
                existing = cur.fetchone()
                ex_st = existing["resolution_status"] if isinstance(existing, dict) else (existing[0] if existing else "UNKNOWN")
                ex_b = existing["base_accession"] if isinstance(existing, dict) else (existing[1] if existing else "UNKNOWN")
                raise RuntimeError(
                    f"AMENDMENT_TRANSITION_CONFLICT: Cannot transition amendment {amendment_accession} "
                    f"from status='{ex_st}', base='{ex_b}' to status='{resolution_status}', base='{base_accession}'."
                )

        if conn is not None:
            with conn.cursor() as cur:
                _exec(cur)
        else:
            with transaction(dsn=self.dsn) as cur:
                _exec(cur)

    # ── Reconciliation Receipts (Append-Only) ─────────────────────────────────

    def save_reconciliation_receipt(self, receipt: Dict[str, Any], conn: Optional[PgConnection] = None) -> str:
        """
        Persists an immutable reconciliation receipt.
        Assigns UUID if not provided. Supports supersedes_receipt_id for corrections.
        Preserves durable issuer and coverage metadata within failure_reasons JSONB.
        """
        receipt_id = receipt.get("receipt_id") or str(uuid.uuid4())
        reasons = receipt.get("failure_reasons", [])
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except Exception:
                reasons = [reasons]
        elif not isinstance(reasons, (list, dict)):
            reasons = [str(reasons)]

        if isinstance(reasons, list) and any(k in receipt for k in ("issuer_cik", "ticker", "covered_ciks", "covered_tickers", "accession_inventory", "source_hashes", "material_gaps")):
            reasons = {
                "reasons": reasons,
                "ticker": receipt.get("ticker"),
                "issuer_cik": receipt.get("issuer_cik"),
                "covered_tickers": receipt.get("covered_tickers") or ([receipt["ticker"]] if "ticker" in receipt else []),
                "covered_ciks": receipt.get("covered_ciks") or ([receipt["issuer_cik"]] if "issuer_cik" in receipt else []),
                "accession_inventory": receipt.get("accession_inventory", []),
                "source_hashes": receipt.get("source_hashes", {}),
                "material_gaps": receipt.get("material_gaps", []),
                "public_availability_cutoff": receipt.get("certified_watermark"),
            }

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

    def record_issuer_coverage_receipt(
        self,
        ticker: str,
        issuer_cik: str,
        interval_start: str,
        interval_end: str,
        certified_watermark: str,
        accession_inventory: Optional[List[str]] = None,
        source_hashes: Optional[Dict[str, str]] = None,
        material_gaps: Optional[List[Dict[str, Any]]] = None,
        coverage_status: str = "COMPLETE",
        receipt_version: int = 1,
        supersedes_receipt_id: Optional[str] = None,
        release_sha: Optional[str] = None,
        receipt_id: Optional[str] = None,
        conn: Optional[PgConnection] = None,
        zero_activity_proof: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Persists a durable issuer-bound historical coverage receipt certifying an observation interval.
        Binds exact issuer ticker, CIK, observation interval, certified watermark, accession inventory,
        cryptographic source hashes, material gaps, and zero-activity proofs within sm_reconciliation_receipts (0-DDL compliant).
        """
        r_id = receipt_id or str(uuid.uuid4())
        acc_inv = sorted(list(set(accession_inventory or [])))
        s_hashes = dict(source_hashes or {})
        m_gaps = list(material_gaps or [])
        clean_ticker = ticker.upper().strip()
        clean_cik = str(issuer_cik).strip().zfill(10)

        # Verified zero-activity check (Finding 1):
        if coverage_status == "COMPLETE" and len(acc_inv) == 0:
            has_catalog_proof = (
                (isinstance(zero_activity_proof, dict) and bool(zero_activity_proof.get("source_catalog_hash")))
                or any("catalog" in str(k).lower() or "submissions" in str(k).lower() for k in s_hashes.keys())
            )
            if not has_catalog_proof:
                raise RuntimeError(
                    "ZERO_ACTIVITY_UNPROVED: Bare empty inventory without authoritative catalog proof "
                    "cannot certify COMPLETE coverage."
                )

        schema_ver = 1
        recon_hash = compute_canonical_receipt_hash(
            schema_version=schema_ver,
            ticker=clean_ticker,
            issuer_cik=clean_cik,
            interval_start=interval_start,
            interval_end=interval_end,
            certified_watermark=certified_watermark,
            receipt_version=receipt_version,
            supersedes_receipt_id=supersedes_receipt_id,
            accession_inventory=acc_inv,
            source_hashes=s_hashes,
            material_gaps=m_gaps,
            zero_activity_proof=zero_activity_proof,
        )

        reasons = [g.get("description", str(g)) if isinstance(g, dict) else str(g) for g in m_gaps]

        structured_metadata = {
            "reasons": reasons,
            "schema_version": schema_ver,
            "ticker": clean_ticker,
            "issuer_cik": clean_cik,
            "covered_tickers": [clean_ticker],
            "covered_ciks": [clean_cik],
            "accession_inventory": acc_inv,
            "source_hashes": s_hashes,
            "material_gaps": m_gaps,
            "public_availability_cutoff": certified_watermark,
        }
        if zero_activity_proof:
            structured_metadata["zero_activity_proof"] = zero_activity_proof

        receipt_payload = {
            "receipt_id": r_id,
            "interval_start": interval_start,
            "interval_end": interval_end,
            "certified_watermark": certified_watermark,
            "receipt_version": receipt_version,
            "supersedes_receipt_id": supersedes_receipt_id,
            "expected_count": len(acc_inv) + len(m_gaps),
            "accounted_count": len(acc_inv),
            "missing_count": len(m_gaps),
            "material_gap_count": len(m_gaps),
            "coverage_status": coverage_status,
            "failure_reasons": structured_metadata,
            "reconciliation_hash": recon_hash,
            "release_sha": release_sha or os.environ.get("ARX_RELEASE_SHA", "b72ea23937cf6ed7b8c4b60096630767a7c4592b"),
        }

        return self.save_reconciliation_receipt(receipt_payload, conn=conn)

    def get_authoritative_receipt_for_interval(
        self,
        interval_start: str,
        interval_end: str,
        certified_watermark: str,
        conn: Optional[PgConnection] = None,
        issuer_cik: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Queries the authoritative reconciliation receipt certifying watermark W.
        Requires full interval coverage, certified watermark >= W, COMPLETE coverage status,
        and zero missing count or material gaps.
        Validates exact issuer binding, canonical hash integrity, supersession chains,
        inventory existence and completeness, and zero-activity proof.
        """
        clean_cik = str(issuer_cik).strip().zfill(10) if issuer_cik else None
        clean_ticker = ticker.upper().strip() if ticker else None

        query = """
            SELECT * FROM sm_reconciliation_receipts
            WHERE interval_start <= %s
              AND (interval_end >= %s OR interval_end >= %s)
              AND (certified_watermark >= %s OR certified_watermark >= %s)
              AND coverage_status = 'COMPLETE'
              AND missing_count = 0
              AND material_gap_count = 0
            ORDER BY receipt_version DESC, created_at DESC;
        """
        eod_end = f"{interval_end[:10]}T23:59:59Z" if len(interval_end) >= 10 else interval_end
        eod_wm = f"{certified_watermark[:10]}T23:59:59Z" if len(certified_watermark) >= 10 else certified_watermark
        params = (interval_start, interval_end, eod_end, certified_watermark, eod_wm)

        def _exec(cur):
            cur.execute(query, params)
            rows = cur.fetchall()
            for r in rows:
                receipt = dict(r)
                if receipt.get("receipt_id"):
                    receipt["receipt_id"] = str(receipt["receipt_id"])
                if receipt.get("supersedes_receipt_id"):
                    receipt["supersedes_receipt_id"] = str(receipt["supersedes_receipt_id"])
                else:
                    receipt["supersedes_receipt_id"] = None
                reasons_val = receipt.get("failure_reasons")
                if isinstance(reasons_val, dict):
                    for k in ("ticker", "issuer_cik", "covered_tickers", "covered_ciks",
                              "accession_inventory", "source_hashes", "material_gaps",
                              "public_availability_cutoff", "zero_activity_proof", "schema_version"):
                        if k in reasons_val and k not in receipt:
                            receipt[k] = reasons_val[k]
                    receipt["failure_reasons"] = reasons_val.get("reasons", [])
                elif isinstance(reasons_val, list):
                    for item in reasons_val:
                        if isinstance(item, dict):
                            for k in ("ticker", "issuer_cik", "covered_tickers", "covered_ciks",
                                      "accession_inventory", "source_hashes", "material_gaps",
                                      "public_availability_cutoff", "zero_activity_proof", "schema_version"):
                                if k in item and k not in receipt:
                                    receipt[k] = item[k]

                # 1. Exact Issuer Binding Validation
                if clean_cik or clean_ticker:
                    r_cik = str(receipt.get("issuer_cik") or "").strip().zfill(10) if receipt.get("issuer_cik") else None
                    c_ciks = [str(c).strip().zfill(10) for c in (receipt.get("covered_ciks") or [])]
                    r_sym = str(receipt.get("ticker") or receipt.get("symbol") or "").strip().upper() if (receipt.get("ticker") or receipt.get("symbol")) else None
                    c_syms = [str(s).strip().upper() for s in (receipt.get("covered_tickers") or receipt.get("covered_symbols") or [])]

                    matched = False
                    if clean_cik and (r_cik == clean_cik or clean_cik in c_ciks):
                        matched = True
                    elif clean_ticker and (r_sym == clean_ticker or clean_ticker in c_syms):
                        matched = True

                    if not matched:
                        continue

                # 2. Supersession Chain Validation
                sup_id = receipt.get("supersedes_receipt_id")
                if sup_id:
                    cur.execute(
                        "SELECT receipt_version, failure_reasons FROM sm_reconciliation_receipts WHERE receipt_id = %s;",
                        (sup_id,),
                    )
                    sup_row = cur.fetchone()
                    if not sup_row:
                        continue
                    if sup_row["receipt_version"] >= receipt["receipt_version"]:
                        continue

                # 3. Canonical Reconciliation Hash Verification
                acc_inv = receipt.get("accession_inventory", [])
                s_hashes = receipt.get("source_hashes", {})
                m_gaps = receipt.get("material_gaps", [])
                z_proof = receipt.get("zero_activity_proof")
                r_sym = receipt.get("ticker") or clean_ticker or ""
                r_cik = receipt.get("issuer_cik") or clean_cik or ""

                expected_hash = compute_canonical_receipt_hash(
                    schema_version=receipt.get("schema_version", 1),
                    ticker=r_sym,
                    issuer_cik=r_cik,
                    interval_start=receipt["interval_start"],
                    interval_end=receipt["interval_end"],
                    certified_watermark=receipt["certified_watermark"],
                    receipt_version=receipt.get("receipt_version", 1),
                    supersedes_receipt_id=receipt.get("supersedes_receipt_id"),
                    accession_inventory=acc_inv,
                    source_hashes=s_hashes,
                    material_gaps=m_gaps,
                    zero_activity_proof=z_proof,
                )
                legacy_payload = (
                    f"{r_sym.upper().strip()}:{str(r_cik).strip().zfill(10)}:{_to_iso(receipt['interval_start'])}:{_to_iso(receipt['interval_end'])}:{_to_iso(receipt['certified_watermark'])}:"
                    f"{json.dumps(sorted(list(set(acc_inv))))}:{json.dumps(s_hashes, sort_keys=True)}:{len(m_gaps)}"
                )
                legacy_hash = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()

                if receipt["reconciliation_hash"] != expected_hash and receipt["reconciliation_hash"] != legacy_hash:
                    continue

                # 4. Inventory & Zero-Activity Validation
                if len(acc_inv) == 0:
                    has_proof = (
                        (isinstance(z_proof, dict) and bool(z_proof.get("source_catalog_hash")))
                        or any("catalog" in str(k).lower() or "submissions" in str(k).lower() for k in s_hashes.keys())
                    )
                    if not has_proof:
                        continue
                else:
                    cur.execute("""
                        SELECT f.accession_number, f.cik, f.ticker, f.content_hash, av.public_availability_at
                        FROM form4_raw_filings f
                        JOIN sm_source_availability_ledger av ON f.accession_number = av.accession_number
                        WHERE f.accession_number = ANY(%s);
                    """, (acc_inv,))
                    found = {row["accession_number"]: row for row in cur.fetchall()}
                    if len(found) != len(acc_inv):
                        continue

                    inv_valid = True
                    for acc in acc_inv:
                        f_row = found[acc]
                        if r_cik and f_row["cik"] != r_cik and (f_row["ticker"] or "").upper() != r_sym.upper():
                            inv_valid = False
                            break
                        if acc in s_hashes and s_hashes[acc] != f_row["content_hash"]:
                            inv_valid = False
                            break
                    if not inv_valid:
                        continue

                    if r_cik:
                        cur.execute("""
                            SELECT count(*) AS cnt FROM form4_raw_filings
                            WHERE cik = %s
                              AND filing_date >= %s::date
                              AND filing_date <= %s::date
                              AND accession_number != ALL(%s);
                        """, (r_cik, _to_iso(receipt["interval_start"])[:10], _to_iso(receipt["interval_end"])[:10], acc_inv))
                        cnt_row = cur.fetchone()
                        if cnt_row and cnt_row["cnt"] > 0:
                            continue

                return receipt
            return None

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
        replay_mode: str = "PUBLIC_AVAILABLE",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Point-in-Time Bulk Transaction Retrieval (SM-DATA-GOV-001 Section 3).
        Supports two distinct replay modes:
        - PUBLIC_AVAILABLE: Enforces av.public_availability_at <= as_of_watermark.
        - ARX_OBSERVED: Enforces COALESCE(av.recorded_at, f.created_at) <= as_of_watermark.
        Resolves amendment supersession point-in-time:
        If an amendment supersedes an original transaction and was available before W,
        the superseding version is used.
        """
        if replay_mode not in ("PUBLIC_AVAILABLE", "ARX_OBSERVED"):
            raise ValueError(f"Invalid replay_mode '{replay_mode}'. Allowed modes: 'PUBLIC_AVAILABLE', 'ARX_OBSERVED'.")
        if as_of_watermark is not None:
            try:
                datetime.fromisoformat(str(as_of_watermark).replace("Z", "+00:00"))
            except Exception as e:
                raise ValueError(f"Invalid as_of_watermark '{as_of_watermark}': must be valid ISO 8601 string.") from e

        if not symbols:
            return {}

        time_predicate = (
            "COALESCE(av.recorded_at, f.created_at) <= %s"
            if replay_mode == "ARX_OBSERVED"
            else "av.public_availability_at IS NOT NULL AND av.public_availability_at <= %s"
        )
        am_time_predicate = (
            "COALESCE(am_av.recorded_at, am_f.created_at) <= %s"
            if replay_mode == "ARX_OBSERVED"
            else "am_av.public_availability_at IS NOT NULL AND am_av.public_availability_at <= %s"
        )
        part_time_predicate = (
            "COALESCE(part_av.recorded_at, part_f.created_at) <= %s"
            if replay_mode == "ARX_OBSERVED"
            else "part_av.public_availability_at IS NOT NULL AND part_av.public_availability_at <= %s"
        )
        re_time_predicate = (
            "COALESCE(re_av.recorded_at, re_f.created_at) <= %s"
            if replay_mode == "ARX_OBSERVED"
            else "re_av.public_availability_at IS NOT NULL AND re_av.public_availability_at <= %s"
        )

        query = f"""
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
                tx.is_derivative,
                tx.signal_type,
                tx.acquired_disposed,
                tx.is_positive_buying_conviction,
                tx.is_open_market_purchase_evidence,
                f.ticker AS symbol,
                f.cik AS issuer_cik,
                f.company_name,
                av.public_availability_at,
                am.amendment_number,
                am.base_accession,
                am.resolution_status
            FROM form4_parsed_transactions tx
            JOIN form4_raw_filings f ON tx.accession_number = f.accession_number
            JOIN sm_source_availability_ledger av ON tx.accession_number = av.accession_number
            LEFT JOIN sm_amendments am ON tx.accession_number = am.amendment_accession
            WHERE {time_predicate}
              AND tx.transaction_date >= %s
              AND tx.transaction_date <= (%s::timestamptz AT TIME ZONE 'UTC')::date
              AND f.ticker = ANY(%s)
              -- 1. Exclude unresolved or ambiguous amendments
              AND NOT EXISTS (
                  SELECT 1 FROM sm_amendments ambig
                  WHERE ambig.amendment_accession = tx.accession_number
                    AND ambig.resolution_status IN ('AMENDMENT_TARGET_AMBIGUOUS', 'UNRESOLVED', 'UNRESOLVED_AMENDMENT')
              )
              -- 2. Exclude transactions superseded by full filing amendments available <= W
              AND NOT EXISTS (
                  SELECT 1 FROM sm_amendments am_full
                  JOIN sm_source_availability_ledger am_av ON am_full.amendment_accession = am_av.accession_number
                  JOIN form4_raw_filings am_f ON am_full.amendment_accession = am_f.accession_number
                  WHERE am_full.base_accession = tx.accession_number
                    AND am_full.resolution_status IN ('SUPERSEDED', 'SUPERSEDED_OR_CORRECTED', 'FULL_REPLACEMENT')
                    AND {am_time_predicate}
              )
              -- 3. Transaction-level reconciliation for partial corrections available <= W
              AND NOT EXISTS (
                  SELECT 1 FROM sm_amendments part_am
                  JOIN form4_parsed_transactions amend_tx ON part_am.amendment_accession = amend_tx.accession_number
                  JOIN sm_source_availability_ledger part_av ON part_am.amendment_accession = part_av.accession_number
                  JOIN form4_raw_filings part_f ON part_am.amendment_accession = part_f.accession_number
                  WHERE part_am.base_accession = tx.accession_number
                    AND part_am.resolution_status IN ('PARTIAL_CORRECTION', 'PARTIAL_AMENDMENT')
                    -- Evidence-backed line mapping: never match solely on mutable date/code/owner
                    AND (
                        (position(':' in amend_tx.transaction_id) > 0 AND position(':' in tx.transaction_id) > 0
                         AND split_part(amend_tx.transaction_id, ':', 2) = split_part(tx.transaction_id, ':', 2)
                         AND split_part(amend_tx.transaction_id, ':', 3) = split_part(tx.transaction_id, ':', 3)
                         AND (
                             split_part(amend_tx.transaction_id, ':', 4) = split_part(tx.transaction_id, ':', 4)
                             OR (
                                 (SELECT count(*) FROM form4_parsed_transactions sub_b
                                  WHERE sub_b.accession_number = tx.accession_number
                                    AND split_part(sub_b.transaction_id, ':', 2) = split_part(tx.transaction_id, ':', 2)
                                    AND split_part(sub_b.transaction_id, ':', 3) = split_part(tx.transaction_id, ':', 3)
                                 ) = 1
                                 AND
                                 (SELECT count(*) FROM form4_parsed_transactions sub_a
                                  WHERE sub_a.accession_number = part_am.amendment_accession
                                    AND split_part(sub_a.transaction_id, ':', 2) = split_part(amend_tx.transaction_id, ':', 2)
                                    AND split_part(sub_a.transaction_id, ':', 3) = split_part(amend_tx.transaction_id, ':', 3)
                                 ) = 1
                             )
                         )
                        )
                        OR (
                            amend_tx.reporting_owner_name = tx.reporting_owner_name
                            AND (
                                (SELECT count(*) FROM form4_parsed_transactions sub_b WHERE sub_b.accession_number = tx.accession_number) = 1
                                AND (SELECT count(*) FROM form4_parsed_transactions sub_a WHERE sub_a.accession_number = part_am.amendment_accession) = 1
                            )
                        )
                    )
                    AND {part_time_predicate}
              )
              -- 4. Exclude amendments that were themselves superseded by subsequent amendments available <= W
              AND NOT EXISTS (
                  SELECT 1 FROM sm_amendments re_am
                  JOIN sm_source_availability_ledger re_av ON re_am.amendment_accession = re_av.accession_number
                  JOIN form4_raw_filings re_f ON re_am.amendment_accession = re_f.accession_number
                  WHERE (
                      re_am.base_accession = tx.accession_number
                      OR EXISTS (
                          SELECT 1 FROM sm_amendments curr_am
                          WHERE curr_am.amendment_accession = tx.accession_number
                            AND re_am.base_accession = curr_am.base_accession
                            AND re_am.amendment_number > curr_am.amendment_number
                      )
                  )
                    AND re_am.resolution_status IN ('SUPERSEDED', 'SUPERSEDED_OR_CORRECTED', 'FULL_REPLACEMENT')
                    AND {re_time_predicate}
              )
            ORDER BY tx.transaction_date ASC, av.public_availability_at ASC;
        """

        def _exec(cur):
            params = (
                as_of_watermark,
                min_date,
                as_of_watermark,
                [s.upper().strip() for s in symbols],
                as_of_watermark,
                as_of_watermark,
                as_of_watermark,
            )
            cur.execute(query, params)
            rows = cur.fetchall()
            out: Dict[str, List[Dict[str, Any]]] = {s.upper().strip(): [] for s in symbols}
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
        replay_mode: str = "PUBLIC_AVAILABLE",
    ) -> Set[str]:
        """
        Identifies symbols having ambiguous or unresolvable amendments available <= W.
        """
        if replay_mode not in ("PUBLIC_AVAILABLE", "ARX_OBSERVED"):
            raise ValueError(f"Invalid replay_mode '{replay_mode}'. Allowed modes: 'PUBLIC_AVAILABLE', 'ARX_OBSERVED'.")
        if as_of_watermark is not None:
            try:
                datetime.fromisoformat(str(as_of_watermark).replace("Z", "+00:00"))
            except Exception as e:
                raise ValueError(f"Invalid as_of_watermark '{as_of_watermark}': must be valid ISO 8601 string.") from e

        if not symbols:
            return set()

        time_predicate = (
            "COALESCE(av.recorded_at, f.created_at) <= %s"
            if replay_mode == "ARX_OBSERVED"
            else "av.public_availability_at IS NOT NULL AND av.public_availability_at <= %s"
        )

        query = f"""
            SELECT DISTINCT f.ticker
            FROM sm_amendments am
            JOIN form4_raw_filings f ON am.amendment_accession = f.accession_number
            JOIN sm_source_availability_ledger av ON am.amendment_accession = av.accession_number
            WHERE {time_predicate}
              AND am.resolution_status IN ('AMENDMENT_TARGET_AMBIGUOUS', 'UNRESOLVED', 'UNRESOLVED_AMENDMENT', 'RESOLVED', 'AMENDMENT_SCOPE_AMBIGUOUS')
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

    def get_snapshot_evidence_as_of(self, as_of_watermark, issuers, min_date):
        """Read a coherent historical input set; never certify zero from a batch receipt."""
        from analyst_dashboard.smart_money.radar_adapter import RadarSmartMoneyAdapter
        symbols = [issuer["ticker"] for issuer in issuers]
        with get_connection(dsn=self.dsn) as conn:
            try:
                conn.rollback()
                with conn.cursor() as cur:
                    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                ambiguous = self.get_ambiguous_amendment_symbols_as_of(
                    as_of_watermark, symbols, conn=conn,
                )
                if ambiguous:
                    raise RuntimeError("SNAPSHOT_AMENDMENT_EFFECT_UNVERIFIED")
                receipt_ids = []
                for issuer in issuers:
                    receipt = self.get_authoritative_receipt_for_interval(
                        interval_start=min_date, interval_end=as_of_watermark,
                        certified_watermark=as_of_watermark, issuer_cik=issuer["cik"], conn=conn,
                    )
                    valid, _ = RadarSmartMoneyAdapter.validate_issuer_coverage_receipt(
                        receipt, issuer["ticker"], issuer["cik"], min_date, as_of_watermark,
                        window_days=30,
                    )
                    if not valid:
                        raise RuntimeError("SNAPSHOT_HISTORICAL_COVERAGE_UNVERIFIED")
                    receipt_ids.append(str(receipt["receipt_id"]))
                transactions = self.get_bulk_transactions_as_of(
                    as_of_watermark, symbols, min_date, conn=conn,
                    replay_mode="PUBLIC_AVAILABLE",
                )
                return {"transactions": transactions, "coverage_receipt_ids": receipt_ids}
            finally:
                # End the read-only transaction on success and failure before pooling.
                conn.rollback()

    # ── Snapshot Publication (Atomic Publication Transaction) ────────────────

    def publish_scanner_snapshot_atomic(
        self,
        snapshot: Dict[str, Any],
        run_request_id: Optional[str] = None,
        lease_token: Optional[str] = None,
        conn: Optional[PgConnection] = None,
        containment_gate: Optional[Any] = None,
    ) -> str:
        """
        Atomic Snapshot Publication Transaction (Section 16).
        In a single transaction:
        0. Enforces backup containment gate (fails closed if backup freshness breached).
        1. If run_request_id and lease_token provided, verifies active lease ownership (fails closed if stale).
        2. Inserts immutable record into scanner_snapshots.
        3. Updates/inserts active pointer in scanner_active_snapshots if decision == 'PUBLISH'.
        4. Marks run_request as COMPLETED within the same atomic transaction if published.
        5. Emits audit event.
        Rolls back entirely if any stage fails.
        """
        gate = containment_gate or self.containment_gate
        if gate is not None:
            gate.enforce_snapshot_publication_gate()

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

    def save_snapshot(
        self,
        snapshot: Dict[str, Any],
        conn: Optional[PgConnection] = None,
        containment_gate: Optional[Any] = None,
    ) -> str:
        """Compatibility alias for publish_scanner_snapshot_atomic adhering to snapshot store contract."""
        return self.publish_scanner_snapshot_atomic(snapshot, conn=conn, containment_gate=containment_gate)

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

    def get_governed_issuer(self, ticker: str, conn: Optional[PgConnection] = None) -> Optional[Dict[str, Any]]:
        """Retrieves authoritative governed issuer record for a ticker."""
        query = """
            SELECT cik, ticker, company_name, sic, fiscal_year_end,
                   scannable, universe_build_id, manifest_hash, updated_at
            FROM governed_issuers
            WHERE ticker = %s LIMIT 1;
        """
        def _exec(cur):
            cur.execute(query, (ticker.upper().strip(),))
            row = cur.fetchone()
            return dict(row) if row else None

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with get_connection(dsn=self.dsn) as c:
                with c.cursor(cursor_factory=RealDictCursor) as cur:
                    return _exec(cur)

    # ── Run Requests (Web -> Worker Queue) ────────────────────────────────────

    def find_active_run_request(
        self,
        request_type: str,
        subject_hash: str,
        conn: Optional[PgConnection] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Checks for an existing in-flight or pending run request with matching type and subject hash.
        Prevents duplicate enqueuing.
        """
        query = """
            SELECT * FROM sm_run_requests
            WHERE request_type = %s
              AND requested_by_subject_hash = %s
              AND status IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE')
            ORDER BY requested_at DESC
            LIMIT 1;
        """

        def _exec(cur):
            cur.execute(query, (request_type, subject_hash))
            row = cur.fetchone()
            return dict(row) if row else None

        if conn is not None:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                return _exec(cur)
        else:
            with get_connection(dsn=self.dsn) as connection:
                with connection.cursor(cursor_factory=RealDictCursor) as cur:
                    return _exec(cur)

    def create_run_request(
        self,
        request_type: str,
        subject_hash: str,
        max_attempts: int = 3,
        conn: Optional[PgConnection] = None,
        idempotency_guard: bool = False,
    ) -> str:
        """Submits an immutable run request into the queue.
        When idempotency_guard=True, returns existing request_id if an active
        (PENDING, RUNNING, or RETRY_ELIGIBLE) request exists with identical
        request_type and subject_hash.
        """
        if idempotency_guard:
            active = self.find_active_run_request(request_type, subject_hash, conn=conn)
            if active:
                logger.info(
                    f"Idempotent request reuse: active request {active['request_id']} "
                    f"found for {request_type}/{subject_hash}."
                )
                return str(active["request_id"])

        request_id = str(uuid.uuid4())
        if idempotency_guard:
            # Atomic conditional insert: prevents concurrent duplicate enqueue races
            query = """
                INSERT INTO sm_run_requests (
                    request_id, request_type, requested_by_subject_hash, status,
                    max_attempts, next_attempt_at
                )
                SELECT %s, %s, %s, 'PENDING', %s, NOW()
                WHERE NOT EXISTS (
                    SELECT 1 FROM sm_run_requests
                    WHERE request_type = %s
                      AND requested_by_subject_hash = %s
                      AND status IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE')
                )
                RETURNING request_id;
            """

            def _exec(cur):
                cur.execute(
                    query,
                    (request_id, request_type, subject_hash, max_attempts, request_type, subject_hash),
                )
                row = cur.fetchone()
                if row:
                    return str(row["request_id"])
                # Concurrent enqueue race won by another caller: fetch winning active request
                cur.execute(
                    """
                    SELECT request_id FROM sm_run_requests
                    WHERE request_type = %s
                      AND requested_by_subject_hash = %s
                      AND status IN ('PENDING', 'RUNNING', 'RETRY_ELIGIBLE')
                    ORDER BY requested_at DESC
                    LIMIT 1;
                    """,
                    (request_type, subject_hash),
                )
                existing = cur.fetchone()
                if existing:
                    return str(existing["request_id"])
                return request_id
        else:
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

    # ── Destruction Protection & Deletion Governance (SM-DATA-GOV-001) ────────

    def delete_canonical_evidence(
        self,
        accession_number: str,
        authorization_token: Optional[str] = None,
        conn: Optional[PgConnection] = None,
    ) -> int:
        """
        Enforces SM-DATA-GOV-001 Section 2:
        Destructive deletion of canonical Smart Money evidence is prohibited.
        Fails closed unconditionally because no trusted legal retention or cryptographic
        deletion authority is configured.
        """
        raise PermissionError(
            "CANONICAL_DELETION_PROHIBITED: Destructive deletion of canonical Smart Money evidence is prohibited. "
            "No authorized legal retention authority is configured."
        )


