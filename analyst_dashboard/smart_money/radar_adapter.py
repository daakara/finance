"""
analyst_dashboard/smart_money/radar_adapter.py

Authoritative Read-Only Radar Smart Money Evidence Adapter.
Directives:
- ARX-RADAR-SM-CONFLUENCE-INTEGRITY-001
- ARX-RADAR-SM-ADAPTER-REMEDIATION-003
- ARX-RADAR-SM-ADAPTER-RELEASE-QUALIFICATION-004

Responsibilities & Governance:
1. Reuses canonical PostgreSQL Form 4 evidence retrieval (PostgresSmartMoneyRepository.get_bulk_transactions_as_of)
   and sm_reconciliation_receipts interval certification.
2. Preserves SEC accession, CIK, amendments, transaction classifications, and source timestamps.
3. Enforces fail-closed treatment of missing, stale, ambiguous, and unavailable evidence.
4. Distinguishes seven explicit evidence states without manufacturing zero activity or neutral signals:
   - VERIFIED_BUY: Authentic open-market purchase conviction (Code P, Acquired, non-derivative, unamended/resolved).
   - VERIFIED_SELL: Open-market insider sales (Code S, Disposed, non-derivative).
   - VERIFIED_ZERO: Active governed issuer with authoritative issuer-level COMPLETE reconciliation receipt certifying zero transactions in trailing window.
   - STALE: Latest observation/receipt exceeds trailing freshness SLA (>90 days).
   - UNAVAILABLE: Database repository disconnected, table missing, or query error.
   - HELD: Operational hold ARX_CONFLUENCE_INTEGRITY_HOLD active without explicit read-only authorization.
   - UNVERIFIED: Ambiguous CIK/ticker mapping, unverified issuer, incomplete coverage receipt, or unresolved amendment.
5. Strict Point-in-Time (PIT) Causality:
   - Invalid as_of timestamps fail closed immediately (never fall back to current time).
   - Date-only as_of strings map to authoritative session-close (T23:59:59Z).
   - Enforces public_availability_at <= as_of_watermark to prevent lookahead bias.
6. Authoritative Coverage Contract:
   - VERIFIED_ZERO is emitted ONLY when the exact issuer, full requested observation interval, and certified watermark
     are covered by an authoritative complete receipt. Unrelated global receipts cannot certify issuer-specific zero activity.
"""

import os
import sqlite3
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RadarSmartMoneyAdapter:
    """Production read-only adapter connecting authentic SEC Form 4 evidence to Radar."""

    DEFAULT_WINDOW_DAYS = 90
    STALENESS_THRESHOLD_DAYS = 90

    @staticmethod
    def validate_issuer_coverage_receipt(
        receipt: Optional[Dict[str, Any]],
        clean_sym: str,
        clean_cik: Optional[str],
        interval_start_iso: str,
        as_of_watermark: str,
        window_days: int = DEFAULT_WINDOW_DAYS,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates whether a reconciliation receipt establishes authoritative complete coverage
        for the exact issuer, full requested observation interval, and certified watermark.
        Strictly prevents using cross-issuer, cross-interval, stale-watermark, or partial-coverage
        receipts to certify zero activity.
        """
        if not receipt:
            return False, f"Governed issuer {clean_sym} lacks authoritative reconciliation receipt."

        # 1. Exact Issuer Binding Check
        receipt_cik = str(receipt.get("issuer_cik") or "").strip().zfill(10) if receipt.get("issuer_cik") else None
        receipt_sym = str(receipt.get("ticker") or receipt.get("symbol") or "").strip().upper() if (receipt.get("ticker") or receipt.get("symbol")) else None
        covered_ciks = [str(c).strip().zfill(10) for c in (receipt.get("covered_ciks") or [])]
        covered_syms = [str(s).strip().upper() for s in (receipt.get("covered_tickers") or receipt.get("covered_symbols") or [])]

        is_issuer_covered = False
        if clean_cik and receipt_cik and receipt_cik == clean_cik:
            is_issuer_covered = True
        elif receipt_sym and receipt_sym == clean_sym:
            is_issuer_covered = True
        elif clean_cik and clean_cik in covered_ciks:
            is_issuer_covered = True
        elif clean_sym in covered_syms:
            is_issuer_covered = True

        if not is_issuer_covered:
            return False, f"Coverage receipt {receipt.get('receipt_id')} does not certify requested issuer {clean_sym} (CIK: {clean_cik})."

        # 2. Full Requested Interval Coverage Check
        r_start = str(receipt.get("interval_start") or "")[:10]
        r_end = str(receipt.get("interval_end") or "")[:10]
        req_start = interval_start_iso[:10]
        req_end = as_of_watermark[:10]

        if r_start > req_start:
            return False, f"Coverage receipt interval start ({r_start}) does not cover full requested interval start ({req_start})."
        if r_end < req_end:
            return False, f"Coverage receipt interval end ({r_end}) does not cover requested interval end ({req_end})."

        # 3. Certified Watermark Freshness Check
        r_wm = str(receipt.get("certified_watermark") or "")[:10]
        if r_wm < req_end:
            return False, f"Coverage receipt certified watermark ({r_wm}) is older than requested watermark ({req_end})."

        # 4. Coverage Completeness & Material Gaps Check
        cov_st = str(receipt.get("coverage_status") or "").upper()
        missing_cnt = receipt.get("missing_count", 0)
        gap_cnt = receipt.get("material_gap_count", 0)
        if cov_st != "COMPLETE":
            return False, f"Coverage receipt status is '{cov_st}', not 'COMPLETE'."
        if missing_cnt > 0 or gap_cnt > 0:
            return False, f"Coverage receipt contains missing filings ({missing_cnt}) or material gaps ({gap_cnt})."

        return True, None

    @classmethod
    def get_radar_insider_trades(
        cls,
        symbol: str,
        as_of: Optional[str] = None,
        window_days: int = DEFAULT_WINDOW_DAYS,
        storage_backend: Optional[str] = None,
        db_path: Optional[str] = None,
        postgres_repo: Optional[Any] = None,
        coverage_receipt: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves authentic point-in-time Form 4 insider transactions for an equity.

        Returns:
            Dict containing:
                - status: "VERIFIED" | "VERIFIED_ZERO" | "UNAVAILABLE" | "HELD" | "STALE" | "UNVERIFIED"
                - evidence_state: canonical evidence state string
                - trades: List of raw transaction dictionaries ready for classify_sec_trades()
                - issuer_cik: str or None
                - as_of_watermark: str or None
                - query_source: "POSTGRES" | "SQLITE" | "HELD" | "UNAVAILABLE"
                - reason: explanatory string
        """
        clean_sym = symbol.strip().upper() if symbol else ""
        if not clean_sym:
            return {
                "status": "UNAVAILABLE",
                "evidence_state": "UNAVAILABLE",
                "trades": [],
                "issuer_cik": None,
                "as_of_watermark": as_of,
                "query_source": "UNAVAILABLE",
                "reason": "EMPTY_SYMBOL_PROVIDED",
            }

        # ── 1. OPERATIONAL HOLD ENFORCEMENT ───────────────────────────────────
        # ARX_CONFLUENCE_INTEGRITY_HOLD is active by default.
        # Unless ARX_ALLOW_READ_ONLY_SM=1 is set (authorized for evaluation/test),
        # return HELD state with empty trades, preventing silent production score mutation.
        confluence_hold = os.environ.get("ARX_CONFLUENCE_INTEGRITY_HOLD", "1") == "1"
        read_only_allowed = os.environ.get("ARX_ALLOW_READ_ONLY_SM", "0") == "1"

        if confluence_hold and not read_only_allowed:
            return {
                "status": "HELD",
                "evidence_state": "HELD",
                "trades": [],
                "issuer_cik": None,
                "as_of_watermark": as_of,
                "query_source": "HELD",
                "reason": "ARX_CONFLUENCE_INTEGRITY_HOLD_ACTIVE",
            }

        # ── 2. DETERMINE EVALUATION CUTOFF & STRICT PIT WATERMARK ─────────────
        # Invariant: Invalid timestamps must fail closed, NEVER silently fall back to current time.
        if as_of is not None:
            as_of_str = str(as_of).strip()
            if not as_of_str:
                return {
                    "status": "UNAVAILABLE",
                    "evidence_state": "UNAVAILABLE",
                    "trades": [],
                    "issuer_cik": None,
                    "as_of_watermark": as_of,
                    "query_source": "UNAVAILABLE",
                    "reason": "INVALID_AS_OF_WATERMARK: Empty timestamp string provided",
                }
            try:
                if len(as_of_str) == 10 and "-" in as_of_str:
                    # Date-only string (e.g. "2026-11-06"): Map to session-close EOD (23:59:59Z)
                    parsed_d = datetime.strptime(as_of_str, "%Y-%m-%d").date()
                    as_of_dt = datetime(parsed_d.year, parsed_d.month, parsed_d.day, 23, 59, 59, tzinfo=timezone.utc)
                elif "T" in as_of_str:
                    as_of_dt = datetime.fromisoformat(as_of_str.replace("Z", "+00:00"))
                    if as_of_dt.tzinfo is None:
                        as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
                    else:
                        as_of_dt = as_of_dt.astimezone(timezone.utc)
                else:
                    return {
                        "status": "UNAVAILABLE",
                        "evidence_state": "UNAVAILABLE",
                        "trades": [],
                        "issuer_cik": None,
                        "as_of_watermark": as_of,
                        "query_source": "UNAVAILABLE",
                        "reason": f"INVALID_AS_OF_WATERMARK: Unrecognized timestamp format '{as_of_str}'",
                    }
            except Exception as parse_err:
                return {
                    "status": "UNAVAILABLE",
                    "evidence_state": "UNAVAILABLE",
                    "trades": [],
                    "issuer_cik": None,
                    "as_of_watermark": as_of,
                    "query_source": "UNAVAILABLE",
                    "reason": f"INVALID_AS_OF_WATERMARK: {str(parse_err)}",
                }
        else:
            as_of_dt = datetime.now(timezone.utc)

        as_of_watermark = as_of_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        eval_date = as_of_dt.date()
        cutoff_date = eval_date - timedelta(days=window_days)
        cutoff_str = cutoff_date.isoformat()

        # ── 3. QUERY REPOSITORY UNDER POINT-IN-TIME WATERMARK ─────────────────
        backend = storage_backend
        if not backend:
            try:
                from analyst_dashboard.smart_money.storage.config import get_storage_backend
                backend = get_storage_backend()
            except Exception:
                backend = "sqlite"

        raw_trades: List[Dict[str, Any]] = []
        issuer_cik: Optional[str] = None
        is_governed_issuer = False
        is_scannable = False
        has_complete_coverage_receipt = False
        coverage_rejection_reason: Optional[str] = None
        has_stale_data = False
        latest_record_date: Optional[str] = None
        query_source = "UNAVAILABLE"

        # Attempt PostgreSQL primary
        if backend == "postgres":
            try:
                if postgres_repo is not None:
                    repo = postgres_repo
                else:
                    from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
                    repo = PostgresSmartMoneyRepository()

                # Query governed_issuers table directly (schema authority)
                gov_info = repo.get_governed_issuer(clean_sym)
                if gov_info:
                    is_governed_issuer = True
                    is_scannable = bool(gov_info.get("scannable", True))
                    issuer_cik = str(gov_info["cik"]).zfill(10)

                # Query point-in-time transactions
                bulk_map = repo.get_bulk_transactions_as_of(
                    as_of_watermark=as_of_watermark,
                    symbols=[clean_sym],
                    min_date=cutoff_str,
                    replay_mode="PUBLIC_AVAILABLE",
                )
                raw_trades = bulk_map.get(clean_sym, [])
                query_source = "POSTGRES"

                # Check if coverage receipt exists certifying the interval
                if coverage_receipt is not None:
                    receipt = coverage_receipt
                else:
                    receipt = repo.get_authoritative_receipt_for_interval(
                        interval_start=cutoff_str,
                        interval_end=eval_date.isoformat(),
                        certified_watermark=as_of_watermark,
                        issuer_cik=issuer_cik,
                    )

                is_cov_valid, cov_reason = cls.validate_issuer_coverage_receipt(
                    receipt=receipt,
                    clean_sym=clean_sym,
                    clean_cik=issuer_cik,
                    interval_start_iso=cutoff_str,
                    as_of_watermark=as_of_watermark,
                    window_days=window_days,
                )
                has_complete_coverage_receipt = is_cov_valid
                coverage_rejection_reason = cov_reason

                # If no transactions in 90d window, check if older records exist to test STALE
                if not raw_trades and is_governed_issuer:
                    from analyst_dashboard.smart_money.storage.postgres_connection import get_connection
                    from psycopg2.extras import RealDictCursor
                    with get_connection(dsn=repo.dsn) as conn:
                        with conn.cursor(cursor_factory=RealDictCursor) as cur:
                            cur.execute("""
                                SELECT tx.transaction_date, f.filing_date, av.public_availability_at
                                FROM form4_parsed_transactions tx
                                JOIN form4_raw_filings f ON tx.accession_number = f.accession_number
                                JOIN sm_source_availability_ledger av ON tx.accession_number = av.accession_number
                                WHERE f.ticker = %s AND av.public_availability_at <= %s
                                ORDER BY tx.transaction_date DESC LIMIT 1;
                            """, (clean_sym, as_of_watermark))
                            row = cur.fetchone()
                            if row:
                                latest_record_date = str(row["transaction_date"] or row["filing_date"])
                                has_stale_data = True

            except Exception as pg_exc:
                logger.debug(f"RadarSmartMoneyAdapter PostgreSQL query failed for {clean_sym}: {pg_exc}")
                # Fall back to SQLite if not explicitly pinned to postgres
                if storage_backend == "postgres":
                    return {
                        "status": "UNAVAILABLE",
                        "evidence_state": "UNAVAILABLE",
                        "trades": [],
                        "issuer_cik": None,
                        "as_of_watermark": as_of_watermark,
                        "query_source": "UNAVAILABLE",
                        "reason": f"POSTGRES_ERROR: {str(pg_exc)}",
                    }
                backend = "sqlite"

        # Fallback to local SQLite store
        if backend == "sqlite":
            effective_db_path = db_path
            if not effective_db_path:
                try:
                    from analyst_dashboard.smart_money.storage.config import get_sqlite_evidence_db_path
                    effective_db_path = get_sqlite_evidence_db_path()
                except Exception:
                    effective_db_path = "analyst_dashboard/data/smart_money_cache.db"

            if not os.path.exists(effective_db_path):
                return {
                    "status": "UNAVAILABLE",
                    "evidence_state": "UNAVAILABLE",
                    "trades": [],
                    "issuer_cik": None,
                    "as_of_watermark": as_of_watermark,
                    "query_source": "UNAVAILABLE",
                    "reason": f"DATABASE_FILE_NOT_FOUND: {effective_db_path}",
                }

            conn = None
            try:
                conn = sqlite3.connect(effective_db_path)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()

                # Check if governed_issuers table exists
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='governed_issuers';")
                has_gov_table = bool(cur.fetchone())
                if has_gov_table:
                    cur.execute("SELECT cik, scannable FROM governed_issuers WHERE ticker = ? LIMIT 1;", (clean_sym,))
                    grow = cur.fetchone()
                    if grow:
                        is_governed_issuer = True
                        is_scannable = bool(grow["scannable"])
                        issuer_cik = str(grow["cik"]).zfill(10)

                if not is_governed_issuer:
                    cur.execute("SELECT issuer_cik FROM form4_parsed_transactions WHERE issuer_symbol = ? LIMIT 1;", (clean_sym,))
                    cik_row = cur.fetchone()
                    if cik_row and cik_row["issuer_cik"]:
                        is_governed_issuer = True
                        is_scannable = True
                        issuer_cik = str(cik_row["issuer_cik"]).zfill(10)

                # Check reconciliation receipts for interval coverage
                if coverage_receipt is not None:
                    receipt = coverage_receipt
                else:
                    receipt = None
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sm_reconciliation_receipts';")
                    has_receipts_table = bool(cur.fetchone())
                    if has_receipts_table:
                        cur.execute("PRAGMA table_info(sm_reconciliation_receipts);")
                        cols = [r["name"] for r in cur.fetchall()]
                        if "issuer_cik" in cols or "ticker" in cols:
                            cur.execute("""
                                SELECT * FROM sm_reconciliation_receipts
                                WHERE (issuer_cik = ? OR ticker = ?)
                                  AND interval_start <= ?
                                  AND (interval_end >= ? OR interval_end >= ?)
                                  AND coverage_status = 'COMPLETE'
                                  AND missing_count = 0
                                  AND material_gap_count = 0
                                ORDER BY receipt_version DESC, created_at DESC LIMIT 1;
                            """, (issuer_cik, clean_sym, cutoff_str, eval_date.isoformat(), f"{eval_date.isoformat()}T23:59:59Z"))
                            r_row = cur.fetchone()
                            if r_row:
                                receipt = dict(r_row)

                is_cov_valid, cov_reason = cls.validate_issuer_coverage_receipt(
                    receipt=receipt,
                    clean_sym=clean_sym,
                    clean_cik=issuer_cik,
                    interval_start_iso=cutoff_str,
                    as_of_watermark=as_of_watermark,
                    window_days=window_days,
                )
                has_complete_coverage_receipt = is_cov_valid
                coverage_rejection_reason = cov_reason

                # Query transactions within cutoff and point-in-time availability
                cur.execute("""
                    SELECT record_key, accession_number, issuer_cik, issuer_symbol, reporting_owner_name,
                           officer_title, is_director, is_officer, is_ten_percent_owner, filing_date,
                           transaction_date, transaction_code, signal_type, shares, price_per_share,
                           total_value_usd, acquired_disposed, is_positive_buying_conviction,
                           is_open_market_purchase_evidence, acceptance_datetime,
                           is_derivative, amendment_number, base_accession, resolution_status
                    FROM form4_parsed_transactions
                    WHERE issuer_symbol = ?
                      AND (filing_date >= ? OR transaction_date >= ?)
                      AND (acceptance_datetime IS NULL OR acceptance_datetime <= ?)
                    ORDER BY transaction_date ASC;
                """, (clean_sym, cutoff_str, cutoff_str, as_of_watermark))

                rows = cur.fetchall()
                raw_trades = [dict(r) for r in rows]
                query_source = "SQLITE"

                # If no transactions in window, check for older records to evaluate STALE
                if not raw_trades and is_governed_issuer:
                    cur.execute("""
                        SELECT transaction_date, filing_date
                        FROM form4_parsed_transactions
                        WHERE issuer_symbol = ?
                          AND (acceptance_datetime IS NULL OR acceptance_datetime <= ?)
                        ORDER BY transaction_date DESC LIMIT 1;
                    """, (clean_sym, as_of_watermark))
                    stale_row = cur.fetchone()
                    if stale_row:
                        latest_record_date = str(stale_row["transaction_date"] or stale_row["filing_date"])
                        has_stale_data = True

            except Exception as e:
                return {
                    "status": "UNAVAILABLE",
                    "evidence_state": "UNAVAILABLE",
                    "trades": [],
                    "issuer_cik": None,
                    "as_of_watermark": as_of_watermark,
                    "query_source": "UNAVAILABLE",
                    "reason": f"SQLITE_QUERY_ERROR: {str(e)}",
                }
            finally:
                if conn:
                    conn.close()

        # ── 4. NORMALIZE & FILTER TRADES THROUGH SEC CLASSIFIER ───────────────
        from api.routes.screener import classify_sec_trades

        # Format records for classify_sec_trades:
        # Expected keys: transaction_code, acquired_disposed_code, is_derivative,
        # transaction_shares, price_per_share, is_amendment, amendment_number, etc.
        normalized_trades: List[Dict[str, Any]] = []
        for t in raw_trades:
            # Enforce CIK identity match against canonical governed issuer CIK
            row_cik = str(t.get("issuer_cik") or "").strip().zfill(10) if t.get("issuer_cik") else None
            if issuer_cik and row_cik and row_cik != issuer_cik:
                # Row identity mismatch — reject transaction from consideration
                continue

            amend_num = t.get("amendment_number") or 0
            try:
                amend_num = int(amend_num)
            except Exception:
                amend_num = 0

            norm_trade = {
                "accession_number": t.get("accession_number"),
                "issuer_symbol": clean_sym,
                "issuer_cik": row_cik or issuer_cik,
                "reporting_owner": t.get("reporting_owner_name") or t.get("reporting_owner", ""),
                "reporting_owner_name": t.get("reporting_owner_name") or t.get("reporting_owner", ""),
                "officer_title": t.get("officer_title", ""),
                "is_officer": bool(t.get("is_officer", False)),
                "is_director": bool(t.get("is_director", False)),
                "is_ten_percent_owner": bool(t.get("is_ten_percent", False) or t.get("is_ten_percent_owner", False)),
                "filing_date": t.get("filing_date"),
                "transaction_date": t.get("transaction_date"),
                "transaction_code": t.get("transaction_code", ""),
                "acquired_disposed_code": t.get("acquired_disposed", ""),
                "is_derivative": bool(t.get("is_derivative", False)),
                "transaction_shares": float(t.get("shares") or 0.0),
                "price_per_share": float(t.get("price_per_share") or 0.0),
                "total_value": float(t.get("total_value_usd") or 0.0),
                "is_amendment": amend_num > 0,
                "amendment_number": amend_num,
                "resolution_status": t.get("resolution_status", "RESOLVED"),
            }
            normalized_trades.append(norm_trade)

        classification = classify_sec_trades(normalized_trades)
        buy_trades = classification.get("buy_trades", [])
        sell_trades = classification.get("sell_trades", [])
        has_buy = classification.get("has_insider_buy", False)
        has_sell = classification.get("has_insider_sell", False)

        # ── 5. RESOLVE EXPLICIT EVIDENCE STATE ────────────────────────────────
        if has_buy:
            evidence_state = "VERIFIED_BUY"
            status = "VERIFIED"
            reason = f"Verified Form 4 open-market purchases: {len(buy_trades)} buys, ${classification['total_buy_value_usd']:,.2f} USD."
        elif has_sell:
            evidence_state = "VERIFIED_SELL"
            status = "VERIFIED"
            reason = f"Verified Form 4 open-market sales: {len(sell_trades)} sales, ${classification['total_sell_value_usd']:,.2f} USD."
        elif is_governed_issuer and is_scannable and has_complete_coverage_receipt:
            # Governed issuer with complete coverage receipt certifying 0 transactions in window
            evidence_state = "VERIFIED_ZERO"
            status = "VERIFIED_ZERO"
            reason = f"Active governed issuer {clean_sym} ({issuer_cik}) certified with 0 open-market transactions in complete {window_days}-day interval."
        elif is_governed_issuer and has_stale_data:
            # Issuer has historical transactions, but newest exceeds window
            evidence_state = "STALE"
            status = "STALE"
            reason = f"Smart money evidence is stale: Newest filing from {latest_record_date} exceeds {window_days}-day threshold."
        elif len(normalized_trades) > 0 and len(buy_trades) == 0 and len(sell_trades) == 0:
            evidence_state = "UNVERIFIED"
            status = "UNVERIFIED"
            reasons = [u.get("reason") for u in classification.get("unqualified_trades", [])]
            reason = f"Transactions excluded by qualification gates: {', '.join(set(reasons))}"
        elif is_governed_issuer and not has_complete_coverage_receipt:
            # Governed issuer present, but lacking complete interval coverage receipt -> FAIL CLOSED
            evidence_state = "UNVERIFIED"
            status = "UNVERIFIED"
            if coverage_rejection_reason:
                reason = f"VERIFIED_ZERO contract rejected: {coverage_rejection_reason}"
            else:
                reason = f"Governed issuer {clean_sym} lacks authoritative issuer-level reconciliation receipt certifying trailing {window_days}-day window."
        else:
            evidence_state = "UNAVAILABLE"
            status = "UNAVAILABLE"
            reason = f"No regulatory filings or repository records found for {clean_sym}."

        return {
            "status": status,
            "evidence_state": evidence_state,
            "trades": normalized_trades,
            "classified": classification,
            "issuer_cik": issuer_cik,
            "as_of_watermark": as_of_watermark,
            "query_source": query_source,
            "reason": reason,
        }
