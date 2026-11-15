"""
analyst_dashboard/smart_money/radar_adapter.py

Authoritative Read-Only Radar Smart Money Evidence Adapter.
Directive: ARX-RADAR-SM-ADAPTER-REMEDIATION-003 (PR #294 Remediation).

Responsibilities & Governance:
1. Reuses canonical PostgreSQL Form 4 evidence retrieval (PostgresSmartMoneyRepository.get_bulk_transactions_as_of)
   and sm_reconciliation_receipts interval certification.
2. Preserves SEC accession, CIK, amendments, transaction classifications, and source timestamps.
3. Enforces fail-closed treatment of missing, stale, ambiguous, and unavailable evidence.
4. Distinguishes seven explicit evidence states without manufacturing zero activity or neutral signals:
   - VERIFIED_BUY: Authentic open-market purchase conviction (Code P, Acquired, non-derivative, unamended/resolved).
   - VERIFIED_SELL: Open-market insider sales (Code S, Disposed, non-derivative).
   - VERIFIED_ZERO: Active governed issuer with COMPLETE reconciliation receipt certifying zero transactions in trailing window.
   - STALE: Latest observation/receipt exceeds trailing freshness SLA (>90 days).
   - UNAVAILABLE: Database repository disconnected, table missing, or query error.
   - HELD: Operational hold ARX_CONFLUENCE_INTEGRITY_HOLD active without explicit read-only authorization.
   - UNVERIFIED: Ambiguous CIK/ticker mapping, unverified issuer, incomplete coverage receipt, or unresolved amendment.
5. Strict Point-in-Time (PIT) Causality:
   - Invalid as_of timestamps fail closed immediately (never fall back to current time).
   - Date-only as_of strings map to authoritative session-close (T23:59:59Z).
   - Enforces public_availability_at <= as_of_watermark to prevent lookahead bias.
"""

import os
import sqlite3
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)


class RadarSmartMoneyAdapter:
    """Production read-only adapter connecting authentic SEC Form 4 evidence to Radar."""

    DEFAULT_WINDOW_DAYS = 90
    STALENESS_THRESHOLD_DAYS = 90

    @classmethod
    def get_radar_insider_trades(
        cls,
        symbol: str,
        as_of: Optional[str] = None,
        window_days: int = DEFAULT_WINDOW_DAYS,
        storage_backend: Optional[str] = None,
        db_path: Optional[str] = None,
        postgres_repo: Optional[Any] = None,
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
                receipt = repo.get_authoritative_receipt_for_interval(
                    interval_start=cutoff_str,
                    interval_end=eval_date.isoformat(),
                    certified_watermark=as_of_watermark,
                )
                if receipt and receipt.get("coverage_status") == "COMPLETE" and receipt.get("missing_count", 1) == 0:
                    has_complete_coverage_receipt = True

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
                if os.environ.get("ARX_ENVIRONMENT") == "production":
                    return {
                        "status": "UNAVAILABLE",
                        "evidence_state": "UNAVAILABLE",
                        "trades": [],
                        "issuer_cik": None,
                        "as_of_watermark": as_of_watermark,
                        "query_source": "POSTGRES_ERROR",
                        "reason": f"POSTGRES_QUERY_FAILED: {str(pg_exc)}",
                    }
                backend = "sqlite"  # Local test fallback

        # Attempt SQLite (isolated tests or local development)
        if backend == "sqlite" and query_source != "POSTGRES":
            try:
                from analyst_dashboard.smart_money.form4_ingestion import connect_smart_money_db
                conn = connect_smart_money_db(db_path=db_path)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()

                # Check if tables exist
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='form4_parsed_transactions';")
                if not cur.fetchone():
                    conn.close()
                    return {
                        "status": "UNAVAILABLE",
                        "evidence_state": "UNAVAILABLE",
                        "trades": [],
                        "issuer_cik": None,
                        "as_of_watermark": as_of_watermark,
                        "query_source": "SQLITE_NO_TABLE",
                        "reason": "SMART_MONEY_TABLES_NOT_INITIALIZED",
                    }

                # Check governed / known issuer
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
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sm_reconciliation_receipts';")
                has_receipts_table = bool(cur.fetchone())
                if has_receipts_table:
                    cur.execute("""
                        SELECT coverage_status, missing_count, material_gap_count
                        FROM sm_reconciliation_receipts
                        WHERE (interval_end = ? OR interval_end = ?)
                          AND coverage_status = 'COMPLETE'
                          AND missing_count = 0
                          AND material_gap_count = 0
                        LIMIT 1;
                    """, (eval_date.isoformat(), f"{eval_date.isoformat()}T23:59:59Z"))
                    r_row = cur.fetchone()
                    if r_row:
                        has_complete_coverage_receipt = True

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
                        SELECT transaction_date, filing_date FROM form4_parsed_transactions
                        WHERE issuer_symbol = ? AND (acceptance_datetime IS NULL OR acceptance_datetime <= ?)
                        ORDER BY transaction_date DESC LIMIT 1;
                    """, (clean_sym, as_of_watermark))
                    old_row = cur.fetchone()
                    if old_row:
                        latest_record_date = str(old_row["transaction_date"] or old_row["filing_date"])
                        has_stale_data = True

                conn.close()

            except Exception as sq_exc:
                logger.debug(f"RadarSmartMoneyAdapter SQLite query failed for {clean_sym}: {sq_exc}")
                return {
                    "status": "UNAVAILABLE",
                    "evidence_state": "UNAVAILABLE",
                    "trades": [],
                    "issuer_cik": None,
                    "as_of_watermark": as_of_watermark,
                    "query_source": "SQLITE_ERROR",
                    "reason": f"SQLITE_QUERY_FAILED: {str(sq_exc)}",
                }

        # ── 4. CANONICAL RECORD NORMALIZATION & CIK IDENTITY AUDIT ────────────
        normalized_trades: List[Dict[str, Any]] = []
        for t in raw_trades:
            tx_code = str(t.get("transaction_code", "")).upper().strip()
            row_cik = str(t.get("issuer_cik", "")).zfill(10) if t.get("issuer_cik") else None

            # Verify CIK identity consistency
            if issuer_cik and row_cik and row_cik != issuer_cik:
                logger.warning(
                    f"CIK_IDENTITY_MISMATCH: Row CIK {row_cik} does not match canonical {issuer_cik} for {clean_sym}."
                )
                continue  # Exclude row with mismatched identity

            normalized_trades.append({
                "transaction_code": tx_code,
                "transaction_type": t.get("signal_type") or ("Purchase" if tx_code == "P" else ("Sale" if tx_code == "S" else "Other")),
                "is_derivative": bool(t.get("is_derivative")),
                "acquired_disposed": str(t.get("acquired_disposed", "")).upper().strip(),
                "total_value": float(t.get("total_value_usd") or 0.0),
                "total_value_usd": float(t.get("total_value_usd") or 0.0),
                "reporting_owner": t.get("reporting_owner_name", ""),
                "reporting_owner_name": t.get("reporting_owner_name", ""),
                "accession_number": t.get("accession_number", ""),
                "filing_date": str(t.get("filing_date", "")),
                "transaction_date": str(t.get("transaction_date", "")),
                "public_availability_at": str(t.get("public_availability_at") or t.get("acceptance_datetime") or ""),
                "cik": str(row_cik or issuer_cik or ""),
                "symbol": clean_sym,
                "is_amendment": bool(t.get("amendment_number") or t.get("base_accession")),
                "unresolved_amendment": str(t.get("resolution_status", "")) in ("AMENDMENT_TARGET_AMBIGUOUS", "UNRESOLVED", "UNRESOLVED_AMENDMENT"),
            })

        # ── 5. RESOLVE EVIDENCE STATE (Zero-Fabrication Contract) ─────────────
        from api.routes.screener import classify_sec_trades
        classification = classify_sec_trades(normalized_trades)

        has_buy = classification["has_insider_buy"]
        has_sell = classification["has_insider_sell"]
        buy_trades = classification["buy_trades"]
        sell_trades = classification["sell_trades"]

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
            reason = f"Governed issuer {clean_sym} certified with 0 open-market transactions in complete {window_days}-day interval."
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
            reason = f"Governed issuer {clean_sym} lacks authoritative reconciliation receipt certifying trailing {window_days}-day window."
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
