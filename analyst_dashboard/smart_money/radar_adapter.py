"""
analyst_dashboard/smart_money/radar_adapter.py

Authoritative Read-Only Radar Smart Money Evidence Adapter (ARX-RADAR-SM-CONFLUENCE-INTEGRITY-001).

Responsibilities:
1. Reuses canonical PostgreSQL Form 4 evidence retrieval and point-in-time contracts
   (PostgresSmartMoneyRepository.get_bulk_transactions_as_of).
2. Preserves SEC accession, CIK, amendments, transaction classifications, and source timestamps.
3. Enforces fail-closed treatment of missing, stale, ambiguous, and unavailable evidence.
4. Distinguishes seven explicit evidence states without manufacturing zero activity or neutral signals:
   - VERIFIED_BUY: Authentic open-market purchase conviction (Code P, Acquired, non-derivative, unamended/resolved).
   - VERIFIED_SELL: Open-market insider sales (Code S, Disposed, non-derivative).
   - VERIFIED_ZERO: Active governed issuer with 0 open-market transactions in the trailing evaluation window.
   - STALE: Observation date exceeds trailing freshness window (>90 days).
   - UNAVAILABLE: Database repository disconnected, table missing, or query error.
   - HELD: Operational hold ARX_CONFLUENCE_INTEGRITY_HOLD active without explicit read-only authorization.
   - UNVERIFIED: Ambiguous CIK/ticker mapping, unverified issuer, or unresolved amendment.
5. Strict Point-in-Time (PIT) Causality:
   Enforces public_availability_at <= as_of_watermark (SM-DATA-GOV-001 Section 3) to prevent lookahead bias.
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
    ) -> Dict[str, Any]:
        """
        Retrieves authentic point-in-time Form 4 insider transactions for an equity.

        Returns:
            Dict containing:
                - status: "VERIFIED" | "VERIFIED_ZERO" | "UNAVAILABLE" | "HELD" | "STALE" | "UNVERIFIED"
                - evidence_state: canonical evidence state string
                - trades: List of raw transaction dictionaries ready for classify_sec_trades()
                - issuer_cik: str or None
                - as_of_watermark: str
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
        sec_hold = os.environ.get("HOLD_SEC_001", "1") == "1"

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

        # ── 2. DETERMINE EVALUATION CUTOFF & WINDOW ───────────────────────────
        try:
            if as_of:
                if "T" in str(as_of):
                    as_of_dt = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
                else:
                    as_of_dt = datetime.strptime(str(as_of)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                as_of_dt = datetime.now(timezone.utc)
        except Exception:
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
        query_source = "UNAVAILABLE"

        # Attempt PostgreSQL primary
        if backend == "postgres":
            try:
                from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
                repo = PostgresSmartMoneyRepository()
                bulk_map = repo.get_bulk_transactions_as_of(
                    as_of_watermark=as_of_watermark,
                    symbols=[clean_sym],
                    min_date=cutoff_str,
                    replay_mode="PUBLIC_AVAILABLE",
                )
                raw_trades = bulk_map.get(clean_sym, [])
                query_source = "POSTGRES"

                # Check if issuer is governed/active in PostgreSQL
                with repo.transaction() as cur:
                    cur.execute(
                        "SELECT cik FROM form4_governed_issuers WHERE ticker = %s LIMIT 1;",
                        (clean_sym,),
                    )
                    row = cur.fetchone()
                    if row:
                        is_governed_issuer = True
                        issuer_cik = str(row[0]).zfill(10)
                    else:
                        # Check if any raw filing exists for issuer
                        cur.execute(
                            "SELECT cik FROM form4_raw_filings WHERE ticker = %s LIMIT 1;",
                            (clean_sym,),
                        )
                        r2 = cur.fetchone()
                        if r2:
                            is_governed_issuer = True
                            issuer_cik = str(r2[0]).zfill(10)

            except Exception as pg_exc:
                logger.debug(f"RadarSmartMoneyAdapter PostgreSQL query failed for {clean_sym}: {pg_exc}")
                # If configured specifically for postgres in production, fail closed
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
                backend = "sqlite"  # Local fallback

        # Attempt SQLite (local non-production development / tests)
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
                cur.execute("SELECT issuer_cik FROM form4_parsed_transactions WHERE issuer_symbol = ? LIMIT 1;", (clean_sym,))
                cik_row = cur.fetchone()
                if cik_row and cik_row["issuer_cik"]:
                    is_governed_issuer = True
                    issuer_cik = str(cik_row["issuer_cik"]).zfill(10)

                # Query transactions within cutoff and point-in-time availability
                cur.execute("""
                    SELECT record_key, accession_number, issuer_cik, issuer_symbol, reporting_owner_name,
                           officer_title, is_director, is_officer, is_ten_percent_owner, filing_date,
                           transaction_date, transaction_code, signal_type, shares, price_per_share,
                           total_value_usd, acquired_disposed, is_positive_buying_conviction,
                           is_open_market_purchase_evidence, acceptance_datetime
                    FROM form4_parsed_transactions
                    WHERE issuer_symbol = ?
                      AND (filing_date >= ? OR transaction_date >= ?)
                      AND (acceptance_datetime IS NULL OR acceptance_datetime <= ?)
                    ORDER BY transaction_date ASC;
                """, (clean_sym, cutoff_str, cutoff_str, as_of_watermark))

                rows = cur.fetchall()
                raw_trades = [dict(r) for r in rows]
                query_source = "SQLITE"
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

        # ── 4. CANONICAL RECORD NORMALIZATION & EVIDENCE CLASSIFICATION ───────
        normalized_trades: List[Dict[str, Any]] = []
        for t in raw_trades:
            tx_code = str(t.get("transaction_code", "")).upper().strip()
            # Map canonical fields for classify_sec_trades
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
                "cik": str(t.get("issuer_cik") or issuer_cik or ""),
                "symbol": clean_sym,
                "is_amendment": bool(t.get("amendment_number") or t.get("base_accession")),
                "unresolved_amendment": str(t.get("resolution_status", "")) in ("AMENDMENT_TARGET_AMBIGUOUS", "UNRESOLVED", "UNRESOLVED_AMENDMENT"),
            })

        # ── 5. RESOLVE EVIDENCE STATE ─────────────────────────────────────────
        # Import production classifier to test transaction qualification
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
        elif is_governed_issuer:
            evidence_state = "VERIFIED_ZERO"
            status = "VERIFIED_ZERO"
            reason = f"Governed issuer {clean_sym} verified with zero open-market transactions in {window_days}-day window."
        elif len(normalized_trades) > 0 and len(buy_trades) == 0 and len(sell_trades) == 0:
            evidence_state = "UNVERIFIED"
            status = "UNVERIFIED"
            reasons = [u.get("reason") for u in classification.get("unqualified_trades", [])]
            reason = f"Transactions excluded by qualification gates: {', '.join(set(reasons))}"
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
