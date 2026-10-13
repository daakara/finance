"""
Deterministic, rate-limited, and cache-aware SEC Form 4 Ingestion Manager.
"""

import os
import re
import time
import json
import logging
import sqlite3
import threading
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import requests

from analyst_dashboard.data.sec_edgar_fetcher import (
    SecEdgarFetcher,
    SEC_USER_AGENT,
    validate_sec_user_agent,
    get_sec_user_agent,
    SecRateLimiter,
    RateLimiter,
    DEFAULT_SEC_RATE_LIMITER,
)
from analyst_dashboard.smart_money.form4_models import (
    SmartMoneyTransaction,
    SmartMoneyFiling,
    OwnershipNature,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser

logger = logging.getLogger(__name__)

DEFAULT_SMART_MONEY_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "smart_money_cache.db",
)


def connect_smart_money_db(db_path: str = DEFAULT_SMART_MONEY_DB) -> sqlite3.Connection:
    """
    Canonical SQLite connection factory for Smart Money subsystem (Gate B6 & Blocker C).
    Enforces WAL journal mode, 5000ms busy timeout, and NORMAL synchronous mode on all connections.
    """
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


class Form4IngestionManager:
    """
    Manages SEC Form 4 document discovery, raw retrieval, rate-limited transport,
    local caching, and parsed transaction persistence.
    """

    BASE_URL = "https://data.sec.gov"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
    SCHEMA_VERSION = "2.2.0"
    PRE_HARDENING_SCHEMA_VERSION = "2.1.0"
    CERTIFICATION_HARDENING_MIGRATION_ID = "SM_MIGRATE_V2_1_TO_V2_2_CERTIFICATION_HARDENING_20261010"
    PRE_WAVE2C_SCHEMA_VERSION = "2.0.0"
    WAVE2C_MIGRATION_ID = "SM_MIGRATE_V2_TO_V2_1_INCREMENTAL_20261010"

    def __init__(
        self,
        db_path: str = DEFAULT_SMART_MONEY_DB,
        rate_limiter: Optional[RateLimiter] = None,
        sec_fetcher: Optional[SecEdgarFetcher] = None,
    ):
        self.db_path = db_path
        self.rate_limiter = rate_limiter or DEFAULT_SEC_RATE_LIMITER
        self.sec_fetcher = sec_fetcher or SecEdgarFetcher(rate_limiter=self.rate_limiter)
        self.session = requests.Session()
        candidate_ua = get_sec_user_agent()
        self.is_configured, self.configuration_error = validate_sec_user_agent(candidate_ua)
        self.user_agent = candidate_ua.strip() if self.is_configured and candidate_ua else None
        if self.is_configured and self.user_agent:
            self.session.headers.update({"User-Agent": self.user_agent})
        else:
            self.session.headers.pop("User-Agent", None)
        self._request_count = 0
        self._cache_hits = 0
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return connect_smart_money_db(self.db_path)

    def _init_db(self):
        """Initializes local SQLite schema for raw filings, normalized transactions, ledger, and backfill checkpoints."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # Raw filings cache
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS form4_raw_filings (
                    accession_number TEXT PRIMARY KEY,
                    issuer_cik TEXT,
                    issuer_symbol TEXT,
                    filing_date TEXT,
                    acceptance_datetime TEXT,
                    document_type TEXT,
                    raw_content TEXT,
                    source_content_hash TEXT,
                    source_url TEXT,
                    fetched_at TEXT,
                    disposition TEXT
                )
                """
            )
            # Normalized parsed transactions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS form4_parsed_transactions (
                    record_key TEXT PRIMARY KEY,
                    accession_number TEXT,
                    issuer_cik TEXT,
                    issuer_symbol TEXT,
                    issuer_name TEXT,
                    reporting_owner_cik TEXT,
                    reporting_owner_name TEXT,
                    is_officer INTEGER,
                    officer_title TEXT,
                    is_director INTEGER,
                    is_ten_percent_owner INTEGER,
                    filing_date TEXT,
                    transaction_date TEXT,
                    transaction_code TEXT,
                    signal_type TEXT,
                    shares REAL,
                    price_per_share REAL,
                    total_value_usd REAL,
                    acquired_disposed TEXT,
                    ownership_nature TEXT,
                    is_derivative INTEGER,
                    is_open_market_purchase_evidence INTEGER,
                    is_positive_buying_conviction INTEGER,
                    acceptance_datetime TEXT,
                    public_availability_at TEXT,
                    source_fetch_time TEXT,
                    content_hash TEXT,
                    parsed_at TEXT,
                    FOREIGN KEY(accession_number) REFERENCES form4_raw_filings(accession_number)
                )
                """
            )
            # Append-only source availability ledger
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_source_availability_ledger (
                    accession_number TEXT PRIMARY KEY,
                    issuer_cik TEXT NOT NULL,
                    public_availability_at TEXT NOT NULL,
                    first_fetched_at TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    form_type TEXT NOT NULL,
                    normalization_version TEXT NOT NULL
                )
                """
            )
            # Lineage tracking for Form 4 vs Form 4/A
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_amendments (
                    accession_number TEXT PRIMARY KEY,
                    amended_accession_number TEXT,
                    relationship TEXT NOT NULL,
                    detected_at TEXT NOT NULL
                )
                """
            )
            # Durable backfill checkpoints for crash/resume
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_backfill_checkpoints (
                    backfill_id TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    symbol TEXT,
                    status TEXT NOT NULL,
                    filings_count INTEGER NOT NULL DEFAULT 0,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (backfill_id, issuer_cik)
                )
                """
            )
            # Backfill manifests catalog
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_backfill_manifests (
                    backfill_id TEXT PRIMARY KEY,
                    spec_hash TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    universe_build_id TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            # Append-only backfill executions catalog (V2 Identity)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_backfill_executions (
                    execution_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    scope_hash TEXT NOT NULL,
                    stage_label TEXT NOT NULL,
                    universe_build_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    sealed_at TEXT
                )
                """
            )
            # Execution checkpoints for distinct execution runs
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_execution_checkpoints (
                    execution_id TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    symbol TEXT,
                    status TEXT NOT NULL,
                    filings_count INTEGER NOT NULL DEFAULT 0,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (execution_id, issuer_cik)
                )
                """
            )
            # Safe schema migrations for existing local databases
            for col_def in [
                ("form4_raw_filings", "source_content_hash TEXT"),
                ("form4_raw_filings", "disposition TEXT"),
                ("form4_parsed_transactions", "is_open_market_purchase_evidence INTEGER"),
                ("form4_parsed_transactions", "acceptance_datetime TEXT"),
                ("form4_parsed_transactions", "public_availability_at TEXT"),
                ("form4_parsed_transactions", "source_fetch_time TEXT"),
            ]:
                try:
                    cursor.execute(f"ALTER TABLE {col_def[0]} ADD COLUMN {col_def[1]}")
                except sqlite3.OperationalError:
                    pass

            # Wave 2C: Incremental Ingestion, Watermark, and Gap Ledger Schema (v2.1.0)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_incremental_runs (
                    cycle_id TEXT PRIMARY KEY,
                    delivery_attempt_id TEXT NOT NULL,
                    execution_attempt_id TEXT NOT NULL,
                    universe_build_id TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    overlap_window_seconds INTEGER NOT NULL,
                    previous_watermark TEXT NOT NULL,
                    partitions_requested INTEGER NOT NULL,
                    partitions_completed INTEGER NOT NULL,
                    partitions_failed INTEGER NOT NULL,
                    filings_discovered INTEGER NOT NULL,
                    new_filings INTEGER NOT NULL,
                    known_filings INTEGER NOT NULL,
                    amendments_discovered INTEGER NOT NULL,
                    fetch_succeeded INTEGER NOT NULL,
                    fetch_failed INTEGER NOT NULL,
                    parse_succeeded INTEGER NOT NULL,
                    parse_failed INTEGER NOT NULL,
                    open_gaps_created INTEGER NOT NULL,
                    open_gaps_resolved INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_incremental_partitions (
                    partition_id TEXT PRIMARY KEY,
                    issuer_cik TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    universe_build_id TEXT NOT NULL,
                    last_poll_started_at TEXT,
                    last_poll_completed_at TEXT,
                    processing_cursor TEXT NOT NULL,
                    candidate_complete_through TEXT NOT NULL,
                    certified_complete_through TEXT NOT NULL,
                    last_successful_discovery_at TEXT,
                    open_gap_count INTEGER NOT NULL DEFAULT 0,
                    oldest_open_gap_start TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_partition_watermarks (
                    watermark_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    universe_build_id TEXT NOT NULL,
                    certified_complete_through TEXT NOT NULL,
                    candidate_complete_through TEXT NOT NULL,
                    open_gaps_count INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY(cycle_id) REFERENCES sm_incremental_runs(cycle_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_source_gaps (
                    gap_id TEXT PRIMARY KEY,
                    incremental_cycle_id TEXT NOT NULL,
                    partition_id TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    universe_build_id TEXT NOT NULL,
                    gap_type TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    accession_number TEXT,
                    first_detected_at TEXT NOT NULL,
                    last_attempted_at TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL,
                    resolution_authority TEXT,
                    resolution_reason TEXT,
                    resolved_at TEXT,
                    source_event_time TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_watermark_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    previous_certified_watermark TEXT NOT NULL,
                    new_certified_watermark TEXT NOT NULL,
                    universe_build_id TEXT NOT NULL,
                    universe_content_hash TEXT NOT NULL,
                    incremental_policy_version TEXT NOT NULL,
                    source_contract_version TEXT NOT NULL,
                    partition_frontier_hash TEXT NOT NULL,
                    open_gap_set_hash TEXT NOT NULL,
                    source_ledger_hash TEXT NOT NULL,
                    corpus_state_hash TEXT NOT NULL,
                    certified_at TEXT NOT NULL,
                    is_sealed INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_late_arrival_events (
                    integrity_event_id TEXT PRIMARY KEY,
                    detected_at TEXT NOT NULL,
                    accession_number TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    public_availability_at TEXT NOT NULL,
                    affected_watermark_receipt TEXT NOT NULL,
                    affected_interval TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    is_integrity_breach INTEGER NOT NULL
                )
                """
            )

            # Wave 2C Hardening: Certification Authority, Enumeration Receipts, Snapshots, Gap Events, and Disputes (v2.2.0)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_partition_enumeration_receipts (
                    partition_enumeration_receipt_id TEXT PRIMARY KEY,
                    partition_id TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    universe_build_id TEXT NOT NULL,
                    required_partition_set_hash TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    source_contract_version TEXT NOT NULL,
                    discovery_policy_version TEXT NOT NULL,
                    pagination_policy_version TEXT NOT NULL,
                    source_snapshot_manifest_hash TEXT NOT NULL,
                    pages_expected_or_discovered INTEGER NOT NULL,
                    pages_consumed INTEGER NOT NULL,
                    pagination_exhausted INTEGER NOT NULL,
                    continuation_token_remaining INTEGER NOT NULL,
                    first_enumeration_accession_set_hash TEXT NOT NULL,
                    post_holdback_accession_set_hash TEXT NOT NULL,
                    accession_set_stable INTEGER NOT NULL,
                    discovery_gap_count INTEGER NOT NULL,
                    enumeration_status TEXT NOT NULL,
                    enumerated_at TEXT NOT NULL,
                    receipt_hash TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_source_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    partition_id TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    request_identity TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    raw_payload TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    pagination_metadata TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    source_contract_version TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_expected_accession_sets (
                    expected_set_id TEXT NOT NULL,
                    candidate_frontier TEXT NOT NULL,
                    accession_number TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    public_availability_at TEXT NOT NULL,
                    source_snapshot_identity TEXT NOT NULL,
                    source_content_hash TEXT NOT NULL,
                    PRIMARY KEY (expected_set_id, accession_number)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_gap_events (
                    event_id TEXT PRIMARY KEY,
                    event_sequence INTEGER NOT NULL,
                    gap_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    partition_id TEXT NOT NULL,
                    issuer_cik TEXT NOT NULL,
                    gap_type TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    event_payload TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    event_hash TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_certification_receipts_v2 (
                    receipt_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    receipt_sequence INTEGER NOT NULL UNIQUE,
                    previous_receipt_id TEXT NOT NULL,
                    previous_receipt_sha256 TEXT NOT NULL,
                    previous_certified_frontier TEXT NOT NULL,
                    candidate_frontier TEXT NOT NULL,
                    certified_frontier TEXT NOT NULL,
                    closure_proof_sha256 TEXT NOT NULL,
                    universe_build_id TEXT NOT NULL,
                    universe_content_hash TEXT NOT NULL,
                    required_partition_count INTEGER NOT NULL,
                    required_partition_set_hash TEXT NOT NULL,
                    certified_at TEXT NOT NULL,
                    certifier_component_version TEXT NOT NULL,
                    certifier_functional_sha TEXT NOT NULL,
                    receipt_sha256 TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_audit_disputes (
                    dispute_id TEXT PRIMARY KEY,
                    receipt_id TEXT NOT NULL,
                    candidate_frontier TEXT NOT NULL,
                    dispute_state TEXT NOT NULL,
                    sla_state TEXT NOT NULL,
                    escalation_level TEXT NOT NULL,
                    certification_state TEXT NOT NULL,
                    first_divergent_step TEXT NOT NULL,
                    minimal_counterexample TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    triage_deadline TEXT NOT NULL,
                    root_cause_deadline TEXT NOT NULL,
                    reconciliation_deadline TEXT NOT NULL,
                    governance_deadline TEXT NOT NULL,
                    sla_policy_version TEXT NOT NULL,
                    sla_policy_sha256 TEXT NOT NULL,
                    resolved_at TEXT,
                    resolution_artifact_hash TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sm_audit_dispute_history (
                    history_id TEXT PRIMARY KEY,
                    dispute_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    from_dispute_state TEXT NOT NULL,
                    to_dispute_state TEXT NOT NULL,
                    from_sla_state TEXT NOT NULL,
                    to_sla_state TEXT NOT NULL,
                    from_escalation_level TEXT NOT NULL,
                    to_escalation_level TEXT NOT NULL,
                    transition_trigger TEXT NOT NULL,
                    transition_payload TEXT NOT NULL,
                    transitioned_at TEXT NOT NULL
                )
                """
            )

            # Indices for rapid querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_f4_symbol ON form4_parsed_transactions(issuer_symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_f4_tx_date ON form4_parsed_transactions(transaction_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_f4_filing_date ON form4_parsed_transactions(filing_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_f4_public_avail ON form4_parsed_transactions(public_availability_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_public_avail ON sm_source_availability_ledger(public_availability_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gaps_status ON sm_source_gaps(status, partition_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gaps_cik ON sm_source_gaps(issuer_cik)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_late_events_breach ON sm_late_arrival_events(is_integrity_breach)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inc_partitions_cik ON sm_incremental_partitions(issuer_cik)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_enum_receipts_cik ON sm_partition_enumeration_receipts(issuer_cik)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_gap_events_seq ON sm_gap_events(event_sequence)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cert_receipts_v2_seq ON sm_certification_receipts_v2(receipt_sequence)")
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, int]:
        return {
            "request_count": self._request_count,
            "cache_hits": self._cache_hits,
        }

    def reset_stats(self):
        self._request_count = 0
        self._cache_hits = 0

    def _fetch_url_with_retry(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Fetches URL with rate limiting and exponential backoff on 429/5xx."""
        if not self.is_configured:
            logger.warning(f"SEC fetch blocked for {url}: SEC_USER_AGENT is not configured or fails validation.")
            return None
        backoff = 1.0
        for attempt in range(max_retries):
            self.rate_limiter.acquire()
            self._request_count += 1
            try:
                resp = self.session.get(url, timeout=10)
                if resp.status_code == 200:
                    return resp.text
                elif resp.status_code in [429, 500, 502, 503, 504]:
                    logger.warning(f"HTTP {resp.status_code} on {url}, retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2.0
                elif resp.status_code == 404:
                    logger.debug(f"HTTP 404 Not Found on {url}")
                    return None
                else:
                    logger.warning(f"Unexpected HTTP {resp.status_code} on {url}")
                    return None
            except Exception as e:
                logger.warning(f"Fetch failed on {url} (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(backoff)
                backoff *= 2.0
        return None

    def get_cached_raw_filing(self, accession_number: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM form4_raw_filings WHERE accession_number = ?", (accession_number,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        finally:
            conn.close()
        return None

    @classmethod
    def validate_timezone_aware_timestamp(cls, ts: str) -> str:
        """
        Validates that a timestamp string is timezone-aware.
        Rejects naive datetimes by raising ValueError.
        Normalizes valid timestamps to ISO 8601 UTC string.
        """
        if not ts:
            raise ValueError("Timestamp cannot be empty")
        cleaned = ts.strip()
        # Reject naive datetimes: must have 'Z' or timezone offset (+/- after date part)
        if "Z" not in cleaned and "+" not in cleaned and "-" not in cleaned[10:]:
            raise ValueError(f"Naive datetime rejected: timestamp '{ts}' must include timezone offset or 'Z'")
        try:
            dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except Exception as e:
            raise ValueError(f"Invalid timestamp format '{ts}': {e}")
        if dt.tzinfo is None:
            raise ValueError(f"Naive datetime rejected: timestamp '{ts}' must have timezone info")
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def save_raw_filing_and_transactions(self, filing: SmartMoneyFiling, source_url: str):
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # 1. Accession duplicate / content hash integrity check
            cursor.execute("SELECT source_content_hash FROM form4_raw_filings WHERE accession_number = ?", (filing.accession_number,))
            existing = cursor.fetchone()
            if existing and existing[0] and filing.source_content_hash:
                if existing[0] != filing.source_content_hash:
                    logger.warning(
                        f"SOURCE_INTEGRITY_AMENDMENT_EVENT: Accession {filing.accession_number} content hash differs! "
                        f"Existing: {existing[0]}, Incoming: {filing.source_content_hash}"
                    )

            # 2. Persist raw filing
            cursor.execute(
                """
                INSERT OR REPLACE INTO form4_raw_filings 
                (accession_number, issuer_cik, issuer_symbol, filing_date, acceptance_datetime, document_type, raw_content, source_content_hash, source_url, fetched_at, disposition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filing.accession_number,
                    filing.issuer_cik,
                    filing.issuer_symbol,
                    filing.filing_date,
                    filing.acceptance_datetime,
                    filing.document_type,
                    filing.raw_xml,
                    filing.source_content_hash,
                    source_url,
                    filing.fetched_at or now_str,
                    filing.disposition,
                ),
            )

            # 3. Source Availability Ledger (Append-only, immutable earlier time)
            # Authority Precedence: EXACT_SEC_ACCEPTANCE > UNKNOWN (Fail-closed, no T22 or fetch_time fallback)
            if filing.acceptance_datetime:
                try:
                    public_avail = self.validate_timezone_aware_timestamp(filing.acceptance_datetime)
                except ValueError:
                    public_avail = "UNKNOWN"
            else:
                public_avail = "UNKNOWN"
            cursor.execute("SELECT public_availability_at FROM sm_source_availability_ledger WHERE accession_number = ?", (filing.accession_number,))
            ledger_row = cursor.fetchone()
            if not ledger_row:
                cursor.execute(
                    """
                    INSERT INTO sm_source_availability_ledger
                    (accession_number, issuer_cik, public_availability_at, first_fetched_at, source_hash, form_type, normalization_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        filing.accession_number,
                        filing.issuer_cik,
                        public_avail,
                        filing.fetched_at or now_str,
                        filing.source_content_hash,
                        filing.document_type,
                        "1.0.0",
                    ),
                )
            # Invariant: SOURCE_AVAILABILITY_CAN_BE_REWRITTEN_EARLIER = NO (leave existing public_availability_at untouched)

            # 4. Lineage / Amendment tracking
            if filing.is_amendment or filing.document_type in ["4/A", "4A"]:
                prior_acc = None
                rel = "UNRESOLVED_AMENDMENT"

                # Check if explicit target is provided in XML
                if getattr(filing, "amended_accession_number", None):
                    cand = filing.amended_accession_number
                    cursor.execute("SELECT 1 FROM form4_raw_filings WHERE accession_number = ?", (cand,))
                    if cursor.fetchone():
                        prior_acc = cand
                        rel = "EXPLICIT_XML_AMENDMENT"

                # If still unresolved, look for unique candidate on same reporting period
                if not prior_acc:
                    cursor.execute(
                        """
                        SELECT accession_number FROM form4_raw_filings 
                        WHERE issuer_cik = ? AND filing_date = ? AND accession_number != ? AND document_type = '4'
                        """,
                        (filing.issuer_cik, filing.filing_date, filing.accession_number),
                    )
                    candidates = [row[0] for row in cursor.fetchall()]
                    if len(candidates) == 1:
                        prior_acc = candidates[0]
                        rel = "AMENDMENT_INFERRED_SINGLE_CANDIDATE"
                    elif len(candidates) > 1:
                        prior_acc = None
                        rel = "AMENDMENT_TARGET_AMBIGUOUS"
                    else:
                        prior_acc = None
                        rel = "UNRESOLVED_AMENDMENT"

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO sm_amendments
                    (accession_number, amended_accession_number, relationship, detected_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (filing.accession_number, prior_acc, rel, now_str),
                )
                if prior_acc and rel != "AMENDMENT_TARGET_AMBIGUOUS":
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO sm_amendments
                        (accession_number, amended_accession_number, relationship, detected_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (prior_acc, filing.accession_number, "SUPERSEDED_OR_CORRECTED", now_str),
                    )

            # 5. Persist normalized transactions
            for tx in filing.transactions:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO form4_parsed_transactions
                    (record_key, accession_number, issuer_cik, issuer_symbol, issuer_name, reporting_owner_cik, reporting_owner_name,
                     is_officer, officer_title, is_director, is_ten_percent_owner, filing_date, transaction_date, transaction_code,
                     signal_type, shares, price_per_share, total_value_usd, acquired_disposed, ownership_nature, is_derivative,
                     is_open_market_purchase_evidence, is_positive_buying_conviction, acceptance_datetime, public_availability_at,
                     source_fetch_time, content_hash, parsed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tx.record_key,
                        tx.accession_number,
                        tx.issuer_cik,
                        tx.issuer_symbol,
                        tx.issuer_name,
                        tx.reporting_owner_cik,
                        tx.reporting_owner_name,
                        1 if tx.is_officer else 0,
                        tx.officer_title,
                        1 if tx.is_director else 0,
                        1 if tx.is_ten_percent_owner else 0,
                        tx.filing_date,
                        tx.transaction_date,
                        tx.transaction_code,
                        tx.signal_type.value if hasattr(tx.signal_type, "value") else str(tx.signal_type),
                        tx.shares,
                        tx.price_per_share,
                        tx.total_value_usd,
                        tx.acquired_disposed,
                        tx.ownership_nature.value if hasattr(tx.ownership_nature, "value") else str(tx.ownership_nature),
                        1 if tx.is_derivative else 0,
                        1 if tx.is_open_market_purchase_evidence else 0,
                        1 if tx.is_positive_buying_conviction else 0,
                        tx.acceptance_datetime,
                        tx.public_availability_at or public_avail,
                        tx.source_fetch_time or now_str,
                        tx.content_hash,
                        tx.normalized_at or now_str,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _row_to_transaction(self, r: sqlite3.Row) -> SmartMoneyTransaction:
        col_names = r.keys()
        is_open_mkt = (
            bool(r["is_open_market_purchase_evidence"])
            if "is_open_market_purchase_evidence" in col_names and r["is_open_market_purchase_evidence"] is not None
            else bool(r["is_positive_buying_conviction"])
        )
        return SmartMoneyTransaction(
            record_key=r["record_key"],
            accession_number=r["accession_number"],
            issuer_cik=r["issuer_cik"],
            issuer_symbol=r["issuer_symbol"],
            issuer_name=r["issuer_name"],
            reporting_owner_cik=r["reporting_owner_cik"],
            reporting_owner_name=r["reporting_owner_name"],
            is_officer=bool(r["is_officer"]),
            officer_title=r["officer_title"],
            is_director=bool(r["is_director"]),
            is_ten_percent_owner=bool(r["is_ten_percent_owner"]),
            filing_date=r["filing_date"],
            transaction_date=r["transaction_date"],
            transaction_code=r["transaction_code"],
            signal_type=r["signal_type"],
            shares=r["shares"],
            price_per_share=r["price_per_share"],
            total_value_usd=r["total_value_usd"],
            acquired_disposed=r["acquired_disposed"],
            ownership_nature=OwnershipNature(r["ownership_nature"]),
            is_derivative=bool(r["is_derivative"]),
            is_open_market_purchase_evidence=is_open_mkt,
            is_positive_buying_conviction=bool(r["is_positive_buying_conviction"]),
            acceptance_datetime=r["acceptance_datetime"] if "acceptance_datetime" in col_names else None,
            public_availability_at=r["public_availability_at"] if "public_availability_at" in col_names else None,
            source_fetch_time=r["source_fetch_time"] if "source_fetch_time" in col_names else None,
            normalized_at=r["parsed_at"] if "parsed_at" in col_names else None,
            footnotes=[],
            source_url="",
            content_hash=r["content_hash"],
        )

    def get_parsed_transactions(
        self,
        symbol: Optional[str] = None,
        cik: Optional[str] = None,
        min_date: Optional[str] = None,
    ) -> List[SmartMoneyTransaction]:
        """Queries normalized transactions from local SQLite cache."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM form4_parsed_transactions WHERE 1=1"
            params = []
            if symbol:
                query += " AND issuer_symbol = ?"
                params.append(symbol.upper().strip())
            if cik:
                query += " AND issuer_cik = ?"
                params.append(str(cik).zfill(10))
            if min_date:
                query += " AND transaction_date >= ?"
                params.append(min_date)
            query += " ORDER BY transaction_date DESC"
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
        finally:
            conn.close()

        return [self._row_to_transaction(r) for r in rows]

    def get_transactions_as_of(
        self,
        as_of_time: str,
        symbol: Optional[str] = None,
        cik: Optional[str] = None,
        min_date: Optional[str] = None,
    ) -> List[SmartMoneyTransaction]:
        """
        Point-in-time query: Returns only transactions whose recorded availability
        occurred at or before as_of_time (l.public_availability_at <= as_of_time).
        Rejects naive datetimes.
        """
        as_of_utc = self.validate_timezone_aware_timestamp(as_of_time)
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = """
                SELECT t.* FROM form4_parsed_transactions t
                JOIN sm_source_availability_ledger l ON t.accession_number = l.accession_number
                WHERE l.public_availability_at IS NOT NULL
                  AND l.public_availability_at != ''
                  AND l.public_availability_at != 'UNKNOWN'
                  AND l.public_availability_at <= ?
            """
            params = [as_of_utc]
            if symbol:
                query += " AND t.issuer_symbol = ?"
                params.append(symbol.upper().strip())
            if cik:
                query += " AND t.issuer_cik = ?"
                params.append(str(cik).zfill(10))
            if min_date:
                query += " AND t.transaction_date >= ?"
                params.append(min_date)
            query += " ORDER BY t.transaction_date DESC, t.record_key ASC"
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
        finally:
            conn.close()

        return [self._row_to_transaction(r) for r in rows]

    def get_bulk_transactions_as_of(
        self,
        as_of_time: str,
        symbols: Optional[List[str]] = None,
        ciks: Optional[List[str]] = None,
        min_date: Optional[str] = None,
    ) -> Dict[str, List[SmartMoneyTransaction]]:
        """
        Bulk Point-in-Time Query (Gate B2 & Phase F):
        Retrieves all transactions across multiple issuers whose recorded availability
        occurred at or before as_of_time (l.public_availability_at <= as_of_time).
        Returns a mapping of symbol (or CIK) -> List[SmartMoneyTransaction].
        Avoids N+1 database queries during market-wide universe evaluation.
        """
        as_of_utc = self.validate_timezone_aware_timestamp(as_of_time)
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = """
                SELECT t.* FROM form4_parsed_transactions t
                JOIN sm_source_availability_ledger l ON t.accession_number = l.accession_number
                WHERE l.public_availability_at IS NOT NULL
                  AND l.public_availability_at != ''
                  AND l.public_availability_at != 'UNKNOWN'
                  AND l.public_availability_at <= ?
            """
            params: List[Any] = [as_of_utc]

            if symbols:
                clean_syms = [s.upper().strip() for s in symbols if s]
                if clean_syms:
                    placeholders = ",".join("?" for _ in clean_syms)
                    query += f" AND t.issuer_symbol IN ({placeholders})"
                    params.extend(clean_syms)

            if ciks:
                clean_ciks = [str(c).zfill(10) for c in ciks if c]
                if clean_ciks:
                    placeholders = ",".join("?" for _ in clean_ciks)
                    query += f" AND t.issuer_cik IN ({placeholders})"
                    params.extend(clean_ciks)

            if min_date:
                query += " AND t.transaction_date >= ?"
                params.append(min_date)

            query += " ORDER BY t.transaction_date DESC, t.record_key ASC"
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

            result: Dict[str, List[SmartMoneyTransaction]] = {}
            for r in rows:
                tx = self._row_to_transaction(r)
                key = tx.issuer_symbol if tx.issuer_symbol else tx.issuer_cik
                if key not in result:
                    result[key] = []
                result[key].append(tx)

            return result
        finally:
            conn.close()

    def get_company_submissions_by_cik(self, cik: str) -> Optional[Dict[str, Any]]:
        """Fetches submissions JSON directly by CIK with rate limiting."""
        cik_str = str(cik).zfill(10)
        url = f"{self.BASE_URL}/submissions/CIK{cik_str}.json"
        text = self._fetch_url_with_retry(url)
        if text:
            try:
                return json.loads(text)
            except Exception as e:
                logger.warning(f"Failed to parse submissions JSON for CIK {cik_str}: {e}")
        return None

    def ingest_filing_by_accession(
        self,
        cik: str,
        accession_number: str,
        filing_date: Optional[str] = None,
        acceptance_dt: Optional[str] = None,
        primary_doc_name: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Optional[SmartMoneyFiling]:
        """
        Retrieves raw XML or submission txt for a specific accession, parses transactions,
        and saves into local SQLite store.
        """
        clean_acc = accession_number.replace("-", "")
        cik_str = str(cik).zfill(10)
        cik_int = int(cik_str)

        if not force_refresh:
            cached = self.get_cached_raw_filing(accession_number)
            if cached:
                self._cache_hits += 1
                # Return parsed filing from cached XML
                filing = Form4XmlParser.parse_filing(
                    raw_content=cached["raw_content"],
                    accession_number=accession_number,
                    filing_date_hint=cached["filing_date"],
                    acceptance_datetime=cached["acceptance_datetime"],
                    source_url=cached["source_url"],
                    source_fetch_time=cached.get("fetched_at"),
                )
                filing.source_content_hash = cached.get("source_content_hash") or ""
                return filing

        fetch_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Build SEC URL
        raw_xml_text = None
        source_url = ""
        if primary_doc_name:
            clean_doc_name = primary_doc_name.split("/")[-1]
            if clean_doc_name.endswith(".xml") or ".xml" in clean_doc_name:
                xml_url = f"{self.ARCHIVES_URL}/{cik_int}/{clean_acc}/{clean_doc_name}"
                raw_xml_text = self._fetch_url_with_retry(xml_url)
                source_url = xml_url

        # Fallback to full submission .txt which contains the XML block
        if not raw_xml_text:
            txt_url = f"{self.ARCHIVES_URL}/{cik_int}/{clean_acc}/{accession_number}.txt"
            raw_xml_text = self._fetch_url_with_retry(txt_url)
            source_url = txt_url

        if not raw_xml_text:
            logger.warning(f"Could not retrieve raw submission for {accession_number} (CIK {cik_str})")
            return None

        filing = Form4XmlParser.parse_filing(
            raw_content=raw_xml_text,
            accession_number=accession_number,
            filing_date_hint=filing_date,
            acceptance_datetime=acceptance_dt,
            source_url=source_url,
            source_fetch_time=fetch_time,
        )

        self.save_raw_filing_and_transactions(filing, source_url)
        return filing

    def ingest_accession(
        self,
        accession_number: str,
        issuer_cik: str,
        filing_date: Optional[str] = None,
        acceptance_datetime: Optional[str] = None,
        raw_content: Optional[str] = None,
        source_url: str = "https://data.sec.gov",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Accession-first canonical ingestion primitive (Phase E).
        Preserves idempotency, verifies content hash integrity, parses normalized transactions,
        and records public availability and lineage.
        """
        if not accession_number:
            raise ValueError("ACCESSION_NUMBER_REQUIRED")

        # 1. If raw_content is provided directly (e.g. from tests or stream)
        if raw_content:
            incoming_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
            cached = self.get_cached_raw_filing(accession_number)
            if cached and not force_refresh:
                existing_hash = cached.get("source_content_hash")
                if existing_hash and existing_hash != incoming_hash:
                    raise RuntimeError(
                        f"SOURCE_INTEGRITY_CONFLICT: Conflicting content for accession {accession_number}! "
                        f"Existing hash: {existing_hash}, Incoming hash: {incoming_hash}"
                    )
                return {
                    "status": "KNOWN_IDEMPOTENT_NOOP",
                    "accession_number": accession_number,
                    "content_hash": existing_hash,
                }

            fetch_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            filing = Form4XmlParser.parse_filing(
                raw_content=raw_content,
                accession_number=accession_number,
                filing_date_hint=filing_date,
                acceptance_datetime=acceptance_datetime,
                source_url=source_url,
                source_fetch_time=fetch_time,
            )
            self.save_raw_filing_and_transactions(filing, source_url)
            return {
                "status": "INGESTED_NEW",
                "accession_number": accession_number,
                "content_hash": filing.source_content_hash,
                "transactions_count": len(filing.transactions),
            }

        # 2. Outbound retrieval by accession
        filing = self.ingest_filing_by_accession(
            cik=issuer_cik,
            accession_number=accession_number,
            filing_date=filing_date,
            acceptance_dt=acceptance_datetime,
            force_refresh=force_refresh,
        )
        if not filing:
            return {
                "status": "RETRIEVAL_FAILED",
                "accession_number": accession_number,
            }

        return {
            "status": "INGESTED_NEW" if filing.transactions else "INGESTED_NO_TRANSACTIONS",
            "accession_number": accession_number,
            "content_hash": filing.source_content_hash,
            "transactions_count": len(filing.transactions),
        }

    def ingest_issuer(
        self,
        symbol: Optional[str] = None,
        cik: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        max_filings: Optional[int] = 5,
        force_refresh: bool = False,
    ) -> List[SmartMoneyFiling]:
        """
        Discovers and ingests Form 4 filings for a single issuer by symbol or CIK.
        Supports date filtering [date_start, date_end].
        """
        if not cik and symbol:
            cik = self.sec_fetcher.resolve_cik(symbol)
        if not cik:
            logger.warning(f"Could not resolve CIK for symbol={symbol}, cik={cik}")
            return []

        cik_str = str(cik).zfill(10)
        subs = self.get_company_submissions_by_cik(cik_str)
        if not subs and symbol:
            subs = self.sec_fetcher.get_company_submissions(symbol)
        if not subs or "filings" not in subs or "recent" not in subs["filings"]:
            return []

        recent = subs["filings"]["recent"]
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        acceptance_dts = recent.get("acceptanceDateTime", [])
        primary_docs = recent.get("primaryDocument", [])

        ingested_filings: List[SmartMoneyFiling] = []
        count = 0
        for i in range(len(forms)):
            if max_filings is not None and count >= max_filings:
                break
            form = forms[i]
            if form in ["4", "4/A"]:
                fdate = filing_dates[i] if i < len(filing_dates) else ""
                if date_start and fdate < date_start:
                    continue
                if date_end and fdate > date_end:
                    continue

                acc = accessions[i] if i < len(accessions) else ""
                acc_dt = acceptance_dts[i] if i < len(acceptance_dts) else ""
                p_doc = primary_docs[i] if i < len(primary_docs) else ""

                if acc:
                    filing = self.ingest_filing_by_accession(
                        cik=cik_str,
                        accession_number=acc,
                        filing_date=fdate,
                        acceptance_dt=acc_dt,
                        primary_doc_name=p_doc,
                        force_refresh=force_refresh,
                    )
                    if filing:
                        ingested_filings.append(filing)
                        count += 1

        return ingested_filings

    def ingest_universe_sample(
        self,
        symbols: List[str],
        max_filings_per_symbol: int = 3,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes bounded ingestion across a list of symbols.
        Returns comprehensive audit statistics.
        """
        self.reset_stats()
        results = {
            "sample_universe_count": len(symbols),
            "symbols_processed": 0,
            "filings_discovered": 0,
            "transactions_parsed": 0,
            "open_market_purchases": 0,
            "excluded_non_directional": 0,
            "duplicates": 0,
            "errors": [],
            "request_count": 0,
            "cache_hits": 0,
        }

        seen_record_keys = set()

        for sym in symbols:
            try:
                filings = self.ingest_issuer(sym, max_filings=max_filings_per_symbol, force_refresh=force_refresh)
                results["symbols_processed"] += 1
                results["filings_discovered"] += len(filings)
                for f in filings:
                    for tx in f.transactions:
                        if tx.record_key in seen_record_keys:
                            results["duplicates"] += 1
                        else:
                            seen_record_keys.add(tx.record_key)

                        results["transactions_parsed"] += 1
                        if tx.is_open_market_purchase_evidence:
                            results["open_market_purchases"] += 1
                        else:
                            results["excluded_non_directional"] += 1
            except Exception as e:
                logger.warning(f"Error ingesting symbol {sym}: {e}")
                results["errors"].append({"symbol": sym, "error": str(e)})

        stats = self.get_stats()
        results["request_count"] = stats["request_count"]
        results["cache_hits"] = stats["cache_hits"]
        return results
