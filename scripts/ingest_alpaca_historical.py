#!/usr/bin/env python3
"""
scripts/ingest_alpaca_historical.py

ARX Terminal — Bounded Alpaca Historical SIP Market Data Ingestion Adapter.
Implements the authoritative Package B market-data acquisition contract against
the persistent MarketDatabaseEngine.

GOVERNING CONTRACT & INVARIANTS (SIP-01 through SIP-06):
1. EXPLICIT FEED ENFORCEMENT: Uses explicit consolidated SIP feed (feed="sip").
   Single-venue IEX volume (~2.4–2.8% market share) is strictly prohibited as a
   substitute for consolidated tape volume under Selection Rule L04.
2. ADJUSTMENT FIDELITY: Requests split-adjusted daily OHLCV bars (adjustment="split")
   leaving dividends unadjusted per canonical contract (AdjustmentState.SPLIT_ADJUSTED_ONLY).
3. COMPLETED TRADING SESSIONS POLICY: Daily bars are anchored strictly to completed
   exchange trading sessions (end <= T_completed, max completed session 2026-10-19).
   Non-trading days (weekends, holidays), unverified future dates, and impossible dates fail closed.
4. ATOMIC PERSISTENCE: Bars and sidecar provenance are persisted atomically per symbol
   within a single SQLite transaction via MarketDatabaseEngine.save_daily_candles_with_evidence.
5. STRICT READ-ONLY DRY-RUN: Default invocation is a non-mutating preparation/dry-run.
   Dry-run uses a strict read-only connection (?mode=ro) and NEVER creates schema, initializes
   tables, alters journal configuration, or mutates an existing file. A 0-byte file remains 0 bytes.
   Apply mode requires an explicit --db-path pointing to an already initialized database.
6. VOLUME & NUMERIC FIDELITY: Volume must be present, finite, non-negative, and strictly
   integer (fractional volumes like 1.7 are rejected, never silently truncated).
7. PAGINATION INTEGRITY: Reaching max_pages with a remaining token fails closed as an
   incomplete acquisition. Repeated tokens trigger cycle detection.
8. PROVENANCE & HISTORY COMPATIBILITY: Existing stored data is checked for window depth,
   earliest date coverage, and adjustment compatibility. Split-adjusted bars are never appended
   to dividend-adjusted or unknown-basis history. A single endpoint bar cannot skip a 1-year window.
9. L03 PRICE FLOOR EQUIVALENCE BRIDGE: On completed session 2026-10-19, split factor to present
   is 1.0; unadjusted price equals split-adjusted price (P_raw = P_split), satisfying L03.
10. HONEST BATCH OUTCOMES: Distinguishes SUCCESS, PARTIAL_SUCCESS, and FAILURE. CLI returns
    nonzero exit codes (1 or 2) when provider requests or validations fail.
11. IMPORT & HELP SAFETY: Module imports and CLI --help execute zero network calls and zero DB writes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.data.alpaca_fetcher import AlpacaMarketFetcher
from analyst_dashboard.data.market_db import MarketDatabaseEngine
from analyst_dashboard.data.market_evidence import (
    AdjustmentState,
    IngestionSource,
    MarketEvidence,
    MarketProvenance,
    ObservationPrecision,
    ObservationSource,
    Provider,
    StructuralQuality,
    assess_structural_quality,
)

logger = logging.getLogger("arx.ingestion.alpaca_sip")

# Authoritative NYSE trading calendar
try:
    import exchange_calendars as ec
    _NYSE_CALENDAR = ec.get_calendar("XNYS")
except Exception:
    _NYSE_CALENDAR = None

# Maximum authorized completed exchange session for Sprint 2 Bounded Expansion
MAX_COMPLETED_SESSION = date(2026, 10, 19)


def is_nyse_trading_session(session_date: date) -> Tuple[bool, Optional[str]]:
    """
    Checks whether a given date is an official NYSE trading session.
    Fails closed if the authoritative exchange calendar is unavailable or out of range.
    Returns: (is_valid_session: bool, rejection_reason: Optional[str])
    """
    if _NYSE_CALENDAR is None:
        return False, "CALENDAR_UNAVAILABLE: Authoritative exchange calendar dependency is unavailable."
    try:
        if not _NYSE_CALENDAR.is_session(session_date):
            return False, f"NON_TRADING_SESSION: Date '{session_date.isoformat()}' is not a recognized NYSE trading session (weekend, holiday, or closure)."
        return True, None
    except Exception as e:
        return False, f"CALENDAR_OUT_OF_RANGE: Exchange calendar cannot evaluate session '{session_date.isoformat()}': {e}"


def get_expected_nyse_sessions(start_date: Union[date, str], end_date: Union[date, str]) -> List[str]:
    """Returns chronological list of official NYSE trading sessions in [start_date, end_date]."""
    if _NYSE_CALENDAR is None:
        raise RuntimeError("CALENDAR_UNAVAILABLE: Authoritative exchange calendar is unavailable.")
    d_start = date.fromisoformat(str(start_date)[:10]) if not isinstance(start_date, date) else start_date
    d_end = date.fromisoformat(str(end_date)[:10]) if not isinstance(end_date, date) else end_date
    sessions = _NYSE_CALENDAR.sessions_in_range(d_start, d_end)
    return [s.strftime("%Y-%m-%d") for s in sessions]


def inspect_existing_market_store_readonly(
    db_path: str,
    symbol: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Inspects existing candles and provenance from the database using a strict
    read-only SQLite connection (?mode=ro).
    Guarantees ZERO schema creation, ZERO table creation, ZERO WAL configuration.
    If the database file is 0 bytes or lacks required tables, returns ([], []) without modifying the file.
    """
    p = Path(db_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"DATABASE_FILE_NOT_FOUND: '{db_path}' does not exist.")

    if p.stat().st_size == 0:
        # Zero-byte file: lacks SQLite header and schema. Do NOT connect or mutate.
        return [], []

    uri = f"{p.as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('asset_ohlcv_daily', 'asset_ohlcv_provenance')"
        )
        found = {row[0] for row in cur.fetchall()}
        if "asset_ohlcv_daily" not in found:
            return [], []

        cur.execute(
            "SELECT trade_date, open, high, low, close, volume FROM asset_ohlcv_daily WHERE symbol = ? ORDER BY trade_date ASC",
            (symbol.upper().strip(),),
        )
        candles = [
            {"time": row[0], "open": row[1], "high": row[2], "low": row[3], "close": row[4], "volume": row[5]}
            for row in cur.fetchall()
        ]

        provenance_rows = []
        if "asset_ohlcv_provenance" in found:
            cur.execute(
                "SELECT trade_date, provider, adjustment_state, structural_quality FROM asset_ohlcv_provenance WHERE symbol = ? ORDER BY trade_date ASC",
                (symbol.upper().strip(),),
            )
            provenance_rows = [
                {"trade_date": row[0], "provider": row[1], "adjustment_state": row[2], "structural_quality": row[3]}
                for row in cur.fetchall()
            ]

        return candles, provenance_rows
    finally:
        conn.close()


def validate_target_database_for_write(db_path: str) -> None:
    """
    Validates that the target database exists and possesses the complete required schema
    BEFORE allowing any write infrastructure to be constructed.
    Fails closed if the database does not exist, is 0 bytes, or lacks required tables/columns.
    Guarantees zero database modifications during validation.
    """
    p = Path(db_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"TARGET_DATABASE_ABSENT: Target database '{db_path}' does not exist.")
    if p.stat().st_size == 0:
        raise ValueError(
            f"TARGET_DATABASE_EMPTY: Target database '{db_path}' is 0 bytes. "
            "Database schema must be explicitly initialized prior to write application."
        )

    uri = f"{p.as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        required = {"asset_ohlcv_daily", "asset_ohlcv_provenance"}
        if not required.issubset(tables):
            raise ValueError(
                f"TARGET_DATABASE_SCHEMA_INCOMPLETE: Missing required tables: {required - tables}. "
                "Initialize repository schema prior to write application."
            )

        cur.execute("PRAGMA table_info(asset_ohlcv_daily)")
        daily_cols = {row[1] for row in cur.fetchall()}
        req_daily = {"symbol", "trade_date", "open", "high", "low", "close", "volume"}
        if not req_daily.issubset(daily_cols):
            raise ValueError(
                f"TARGET_DATABASE_SCHEMA_INCOMPLETE: Table 'asset_ohlcv_daily' missing required columns: {req_daily - daily_cols}."
            )

        cur.execute("PRAGMA table_info(asset_ohlcv_provenance)")
        prov_cols = {row[1] for row in cur.fetchall()}
        req_prov = {
            "symbol", "trade_date", "provider", "ingestion_source", "observed_at",
            "observed_date", "observation_precision", "observation_source", "ingested_at",
            "adjustment_state", "structural_quality", "fallback_status"
        }
        if not req_prov.issubset(prov_cols):
            raise ValueError(
                f"TARGET_DATABASE_SCHEMA_INCOMPLETE: Table 'asset_ohlcv_provenance' missing required columns: {req_prov - prov_cols}."
            )
    finally:
        conn.close()


@dataclass
class ManifestScopeInfo:
    """Parsed manifest information supporting backward-compatible unpacking."""
    symbols: List[str]
    manifest_sha256: str
    scope_type: str  # PACKAGE_B | CANDIDATE_MANIFEST | SYMBOL_LIST
    per_symbol_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __iter__(self):
        """Allows `symbols, sha = load_manifest(...)` for complete backward compatibility."""
        return iter((self.symbols, self.manifest_sha256))


@dataclass
class BarValidationResult:
    """Outcome of structural and semantic validation on a returned bar series."""
    is_valid: bool
    rejection_reason: Optional[str]
    validated_bars: List[Dict[str, Any]]
    bar_count: int
    first_session: Optional[str]
    last_session: Optional[str]
    duplicate_dates: List[str] = field(default_factory=list)
    out_of_window_dates: List[str] = field(default_factory=list)
    non_finite_fields: List[str] = field(default_factory=list)
    ohlc_inconsistencies: List[str] = field(default_factory=list)
    structural_quality: str = StructuralQuality.UNKNOWN


@dataclass
class CandidateIngestionDisposition:
    """Attributable per-candidate processing disposition."""
    symbol: str
    action_type: str  # BACKFILL | REFRESH | SKIP_ALREADY_CURRENT | BLOCKED | REJECTED | UNPROCESSED
    status: str       # APPLIED | PREPARED_DRY_RUN | SKIPPED | FAILED_VALIDATION | PROVIDER_ERROR | FAILED_PAGINATION | PERSISTENCE_ERROR | UNPROCESSED
    bars_retrieved: int = 0
    bars_ingested: int = 0
    first_session: Optional[str] = None
    last_session: Optional[str] = None
    existing_sessions_in_store: int = 0
    earliest_stored_date_before: Optional[str] = None
    latest_stored_date_before: Optional[str] = None
    feed: str = "sip"
    adjustment: str = "split"
    structural_quality: str = StructuralQuality.UNKNOWN
    stored_provenance_adjustment_state: Optional[str] = None
    provenance_compatible: bool = True
    rejection_reason: Optional[str] = None
    retrieval_timestamp_utc: Optional[str] = None
    ingestion_timestamp_utc: Optional[str] = None
    pages_retrieved: int = 1
    l03_equivalence_established: bool = False
    l03_anchor_session: Optional[str] = None
    l03_observation_price: Optional[float] = None
    l03_derivation_note: Optional[str] = None


@dataclass
class IngestionBatchSummary:
    """Comprehensive summary of the bounded ingestion execution."""
    manifest_path: str
    manifest_sha256: str
    manifest_scope_type: str
    feed: str
    adjustment: str
    requested_start: str
    requested_end: str
    mode: str  # DRY_RUN | APPLIED
    batch_outcome: str  # SUCCESS | PARTIAL_SUCCESS | FAILURE
    target_db_path: Optional[str]
    total_candidates_evaluated: int = 0
    backfilled_count: int = 0
    refreshed_count: int = 0
    skipped_current_count: int = 0
    failed_validation_count: int = 0
    provider_error_count: int = 0
    total_bars_ingested: int = 0
    execution_outcome: str = "PENDING"
    completed_writes: List[str] = field(default_factory=list)
    failed_candidate: Optional[str] = None
    unprocessed_candidates: List[str] = field(default_factory=list)
    l03_bridge_status: Dict[str, Any] = field(default_factory=dict)
    l03_price_floor_basis_bridge: str = (
        "REFERENCE_DATE_EQUIVALENCE: On completed anchor session 2026-10-19 with adjustment_state='SPLIT_ADJUSTED_ONLY', "
        "split factor to present is 1.0; unadjusted price equals split-adjusted price (P_raw = P_split). "
        "Ingestion provides candidate market data evidence only; it does not certify L03 price-floor or L04 "
        "turnover compliance, and confers zero universe admission or publication authority."
    )
    effective_parameters: Dict[str, Any] = field(default_factory=dict)
    candidate_dispositions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    executed_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlpacaHistoricalSipIngestionAdapter:
    """
    Production-ready ingestion adapter for historical consolidated SIP bars via Alpaca v2.
    Interoperates with MarketDatabaseEngine without inventing parallel writers.
    """

    def __init__(
        self,
        fetcher: Optional[AlpacaMarketFetcher] = None,
        db_engine: Optional[MarketDatabaseEngine] = None,
        feed: str = "sip",
        adjustment: str = "split",
    ):
        if feed.lower() != "sip":
            raise ValueError(
                f"INVALID_FEED: Adapter strictly requires feed='sip' (consolidated tape); got '{feed}'."
            )
        if adjustment.lower() != "split":
            raise ValueError(
                f"INVALID_ADJUSTMENT: Adapter strictly requires adjustment='split'; got '{adjustment}'."
            )

        self.feed = feed.lower()
        self.adjustment = adjustment.lower()
        self.fetcher = fetcher or AlpacaMarketFetcher(feed=self.feed)
        self.db_engine = db_engine

    @staticmethod
    def load_manifest(manifest_path: Path) -> ManifestScopeInfo:
        """Loads symbols and parameters from a candidate or Package B manifest and computes its SHA-256."""
        p = Path(manifest_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"MANIFEST_NOT_FOUND: Manifest path '{manifest_path}' does not exist.")

        raw_bytes = p.read_bytes()
        manifest_sha = hashlib.sha256(raw_bytes).hexdigest()
        data = json.loads(raw_bytes.decode("utf-8"))

        symbols: List[str] = []
        scope_type = "UNKNOWN"
        per_symbol_parameters: Dict[str, Dict[str, Any]] = {}

        if "ingestion_scope" in data:
            # Package B structure
            scope_type = "PACKAGE_B"
            scope = data["ingestion_scope"]

            # 1. Process production_target_refresh first
            refresh_items = scope.get("production_target_refresh", {}).get("per_symbol_parameters", [])
            for item in refresh_items:
                if isinstance(item, dict):
                    sym = item.get("symbol", "").upper().strip()
                    if sym:
                        symbols.append(sym)
                        per_symbol_parameters[sym] = dict(item)

            # 2. Process production_target_backfill
            backfill = scope.get("production_target_backfill", {}).get("missing_symbols", [])
            for sym in backfill:
                clean_sym = sym.upper().strip()
                symbols.append(clean_sym)
                if clean_sym not in per_symbol_parameters:
                    per_symbol_parameters[clean_sym] = {"action": "BACKFILL"}

            # 3. Process candidate_bindings with strict precedence (e.g. action: REPLACE / BACKFILL)
            candidate_bindings = scope.get("candidate_bindings", {})
            for sym, binding in candidate_bindings.items():
                if isinstance(binding, dict):
                    clean_sym = sym.upper().strip()
                    if clean_sym.startswith("SEC_") or not re.match(r"^[A-Za-z]{1,5}$", clean_sym):
                        raise ValueError(f"NONCANONICAL_IDENTITY_ALIAS_REJECTED: Symbol '{clean_sym}' is a noncanonical alias.")
                    sec_id = binding.get("internal_security_id") or binding.get("durable_security_id") or binding.get("security_id")
                    if sec_id and not re.match(r"^ARX_SEC_[0-9A-F]{12}$", sec_id):
                        raise ValueError(f"INVALID_CANONICAL_SECURITY_ID: Security ID '{sec_id}' does not match canonical format ARX_SEC_<12_HEX>.")
                    assoc_id = binding.get("association_id") or binding.get("listing_association_id")
                    if assoc_id and not re.match(r"^ASSOC_ARX_SEC_[0-9A-F]{12}_[A-Z]{4}_[A-Z0-9]+_\d{4}-\d{2}-\d{2}$", assoc_id):
                        raise ValueError(f"INVALID_CANONICAL_ASSOCIATION_ID: Association ID '{assoc_id}' does not match canonical format ASSOC_ARX_SEC_...")
                    symbols.append(clean_sym)
                    if clean_sym in per_symbol_parameters:
                        merged = dict(per_symbol_parameters[clean_sym])
                        merged.update(binding)
                        action = binding.get("action") or binding.get("mandated_action")
                        if action:
                            merged["action"] = str(action).upper().strip()
                        per_symbol_parameters[clean_sym] = merged
                    else:
                        per_symbol_parameters[clean_sym] = dict(binding)

            symbols = sorted(list(set(symbols)))
        elif "pilot_candidates" in data or "candidates" in data:
            c_key = "pilot_candidates" if "pilot_candidates" in data else "candidates"
            scope_type = "PILOT_MANIFEST" if c_key == "pilot_candidates" else "CANDIDATE_MANIFEST"
            for c in data[c_key]:
                if not isinstance(c, dict):
                    continue
                raw_sym = c.get("symbol", "").strip()
                if not raw_sym:
                    continue
                if raw_sym.upper().startswith("SEC_") or not re.match(r"^[A-Za-z]{1,5}$", raw_sym):
                    raise ValueError(f"NONCANONICAL_IDENTITY_ALIAS_REJECTED: Symbol '{raw_sym}' is a noncanonical alias.")

                clean_sym = raw_sym.upper()
                sec_id = c.get("durable_security_id") or c.get("security_id")
                if sec_id and not re.match(r"^ARX_SEC_[0-9A-F]{12}$", sec_id):
                    raise ValueError(f"INVALID_CANONICAL_SECURITY_ID: Security ID '{sec_id}' does not match canonical format ARX_SEC_<12_HEX>.")

                assoc_id = c.get("listing_association_id") or c.get("association_id")
                if assoc_id and not re.match(r"^ASSOC_ARX_SEC_[0-9A-F]{12}_[A-Z]{4}_[A-Z0-9]+_\d{4}-\d{2}-\d{2}$", assoc_id):
                    raise ValueError(f"INVALID_CANONICAL_ASSOCIATION_ID: Association ID '{assoc_id}' does not match canonical format ASSOC_ARX_SEC_...")

                symbols.append(clean_sym)
                per_symbol_parameters[clean_sym] = c
            symbols = sorted(list(set(symbols)))
        elif isinstance(data, list):
            # Plain list of symbols or dicts
            scope_type = "SYMBOL_LIST"
            for item in data:
                if isinstance(item, str):
                    clean = item.strip()
                    if clean.upper().startswith("SEC_") or not re.match(r"^[A-Za-z]{1,5}$", clean):
                        raise ValueError(f"NONCANONICAL_IDENTITY_ALIAS_REJECTED: Symbol '{clean}' is a noncanonical alias.")
                    symbols.append(clean.upper())
                elif isinstance(item, dict) and "symbol" in item:
                    raw_sym = item["symbol"].strip()
                    if raw_sym.upper().startswith("SEC_") or not re.match(r"^[A-Za-z]{1,5}$", raw_sym):
                        raise ValueError(f"NONCANONICAL_IDENTITY_ALIAS_REJECTED: Symbol '{raw_sym}' is a noncanonical alias.")
                    symbols.append(raw_sym.upper())
            symbols = sorted(list(set(symbols)))
        else:
            raise ValueError(f"UNRECOGNIZED_MANIFEST_FORMAT: Could not extract symbols from '{manifest_path}'.")

        return ManifestScopeInfo(
            symbols=symbols,
            manifest_sha256=manifest_sha,
            scope_type=scope_type,
            per_symbol_parameters=per_symbol_parameters,
        )

    @staticmethod
    def is_prov_compatible(
        existing_provenance: List[Dict[str, Any]],
        existing_candles: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether existing stored provenance is compatible with Alpaca SIP ingestion.
        Enforces:
        1. Pure Alpaca lineage: Provider must match ALPACA. No mixing with MASSIVE_POLYGON or other sources.
        2. Adjustment fidelity: Must be SPLIT_ADJUSTED_ONLY.
        3. Zero missing sidecars: Every candle must have a corresponding provenance row.
        Returns: (is_compatible: bool, rejection_reason: Optional[str])
        """
        if not existing_provenance:
            return False, "UNPROVENANCED: No existing provenance records found."

        if existing_candles is not None:
            if len(existing_provenance) != len(existing_candles):
                return False, f"ROW_COUNT_MISMATCH: {len(existing_provenance)} provenance rows != {len(existing_candles)} candles."
            prov_by_date = {p.get("trade_date"): p for p in existing_provenance}
            missing_dates = [c.get("time") for c in existing_candles if c.get("time") not in prov_by_date]
            if missing_dates:
                return False, f"UNPROVENANCED_CANDLES: {len(missing_dates)} candles lack provenance sidecars."

        non_matching_provider = [
            p.get("trade_date") for p in existing_provenance
            if str(p.get("provider", "")).upper() not in ("ALPACA", str(Provider.ALPACA).upper())
        ]
        if non_matching_provider:
            providers = sorted(list({str(p.get("provider", "")) for p in existing_provenance}))
            return False, f"INCOMPATIBLE_PROVIDER: Stored lineage has provider(s) {providers}; Alpaca SIP forbids provider blending."

        non_split = [
            p.get("trade_date") for p in existing_provenance
            if p.get("adjustment_state") != AdjustmentState.SPLIT_ADJUSTED_ONLY
        ]
        if non_split:
            return False, f"INCOMPATIBLE_ADJUSTMENT: {len(non_split)} records are not SPLIT_ADJUSTED_ONLY."

        return True, None

    @staticmethod
    def validate_bar_series(
        symbol: str,
        bars: List[Dict[str, Any]],
        start_date: str,
        end_date: str,
    ) -> BarValidationResult:
        """
        Validates individual bars and overall series consistency per Package B
        bar_validation_specification.
        Enforces true ISO timestamp parsing, NYSE trading session verification,
        strict integer non-negative volume, strictly positive finite prices,
        and High/Low geometric boundaries.
        """
        if not bars:
            return BarValidationResult(
                is_valid=False,
                rejection_reason="EMPTY_BAR_SERIES: Provider returned zero bars.",
                validated_bars=[],
                bar_count=0,
                first_session=None,
                last_session=None,
                structural_quality=StructuralQuality.UNKNOWN,
            )

        validated: List[Dict[str, Any]] = []
        seen_dates: Set[str] = set()
        duplicate_dates: List[str] = []
        out_of_window: List[str] = []
        non_finite: List[str] = []
        ohlc_inconsistencies: List[str] = []

        try:
            start_d = date.fromisoformat(str(start_date)[:10])
            end_d = date.fromisoformat(str(end_date)[:10])
        except Exception as e:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=f"INVALID_BOUND_DATES: Could not parse bounds '{start_date}' or '{end_date}': {e}",
                validated_bars=[],
                bar_count=len(bars),
                first_session=None,
                last_session=None,
                structural_quality=StructuralQuality.CORRUPT,
            )

        if end_d > MAX_COMPLETED_SESSION:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=(
                    f"FUTURE_OR_INCOMPLETE_SESSION_BOUND: Requested end date '{end_d.isoformat()}' "
                    f"exceeds maximum allowed completed session '{MAX_COMPLETED_SESSION.isoformat()}'."
                ),
                validated_bars=[],
                bar_count=len(bars),
                first_session=None,
                last_session=None,
                structural_quality=StructuralQuality.CORRUPT,
            )

        for idx, bar in enumerate(bars):
            if not isinstance(bar, dict):
                return BarValidationResult(
                    is_valid=False,
                    rejection_reason=f"MALFORMED_BAR_AT_INDEX_{idx}: Expected dict, got {type(bar).__name__}.",
                    validated_bars=[],
                    bar_count=len(bars),
                    first_session=None,
                    last_session=None,
                    structural_quality=StructuralQuality.CORRUPT,
                )

            # Check bar symbol if present
            bar_sym = bar.get("S") or bar.get("symbol")
            if bar_sym and str(bar_sym).upper().strip() != symbol.upper().strip():
                return BarValidationResult(
                    is_valid=False,
                    rejection_reason=(
                        f"SYMBOL_IDENTITY_MISMATCH_AT_INDEX_{idx}: Bar symbol '{bar_sym}' "
                        f"does not match expected candidate '{symbol}'."
                    ),
                    validated_bars=[],
                    bar_count=len(bars),
                    first_session=None,
                    last_session=None,
                    structural_quality=StructuralQuality.CORRUPT,
                )

            # Extract timestamp/date with true ISO parsing
            raw_t = bar.get("t") or bar.get("time") or bar.get("trade_date") or bar.get("date")
            if not raw_t:
                return BarValidationResult(
                    is_valid=False,
                    rejection_reason=f"MISSING_TIMESTAMP_AT_INDEX_{idx}",
                    validated_bars=[],
                    bar_count=len(bars),
                    first_session=None,
                    last_session=None,
                    structural_quality=StructuralQuality.CORRUPT,
                )

            try:
                clean_t = str(raw_t).strip().replace("Z", "+00:00")
                parsed_dt = datetime.fromisoformat(clean_t)
                session_d = parsed_dt.date()
                date_str = session_d.isoformat()
            except Exception as e:
                return BarValidationResult(
                    is_valid=False,
                    rejection_reason=f"INVALID_ISO_TIMESTAMP_AT_INDEX_{idx}: '{raw_t}': {e}",
                    validated_bars=[],
                    bar_count=len(bars),
                    first_session=None,
                    last_session=None,
                    structural_quality=StructuralQuality.CORRUPT,
                )

            # Window check: check against maximum completed session and requested date range
            if session_d > MAX_COMPLETED_SESSION:
                out_of_window.append(f"{date_str} exceeds max completed session {MAX_COMPLETED_SESSION.isoformat()}")
            elif session_d > end_d:
                out_of_window.append(f"{date_str} > {end_d.isoformat()}")
            elif session_d < start_d:
                out_of_window.append(f"{date_str} < {start_d.isoformat()}")
            else:
                # Validate date against official NYSE exchange calendar
                is_valid_sess, sess_reason = is_nyse_trading_session(session_d)
                if not is_valid_sess:
                    return BarValidationResult(
                        is_valid=False,
                        rejection_reason=f"{sess_reason}_AT_INDEX_{idx}",
                        validated_bars=[],
                        bar_count=len(bars),
                        first_session=None,
                        last_session=None,
                        structural_quality=StructuralQuality.CORRUPT,
                    )

            # Duplicate check
            if date_str in seen_dates:
                duplicate_dates.append(date_str)
            seen_dates.add(date_str)

            # Volume validation (SIP-02: Volume must be present, never defaulted to 0, never truncated)
            if "v" not in bar and "volume" not in bar:
                return BarValidationResult(
                    is_valid=False,
                    rejection_reason=f"MISSING_VOLUME_AT_INDEX_{idx}: Bar missing required volume field.",
                    validated_bars=[],
                    bar_count=len(bars),
                    first_session=None,
                    last_session=None,
                    structural_quality=StructuralQuality.CORRUPT,
                )
            raw_v = bar.get("v") if "v" in bar else bar.get("volume")
            if raw_v is None:
                return BarValidationResult(
                    is_valid=False,
                    rejection_reason=f"NULL_VOLUME_AT_INDEX_{idx}: Volume cannot be null.",
                    validated_bars=[],
                    bar_count=len(bars),
                    first_session=None,
                    last_session=None,
                    structural_quality=StructuralQuality.CORRUPT,
                )

            try:
                float_v = float(raw_v)
            except (ValueError, TypeError) as e:
                non_finite.append(f"volume_parse_error_{idx}: {e}")
                continue

            if math.isnan(float_v) or math.isinf(float_v) or float_v < 0:
                non_finite.append(f"volume={float_v}_at_{date_str}")

            if not float_v.is_integer():
                return BarValidationResult(
                    is_valid=False,
                    rejection_reason=(
                        f"NON_INTEGER_VOLUME_AT_INDEX_{idx}: Volume {float_v} must be an integer share count; "
                        "silent truncation is strictly prohibited."
                    ),
                    validated_bars=[],
                    bar_count=len(bars),
                    first_session=None,
                    last_session=None,
                    structural_quality=StructuralQuality.CORRUPT,
                )
            v_int = int(float_v)

            # Numeric price validation
            try:
                o = float(bar.get("o", bar.get("open", 0.0)))
                h = float(bar.get("h", bar.get("high", 0.0)))
                l = float(bar.get("l", bar.get("low", 0.0)))
                c = float(bar.get("c", bar.get("close", 0.0)))
            except (ValueError, TypeError) as e:
                non_finite.append(f"bar_{idx}_price_parse_error: {e}")
                continue

            for field_name, val in [("open", o), ("high", h), ("low", l), ("close", c)]:
                if math.isnan(val) or math.isinf(val) or val <= 0:
                    non_finite.append(f"{field_name}={val}_at_{date_str}")

            # Geometric consistency
            if h < l:
                ohlc_inconsistencies.append(f"high_{h} < low_{l} at {date_str}")
            if h < o:
                ohlc_inconsistencies.append(f"high_{h} < open_{o} at {date_str}")
            if h < c:
                ohlc_inconsistencies.append(f"high_{h} < close_{c} at {date_str}")
            if l > o:
                ohlc_inconsistencies.append(f"low_{l} > open_{o} at {date_str}")
            if l > c:
                ohlc_inconsistencies.append(f"low_{l} > close_{c} at {date_str}")

            validated.append({
                "time": date_str,
                "open": round(o, 4),
                "high": round(h, 4),
                "low": round(l, 4),
                "close": round(c, 4),
                "volume": v_int,
            })

        # Evaluate validation failures
        if out_of_window:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=f"OUT_OF_WINDOW_SESSIONS: {out_of_window[:3]}",
                validated_bars=[],
                bar_count=len(bars),
                first_session=validated[0]["time"] if validated else None,
                last_session=validated[-1]["time"] if validated else None,
                out_of_window_dates=out_of_window,
                structural_quality=StructuralQuality.CORRUPT,
            )

        if duplicate_dates:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=f"DUPLICATE_SESSION_DATES: {duplicate_dates[:3]}",
                validated_bars=[],
                bar_count=len(bars),
                first_session=validated[0]["time"] if validated else None,
                last_session=validated[-1]["time"] if validated else None,
                duplicate_dates=duplicate_dates,
                structural_quality=StructuralQuality.CORRUPT,
            )

        if non_finite:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=f"NON_FINITE_OR_INVALID_NUMERIC_VALUES: {non_finite[:3]}",
                validated_bars=[],
                bar_count=len(bars),
                first_session=validated[0]["time"] if validated else None,
                last_session=validated[-1]["time"] if validated else None,
                non_finite_fields=non_finite,
                structural_quality=StructuralQuality.CORRUPT,
            )

        if ohlc_inconsistencies:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=f"OHLC_GEOMETRIC_INCONSISTENCIES: {ohlc_inconsistencies[:3]}",
                validated_bars=[],
                bar_count=len(bars),
                first_session=validated[0]["time"] if validated else None,
                last_session=validated[-1]["time"] if validated else None,
                ohlc_inconsistencies=ohlc_inconsistencies,
                structural_quality=StructuralQuality.CORRUPT,
            )

        if not validated:
            return BarValidationResult(
                is_valid=False,
                rejection_reason="ZERO_VALIDATED_BARS: Bar series contains zero valid trading sessions.",
                validated_bars=[],
                bar_count=len(bars),
                first_session=None,
                last_session=None,
                structural_quality=StructuralQuality.CORRUPT,
            )

        # R1: Check NYSE trading calendar and complete session acquisition coverage
        latest_session = validated[-1]["time"]
        first_session = validated[0]["time"]
        try:
            exp_sessions = get_expected_nyse_sessions(start_date, end_date)
        except Exception as cal_err:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=f"CALENDAR_UNAVAILABLE: Cannot verify acquisition completeness: {cal_err}",
                validated_bars=[],
                bar_count=len(bars),
                first_session=first_session,
                last_session=latest_session,
                structural_quality=StructuralQuality.CORRUPT,
            )

        if not exp_sessions:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=f"ZERO_EXPECTED_SESSIONS: Calendar resolved zero trading sessions in [{start_date}, {end_date}].",
                validated_bars=[],
                bar_count=len(bars),
                first_session=first_session,
                last_session=latest_session,
                structural_quality=StructuralQuality.CORRUPT,
            )

        # 1. Target endpoint session acquisition check
        if latest_session != exp_sessions[-1]:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=(
                    f"INCOMPLETE_ACQUISITION_MISSING_ENDPOINT: Requested target endpoint was '{exp_sessions[-1]}', "
                    f"but latest returned session was '{latest_session}'."
                ),
                validated_bars=[],
                bar_count=len(bars),
                first_session=first_session,
                last_session=latest_session,
                structural_quality=StructuralQuality.CORRUPT,
            )

        # 2. Window start / truncated history check (R1)
        if first_session > exp_sessions[0]:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=(
                    f"INCOMPLETE_ACQUISITION_TRUNCATED_HISTORY: Requested acquisition from '{exp_sessions[0]}' "
                    f"({len(exp_sessions)} expected sessions), but series starts at '{first_session}' "
                    f"({len(validated)} returned sessions). Missing {len(exp_sessions) - len(validated)} sessions."
                ),
                validated_bars=[],
                bar_count=len(bars),
                first_session=first_session,
                last_session=latest_session,
                structural_quality=StructuralQuality.CORRUPT,
            )

        # 3. Interior session gaps check (R1)
        returned_dates_set = {b["time"] for b in validated}
        interior_expected = {s for s in exp_sessions if first_session <= s <= latest_session}
        interior_gaps = interior_expected - returned_dates_set
        if interior_gaps:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=(
                    f"INCOMPLETE_ACQUISITION_INTERIOR_GAPS: Series has {len(interior_gaps)} missing interior sessions "
                    f"between '{first_session}' and '{latest_session}': {sorted(list(interior_gaps))[:5]}."
                ),
                validated_bars=[],
                bar_count=len(bars),
                first_session=first_session,
                last_session=latest_session,
                structural_quality=StructuralQuality.CORRUPT,
            )

        # 4. Coverage deficit check (R1)
        expected_sessions_set = set(exp_sessions)
        missing_sessions = expected_sessions_set - returned_dates_set
        if missing_sessions:
            return BarValidationResult(
                is_valid=False,
                rejection_reason=(
                    f"INCOMPLETE_ACQUISITION_COVERAGE_DEFICIT: Series lacks {len(missing_sessions)} expected sessions "
                    f"in requested window [{start_date}, {end_date}]: {sorted(list(missing_sessions))[:5]}."
                ),
                validated_bars=[],
                bar_count=len(bars),
                first_session=first_session,
                last_session=latest_session,
                structural_quality=StructuralQuality.CORRUPT,
            )

        quality = assess_structural_quality(validated)
        return BarValidationResult(
            is_valid=True,
            rejection_reason=None,
            validated_bars=validated,
            bar_count=len(validated),
            first_session=validated[0]["time"] if validated else None,
            last_session=validated[-1]["time"] if validated else None,
            structural_quality=quality,
        )

    def fetch_all_candidate_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[bool, List[Dict[str, Any]], int, Optional[str]]:
        """
        Fetches historical bars for a candidate across all pages until pagination
        is complete. Enforces cycle detection, symbol identity verification, and
        fails closed if max_pages is exhausted while next_page_token remains.
        """
        all_bars: List[Dict[str, Any]] = []
        next_page_token: Optional[str] = None
        seen_tokens: Set[str] = set()
        pages_retrieved = 0
        max_pages = 25  # Bounded pagination safety threshold

        while pages_retrieved < max_pages:
            res = self.fetcher.fetch_historical_bars(
                symbol=symbol,
                start=start_date,
                end=end_date,
                feed=self.feed,
                adjustment=self.adjustment,
                limit=1000,
                page_token=next_page_token,
            )
            pages_retrieved += 1

            if not res.get("success"):
                return False, [], pages_retrieved, f"PROVIDER_ERROR: {res.get('error', 'UNKNOWN_FETCH_ERROR')}"

            # Identity validation: ensure returned symbol matches requested candidate
            ret_sym = res.get("symbol")
            if ret_sym and ret_sym.upper().strip() != symbol.upper().strip():
                return False, [], pages_retrieved, (
                    f"SYMBOL_IDENTITY_MISMATCH: Requested candidate '{symbol}', "
                    f"provider returned response for '{ret_sym}'."
                )

            bars = res.get("bars")
            if not isinstance(bars, list):
                return False, [], pages_retrieved, f"MALFORMED_PAGE: 'bars' is not a list (got {type(bars).__name__})."

            all_bars.extend(bars)

            next_page_token = res.get("next_page_token")
            if not next_page_token:
                break

            if next_page_token in seen_tokens:
                return False, [], pages_retrieved, f"PAGINATION_CYCLE_DETECTED: Repeated page_token '{next_page_token}'."
            seen_tokens.add(next_page_token)

        # SIP-03: Reaching max_pages with a remaining token is an INCOMPLETE acquisition
        if next_page_token:
            return False, [], pages_retrieved, (
                f"PAGINATION_BUDGET_EXHAUSTED: Exceeded {max_pages} pages with remaining page_token '{next_page_token}'. "
                "Acquisition cannot be certified complete."
            )

        return True, all_bars, pages_retrieved, None

    def execute_bounded_ingestion(
        self,
        manifest_path: Path,
        symbols_override: Optional[List[str]] = None,
        start_date: str = "2025-09-15",
        end_date: str = "2026-10-19",
        apply_writes: bool = False,
        db_path: Optional[str] = None,
        max_candidates: Optional[int] = None,
    ) -> IngestionBatchSummary:
        """
        Executes bounded ingestion workflow across candidates.
        Enforces strict read-only dry-run, target database schema pre-validation on apply,
        candidate symbol validation against manifest, adjustment compatibility inspection,
        and honest batch outcome calculation.
        """
        # Validate dates against maximum completed session
        try:
            d_start = date.fromisoformat(str(start_date)[:10])
            d_end = date.fromisoformat(str(end_date)[:10])
        except Exception as e:
            raise ValueError(f"INVALID_REQUEST_DATES: Could not parse dates: {e}")

        if d_end > MAX_COMPLETED_SESSION:
            raise ValueError(
                f"INVALID_END_DATE: Requested end date '{end_date}' exceeds maximum completed session "
                f"'{MAX_COMPLETED_SESSION.isoformat()}'."
            )
        if d_start > d_end:
            raise ValueError(f"INVALID_DATE_ORDER: start_date '{start_date}' cannot be after end_date '{end_date}'.")

        # SIP-01 & SIP-05: Load and validate manifest & requested scope BEFORE touching write infrastructure
        manifest_info = self.load_manifest(manifest_path)
        manifest_symbols_set = set(manifest_info.symbols)

        # SIP-05: Reject requested symbols absent from the manifest
        if symbols_override:
            upper_req = [s.upper().strip() for s in symbols_override]
            missing_syms = [s for s in upper_req if s not in manifest_symbols_set]
            if missing_syms:
                raise ValueError(
                    f"REQUESTED_SYMBOLS_ABSENT_FROM_MANIFEST: Symbols {missing_syms} are not present "
                    f"in manifest '{manifest_path}'."
                )
            target_symbols = [s for s in manifest_info.symbols if s in set(upper_req)]
        else:
            target_symbols = list(manifest_info.symbols)

        if not target_symbols:
            raise ValueError("EMPTY_EFFECTIVE_SCOPE: Zero candidates selected for ingestion.")

        if max_candidates and max_candidates > 0:
            target_symbols = target_symbols[:max_candidates]

        # SIP-01: In apply mode, validate target database schema exists in ?mode=ro BEFORE constructing write engine
        # and construct with auto_init=False to guarantee zero table migrations/creations
        if apply_writes:
            if not db_path:
                raise ValueError("MISSING_DB_PATH: --db-path is required when --apply is specified.")
            validate_target_database_for_write(db_path)
            if self.db_engine is None or self.db_engine.db_path != str(Path(db_path).resolve()):
                self.db_engine = MarketDatabaseEngine(db_path=str(Path(db_path).resolve()), auto_init=False)

        summary = IngestionBatchSummary(
            manifest_path=str(manifest_path),
            manifest_sha256=manifest_info.manifest_sha256,
            manifest_scope_type=manifest_info.scope_type,
            feed=self.feed,
            adjustment=self.adjustment,
            requested_start=start_date,
            requested_end=end_date,
            mode="APPLIED" if apply_writes else "DRY_RUN",
            execution_outcome="PENDING",
            batch_outcome="FAILURE",  # Initialized to FAILURE; updated at end
            target_db_path=db_path if apply_writes else None,
            total_candidates_evaluated=len(target_symbols),
            effective_parameters={
                "feed": self.feed,
                "adjustment": self.adjustment,
                "start_date": start_date,
                "end_date": end_date,
                "max_candidates": max_candidates,
                "apply_writes": apply_writes,
            },
        )

        try:
            expected_window_sessions = get_expected_nyse_sessions(start_date, end_date)
            calendar_available = bool(expected_window_sessions)
        except Exception as e:
            expected_window_sessions = []
            calendar_available = False
            logger.warning(f"Authoritative calendar unavailable: {e}")
        expected_sessions_set = set(expected_window_sessions)

        for sym in target_symbols:
            # SIP-01 & SIP-04: Inspect existing state using strict read-only inspection
            existing_candles: List[Dict[str, Any]] = []
            existing_provenance: List[Dict[str, Any]] = []
            if db_path and Path(db_path).is_file():
                existing_candles, existing_provenance = inspect_existing_market_store_readonly(db_path, sym)

            existing_count = len(existing_candles)
            latest_stored = existing_candles[-1]["time"] if existing_candles else None
            earliest_stored = existing_candles[0]["time"] if existing_candles else None
            stored_dates_set = {c["time"] for c in existing_candles}

            # R3: Inspect 100% of stored candles and provenance records for 1-to-1 correspondence and purity
            is_prov_compatible, prov_incompat_reason = self.is_prov_compatible(
                existing_provenance, existing_candles
            ) if existing_count > 0 else (True, None)
            primary_stored_adj = existing_provenance[-1].get("adjustment_state") if existing_provenance else None

            # R3: Calendar authority fail-closed check
            if not calendar_available:
                reason = "CALENDAR_AUTHORITY_UNAVAILABLE: Authoritative exchange calendar unavailable; cannot certify stored coverage or execute ingestion."
                disp = CandidateIngestionDisposition(
                    symbol=sym,
                    action_type="BLOCKED",
                    status="FAILED_VALIDATION",
                    rejection_reason=reason,
                    existing_sessions_in_store=existing_count,
                    earliest_stored_date_before=earliest_stored,
                    latest_stored_date_before=latest_stored,
                    feed=self.feed,
                    adjustment=self.adjustment,
                    stored_provenance_adjustment_state=primary_stored_adj,
                    provenance_compatible=is_prov_compatible,
                )
                summary.candidate_dispositions[sym] = asdict(disp)
                summary.failed_validation_count += 1
                continue

            # R3 & L03: Determine whether truly already current
            # Must satisfy: calendar authority available AND compatible provenance AND all expected sessions present without gaps
            is_already_current = False
            if (
                is_prov_compatible
                and existing_count > 0
                and latest_stored is not None
                and latest_stored >= end_date
                and earliest_stored is not None
                and earliest_stored <= start_date
                and expected_sessions_set
            ):
                missing_sessions = expected_sessions_set - stored_dates_set
                if not missing_sessions:
                    is_already_current = True

            if is_already_current:
                anchor_bar = next((c for c in existing_candles if c["time"] == "2026-10-19"), None)
                has_anchor = anchor_bar is not None
                obs_close = float(anchor_bar["close"]) if has_anchor else None
                disp = CandidateIngestionDisposition(
                    symbol=sym,
                    action_type="SKIP_ALREADY_CURRENT",
                    status="SKIPPED",
                    bars_retrieved=0,
                    bars_ingested=0,
                    first_session=earliest_stored,
                    last_session=latest_stored,
                    existing_sessions_in_store=existing_count,
                    earliest_stored_date_before=earliest_stored,
                    latest_stored_date_before=latest_stored,
                    feed=self.feed,
                    adjustment=self.adjustment,
                    stored_provenance_adjustment_state=primary_stored_adj,
                    provenance_compatible=is_prov_compatible,
                    l03_equivalence_established=False,  # Conditional: candidate market data evidence only
                    l03_anchor_session="2026-10-19" if has_anchor else None,
                    l03_observation_price=obs_close,
                    l03_derivation_note=(
                        f"Candidate {sym} has verified split-adjusted bar on reference anchor session 2026-10-19 (close={obs_close}). "
                        "Equivalence is conditional: factor 1.0 assumed without corporate action verification. "
                        "Candidate market data evidence only; does not certify L03/L04 compliance."
                        if has_anchor else
                        f"Anchor session 2026-10-19 not present in store for {sym}; L03 price-floor equivalence cannot be established."
                    ),
                )
                summary.skipped_current_count += 1
                summary.candidate_dispositions[sym] = asdict(disp)
                continue

            # Determine action type and query start date
            sym_start = start_date
            action = "BACKFILL"

            # Defect A: Check if candidate can be refreshed incrementally
            # 1. Existing stored history must extend back to start_date
            # 2. Existing stored history must have latest_stored < end_date
            # 3. Provenance must be 100% compatible
            # 4. Retained prefix [start_date, latest_stored] MUST be complete without interior gaps
            prefix_expected_sessions = [s for s in expected_window_sessions if start_date <= s <= latest_stored] if latest_stored else []
            prefix_missing = set(prefix_expected_sessions) - stored_dates_set

            manifest_param = manifest_info.per_symbol_parameters.get(sym, {})
            manifest_action = str(manifest_param.get("action") or manifest_param.get("mandated_action") or "").upper().strip()

            can_refresh = (
                manifest_action not in ("BACKFILL", "REPLACE")
                and existing_count > 0
                and latest_stored is not None
                and latest_stored < end_date
                and earliest_stored is not None
                and earliest_stored <= start_date
                and is_prov_compatible
                and len(prefix_missing) == 0
            )

            if manifest_action == "REPLACE":
                action = "REPLACE"
                sym_start = start_date
            elif can_refresh and latest_stored:
                try:
                    last_d = date.fromisoformat(latest_stored)
                    next_d = date.fromordinal(last_d.toordinal() + 1).isoformat()
                    manifest_start = manifest_param.get("requested_start_session")
                    if manifest_start and manifest_start > next_d:
                        try:
                            skipped = [s for s in get_expected_nyse_sessions(next_d, manifest_start) if s < manifest_start]
                        except Exception:
                            skipped = []
                        if skipped:
                            action = "BACKFILL"
                            sym_start = start_date
                        else:
                            sym_start = manifest_start
                            action = "REFRESH"
                    else:
                        sym_start = next_d
                        action = "REFRESH"
                except Exception:
                    sym_start = start_date
                    action = "BACKFILL"
            else:
                action = "BACKFILL"
                sym_start = start_date

            # R2 & Scope-edge: Check if existing history outside requested scope has incompatible provenance or missing sidecars
            if existing_count > 0:
                outside_candles = [c for c in existing_candles if c["time"] < sym_start or c["time"] > end_date]
                outside_prov = [p for p in existing_provenance if p.get("trade_date") < sym_start or p.get("trade_date") > end_date]
                outside_prov_dates = {p.get("trade_date") for p in outside_prov}
                outside_unprovenanced = [c["time"] for c in outside_candles if c["time"] not in outside_prov_dates]
                incompatible_outside = [
                    p for p in outside_prov
                    if p.get("adjustment_state") != AdjustmentState.SPLIT_ADJUSTED_ONLY
                    or str(p.get("provider", "")).upper() not in ("ALPACA", str(Provider.ALPACA).upper())
                ]
                if incompatible_outside or outside_unprovenanced:
                    reason = (
                        f"INCOMPATIBLE_HISTORY_OUTSIDE_REVIEWED_SCOPE: Candidate '{sym}' has "
                        f"{len(incompatible_outside)} incompatible and {len(outside_unprovenanced)} unprovenanced "
                        f"records outside reviewed window [{sym_start}, {end_date}]. Bounded replacement cannot proceed "
                        f"without explicit wider repair from '{earliest_stored}'."
                    )
                    logger.error(reason)
                    disp = CandidateIngestionDisposition(
                        symbol=sym,
                        action_type=action,
                        status="BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE",
                        rejection_reason=reason,
                        bars_retrieved=0,
                        bars_ingested=0,
                        first_session=earliest_stored,
                        last_session=latest_stored,
                        existing_sessions_in_store=existing_count,
                        earliest_stored_date_before=earliest_stored,
                        latest_stored_date_before=latest_stored,
                        feed=self.feed,
                        adjustment=self.adjustment,
                        stored_provenance_adjustment_state=primary_stored_adj,
                        provenance_compatible=False,
                    )
                    summary.candidate_dispositions[sym] = asdict(disp)
                    summary.failed_validation_count += 1
                    continue

            retrieval_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            fetch_ok, raw_bars, pages_count, err_msg = self.fetch_all_candidate_bars(sym, sym_start, end_date)

            if not fetch_ok:
                err_str = str(err_msg)
                err_status = "FAILED_PAGINATION" if "PAGINATION" in err_str else "PROVIDER_ERROR"
                disp = CandidateIngestionDisposition(
                    symbol=sym,
                    action_type=action,
                    status=err_status,
                    bars_retrieved=0,
                    bars_ingested=0,
                    existing_sessions_in_store=existing_count,
                    earliest_stored_date_before=earliest_stored,
                    latest_stored_date_before=latest_stored,
                    feed=self.feed,
                    adjustment=self.adjustment,
                    stored_provenance_adjustment_state=primary_stored_adj,
                    provenance_compatible=is_prov_compatible,
                    rejection_reason=err_str,
                    retrieval_timestamp_utc=retrieval_utc,
                    pages_retrieved=pages_count,
                )
                summary.provider_error_count += 1
                summary.candidate_dispositions[sym] = asdict(disp)
                continue

            # Validate bar series
            val_res = self.validate_bar_series(sym, raw_bars, sym_start, end_date)
            if not val_res.is_valid:
                disp = CandidateIngestionDisposition(
                    symbol=sym,
                    action_type=action,
                    status="FAILED_VALIDATION",
                    bars_retrieved=len(raw_bars),
                    bars_ingested=0,
                    existing_sessions_in_store=existing_count,
                    earliest_stored_date_before=earliest_stored,
                    latest_stored_date_before=latest_stored,
                    feed=self.feed,
                    adjustment=self.adjustment,
                    structural_quality=val_res.structural_quality,
                    stored_provenance_adjustment_state=primary_stored_adj,
                    provenance_compatible=is_prov_compatible,
                    rejection_reason=val_res.rejection_reason,
                    retrieval_timestamp_utc=retrieval_utc,
                    pages_retrieved=pages_count,
                )
                summary.failed_validation_count += 1
                summary.candidate_dispositions[sym] = asdict(disp)
                continue

            # Defect A: Validate that the final union of retained and newly acquired observations
            # covers 100% of the requested contract [start_date, end_date] before applying or certifying success
            combined_dates_set = (
                (stored_dates_set | {b["time"] for b in val_res.validated_bars})
                if action == "REFRESH" else {b["time"] for b in val_res.validated_bars}
            )
            final_missing = expected_sessions_set - combined_dates_set
            if final_missing:
                reason = (
                    f"INCOMPLETE_FINAL_STORE_WINDOW: Requested window [{start_date}, {end_date}] "
                    f"for candidate '{sym}' remains incomplete ({len(final_missing)} missing sessions): "
                    f"{sorted(list(final_missing))[:5]}. Cannot certify full-window readiness."
                )
                logger.error(reason)
                disp = CandidateIngestionDisposition(
                    symbol=sym,
                    action_type=action,
                    status="FAILED_VALIDATION",
                    bars_retrieved=len(raw_bars),
                    bars_ingested=0,
                    first_session=val_res.first_session,
                    last_session=val_res.last_session,
                    existing_sessions_in_store=existing_count,
                    earliest_stored_date_before=earliest_stored,
                    latest_stored_date_before=latest_stored,
                    feed=self.feed,
                    adjustment=self.adjustment,
                    structural_quality=val_res.structural_quality,
                    stored_provenance_adjustment_state=primary_stored_adj,
                    provenance_compatible=is_prov_compatible,
                    rejection_reason=reason,
                    retrieval_timestamp_utc=retrieval_utc,
                    pages_retrieved=pages_count,
                )
                summary.failed_validation_count += 1
                summary.candidate_dispositions[sym] = asdict(disp)
                continue

            # Persist if apply_writes is active
            ingested_count = 0
            ingest_utc = None
            if apply_writes and self.db_engine:
                ingest_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                prov = MarketProvenance(
                    provider=Provider.ALPACA,
                    ingestion_source=IngestionSource.DIRECT_PROVIDER,
                    observed_at=retrieval_utc,
                    observed_date=val_res.last_session,
                    observation_precision=ObservationPrecision.DATE,
                    observation_source=ObservationSource.PROVIDER_BAR_TIMESTAMP,
                    ingested_at=ingest_utc,
                    adjustment_state=AdjustmentState.SPLIT_ADJUSTED_ONLY,
                    structural_quality=val_res.structural_quality,
                    fallback_status=False,
                )
                try:
                    # Atomic write of bars and sidecar provenance for this candidate
                    # R2: on BACKFILL or REPLACE, atomically replace existing scope within reviewed bounds to preserve outside-scope data
                    self.db_engine.save_daily_candles_with_evidence(
                        sym,
                        val_res.validated_bars,
                        prov,
                        replace_existing_scope=(action in ("BACKFILL", "REPLACE")),
                        scope_start_date=sym_start,
                        scope_end_date=end_date,
                    )
                    ingested_count = len(val_res.validated_bars)
                    summary.total_bars_ingested += ingested_count
                    summary.completed_writes.append(sym)
                except Exception as write_err:
                    # SIP-05: Handle persistence error gracefully and record truth
                    logger.error(f"Persistence error for candidate '{sym}': {write_err}")
                    summary.failed_candidate = sym
                    disp = CandidateIngestionDisposition(
                        symbol=sym,
                        action_type=action,
                        status="PERSISTENCE_ERROR",
                        bars_retrieved=len(raw_bars),
                        bars_ingested=0,
                        first_session=val_res.first_session,
                        last_session=val_res.last_session,
                        existing_sessions_in_store=existing_count,
                        earliest_stored_date_before=earliest_stored,
                        latest_stored_date_before=latest_stored,
                        feed=self.feed,
                        adjustment=self.adjustment,
                        structural_quality=val_res.structural_quality,
                        stored_provenance_adjustment_state=primary_stored_adj,
                        provenance_compatible=is_prov_compatible,
                        rejection_reason=f"PERSISTENCE_ERROR: {str(write_err)}",
                        retrieval_timestamp_utc=retrieval_utc,
                        pages_retrieved=pages_count,
                    )
                    summary.candidate_dispositions[sym] = asdict(disp)

                    # Mark remaining symbols as UNPROCESSED
                    curr_idx = target_symbols.index(sym)
                    for remaining_sym in target_symbols[curr_idx + 1:]:
                        summary.unprocessed_candidates.append(remaining_sym)
                        rem_disp = CandidateIngestionDisposition(
                            symbol=remaining_sym,
                            action_type="UNPROCESSED",
                            status="UNPROCESSED",
                            rejection_reason=f"ABORTED_AFTER_PRIOR_PERSISTENCE_FAILURE: Processing aborted following write error on '{sym}'.",
                        )
                        summary.candidate_dispositions[remaining_sym] = asdict(rem_disp)
                    break

            status = "APPLIED" if apply_writes else "PREPARED_DRY_RUN"
            if action in ("BACKFILL", "REPLACE"):
                summary.backfilled_count += 1
            else:
                summary.refreshed_count += 1

            anchor_bar = next((b for b in val_res.validated_bars if b["time"] == "2026-10-19"), None)
            has_anchor = anchor_bar is not None
            obs_close = float(anchor_bar["close"]) if has_anchor else None

            disp = CandidateIngestionDisposition(
                symbol=sym,
                action_type=action,
                status=status,
                bars_retrieved=len(raw_bars),
                bars_ingested=ingested_count,
                first_session=val_res.first_session,
                last_session=val_res.last_session,
                existing_sessions_in_store=existing_count,
                earliest_stored_date_before=earliest_stored,
                latest_stored_date_before=latest_stored,
                feed=self.feed,
                adjustment=self.adjustment,
                structural_quality=val_res.structural_quality,
                stored_provenance_adjustment_state=primary_stored_adj,
                provenance_compatible=is_prov_compatible,
                retrieval_timestamp_utc=retrieval_utc,
                ingestion_timestamp_utc=ingest_utc,
                pages_retrieved=pages_count,
                l03_equivalence_established=False,  # Conditional: candidate market data evidence only
                l03_anchor_session="2026-10-19" if has_anchor else None,
                l03_observation_price=obs_close,
                l03_derivation_note=(
                    f"Candidate {sym} has verified split-adjusted bar on reference anchor session 2026-10-19 (close={obs_close}). "
                    "Equivalence is conditional: factor 1.0 assumed without corporate action verification. "
                    "Candidate market data evidence only; does not certify L03/L04 compliance."
                    if has_anchor else
                    f"Anchor session 2026-10-19 not present in acquired series for {sym}; L03 price-floor equivalence cannot be established."
                ),
            )
            summary.candidate_dispositions[sym] = asdict(disp)

        # SIP-05: Compute honest batch outcome and execution outcome
        if summary.failed_candidate is not None:
            summary.execution_outcome = "PARTIAL_SUCCESS" if summary.completed_writes else "FAILURE"
            summary.batch_outcome = summary.execution_outcome
        else:
            total_eval = summary.total_candidates_evaluated
            total_failed = summary.failed_validation_count + summary.provider_error_count
            if total_eval == 0 or total_failed == total_eval:
                summary.execution_outcome = "FAILURE"
            elif total_failed > 0:
                summary.execution_outcome = "PARTIAL_SUCCESS"
            else:
                summary.execution_outcome = "SUCCESS"
            summary.batch_outcome = summary.execution_outcome

        # SIP-04: Populate attributable L03 bridge status
        candidates_with_anchor = [
            sym for sym, d in summary.candidate_dispositions.items()
            if d.get("l03_equivalence_established") is True
        ]
        summary.l03_bridge_status = {
            "anchor_session": "2026-10-19",
            "adjustment_state": "SPLIT_ADJUSTED_ONLY",
            "anchor_split_factor": 1.0,
            "candidates_with_anchor_observation": len(candidates_with_anchor),
            "anchor_candidates": candidates_with_anchor,
            "disclaimer": (
                "Ingestion provides candidate market data evidence only; it does not certify L03 price-floor or L04 "
                "turnover compliance, and confers zero universe admission or publication authority."
            ),
        }

        return summary


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments without triggering network calls or database writes."""
    parser = argparse.ArgumentParser(
        description="ARX Terminal: Bounded Alpaca Historical SIP Market Data Ingestion Adapter"
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(REPO_ROOT / "evidence" / "coverage_expansion" / "package_b_market_data_ingestion.json"),
        help="Path to reviewed candidate manifest or Package B specification JSON.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Optional comma-separated list of symbols to ingest (overriding manifest universe).",
    )
    parser.add_argument(
        "--feed",
        type=str,
        default="sip",
        choices=["sip"],
        help="Market data tape feed (strictly 'sip' for consolidated tape).",
    )
    parser.add_argument(
        "--adjustment",
        type=str,
        default="split",
        choices=["split"],
        help="Adjustment convention (strictly 'split' per Package B contract).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-09-15",
        help="Inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-10-19",
        help="Inclusive end date (YYYY-MM-DD, completed trading session).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum number of candidates to process.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Explicit path to target SQLite database (required when --apply is active).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Explicitly commit writes to the target database. Default is non-mutating dry-run.",
    )
    parser.add_argument(
        "--output-summary-path",
        type=str,
        default=None,
        help="Path to write execution summary JSON artifact.",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main CLI entrypoint.
    Returns:
      0: SUCCESS (all candidates succeeded)
      1: FAILURE (fatal error or all candidates failed)
      2: PARTIAL_SUCCESS (some candidates succeeded, some failed)
    """
    parsed = parse_args(args)

    symbols_list = [s.strip().upper() for s in parsed.symbols.split(",")] if parsed.symbols else None
    adapter = AlpacaHistoricalSipIngestionAdapter(feed=parsed.feed, adjustment=parsed.adjustment)

    try:
        summary = adapter.execute_bounded_ingestion(
            manifest_path=Path(parsed.manifest_path),
            symbols_override=symbols_list,
            start_date=parsed.start_date,
            end_date=parsed.end_date,
            apply_writes=parsed.apply,
            db_path=parsed.db_path,
            max_candidates=parsed.max_candidates,
        )

        out_json = json.dumps(summary.to_dict(), indent=2)
        if parsed.output_summary_path:
            p_out = Path(parsed.output_summary_path)
            p_out.parent.mkdir(parents=True, exist_ok=True)
            p_out.write_text(out_json, encoding="utf-8")
            print(f"Execution summary written to: {p_out}")
        else:
            print(out_json)

        # SIP-05: Exit code reflects honest batch outcome
        if summary.batch_outcome == "SUCCESS":
            return 0
        elif summary.batch_outcome == "PARTIAL_SUCCESS":
            return 2
        else:
            return 1
    except Exception as e:
        logger.error(f"Ingestion execution aborted: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
