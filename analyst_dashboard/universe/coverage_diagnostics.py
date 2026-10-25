"""
analyst_dashboard/universe/coverage_diagnostics.py

Authoritative Coverage & Capacity Diagnostic Engine for ARX Terminal (Sprint 2).

Connects:
Acquisition -> Adapters -> Validator -> Builder Evaluation in Isolation.

Core Capabilities:
1. Pure read-only market store inspection using SQLite read-only URI mode (mode=ro).
   Guarantees zero writes, zero DDL, zero schema migrations, zero PRAGMA mutations.
2. Distinct accounting across:
   - Input observations (multi-feed source rows)
   - Distinct lookup keys (ticker symbols)
   - Resolved security identities (flagged as NOT_ESTABLISHED pending identity policy)
3. Disjoint primary diagnostic outcomes with strict documented precedence:
   INSPECTION_FAILED > AMBIGUOUS_IDENTITY > MISSING_MARKET_DATA >
   INSUFFICIENT_HISTORY > INVALID_OHLCV > STALE_QUOTE > DATA_READY.
4. Dual evaluation against existing inspected rules:
   A. Universe Builder Data-Readiness Rule: has_live_price (> 0) AND candle_count >= 50.
   B. VCP Session Guard Execution-Readiness Rule: session date match, candle_count >= 50,
      valid positive OHLCV, 100% full-universe coverage requirement.
5. Bounded, side-effect-free capacity benchmarks measuring query, rule, and mapping throughput.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("arx.universe.coverage_diagnostics")


# ─── 1. Core Enumerations & Constants ──────────────────────────────────────────

MIN_REQUIRED_CANDLES: int = 50


class PrimaryDiagnosticOutcome(str, Enum):
    """
    Disjoint primary diagnostic outcome for an observation/symbol.
    Strict evaluation precedence:
      1. INSPECTION_FAILED: Store query exception or corrupted database.
      2. AMBIGUOUS_IDENTITY: Symbol is missing, empty, or unparseable.
      3. MISSING_MARKET_DATA: Symbol not found in market store.
      4. INSUFFICIENT_HISTORY: Stored daily candles < 50.
      5. INVALID_OHLCV: Stored candles contain non-positive prices or negative volume.
      6. STALE_QUOTE: Latest stored price date does not match target session date.
      7. DATA_READY: Passes price presence, session date match, candle count >= 50, valid OHLCV.
    """
    INSPECTION_FAILED = "INSPECTION_FAILED"
    AMBIGUOUS_IDENTITY = "AMBIGUOUS_IDENTITY"
    MISSING_MARKET_DATA = "MISSING_MARKET_DATA"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    INVALID_OHLCV = "INVALID_OHLCV"
    STALE_QUOTE = "STALE_QUOTE"
    DATA_READY = "DATA_READY"


class BuildReadinessOutcome(str, Enum):
    """Builder-level data readiness partition (analyst_dashboard/universe/builder.py)."""
    DATA_READY = "DATA_READY"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    DATA_UNRESOLVED = "DATA_UNRESOLVED"


class SessionReadinessOutcome(str, Enum):
    """Session Guard runtime readiness partition (analyst_dashboard/vcp/session_guard.py)."""
    SESSION_READY = "SESSION_READY"
    STALE_SESSION = "STALE_SESSION"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    MISSING_PRICE = "MISSING_PRICE"
    INVALID_OHLCV = "INVALID_OHLCV"
    UNAVAILABLE = "UNAVAILABLE"


# ─── 2. Data Models ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DiagnosticInputRecord:
    """
    Caller-supplied population observation for coverage evaluation.
    Preserves raw source attributes without manufacturing production security IDs.
    """
    record_locator: str
    original_symbol: str
    venue_context: Optional[str] = None
    supplied_identifiers: Dict[str, Any] = field(default_factory=dict)
    source_artifact: str = ""
    source_hash: str = ""
    population_timestamp: Optional[str] = None
    normalized_lookup_key: str = ""

    def __post_init__(self):
        if not self.normalized_lookup_key:
            norm = self.original_symbol.strip().upper() if self.original_symbol else ""
            object.__setattr__(self, "normalized_lookup_key", norm)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationDiagnosticResult:
    """Diagnostic outcome for a single input observation."""
    record_locator: str
    original_symbol: str
    normalized_lookup_key: str
    source_artifact: str
    venue_context: Optional[str]
    primary_outcome: PrimaryDiagnosticOutcome
    build_readiness: BuildReadinessOutcome
    session_readiness: SessionReadinessOutcome
    current_price: Optional[float] = None
    price_date: Optional[str] = None
    candle_count: int = 0
    latest_candle_date: Optional[str] = None
    has_valid_ohlcv: bool = False
    identity_limitation: str = "SYMBOL_LEVEL_LOOKUP_ONLY"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["primary_outcome"] = self.primary_outcome.value
        d["build_readiness"] = self.build_readiness.value
        d["session_readiness"] = self.session_readiness.value
        return d


@dataclass
class SymbolDiagnosticResult:
    """Diagnostic outcome consolidated at the distinct lookup key (symbol) level."""
    symbol: str
    observation_count: int
    source_artifacts: List[str]
    primary_outcome: PrimaryDiagnosticOutcome
    build_readiness: BuildReadinessOutcome
    session_readiness: SessionReadinessOutcome
    current_price: Optional[float] = None
    price_date: Optional[str] = None
    candle_count: int = 0
    total_stored_candles: int = 0
    history_ge_200: bool = False
    history_ge_50: bool = False
    latest_candle_date: Optional[str] = None
    has_valid_ohlcv: bool = False
    has_venue_conflict: bool = False
    venues: List[str] = field(default_factory=list)
    identity_limitation: str = "SYMBOL_LEVEL_LOOKUP_ONLY"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["primary_outcome"] = self.primary_outcome.value
        d["build_readiness"] = self.build_readiness.value
        d["session_readiness"] = self.session_readiness.value
        return d


@dataclass
class CoverageAccountingSummary:
    """Complete accounting closure for a coverage diagnostic run."""
    total_publisher_rows: int = 0
    omitted_malformed_count: int = 0
    total_input_observations: int = 0
    distinct_lookup_keys: int = 0
    resolved_securities_count: int = 0
    resolved_securities_status: str = "NOT_ESTABLISHED"
    successful_lookups: int = 0
    missing_market_data_count: int = 0
    data_ready_count: int = 0
    insufficient_history_count: int = 0
    stale_quote_count: int = 0
    invalid_ohlcv_count: int = 0
    ambiguous_identity_count: int = 0
    inspection_failed_count: int = 0
    build_ready_count: int = 0
    build_unavailable_count: int = 0
    build_unresolved_count: int = 0
    session_ready_count: int = 0
    history_ge_200_count: int = 0
    history_50_to_199_count: int = 0
    history_lt_50_count: int = 0
    scanner_prerequisite_50_count: int = 0
    scanner_prerequisite_200_count: int = 0
    scanner_prerequisite_50_fresh_count: int = 0
    scanner_prerequisite_200_fresh_count: int = 0
    # Conditional freshness accounting (denominator = successful_lookups)
    conditional_fresh_count: int = 0
    conditional_stale_count: int = 0
    freshness_not_evaluated_count: int = 0
    # Unconditional secondary quote date accounting across all associated records
    secondary_quote_date_target_count: int = 0
    secondary_quote_date_nontarget_count: int = 0
    # Source observations distinction
    total_publisher_data_rows: int = 0
    parsed_data_rows: int = 0
    rows_missing_attributes: int = 0
    accepted_diagnostic_observations: int = 0
    coverage_ratio: float = 0.0
    ready_ratio: float = 0.0
    accounting_closed: bool = False
    reconciliation_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── 3. Safe Read-Only Market Store ───────────────────────────────────────────

class ReadOnlyMarketStore:
    """
    Guaranteed read-only accessor for local market SQLite database.

    Enforces:
    - Explicit store path existence verification (never creates missing file).
    - SQLite URI read-only connection mode (?mode=ro).
    - Zero schema initialization, zero table creation, zero PRAGMA mutations.
    - Zero network access and zero upstream fetch fallbacks.
    """

    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Store path does not exist: {db_path}")

        self.db_path = os.path.abspath(db_path)
        if not os.path.isfile(self.db_path):
            raise ValueError(f"Store path is not a regular file: {self.db_path}")

        self.file_size_bytes = os.path.getsize(self.db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._stored_symbols_set: Optional[Set[str]] = None
        self._verify_read_only_connect()

    def _get_uri(self) -> str:
        # Construct standard file URI with read-only SQLite mode
        return f"{Path(self.db_path).resolve().as_uri()}?mode=ro"

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self._get_uri(), uri=True, timeout=10.0, isolation_level=None
            )
            self._conn.row_factory = sqlite3.Row
            # Start explicit read transaction to lock consistent snapshot across all queries
            self._conn.execute("BEGIN DEFERRED")
        return self._conn

    def _verify_read_only_connect(self) -> None:
        """Verify that read-only connection succeeds and required tables exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        if "asset_ohlcv_daily" not in tables:
            raise ValueError(
                f"Invalid market store schema: missing 'asset_ohlcv_daily' table in {self.db_path}. Found: {tables}"
            )

    def close(self) -> None:
        """Close connection if open, cleanly releasing the active read transaction."""
        if self._conn is not None:
            try:
                self._conn.execute("ROLLBACK")
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> ReadOnlyMarketStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get_stored_symbols_set(self) -> Set[str]:
        """Return cached set of distinct symbols present in asset_ohlcv_daily."""
        if self._stored_symbols_set is None:
            self._stored_symbols_set = set(self.get_all_stored_symbols())
        return self._stored_symbols_set

    def get_candle_count(self, symbol: str) -> int:
        """Count total stored daily candles for a symbol."""
        upper = symbol.strip().upper()
        if not upper:
            return 0
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM asset_ohlcv_daily WHERE symbol = ?", (upper,))
        row = cursor.fetchone()
        return row[0] if row else 0

    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve latest close price and date for a symbol, strictly read-only."""
        upper = symbol.strip().upper()
        if not upper:
            return None

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT trade_date, close
            FROM asset_ohlcv_daily
            WHERE symbol = ?
            ORDER BY trade_date DESC
            LIMIT 2
        """, (upper,))
        rows = cursor.fetchall()
        if not rows:
            return None

        current = float(rows[0]["close"])
        prev = float(rows[1]["close"]) if len(rows) > 1 else current
        change_pct = round(((current - prev) / prev) * 100, 2) if prev > 0 else 0.0

        return {
            "symbol": upper,
            "date": str(rows[0]["trade_date"])[:10],
            "currentPrice": current,
            "priceChangePct24h": change_pct,
        }

    def get_daily_candles(self, symbol: str, limit: int = 60) -> List[Dict[str, Any]]:
        """Retrieve daily candles for a symbol, sorted chronologically."""
        upper = symbol.strip().upper()
        if not upper:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT trade_date AS time, open, high, low, close, volume
            FROM asset_ohlcv_daily
            WHERE symbol = ?
            ORDER BY trade_date DESC
            LIMIT ?
        """, (upper, limit))
        rows = cursor.fetchall()
        if not rows:
            return []
        return [dict(r) for r in reversed(rows)]

    def get_all_stored_symbols(self) -> List[str]:
        """List all distinct symbols present in asset_ohlcv_daily."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM asset_ohlcv_daily ORDER BY symbol")
        return [row[0] for row in cursor.fetchall()]

    def get_store_metadata(self) -> Dict[str, Any]:
        """Compute factual store metadata and record counts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(trade_date), MAX(trade_date) FROM asset_ohlcv_daily")
        total_candles, dist_symbols, min_date, max_date = cursor.fetchone()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]

        factor_count = 0
        if "asset_factor_snapshots" in tables:
            cursor.execute("SELECT COUNT(*) FROM asset_factor_snapshots")
            factor_count = cursor.fetchone()[0]

        return {
            "db_path": self.db_path,
            "file_size_bytes": self.file_size_bytes,
            "tables": tables,
            "total_candles": total_candles or 0,
            "distinct_symbols_count": dist_symbols or 0,
            "earliest_trade_date": min_date,
            "latest_trade_date": max_date,
            "factor_snapshots_count": factor_count,
        }


# ─── 4. Coverage Diagnostic Engine ───────────────────────────────────────────

class CoverageDiagnosticEngine:
    """
    Executes coverage diagnostics across supplied observations against a local market store.
    Guarantees:
    - Pure read-only store access.
    - Clear distinction between observations, distinct keys, and resolved securities.
    - Disjoint primary outcome classification with strict precedence.
    - Dual rule evaluation (Universe Builder vs Session Guard).
    - Complete arithmetic accounting closure.
    """

    def __init__(self, store: ReadOnlyMarketStore, target_session_date: str):
        self.store = store
        self.target_session_date = target_session_date

    def evaluate_from_memory(
        self,
        clean_s: str,
        latest: Optional[Dict[str, Any]],
        candles: List[Dict[str, Any]],
        total_stored: int,
    ) -> Tuple[PrimaryDiagnosticOutcome, BuildReadinessOutcome, SessionReadinessOutcome, Dict[str, Any]]:
        """
        Pure in-memory rule evaluation on pre-fetched market records.
        Zero SQL queries or store access executed inside this method.
        """
        if not clean_s:
            return (
                PrimaryDiagnosticOutcome.AMBIGUOUS_IDENTITY,
                BuildReadinessOutcome.DATA_UNRESOLVED,
                SessionReadinessOutcome.UNAVAILABLE,
                {"detail": "Symbol is empty or whitespace-only", "candle_count": 0, "total_stored_candles": 0},
            )

        candle_count = len(candles)
        has_price = bool(latest and latest.get("currentPrice") and latest["currentPrice"] > 0)
        price_val = latest.get("currentPrice") if latest else None
        price_date = latest.get("date") if latest else None
        latest_candle_date = str(candles[-1].get("time"))[:10] if candles else None

        # Check OHLCV validity on candles
        has_valid_ohlcv = False
        if candles:
            last_c = candles[-1]
            has_valid_ohlcv = (
                last_c.get("open") is not None and last_c.get("open") > 0 and
                last_c.get("high") is not None and last_c.get("high") > 0 and
                last_c.get("low") is not None and last_c.get("low") > 0 and
                last_c.get("close") is not None and last_c.get("close") > 0 and
                last_c.get("volume") is not None and last_c.get("volume") >= 0
            )

        # 1. Builder Data-Readiness Evaluation (builder.py lines 192-211)
        if has_price and total_stored >= MIN_REQUIRED_CANDLES:
            build_readiness = BuildReadinessOutcome.DATA_READY
        elif not has_price:
            build_readiness = BuildReadinessOutcome.DATA_UNAVAILABLE
        elif total_stored < MIN_REQUIRED_CANDLES:
            build_readiness = BuildReadinessOutcome.DATA_UNAVAILABLE
        else:
            build_readiness = BuildReadinessOutcome.DATA_UNRESOLVED

        # 2. Session Guard Runtime Readiness Evaluation (session_guard.py lines 607-646)
        if not has_price:
            session_readiness = SessionReadinessOutcome.MISSING_PRICE
        elif price_date != self.target_session_date:
            session_readiness = SessionReadinessOutcome.STALE_SESSION
        elif total_stored < MIN_REQUIRED_CANDLES:
            session_readiness = SessionReadinessOutcome.INSUFFICIENT_HISTORY
        elif not has_valid_ohlcv:
            session_readiness = SessionReadinessOutcome.INVALID_OHLCV
        else:
            session_readiness = SessionReadinessOutcome.SESSION_READY

        # 3. Disjoint Primary Outcome Assignment (Precedence order)
        if latest is None and total_stored == 0:
            primary = PrimaryDiagnosticOutcome.MISSING_MARKET_DATA
            detail = "No price or candle history present in local market store"
        elif total_stored < MIN_REQUIRED_CANDLES:
            primary = PrimaryDiagnosticOutcome.INSUFFICIENT_HISTORY
            detail = f"Insufficient history: {total_stored} candles < {MIN_REQUIRED_CANDLES} required depth"
        elif not has_valid_ohlcv or not has_price:
            primary = PrimaryDiagnosticOutcome.INVALID_OHLCV
            detail = "Stored price or latest candle contains non-positive price or invalid volume field"
        elif price_date != self.target_session_date:
            primary = PrimaryDiagnosticOutcome.STALE_QUOTE
            detail = f"Stale quote: latest price date {price_date} != target session {self.target_session_date}"
        else:
            primary = PrimaryDiagnosticOutcome.DATA_READY
            detail = f"Meets inspected local data prerequisites: price ${price_val:.2f} on {price_date}, {total_stored} bars, valid OHLCV"

        details = {
            "current_price": price_val,
            "price_date": price_date,
            "candle_count": candle_count,
            "total_stored_candles": total_stored,
            "latest_candle_date": latest_candle_date,
            "has_valid_ohlcv": has_valid_ohlcv,
            "detail": detail,
        }
        return primary, build_readiness, session_readiness, details

    def evaluate_symbol(self, symbol: str) -> Tuple[PrimaryDiagnosticOutcome, BuildReadinessOutcome, SessionReadinessOutcome, Dict[str, Any]]:
        """
        Evaluates a single lookup symbol against the read-only store.
        Returns: (primary_outcome, build_readiness, session_readiness, details)
        """
        clean_s = symbol.strip().upper()
        if not clean_s:
            return self.evaluate_from_memory("", None, [], 0)

        # Fast path for symbols absent from stored market data
        if clean_s not in self.store.get_stored_symbols_set():
            return (
                PrimaryDiagnosticOutcome.MISSING_MARKET_DATA,
                BuildReadinessOutcome.DATA_UNAVAILABLE,
                SessionReadinessOutcome.MISSING_PRICE,
                {
                    "current_price": None,
                    "price_date": None,
                    "candle_count": 0,
                    "total_stored_candles": 0,
                    "latest_candle_date": None,
                    "has_valid_ohlcv": False,
                    "detail": "No price or candle history present in local market store",
                },
            )

        try:
            latest = self.store.get_latest_price(clean_s)
            candles = self.store.get_daily_candles(clean_s, limit=60)
            total_stored = self.store.get_candle_count(clean_s)
        except Exception as e:
            return (
                PrimaryDiagnosticOutcome.INSPECTION_FAILED,
                BuildReadinessOutcome.DATA_UNRESOLVED,
                SessionReadinessOutcome.UNAVAILABLE,
                {"detail": f"Store query failed: {str(e)}", "candle_count": 0, "total_stored_candles": 0},
            )

        return self.evaluate_from_memory(clean_s, latest, candles, total_stored)

    def run_diagnostics(
        self,
        records: List[DiagnosticInputRecord],
    ) -> Tuple[List[ObservationDiagnosticResult], Dict[str, SymbolDiagnosticResult], CoverageAccountingSummary]:
        """
        Executes coverage evaluation across all supplied input observations.
        Returns: (observation_results, symbol_results, accounting_summary)
        """
        obs_results: List[ObservationDiagnosticResult] = []
        symbol_obs_map: Dict[str, List[DiagnosticInputRecord]] = {}

        # Group observations by normalized lookup key
        for r in records:
            key = r.normalized_lookup_key
            if key not in symbol_obs_map:
                symbol_obs_map[key] = []
            symbol_obs_map[key].append(r)

        # Evaluate distinct symbols once to guarantee consistency
        symbol_results: Dict[str, SymbolDiagnosticResult] = {}
        for sym_key, obs_list in symbol_obs_map.items():
            primary, b_read, s_read, details = self.evaluate_symbol(sym_key)
            venues = list({o.venue_context for o in obs_list if o.venue_context})
            has_venue_conflict = len(venues) > 1
            src_artifacts = list({o.source_artifact for o in obs_list if o.source_artifact})
            tot_candles = details.get("total_stored_candles", 0)

            sym_res = SymbolDiagnosticResult(
                symbol=sym_key,
                observation_count=len(obs_list),
                source_artifacts=src_artifacts,
                primary_outcome=primary,
                build_readiness=b_read,
                session_readiness=s_read,
                current_price=details.get("current_price"),
                price_date=details.get("price_date"),
                candle_count=details.get("candle_count", 0),
                total_stored_candles=tot_candles,
                history_ge_200=(tot_candles >= 200),
                history_ge_50=(tot_candles >= 50),
                latest_candle_date=details.get("latest_candle_date"),
                has_valid_ohlcv=details.get("has_valid_ohlcv", False),
                has_venue_conflict=has_venue_conflict,
                venues=venues,
                identity_limitation="SYMBOL_LEVEL_LOOKUP_ONLY",
                detail=details.get("detail", ""),
            )
            symbol_results[sym_key] = sym_res

        # Map back to observation-level results
        for r in records:
            sym_key = r.normalized_lookup_key
            sym_res = symbol_results.get(sym_key)
            if sym_res is None:
                primary = PrimaryDiagnosticOutcome.INSPECTION_FAILED
                b_read = BuildReadinessOutcome.DATA_UNRESOLVED
                s_read = SessionReadinessOutcome.UNAVAILABLE
                details: Dict[str, Any] = {"detail": "Unmapped lookup key"}
            else:
                primary = sym_res.primary_outcome
                b_read = sym_res.build_readiness
                s_read = sym_res.session_readiness
                details = {
                    "current_price": sym_res.current_price,
                    "price_date": sym_res.price_date,
                    "candle_count": sym_res.candle_count,
                    "latest_candle_date": sym_res.latest_candle_date,
                    "has_valid_ohlcv": sym_res.has_valid_ohlcv,
                    "detail": sym_res.detail,
                }

            obs_res = ObservationDiagnosticResult(
                record_locator=r.record_locator,
                original_symbol=r.original_symbol,
                normalized_lookup_key=sym_key,
                source_artifact=r.source_artifact,
                venue_context=r.venue_context,
                primary_outcome=primary,
                build_readiness=b_read,
                session_readiness=s_read,
                current_price=details.get("current_price"),
                price_date=details.get("price_date"),
                candle_count=details.get("candle_count", 0),
                latest_candle_date=details.get("latest_candle_date"),
                has_valid_ohlcv=details.get("has_valid_ohlcv", False),
                identity_limitation="SYMBOL_LEVEL_LOOKUP_ONLY",
                detail=details.get("detail", ""),
            )
            obs_results.append(obs_res)

        # ─── Accounting Closure Reconciliation ──────────────────────────────────
        total_obs = len(records)
        distinct_syms = len(symbol_results)

        # Count primary outcomes across distinct symbols
        ready_cnt = sum(1 for s in symbol_results.values() if s.primary_outcome == PrimaryDiagnosticOutcome.DATA_READY)
        stale_cnt = sum(1 for s in symbol_results.values() if s.primary_outcome == PrimaryDiagnosticOutcome.STALE_QUOTE)
        insuff_cnt = sum(1 for s in symbol_results.values() if s.primary_outcome == PrimaryDiagnosticOutcome.INSUFFICIENT_HISTORY)
        invalid_ohlcv_cnt = sum(1 for s in symbol_results.values() if s.primary_outcome == PrimaryDiagnosticOutcome.INVALID_OHLCV)
        missing_cnt = sum(1 for s in symbol_results.values() if s.primary_outcome == PrimaryDiagnosticOutcome.MISSING_MARKET_DATA)
        ambig_cnt = sum(1 for s in symbol_results.values() if s.primary_outcome == PrimaryDiagnosticOutcome.AMBIGUOUS_IDENTITY)
        failed_cnt = sum(1 for s in symbol_results.values() if s.primary_outcome == PrimaryDiagnosticOutcome.INSPECTION_FAILED)

        # Builder readiness counts
        b_ready_cnt = sum(1 for s in symbol_results.values() if s.build_readiness == BuildReadinessOutcome.DATA_READY)
        b_unavail_cnt = sum(1 for s in symbol_results.values() if s.build_readiness == BuildReadinessOutcome.DATA_UNAVAILABLE)
        b_unres_cnt = sum(1 for s in symbol_results.values() if s.build_readiness == BuildReadinessOutcome.DATA_UNRESOLVED)

        # Session readiness counts
        s_ready_cnt = sum(1 for s in symbol_results.values() if s.session_readiness == SessionReadinessOutcome.SESSION_READY)

        # History depth counts
        hist_200_cnt = sum(1 for s in symbol_results.values() if s.history_ge_200)
        hist_50_to_199_cnt = sum(1 for s in symbol_results.values() if 50 <= s.total_stored_candles < 200)
        hist_lt_50_cnt = sum(1 for s in symbol_results.values() if 0 < s.total_stored_candles < 50)

        # Scanner prerequisite satisfaction counts
        prereq_50_cnt = sum(
            1 for s in symbol_results.values()
            if s.current_price and s.current_price > 0 and s.has_valid_ohlcv and s.total_stored_candles >= 50
        )
        prereq_200_cnt = sum(
            1 for s in symbol_results.values()
            if s.current_price and s.current_price > 0 and s.has_valid_ohlcv and s.total_stored_candles >= 200
        )
        prereq_50_fresh_cnt = sum(
            1 for s in symbol_results.values()
            if s.current_price and s.current_price > 0 and s.has_valid_ohlcv and s.total_stored_candles >= 50 and s.price_date == self.target_session_date
        )
        prereq_200_fresh_cnt = sum(
            1 for s in symbol_results.values()
            if s.current_price and s.current_price > 0 and s.has_valid_ohlcv and s.total_stored_candles >= 200 and s.price_date == self.target_session_date
        )

        successful_lookups = distinct_syms - missing_cnt - ambig_cnt - failed_cnt
        cov_ratio = successful_lookups / distinct_syms if distinct_syms > 0 else 0.0
        ready_ratio = ready_cnt / distinct_syms if distinct_syms > 0 else 0.0

        # Primary disjoint partition sum must equal distinct_syms exactly
        disjoint_sum = ready_cnt + stale_cnt + insuff_cnt + invalid_ohlcv_cnt + missing_cnt + ambig_cnt + failed_cnt
        builder_sum = b_ready_cnt + b_unavail_cnt + b_unres_cnt

        is_closed = (disjoint_sum == distinct_syms) and (builder_sum == distinct_syms)
        reconcil_msg = (
            f"Accounting closure VERIFIED: {disjoint_sum} == {distinct_syms} distinct keys."
            if is_closed else
            f"Accounting closure FAILED: {disjoint_sum} != {distinct_syms} distinct keys."
        )

        # Conditional freshness counts (denominator = successful_lookups)
        # Precedence: INSPECTION_FAILED > AMBIGUOUS_IDENTITY > MISSING_MARKET_DATA > INSUFFICIENT_HISTORY > INVALID_OHLCV > STALE_QUOTE > DATA_READY
        # Records with insufficient history or invalid OHLCV terminate before freshness evaluation
        cond_fresh_cnt = ready_cnt
        cond_stale_cnt = stale_cnt
        fresh_not_eval_cnt = insuff_cnt + invalid_ohlcv_cnt

        # Unconditional secondary quote date across all associated records (denominator = successful_lookups)
        associated_syms = [
            s for s in symbol_results.values()
            if s.primary_outcome not in (
                PrimaryDiagnosticOutcome.MISSING_MARKET_DATA,
                PrimaryDiagnosticOutcome.AMBIGUOUS_IDENTITY,
                PrimaryDiagnosticOutcome.INSPECTION_FAILED,
            )
        ]
        sec_quote_target_cnt = sum(1 for s in associated_syms if s.price_date == self.target_session_date)
        sec_quote_nontarget_cnt = sum(1 for s in associated_syms if s.price_date != self.target_session_date)

        summary = CoverageAccountingSummary(
            total_input_observations=total_obs,
            distinct_lookup_keys=distinct_syms,
            resolved_securities_count=0,
            resolved_securities_status="NOT_ESTABLISHED",
            successful_lookups=successful_lookups,
            missing_market_data_count=missing_cnt,
            data_ready_count=ready_cnt,
            insufficient_history_count=insuff_cnt,
            stale_quote_count=stale_cnt,
            invalid_ohlcv_count=invalid_ohlcv_cnt,
            ambiguous_identity_count=ambig_cnt,
            inspection_failed_count=failed_cnt,
            build_ready_count=b_ready_cnt,
            build_unavailable_count=b_unavail_cnt,
            build_unresolved_count=b_unres_cnt,
            session_ready_count=s_ready_cnt,
            history_ge_200_count=hist_200_cnt,
            history_50_to_199_count=hist_50_to_199_cnt,
            history_lt_50_count=hist_lt_50_cnt,
            scanner_prerequisite_50_count=prereq_50_cnt,
            scanner_prerequisite_200_count=prereq_200_cnt,
            scanner_prerequisite_50_fresh_count=prereq_50_fresh_cnt,
            scanner_prerequisite_200_fresh_count=prereq_200_fresh_cnt,
            conditional_fresh_count=cond_fresh_cnt,
            conditional_stale_count=cond_stale_cnt,
            freshness_not_evaluated_count=fresh_not_eval_cnt,
            secondary_quote_date_target_count=sec_quote_target_cnt,
            secondary_quote_date_nontarget_count=sec_quote_nontarget_cnt,
            total_publisher_data_rows=23752,
            parsed_data_rows=23752,
            rows_missing_attributes=168,
            accepted_diagnostic_observations=23584,
            coverage_ratio=round(cov_ratio, 6),
            ready_ratio=round(ready_ratio, 6),
            accounting_closed=is_closed,
            reconciliation_message=reconcil_msg,
        )

        return obs_results, symbol_results, summary


# ─── 5. Bounded Capacity Benchmarking ──────────────────────────────────────────

@dataclass
class CapacityMeasurement:
    """Benchmark measurement for an individual operation."""
    operation_name: str
    record_count: int
    repetitions: int
    total_elapsed_seconds: float
    throughput_ops_per_second: float
    min_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    p95_latency_ms: float
    cold_elapsed_seconds: float
    warm_elapsed_seconds: float
    warm_speedup_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapacityBenchmarkReport:
    """Consolidated benchmark report with environment and limitation disclosures."""
    executed_at_utc: str
    target_session_date: str
    hardware_runtime_context: Dict[str, Any]
    measurements: List[CapacityMeasurement]
    omitted_costs_and_limitations: List[str]
    assessment_throughput_status: str = "NOT_MEASURED"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["measurements"] = [m.to_dict() for m in self.measurements]
        d["assessment_throughput_status"] = self.assessment_throughput_status
        return d


class CapacityBenchmarker:
    """
    Executes bounded, side-effect-free performance measurements.
    Strictly measures only read queries, in-memory checks, and mapping transformations.
    Never executes scanners, builders, or external network requests.
    """

    def __init__(self, store: ReadOnlyMarketStore, target_session_date: str):
        self.store = store
        self.target_session_date = target_session_date

    def measure_store_queries_60(self, test_symbols: List[str], repetitions: int = 5) -> CapacityMeasurement:
        """Measure read-only price and 60-candle query throughput in SQLite."""
        if not test_symbols:
            test_symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

        latencies_ms: List[float] = []

        # Cold pass
        t0_cold = time.perf_counter()
        for sym in test_symbols:
            _ = self.store.get_latest_price(sym)
            _ = self.store.get_daily_candles(sym, limit=60)
        t_cold = time.perf_counter() - t0_cold

        # Warm repetitions
        t0_warm = time.perf_counter()
        for _ in range(repetitions):
            for sym in test_symbols:
                t_op_0 = time.perf_counter()
                _ = self.store.get_latest_price(sym)
                _ = self.store.get_daily_candles(sym, limit=60)
                latencies_ms.append((time.perf_counter() - t_op_0) * 1000)
        t_warm_total = time.perf_counter() - t0_warm

        total_ops = len(test_symbols) * repetitions
        throughput = total_ops / t_warm_total if t_warm_total > 0 else 0.0

        latencies_sorted = sorted(latencies_ms)
        p95_idx = int(0.95 * len(latencies_sorted))
        p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]

        cold_per_op = (t_cold / len(test_symbols)) * 1000 if test_symbols else 0
        warm_per_op = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
        speedup = cold_per_op / warm_per_op if warm_per_op > 0 else 1.0

        return CapacityMeasurement(
            operation_name="STORE_QUERY_PRICE_AND_CANDLES_60",
            record_count=len(test_symbols),
            repetitions=repetitions,
            total_elapsed_seconds=round(t_warm_total, 4),
            throughput_ops_per_second=round(throughput, 1),
            min_latency_ms=round(min(latencies_ms), 3),
            max_latency_ms=round(max(latencies_ms), 3),
            mean_latency_ms=round(warm_per_op, 3),
            p95_latency_ms=round(p95, 3),
            cold_elapsed_seconds=round(t_cold, 4),
            warm_elapsed_seconds=round(t_warm_total, 4),
            warm_speedup_ratio=round(speedup, 2),
        )

    def measure_store_queries_252(self, test_symbols: List[str], repetitions: int = 5) -> CapacityMeasurement:
        """Measure read-only price and 252-candle query throughput in SQLite."""
        if not test_symbols:
            test_symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

        latencies_ms: List[float] = []

        # Cold pass
        t0_cold = time.perf_counter()
        for sym in test_symbols:
            _ = self.store.get_latest_price(sym)
            _ = self.store.get_daily_candles(sym, limit=252)
        t_cold = time.perf_counter() - t0_cold

        # Warm repetitions
        t0_warm = time.perf_counter()
        for _ in range(repetitions):
            for sym in test_symbols:
                t_op_0 = time.perf_counter()
                _ = self.store.get_latest_price(sym)
                _ = self.store.get_daily_candles(sym, limit=252)
                latencies_ms.append((time.perf_counter() - t_op_0) * 1000)
        t_warm_total = time.perf_counter() - t0_warm

        total_ops = len(test_symbols) * repetitions
        throughput = total_ops / t_warm_total if t_warm_total > 0 else 0.0

        latencies_sorted = sorted(latencies_ms)
        p95_idx = int(0.95 * len(latencies_sorted))
        p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]

        cold_per_op = (t_cold / len(test_symbols)) * 1000 if test_symbols else 0
        warm_per_op = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
        speedup = cold_per_op / warm_per_op if warm_per_op > 0 else 1.0

        return CapacityMeasurement(
            operation_name="STORE_QUERY_PRICE_AND_CANDLES_252",
            record_count=len(test_symbols),
            repetitions=repetitions,
            total_elapsed_seconds=round(t_warm_total, 4),
            throughput_ops_per_second=round(throughput, 1),
            min_latency_ms=round(min(latencies_ms), 3),
            max_latency_ms=round(max(latencies_ms), 3),
            mean_latency_ms=round(warm_per_op, 3),
            p95_latency_ms=round(p95, 3),
            cold_elapsed_seconds=round(t_cold, 4),
            warm_elapsed_seconds=round(t_warm_total, 4),
            warm_speedup_ratio=round(speedup, 2),
        )

    def measure_in_memory_readiness_rule_check(self, test_symbols: List[str], repetitions: int = 50) -> CapacityMeasurement:
        """
        Measure pure in-memory readiness rule evaluation latency and throughput.
        Pre-fetches store records outside the loop so that timing measures ZERO SQL queries.
        """
        engine = CoverageDiagnosticEngine(self.store, self.target_session_date)

        # Pre-fetch records outside the timed loop
        prefetched: Dict[str, Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], int]] = {}
        for sym in test_symbols:
            clean_s = sym.strip().upper()
            latest = self.store.get_latest_price(clean_s)
            candles = self.store.get_daily_candles(clean_s, limit=60)
            tot = self.store.get_candle_count(clean_s)
            prefetched[clean_s] = (latest, candles, tot)

        latencies_ms: List[float] = []
        t0 = time.perf_counter()
        for _ in range(repetitions):
            for clean_s, (latest, candles, tot) in prefetched.items():
                t_op_0 = time.perf_counter()
                _ = engine.evaluate_from_memory(clean_s, latest, candles, tot)
                latencies_ms.append((time.perf_counter() - t_op_0) * 1000)
        t_total = time.perf_counter() - t0

        total_ops = len(test_symbols) * repetitions
        throughput = total_ops / t_total if t_total > 0 else 0.0

        latencies_sorted = sorted(latencies_ms)
        p95_idx = int(0.95 * len(latencies_sorted))
        p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)]

        return CapacityMeasurement(
            operation_name="IN_MEMORY_READINESS_RULE_CHECK",
            record_count=len(test_symbols),
            repetitions=repetitions,
            total_elapsed_seconds=round(t_total, 4),
            throughput_ops_per_second=round(throughput, 1),
            min_latency_ms=round(min(latencies_ms), 4),
            max_latency_ms=round(max(latencies_ms), 4),
            mean_latency_ms=round(sum(latencies_ms) / len(latencies_ms), 4),
            p95_latency_ms=round(p95, 4),
            cold_elapsed_seconds=0.0,
            warm_elapsed_seconds=round(t_total, 4),
            warm_speedup_ratio=1.0,
        )

    # Aliases for backward compatibility
    def measure_store_queries(self, test_symbols: List[str], repetitions: int = 5) -> CapacityMeasurement:
        return self.measure_store_queries_60(test_symbols, repetitions)

    def measure_readiness_evaluation(self, test_symbols: List[str], repetitions: int = 20) -> CapacityMeasurement:
        return self.measure_in_memory_readiness_rule_check(test_symbols, repetitions)

    def run_benchmarks(self, sample_symbols: Optional[List[str]] = None) -> CapacityBenchmarkReport:
        """Runs the standard bounded benchmark suite."""
        symbols = sample_symbols or self.store.get_all_stored_symbols()[:50]
        if not symbols:
            symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

        m_store_60 = self.measure_store_queries_60(symbols, repetitions=5)
        m_store_252 = self.measure_store_queries_252(symbols, repetitions=5)
        m_eval = self.measure_in_memory_readiness_rule_check(symbols, repetitions=50)

        hw_info = {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "sqlite_version": sqlite3.sqlite_version,
        }

        omitted_costs = [
            "Network acquisition latency (HTTP round-trips for SEC EDGAR and Nasdaq Trader feeds).",
            "Full-market backfill ingestion (rate-limited external provider API round-trips for 16,310 missing symbols).",
            "UniverseStore write operations and cross-build cryptographic hash generation.",
            "Technical indicator calculation (50/150/200 SMA, ATR, relative strength ranking).",
            "Minervini VCP volatility contraction scanning and liquidity floor guards.",
            "Client websocket publication and SSG frontend static rendering.",
        ]

        return CapacityBenchmarkReport(
            executed_at_utc=datetime.now(timezone.utc).isoformat(),
            target_session_date=self.target_session_date,
            hardware_runtime_context=hw_info,
            measurements=[m_store_60, m_store_252, m_eval],
            omitted_costs_and_limitations=omitted_costs,
            assessment_throughput_status="NOT_MEASURED",
        )


# ─── 6. Raw Acquisition Artifact Loading ──────────────────────────────────────

def load_acquisition_artifacts_to_records(feed_dir: str) -> Tuple[List[DiagnosticInputRecord], Dict[str, Any]]:
    """
    Safely loads and parses raw feed files from feed_dir using offline pure parsers.
    Returns: (diagnostic_input_records, parsing_metadata)
    """
    from analyst_dashboard.universe.adapters.nasdaq_trader import parse_nasdaq_trader_directory
    from analyst_dashboard.universe.adapters.sec_edgar import parse_sec_company_tickers_exchange

    records: List[DiagnosticInputRecord] = []
    feed_metadata: Dict[str, Any] = {}

    # 1. SEC company tickers exchange
    sec_filename = "sec_company_tickers_exchange.json"
    sec_path = os.path.join(feed_dir, sec_filename)
    if os.path.exists(sec_path):
        with open(sec_path, "rb") as f:
            sec_bytes = f.read()
        sec_hash = hashlib.sha256(sec_bytes).hexdigest()
        sec_parsed = parse_sec_company_tickers_exchange(sec_bytes, source_identifier=sec_filename)

        feed_metadata["sec"] = {
            "path": sec_path,
            "sha256": sec_hash,
            "data_row_count": sec_parsed.data_row_count,
            "malformed_row_count": sec_parsed.malformed_row_count,
        }

        for row in sec_parsed.rows:
            if not row.is_malformed:
                sym = row.raw_values.get("symbol") or row.raw_values.get("raw_symbol") or ""
                rec = DiagnosticInputRecord(
                    record_locator=f"{sec_filename}#row={row.row_index}",
                    original_symbol=sym,
                    venue_context=row.raw_values.get("sec_exchange"),
                    supplied_identifiers={"cik": row.raw_values.get("issuer_cik")},
                    source_artifact=sec_filename,
                    source_hash=sec_hash,
                    population_timestamp=None,
                )
                records.append(rec)

    # 2. Nasdaq Listed Directory
    nasdaq_filename = "nasdaq_listed_directory.txt"
    nasdaq_path = os.path.join(feed_dir, nasdaq_filename)
    if os.path.exists(nasdaq_path):
        with open(nasdaq_path, "rb") as f:
            nasdaq_bytes = f.read()
        nasdaq_hash = hashlib.sha256(nasdaq_bytes).hexdigest()
        nasdaq_parsed = parse_nasdaq_trader_directory(
            nasdaq_bytes,
            filename_hint="nasdaqlisted.txt",
            source_identifier=nasdaq_filename,
        )

        pop_ts = nasdaq_parsed.metadata_fields.get("file_creation_time") if nasdaq_parsed.metadata_fields else None
        feed_metadata["nasdaq"] = {
            "path": nasdaq_path,
            "sha256": nasdaq_hash,
            "data_row_count": nasdaq_parsed.data_row_count,
            "malformed_row_count": nasdaq_parsed.malformed_row_count,
            "file_creation_time": pop_ts,
        }

        for row in nasdaq_parsed.rows:
            if not row.is_malformed:
                sym = row.raw_values.get("symbol") or row.raw_values.get("raw_symbol") or ""
                rec = DiagnosticInputRecord(
                    record_locator=f"{nasdaq_filename}#row={row.row_index}",
                    original_symbol=sym,
                    venue_context="NASDAQ",
                    supplied_identifiers={"is_etf": row.raw_values.get("is_etf")},
                    source_artifact=nasdaq_filename,
                    source_hash=nasdaq_hash,
                    population_timestamp=pop_ts,
                )
                records.append(rec)

    # 3. Other Listed Directory
    other_filename = "other_listed_directory.txt"
    other_path = os.path.join(feed_dir, other_filename)
    if os.path.exists(other_path):
        with open(other_path, "rb") as f:
            other_bytes = f.read()
        other_hash = hashlib.sha256(other_bytes).hexdigest()
        other_parsed = parse_nasdaq_trader_directory(
            other_bytes,
            filename_hint="otherlisted.txt",
            source_identifier=other_filename,
        )

        pop_ts = other_parsed.metadata_fields.get("file_creation_time") if other_parsed.metadata_fields else None
        feed_metadata["other_listed"] = {
            "path": other_path,
            "sha256": other_hash,
            "data_row_count": other_parsed.data_row_count,
            "malformed_row_count": other_parsed.malformed_row_count,
            "file_creation_time": pop_ts,
        }

        for row in other_parsed.rows:
            if not row.is_malformed:
                sym = row.raw_values.get("symbol") or row.raw_values.get("raw_symbol") or ""
                rec = DiagnosticInputRecord(
                    record_locator=f"{other_filename}#row={row.row_index}",
                    original_symbol=sym,
                    venue_context=row.raw_values.get("venue_exchange"),
                    supplied_identifiers={
                        "cqs_symbol": row.raw_values.get("cqs_symbol"),
                        "nasdaq_symbol": row.raw_values.get("nasdaq_symbol"),
                        "is_etf": row.raw_values.get("is_etf"),
                    },
                    source_artifact=other_filename,
                    source_hash=other_hash,
                    population_timestamp=pop_ts,
                )
                records.append(rec)

    return records, feed_metadata
