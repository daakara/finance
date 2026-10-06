"""
analyst_dashboard/security_master/persistence.py

Hybrid SQLite + In-Memory LRU Cache Persistence Repository for ARX Security Master.
Guarantees full provenance preservation, CWD-independence, and auditability.

Invariants Enforced:
- PERSISTENCE_MODEL = HYBRID_SQLITE_PERSISTENCE_WITH_LRU_CACHE
- Stored records preserve complete provider provenance, conflict state, and identifiers.
- Cache hits never strip provenance or modify eligibility.
- Freshness TTL enforcement ensures stale data fails closed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, Optional, Union
from collections import OrderedDict

from .models import (
    AssetClass,
    SecurityType,
    ListingStatus,
    ClassificationStatus,
    ExecutionEligibility,
    AnalyticsCapability,
    CanonicalInstrument,
)
from .config import (
    resolve_security_master_db_path,
    get_security_master_ttl,
)

logger = logging.getLogger("arx.security_master.persistence")


class ThreadSafeLRUCache:
    """Thread-safe LRU cache with capacity limit."""

    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.cache: OrderedDict[str, CanonicalInstrument] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[CanonicalInstrument]:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: CanonicalInstrument) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()


class SecurityMasterRepository:
    """
    Hybrid SQLite + LRU Cache persistence manager for CanonicalInstrument records.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        ttl_seconds: Optional[float] = None,
        lru_capacity: int = 1024,
        clock: Optional[Any] = None,
    ):
        self.db_path = resolve_security_master_db_path(db_path)
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else get_security_master_ttl()
        self.clock = clock or time.time
        self.lru_cache = ThreadSafeLRUCache(capacity=lru_capacity)
        self._schema_initialized = False
        self._lock = threading.Lock()

        # In-memory connections must be persistent per instance
        self._is_memory = str(self.db_path) == ":memory:" or str(self.db_path).startswith(":memory:")
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self._is_memory:
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
            self._ensure_schema(self._memory_conn)

    def _get_connection(self) -> sqlite3.Connection:
        if self._is_memory and self._memory_conn is not None:
            return self._memory_conn

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if not self._schema_initialized:
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS canonical_instruments (
                    symbol TEXT PRIMARY KEY,
                    provider_symbol TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    security_type TEXT NOT NULL,
                    primary_exchange TEXT NOT NULL,
                    listing_status TEXT NOT NULL,
                    classification_status TEXT NOT NULL,
                    execution_eligibility TEXT NOT NULL,
                    analytics_capability TEXT NOT NULL,
                    classification_authority TEXT NOT NULL,
                    classification_timestamp TEXT NOT NULL,
                    resolved_at_epoch REAL NOT NULL,
                    stable_identifiers_json TEXT NOT NULL,
                    source_provenance_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sec_master_eligibility
                ON canonical_instruments(execution_eligibility);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sec_master_type
                ON canonical_instruments(security_type);
            """)
            conn.commit()
            self._schema_initialized = True

    def save(self, instrument: CanonicalInstrument, now_epoch: Optional[float] = None) -> None:
        """Saves canonical instrument to SQLite and updates LRU cache."""
        clean_symbol = instrument.symbol.strip().upper()
        current_epoch = now_epoch if now_epoch is not None else self.clock()

        stable_json = json.dumps(instrument.stable_identifiers or {})
        prov_json = json.dumps(instrument.source_provenance or {})

        with self._lock:
            conn = self._get_connection()
            try:
                self._ensure_schema(conn)
                conn.execute("""
                    INSERT OR REPLACE INTO canonical_instruments (
                        symbol,
                        provider_symbol,
                        asset_class,
                        security_type,
                        primary_exchange,
                        listing_status,
                        classification_status,
                        execution_eligibility,
                        analytics_capability,
                        classification_authority,
                        classification_timestamp,
                        resolved_at_epoch,
                        stable_identifiers_json,
                        source_provenance_json,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    clean_symbol,
                    instrument.provider_symbol,
                    instrument.asset_class,
                    instrument.security_type,
                    instrument.primary_exchange,
                    instrument.listing_status,
                    instrument.classification_status,
                    instrument.execution_eligibility,
                    instrument.analytics_capability,
                    instrument.classification_authority,
                    instrument.classification_timestamp,
                    current_epoch,
                    stable_json,
                    prov_json,
                    current_epoch,
                ))
                if not self._is_memory:
                    conn.commit()
            finally:
                if not self._is_memory:
                    conn.close()

        # Update in-memory LRU cache
        self.lru_cache.put(clean_symbol, (instrument, current_epoch))

    def get(
        self,
        symbol: str,
        now_epoch: Optional[float] = None,
        allow_stale: bool = False,
    ) -> Optional[CanonicalInstrument]:
        """
        Retrieves canonical instrument by symbol.
        Checks LRU cache first, then SQLite.
        Enforces freshness TTL unless allow_stale=True.
        """
        clean_symbol = symbol.strip().upper()
        if not clean_symbol:
            return None

        current_epoch = now_epoch if now_epoch is not None else self.clock()

        # 1. LRU Cache probe
        cached_entry = self.lru_cache.get(clean_symbol)
        if cached_entry is not None:
            cached_inst, cached_resolved_at = cached_entry
            if allow_stale or (current_epoch - cached_resolved_at) <= self.ttl_seconds:
                return cached_inst

        # 2. SQLite probe
        with self._lock:
            conn = self._get_connection()
            try:
                self._ensure_schema(conn)
                cur = conn.execute("""
                    SELECT * FROM canonical_instruments WHERE symbol = ?;
                """, (clean_symbol,))
                row = cur.fetchone()
                if not row:
                    return None

                resolved_at = row["resolved_at_epoch"]
                if not allow_stale and (current_epoch - resolved_at) > self.ttl_seconds:
                    logger.debug(f"Cached record for {clean_symbol} is stale ({current_epoch - resolved_at:.1f}s > {self.ttl_seconds}s)")
                    return None

                try:
                    stable_ids = json.loads(row["stable_identifiers_json"])
                except Exception:
                    stable_ids = {}

                try:
                    prov = json.loads(row["source_provenance_json"])
                except Exception:
                    prov = {}

                instrument = CanonicalInstrument(
                    symbol=row["symbol"],
                    provider_symbol=row["provider_symbol"],
                    asset_class=AssetClass(row["asset_class"]),
                    security_type=SecurityType(row["security_type"]),
                    primary_exchange=row["primary_exchange"],
                    listing_status=ListingStatus(row["listing_status"]),
                    classification_status=ClassificationStatus(row["classification_status"]),
                    execution_eligibility=ExecutionEligibility(row["execution_eligibility"]),
                    analytics_capability=AnalyticsCapability(row["analytics_capability"]),
                    classification_authority=row["classification_authority"],
                    classification_timestamp=row["classification_timestamp"],
                    source_provenance=prov,
                    stable_identifiers=stable_ids,
                )

                # Populate LRU
                self.lru_cache.put(clean_symbol, (instrument, resolved_at))
                return instrument
            finally:
                if not self._is_memory:
                    conn.close()

    def clear_cache(self) -> None:
        self.lru_cache.clear()
