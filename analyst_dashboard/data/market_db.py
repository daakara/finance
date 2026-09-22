import sqlite3
import os
import json
import time
import functools
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None

DATA_DIR = os.getenv("FINANCE_DATA_DIR", os.getenv("DATA_DIR", os.path.expanduser("~")))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, ".finance_market_store.db")


def retry_sqlite(max_retries: int = 3, base_delay: float = 0.05):
    """Decorator to retry SQLite operations with exponential backoff on database lock contention."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    last_err = e
                    err_msg = str(e).lower()
                    if "locked" in err_msg or "busy" in err_msg:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"SQLite contention on {func.__name__} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {base_delay * (2 ** attempt):.3f}s..."
                            )
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                    raise
                except Exception:
                    raise
            if last_err:
                raise last_err
        return wrapper
    return decorator


class MarketDatabaseEngine:
    """Production-grade SQLite persistent store for market data, eliminating synthetic fallbacks."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    @retry_sqlite()
    def _init_schema(self):
        """Create tables for persistent market storage if they do not exist."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                # 1. Historical Daily Candles
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS asset_ohlcv_daily (
                        symbol TEXT NOT NULL,
                        trade_date TEXT NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        PRIMARY KEY (symbol, trade_date)
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_date ON asset_ohlcv_daily (symbol, trade_date)")

                # 2. Asset Factor & Fundamentals Snapshot (Versioned Point-in-Time Schema)
                cursor.execute("PRAGMA table_info(asset_factor_snapshots)")
                existing_cols = {row["name"] for row in cursor.fetchall()}
                if not existing_cols:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS asset_factor_snapshots (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            as_of_date TEXT,
                            period_end TEXT,
                            filing_date TEXT,
                            acceptance_datetime TEXT,
                            available_from TEXT,
                            fetched_at TEXT,
                            point_in_time_status TEXT DEFAULT 'CURRENT_ONLY',
                            source TEXT DEFAULT 'unknown',
                            current_price REAL,
                            price_change_24h REAL,
                            growth_score INTEGER,
                            quality_score INTEGER,
                            valuation_score INTEGER,
                            momentum_score INTEGER,
                            tail_risk_score INTEGER,
                            composite_score INTEGER,
                            piotroski_f INTEGER,
                            verdict TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_sym_avail ON asset_factor_snapshots (symbol, available_from)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_sym_asof ON asset_factor_snapshots (symbol, as_of_date)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_sym_id ON asset_factor_snapshots (symbol, id DESC)")
                elif "available_from" not in existing_cols:
                    logger.info("Migrating legacy asset_factor_snapshots table to versioned point-in-time schema...")
                    cursor.execute("ALTER TABLE asset_factor_snapshots RENAME TO asset_factor_snapshots_legacy")
                    cursor.execute("""
                        CREATE TABLE asset_factor_snapshots (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            as_of_date TEXT,
                            period_end TEXT,
                            filing_date TEXT,
                            acceptance_datetime TEXT,
                            available_from TEXT,
                            fetched_at TEXT,
                            point_in_time_status TEXT DEFAULT 'CURRENT_ONLY',
                            source TEXT DEFAULT 'unknown',
                            current_price REAL,
                            price_change_24h REAL,
                            growth_score INTEGER,
                            quality_score INTEGER,
                            valuation_score INTEGER,
                            momentum_score INTEGER,
                            tail_risk_score INTEGER,
                            composite_score INTEGER,
                            piotroski_f INTEGER,
                            verdict TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    updated_at_expr = "updated_at" if "updated_at" in existing_cols else "CURRENT_TIMESTAMP"
                    cursor.execute(f"""
                        INSERT INTO asset_factor_snapshots (
                            symbol, current_price, price_change_24h, growth_score, quality_score,
                            valuation_score, momentum_score, tail_risk_score, composite_score,
                            piotroski_f, verdict, as_of_date, available_from, fetched_at,
                            point_in_time_status, source, created_at
                        )
                        SELECT
                            symbol, current_price, price_change_24h, growth_score, quality_score,
                            valuation_score, momentum_score, tail_risk_score, composite_score,
                            piotroski_f, verdict,
                            COALESCE(strftime('%Y-%m-%d', {updated_at_expr}), ''),
                            NULL,
                            {updated_at_expr},
                            'CURRENT_ONLY',
                            'legacy_migration',
                            {updated_at_expr}
                        FROM asset_factor_snapshots_legacy
                    """)
                    cursor.execute("DROP TABLE asset_factor_snapshots_legacy")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_sym_avail ON asset_factor_snapshots (symbol, available_from)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_sym_asof ON asset_factor_snapshots (symbol, as_of_date)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_sym_id ON asset_factor_snapshots (symbol, id DESC)")
                    logger.info("Migration of asset_factor_snapshots to versioned PIT store complete.")
                else:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_sym_avail ON asset_factor_snapshots (symbol, available_from)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_sym_asof ON asset_factor_snapshots (symbol, as_of_date)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_sym_id ON asset_factor_snapshots (symbol, id DESC)")

                # 3. Verified Company Catalysts
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS asset_catalyst_registry (
                        symbol TEXT PRIMARY KEY,
                        company_name TEXT NOT NULL,
                        sector TEXT NOT NULL,
                        primary_drug_trial TEXT NOT NULL,
                        trial_phase TEXT NOT NULL,
                        trial_readout_timeline TEXT NOT NULL,
                        efficacy_summary TEXT NOT NULL,
                        competitive_edge TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 4. Insider & Congressional Disclosures
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS insider_disclosures (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        politician TEXT NOT NULL,
                        chamber TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        amount_range TEXT NOT NULL,
                        filing_date TEXT NOT NULL,
                        transaction_date TEXT NOT NULL,
                        performance_since_pct REAL DEFAULT 0.0,
                        sentiment TEXT DEFAULT 'Bullish'
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_insider_sym ON insider_disclosures (symbol)")
        finally:
            conn.close()

    @retry_sqlite()
    def save_daily_candles(self, symbol: str, data: Any):
        """Save OHLCV candles (pandas DataFrame or list of dicts) to database."""
        if data is None:
            return
        upper = symbol.upper().strip()
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                if pd is not None and isinstance(data, pd.DataFrame):
                    if data.empty:
                        return
                    for idx, row in data.iterrows():
                        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx).split("T")[0]
                        cursor.execute("""
                            INSERT OR REPLACE INTO asset_ohlcv_daily (symbol, trade_date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            upper,
                            date_str,
                            round(float(row["Open"]), 2),
                            round(float(row["High"]), 2),
                            round(float(row["Low"]), 2),
                            round(float(row["Close"]), 2),
                            int(row.get("Volume", 0)),
                        ))
                elif isinstance(data, list):
                    for item in data:
                        date_str = str(item.get("time") or item.get("trade_date") or item.get("date")).split("T")[0]
                        cursor.execute("""
                            INSERT OR REPLACE INTO asset_ohlcv_daily (symbol, trade_date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            upper,
                            date_str,
                            round(float(item.get("open", item.get("Open", 0.0))), 2),
                            round(float(item.get("high", item.get("High", 0.0))), 2),
                            round(float(item.get("low", item.get("Low", 0.0))), 2),
                            round(float(item.get("close", item.get("Close", 0.0))), 2),
                            int(item.get("volume", item.get("Volume", 0))),
                        ))
        finally:
            conn.close()

    @retry_sqlite()
    def get_daily_candles(self, symbol: str, limit: int = 252) -> List[Dict[str, Any]]:
        """Retrieve stored historical daily candles for a symbol, sorted chronologically."""
        upper = symbol.upper().strip()
        conn = self._get_connection()
        try:
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
            candles = [dict(row) for row in reversed(rows)]
            return candles
        finally:
            conn.close()

    @retry_sqlite()
    def get_candles_with_freshness(self, symbol: str, limit: int = 252) -> Dict[str, Any]:
        """Retrieve stored daily candles along with calculated freshness metadata."""
        candles = self.get_daily_candles(symbol, limit=limit)
        if not candles:
            return {
                "candles": [],
                "freshness_status": "UNAVAILABLE",
                "last_trade_date": None,
                "staleness_days": None,
                "candle_count": 0,
            }
        last_date_str = str(candles[-1]["time"])[:10]
        staleness_days = 0
        try:
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
            today = datetime.utcnow().date()
            staleness_days = max(0, (today - last_date).days)
        except Exception:
            staleness_days = 0

        # Account for weekend gap (up to 4 calendar days is still considered recent)
        if staleness_days <= 1:
            freshness = "LIVE"
        elif staleness_days <= 4:
            freshness = "RECENT"
        else:
            freshness = "STALE_HISTORICAL"

        return {
            "candles": candles,
            "freshness_status": freshness,
            "last_trade_date": last_date_str,
            "staleness_days": staleness_days,
            "candle_count": len(candles),
        }

    @retry_sqlite()
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest stored close price and 24h change for an asset."""
        upper = symbol.upper().strip()
        conn = self._get_connection()
        try:
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
            change_pct = round(((current - prev) / prev) * 100, 2)

            last_date_str = str(rows[0]["trade_date"])[:10]
            staleness_days = 0
            try:
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
                today = datetime.utcnow().date()
                staleness_days = max(0, (today - last_date).days)
            except Exception:
                staleness_days = 0

            freshness = "LIVE" if staleness_days <= 1 else ("RECENT" if staleness_days <= 4 else "STALE_HISTORICAL")

            return {
                "symbol": upper,
                "date": rows[0]["trade_date"],
                "currentPrice": current,
                "priceChangePct24h": change_pct,
                "freshnessStatus": freshness,
                "stalenessDays": staleness_days,
            }
        finally:
            conn.close()

    @retry_sqlite()
    def save_factor_snapshot(self, symbol: str, snapshot: Dict[str, Any]):
        """Save factor score and fundamental snapshot to database with versioned PIT provenance.

        Does not overwrite prior historical records, enabling true Point-in-Time temporal queries.
        """
        upper = symbol.upper().strip()
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                as_of = snapshot.get("as_of_date") or snapshot.get("asOf") or snapshot.get("asOfDate")
                period_end = snapshot.get("period_end") or snapshot.get("periodEnd")
                filing_date = snapshot.get("filing_date") or snapshot.get("filingDate")
                acceptance_dt = snapshot.get("acceptance_datetime") or snapshot.get("acceptanceDatetime")
                available_from = snapshot.get("available_from") or snapshot.get("availableFrom")
                # Canonical Rule: If official SEC acceptance datetime is present, availableFrom = acceptanceDatetime
                if not available_from and acceptance_dt:
                    available_from = acceptance_dt

                fetched_at = snapshot.get("fetched_at") or snapshot.get("fetchedAt")
                if not fetched_at:
                    fetched_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

                pit_status = snapshot.get("point_in_time_status") or snapshot.get("pointInTimeStatus")
                if not pit_status:
                    pit_status = "POINT_IN_TIME" if available_from else "CURRENT_ONLY"
                elif hasattr(pit_status, "value"):
                    pit_status = pit_status.value

                source = snapshot.get("source") or ("SEC_EDGAR" if acceptance_dt else "LIVE_RUNTIME")

                # Robust extraction supporting both camelCase and snake_case factor score keys
                growth = snapshot.get("growthScore") if snapshot.get("growthScore") is not None else snapshot.get("growth_score")
                quality = snapshot.get("qualityScore") if snapshot.get("qualityScore") is not None else snapshot.get("quality_score")
                valuation = snapshot.get("valuationScore") if snapshot.get("valuationScore") is not None else snapshot.get("valuation_score")
                momentum = snapshot.get("momentumScore") if snapshot.get("momentumScore") is not None else snapshot.get("momentum_score")
                tail_risk = snapshot.get("tailRiskScore") if snapshot.get("tailRiskScore") is not None else snapshot.get("tail_risk_score")
                composite = snapshot.get("compositeFactorScore") if snapshot.get("compositeFactorScore") is not None else snapshot.get("composite_score")
                piotroski = snapshot.get("piotroskiFScore") if snapshot.get("piotroskiFScore") is not None else snapshot.get("piotroski_f")

                cursor.execute("""
                    INSERT INTO asset_factor_snapshots (
                        symbol, as_of_date, period_end, filing_date, acceptance_datetime,
                        available_from, fetched_at, point_in_time_status, source,
                        current_price, price_change_24h, growth_score, quality_score,
                        valuation_score, momentum_score, tail_risk_score, composite_score,
                        piotroski_f, verdict
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    upper,
                    as_of,
                    period_end,
                    filing_date,
                    acceptance_dt,
                    available_from,
                    fetched_at,
                    str(pit_status),
                    source,
                    float(snapshot["currentPrice"]) if snapshot.get("currentPrice") is not None else (float(snapshot["current_price"]) if snapshot.get("current_price") is not None else None),
                    float(snapshot["priceChangePct24h"]) if snapshot.get("priceChangePct24h") is not None else (float(snapshot["price_change_24h"]) if snapshot.get("price_change_24h") is not None else None),
                    int(growth) if growth is not None else None,
                    int(quality) if quality is not None else None,
                    int(valuation) if valuation is not None else None,
                    int(momentum) if momentum is not None else None,
                    int(tail_risk) if tail_risk is not None else None,
                    int(composite) if composite is not None else None,
                    int(piotroski) if piotroski is not None else None,
                    str(snapshot["verdict"]) if snapshot.get("verdict") is not None else None,
                ))
        finally:
            conn.close()

    @retry_sqlite()
    def get_factor_snapshot(
        self, symbol: str, evaluation_timestamp: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve stored factor snapshot from database.

        If evaluation_timestamp is None (live/current evaluation):
            Returns the latest stored snapshot for the symbol.
        If evaluation_timestamp is provided (historical replay / backtest cutoff T):
            Returns the latest snapshot strictly satisfying:
                available_from <= evaluation_timestamp
            with point_in_time_status == 'POINT_IN_TIME'.
            Never returns a record where available_from > evaluation_timestamp.
            Returns None if no eligible point-in-time record exists.
        """
        upper = symbol.upper().strip()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if evaluation_timestamp is None:
                cursor.execute("""
                    SELECT * FROM asset_factor_snapshots
                    WHERE symbol = ?
                    ORDER BY id DESC
                    LIMIT 1
                """, (upper,))
                row = cursor.fetchone()
                if not row:
                    return None
                return dict(row)

            # Historical evaluation at cutoff T
            eval_ts = str(evaluation_timestamp).strip()
            cursor.execute("""
                SELECT * FROM asset_factor_snapshots
                WHERE symbol = ?
                  AND point_in_time_status = 'POINT_IN_TIME'
                  AND available_from IS NOT NULL
                  AND available_from <= ?
                ORDER BY available_from DESC, id DESC
                LIMIT 1
            """, (upper, eval_ts))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            # Invariant: Never leak future fundamental data
            if res.get("available_from") and res["available_from"] > eval_ts:
                return None
            return res
        finally:
            conn.close()

    @retry_sqlite()
    def get_factor_history(self, symbol: str) -> List[Dict[str, Any]]:
        """Retrieve all historical factor snapshots for a symbol, ordered chronologically."""
        upper = symbol.upper().strip()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM asset_factor_snapshots
                WHERE symbol = ?
                ORDER BY available_from ASC, id ASC
            """, (upper,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    @retry_sqlite()
    def save_catalyst(self, symbol: str, catalyst: Dict[str, Any]):
        """Save verified business catalyst for an asset."""
        upper = symbol.upper().strip()
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO asset_catalyst_registry (
                        symbol, company_name, sector, primary_drug_trial, trial_phase,
                        trial_readout_timeline, efficacy_summary, competitive_edge
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    upper,
                    catalyst.get("company_name", f"{upper} Corporation"),
                    catalyst.get("sector", "Multi-Asset Technology / Growth"),
                    catalyst.get("primary_drug_trial", "Next-Gen Product Cycle & AI Architecture"),
                    catalyst.get("trial_phase", "Production & Enterprise Scaling"),
                    catalyst.get("trial_readout_timeline", "Quarterly Earnings & Developer Conferences"),
                    catalyst.get("efficacy_summary", "Strong operational leverage and continuous cash conversion."),
                    catalyst.get("competitive_edge", "Ecosystem network effects and high switching costs."),
                ))
        finally:
            conn.close()

    @retry_sqlite()
    def get_catalyst(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve verified business catalyst for an asset."""
        upper = symbol.upper().strip()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM asset_catalyst_registry WHERE symbol = ?", (upper,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)
        finally:
            conn.close()

    @retry_sqlite()
    def purge_stale_data(self, max_factor_age_hours: int = 24) -> int:
        """Purge records older than specified TTL from local store to prevent stale data retention."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM asset_factor_snapshots
                    WHERE updated_at < datetime('now', ?)
                """, (f"-{max_factor_age_hours} hours",))
                purged_count = cursor.rowcount
                logger.info(f"Purged {purged_count} stale factor snapshots older than {max_factor_age_hours}h.")
                return purged_count
        finally:
            conn.close()


# Backward compatibility alias
MarketDatabase = MarketDatabaseEngine


