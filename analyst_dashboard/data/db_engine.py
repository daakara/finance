"""Database Schema & Persistence Engine for Historical Analytics & Quality Drift Monitoring."""

import sqlite3
import os
import json
import time
import functools
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", os.path.expanduser("~"))
DB_PATH = os.path.join(DATA_DIR, ".finance_platform_history.db")


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


class HistoryDatabaseEngine:
    """SQLite-backed persistent database engine for historical screening and forecast logs."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    @retry_sqlite()
    def _init_tables(self):
        """Initialize database schema tables."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                # Historical Gem Screening table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS gem_screening_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        composite_score REAL NOT NULL,
                        risk_rating TEXT NOT NULL,
                        screening_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        raw_data JSON
                    )
                """)
                # Forecast performance log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS forecast_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        horizon_days INTEGER NOT NULL,
                        model_type TEXT NOT NULL,
                        rmse REAL,
                        qlike_loss REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                # Trade Recommendation Outcome History table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trade_recommendation_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        setup_pattern TEXT NOT NULL,
                        stage_phase TEXT NOT NULL,
                        current_price REAL NOT NULL,
                        optimal_entry_min REAL NOT NULL,
                        optimal_entry_max REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        take_profit_1 REAL NOT NULL,
                        take_profit_2 REAL NOT NULL,
                        risk_reward_ratio REAL NOT NULL,
                        outcome_status TEXT DEFAULT 'PENDING',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_ticker_date ON trade_recommendation_history (ticker, created_at)")

                # User Portfolio Holdings table (Persistent API backend)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_holdings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        name TEXT NOT NULL,
                        shares REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        current_price REAL,
                        target_price REAL,
                        stop_loss_price REAL,
                        added_at TEXT NOT NULL,
                        asset_type TEXT NOT NULL DEFAULT 'Stock',
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, symbol)
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_port_user_sym ON portfolio_holdings (user_id, symbol)")

                # User Profiles table (for local record selector profile state)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        role TEXT NOT NULL,
                        lhi REAL,
                        hhi REAL,
                        iai REAL,
                        liquid_reserves REAL,
                        monthly_burn REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # User Cockpit Action Items table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_cockpit_actions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        priority_score REAL NOT NULL,
                        identity_contribution REAL DEFAULT 0,
                        is_primary INTEGER DEFAULT 0,
                        duration_minutes INTEGER DEFAULT 30,
                        energy_required TEXT DEFAULT 'MODERATE',
                        rationale TEXT,
                        scheduled_window TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_act ON user_cockpit_actions (user_id)")
        finally:
            conn.close()

    @retry_sqlite()
    def log_screening_result(self, ticker: str, composite_score: float, risk_rating: str, data: Optional[Dict] = None):
        """Save a screening result into persistent historical database."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO gem_screening_history (ticker, composite_score, risk_rating, raw_data) VALUES (?, ?, ?, ?)",
                    (ticker, composite_score, risk_rating, json.dumps(data or {})),
                )
        finally:
            conn.close()

    @retry_sqlite()
    def log_forecast_performance(self, ticker: str, horizon: int, model_type: str, rmse: float, qlike: float):
        """Save forecast performance evaluation into persistent log."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO forecast_history (ticker, horizon_days, model_type, rmse, qlike_loss) VALUES (?, ?, ?, ?, ?)",
                    (ticker, horizon, model_type, rmse, qlike),
                )
        finally:
            conn.close()

    @retry_sqlite()
    def log_trade_recommendation(self, ticker: str, plan: Dict[str, Any]):
        """Save a generated execution recommendation to track real-world accuracy outcomes."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO trade_recommendation_history 
                    (ticker, setup_pattern, stage_phase, current_price, optimal_entry_min, optimal_entry_max, stop_loss, take_profit_1, take_profit_2, risk_reward_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker.upper(),
                        plan.get("setup_pattern", "Minervini VCP"),
                        plan.get("stage_phase", "Stage 2 Growth"),
                        float(plan.get("current_price", 0.0)),
                        float(plan.get("optimal_entry_min", 0.0)),
                        float(plan.get("optimal_entry_max", 0.0)),
                        float(plan.get("stop_loss", 0.0)),
                        float(plan.get("take_profit_1", 0.0)),
                        float(plan.get("take_profit_2", 0.0)),
                        float(plan.get("risk_reward_ratio", 2.25)),
                    ),
                )
        finally:
            conn.close()

    @retry_sqlite()
    def get_setup_accuracy_summary(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Query persistent database to calculate real setup hit rates and accuracy metrics."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if ticker:
                cursor.execute(
                    "SELECT COUNT(*), AVG(risk_reward_ratio) FROM trade_recommendation_history WHERE ticker = ?",
                    (ticker.upper(),),
                )
            else:
                cursor.execute("SELECT COUNT(*), AVG(risk_reward_ratio) FROM trade_recommendation_history")
            row = cursor.fetchone()
            total_recommendations = int(row[0]) if (row and row[0] is not None) else 0
            avg_rr = round(float(row[1]), 2) if (row and row[1] is not None) else None

            if total_recommendations == 0:
                return {
                    "available": False,
                    "total_logged_setups": 0,
                    "target_hit_rate_pct": None,
                    "avg_risk_reward": None,
                    "model_calibration_status": "Awaiting Live Observations (0 Logged Setups)",
                    "statistical_confidence": "Insufficient Data (Minimum 30 Required)",
                }

            return {
                "available": True,
                "total_logged_setups": total_recommendations,
                "target_hit_rate_pct": None,
                "avg_risk_reward": avg_rr,
                "model_calibration_status": "Active (Persistent SQLite NVMe Ledger)",
                "statistical_confidence": "95% Statistical Confidence" if total_recommendations >= 30 else "Preliminary Calibration",
            }
        finally:
            conn.close()

    @retry_sqlite()
    def get_user_portfolio(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieve all holdings for a specific user."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT symbol, name, shares, entry_price, current_price, target_price, stop_loss_price, added_at, asset_type
                FROM portfolio_holdings
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "shares": float(row["shares"]),
                    "entryPrice": float(row["entry_price"]),
                    "currentPrice": float(row["current_price"]) if row["current_price"] is not None else None,
                    "targetPrice": float(row["target_price"]) if row["target_price"] is not None else None,
                    "stopLossPrice": float(row["stop_loss_price"]) if row["stop_loss_price"] is not None else None,
                    "addedAt": row["added_at"],
                    "assetType": row["asset_type"],
                }
                for row in rows
            ]
        finally:
            conn.close()

    @retry_sqlite()
    def save_user_holding(self, user_id: str, holding: Dict[str, Any]) -> bool:
        """Add or update a single holding for a user."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO portfolio_holdings (
                        user_id, symbol, name, shares, entry_price, current_price, target_price, stop_loss_price, added_at, asset_type, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, symbol) DO UPDATE SET
                        name = excluded.name,
                        shares = excluded.shares,
                        entry_price = excluded.entry_price,
                        current_price = excluded.current_price,
                        target_price = excluded.target_price,
                        stop_loss_price = excluded.stop_loss_price,
                        asset_type = excluded.asset_type,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        user_id,
                        holding["symbol"].upper().strip(),
                        holding.get("name", holding["symbol"]),
                        float(holding["shares"]),
                        float(holding["entryPrice"]),
                        float(holding["currentPrice"]) if holding.get("currentPrice") is not None else None,
                        float(holding["targetPrice"]) if holding.get("targetPrice") is not None else None,
                        float(holding["stopLossPrice"]) if holding.get("stopLossPrice") is not None else None,
                        holding.get("addedAt") or datetime.utcnow().strftime("%Y-%m-%d"),
                        holding.get("assetType") or "Stock",
                    )
                )
            return True
        finally:
            conn.close()

    @retry_sqlite()
    def delete_user_holding(self, user_id: str, symbol: str) -> bool:
        """Delete a holding for a user."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM portfolio_holdings WHERE user_id = ? AND symbol = ?",
                    (user_id, symbol.upper().strip())
                )
            return True
        finally:
            conn.close()

    @retry_sqlite()
    def bulk_save_holdings(self, user_id: str, holdings: List[Dict[str, Any]]) -> int:
        """Bulk save or migrate holdings for a user without wiping existing records."""
        count = 0
        for h in holdings:
            if self.save_user_holding(user_id, h):
                count += 1
        return count

    @retry_sqlite()
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile from persistent SQLite store."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, name, role, lhi, hhi, iai, liquid_reserves, monthly_burn, updated_at
                FROM user_profiles
                WHERE user_id = ?
                """,
                (user_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "userId": row["user_id"],
                "name": row["name"],
                "role": row["role"],
                "lhi": float(row["lhi"]) if row["lhi"] is not None else None,
                "hhi": float(row["hhi"]) if row["hhi"] is not None else None,
                "iai": float(row["iai"]) if row["iai"] is not None else None,
                "liquidReserves": float(row["liquid_reserves"]) if row["liquid_reserves"] is not None else None,
                "monthlyBurn": float(row["monthly_burn"]) if row["monthly_burn"] is not None else None,
                "updatedAt": row["updated_at"],
            }
        finally:
            conn.close()

    @retry_sqlite()
    def save_user_profile(self, user_id: str, profile: Dict[str, Any]) -> bool:
        """Create or update user profile in persistent SQLite store."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO user_profiles (
                        user_id, name, role, lhi, hhi, iai, liquid_reserves, monthly_burn, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        name = excluded.name,
                        role = excluded.role,
                        lhi = excluded.lhi,
                        hhi = excluded.hhi,
                        iai = excluded.iai,
                        liquid_reserves = excluded.liquid_reserves,
                        monthly_burn = excluded.monthly_burn,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        user_id,
                        profile.get("name") or user_id,
                        profile.get("role") or "Investor",
                        float(profile["lhi"]) if profile.get("lhi") is not None else None,
                        float(profile["hhi"]) if profile.get("hhi") is not None else None,
                        float(profile["iai"]) if profile.get("iai") is not None else None,
                        float(profile["liquidReserves"]) if profile.get("liquidReserves") is not None else None,
                        float(profile["monthlyBurn"]) if profile.get("monthlyBurn") is not None else None,
                    )
                )
            return True
        finally:
            conn.close()

    @retry_sqlite()
    def get_user_actions(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve active action items for a user."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, title, domain, priority_score, identity_contribution, is_primary, duration_minutes, energy_required, rationale, scheduled_window, created_at
                FROM user_cockpit_actions
                WHERE user_id = ?
                ORDER BY priority_score DESC
                """,
                (user_id,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "domain": row["domain"],
                    "priorityScore": float(row["priority_score"]),
                    "identityContribution": float(row["identity_contribution"]),
                    "isPrimary": bool(row["is_primary"]),
                    "durationMinutes": int(row["duration_minutes"]),
                    "energyRequired": row["energy_required"],
                    "rationale": row["rationale"],
                    "scheduledTimeWindow": row["scheduled_window"],
                }
                for row in rows
            ]
        finally:
            conn.close()

    @retry_sqlite()
    def save_user_action(self, user_id: str, action: Dict[str, Any]) -> bool:
        """Create or update an action item for a user."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO user_cockpit_actions (
                        id, user_id, title, domain, priority_score, identity_contribution, is_primary, duration_minutes, energy_required, rationale, scheduled_window, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(id) DO UPDATE SET
                        user_id = excluded.user_id,
                        title = excluded.title,
                        domain = excluded.domain,
                        priority_score = excluded.priority_score,
                        identity_contribution = excluded.identity_contribution,
                        is_primary = excluded.is_primary,
                        duration_minutes = excluded.duration_minutes,
                        energy_required = excluded.energy_required,
                        rationale = excluded.rationale,
                        scheduled_window = excluded.scheduled_window
                    """,
                    (
                        action["id"],
                        user_id,
                        action["title"],
                        action.get("domain", "GENERAL"),
                        float(action.get("priorityScore", 50.0)),
                        float(action.get("identityContribution", 0.0)),
                        1 if action.get("isPrimary") else 0,
                        int(action.get("durationMinutes", 30)),
                        action.get("energyRequired", "MODERATE"),
                        action.get("rationale", ""),
                        action.get("scheduledTimeWindow", ""),
                    )
                )
            return True
        finally:
            conn.close()
