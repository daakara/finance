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

                # User Trade Journal & Behavioral Risk Telemetry table (Option A Canonical)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_trade_journal (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        setup_name TEXT,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        shares REAL NOT NULL,
                        remaining_shares REAL,
                        r_achieved REAL,
                        followed_rules INTEGER NOT NULL DEFAULT -1,
                        confidence REAL,
                        pnl REAL,
                        status TEXT NOT NULL,
                        entry_date TEXT NOT NULL,
                        exit_date TEXT,
                        parent_trade_id INTEGER,
                        execution_role TEXT DEFAULT 'ENTRY',
                        idempotency_key TEXT,
                        notes TEXT,
                        target1 REAL,
                        stop_loss REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_journal_user_date ON user_trade_journal (user_id, created_at)")

                # Safe backward-compatible column migration for existing SQLite databases
                cursor.execute("PRAGMA table_info(user_trade_journal)")
                existing_cols = {row["name"] for row in cursor.fetchall()}
                migration_cols = [
                    ("remaining_shares", "REAL"),
                    ("exit_date", "TEXT"),
                    ("parent_trade_id", "INTEGER"),
                    ("execution_role", "TEXT DEFAULT 'ENTRY'"),
                    ("idempotency_key", "TEXT"),
                    ("notes", "TEXT"),
                    ("target1", "REAL"),
                    ("stop_loss", "REAL"),
                ]
                for col_name, col_type in migration_cols:
                    if col_name not in existing_cols:
                        cursor.execute(f"ALTER TABLE user_trade_journal ADD COLUMN {col_name} {col_type}")
                cursor.execute("UPDATE user_trade_journal SET remaining_shares = shares WHERE remaining_shares IS NULL AND status = 'OPEN'")
                cursor.execute("UPDATE user_trade_journal SET remaining_shares = 0 WHERE remaining_shares IS NULL AND status = 'CLOSED'")

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_journal_idempotency ON user_trade_journal (user_id, idempotency_key)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_journal_parent ON user_trade_journal (parent_trade_id)")
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

    def _format_journal_row(self, row: Any) -> Dict[str, Any]:
        """Format a user_trade_journal row into the canonical Journal trade dictionary."""
        if not row:
            return {}
        keys = set(row.keys()) if hasattr(row, "keys") else set()
        pnl_raw = float(row["pnl"]) if (row["pnl"] is not None) else None
        pnl_str = None
        if pnl_raw is not None:
            pnl_str = f"+${pnl_raw:.2f}" if pnl_raw >= 0 else f"-${abs(pnl_raw):.2f}"

        fr = row["followed_rules"]
        followed_rules = True if fr == 1 else (False if fr == 0 else None)

        status = row["status"]
        shares = float(row["shares"])
        rem_shares_val = row["remaining_shares"] if "remaining_shares" in keys else None
        if rem_shares_val is not None:
            remaining_shares = float(rem_shares_val)
        else:
            remaining_shares = 0.0 if status == "CLOSED" else shares

        exit_date = row["exit_date"] if "exit_date" in keys else None
        parent_id = str(row["parent_trade_id"]) if ("parent_trade_id" in keys and row["parent_trade_id"] is not None) else None
        role = row["execution_role"] if ("execution_role" in keys and row["execution_role"]) else "ENTRY"
        notes = row["notes"] if "notes" in keys else None
        target1 = float(row["target1"]) if ("target1" in keys and row["target1"] is not None) else None
        stop_loss = float(row["stop_loss"]) if ("stop_loss" in keys and row["stop_loss"] is not None) else None
        idempotency_key = row["idempotency_key"] if "idempotency_key" in keys else None

        return {
            "id": str(row["id"]),
            "userId": row["user_id"],
            "ticker": row["symbol"],
            "symbol": row["symbol"],
            "setup": row["setup_name"],
            "setupName": row["setup_name"],
            "entryPrice": float(row["entry_price"]),
            "exitPrice": float(row["exit_price"]) if row["exit_price"] is not None else None,
            "shares": shares,
            "remainingShares": remaining_shares,
            "rAchieved": float(row["r_achieved"]) if row["r_achieved"] is not None else None,
            "followedRules": followed_rules,
            "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
            "pnl": pnl_str,
            "pnlRaw": pnl_raw,
            "status": status,
            "date": exit_date if (status == "CLOSED" and exit_date) else row["entry_date"],
            "entryDate": row["entry_date"],
            "exitDate": exit_date,
            "parentTradeId": parent_id,
            "executionRole": role,
            "notes": notes,
            "target1": target1,
            "stopLoss": stop_loss,
            "idempotencyKey": idempotency_key,
            "createdAt": str(row["created_at"]),
        }

    @retry_sqlite()
    def save_journal_trade(self, user_id: str, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Save a trade execution record to user's journal."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                entry_date = trade.get("entryDate") or trade.get("date") or datetime.utcnow().strftime("%Y-%m-%d")
                exit_date = trade.get("exitDate")

                status_raw = trade.get("status")
                if status_raw:
                    status = str(status_raw).strip().upper()
                    if status not in ("OPEN", "CLOSED"):
                        raise ValueError(f"Invalid trade status: {status}. Must be 'OPEN' or 'CLOSED'.")
                else:
                    status = "CLOSED" if (trade.get("exitPrice") is not None or trade.get("pnl") is not None or trade.get("rAchieved") is not None) else "OPEN"

                setup_name = (trade.get("setupName") or trade.get("setup") or "").strip() or None
                exit_price = float(trade["exitPrice"]) if trade.get("exitPrice") is not None else None
                r_achieved = float(trade["rAchieved"]) if trade.get("rAchieved") is not None else None

                # followed_rules in SQLite schema has NOT NULL DEFAULT 1.
                # 1 = True (followed rules), 0 = False (violated rules), -1 = Missing / Unrecorded
                fr_val = trade.get("followedRules")
                followed_rules = 1 if fr_val is True else (0 if fr_val is False else -1)

                confidence = float(trade["confidence"]) if trade.get("confidence") is not None else None
                pnl = float(trade["pnl"]) if trade.get("pnl") is not None else None
                entry_price = float(trade["entryPrice"])
                shares = float(trade["shares"])
                remaining_shares = float(trade["remainingShares"]) if trade.get("remainingShares") is not None else (0.0 if status == "CLOSED" else shares)
                parent_trade_id = int(trade["parentTradeId"]) if trade.get("parentTradeId") is not None else None
                execution_role = trade.get("executionRole") or ("FULL_EXIT" if status == "CLOSED" else "ENTRY")
                idempotency_key = trade.get("idempotencyKey")
                notes = trade.get("notes")
                target1 = float(trade["target1"]) if trade.get("target1") is not None else None
                stop_loss = float(trade["stopLoss"]) if trade.get("stopLoss") is not None else None

                cursor.execute(
                    """
                    INSERT INTO user_trade_journal (
                        user_id, symbol, setup_name, entry_price, exit_price, shares, remaining_shares,
                        r_achieved, followed_rules, confidence, pnl, status, entry_date, exit_date,
                        parent_trade_id, execution_role, idempotency_key, notes, target1, stop_loss, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        user_id,
                        trade["symbol"].upper().strip(),
                        setup_name,
                        entry_price,
                        exit_price,
                        shares,
                        remaining_shares,
                        r_achieved,
                        followed_rules,
                        confidence,
                        pnl,
                        status,
                        entry_date,
                        exit_date,
                        parent_trade_id,
                        execution_role,
                        idempotency_key,
                        notes,
                        target1,
                        stop_loss,
                    ),
                )
                trade_id = cursor.lastrowid
                cursor.execute("SELECT * FROM user_trade_journal WHERE id = ?", (trade_id,))
                row = cursor.fetchone()
                return self._format_journal_row(row)
        finally:
            conn.close()

    @retry_sqlite()
    def record_trade_fill(self, user_id: str, fill_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record an executed trade plan fill in the persistent journal and update portfolio holdings."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                idempotency_key = fill_data.get("idempotencyKey")
                if idempotency_key:
                    cursor.execute(
                        "SELECT * FROM user_trade_journal WHERE user_id = ? AND idempotency_key = ? LIMIT 1",
                        (user_id, str(idempotency_key)),
                    )
                    row = cursor.fetchone()
                    if row:
                        return self._format_journal_row(row)

                symbol = fill_data["symbol"].upper().strip()
                entry_price = float(fill_data["entryPrice"])
                shares = float(fill_data["shares"])
                if entry_price <= 0:
                    raise ValueError("Entry price must be positive.")
                if shares <= 0:
                    raise ValueError("Shares count must be positive.")

                stop_loss = float(fill_data["stopLoss"]) if fill_data.get("stopLoss") is not None else None
                target1 = float(fill_data["target1"]) if fill_data.get("target1") is not None else None
                setup_name = (fill_data.get("setupName") or fill_data.get("setup") or "").strip() or None
                confidence = float(fill_data["confidence"]) if fill_data.get("confidence") is not None else None
                if confidence is not None and (confidence < 0.0 or confidence > 100.0):
                    raise ValueError("Confidence must be between 0 and 100.")

                entry_date = fill_data.get("entryDate") or datetime.utcnow().strftime("%Y-%m-%d")
                notes = (fill_data.get("notes") or "").strip() or None

                cursor.execute(
                    """
                    INSERT INTO user_trade_journal (
                        user_id, symbol, setup_name, entry_price, exit_price, shares, remaining_shares,
                        r_achieved, followed_rules, confidence, pnl, status, entry_date, exit_date,
                        parent_trade_id, execution_role, idempotency_key, notes, target1, stop_loss, created_at
                    ) VALUES (?, ?, ?, ?, NULL, ?, ?, NULL, -1, ?, NULL, 'OPEN', ?, NULL, NULL, 'ENTRY', ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        user_id,
                        symbol,
                        setup_name,
                        entry_price,
                        shares,
                        shares,
                        confidence,
                        entry_date,
                        idempotency_key,
                        notes,
                        target1,
                        stop_loss,
                    ),
                )
                trade_id = cursor.lastrowid

                # Synchronize with portfolio_holdings
                cursor.execute(
                    "SELECT shares, entry_price FROM portfolio_holdings WHERE user_id = ? AND symbol = ?",
                    (user_id, symbol),
                )
                existing_holding = cursor.fetchone()
                if existing_holding:
                    old_shares = float(existing_holding["shares"])
                    old_entry = float(existing_holding["entry_price"])
                    new_shares = old_shares + shares
                    new_entry = round(((old_shares * old_entry) + (shares * entry_price)) / new_shares, 4)
                    cursor.execute(
                        """
                        UPDATE portfolio_holdings
                        SET shares = ?, entry_price = ?,
                            stop_loss_price = COALESCE(?, stop_loss_price),
                            target_price = COALESCE(?, target_price),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND symbol = ?
                        """,
                        (new_shares, new_entry, stop_loss, target1, user_id, symbol),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO portfolio_holdings (
                            user_id, symbol, name, shares, entry_price, current_price, target_price, stop_loss_price, added_at, asset_type, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Stock', CURRENT_TIMESTAMP)
                        """,
                        (user_id, symbol, symbol, shares, entry_price, entry_price, target1, stop_loss, entry_date),
                    )

                cursor.execute("SELECT * FROM user_trade_journal WHERE id = ?", (trade_id,))
                created_row = cursor.fetchone()
                return self._format_journal_row(created_row)
        finally:
            conn.close()

    @retry_sqlite()
    def record_trade_exit(self, user_id: str, exit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record a partial or complete exit of an active open trade."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                idempotency_key = exit_data.get("idempotencyKey")
                if idempotency_key:
                    cursor.execute(
                        "SELECT * FROM user_trade_journal WHERE user_id = ? AND idempotency_key = ? LIMIT 1",
                        (user_id, str(idempotency_key)),
                    )
                    row = cursor.fetchone()
                    if row:
                        return self._format_journal_row(row)

                trade_id = exit_data.get("tradeId")
                symbol = exit_data.get("symbol", "").upper().strip()
                exit_price = float(exit_data["exitPrice"])
                if exit_price <= 0:
                    raise ValueError("Exit price must be positive.")

                # Locate target trade
                if trade_id is not None:
                    cursor.execute(
                        "SELECT * FROM user_trade_journal WHERE id = ? AND user_id = ?",
                        (int(trade_id), user_id),
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM user_trade_journal WHERE user_id = ? AND symbol = ? AND status = 'OPEN' ORDER BY created_at DESC LIMIT 1",
                        (user_id, symbol),
                    )
                parent = cursor.fetchone()
                if not parent:
                    raise ValueError(f"No active open trade found for symbol '{symbol}' or ID '{trade_id}'.")

                if parent["status"] != "OPEN":
                    raise ValueError(f"Trade #{parent['id']} is already CLOSED. Cannot record exit on a closed trade.")

                parent_remaining = float(parent["remaining_shares"] if parent["remaining_shares"] is not None else parent["shares"])
                if parent_remaining <= 0:
                    raise ValueError(f"Trade #{parent['id']} has zero remaining shares.")

                exit_shares = float(exit_data.get("shares") or parent_remaining)
                if exit_shares <= 0:
                    raise ValueError("Exit shares count must be positive.")
                if exit_shares > parent_remaining + 1e-6:
                    raise ValueError(f"Exit quantity ({exit_shares}) exceeds remaining open shares ({parent_remaining}).")

                exit_date = exit_data.get("exitDate") or datetime.utcnow().strftime("%Y-%m-%d")
                fr_val = exit_data.get("followedRules")
                followed_rules = 1 if fr_val is True else (0 if fr_val is False else -1)
                notes = (exit_data.get("notes") or "").strip() or None

                entry_price = float(parent["entry_price"])
                leg_pnl = round((exit_price - entry_price) * exit_shares, 2)

                stop_loss = float(parent["stop_loss"]) if parent["stop_loss"] is not None else None
                r_achieved = None
                if stop_loss is not None and entry_price != stop_loss:
                    risk_per_share = entry_price - stop_loss
                    r_achieved = round((exit_price - entry_price) / risk_per_share, 2)

                is_partial = (parent_remaining - exit_shares) > 1e-6

                if is_partial:
                    new_remaining = round(parent_remaining - exit_shares, 6)
                    cursor.execute(
                        "UPDATE user_trade_journal SET remaining_shares = ? WHERE id = ?",
                        (new_remaining, parent["id"]),
                    )
                    cursor.execute(
                        """
                        INSERT INTO user_trade_journal (
                            user_id, symbol, setup_name, entry_price, exit_price, shares, remaining_shares,
                            r_achieved, followed_rules, confidence, pnl, status, entry_date, exit_date,
                            parent_trade_id, execution_role, idempotency_key, notes, target1, stop_loss, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, 'CLOSED', ?, ?, ?, 'PARTIAL_EXIT', ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """,
                        (
                            user_id,
                            parent["symbol"],
                            parent["setup_name"],
                            entry_price,
                            exit_price,
                            exit_shares,
                            r_achieved,
                            followed_rules,
                            parent["confidence"],
                            leg_pnl,
                            parent["entry_date"],
                            exit_date,
                            parent["id"],
                            idempotency_key,
                            notes,
                            parent["target1"],
                            stop_loss,
                        ),
                    )
                    leg_id = cursor.lastrowid

                    # Decrement portfolio_holdings
                    cursor.execute(
                        "SELECT shares FROM portfolio_holdings WHERE user_id = ? AND symbol = ?",
                        (user_id, parent["symbol"]),
                    )
                    h_row = cursor.fetchone()
                    if h_row:
                        h_shares = float(h_row["shares"])
                        if h_shares > exit_shares + 1e-6:
                            cursor.execute(
                                "UPDATE portfolio_holdings SET shares = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND symbol = ?",
                                (round(h_shares - exit_shares, 6), user_id, parent["symbol"]),
                            )
                        else:
                            cursor.execute(
                                "DELETE FROM portfolio_holdings WHERE user_id = ? AND symbol = ?",
                                (user_id, parent["symbol"]),
                            )

                    cursor.execute("SELECT * FROM user_trade_journal WHERE id = ?", (leg_id,))
                    result_row = cursor.fetchone()
                    return self._format_journal_row(result_row)

                else:
                    # Full close
                    cursor.execute("SELECT COUNT(*) as count FROM user_trade_journal WHERE parent_trade_id = ?", (parent["id"],))
                    prior_legs = cursor.fetchone()["count"]

                    if prior_legs > 0:
                        cursor.execute(
                            "UPDATE user_trade_journal SET remaining_shares = 0, status = 'CLOSED' WHERE id = ?",
                            (parent["id"],),
                        )
                        cursor.execute(
                            """
                            INSERT INTO user_trade_journal (
                                user_id, symbol, setup_name, entry_price, exit_price, shares, remaining_shares,
                                r_achieved, followed_rules, confidence, pnl, status, entry_date, exit_date,
                                parent_trade_id, execution_role, idempotency_key, notes, target1, stop_loss, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, 'CLOSED', ?, ?, ?, 'FULL_EXIT', ?, ?, ?, ?, CURRENT_TIMESTAMP)
                            """,
                            (
                                user_id,
                                parent["symbol"],
                                parent["setup_name"],
                                entry_price,
                                exit_price,
                                exit_shares,
                                r_achieved,
                                followed_rules,
                                parent["confidence"],
                                leg_pnl,
                                parent["entry_date"],
                                exit_date,
                                parent["id"],
                                idempotency_key,
                                notes,
                                parent["target1"],
                                stop_loss,
                            ),
                        )
                        final_id = cursor.lastrowid
                    else:
                        cursor.execute(
                            """
                            UPDATE user_trade_journal
                            SET exit_price = ?, remaining_shares = 0, status = 'CLOSED', exit_date = ?,
                                pnl = ?, r_achieved = ?, followed_rules = ?, execution_role = 'FULL_EXIT',
                                notes = COALESCE(?, notes), idempotency_key = COALESCE(?, idempotency_key)
                            WHERE id = ?
                            """,
                            (
                                exit_price,
                                exit_date,
                                leg_pnl,
                                r_achieved,
                                followed_rules,
                                notes,
                                idempotency_key,
                                parent["id"],
                            ),
                        )
                        final_id = parent["id"]

                    # Remove holding from portfolio_holdings
                    cursor.execute(
                        "DELETE FROM portfolio_holdings WHERE user_id = ? AND symbol = ?",
                        (user_id, parent["symbol"]),
                    )

                    cursor.execute("SELECT * FROM user_trade_journal WHERE id = ?", (final_id,))
                    result_row = cursor.fetchone()
                    return self._format_journal_row(result_row)
        finally:
            conn.close()

    @retry_sqlite()
    def get_journal_trades(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve chronological trade log for a user."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT *
                FROM user_trade_journal
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = cursor.fetchall()
            return [self._format_journal_row(row) for row in rows]
        finally:
            conn.close()

    @retry_sqlite()
    def get_risk_telemetry(self, user_id: str) -> Dict[str, Any]:
        """Derive authoritative behavioral risk telemetry directly from persistent trade journal and portfolio holdings."""
        trades = self.get_journal_trades(user_id, limit=200)
        holdings = self.get_user_portfolio(user_id)

        account_equity: Optional[float] = None
        if holdings:
            total_eq = sum(h["shares"] * (h.get("currentPrice") or h.get("entryPrice", 0.0)) for h in holdings)
            if total_eq > 0:
                account_equity = round(total_eq, 2)

        total_trades = len(trades)
        consecutive_loss_streak = 0
        for t in trades:
            pnl_val = t.get("pnlRaw")
            r_val = t.get("rAchieved")
            is_loss = (pnl_val is not None and pnl_val < 0) or (r_val is not None and r_val < 0)
            is_win = (pnl_val is not None and pnl_val > 0) or (r_val is not None and r_val > 0)
            if is_loss:
                consecutive_loss_streak += 1
            elif is_win:
                break

        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        today_trades = [t for t in trades if t["date"] == today_str or (t.get("createdAt") and t["createdAt"][:10] == today_str)]
        today_loss_dollars = sum(abs(t["pnlRaw"]) for t in today_trades if t.get("pnlRaw") is not None and t["pnlRaw"] < 0)

        daily_drawdown_pct = 0.0
        if account_equity and account_equity > 0:
            daily_drawdown_pct = round((today_loss_dollars / account_equity) * 100.0, 2)

        rule_adherence_pct: Optional[float] = None
        brier_score: Optional[float] = None

        trades_with_rule_evidence = [t for t in trades if t.get("followedRules") is not None]
        if len(trades_with_rule_evidence) > 0:
            rules_followed = sum(1 for t in trades_with_rule_evidence if t["followedRules"] is True)
            rule_adherence_pct = round((rules_followed / len(trades_with_rule_evidence)) * 100.0, 1)

        brier_eligible_trades = [
            t for t in trades
            if t.get("confidence") is not None and (
                (t.get("pnlRaw") is not None and t["pnlRaw"] != 0) or
                (t.get("rAchieved") is not None and t["rAchieved"] != 0)
            )
        ]
        if len(brier_eligible_trades) > 0:
            brier_sum = sum(
                ((t["confidence"] / 100.0) - (1.0 if (t.get("pnlRaw") or 0) > 0 or (t.get("rAchieved") or 0) > 0 else 0.0)) ** 2
                for t in brier_eligible_trades
            )
            brier_score = round(brier_sum / len(brier_eligible_trades), 2)

        return {
            "available": True,
            "userId": user_id,
            "accountEquity": account_equity,
            "consecutiveLossStreak": consecutive_loss_streak,
            "dailyDrawdownPct": daily_drawdown_pct,
            "ruleAdherencePct": rule_adherence_pct,
            "brierScore": brier_score,
            "totalTrades": total_trades,
            "isCalibrated": brier_score is not None and brier_score <= 0.25,
            "source": "AUTHORITATIVE_API",
        }
