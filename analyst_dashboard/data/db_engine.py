"""Database Schema & Persistence Engine for Historical Analytics & Quality Drift Monitoring."""

import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.expanduser("~"), ".finance_platform_history.db")


class HistoryDatabaseEngine:
    """SQLite-backed persistent database engine for historical screening and forecast logs."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        """Initialize database schema tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize history database: {e}")

    def log_screening_result(self, ticker: str, composite_score: float, risk_rating: str, data: Optional[Dict] = None):
        """Save a screening result into persistent historical database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO gem_screening_history (ticker, composite_score, risk_rating, raw_data) VALUES (?, ?, ?, ?)",
                    (ticker, composite_score, risk_rating, json.dumps(data or {})),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging screening result for {ticker}: {e}")

    def log_forecast_performance(self, ticker: str, horizon: int, model_type: str, rmse: float, qlike: float):
        """Save forecast performance evaluation into persistent log."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO forecast_history (ticker, horizon_days, model_type, rmse, qlike_loss) VALUES (?, ?, ?, ?, ?)",
                    (ticker, horizon, model_type, rmse, qlike),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging forecast performance for {ticker}: {e}")

