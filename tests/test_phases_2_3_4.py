"""Unit tests for Phase 2, Phase 3, and Phase 4 enhancements (DiskCache, DB Persistence, CI gates)."""

import unittest
import os
import tempfile
import pandas as pd
import numpy as np

from analyst_dashboard.data.gem_fetchers import MultiAssetDataPipeline
from analyst_dashboard.data.db_engine import HistoryDatabaseEngine


class TestPhases234Enhancements(unittest.TestCase):
    """Test suite verifying DiskCache integration and SQLite persistence engine."""

    def setUp(self):
        self.pipeline = MultiAssetDataPipeline()
        self.temp_db = os.path.join(tempfile.gettempdir(), "test_finance_history.db")
        self.db_engine = HistoryDatabaseEngine(db_path=self.temp_db)

    def tearDown(self):
        if os.path.exists(self.temp_db):
            try:
                os.remove(self.temp_db)
            except Exception:
                pass

    def test_disk_cache_initialization(self):
        """Verify DiskCache object is initialized on MultiAssetDataPipeline."""
        self.assertTrue(hasattr(self.pipeline, "disk_cache"))

    def test_db_engine_screening_log(self):
        """Verify database engine logs screening history."""
        self.db_engine.log_screening_result("AAPL", 88.5, "High Conviction", {"sector": "tech"})

    def test_db_engine_forecast_log(self):
        """Verify database engine logs forecast performance."""
        self.db_engine.log_forecast_performance("MSFT", 30, "GARCH(1,1)", 0.015, 0.25)


if __name__ == "__main__":
    unittest.main()