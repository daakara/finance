"""Unit tests for Phase 1 AppSettings configuration validation."""

import unittest
from config import Config, AppSettings


class TestPhase1ConfigValidation(unittest.TestCase):
    """Test suite verifying AppSettings validation and defaults."""

    def test_app_settings_types_and_defaults(self):
        """Verify AppSettings initializes with correct default types."""
        settings = AppSettings()
        self.assertIsInstance(settings.cache_ttl_seconds, int)
        self.assertIsInstance(settings.max_cache_size, int)
        self.assertIsInstance(settings.max_symbols_per_request, int)
        self.assertIsInstance(settings.disable_ssl_verify, bool)
        self.assertIsInstance(settings.use_sample_data, bool)

    def test_config_class_exports(self):
        """Verify Config class exports validated AppSettings attributes."""
        self.assertFalse(Config.DISABLE_SSL_VERIFY)
        self.assertEqual(Config.DEFAULT_PERIOD, "1y")
        self.assertEqual(Config.DEFAULT_INTERVAL, "1d")


if __name__ == "__main__":
    unittest.main()

