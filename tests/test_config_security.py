"""Unit tests for verifying SSL and security configurations across the platform."""

import unittest
import os
import ssl

from ssl_config import create_ssl_context, create_session_with_retries
from config import Config


class TestSecurityConfiguration(unittest.TestCase):
    """Test suite to ensure security defaults are enforced."""

    def test_ssl_environment_variables_not_disabled(self):
        """Verify that global SSL bypass environment variables are not set to disable verification."""
        self.assertNotEqual(os.environ.get("CURL_DISABLE_SSL_VERIFY"), "1")
        self.assertNotEqual(os.environ.get("PYTHONHTTPSVERIFY"), "0")

    def test_create_ssl_context_enforces_verification(self):
        """Verify that default SSL context requires certificate verification."""
        context = create_ssl_context()
        self.assertIsNotNone(context)
        self.assertNotEqual(context.verify_mode, ssl.CERT_NONE)

    def test_create_session_with_retries_has_verification_enabled(self):
        """Verify that requests sessions verify SSL certificates."""
        session = create_session_with_retries()
        self.assertNotEqual(session.verify, False)

    def test_config_defaults(self):
        """Verify default configuration flags."""
        self.assertFalse(Config.DISABLE_SSL_VERIFY)


if __name__ == "__main__":
    unittest.main()

