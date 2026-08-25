"""Unit test verifying Next.js TypeScript frontend structure and build components."""

import unittest
import os
import json


class TestNextJsFrontendStructure(unittest.TestCase):
    """Test suite verifying Next.js application scaffold."""

    def test_package_json_dependencies(self):
        """Verify package.json specifies Next.js, React, and TradingView lightweight-charts."""
        pkg_path = os.path.join("frontend", "package.json")
        self.assertTrue(os.path.exists(pkg_path))

        with open(pkg_path, "r") as f:
          data = json.load(f)

        deps = data.get("dependencies", {})
        self.assertIn("next", deps)
        self.assertIn("react", deps)
        self.assertIn("lightweight-charts", deps)

    def test_app_routes_exist(self):
        """Verify main Next.js routes exist."""
        self.assertTrue(os.path.exists(os.path.join("frontend", "app", "layout.tsx")))
        self.assertTrue(os.path.exists(os.path.join("frontend", "app", "page.tsx")))
        self.assertTrue(os.path.exists(os.path.join("frontend", "app", "screener", "page.tsx")))
        self.assertTrue(os.path.exists(os.path.join("frontend", "components", "TradingViewChart.tsx")))
        self.assertTrue(os.path.exists(os.path.join("frontend", "lib", "api.ts")))


if __name__ == "__main__":
    unittest.main()

