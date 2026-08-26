"""Unit test verifying Next.js TypeScript frontend structure, route contracts, and deep-link query parameter bindings."""

import unittest
import os
import json
import re


class TestNextJsFrontendStructure(unittest.TestCase):
    """Test suite verifying Next.js application scaffold, deep links, and route parameter bindings."""

    def test_package_json_dependencies(self):
        """Verify package.json specifies Next.js, React, and TradingView lightweight-charts."""
        pkg_path = os.path.join("frontend", "package.json")
        self.assertTrue(os.path.exists(pkg_path))

        with open(pkg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        deps = data.get("dependencies", {})
        self.assertIn("next", deps)
        self.assertIn("react", deps)
        self.assertIn("lightweight-charts", deps)

    def test_app_routes_exist(self):
        """Verify all core Next.js routes exist."""
        self.assertTrue(os.path.exists(os.path.join("frontend", "app", "layout.tsx")))
        self.assertTrue(os.path.exists(os.path.join("frontend", "app", "page.tsx")))
        self.assertTrue(os.path.exists(os.path.join("frontend", "app", "screener", "page.tsx")))
        self.assertTrue(os.path.exists(os.path.join("frontend", "app", "compare", "page.tsx")))
        self.assertTrue(os.path.exists(os.path.join("frontend", "lib", "api.ts")))

    def test_terminal_deep_link_query_param_binding(self):
        """Regression Quality Gate: Ensure frontend/app/page.tsx strictly extracts and syncs ?symbol= query parameters via useSearchParams."""
        page_path = os.path.join("frontend", "app", "page.tsx")
        self.assertTrue(os.path.exists(page_path))

        with open(page_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Must import useSearchParams and Suspense
        self.assertIn("useSearchParams", content, "Terminal page must import and use useSearchParams for deep linking")
        self.assertIn("Suspense", content, "Terminal page must wrap search param content in Next.js Suspense")

        # Must extract 'symbol' query parameter
        self.assertIn('searchParams.get("symbol")', content, "Terminal page must extract symbol parameter from URL query")

        # Must synchronize state when urlSymbol changes
        self.assertIn("setSelectedSymbol", content)

    def test_cross_page_links_match_destination_parameters(self):
        """Verify all Link href destinations in screener and compare pages pass valid symbol query format."""
        for subpage in ["screener", "compare"]:
            page_path = os.path.join("frontend", "app", subpage, "page.tsx")
            self.assertTrue(os.path.exists(page_path))

            with open(page_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Must contain links to root terminal with symbol query
            link_pattern = re.compile(r'href=\{`/\?symbol=\${[a-zA-Z0-9_\.]+\}`\}')
            matches = link_pattern.findall(content)
            self.assertTrue(len(matches) > 0, f"Page {subpage} must contain dynamic deep links in format /?symbol=${{...}}")

    def test_dual_horizon_lens_implementation_on_all_views(self):
        """Verify that all core interactive views implement the Dual-Horizon role toggle."""
        views_to_check = [
            os.path.join("frontend", "app", "page.tsx"),
            os.path.join("frontend", "app", "screener", "page.tsx"),
            os.path.join("frontend", "app", "compare", "page.tsx"),
        ]

        for view_path in views_to_check:
            self.assertTrue(os.path.exists(view_path))
            with open(view_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Must check or bind localStorage FINANCE_USER_ROLE
            self.assertTrue(
                "FINANCE_USER_ROLE" in content or "userRole" in content,
                f"View {view_path} must implement Dual-Horizon user role integration"
            )


if __name__ == "__main__":
    unittest.main()
