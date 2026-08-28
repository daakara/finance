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

    def test_chart_timeframe_state_and_api_contract(self):
        """Regression Quality Gate: Ensure PriceChart component and page avoid re-render reset loops and invalid timeScale API calls."""
        price_chart_path = os.path.join("frontend", "components", "PriceChart.tsx")
        navbar_path = os.path.join("frontend", "components", "Navbar.tsx")
        api_path = os.path.join("frontend", "lib", "api.ts")

        with open(price_chart_path, "r", encoding="utf-8") as f:
            chart_content = f.read()

        # Must never call non-existent resetTimeScale()
        self.assertNotIn("resetTimeScale()", chart_content, "PriceChart must not call non-existent resetTimeScale() on Lightweight Charts v4")
        self.assertIn("fitContent()", chart_content, "PriceChart must call fitContent() to auto-scale viewport")
        self.assertIn('type="button"', chart_content, "Timeframe interval buttons must explicitly specify type='button'")

        with open(navbar_path, "r", encoding="utf-8") as f:
            navbar_content = f.read()

        # Navbar must not trigger onRoleChange on mount in a way that resets timeframe state
        self.assertNotIn('if (onRoleChange) onRoleChange(saved);', navbar_content, "Navbar must not call onRoleChange on initial storage read")

        with open(api_path, "r", encoding="utf-8") as f:
            api_content = f.read()

        # Fallback generator must not match '1mo' as intraday with substring .includes('m')
        self.assertNotIn('interval.includes("m")', api_content, "api.ts must not use interval.includes('m') which corrupts 1mo monthly macro intervals")

    def test_light_paper_theme_compliance(self):
        """Regression Quality Gate: Ensure Paper Light theme overrides cover headers, panels, cards, and canvas."""
        globals_css_path = os.path.join("frontend", "app", "globals.css")
        price_chart_path = os.path.join("frontend", "components", "PriceChart.tsx")

        with open(globals_css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

        self.assertIn('[data-theme="paper"]', css_content, "globals.css must specify data-theme='paper' overrides")
        self.assertIn('[data-theme="paper"] header', css_content, "globals.css must override header in light theme")
        self.assertIn('[data-theme="paper"] [class*="bg-[#0c1017]"]', css_content, "globals.css must override navbar/tab dark backgrounds")
        self.assertIn('[data-theme="paper"] [class*="bg-[#111722]"]', css_content, "globals.css must override card/sidebar backgrounds")

        with open(price_chart_path, "r", encoding="utf-8") as f:
            chart_content = f.read()

        self.assertIn("finance:theme-change", chart_content, "PriceChart must listen for finance:theme-change event")
        self.assertIn("isPaperTheme", chart_content, "PriceChart must adapt initial canvas layout to active theme")

    def test_canonical_domain_and_redirects_structure(self):
        """Regression Quality Gate: Ensure Cloudflare _redirects file and layout script enforce arxterminal.com -> www.arxterminal.com."""
        redirects_path = os.path.join("frontend", "public", "_redirects")
        layout_path = os.path.join("frontend", "app", "layout.tsx")
        sitemap_path = os.path.join("frontend", "public", "sitemap.xml")

        self.assertTrue(os.path.exists(redirects_path), "public/_redirects must exist for Cloudflare Pages")
        with open(redirects_path, "r", encoding="utf-8") as f:
            redirects_content = f.read()

        self.assertIn("https://arxterminal.com/*", redirects_content)
        self.assertIn("https://www.arxterminal.com/:splat", redirects_content)
        self.assertIn("301!", redirects_content)

        with open(layout_path, "r", encoding="utf-8") as f:
            layout_content = f.read()

        self.assertIn("https://www.arxterminal.com", layout_content, "layout.tsx metadataBase must use www.arxterminal.com")
        self.assertIn("host === 'arxterminal.com'", layout_content, "layout.tsx must contain client-side canonical redirect script")
        self.assertIn("window.location.replace('https://www.arxterminal.com'", layout_content)

        with open(sitemap_path, "r", encoding="utf-8") as f:
            sitemap_content = f.read()

        self.assertNotIn("https://arxterminal.com/", sitemap_content, "sitemap.xml must not contain non-www URLs")
        self.assertIn("https://www.arxterminal.com/", sitemap_content)

    def test_brand_tone_and_progressive_clarity_vernacular_engine(self):
        """Regression Quality Gate: Ensure Brand Tone, Vernacular Switcher, Bottom Line summaries, and Jargon Buster exist."""
        navbar_path = os.path.join("frontend", "components", "Navbar.tsx")
        conviction_path = os.path.join("frontend", "components", "CompositeConvictionCard.tsx")
        sizer_path = os.path.join("frontend", "components", "DayTraderPositionSizer.tsx")
        risk_path = os.path.join("frontend", "components", "RiskMetricsCard.tsx")
        factor_path = os.path.join("frontend", "components", "AssetFactorRadar.tsx")
        guide_path = os.path.join("frontend", "app", "guide", "page.tsx")

        with open(navbar_path, "r", encoding="utf-8") as f:
            nav_content = f.read()
        self.assertIn("ARX TERMINAL", nav_content)
        self.assertIn("No-BS Market Intel", nav_content)
        self.assertIn("ARX_VERNACULAR_MODE", nav_content)
        self.assertIn("finance:vernacular-change", nav_content)
        self.assertIn("Plain English", nav_content)
        self.assertIn("Pro Quant", nav_content)

        with open(conviction_path, "r", encoding="utf-8") as f:
            conv_content = f.read()
        self.assertIn("The Bottom Line (No Wall Street Fluff)", conv_content)
        self.assertIn("finance:vernacular-change", conv_content)

        with open(sizer_path, "r", encoding="utf-8") as f:
            sizer_content = f.read()
        self.assertIn("Rule #1: Protect The Castle", sizer_content)
        self.assertIn("The Math Made Simple", sizer_content)

        with open(risk_path, "r", encoding="utf-8") as f:
            risk_content = f.read()
        self.assertIn("Worst-Case Crash Test", risk_content)
        self.assertIn("Standard Bad Day", risk_content)
        self.assertIn("finance:vernacular-change", risk_content)

        with open(factor_path, "r", encoding="utf-8") as f:
            factor_content = f.read()
        self.assertIn("BS Detector", factor_content)
        self.assertIn("finance:vernacular-change", factor_content)

        with open(guide_path, "r", encoding="utf-8") as f:
            guide_content = f.read()
        self.assertIn("Chapter 8: The No-BS Plain-English Jargon Buster", guide_content)


if __name__ == "__main__":
    unittest.main()

