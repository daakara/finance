import os
import sys
import json
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from playwright.sync_api import sync_playwright

def verify_poc_page():
    console_logs = []
    page_errors = []
    failed_requests = []

    artifact_dir = r"C:\Users\akara\.gemini\antigravity\brain\1f9bcc84-ac95-4b1d-baf9-b4d4d76f13c7"
    screenshot_dir = os.path.join(artifact_dir, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    evidence_dir = r"C:\Users\akara\Documents\Projects\finance-arx-market-wide-top35\evidence\visual_acceptance\poc_market_wide"
    os.makedirs(evidence_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1440, 'height': 1050},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        
        # Bypass first-time onboarding tour modal
        context.add_init_script("""
            try {
                localStorage.setItem("FINANCE_ONBOARDING_COMPLETED", "true");
                localStorage.setItem("ARX_TOUR_DISMISSED", "true");
            } catch (e) {}
        """)

        page = context.new_page()

        page.on('console', lambda msg: console_logs.append({'type': msg.type, 'text': msg.text}))
        page.on('pageerror', lambda exc: page_errors.append(str(exc)))
        page.on('requestfailed', lambda req: failed_requests.append({'url': req.url, 'failure': str(req.failure)}))

        target_url = "http://127.0.0.1:3042/poc-market-wide/"
        print(f"Navigating to {target_url}...")
        resp = page.goto(target_url, wait_until='networkidle', timeout=15000)
        print("Page HTTP Status:", resp.status if resp else "None")
        print("Page Title:", page.title())

        # Wait for candidate cards
        page.wait_for_selector("text=EXPERIMENTAL MARKET-WIDE RADAR", timeout=10000)
        
        # Verify text content
        has_banner = page.locator("text=EXPERIMENTAL MARKET-WIDE RADAR — SHADOW POC").is_visible()
        has_warning = page.locator("text=MARKET_WIDE_COVERAGE_UNPROVEN").is_visible()
        has_funnel = page.locator("text=Source Market").is_visible()
        has_incremental_tab = page.locator("text=Incremental Discoveries (30)").is_visible()

        print(f"Banner visible: {has_banner}")
        print(f"Warning visible: {has_warning}")
        print(f"Funnel metrics visible: {has_funnel}")
        print(f"Incremental tab visible: {has_incremental_tab}")

        # Capture primary screenshot
        screen_path_brain = os.path.join(screenshot_dir, "poc_market_wide_dashboard.png")
        screen_path_evidence = os.path.join(evidence_dir, "poc_market_wide_dashboard.png")
        page.screenshot(path=screen_path_brain, full_page=True)
        page.screenshot(path=screen_path_evidence, full_page=True)
        print(f"Screenshot captured to {screen_path_brain}")

        # Click incremental tab and capture that view too
        page.locator("text=Incremental Discoveries (30)").click()
        time.sleep(1.0)
        screen_inc_brain = os.path.join(screenshot_dir, "poc_market_wide_incremental.png")
        screen_inc_evidence = os.path.join(evidence_dir, "poc_market_wide_incremental.png")
        page.screenshot(path=screen_inc_brain, full_page=True)
        page.screenshot(path=screen_inc_evidence, full_page=True)
        print(f"Incremental tab screenshot captured to {screen_inc_brain}")

        browser.close()

        assert has_banner, "Banner was not visible"
        assert has_warning, "Warning tag was not visible"
        assert has_funnel, "Funnel metrics were not visible"
        assert len(page_errors) == 0, f"Page errors found: {page_errors}"
        print("PLAYWRIGHT_VERIFICATION_SUCCESSFUL")

if __name__ == "__main__":
    verify_poc_page()
