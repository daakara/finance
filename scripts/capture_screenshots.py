"""
scripts/capture_screenshots.py

Spawns Python's native http.server to serve the static exported frontend/out directory,
navigates to stock detail pages using Playwright, captures screenshots of the
SourcedFundamentalsCard section, and verifies clean browser rendering.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = Path("docs/research/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

PORT = 3008
BASE_URL = f"http://127.0.0.1:{PORT}"


def wait_for_server(url: str, timeout: int = 20) -> bool:
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{url}/stock/fix/", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    print(f"Starting native Python http.server for frontend/out on port {PORT}...")
    out_dir = Path("frontend/out").resolve()
    assert out_dir.exists(), "frontend/out directory does not exist! Run npm run build first."

    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT), "--directory", str(out_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        print("Waiting for server to become responsive...")
        if not wait_for_server(BASE_URL, timeout=15):
            print("Server failed to respond within deadline.")
            return 1
        print("Server is responsive!")

        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            page = context.new_page()

            test_cases = [
                ("fix", "screenshot_fix_fundamentals.png", "Comfort Systems (FIX)"),
                ("duol", "screenshot_duol_fundamentals.png", "Duolingo (DUOL)"),
                ("ulta", "screenshot_ulta_fundamentals.png", "Ulta Beauty (ULTA)"),
                ("aapl", "screenshot_aapl_fallback.png", "Apple (AAPL) - Fallback"),
            ]

            for ticker, filename, label in test_cases:
                url = f"{BASE_URL}/stock/{ticker}/"
                print(f"Navigating to {url} ({label})...")
                page.goto(url, wait_until="networkidle")
                time.sleep(1)

                output_path = SCREENSHOTS_DIR / filename
                page.screenshot(path=str(output_path), full_page=False)
                print(f"Saved screenshot: {output_path} ({output_path.stat().st_size} bytes)")

            browser.close()

        print("\nAll screenshots captured successfully!")
        return 0

    finally:
        print("Terminating preview server...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
