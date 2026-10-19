"""
scripts/verify_html_rendering.py

Inspects rendered static HTML files to confirm proper rendering of:
- Populated company (FIX)
- Incomplete debt / suppressed valuation company (DUOL)
- Incomplete debt company (ULTA)
- Uncataloged / fallback company (AAPL)
"""

from pathlib import Path

def check_html():
    stock_dir = Path("frontend/.next/server/app/stock")

    # 1. FIX
    fix_path = stock_dir / "fix.html"
    assert fix_path.exists(), "fix.html does not exist"
    fix_html = fix_path.read_text(encoding="utf-8")
    assert "SEC EDGAR Sourced Fundamentals" in fix_html
    assert "TOTAL DEBT FULLY EVIDENCED" in fix_html
    assert "0001035983" in fix_html  # CIK
    assert "REPORTED (XBRL)" in fix_html
    assert "DERIVED FORMULA" in fix_html
    # Lease label follows broader concept without noncurrent restriction (P3)
    assert "Operating Lease Liabilities" in fix_html
    assert "ASC 842" in fix_html
    assert "Operating Lease Liabilities (Noncurrent)" not in fix_html
    # Strictly facts-only: No DCF or valuation rankings
    assert "Research Scenario DCF" not in fix_html
    assert "fair_value" not in fix_html
    print("[OK] FIX static HTML verified: Total Debt Evidenced, Broader Leases & Facts-Only disclosures rendered.")

    # 2. DUOL
    duol_path = stock_dir / "duol.html"
    assert duol_path.exists(), "duol.html does not exist"
    duol_html = duol_path.read_text(encoding="utf-8")
    assert "TOTAL DEBT UNAVAILABLE IN XBRL" in duol_html
    assert "0001562088" in duol_html  # CIK
    assert "Operating Lease Liabilities (Noncurrent)" in duol_html
    assert "Deep Valuation Model Suppressed" not in duol_html
    print("[OK] DUOL static HTML verified: Missing Debt warning & Noncurrent Leases rendered.")

    # 3. ULTA
    ulta_path = stock_dir / "ulta.html"
    assert ulta_path.exists(), "ulta.html does not exist"
    ulta_html = ulta_path.read_text(encoding="utf-8")
    assert "TOTAL DEBT UNAVAILABLE IN XBRL" in ulta_html
    assert "0001403568" in ulta_html  # CIK
    assert "Operating Lease Liabilities (Noncurrent)" in ulta_html
    assert "Deep Valuation Model Suppressed" not in ulta_html
    print("[OK] ULTA static HTML verified: Partial Debt warning & Facts-Only disclosures rendered.")

    # 4. AAPL (Fallback / Uncataloged in pilot)
    aapl_path = stock_dir / "aapl.html"
    assert aapl_path.exists(), "aapl.html does not exist"
    aapl_html = aapl_path.read_text(encoding="utf-8")
    assert "Primary filing XBRL extraction is currently available for the 8 deep-dive pilot companies" in aapl_html
    assert "Sourced financial evidence has not yet been cataloged for" in aapl_html
    print("[OK] AAPL static HTML verified: Graceful fallback and pilot links rendered.")

    print("\nALL 4 HTML RENDERING VERIFICATIONS PASSED!")

if __name__ == "__main__":
    check_html()
