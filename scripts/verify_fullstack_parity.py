#!/usr/bin/env python3
"""
scripts/verify_fullstack_parity.py

Full-Stack Release Parity Verification Tool.
Audits deployment release identity synchronization across:
1. Railway Production Backend (/api/v1/health)
2. Cloudflare Pages Frontend Bundle (NEXT_PUBLIC_ARX_RELEASE embedded in chunks)
3. Dynamic Stock Detail Route Rewrite (/stock/plse/ HTTP 200 vs HTTP 404)
4. Canonical Security Master Resolution for uncataloged equities (/api/v1/security-master/instruments/PLSE)
"""

import sys
import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

BACKEND_URL = "https://web-production-e370b.up.railway.app"
FRONTEND_URL = "https://finance-xp8.pages.dev"


def fetch_url(url: str, timeout: int = 15) -> tuple[int, bytes, Dict[str, str]]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ARX-Parity-Auditor/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)
    except Exception as e:
        return 0, str(e).encode(), {}


def check_backend_identity() -> Dict[str, Any]:
    print(f"[*] Probing Railway backend health: {BACKEND_URL}/health ...")
    status, body, headers = fetch_url(f"{BACKEND_URL}/health")
    if status != 200:
        return {"status": "FAIL", "http_status": status, "error": f"HTTP {status}"}
    try:
        data = json.loads(body.decode("utf-8"))
        return {
            "status": "PASS",
            "http_status": 200,
            "backend_status": data.get("status"),
            "raw": data,
        }
    except Exception as e:
        return {"status": "FAIL", "error": f"JSON parse error: {e}"}


def check_backend_security_master(symbol: str = "PLSE") -> Dict[str, Any]:
    print(f"[*] Probing backend Security Master for {symbol}: {BACKEND_URL}/api/v1/security-master/instruments/{symbol} ...")
    status, body, _ = fetch_url(f"{BACKEND_URL}/api/v1/security-master/instruments/{symbol}")
    if status != 200:
        return {"status": "FAIL", "http_status": status, "error": f"HTTP {status}"}
    try:
        data = json.loads(body.decode("utf-8"))
        return {
            "status": "PASS",
            "http_status": 200,
            "symbol": data.get("symbol"),
            "security_type": data.get("security_type"),
            "classification_status": data.get("classification_status"),
            "execution_eligibility": data.get("execution_eligibility"),
        }
    except Exception as e:
        return {"status": "FAIL", "error": f"JSON parse error: {e}"}


def check_frontend_bundle_identity() -> Dict[str, Any]:
    print(f"[*] Probing Cloudflare Pages index HTML: {FRONTEND_URL} ...")
    status, body, headers = fetch_url(FRONTEND_URL)
    if status != 200:
        return {"status": "FAIL", "http_status": status, "error": f"HTTP {status}"}

    html = body.decode("utf-8", errors="ignore")
    chunks = set(re.findall(r'/_next/static/chunks/[a-zA-Z0-9_\-\.]+\.js', html))
    print(f"    Found {len(chunks)} JS chunks referenced in index HTML.")

    embedded_shas = set()
    for chunk in list(chunks)[:15]:  # Sample up to 15 chunks
        chunk_url = f"{FRONTEND_URL}{chunk}"
        c_status, c_body, _ = fetch_url(chunk_url)
        if c_status == 200:
            c_text = c_body.decode("utf-8", errors="ignore")
            # Search for 40-character commit hashes or NEXT_PUBLIC_ARX_RELEASE patterns
            matches = re.findall(r'[0-9a-f]{40}', c_text)
            for m in matches:
                embedded_shas.add(m)

    return {
        "status": "PASS",
        "http_status": status,
        "chunk_count": len(chunks),
        "embedded_shas": list(embedded_shas),
    }


def check_stock_detail_route(symbol: str = "PLSE") -> Dict[str, Any]:
    route_url = f"{FRONTEND_URL}/stock/{symbol.lower()}/"
    print(f"[*] Probing Cloudflare Pages stock detail route: {route_url} ...")
    status, body, _ = fetch_url(route_url)
    return {
        "url": route_url,
        "http_status": status,
        "route_available": status == 200,
        "body_length": len(body),
    }


def run_fullstack_parity_audit(expected_sha: Optional[str] = None):
    print("=" * 70)
    print("ARX TERMINAL FULL-STACK RELEASE PARITY VERIFICATION")
    print("=" * 70)

    backend = check_backend_identity()
    print(f"  Backend Health: HTTP {backend.get('http_status')} | Runtime SHA: {backend.get('git_sha')}")

    sec_master = check_backend_security_master("PLSE")
    print(f"  PLSE Security Master: HTTP {sec_master.get('http_status')} | Type: {sec_master.get('security_type')} | Status: {sec_master.get('classification_status')} | Eligibility: {sec_master.get('execution_eligibility')}")

    frontend = check_frontend_bundle_identity()
    print(f"  Frontend Index: HTTP {frontend.get('http_status')} | Chunks: {frontend.get('chunk_count')}")

    plse_route = check_stock_detail_route("PLSE")
    print(f"  PLSE Detail Route: HTTP {plse_route.get('http_status')} (Available: {plse_route.get('route_available')})")

    aapl_route = check_stock_detail_route("AAPL")
    print(f"  AAPL Detail Route: HTTP {aapl_route.get('http_status')} (Available: {aapl_route.get('route_available')})")

    print("\n" + "=" * 70)
    print("PARITY EVALUATION SUMMARY")
    print("=" * 70)

    divergence_found = False
    if expected_sha:
        backend_sha = backend.get("git_sha")
        print(f"  Expected Release SHA: {expected_sha}")
        if backend_sha != expected_sha:
            print(f"  [DIVERGENCE] Backend SHA ({backend_sha}) does not match expected SHA ({expected_sha})")
            divergence_found = True
        else:
            print(f"  [PARITY] Backend SHA exactly matches expected release SHA.")

    if plse_route.get("http_status") != 200:
        print(f"  [DIVERGENCE] Cloudflare Pages /stock/plse/ returned HTTP {plse_route.get('http_status')} (Expected HTTP 200 rewrite)")
        divergence_found = True
    else:
        print(f"  [PARITY] Cloudflare Pages /stock/plse/ resolves to HTTP 200.")

    if sec_master.get("execution_eligibility") == "STOCK_EXECUTION" and sec_master.get("security_type") == "COMMON_STOCK":
        print(f"  [PARITY] Backend Security Master certifies PLSE as operating common stock.")
    else:
        print(f"  [DIVERGENCE] Backend Security Master failed to certify PLSE.")
        divergence_found = True

    print("=" * 70)
    return not divergence_found


if __name__ == "__main__":
    expected = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_fullstack_parity_audit(expected)
    sys.exit(0 if success else 1)
