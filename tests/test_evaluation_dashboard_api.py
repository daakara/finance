"""Adversarial and functional test suite for the Model Governance & Prospective Evaluation API."""

import os
import json
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.routes import governance

client = TestClient(app)

VALID_DEV_KEY = "arx-eval-prospective-2026-secret"


def test_adversarial_1_no_credential():
    """1. No credential -> 401."""
    response = client.get("/api/v1/governance/evaluation-summary")
    assert response.status_code == 401
    assert "Unauthorized" in response.json().get("detail", "")
    assert response.headers.get("X-Robots-Tag") == "noindex, nofollow, noarchive, nosnippet"
    assert response.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate, max-age=0"
    assert response.headers.get("Cloudflare-CDN-Cache-Control") == "no-store"


def test_adversarial_2_wrong_credential():
    """2. Wrong credential -> 401."""
    response = client.get(
        "/api/v1/governance/evaluation-summary",
        headers={"Authorization": "Bearer totally-wrong-invalid-credential-12345"}
    )
    assert response.status_code == 401
    assert "Unauthorized" in response.json().get("detail", "")


def test_adversarial_3_empty_credential():
    """3. Empty credential -> 401."""
    # Empty Authorization header
    res1 = client.get("/api/v1/governance/evaluation-summary", headers={"Authorization": ""})
    assert res1.status_code == 401

    # Empty Bearer header
    res2 = client.get("/api/v1/governance/evaluation-summary", headers={"Authorization": "Bearer "})
    assert res2.status_code == 401

    # Empty X-Evaluation-Key
    res3 = client.get("/api/v1/governance/evaluation-summary", headers={"X-Evaluation-Key": ""})
    assert res3.status_code == 401


def test_adversarial_4_malformed_bearer_token():
    """4. Malformed bearer token -> 401."""
    malformed_headers = [
        "Bearer",
        "bearer",
        "Bearer   ",
        "Token some-random-token",
        "Basic YWRtaW46cGFzc3dvcmQ=",
        "Bearer key1 key2 key3",
        "Bearer \x00\x01\x02",
    ]
    for h in malformed_headers:
        resp = client.get("/api/v1/governance/evaluation-summary", headers={"Authorization": h})
        assert resp.status_code == 401, f"Failed on malformed header: {h}"


def test_adversarial_5_correct_credential_allowed(monkeypatch):
    """5. Correct credential -> allowed (200 OK with complete scorecard)."""
    monkeypatch.setattr(governance, "SERVER_EVAL_KEY", VALID_DEV_KEY)
    response = client.get(
        "/api/v1/governance/evaluation-summary",
        headers={"Authorization": f"Bearer {VALID_DEV_KEY}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"

    # Anti-caching and Vary headers
    assert response.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate, max-age=0"
    assert response.headers.get("Cloudflare-CDN-Cache-Control") == "no-store"
    assert "Authorization" in response.headers.get("Vary", "")


def test_adversarial_6_unsupported_methods_rejected():
    """6. Correct credential with unsupported method -> rejected (405)."""
    headers = {"Authorization": f"Bearer {VALID_DEV_KEY}"}

    for endpoint in ["/api/v1/governance/evaluation-summary", "/api/v1/governance/prospective-ledger"]:
        assert client.post(endpoint, headers=headers).status_code == 405
        assert client.put(endpoint, headers=headers).status_code == 405
        assert client.patch(endpoint, headers=headers).status_code == 405
        assert client.delete(endpoint, headers=headers).status_code == 405


def test_adversarial_7_direct_backend_access_without_credential():
    """7. Attempted direct backend access without credential -> rejected."""
    # Direct calls without Cloudflare headers
    resp1 = client.get("/api/v1/governance/evaluation-summary")
    assert resp1.status_code == 401

    resp2 = client.get("/api/v1/governance/prospective-ledger")
    assert resp2.status_code == 401


def test_adversarial_8_public_routes_do_not_leak_evaluation_data():
    """8. Evaluation data cannot be obtained through an alternative public route."""
    public_endpoints = [
        "/health",
        "/api/v1/screener",
        "/api/v1/smart-money",
        "/api/v1/regimes/current",
    ]
    for ep in public_endpoints:
        res = client.get(ep)
        text = res.text
        # Ensure no evaluation ledger signal IDs or frozen hashes are leaked
        assert "ANET_2026-09-04" not in text
        assert "2cda1c986bf49e16a0852b968a55d38859849244f4788a6ef90d7bbfa09cf614" not in text
        assert "v2.4.0-phase24-freeze" not in text


def test_adversarial_9_static_build_leakage_audit():
    """9. Confirm static /evaluation assets contain zero evaluation data or secrets."""
    repo_root = os.path.dirname(os.path.dirname(__file__))
    out_eval_path = os.path.join(repo_root, "frontend", "out", "evaluation", "index.html")
    if os.path.exists(out_eval_path):
        with open(out_eval_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Secret audit
        assert VALID_DEV_KEY not in html
        assert "ARX_EVALUATION_KEY" not in html

        # Ledger data audit
        assert "ANET_2026-09-04" not in html
        assert "191.44" not in html
        assert "218.36" not in html
        assert "2cda1c986bf49e16a0852b968a55d38859849244f4788a6ef90d7bbfa09cf614" not in html

        # Anti-indexing audit
        assert 'name="robots" content="noindex, nofollow, nocache"' in html


def test_adversarial_10_cache_prevention_headers_enforced():
    """10. Authenticated response cannot be retrieved through an unauthenticated cache hit."""
    res = client.get(
        "/api/v1/governance/evaluation-summary",
        headers={"Authorization": f"Bearer {VALID_DEV_KEY}"}
    )
    # The server strictly dictates no-store, preventing edge/proxy caching
    cc = res.headers.get("Cache-Control", "")
    assert "no-store" in cc
    assert "no-cache" in cc
    assert res.headers.get("Cloudflare-CDN-Cache-Control") == "no-store"
    assert res.headers.get("CDN-Cache-Control") == "no-store"
    assert "Authorization" in res.headers.get("Vary", "")


def test_adversarial_11_fail_closed_when_key_unconfigured_in_production(monkeypatch):
    """11. When ARX_EVALUATION_KEY is not configured on server, it strictly fails closed (refuses all access)."""
    monkeypatch.setattr(governance, "SERVER_EVAL_KEY", "")
    monkeypatch.setenv("ARX_EVALUATION_KEY", "")
    
    # Even if client sends something, server must reject because it has no key configured
    res = client.get(
        "/api/v1/governance/evaluation-summary",
        headers={"Authorization": "Bearer some-guess"}
    )
    assert res.status_code == 401
    assert "Unauthorized" in res.json().get("detail", "")
