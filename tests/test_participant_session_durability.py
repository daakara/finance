"""
ARX Terminal — Phase 2 Research Instrumentation Nonhuman Durability & Custody Suite
Defect: ARX-UX-P00-INSTR-001
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
Governing Authorization: RO-DEC-2026-10-23-012

Test Groups:
- Group A: Event Durability & Schema Ordering
- Group B: Crash & Interruption Handling
- Group C: Cryptographic Receipt Integrity & Server HMAC Verification
- Group D: Idempotency & Replay Protection
- Group E: Security, Clean-URL Bootstrap & Network Boundaries
- Group F: Synthetic vs. Human Boundary Isolation
- Group G: Moderator Readback & Withdrawal Taxonomy
- Group H: Baseline Regression Suite Preservation
"""

import os
import json
import time
import hmac
import socket
import secrets
import hashlib
import tempfile
import threading
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any

import pytest

from analyst_dashboard.research.custody.vault import LocalRestrictedVault
from analyst_dashboard.research.custody.crypto import CustodyCrypto, generate_key, compute_sha256
from analyst_dashboard.research.custody.ledger import TamperEvidentLedger
from analyst_dashboard.research.custody.session_intake import (
    ParticipantSessionIntakePipeline,
    CustodyConflictError,
    IntakeValidationError,
)
from scripts.research_server import create_research_server


@pytest.fixture
def temp_vault_env():
    """Create a temporary isolated vault directory and pipeline instance."""
    temp_dir = tempfile.mkdtemp(prefix="arx_test_vault_")
    vault = LocalRestrictedVault(storage_root=temp_dir, enforce_acls=False, strict_permissions_check=False)
    crypto = CustodyCrypto(generate_key())
    ledger = TamperEvidentLedger(vault.ledger_dir / "ledger.jsonl")
    pipeline = ParticipantSessionIntakePipeline(vault, crypto, ledger, is_synthetic_mode=True)
    return {
        "root": temp_dir,
        "vault": vault,
        "crypto": crypto,
        "ledger": ledger,
        "pipeline": pipeline,
    }


def make_sample_record(
    scenario_id: str = "SCN-NORM-PLSE-SWING",
    participant_id: str = "P00-SYNTHETIC",
    seq: int = 1,
    event_count: int = 3,
) -> Dict[str, Any]:
    events = []
    for i in range(1, event_count + 1):
        events.append({
            "event_seq": i,
            "event_type": "INPUT_CHANGED" if i > 1 else "TASK_RENDERED",
            "monotonic_ms": round(i * 150.5, 2),
            "wall_clock_iso": "2026-10-23T20:00:00Z",
            "details": {"step": i},
        })
    return {
        "scenario_id": scenario_id,
        "scenario_version": "1.0",
        "participant_id": participant_id,
        "participant_archetype": "Quantitative Trader",
        "experiment_block": "BLOCK_1_TRADE_PLANNING",
        "architecture_variant": "CONCEPT_A",
        "sequence_position": seq,
        "authoritative_expected_state": "ACTIONABLE",
        "participant_stated_state": "ACTIONABLE",
        "participant_proposed_action": "BUY_LIMIT",
        "participant_explanation": "Sample explanation",
        "risk_identification_result": "DETECTED",
        "data_freshness_interpretation": "CORRECT_LIVE",
        "confidence_rating": 4,
        "response_elapsed_ms": 1200.0,
        "completion_status": "SUCCESS",
        "canonical_cm_codes": ["CM-01"],
        "supplemental_sm_codes": ["SM-01"],
        "events": events,
    }


# ==============================================================================
# GROUP A: Event Durability & Schema Ordering
# ==============================================================================

def test_group_a_canonical_payload_intake(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    rec = make_sample_record()
    payload_bytes = (json.dumps(rec) + "\n").encode("utf-8")

    receipt = pipeline.ingest_session_payload(payload_bytes, session_id="SESS-DURABLE-01")
    assert receipt["status"] == "ACCEPTED_AND_SEALED"
    assert receipt["session_id"] == "SESS-DURABLE-01"
    assert receipt["participant_id"] == "P00-SYNTHETIC"
    assert receipt["record_count"] == 1
    assert receipt["event_count"] == 3
    assert receipt["source_payload_sha256"] == compute_sha256(payload_bytes)


def test_group_a_event_monotonicity_enforcement(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    rec = make_sample_record()
    # Invert event sequence order
    rec["events"][0]["event_seq"] = 5
    rec["events"][1]["event_seq"] = 2
    payload_bytes = (json.dumps(rec) + "\n").encode("utf-8")

    with pytest.raises(IntakeValidationError) as exc:
        pipeline.ingest_session_payload(payload_bytes, session_id="SESS-NON-MONOTONIC")
    assert "non-monotonic event_seq" in str(exc.value)


# ==============================================================================
# GROUP B: Crash & Interruption Handling
# ==============================================================================

def test_group_b_malformed_partial_json_rejection(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    partial_bytes = b'{"scenario_id": "SCN-NORM-PLSE-SWING", "participant_id": "P00-SYN'

    with pytest.raises(IntakeValidationError) as exc:
        pipeline.ingest_session_payload(partial_bytes, session_id="SESS-CRASH-01")
    assert "Malformed JSON" in str(exc.value)

    # Confirm vault has zero sealed records for this session
    assert not (temp_vault_env["vault"].sealed_dir / "SUBM-SESS-SESS-CRASH-01.enc").exists()


def test_group_b_empty_payload_rejection(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    with pytest.raises(IntakeValidationError) as exc:
        pipeline.ingest_session_payload(b"", session_id="SESS-EMPTY")
    assert "payload is empty" in str(exc.value)


# ==============================================================================
# GROUP C: Cryptographic Receipt Integrity & Server HMAC Verification
# ==============================================================================

def test_group_c_valid_receipt_hmac_signature(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    rec = make_sample_record()
    payload_bytes = (json.dumps(rec) + "\n").encode("utf-8")

    receipt = pipeline.ingest_session_payload(payload_bytes, session_id="SESS-HMAC-01")
    assert pipeline.verify_server_receipt_signature(receipt) is True


def test_group_c_tampered_receipt_signature_rejection(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    rec = make_sample_record()
    payload_bytes = (json.dumps(rec) + "\n").encode("utf-8")

    receipt = pipeline.ingest_session_payload(payload_bytes, session_id="SESS-HMAC-02")

    # Tamper with record_count
    tampered_receipt = dict(receipt)
    tampered_receipt["record_count"] = 99
    assert pipeline.verify_server_receipt_signature(tampered_receipt) is False

    # Tamper with payload_sha256
    tampered_sha = dict(receipt)
    tampered_sha["source_payload_sha256"] = "0" * 64
    assert pipeline.verify_server_receipt_signature(tampered_sha) is False

    # Tamper with forged signature
    tampered_sig = dict(receipt)
    tampered_sig["server_receipt_signature"] = "a" * 64
    assert pipeline.verify_server_receipt_signature(tampered_sig) is False


def test_group_c_ledger_hash_chain_continuity(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    ledger = temp_vault_env["ledger"]

    for i in range(1, 4):
        rec = make_sample_record(seq=i)
        payload_bytes = (json.dumps(rec) + "\n").encode("utf-8")
        pipeline.ingest_session_payload(payload_bytes, session_id=f"SESS-CHAIN-{i:02d}")

    # Ledger audit
    is_valid, err = ledger.verify_chain_integrity()
    assert is_valid is True, f"Ledger integrity check failed: {err}"
    assert len(ledger.get_all_receipts()) == 3


# ==============================================================================
# GROUP D: Idempotency & Replay Protection
# ==============================================================================

def test_group_d_identical_replay_returns_original_receipt(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    rec = make_sample_record()
    payload_bytes = (json.dumps(rec) + "\n").encode("utf-8")

    r1 = pipeline.ingest_session_payload(payload_bytes, session_id="SESS-IDEM-01")
    r2 = pipeline.ingest_session_payload(payload_bytes, session_id="SESS-IDEM-01")

    assert r2["status"] == "IDEMPOTENT_REPLAY_SUCCESS"
    assert r1["receipt_id"] == r2["receipt_id"]
    assert r1["ledger_ref"] == r2["ledger_ref"]
    assert r1["server_receipt_signature"] == r2["server_receipt_signature"]
    assert pipeline.verify_server_receipt_signature(r2) is True


def test_group_d_conflicting_replay_fails_closed(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    rec1 = make_sample_record(seq=1)
    rec2 = make_sample_record(seq=2)
    payload1 = (json.dumps(rec1) + "\n").encode("utf-8")
    payload2 = (json.dumps(rec2) + "\n").encode("utf-8")

    pipeline.ingest_session_payload(payload1, session_id="SESS-CONFLICT-01")

    with pytest.raises(CustodyConflictError) as exc:
        pipeline.ingest_session_payload(payload2, session_id="SESS-CONFLICT-01")
    assert "already sealed with a different payload digest" in str(exc.value)


# ==============================================================================
# GROUP E: Security, Clean-URL Bootstrap & Network Boundaries
# ==============================================================================

def test_group_e_payload_size_ceiling_enforced(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    # 2.1 MB payload
    oversized = b"a" * (2 * 1024 * 1024 + 100)
    with pytest.raises(IntakeValidationError) as exc:
        pipeline.ingest_session_payload(oversized, session_id="SESS-OVERSIZE")
    assert "exceeds maximum allowed size" in str(exc.value)


def test_group_e_unapproved_scenario_rejected(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    rec = make_sample_record(scenario_id="SCN-FORGED-INVASIVE")
    payload_bytes = (json.dumps(rec) + "\n").encode("utf-8")

    with pytest.raises(IntakeValidationError) as exc:
        pipeline.ingest_session_payload(payload_bytes, session_id="SESS-BAD-SCENARIO")
    assert "unapproved scenario_id" in str(exc.value)


# ==============================================================================
# GROUP F: Synthetic vs. Human Boundary Isolation
# ==============================================================================

def test_group_f_human_tokens_rejected_in_synthetic_mode(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    rec = make_sample_record(participant_id="P01")
    payload_bytes = (json.dumps(rec) + "\n").encode("utf-8")

    with pytest.raises(IntakeValidationError) as exc:
        pipeline.ingest_session_payload(payload_bytes, session_id="SESS-HUMAN-LEAK")
    assert "human session intake is NOT authorized" in str(exc.value)


def test_group_f_synthetic_mode_indicator_in_receipt(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    rec = make_sample_record(participant_id="P00-SYNTHETIC")
    payload_bytes = (json.dumps(rec) + "\n").encode("utf-8")

    receipt = pipeline.ingest_session_payload(payload_bytes, session_id="SESS-SYNTHETIC-FLAG")
    assert receipt["synthetic_mode"] is True


# ==============================================================================
# GROUP G: Moderator Readback & Withdrawal Taxonomy
# ==============================================================================

def test_group_g_withdrawal_purges_ciphertext_and_leaves_tombstone(temp_vault_env):
    pipeline = temp_vault_env["pipeline"]
    vault = temp_vault_env["vault"]
    ledger = temp_vault_env["ledger"]

    rec = make_sample_record()
    payload_bytes = (json.dumps(rec) + "\n").encode("utf-8")
    session_id = "SESS-WITHDRAW-01"

    pipeline.ingest_session_payload(payload_bytes, session_id=session_id)
    sealed_file = vault.sealed_dir / f"SUBM-SESS-{session_id}.enc"
    assert sealed_file.exists() is True

    # Execute withdrawal
    tombstone = pipeline.execute_session_withdrawal(
        session_id=session_id,
        participant_id="P00-SYNTHETIC",
        withdrawal_reason="PARTICIPANT_GDPR_ERASURE_REQUEST",
    )

    assert tombstone["status"] == "WITHDRAWAL_COMPLETED"
    assert not sealed_file.exists()  # Shredded and purged

    # Verify ledger integrity remains continuous with tombstone block
    is_valid, err = ledger.verify_chain_integrity()
    assert is_valid is True
    last_rcpt = ledger.get_last_receipt()
    assert last_rcpt["custody_state"] == "WITHDRAWN_TOMBSTONE"


# ==============================================================================
# HTTP Loopback Server Integration Tests (Endpoints & Authentication Boundary)
# ==============================================================================

def test_http_server_custody_integration(temp_vault_env):
    port = 3488
    token = "moderator_test_token_8899aabbcc"
    server = create_research_server(
        port=port,
        bootstrap_token=token,
        pipeline=temp_vault_env["pipeline"],
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Handshake without token fails 403
        req = urllib.request.Request(
            f"{base_url}/api/research/custody/handshake",
            data=b"{}",
            headers={"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 403

        # 2. Handshake with valid token returns session_token
        handshake_payload = json.dumps({
            "bootstrap_token": token,
            "session_id": "SESS-HTTP-01",
            "participant_id": "P00-SYNTHETIC",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/research/custody/handshake",
            data=handshake_payload,
            headers={"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"},
        )
        res = urllib.request.urlopen(req)
        assert res.code == 200
        auth_data = json.loads(res.read().decode("utf-8"))
        session_token = auth_data["session_token"]
        assert len(session_token) == 64

        # 3. Direct unauthenticated ingest fails 401
        rec = make_sample_record()
        raw_payload = (json.dumps(rec) + "\n").encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/research/custody/session_ingest",
            data=raw_payload,
            headers={"Content-Type": "application/x-jsonlines", "Host": f"127.0.0.1:{port}"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 401

        # 4. Authenticated ingest succeeds 200 with server-signed receipt
        req = urllib.request.Request(
            f"{base_url}/api/research/custody/session_ingest",
            data=raw_payload,
            headers={
                "Content-Type": "application/x-jsonlines",
                "Authorization": f"Bearer {session_token}",
                "Host": f"127.0.0.1:{port}",
            },
        )
        res = urllib.request.urlopen(req)
        assert res.code == 200
        rcpt = json.loads(res.read().decode("utf-8"))
        assert rcpt["status"] == "ACCEPTED_AND_SEALED"
        assert "server_receipt_signature" in rcpt

        # 5. Verify receipt via verify_receipt endpoint
        req = urllib.request.Request(
            f"{base_url}/api/research/custody/verify_receipt",
            data=json.dumps(rcpt).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {session_token}",
                "Host": f"127.0.0.1:{port}",
            },
        )
        res = urllib.request.urlopen(req)
        assert res.code == 200
        v_data = json.loads(res.read().decode("utf-8"))
        assert v_data["verified"] is True

        # 6. Readback receipt
        req = urllib.request.Request(
            f"{base_url}/api/research/custody/receipt?receipt_id={rcpt['receipt_id']}&session_id=SESS-HTTP-01",
            headers={
                "Authorization": f"Bearer {session_token}",
                "Host": f"127.0.0.1:{port}",
            },
        )
        res = urllib.request.urlopen(req)
        assert res.code == 200
        rb_data = json.loads(res.read().decode("utf-8"))
        assert rb_data["receipt_id"] == rcpt["receipt_id"]

    finally:
        server.shutdown()


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ==============================================================================
# Adversarial Security & Boundary Enforcement Tests
# ==============================================================================

def test_http_adversarial_host_rejection(temp_vault_env):
    port = get_free_port()
    token = "mod_token_" + secrets.token_hex(16)
    server = create_research_server(port=port, bootstrap_token=token, pipeline=temp_vault_env["pipeline"])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/research/custody/handshake",
            data=b"{}",
            headers={"Content-Type": "application/json", "Host": "malicious.attacker.com"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 403
    finally:
        server.shutdown()


def test_http_adversarial_origin_rejection(temp_vault_env):
    port = get_free_port()
    token = "mod_token_" + secrets.token_hex(16)
    server = create_research_server(port=port, bootstrap_token=token, pipeline=temp_vault_env["pipeline"])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/research/custody/handshake",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Host": f"127.0.0.1:{port}",
                "Origin": "http://evil.domain.com",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 403
    finally:
        server.shutdown()


def test_http_adversarial_sec_fetch_site_rejection(temp_vault_env):
    port = get_free_port()
    token = "mod_token_" + secrets.token_hex(16)
    server = create_research_server(port=port, bootstrap_token=token, pipeline=temp_vault_env["pipeline"])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/research/custody/handshake",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Host": f"127.0.0.1:{port}",
                "Sec-Fetch-Site": "cross-site",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 403
    finally:
        server.shutdown()


def test_http_adversarial_invalid_bootstrap_token(temp_vault_env):
    port = get_free_port()
    token = "legitimate_mod_token_" + secrets.token_hex(16)
    server = create_research_server(port=port, bootstrap_token=token, pipeline=temp_vault_env["pipeline"])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        handshake_payload = json.dumps({
            "bootstrap_token": "forged_or_invalid_passkey",
            "session_id": "SESS-BAD-TOKEN",
            "participant_id": "P00-SYNTHETIC",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/research/custody/handshake",
            data=handshake_payload,
            headers={"Content-Type": "application/json", "Host": f"127.0.0.1:{port}"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 403
    finally:
        server.shutdown()


def test_http_adversarial_expired_session_token(temp_vault_env):
    port = get_free_port()
    token = "mod_token_" + secrets.token_hex(16)
    server = create_research_server(port=port, bootstrap_token=token, pipeline=temp_vault_env["pipeline"])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        # Issue valid session token
        session_token = server.research_state.issue_session_token("SESS-EXP-01", "P00-SYNTHETIC")
        # Artificially expire the token
        server.research_state.session_tokens[session_token]["expires_at"] = time.time() - 100

        rec = make_sample_record()
        raw_payload = (json.dumps(rec) + "\n").encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/research/custody/session_ingest",
            data=raw_payload,
            headers={
                "Content-Type": "application/x-jsonlines",
                "Authorization": f"Bearer {session_token}",
                "Host": f"127.0.0.1:{port}",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 401
    finally:
        server.shutdown()


def test_http_adversarial_revoked_session_token(temp_vault_env):
    port = get_free_port()
    token = "mod_token_" + secrets.token_hex(16)
    server = create_research_server(port=port, bootstrap_token=token, pipeline=temp_vault_env["pipeline"])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    try:
        session_token = server.research_state.issue_session_token("SESS-REVOKE-01", "P00-SYNTHETIC")
        # Invalidate / revoke token
        server.research_state.invalidate_session_token(session_token)

        rec = make_sample_record()
        raw_payload = (json.dumps(rec) + "\n").encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/research/custody/session_ingest",
            data=raw_payload,
            headers={
                "Content-Type": "application/x-jsonlines",
                "Authorization": f"Bearer {session_token}",
                "Host": f"127.0.0.1:{port}",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 401
    finally:
        server.shutdown()


# ==============================================================================
# Real Browser Playwright Integration Tests (Crash Recovery, Wiping, UI Badges)
# ==============================================================================

def test_playwright_browser_process_restart_recovery(temp_vault_env):
    from playwright.sync_api import sync_playwright

    port = get_free_port()
    passkey = "passkey_" + secrets.token_hex(16)
    server = create_research_server(port=port, bootstrap_token=passkey, pipeline=temp_vault_env["pipeline"])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)

    user_data_dir = tempfile.mkdtemp(prefix="arx_pw_recovery_")

    try:
        # Step 1: Open browser, enroll, complete Trial 1, verify IndexedDB
        with sync_playwright() as p:
            browser = p.chromium.launch_persistent_context(user_data_dir, headless=True)
            page = browser.pages[0] if browser.pages else browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/prototype/")
            page.wait_for_load_state("networkidle")

            page.locator("input").first.fill(passkey)
            page.get_by_role("button", name="Start Assigned Protocol").click()
            page.wait_for_selector("#statedDecision", state="visible")

            page.fill("#statedDecision", "BUY_LIMIT approved at support level")
            page.fill("#explanation", "Clear technical corridor observed with active volume confirmation.")
            page.get_by_role("button", name="Submit Decision & Continue").click()
            page.wait_for_timeout(1000)

            # Assert current trial advanced to SYN-02
            assert "SYN-02" in page.content() or "FIX Volatility" in page.content()

            # Verify events exist in IndexedDB events_journal
            count = page.evaluate('''async () => {
                return new Promise((res) => {
                    const req = indexedDB.open("ARX_RESEARCH_JOURNAL_V1", 1);
                    req.onsuccess = () => {
                        const tx = req.result.transaction("events_journal", "readonly");
                        const cr = tx.objectStore("events_journal").count();
                        cr.onsuccess = () => res(cr.result);
                    };
                });
            }''')
            assert count > 0, "IndexedDB events_journal should contain committed events"
            browser.close()

        # Step 2: Relaunch browser with the SAME user_data_dir (process restart)
        with sync_playwright() as p:
            browser = p.chromium.launch_persistent_context(user_data_dir, headless=True)
            page = browser.pages[0] if browser.pages else browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/prototype/")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1500)

            # Verify session resumed at SYN-02 with active prompter
            content = page.content()
            assert "SYN-02" in content or "FIX Volatility" in content
            assert page.locator("#statedDecision").is_visible()
            assert not page.get_by_role("button", name="Start Assigned Protocol").is_visible()
            browser.close()

    finally:
        server.shutdown()


def test_playwright_participant_withdrawal_and_zero_fill(temp_vault_env):
    from playwright.sync_api import sync_playwright

    port = get_free_port()
    passkey = "passkey_" + secrets.token_hex(16)
    server = create_research_server(port=port, bootstrap_token=passkey, pipeline=temp_vault_env["pipeline"])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)

    user_data_dir = tempfile.mkdtemp(prefix="arx_pw_withdraw_")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch_persistent_context(user_data_dir, headless=True)
            page = browser.pages[0] if browser.pages else browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/prototype/")
            page.wait_for_load_state("networkidle")

            page.locator("input").first.fill(passkey)
            page.get_by_role("button", name="Start Assigned Protocol").click()
            page.wait_for_selector("#statedDecision", state="visible")

            # Auto-accept confirm dialog
            page.on("dialog", lambda dialog: dialog.accept())
            page.get_by_role("button", name="Withdraw").click()
            page.wait_for_timeout(1000)

            # Assert DOM renders withdrawal confirmed banner
            assert page.locator("text=PARTICIPANT WITHDRAWAL CONFIRMED").is_visible()

            # Verify IndexedDB has been wiped (zero entries)
            idb_counts = page.evaluate('''async () => {
                return new Promise((res) => {
                    const req = indexedDB.open("ARX_RESEARCH_JOURNAL_V1", 1);
                    req.onsuccess = () => {
                        const db = req.result;
                        const tx = db.transaction(["events_journal", "crypto_keys", "session_state"], "readonly");
                        let ej = 0, ck = 0, ss = 0;
                        const r1 = tx.objectStore("events_journal").count();
                        r1.onsuccess = () => { ej = r1.result; };
                        const r2 = tx.objectStore("crypto_keys").count();
                        r2.onsuccess = () => { ck = r2.result; };
                        const r3 = tx.objectStore("session_state").count();
                        r3.onsuccess = () => { ss = r3.result; };
                        tx.oncomplete = () => res({ events: ej, keys: ck, state: ss });
                    };
                });
            }''')
            assert idb_counts["events"] == 0, "events_journal should be empty after withdrawal"
            assert idb_counts["keys"] == 0, "crypto_keys should be empty after withdrawal"
            assert idb_counts["state"] == 0, "session_state should be empty after withdrawal"

            browser.close()

    finally:
        server.shutdown()


def test_playwright_custody_hold_offline_intake(temp_vault_env):
    from playwright.sync_api import sync_playwright

    port = get_free_port()
    passkey = "passkey_" + secrets.token_hex(16)
    server = create_research_server(port=port, bootstrap_token=passkey, pipeline=temp_vault_env["pipeline"])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)

    user_data_dir = tempfile.mkdtemp(prefix="arx_pw_hold_")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch_persistent_context(user_data_dir, headless=True)
            page = browser.pages[0] if browser.pages else browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/prototype/")
            page.wait_for_load_state("networkidle")

            # Intentionally provide wrong passkey to trigger custody hold
            page.locator("input").first.fill("wrong_bootstrap_passkey")
            page.get_by_role("button", name="Start Assigned Protocol").click()
            page.wait_for_timeout(1000)

            # Assert UI enters SESSION_HOLD_EVIDENCE_INTEGRITY
            assert page.locator("text=EVIDENCE CUSTODY HOLD").is_visible()
            assert page.locator("text=Bootstrap enrollment failed").is_visible()

            browser.close()

    finally:
        server.shutdown()


def test_playwright_end_to_end_custody_sealing(temp_vault_env):
    from playwright.sync_api import sync_playwright

    port = get_free_port()
    passkey = "passkey_" + secrets.token_hex(16)
    server = create_research_server(port=port, bootstrap_token=passkey, pipeline=temp_vault_env["pipeline"])
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)

    user_data_dir = tempfile.mkdtemp(prefix="arx_pw_sealing_")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch_persistent_context(user_data_dir, headless=True)
            page = browser.pages[0] if browser.pages else browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/prototype/")
            page.wait_for_load_state("networkidle")

            page.locator("input").first.fill(passkey)
            page.get_by_role("button", name="Start Assigned Protocol").click()
            page.wait_for_selector("#statedDecision", state="visible")

            # Complete rehearsal trials dynamically handling prompts, TLX modal, and break
            max_steps = 150
            step = 0
            while step < max_steps and not page.locator("text=SEALED CUSTODY VERIFIED").is_visible():
                step += 1
                tlx_btn = page.get_by_role("button", name="Submit Workload Assessment & Continue")
                if tlx_btn.is_visible():
                    tlx_btn.click()
                    page.wait_for_timeout(350)
                    continue

                skip_btn = page.get_by_role("button", name="Skip Break Countdown (Rehearsal Only)")
                if skip_btn.is_visible():
                    skip_btn.click()
                    page.wait_for_timeout(350)
                    continue

                prompter = page.locator("#statedDecision")
                if prompter.is_visible():
                    prompter.fill("Rehearsal trial actionable decision")
                    page.fill("#explanation", "Rehearsal evidence documentation")
                    page.get_by_role("button", name="Submit Decision & Continue").click()
                    page.wait_for_timeout(350)
                    continue

                page.wait_for_timeout(200)

            # Wait for automated push to custody endpoint and UI verification
            page.wait_for_selector("text=SEALED CUSTODY VERIFIED", timeout=12000)
            assert page.locator("text=SEALED CUSTODY VERIFIED").is_visible()
            assert page.locator("text=Server HMAC Signature:").is_visible()

            # Verify sealed file exists in vault
            sealed_files = list(temp_vault_env["vault"].sealed_dir.glob("SUBM-SESS-*.enc"))
            assert len(sealed_files) == 1, "Expected 1 sealed submission in vault"

            # Verify ledger record
            receipts = temp_vault_env["ledger"].get_all_receipts()
            assert len(receipts) == 1
            assert receipts[0]["custody_state"] == "SEALED"

            browser.close()

    finally:
        server.shutdown()
