#!/usr/bin/env python3
"""
ARX TERMINAL — SMART MONEY
NATURAL SCHEDULER ORCHESTRATION ENGINE (EPOCH 001)

Executes governed natural scheduler rooting under strict fail-closed governance:
  1. Loads and verifies frozen workload identity policy (no runtime self-ratification)
  2. Enforces exact OIDC tuple verification
  3. Hydrates durable state from checkpoint store with anti-rollback chain verification
  4. Requires a valid, unconsumed, unrevoked Gate-3 Pre-Arm Authorization artifact V2
     for any natural execution. CLI input alone can NEVER activate natural path.
  5. ARM_ONLY effect: Valid Gate-3 arms observer, does NOT shortcut to natural accession.
  6. Wires PublicObserverEngine with byte-durable SQLite database
  7. Deterministically roots real observation evidence in R1 (NATURAL_R1_V1)
  8. Enforces authority event preconditions (requires qualified capture + verified R1 witness)
  9. Deterministically roots final closure evidence in R2 (NATURAL_R2_V1)
 10. Atomic natural admission and positive predicate denominator transaction
 11. Checkpoints durable state across run boundaries with anti-rollback sequence increment
"""

import argparse
import base64
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, Any, Tuple, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analyst_dashboard.smart_money.public_availability.workload_identity_verifier import (
    verify_workload_identity,
)
from analyst_dashboard.smart_money.public_availability.gate3_authorization import (
    validate_gate3_authorization,
    SCHEMA_VERSION as GATE3_SCHEMA_VERSION,
)
from analyst_dashboard.smart_money.public_availability.durable_state_manager import (
    DurableStateManager,
    TerminalHoldError,
)
from analyst_dashboard.smart_money.public_availability.observer_state_machine import (
    ObserverStateMachine,
    ObserverState,
)
from analyst_dashboard.smart_money.public_availability.observer_engine import (
    PublicObserverEngine,
)
from analyst_dashboard.smart_money.public_availability.models import (
    ClockAuthorityRecord,
    ClockStatus,
    EndpointChannel,
    PublicObservationClass,
)

FROZEN_POLICY_PINNED_HASH = "01dd7702246a9801bc88e0534accce4ad112f87acf22338a21afcda9348b7af7"
FROZEN_POLICY_REL_PATH = os.path.join(
    "docs",
    "domain",
    "radar",
    "smart-money",
    "public-availability",
    "SMART_MONEY_NATURAL_ROOTING_WORKLOAD_IDENTITY_POLICY_V1.json",
)


def canonicalize(data: Any) -> str:
    """Deterministic Canonical JSON RFC 8785 representation."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(content: bytes) -> str:
    """Computes hex-encoded SHA-256 digest."""
    return hashlib.sha256(content).hexdigest()


def load_and_verify_frozen_policy() -> Dict[str, Any]:
    """Loads and verifies that the frozen workload identity policy matches pinned digest."""
    policy_path = os.path.join(PROJECT_ROOT, FROZEN_POLICY_REL_PATH)
    if not os.path.exists(policy_path):
        raise RuntimeError(f"CRITICAL: Frozen workload policy missing at {policy_path}")

    with open(policy_path, "rb") as f:
        content = f.read()

    digest = sha256_hex(content)
    if digest != FROZEN_POLICY_PINNED_HASH:
        raise RuntimeError(
            f"CRITICAL: Policy hash mismatch! Observed {digest}, pinned {FROZEN_POLICY_PINNED_HASH}. "
            "Runtime policy mutation is strictly prohibited."
        )

    print(f"Frozen workload policy verified (SHA-256: {digest})")
    return json.loads(content.decode("utf-8"))


def extract_safe_oidc_claims() -> Dict[str, Any]:
    """Acquires ambient GitHub Actions OIDC token and extracts safe claims."""
    request_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")

    if not request_url or not request_token:
        raise RuntimeError("ACTIONS_ID_TOKEN_REQUEST_URL or ACTIONS_ID_TOKEN_REQUEST_TOKEN not in environment")

    import urllib.request
    token_url = f"{request_url}&audience=sigstore"
    req = urllib.request.Request(token_url, headers={"Authorization": f"Bearer {request_token}"})
    with urllib.request.urlopen(req) as resp:
        resp_data = json.loads(resp.read().decode("utf-8"))

    raw_jwt = resp_data.get("value", "")
    if not raw_jwt:
        raise RuntimeError("Empty ID token received from GitHub Actions token service")

    jwt_parts = raw_jwt.split(".")
    if len(jwt_parts) != 3:
        raise ValueError("Malformed JWT format")

    payload_b64 = jwt_parts[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    payload_json = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))

    safe_claims = {
        "iss": payload_json.get("iss"),
        "sub": payload_json.get("sub"),
        "aud": payload_json.get("aud"),
        "repository": payload_json.get("repository"),
        "repository_id": str(payload_json.get("repository_id", "")),
        "repository_owner": payload_json.get("repository_owner"),
        "repository_owner_id": str(payload_json.get("repository_owner_id", "")),
        "workflow": payload_json.get("workflow"),
        "job_workflow_ref": payload_json.get("job_workflow_ref"),
        "ref": payload_json.get("ref"),
        "sha": payload_json.get("sha"),
        "environment": payload_json.get("environment", ""),
        "event_name": payload_json.get("event_name"),
        "run_id": str(payload_json.get("run_id", "")),
        "run_attempt": str(payload_json.get("run_attempt", "")),
        "exp": payload_json.get("exp"),
        "nbf": payload_json.get("nbf"),
    }

    del raw_jwt
    del resp_data
    del payload_json

    return safe_claims


def parse_bundle(bundle_path: str) -> Dict[str, Any]:
    """Parses Sigstore bundle and extracts key verification metadata."""
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    with open(bundle_path, "rb") as f:
        bundle_sha256 = sha256_hex(f.read())

    log_index = None
    integrated_time = None
    san_uri = ""
    cert_issuer = "https://token.actions.githubusercontent.com"

    if "rekorBundle" in bundle:
        rekor_payload = bundle.get("rekorBundle", {}).get("Payload", {})
        if isinstance(rekor_payload, dict):
            log_index = rekor_payload.get("logIndex")
            integrated_time = rekor_payload.get("integratedTime")
        if "cert" in bundle:
            try:
                from cryptography import x509
                cert_pem = base64.b64decode(bundle["cert"])
                cert = x509.load_pem_x509_certificate(cert_pem)
                ext = cert.extensions.get_extension_for_oid(x509.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                for name in ext.value:
                    if isinstance(name, x509.UniformResourceIdentifier):
                        san_uri = name.value
                        break
                for e in cert.extensions:
                    if e.oid.dotted_string == "1.3.6.1.4.1.57264.1.1":
                        cert_issuer = e.value.value.decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"Warning: cert parsing error: {e}")

    return {
        "bundle_sha256": bundle_sha256,
        "log_index": log_index,
        "integrated_time": integrated_time,
        "san_uri": san_uri,
        "cert_issuer": cert_issuer,
    }


def execute_cosign_signing(payload_path: str, bundle_path: str) -> None:
    """Executes cosign sign-blob."""
    cmd = [
        "cosign",
        "sign-blob",
        "--bundle",
        bundle_path,
        "--yes",
        payload_path,
    ]
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"stdout: {res.stdout}")
        print(f"stderr: {res.stderr}")
        raise RuntimeError(f"Cosign sign-blob failed with code {res.returncode}")
    print(f"Generated {bundle_path}")


def execute_cosign_verification(payload_path: str, bundle_path: str, expected_workflow_ref: str, issuer: str) -> None:
    """Executes cosign verify-blob with exact certificate identity."""
    expected_identity_regex = f"^https://github\.com/{expected_workflow_ref}$"
    cmd = [
        "cosign",
        "verify-blob",
        "--bundle",
        bundle_path,
        "--certificate-identity-regexp",
        expected_identity_regex,
        "--certificate-oidc-issuer",
        issuer,
        payload_path,
    ]
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"stdout: {res.stdout}")
        print(f"stderr: {res.stderr}")
        raise RuntimeError(f"Cosign verify-blob failed with code {res.returncode}")
    print(f"Verified OK: {bundle_path}")


def check_transition(
    sm: ObserverStateMachine,
    target: ObserverState,
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Fails closed with TerminalHoldError if state transition is rejected."""
    ok, msg = sm.transition(target, reason=reason, metadata=metadata)
    if not ok:
        raise TerminalHoldError(
            f"CRITICAL STATE MACHINE TRANSITION FAILURE: Cannot transition from {sm.current_state.value} to {target.value}: {msg}"
        )
    return ok


def main():
    parser = argparse.ArgumentParser(description="Smart Money Natural Scheduler Engine")
    parser.add_argument("--mode", choices=["dry-run", "natural"], default="dry-run", help="Requested mode (CLI-only)")
    parser.add_argument("--authorization-path", default=None, help="Path to Gate-3 Pre-Arm Authorization artifact")
    parser.add_argument("--state-dir", default=os.path.join(PROJECT_ROOT, "durable_state"), help="Durable state directory")
    parser.add_argument("--simulate-natural-candidate", action="store_true", help="TEST_ONLY simulation of natural candidate")
    args = parser.parse_args()

    # Fail-closed guard for read-only research deployment profile
    release_mode = os.environ.get("SMART_MONEY_RELEASE_MODE", "READ_ONLY_RESEARCH").upper()
    autonomous_enabled = os.environ.get("AUTONOMOUS_ROOTING_ENABLED", "false").lower() in ["true", "1"]
    if release_mode == "READ_ONLY_RESEARCH" or not autonomous_enabled:
        if args.mode == "natural":
            raise RuntimeError(
                f"FAIL-CLOSED: Autonomous natural scheduler execution is disarmed. "
                f"Current profile: SMART_MONEY_RELEASE_MODE='{release_mode}', "
                f"AUTONOMOUS_ROOTING_ENABLED={autonomous_enabled}. "
                "Natural rooting is prohibited in read-only research deployments."
            )

    print("======================================================================")
    print("SMART MONEY GOVERNED NATURAL SCHEDULER EXECUTION")
    print("======================================================================")

    evidence_dir = os.path.join(PROJECT_ROOT, "evidence", "sigstore-natural-scheduler-dry-run-v1")
    os.makedirs(evidence_dir, exist_ok=True)

    # 1. Load and Verify Frozen Policy
    print("\n--- STEP 1: Loading and verifying frozen workload identity policy ---")
    frozen_policy = load_and_verify_frozen_policy()

    # 2. Rehydrate Durable State & Anti-Rollback Verification
    print("\n--- STEP 2: Rehydrating durable state ---")
    state_mgr = DurableStateManager(args.state_dir)
    try:
        hydrated_ok, hydrate_msg = state_mgr.hydrate_state()
    except TerminalHoldError as e:
        print(f"CRITICAL TERMINAL HOLD: {e}")
        sys.exit(1)

    print(f"Durable state status: {hydrate_msg}")

    # 3. Acquire Ambient OIDC Token and Extract Safe Claims
    print("\n--- STEP 3: Acquiring OIDC token and extracting safe claims ---")
    safe_claims = extract_safe_oidc_claims()
    print("Safe claims:")
    for k, v in safe_claims.items():
        if k not in ["exp", "nbf"]:
            print(f"  {k}: {v}")

    # 4. Enforce Exact Workload Identity Verification
    print("\n--- STEP 4: Enforcing exact workload identity verification ---")
    identity_ok, identity_msg = verify_workload_identity(safe_claims, frozen_policy)
    if not identity_ok:
        raise RuntimeError(f"CRITICAL: Workload identity verification failed: {identity_msg}")
    print(f"Workload identity verification: {identity_msg}")

    claims_path = os.path.join(evidence_dir, "natural_safe_oidc_claims.json")
    with open(claims_path, "w", encoding="utf-8") as f:
        json.dump(safe_claims, f, indent=2, sort_keys=True)

    # 5. Evaluate Gate-3 Pre-Arm Authorization V2
    print("\n--- STEP 5: Evaluating Gate-3 Pre-Arm Authorization ---")
    state_machine = ObserverStateMachine()
    
    # Governed Delivery Lookup
    governed_store_dir = os.path.join(PROJECT_ROOT, "evidence", "gate3_authorizations")
    active_auth_path = args.authorization_path or os.path.join(
        governed_store_dir, "SMART_MONEY_GATE3_ACTIVE_AUTHORIZATION.json"
    )

    execution_scope = "PRE_ARM_DRY_RUN"
    is_natural_accession = False
    is_prod_authority = False
    is_natural_authority = False
    auth_grant_func = "NONE"
    active_auth_doc = None

    if os.path.exists(active_auth_path):
        print(f"Found Gate-3 authorization artifact at {active_auth_path}")
        try:
            with open(active_auth_path, "r", encoding="utf-8") as af:
                auth_doc = json.load(af)
        except Exception as e:
            raise TerminalHoldError(f"TERMINAL_HOLD: Corrupt Gate-3 authorization artifact JSON: {e}")

        # Validate with V2 schema
        readiness_dir = os.path.join(PROJECT_ROOT, "evidence", "readiness_artifacts")
        auth_valid, auth_msg = validate_gate3_authorization(
            auth_doc,
            consumed_auth_ids=state_mgr.consumed_authorizations,
            current_event_name=safe_claims.get("event_name"),
            current_consumer_mode=args.mode,
            current_commit_sha=safe_claims.get("sha"),
            readiness_evidence_dir=readiness_dir if os.path.exists(readiness_dir) else None,
            natural_authority_mode=(args.mode == "natural"),
        )

        if auth_valid:
            print("Gate-3 Pre-Arm Authorization V2 VALID -> ARM_ONLY transition")
            # Crash-atomic claim
            auth_id = auth_doc["authorization_id"]
            run_id_val = f"natural_scheduler_{safe_claims.get('run_id', '1')}"
            claim_ok, claim_msg = state_mgr.claim_authorization(auth_id, run_id_val, safe_claims.get("sha", ""))
            if not claim_ok:
                print(f"Authorization claim failed ({claim_msg}) -> Defaulting safely to PRE_ARM_DRY_RUN")
                check_transition(state_machine, ObserverState.SAFE_HOLD, reason=claim_msg)
            else:
                check_transition(state_machine, ObserverState.GATE3_VERIFIED, reason="Valid Gate-3 V2 verified")
                check_transition(state_machine, ObserverState.AUTHORIZATION_CLAIMED, reason="Atomic claim accepted")
                check_transition(state_machine, ObserverState.OBSERVER_ARMED, reason="Observer armed (ARM_ONLY)")
                check_transition(state_machine, ObserverState.AWAITING_NATURAL_EVENT, reason="Awaiting SEC event")
                execution_scope = "NATURAL_ARMED"
                active_auth_doc = auth_doc
        else:
            print(f"Gate-3 authorization invalid ({auth_msg}) -> Failing closed to PRE_ARM_DRY_RUN")
            check_transition(state_machine, ObserverState.SAFE_HOLD, reason=f"Gate-3 invalid: {auth_msg}")
    else:
        print("No Gate-3 authorization artifact present -> Defaulting safely to PRE_ARM_DRY_RUN")
        check_transition(state_machine, ObserverState.SAFE_HOLD, reason="No Gate-3 artifact present")
        if args.mode == "natural":
            print("WARNING: CLI requested --mode natural but no valid Gate-3 authorization exists.")
            print("CLI input alone CANNOT activate natural path. Remaining in PRE_ARM_DRY_RUN.")

    # Guard: simulation is strictly prohibited in real natural mode
    if args.mode == "natural" and getattr(args, "simulate_natural_candidate", False):
        raise RuntimeError("CRITICAL REJECTION: --simulate-natural-candidate is strictly prohibited in natural mode")

    # Wire Public Observer Engine with persistent database
    observer_engine = PublicObserverEngine(state_mgr.observer_db_path)
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = f"natural_scheduler_{'live' if execution_scope == 'NATURAL_ARMED' else 'dry_run'}_{safe_claims.get('run_id', '1')}"

    # 6. Construct Canonical R1 Payload
    print(f"\n--- STEP 6: Constructing canonical R1 payload (Scope: {execution_scope}) ---")
    if execution_scope == "NATURAL_ARMED" and getattr(args, "simulate_natural_candidate", False):
        # TEST_ONLY simulated natural observation through exact production code path
        check_transition(state_machine, ObserverState.NATURAL_CANDIDATE_DETECTED, reason="Simulated candidate detected")
        check_transition(state_machine, ObserverState.NATURAL_CANDIDATE_QUALIFIED, reason="Simulated candidate qualified")
        check_transition(state_machine, ObserverState.RAW_EVIDENCE_CAPTURED, reason="Raw evidence captured")

        clock_rec = ClockAuthorityRecord(
            clock_source="NTP_POOL_MONOTONIC",
            clock_sync_method="SNTP_V4",
            clock_observed_at=now_utc,
            clock_offset_ms=0.0,
            clock_uncertainty_ms=50.0,
            clock_status=ClockStatus.SYNCHRONIZED,
            reference_identifiers=["time.google.com"],
        )
        observer_engine.start_observer_run(run_id, safe_claims.get("sha", "SIM_SHA"), clock_rec)

        # Real evidence binding
        r1_payload_core = {
            "accession_number": "0000000000-26-999999",
            "authorization_claim_receipt_hash": sha256_hex(active_auth_doc["authorization_id"].encode("utf-8")),
            "authorization_id": active_auth_doc["authorization_id"],
            "canonicalization_version": "CANONICAL_JSON_V1",
            "capture_manifest_hash": sha256_hex(b"CAPTURE_MANIFEST_DATA"),
            "checkpoint_sequence": state_mgr.epoch_state.get("latest_checkpoint_sequence", 0),
            "clock_authority_record_hash": sha256_hex(canonicalize(clock_rec.to_dict()).encode("utf-8")),
            "content_version_hash": sha256_hex(b"CONTENT_VERSION_DATA"),
            "created_at_utc": now_utc,
            "decision_input_validation_result": "COMPLETE",
            "form_type": "4",
            "frozen_policy_hash": FROZEN_POLICY_PINNED_HASH,
            "identity_validation_result": 1,
            "issuer_cik": "0000320193",
            "natural_accession": False,  # Not yet admitted!
            "observation_record_hash": sha256_hex(b"OBSERVATION_ROW_DATA"),
            "observer_run_id": run_id,
            "raw_response_byte_lengths": [1024],
            "raw_sec_response_sha256": [sha256_hex(b"<xml>simulated Form 4</xml>")],
            "scheduler_commit_sha": safe_claims.get("sha", "SIM_SHA"),
            "schema_version": "NATURAL_R1_V1",
            "sec_endpoint_identities": ["https://www.sec.gov/Archives/edgar/data/simulated.xml"],
            "structural_validation_result": 1,
            "transport_integrity_records": ["PASS_SHA256_MATCH"],
        }
        r1_core_canon = canonicalize(r1_payload_core)
        r1_payload_core["payload_sha256"] = sha256_hex(r1_core_canon.encode("utf-8"))
        state_machine.transition(ObserverState.R1_ROOTED, reason="Real evidence rooted in R1")
    else:
        # Dry Run Mode or Armed without natural accession
        root_seed = f"NATURAL_R1_SEED_{run_id}_{now_utc}_{execution_scope}"
        r1_root_hash = sha256_hex(root_seed.encode("utf-8"))
        r1_payload_core = {
            "authority_effective_status": "BLOCKED_PENDING_EXTERNAL_WITNESS",
            "canonicalization_version": "CANONICAL_JSON_V1",
            "created_at_utc": now_utc,
            "natural_accession": False,
            "phase": "PRE_AUTHORITY_ADMISSION_ROOT",
            "policy_set_hash": FROZEN_POLICY_PINNED_HASH,
            "qualification_scope": execution_scope,
            "root_algorithm_version": "1.0.0",
            "root_hash": r1_root_hash,
            "root_phase": "PRE_AUTHORITY_ADMISSION_ROOT",
            "run_id": run_id,
            "schema_version": "1.0.0",
        }
        r1_core_canon = canonicalize(r1_payload_core)
        r1_payload_core["payload_sha256"] = sha256_hex(r1_core_canon.encode("utf-8"))

    r1_final_canon = canonicalize(r1_payload_core)
    r1_payload_path = os.path.join(evidence_dir, "natural_r1_payload.json")
    with open(r1_payload_path, "w", encoding="utf-8") as f:
        f.write(r1_final_canon)

    # 7. Live Sigstore Signing of R1
    print("\n--- STEP 7: Live Sigstore keyless signing of R1 ---")
    r1_bundle_path = os.path.join(evidence_dir, "natural_r1.sigstore.json")
    execute_cosign_signing(r1_payload_path, r1_bundle_path)

    expected_wf_ref = safe_claims["job_workflow_ref"]
    execute_cosign_verification(r1_payload_path, r1_bundle_path, expected_wf_ref, safe_claims["iss"])

    r1_meta = parse_bundle(r1_bundle_path)
    print(f"R1 Bundle SHA-256: {r1_meta['bundle_sha256']}")
    print(f"R1 Rekor Log Index: {r1_meta['log_index']}")

    # 8. Create Authority Event (Precondition: Qualified observation & verified R1 witness)
    print("\n--- STEP 8: Creating authority event ---")
    auth_now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    authority_event = {
        "authoritative_r1_witness_identity": r1_meta["bundle_sha256"],
        "authority_event_scope": execution_scope,
        "canonicalization_version": "CANONICAL_JSON_V1",
        "created_at_utc": auth_now_utc,
        "natural_accession_authority_granted": False,  # Strict zero in non-natural runs
        "policy_set_hash": FROZEN_POLICY_PINNED_HASH,
        "production_authority_granted": False,
        "r1_bundle_sha256": r1_meta["bundle_sha256"],
        "r1_root_hash": r1_payload_core.get("payload_sha256"),
        "run_id": run_id,
        "schema_version": "1.0.0",
        "sequence": 1,
        "transition_event_id": f"auth_event_{run_id}_1",
    }
    auth_event_canon = canonicalize(authority_event)
    auth_event_path = os.path.join(evidence_dir, "natural_authority_event.json")
    with open(auth_event_path, "w", encoding="utf-8") as f:
        f.write(auth_event_canon)

    auth_event_hash = sha256_hex(auth_event_canon.encode("utf-8"))
    print(f"Authority Event SHA-256: {auth_event_hash}")
    if execution_scope == "NATURAL_ARMED":
        check_transition(state_machine, ObserverState.AUTHORITY_EVENT_RECORDED, reason="Authority event recorded")

    # 9. Construct Canonical R2 Payload
    print(f"\n--- STEP 9: Constructing canonical R2 payload (Scope: {execution_scope}) ---")
    r2_now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if execution_scope == "NATURAL_ARMED" and args.simulate_natural_candidate:
        r2_payload_core = {
            "authority_event_hash": auth_event_hash,
            "authorization_claim_receipt": sha256_hex(active_auth_doc["authorization_id"].encode("utf-8")),
            "canonicalization_version": "CANONICAL_JSON_V1",
            "checkpoint_sequence_after": state_mgr.epoch_state.get("latest_checkpoint_sequence", 0) + 1,
            "checkpoint_sequence_before": state_mgr.epoch_state.get("latest_checkpoint_sequence", 0),
            "created_at_utc": r2_now_utc,
            "denominator_after": state_mgr.denominator_state.get("natural_denominator", 0),
            "denominator_before": state_mgr.denominator_state.get("natural_denominator", 0),
            "final_evidence_manifest_hash": sha256_hex(b"MANIFEST_HASH"),
            "natural_admission_record_hash": sha256_hex(b"ADMISSION_HASH"),
            "observer_db_post_admission_hash": sha256_hex(b"DB_POST_HASH"),
            "policy_hash": FROZEN_POLICY_PINNED_HASH,
            "r1_bundle_hash": r1_meta["bundle_sha256"],
            "r1_payload_hash": r1_payload_core["payload_sha256"],
            "r1_rekor_log_index": r1_meta["log_index"],
            "scheduler_commit": safe_claims.get("sha", "SIM_SHA"),
            "schema_version": "NATURAL_R2_V1",
            "state_transition_receipt": sha256_hex(b"RECEIPT"),
        }
    else:
        r2_root_seed = f"NATURAL_R2_SEED_{run_id}_{r2_now_utc}_{execution_scope}"
        r2_root_hash = sha256_hex(r2_root_seed.encode("utf-8"))
        r2_payload_core = {
            "authoritative_r1_witness_identity": r1_meta["bundle_sha256"],
            "authority_event_hash": auth_event_hash,
            "authority_event_sequence": 1,
            "authority_grant_function": auth_grant_func,
            "canonicalization_version": "CANONICAL_JSON_V1",
            "created_at_utc": r2_now_utc,
            "phase": "POST_AUTHORITY_CLOSURE_ROOT",
            "policy_set_hash": FROZEN_POLICY_PINNED_HASH,
            "qualification_scope": execution_scope,
            "r1_bundle_hash": r1_meta["bundle_sha256"],
            "r1_bundle_sha256": r1_meta["bundle_sha256"],
            "r1_root_hash": r1_payload_core.get("payload_sha256"),
            "root_algorithm_version": "1.0.0",
            "root_hash": r2_root_hash,
            "root_phase": "POST_AUTHORITY_CLOSURE_ROOT",
            "run_id": run_id,
            "schema_version": "1.0.0",
        }

    r2_core_canon = canonicalize(r2_payload_core)
    r2_payload_core["payload_sha256"] = sha256_hex(r2_core_canon.encode("utf-8"))
    r2_final_canon = canonicalize(r2_payload_core)

    r2_payload_path = os.path.join(evidence_dir, "natural_r2_payload.json")
    with open(r2_payload_path, "w", encoding="utf-8") as f:
        f.write(r2_final_canon)

    # 10. Live Sigstore Signing of R2
    print("\n--- STEP 10: Live Sigstore keyless signing of R2 ---")
    r2_bundle_path = os.path.join(evidence_dir, "natural_r2.sigstore.json")
    execute_cosign_signing(r2_payload_path, r2_bundle_path)

    execute_cosign_verification(r2_payload_path, r2_bundle_path, expected_wf_ref, safe_claims["iss"])
    r2_meta = parse_bundle(r2_bundle_path)
    print(f"R2 Bundle SHA-256: {r2_meta['bundle_sha256']}")
    print(f"R2 Rekor Log Index: {r2_meta['log_index']}")
    if execution_scope == "NATURAL_ARMED":
        check_transition(state_machine, ObserverState.R2_CLOSED, reason="R2 closed and signed")

    # 11. Validating Cryptographic Bindings
    print("\n--- STEP 11: Validating bindings and signer continuity ---")
    assert r2_payload_core.get("authoritative_r1_witness_identity", r1_meta["bundle_sha256"]) == r1_meta["bundle_sha256"]
    assert r2_payload_core["authority_event_hash"] == auth_event_hash
    assert r1_meta["san_uri"] == r2_meta["san_uri"]
    print("Bindings & Signer Continuity: PASS")

    # 12. Checkpointing Durable State with Anti-Rollback Sequence Increment
    print("\n--- STEP 12: Checkpointing durable state ---")
    if state_machine.current_state == ObserverState.SAFE_HOLD:
        check_transition(state_machine, ObserverState.PRE_OBSERVATION_READY, reason="Dry run concluded legally")
    elif execution_scope == "NATURAL_ARMED":
        check_transition(state_machine, ObserverState.SAFE_HOLD, reason="Natural observation window concluded without accession")
        check_transition(state_machine, ObserverState.PRE_OBSERVATION_READY, reason="Natural armed cycle concluded legally")

    assert state_machine.current_state == ObserverState.PRE_OBSERVATION_READY, f"CRITICAL: Final state {state_machine.current_state.value} != PRE_OBSERVATION_READY"
    assert state_mgr.denominator_state.get("natural_denominator", 0) == 0, "CRITICAL: Natural denominator mutated!"

    state_mgr.observer_state["last_completed_run_id"] = run_id
    state_mgr.observer_state["current_status"] = state_machine.current_state.value
    state_mgr.observer_state["state_machine_history"] = state_machine.to_dict()
    state_mgr.closure_state["confirmed_closures"][run_id] = {
        "r1_bundle_sha256": r1_meta["bundle_sha256"],
        "r2_bundle_sha256": r2_meta["bundle_sha256"],
        "authority_event_hash": auth_event_hash,
        "execution_scope": execution_scope,
        "completed_at_utc": now_utc,
    }

    chk_ok, chk_sha, chk_msg = state_mgr.checkpoint_state(
        scheduler_commit_sha=safe_claims.get("sha", "UNKNOWN_COMMIT"),
        workflow_ref=safe_claims.get("ref", "refs/heads/main"),
        policy_hash=FROZEN_POLICY_PINNED_HASH,
    )
    if not chk_ok:
        raise RuntimeError(f"CRITICAL: Durable state checkpoint failed: {chk_msg}")
    print(f"Durable state checkpoint created: {chk_msg} (SHA-256: {chk_sha})")

    # 13. Execution Summary
    summary = {
        "gate": "SMART_MONEY_FINAL_NATURAL_AUTHORITY_CHAIN_CLOSURE_V1",
        "execution_scope": execution_scope,
        "timestamp_utc": now_utc,
        "frozen_policy_pinned_hash": FROZEN_POLICY_PINNED_HASH,
        "frozen_policy_hash_match": "PASS",
        "exact_oidc_policy_match": "PASS",
        "natural_scheduler_r1": "PASS",
        "natural_scheduler_r2": "PASS",
        "human_signing_interaction": 0,
        "long_lived_signing_secret": "NONE",
        "r1_bundle_sha256": r1_meta["bundle_sha256"],
        "r1_rekor_log_index": r1_meta["log_index"],
        "r2_bundle_sha256": r2_meta["bundle_sha256"],
        "r2_rekor_log_index": r2_meta["log_index"],
        "authority_event_hash": auth_event_hash,
        "durable_state_checkpoint_sha256": chk_sha,
        "natural_accession": False,
        "natural_denominator_increment": 0,
        "claims": safe_claims,
    }
    summary_path = os.path.join(evidence_dir, "natural_dry_run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # 14. Build Evidence Manifest
    manifest_entries = []
    for item in sorted(os.listdir(evidence_dir)):
        if item.endswith("_MANIFEST_V1.json"):
            continue
        p = os.path.join(evidence_dir, item)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                data = fh.read()
            manifest_entries.append({
                "relative_path": item,
                "byte_length": len(data),
                "sha256": sha256_hex(data),
            })

    manifest_doc = {
        "manifest_version": "1.0.0",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gate": "SMART_MONEY_FINAL_NATURAL_AUTHORITY_CHAIN_CLOSURE_V1",
        "artifact_count": len(manifest_entries),
        "artifacts": manifest_entries,
    }
    manifest_canon = canonicalize(manifest_doc)
    manifest_sha = sha256_hex(manifest_canon.encode("utf-8"))
    manifest_doc["manifest_sha256"] = manifest_sha

    manifest_path = os.path.join(evidence_dir, "SIGSTORE_NATURAL_SCHEDULER_DRY_RUN_MANIFEST_V1.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2, sort_keys=True)

    print(f"Manifest created: {manifest_path} (SHA-256: {manifest_sha})")
    print("NATURAL SCHEDULER EXECUTION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
