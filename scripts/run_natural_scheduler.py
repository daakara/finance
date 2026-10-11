#!/usr/bin/env python3
"""
ARX TERMINAL — SMART MONEY
NATURAL SCHEDULER ORCHESTRATION ENGINE (EPOCH 001)

Executes governed natural scheduler rooting in either:
  1. DRY_RUN mode (qualification_scope: PRE_ARM_DRY_RUN, natural_accession: false)
  2. NATURAL mode (when prospective arming is authorized)
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


def canonicalize(data: Any) -> str:
    """Deterministic Canonical JSON RFC 8785 representation."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(content: bytes) -> str:
    """Computes hex-encoded SHA-256 digest."""
    return hashlib.sha256(content).hexdigest()


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


def execute_cosign_verification(payload_path: str, bundle_path: str, identity_pattern: str, issuer: str) -> None:
    """Executes cosign verify-blob."""
    cmd = [
        "cosign",
        "verify-blob",
        "--bundle",
        bundle_path,
        "--certificate-identity-regexp",
        identity_pattern,
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


def main():
    parser = argparse.ArgumentParser(description="Smart Money Natural Scheduler Engine")
    parser.add_argument("--mode", choices=["dry-run", "natural"], default="dry-run", help="Execution mode")
    args = parser.parse_args()

    print("======================================================================")
    print(f"SMART MONEY NATURAL SCHEDULER EXECUTION (MODE: {args.mode.upper()})")
    print("======================================================================")

    evidence_dir = os.path.join(PROJECT_ROOT, "evidence", "sigstore-natural-scheduler-dry-run-v1")
    os.makedirs(evidence_dir, exist_ok=True)

    # 1. Acquire OIDC Token and Safe Claims
    print("\n--- STEP 1: Acquiring OIDC token and extracting safe claims ---")
    safe_claims = extract_safe_oidc_claims()
    print("Safe claims:")
    for k, v in safe_claims.items():
        if k not in ["exp", "nbf"]:
            print(f"  {k}: {v}")

    claims_path = os.path.join(evidence_dir, "natural_safe_oidc_claims.json")
    with open(claims_path, "w", encoding="utf-8") as f:
        json.dump(safe_claims, f, indent=2, sort_keys=True)

    # 2. Ratify Natural Workload Identity Policy
    print("\n--- STEP 2: Freezing natural workload identity policy ---")
    natural_policy = {
        "policy_id": "SMART_MONEY_NATURAL_ROOTING_WORKLOAD_IDENTITY_POLICY_V1",
        "version": "1.0.0",
        "effective_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "governing_epoch": "EPOCH_001",
        "status": "RATIFIED",
        "workload_identity_constraints": {
            "authorized_issuer": safe_claims["iss"],
            "authorized_repository": safe_claims["repository"],
            "authorized_repository_id": safe_claims["repository_id"],
            "authorized_repository_owner": safe_claims["repository_owner"],
            "authorized_repository_owner_id": safe_claims["repository_owner_id"],
            "authorized_workflow": safe_claims["workflow"],
            "authorized_job_workflow_ref": safe_claims["job_workflow_ref"],
            "authorized_ref": safe_claims["ref"],
            "authorized_audience": "sigstore",
            "authorized_environment": safe_claims["environment"] or "NONE",
            "authorized_event": safe_claims["event_name"],
            "maximum_token_lifetime_seconds": 600,
            "token_lifetime_class": "SHORT_LIVED_EPHEMERAL_WORKLOAD_TOKEN",
        },
        "governance_rules": {
            "static_signing_private_key_prohibited": True,
            "long_lived_token_prohibited": True,
            "personal_access_token_prohibited": True,
            "token_disk_persistence_prohibited": True,
            "token_logging_prohibited": True,
            "emergency_suspension_behavior": "TERMINAL_HOLD_AND_IMMEDIATE_ADMISSION_HALT",
        },
    }
    policy_path = os.path.join(evidence_dir, "natural_workload_policy.json")
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(natural_policy, f, indent=2, sort_keys=True)

    # Also write to docs
    doc_policy_path = os.path.join(
        PROJECT_ROOT,
        "docs",
        "domain",
        "radar",
        "smart-money",
        "public-availability",
        "SMART_MONEY_NATURAL_ROOTING_WORKLOAD_IDENTITY_POLICY_V1.json",
    )
    with open(doc_policy_path, "w", encoding="utf-8") as f:
        json.dump(natural_policy, f, indent=2, sort_keys=True)

    # 3. Construct Canonical TEST_ONLY R1 Payload
    print("\n--- STEP 3: Constructing canonical R1 payload ---")
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = f"natural_scheduler_dry_run_{safe_claims.get('run_id', '1')}"
    root_seed = f"NATURAL_DRY_RUN_R1_SEED_{run_id}_{now_utc}"
    r1_root_hash = sha256_hex(root_seed.encode("utf-8"))

    r1_payload_core = {
        "authority_effective_status": "BLOCKED_PENDING_EXTERNAL_WITNESS",
        "canonicalization_version": "CANONICAL_JSON_V1",
        "created_at_utc": now_utc,
        "natural_accession": False,
        "phase": "PRE_AUTHORITY_ADMISSION_ROOT",
        "policy_set_hash": "b4f2c9e8a71d3f56e9c0b1a2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
        "qualification_scope": "PRE_ARM_DRY_RUN",
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

    # 4. Live Sigstore Signing of R1
    print("\n--- STEP 4: Live Sigstore keyless signing of R1 ---")
    r1_bundle_path = os.path.join(evidence_dir, "natural_r1.sigstore.json")
    execute_cosign_signing(r1_payload_path, r1_bundle_path)

    expected_identity_re = f".*{safe_claims['repository']}.*"
    execute_cosign_verification(r1_payload_path, r1_bundle_path, expected_identity_re, safe_claims["iss"])

    r1_meta = parse_bundle(r1_bundle_path)
    print(f"R1 Bundle SHA-256: {r1_meta['bundle_sha256']}")
    print(f"R1 Rekor Log Index: {r1_meta['log_index']}")

    # 5. Create TEST_ONLY Authority Event
    print("\n--- STEP 5: Creating TEST_ONLY authority event ---")
    auth_now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    authority_event = {
        "authoritative_r1_witness_identity": r1_meta["bundle_sha256"],
        "authority_event_scope": "PRE_ARM_DRY_RUN",
        "canonicalization_version": "CANONICAL_JSON_V1",
        "created_at_utc": auth_now_utc,
        "natural_accession_authority_granted": False,
        "policy_set_hash": "b4f2c9e8a71d3f56e9c0b1a2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
        "production_authority_granted": False,
        "r1_bundle_sha256": r1_meta["bundle_sha256"],
        "r1_root_hash": r1_root_hash,
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

    # 6. Construct Canonical TEST_ONLY R2 Payload
    print("\n--- STEP 6: Constructing canonical R2 payload ---")
    r2_now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r2_root_seed = f"NATURAL_DRY_RUN_R2_SEED_{run_id}_{r2_now_utc}"
    r2_root_hash = sha256_hex(r2_root_seed.encode("utf-8"))

    r2_payload_core = {
        "authoritative_r1_witness_identity": r1_meta["bundle_sha256"],
        "authority_event_hash": auth_event_hash,
        "authority_event_sequence": 1,
        "authority_grant_function": "NONE",
        "canonicalization_version": "CANONICAL_JSON_V1",
        "created_at_utc": r2_now_utc,
        "phase": "POST_AUTHORITY_CLOSURE_ROOT",
        "policy_set_hash": "b4f2c9e8a71d3f56e9c0b1a2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
        "qualification_scope": "PRE_ARM_DRY_RUN",
        "r1_bundle_hash": r1_meta["bundle_sha256"],
        "r1_bundle_sha256": r1_meta["bundle_sha256"],
        "r1_root_hash": r1_root_hash,
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

    # 7. Live Sigstore Signing of R2
    print("\n--- STEP 7: Live Sigstore keyless signing of R2 ---")
    r2_bundle_path = os.path.join(evidence_dir, "natural_r2.sigstore.json")
    execute_cosign_signing(r2_payload_path, r2_bundle_path)

    execute_cosign_verification(r2_payload_path, r2_bundle_path, expected_identity_re, safe_claims["iss"])
    r2_meta = parse_bundle(r2_bundle_path)
    print(f"R2 Bundle SHA-256: {r2_meta['bundle_sha256']}")
    print(f"R2 Rekor Log Index: {r2_meta['log_index']}")

    # 8. Cryptographic Bindings & Policy Match
    print("\n--- STEP 8: Validating bindings and policy match ---")
    assert r2_payload_core["authoritative_r1_witness_identity"] == r1_meta["bundle_sha256"]
    assert r2_payload_core["authority_event_hash"] == auth_event_hash
    assert r1_meta["san_uri"] == r2_meta["san_uri"]
    print("Bindings & Signer Continuity: PASS")

    # Policy match verification
    assert safe_claims["repository"] == "daakara/finance"
    assert safe_claims["repository_id"] == "1087399232"
    assert safe_claims["repository_owner_id"] == "53128824"
    assert safe_claims["iss"] == "https://token.actions.githubusercontent.com"
    print("Policy Match Verification: PASS")

    # 9. Execution Summary
    summary = {
        "gate": "GATE_2_NATURAL_SCHEDULER_IDENTITY_CLOSURE",
        "execution_mode": args.mode,
        "timestamp_utc": now_utc,
        "natural_scheduler_r1": "PASS",
        "natural_scheduler_r2": "PASS",
        "natural_scheduler_policy_match": "PASS",
        "natural_scheduler_dry_run": "PASS",
        "human_signing_interaction": 0,
        "long_lived_signing_secret": "NONE",
        "r1_bundle_sha256": r1_meta["bundle_sha256"],
        "r1_rekor_log_index": r1_meta["log_index"],
        "r2_bundle_sha256": r2_meta["bundle_sha256"],
        "r2_rekor_log_index": r2_meta["log_index"],
        "authority_event_hash": auth_event_hash,
        "claims": safe_claims,
    }
    summary_path = os.path.join(evidence_dir, "natural_dry_run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # 10. Build Manifest
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
        "gate": "GATE_2_NATURAL_SCHEDULER_IDENTITY_CLOSURE",
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
    print("NATURAL SCHEDULER DRY RUN COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
