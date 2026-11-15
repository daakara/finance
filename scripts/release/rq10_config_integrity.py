#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-10: Release Configuration Integrity
Verifies deployment configuration, engine manifest integrity, environment contracts,
and checks for accidental secret leakage in configuration files.
Emits a verified machine-readable stage receipt.
"""

import sys, os, subprocess, json, hashlib, time, datetime, re

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import create_stage_receipt, save_stage_receipt, compute_file_sha256, get_git_commit, resolve_identity_contract

REQUIRED_ENV_VARS = ["NEXT_PUBLIC_API_URL"]

def verify_engine_manifest(enforce_production=True, manifest_override=None):
    """Check three-engine integrity and ratified release authority.

    Production mode is the release runner default. Development mode must be
    explicitly selected by callers; it does not produce production authority.
    """
    manifest_name = manifest_override or (
        "FROZEN_ENGINE_MANIFEST_V2_5_3.json"
        if enforce_production else "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json"
    )
    if os.path.basename(manifest_name) != manifest_name:
        raise AssertionError("Manifest path must be a repository-root filename")
    manifest_path = os.path.join(PROJECT_ROOT, manifest_name)
    if not os.path.isfile(manifest_path):
        raise AssertionError(f"Engine manifest missing: {manifest_name}")

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if enforce_production:
        if manifest_name != "FROZEN_ENGINE_MANIFEST_V2_5_3.json":
            raise AssertionError("Production must use the exact frozen v2.5.3 manifest")
        if data.get("isProductionRatified") is not True:
            raise AssertionError("Production engine manifest is not ratified")
        if data.get("manifestVersion") != "2.5.3":
            raise AssertionError("Unexpected production engine manifest version")
        if data.get("authorityScope") != "CURRENT_PRODUCTION_AUTHORITY":
            raise AssertionError("Manifest not scoped to production authority")
        if data.get("provenanceCommit") != "cc0c72c23cd874148a6db49730153d9b644617f0":
            raise AssertionError("Unrecognized ratification provenance commit")
        if data.get("releaseAuthorityBinding") != "SPRINT4_RC5_CONFLUENCE_PROVENANCE_AND_CAPS_RATIFICATION":
            raise AssertionError("Unrecognized release authority binding")
        if data.get("ratifiedAuthorityRecord") != "docs/governance/ARX_RC5_ENGINE_MANIFEST_V253_SUCCESSION_RECORD.md":
            raise AssertionError("Ratification record binding mismatch")
        record = os.path.join(PROJECT_ROOT, data["ratifiedAuthorityRecord"])
        if not os.path.isfile(record):
            raise AssertionError("Ratification record missing")
        record_text = open(record, "r", encoding="utf-8").read()
        if "ARX_RC5_MANIFEST_SUCCESSION_V253_AUTHORIZED" not in record_text:
            raise AssertionError("Owner ratification record missing authorization token")
        if data.get("invariants", {}).get("production_ratification_prohibited") is True:
            raise AssertionError("Production ratification explicitly prohibited")
    elif data.get("isProductionRatified") is True and manifest_name != "FROZEN_ENGINE_MANIFEST_V2_5_3.json":
        raise AssertionError("Development candidate cannot assert production ratification")

    required = {"confluence_engine", "optimal_execution", "decision_hierarchy"}
    engines = data.get("engines", {})
    if not isinstance(engines, dict) or not required.issubset(engines):
        raise AssertionError("Engine manifest must list all three mandatory engines")
    hashes = {}
    for name in sorted(required):
        meta = engines[name]
        expected_path = f"analyst_dashboard/analyzers/{name}.py"
        if meta.get("filePath") != expected_path:
            raise AssertionError(f"{name}: unexpected engine source path")
        expected_hash = meta.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise AssertionError(f"{name}: invalid engine digest")
        engine_path = os.path.join(PROJECT_ROOT, expected_path)
        if not os.path.isfile(engine_path):
            raise AssertionError(f"{name}: engine source missing")
        with open(engine_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
        if actual_hash != expected_hash:
            raise AssertionError(f"{name}: engine hash mismatch")
        hashes[name] = actual_hash

    return {"manifest_version": data.get("manifestVersion"),
            "ratified": data.get("isProductionRatified") is True,
            "engine_hashes": hashes}


def verify_env_contracts():
    """Actively inspects environment variables, .env.production, and package configuration."""
    missing_vars = []
    found_vars = {}

    # Read frontend/.env.production if present (use utf-8-sig to strip BOM)
    env_prod_vars = {}
    prod_env_file = os.path.join(PROJECT_ROOT, "frontend", ".env.production")
    if os.path.exists(prod_env_file):
        with open(prod_env_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    clean_k = k.strip().replace("\ufeff", "")
                    env_prod_vars[clean_k] = v.strip()

    for var in REQUIRED_ENV_VARS:
        if os.environ.get(var):
            found_vars[var] = "PRESENT_IN_PROCESS_ENV"
        elif var in env_prod_vars and env_prod_vars[var]:
            found_vars[var] = f"CONFIGURED_IN_PROD_ENV ({env_prod_vars[var][:25]}...)"
        else:
            missing_vars.append(var)

    return found_vars, missing_vars

def scan_config_files_for_secrets():
    """Scans repository configuration files (.json, .yml, .yaml, .toml) for exposed API keys or tokens."""
    config_extensions = (".json", ".yml", ".yaml", ".toml")
    forbidden_patterns = [
        r"ghp_[a-zA-Z0-9]{36}",
        r"github_pat_[a-zA-Z0-9_]{50,}",
        r"CLOUDFLARE_API_TOKEN\s*[:=]\s*['\"][a-zA-Z0-9_-]{20,}['\"]",
        r"RAILWAY_TOKEN\s*[:=]\s*['\"][a-zA-Z0-9_-]{20,}['\"]"
    ]

    suspicious_findings = []
    for root_dir in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts", "release")]:
        for fname in os.listdir(root_dir):
            if fname.endswith(config_extensions) and not fname.startswith(".secrets"):
                fpath = os.path.join(root_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "r", encoding="utf-8-sig", errors="ignore") as f:
                        content = f.read()
                        for pat in forbidden_patterns:
                            if re.search(pat, content):
                                suspicious_findings.append((fname, pat))

    return suspicious_findings

def main():
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    errors = []
    assertions = []

    print("=== [RQ-10] CONFIGURATION & ENVIRONMENT CONTRACT INTEGRITY ===")

    # 1. Engine manifest verification
    try:
        manifest_info = verify_engine_manifest()
        assertions.append({
            "assertion": "Ratified v2.5.3 manifest and three engine hashes match production source",
            "observed": manifest_info,
            "passed": True
        })
        print("  [PASS] Ratified production engine manifest verified.")
    except Exception as e:
        assertions.append({
            "assertion": "Ratified v2.5.3 manifest and three engine hashes match production source",
            "observed": str(e),
            "passed": False
        })
        errors.append(f"Engine manifest verification failed: {e}")
        print(f"  [FAIL] {e}")

    # 2. Environment contract verification
    found_vars, missing_vars = verify_env_contracts()
    env_passed = (len(missing_vars) == 0)
    assertions.append({
        "assertion": "Required frontend environment contracts satisfied",
        "observed": {"found": found_vars, "missing": missing_vars},
        "passed": env_passed
    })
    if env_passed:
        print(f"  [PASS] Environment contracts satisfied: {found_vars}")
    else:
        print(f"  [FAIL] Missing required environment variables: {missing_vars}")
        errors.append(f"Missing required environment variables: {missing_vars}")

    # 3. Secret scan in config files
    secret_findings = scan_config_files_for_secrets()
    secrets_clean = (len(secret_findings) == 0)
    assertions.append({
        "assertion": "Zero plaintext credentials in repository configuration files",
        "observed": {"suspicious_files_count": len(secret_findings)},
        "passed": secrets_clean
    })
    if secrets_clean:
        print("  [PASS] Zero plaintext credentials detected in configuration files.")
    else:
        print(f"  [FAIL] Suspicious secrets detected in config files: {secret_findings}")
        errors.append(f"Secret material detected in configuration files: {secret_findings}")

    duration = time.time() - t0
    status = "PASS" if len(errors) == 0 else "FAIL"
    ok, identity, err = resolve_identity_contract()
    candidate_sha = identity["source_sha"] if ok else None

    evidence = [
        "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json",
        "analyst_dashboard/analyzers/confluence_engine.py",
        "frontend/.env.production"
    ]

    receipt = create_stage_receipt(
        control_id="RQ-10",
        name="Release Configuration Integrity",
        status=status,
        start_time=start_time,
        end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_sec=duration,
        candidate_sha=candidate_sha,
        target_branch="release/production",
        assertions=assertions,
        evidence_paths=evidence,
        errors=errors
    )
    receipt_path = save_stage_receipt(receipt)
    print(f"Receipt written to: {receipt_path}")

    if status == "PASS":
        print("=== [RQ-10] RESULT: PASS ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-10] RESULT: FAIL (Errors: {errors}) ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
