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
from rq_receipt_utils import create_stage_receipt, save_stage_receipt, compute_file_sha256, get_git_commit

REQUIRED_ENV_VARS = ["NEXT_PUBLIC_API_URL"]

def verify_engine_manifest():
    manifest_path = os.path.join(PROJECT_ROOT, "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json")
    if not os.path.exists(manifest_path):
        raise AssertionError("CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json not found in repository root.")
    
    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    
    confluence_meta = data.get("engines", {}).get("confluence_engine", {})
    expected_hash = confluence_meta.get("sha256")
    if not expected_hash:
        raise AssertionError("confluence_engine entry in CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json missing sha256.")

    confluence_file = os.path.join(PROJECT_ROOT, "analyst_dashboard", "analyzers", "confluence_engine.py")
    if not os.path.exists(confluence_file):
        raise AssertionError(f"Confluence engine file not found at {confluence_file}")

    actual_hash = compute_file_sha256(confluence_file)
    if actual_hash.lower() != expected_hash.lower():
        raise AssertionError(f"Confluence engine hash mismatch: expected {expected_hash[:10]}... got {actual_hash[:10]}...")

    return {"manifest_version": data.get("manifest_version"), "engine_hash": actual_hash}

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
            "assertion": "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json hash matches source",
            "observed": manifest_info,
            "passed": True
        })
        print("  [PASS] Candidate engine manifest verified.")
    except Exception as e:
        assertions.append({
            "assertion": "CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json hash matches source",
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
    candidate_sha = get_git_commit("HEAD")

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