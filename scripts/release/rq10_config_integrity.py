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

# Owner-ratified v2.5.3 source binding; changing these constants requires a reviewed
# successor ratification, not merely toggling a JSON Boolean.
EXPECTED_V253_COMMIT = "cc0c72c23cd874148a6db49730153d9b644617f0"
EXPECTED_V253_AUTHORITY = "SPRINT4_RC5_CONFLUENCE_PROVENANCE_AND_CAPS_RATIFICATION"
EXPECTED_V253_TOKEN = "ARX_RC5_MANIFEST_SUCCESSION_V253_AUTHORIZED"
EXPECTED_V253_DIGESTS = {
    "confluence_engine": ("analyst_dashboard/analyzers/confluence_engine.py", "544851fadf3e8c5e0ae3a4390cb7c94ef223c1d4aacf9e7c735ee1eda8404ce2"),
    "optimal_execution": ("analyst_dashboard/analyzers/optimal_execution.py", "df2dc6050bf6fe91cae442e1872a3eccc40520b990f4efb4af63f79311a31dec"),
    "decision_hierarchy": ("analyst_dashboard/analyzers/decision_hierarchy.py", "b0d41de56c34a46000496e2cf80afdfd76a3ff19ffc18fbdd80f430050c375b3"),
}
V253_MANIFEST = "FROZEN_ENGINE_MANIFEST_V2_5_3.json"
V253_ADR = "docs/governance/ARX_RC5_ENGINE_MANIFEST_V253_SUCCESSION_RECORD.md"

def verify_engine_manifest(target_branch="release/production"):
    """RQ-10: reject candidate-only authority and verify three owner-approved hashes.

    The approved chat decision is recorded in the ADR. This check establishes
    content consistency, not a cryptographic signature; repository approval and
    signed-commit protection remain independent release obligations.
    """
    clean_target = (target_branch or "").replace("refs/heads/", "").strip()
    if clean_target not in ("release/production", "main"):
        raise AssertionError(f"Unrecognized manifest qualification target: {target_branch!r}")

    manifest_path = os.path.join(PROJECT_ROOT, V253_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise AssertionError(f"Ratified production manifest missing: {V253_MANIFEST}")

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if data.get("isProductionRatified") is not True:
        raise AssertionError("Production release candidate rejected: manifest is not ratified")
    expected_fields = {
        "manifestVersion": "2.5.3",
        "frozenStrategyVersion": "2.5.3",
        "authorityScope": "CURRENT_PRODUCTION_AUTHORITY",
        "provenanceCommit": EXPECTED_V253_COMMIT,
        "releaseAuthorityBinding": EXPECTED_V253_AUTHORITY,
        "ratifiedAuthorityRecord": V253_ADR,
        "supersedesManifest": "FROZEN_ENGINE_MANIFEST_V2_5_2.json",
    }
    for field, expected in expected_fields.items():
        if data.get(field) != expected:
            raise AssertionError(f"Unapproved manifest field {field}: {data.get(field)!r}")

    if data.get("invariants", {}).get("zero_production_mutation") is not True:
        raise AssertionError("Missing zero-production-mutation invariant")

    engines = data.get("engines")
    if not isinstance(engines, dict) or set(engines) != set(EXPECTED_V253_DIGESTS):
        raise AssertionError("Ratified manifest must contain exactly the three authorized engines")

    actual_hashes = {}
    for name, (approved_path, approved_hash) in EXPECTED_V253_DIGESTS.items():
        meta = engines[name]
        if (meta.get("filePath") != approved_path
                or meta.get("sha256") != approved_hash
                or meta.get("ratificationCommit") != EXPECTED_V253_COMMIT):
            raise AssertionError(f"Unauthorized manifest binding for {name}")
        full_path = os.path.join(PROJECT_ROOT, approved_path)
        if not os.path.isfile(full_path):
            raise AssertionError(f"Missing authorized engine source: {approved_path}")
        actual = compute_file_sha256(full_path)
        if actual.lower() != approved_hash:
            raise AssertionError(f"Engine digest mismatch for {name}: expected {approved_hash[:12]}, got {actual[:12]}")
        actual_hashes[name] = actual

    adr_path = os.path.join(PROJECT_ROOT, V253_ADR)
    if not os.path.isfile(adr_path):
        raise AssertionError("Owner succession decision record is missing")
    with open(adr_path, "r", encoding="utf-8") as f:
        adr = f.read()
    if EXPECTED_V253_TOKEN not in adr or EXPECTED_V253_COMMIT not in adr:
        raise AssertionError("Owner succession decision record lacks expected binding")
    if any(expected_hash not in adr for _, expected_hash in EXPECTED_V253_DIGESTS.values()):
        raise AssertionError("Owner succession decision record has incomplete engine digest binding")

    # Check the ledger registration without importing optional analytics dependencies.
    import ast
    ledger_path = os.path.join(PROJECT_ROOT, "analyst_dashboard", "governance", "experiment_ledger.py")
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger_tree = ast.parse(f.read(), filename=ledger_path)
    cls = next((node for node in ledger_tree.body if isinstance(node, ast.ClassDef) and node.name == "ExperimentLedger"), None)
    if cls is None:
        raise AssertionError("ExperimentLedger governance binding class missing")
    attributes = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("AUTHORITY_VERSION_BINDINGS", "CURRENT_AUTHORITY_MANIFEST", "CURRENT_RATIFIED_AUTHORITY"):
                attributes[node.targets[0].id] = ast.literal_eval(node.value)
    binding = attributes.get("AUTHORITY_VERSION_BINDINGS", {}).get("2.5.3", {})
    if (binding.get("manifestFile") != V253_MANIFEST
            or binding.get("provenanceCommit") != EXPECTED_V253_COMMIT
            or binding.get("releaseAuthorityBinding") != EXPECTED_V253_AUTHORITY
            or binding.get("ratifiedAuthorityRecord") != V253_ADR
            or attributes.get("CURRENT_AUTHORITY_MANIFEST") != V253_MANIFEST
            or attributes.get("CURRENT_RATIFIED_AUTHORITY") != EXPECTED_V253_AUTHORITY):
        raise AssertionError("ExperimentLedger ratified production authority registration mismatch")
    declared_manifest_digest = binding.get("expectedDigest")
    actual_manifest_digest = compute_file_sha256(manifest_path)
    if not declared_manifest_digest or declared_manifest_digest != actual_manifest_digest:
        raise AssertionError("Frozen manifest content digest mismatch against ledger authority binding")

    return {"manifest_version": "2.5.3", "engine_hashes": actual_hashes,
            "manifest_digest": actual_manifest_digest, "authority": EXPECTED_V253_AUTHORITY}

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
        ok, identity, identity_error = resolve_identity_contract()
        if not ok:
            raise AssertionError(f"Release identity unavailable for manifest verification: {identity_error}")
        manifest_info = verify_engine_manifest(identity["target_branch"])
        assertions.append({
            "assertion": "Ratified v2.5.3 three-engine and governance authority matches source",
            "observed": manifest_info,
            "passed": True
        })
        print("  [PASS] Ratified v2.5.3 engine manifest and ledger authority verified.")
    except Exception as e:
        assertions.append({
            "assertion": "Ratified v2.5.3 three-engine and governance authority matches source",
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
        V253_MANIFEST,
        V253_ADR,
        "analyst_dashboard/governance/experiment_ledger.py",
        "analyst_dashboard/analyzers/confluence_engine.py",
        "analyst_dashboard/analyzers/optimal_execution.py",
        "analyst_dashboard/analyzers/decision_hierarchy.py",
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
