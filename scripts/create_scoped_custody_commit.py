#!/usr/bin/env python3
"""
Create scoped, isolated commit and branch for Phase 2 Evidence Custody.
Does not disrupt or modify the primary Git index or concurrent working tree changes.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_COMMIT = "1d089301940b8b4125c0d060af6e89926b11a20f"
TARGET_BRANCH = "arx/ux-phase2-secure-evidence-custody"
COMMIT_MSG = """feat(ux-research): Phase 2 secure evidence custody foundation and verification

- Gate A: Repository exposure containment (.gitignore + verifier script)
- Gate B: Custody threat model (14 threat scenarios)
- Gate C: Local restricted evidence vault (off-Git root, NTFS/POSIX ACLs)
- Gate D: Cryptographic contract (AES-256-GCM AEAD, pre-parse SHA-256)
- Gate E: Controlled synthetic intake pipeline (10-step atomic intake)
- Gate F: Tamper-evident hash-chained custody receipt ledger
- Gate G: Mutual blinding enforcement and qualification scoring bridge
- Gate H: Data retention and GDPR Article 17 participant withdrawal workflow
- Gate I: External custody architecture specification (Cloudflare R2 EU)
- Gate J: 24/24 adversarial security test suite passed
- Gate K: 51/51 regression tests passed, canonical package byte-identical

Authority: ARX Closure Model v2
Disposition: ARX_UX_CUSTODY_LOCAL_IMPLEMENTATION_VERIFIED
"""

SCOPED_FILES = [
    ".gitignore",
    "analyst_dashboard/research/custody/__init__.py",
    "analyst_dashboard/research/custody/exceptions.py",
    "analyst_dashboard/research/custody/crypto.py",
    "analyst_dashboard/research/custody/vault.py",
    "analyst_dashboard/research/custody/ledger.py",
    "analyst_dashboard/research/custody/intake.py",
    "analyst_dashboard/research/custody/scoring_bridge.py",
    "analyst_dashboard/research/custody/retention.py",
    "scripts/verify_evidence_custody_containment.py",
    "scripts/create_scoped_custody_commit.py",
    "tests/test_evidence_custody_security.py",
    "tests/test_observer_distribution_integrity.py",
    "tests/test_scoring_utility_regression.py",
    "tests/fixtures/qualification/ARX_OBSERVER_QUALIFICATION_BUNDLE.zip",
    "tests/fixtures/qualification/custodian_package/ARX_UX_CALIBRATION_GOLD_KEY_CORRECTION_LEDGER.md",
    "tests/fixtures/qualification/custodian_package/gold_key_alpha_v2.json",
    "tests/fixtures/qualification/custodian_package/gold_key_beta_v2.json",
    "tests/fixtures/qualification/custodian_package/gold_key_consolidated_30.json",
    "tests/fixtures/qualification/custodian_package/gold_key_gamma_v1.json",
    "tests/fixtures/qualification/custodian_package/PACKAGE_INTEGRITY_MANIFEST.json",
    "tests/fixtures/qualification/custodian_package/scoring_utility.py",
    "tests/fixtures/qualification/observer_package/ARX_UX_OBSERVER_SCORING_HANDBOOK_V2.md",
    "tests/fixtures/qualification/observer_package/OBSERVER_RATING_SCHEMA.json",
    "tests/fixtures/qualification/observer_package/SUBMISSION_INSTRUCTIONS.md",
    "tests/fixtures/qualification/observer_package/TRANS_ALPHA_BLINDED.json",
    "tests/fixtures/qualification/observer_package/TRANS_BETA_BLINDED.json",
    "tests/fixtures/qualification/observer_package/TRANS_GAMMA_BLINDED.json",
    "tests/fixtures/qualification/submissions/SUBMISSION_LOCKBOX_POLICY.md",
    "ARX_UX_CUSTODY_IMPLEMENTATION_SOURCE_ATTESTATION.md",
    "ARX_UX_GIT_EXPOSURE_CONTAINMENT_RECEIPT.md",
    "ARX_UX_CUSTODY_THREAT_MODEL.md",
    "ARX_UX_LOCAL_RESTRICTED_VAULT_IMPLEMENTATION.md",
    "ARX_UX_CUSTODY_CRYPTOGRAPHIC_CONTRACT.md",
    "ARX_UX_SYNTHETIC_INTAKE_PIPELINE_VERIFICATION.md",
    "ARX_UX_CUSTODY_RECEIPT_INTEGRITY_AUDIT.md",
    "ARX_UX_MUTUAL_BLINDING_SECURITY_TESTS.md",
    "ARX_UX_DATA_RETENTION_AND_WITHDRAWAL.md",
    "ARX_UX_EXTERNAL_CUSTODY_ARCHITECTURE_DECISION.md",
    "ARX_UX_CUSTODY_REGRESSION_EVIDENCE.md",
    "ARX_UX_CUSTODY_FINAL_IMPLEMENTATION_DECISION.md",
]


def main():
    isolated_index = REPO_ROOT / ".git" / "custody_temp_index"
    if isolated_index.exists():
        isolated_index.unlink()

    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = str(isolated_index)

    try:
        # Step 1: Read base tree into isolated index
        print(f"Reading base commit tree: {BASE_COMMIT}...")
        subprocess.run(
            ["git", "read-tree", f"{BASE_COMMIT}^{{tree}}"],
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
        )

        # Step 2: Add scoped files into isolated index
        print(f"Adding {len(SCOPED_FILES)} scoped files to isolated index...")
        for file_path in SCOPED_FILES:
            full_path = REPO_ROOT / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"Scoped file not found: {file_path}")
            subprocess.run(
                ["git", "add", file_path],
                cwd=str(REPO_ROOT),
                env=env,
                check=True,
            )

        # Step 3: Write isolated tree
        res = subprocess.run(
            ["git", "write-tree"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        new_tree = res.stdout.strip()
        print(f"Isolated tree created: {new_tree}")

        # Step 4: Commit tree with parent BASE_COMMIT
        res = subprocess.run(
            ["git", "commit-tree", new_tree, "-p", BASE_COMMIT, "-m", COMMIT_MSG],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        new_commit = res.stdout.strip()
        print(f"Scoped commit created: {new_commit}")

        # Step 5: Update or create target branch
        subprocess.run(
            ["git", "branch", "-f", TARGET_BRANCH, new_commit],
            cwd=str(REPO_ROOT),
            check=True,
        )
        print(f"Branch '{TARGET_BRANCH}' pointing to commit {new_commit}")

        # Step 6: Verify commit contents
        res = subprocess.run(
            ["git", "show", "--stat", "--oneline", new_commit],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        print("\nCommit verification:")
        print(res.stdout[:500] + "...")

    finally:
        if isolated_index.exists():
            isolated_index.unlink()


if __name__ == "__main__":
    main()
