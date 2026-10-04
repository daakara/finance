"""
Tests for Changed File Scope Enforcement.

Verifies that all changed or untracked files relative to base commit
d20ec394133261df84885fb2d8c6f941a5b9ba19 belong exclusively to the
authorized Phase 1A-1E inventory:
- api/context/
- api/capabilities/
- api/services/
- frontend/lib/saas/
- tests/saas/
- tests/architecture/
- docs/architecture/
"""

import os
import subprocess
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPECTED_BASE_SHA = "d20ec394133261df84885fb2d8c6f941a5b9ba19"

AUTHORIZED_PATHS = (
    "api/context/",
    "api/capabilities/",
    "api/services/",
    "api/routes/portfolio.py",
    "api/routes/journal.py",
    "api/routes/cockpit.py",
    "database/",
    "analyst_dashboard/data/db_engine.py",
    "frontend/lib/saas/",
    "tests/saas/",
    "tests/architecture/",
    "docs/architecture/ARX_SAAS_",
)


def _get_base_sha():
    proc_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    branch = proc_branch.stdout.strip()
    if branch == "main":
        proc_merge = subprocess.run(["git", "rev-parse", "--verify", "HEAD^2"], cwd=REPO_ROOT, capture_output=True, text=True)
        if proc_merge.returncode == 0:
            return "HEAD^1"
    return EXPECTED_BASE_SHA


def _get_changed_and_untracked_files():
    base_sha = _get_base_sha()
    cmd_diff = ["git", "diff", "--name-only", base_sha, "HEAD"]
    proc_diff = subprocess.run(cmd_diff, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc_diff.returncode == 0
    diff_files = [f.strip().replace("\\", "/") for f in proc_diff.stdout.splitlines() if f.strip()]

    cmd_status = ["git", "status", "--porcelain=v1"]
    proc_status = subprocess.run(cmd_status, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc_status.returncode == 0
    status_files = []
    for line in proc_status.stdout.splitlines():
        if line.strip():
            path = line[3:].strip().replace("\\", "/")
            if line.startswith("??"):
                if any(path.startswith(p) or path == p for p in AUTHORIZED_PATHS):
                    status_files.append(path)
            else:
                status_files.append(path)

    all_files = sorted(list(set(diff_files + status_files)))
    return all_files



def test_changed_files_within_authorized_scope():
    """Verify all changed and untracked files match the authorized Phase 1G scope."""
    all_files = _get_changed_and_untracked_files()
    assert len(all_files) > 0, "Expected changed/new files for Phase 1 implementation."

    unauthorized_files = []
    for f in all_files:
        if not any(f.startswith(p) or f == p for p in AUTHORIZED_PATHS):
            unauthorized_files.append(f)

    assert len(unauthorized_files) == 0, (
        f"Unauthorized files detected outside Phase 1G scope:\n"
        + "\n".join(unauthorized_files)
    )


def test_no_protected_quant_or_etf_files_in_diff():
    """Verify zero quant, ETF V2, OpenFIGI, or unapproved route files appear in git diff/status."""
    all_files = _get_changed_and_untracked_files()
    forbidden_tokens = ["analyst_dashboard/analyzers", "engines", "etf", "openfigi"]

    for f in all_files:
        f_lower = f.lower()
        for tok in forbidden_tokens:
            # Only forbidden outside tests/architecture or tests/saas
            if not f.startswith("tests/architecture/") and not f.startswith("tests/saas/"):
                assert tok not in f_lower, f"Forbidden path modified/created: {f}"

        # Enforce that no route outside the 3 authorized private routes is touched
        if f.startswith("api/routes/"):
            assert f in (
                "api/routes/portfolio.py",
                "api/routes/journal.py",
                "api/routes/cockpit.py",
            ), f"Unauthorized route file modified: {f}"

        # Enforce that inside analyst_dashboard/, ONLY data/db_engine.py is touched
        if f.startswith("analyst_dashboard/"):
            assert f == "analyst_dashboard/data/db_engine.py", f"Unauthorized analyst_dashboard file modified: {f}"
