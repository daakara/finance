"""
Tests for Phase 1A-1E Scope Boundaries.

Verifies:
1. Zero authentication middleware or session handling introduced.
2. Zero billing, checkout, or stripe integration introduced.
3. Zero database schema migrations or DDL changes introduced.
4. Zero existing API routes modified or rewired.
5. Zero frontend UI pages or components created or modified.
"""

import os
import subprocess
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPECTED_BASE_SHA = "d20ec394133261df84885fb2d8c6f941a5b9ba19"


def test_no_auth_middleware():
    """Verify api/ directory contains no auth/session middleware."""
    api_dir = os.path.join(REPO_ROOT, "api")
    for root, _, files in os.walk(api_dir):
        for fname in files:
            fname_lower = fname.lower()
            assert not any(
                bad in fname_lower
                for bad in ["auth_middleware", "session_middleware", "jwt", "oauth", "passport"]
            ), f"Prohibited auth module found: {fname}"


def test_no_billing_or_payment_modules():
    """Verify api/ and frontend/ contain zero billing or payment files."""
    for sub in ["api", "frontend"]:
        target_dir = os.path.join(REPO_ROOT, sub)
        for root, _, files in os.walk(target_dir):
            for fname in files:
                fname_lower = fname.lower()
                assert not any(
                    bad in fname_lower
                    for bad in ["stripe", "billing", "checkout", "subscription_service", "invoice"]
                ), f"Prohibited billing file found: {os.path.join(root, fname)}"


def test_no_database_migrations_added():
    """Verify no migration files were added in this branch."""
    cmd = ["git", "diff", "--name-only", EXPECTED_BASE_SHA, "HEAD"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    changed_files = [f for f in proc.stdout.splitlines() if f.strip()]

    # Also check untracked files
    cmd_untracked = ["git", "status", "--porcelain=v1", "--untracked-files=all"]
    proc_untracked = subprocess.run(cmd_untracked, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc_untracked.returncode == 0
    all_files = changed_files + [
        line[3:].strip() for line in proc_untracked.stdout.splitlines() if line.startswith("??")
    ]

    for f in all_files:
        assert not any(
            migration_marker in f.lower()
            for migration_marker in ["migration", "alembic", "schema.sql", "upgrade.sql"]
        ), f"Migration file detected: {f}"


def test_no_unauthorized_routes_modified():
    """Verify only explicitly authorized private routes were modified, and all public routes remain untouched."""
    cmd = ["git", "diff", "--name-only", EXPECTED_BASE_SHA, "api/routes"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    modified_routes = [f.strip().replace("\\", "/") for f in proc.stdout.splitlines() if f.strip()]
    authorized_routes = {
        "api/routes/portfolio.py",
        "api/routes/journal.py",
        "api/routes/cockpit.py",
    }
    for route in modified_routes:
        assert route in authorized_routes, f"Unauthorized route file modified: {route}"


def test_no_frontend_pages_or_components_modified():
    """Verify frontend pages and components are completely untouched."""
    for folder in ["frontend/app", "frontend/components"]:
        cmd = ["git", "diff", "--name-only", EXPECTED_BASE_SHA, folder]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        assert proc.returncode == 0
        diff_output = proc.stdout.strip()
        assert diff_output == "", f"Files in {folder} were modified:\n{diff_output}"
