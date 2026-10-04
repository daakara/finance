"""
Tests for Repository Isolation and Base Lineage.

Verifies:
1. Base commit is exactly d20ec394133261df84885fb2d8c6f941a5b9ba19.
2. Branch is feat/arx-saas-foundation-phase1-seams.
3. Worktree path is isolated from main working tree.
"""

import os
import subprocess
import pytest

EXPECTED_BASE_SHA = "d20ec394133261df84885fb2d8c6f941a5b9ba19"
EXPECTED_BRANCH = "feat/arx-saas-foundation-phase1-seams"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_base_commit_lineage():
    """Verify the current branch descends directly from the verified base SHA."""
    cmd = ["git", "merge-base", "HEAD", EXPECTED_BASE_SHA]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"git merge-base failed: {proc.stderr}"
    base_sha = proc.stdout.strip()
    assert base_sha == EXPECTED_BASE_SHA, (
        f"Base commit mismatch. Expected {EXPECTED_BASE_SHA}, got {base_sha}"
    )


def test_current_branch():
    """Verify working on the dedicated feature branch."""
    cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"git rev-parse failed: {proc.stderr}"
    branch = proc.stdout.strip()
    assert branch == EXPECTED_BRANCH, (
        f"Branch mismatch. Expected {EXPECTED_BRANCH}, got {branch}"
    )


def test_worktree_isolation():
    """Verify repository path is in an isolated worktree directory."""
    worktree_name = os.path.basename(REPO_ROOT)
    assert "phase1" in worktree_name or "arx-saas" in worktree_name, (
        f"Repository {REPO_ROOT} does not appear to be an isolated worktree."
    )
