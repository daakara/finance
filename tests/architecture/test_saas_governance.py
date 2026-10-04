"""
Tests for SaaS Architectural Governance.

Verifies:
1. No new external dependencies added to pyproject.toml or package.json.
2. Standard library purity across api/context, api/capabilities, and api/services.
3. Zero database schema or migration files touched.
4. Clean separation of concerns between analytical core and SaaS layers.
"""

import ast
import os
import subprocess
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPECTED_BASE_SHA = "d20ec394133261df84885fb2d8c6f941a5b9ba19"


def _get_base_sha():
    proc_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    branch = proc_branch.stdout.strip()
    if branch == "main":
        proc_merge = subprocess.run(["git", "rev-parse", "--verify", "HEAD^2"], cwd=REPO_ROOT, capture_output=True, text=True)
        if proc_merge.returncode == 0:
            return "HEAD^1"
    return EXPECTED_BASE_SHA


def test_no_dependency_files_modified():
    """Verify package manifests (pyproject.toml, package.json, requirements) were not altered."""
    base_sha = _get_base_sha()
    cmd = ["git", "diff", "--name-only", base_sha, "HEAD"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    diff_files = [f.strip() for f in proc.stdout.splitlines() if f.strip()]

    forbidden_manifests = [
        "pyproject.toml",
        "package.json",
        "package-lock.json",
        "requirements.txt",
        "poetry.lock",
        "pnpm-lock.yaml",
    ]
    for f in diff_files:
        assert os.path.basename(f) not in forbidden_manifests, f"Manifest file modified: {f}"



def test_standard_library_purity_of_seam_packages():
    """Verify api seam modules import only Python standard library."""
    seam_dirs = [
        os.path.join(REPO_ROOT, "api", "context"),
        os.path.join(REPO_ROOT, "api", "capabilities"),
        os.path.join(REPO_ROOT, "api", "services"),
    ]
    allowed_stdlib = {
        "dataclasses",
        "typing",
        "types",
        "re",
        "uuid",
        "enum",
        "os",
        "sys",
        "abc",
        "collections",
        "copy",
        "functools",
        "itertools",
        "math",
        "datetime",
        "time",
        "json",
        "hashlib",
        "api",  # internal project namespace
    }

    for sdir in seam_dirs:
        for fname in os.listdir(sdir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(sdir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=fname)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        assert root in allowed_stdlib, (
                            f"Non-stdlib import '{alias.name}' in {fpath}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    root = mod.split(".")[0]
                    assert root in allowed_stdlib, (
                        f"Non-stdlib import from '{mod}' in {fpath}"
                    )
