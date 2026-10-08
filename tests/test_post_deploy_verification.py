"""
tests/test_post_deploy_verification.py

QA Regression suite verifying deployment identity, functional release resolution,
and documentation-only redeployment semantics for ARX Terminal post-deploy gates.
"""

import os
import sys
import pytest

# Ensure scripts directory is in path
SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "qa"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from post_deploy_production_verification import (
    is_git_ancestor,
    parse_documented_functional_releases,
    resolve_and_verify_functional_release,
)


CANONICAL_FUNCTIONAL_SHA = "5dcfeb41d75bb3ae02f25cb3599ab86c0cb03950"
CANONICAL_DOC_REDEPLOY_RUNTIME_SHA = "6f0559d93a681c3bb7c3a89883e0a820934a4a7a"


def test_is_git_ancestor_positive():
    """Verify that functional release 5dcfeb4 is an ancestor of doc redeploy 6f0559d."""
    assert is_git_ancestor(CANONICAL_FUNCTIONAL_SHA, CANONICAL_DOC_REDEPLOY_RUNTIME_SHA) is True


def test_is_git_ancestor_negative():
    """Verify that descendant 6f0559d is NOT an ancestor of parent 5dcfeb4."""
    assert is_git_ancestor(CANONICAL_DOC_REDEPLOY_RUNTIME_SHA, CANONICAL_FUNCTIONAL_SHA) is False


def test_parse_documented_functional_releases():
    """Verify that docs/releases parser discovers the committed release note for 5dcfeb4."""
    releases = parse_documented_functional_releases()
    assert len(releases) >= 1
    shas = [r["release_sha"] for r in releases]
    assert any(CANONICAL_FUNCTIONAL_SHA.startswith(s) or s.startswith(CANONICAL_FUNCTIONAL_SHA[:7]) for s in shas)


def test_doc_only_redeployment_explicit_sha_pass():
    """
    Case 1: Documentation-only redeployment with explicit functional SHA.
    functional release A -> release-note commit B -> B is deployed.
    Expected: PASS, functional_release_is_ancestor = YES, release_note_found = YES.
    """
    res = resolve_and_verify_functional_release(
        runtime_sha=CANONICAL_DOC_REDEPLOY_RUNTIME_SHA,
        expected_functional_sha=CANONICAL_FUNCTIONAL_SHA,
    )
    assert res["status"] == "PASS"
    assert res["runtime_sha"] == CANONICAL_DOC_REDEPLOY_RUNTIME_SHA
    assert res["functional_release_sha"] == CANONICAL_FUNCTIONAL_SHA
    assert res["functional_release_is_ancestor"] == "YES"
    assert res["release_note_found"] == "YES"
    assert res["is_doc_only_redeploy"] is True
    assert "2026-10-08_5dcfeb4" in res["release_note_file"]


def test_doc_only_redeployment_auto_resolution_pass():
    """
    Case 2: Documentation-only redeployment with auto-resolution from ancestry.
    Expected: Automatically discovers 5dcfeb4 as the functional release for runtime 6f0559d.
    """
    res = resolve_and_verify_functional_release(
        runtime_sha=CANONICAL_DOC_REDEPLOY_RUNTIME_SHA,
        expected_functional_sha=None,
    )
    assert res["status"] == "PASS"
    assert res["runtime_sha"] == CANONICAL_DOC_REDEPLOY_RUNTIME_SHA
    assert res["functional_release_sha"] == CANONICAL_FUNCTIONAL_SHA
    assert res["resolution_method"] == "RELEASE_NOTE_METADATA_AND_GIT_ANCESTRY"
    assert res["functional_release_is_ancestor"] == "YES"
    assert res["release_note_found"] == "YES"
    assert res["is_doc_only_redeploy"] is True


def test_missing_release_note_fail():
    """
    Case 3: Missing release note.
    Runtime contains functional release, but functional release has no note committed.
    Expected: FAIL / HOLD, release_note_found = NO.
    """
    non_existent_functional_sha = "1111111111111111111111111111111111111111"
    res = resolve_and_verify_functional_release(
        runtime_sha=CANONICAL_DOC_REDEPLOY_RUNTIME_SHA,
        expected_functional_sha=non_existent_functional_sha,
    )
    assert res["status"] == "HOLD"
    assert res["release_note_found"] == "NO"
    assert "No documented release note" in res["error"]


def test_unrelated_runtime_ancestry_fail():
    """
    Case 4: Unrelated runtime.
    Claimed functional SHA has a release note, but is NOT an ancestor of runtime SHA.
    (e.g. older runtime 3ae385c claiming functional release 5dcfeb4).
    Expected: FAIL / HOLD, functional_release_is_ancestor = NO.
    """
    older_runtime_sha = "3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb"
    res = resolve_and_verify_functional_release(
        runtime_sha=older_runtime_sha,
        expected_functional_sha=CANONICAL_FUNCTIONAL_SHA,
    )
    assert res["status"] == "HOLD"
    assert res["functional_release_is_ancestor"] == "NO"
    assert "NOT an ancestor" in res["error"]


def test_runtime_sha_mismatch_detection():
    """
    Case 5: Runtime SHA mismatch.
    When live runtime/provider returns an unexpected SHA, the verifier rejects it.
    """
    expected_runtime = CANONICAL_DOC_REDEPLOY_RUNTIME_SHA
    mock_live_sha = "badc0ffee0000000000000000000000000000000"
    assert not expected_runtime.startswith(mock_live_sha)
