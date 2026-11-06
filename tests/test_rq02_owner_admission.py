"""Fail-closed RQ-02 solo-owner admission adversarial controls."""
import datetime as dt
import pathlib
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "release"))
import rq02_owner_admission as admission

pytestmark = pytest.mark.tier2b
HEAD = "a" * 40
OWNER_ID = 53128824
NOW = dt.datetime(2026, 11, 6, 12, 0, tzinfo=dt.timezone.utc)


def comment(*, body=None, user_id=OWNER_ID, when="2026-11-06T11:00:00Z"):
    return {
        "id": 123,
        "html_url": "https://github.com/daakara/finance/pull/207#issuecomment-123",
        "body": body or f"ARX-RQ02-APPROVE v1 pr=207 sha={HEAD} base=main",
        "user": {"id": user_id, "type": "User"},
        "created_at": when,
        "updated_at": when,
    }


@pytest.mark.parametrize("override,reason", [
    ({"user_id": 111}, "NOT_REPOSITORY_OWNER"),
    ({"body": "ARX-RQ02-APPROVE v1 pr=207 sha=" + "b"*40 + " base=main"}, "APPROVAL_NOT_SHA_BOUND"),
    ({"body": "ARX-RQ02-APPROVE v1 pr=208 sha=" + HEAD + " base=main"}, "APPROVAL_NOT_SHA_BOUND"),
    ({"body": "ARX-RQ02-APPROVE v1 pr=207 sha=" + HEAD + " base=release/production"}, "APPROVAL_NOT_SHA_BOUND"),
    ({"when": "2026-10-20T11:00:00Z"}, "EXPIRED_OR_FUTURE_APPROVAL"),
    ({"when": "2026-11-07T11:00:00Z"}, "EXPIRED_OR_FUTURE_APPROVAL"),
])
def test_reject_unapproved_comment(override, reason):
    valid, observed = admission.evaluate_owner_comment(
        comment(**override), owner_id=OWNER_ID, pr_number=207, head_sha=HEAD, now=NOW,
    )
    assert valid is False
    assert observed == reason


def test_owner_exact_sha_approval_accepted():
    valid, evidence = admission.evaluate_owner_comment(
        comment(), owner_id=OWNER_ID, pr_number=207, head_sha=HEAD, now=NOW,
    )
    assert valid is True
    assert "issuecomment-123" in evidence


def test_missing_event_or_production_target_fails_closed():
    assert admission.verify_owner_admission(pr_number=207, head_sha=HEAD, target="release/production") == (False, "INVALID_MAIN_CANDIDATE_IDENTITY")
    with patch.dict("os.environ", {"GITHUB_ACTIONS": "false", "GITHUB_REPOSITORY": "daakara/finance"}):
        assert admission.verify_owner_admission(pr_number=207, head_sha=HEAD, target="main") == (False, "NON_HOSTED_OWNER_ADMISSION_DISALLOWED")


def test_exact_owner_approval_from_live_repository_metadata():
    repo = {"full_name": "daakara/finance", "owner": {"id": OWNER_ID}}
    pr = {
        "number": 207, "state": "open",
        "head": {"sha": HEAD, "repo": {"full_name": "daakara/finance"}},
        "base": {"ref": "main"},
    }
    fresh = dt.datetime.now(dt.timezone.utc).isoformat()
    with patch.dict("os.environ", {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "daakara/finance"}):
        with patch.object(admission, "_github_json", side_effect=[repo, pr, [comment(when=fresh)]]):
            passed, evidence = admission.verify_owner_admission(pr_number=207, head_sha=HEAD, target="main")
    assert passed is True
    assert "issuecomment-123" in evidence


def test_pr_head_substitution_is_rejected():
    repo = {"full_name": "daakara/finance", "owner": {"id": OWNER_ID}}
    pr = {"number": 207, "state": "open", "head": {"sha": "b"*40, "repo": {"full_name": "daakara/finance"}}, "base": {"ref": "main"}}
    with patch.dict("os.environ", {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "daakara/finance"}):
        with patch.object(admission, "_github_json", side_effect=[repo, pr]):
            assert admission.verify_owner_admission(pr_number=207, head_sha=HEAD, target="main") == (False, "PR_HEAD_CHANGED_SINCE_APPROVAL")


def test_api_error_fails_closed():
    with patch.dict("os.environ", {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "daakara/finance"}):
        with patch.object(admission, "_github_json", side_effect=OSError("API unavailable")):
            assert admission.verify_owner_admission(pr_number=207, head_sha=HEAD, target="main") == (False, "GITHUB_OWNER_ATTESTATION_UNAVAILABLE")
