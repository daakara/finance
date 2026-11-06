"""Adversarial tests for trusted-base RQ-02 file-scoping gate."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "release"))
from rq02_trusted_admission import validate_files, get_all_files

pytestmark = pytest.mark.tier2b


def f(path, status="modified"):
    return {"filename": path, "status": status}


@pytest.mark.parametrize("files,reason", [
    ([], "EMPTY_CANDIDATE_DIFF"),
    ([f("api/routes/screener.py"), f("scripts/release/rq02_diff_integrity.py")], "PROTECTED_PATH_MODIFICATION"),
    ([f(".github/workflows/ci.yml")], "PROTECTED_PATH_MODIFICATION"),
    ([f(".secrets/creds.json")], "PROTECTED_PATH_MODIFICATION"),
    ([f("railway.json")], "PROTECTED_PATH_MODIFICATION"),
    ([f("frontend/lib/api.ts", "removed")], "DESTRUCTIVE_OR_UNEXPECTED_OPERATION"),
    ([f("README.md")], "OUTSIDE_APPLICATION_BOUNDARY"),
    ([f("frontend/../api/routes/foo.py")], "MALFORMED_DIFF_PATH"),
    ([f("tests/a.py"), f("tests/a.py")], "DUPLICATE_OR_INVALID_DIFF_PATH"),
])
def test_out_of_scope_diffs_fail_closed(files, reason):
    accepted, detail = validate_files(files)
    assert not accepted and detail.startswith(reason)


def test_two_file_statistical_claims_pr_allowed_after_owner_approval():
    allowed, reason = validate_files([
        f("analyst_dashboard/analyzers/self_healing_engine.py"),
        f("tests/test_self_healing_audit_claims.py", "added"),
    ])
    assert allowed and reason == "SCOPED_APPLICATION_DIFF"


def test_pagination_limit_fails_closed(monkeypatch):
    import rq02_trusted_admission as gate
    monkeypatch.setattr(gate, "_github_json", lambda _: [f("frontend/one.ts")] * 100)
    assert get_all_files("daakara/finance", 207) is None



def test_trusted_commit_status_binds_success_to_exact_head(monkeypatch):
    import rq02_trusted_admission as gate
    from unittest.mock import call, patch
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with patch.object(gate, "_event_head", return_value=("daakara/finance", "a"*40)):
        with patch.object(gate, "_evaluate_main", return_value=0):
            with patch.object(gate, "post_commit_status") as status:
                assert gate.main() == 0
                assert status.call_count == 2
                assert status.call_args_list[0].args[:3] == ("daakara/finance", "a"*40, "pending")
                assert status.call_args_list[1].args[:3] == ("daakara/finance", "a"*40, "success")


def test_trusted_commit_status_fails_closed_on_rejection(monkeypatch):
    import rq02_trusted_admission as gate
    from unittest.mock import patch
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with patch.object(gate, "_event_head", return_value=("daakara/finance", "a"*40)):
        with patch.object(gate, "_evaluate_main", return_value=1):
            with patch.object(gate, "post_commit_status") as status:
                assert gate.main() == 1
                assert status.call_args_list[-1].args[2] == "failure"


def test_status_api_failure_never_turns_into_pass(monkeypatch):
    import rq02_trusted_admission as gate
    from unittest.mock import patch
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with patch.object(gate, "_event_head", return_value=("daakara/finance", "a"*40)):
        with patch.object(gate, "post_commit_status", side_effect=OSError("status API down")):
            assert gate.main() == 1


def test_status_write_requires_trusted_token(monkeypatch):
    import rq02_trusted_admission as gate
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(ValueError, match="token unavailable"):
        gate.post_commit_status("daakara/finance", "a"*40, "success", "test")
