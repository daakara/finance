"""Single-use, fail-closed RQ-02 bootstrap for owner-authorized PR #206.

Do not generalize this exception to application PRs. Verification compares the
entire candidate qualification source against the original mainline source
after reversing ONLY the registered PR number and the exact bootstrap hook.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Tuple

BASE_ANCHOR = "41897bd2714301bd4ea67d9c6932493dd256e38c"
SOURCE = "scripts/release/rq02_diff_integrity.py"
SELF = "scripts/release/rq02_registration_206.py"
TEST = "scripts/release/test_rq02_registration_206.py"
EXPECTED_DIFF = {
    SOURCE: "M",
    SELF: "A",
    TEST: "A",
}
START = "        # Bounded administrative registration bootstrap: PR #206 only.\n"
END = "        # Candidate PR Authorization check\n"
OLD_AUTH = "is_approved_pr = pr_val in APPROVED_PRS"
NEW_AUTH = "is_approved_pr = (pr_val in APPROVED_PRS) or registration_bootstrap"
ALLOW_PATTERN = re.compile(r"^APPROVED_PRS = \{[^\n]*\}$", re.MULTILINE)


def _run_git(root: str, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args], cwd=root, stderr=subprocess.PIPE, timeout=15
    )


def verify_candidate_contents(base: str, candidate: str) -> Tuple[bool, str]:
    """Admit only a single addition, 205, plus the exact bootstrap hook."""
    base_lines = ALLOW_PATTERN.findall(base)
    head_lines = ALLOW_PATTERN.findall(candidate)
    if len(base_lines) != 1 or len(head_lines) != 1:
        return False, "MISSING_OR_AMBIGUOUS_ALLOWLIST"
    old = base_lines[0]
    new = head_lines[0]
    if not old.endswith("201}") or new != old[:-1] + ", 205}":
        return False, "UNAUTHORIZED_APPROVED_PR_SET_CHANGE"
    if candidate.count(START) != 1 or candidate.count(END) != 1:
        return False, "BOOTSTRAP_BLOCK_MISSING_OR_DUPLICATE"
    begin = candidate.index(START)
    end = candidate.index(END, begin)
    block = candidate[begin:end]
    # The change is limited to the explicit registration call and audit
    # assertion. Tests challenge this for tampering and scope widening.
    if (
        'str(pr_number) == "206" and clean_target == "main"' not in block
        or 'from rq02_registration_206 import verify_registration_only' not in block
        or 'if not registration_bootstrap:' not in block
        or 'REGISTRATION_BOOTSTRAP_REJECTED' not in block
    ):
        return False, "BOOTSTRAP_HOOK_SCOPE_INVALID"
    normalized = candidate[:begin] + candidate[end:]
    if normalized.count(NEW_AUTH) != 1:
        return False, "BOOTSTRAP_AUTH_SCOPE_INVALID"
    normalized = normalized.replace(NEW_AUTH, OLD_AUTH, 1)
    normalized = normalized.replace(new, old, 1)
    if normalized != base:
        return False, "UNRELATED_RELEASE_GATE_EDIT"
    return True, "EXACT_PR205_REGISTRATION_ONLY"


def verify_registration_only(
    *, base_sha: str, head_sha: str, repo_root: str
) -> Tuple[bool, str]:
    """Verify immutable Git objects, exact changed files and source diff."""
    if base_sha != BASE_ANCHOR:
        return False, "UNEXPECTED_REGISTRATION_BASE_ANCHOR"
    if not re.fullmatch(r"[a-f0-9]{40}", head_sha):
        return False, "INVALID_CANDIDATE_HEAD"
    try:
        diff = _run_git(repo_root, "diff", "--name-status", base_sha, head_sha).decode("utf-8")
        changes = {}
        for line in diff.splitlines():
            fields = line.split("\t")
            if len(fields) != 2 or fields[0] not in ("A", "M"):
                return False, "FORBIDDEN_DIFF_OPERATION"
            if fields[1] in changes:
                return False, "DUPLICATED_DIFF_PATH"
            changes[fields[1]] = fields[0]
        if changes != EXPECTED_DIFF:
            return False, "UNEXPECTED_REGISTRATION_SCOPE"
        base = _run_git(repo_root, "show", f"{base_sha}:{SOURCE}").decode("utf-8")
        head = _run_git(repo_root, "show", f"{head_sha}:{SOURCE}").decode("utf-8")
        return verify_candidate_contents(base, head)
    except (subprocess.CalledProcessError, UnicodeDecodeError, OSError, subprocess.TimeoutExpired):
        return False, "GIT_ATTESTATION_FAILED"
