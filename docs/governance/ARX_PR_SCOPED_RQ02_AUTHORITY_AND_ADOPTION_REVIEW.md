# ARX — PR-scoped RQ-02 qualification: authority and adoption review

**Status:** DESIGN_REVIEW_ONLY / NOT_ACTIVATED / NOT_A_PRODUCTION_RELEASE_QUALIFICATION

**Candidate PR:** #141. **Dependent UX PR:** #139. Existing cumulative-release RQ-02, its historical manifest, and deployment controls remain unchanged.

## Verified source behaviors

- The proposed verifier inspects an exact base/head Git diff, permitted operations, SHA-256 of immutable head-tree blobs, ancestry, and a candidate-independent approval manifest read via `git show <base>:scripts/release/pr_approvals/<PR>.json`.
- In GitHub Actions it verifies event PR number, head and base SHA, and the originating repository identity (same-repository PRs only). External forks fail closed.
- The candidate cannot add or modify its own trusted-base approval and cannot modify `.github/` or `scripts/release/` under this verifier.
- These controls do **not** constitute owner authorization or production-release qualification.

## Material adoption problem: approval-first branch lifecycle

For an already opened PR #139, the base commit `989070fcf2209a3375b5f5100cd79b53e3f5ed7a` does not contain `scripts/release/pr_approvals/139.json`. Merely merging #141 would not fix this. A trusted approval must exist in a **new** base commit; the candidate then needs a controlled rebase or branch update, and a new exact-head evaluation. That operation is not authorized by this draft.

An approval manifest in the base cannot contain a future descendant head SHA without circular Git hashing. It can approve exact file operations and SHA-256 blob contents. GitHub's PR event must independently bind the observed head/base identities. Subsequent changes require renewed content approval if the approved file hashes or inventory differ.

## Minimum governance authority contract before activation

1. **Approval issuer:** designate the owner or delegated reviewer permitted to approve PR-scoped inventories; separate the issuer from candidate authors.
2. **Evidence source:** capture exact proposed file paths, operation types, resulting Git blob contents, review scope, security review and decision receipt. Reject empty/ambiguous approvals.
3. **Trusted publication:** land an approved immutable record in protected base through a separately reviewed change. Candidate branches may not write approvals or the policy verifier.
4. **Base advancement:** define how an approved manifest survives base movement and how rebasing/pulling a new base invalidates old receipts; retain an exact commit bound to every verification.
5. **CI checkout:** use `pull_request` event, pinned exact source commit, full history, read-only token and no untrusted head workflow execution. Compare source SHA with event head, not the default synthetic merge SHA.
6. **Protected paths:** verify branch protection, CODEOWNERS/reviewer constraints and cross-repository restrictions. Separate PR-scoped proof from cumulative production release proof.
7. **Receipt semantics:** emit immutable base, head, approval path, file hashes, policy version and pass/fail. The PR receipt must explicitly say `PR_DIFFERENTIAL_ONLY_NOT_PRODUCTION_RELEASE`.
8. **Negative controls:** missing approval; unauthorized addition/removal; tampered content hash; stale/wrong base or head; unrelated ancestry; changed approval/policy/workflow; forked head; CI event spoof or absent repo; duplicate JSON keys.
9. **Independent acceptance:** run the test suite against actual Git repositories and hosted Python 3.11/3.12. An acceptance reviewer must inspect the test results and authorize any new workflow integration.
10. **Stop conditions:** historical RQ-02 failure remains binding, no automatic merge of #139/#141, no production deployment, no Sprint 5 activation or modification to existing HOLDs.

## Integration decision paths

**Option A — trusted-base approval registry (current prototype):** preserves strong separation but requires a prior approved commit to main and controlled rebasing. It cannot retroactively qualify #139 at its current base without a separately authorized lifecycle.

**Option B — independently signed external approval artifact:** could avoid modifying base, but introduces custody, signature-verification, reviewer identity and replay-risk requirements. No such alternate authority is implemented or authorized.

**Recommendation:** retain Option A for further governance review only, complete hosted negative controls and independently audit whether the approval-issuance/branch-protection process is operationally feasible. Do not wire it to the existing release workflow until the owner ratifies the exact authority and migration sequence.

## Distinct final statuses

- `UNIT_AND_GIT_TESTS_QUALIFIED`: only the prototype test contract passes.
- `TRUST_MODEL_RATIFIED`: named issuer, protected approval publication and replay rules authorized.
- `PR_GATE_INTEGRATED`: separate workflow implementation passes hosted checks and negative controls.
- `UX_PR139_RELEASE_READY`: still requires all independent UX, Radar and governance acceptance gates.
