# ARX — Trusted-Base PR Qualification: Approval Issuance and CI Adoption Specification

**Decision status:** CONDITIONAL_DESIGN_APPROVAL / NOT_OPERATIONALLY_RATIFIED
**Reference:** PR #141; dependent UX PR #139
**Implementation baseline:** governance candidate `88d3320fcadffa06786826f30f7647b64e2cd4db`
**Current historical release gate:** RQ-02 and RQ-12 remain unchanged and binding.

## 1. Separation of authorities (proposed; not granted)

| Role | Responsibility | Explicit prohibition |
| --- | --- | --- |
| Candidate author | Submit PR code, scope, test results, and requested file inventory | Cannot authorize their own manifest or modify protected qualification policy |
| Independent qualification reviewer | Validate base/head source identity, diff, SHA-256, risk and negative controls | Cannot substitute candidate assertions for observed evidence |
| Product Owner / delegated governance authority | Authorize scope and designate manifest issuer; approve exceptional risk | Conditional design approval does not itself appoint an issuer |
| Manifest publisher | Publish reviewed manifest to protected base by independent, reviewable change | Cannot alter candidate head or release qualification silently |
| CI verifier | Evaluate trusted-base approval against immutable source commit and authenticated PR event | Cannot generate or approve manifests dynamically |
| Release authority | Independently certify cumulative production-release gates | Cannot infer production approval from PR differential PASS |

No person, bot or GitHub App currently receives new permissions under this document.
Issuer, reviewer, CODEOWNERS and branch-protection evidence must be recorded before operational adoption.

## 2. Candidate inventory and proposed approval schema

Only a separately approved record under `scripts/release/pr_approvals/<PR>.json` in a trusted base is valid:

```json
{
  "schema_version": 1,
  "pr_number": 139,
  "files": {
    "frontend/app/compare/page.tsx": {
      "operation": "M",
      "sha256": "<exact 64-character SHA-256 of head-tree file content>"
    }
  }
}
```

This is a schema illustration, **not an approval of PR #139**. Never commit this illustrative placeholder. Each changed file requires an exact operation and content digest. Missing, extra, duplicate, renamed, or ambiguous entries must fail closed.

An approval record is valid only for its base ancestry and exact file inventory/content. Its inclusion in a base commit is not proof of independent human review; protected publication and review receipts are mandatory. Approval must be revoked or superseded when its issuer, applicable scope, baseline, policy version or candidate content changes.

## 3. Required operational lifecycle — not yet authorized

1. The candidate author proposes a frozen diff, full source head SHA, target base SHA, and recorded tests without modifying protected approval paths.
2. Independent reviewer reproduces the approved file inventory and immutable head-tree SHA-256 digests; validates the authorized purpose, UX/Radar file ownership and no held capability activation.
3. Product Owner or separately delegated governance authority explicitly authorizes a **specific** inventory and publication transaction with its owner/review receipts.
4. Publisher opens a separate approval-manifest PR targeting protected main; independent code owner reviews and merges only under separately granted authorization.
5. After approval manifest lands on main, the candidate branch must be reconciled with that newer main through an authorized branch-update operation. Recompute head/base and observed changes. Reapprove if scope or final content differs.
6. The CI PR-only qualifier runs on immutable head, trusted base and authenticated event identity. It reads manifest **from base Git tree**, never from the candidate, an untrusted artifact or mutable working tree.
7. The resulting immutable receipt records observed base/head SHAs, PR ID, repository identity, approved-manifest path, content hashes, policy version, CI run/job ID and verdict. Output is labeled `PR_DIFFERENTIAL_ONLY_NOT_PRODUCTION_RELEASE`.
8. Production release qualification still separately runs full historical cumulative RQ-02/RQ-12 and required release gates; no PR-only PASS may replace it.

## 4. Required CI integration design (proposal only)

- A distinct named PR-only job must not overwrite the existing `ARX Release Qualification (RQ-01..12)` check name or reuse production-success receipts.
- Restrict to trusted same-repository `pull_request` events. Require read-only contents permissions, no write token, no secrets, no `pull_request_target`, and no candidate-provided executable workflow code in a privileged context.
- Fetch full Git history and explicitly checkout `github.event.pull_request.head.sha`, avoiding confusion with GitHub's synthetic merge SHA.
- Validate GitHub event PR number, source/base full SHA, source/base repository identity and actual checked-out `HEAD`; reject stale or moving refs.
- Read an approval record from the event's exact base SHA and enforce base ancestry, strict allowed inventory/operation and SHA-256 digests.
- Protect workflow, validator, registry, test harness, governance and release-configuration paths from candidate changes. Review whether other scripts can alter the trust boundary.
- On approvals changing or base advancement: invalidate prior evidence, rerun both positive and negative controls, and recalculate receipts.
- Keep production RQ-12 aggregation logically independent; do not render a blocked release acceptable because PR-specific checks pass.

## 5. Adversarial acceptance matrix

| Scenario | Required verdict |
| --- | --- |
| Correct trusted-base approval, same repo and exact immutable candidate | PASS for PR differential only |
| No manifest in actual base commit | FAIL |
| Candidate adds approval to its own head | FAIL |
| Candidate modifies manifest / workflow / release validator | FAIL |
| Wrong PR number, wrong base, stale or wrong head | FAIL |
| Forked or mismatched head/base repository | FAIL |
| Unrelated head not descended from base | FAIL |
| File unapproved, missing, renamed or operation changed | FAIL |
| SHA-256 content differs from approved content | FAIL |
| Duplicate JSON approval key or malformed schema | FAIL |
| Modified approval issuer/registry controls without independent approval | FAIL |
| No authenticated CI event in an activated CI context | FAIL |
| Branch push after prior PASS / base advances | Reevaluate exact event identity; prior PASS cannot transfer |
| Approval grants PR-level qualification but production cumulative RQ-02 fails | Production release remains BLOCKED |

Positive control must use a real disposable Git repository. No tests may rely solely on literal strings or mocked subprocesses.

## 6. Adoption dependency and decision boundaries

**Important:** PR #139 currently targets base `989070fcf2209a3375b5f5100cd79b53e3f5ed7a`, which has no independently issued trusted-base approval for PR #139. Merging the prototype alone cannot fix this, and this document does not authorize approval-publication or rebase.

**Gate A — Design qualification:** Verify source, negative controls, exact SHA hosted Python 3.11/3.12 and SSG. Independent reviewers assess trust model. Conditional owner design approval already exists.

**Gate B — Operational governance ratification:** Explicitly designate independent reviewer, issuer, publisher and protection policies; approve the publication lifecycle, revocation rules and source of immutable decision receipts. NOT YET GRANTED.

**Gate C — Workflow adoption:** Independently authorize implementation of PR-only CI and integration tests; preserve historical cumulative-release checks. NOT YET GRANTED.

**Gate D — PR #139 approval issuance:** Approve exact PR #139 inventory and trusted publication/rebase sequence after independently reviewing its final candidate and Radar overlap. NOT YET GRANTED.

**Gate E — Merge / deployment / acceptance:** Require separate owner review, all hosted CI and Radar/UX acceptance, rollback verification and production parity evidence. NOT YET GRANTED.

## 7. Current verdict

`TRUSTED_BASE_APPROVAL_LIFECYCLE_SPECIFIED_NOT_RATIFIED`

No manifest was issued, no approval registry was written, no protected workflow was changed, no PR merged, and no production/governance state was activated.
