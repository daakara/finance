# ARX — RQ-02 Independent Review and Approval Registry Enforcement Proposal

**Directive:** ARX_RQ02_INDEPENDENT_REVIEW_ENFORCEMENT_PROPOSAL_001  
**Status:** PROPOSAL_ONLY / NOT_RATIFIED / NO_POLICY_MUTATION  
**Parent:** PR #141 (trusted-base PR qualification prototype)  
**Dependent work:** PR #139 (approved UX remediation)  
**Baseline:** `989070fcf2209a3375b5f5100cd79b53e3f5ed7a` (main)  
**Do not interpret this document as permission to merge, activate a gate, publish an approval manifest, grant permissions or deploy.**

## 1. Verified current-state evidence

GitHub REST ruleset `25187391`, `main-production-release-gate`, is active for `refs/heads/main`. It enforces PR-based updates, rejects deletion and non-fast-forward changes, and requires backend Python 3.11, Python 3.12, linter and SSG checks. The rule's PR parameters observed on 30 October 2026 were:

- `required_approving_review_count: 0`
- `require_code_owner_review: false`
- `require_last_push_approval: false`
- `dismiss_stale_reviews_on_push: true`
- `required_review_thread_resolution: true`

Its bypass-actor list was empty in the returned ruleset. This DOES NOT prove that all possible organization/repository/administrator protections are absent: the branch-protection API returned 403 to the connected integration. `.github/CODEOWNERS` returned 404 in mainline contents, and `scripts/release/pr_approvals/` did not exist.

`release/production` has a separate active ruleset `26024894`; neither it nor cumulative-release qualification may be weakened to enable this proposal.

## 2. Authority and operational roles: owner decision required

| Role | Minimum control | Decision state |
| --- | --- | --- |
| Product Owner or formally delegated authority | Approves *specific* inventory, scope and issuance transaction; cannot be inferred from generic "proceed" | NOT_APPOINTED_FOR_ISSUANCE |
| Independent governance reviewer | Reviews manifest, candidate digest, policy and negative controls; must not be the candidate author or last pusher | IDENTITY_AND_ACCESS_UNVERIFIED |
| Approval manifest publisher | Submits independently approved manifest via separate PR to protected main; cannot self-approve | UNDESIGNATED |
| Qualification verifier | Read-only GitHub Actions job using immutable event PR identity and a manifest in trusted base | PROTOTYPE_ONLY |
| Release authority | Keeps cumulative production RQ-02 and RQ-12 separate from any PR-differential pass | UNCHANGED |

An actual independently eligible human reviewer and suitable repository role MUST be verified before setting enforcement rules. Do not choose an identity automatically, add new collaborators or claim that an owner can approve their own push.

## 3. Proposed GitHub repository review enforcement (not applied)

**Scope:** `refs/heads/main`, including every PR that adds, edits or removes `scripts/release/pr_approvals/**`, `scripts/release/pr_diff_qualification.py`, related tests, `.github/workflows/**`, `.github/CODEOWNERS`, release receipts/manifest policy and release-governance records.

**Minimum ruleset proposal**, subject to repository-authority review:
- Required approving review count **at least one** from an eligible independent reviewer; stronger count may be warranted for authority-changing modifications.
- Enable mandatory code-owner review; keep stale-review dismissal on new commits.
- Require last-push approval by somebody other than the last pusher if supported and consistent with existing settings.
- Continue required check success on exact new head, branch up-to-date policy and review-thread resolution.
- Retain no bypass actors unless explicitly authorized and evidenced; prohibit direct pushes, force pushes and deletion.
- Validate that existing rule overlaps are compositional, and that rule changes do not inadvertently affect emergency rollback workflows.
- Recheck permissions against actual organization and repo policy. This proposal DOES NOT write ruleset settings.

**Proposed CODEOWNERS snippet for independent review only (NOT a committed CODEOWNERS file):**

```text
# Replace placeholder with a verified independent owner/team before enactment.
# /scripts/release/pr_approvals/  @VERIFIED-INDEPENDENT-GOVERNANCE-REVIEWER
# /scripts/release/pr_diff_qualification.py  @VERIFIED-INDEPENDENT-GOVERNANCE-REVIEWER
# /scripts/release/rq02_diff_integrity.py  @VERIFIED-INDEPENDENT-GOVERNANCE-REVIEWER
# /scripts/release/release_manifest.json  @VERIFIED-INDEPENDENT-GOVERNANCE-REVIEWER
# /.github/workflows/  @VERIFIED-INDEPENDENT-GOVERNANCE-REVIEWER
# /.github/CODEOWNERS  @VERIFIED-INDEPENDENT-GOVERNANCE-REVIEWER
```

The above is **illustrative comment-only text**. It is not a working GitHub policy and must not be copied into a live CODEOWNERS file unchanged. Actual CODEOWNERS match behavior, team visibility, write access and inherited owner entries must be validated.

## 4. Approval-manifest issuance protocol

1. Freeze source candidate, identify PR number, trusted base and exact head SHA; collect exact changed-file operations and SHA-256 of head-tree blobs.
2. Independently replay and review the changed inventory and the authority of each data source, especially any UX/Radar shared files. Reject altered framework, approval-registry, CI workflow or protected governance paths.
3. Obtain a specific recorded owner authorization covering inventory, PR number, purpose and reviewer. Generic preapproval authorizes engineering tests, not issuance.
4. Publisher proposes a manifest-only PR to protected main. The publisher and candidate author cannot self-approve. Independent code-owner approval and required status checks must pass.
5. After separately authorized manifest publication, rebase or reconcile candidate against new trusted base; do NOT force-push or rewrite without authorization. Recompute the diff/hash inventory and expire old exact-head receipts.
6. Evaluate PR differential only using the trusted base manifest and GitHub event; output immutable, separately named `PR_DIFFERENTIAL_ONLY_NOT_PRODUCTION_RELEASE` receipt.
7. Retain cumulative production-release RQ-02/RQ-12 independently, and require separate UX, Radar/Smart Money, release and owner acceptance gates.

**Revocation and supersession:** record previous approval ID/base SHA, reason, new approval ID, effective source SHA, independent reviewer, and linkage to obsolete receipts. Since an approval is embedded in Git history it cannot be physically revoked from history; the active policy must use a newer trusted-base state plus explicit expiry/revocation semantics before relying on older manifests. **Current prototype does not implement revocation or policy-version pinning.** This is an unresolved implementation dependency.

## 5. Adversarial enforcement acceptance matrix

| Negative case | Expected result |
| --- | --- |
| Candidate author issues and approves own manifest | Rejected by independent review enforcement |
| Manifest published without required qualified reviewer | Rejected by repository ruleset |
| CODEOWNERS placeholder, nonexistent team, non-write-access owner | Review policy adoption blocked |
| Review completed before the last push and not revalidated | Stale approval invalidated |
| Repository workflow or qualification validator tampered | Rejected by protected-file rules and negative-control tests |
| PR head from external fork / mismatched repository | CI rejects |
| Head SHA, base SHA, base ancestry or manifest mismatch | CI rejects |
| Candidate uses trusted-base approval from another PR | CI rejects |
| Approved contents change, file added, removed or renamed | CI rejects, new approval required |
| Earlier base contains superseded approval | Policy must reject/revalidate; prototype gap |
| PR-only verifier passes but cumulative release gate fails | Production release stays BLOCKED |
| Reviewer absent or cannot be independently enforced | Operational adoption BLOCKED |
| Emergency rollback procedure potentially harmed by ruleset change | Stop for explicit security/release owner review |

For all accepted positive cases, require full Git tests, Python 3.11/3.12 exact-head receipts, code-owner review and meaningful release-protection simulations. The GitHub permissions and reviewer restrictions must be independently verified, not inferred from a successful unit test.

## 6. Approved sequencing boundaries

- **Now permitted:** inspect policies, gather reviewer-access evidence read-only, run test-only dry runs, create governance proposals and draft PRs, critique design.
- **Requires fresh explicit governance authorization:** appoint qualified reviewer/issuer/publisher; add working CODEOWNERS; change rulesets or permissions; create/merge approval manifests; implement or enable release workflow routing; supersession semantics.
- **Requires distinct release authorization:** merge PR #141, PR #139, deploy, lift any HOLD, activate research ingestion/scanner scheduling, or begin Sprint 5 execution.

## 7. Owner decision package

**GOV-RQ02-D1:** Ratify named/eligible independent reviewer and issuer roles, with actual GitHub access evidence.  
**GOV-RQ02-D2:** Approve proposed review-enforcement changes and explicit affected branches/paths.  
**GOV-RQ02-D3:** Authorize tested revocation/supersession semantics and immutable approval custody.  
**GOV-RQ02-D4:** Authorize separate PR-only CI adoption after D1–D3 pass.  
**GOV-RQ02-D5:** Separately authorize issuance of the exact inventory for UX PR #139 and any base update/rebase.

**Current disposition:** `REVIEW_ENFORCEMENT_PROPOSAL_READY_FOR_OWNER_REVIEW`, not `TRUST_MODEL_OPERATIONALLY_RATIFIED`.

## 8. Source locations

- GitHub active rulesets: https://github.com/daakara/finance/rules/25187391 and https://github.com/daakara/finance/rules/26024894
- Governance prototype: https://github.com/daakara/finance/pull/141
- UX candidate: https://github.com/daakara/finance/pull/139

No setting is changed by this specification.
