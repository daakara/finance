# ARX TERMINAL — SPRINT 2 PACKAGE B
# CANONICAL CODE-RELEASE PULL REQUEST SUBMISSION RECORD

## 1. Authority, Governance & Gating Status

- **Governing Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Accepted Input Disposition**: `PACKAGE_B_CODE_RELEASE_READY_FOR_EXPLICIT_APPROVAL`
- **Current Operational Disposition**: `PACKAGE_B_CODE_PR_READY_FOR_MERGE_REVIEW`
- **Sprint 2 Operating Gate**: `G-S2-ACT` (Governs code release, PR submission, and historical pilot ingestion).
- **VCP Holdout Authority**: `G-VCP-012 = ACTIVATION_DEFERRED` (Independently governed; production historical ingestion does NOT require Gate 12).
- **Operator Authorization Token**: `G-S2-ACT-CODE-PR-001: AUTHORIZE_BRANCH_PUSH_AND_PR_CREATION_ONLY`
- **Token Status**: **GRANTED BY OPERATOR & EXECUTED**
- **Authorization Boundary Matrix**:

| Operation | Status | Authorization Requirement |
| :--- | :--- | :--- |
| **Local Commit Preparation** | **AUTHORIZED & COMPLETED** | Standard engineering readiness |
| **Local Pre-PR Rehearsal & Regressions (99 tests)** | **AUTHORIZED & COMPLETED** | All 99 passed (100% green) |
| **Cross-Boundary Regressions (159 tests)** | **AUTHORIZED & COMPLETED** | All 159 passed (100% green) |
| **Remote Branch Push (`origin/fix/...`)** | **AUTHORIZED & COMPLETED** | Executed under `G-S2-ACT-CODE-PR-001` |
| **Pull Request Creation (`PR #35`)** | **AUTHORIZED & COMPLETED** | Opened targeting `main`: [PR #35](https://github.com/daakara/finance/pull/35) |
| **GitHub Actions CI Pipeline** | **PASSED (100% GREEN)** | All 5 checks completed with success |
| **Pull Request Merge into `main`** | **STRICTLY PROHIBITED** | Requires separate merge review gate under `G-S2-ACT` |
| **Production Code Deployment (Railway)** | **STRICTLY PROHIBITED** | Automated on merge to `main` (requires separate authorization) |
| **Production Market-Data Ingestion** | **STRICTLY PROHIBITED** | Requires separate `G-S2-ACT` execution gate; blocked by `AAPL_OUTSIDE_SCOPE_UNPROVENANCED_TAIL` |
| **Production Database Restoration** | **STRICTLY PROHIBITED** | Guarded by recovery tokens & zero-connection fence |

---

## 2. Remote GitHub Pull Request Verification

- **Repository**: `daakara/finance`
- **Pull Request Number**: `35`
- **Pull Request URL**: https://github.com/daakara/finance/pull/35
- **PR Title**: `fix(package-b): canonical SIP ingestion adapter and guarded recovery controls`
- **Base Branch**: `main`
- **Head Branch**: `fix/sprint2-package-b-canonical-ingestion-release`
- **Remote Branch HEAD SHA**: `695cd986f10b46d315ada8afa13058e81f4b4652` (Verified via `git ls-remote`)
- **Git Tree SHA**: `91e02c4c307a7718f3838f3e545d0728261d494f`
- **Base Commit SHA**: `f6c5755126059b2f227e28ce90ddaad43997884c`
- **Auto-Merge**: `Disabled` (None)
- **Reviewer Status**: Awaiting independent repository review

---

## 3. GitHub Actions CI Status & Check Runs

- **Workflow Run**: `ARX Terminal CI/CD Pipeline` ([Run #39566286992](https://github.com/daakara/finance/actions/runs/39566286992))
- **Event**: `pull_request` (target: `main`)
- **Overall CI Conclusion**: **`SUCCESS` (100% Green)**

### Check Run Breakdown

| Check Run Name | Status | Conclusion | Notes |
| :--- | :--- | :--- | :--- |
| **Backend Test Suite (Python 3.11)** | `completed` | `success` | Automated pytest suite |
| **Backend Test Suite (Python 3.12)** | `completed` | `success` | Multi-runtime compatibility |
| **Code Quality & Security Linter** | `completed` | `success` | Flake8, Black, Bandit, invariant linter |
| **Static SSG Export & Integrity Gate** | `completed` | `success` | Next.js SSG build & contract verification |
| **Cloudflare Pages** | `completed` | `success` | Preview build verification |
| **Production Deployment & Edge Cache** | `completed` | `skipped` | **Zero deployment side effect (correctly skipped on PR)** |

---

## 4. Changed-File Inventory (21 Files Verified Remotely)

Verified via GitHub API (`/repos/daakara/finance/pulls/35/files`):

```
added    analyst_dashboard/data/guarded_recovery.py
added    docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_CANONICAL_RELEASE_CANDIDATE.md
added    docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_CROSS_WORKTREE_CONVERGENCE.md
added    docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_FINAL_PILOT_MANIFEST_RECONCILIATION.md
added    docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_NEXT_AUTHORIZATION_DECISION.md
added    docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_RELEASE_PARITY_TEST_EVIDENCE.md
added    docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST.json
added    docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json
added    docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_CLEAN_RELEASE_DIFF.md
added    docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_CODE_ONLY_STARTUP_SAFETY.md
added    docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_CODE_PR_SUBMISSION_RECORD.md
added    docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_CODE_RELEASE_AUTHORIZATION_DECISION.md
added    docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_MANIFEST_V3_COMPATIBILITY.md
added    docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_PRODUCTION_RELEASE_PLAN.md
added    docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_RELEASE_REGRESSION_RESULTS.md
added    docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_RELEASE_SOURCE_PROVENANCE.md
added    scripts/guarded_market_recovery.py
modified scripts/ingest_alpaca_historical.py
modified tests/test_alpaca_historical_sip_ingestion.py
added    tests/test_package_b_deployment_rehearsal.py
added    tests/test_pilot_identity_and_guarded_recovery.py
```

*Verification*: Zero binary store files, zero credentials, zero Smart Money Candidate 006 files, zero Package C files.

---

## 5. Cryptographic Artifact Digests

| Component | Path | SHA-256 Digest | Size (Bytes) | Lines | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ingestion Adapter** | `scripts/ingest_alpaca_historical.py` | `7DA7CFC6BE8717C86EC15568F6D16E7173A44A7F33084653E889800F80C273C7` | 74,361 | 1,579 | Included in PR #35 |
| **Guarded Recovery Engine** | `analyst_dashboard/data/guarded_recovery.py` | `217377EC4BFF02C86B7B7122D39FC77A4112820B9F4BCA2E8C077888EA0DA684` | 21,241 | 466 | Updated in local safety correction |
| **Recovery CLI Tool** | `scripts/guarded_market_recovery.py` | `7B0F79B6F57BB94E5D5CA77B99D5B7237CD54B1E5FDF5A317429701F81917D82` | 2,829 | 73 | Updated in local safety correction |
| **Pilot Manifest Canonical V3** | `docs/.../PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` | `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` | 9,972 | 226 | `PROPOSED_NOT_AUTHORIZED` (Unsigned) |
| **Pilot Proposed Manifest V2** | `docs/.../PILOT_PROPOSED_MANIFEST.json` | `8923DAD988A10AE07A2B7BDA1044127130BF9C90539424D98F25676A9A59F07B` | 13,833 | 320 | `SUPERSEDED / NOT FOR EXECUTION` |

---

## 6. Railway Production State Attestation

- **Platform**: Railway (`tranquil-radiance`, env `production`, service `web`)
- **Active Deployment ID**: `4aa235d8-b571-4f67-bb41-2cdfa103fa00` (unchanged)
- **Deployment Trigger Check**: Branch push and PR creation triggered **zero** Railway deployments.
- **Production Database Writes**: **0 (ZERO)**.

---

## 7. AAPL Tail Ingestion Hold Record

```
AAPL_OUTSIDE_SCOPE_UNPROVENANCED_TAIL = BLOCKING_FOR_INGESTION
```

- Production contains trade date `2026-10-20` for AAPL (unprovenanced).
- Canonical V3 manifest specifies window `2025-08-27` to `2026-10-19`.
- Current adapter fail-closed check (`outside_unprovenanced > 0`) detects this session and rejects ingestion with `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE` with zero database writes.
- Manifest V3 is NOT automatically expanded; the production row is NOT deleted or quarantined; V3 digest (`0F54C702...`) is preserved.
- Blocks production historical ingestion, NOT code release.

---

## 8. Exact Next Approval Gate

1. **Gate**: Merge review under `G-S2-ACT`.
2. **Pre-condition**: Repository maintainer approval of PR #35.
3. **Action upon authorization**: Squash-and-merge or merge commit into `main`.
4. **Subsequent automated trigger**: Railway will build and deploy container on `main`.
5. **Post-deploy verification**: Run container script parity check (`7DA7CFC6...` and `8BE7D6A6...`).
6. **Separate Pilot Gate**: Ingestion remains `NOT_AUTHORIZED` pending AAPL tail scope resolution.
