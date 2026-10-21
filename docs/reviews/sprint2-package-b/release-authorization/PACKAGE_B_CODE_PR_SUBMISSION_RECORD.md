# ARX TERMINAL — SPRINT 2 PACKAGE B
# CANONICAL CODE-RELEASE PULL REQUEST SUBMISSION RECORD

## 1. Authority, Governance & Gating Status

- **Governing Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Accepted Input Disposition**: `PACKAGE_B_CODE_RELEASE_READY_FOR_EXPLICIT_APPROVAL`
- **Sprint 2 Operating Gate**: `G-S2-ACT` (Governs code release, PR submission, and historical pilot ingestion).
- **VCP Holdout Authority**: `G-VCP-012 = ACTIVATION_DEFERRED` (Independently governed; production historical ingestion does NOT require Gate 12).
- **Required Operator Gating Token**: `G-S2-ACT-CODE-PR-001: AUTHORIZE_BRANCH_PUSH_AND_PR_CREATION_ONLY`
- **Token Status**: **HELD / PENDING OPERATOR AUTHORIZATION**
- **Authorization Boundary Matrix**:

| Operation | Status | Authorization Requirement |
| :--- | :--- | :--- |
| **Local Commit Preparation** | **AUTHORIZED & COMPLETED** | Standard engineering readiness |
| **Local Pre-PR Rehearsal & Regressions (94 tests)** | **AUTHORIZED & COMPLETED** | All 94 passed (100% green) |
| **Cross-Boundary Regressions (159 tests)** | **AUTHORIZED & COMPLETED** | All 159 passed (100% green) |
| **Remote Branch Push (`origin/fix/...`)** | **NOT_AUTHORIZED** | Blocked on `G-S2-ACT-CODE-PR-001` |
| **Pull Request Creation (`gh pr create` / Web)** | **NOT_AUTHORIZED** | Blocked on `G-S2-ACT-CODE-PR-001` |
| **Pull Request Merge into `main`** | **STRICTLY PROHIBITED** | Requires separate merge review gate under `G-S2-ACT` |
| **Production Code Deployment (Railway)** | **STRICTLY PROHIBITED** | Automated on merge to `main` (requires separate authorization) |
| **Production Market-Data Ingestion** | **STRICTLY PROHIBITED** | Requires separate `G-S2-ACT` execution gate; blocked by `AAPL_OUTSIDE_SCOPE_UNPROVENANCED_TAIL` |
| **Production Database Restoration** | **STRICTLY PROHIBITED** | Guarded by recovery tokens & zero-connection fence |

---

## 2. Release Source Authority & Worktree State

- **Isolated Release Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint2-package-b-canonical-release`
- **Release Branch**: `fix/sprint2-package-b-canonical-ingestion-release`
- **Base Commit**: `origin/main` (`f6c5755126059b2f227e28ce90ddaad43997884c`)
- **Upstream Verification**: Verified `origin/main` has not advanced (`git fetch origin main`). Matches active Railway production deployment `4aa235d8-b571-4f67-bb41-2cdfa103fa00`.
- **Worktree Cleanliness**: Working tree is 100% clean (`nothing to commit, working tree clean`). Zero untracked files.
- **Cross-Worktree Segregation**: Base worktree `c:\Users\akara\Documents\Projects\finance` and all 218 Smart Money files remain completely isolated with zero bleed.

---

## 3. Commit Metadata & Scope Inventory

- **Commit Subject**: `fix(package-b): canonical historical SIP ingestion adapter and guarded recovery controls (#34)`
- **Author**: `Antigravity AI <akara@finance.app>`
- **Total Files in Release Scope**: 21 files (Code, tests, configurations, documentation only; zero binary/store files).

### File Inventory & Breakdown

```
analyst_dashboard/data/guarded_recovery.py
docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_CANONICAL_RELEASE_CANDIDATE.md
docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_CROSS_WORKTREE_CONVERGENCE.md
docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_FINAL_PILOT_MANIFEST_RECONCILIATION.md
docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_NEXT_AUTHORIZATION_DECISION.md
docs/reviews/sprint2-package-b/pilot-authorization/PACKAGE_B_RELEASE_PARITY_TEST_EVIDENCE.md
docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST.json
docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json
docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_CLEAN_RELEASE_DIFF.md
docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_CODE_ONLY_STARTUP_SAFETY.md
docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_CODE_PR_SUBMISSION_RECORD.md
docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_CODE_RELEASE_AUTHORIZATION_DECISION.md
docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_MANIFEST_V3_COMPATIBILITY.md
docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_PRODUCTION_RELEASE_PLAN.md
docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_RELEASE_REGRESSION_RESULTS.md
docs/reviews/sprint2-package-b/release-authorization/PACKAGE_B_RELEASE_SOURCE_PROVENANCE.md
scripts/guarded_market_recovery.py
scripts/ingest_alpaca_historical.py
tests/test_alpaca_historical_sip_ingestion.py
tests/test_package_b_deployment_rehearsal.py
tests/test_pilot_identity_and_guarded_recovery.py
```

---

## 4. Cryptographic Artifact Digests & Manifest Ambiguity Elimination

| Component | Path | SHA-256 Digest | Size (Bytes) | Lines | Operational Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ingestion Adapter** | `scripts/ingest_alpaca_historical.py` | `7DA7CFC6BE8717C86EC15568F6D16E7173A44A7F33084653E889800F80C273C7` | 74,361 | 1,579 | Production candidate |
| **Guarded Recovery Engine** | `analyst_dashboard/data/guarded_recovery.py` | `8BE7D6A69A3D4316C3CF57793C8921C4E36E5E8A5BF6A78DA91281D6C4A6299D` | 16,446 | 377 | Production candidate |
| **Recovery CLI Tool** | `scripts/guarded_market_recovery.py` | `73E940354777D51F5DBEEB51A86BFF441828BE9138AA3E26FFFCD8614398A7BF` | 2,867 | 73 | Production candidate |
| **Pilot Manifest Canonical V3** | `docs/.../PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` | `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` | 9,972 | 226 | `PROPOSED_NOT_AUTHORIZED` (Sole candidate for eventual pilot ingestion) |
| **Pilot Proposed Manifest V2** | `docs/.../PILOT_PROPOSED_MANIFEST.json` | `8923DAD988A10AE07A2B7BDA1044127130BF9C90539424D98F25676A9A59F07B` | 13,833 | 320 | `SUPERSEDED / NOT FOR EXECUTION` (Retained solely for regression digest verification) |

### Manifest Selection Rules & Invariants
1. **Canonical Pilot Manifest V3 is the SOLE execution candidate**:
   - Every pilot execution runbook and dry-run instruction targets `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` (`0F54C702...`).
   - Status remains strictly `PROPOSED_NOT_AUTHORIZED`.
   - **No manifest is signed during PR preparation.**
2. **Predecessor Manifest V2 is Formally SUPERSEDED**:
   - Status is `SUPERSEDED / NOT FOR EXECUTION`.
   - Retained only to satisfy deterministic freeze regression test `test_deterministic_manifest_digest_verification`.
   - The adapter will never silently substitute V2 for V3. Any execution command referencing V2 is rejected.

---

## 5. AAPL Out-of-Scope Production Tail Ingestion Blocker

### Formal Holding Record:
```
AAPL_OUTSIDE_SCOPE_UNPROVENANCED_TAIL = BLOCKING_FOR_INGESTION
```

### Fact Finding & Fail-Closed Behavior
- **Authoritative Production Baseline**:
  - AAPL stored OHLCV rows: `289` daily sessions.
  - Earliest stored session: `2025-08-27`.
  - Latest stored session: `2026-10-20`.
  - Total provenance rows: `0`.
- **Approved Pilot Manifest V3 Scope**:
  - `start_date`: `2025-08-27`
  - `end_date`: `2026-10-19` (288 NYSE trading sessions).
- **Invariant Conflict**:
  - The production database contains an unprovenanced bar for `2026-10-20`, which is strictly outside the approved V3 window ending on `2026-10-19`.
- **Adapter Safety Guarantee**:
  - The adapter's pre-apply validation checks (`outside_unprovenanced > 0`) evaluate to `1` and fail closed with `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE`.
  - Zero database writes occur; the store remains 100% untouched.
  - Verified by regression test `test_unprovenanced_historical_tail_outside_approved_scope_fails_closed`.
- **Operational Discipline**:
  - Manifest V3 is **NOT** automatically expanded to `2026-10-20`.
  - The production `2026-10-20` record is **NOT** deleted, quarantined, or mutated.
  - Canonical V3 digest (`0F54C702...`) is preserved unchanged.
  - Classification: **This is a production ingestion blocker, NOT an automatic code-only release blocker.**

---

## 6. Test Execution & Rehearsal Results

- **Release Worktree Subtotal**: **94 / 94 Passed (100% Green)**
  - `tests/test_package_b_deployment_rehearsal.py`: 8 passed in ~5.5s
  - `tests/test_pilot_identity_and_guarded_recovery.py`: 15 passed in ~0.5s
  - `tests/test_alpaca_historical_sip_ingestion.py`: 71 passed in ~8.4s
- **Base Repository Cross-Boundary Subtotal**: **159 / 159 Passed (100% Green)**
- **Grand Regression Total**: **253 / 253 Passed (100% Green, 0 Regressions)**

---

## 7. Railway Deployment Trigger Audit

- **Deployment Platform**: Railway (`tranquil-radiance`, env `production`, service `web`)
- **Active Deployment ID**: `4aa235d8-b571-4f67-bb41-2cdfa103fa00` (HEAD `f6c5755126059b2f227e28ce90ddaad43997884c`)
- **Trigger Verification**:
  - `git push origin fix/sprint2-package-b-canonical-ingestion-release`: **Zero deployment side effect**. Railway builds only on merges into `main`.
  - PR Creation: **Zero deployment side effect**. PR preview environments are disabled.
  - Production deployment can only occur upon explicit merge into `main` after independent review.

---

## 8. Pre-Deployment Live Production Logical Content Baseline & Non-Mutation Attestation

Captured from Railway container (`/root/.finance_market_store.db`):
- Total daily candle rows: 36,673
- Total provenance rows: 287 (GOOG only, provider `MASSIVE_POLYGON`)

### Stable Row-Content Logical Hashing

| Symbol | Candle Rows | Provenance Rows | Earliest Date | Latest Date | Candle Logical SHA-256 | Provenance Logical SHA-256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | 289 | 0 | `2025-08-27` | `2026-10-20` | `BD96272C784A6CD1E85016A9878BC4C91B46CA68BA164A87E97CE6BB5F62124E` | `4F53CDA18C2BAA0C0354BB5F9A3ECBE5ED12AB4D8E11BA873C2F11161202B945` |
| **GOOG** | 287 | 287 | `2025-08-27` | `2026-10-16` | `C149DE4587DC9C1CB8021C09C3AA61D03F17F9A83C972E34D8ABE454583EEBB0` | `27A9A825EF160C94519D4BFA5B83C1F176D077285033A6B1D8B6336F4A2DE4B2` |
| **GOOGL** | 288 | 0 | `2025-08-27` | `2026-10-19` | `5CFA99858CFBCE97339B51CB448176459668543E4C2CB26A3C91F5FFD667D496` | `4F53CDA18C2BAA0C0354BB5F9A3ECBE5ED12AB4D8E11BA873C2F11161202B945` |
| **ABBV** | 0 | 0 | None | None | `4F53CDA18C2BAA0C0354BB5F9A3ECBE5ED12AB4D8E11BA873C2F11161202B945` | `4F53CDA18C2BAA0C0354BB5F9A3ECBE5ED12AB4D8E11BA873C2F11161202B945` |

*Attestation*: Raw SQLite file-byte equality is NOT required or claimed across live WAL activity. Separate stable row-content hashing proves **ZERO DATABASE MUTATION** occurred during pre-flight diagnostics. Zero database writes executed.

---

## 9. Pull Request Specification (Awaiting Operator Token)

When the operator separately grants token `G-S2-ACT-CODE-PR-001`, the following pull request will be opened:

- **Target Base**: `main`
- **Head Branch**: `fix/sprint2-package-b-canonical-ingestion-release`
- **PR Title**: `fix(package-b): canonical historical SIP ingestion adapter and guarded recovery controls (#34)`
- **PR Body**:
```markdown
## Summary

Package B canonical code release delivering historical SIP market-data ingestion adapter and guarded emergency recovery controls for ARX Sprint 2.

### Scope of Changes
- Canonical Historical SIP Ingestion Adapter (`scripts/ingest_alpaca_historical.py`) with strict NYSE calendar validation, feed/adjustment enforcement, out-of-scope fail-closed protections, and adversarial precedence handling.
- Guarded Recovery Engine (`analyst_dashboard/data/guarded_recovery.py`) enforcing pre-restore WAL file handle scanning, zero-connection verification, checksum validations, and mandatory confirmation tokens.
- Guarded Recovery CLI (`scripts/guarded_market_recovery.py`).
- 94 comprehensive deployment rehearsal, pilot identity, and adversarial safety tests (`tests/`).
- Authoritative Sprint 2 Package B release runbooks, rollback procedures, and canonical pilot manifests (V2 & V3).

### Verification
- 94/94 isolated tests passed in release worktree.
- 159/159 cross-boundary regression tests passed in base repo.
- 253/253 total tests passed (100% green).
- Zero database mutations, zero historical ingestion executed.
- Verified passive code-only startup safety with zero boot-time writes.

### Operational Hold
- Production historical ingestion remains NOT_AUTHORIZED under G-S2-ACT.
- AAPL_OUTSIDE_SCOPE_UNPROVENANCED_TAIL = BLOCKING_FOR_INGESTION.
```
