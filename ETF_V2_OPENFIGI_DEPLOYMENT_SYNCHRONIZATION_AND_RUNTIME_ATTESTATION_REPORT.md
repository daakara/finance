# ETF V2 — OpenFIGI Deployment Synchronization & Production Runtime Attestation Report

**Gate Identifier**: `ETF_V2_OPENFIGI_DEPLOYMENT_SYNCHRONIZATION_AND_RUNTIME_ATTESTATION_GATE`  
**Execution Timestamp**: `2026-10-06T03:36:00Z`  
**Predecessor Gate**: `HOLD_ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION`  
**Baseline Commit**: `f5ba5b28fb08091e1adc2ef2e704db1dc938e91a`  
**Remediation Code SHA**: `6aadcdf5d69ad0811ada4e6215233fa4e9abbd90`  
**Production Release Candidate SHA (Branch Head)**: `ac6461c5615b9793c32bde69e7d583ce30e71102`  
**Current Production Runtime SHA**: `2a239fb3ecf3e88744a42ae64a4443c039a8d626`  
**Final Gate Verdict**: `HOLD_ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION`  

---

## 1. Executive Summary & Purpose

This gate investigates and resolves the deployment synchronization and production runtime attestation boundary for the OpenFIGI operational DB-path remediation.

In the preceding gate, the remediation scope was proven clean and verified across all 121 tests (62 OpenFIGI + 59 ETF v2 regressions), committed, and published to `origin/arx/etf-v2-openfigi-remediation`. However, the live Railway production container (`web-production-470560.up.railway.app`) was found to be executing commit `2a239fb3ecf3e88744a42ae64a4443c039a8d626`.

This gate performed:
1. Re-attestation of repository boundary and local/remote parity;
2. Resolution of the SHA delta between code commit `6aadcdf` and documentation commit `ac6461c`;
3. Discovery of the Railway production deployment source;
4. Non-mutating pre-deployment merge safety analysis (`git merge-tree`);
5. Live production container inspection and file analysis;
6. Formal re-adjudication of criteria ETF-RELEASE-01 through ETF-RELEASE-20 under the mandatory evidence rule (*"ETF-RELEASE-08 through ETF-RELEASE-13 are production-runtime criteria and may be PASS only from deployed-candidate production evidence"*).

---

## 2. Repository Boundary Re-Attestation (Section 1)

Executed strictly from `C:/Users/akara/Documents/Projects/finance-etf-v2`:

```ini
BRANCH =
  arx/etf-v2-openfigi-remediation
WORKTREE_CLEAN =
  YES
LOCAL_HEAD =
  ac6461c5615b9793c32bde69e7d583ce30e71102
REMOTE_HEAD =
  ac6461c5615b9793c32bde69e7d583ce30e71102
LOCAL_REMOTE_PARITY =
  YES
SECURITY_MASTER_FILES_PRESENT =
  NO
UNRELATED_ARX_FILES_PRESENT =
  NO
```

---

## 3. Candidate SHA Ambiguity Resolution (Section 2)

Inspected diff between `6aadcdf5d69ad0811ada4e6215233fa4e9abbd90` and `ac6461c5615b9793c32bde69e7d583ce30e71102`:

```ini
git diff --name-status 6aadcdf5d69ad0811ada4e6215233fa4e9abbd90..ac6461c5615b9793c32bde69e7d583ce30e71102
```

Outputs:
* `A ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION_MANIFEST.json`
* `A ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION_REPORT.md`

```ini
RUNTIME_CODE_DIFFERENCE =
  NO
RUNTIME_EQUIVALENCE =
  VERIFIED_DOCUMENTATION_ONLY_SUCCESSOR
PRODUCTION_RELEASE_CANDIDATE_SHA =
  ac6461c5615b9793c32bde69e7d583ce30e71102
```

The branch head `ac6461c5615b9793c32bde69e7d583ce30e71102` is the authoritative candidate so that branch, remote, and production converge to one exact SHA.

---

## 4. Test Attestation (Section 3)

Because `ac6461c` is a documentation-only successor to `6aadcdf`:
* `TESTED_RUNTIME_SHA = 6aadcdf5d69ad0811ada4e6215233fa4e9abbd90`
* `DEPLOYMENT_SHA = ac6461c5615b9793c32bde69e7d583ce30e71102`
* `RUNTIME_EQUIVALENCE = VERIFIED_DOCUMENTATION_ONLY_SUCCESSOR`
* All 121 tests pass with 0 failures:
  - 62 OpenFIGI tests pass
  - 59 ETF v2 regression tests (42 pass, 17 skip) pass

---

## 5. Deployment Workflow Discovery (Section 4)

Railway environment inspection via `railway status`, `railway service list --json`, and `railway variables --json`:

* Service `web` is linked to GitHub repository `daakara/finance` (`source.repo = daakara/finance`).
* Production autodeploy tracks `refs/heads/main`.
* Pushing to branch `arx/etf-v2-openfigi-remediation` published the branch to GitHub, but did not trigger Railway deployment.

```ini
RAILWAY_DEPLOYMENT_SOURCE =
  github://daakara/finance@refs/heads/main
MERGE_TO_MAIN_REQUIRED_FOR_DEPLOYMENT =
  YES
CLI_DEPLOY_FROM_BRANCH_SUPPORTED =
  NO (CLI upload from baseline worktree would overwrite container with f5ba5b2 state, regressing 26 mainline commits)
```

---

## 6. Pre-Deployment Merge Safety Analysis (Section 5)

Non-mutating merge analysis executed from `finance-etf-v2`:

```ini
git merge-base origin/main arx/etf-v2-openfigi-remediation
-> f5ba5b28fb08091e1adc2ef2e704db1dc938e91a

git merge-tree --write-tree origin/main arx/etf-v2-openfigi-remediation
```

### Conflict Findings:
`git merge-tree` revealed **6 merge conflicts**:
1. `scripts/research/etf_v2/openfigi_config.py` (add/add conflict)
2. `scripts/research/etf_v2/openfigi_persistence.py` (content conflict)
3. `scripts/research/etf_v2/openfigi_rate_limiter.py` (content conflict)
4. `tests/test_etf_v2_openfigi_contract.py` (content conflict)
5. `tests/test_etf_v2_openfigi_global_rate_limiter.py` (content conflict)
6. `tests/test_etf_v2_openfigi_path_resolution.py` (add/add conflict)

**Root Cause**: An intermediate, unhardened commit (`c25553db6b5963caeddc10b2aeb65e855042fb40`) was previously committed to `main`. When `arx/etf-v2-openfigi-remediation` was branched cleanly from the official baseline `f5ba5b28`, Git flagged conflicts between the intermediate commit on `main` and the completed remediation on `arx/etf-v2-openfigi-remediation`.

Furthermore, `origin/main` contains 26 commits, including Security Master governance files (`2a239fb`), which must not be merged into the remediation branch.

```ini
MERGE_CONFLICTS =
  6
UNRELATED_MAIN_CHANGES =
  26 commits (SaaS foundation, Analysis UX, Radar portfolio-aware status, Security Master design 2a239fb)
MERGE_SAFETY =
  HOLD
```

Per Section 5 instruction: *"If merging to main is required, do not perform it until the release diff and conflict analysis are clean. If normal repository governance requires an explicit reviewed merge, follow that process."*

---

## 7. Live Production Inspection (Section 6 & 7)

Railway container inspected via Railway CLI (`railway service files download /app/scripts/research/etf_v2/openfigi_config.py`):
* Downloaded and verified the container file `openfigi_config.py`: it contains the 85-line intermediate implementation from commit `c25553d`.
* It resolves relative paths relative to `REPO_ROOT`, but lacks `validate_openfigi_operational_db_path`, `validate_store_path_parity`, and `RELATIVE_OVERRIDE_ALLOWED = NO` validation.
* `/health` HTTP telemetry probe confirmed:
  - `status`: `online` (200 OK)
  - `backend_release_sha`: `"2a239fb3ecf3e88744a42ae64a4443c039a8d626"`

```ini
DEPLOYMENT_ID =
  f0270890-4d5d-482f-b570-b49004baf49b
DEPLOYED_SHA =
  2a239fb3ecf3e88744a42ae64a4443c039a8d626
PRODUCTION_RUNTIME_SHA =
  2a239fb3ecf3e88744a42ae64a4443c039a8d626
PRODUCTION_RELEASE_CANDIDATE_SHA =
  ac6461c5615b9793c32bde69e7d583ce30e71102
RUNTIME_CANDIDATE_PARITY =
  NO
```

---

## 8. Corrected Acceptance Matrix Re-Adjudication (Section 16)

> **Mandatory Evidence Rule**:
> ETF-RELEASE-08 through ETF-RELEASE-13 are production-runtime criteria.
> They may be PASS only from deployed-candidate production evidence.
> Local tests, source inspection, or undeployed candidate behavior cannot satisfy them.

| Criterion ID | Criterion Description | Status | Evidence / Notes |
| :--- | :--- | :---: | :--- |
| **ETF-RELEASE-01** | Dedicated branch/worktree verified | **PASS** | Branch `arx/etf-v2-openfigi-remediation`, isolated clean worktree |
| **ETF-RELEASE-02** | Remediation diff scope clean | **PASS** | Exactly 10 files in allowed classes against `f5ba5b2`, 0 unrelated files |
| **ETF-RELEASE-03** | Local/remote commit parity verified | **PASS** | Local and remote SHA `ac6461c5...` equal |
| **ETF-RELEASE-04** | Exact release SHA tests pass | **PASS** | 121 tests pass on candidate code, 0 failures |
| **ETF-RELEASE-05** | Deployment completed successfully | **HOLD** | Candidate `ac6461c5...` not yet deployed to Railway |
| **ETF-RELEASE-06** | Deployed SHA equals release candidate SHA | **HOLD** | Deployed `2a239fb...` != Candidate `ac6461c5...` |
| **ETF-RELEASE-07** | Runtime SHA equals deployed SHA | **HOLD** | Runtime matches deployed `2a239fb...`, but != candidate SHA |
| **ETF-RELEASE-08** | Production canonical path absolute | **HOLD** | Candidate not yet active on production container |
| **ETF-RELEASE-09** | Production path CWD independent | **HOLD** | Candidate not yet active on production container |
| **ETF-RELEASE-10** | Limiter/persistence path parity verified | **HOLD** | Candidate not yet active on production container |
| **ETF-RELEASE-11** | Startup validation passes | **HOLD** | Candidate not yet active on production container |
| **ETF-RELEASE-12** | No fallback operational DB created | **HOLD** | Candidate not yet active on production container |
| **ETF-RELEASE-13** | Production artifact uses canonical resolver | **HOLD** | Deployed container currently runs older intermediate code from `2a239fb` |
| **ETF-RELEASE-14** | No canonical ETF mutation | **PASS** | Canonical DB digests bit-for-bit intact |
| **ETF-RELEASE-15** | No mapping semantics change | **PASS** | Zero mapping, classification, or normalization changes |
| **ETF-RELEASE-16** | No rate-limit threshold change | **PASS** | 20 requests / rolling 60s strictly enforced |
| **ETF-RELEASE-17** | No remediation-related production errors | **PASS** | Zero errors in Railway container logs |
| **ETF-RELEASE-18** | No live OpenFIGI requests required | **PASS** | 0 live provider network calls dispatched |
| **ETF-RELEASE-19** | Security Master scope untouched | **PASS** | 0 Security Master files or dependencies in remediation branch |
| **ETF-RELEASE-20** | No automatic provider activation performed | **PASS** | Provider activation remains `HOLD` |

---

## 9. Formal Gate Verdict (Section 18)

```ini
GATE =
  HOLD_ETF_V2_OPENFIGI_REMEDIATION_RELEASE_VERIFICATION
PRIMARY_REASON =
  PRODUCTION_DEPLOYMENT_PARITY_PENDING_MERGE_RECONCILIATION
UNVERIFIED_ACCEPTANCE_CRITERIA =
  ETF-RELEASE-05, ETF-RELEASE-06, ETF-RELEASE-07, ETF-RELEASE-08, ETF-RELEASE-09, ETF-RELEASE-10, ETF-RELEASE-11, ETF-RELEASE-12, ETF-RELEASE-13
MISSING_EVIDENCE =
  Railway production runtime is executing commit 2a239fb3ecf3e88744a42ae64a4443c039a8d626; candidate ac6461c5615b9793c32bde69e7d583ce30e71102 has 6 merge conflicts against origin/main due to predecessor intermediate commit c25553d. Deployed candidate runtime evidence (ETF-RELEASE-08..13) cannot be established until reviewed merge and deployment synchronization are performed.
OPENFIGI_ACTIVATION =
  HOLD
NEXT_ACTION =
  CONDUCT_REVIEWED_MERGE_RECONCILIATION_AND_SYNCHRONIZE_RAILWAY_DEPLOYMENT
AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```

---

## 10. Mandatory Stop Enforcement (Section 20)

* **Unrestricted OpenFIGI Usage**: NOT ACTIVATED.
* **ETF Population Regeneration**: NOT EXECUTED.
* **Bulk Mappings**: NOT EXECUTED.
* **Local Limit**: Strictly preserved at 20 requests / rolling 60 seconds.
* **Security Master Track**: Strictly isolated.
* **Controlled Live Validation**: NOT EXECUTED.
