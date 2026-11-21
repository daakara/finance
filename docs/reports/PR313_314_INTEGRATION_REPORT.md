# ARX Integration Report: PR #313 & PR #314 Consolidation
**Protocol Reference:** ARX-INTEGRATION-313-314  
**Issue References:** #165 (SEC Form 4 Disclosure View) & #166 (Radar Evidence/Ranking/Handoff)  
**Consolidated PR:** PR #315 / Consolidated Release Integration  
**Base Anchor Commit:** `041ab6e7930037531853362b7b06139e9a54cb1c` (reconciled after PR #314 merge)  
**Historical Anchor:** `868d7afcd528006c2323e6b79b5a17f2f20f677c`  
**Timestamp:** 2026-11-21T07:50:00Z  

---

## 1. Executive Summary & Objective

This integration consolidates two independently qualified, pre-approved release workstreams into a single unified release candidate:
1. **SEC Form 4 Disclosure Evidence View (PR #313)**: Connects verified Form 4 transaction evidence to the institutional disclosure panel with neutral latest-event mapping, fail-closed receipt certification, amendment lineage semantics, and aggregate withholding.
2. **Radar Evidence, Ranking & Analysis-Handoff (PR #314)**: Eliminates synthetic accumulation claims, enforces exact unrounded score ranking via a canonical comparator, and provides transparent discovery-to-analysis handoff disclosures.

Both changesets originally branched from verified production baseline `868d7afcd528006c2323e6b79b5a17f2f20f677c` and achieved 100% success on hosted GitHub Actions:
- PR #313: Hosted Run [44873834199](https://github.com/daakara/finance/actions/runs/44873834199) (Head `668d8a7f5701694660c1f331f1aeb4a76675d8d2`)
- PR #314: Hosted Run [44874232507](https://github.com/daakara/finance/actions/runs/44874232507) (Head `45dcf0e7eb2834de546026bdac77a02e704db3fc`)

During qualification, PR #314 was merged into `release/production` as commit `041ab6e7930037531853362b7b06139e9a54cb1c`. In accordance with Phase 1 Step 6 ("If production has moved, reconcile with the new base before proceeding"), the consolidated integration branch was reconciled onto the new production base `041ab6e7`. Because `041ab6e7` already contains all PR #314 deliverables, integrating PR #313 onto `041ab6e7` produces a unified release containing both projects with zero dropped changes or duplicate work.

---

## 2. Reconciled Files & Conflict Resolution

Four overlapping files were carefully reconciled to preserve all requirements from both workstreams:

### 2.1 `tests/test_smart_money_two_issuer_trial.py`
- **Issue**: Standard runs on non-dissemination days (weekends) evaluate `WEEKEND_EXCLUDED` before quota/slot assertions.
- **Resolution**: Unified on the calendar-deterministic implementation from PR #314:
  - Explicitly pinned New York time to Friday `2026-11-20 09:20:00 EST`.
  - Configured `trial_scheduler.start_date` and `end_date` explicitly covering the test date.
  - Aligned receipt interval timestamps (`2026-11-20T14:15:00Z` to `2026-11-20T15:00:00Z`).
  - Passed `as_of=ny_today` into `_step3_check_durable_date_eligibility_under_lock`.
- **Result**: All 84 test items pass deterministically on any calendar day.

### 2.2 `scripts/release/rq02_diff_integrity.py`
- **Issue**: PR #313 added `313` to `APPROVED_PRS`, while PR #314 added `314`.
- **Resolution**: Combined both approved PR IDs into `APPROVED_PRS = {..., 311, 312, 313, 314, 315}`.
- **Result**: Preserves strict fail-closed PR attribution for files originating in either workstream.

### 2.3 `scripts/release/rq04_frontend_regression.py`
- **Issue**: PR #313 introduced Test 5 for SEC Form 4 Disclosure View, while PR #314 introduced Test 5 for Radar Evidence, Ranking & Handoff.
- **Resolution**: Sequentially chained both test suites:
  - **Step 5**: SEC Form 4 Disclosure Evidence View (`frontend/tests/secDisclosureEvidenceView.test.ts`)
  - **Step 6**: Radar Evidence, Ranking & Analysis-Handoff Qualification (`frontend/tests/radarEvidenceRankingHandoffQualification.test.ts`)
  - Merged evidence path manifests to track all frontend components and test files from both PRs.
- **Result**: `rq04_frontend_regression.py` executes all 6 suites and passes completely with `tsc --noEmit` clean.

### 2.4 `scripts/release/release_manifest.json`
- **Issue**: Production base moved to `041ab6e7930037531853362b7b06139e9a54cb1c`.
- **Resolution**: Generated a unified cumulative manifest under directive `ARX-INTEGRATION-313-314`, binding:
  - Base anchor: `041ab6e7930037531853362b7b06139e9a54cb1c`.
  - Release candidate: The reconciled candidate commit.
  - Full inventory of all candidate files with exact cryptographic digests, sizes, operations, and originating PR attributions.

---

## 3. Invariant Verification Matrix

### 3.1 SEC Form 4 Disclosure View (PR #313 Domain)
- **Issuer/CIK Binding**: Conflicting issuer ticker or CIK is strictly rejected (`CONFLICTING_ISSUER_IDENTITY`, `INVENTORY_ISSUER_MISMATCH`).
- **Accession Evidence & Availability**: Every accession requires source-hash binding in `source_hashes`, exact match in `form4_raw_filings`, and verified public availability in `sm_source_availability_ledger` prior to watermark cutoff.
- **Fail-Closed Certification**: Database query exceptions during supersession or inventory checks fail closed with `VERIFICATION_UNAVAILABLE`.
- **Legitimate PARTIAL Semantics**: Preserved as `PARTIAL` without conversion to `COMPLETE`.
- **Zero Read-Path Side Effects**: 0 live SEC HTTP requests, 0 database writes, SEC advisory lock `401007311509093371` remains completely unheld.

### 3.2 Radar Evidence, Ranking & Handoff (PR #314 Domain)
- **Exact Unrounded Ranking**: Preserved via `compareRadarAssets` in `frontend/lib/radarRanking.ts`. Higher unrounded score (e.g. 45.4 vs 44.6) strictly ranks higher without integer rounding inversion.
- **Verified Evidence Coverage**: Candidate mapping strictly decouples evidence coverage from positive/negative signal status.
- **No Fabricated Narratives**: Removed unsupported accumulation and absorption assertions from VCP+GARP, VCP-only, GARP-only, and VCP+SM narratives.
- **Transparent Analysis Handoff**: `resolveDiscoveryHandoff` names the displayed asset and discloses separate single-asset assessment without implying reproduction of unavailable discovery snapshots.

### 3.3 Production Isolation & Smart Money Ownership
- Both workstreams maintain strict separation from the active Smart Money two-issuer trial (`ARX-TRIAL-2026-NVDA-AAPL-001`).
- Preserves `HOLD_SEC_001` standby boundary and `HOLD_SCAN_001` publication hold.

---

## 4. Test Verification Summary

| Suite | Scope | Status | Notes |
| :--- | :--- | :---: | :--- |
| `rq01_candidate_identity.py` | Ancestry & Target | **PASS** | Base anchor `041ab6e7` verified |
| `rq02_diff_integrity.py` | Changed files & digests | **PASS** | 11 adversarial controls + 12 Gate H attacks pass |
| `rq03_backend_regression.py` | Full backend regressions | **PASS** | 257/257 passed in 32.8s |
| `rq04_frontend_regression.py` | Full frontend regressions | **PASS** | All 6 steps pass, TypeScript clean |
| `secDisclosureEvidenceView.test.ts` | SEC Form 4 disclosure | **PASS** | 5/5 passed |
| `radarEvidenceRankingHandoffQualification.test.ts` | Radar qualification | **PASS** | All checks passed |
| `radarEvidenceRankingHandoffReproduction.test.ts` | Defect reproduction | **PASS** | All 3 defects resolved |
| `test_sec_disclosure_postgres_integration.py` | PostgreSQL disclosure integration | **PASS** | 13/13 passed |
| `test_sec_disclosure_view.py` | SQLite/unit disclosure | **PASS** | 7/7 passed |
| `test_fred_sec_presentation_integrity.py` | Macro presentation integrity | **PASS** | 14/14 passed |
