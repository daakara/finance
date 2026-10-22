# ARX TERMINAL — SPRINT 2 S2-09
# COUNT AUTHORITY CLOSURE & SCANNER MATCH SEPARATION REPORT

**Governing authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Execution mode:** TARGETED VERIFICATION / LOCAL REMEDIATION ONLY  
**Worktree:** `finance-sprint-2-accounting-closure-remediation`  
**Branch:** `fix/sprint2-publication-accounting-closure-semantics`  
**Remediation commit:** `248b7dd`  
**Railway backend:** OUTAGE / DEPLOYMENT HOLD (`f6c5755126059b2f227e28ce90ddaad43997884c`)  
**Production mutations:** PROHIBITED / NOT AUTHORIZED  
**Final disposition:** `MATERIAL_DEFECT_REMEDIATED`

---

## 1. Executive Disposition & Summary

Investigation confirmed a **material defect** in the interaction between authoritative backend scanner match counts and frontend-filtered display counts, as well as two related count-authority edge cases in the disclosure engine and API routes.

All defects were surgically repaired, verified with deterministic offline fixtures, and committed to the isolated Sprint 2 worktree (`248b7dd`).

### Summary of Identified Defects & Remediations

| ID | Component | Defect Description | Remediation Applied |
|:---|:---|:---|:---|
| **DEF-CNT-01** | `frontend/app/radar/page.tsx` | **Frontend Count Conflation:** The VCP header banner rendered `{filteredAssets.length} matches · {denominator} scannable population`. When user-facing filters (search query or portfolio ownership filter) were applied, `filteredAssets.length` changed, causing the banner to misrepresent the filtered display count as the total scanner match count across the scannable population (e.g. searching "AAPL" rendered `1 matches · 150 scannable population`). | Decoupled display count from authoritative match count. When user filters reduce the count, the banner displays: `Showing {filteredAssets.length} of {capabilities.VCP.matchedCount} scanner matches · {denominator} scannable population`. When unfiltered: `{capabilities.VCP.matchedCount} scanner matches · {denominator} scannable population`. |
| **DEF-CNT-02** | `analyst_dashboard/universe/isolated_publication.py` & `api/routes/screener.py` | **Zero-Count Fabrication:** Incomplete snapshots lacking `matched_count` defaulted to `0` (`.get("matched_count", 0)`), violating Invariant 3 by fabricating a zero-match result for unverified or incomplete runs. | Enforced strict integer verification (`isinstance(raw_matched_count, int) and raw_matched_count >= 0`). Missing or null `matched_count` fails closed to `matchedCount: None` and `universeScreening: PIPELINE_PENDING`. |
| **DEF-CNT-03** | `analyst_dashboard/universe/isolated_publication.py` & `api/routes/screener.py` | **Rollback Preemption Leak:** A publication pointer with `is_rollback=True` was not preempting operational status if valid strategy admission or dispatch evidence was provided. | Elevated `is_rollback` check to an immediate preemption gate at the entry of both disclosure functions. A rolled-back pointer strictly demotes to `PIPELINE_PENDING`, `matchedCount: None`, and `lifecycleState: STALE_OR_ROLLED_BACK_PUBLICATION`. |

---

## 2. Invariant Verification Matrix

| Required Invariant | Status | Verification Evidence |
|:---|:---:|:---|
| **1. `matchedCount` sourced from completed, authorized scanner result** | **VERIFIED** | `test_matched_count_sourced_from_completed_scanner_result` proves missing `matched_count` in an AVAILABLE snapshot fails closed to `matchedCount: None` and `PIPELINE_PENDING`. |
| **2. `filteredAssets.length` treated as display count, not authoritative scanner output** | **VERIFIED** | `radarCountAuthority.test.ts` (Test 1.1 & 1.2) proves elimination of `{filteredAssets.length} matches` and verifies `Showing {filteredAssets.length} of {matchedCount} scanner matches`. |
| **3. Pending, held, unauthorized, or stale states cannot display fabricated zero-match** | **VERIFIED** | `radarCountAuthority.test.ts` (Test 3.2 & 3.3) and Python test suite confirm pending states render `Scanner Pending` with strictly `matchedCount: null`. |
| **4. Operational zero matches remain valid when confirmed by scanner snapshot** | **VERIFIED** | `test_tier5_scanner_operational_with_truthful_zero_count` and `radarCountAuthority.test.ts` (Test 3.1) verify that an authentic snapshot with `matched_count=0` renders `0 scanner matches`. |
| **5. Population denominators associated with publication build & digest** | **VERIFIED** | `populationDenominator` remains strictly bound to `universeBuildId` and `publicationDigest`, rendered in banner and tooltips. |
| **6. Stale or rolled-back pointers cannot imply operational scanner availability** | **VERIFIED** | `test_rolled_back_pointer_strictly_blocks_operational_availability` and `radarCountAuthority.test.ts` (Test 4.1) prove immediate demotion on `isRollback: true`. |
| **7. User-facing filters change display counts without altering authoritative match counts** | **VERIFIED** | `radarCountAuthority.test.ts` (Test 5.1 & 5.2) verifies that searching or filtering by ownership alters `filteredAssets` without mutating `capabilities.VCP.matchedCount`. |

---

## 3. Code Modifications Ledger

### 3.1. Frontend Header Banner (`frontend/app/radar/page.tsx`)
```tsx
// Lines 978-998
<span className="text-slate-400">
  {capabilities.VCP?.universeScreening === 'AVAILABLE' ? (
    <>
      {typeof capabilities.VCP?.matchedCount === 'number' ? (
        filteredAssets.length !== capabilities.VCP.matchedCount ? (
          <>
            Showing <strong className="text-white">{filteredAssets.length}</strong> of <strong className="text-white">{capabilities.VCP.matchedCount}</strong> scanner matches · {capabilities.VCP?.populationDenominator ? `${capabilities.VCP.populationDenominator} scannable population` : '100% coverage'}
          </>
        ) : (
          <>
            <strong className="text-white">{capabilities.VCP.matchedCount}</strong> scanner matches · {capabilities.VCP?.populationDenominator ? `${capabilities.VCP.populationDenominator} scannable population` : '100% coverage'}
          </>
        )
      ) : (
        <>
          Scanner Pending · {capabilities.VCP?.populationDenominator ? `${capabilities.VCP.populationDenominator} scannable population` : 'Universe Pending'}
        </>
      )}
    </>
  ) : (
    <>
      Scanner Pending · {capabilities.VCP?.populationDenominator ? `${capabilities.VCP.populationDenominator} scannable population` : 'Universe Pending'}
    </>
  )}
</span>
```

### 3.2. Backend Disclosure Engine (`analyst_dashboard/universe/isolated_publication.py`)
```python
# Rollback immediate preemption (Lines 1198-1218)
if is_rollback:
    return {
        "status": "PIPELINE_PENDING",
        "universeScreening": "PIPELINE_PENDING",
        "singleAssetAnalysis": "AVAILABLE",
        "matchedCount": None,
        "populationDenominator": population_denominator,
        "universeBuildId": build_id,
        "universeId": universe_id,
        "publicationDigest": digest,
        "isRollback": True,
        "rollbackReason": rollback_reason,
        "membershipStatus": "AVAILABLE",
        "lifecycleState": "STALE_OR_ROLLED_BACK_PUBLICATION",
        "reason": "ROLLED_BACK_PUBLICATION_POINTER",
        ...
    }

# Strict match count validation in Tier 4 (Lines 1288-1296)
raw_matched_count = active_snapshot.get("matched_count") if active_snapshot else None
if snapshot_status == "AVAILABLE" and isinstance(raw_matched_count, int) and raw_matched_count >= 0:
    ...
    "matchedCount": raw_matched_count,
```

### 3.3. API Route Handler (`api/routes/screener.py`)
- Added entry-level `is_rollback` preemption gate in `get_radar_capabilities`.
- Enforced strict integer verification on `vcp_snapshot.get("matched_count")`.

---

## 4. Focused Verification Test Evidence

### 4.1. Frontend Test Suite (`radarCountAuthority.test.ts`)
```bash
> npx tsx tests/radarCountAuthority.test.ts
Starting Radar Count Authority & Filtered Display Verification Suite...

1. Verifying static count authority invariants in frontend/app/radar/page.tsx...
   [PASS] 1.1: Conflation of filteredAssets.length with scanner matches eliminated
   [PASS] 1.2: Filtered display count clearly distinguished from authoritative scanner match count
   [PASS] 1.3: Authoritative scanner matches displayed when no reducing filter is applied
   [PASS] 1.4: Pending states render 'Scanner Pending', never fabricated zero matches

2. Simulating Radar banner count formatting logic across filter states...
   [PASS] 2.1: Unfiltered operational scanner: '12 scanner matches · 150 scannable population'
   [PASS] 2.2: User search filter: 'Showing 1 of 12 scanner matches · 150 scannable population' (matchedCount=12 intact)
   [PASS] 2.3: Restrictive filter: 'Showing 0 of 12 scanner matches' (distinguished from scanner zero-match)

3. Testing truthful operational zero matches vs pending states...
   [PASS] 3.1: Authoritative operational zero matches: '0 scanner matches · 150 scannable population'
   [PASS] 3.2: Pending scanner: 'Scanner Pending · 150 scannable population' (zero fabricated 0)
   [PASS] 3.3: Incomplete snapshot fail-closed to 'Scanner Pending'

4. Testing rollback pointer containment...
   [PASS] 4.1: Rolled-back pointer enforces PIPELINE_PENDING and matchedCount=null

5. Testing filter independence across sort and ownership states...
   [PASS] 5.1: Ownership filter reduces displayed assets (2) without mutating authoritative matches (3)
   [PASS] 5.2: Search filter reduces displayed assets (1) without mutating authoritative matches (3)

===============================================================================
ALL RADAR COUNT AUTHORITY & FILTERED DISPLAY TESTS PASSED SUCCESSFULLY! ✅
===============================================================================
```

### 4.2. TypeScript Compilation Check
```bash
> tsc --noEmit
# Exit Code: 0 (Zero errors)
```

### 4.3. Python Focused Test Execution (17/17 Passed)
```bash
> python -m pytest tests/test_radar_coverage_disclosure_semantics.py tests/test_vcp_inactive_read_safety.py -v
============================= 17 passed, 1 warning in 18.34s =============================
- test_tier1_no_committed_publication_pointer .................................... PASSED
- test_tier2_committed_publication_scanner_not_enrolled .......................... PASSED
- test_tier3_scanner_enrolled_dispatch_unauthorized .............................. PASSED
- test_tier4_operational_deployment_hold .......................................... PASSED
- test_tier5_scanner_operational_with_truthful_zero_count ........................ PASSED
- test_tier6_rolled_back_pointer_disclosure ...................................... PASSED
- test_api_route_get_radar_capabilities_integration .............................. PASSED
- test_matched_count_sourced_from_completed_scanner_result ....................... PASSED
- test_rolled_back_pointer_strictly_blocks_operational_availability .............. PASSED
- test_api_get_radar_capabilities_count_authority_and_rollback_containment ....... PASSED
- test_01_existing_usable_snapshot ............................................... PASSED
- test_02_empty_storage_cache_miss ............................................... PASSED
- test_03_snapshot_release_mismatch .............................................. PASSED
- test_04_stale_snapshot_policy .................................................. PASSED
- test_05_storage_failure ........................................................ PASSED
- test_06_repeated_reads ......................................................... PASSED
- test_07_complete_status_propagation_matrix ..................................... PASSED
```

### 4.4. Existing Radar Regression Suites Re-verified
- `npx tsx tests/preflightRadarCopyRemediation.test.ts`: PASSED
- `npx tsx tests/governorSizingEngine.test.ts`: PASSED
- `npx tsx tests/radarPortfolioContext.test.ts`: PASSED
- `npx tsx tests/radarMetricCleanup.test.ts`: PASSED

---

## 5. Remaining Limitations

1. **Railway Outage Containment:** Live execution of the scanner remains held on the production deployment barrier (`f6c5755126059b2f227e28ce90ddaad43997884c`).
2. **PR #40 Isolation:** All changes reside strictly on `fix/sprint2-publication-accounting-closure-semantics` in `finance-sprint-2-accounting-closure-remediation`. Production `main` and PR #40 release branches remain untouched.

---

## 6. Closing Disposition

$$\mathbf{DISPOSITION:\ MATERIAL\_DEFECT\_REMEDIATED}$$

All required count-authority invariants have been verified, repaired, and confirmed by deterministic test suites.
