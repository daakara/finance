# ARX TERMINAL — RADAR MINERVINI VCP LOCAL CANDIDATE
## ACCEPTANCE AND VERIFICATION REPORT

**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Workstream:** Sprint 3 / Radar Operational Minervini VCP Screening  
**Branch:** `feat/radar-minervini-vcp-activation`  
**Base:** `d19b2c4cc7bb9122fdfe323284de0ae0d42fbcc2`  
**Integrated Main Baseline:** `eb6fbfacc1a3e8784704567301244266475468d3`  
**Authorization:** ARX_VCP_COMMIT_RECONCILIATION_PUSH_AND_DRAFT_PR_AUTHORIZED  
**Execution Posture:** Local Candidate Reconciliation, Main Integration & Draft PR Publication  

---

### 1. Executive Summary

Under explicit Product Owner authorization, the Minervini Volatility Contraction Pattern (VCP) scanner candidate has been fully reconciled, integrated with current `origin/main`, and verified against all authorized Python and frontend regression test suites.

The candidate replaces the placeholder/proxy scanner implementation with the canonical bitemporal `VCPClassifier` and `VCPTemporalCaseCompiler`, enforces strict mathematical history boundary gates, ensures truthful capability and count disclosures across all lifecycle states, and preserves all research holdout commitments.

---

### 2. Candidate Commit Lineage & Reconciled Hierarchy

The candidate branch `feat/radar-minervini-vcp-activation` incorporates the following explicit, non-rewritten commits:

1. **`b10ede531a9618c9a214b00de32e64d6a6635f1d`**  
   *`feat(radar): activate operational Minervini VCP screening with canonical classifier`*  
   Wires canonical `VCPClassifier` and `VCPTemporalCaseCompiler` into `VCPScannerRunner` with 252-bar window, defensive candle validation, authoritative evaluation tracking, and frontend badge synchronization.

2. **`681e31efed9bfe30c0a053bee794b4c8dab07841`**  
   *`fix(radar): enforce 200-bar history boundary and add permanent VCP regression tests`*  
   Elevates candle pre-filter threshold from 50 to 200 bars in `VCPScannerRunner`, publishes `PIPELINE_PENDING` with `matched_count=None` when all symbols have insufficient history, and adds boundary test suite in `tests/test_vcp_activation.py`.

3. **`ea7ffe1163983998db061d83f19f8c07be687903`**  
   *`test(radar): replace brittle VCP badge string assertion with semantic capability contract test`*  
   Replaces fragile static string assertion in `frontend/tests/preflightRadarCopyRemediation.test.ts` with comprehensive semantic capability-to-badge contract tests across all 6 lifecycle states and 4 negative adversarial cases.

4. **`ccb4f14a74eac4f75d7773557a5c0859ccf4f996`**  
   *`Merge branch 'origin/main' into feat/radar-minervini-vcp-activation`*  
   Clean integration of `origin/main` (`eb6fbfacc1a3e8784704567301244266475468d3`), reconciling atomic reliability, candidate truth provenance (#71), and VCP scanner activation with zero merge conflicts.

---

### 3. Exact History-Boundary Mathematical Analysis & Behavior

The canonical classifier (`VCPClassifier`) and runner (`VCPScannerRunner`) distinguish evaluable from excluded assets across all 6 exact history boundaries:

| Bar Count | Canonical Status | Predicate Outcome | Runner Handling & Mathematical Rationale |
| :---: | :---: | :---: | :--- |
| **199** | **EXCLUDED** | `INSUFFICIENT_DATA` | **Excluded Pre-Scan:** `session_count < 200`. The 200-SMA requires $\ge 200$ trading sessions. `PRED_SUFFICIENT_HISTORY` fails closed. The runner records `unavailable_reasons["INSUFFICIENT_HISTORY"] += 1` without executing the classifier. If all assets are < 200, publishes `PIPELINE_PENDING` (`matched_count=None`). |
| **200** | **EVALUABLE** | `PASS` | **Exact Minimum 200-SMA:** `session_count == 200`. Exact minimum window for `sma_200 = sum(200) / 200`. The 22-bar slope uses contract fallback `prior = sma_series[199] == current`, yielding baseline `slope_22 = 0.0`. Trend template is evaluable. |
| **221** | **EVALUABLE** | `PASS` | **Partial Slope Fallback:** `session_count == 221`. 200-SMA exists across 22 observations. The 22-bar slope uses the earliest available SMA (`sma_series[199]`), representing a 21-session lag. Trend template and contraction stages evaluate authoritatively. |
| **222** | **EVALUABLE** | `PASS` | **Full 22-Session Slope:** `session_count == 222`. Exact boundary for true 22-session SMA slope with zero fallback: `sma_series[222-23] = sma_series[199]` ($221 - 199 = 22$). Complete uncompromised moving-average trend template. |
| **251** | **EVALUABLE** | `PASS` | **High-Fidelity Window:** `session_count == 251`. Full 200-SMA and 22-session slope. 52-week window evaluates over `min(251, 252) = 251` sessions. Evaluates authoritatively. |
| **252** | **EVALUABLE** | `PASS` | **Standard 1-Year Benchmark:** Full 252-session trading year. Exact 52-week high/low boundaries and 200-SMA slope. Benchmark sample size per `ConsumerReadinessEvaluator`. |

---

### 4. Verification Evidence

#### 4.1 Python Regression Suite (157 / 157 PASSED)
Executed across `tests/test_vcp_activation.py`, `tests/test_radar_coverage_disclosure_semantics.py`, `tests/test_radar_scanner_pipeline.py`, `tests/test_scanner_truthfulness_remediation.py`, `tests/test_vcp_c004_universe_wiring.py`, `tests/test_vcp_natural_control_plane.py`, `tests/test_sprint_2b_vcp_domain_authority.py`, `tests/test_universe_admission_mappings.py`, `tests/test_radar_candidate_truth_remediation.py`, and `tests/test_radar_price_provenance.py`:

```
====================== 157 passed, 3 warnings in 27.22s =======================
```

#### 4.2 Frontend Test Suite (100% PASSED)
Executed via `npm test` (`tsc --noEmit`, `vitest run`, and `test:arch` across 31 architectural test files):
- `tsc --noEmit`: 0 type errors.
- `vitest run`: 41 tests passed across command strip, navigation, and telemetry.
- `test:arch`: All 31 test suites passed cleanly, including `preflightRadarCopyRemediation.test.ts`, `radarCandidateTruthRemediation.test.ts`, `radarErrorAndZeroSemantics.test.ts`, `radarPriceProvenance.test.ts`, and `marketDataProvenance.test.ts`.

#### 4.3 Authentic-Data Isolation Verification
Evaluated against an isolated copy of `C:\Users\akara\.finance_market_store.db`:
- **Population:** 35 canonical assets.
- **Observations:** All 35 assets have between 286 and 489 bars ($\ge 286$ bars $\ge 252$).
- **Outcome:** 35 evaluated, 0 excluded, 0 matches.
- **Status:** Truthful `AVAILABLE`, `matched_count = 0`, `SCANNER_OPERATIONAL` with 100% complete coverage disclosure.

---

### 5. Complete PR Diff Inventory Against `origin/main`

A clean comparison of `origin/main` (`eb6fbfa`) against `HEAD` reveals changes strictly confined to the intended VCP operational scope:

| File | Nature of Change |
| :--- | :--- |
| `analyst_dashboard/analyzers/scanner_contract.py` | Serializes `snapshot_id`, `matched_count`, `data_as_of`, `generated_at` in snapshot dict |
| `analyst_dashboard/analyzers/scanner_runner.py` | Wires canonical `VCPClassifier`, enforces 200-bar floor, and attributes `PIPELINE_PENDING` on insufficient history |
| `api/routes/screener.py` | Extracts `matched_count` and `snapshot_id` from top-level or nested snapshot structures |
| `frontend/app/radar/page.tsx` | Synchronizes VCP badge with dynamic `matchedCount` when available |
| `frontend/tests/preflightRadarCopyRemediation.test.ts` | Permanent semantic capability-to-badge contract tests (6 states, 4 negative tests) |
| `tests/test_vcp_activation.py` | Permanent regression tests covering all 6 history boundaries and runner lifecycle states |
| `docs/reports/ARX_VCP_LOCAL_CANDIDATE_ACCEPTANCE_REPORT.md` | Formal candidate acceptance report |

---

### 6. Deployment-Trigger Isolation

- **Railway Production:** Auto-deploy is restricted to pushes to branch `main`. Feature branch pushes do not deploy.
- **Cloudflare Edge Cache Purge:** In `ci.yml`, `deploy_and_purge_cache` is strictly gated on `if: github.ref == 'refs/heads/main' && github.event_name == 'push'`.
- **Scheduled Automations:** `vcp-natural-scheduler.yml` is `workflow_dispatch` only.
- **Production Data:** Zero production database mutation or active scanning authorized.
