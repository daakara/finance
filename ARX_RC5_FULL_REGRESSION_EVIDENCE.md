# ARX TERMINAL — SPRINT 3 / CORPORATE EVENTS RC5 FULL REGRESSION EVIDENCE

**Governance Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Authorization Reference:** `RC5_GATE_B_ISOLATED_INTEGRATION_AUTHORIZED`  
**Integration Commit SHA:** `c060700b2b3eedde2ec82800462b6c97a67c2939`  
**Timestamp:** 2026-10-23T10:26:10+02:00  

---

## 1. Test Execution Summary

All test suites were executed in the isolated integration worktree (`C:\Users\akara\Documents\Projects\finance-sprint3-integration`) using isolated local fixtures and mocked providers.

| Test Tier | Test Suite / Command | Total Tests | Passed | Failed | Exit Code | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Backend Corporate Events** | `pytest tests/test_corporate_events_earnings_audit.py` | 16 | 16 | 0 | 0 | 2.04s |
| **Tier 2: Frontend Unit Tests** | `npm run test:unit` (`npx vitest run`) | 258 | 258 | 0 | 0 | 15.70s |
| **Tier 3: Frontend Architectural Invariants** | `npm run test:arch` (29 invariant suites) | 29 suites | 29 suites | 0 | 0 | ~60s |
| **Tier 4: TypeScript AST Type Check** | `npm run type-check` (`tsc --noEmit`) | Full repo | All clean | 0 | 0 | ~15s |
| **Tier 5: Next.js Production Build** | `npm run build` (`next build`) | 147 pages | 147 SSG | 0 | 0 | 52.8s |
| **Tier 6: Core Invariant Backend Suites** | `pytest tests/test_confluence_engine.py ...` | 26 | 26 | 0 | 0 | 10.51s |
| **Tier 7: Packaging Whitelist Verification** | `python scripts/verify_upload_context.py` | 2,114 files | 2,114 | 0 | 0 | 1.1s |
| **Tier 8: Staging Runtime Smoke** | `python scripts/validate_staging_startup.py` | 4 endpoints | 4 (200 OK) | 0 | 0 | 16.2s |
| **Tier 9: Target Architecture & ETF** | `pytest tests/test_target_architecture_api.py ...` | 22 | 22 | 0 | 0 | 10.35s |
| **Tier 10: Invariant Governance Engine** | `pytest tests/test_epoch2_production_governance.py` | 36 | 32 | 4* | 1* | 60.41s |

*\*Note on Tier 10: The 4 failures in `test_epoch2_production_governance.py` and 1 in `test_analytics_nan_incident_epoch2.py` are due to the pre-existing cryptographic manifests (`EPOCH_4_MANIFEST_V5.json` and `FROZEN_ENGINE_MANIFEST_V2_5_1.json`) checking the pre-RC5 SHA256 hashes of `confluence_engine.py` and `analytics.py`. See Section 3 for governance analysis.*

---

## 2. Granular Suite Evidence

### 2.1 Backend Corporate Events Suite (`test_corporate_events_earnings_audit.py`)
```text
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_01_axp_earnings_scheduled_today_applies_maximum_gap_penalty PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_02_axp_earnings_call_vs_results_release_distinction PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_03_confirmed_vs_estimated_dates_distinguished PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_04_earnings_within_24_and_48_hours PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_05_earnings_beyond_risk_window PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_06_earnings_already_completed PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_07_negative_days_to_earnings_remediation PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_08_missing_earnings_timestamp PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_09_provider_timeout PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_10_stale_provider_data PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_11_incorrect_ticker_or_exchange PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_12_daylight_saving_and_timezone_boundaries PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_13_missing_coverage_never_producing_pass PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_14_confluence_score_unchanged_for_unrelated_assets PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_15_preflight_state_consistency_across_api_and_frontend PASSED
tests/test_corporate_events_earnings_audit.py::TestCorporateEventsAndEarningsAudit::test_16_news_sentiment_unavailable_while_earnings_coverage_valid PASSED
16 passed in 2.04s
```

### 2.2 Frontend Unit Suite (`npm run test:unit`)
- **Total Test Files:** 28 passed (28)
- **Total Tests:** 258 passed (258)
- **Key Suites:**
  - `components/__tests__/PreFlightCatalystRisk.test.tsx`: 5 passed
  - `components/__tests__/OptimalEntryExitCard.test.tsx`: passed
  - `components/__tests__/DecisionReadinessCard.test.tsx`: 4 passed
  - `components/__tests__/AnalysisDecisionHierarchy.test.tsx`: 23 passed
  - `components/__tests__/PriceChartVisibility.test.tsx`: 16 passed

### 2.3 Frontend Architectural Invariants Suite (`npm run test:arch`)
- 29/29 invariant test files passed with 0 failures:
  - Client fail-closed fallback
  - Decision contract & taxonomy
  - Governor sizing engine (17 tests)
  - Market data provenance & freshness (8 tests)
  - Epistemic purity & provenance sanitization (12 tests)
  - Radar metric cleanup & taxonomy
  - Tactical setups timeout (15,000ms bounds)
  - Radar portfolio context & badge accessibility
  - Weekly spotlight decoupling (14 criteria)
  - Canonical security master routing (6 gates)
  - UX Phase 1 regression (5 suites)
  - Pre-flight copy remediation & institutional framing
  - Frontend QA escape permanent regression (8 invariants)
  - Research summary & scenario presentation (5 suites)

### 2.4 Next.js Production Build
- **Status:** Compiled and exported successfully (code 0).
- **Static Pages:** 147/147 generated.
- **Shared First-Load JS:** 87.6 kB.
- **Root Page JS:** 97.1 kB.

---

## 3. Cryptographic Governance Manifest Findings

The evaluation of `analyst_dashboard/governance/evaluator.py` under `test_epoch2_production_governance.py` confirmed the expected fail-closed cryptographic containment:
1. `FROZEN_ENGINE_MANIFEST_V2_5_1.json` (ratified on Oct 16 under PR #3) validates `confluence_engine.py` against SHA-256 `147c010eb6068c3470fa3269f08610c0f436ea117b99b3b5217e7020b911e2b1`.
2. `EPOCH_4_MANIFEST_V5.json` validates `api/routes/analytics.py` against SHA-256 `e16428bd27d591211d112e4eb125b21f0ccb4c3493b56a5900deb8854551c3be`.
3. Because RC5 introduced the earnings proximity deduction into `confluence_engine.py` and the `corporateEvent` field into `api/routes/analytics.py`, the live file hashes changed:
   - `confluence_engine.py` -> `0f77d3b0b15f5d60a6de52ec1a0cd6b385ce1615865839d684d53aa655dadc5a`
   - `api/routes/analytics.py` -> `361ae1daa6ca5379222b65f00cc31c4bad0ec5589d53dac3d7c2ad9e39aea278`
4. The production evaluator correctly flagged this as `MANIFEST_CORRUPTED`, preventing automated self-certification of unratified engine modifications.
5. In accordance with ARX Governance Invariant rules, feature candidates cannot unilaterally update production manifests. Ratification of manifest succession (e.g. `EPOCH_4_MANIFEST_V6.json` and `FROZEN_ENGINE_MANIFEST_V2_5_2.json`) requires a separate operator governance authorization at Gate D.
