# ARX TERMINAL — PRODUCTION QA ESCAPE REGISTRY

```text
DOCUMENT_TYPE =
  CANONICAL_GOVERNANCE_REGISTRY
AUTHORITY =
  PRODUCT_GOVERNANCE + QA_SYSTEM_ARCHITECTURE + RELEASE_MANAGEMENT
STATUS =
  RECONCILED_AND_ACTIVE
LOCATION =
  docs/governance/ARX_QA_ESCAPE_REGISTRY.md
ESTABLISHED =
  2026-10-08
CANONICAL_INTEGRATION_BASE =
  6f0559d93a681c3bb7c3a89883e0a820934a4a7a
CURRENT_PRODUCTION_BASELINE =
  5dcfeb41d75bb3ae02f25cb3599ab86c0cb03950
```

---

## 1. Executive Purpose & Governance Boundary

The **ARX Production QA Escape Registry** is the authoritative, permanent ledger of all confirmed or suspected defects, product-quality escapes, and epistemic contradictions discovered in production after technical quality assurance, integration tests, or release checks had passed.

### 1.1 Canonical Release Quality Model
```text
RELEASE_READY =
    TECHNICAL_CORRECTNESS
AND DATA_CORRECTNESS
AND SEMANTIC_CORRECTNESS
AND RENDERED_JOURNEY_CORRECTNESS

PRODUCTION_RELEASE_COMPLETE =
    DEPLOYED
AND PRODUCTION_VERIFIED
AND RELEASE_NOTES_CREATED
AND RELEASE_NOTES_COMMITTED
```

### 1.2 The Escape Conversion Invariant: Defect Fixed vs. Test Exists
Adding a regression test is **necessary but not sufficient** for closure.
This registry strictly decouples four independent lifecycle states for each escape:
1. `DEFECT_STATUS`: Whether the code defect has been remediated (`PREVIOUSLY_REMEDIATED`, `REMEDIATED_IN_SEPARATE_CHANGE`, `OPEN`, `NOT_A_DEFECT`, `INSUFFICIENT_EVIDENCE`).
2. `REGRESSION_COVERAGE_STATUS`: Whether automated regression tests exist and pass (`NOT_COVERED`, `REGRESSION_ADDED`, `REGRESSION_VERIFIED`).
3. `PREVENTION_CONTROL_STATUS`: Whether architectural gates enforce prevention (`ENFORCED`, `PENDING`).
4. `PRODUCTION_VERIFICATION_STATUS`: Whether the remediation is verified on live production infrastructure (`VERIFIED`, `NOT_VERIFIED`, `NOT_APPLICABLE`).

---

## 2. Detection Failure Taxonomy

Every escape is classified according to its primary detection failure mechanisms:
* `UNIT_COVERAGE_GAP`: Specific component or function logic unexercised by unit tests.
* `CONTRACT_COVERAGE_GAP`: API schema, serialization type, or cross-boundary payload mismatch unverified.
* `INTEGRATION_COVERAGE_GAP`: Independent units function in isolation but break when composed together.
* `PRODUCTION_DATA_GAP`: Real market data distributions, volume zeroes, or multi-feed dynamics not represented.
* `FIXTURE_REALISM_GAP`: Test fixtures relied on over-simplified, synthetic, or happy-path static data.
* `SEMANTIC_ASSERTION_GAP`: Tests only asserted presence (`toBeDefined()`), not meaning, direction, or non-contradiction.
* `VISUAL_RENDERING_GAP`: Headless or node-rendered HTML succeeded while browser layout engine clipped or hid elements.
* `END_TO_END_JOURNEY_GAP`: Multi-step user workflow (e.g. click -> navigate -> reload) never simulated sequentially.
* `EMPTY_STATE_GAP`: Incomplete, pending, or empty states treated as runtime failures or collapsed into zeroes.
* `PROVENANCE_GAP`: Stale, cached, or synthetic data rendered without required provenance metadata or badges.
* `OBSERVABILITY_GAP`: Production metrics lacked telemetry or structured logs to detect anomalies passively.
* `RELEASE_SMOKE_GAP`: Pre-promotion verification failed to exercise critical user journeys against deployed candidates.
* `HUMAN_ACCEPTANCE_GAP`: Discrepancy between developer technical specification and human decision-support expectations.
* `TEST_ORACLE_WEAKNESS`: Test assertion checked a tautology (e.g. asserted code output matches its own unverified formula).

---

## 3. Cumulative Production Escape Register

### QA-ESC-001: Quick Tour Skip Persistence & Auto-Open State Bleed
* **ESCAPE_ID**: `QA-ESC-001`
* **DATE**: `2026-10-08`
* **PRODUCTION_SYMPTOM**: Quick Tour modal reappeared automatically on page refresh, route navigation, or next visit after user clicked "Skip".
* **ROOT_CAUSE**: `onClose` handler closed the visual modal state (`setIsOnboardingOpen(false)`) without updating `localStorage`, leaving `FINANCE_ONBOARDING_COMPLETED` unset; unmanaged background `setTimeout` fired asynchronously.
* **DEFECT_STATUS**: `PREVIOUSLY_REMEDIATED`
* **FIX_COMMIT**: `3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb`
* **FIX_FILE**: `frontend/components/OnboardingTourModal.tsx`, `frontend/components/Navbar.tsx`
* **CURRENT_MAIN_CONTAINS_FIX**: `YES` (Commit `3ae385c` is in canonical `main` history)
* **REGRESSION_TEST**: `frontend/components/__tests__/OnboardingTourPersistence.test.tsx` + `frontend/tests/qaEscapeSemanticInvariants.test.ts`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `VERIFIED` (Verified live on `https://www.arxterminal.com`)
* **REMAINING_ACTION**: `NONE`

---

### QA-ESC-002: Mobile Overflow Menu iOS WebKit CoreAnimation Hardware Clipping
* **ESCAPE_ID**: `QA-ESC-002`
* **DATE**: `2026-10-07`
* **PRODUCTION_SYMPTOM**: On physical iPhone (iOS Safari), tapping the `...` overflow navigation button caused the drawer to flicker, close, or remain invisible.
* **ROOT_CAUSE**: Hardware layer compositing clipping in WebKit CoreAnimation engine (`masksToBounds`) caused by ancestor `<header>` combining `overflow-x: clip` and `backdrop-filter: blur()`, plus touch-outside listeners emitting `relatedTarget === null` on blur.
* **DEFECT_STATUS**: `PREVIOUSLY_REMEDIATED`
* **FIX_COMMIT**: `2924391` (preceded by `8a32366`, `a10476a`)
* **FIX_FILE**: `frontend/components/Navbar.tsx`
* **CURRENT_MAIN_CONTAINS_FIX**: `YES` (Commit `2924391` is in canonical `main` history)
* **REGRESSION_TEST**: `frontend/components/__tests__/MobileNavbarOverflowMenu.test.tsx` + `frontend/tests/qaEscapeSemanticInvariants.test.ts`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `VERIFIED` (Automated WebKit & Chromium production verification); `PHYSICAL_IOS_RENDERING: NOT_VERIFIED` (Requires physical device check on next deploy)
* **REMAINING_ACTION**: `PHYSICAL_DEVICE_HUMAN_VERIFICATION_ON_NEXT_DEPLOY`

---

### QA-ESC-003: Smart Money & VCP Radar Universe Screening Epistemic Collapsing
* **ESCAPE_ID**: `QA-ESC-003`
* **DATE**: `2026-10-07`
* **PRODUCTION_SYMPTOM**: On Radar surface, switching to "Smart Money" or "Minervini VCP" tabs showed empty tables or "0 candidates found", falsely implying market scans had completed and found zero matching setups.
* **ROOT_CAUSE**: Epistemic state collapse violating Domain Invariant `ARX_INV_001` and PRD Addendum 001 Section 1.2: collapsing `PIPELINE_PENDING` into `EMPTY_RESULT` / `ZERO`.
* **DEFECT_STATUS**: `PREVIOUSLY_REMEDIATED`
* **FIX_COMMIT**: `d194c15`, `9d5fc2b`
* **FIX_FILE**: `frontend/components/TerminalMarketRadar.tsx`, `frontend/components/RadarView.tsx`
* **CURRENT_MAIN_CONTAINS_FIX**: `YES`
* **REGRESSION_TEST**: `frontend/tests/preflightRadarCopyRemediation.test.ts` + `frontend/tests/qaEscapeSemanticInvariants.test.ts` + `tests/test_qa_escape_invariants.py`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `VERIFIED`
* **REMAINING_ACTION**: `NONE`

---

### QA-ESC-004: Pre-Flight Retail Emotional Copy Leakage & False Revocation Alarmism
* **ESCAPE_ID**: `QA-ESC-004`
* **DATE**: `2026-10-06`
* **PRODUCTION_SYMPTOM**: Emotive warnings ("hard-earned money") and alarmist phrases ("FLIGHT CLEARANCE REVOKED") appeared on normal prospective setups forming healthy bases.
* **ROOT_CAUSE**: Unvetted copy written without institutional voice guidelines or semantic posture alignment; treated normal setup wait states as punitive failures.
* **DEFECT_STATUS**: `PREVIOUSLY_REMEDIATED`
* **FIX_COMMIT**: `d194c15`
* **FIX_FILE**: `frontend/components/PreFlightChecklistModal.tsx`
* **CURRENT_MAIN_CONTAINS_FIX**: `YES`
* **REGRESSION_TEST**: `frontend/tests/preflightRadarCopyRemediation.test.ts` + `frontend/tests/qaEscapeSemanticInvariants.test.ts`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `VERIFIED`
* **REMAINING_ACTION**: `NONE`

---

### QA-ESC-005: Execution Ladder Prospective Target-State Conflation & Card/Chart Drift
* **ESCAPE_ID**: `QA-ESC-005`
* **DATE**: `2026-10-07`
* **PRODUCTION_SYMPTOM**: Prospective un-entered assets (e.g. NAUT, spot \$1.96, planned entry \$1.46, TP1 \$1.89) displayed `TARGET_REACHED` on execution ladder card while simulation summary reported `WAITING_PULLBACK`; long-horizon targets showed TP1 $\ge$ TP2 drift.
* **ROOT_CAUSE**: Status precedence inversion in `OptimalExecutionEngine._enforce_execution_invariants` evaluating `eval_price >= take_profit_1` before extension threshold, conflating spatial market location with active entered trade outcomes.
* **DEFECT_STATUS**: `PREVIOUSLY_REMEDIATED`
* **FIX_COMMIT**: `7bcb7780221f58cf596dabce484d83276e0a3c50`
* **FIX_FILE**: `analyst_dashboard/analyzers/optimal_execution.py`, `frontend/components/OptimalEntryExitCard.tsx`
* **CURRENT_MAIN_CONTAINS_FIX**: `YES` (Commit `7bcb778` is in canonical `main` history)
* **REGRESSION_TEST**: `tests/analyzers/test_execution_ladder_remediation.py` + `tests/test_qa_escape_invariants.py` + `frontend/tests/qaEscapeSemanticInvariants.test.ts`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `VERIFIED`
* **REMAINING_ACTION**: `NONE`

---

### QA-ESC-006: Synthetic MiniSparkline Fabrication (Rule D04)
* **ESCAPE_ID**: `QA-ESC-006`
* **DATE**: `2026-10-06`
* **PRODUCTION_SYMPTOM**: Table rows for instruments with missing or unobserved price series rendered smooth synthetic upward or linear sparklines instead of indicating missing data.
* **ROOT_CAUSE**: Defensive UI programming in `MiniSparkline.tsx` that fabricated array `[10, 11, 12, ...]` to avoid SVG NaN errors when data was missing.
* **DEFECT_STATUS**: `PREVIOUSLY_REMEDIATED`
* **FIX_COMMIT**: `9d5fc2b`
* **FIX_FILE**: `frontend/components/MiniSparkline.tsx`
* **CURRENT_MAIN_CONTAINS_FIX**: `YES`
* **REGRESSION_TEST**: `frontend/tests/wave3DecisionIntegrity.test.ts` + `frontend/tests/qaEscapeSemanticInvariants.test.ts`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `VERIFIED`
* **REMAINING_ACTION**: `NONE`

---

### QA-ESC-007: Closed-Market / Weekend Stale Tape Realtime Masquerading
* **ESCAPE_ID**: `QA-ESC-007`
* **DATE**: `2026-10-05`
* **PRODUCTION_SYMPTOM**: On weekends and outside exchange trading hours, setups either disappeared completely or displayed Friday settlement quotes with "LIVE REALTIME" badges.
* **ROOT_CAUSE**: Conflation of HTTP 200 delivery with market session freshness; client defaulted timestamp to `new Date()`.
* **DEFECT_STATUS**: `PREVIOUSLY_REMEDIATED`
* **FIX_COMMIT**: `49d5d5a`, `9d5fc2b`
* **FIX_FILE**: `frontend/components/nav/MarketCommandRibbon.tsx`, `api/routes/analytics.py`
* **CURRENT_MAIN_CONTAINS_FIX**: `YES`
* **REGRESSION_TEST**: `tests/test_price_provenance_truthfulness.py` + `tests/test_qa_escape_invariants.py`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `VERIFIED`
* **REMAINING_ACTION**: `NONE`

---

### QA-ESC-008: Generic Security Master Monolithic Filing Disqualification (ETF/ADR 10-K)
* **ESCAPE_ID**: `QA-ESC-008`
* **DATE**: `2026-10-06`
* **PRODUCTION_SYMPTOM**: Major ETFs (SPY, QQQ) and foreign ADRs (TSM) were disqualified with error `"Core SEC Form 10-Q/10-K financial filings are unverified"`.
* **ROOT_CAUSE**: Monolithic evidence evaluation failing to route evidence requirements through instrument regulatory taxonomy (1940 Act funds vs 1934 Act equities).
* **DEFECT_STATUS**: `PREVIOUSLY_REMEDIATED`
* **FIX_COMMIT**: `9d5fc2b`
* **FIX_FILE**: `analyst_dashboard/security_master/applicability.py`, `analyst_dashboard/analyzers/decision_hierarchy.py`
* **CURRENT_MAIN_CONTAINS_FIX**: `YES`
* **REGRESSION_TEST**: `tests/test_wave3_decision_integrity.py` + `tests/test_qa_escape_invariants.py` + `frontend/tests/qaEscapeSemanticInvariants.test.ts`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `VERIFIED`
* **REMAINING_ACTION**: `NONE`

---

### QA-ESC-009: Unchecked Directional Sizing for Spot-Only Long Asset Engine
* **ESCAPE_ID**: `QA-ESC-009`
* **DATE**: `2026-10-04`
* **PRODUCTION_SYMPTOM**: If a short trade setup was requested or encountered, position sizing emitted positive profit targets above entry and inverted risk allocations on an engine strictly configured for long equity spot execution.
* **ROOT_CAUSE**: Unasserted assumption of long geometry without fail-closed precondition check in sizing engine.
* **DEFECT_STATUS**: `PREVIOUSLY_REMEDIATED`
* **FIX_COMMIT**: `443c70d77cdb31fb3a17c88eb032a77a3ebd1c02`
* **FIX_FILE**: `frontend/components/DayTraderPositionSizer.tsx`, `frontend/lib/decisionHierarchyUtils.ts`
* **CURRENT_MAIN_CONTAINS_FIX**: `YES`
* **REGRESSION_TEST**: `frontend/components/__tests__/AnalysisDecisionHierarchy.test.tsx` (`INV-ANALYSIS-07`) + `frontend/tests/governorSizingEngine.test.ts` + `tests/test_qa_escape_invariants.py`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `VERIFIED`
* **REMAINING_ACTION**: `NONE`

---

### QA-ESC-010: Indefinite In-Memory Ticker Cache Map Without TTL Invalidation
* **ESCAPE_ID**: `QA-ESC-010`
* **DATE**: `2026-10-05`
* **PRODUCTION_SYMPTOM**: Navigating away from a ticker (e.g. AAPL $\to$ TSLA $\to$ AAPL) during active trading hours served stale prices from the initial visit hours earlier.
* **ROOT_CAUSE**: In-memory `cacheRef` Map in `app/page.tsx` stored API responses indefinitely without TTL, timestamp invalidation, or session awareness.
* **DEFECT_STATUS**: `PREVIOUSLY_REMEDIATED`
* **FIX_COMMIT**: `b89b3586bfc4a632e335ee8eb47ae0f56ebd1065`, `0b7dedaf11b2734ba075d10633f751388291473f`
* **FIX_FILE**: `frontend/app/page.tsx`, `frontend/components/Navbar.tsx`
* **CURRENT_MAIN_CONTAINS_FIX**: `YES`
* **REGRESSION_TEST**: `frontend/tests/marketDataProvenance.test.ts`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `VERIFIED`
* **REMAINING_ACTION**: `NONE`

---

### QA-ESC-011: Analytical Verdict Decision-Surface Multi-Authority Duplication & Contradictory Actionability Collapse
* **ESCAPE_ID**: `QA-ESC-011`
* **DATE**: `2026-10-08`
* **PRODUCTION_SYMPTOM**: On ARX Analytical Verdict card (notably observed on NAUT production screenshot), the same decision state ("Wait for Trigger") rendered three times (Headline, Amber Badge, Secondary Grey Badge). On `AVOID` setups, `OWNED`/`HOLD` positions, and `UNVERIFIED` assets, non-actionable state collapsed to `"WAIT FOR TRIGGER"`, presenting contradictory guidance.
* **ROOT_CAUSE**:
  1. Backend-frontend contract misalignment: `DecisionHierarchyEngine` emitted `decisionStateLabel`, while frontend `assessmentEngine.ts` read `decisionTrace?.stateLabel`, causing fallback to generic `"Wait for Trigger"`.
  2. Generic fallback on missing state: `assessmentEngine.ts` line 216 defaulted undefined `decisionTrace` to `"Wait for Trigger"` rather than failing closed to an explicit neutral/unassessed condition (`"Setup Evaluation Pending"`).
  3. Actionability-trigger presentation collapse: `StandardTerminalView`, `GuidedTerminalView`, and `AdvancedTerminalView` hardcoded `{isActionable ? "ACTIONABLE" : "WAIT FOR TRIGGER"}`, falsely equating `!isActionable` with awaiting a trade trigger.
  4. Redundant secondary badge: Terminal views rendered `{insight.verdictLabel}` as headline and `{insight.terminalState.uiStateLabel}` as secondary grey badge, repeating identical text when `verdictLabel` was mapped to `uiStateLabel`.
* **DEFECT_STATUS**: `REMEDIATED_PENDING_PRODUCTION_VERIFICATION`
* **FIX_COMMIT**: `LOCAL_CANDIDATE`
* **FIX_FILE**: `frontend/types/insight.ts`, `frontend/lib/assessmentEngine.ts`, `frontend/lib/insightGenerator.ts`, `frontend/components/terminal/StandardTerminalView.tsx`, `frontend/components/terminal/GuidedTerminalView.tsx`, `frontend/components/terminal/AdvancedTerminalView.tsx`
* **CURRENT_MAIN_CONTAINS_FIX**: `PENDING_INTEGRATION`
* **REGRESSION_TEST**: `frontend/components/__tests__/DecisionSurfaceIntegrity.test.tsx`, `frontend/components/__tests__/AnalysisDecisionHierarchy.test.tsx`
* **REGRESSION_COVERAGE_STATUS**: `REGRESSION_VERIFIED`
* **PREVENTION_CONTROL_STATUS**: `ENFORCED`
* **PRODUCTION_VERIFICATION_STATUS**: `NOT_YET_VERIFIED_IN_PRODUCTION`
* **REMAINING_ACTION**: `DEPLOY_AND_VERIFY_PRODUCTION_DECISION_SURFACE`
