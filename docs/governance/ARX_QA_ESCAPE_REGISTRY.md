# ARX TERMINAL — PRODUCTION QA ESCAPE REGISTRY

```text
DOCUMENT_TYPE =
  CANONICAL_GOVERNANCE_REGISTRY
AUTHORITY =
  PRODUCT_GOVERNANCE + QA_SYSTEM_ARCHITECTURE + RELEASE_MANAGEMENT
STATUS =
  ACTIVE_AND_CUMULATIVE
LOCATION =
  docs/governance/ARX_QA_ESCAPE_REGISTRY.md
ESTABLISHED =
  2026-10-08
CANONICAL_INTEGRATION_BASE =
  3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb
```

---

## 1. Executive Purpose & Governance Boundary

The **ARX Production QA Escape Registry** is the authoritative, permanent ledger of all confirmed or suspected defects, product-quality escapes, and epistemic contradictions discovered in production after technical quality assurance, integration tests, or release checks had passed.

### 1.1 Core Epistemic & Release Invariant
```text
RELEASE_READY =
    TECHNICAL_CORRECTNESS
AND DATA_CORRECTNESS
AND SEMANTIC_CORRECTNESS
AND RENDERED_JOURNEY_CORRECTNESS

PRODUCTION_RELEASE_COMPLETE =
    DEPLOYED
AND PRODUCTION_VERIFIED
```

### 1.2 The Escape Conversion Invariant
No escape entry may be closed merely because a code patch was deployed. Closure strictly requires:
1. **Known Root Cause**: Forensic mechanism established with high confidence.
2. **Detection Failure Accounting**: Explicit documentation of why existing tests missed it.
3. **Missing Layer Identification**: Architecture level identified (unit, contract, fixture, semantic, rendered journey, release smoke).
4. **Permanent Regression Coverage**: Durable automated tests operating at the lowest appropriate test layer.
5. **Prevention Control**: Gatekeeper rule preventing repetition of the entire defect class.

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
* **USER_VISIBLE_SYMPTOM**: User clicks "Skip" on Quick Tour modal; on reload, route change, or next visit, Quick Tour re-opens automatically.
* **AFFECTED_SURFACE**: `OnboardingTourModal.tsx`, `Navbar.tsx`, Global App Shell
* **AFFECTED_TICKER_OR_CONTEXT**: Universal client app shell, all routes
* **FIRST_KNOWN_PRODUCTION_OBSERVATION**: User observed Quick Tour modal repeatedly opening after dismissal on production domain `https://www.arxterminal.com`.
* **EXPECTED_BEHAVIOR**: Clicking "Skip", "Cancel", or close button must permanently record completion in `localStorage` (`FINANCE_ONBOARDING_COMPLETED = "true"`) and clear all pending auto-open timeouts, suppressing automatic reappearance across all sessions unless deliberately triggered via "Quick Tour" menu item.
* **ACTUAL_BEHAVIOR**: `OnboardingTourModal.tsx` `onClose` handler closed the visual modal state (`setIsOnboardingOpen(false)`) without updating `localStorage`, leaving `FINANCE_ONBOARDING_COMPLETED` unset. Additionally, an unmanaged background timer fired asynchronously, re-triggering the tour.
* **ROOT_CAUSE**: Architectural omission in modal dismiss handler: split-brain between component local visibility state and persistent storage layer; failure to cancel unmounted timeout references.
* **ROOT_CAUSE_CONFIDENCE**: `HIGH`
* **CLASSIFICATION**: `CONFIRMED_UI_STATE_DEFECT`
* **EXISTING_TESTS_THAT_SHOULD_HAVE_CAUGHT_IT**: `frontend/components/__tests__/OnboardingTourPersistence.test.tsx`
* **WHY_EXISTING_TESTS_DID_NOT_CATCH_IT**: Prior tests only asserted that when `localStorage` has `"true"`, modal does not render, and when absent, modal mounts. Tests never executed the interactive user click journey of "Skip" followed by remount/reload verification.
* **DETECTION_FAILURE_CLASSIFICATION**: `END_TO_END_JOURNEY_GAP`, `TEST_ORACLE_WEAKNESS`
* **TEST_LAYER_MISSING**: Component interaction test simulating button click $\to$ storage write $\to$ timer cancellation $\to$ remount.
* **RELEASE_GATE_MISSING**: Production-Candidate Onboarding State Verification Gate.
* **OBSERVABILITY_MISSING**: Telemetry event on tour dismissal recording storage success.
* **CODE_FIX_STATUS**: `REMEDIATED`
* **FIX_COMMIT**: `3ae385c7d9b338d0dde96e0e0d6ecfebfc36debb`
* **REGRESSION_TEST**: `frontend/components/__tests__/OnboardingTourPersistence.test.tsx`
* **REGRESSION_TEST_STATUS**: `VERIFIED_PASS`
* **PREVENTION_CONTROL**: All persistent client preference toggles must have automated tests verifying both write-on-action and read-on-subsequent-mount contracts.
* **PREVENTION_STATUS**: `ENFORCED`
* **STATUS**: `CLOSED`

---

### QA-ESC-002: Mobile Overflow Menu iOS WebKit CoreAnimation Hardware Clipping
* **ESCAPE_ID**: `QA-ESC-002`
* **DATE**: `2026-10-07`
* **USER_VISIBLE_SYMPTOM**: On physical iPhone (iOS Safari), tapping the `...` overflow navigation button caused the menu to either not open, flicker instantaneously, or remain invisible.
* **AFFECTED_SURFACE**: `Navbar.tsx`, `layout.tsx` Header & Viewport Shell
* **AFFECTED_TICKER_OR_CONTEXT**: Mobile viewports ($\le 768\text{px}$), physical iOS WebKit
* **FIRST_KNOWN_PRODUCTION_OBSERVATION**: User manual test on physical iPhone 13 Pro (iOS 17 Safari) on `https://www.arxterminal.com`.
* **EXPECTED_BEHAVIOR**: Tapping `...` opens the navigation drawer; drawer renders fully on top of all page content, respects safe area insets, and remains open until explicit item selection or outside dismissal.
* **ACTUAL_BEHAVIOR**: Headless Chromium and Playwright tests passed completely. On physical iOS devices, ancestor `<header>` possessed `overflow-x: clip` combined with `backdrop-filter: blur(12px)`. WebKit's CoreAnimation hardware layer compositing triggered `masksToBounds` hardware clipping, slicing the portalless dropdown below the header boundary. In parallel, touch-outside listeners lost focus because WebKit emitted `relatedTarget === null` on blur.
* **ROOT_CAUSE**: Hardware layer compositing clipping in WebKit CoreAnimation engine combined with touch-to-mouse synthetic event sequence differences.
* **ROOT_CAUSE_CONFIDENCE**: `HIGH`
* **CLASSIFICATION**: `CONFIRMED_UI_STATE_DEFECT`
* **EXISTING_TESTS_THAT_SHOULD_HAVE_CAUGHT_IT**: `frontend/components/__tests__/MobileNavbarOverflowMenu.test.tsx`
* **WHY_EXISTING_TESTS_DID_NOT_CATCH_IT**: Tests ran in Chromium desktop and simulated mobile emulation. Chromium handles CSS `clip` and `backdrop-filter` compositing differently than Apple WebKit CoreAnimation; synthetic mouse events did not reproduce iOS touch event blur timing.
* **DETECTION_FAILURE_CLASSIFICATION**: `VISUAL_RENDERING_GAP`, `FIXTURE_REALISM_GAP`
* **TEST_LAYER_MISSING**: WebKit device-runtime rendering and layout geometry boundary test.
* **RELEASE_GATE_MISSING**: Target-Platform Physical / WebKit Rendering Pre-Flight Gate.
* **OBSERVABILITY_MISSING**: Client-side interaction telemetry measuring menu open/close duration.
* **CODE_FIX_STATUS**: `REMEDIATED`
* **FIX_COMMIT**: `2924391` (preceded by `8a32366`, `a10476a`)
* **REGRESSION_TEST**: `frontend/components/__tests__/MobileNavbarOverflowMenu.test.tsx` + WebKit automated journey suite
* **REGRESSION_TEST_STATUS**: `VERIFIED_PASS`
* **PREVENTION_CONTROL**: All dropdown menus and fixed overlays must be rendered via fixed portals or have explicit overflow-escape geometry, tested against WebKit layout engines.
* **PREVENTION_STATUS**: `ENFORCED`
* **STATUS**: `CLOSED`

---

### QA-ESC-003: Smart Money & VCP Radar Universe Screening Epistemic Collapsing
* **ESCAPE_ID**: `QA-ESC-003`
* **DATE**: `2026-10-07`
* **USER_VISIBLE_SYMPTOM**: On Radar surface, switching to "Smart Money" or "Minervini VCP" tabs showed empty tables or "0 candidates found", falsely implying market scans had completed and found zero matching setups.
* **AFFECTED_SURFACE**: `TerminalMarketRadar.tsx`, `RadarView.tsx`, Discovery Dashboard
* **AFFECTED_TICKER_OR_CONTEXT**: Market-wide screener views across all tickers
* **FIRST_KNOWN_PRODUCTION_OBSERVATION**: Production UX audit of discovery tabs on live terminal.
* **EXPECTED_BEHAVIOR**: When market-wide scanning is not yet implemented or pending background batch worker deployment, the interface must explicitly communicate capability status: `"⚡ Minervini VCP Universe Scanner Pending"` or `"🐋 Smart Money Universe Scanner Pending"`. It must never collapse `PIPELINE_PENDING` into `0 candidates` or an empty error box.
* **ACTUAL_BEHAVIOR**: Frontend components initialized tabs with empty arrays (`items = []`), rendering the default empty-filter state ("No candidates match your filters") when the background scanner had not yet executed.
* **ROOT_CAUSE**: Epistemic state collapse violating Domain Invariant `ARX_INV_001` and PRD Addendum 001 Section 1.2: collapsing `PIPELINE_PENDING` into `EMPTY_RESULT` / `ZERO`.
* **ROOT_CAUSE_CONFIDENCE**: `HIGH`
* **CLASSIFICATION**: `CONFIRMED_SEMANTIC_QUALITY_DEFECT` / `EXPECTED_STATE_POORLY_COMMUNICATED`
* **EXISTING_TESTS_THAT_SHOULD_HAVE_CAUGHT_IT**: `frontend/tests/preflightRadarCopyRemediation.test.ts`, `tests/test_radar_taxonomy_remediation.py`
* **WHY_EXISTING_TESTS_DID_NOT_CATCH_IT**: Existing tests provided mock lists of 5 symbols, asserting that rows rendered correctly. Zero tests validated the unpopulated / pending pipeline state contract.
* **DETECTION_FAILURE_CLASSIFICATION**: `EMPTY_STATE_GAP`, `SEMANTIC_ASSERTION_GAP`, `FIXTURE_REALISM_GAP`
* **TEST_LAYER_MISSING**: Semantic state contract assertions enforcing `ZERO != EMPTY_RESULT != PIPELINE_PENDING`.
* **RELEASE_GATE_MISSING**: Production-Candidate Empty State Audit Gate.
* **OBSERVABILITY_MISSING**: Radar scanner execution state telemetry.
* **CODE_FIX_STATUS**: `REMEDIATED`
* **FIX_COMMIT**: `d194c15` + `9d5fc2b`
* **REGRESSION_TEST**: `frontend/tests/preflightRadarCopyRemediation.test.ts`, `frontend/tests/wave3DecisionIntegrity.test.ts`
* **REGRESSION_TEST_STATUS**: `VERIFIED_PASS`
* **PREVENTION_CONTROL**: Mandatory empty-state triad auditing: all list surfaces must define and test `LOADING`, `EMPTY_RESULT`, `PIPELINE_PENDING`, and `ERROR` distinct visual contracts.
* **PREVENTION_STATUS**: `ENFORCED`
* **STATUS**: `CLOSED`

---

### QA-ESC-004: Pre-Flight Retail Emotional Copy Leakage & False Revocation Alarmism
* **ESCAPE_ID**: `QA-ESC-004`
* **DATE**: `2026-10-06`
* **USER_VISIBLE_SYMPTOM**: Pre-Flight checklist banner displayed emotive, retail-style warning copy ("Protect your hard-earned money") and alarmist phrases ("FLIGHT CLEARANCE REVOKED") when an asset was merely in a normal prospective setup stage awaiting trigger confirmation.
* **AFFECTED_SURFACE**: `PreFlightChecklistModal.tsx`, `OptimalEntryExitCard.tsx`
* **AFFECTED_TICKER_OR_CONTEXT**: All tickers evaluated in non-actionable setup states (e.g. `VALID_SETUP`, `WAITING_PULLBACK`)
* **FIRST_KNOWN_PRODUCTION_OBSERVATION**: User review of Pre-Flight modal on non-cleared assets.
* **EXPECTED_BEHAVIOR**: Institutional decision-support voice: objective, calm, factual, non-patronizing. A non-actionable state is a normal stage in setup development, not a moral failure or emergency. It must display `TRADE NOT CLEARED: AWAITING CONFIRMATION` with clear risk-hygiene framing.
* **ACTUAL_BEHAVIOR**: Modal rendered emotional strings and conflated lack of immediate trigger confirmation with an active flight revocation emergency.
* **ROOT_CAUSE**: Unreviewed draft UI copy written without institutional voice guidelines or semantic posture alignment.
* **ROOT_CAUSE_CONFIDENCE**: `HIGH`
* **CLASSIFICATION**: `CONFIRMED_SEMANTIC_QUALITY_DEFECT`
* **EXISTING_TESTS_THAT_SHOULD_HAVE_CAUGHT_IT**: `frontend/components/__tests__/AnalysisDecisionHierarchy.test.tsx`
* **WHY_EXISTING_TESTS_DID_NOT_CATCH_IT**: Tests checked `expect(screen.getByText(/Pre-Flight/i)).toBeInTheDocument()` and validated schema shape. They did not validate semantic copy invariants, lexicon guidelines, or posture alignment.
* **DETECTION_FAILURE_CLASSIFICATION**: `SEMANTIC_ASSERTION_GAP`, `TEST_ORACLE_WEAKNESS`
* **TEST_LAYER_MISSING**: Semantic copy invariant test checking institutional terminology and forbidding emotive buzzwords.
* **RELEASE_GATE_MISSING**: Brand & Decision-Support Semantic Gate.
* **OBSERVABILITY_MISSING**: Static analysis linter for prohibited strings.
* **CODE_FIX_STATUS**: `REMEDIATED`
* **FIX_COMMIT**: `d194c15`
* **REGRESSION_TEST**: `frontend/tests/preflightRadarCopyRemediation.test.ts`
* **REGRESSION_TEST_STATUS**: `VERIFIED_PASS`
* **PREVENTION_CONTROL**: Automated regex scan across all modal and card templates checking against `PROHIBITED_COPY_PATTERNS`.
* **PREVENTION_STATUS**: `ENFORCED`
* **STATUS**: `CLOSED`

---

### QA-ESC-005: Execution Ladder Prospective Target-State Conflation & Card/Chart Drift
* **ESCAPE_ID**: `QA-ESC-005`
* **DATE**: `2026-10-07`
* **USER_VISIBLE_SYMPTOM**: Prospective, un-entered assets (e.g. NAUT, spot \$1.96, planned entry \$1.46, TP1 \$1.89) displayed `TARGET_REACHED` on the execution ladder card, while the technical chart and simulation engine simultaneously reported `WAITING_PULLBACK`. In parallel, long-horizon setups showed TP1 $\ge$ TP2 anomalies.
* **AFFECTED_SURFACE**: `OptimalEntryExitCard.tsx`, `analyst_dashboard/analyzers/optimal_execution.py`, TradingView Chart Overlays
* **AFFECTED_TICKER_OR_CONTEXT**: Extended momentum assets with no open position (e.g. NAUT)
* **FIRST_KNOWN_PRODUCTION_OBSERVATION**: User observed NAUT showing contradictory status labels between execution card and simulation summary.
* **EXPECTED_BEHAVIOR**: For prospective setups where the user holds no active position (`NO_ACTIVE_POSITION`), the status must represent **entry readiness** (`WAITING_PULLBACK` or `EXTENDED_ABOVE_BUY_ZONE`). `TARGET_REACHED` is an outcome state of an active, entered trade position and must never be emitted on prospective scans. Ladder targets must obey $\text{Stop} < \text{Entry} < \text{TP1} < \text{TP2}$.
* **ACTUAL_BEHAVIOR**: Status determination evaluated `eval_price >= take_profit_1` before checking extension threshold, conflating spatial market location with active position outcome.
* **ROOT_CAUSE**: Status precedence inversion and failure to decouple the 4 orthogonal dimensions: Position Lifecycle, Entry Readiness, Market Location, and Target Progress.
* **ROOT_CAUSE_CONFIDENCE**: `HIGH`
* **CLASSIFICATION**: `CONFIRMED_PRODUCT_DEFECT`
* **EXISTING_TESTS_THAT_SHOULD_HAVE_CAUGHT_IT**: `tests/test_optimal_execution.py`
* **WHY_EXISTING_TESTS_DID_NOT_CATCH_IT**: Tests only provided happy-path setups where spot price was inside the buy zone, never asserting status precedence on extended assets where spot exceeded prospective target levels.
* **DETECTION_FAILURE_CLASSIFICATION**: `FIXTURE_REALISM_GAP`, `SEMANTIC_ASSERTION_GAP`
* **TEST_LAYER_MISSING**: Boundary value test for extended assets evaluating entry readiness vs target attainment.
* **RELEASE_GATE_MISSING**: Cross-Surface State Consistency Gate.
* **OBSERVABILITY_MISSING**: Telemetry alerting on card vs chart status divergence.
* **CODE_FIX_STATUS**: `REMEDIATED`
* **FIX_COMMIT**: `7bcb778` (preceded by `3896464`, `64a080d`)
* **REGRESSION_TEST**: `tests/analyzers/test_execution_ladder_remediation.py`
* **REGRESSION_TEST_STATUS**: `VERIFIED_PASS`
* **PREVENTION_CONTROL**: Strict mathematical invariant enforcement: $\text{eval\_price} > \text{ext\_threshold} \implies \text{WAITING\_PULLBACK}$, and $\text{TP2} > \text{TP1} > \text{Entry} > \text{Stop}$ for long equity.
* **PREVENTION_STATUS**: `ENFORCED`
* **STATUS**: `CLOSED`

---

### QA-ESC-006: Synthetic MiniSparkline Fabrication on Missing Series (D04 Invariant)
* **ESCAPE_ID**: `QA-ESC-006`
* **DATE**: `2026-10-06`
* **USER_VISIBLE_SYMPTOM**: Table rows for instruments with missing or unobserved price series rendered smooth synthetic upward or linear sparklines instead of indicating missing data.
* **AFFECTED_SURFACE**: `MiniSparkline.tsx`, Discovery Tables, Watchlist
* **AFFECTED_TICKER_OR_CONTEXT**: Newly listed securities or feed-interrupted symbols
* **FIRST_KNOWN_PRODUCTION_OBSERVATION**: Forensic audit of visual evidence components during Wave 1/2 declutter.
* **EXPECTED_BEHAVIOR**: In accordance with `ARX_INV_002` (Synthetic evidence never becomes natural evidence) and Rule D04, missing price history must render an honest placeholder: `—` or empty state. It must never fabricate synthetic curves.
* **ACTUAL_BEHAVIOR**: Component contained fallback array `[10, 11, 12, ...]` to prevent SVG `<path>` rendering exceptions when `data` was null or length was 0.
* **ROOT_CAUSE**: Defensive UI programming that prioritized avoiding SVG NaN errors over epistemic truthfulness.
* **ROOT_CAUSE_CONFIDENCE**: `HIGH`
* **CLASSIFICATION**: `CONFIRMED_SEMANTIC_QUALITY_DEFECT`
* **EXISTING_TESTS_THAT_SHOULD_HAVE_CAUGHT_IT**: `frontend/tests/radarMetricCleanup.test.ts`
* **WHY_EXISTING_TESTS_DID_NOT_CATCH_IT**: Tests checked that the component mounted and SVG element was present; tests did not assert that the rendered coordinates derived from authentic series points.
* **DETECTION_FAILURE_CLASSIFICATION**: `SEMANTIC_ASSERTION_GAP`, `TEST_ORACLE_WEAKNESS`
* **TEST_LAYER_MISSING**: Visual truthfulness invariant test verifying that `data === null || data.length === 0` renders `—` and zero SVG paths.
* **RELEASE_GATE_MISSING**: Truthful Presentation & Epistemic Audit Gate.
* **OBSERVABILITY_MISSING**: Client logging when fallbacks trigger.
* **CODE_FIX_STATUS**: `REMEDIATED`
* **FIX_COMMIT**: `9d5fc2b`
* **REGRESSION_TEST**: `frontend/tests/wave3DecisionIntegrity.test.ts`
* **REGRESSION_TEST_STATUS**: `VERIFIED_PASS`
* **PREVENTION_CONTROL**: Strict prohibition of synthetic numerical or coordinate fallbacks across all charting primitives.
* **PREVENTION_STATUS**: `ENFORCED`
* **STATUS**: `CLOSED`

---

### QA-ESC-007: Closed-Market / Weekend Stale Tape Realtime Masquerading
* **ESCAPE_ID**: `QA-ESC-007`
* **DATE**: `2026-10-05`
* **USER_VISIBLE_SYMPTOM**: On weekends and outside exchange trading hours, setups either disappeared completely or displayed Friday settlement quotes with "LIVE REALTIME" badges.
* **AFFECTED_SURFACE**: `MarketCommandRibbon.tsx`, Asset Header, Live Tape Pipeline
* **AFFECTED_TICKER_OR_CONTEXT**: All assets accessed during exchange-closed hours
* **FIRST_KNOWN_PRODUCTION_OBSERVATION**: Weekend user session observation on production domain.
* **EXPECTED_BEHAVIOR**: Pursuant to `ARX_INV_008`, closed-market quotes must be explicitly labeled `SETTLEMENT_PINNED` or `STALE` with the authentic observation timestamp. Setup screening must remain available for research rather than crashing or claiming live streaming.
* **ACTUAL_BEHAVIOR**: Market ribbon checked `status === 200` and defaulted timestamp to client clock `new Date()`, disguising closed-market quotes as live realtime ticks.
* **ROOT_CAUSE**: Conflation of HTTP delivery success with market session freshness; client-side timestamp fabrication.
* **ROOT_CAUSE_CONFIDENCE**: `HIGH`
* **CLASSIFICATION**: `CONFIRMED_DATA_PIPELINE_DEFECT`
* **EXISTING_TESTS_THAT_SHOULD_HAVE_CAUGHT_IT**: `tests/test_live_api_provenance.py`
* **WHY_EXISTING_TESTS_DID_NOT_CATCH_IT**: Automated CI tests executed during business hours or with static mock dates matching the current date, never testing off-hours and weekend timestamp deltas.
* **DETECTION_FAILURE_CLASSIFICATION**: `FIXTURE_REALISM_GAP`, `PROVENANCE_GAP`
* **TEST_LAYER_MISSING**: Time-travel provenance test verifying off-hours session labeling.
* **RELEASE_GATE_MISSING**: Off-Market Hours Verification Gate.
* **OBSERVABILITY_MISSING**: Telemetry measuring observation age vs current UTC time.
* **CODE_FIX_STATUS**: `REMEDIATED`
* **FIX_COMMIT**: `49d5d5a` + `9d5fc2b`
* **REGRESSION_TEST**: `tests/test_price_provenance_truthfulness.py`
* **REGRESSION_TEST_STATUS**: `VERIFIED_PASS`
* **PREVENTION_CONTROL**: Mandatory observation timestamp tagging and provenance badge rendering based on server exchange session calendar.
* **PREVENTION_STATUS**: `ENFORCED`
* **STATUS**: `CLOSED`

---

### QA-ESC-008: Generic Security Master Monolithic Filing Disqualification (ETF/ADR 10-K)
* **ESCAPE_ID**: `QA-ESC-008`
* **DATE**: `2026-10-06`
* **USER_VISIBLE_SYMPTOM**: Major ETFs (SPY, QQQ, XLK) and foreign ADRs (TSM, ASML) evaluated in Decision Hierarchy were disqualified with error `"Core SEC Form 10-Q/10-K financial filings are unverified"`, preventing valid technical setups from becoming actionable.
* **AFFECTED_SURFACE**: `DecisionHierarchyEngine.resolve_decision_state`, Asset Detail
* **AFFECTED_TICKER_OR_CONTEXT**: ETFs, Registered Investment Funds, Foreign ADRs
* **FIRST_KNOWN_PRODUCTION_OBSERVATION**: User observed SPY showing `EVIDENCE_INCOMPLETE` despite flawless technical setup and authentic pricing.
* **EXPECTED_BEHAVIOR**: In accordance with PRD Addendum 001 Section 5, evidence requirements must be routed through canonical security master taxonomy: ETFs are governed by the 1940 Act (Forms N-CSR/N-PORT); corporate 10-K/10-Q filings are `NOT_APPLICABLE` and must never disqualify an ETF.
* **ACTUAL_BEHAVIOR**: `DecisionHierarchyEngine` evaluated a single monolithic rule: `if not has_fundamentals: return EVIDENCE_INCOMPLETE`, treating ETFs as if they were domestic common operating companies.
* **ROOT_CAUSE**: Monolithic evidence evaluation failing to decouple instrument regulatory structures.
* **ROOT_CAUSE_CONFIDENCE**: `HIGH`
* **CLASSIFICATION**: `CONFIRMED_PRODUCT_DEFECT`
* **EXISTING_TESTS_THAT_SHOULD_HAVE_CAUGHT_IT**: `tests/test_recommendation_consistency.py`
* **WHY_EXISTING_TESTS_DID_NOT_CATCH_IT**: Test fixtures in `test_recommendation_consistency.py` only evaluated common stock tickers (AAPL, NVDA), never running full hierarchy resolution on ETF or ADR symbols.
* **DETECTION_FAILURE_CLASSIFICATION**: `FIXTURE_REALISM_GAP`, `CONTRACT_COVERAGE_GAP`
* **TEST_LAYER_MISSING**: Instrument-aware evidence applicability router test across all 5 canonical security types.
* **RELEASE_GATE_MISSING**: Multi-Asset Class Pre-Flight Release Gate.
* **OBSERVABILITY_MISSING**: Metric tracking disqualification reason by asset class.
* **CODE_FIX_STATUS**: `REMEDIATED`
* **FIX_COMMIT**: `9d5fc2b`
* **REGRESSION_TEST**: `tests/test_wave3_decision_integrity.py` (`test_etf_not_disqualified_for_missing_corporate_fundamentals`)
* **REGRESSION_TEST_STATUS**: `VERIFIED_PASS`
* **PREVENTION_CONTROL**: `get_required_evidence_for_instrument(security_type)` must be the exclusive authority for prerequisite evidence.
* **PREVENTION_STATUS**: `ENFORCED`
* **STATUS**: `CLOSED`

---

### QA-ESC-009: Unchecked Directional Sizing for Spot-Only Long Asset Engine
* **ESCAPE_ID**: `QA-ESC-009`
* **DATE**: `2026-10-04`
* **USER_VISIBLE_SYMPTOM**: If a short trade setup was requested or encountered, position sizing emitted positive profit targets above entry and inverted risk allocations on an engine strictly configured for long equity spot execution.
* **AFFECTED_SURFACE**: `governorSizingEngine.ts`, `analyst_dashboard/analyzers/optimal_execution.py`
* **AFFECTED_TICKER_OR_CONTEXT**: Short setups or inverted corridor tests
* **FIRST_KNOWN_PRODUCTION_OBSERVATION**: Latent defect discovery during quantitative sizing engine audit.
* **EXPECTED_BEHAVIOR**: ARX spot trading engine must fail closed on short trade requests, explicitly certifying `ARX_SPOT_LONG_ONLY` invariant. Sizing must never calculate shares for unsupported directional archetypes.
* **ACTUAL_BEHAVIOR**: Mathematical formulas assumed long geometry without asserting `direction === "LONG"`, producing undefined behavior if `direction === "SHORT"`.
* **ROOT_CAUSE**: Unasserted assumption of long-only trading without fail-closed precondition check.
* **ROOT_CAUSE_CONFIDENCE**: `HIGH`
* **CLASSIFICATION**: `CONFIRMED_PRODUCT_DEFECT`
* **EXISTING_TESTS_THAT_SHOULD_HAVE_CAUGHT_IT**: `frontend/tests/governorSizingEngine.test.ts`
* **WHY_EXISTING_TESTS_DID_NOT_CATCH_IT**: Unit tests only passed long trade parameter fixtures into sizing calculations.
* **DETECTION_FAILURE_CLASSIFICATION**: `FIXTURE_REALISM_GAP`, `UNIT_COVERAGE_GAP`
* **TEST_LAYER_MISSING**: Directional invariant boundary test verifying rejection of short parameters.
* **RELEASE_GATE_MISSING**: Quant Execution Boundary Gate.
* **OBSERVABILITY_MISSING**: Engine error telemetry when unsupported direction is requested.
* **CODE_FIX_STATUS**: `REMEDIATED`
* **FIX_COMMIT**: `9d5fc2b` + `7bcb778`
* **REGRESSION_TEST**: `frontend/tests/governorSizingEngine.test.ts`
* **REGRESSION_TEST_STATUS**: `VERIFIED_PASS`
* **PREVENTION_CONTROL**: Strict directional guard in all execution sizing calculations enforcing long-only spot semantics.
* **PREVENTION_STATUS**: `ENFORCED`
* **STATUS**: `CLOSED`

---

### QA-ESC-010: Indefinite In-Memory Ticker Cache Map Without TTL Invalidation
* **ESCAPE_ID**: `QA-ESC-010`
* **DATE**: `2026-10-05`
* **USER_VISIBLE_SYMPTOM**: In the main workstation, switching from AAPL to TSLA and back to AAPL during active trading hours served stale prices from the initial visit hours earlier, never refreshing from the live API.
* **AFFECTED_SURFACE**: `app/page.tsx`
* **AFFECTED_TICKER_OR_CONTEXT**: High-frequency intra-session ticker switching
* **FIRST_KNOWN_PRODUCTION_OBSERVATION**: Internal trading session audit discovering price freezing after symbol navigation.
* **EXPECTED_BEHAVIOR**: In-memory caching must enforce statutory TTL ($< 60\text{ seconds}$ intraday) or invalidate when market session ticks advance, ensuring user views fresh market quotes upon returning to an asset.
* **ACTUAL_BEHAVIOR**: In-memory `cacheRef` Map stored full API responses keyed solely on `symbol` with zero timestamp tracking, zero TTL, and no invalidation triggers.
* **ROOT_CAUSE**: Missing cache expiration policy violating `ARX_INV_008` and `ARX_INV_012`.
* **ROOT_CAUSE_CONFIDENCE**: `HIGH`
* **CLASSIFICATION**: `CONFIRMED_DATA_PIPELINE_DEFECT`
* **EXISTING_TESTS_THAT_SHOULD_HAVE_CAUGHT_IT**: `frontend/components/__tests__/ArxCockpitUxRefinement.test.tsx`
* **WHY_EXISTING_TESTS_DID_NOT_CATCH_IT**: Tests mounted the component once, checked single-render behavior, and unmounted; multi-symbol navigation over elapsed time was never simulated.
* **DETECTION_FAILURE_CLASSIFICATION**: `END_TO_END_JOURNEY_GAP`, `INTEGRATION_COVERAGE_GAP`
* **TEST_LAYER_MISSING**: Cache lifecycle and TTL expiration integration test.
* **RELEASE_GATE_MISSING**: Production Data Lifecycle Gate.
* **OBSERVABILITY_MISSING**: Cache hit/miss age telemetry.
* **CODE_FIX_STATUS**: `REMEDIATED`
* **FIX_COMMIT**: `49d5d5a` + `9d5fc2b`
* **REGRESSION_TEST**: `frontend/tests/marketDataProvenance.test.ts`
* **REGRESSION_TEST_STATUS**: `VERIFIED_PASS`
* **PREVENTION_CONTROL**: All client-side caches must include observation timestamp and enforce TTL $\le 60\text{s}$ during active market sessions.
* **PREVENTION_STATUS**: `ENFORCED`
* **STATUS**: `CLOSED`

---

## 4. Systemic QA Failure Patterns & Systemic Remedies

The analysis of QA-ESC-001 through QA-ESC-010 reveals 5 systemic failure modes in the previous QA methodology:

| Pattern | Escapes Affected | Systemic Root Cause | Mandatory Systemic Remedy |
| :--- | :--- | :--- | :--- |
| **P1: Weak Presence Assertions** | QA-ESC-003, 004, 006 | Tests asserted that elements existed in DOM (`toBeDefined()`) without asserting semantic correctness, directional truth, or absence of prohibited terms. | **Semantic Invariant Layer**: Tests must assert mathematical order, non-contradiction, and explicit string bans. |
| **P2: Fixture Realism & Happy-Path Bias** | QA-ESC-005, 007, 008, 009 | Fixtures provided rich, compliant common-stock data during live market hours, never simulating weekends, extended prices, or missing optional metrics. | **Realistic Fixture Matrix**: Standardized fixtures covering all 14 boundary conditions (weekend, ETF, extended, partial). |
| **P3: Epistemic State Collapsing** | QA-ESC-003, 006, 007 | Missing, pending, or unapplicable data collapsed into `0`, `false`, or empty arrays. | **Non-Collapsing State Triads**: Explicit status modeling (`PIPELINE_PENDING != ZERO != UNAVAILABLE`). |
| **P4: Client State & Journey Omission** | QA-ESC-001, 010 | Tests mounted a single component in isolation without testing interactive state persistence across navigation and reload. | **Rendered Journey QA**: Acceptance journeys covering multi-step user workflows and persistence. |
| **P5: Target-Device Runtime Reality Gap** | QA-ESC-002 | Automated tests ran exclusively in desktop Chromium; mobile WebKit hardware layer compositing was never exercised. | **Physical/WebKit Verification**: Target-platform WebKit layout bounding checks in pre-release gates. |
