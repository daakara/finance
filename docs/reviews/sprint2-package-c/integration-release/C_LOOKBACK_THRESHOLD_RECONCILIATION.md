# ARX TERMINAL — SPRINT 2 PACKAGE C
# LOOKBACK THRESHOLD HIERARCHY RECONCILIATION & BOUNDARY RESULTS

**Document Identifier:** `C_LOOKBACK_THRESHOLD_RECONCILIATION.md`  
**Assessment Timestamp:** `2026-10-21T09:25:00Z` (`2026-10-21T11:25:00 Europe/Berlin, UTC+2`)  
**Governing Authority:** ARX Closure Model v2 (*Decision-First, Minimum Sufficient Evidence*)  
**Test Suite:** `tests/test_lookback_threshold_boundaries.py`  
**Formal Reconciliation Disposition:** `LOOKBACK_THRESHOLD_HIERARCHY_RECONCILED`

---

## 1. Executive Summary & Problem Statement

Historical bar thresholds in quantitative platforms often suffer from conflation between storage-level technical validation, universe builder publication contracts, and strategy-specific lookback requirements.

In Sprint 2 Track C, the three-tier lookback threshold hierarchy is formally reconciled and bounded by automated tests:
1. **Tier 1 — Technical Validation Floor (50 daily bars):** Enforced by `MarketDataQualityEvidence` during State 2 -> State 3 transition (`PROVIDER_RESOLVABLE -> DATA_QUALIFIED`).
2. **Tier 2 — Operational Builder Readiness Contract (60 daily bars):** Enforced during Universe Publication evaluation to confer `DATA_READY` status for universe inclusion.
3. **Tier 3 — Strategy Benchmark Readiness (260 daily bars):** Enforced by `StrategyAdmissionEvidence` during State 5 -> State 6 transition (`PUBLISHED_UNIVERSE -> STRATEGY_ADMITTED`) for 52-week trend strategies (e.g. Minervini Trend Template).

---

## 2. Three-Tier Threshold Hierarchy Specification

```text
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       ARX LOOKBACK THRESHOLD HIERARCHY                                                │
├──────┬──────────────────────┬───────────────┬────────────────────────────┬────────────────────────────────────────────┤
│ Tier │ Contract Tier        │ Required Bars │ Governing Authority & Gate │ Functional Role & Consumer Target          │
├──────┼──────────────────────┼───────────────┼────────────────────────────┼────────────────────────────────────────────┤
│ 1    │ Technical Validation │ 50 daily bars │ MarketDataQualityEvidence  │ Core Market Store data integrity floor.    │
│      │ Minimum Floor        │               │ Gate: State 2 -> State 3   │ Below 50 bars fails closed in data quality.│
├──────┼──────────────────────┼───────────────┼────────────────────────────┼────────────────────────────────────────────┤
│ 2    │ Operational Builder  │ 60 daily bars │ UniverseBuilder Readiness  │ Confers DATA_READY for universe inclusion  │
│      │ Readiness Contract   │               │ Gate: State 4 -> State 5   │ (50-day SMA + 10-day volatility buffer).   │
├──────┼──────────────────────┼───────────────┼────────────────────────────┼────────────────────────────────────────────┤
│ 3    │ Strategy Benchmark   │ 260 daily bars│ StrategyAdmissionEvidence  │ Algorithmic strategy evaluation floor.     │
│      │ Readiness            │               │ Gate: State 5 -> State 6   │ 52-week lookback (200 DMA, 52w high/low).  │
└──────┴──────────────────────┴───────────────┴────────────────────────────┴────────────────────────────────────────────┘
```

### 2.1 Tier 1: Technical Validation Floor (50 bars)
- **Role:** Pure structural validation floor. Guarantees that the market database contains a minimally sufficient sample of trading days to calculate basic volatility metrics and check for geometric integrity.
- **Fail-Closed Behavior:** Any candidate with $< 50$ bars (e.g. 49 bars) raises `PolicyViolationError: Insufficient historical coverage... less than required builder floor (50 bars)`. The candidate cannot reach `DATA_QUALIFIED` and remains in `PROVIDER_RESOLVABLE` with status `DATA_UNAVAILABLE`.

### 2.2 Tier 2: Operational Builder Readiness Contract (60 bars)
- **Role:** Universe publication contract. Even if a security passes technical validation ($\ge 50$ bars), `UniverseBuilder` requires 60 bars (95 calendar days) to calculate the 50-day Simple Moving Average (SMA) plus a 10-day buffer for trading volume dry-up checks.
- **Fail-Closed Behavior:** If a security has $50 \le bars < 60$ (e.g. 50 to 59 bars), it reaches `DATA_QUALIFIED` (and `PUBLICATION_ELIGIBLE` if R01-R07 pass), but its data readiness decision is `DATA_UNAVAILABLE` with reason `INSUFFICIENT_BUILDER_READINESS_HISTORY: <bars> bars < 60 bars operational contract`. It is **NOT** included in `published_symbols`.

### 2.3 Tier 3: Strategy Benchmark Readiness (260 bars)
- **Role:** Downstream strategy admission floor. Algorithmic strategies requiring 52-week trend templates (such as Minervini VCP) mandate 260 trading sessions (400 calendar days) to compute 200-day DMA, 150-day DMA, 50-day DMA, and 52-week highs/lows.
- **Fail-Closed Behavior:** A security with $60 \le bars < 260$ is successfully published to the universe (`DATA_READY`), but if submitted to a 52-week trend strategy, `StrategyAdmissionEvidence` raises `PolicyViolationError: Strategy benchmark violation: history <bars> bars < 260 bars`. The candidate remains in State 5 (`PUBLISHED_UNIVERSE`) and cannot enter State 6 (`STRATEGY_ADMITTED`).

---

## 3. Empirical Boundary Test Execution Results

All 6 boundary tests executed in `tests/test_lookback_threshold_boundaries.py` with 100% pass rate:

```text
====================================================================================================
                                 LOOKBACK BOUNDARY TEST RESULTS
====================================================================================================
  TEST CASE         BAR COUNT  TECHNICAL FLOOR (50)  BUILDER READY (60)  STRAT BENCHMARK (260)  OUTCOME
────────────────────────────────────────────────────────────────────────────────────────────────────
  Boundary 1        49 bars    FAIL (Blocked)        FAIL (Blocked)      FAIL (Blocked)         PASSED
  Boundary 2        50 bars    PASS (DATA_QUALIFIED) FAIL (Unseasoned)   FAIL (Blocked)         PASSED
  Boundary 3        59 bars    PASS (DATA_QUALIFIED) FAIL (Unseasoned)   FAIL (Blocked)         PASSED
  Boundary 4        60 bars    PASS (DATA_QUALIFIED) PASS (PUBLISHED)    FAIL (Blocked)         PASSED
  Boundary 5        259 bars   PASS (DATA_QUALIFIED) PASS (PUBLISHED)    FAIL (Blocked)         PASSED
  Boundary 6        260 bars   PASS (DATA_QUALIFIED) PASS (PUBLISHED)    PASS (ADMITTED)        PASSED
====================================================================================================
```

### Detailed Boundary Analysis:
1. **49 Daily Bars (`test_boundary_49_bars_fails_technical_floor`):**
   - `MarketDataQualityEvidence.validate()` raises `PolicyViolationError`.
   - `CandidateEvaluationResult`: `current_publication_state = PROVIDER_RESOLVABLE`, `data_readiness_decision = "DATA_UNAVAILABLE"`.
   - Dry-run published symbols: `[]`.
2. **50 Daily Bars (`test_boundary_50_bars_passes_technical_floor_but_fails_builder_contract`):**
   - State machine transitions to `DATA_QUALIFIED`.
   - Candidate evaluated: `eligibility_decision = "ELIGIBLE"`, `data_readiness_decision = "DATA_UNAVAILABLE"`.
   - Reason: `INSUFFICIENT_BUILDER_READINESS_HISTORY: 50 bars < 60 bars operational contract`.
   - Dry-run published symbols: `[]`.
3. **59 Daily Bars (`test_boundary_59_bars_fails_builder_contract`):**
   - `data_readiness_decision = "DATA_UNAVAILABLE"`.
   - Dry-run published symbols: `[]`.
4. **60 Daily Bars (`test_boundary_60_bars_satisfies_builder_contract_published`):**
   - Exactly satisfies builder contract.
   - `data_readiness_decision = "DATA_READY"`.
   - Dry-run published symbols: `["AAPL"]`.
   - Strategy admission check at 60 bars raises `PolicyViolationError: Strategy benchmark violation: history 60 bars < 260 bars`.
5. **259 Daily Bars (`test_boundary_259_bars_published_fails_strategy_benchmark`):**
   - Published in universe (`DATA_READY`).
   - Strategy admission check at 259 bars raises `PolicyViolationError: Strategy benchmark violation: history 259 bars < 260 bars`.
6. **260 Daily Bars (`test_boundary_260_bars_satisfies_all_three_thresholds`):**
   - Satisfies technical floor (50), builder readiness (60), and 52-week strategy benchmark (260).
   - Successfully transitions State 5 -> State 6 (`STRATEGY_ADMITTED`).

---

## 4. Conclusion & Invariant Certification

The lookback threshold hierarchy is fully reconciled and verified:
- No conflation between data qualification, universe publication, and strategy admission.
- Exact fail-closed boundaries at 49, 50, 59, 60, 259, and 260 bars operate deterministically.
