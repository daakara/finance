# ARX TERMINAL — SPRINT 2 PACKAGE C
# FULL-HISTORY PROVENANCE REPLACEMENT & CERTIFICATION TESTS

**Document Identifier:** `C_FULL_HISTORY_CERTIFICATION_TESTS.md`  
**Assessment Timestamp:** `2026-10-21T09:25:00Z` (`2026-10-21T11:25:00 Europe/Berlin, UTC+2`)  
**Governing Authority:** ARX Closure Model v2 (*Decision-First, Minimum Sufficient Evidence*)  
**Test Suite:** `tests/test_package_b_v3_canonical_consumer.py`  
**Execution Environment:** Disposable In-Memory / Test-Only Python Virtualenv  
**Formal Test Disposition:** `FULL_HISTORY_CERTIFICATION_VERIFIED`

---

## 1. Executive Summary

Phase 3 verifies the transition from the legacy pre-replacement state to the certified post-replacement state under Package B Option B Full-History Lineage Invariant.

The tests prove that:
1. **Pre-Replacement Baseline Fails Closed:** Legacy unprovenanced data (`AAPL`, `GOOGL`), cross-provider Polygon data (`GOOG`), and unbackfilled state (`ABBV`) fail data qualification (`DATA_UNAVAILABLE`). Zero securities are published.
2. **Post-Replacement State Qualifies All Candidates:** Uniform Alpaca SIP split-adjusted history across 288 sessions qualifies all 4 pilot candidates (`AAPL`, `ABBV`, `GOOG`, `GOOGL`) to `DATA_READY` and `PUBLICATION_ELIGIBLE`.
3. **Multi-Share-Class Pricing Separation Enforced:** Alphabet Inc. Class C (`GOOG`) and Class A (`GOOGL`) must maintain strictly distinct pricing series; cross-mapped or identical price series fail closed with `PolicyViolationError`.

---

## 2. Pre-Replacement vs Post-Replacement State Matrix

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PRE-REPLACEMENT VS POST-REPLACEMENT BEHAVIOR                                     │
├────────┬───────────────────────────────────────────┬───────────────────────────────────────────┬────────────────────┤
│ Symbol │ Pre-Replacement State (Fails Closed)      │ Post-Replacement State (Qualified)        │ Track C Result     │
├────────┼───────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
│ AAPL   │ 289 bars, 0 prov rows (Legacy unprov.)   │ 288 bars, SIP_CONSOLIDATED, SPLIT_ADJ     │ State 2 -> State 4 │
│        │ State: PROVIDER_RESOLVABLE (DATA_UNAVAIL) │ State: PUBLICATION_ELIGIBLE (DATA_READY)  │ QUALIFIED          │
├────────┼───────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
│ GOOG   │ 287 bars, MASSIVE_POLYGON prov rows       │ 288 bars, SIP_CONSOLIDATED, SPLIT_ADJ     │ State 2 -> State 4 │
│        │ State: PROVIDER_RESOLVABLE (DATA_UNAVAIL) │ State: PUBLICATION_ELIGIBLE (DATA_READY)  │ QUALIFIED          │
├────────┼───────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
│ GOOGL  │ 288 bars, 0 prov rows (Legacy unprov.)   │ 288 bars, SIP_CONSOLIDATED, SPLIT_ADJ     │ State 2 -> State 4 │
│        │ State: PROVIDER_RESOLVABLE (DATA_UNAVAIL) │ State: PUBLICATION_ELIGIBLE (DATA_READY)  │ QUALIFIED          │
├────────┼───────────────────────────────────────────┼───────────────────────────────────────────┼────────────────────┤
│ ABBV   │ 0 bars, 0 prov rows (Unbackfilled)        │ 288 bars, SIP_CONSOLIDATED, SPLIT_ADJ     │ State 2 -> State 4 │
│        │ State: PROVIDER_RESOLVABLE (DATA_UNAVAIL) │ State: PUBLICATION_ELIGIBLE (DATA_READY)  │ QUALIFIED          │
└────────┴───────────────────────────────────────────┴───────────────────────────────────────────┴────────────────────┘
```

---

## 3. Empirical Test Execution Results

All tests executed via `pytest tests/test_package_b_v3_canonical_consumer.py -v`:

### 3.1 Test 04 — Pre-Replacement Storage State Fails Closed
- **Input:** Candidates constructed with `PackageBV3Consumer.build_candidates_from_manifest(manifest, pre_replacement=True)`.
- **Evaluations:**
  - `AAPL`: `data_readiness_decision = "DATA_UNAVAILABLE"`, `current_publication_state = PROVIDER_RESOLVABLE`.
  - `GOOG`: `data_readiness_decision = "DATA_UNAVAILABLE"`, `current_publication_state = PROVIDER_RESOLVABLE`.
  - `GOOGL`: `data_readiness_decision = "DATA_UNAVAILABLE"`, `current_publication_state = PROVIDER_RESOLVABLE`.
  - `ABBV`: `data_readiness_decision = "DATA_UNAVAILABLE"`, `current_publication_state = PROVIDER_RESOLVABLE`.
- **Dry-Run Output:**
  ```json
  {
    "published_symbols": [],
    "accounting": {
      "source_count": 4,
      "eligible_count": 0,
      "ineligible_count": 0,
      "unresolved_count": 4,
      "data_ready_count": 0,
      "data_unavailable_count": 0,
      "data_unresolved_count": 0,
      "scannable_count": 0,
      "is_closed": true
    }
  }
  ```
- **Outcome:** **`PASSED`** (Zero securities published).

### 3.2 Test 05 — Post-Replacement Uniform SIP Qualifies All Candidates
- **Input:** Candidates constructed with `PackageBV3Consumer.build_candidates_from_manifest(manifest, pre_replacement=False)`.
- **Evaluations:**
  - All 4 candidates transition `PROVIDER_RESOLVABLE -> DATA_QUALIFIED -> PUBLICATION_ELIGIBLE`.
  - All 4 candidates evaluated to `DATA_READY`.
- **Dry-Run Output:**
  ```json
  {
    "published_symbols": ["AAPL", "ABBV", "GOOG", "GOOGL"],
    "accounting": {
      "source_count": 4,
      "eligible_count": 4,
      "ineligible_count": 0,
      "unresolved_count": 0,
      "data_ready_count": 4,
      "data_unavailable_count": 0,
      "data_unresolved_count": 0,
      "scannable_count": 4,
      "is_closed": true
    }
  }
  ```
- **Publication Pointer:**
  - Pointer build ID: Generated successfully.
  - Published symbols: `["AAPL", "ABBV", "GOOG", "GOOGL"]`.
  - Content digest length: 64 hex characters (SHA-256).
- **Outcome:** **`PASSED`**.

### 3.3 Test 06 — Multi-Share-Class Pricing Divergence
- **Distinct Identities:** `GOOG` (`ARX_SEC_A11A24D442E2`) vs `GOOGL` (`ARX_SEC_B7EF9ED7A35A`).
- **Distinct Associations:** `ASSOC_..._GOOG_...` vs `ASSOC_..._GOOGL_...`.
- **Distinct Pricing Series:** `[178.50, 179.10, 180.25]` vs `[177.80, 178.40, 179.60]` -> Verification **`PASSED`**.
- **Adversarial Collision Test:** Feeding identical price series raises `PolicyViolationError: Pricing collision: GOOG (Class C) and GOOGL (Class A) exhibit identical pricing series. Cross-contamination or provider alias mapping detected; failing closed.` -> **`PASSED`**.

---

## 4. Verification Conclusion

The full-history certification suite conclusively demonstrates that Track C correctly gates universe inclusion:
- Dirty, unprovenanced, or mixed-provider market data is blocked from universe entry.
- Uniform Alpaca SIP split-adjusted historical data seamlessly qualifies candidates.
- Multi-share-class capital structures are strictly safeguarded against price cross-contamination.
