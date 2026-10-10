# ARX TERMINAL — RADAR SMART MONEY WAVE 2A TEST EVIDENCE

**Document ID**: `SMART-MONEY-WAVE2A-EVIDENCE-001`  
**Date**: `2026-10-10`  
**Execution Context**: Weekend Isolated Workstream — Wave 2A  
**Status**: `LOCAL_VERIFICATION_COMPLETE`  
**Authoritative Scope**: Unit Tests, 500-Permutation Property Tests, Bounded Parity, Accounting Closure, Pipeline Integrity

---

## 1. Executive Summary

This document certifies the test and verification evidence for **Smart Money Wave 2A: Market-Wide Universe Authority & Security Master / CIK Mapping**.

All unit, integration, accounting closure, and invariance property tests pass unconditionally without degrading or mutating any frozen Candidate 004 production shadow assets.

```
Total Test Corpus (Python Backend): 35 passed in 10.22s
- tests/test_smart_money_universe_authority.py: 7 passed
- tests/test_smart_money_form4_pipeline.py:     8 passed
- tests/test_smart_money_api.py:                3 passed
- tests/test_smart_money_legislative.py:        8 passed
- tests/test_radar_scanner_pipeline.py:         9 passed
```

---

## 2. Universe Authority Test Matrix

| Test Identifier | Method | Invariant / Rule Tested | Result |
| :--- | :--- | :--- | :--- |
| **SM-U-01** | `test_universe_builder_accounting_closure` | **Arithmetic Closure**: `source_count == eligible + ineligible + unresolved` exactly holds. Scannable closure `scannable_count == eligible_count` verified. | **PASS** |
| **SM-U-02** | `test_universe_build_id_and_hash_parity_across_runs` | **Determinism & Idempotency**: Building twice from identical inputs yields 100% identical `universe_build_id`, `member_identity_hash`, `source_snapshot_hash`, and symbol ordering. | **PASS** |
| **SM-U-03** | `test_ambiguous_cik_collision_fails_closed` | **Collision Isolation (Fail-Closed)**: Ticker colliding with multiple CIKs is quarantined to `UNRESOLVED` and never admitted to scannable universe. | **PASS** |
| **SM-U-04** | `test_missing_cik_fails_closed` | **Missing CIK Isolation**: Security with `cik: None` is quarantined to `UNRESOLVED`. Scannable count is 0. | **PASS** |
| **SM-U-05** | `test_scanner_market_wide_requires_universe_build_id` | **Provenance Gate**: `SmartMoneyForm4Scanner.scan_issuers(is_bounded_sample=False)` strictly raises `ValueError` if `universe_build_id` is missing; binds manifest on supply. | **PASS** |
| **SM-U-06** | `test_property_500_permutations_member_hash_invariance` | **Property Test (500 Permutations)**: 500 randomized shuffles of source input records produce 0.00% hash drift and identical lexicographical membership order. | **PASS** |
| **SM-U-07** | `test_bounded_universe_build_parity_against_35_benchmark_tickers` | **Bounded Parity & ADR Exclusion**: Historical 35 tickers treated as a bounded test subset; ADRs (NVO, ASML, ONON) excluded under Rule SM-E03 (Exchange Act Rule 3a12-3); accounting closure holds (32 eligible, 3 ineligible). | **PASS** |

---

## 3. Property Test: 500 Permutation Invariance

### Invariant Tested
The membership fingerprint of an immutable universe build must be invariant under any ordering of raw upstream data.

$$\text{Hash}(P(\text{Records})) = \text{Hash}(\text{Records}) \quad \forall P \in S_N$$

### Execution Methodology
1. Built a baseline universe manifest from a canonical heterogeneous source population of 17 instruments.
2. Extracted `manifest_base.member_identity_hash` and the canonical symbol sequence.
3. Executed 500 consecutive trials:
   - Randomly shuffled the input records array with Python's PRNG (`random.shuffle()`).
   - Invoked `SmartMoneyUniverseBuilder.build_universe()`.
   - Compared resulting `member_identity_hash` against baseline.
   - Compared resulting `[s.symbol for s in scannable]` against baseline.

### Empirical Outcome
- **Total Permutations Tested**: 500 / 500
- **Hash Drift Incidents**: 0 (0.00%)
- **Ordering Drift Incidents**: 0 (0.00%)
- **Conclusion**: Canonical sorting by `(symbol, cik, exchange)` prior to hashing guarantees deterministic member identity hashing.

---

## 4. Pipeline Readiness & Anti-False-Zero Enforcement

### Invariant Tested
Under no circumstances may an incomplete, unmapped, or unready scanner return an empty result set (`[]` or `0 items`) as a genuine zero-hit scan. It MUST fail closed to `PIPELINE_PENDING` with `matched_count: None`.

### Verification Suite
In `tests/test_smart_money_form4_pipeline.py`:
- `test_readiness_contract_false_zero_prevention`:
  1. Rejects market-wide scan without `universe_build_id` via `ValueError`.
  2. With governed `universe_build_id`, produces `status == "PIPELINE_PENDING"` and `matched_count is None`.
  3. `items` array is empty (`[]`), but distinguishable from `NO_MATCHES` (`matched_count: 0`).
  4. Readiness evaluation returns `checks["market_wide_universe_governed"] = False` until full production authorization.
- `test_bounded_sample_snapshot_with_zero_matches`:
  - When explicitly running in `is_bounded_sample=True` mode with zero qualifying filings, reports `status == "NO_MATCHES"` with `matched_count == 0`.

---

## 5. Performance Budget & SEC EDGAR Rate Limit Analysis

### 5.1 Universe Scale & Ingestion Architecture
| Metric | Value | Derived Rationale |
| :--- | :--- | :--- |
| **Source Entities (SEC Tickers Exchange)** | 10,435 | Raw SEC `company_tickers_exchange.json` population |
| **Major US Exchange Listings** | 7,710 | NASDAQ (4,376), NYSE (3,290), CBOE (44) |
| **Estimated Eligible Operating Equities** | ~4,800 – 5,200 | Excludes ADRs/FPIs (~1,200), Closed-End Funds/ETFs (~1,500), Preferreds |
| **Daily Incremental Filings (Form 4)** | ~100 – 300 | Average weekday Form 4 volume across market |

### 5.2 Rate Limiting & SEC Compliance
- **SEC Fair Access Limit**: Maximum 10 requests per second with registered `User-Agent`.
- **ARX Safe Conservative Operating Limit**: **5.0 requests per second** (50% safety ceiling, implemented via token bucket `RateLimiter(rate=5.0)`).
- **Cold-Start Universe Sync Duration**:
  $$\text{Time} = \frac{5,000 \text{ entities}}{5.0 \text{ req/sec}} = 1,000 \text{ seconds} \approx 16.7 \text{ minutes}$$
- **Daily Incremental Update Duration**:
  $$\text{Time} = \frac{300 \text{ filings}}{5.0 \text{ req/sec}} = 60 \text{ seconds} = 1.0 \text{ minute}$$
- **Request Budget Feasibility**:
  Incremental daily scanning can execute in 1 to 2 minutes per day via scheduled background task or cron, easily fitting within standard cloud execution budgets without risk of SEC IP throttling.

---

## 6. Candidate 004 Production Shadow Firewall Attestation

```
======================================================================
CANDIDATE 004 ISOLATION VERIFICATION
======================================================================
PROTECTED_FILES_CHANGED = 0
CANDIDATE_004_FUNCTIONAL_SHA = fafba93ccd60f40792bb3f5ea6702804eeea40cf (UNTOUCHED)
ACTIVE_EPOCH_DATABASE = shadow_evidence.db (UNTOUCHED)
PRODUCTION_RADAR_STATUS = PIPELINE_PENDING (UNCHANGED)
PRODUCTION_DEPLOYMENTS = 0 (LOCAL ONLY)
GIT_BRANCH = arx/weekend-smart-money-investigation (ISOLATED)
======================================================================
```
