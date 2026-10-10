# SMART MONEY WAVE 1 — READINESS & GOVERNANCE CONTRACT

## 1. Executive Summary

This document governs the lifecycle states and transitions for the Smart Money scanner in ARX Terminal. It defines the exact prerequisites required to transition between `PIPELINE_PENDING`, `DATA_INCOMPLETE`, `AVAILABLE`, `NO_MATCHES`, and `UNAVAILABLE`.

It formally outlaws the "false-zero" defect where an incomplete or unready scanner outputs 0 matches.

---

## 2. Governed Scanner Lifecycle States

| State | Description | `matched_count` | `candidates` | UI Badge / Semantics |
|:---|:---|:---|:---|:---|
| `PIPELINE_PENDING` | Pipeline unprovisioned, in development, or awaiting market-wide universe governance. | `null` (`None`) | `[]` | ⏳ Pipeline Pending (Amber) |
| `DATA_INCOMPLETE` | Ingestion in progress, partial universe coverage, or stale data beyond threshold. | `null` or explicit partial | `[]` or partial | ⚠️ Data Incomplete (Orange) |
| `AVAILABLE` | Governed universe, complete scan, fresh data, >= 1 valid matches found. | Integer `>= 1` | List of items | 🎯 Active Matches (Emerald) |
| `NO_MATCHES` | Governed universe, complete scan, fresh data, exactly 0 valid matches found. | Integer `0` | `[]` | 🔍 No Setups Found (Slate) |
| `UNAVAILABLE` | External source API failure, rate-limit lockout, or schema break. | `null` (`None`) | `[]` | ❌ Feed Offline (Rose) |

---

## 3. Strict Transition Prerequisites

### Invariant 1: `AVAILABLE` Gate
A scanner snapshot may declare `status: AVAILABLE` **if and only if** ALL of the following criteria are verified:
1. `source_ingestion_operational == True`: Source adapters are actively ingesting without critical errors.
2. `market_wide_universe_governed == True`: An authoritative security master / universe exists and provides an immutable membership hash.
3. `universe_denominator > 0`: Total scannable universe count is known and verified.
4. `freshness_policy_satisfied == True`: Data as-of timestamp is within the allowable staleness threshold (e.g. <= 48 hours for market days).
5. `full_scan_completed == True`: The scanner processed 100% of the scannable population without unhandled execution aborts.
6. `snapshot_persisted == True`: An immutable snapshot envelope has been generated and signed with release SHA and content hash.

### Invariant 2: `NO_MATCHES` Gate
A scanner snapshot may declare `status: NO_MATCHES` (and output `matched_count: 0`) **if and only if**:
1. All criteria for `AVAILABLE` are satisfied.
2. The evaluated candidate count following all deterministic filters is exactly 0.

### Invariant 3: `PIPELINE_PENDING` Protection
Whenever `market_wide_universe_governed == False` OR `source_ingestion_operational == False`:
- The scanner **must fail closed to `PIPELINE_PENDING`**.
- `matched_count` **must be `null` (`None`)**.
- Emitting `matched_count: 0` while in `PIPELINE_PENDING` is a critical governance violation.

---

## 4. Wave 1 Authority Ruling

In Wave 1:
```text
SMART_MONEY_SOURCE_UNIVERSE = NOT_ESTABLISHED (Market-wide US equities master unprovisioned in UniverseStore)
SMART_MONEY_ELIGIBLE_UNIVERSE = NOT_ESTABLISHED
SMART_MONEY_UNIVERSE_COUNT = NOT_ESTABLISHED
35-ASSET_REFERENCE_SET_IS_MARKET_WIDE_UNIVERSE = NO
SMART_MONEY_WAVE1_UNIVERSE_GATE = BLOCKED

VALIDATION_SCOPE = BOUNDED_ENGINEERING_SAMPLE
SMART_MONEY_COVERAGE = NOT_ESTABLISHED
```

**Binding Ruling**: Because the market-wide universe authority is not yet established for 5,000+ equities, the production radar status of Smart Money **remains strictly `PIPELINE_PENDING`**. The Wave 1 engineering prototype validates the end-to-end ingestion, parsing, clustering, and ranking logic on a bounded engineering sample without prematurely flipping production status to `AVAILABLE`.
