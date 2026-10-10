# ARX TERMINAL — SMART MONEY WAVE 2C
## INCREMENTAL FORM 4 INGESTION CONTRACT

```yaml
CONTRACT_ID: SMC-WAVE2C-INC-001
VERSION: 2.1.0
STATUS: LOCAL_ENGINEERING_VERIFIED
UNIVERSE_BUILD_ID: UB_SM_20261010_R1_19ef857fec1f
SCANNABLE_PARTITIONS: 106
BASELINE_ID: BASE-SM-WAVE2C-20261010-R1
BASELINE_WATERMARK: 2026-10-10T01:02:33.000Z
FIREWALL_STATUS: CANDIDATE_004_ISOLATED
```

---

### 1. Architectural Scope & Operating Principles

1. **Local-Only Execution**: All incremental machinery operates exclusively within the local development environment. Production crons, Railway workers, remote pushes, and `main` merges are strictly prohibited.
2. **Deterministic Partitioning**: The scannable universe of 106 common equities (`UB_SM_20261010_R1_19ef857fec1f`) is partitioned by issuer CIK (`PART-{issuer_cik}`).
3. **Partition Isolation**: Each partition maintains its independent poll cursor, candidate watermark, and gap ledger.
4. **Three Distinct Progress Metrics**:
   - `MAX_OBSERVED_SOURCE_TIME`: Highest `public_availability_at` observed anywhere in raw filings or ledger.
   - `GLOBAL_PROCESSING_CURSOR`: Minimum latest attempted poll timestamp across all 106 partitions.
   - `GLOBAL_CERTIFIED_COMPLETENESS_WATERMARK`: Strictly defined as `MIN(certified_complete_through)` across all 106 partitions. Later filings NEVER imply completeness.

---

### 2. Incremental Cycle Specification

Each incremental polling cycle is governed by an immutable `IncrementalCycle` identity:
- `cycle_id = "CYCLE-" + sha256(payload)[:16]`
- Parameters:
  - `universe_build_id`: Bound to active universe build.
  - `window_start`: Derived from `previous_certified_watermark - overlap_window_seconds` (fail-safe lookback).
  - `window_end`: Deterministic cycle poll boundary (UTC).
  - `partition_set_hash`: SHA256 of canonical sorted 10-digit zero-padded CIK list.
  - `incremental_policy_version`: `inc-v2.1`.
  - `source_contract_version`: `1.0.0`.

---

### 3. Concurrency & Mutex Semantics

1. **Cycle Leases**: Cycles acquire an exclusive atomic lease in `sm_incremental_runs`.
2. **Collision Prevention**: Attempting to acquire a lease for an active `RUNNING` cycle fails closed immediately.
3. **Lease Expiry / Heartbeats**: Leases without heartbeats older than TTL fail closed and require explicit audit recovery.

---

### 4. Zero False Completeness Invariant

`GLOBAL_CERTIFIED_COMPLETENESS_WATERMARK` advances if and only if:
1. Every one of the 106 partitions has polled and verified through the target time.
2. Zero open or retrying source gaps exist with a blocking frontier at or before the target time.
3. Zero uncertified late-arrival breaches exist.
4. An immutable `WatermarkReceipt` is cryptographically sealed in SQLite.
