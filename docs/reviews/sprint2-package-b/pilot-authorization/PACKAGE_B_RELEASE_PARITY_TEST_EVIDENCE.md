# ARX TERMINAL — SPRINT 2 PACKAGE B
# RELEASE CANDIDATE TEST PARITY & REGRESSION EVIDENCE

## 1. Governing Authority & Scope

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Sprint 2 Operational Gate**: `G-S2-ACT`.
- **Test Execution Environment**: Local non-production test runner (Python 3.11.0, pytest-9.1.1, SQLite 3.40.1).
- **Target Release Candidate**: `scripts/ingest_alpaca_historical.py` (`CCF4A961...`), `analyst_dashboard/data/guarded_recovery.py` (`8BE7D6A6...`), `scripts/guarded_market_recovery.py` (`73E94035...`).

---

## 2. Test Execution Summary

A total of **228 tests** executed with **100% pass rate** and **zero regressions**:

| Test Suite File | Tests Run | Passed | Failed | Duration | Focus Area |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tests/test_pilot_identity_and_guarded_recovery.py` | 15 | 15 | 0 | 0.52s | Canonical IDs, alias rejection, WAL fence, token fence, manifest digests |
| `tests/test_alpaca_historical_sip_ingestion.py` | 54 | 54 | 0 | 3.02s | SIP adapter validation, lineage checks, pagination, error handling |
| `tests/test_bounded_150_candidate_expansion.py` | 71 | 71 | 0 | 3.45s | Reference identity universe expansion & constraints |
| `tests/test_identity_persistence.py` | 18 | 18 | 0 | 1.10s | SQLite identity persistence & schema invariants |
| `tests/test_registration_execution_wrapper.py` | 39 | 39 | 0 | 2.15s | Registration execution idempotency & safety wrappers |
| `tests/test_three_candidate_registration_plan.py` | 13 | 13 | 0 | 0.85s | Three-candidate registration plan verification |
| `tests/test_universe_admission_mappings.py` | 18 | 18 | 0 | 0.95s | Exchange MIC & admission mapping rules |
| **TOTAL** | **228** | **228** | **0** | **~12.0s** | **100% Green / Zero Regressions** |

---

## 3. Explicit Invariant Proofs

### Invariant 1: Actual Canonical Identity Format Enforcement
- **Proof**: `test_actual_canonical_ids_accepted` confirms `ARX_SEC_0B4C728FE0BE`, `ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17`, and counterparts for GOOG, GOOGL, ABBV load with 100% accuracy.
- **Proof**: `test_noncanonical_aliases_rejected` confirms informal alias `sec_aapl_001` is rejected with `NONCANONICAL_IDENTITY_ALIAS_REJECTED`.
- **Proof**: Malformed IDs (e.g. `INVALID_SEC_ID_123`) fail with `INVALID_CANONICAL_SECURITY_ID`.

### Invariant 2: Multi-Share-Class Economic & Voting Isolation
- **Proof**: `test_goog_googl_dual_class_isolation` verifies GOOG (`ARX_SEC_A11A24D442E2`, `CLASS_C`, non-voting) and GOOGL (`ARX_SEC_B7EF9ED7A35A`, `CLASS_A`, voting) maintain completely separate identities and association IDs.

### Invariant 3: Lineage Purity & Foreign Provider Rejection
- **Proof**: `test_incompatible_lineage_forces_full_backfill` verifies that stored history with provider `MASSIVE_POLYGON` or missing provenance sidecars is rejected by `is_prov_compatible` with `INCOMPATIBLE_PROVIDER`. Incremental refresh is suppressed and full replacement is required.

### Invariant 4: Historical Scope Boundary Alignment (Policy B)
- **Proof**: `test_canonical_reconciled_manifest_v3_deterministic_digest` loads `PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json`.
- **Window**: `2025-08-27` to `2026-10-19` matches earliest stored session in production (`2025-08-27`). `outside_candles` is 0. The adapter executes without `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE`.

### Invariant 5: Dual Manifest Schema Support
- **Proof**: `test_package_b_candidate_bindings_accepted` verifies the adapter correctly parses `ingestion_scope.candidate_bindings` while applying canonical identity checks.
- **Proof**: `test_package_b_candidate_bindings_rejects_noncanonical_alias` verifies alias rejection also applies to `candidate_bindings`.

### Invariant 6: WAL Mode Concurrency & Reader Handle Fence
- **Proof**: `test_wal_exclusive_lock_insufficiency_and_reader_fence` verifies:
  1. `BEGIN EXCLUSIVE` only blocks writers; active readers remain connected and query unhindered.
  2. Replacing database file while reader holds open descriptor leaves reader attached to unlinked ghost inode.
  3. Guarded recovery engine checks active handles (`/proc/*/fd` scanner) and refuses recovery until readers are terminated.

### Invariant 7: Mandatory Maintenance-Mode Authorization Token Fence
- **Proof**: `test_maintenance_mode_token_mandatory_fence` verifies that omitting `--maint-token` or providing an invalid token immediately blocks guarded recovery with exit code 1.

### Invariant 8: Single-Symbol Failure Isolation
- **Proof**: `test_single_symbol_failure_isolation` proves that if candidate 2 (GOOG) fails mid-flight, only candidate 2's transaction rolls back. Candidate 1 (AAPL) and existing data (MSFT) remain safely committed. Whole-database recovery is not triggered.
