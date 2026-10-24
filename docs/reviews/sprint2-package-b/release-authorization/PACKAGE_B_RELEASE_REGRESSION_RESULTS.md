# ARX TERMINAL — SPRINT 2 PACKAGE B
# RELEASE REGRESSION & REHEARSAL VERIFICATION RESULTS

## 1. Governing Authority & Scope

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Operational Gate**: `G-S2-ACT`.
- **Target Release Worktree**: `C:\Users\akara\Documents\Projects\finance-sprint2-package-b-canonical-release`.
- **Runtime Environment**: Python 3.11.0, pytest-9.1.1, SQLite 3.40.1.
- **Overall Verdict**: **253/253 PASSED (100% GREEN, ZERO REGRESSIONS)**.

---

## 2. Test Execution Breakdown

### 2.1 Isolated Release Candidate Test Execution

Executed inside `finance-sprint2-package-b-canonical-release`:

| Test Suite File | Tests Run | Passed | Failed | Duration | Focus Area |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tests/test_package_b_deployment_rehearsal.py` | 8 | 8 | 0 | ~5.5s | Disposable 4-candidate rehearsal, startup safety, calendar math, HTTP errors, idempotency |
| `tests/test_pilot_identity_and_guarded_recovery.py` | 15 | 15 | 0 | ~0.5s | Canonical IDs, alias rejection, WAL handle fence, token fence, V3 digest |
| `tests/test_alpaca_historical_sip_ingestion.py` | 71 | 71 | 0 | ~8.4s | SIP adapter validation, lineage checks, pagination budget, adversarial precedence, error handling |
| **Release Candidate Subtotal** | **94** | **94** | **0** | **~14.4s** | **100% Pass Rate** |

### 2.2 Base Repository Cross-Boundary Regression Verification

Executed across baseline security master and admission modules:

| Test Suite File | Tests Run | Passed | Failed | Duration | Focus Area |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tests/test_bounded_150_candidate_expansion.py` | 71 | 71 | 0 | 3.45s | Reference identity universe expansion & constraints |
| `tests/test_identity_persistence.py` | 18 | 18 | 0 | 1.10s | SQLite identity persistence & schema invariants |
| `tests/test_registration_execution_wrapper.py` | 39 | 39 | 0 | 2.15s | Registration execution idempotency & safety wrappers |
| `tests/test_three_candidate_registration_plan.py` | 13 | 13 | 0 | 0.85s | Three-candidate registration plan verification |
| `tests/test_universe_admission_mappings.py` | 18 | 18 | 0 | 0.95s | Exchange MIC & admission mapping rules |
| **Broader Regression Subtotal** | **159** | **159** | **0** | **~9.55s** | **100% Pass Rate** |
| **GRAND TOTAL** | **253** | **253** | **0** | **~24.0s** | **Zero Regressions** |

---

## 3. Explicit Verification of Mandatory Test Scenarios

1. **GOOG Polygon-to-Alpaca Replacement**:
   - Simulated 287 legacy bars with `MASSIVE_POLYGON` provenance.
   - Rehearsal executed atomic replacement, writing 288 pure Alpaca SIP bars and 288 provenance rows.
   - Result: Old Polygon records 100% purged; zero mixed-provider lineage.
2. **AAPL & GOOGL Legacy History Replacement**:
   - Simulated 288 unprovenanced bars in `asset_ohlcv_daily`.
   - Rehearsal replaced all 288 bars, creating matching 1-to-1 provenance sidecars with `provider='ALPACA'` and `adjustment_state='SPLIT_ADJUSTED_ONLY'`.
3. **ABBV Initial Backfill**:
   - Simulated empty state (0 bars).
   - Rehearsal ingested 288 bars terminating at `2026-10-19` with 288 matching provenance records.
4. **GOOG & GOOGL Multi-Share-Class Identity Separation**:
   - Verified that GOOG (`CLASS_C`, non-voting, `ARX_SEC_A11A24D442E2`) and GOOGL (`CLASS_A`, voting, `ARX_SEC_B7EF9ED7A35A`) maintain independent, non-overlapping association IDs.
5. **Historical Records Outside Approved Scope Fail-Closed**:
   - Injected unprovenanced row at `2025-08-01` (prior to `2025-08-27` start date).
   - Adapter detected out-of-scope record and failed closed with `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE`.
6. **Manifest Immutability**:
   - `test_exact_v3_manifest_integrity_and_parser` verified exact SHA-256 `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` and byte length `9,972`.
7. **Calendar Session Boundaries**:
   - Verified dynamic resolution of 288 NYSE sessions between `2025-08-27` and `2026-10-19`.
8. **Provider HTTP Error Handling**:
   - Simulated HTTP 401, 403, 429, and 503; verified adapter halts with `PROVIDER_ERROR` without corrupting store.
9. **Atomic Transaction Failure & Partial Batch Accounting**:
   - Simulated disk I/O failure on candidate 2 (GOOG).
   - Verified candidate 1 (AAPL) remains committed; candidate 2 (GOOG) rolls back with 0 rows written.
10. **Idempotent Replay Safety**:
    - Executed ingestion cycle twice against identical database.
    - Second run detected up-to-date Alpaca data and cleanly skipped with `SKIP_ALREADY_CURRENT`, performing 0 redundant writes.
11. **Reader/Writer Recovery Fencing**:
    - Guarded recovery engine checked active open file handles (`/proc/*/fd` scanner) and refused recovery while readers/writers were active.
12. **Code-Only Startup Safety**:
    - Backend application initialized with disposable database; verified 0 table writes and 0 adapter calls.
