# ARX TERMINAL — SPRINT 2 PACKAGE B
# CANONICAL INGESTION ADAPTER RELEASE CANDIDATE

## 1. Governing Authority & Scope

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Sprint 2 Operational Gate**: `G-S2-ACT`.
- **Deferred Authority**: `G-VCP-012` (separately governed).
- **Artifact Category**: Code Release Candidate Specification.
- **Production Status**: Code deployment `NOT_AUTHORIZED`; Ingestion execution `NOT_AUTHORIZED`.

---

## 2. Release Candidate Exact Inventory

The minimal canonical release candidate consists of exactly three Python source files:

| File Path | Lines | Bytes | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- |
| `scripts/ingest_alpaca_historical.py` | `1,567` | `73,674` | `CCF4A96197B0AF689B119C99C8E6CE7598D74DC619835945D423657F9A1D137D` | **MODIFIED (RECONCILED)** |
| `analyst_dashboard/data/guarded_recovery.py` | `377` | `16,446` | `8BE7D6A69A3D4316C3CF57793C8921C4E36E5E8A5BF6A78DA91281D6C4A6299D` | **NEW ENGINE** |
| `scripts/guarded_market_recovery.py` | `73` | `2,867` | `73E940354777D51F5DBEEB51A86BFF441828BE9138AA3E26FFFCD8614398A7BF` | **NEW CLI WRAPPER** |

---

## 3. Surgical Code Enhancements

### 3.1 `scripts/ingest_alpaca_historical.py`

1. **Dual Manifest Ingestion**:
   - Parses both `ingestion_scope.candidate_bindings` (Package B format) and `pilot_candidates` (Pilot format).
   - Extracts symbol directives (`REPLACE`, `BACKFILL`), durable security IDs, and listing association IDs.

2. **Strict Canonical Identity Validation**:
   - Rejects noncanonical aliases (`sec_aapl_001`, etc.) via `NONCANONICAL_IDENTITY_ALIAS_REJECTED`.
   - Enforces format `ARX_SEC_[0-9A-F]{12}` for durable security IDs.
   - Enforces format `ASSOC_ARX_SEC_[0-9A-F]{12}_[A-Z]{4}_[A-Z0-9]+_\d{4}-\d{2}-\d{2}` for listing associations.

3. **Provider Lineage Purity (`is_prov_compatible`)**:
   - Static method verifies 1:1 candle-to-provenance correspondence.
   - Rejects non-Alpaca stored lineage (`MASSIVE_POLYGON`, etc.) with `INCOMPATIBLE_PROVIDER`.
   - Enforces `SPLIT_ADJUSTED_ONLY` basis with `INCOMPATIBLE_ADJUSTMENT`.

4. **Explicit `REPLACE` Action & Historical Scope**:
   - Manifest directive `action="REPLACE"` unconditionally suppresses incremental refresh and queries full window from `start_date`.
   - Executes atomic replacement: `replace_existing_scope=(action in ("BACKFILL", "REPLACE"))`.
   - Verifies outside-scope records: triggers `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE` if unprovenanced, foreign provider, or non-split records exist outside `[sym_start, end_date]`.

### 3.2 `analyst_dashboard/data/guarded_recovery.py`

- **10-Step Non-Destructive Guarded Recovery Protocol**:
  1. Mandatory incident authorization token (`G-S2-ACT` or `INC-G-S2-...`).
  2. Mandatory maintenance-mode authorization token (`MAINT-...`).
  3. Multi-layer process and handle fencing: scans `/proc/*/fd` (Linux) and handle tables to guarantee zero active reader or writer connections to SQLite database or WAL.
  4. Non-destructive quarantine: moves affected `.db`, `.db-wal`, and `.db-shm` into timestamped quarantine directory (`/root/incident_quarantine/<incident_id>_<timestamp>/`).
  5. Cryptographic pre-restore verification: validates backup existence, SHA-256 agreement, and `PRAGMA integrity_check == 'ok'`.
  6. Staged restoration: restores to `.restore_candidate` temporary file before atomic rename.
  7. Post-restoration verification: validates schema, tables, candle count, and provenance sidecars.
  8. Atomic rename into target path.
  9. Structured receipt emission (`GuardedRecoveryReceipt`).
  10. Fail-closed rollback if any verification fails.

### 3.3 `scripts/guarded_market_recovery.py`

- CLI wrapper exposing parameter flags:
  `--target-db`, `--backup-db`, `--expected-backup-sha256`, `--incident-id`, `--auth-token`, `--maint-token`, `--quarantine-dir`.
- Strictly enforces non-zero exit codes on recovery abort or fence violation.

---

## 4. Boundary Statement & Blast Radius Containment

- **Zero Schema Alteration**: Modifies no database schemas, tables, or columns.
- **Zero Live Pipeline Exposure**: Scanner runner, VCP scheduler, and live stream handlers are not touched.
- **Zero Package C Advancement**: Gate 12 remains locked; universe publication remains unexecuted.
- **Strictly Contained**: Changes are isolated to historical REST ingestion and disaster recovery tooling.
