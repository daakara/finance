# ARX Terminal — SM-ACT-026 / SM-ACT-026R SEC Evidence Visibility Repair & Qualification
## Directive: SM-ACT-026 / SM-ACT-026R PostgreSQL Evidence Visibility Repair & Production Qualification

**Tracker:** GitHub Issue #165 (`daakara/finance`)  
**Production Commit Baseline:** [`649d31511c79324dbd1cb7369ffacac1c36ea75f`](https://github.com/daakara/finance/commit/649d31511c79324dbd1cb7369ffacac1c36ea75f) (PR #190)  
**Evaluated Filing:** Accession `0001696841-26-000012` (NVDA, Timothy S. Teter, Code F)  
**Reconciliation Directive:** `SM-ACT-026R`  
**Execution Timestamp (UTC):** `2026-11-05T01:30:00+00:00`  
**Binding Disposition:** **`SMART_MONEY_READ_ONLY_EVIDENCE_VISIBILITY_QUALIFIED_FOR_RELEASE`**  

---

### 1. Production Baseline Reconciliation

- **Production Anchor:** `649d31511c79324dbd1cb7369ffacac1c36ea75f` (PR #190 merged into `release/production`).
- **Included Anchors:** Contains PR #184 (`d247ee78`: fail-closed retention protection) and PR #190 (VCP bitemporal cutoff fix).
- **Branch Lineage:** Review branch `fix/sm-act-026-postgres-evidence-visibility` branched directly from `649d31511c79324dbd1cb7369ffacac1c36ea75f`.
- **Target PR:** [PR #192](https://github.com/daakara/finance/pull/192).

---

### 2. Surgical Projection Implementation & SM-ACT-026R Reconciliation

1. **Authoritative Storage Backend Selection:**  
   `SecEdgarFetcher._get_cached_recent_filings_envelope()` calls `get_storage_backend()`. When configured with `ARX_STORAGE_BACKEND=postgres` or production URL, it executes parameterized queries against PostgreSQL `form4_raw_filings` with a LEFT JOIN on `sm_source_availability_ledger`.
2. **Backwards-Compatible SQLite Fallback:**  
   When `ARX_STORAGE_BACKEND=sqlite` in non-production environments, it safely checks for `sm_source_availability_ledger` and `sm_amendments` existence before querying, maintaining backwards compatibility with legacy stores.
3. **Canonical SEC Point-in-Time Timestamps & Timezone Conversion:**  
   Helper `_format_canonical_sec_timestamp()` parses ISO 8601 strings and converts all valid timezone-aware representations (both positive and negative offsets, e.g. `+05:30`, `-04:00`, `+00:00`) into canonical UTC with `'Z'` suffix (e.g. `2026-09-18T21:13:29Z`). It strictly refuses date-only fabrication (e.g. `T18:00:00Z` or `T00:00:00Z`) and invalid strings, returning `None`.
4. **Form 4 vs Form 4/A Provenance & Amendment Metadata:**  
   - Inspects `sm_amendments` table to identify amendment accessions, extracting `base_accession` and `amendment_number`.
   - Inspects raw XML `<documentType>` as fallback.
   - Accurately classifies Form 4 (`form: "4"`, `is_amendment: False`) vs Form 4/A (`form: "4/A"`, `is_amendment: True`).
   - Strictly enforces form filtering (`form_types=["4"]` excludes amendments; `form_types=["4/A"]` returns amendments).
5. **Database Exception Sanitization:**  
   User-facing `error_detail` returns safe, generic status messages (`Failed to query filings store for {symbol}.`) without leaking raw database driver exceptions, database usernames, connection strings, or SQL syntax. Full diagnostic information and stack traces are preserved via `logger.error(..., exc_info=True)`.
6. **NVDA `arx_observed_at` Authoritative State Reconciliation:**  
   Authoritative PostgreSQL state confirms:
   - Accession `0001696841-26-000012` (NVDA) was accepted by SEC EDGAR at `2026-09-18T21:13:29Z`.
   - The production canary execution occurred at `2026-11-04T09:23:57Z` (UTC).
   - In PostgreSQL `sm_source_availability_ledger`, `recorded_at = 2026-11-04T09:23:57Z` and `public_availability_at = 2026-09-18T21:13:29Z`.
   - Reconciles conversational prose notation against the authoritative database and canary execution telemetry.
7. **Truthfully Distinguishable Analytical States:**  
   - `AVAILABLE_WITH_RESULTS` (HTTP 200, items: `[...]`): Filings present in store.
   - `AVAILABLE_NO_RESULTS` (HTTP 200, items: `[]`): Known issuer with 0 filings matching requested form types.
   - `CONFIGURATION_BLOCKED` (`SOURCE_NOT_INGESTED`, items: `None`): Unknown or uningested symbol.
   - `BACKEND_STORAGE_ERROR` (`STORAGE_QUERY_FAILED`, items: `None`): Storage connection or query failure.
8. **Zero Outbound SEC Egress in Cache Mode:**  
   Verified via `requests.Session.send` barrier that zero network requests are issued.

---

### 3. Verification & Qualification Results

1. **Targeted SM-ACT-026R Qualification Suite (`tests/test_smart_money_postgres_runtime_integration.py`):**
   - `test_sec_fetcher_cached_envelope_queries_postgres_when_configured`: **PASS**
   - `test_nvda_canary_acceptance_time_and_canonical_provenance`: **PASS** (`0001696841-26-000012` carries `2026-09-18T21:13:29Z`)
   - `test_missing_and_invalid_timestamps_never_invented`: **PASS** (Zero invented timestamps; positive/negative offsets convert to UTC Z; date-only rejected)
   - `test_truthful_distinguishable_states`: **PASS** (`AVAILABLE_WITH_RESULTS` vs `AVAILABLE_NO_RESULTS` vs `CONFIGURATION_BLOCKED` vs `BACKEND_STORAGE_ERROR`)
   - `test_sqlite_compatibility_and_timestamp_handling`: **PASS**
   - `test_zero_sec_network_egress_in_cache_mode`: **PASS** (Zero network egress)
   - `test_advisory_scoring_and_snapshot_containment_preserved`: **PASS** (`PIPELINE_HELD`, `HOLD_SEC_001` preserved)
   - `test_form4_vs_form4a_provenance_and_metadata`: **PASS** (Form 4/A amendment metadata preserved, form filtering exact)
   - `test_database_exception_sanitization_in_api_envelope`: **PASS** (Zero credential / driver leakage in API envelopes)
   - **Result:** **16 passed in 13.84s**.

2. **Existing SEC Failure-Path Regression Suite:**
   - `tests/test_sec_transport_failure_semantics.py` (7 tests): **PASS**
   - `tests/test_sm_act_001_falsification.py` (20 tests): **PASS**
   - **Result:** **27 passed in 11.77s**.

3. **Backend Confluence & Sizing Regression Suite (RQ-03):**
   - `tests/test_directive_057_remediation.py` (20 tests): **PASS**
   - `tests/test_confluence_engine.py` (5 tests): **PASS**
   - `tests/test_smart_money_backup_retention_safety.py` (5 tests): **PASS**
   - **Result:** **30 passed in 2.73s**.

---

### 4. Operational Invariants & Safeguards Preserved

- `HOLD_SEC_001`: **ACTIVE** (Continuous SEC ingestion disabled; worker standby; zero unapproved SEC network egress)
- `HOLD_SCAN_001`: **ACTIVE** (Radar scanner live loop held)
- `HOLD_EXEC_001`: **ACTIVE** (Trade execution locked)
- Sprint 5 Execution Lock: **ACTIVE**
- Active Snapshot `SNAP_SM_1792939989_48f30db6`: **UNMODIFIED**
- Database DDL / Migrations: **0 DDL** (No schema mutation)
