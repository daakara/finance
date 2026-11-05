# ARX Terminal — SM-ACT-026 SEC Evidence Visibility Repair & Qualification
## Directive: SM-ACT-026 PostgreSQL Evidence Visibility Repair & Production Qualification

**Tracker:** GitHub Issue #165 (`daakara/finance`)  
**Production Commit Baseline:** [`649d31511c79324dbd1cb7369ffacac1c36ea75f`](https://github.com/daakara/finance/commit/649d31511c79324dbd1cb7369ffacac1c36ea75f) (PR #190)  
**Evaluated Filing:** Accession `0001696841-26-000012` (NVDA, Timothy S. Teter, Code F)  
**Execution Timestamp (UTC):** `2026-11-05T00:15:00+00:00`  
**Binding Disposition:** **`SMART_MONEY_READ_ONLY_EVIDENCE_VISIBILITY_QUALIFIED_FOR_RELEASE`**  

---

### 1. Production Baseline Reconciliation

- **Production Anchor:** `649d31511c79324dbd1cb7369ffacac1c36ea75f` (PR #190 merged into `release/production`).
- **Included Anchors:** Contains PR #184 (`d247ee78`: fail-closed retention protection) and PR #190 (VCP bitemporal cutoff fix).
- **Branch Lineage:** Review branch `fix/sm-act-026-postgres-evidence-visibility` branched directly from `649d31511c79324dbd1cb7369ffacac1c36ea75f`.

---

### 2. Surgical Projection Implementation

1. **Authoritative Storage Backend Selection:**  
   `SecEdgarFetcher._get_cached_recent_filings_envelope()` calls `get_storage_backend()`. When configured with `ARX_STORAGE_BACKEND=postgres` or production URL, it executes parameterized queries against PostgreSQL `form4_raw_filings` with a LEFT JOIN on `sm_source_availability_ledger`.
2. **Backwards-Compatible SQLite Fallback:**  
   When `ARX_STORAGE_BACKEND=sqlite` in non-production environments, it safely checks for `sm_source_availability_ledger` existence before querying, maintaining backwards compatibility with legacy stores.
3. **Canonical SEC Point-in-Time Timestamps:**  
   Helper `_format_canonical_sec_timestamp()` enforces ISO 8601 UTC representation with `'Z'` suffix. It strictly refuses date-only fabrication (e.g. `T18:00:00Z` or `T00:00:00Z`), returning `None` when timestamps are absent.
4. **Truthfully Distinguishable Analytical States:**  
   - `AVAILABLE_WITH_RESULTS` (HTTP 200, items: `[...]`): Filings present in store.
   - `AVAILABLE_NO_RESULTS` (HTTP 200, items: `[]`): Known issuer with 0 filings matching requested form types.
   - `CONFIGURATION_BLOCKED` (`SOURCE_NOT_INGESTED`, items: `None`): Unknown or uningested symbol.
   - `BACKEND_STORAGE_ERROR` (`STORAGE_QUERY_FAILED`, items: `None`): Storage connection or query failure.
5. **Zero Outbound SEC Egress in Cache Mode:**  
   Verified via `requests.Session.send` barrier that zero network requests are issued.

---

### 3. Verification & Qualification Results

1. **Targeted SM-ACT-026 Qualification Suite (`tests/test_smart_money_postgres_runtime_integration.py`):**
   - `test_sec_fetcher_cached_envelope_queries_postgres_when_configured`: **PASS**
   - `test_nvda_canary_acceptance_time_and_canonical_provenance`: **PASS** (`0001696841-26-000012` carries `2026-09-18T21:13:29Z`)
   - `test_missing_and_invalid_timestamps_never_invented`: **PASS** (Zero invented timestamps)
   - `test_truthful_distinguishable_states`: **PASS** (`AVAILABLE_WITH_RESULTS` vs `AVAILABLE_NO_RESULTS` vs `CONFIGURATION_BLOCKED` vs `BACKEND_STORAGE_ERROR`)
   - `test_sqlite_compatibility_and_timestamp_handling`: **PASS**
   - `test_zero_sec_network_egress_in_cache_mode`: **PASS** (Zero network egress)
   - `test_advisory_scoring_and_snapshot_containment_preserved`: **PASS** (`PIPELINE_HELD`, `HOLD_SEC_001` preserved)
   - **Result:** **14 passed in 12.72s**.

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
