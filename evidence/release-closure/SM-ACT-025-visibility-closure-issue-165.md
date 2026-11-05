# ARX Terminal — SM-ACT-025 Production Visibility & Advisory Authority Closure
## Directive: SM-ACT-025 Production Visibility & Advisory Authority Closure

**Tracker:** GitHub Issue #165 (`daakara/finance`)  
**Production Commit Baseline:** [`d247ee78234139308e2c611807d5efb8dd6ccc63`](https://github.com/daakara/finance/commit/d247ee78234139308e2c611807d5efb8dd6ccc63) (SM-ACT-020, PR #184)  
**Evaluated Filing:** Accession `0001696841-26-000012` (NVDA, Timothy S. Teter, Code F)  
**Execution Timestamp (UTC):** `2026-11-05T00:00:00+00:00`  
**Primary Technical Verdict:** **`SMART_MONEY_INTEGRATION_DEFECT_CONFIRMED`**  
**Advisory Authority Governance Verdict:** **`SMART_MONEY_PUBLICATION_AUTHORIZATION_REQUIRED`**  

---

### 1. Evidence & Scoring Reconciliation

- **Non-Conviction Invariant:** Accession `0001696841-26-000012` ($7.58M tax withholding) is parsed as `TAX_WITHHOLDING`, `is_positive_buying_conviction = False`. In `SmartMoneyEngine.evaluate_smart_money_feeds('NVDA')`, Code F is strictly excluded from buy/sell volume, producing `directional_bias: NEUTRAL`.
- **Neutral Confluence Scoring:** Score 55.0 yields `status: "neutral"` in `ConfluenceEngine.calculate_confluence()`. It cannot trigger `has_insider_buy` or positive confluence conviction.
- **Dynamic Reweighting Preserved:** Under `HOLD_SEC_001`, `_build_smart_money_confluence_inputs` returns `None`. `ConfluenceEngine` redistributes pillar weights proportionally (Technical 35%, Fundamental 35%, Macro 30%) with zero false score dilution.
- **PIT Temporal Invariant Empirically Proven:**
  - `PUBLIC_AVAILABLE` as-of `2026-09-17` (pre-acceptance): Canary **absent** (0 total obs).
  - `PUBLIC_AVAILABLE` as-of `2026-09-19` (post-acceptance): Canary **present** (3 total obs).
  - `ARX_OBSERVED` as-of `2026-11-03` (pre-canary ingestion): Canary **absent** (7 total obs).
  - `ARX_OBSERVED` as-of `2026-11-04T12:00Z` (post-canary ingestion): Canary **present** (9 total obs).

---

### 2. Visibility Boundary Resolution (Controls A–D)

- **Control A (Intentional Publication Hold):** `HOLD_SEC_001` intentionally pauses live continuous SEC polling and advisory scoring. Single-issuer canary scoring is quarantined from mutating active snapshot `SNAP_SM_1792939989_48f30db6`. **OPERATING CORRECTLY.**
- **Control B (Read-Only Historical Evidence Projection):** `SecEdgarFetcher._get_cached_recent_filings_envelope()` hardcoded SQLite queries and lacked PostgreSQL backend query code. In production (`SMART_MONEY_STORAGE_BACKEND=postgres`), `GET /api/v1/smart-money/sec-filings/NVDA` emitted `CONFIGURATION_BLOCKED` despite 7 valid NVDA filings in PostgreSQL. **CONFIRMED INTEGRATION DEFECT.**
- **Control C (Snapshot-to-API Integration):** `GET /api/v1/screener/smart-money/snapshot` serves active snapshot `SNAP_SM_1792939989_48f30db6` with 10 results. **OPERATING CORRECTLY.**
- **Control D (Assessment Authority):** Smart Money is purely advisory; execution is gated by `HOLD_EXEC_001`. **OPERATING CORRECTLY.**

---

### 3. Minimum-Change Implementation Preparation

- **Surgical Patch:** Added PostgreSQL query support to `SecEdgarFetcher._get_cached_recent_filings_envelope()` in `analyst_dashboard/data/sec_edgar_fetcher.py`. When `get_storage_backend() == "postgres"`, cached filings are projected directly from `form4_raw_filings` with full provenance, timestamps, and SEC URLs in cache-only mode.
- **Automated Test Results:**
  - `tests/test_sec_transport_failure_semantics.py` + `tests/test_sm_act_001_falsification.py`: **27 passed, 0 failed** (12.32s).
  - `tests/test_smart_money_postgres_runtime_integration.py` (`test_sec_fetcher_cached_envelope_queries_postgres_when_configured`): **1 passed, 0 failed** (14.68s).

---

### 4. Release Decision & Governance State

- **Remaining Work:**
  1. For user-facing read-only visibility: **A small API presentation correction** (deploying the surgical `SecEdgarFetcher` patch).
  2. For advisory trading signals and snapshot publication: **Controlled publication authorization only** (requires owner decision to authorize 10-issuer universe calibration or maintain operational containment).
- **Active Operational Holds:**
  - `HOLD_SEC_001`: **ACTIVE** (Continuous ingestion disabled; worker in `STANDBY`)
  - `HOLD_SCAN_001`: **ACTIVE** (Radar live loop held)
  - `HOLD_EXEC_001`: **ACTIVE** (Trade execution locked)
  - Sprint 5 Execution Lock: **ACTIVE**
- **Binding Verdicts:**
  - **`SMART_MONEY_INTEGRATION_DEFECT_CONFIRMED`** (Technical diagnosis of deployed baseline `d247ee78`)
  - **`SMART_MONEY_PUBLICATION_AUTHORIZATION_REQUIRED`** (Advisory activation governance boundary)
