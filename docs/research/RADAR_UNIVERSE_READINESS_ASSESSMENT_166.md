# ARX Terminal — Authoritative Radar Universe Readiness Assessment (Issue #166)

> **Directive Reference:** ISSUE-166 (Authoritative ARX Radar Universe Readiness Assessment (Issue #166))  
> **Topic Branch:** `audit/radar-readiness-assessment-166`  
> **Base Anchor Commit:** `868d7afcd528006c2323e6b79b5a17f2f20f677c`  
> **Branch Head Commit:** `ef63761a44848fddf795edcb9701614ba61cd076`  
> **Authoritative JSON Artifact:** [`docs/research/radar_universe_readiness_assessment_166.json`](radar_universe_readiness_assessment_166.json)  
> **Reproducible Auditor Script:** [`scripts/diagnostics/audit_radar_universe_readiness_166.py`](../../scripts/diagnostics/audit_radar_universe_readiness_166.py)  
> **Assessment Cutoff:** `2026-11-20`  
> **Expected Completed Trading Session:** `2026-11-20` (exchange_calendars:XNYS)  
> **Historical Rehearsal Benchmark Cutoff:** `2026-11-06` (RADAR-P2-005R, explicitly separate)  
> **Final Readiness Disposition:** **`ASSESSED_AS_OF_EVIDENCE / BLOCKED_BY_EVIDENCE`**  
> **Production Authority Status:** **`NOT_CHECKED`**  
> *(Operational activation prohibited; `HOLD_SCAN_001` remains strictly ACTIVE)*

---

## 1. Executive Summary & Quality Gate Status

This authoritative readiness assessment provides a reproducible, non-mutating forensic evaluation of the ARX Radar screening universe populations under Issue #166. It reconciles universe definitions across the **100 Stage 2 symbols**, the **250 promoted operational reference cohort**, and the **500 candidate expansion universe**, while establishing exact symbol-level findings for the inspected store, evaluating snapshot compatibility against the relevant code contract, and documenting explicit limitations regarding production authority.

All diagnostic evaluations were executed under strict SQLite read-only URI mode (`?mode=ro`) without external network calls, without PostgreSQL queries, without modifying session advisory lock `401007311509093371`, and without disturbing the armed Day 1 SEC ingestion trial (`ARX-TRIAL-2026-NVDA-AAPL-001`, Issue #165).

### Quality Gate Results Summary

| Gate | Description | Status | Evidence & Forensic Resolution |
| :--- | :--- | :---: | :--- |
| **GATE 1** | **Evidence Custody & Store Lineage** | **PASS** *(Store Identified)*<br>**NOT_CHECKED** *(Production Authority)* | Inspected SQLite database (`/Users/iDavid/.finance_market_store.db`, SHA-256 `ce911318...`, 61 symbols, 7951 daily bars). Relationship to earlier reported baseline (585 symbols, 228937 bars) cannot be established from local artifacts; production authority is marked **NOT_CHECKED**. Filesystem mtime recorded; acquisition time is unrecorded. |
| **GATE 2** | **Population Reconciliation** | **PASS** | Formally resolved universe definitions: the '100 Stage 2 symbols' is an **analytical classification result** from RADAR-015 (`run-00cc89d2a6b9`), not an independent universe build. Historical promotion evidence (RADAR-014 / PR #164) is described separately from current production membership (**NOT_CHECKED**). Overlap (250 symbols) does not constitute production promotion for the 500 candidate universe. |
| **GATE 3** | **Symbol-Level Readiness Audit** | **PASS** | Reconciled symbol-level records dynamically for all candidate symbols: **445 symbols** have 0 daily bars (`MISSING_MARKET_DATA`); **55 symbols** have insufficient 200 SMA history; **0 symbols** have insufficient slope history; **0 symbols** have stale completed sessions; **0 symbols** are ready. Zero symbols produce READY without complete evidence. |
| **GATE 4** | **Authoritative Report & Disposition** | **PASS** | Emitted machine-readable JSON and Markdown assessment. Arithmetic closure verified dynamically. Snapshot compatibility evaluated against the 6-component version tuple contract rather than commit SHAs alone. Operational holds preserved. |
| **FINAL** | **Overall Readiness Verdict** | **ASSESSED_AS_OF_EVIDENCE / BLOCKED_BY_EVIDENCE** | Complete as of inspected evidence. Blocked from operational activation by confirmed data deficits in inspected store and unverified production authority (`NOT_CHECKED`). Next dependency is an attributable, consistent export of the actual Radar stores. |

---

## 2. Evidence Custody & Store Relationship (Gate 1)

All inputs were accessed strictly offline using read-only access. Observations were filtered to `trade_date <= observation_cutoff` before assessing history.

### Inspected Market Store Custody Audit

| Attribute | Inspected Market Store Evidence |
| :--- | :--- |
| **Filesystem Path** | `/Users/iDavid/.finance_market_store.db` |
| **Source Type** | Local developer SQLite file |
| **Filesystem Mtime (UTC)** | `2026-11-14T19:26:22.069878+00:00` |
| **Acquisition Time Status** | NOT_RECORDED (filesystem mtime only; provenance/acquisition time unrecorded in store) |
| **SHA-256 Digest** | `ce911318619d9cdfb8984ada5ed72d3b1f1b2e1a72ddc9e6cecf0b1ae0ef1ba9` |
| **File Size** | 1,028,096 bytes |
| **Total Rows (`asset_ohlcv_daily` all-time)** | 7,951 rows |
| **Total Rows (filtered to cutoff `2026-11-20`)** | 7,951 rows |
| **Distinct Symbols in Store** | 61 distinct symbols |
| **Date Range (All-Time)** | 2025-02-10 to 2026-11-13 |
| **Date Range (Filtered to Cutoff)** | 2025-02-10 to 2026-11-13 |
| **Universe Build Binding** | UNBOUND (no build ID or manifest hash recorded in store) |
| **Snapshot Binding** | UNBOUND (0 rows in asset_factor_snapshots, 0 rows in asset_ohlcv_provenance) |
| **Access Mode** | SQLITE_URI_READ_ONLY (mode=ro) |

### Relationship to Earlier Reported Baseline (585 Symbols, 228,937 Bars)
- An earlier reported store baseline contained **585 distinct symbols and 228,937 bars** (reflecting a fully hydrated market dataset from historical testing).
- The inspected store contains **61 distinct symbols and 7,951 daily bars**.
- **Lineage Determination:** The relationship between the inspected file and the earlier baseline store cannot be established from available local artifacts.
- **Governance Verdict on Production Authority:** **`NOT_CHECKED`**.
  Because local artifacts cannot verify that the inspected store reflects authoritative production market data, production market data authority is recorded as **NOT_CHECKED**.

---

## 3. Universe Population Reconciliation & Promotion Distinction (Gate 2)

### Historical Promotion Evidence vs. Current Production Membership
- **Historical Promotion Evidence:** RADAR-014 / PR #164 merged the 250-symbol manifest (`UB-ARX-EXPANSION-250-20261107`) as the operational reference universe.
- **Current Production Membership Status:** **`NOT_CHECKED`**. Live production membership cannot be confirmed without live database verification or an attributable signed export.
- **Candidate Universe Status:** **`PROVISIONAL_SPECIFICATION / REHEARSAL_QUALIFIED (UNPROMOTED)`**. Provisional expansion specification under RADAR-P2-003. Membership overlap (250 symbols) does NOT constitute production promotion; the 500 candidate universe remains unpromoted.

| Population | Manifest / Identifier | Membership SHA-256 | Count | Historical Promotion Status | Current Production Membership |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **100 Stage 2 Symbols** | `RADAR-015-FORENSIC-S2` | N/A (Analytical projection) | **100** | Analytical classification result from RADAR-015 (`run-00cc89d2a6b9`). | Analytical result; not a universe build. |
| **250 Promoted Reference** | `UB-ARX-EXPANSION-250-20261107` | `889c86cd6d3f7c6a...` | **250** | Confirmed historical promotion under RADAR-014 / PR #164. | **NOT_CHECKED** (unverified live state) |
| **500 Expansion Candidate** | `UB-ARX-EXPANSION-500-REHEARSAL` | `5f60b47d5402c0fc...` | **500** | Provisional specification (RADAR-P2-003). | **NOT_PROMOTED** (rehearsal only) |

---

## 4. Minimum History Data Requirements Matrix

| Metric / Classification Rule | Minimum Trading Days | Applicable Domain Invariant | Satisfying in Inspected Store | Failing in Inspected Store | Purpose & Consequence of Deficit |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Universe Builder Screening Readiness** | **50** | `INV-DS-01` (Active Price & Liquidity) | **55** | **445** | Basic screening eligibility. Assets with $<50$ bars fail to `DATA_UNAVAILABLE`. |
| **Weinstein 200-Day SMA** | **200** | `INV-TA-01` (200 SMA Boundary) | **0** | **500** | `close > sma_200`. Assets with $<200$ bars fail closed to `STAGE_UNRESOLVED`. |
| **Weinstein Stage 2 Slope Boundary** | **222** | `INV-TA-02` (22-Day Moving Slope) | **0** | **500** | Requires 200 SMA + 22-day slope. Assets with $<222$ bars cannot compute Stage 2. |
| **Minervini Trend Template** | **252** | `INV-TA-03` (52-Week High/Low Bounds) | **0** | **500** | Requires 52w high/low and moving averages. Assets with $<252$ bars fail closed. |
| **Strict VCP Consolidation Pattern** | **252** | `INV-VCP-01` (Multi-Base Contraction) | **0** | **500** | Requires multi-month price contraction analysis and base count. |

---

## 5. Audit Findings for Inspected Store (Gate 3)

### Finding 1: Market Data Deficit in Inspected Database
- In the inspected store, 445 candidate symbols have 0 bars (MISSING_MARKET_DATA), 55 have insufficient 200 SMA history, and 0 have insufficient slope history. Only 0 symbols satisfy the 200 SMA threshold.
- **Scope Limitation:** Finding applies strictly to inspected store (/Users/iDavid/.finance_market_store.db). Production market store authority is NOT_CHECKED.
- **Historical Claim Status:** The historical '439/500 stale' claim from prior status reports remains an unverified historical measurement whose calculation basis is not established by local artifacts.
- **Local Verdict:** **LOCAL_DATA_BLOCKER**.

### Finding 2: Missing Provenance Records in Inspected Database
- In the inspected store, asset_ohlcv_provenance contains 0 rows. 500 of 500 candidate symbols lack table-level provenance.
- **Scope Limitation:** Finding applies strictly to inspected store. Production provenance authority is NOT_CHECKED.
- **Historical Claim Status:** The historical '67 symbols lacked provenance' claim remains an unverified historical figure whose specific symbol roster cannot be confirmed from local artifacts.
- **Local Verdict:** **LOCAL_DATA_BLOCKER**.

### Finding 3: Snapshot Compatibility Under Relevant Code Contract
- Under the publication integrity contract (ScannerPublicationIntegrityEngine), snapshot compatibility is governed by the 6-component ScannerVersionTuple, component content hashes, and semantic fingerprint, rather than git commit SHAs alone. Different commit SHAs alone across non-functional PRs do not establish incompatibility. However, the inspected scanner store contains 1 snapshots (0 VCP snapshots). Local snapshot compatibility cannot be established.
- **Scope Limitation:** Production snapshot serving state is NOT_CHECKED.
- **Local Verdict:** **GOVERNANCE_AND_STORAGE_BLOCKER**.

---

## 6. Symbol-Level Summary Metrics (Calculated Dynamically)

| Category / Metric | Candidate Universe (Inspected Store) | Promoted Cohort (Inspected Store) | Requirement for Stage 2 Discovery |
| :--- | :---: | :---: | :--- |
| **Total Evaluated Symbols** | **500** | **250** | Evaluated denominator |
| **Missing Market Data (0 bars)** | **445** | **195** | 0 bars in store |
| **Insufficient History (1–199 bars)** | **55** | **55** | Daily candles present, but $<200$ bars |
| **Insufficient Slope History (200–221 bars)** | **0** | **0** | Has 200 SMA, fails 22-day slope |
| **Stale Market Data (Completed Session Lag)** | **0** | **0** | Has history, but latest bar < completed session |
| **Missing Provenance** | **0** | **0** | Fresh history, but lacks provenance tracking |
| **Data Ready (Full Requirements Met)** | **0** | **0** | Requires $\ge 222$ bars + fresh session + provenance |
| — History: $<50$ bars | 0 | 0 | Fails Universe Builder threshold |
| — History: $50–199$ bars | 55 | 55 | Passes Builder, fails 200 SMA |
| — History: $200–221$ bars | 0 | 0 | Has 200 SMA, fails 22-day slope |
| — History: $222–251$ bars | 0 | 0 | Meets Stage 2, fails 52-week metrics |
| — History: $\ge 252$ bars | 0 | 0 | Meets full Minervini Trend Template |
| **Universe Builder Data Ready** | **55** | **55** | Price $>0$ and bars $\ge 50$ (`builder.py`) |
| **Stage 2 Eligible by History Depth** | **0** | **0** | Requires $\ge 222$ completed trading sessions |
| **Trend Template Eligible by Depth** | **0** | **0** | Requires $\ge 252$ completed trading sessions |
| **Symbols with Table Provenance** | **0** | **0** | Provenance established |

---

## 7. Arithmetic Closure Verification (Gate 4)

Complete arithmetic closure is formally verified across all evaluations:

### Candidate Universe Closure Proof: `500 == INSUFFICIENT_HISTORY (55) + MISSING_MARKET_DATA (445)`
- Closure Status: **CLOSED (PASS)**

### Promoted Cohort Closure Proof: `250 == INSUFFICIENT_HISTORY (55) + MISSING_MARKET_DATA (195)`
- Closure Status: **CLOSED (PASS)**

---

## 8. Prioritized Blocker Ledger & Next Production Dependencies

| Rank | Blocker ID | Description | Severity | Next Dependency Specification | Operational Boundary |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | **BLK-RADAR-001** | **Production Authority Unverified (NOT_CHECKED)** | **P0** | **Obtain an attributable, consistent export of the actual production Radar stores.** Production hydration must NOT be recommended or executed based on local developer store deficits. | Preserves SEC trial isolation during active trading window. |
| **2** | **BLK-RADAR-002** | **Market Data Deficits in Inspected Store** | **P0** | Upon receiving the attributable production export, assess whether production market stores satisfy the 222-session Stage 2 threshold. | `HOLD_SCAN_001` remains strictly ACTIVE. |
| **3** | **BLK-RADAR-003** | **Zero Table-Level Provenance in SQLite** | **P0** | Verify whether production stores maintain `asset_ohlcv_provenance` records with vendor attribution and observation precision. | Enforces `INV-PROV-01` (Attributable Market Evidence). |
| **4** | **BLK-RADAR-004** | **Snapshot Contract Compatibility** | **P1** | Verify snapshot serving state against the 6-component `ScannerVersionTuple` under `ScannerPublicationIntegrityEngine`. | Aligns scanner endpoint with publication integrity contract. |

---

## 9. Operational Holds & Explicit Limitations

### Explicit Limitations of this Assessment
1. **Store Scope:** Findings describe the inspected store (`/Users/iDavid/.finance_market_store.db`).
2. **Production Authority Unverified:** Production database status is **`NOT_CHECKED`** because live production queries were not executed in this turn.
3. **Zero Production Mutation:** No market data hydration, database alteration, or scanner activation is recommended or authorized based on local findings.

### Operational Holds Retained Intact
- **`HOLD_SCAN_001=ACTIVE`:** Natural recurring background scanning, cron schedulers, and periodic scanner triggers remain **strictly disabled**.
- **`HOLD_SEC_001=ACTIVE`:** Continuous SEC Form 4 polling remains disabled; the Day 1 bounded two-issuer trial (`ARX-TRIAL-2026-NVDA-AAPL-001`) runs independently.
- **`HOLD_EXEC_001=ACTIVE`:** Zero financial trade execution code exists or is authorized.
- **`ARX_CONFLUENCE_INTEGRITY_HOLD=ACTIVE`:** Confluence scoring remains in governed research mode.
- **`PHASE_A3_UX_HOLD=ACTIVE`:** Progressive disclosure UI redesign remains held in draft awaiting human usability validation.

---

> **Authoritative Attestation:**  
> This readiness assessment is complete and reconciled as of inspected evidence. Final disposition: **`ASSESSED_AS_OF_EVIDENCE / BLOCKED_BY_EVIDENCE`**. Production authority: **`NOT_CHECKED`**. Zero external fetches, zero database mutations, zero live production queries, and zero operational hold releases have occurred.
