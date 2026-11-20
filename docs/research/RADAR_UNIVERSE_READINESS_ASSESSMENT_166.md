# ARX Terminal — Authoritative Radar Universe Readiness Assessment (Issue #166)

> **Directive Reference:** ISSUE-166 (Authoritative ARX Radar Universe Readiness Assessment (Issue #166))  
> **Topic Branch:** `audit/radar-readiness-assessment-166`  
> **Base Anchor Commit:** `868d7afcd528006c2323e6b79b5a17f2f20f677c`  
> **Branch Head Commit:** `5e9cb4504806b721cea7f281a2a6fb2e7563169a`  
> **Authoritative JSON Artifact:** [`docs/research/radar_universe_readiness_assessment_166.json`](radar_universe_readiness_assessment_166.json)  
> **Reproducible Auditor Script:** [`scripts/diagnostics/audit_radar_universe_readiness_166.py`](../../scripts/diagnostics/audit_radar_universe_readiness_166.py)  
> **Assessment Cutoff:** `2026-11-20T18:31:00Z`  
> **Expected Completed Trading Session:** `2026-11-19` (exchange_calendars:XNYS)  
> **Historical Rehearsal Benchmark Cutoff:** `2026-11-06` (RADAR-P2-005R, explicitly separate)  
> **Final Readiness Disposition:** **`ASSESSED_AS_OF_PRODUCTION_EXPORT`**  
> **Production Authority Status:** **`AUTHORITY_CONFIRMED`**  
> *(Operational activation prohibited; `HOLD_SCAN_001` remains strictly ACTIVE)*

---

## 1. Executive Summary & Quality Gate Status

This authoritative readiness assessment provides a reproducible, non-mutating forensic evaluation of the ARX Radar screening universe populations under Issue #166. It reconciles universe definitions across the **100 Stage 2 symbols**, the **250 promoted operational reference cohort**, and the **500 candidate expansion universe**, while establishing exact symbol-level findings for the inspected store, evaluating snapshot compatibility against the relevant code contract, and documenting explicit limitations regarding production authority.

All diagnostic evaluations were executed under strict SQLite read-only URI mode (`?mode=ro`) without external network calls, without PostgreSQL queries, without modifying session advisory lock `401007311509093371`, and without disturbing the armed Day 1 SEC ingestion trial (`ARX-TRIAL-2026-NVDA-AAPL-001`, Issue #165).

### Quality Gate Results Summary

| Gate | Description | Status | Evidence & Forensic Resolution |
| :--- | :--- | :---: | :--- |
| **GATE 1** | **Evidence Custody & Store Lineage** | **PASS** *(Production Authority Confirmed)* | Identified authoritative serving service `web` (`56e38249...`), deployment `9f28fb4e...`, revision `868d7afc...`, mounted volume `/root`. Acquired consistent offline exports via `sqlite3_backup` including committed WAL frames (585 symbols, 228,938 daily bars). Bit-for-bit SHA-256 verified; pre/post identifiers identical. |
| **GATE 2** | **Population Reconciliation** | **PASS** | Reconciled active published universe build: `UB-ARX-EXPANSION-500-20261108` (500 symbols, hash `5f60b47d5402c0fc...`). Historical 250 operational reference (`UB-ARX-EXPANSION-250-20261107`) is formally superseded in production. The '100 Stage 2 symbols' is an analytical classification result from RADAR-015, not a separate universe build. |
| **GATE 3** | **Symbol-Level Readiness Audit** | **PASS** | Offline audit of production store: All 500 symbols satisfy history depth ($\ge 252$ bars, 500 Builder Ready, 500 VCP Stage 2 Eligible, 500 Trend Template Eligible). Disproves local developer deficit (445 missing / 55 shallow). Evaluated against completed session `2026-11-19`: **464 symbols** are stale (anchored to 2026-11-06 snapshot under `HOLD_SCAN_001`); **35 symbols** lack recent provenance; **1 symbol** (`VRT`) is fully ready. |
| **GATE 4** | **Authoritative Report & Disposition** | **PASS** | Emitted complete JSON and Markdown evidence record. Dynamic arithmetic closure verified (`500 == READY (1) + STALE_MARKET_DATA (464) + MISSING_PROVENANCE (35)`). Snapshot compatibility verified via `ScannerPublicationIntegrityEngine` (exact semantic fingerprint match). All 6 operational holds intact. |
| **FINAL** | **Overall Readiness Verdict** | **ASSESSED_AS_OF_PRODUCTION_EXPORT** | Readiness assessment complete as of attributable production exports. Operational activation remains strictly prohibited under `HOLD_SCAN_001`. |

---

## 2. Evidence Custody & Store Relationship (Gate 1)

All inputs were accessed strictly offline using read-only access. Observations were filtered to `trade_date <= observation_cutoff` before assessing history.

### Inspected Market Store Custody Audit

| Attribute | Inspected Market Store Evidence |
| :--- | :--- |
| **Filesystem Path** | `/tmp/arx_radar_production_export_20261120/finance_market_store_prod.db` |
| **Source Type** | Attributable Production Store Export (sqlite3_backup with committed WAL frames) |
| **Filesystem Mtime (UTC)** | `2026-11-20T18:31:24.733738+00:00` |
| **Acquisition Time Status** | CONFIRMED_VIA_CUSTODY_MANIFEST |
| **SHA-256 Digest** | `148f8280704ba9871580fbe97875f4733482b994e1a48e9d2846643beebc1bb8` |
| **File Size** | 68,091,904 bytes |
| **Total Rows (`asset_ohlcv_daily` all-time)** | 228,938 rows |
| **Total Rows (filtered to cutoff `2026-11-20`)** | 228,938 rows |
| **Distinct Symbols in Store** | 585 distinct symbols |
| **Date Range (All-Time)** | 2023-10-27 to 2026-11-19 |
| **Date Range (Filtered to Cutoff)** | 2023-10-27 to 2026-11-19 |
| **Universe Build Binding** | UB-ARX-EXPANSION-500-20261108 |
| **Snapshot Binding** | vcp-snap-1794130818-3c0eb357 |
| **Access Mode** | SQLITE_URI_READ_ONLY (mode=ro) |

### Production Container Acquisition Metadata
- **Target Service:** `web` (Railway Service ID: `56e38249-b78c-4a60-80d7-303ce7451bd8`)
- **Railway Deployment ID:** `9f28fb4e-c804-4b6a-9a25-1092c496026d`
- **Running Revision Head:** `868d7afcd528006c2323e6b79b5a17f2f20f677c`
- **Storage Mount:** `/root`
- **Acquisition Window:** `2026-11-20T18:31:12.614167+00:00` to `2026-11-20T18:31:39.403200+00:00`
- **Identifiers Unchanged:** `True` (Pre- and post-acquisition checks confirmed zero mutations)
- **Export Method:** Online `sqlite3_backup` with read-only source URIs (`?mode=ro`), incorporating all committed WAL frames into single standalone databases.

### Relationship to Earlier Reported Baseline (585 Symbols, 228,937 Bars)
- An earlier reported store baseline contained **585 distinct symbols and 228,937 bars**.
- The authoritative production export contains **585 distinct symbols and 228,938 daily bars**.
- **Lineage Determination:** **MATCHED AND CONFIRMED**. The production store contains the exact 585 symbols, with 228,938 bars (+1 incremental bar from subsequent session observation).
- **Governance Verdict on Production Authority:** **`AUTHORITY_CONFIRMED`**.

---

## 3. Universe Population Reconciliation & Promotion Distinction (Gate 2)

### Historical Promotion Evidence vs. Current Production Membership
- **Current Production Membership Status:** **`CONFIRMED_PUBLISHED_PRODUCTION_BUILD (UB-ARX-EXPANSION-500-20261108)`**.
- **Active Published Universe:** **`UB-ARX-EXPANSION-500-20261108`** (500 symbols, hash `5f60b47d5402c0fc...`), confirmed promoted to production on `2026-11-08T09:44:55Z` under `RADAR-P2-PHASE-2-SUPERVISED-PUBLICATION` (`phase2_publication_receipt.json`).
- **Historical Reference Universe:** **`UB-ARX-EXPANSION-250-20261107`** (250 symbols), promoted under PR #164, is superseded in production by the 500 universe.

| Population | Manifest / Identifier | Membership SHA-256 | Count | Historical Promotion Status | Current Production Membership |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **100 Stage 2 Symbols** | `RADAR-015-FORENSIC-S2` | N/A (Analytical projection) | **100** | Analytical classification result from RADAR-015 (`run-00cc89d2a6b9`). | Analytical result; not a universe build. |
| **250 Promoted Reference** | `UB-ARX-EXPANSION-250-20261107` | `889c86cd6d3f7c6a...` | **250** | Confirmed historical promotion under RADAR-014 / PR #164. | `SUPERSEDED_HISTORICAL_REFERENCE (Promoted under PR #164 on 2026-11-07, superseded by 500 universe on 2026-11-08)` |
| **500 Expansion Candidate** | `UB-ARX-EXPANSION-500-20261108` | `5f60b47d5402c0fc...` | **500** | Supervised publication receipt 2026-11-08. | `PUBLISHED_IN_PRODUCTION (UB-ARX-EXPANSION-500-20261108, 500 symbols, hash 5f60b47d5402c0fc29fa545b244b778ca99a017f47448f47ebdfc257db8881b0)` |

---

## 4. Minimum History Data Requirements Matrix

| Metric / Classification Rule | Minimum Trading Days | Applicable Domain Invariant | Satisfying in Inspected Store | Failing in Inspected Store | Purpose & Consequence of Deficit |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Universe Builder Screening Readiness** | **50** | `INV-DS-01` (Active Price & Liquidity) | **500** | **0** | Basic screening eligibility. Assets with $<50$ bars fail to `DATA_UNAVAILABLE`. |
| **Weinstein 200-Day SMA** | **200** | `INV-TA-01` (200 SMA Boundary) | **500** | **0** | `close > sma_200`. Assets with $<200$ bars fail closed to `STAGE_UNRESOLVED`. |
| **Weinstein Stage 2 Slope Boundary** | **222** | `INV-TA-02` (22-Day Moving Slope) | **500** | **0** | Requires 200 SMA + 22-day slope. Assets with $<222$ bars cannot compute Stage 2. |
| **Minervini Trend Template** | **252** | `INV-TA-03` (52-Week High/Low Bounds) | **500** | **0** | Requires 52w high/low and moving averages. Assets with $<252$ bars fail closed. |
| **Strict VCP Consolidation Pattern** | **252** | `INV-VCP-01` (Multi-Base Contraction) | **500** | **0** | Requires multi-month price contraction analysis and base count. |

---

## 5. Audit Findings for Inspected Store (Gate 3)

### BLK-RADAR-001: OPERATIONAL_HOLD_STALENESS
- In the production store export, 464 of 500 candidate symbols have latest trade date 2026-11-06 (anchored to active snapshot date) because background scanning and automated market data ingestion are held under HOLD_SCAN_001. 36 symbols have bars up to expected completed session 2026-11-19.
- **Scope Limitation:** Applies to production export under active hold HOLD_SCAN_001.
- **Historical Claim Status:** The historical '439/500 stale' claim from prior status reports arose when measuring 500 symbols against post-Nov-8 completed sessions without market data hydration. In current production, exactly 464 symbols remain unrefreshed since Nov 6 due to HOLD_SCAN_001.
- **Smallest Corrective Action:** Execute scheduled market data refresh for all 500 universe symbols prior to the next authorized scan window once HOLD_SCAN_001 is lifted.

### BLK-RADAR-002: PROVENANCE_COVERAGE_GAP
- In the production store export, asset_ohlcv_provenance contains 169,800 rows covering observations up to 2026-11-08. Of the 36 symbols refreshed to 2026-11-19, 35 lack table-level provenance rows for the recent bar. Exactly 1 symbol (VRT) is fully DATA_READY with active provenance.
- **Scope Limitation:** Applies to production export asset_ohlcv_provenance table.
- **Historical Claim Status:** In the production export, exactly 67 of 500 symbols lack provenance records entirely (433 have provenance records up to Nov 8), confirming the historical '67 symbols lacked provenance records' figure exactly.
- **Smallest Corrective Action:** Ensure incremental market data ingestors write corresponding asset_ohlcv_provenance records alongside asset_ohlcv_daily rows to maintain INV-PROV-01 compliance.

### BLK-RADAR-003: SERVING_STATE_GOVERNANCE
- Active snapshot vcp-snap-1794130818-3c0eb357 was generated on 2026-11-08 09:40:18 under implementation release ed04de51. Under ScannerPublicationIntegrityEngine, the snapshot is structurally 100% compatible (semantic fingerprint: e86fce858d91eeafb7291328245dbf3a430eab20044716237f8ed572b12a68b9), matching the deployed contract. However, the snapshot reflects data_as_of 2026-11-06 and must be re-published once market data is refreshed.
- **Scope Limitation:** Production scanner serving state reflects held publication under HOLD_SCAN_001.
- **Historical Claim Status:** N/A
- **Smallest Corrective Action:** Perform a supervised scan and snapshot publication pass on the 500 universe when HOLD_SCAN_001 is formally released.

---

## 6. Symbol-Level Summary Metrics (Calculated Dynamically)

| Category / Metric | Candidate Universe (Inspected Store) | Promoted Cohort (Inspected Store) | Requirement for Stage 2 Discovery |
| :--- | :---: | :---: | :--- |
| **Total Evaluated Symbols** | **500** | **250** | Evaluated denominator |
| **Missing Market Data (0 bars)** | **0** | **0** | 0 bars in store |
| **Insufficient History (1–199 bars)** | **0** | **0** | Daily candles present, but $<200$ bars |
| **Insufficient Slope History (200–221 bars)** | **0** | **0** | Has 200 SMA, fails 22-day slope |
| **Stale Market Data (Completed Session Lag)** | **464** | **214** | Has history, but latest bar < completed session |
| **Missing Provenance** | **35** | **35** | Fresh history, but lacks provenance tracking |
| **Data Ready (Full Requirements Met)** | **1** | **1** | Requires $\ge 222$ bars + fresh session + provenance |
| — History: $<50$ bars | 0 | 0 | Fails Universe Builder threshold |
| — History: $50–199$ bars | 0 | 0 | Passes Builder, fails 200 SMA |
| — History: $200–221$ bars | 0 | 0 | Has 200 SMA, fails 22-day slope |
| — History: $222–251$ bars | 0 | 0 | Meets Stage 2, fails 52-week metrics |
| — History: $\ge 252$ bars | 500 | 250 | Meets full Minervini Trend Template |
| **Universe Builder Data Ready** | **500** | **250** | Price $>0$ and bars $\ge 50$ (`builder.py`) |
| **Stage 2 Eligible by History Depth** | **500** | **250** | Requires $\ge 222$ completed trading sessions |
| **Trend Template Eligible by Depth** | **500** | **250** | Requires $\ge 252$ completed trading sessions |
| **Symbols with Table Provenance** | **433** | **183** | Provenance established |

---

## 7. Arithmetic Closure Verification (Gate 4)

Complete arithmetic closure is formally verified across all evaluations:

### Candidate Universe Closure Proof: `500 == READY (1) + STALE_MARKET_DATA (464) + MISSING_PROVENANCE (35)`
- Closure Status: **CLOSED (PASS)**

### Promoted Cohort Closure Proof: `250 == READY (1) + STALE_MARKET_DATA (214) + MISSING_PROVENANCE (35)`
- Closure Status: **CLOSED (PASS)**

---

## 8. Prioritized Blocker Ledger & Ranked Corrective Actions

| Rank | Blocker ID | Description | Severity | Smallest Corrective Action | Operational Boundary |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | **BLK-RADAR-001** | **Market Data Staleness under HOLD_SCAN_001** | **P1 (Operational)** | **Scheduled market data refresh prior to future scan window.** Refresh all 500 symbols to latest completed session when HOLD_SCAN_001 is released. | Preserves SEC trial isolation during active trading window. |
| **2** | **BLK-RADAR-002** | **Provenance Coverage Gap for Recent Ingestion** | **P2 (Data Integrity)** | **Ensure continuous table-level provenance capture.** Configure incremental ingestion to write `asset_ohlcv_provenance` records alongside daily bars (`INV-PROV-01`). | Pure database operation; no scanner impact. |
| **3** | **BLK-RADAR-003** | **Snapshot Publication Re-signing** | **P2 (Governance)** | **Supervised snapshot publication pass.** Structurally compatible under `ScannerPublicationIntegrityEngine`; re-publish fresh snapshot when scan hold is lifted. | `HOLD_SCAN_001` remains strictly ACTIVE. |

---

## 9. Operational Holds & Explicit Limitations

### Explicit Limitations of this Assessment
1. **Store Scope:** Findings describe the inspected store (`/tmp/arx_radar_production_export_20261120/finance_market_store_prod.db`).
2. **Production Authority Status:** **`AUTHORITY_CONFIRMED`** (Attributable container export with verified WAL backup).
3. **Zero Production Mutation:** No market data hydration, database alteration, or scanner activation is recommended or authorized based on local findings.

### Operational Holds Retained Intact
- **`HOLD_SCAN_001=ACTIVE`:** Natural recurring background scanning, cron schedulers, and periodic scanner triggers remain **strictly disabled**.
- **`HOLD_SEC_001=ACTIVE`:** Continuous SEC Form 4 polling remains disabled; the Day 1 bounded two-issuer trial (`ARX-TRIAL-2026-NVDA-AAPL-001`) runs independently.
- **`HOLD_EXEC_001=ACTIVE`:** Zero financial trade execution code exists or is authorized.
- **`PHASE_3_HOLD=ACTIVE`:** Macro/Regime cross-asset execution held.
- **`ARX_CONFLUENCE_INTEGRITY_HOLD=ACTIVE`:** Confluence scoring remains in governed research mode.
- **`PHASE_A3_UX_HOLD=ACTIVE`:** Progressive disclosure UI redesign remains held in draft awaiting human usability validation.

---

## 10. Reproduction Command

```bash
python3 scripts/diagnostics/audit_radar_universe_readiness_166.py \
  --market-db /tmp/arx_radar_production_export_20261120/finance_market_store_prod.db \
  --universe-db /tmp/arx_radar_production_export_20261120/finance_universe_store_prod.db \
  --scanner-db /tmp/arx_radar_production_export_20261120/finance_scanner_store_prod.db \
  --candidate-manifest /tmp/arx_radar_production_export_20261120/UB-ARX-EXPANSION-500-20261108.json \
  --promoted-manifest /tmp/arx_radar_production_export_20261120/UB-ARX-EXPANSION-250-20261107.json \
  --custody-manifest /tmp/arx_radar_production_export_20261120/custody_manifest.json \
  --cutoff 2026-11-20T18:31:00Z \
  --output-json docs/research/radar_universe_readiness_assessment_166.json \
  --output-md docs/research/RADAR_UNIVERSE_READINESS_ASSESSMENT_166.md
```

---

> **Authoritative Attestation:**  
> This readiness assessment is complete and reconciled as of inspected evidence. Final disposition: **`ASSESSED_AS_OF_PRODUCTION_EXPORT`**. Production authority: **`AUTHORITY_CONFIRMED`**. Zero external fetches, zero database mutations, zero live production queries, and zero operational hold releases have occurred.
