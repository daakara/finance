# ARX Terminal — Authoritative Radar Universe Readiness Assessment (Issue #166)

> **Directive Reference:** Issue #166 (ARX Radar Screening Universe & Readiness Re-evaluation)  
> **Topic Branch:** `audit/radar-readiness-assessment-166`  
> **Base Anchor Commit:** `868d7afcd528006c2323e6b79b5a17f2f20f677c` (`origin/release/production`, PR #312)  
> **Authoritative JSON Artifact:** [`docs/research/radar_universe_readiness_assessment_166.json`](radar_universe_readiness_assessment_166.json)  
> **Reproducible Auditor Script:** [`scripts/diagnostics/audit_radar_universe_readiness_166.py`](../../scripts/diagnostics/audit_radar_universe_readiness_166.py)  
> **Evaluation As-Of Date:** `2026-11-20`  
> **Target Rehearsal Session Date:** `2026-11-06` (RADAR-P2-005R)  
> **Latest Inspected Store Date:** `2026-11-13`  
> **Final Readiness Disposition:** **`ASSESSED_AS_OF_EVIDENCE` / `BLOCKED_BY_EVIDENCE`**  
> **Production Authority Status:** **`NOT_CHECKED`**  
> *(Operational activation prohibited; `HOLD_SCAN_001` remains strictly ACTIVE)*

---

## 1. Executive Summary & Quality Gate Status

This authoritative readiness assessment provides a reproducible, non-mutating forensic evaluation of the ARX Radar screening universe populations under Issue #166. It reconciles universe definitions across the **100 Stage 2 symbols**, the **250 promoted operational universe**, and the **500 candidate expansion universe**, while establishing exact symbol-level findings for the inspected local store, evaluating snapshot compatibility against the relevant code contract, and documenting explicit limitations regarding production authority.

All diagnostic evaluations were executed under strict SQLite read-only URI mode (`?mode=ro`) against local SQLite stores without external network calls, without PostgreSQL queries, without modifying session advisory lock `401007311509093371`, and without disturbing the armed Day 1 SEC ingestion trial (`ARX-TRIAL-2026-NVDA-AAPL-001`, Issue #165).

### Quality Gate Results Summary

| Gate | Description | Status | Evidence & Forensic Resolution |
| :--- | :--- | :---: | :--- |
| **GATE 1** | **Evidence Custody & Store Lineage** | **PASS** *(Store Identified)*<br>**NOT_CHECKED** *(Production Authority)* | Inspected local SQLite database (`/Users/iDavid/.finance_market_store.db`, SHA-256 `ce911318...`, 61 symbols, 7,951 daily bars). Its relationship to the earlier reported store (585 symbols, 228,937 bars) cannot be established from local artifacts; production authority is therefore marked **NOT_CHECKED**. Pre/post store hash verified identical (zero mutations). |
| **GATE 2** | **Population Reconciliation** | **PASS** | Formally resolved universe definitions: the "100 Stage 2 symbols" is an **analytical classification result** from RADAR-015 (`run-00cc89d2a6b9`), not an independent universe build. The 250 cohort is confirmed promoted under RADAR-014 (PR #164). The 500 candidate universe is a provisional rehearsal specification under RADAR-P2-003; **membership overlap does not constitute production promotion**. |
| **GATE 3** | **Symbol-Level Readiness Audit** | **PASS** | Reconciled symbol-level records for all 500 candidates strictly within the inspected local database: **445 symbols** (89.0%) have 0 daily bars (`MISSING_MARKET_DATA`); **55 symbols** (11.0%) have 50–199 bars (`INSUFFICIENT_HISTORY` for 200 SMA); **0 symbols** satisfy the 200/222/252 bar thresholds. Table-level provenance rows = 0. Historical 439 and 67 figures noted as unverified historical claims. |
| **GATE 4** | **Authoritative Report & Disposition** | **PASS** | Emitted machine-readable JSON (`radar_universe_readiness_assessment_166.json`) and Markdown assessment. Arithmetic closure verified ($500 = 445 + 55$). Evaluated snapshot compatibility against the 6-component version tuple contract rather than commit SHAs alone. Operational holds preserved. |
| **FINAL** | **Overall Readiness Verdict** | **ASSESSED_AS_OF_EVIDENCE**<br>*(BLOCKED_BY_EVIDENCE)* | Complete as of inspected evidence. Blocked from operational activation by confirmed data deficits in inspected local store and unverified production authority (`NOT_CHECKED`). |

---

## 2. Evidence Custody & Store Relationship (Gate 1)

All inputs were accessed strictly offline using read-only access. Zero application startup paths warming caches or invoking background fetchers were executed.

### Inspected Market Store Custody Audit

| Attribute | Inspected Local Market Store Evidence |
| :--- | :--- |
| **Filesystem Path** | `/Users/iDavid/.finance_market_store.db` |
| **Source Type** | Local developer SQLite file |
| **Acquisition / Mtime (UTC)** | `2026-11-14T19:26:22Z` |
| **SHA-256 Digest** | `ce911318619d9cdfb8984ada5ed72d3b1f1b2e1a72ddc9e6cecf0b1ae0ef1ba9` |
| **File Size** | 1,028,096 bytes (1.0 MB) |
| **Row Count (`asset_ohlcv_daily`)** | 7,951 rows |
| **Distinct Symbols Count** | 61 distinct symbols |
| **Date Range** | `2025-02-10` to `2026-11-13` |
| **Universe Build Binding** | **UNBOUND** (Zero universe manifest metadata or build ID stored in DB) |
| **Snapshot Binding** | **UNBOUND** (0 rows in `asset_factor_snapshots`, 0 rows in `asset_ohlcv_provenance`) |
| **Access Mode** | Strict SQLite URI read-only mode (`?mode=ro`) |

### Relationship to Earlier Reported Store (585 Symbols, 228,937 Bars)
- An earlier reported store baseline contained **585 distinct symbols and 228,937 bars** (reflecting a fully hydrated market dataset from historical testing and qualification).
- The inspected local database contains only **61 distinct symbols and 7,951 daily bars**.
- **Lineage Determination:** The relationship between the inspected local file and the 585-symbol / 228,937-bar store (e.g., whether it represents an unverified developer slice, test fixture, or partial extraction) **cannot be established from available local artifacts**.
- **Governance Verdict on Production Authority:** **`NOT_CHECKED`**.  
  Because local artifacts cannot verify that the inspected store reflects authoritative production market data, and because live production database queries were strictly avoided to prevent disturbing the armed SEC trial, production market data authority is recorded as **NOT_CHECKED**.

### Other Inspected Manifests and Auxiliary Stores

| Artifact / Data Store | Filesystem Path | SHA-256 Digest | Size (Bytes) | Role / Status |
| :--- | :--- | :--- | :---: | :--- |
| **500 Rehearsal Manifest** | `docs/research/UB-ARX-EXPANSION-500-REHEARSAL.json` | `cfacda2ca29eb5bf8c4a7604ae3e37f3c368cc1a2a3bbf3c7299840aa0fc0c9a` | 7,915 | Provisional 500-Equity Candidate Specification (Build: `UB-ARX-EXPANSION-500-REHEARSAL`, Hash: `5f60b47d...`, Unpromoted) |
| **250 Promoted Manifest** | `docs/research/UB-ARX-EXPANSION-250-20261107.json` | `507f6ea9cd6a77d1b773fdcf0578060e33b305db3b1a953f7834f85afe610548` | 4,561 | Promoted Operational Reference Cohort (Build: `UB-ARX-EXPANSION-250-20261107`, Hash: `889c86cd...`, RADAR-014 / PR #164) |
| **Local Universe Store** | `/Users/iDavid/.finance_universe_store.db` | `1a920f963f95b95db2bed1dc04755076beff76d493c1895c10a07b874863682f` | 53,248 | SQLite store (0 rows across `universe_source_snapshots`, `universe_builds`, `universe_eligibility_ledger`, `universe_readiness_ledger`) |
| **Local Scanner Store** | `/Users/iDavid/.finance_scanner_store.db` | `3b221fa51e91c0cc94846e49dbefbe108973d1f5cff307c19d928eaff5785a17` | 122,880 | SQLite snapshot store (1 Smart Money snapshot `SNAP_SM_1794978903_1dc4b96f`, 0 VCP snapshots) |

---

## 3. Universe Population Reconciliation & Promotion Distinction (Gate 2)

Prior discussions referenced three distinct populations: **"100 Stage 2 symbols"**, **"250 promoted symbols"**, and **"500 candidate symbols"**. Forensic inspection of historical execution records resolves their precise relationship:

### Population Structure & Promotion Status

```
┌────────────────────────────────────────────────────────────────────────┐
│             ARX RADAR UNIVERSE POPULATION RELATIONSHIP                 │
│                                                                        │
│   500 Candidate Expansion Universe (UB-ARX-EXPANSION-500-REHEARSAL)    │
│   [STATUS: PROVISIONAL SPECIFICATION / UNPROMOTED REHEARSAL CANDIDATE] │
│                                                                        │
│   ├── 250 Preserved Promoted Operational Universe                      │
│   │   [STATUS: CONFIRMED PRODUCTION PROMOTION — RADAR-014 / PR #164]   │
│   │   ├── 133 Curated Baseline Equities                                │
│   │   └── 117 Additional Hydrated S&P 500 Components                   │
│   │                                                                    │
│   │   * RADAR-015 Forensic Scan Evaluation Result (run-00cc89d2a6b9):  │
│   │     ├── 100 Stage 2 Qualified Matches ("100 Stage 2 Symbols")      │
│   │     └── 150 Non-Stage-2 or Unresolved:                             │
│   │         ├── 65 Stage Unresolved (insufficient history)             │
│   │         ├── 52 Stage 4 (downtrend phase)                           │
│   │         ├── 33 Stage 1 (basing consolidation)                      │
│   │         └──  0 Stage 3 (distribution top)                          │
│   │     └──  0 Minervini VCP Setups Qualified                          │
│   │                                                                    │
│   └── 250 Additional Provisional Candidates [UNPROMOTED]               │
│       ├── 225 S&P 500 Remaining Institutional Components (Tier A)      │
│       └──  25 S&P MidCap 400 Senior CIK Constituents (Tier B)          │
└────────────────────────────────────────────────────────────────────────┘
```

| Population | Manifest / Identifier | Membership SHA-256 | Count | Derivation & Production Status | Overlap with 250 | Overlap with 500 |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| **100 Stage 2 Symbols** | `RADAR-015-FORENSIC-S2` | N/A (Analytical projection) | **100** | **Analytical classification result** from RADAR-015 scan (`run-00cc89d2a6b9`) on the 250 promoted universe. It is **NOT** an independent universe build or static list. | 100 (Subset of 250) | 100 (Subset of 500) |
| **250 Promoted Universe** | `UB-ARX-EXPANSION-250-20261107` | `889c86cd6d3f7c6aaef98852419fe16a546441e38171370e731bedab0947e313` | **250** | **Confirmed production promotion** under RADAR-014 (PR #164). Composed of 133 baseline + 117 hydrated S&P 500 constituents. | **250 (100%)** | **250 (100%)** |
| **500 Expansion Candidate** | `UB-ARX-EXPANSION-500-REHEARSAL` | `5f60b47d5402c0fc29fa545b244b778ca99a017f47448f47ebdfc257db8881b0` | **500** | **Provisional specification / rehearsal candidate** under RADAR-P2-003. Strict superset containing the 250 promoted cohort + 225 Tier A + 25 Tier B. **Membership overlap does NOT constitute production promotion; this universe remains unpromoted.** | **250 (100% of 250)** | **500 (100%)** |

### Key Governance Distinctions
1. **Output vs. Input:** The "100 Stage 2 symbols" represents the subset of equities within the 250 promoted universe that satisfied Weinstein Stage 2 criteria during historical scan RADAR-015. Primary Radar discovery projects this subset.
2. **Promotion vs. Membership Overlap:** While all 250 promoted symbols are preserved as a subset of the 500 candidate universe, **the 500 candidate universe as a whole has never received operational production promotion**. It remains a provisional specification under RADAR-P2-003.

---

## 4. Minimum History Data Requirements Matrix

Under Weinstein Stage Analysis and Minervini Trend Template domain rules, screening metrics enforce strict non-negotiable minimum trading history depths:

| Metric / Classification Rule | Minimum Trading Days | Applicable Domain Invariant | Code Authority | Purpose & Consequence of Deficit |
| :--- | :---: | :--- | :--- | :--- |
| **Universe Builder Screening Readiness** | **50** | `INV-DS-01` (Active Price & Liquidity) | `builder.py` (lines 192–211) | Basic screening eligibility. Assets with $<50$ bars fail to `DATA_UNAVAILABLE`. |
| **Weinstein 200-Day SMA** | **200** | `INV-TA-01` (200 SMA Boundary) | `predicate_registry.py` (`PRED_STAGE_2`) | `close > sma_200`. Assets with $<200$ bars fail closed to `STAGE_UNRESOLVED`. |
| **Weinstein Stage 2 Slope Boundary** | **222** | `INV-TA-02` (22-Day Moving Slope) | `predicate_registry.py` (`PRED_STAGE_2`) | Requires 200 SMA + 22-day lookback slope (`sma_200_slope_22 > 0`). Assets with $<222$ bars cannot compute Stage 2. |
| **Minervini Trend Template** | **252** | `INV-TA-03` (52-Week High/Low Bounds) | `predicate_registry.py` (`PRED_TREND_TEMPLATE`) | Requires 52-week high, 52-week low, 150 SMA, and 200 SMA (`close >= 1.25 * 52w_low`, within 25% of 52w high). Assets with $<252$ bars fail closed to `UNQUALIFIED`. |
| **Strict VCP Consolidation Pattern** | **252** | `INV-VCP-01` (Multi-Base Contraction) | `classifier.py` | Requires multi-month price contraction analysis, volume contraction, and overhead resistance checks. |

---

## 5. Audit Findings for Inspected Local Database (Gate 3)

### Finding 1: Market Data Deficit in Inspected Local Database
- In the inspected local database (`/Users/iDavid/.finance_market_store.db`), exact counts across the 500 candidate symbols are:
  - **445 symbols (89.0%)** have **0 daily bars** (`MISSING_MARKET_DATA`).
  - **55 symbols (11.0%)** have **50–199 daily bars** (`INSUFFICIENT_HISTORY` for 200 SMA). Specifically, 53 symbols have 128 bars, 1 has 65 bars, and 1 has 145 bars.
  - **0 symbols (0.0%)** have $\ge 200$ bars, $\ge 222$ bars, or $\ge 252$ bars.
  - Latest quote dates among the 55 present symbols: 54 have latest date `2026-11-12`, and 1 (`VRTX`) has latest date `2026-09-03`.
- **Status of Historical "439 Stale" Claim:** The historical claim that "439/500 symbols had stale market data" remains an unverified historical audit measurement whose calculation formula is not established by local artifacts. The verified finding for the inspected local store is 445 missing and 55 insufficient-history. Production market store coverage is **NOT_CHECKED**.
- **Local Verdict:** **LOCAL_DATA_BLOCKER**.

### Finding 2: Missing Provenance Records in Inspected Local Database
- In the inspected local database, `asset_ohlcv_provenance` contains **0 rows** (100% of symbols lack table-level provenance), and the universe store tables contain **0 rows**.
- **Status of Historical "67 Symbols" Claim:** The historical claim that "67 symbols lacked provenance records" remains an unverified historical figure whose specific symbol roster cannot be confirmed from local artifacts without live production access.
- **Local Verdict:** **LOCAL_DATA_BLOCKER**.

### Finding 3: Snapshot Compatibility Under Relevant Code Contract
- **Contract Authority:** Under the publication integrity contract (`ScannerPublicationIntegrityEngine` in `scanner_publication_integrity.py`), snapshot compatibility is governed by:
  1. The 6-component `ScannerVersionTuple` (`api_contract_version`, `ruleset_version`, `evidence_schema_version`, `score_model_version`, `data_provenance_version`, `universe_version`, `freshness_policy_version`).
  2. Component content hashes matching `AUTHORIZED_VERSION_REGISTRY`.
  3. The canonical semantic fingerprint matching `compute_semantic_fingerprint(...)`.
- **Commit SHA Distinction:** While `scanner_runner.py` (line 1009) contains a release SHA comparison against `CURRENT_IMPLEMENTATION_RELEASE_SHA`, **different git commit SHAs alone (e.g. across non-functional documentation PRs) do NOT establish semantic or structural incompatibility** under the code contract. A snapshot remains structurally compatible if its version tuple, schema, and ruleset match the ratified contract.
- **Local Store Reality:** In the inspected local scanner store (`/Users/iDavid/.finance_scanner_store.db`), `scanner_snapshots` contains **0 VCP snapshot rows** (only 1 Smart Money snapshot is present). Therefore, local snapshot presence and compatibility cannot be established.
- **Local Verdict:** **GOVERNANCE_AND_STORAGE_BLOCKER**.

---

## 6. Symbol-Level Summary for Inspected Local Database (500 Candidates)

| Category / Metric | 500 Expansion Universe (Local Store) | 250 Promoted Universe (Local Store) | Requirement for Stage 2 Discovery |
| :--- | :---: | :---: | :--- |
| **Total Evaluated Symbols** | **500** | **250** | Reference Universe Denominator |
| **Missing Market Data (0 bars)** | **445 (89.0%)** | **195 (78.0%)** | 0 bars in inspected local SQLite store |
| **Insufficient History (1–199 bars)** | **55 (11.0%)** | **55 (22.0%)** | Daily candles present, but $<200$ bars |
| — History: $<50$ bars | 0 (0.0%) | 0 (0.0%) | Fails Universe Builder threshold |
| — History: $50–199$ bars | 55 (11.0%) | 55 (22.0%) | Passes Builder, fails 200 SMA |
| — History: $200–221$ bars | 0 (0.0%) | 0 (0.0%) | Has 200 SMA, fails 22-day slope |
| — History: $222–251$ bars | 0 (0.0%) | 0 (0.0%) | Meets Stage 2, fails 52-week metrics |
| — History: $\ge 252$ bars | 0 (0.0%) | 0 (0.0%) | Meets full Minervini Trend Template |
| **Data Ready (Full Requirements Met)** | **0 (0.0%)** | **0 (0.0%)** | Requires $\ge 222$ bars + fresh session + provenance |
| **Universe Builder Data Ready** | **55 (11.0%)** | **55 (22.0%)** | Price $>0$ and bars $\ge 50$ (`builder.py`) |
| **Stage 2 Eligible by History Depth** | **0 (0.0%)** | **0 (0.0%)** | Requires $\ge 222$ completed trading sessions |
| **Trend Template Eligible by Depth** | **0 (0.0%)** | **0 (0.0%)** | Requires $\ge 252$ completed trading sessions |
| **Table-Level Provenance Rows** | **0 (0.0%)** | **0 (0.0%)** | Rows in `asset_ohlcv_provenance` |

*(Full 500-symbol evaluation records are serialized in [`docs/research/radar_universe_readiness_assessment_166.json`](radar_universe_readiness_assessment_166.json)).*

---

## 7. Arithmetic Closure Verification (Gate 4)

Complete arithmetic closure is formally verified across both populations:

### 500 Candidate Universe Closure Proof
$$\text{Total Evaluated} = 500$$
$$\text{Missing Market Data (445)} + \text{Insufficient History (55)} + \text{Ready (0)} + \text{Unknown (0)} = 500 \quad \checkmark$$
$$\text{Zero Bars (445)} + \text{50–199 Bars (55)} + \text{200–221 Bars (0)} + \text{222–251 Bars (0)} + \ge \text{252 Bars (0)} = 500 \quad \checkmark$$
$$\text{Builder Data Unavailable (445)} + \text{Builder Data Ready (55)} = 500 \quad \checkmark$$

### 250 Promoted Operational Universe Closure Proof
$$\text{Total Promoted Evaluated} = 250$$
$$\text{Missing Market Data (195)} + \text{Insufficient History (55)} = 250 \quad \checkmark$$
$$\text{Preserved in 500 Universe} = 250 \quad (100\%) \quad \checkmark$$

---

## 8. Prioritized Blocker Ledger & Remediation Plan

| Rank | Blocker ID | Description | Severity | Remediation Specification | Hold Gate Dependency |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | **BLK-RADAR-001** | **445 Missing & 55 Shallow Equities in Inspected Store** | **P0** | In an authorized maintenance window, hydrate $\ge 252$ daily OHLCV bars per symbol via authorized market data pipeline (start date: `2025-01-01`, end date: current completed session). | `HOLD_SCAN_001` remains ACTIVE during hydration; schedulers remain disabled. |
| **2** | **BLK-RADAR-002** | **Zero Table-Level Provenance Records in SQLite** | **P0** | Ensure market data ingestion writes corresponding records to `asset_ohlcv_provenance` per symbol transaction with vendor name, feed, `observation_precision`, and timestamps. | Enforces `INV-PROV-01` (Attributable Market Evidence). |
| **3** | **BLK-RADAR-003** | **Production Authority Unverified (NOT_CHECKED)** | **P0** | Reconcile local store against the authoritative production/staging database (585 symbols, 228,937 bars) in an authorized maintenance window. | Preserves SEC trial isolation during active trading window. |
| **4** | **BLK-RADAR-004** | **Snapshot Contract Verification** | **P1** | Verify active snapshot against the 6-component `ScannerVersionTuple` under `ScannerPublicationIntegrityEngine` and publish an authorized snapshot. | Aligns scanner endpoint with publication integrity contract. |

---

## 9. Operational Holds & Explicit Limitations

### Explicit Limitations of this Assessment
1. **Local Database Scope:** Findings (445 missing, 55 shallow history, 0 provenance rows) describe the inspected local developer database (`/Users/iDavid/.finance_market_store.db`, SHA-256 `ce911318...`).
2. **Production Authority Unverified:** Production database status is **`NOT_CHECKED`** because live production queries were not executed in this turn.
3. **Deployment Claims Excluded:** Stale claims regarding live deployed worker container PIDs or transient execution states are excluded in the absence of fresh, timestamped deployment telemetry.

### Operational Holds Retained Intact
- **`HOLD_SCAN_001=ACTIVE`:** Natural recurring background scanning, cron schedulers, and periodic scanner triggers remain **strictly disabled**.
- **`HOLD_SEC_001=ACTIVE`:** Continuous SEC Form 4 polling remains disabled; the Day 1 bounded two-issuer trial (`ARX-TRIAL-2026-NVDA-AAPL-001`) runs independently.
- **`HOLD_EXEC_001=ACTIVE`:** Zero financial trade execution code exists or is authorized.
- **`ARX_CONFLUENCE_INTEGRITY_HOLD=ACTIVE`:** Confluence scoring remains in governed research mode.
- **`PHASE_A3_UX_HOLD=ACTIVE`:** Progressive disclosure UI redesign remains held in draft awaiting human usability validation.

---

> **Authoritative Attestation:**  
> This readiness assessment is complete and reconciled as of inspected evidence. Final disposition: **`ASSESSED_AS_OF_EVIDENCE` / `BLOCKED_BY_EVIDENCE`**. Production authority: **`NOT_CHECKED`**. Zero external fetches, zero database mutations, zero live production queries, and zero operational hold releases have occurred.
