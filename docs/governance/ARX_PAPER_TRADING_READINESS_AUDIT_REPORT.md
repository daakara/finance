# ARX Paper-Trading & Research Readiness Audit Report

**Audit Target:** Production Web Service `56e38249-b78c-4a60-80d7-303ce7451bd8`  
**Railway Project:** `a339a92a-7236-4ac0-9a3a-d7ad440f9690` (`tranquil-radiance`)  
**Railway Environment:** `11cad5e1-358e-430e-84fc-189953cff48a` (`production`)  
**Container Mount Path:** `/root` (Persistent Volume `7867d923-89e9-4619-955a-cc7a43344f73`, `web-volume`)  
**Audit Timestamp:** `2026-11-22T22:55:00Z` (Local: `2026-11-22T23:55:00+01:00`)  
**Inspection Mode:** Read-Only Terminal Inspection (`mode=ro`, `PRAGMA query_only=ON;`, zero application imports)  

---

## Executive Readiness Scorecard

| Dimension | Audit Status | Key Operational Finding |
| :--- | :---: | :--- |
| **Research Tracking Readiness** | **PARTIAL (PASSIVE LOGGING ONLY)** | 122 prospective decision events and 5,522 platform recommendations logged; however, deployed release `2eccc64a` lacks epoch authorization, and all 122 decisions stem from user on-demand requests (systematic denominator remains 0). |
| **Settlement Readiness** | **NOT READY (HELD / INACTIVE)** | Exactly **0 outcome links** exist in `prospective_outcome_links`; 0 settled recommendations in `paper_trading_ledger.json` (all 19 `OPEN`); automatic periodic settlement workers remain disabled per `HOLD_SCAN_001` (`VCP_NATURAL_SCHEDULER_ENABLED=0`). |
| **Broker-Paper Readiness** | **NOT IMPLEMENTED (PROHIBITED)** | **No broker paper-order adapter exists** in the codebase. Alpaca credentials power read-only market price bars (`ALPACA_IEX`) and security master directory lookups only. Order execution actions (`SUBMIT_ORDER`, `EXECUTE_TRADE`) are excluded from domain governance contracts. |

---

## 1. Storage Path Resolution & Volume Architecture

*All storage paths were resolved authoritatively from container environment variables and directory inspection without importing any application code.*

- **Mount Point:** `RAILWAY_VOLUME_MOUNT_PATH = /root`
- **Data Root:** `/root/analyst_dashboard/data` (persistent ext4 volume mount, physically distinct from container overlayfs)
- **Database Access Contract:** Opened with standard library `sqlite3` using URI syntax `file:{path}?mode=ro` followed immediately by `PRAGMA query_only=ON;`. No missing files were initialized.

### Inventory of Evaluated Storage Artifacts

| Path | Artifact Type | Size (Bytes) | Last Modified (UTC) |
| :--- | :--- | :---: | :--- |
| `/root/analyst_dashboard/data/governance.db` | SQLite Database | 1,589,248 | 2026-11-22T14:44 |
| `/root/analyst_dashboard/data/paper_trading_ledger.json` | JSON Ledger | 276,238 | 2026-11-20T16:48 |
| `/root/analyst_dashboard/data/prospective_capture_stream.jsonl` | JSONL Stream | 190,165 | 2026-11-22T22:53 |
| `/root/analyst_dashboard/data/shadow_evidence.db` | SQLite Database | 622,592 | 2026-10-18T05:18 |
| `/root/analyst_dashboard/data/identity.db` | SQLite Database | 114,688 | 2026-10-20T22:33 |
| `/root/.finance_platform_history.db` | SQLite Database | 1,466,368 | 2026-11-22T14:44 |
| `/root/.finance_scanner_store.db` | SQLite Database | 360,448 | 2026-11-08T09:40 |
| `/root/.finance_coordination_store.db` | SQLite Database | 98,304 | 2026-11-06T09:51 |
| `/root/.finance_universe_store.db` | SQLite Database | 1,810,432 | 2026-11-08T09:37 |
| `/root/data/operational/openfigi_operational.db` | SQLite Database | 81,920 | 2026-11-22T14:07 |

---

## 2. Epoch Lifecycle, Release Authorization & Revocation Audit

*Inspected directly from `/root/analyst_dashboard/data/governance.db`:*

### 2.1 Epoch Activation Records (`epoch_activation_records`)
- **Total Rows:** 3 activations recorded in production history:
  1. `ARX_PROSPECTIVE_VALIDATION_EPOCH_2`: Release `f3b4e16d96f8686ae175874a9a6cde441ddd94a1`, Deployment `99e43e89-a0a4-4d2f-b082-c8b5d0100a4a`, Activated: `2026-09-24T07:22:51.654664+00:00`.
  2. `ARX_PROSPECTIVE_VALIDATION_EPOCH_3`: Release `ce30cfa95956541a67085d9bb2af6e41160f80b5`, Deployment `ec5d1724-4a6b-4bce-9151-84ed71d9968d`, Activated: `2026-09-24T21:45:31.641142+00:00`.
  3. `ARX_PROSPECTIVE_VALIDATION_EPOCH_4`: Release `5efc217950a6009270344122bfc07fca854ff788`, Deployment `cc2c94bd-2b4a-43d1-8b36-06d1129c8e6e`, Activated: `2026-09-25T05:15:34.204324+00:00`.
- **Current Deployed Production Release Alignment:**
  - Container environment: `ARX_RELEASE_SHA = 2eccc64a2cb3c649ad4d584a942285a6406d1ce0`.
  - Current release `2eccc64a` is **NOT ACTIVATED** in `epoch_activation_records`.

### 2.2 Release Authorizations (`epoch_release_authorizations`)
- **Total Rows:** 6 authorizations recorded (for releases `f3b4e16d`, `49d5d5a9`, `e9eec914`, `ce30cfa9`, `5efc2179`).
- Current production release `2eccc64a` has **zero certification records** and **zero authorizations** in `epoch_release_authorizations`.

### 2.3 Runtime Revocations (`epoch_release_revocations`)
- **Total Rows:** **0**. No release or deployment revocations have been registered.

### 2.4 Epoch Supersessions (`epoch_supersession_records`)
- **Total Rows:** 1 row: `ARX_PROSPECTIVE_VALIDATION_EPOCH_3` superseded by `ARX_PROSPECTIVE_VALIDATION_EPOCH_4` on `2026-09-25T05:15:34.204324+00:00` with reason `UPSTREAM_OPPORTUNITY_GENERATION_SELECTION_REFINEMENT` (`clean_prospective_signals_captured = 0`, `empirical_evidence_lost = 0`).

---

## 3. Census of Recommendations, Decisions & Outcome Links

*Reconciled across `paper_trading_ledger.json`, `governance.db`, `prospective_capture_stream.jsonl`, and `.finance_platform_history.db`:*

### 3.1 Paper Trading Ledger Signals (`paper_trading_ledger.json`)
The ledger records exactly **19 signals**, completely categorized into three cohorts:

| Cohort | Count | Status | Symbols | Timestamps | Description |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Historical Launch Baseline** | **9** | `OPEN` (0 settled) | `ANET`, `CELH`, `CRSP`, `GOOGL`, `LULU`, `MDB`, `NET`, `SNPS`, `TMDX` | `2026-09-04` | Frozen point-in-time launch baseline (`v2.4.0-phase24-freeze`, commit `4e36862`). All `sessionsObserved = 0`, `resolvedOutcome = null`. |
| **Synthetic / Certification Validation** | **1** | `OPEN` (0 settled) | `AAPL` | `2026-09-19T01:08:04Z` | `provenanceCohort = CERTIFICATION_VALIDATION`. Quarantined (`certification_generated: locally induced during Step 2 verification, excluded from primary natural prospective denominator`). |
| **Prospective Clean Observations** | **9** | `OPEN` (0 settled) | `AAPL`, `META`, `QQQM`, `VOO`, `FTXL`, `JEPI`, `QQQI`, `MSFT`, `NVDA` | `2026-09-24T09:10:36Z` to `2026-09-24T17:14:05Z` | `provenanceCohort = PROSPECTIVE_CLEAN`, `epochId = ARX_PROSPECTIVE_VALIDATION_EPOCH_2`. All `sessionsObserved = 0`, `resolvedOutcome = null`. |

- **Execution Ladder Plans (`paper_trading_ledger.json` & `governance.db`):** 125 total plans (35 `IN_BUY_ZONE`, 35 `EXTENDED_ABOVE_BUY_ZONE`, 36 `WAITING_PULLBACK`, 16 `STOPPED_OUT`, 3 `IN_BUY_ZONE_AWAITING_TRIGGER`). Timestamps range: `1791554903808` to `1795193318079`.

### 3.2 Prospective Decisions (`governance.db: prospective_decision_events`)
- **Total Events:** **122 rows**
- **Breakdown by Decision & Actionability State:**
  - `ACTIONABLE_SETUP` / `ACTIONABLE`: **9** (`AAPL` [3], `A` [2], `LNTH` [1], `MSFT` [1], `MNST` [2])
  - `VALID_SETUP` / `NON_ACTIONABLE`: **44**
  - `EVIDENCE_INCOMPLETE` / `NON_ACTIONABLE`: **50**
  - `INSUFFICIENT_DATA` / `NON_ACTIONABLE`: **17**
  - `STALE_DATA` / `NON_ACTIONABLE`: **1**
  - `UNVERIFIED` / `NON_ACTIONABLE`: **1**
- **Opportunity Capture Eligibility:** All 122 events were logged under evaluation cycles prefixed with `CYC_EXPLICIT_USER_REQUEST_*` (`POPULATION_SAMPLING_MODE = USER_SELECTED`, `eligible_for_opportunity_capture = False`). Systematic scheduled scans (`SCHEDULED_UNIVERSE_SCAN`) = **0**.
- **Sequential Recovery Stream (`prospective_capture_stream.jsonl`):** 545 total events (16 cycle creations, 122 decision recordings, 407 evaluation attempts).

### 3.3 Platform History Recommendations (`.finance_platform_history.db: trade_recommendation_history`)
- **Total Recommendations Logged:** **5,522 rows**
- **Outcome Status:** **100% `PENDING` (5,522 rows)**. Exactly **0 settled recommendations**.

### 3.4 Outcome Links (`governance.db: prospective_outcome_links`)
- **Total Rows:** **0**. Zero outcome links have been recorded.
- **Settlement Records in Stream:** **0**.

---

## 4. Outcome Settlement & Scheduling Audit

1. **Scheduling Configuration:**
   - Container environment variable: `VCP_NATURAL_SCHEDULER_ENABLED = 0` (Disabled).
   - No systemd timers, cron jobs, or active background settlement tasks exist in the container.
2. **Codebase Invariant:**
   - As codified in `analyst_dashboard/governance/shadow_replay_runner.py`:
     > *"PERIODIC WORKER CONTAINMENT: Settlement is strictly manual/dry-run replay only. Automatic periodic settlement workers remain disabled per HOLD_SCAN_001."*
3. **Settlement Timestamps:**
   - `prospective_outcome_links`: None (0 rows).
   - `paper_trading_ledger.json`: None (`resolutionDate: null` for all 19 signals).
   - `trade_recommendation_history`: None (`outcome_status: 'PENDING'` for all 5,522 rows).
   - Portfolio Exits: Exactly 1 manual exit event in `portfolio_holding_exit_events` on `2026-10-09 03:04:39` (manual user trade close of `IREN`, not automated recommendation settlement).

---

## 5. Broker Paper-Order Adapter Audit

1. **Credential Inspection:**
   - `ALPACA_API_KEY_ID`: Configured (`PKYUBBRKCK4ZUFH4LOZQ6XERJR`)
   - `ALPACA_API_SECRET_KEY`: Configured
   - `ALPACA_DATA_FEED`: `iex`
2. **Implementation Scope Analysis:**
   - Static analysis of all Alpaca usage across the repository demonstrates that credentials are used exclusively for:
     - Market data ingestion: `analyst_dashboard/data/alpaca_fetcher.py` (`source: "ALPACA_IEX"`, fetches bars/quotes).
     - Security Master reference: `analyst_dashboard/security_master/alpaca_adapter.py` (`ALPACA_ASSET_DIRECTORY`).
3. **Absence of Broker Order Adapter:**
   - There is **no broker order placement client, paper-trading execution adapter, or order routing engine** in the repository.
   - Sizing and order placement actions (`SIZE_TRADE`, `EXECUTE_TRADE`, `SUBMIT_ORDER`) are explicitly excluded and prohibited by governance rules (`ARX_058B_BATCH2_QUALIFICATION_PACKAGE.md`, `test_radar_error_zero_semantics.py`).
   - No broker API endpoints were contacted, and no order placement was attempted.

---

## 6. Tri-Partite Readiness Determinations

### 6.1 Research Tracking Readiness: **`PARTIAL_PASSIVE_ONLY`**
- **Operational:** Passive telemetry and decision capture engines are operational in production (`governance.db` with 122 events, `prospective_capture_stream.jsonl` with 545 entries, and `trade_recommendation_history` with 5,522 rows).
- **Gaps to Address Before Certified Research Tracking:**
  1. Current deployed release `2eccc64a` is not certified or authorized in `epoch_release_authorizations`.
  2. All captured decisions were generated via on-demand user analysis (`EXPLICIT_USER_REQUEST`), leaving the systematic opportunity capture denominator at 0.

### 6.2 Settlement Readiness: **`INACTIVE_HELD`**
- **Operational:** SQLite table `prospective_outcome_links` and `OutcomeSettlementLedger` schema exist.
- **Gaps to Address Before Settlement Activation:**
  1. Exactly 0 outcomes have ever been settled across all storage mechanisms.
  2. Automatic periodic settlement workers are disabled under `HOLD_SCAN_001` (`VCP_NATURAL_SCHEDULER_ENABLED=0`).
  3. Price bars and settlement evaluation must be operated via supervised dry-run replay before autonomous scheduler activation.

### 6.3 Broker-Paper Readiness: **`NOT_IMPLEMENTED`**
- **Operational:** None.
- **Gaps to Address Before Broker-Paper Trading:**
  1. Broker order adapter architecture is entirely unbuilt.
  2. Order submission actions are strictly barred by active governance boundaries (`HOLD_TRADING_001`).
  3. Existing Alpaca credentials power market data only.

---

## 7. Post-Audit Resolution Addendum (2026-11-23)

Following the delivery of this initial readiness audit, the platform team executed the **ARX Bounded Offline Recommendation Outcome Pilot (`ARX_BOUNDED_OFFLINE_OUTCOME_PILOT_001`)** under frozen Contract V1.0.1 ([ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1.json](file:///Users/iDavid/Projects/finance-clean/docs/governance/ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1.json)):

1. **Offline Evaluation Feasibility Proved:**
   - The frozen `PaperTradingOutcomeEvaluator` was exercised across all 19 recorded production signals without mutating production data or activating online schedulers.
   - 100% of signals were evaluated offline with bit-for-bit repeat-run determinism (SHA-256 `a9675c9592c474c2b3ad94998922c3c184d4a27305b091e42cdcd9e92eda2895`).
2. **Data Coverage Gap Resolved via Bounded Acquisition:**
   - Missing ETF price history for `VOO`, `QQQM`, and `JEPI` was acquired under `ARX_THREE_SYMBOL_ACQUISITION_MANIFEST_002` using persistent HTTP-attempt reservation (consuming 3 of 6 shared attempt slots, with 0 errors/retries/redirects).
   - This converted the previous data completeness gap into 100% resolved outcomes ($D_5 = 10$ of 10 triggered natural signals resolved under intraday bar sequencing).
3. **Preservation of Operational Holds:**
   - `HOLD_SCAN_001` remains **`ACTIVE`** (`VCP_NATURAL_SCHEDULER_ENABLED=0`).
   - `HOLD_EXEC_001` remains **`ACTIVE`** (zero broker connectivity).
   - All outcome evidence and evaluation receipts are archived in [ARX_BOUNDED_OFFLINE_OUTCOME_PILOT_REPORT.md](file:///Users/iDavid/Projects/finance-clean/docs/governance/ARX_BOUNDED_OFFLINE_OUTCOME_PILOT_REPORT.md) and [ARX_BOUNDED_OFFLINE_OUTCOME_PILOT_RECEIPT.json](file:///Users/iDavid/Projects/finance-clean/docs/governance/ARX_BOUNDED_OFFLINE_OUTCOME_PILOT_RECEIPT.json).

---
*Zero mutations, analytics calls, replay/settlement runs, activations, deployments, SEC requests, grant updates, or hold alterations occurred.*
