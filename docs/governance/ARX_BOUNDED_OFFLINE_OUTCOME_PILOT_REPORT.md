# ARX — Bounded Offline Recommendation Outcome Pilot Report
## Final Reconciled Audit & Executed Acquisition Results

**Evaluation Id:** `ARX_BOUNDED_OFFLINE_OUTCOME_PILOT_001`  
**Governing Contract:** ARX Paper-Trading Outcome Evaluation Contract V1.0.1 ([ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1.json](file:///Users/iDavid/Projects/finance-clean/docs/governance/ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1.json))  
**Contract SHA-256:** `0f13677667d6755ad97a63de9264bff8e3c77fd37642eca0f80bf3b95166c2c1`  
**Evaluation Engine:** `PaperTradingOutcomeEvaluator` ([paper_trading_outcome_evaluator.py](file:///Users/iDavid/Projects/finance-clean/analyst_dashboard/governance/paper_trading_outcome_evaluator.py))  
**Target Web Service:** `56e38249-b78c-4a60-80d7-303ce7451bd8` (Railway project `a339a92a-7236-4ac0-9a3a-d7ad440f9690`, environment `11cad5e1-358e-430e-84fc-189953cff48a`)  
**Acquisition Manifest:** `ARX_THREE_SYMBOL_ACQUISITION_MANIFEST_002`  
**Execution Timestamp:** `2026-11-23T06:45:00Z` (Local: `2026-11-23T07:45:00+01:00`)  
**Status:** **`FINAL_PILOT_COMPLETE`**  
**Repeat-Run Determinism SHA-256:** `a9675c9592c474c2b3ad94998922c3c184d4a27305b091e42cdcd9e92eda2895` (100% Bit-for-Bit Deterministic)  

---

## 1. Authoritative Source & Client Hash Verification

Prior to executing network requests, all local client modules, test suites, and source ledgers were cryptographically verified against their recorded references:

| Component | Authoritative File / Volume Path | Raw-Byte SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **Hardened Client** | `scratch/bounded_acquisition_client.py` | `c16b35962b9e44303bb2c9b44170d581943a38c9fdc5ebd70b6f05147e07533d` | **VERIFIED** |
| **Comprehensive Test Suite** | `scratch/test_bounded_acquisition_client.py` | `109a3a5bd7766a291504ec7b4fde6ed9d01d469c6587aca3997356018930556b` | **VERIFIED** |
| **Durable Attempt Ledger** | `scratch/acquisition_attempt_ledger.json` | `a7306925875c16cd2f284bf89a99b5a39a77c3b25ae3c4863839c4d73809f1a1` | **VERIFIED (Continuation Mode)** |
| **Production Source Ledger** | `/root/analyst_dashboard/data/paper_trading_ledger.json` | `5a3a390e81cfcddf7177609a8a1ae21d299032c35c0d9425047712b51bae3d3b` | **VERIFIED** |
| **Governing Contract V1.0.1** | `docs/governance/ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1.json` | `0f13677667d6755ad97a63de9264bff8e3c77fd37642eca0f80bf3b95166c2c1` | **VERIFIED** |
| **Historical Frozen Evidence** | `docs/governance/ARX_HISTORICAL_OUTCOME_FROZEN_EVIDENCE_V1.json` | `012643fc53cfc80cc615ba364ab5101d2fb0d791c7361e4b0967c3f8766d3aa0` | **VERIFIED** |
| **Historical Audit Record** | `docs/governance/ARX_HISTORICAL_RECOMMENDATION_OUTCOME_AUDIT_V1.json` | `c1023530e0c95fe7a4897ece44e8c66a8cae10297ac761a78c01f25eb0fdfce0` | **VERIFIED** |

---

## 2. Reconciled Per-Signal Census (All 19 Recorded Signals)

Every signal recorded contemporaneously in production `paper_trading_ledger.json` ($D_1 = 19$) has now been evaluated with full post-signal trading session coverage. With the authorized execution of `ARX_THREE_SYMBOL_ACQUISITION_MANIFEST_002`, **100% of data gaps are resolved**.

### Comprehensive Per-Signal Ledger

| Id | Symbol | Cohort | Verified Signal Timestamp | Entry Corridor | Stop Floor | Take Profit 1 | Price History Post-Signal | Final Entry State | Pure Daily Outcome | Intraday Bars Resolved Outcome | Reconciled Return (Gross / Net / R) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| 1 | **ANET** | Historical | `2026-09-04T17:32:00Z` | 182.36–194.83 | 179.00 | 218.36 | 55 sessions | Triggered | `NEUTRAL` | `NEUTRAL` | +6.20% / +5.95% / +0.76R |
| 2 | **CELH** | Historical | `2026-09-04T17:32:00Z` | 30.15–32.25 | 29.56 | 35.98 | 51 sessions | Triggered | `UNRESOLVED_INTRABAR` | `FAILURE` | -3.40% / -3.65% / -1.00R |
| 3 | **CRSP** | Historical | `2026-09-04T17:32:00Z` | 53.60–57.52 | 53.05 | 65.53 | 45 sessions | Triggered | `UNRESOLVED_INTRABAR` | `FAILURE` | -2.57% / -2.82% / -1.00R |
| 4 | **GOOGL** | Historical | `2026-09-04T17:32:00Z` | 340.64–347.16 | 330.42 | 364.80 | 50 sessions | Untriggered | `ENTRY_NOT_TRIGGERED` | `ENTRY_NOT_TRIGGERED` | — |
| 5 | **LULU** | Historical | `2026-09-04T17:32:00Z` | 117.27–122.00 | 113.85 | 136.61 | 55 sessions | Untriggered | `ENTRY_NOT_TRIGGERED` | `ENTRY_NOT_TRIGGERED` | — |
| 6 | **MDB** | Historical | `2026-09-04T17:32:00Z` | 371.95–422.66 | 360.81 | 453.19 | 51 sessions | Untriggered | `ENTRY_NOT_TRIGGERED` | `ENTRY_NOT_TRIGGERED` | — |
| 7 | **NET** | Historical | `2026-09-04T17:32:00Z` | 279.18–298.04 | 266.02 | 319.90 | 51 sessions | Triggered | `SUCCESS` | `SUCCESS` | +14.59% / +14.34% / +3.09R |
| 8 | **SNPS** | Historical | `2026-09-04T17:32:00Z` | 408.30–424.02 | 396.04 | 453.81 | 55 sessions | Untriggered | `ENTRY_NOT_TRIGGERED` | `ENTRY_NOT_TRIGGERED` | — |
| 9 | **TMDX** | Historical | `2026-09-04T17:32:00Z` | 79.89–88.83 | 79.46 | 104.54 | 55 sessions | Triggered | `FAILURE` | `FAILURE` | -10.55% / -10.80% / -1.00R |
| 10 | **AAPL** | Synthetic* | `2026-09-19T01:08:04Z` | 320.03–328.85 | 314.28 | 383.66 | 55 sessions | Untriggered | `ENTRY_NOT_TRIGGERED` | `ENTRY_NOT_TRIGGERED` | — |
| 11 | **AAPL** | Prospective | `2026-09-24T09:10:36Z` | 321.41–332.13 | 315.11 | 383.73 | 42 sessions | Triggered | `NEUTRAL` | `NEUTRAL` | +3.47% / +3.22% / +0.68R |
| 12 | **META** | Prospective | `2026-09-24T09:23:20Z` | 611.63–671.93 | 607.91 | 996.05 | 38 sessions | Untriggered | `ENTRY_NOT_TRIGGERED` | `ENTRY_NOT_TRIGGERED` | — |
| 13 | **QQQM** | Prospective | `2026-09-24T10:07:27Z` | 292.64–298.68 | 285.31 | 344.50 | 24 sessions | Untriggered | `ENTRY_NOT_TRIGGERED` | `ENTRY_NOT_TRIGGERED` | **[RECONCILED]** |
| 14 | **VOO** | Prospective | `2026-09-24T10:12:04Z` | 699.14–706.19 | 678.16 | 762.09 | 24 sessions | Triggered | `NEUTRAL` | `NEUTRAL` | **-0.55% / -0.80% / -0.14R** |
| 15 | **FTXL** | Prospective | `2026-09-24T14:20:59Z` | 228.45–235.94 | 227.19 | 309.78 | 6 sessions | Untriggered | `ENTRY_NOT_TRIGGERED` | `ENTRY_NOT_TRIGGERED` | — |
| 16 | **JEPI** | Prospective | `2026-09-24T14:24:16Z` | 56.47–56.92 | 54.60 | 60.25 | 20 sessions | Triggered | `NEUTRAL` | `NEUTRAL` | **+1.29% / +1.04% / +0.37R** |
| 17 | **QQQI** | Prospective | `2026-09-24T14:25:00Z` | 53.44–54.52 | 51.94 | 62.41 | 16 sessions | Untriggered | `ENTRY_NOT_TRIGGERED` | `ENTRY_NOT_TRIGGERED` | — |
| 18 | **MSFT** | Prospective | `2026-09-24T17:14:04Z` | 471.03–500.39 | 468.05 | 581.44 | 38 sessions | Triggered | `NEUTRAL` | `NEUTRAL` | +4.42% / +4.17% / +0.71R |
| 19 | **NVDA** | Prospective | `2026-09-24T17:14:05Z` | 215.19–223.41 | 210.85 | 256.54 | 42 sessions | Triggered | `NEUTRAL` | `NEUTRAL` | -5.01% / -5.26% / -0.89R |

*\*Quarantined: Synthetically induced during Step 2 verification; strictly excluded from performance denominators.*

### Key Before vs. After Reconciliation Summary:
1. **`QQQM`:** Prior state `ENTRY_WINDOW_STILL_OPEN` (3 of 5 entry sessions). With all 5 entry sessions acquired, verified that spot prices never satisfied the entry corridor (292.64–298.68). Final outcome resolved to **`ENTRY_NOT_TRIGGERED`** (`ENTRY_WINDOW_EXPIRED`).
2. **`VOO`:** Prior state `OUTCOME_DATA_MISSING` (0 post-signal sessions). With 24 sessions acquired, triggered on 2026-09-25 at `$706.19` (corridor ceiling limit fill). Survived all 20 sessions without breaching stop floor (`678.16`) or TP1 (`762.09`). Final outcome resolved to **`NEUTRAL`** (Horizon Exit @ `$702.30`: Gross -0.55%, Net -0.80%, R -0.14R).
3. **`JEPI`:** Prior state `UNMATURED` (3 of 20 holding sessions). With all 20 sessions acquired, survived full holding horizon without breaching stop floor (`54.60`) or TP1 (`60.25`). Final outcome resolved to **`NEUTRAL`** (Horizon Exit @ `$57.29`: Gross +1.29%, Net +1.04%, R +0.37R).

---

## 3. Reconciled Denominators & Dual Intraday Sequencing

Contract V1.0.1 mandates rigorous separation between fail-closed Daily OHLC evaluation and Intraday Bars Resolved evaluation.

| Governance Metric | Pure Daily OHLC Evaluation | Intraday Bars Resolved Evaluation | Reconciliation Notes |
| :--- | :---: | :---: | :--- |
| **Total Recorded Inventory ($D_1$)** | **19** | **19** | Complete inventory preserved |
| **Synthetic Quarantined Excluded** | **1** (`AAPL` 2026-09-19) | **1** (`AAPL` 2026-09-19) | Excluded from performance denominators |
| **Natural Universe ($D_1$)** | **18** | **18** | 9 Historical Baseline + 9 Prospective Clean |
| **Data Eligible ($D_2$)** | **18** (100.0%) | **18** (100.0%) | 100% data coverage; zero missing symbols |
| **Triggered Entries ($D_3$)** | **10** (55.6%) | **10** (55.6%) | 5 Historical + 5 Prospective |
| **Mature Triggered ($D_4$)** | **10** (100.0%) | **10** (100.0%) | All 10 reached terminal horizon or stop/target |
| **Resolved Mature ($D_5$)** | **8** (80.0% of $D_3$) | **10** (100.0% of $D_3$) | All 10 resolved under intraday bars |
| **Take Profit 1 (`SUCCESS`)** | **1** (`NET`) | **1** (`NET`) | 10.0% of $D_5$ |
| **Stop Loss (`FAILURE`)** | **1** (`TMDX`) | **3** (`TMDX`, `CELH`, `CRSP`) | 30.0% of $D_5$ |
| **Horizon Exit (`NEUTRAL`)** | **6** (`ANET`, `AAPL`, `MSFT`, `NVDA`, `VOO`, `JEPI`) | **6** (`ANET`, `AAPL`, `MSFT`, `NVDA`, `VOO`, `JEPI`) | 60.0% of $D_5$ |
| **Untriggered Expired** | **8** (44.4% of $D_1$) | **8** (44.4% of $D_1$) | 4 Historical + 4 Prospective |
| **Unresolved Intrabar Collision** | **2** (`CELH`, `CRSP`) | **0** | Conclusively resolved via 1-min intraday bars |
| **Blocked / Incomplete Data** | **0** | **0** | **100% data gaps eliminated** |
| **Average Gross Return (Resolved)** | **+1.73%** | **+0.79%** | Pure Daily excludes CELH/CRSP; Intraday includes both |
| **Average Net Return (Resolved)** | **+1.48%** | **+0.54%** | Applies mandatory 25 bps contract friction |
| **Average R-Multiple (Resolved)** | **+0.45R** | **+0.16R** | Pure Daily: +0.45R (N=8); Intraday: +0.16R (N=10) |

---

## 4. Cohort Performance Breakdown (Historical Baseline vs. Prospective Clean)

Separating the initial 9 historical baseline recommendations from the 9 prospective clean recommendations reveals distinct execution characteristics:

| Metric | Historical Baseline Cohort | Prospective Clean Cohort | Combined Natural Universe |
| :--- | :---: | :---: | :---: |
| **Cohort Universe ($D_1$)** | 9 | 9 | 18 |
| **Untriggered Setups** | 4 (44.4%) | 4 (44.4%) | 8 (44.4%) |
| **Triggered Executions ($D_3$)** | 5 (55.6%) | 5 (55.6%) | 10 (55.6%) |
| **Take Profit 1 (`SUCCESS`)** | 1 (20.0%) | 0 (0.0%) | 1 (10.0%) |
| **Stop Loss (`FAILURE`)** | 3 (60.0%) | 0 (0.0%) | 3 (30.0%) |
| **Horizon Exit (`NEUTRAL`)** | 1 (20.0%) | 5 (100.0%) | 6 (60.0%) |
| **Unresolved Setups** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **Average Gross Return** | **+0.85%** | **+0.72%** | **+0.79%** |
| **Average Net Return** | **+0.60%** | **+0.47%** | **+0.54%** |
| **Average R-Multiple** | **+0.17R** | **+0.15R** | **+0.16R** |
| **Win / Loss / Neutral Ratio** | 20% / 60% / 20% | 0% / 0% / 100% | 10% / 30% / 60% |

> [!NOTE]
> **Cohort Divergence Analysis, Fill Price & Return Reconciliation:**
> - **Historical Baseline:** Single-stock setups (`NET` +14.59% gross / +14.34% net / +3.09R; `ANET` +6.20% gross / +5.95% net / +0.76R; `TMDX` -10.55% gross / -10.80% net / -1.00R; `CELH` -3.40% gross / -3.65% net / -1.00R; `CRSP` -2.57% gross / -2.82% net / -1.00R). Recomputed directly from verified fills under Contract V1.0.1, the cohort average net return is **+0.60%** (gross +0.85%) and the average R-multiple is **+0.17R**. Prior reported values of -1.78% net and -0.01R reflected legacy summary table transcription errors: `CELH` and `CRSP` were transcribed as -10.38% and -8.02% (contradicting verified fills $30.60->$29.56 and $54.45->$53.05), while `NET` (+3.09R) and `ANET` (+0.76R) were transcribed as +1.77R and +1.18R.
> - **Prospective Clean:** 100% of triggered setups survived to the 20-session horizon exit (`NEUTRAL`), yielding zero stop-outs, average net return of **+0.47%** (gross +0.72%), and an average R-multiple of **+0.15R** (`AAPL` +0.68R, `MSFT` +0.71R, `NVDA` -0.89R, `VOO` -0.14R, `JEPI` +0.37R; prior +0.18R reflected manual transcription errors where AAPL was transcribed as +0.47R, MSFT as +1.03R, and NVDA as -0.84R).
> - **Combined Natural Expectancy:** Across all 10 resolved trades ($D_5=10$), the combined pooled average net return is **+0.54%** (gross +0.79%) and the reconciled average R-multiple is **+0.16R** (prior -0.65% net and +0.08R were the arithmetic propagation of the transcribed errors). Under Pure Daily evaluation ($D_5=8$, excluding unsequenced CELH/CRSP), average net return is **+1.48%** (gross +1.73%) and average R-multiple is **+0.45R**.

---

## 5. Executed Bounded Acquisition Summary (`ARX_THREE_SYMBOL_ACQUISITION_MANIFEST_002`)

The acquisition was executed strictly under the governing 6-attempt HTTP ceiling with persistent on-disk reservations:

```
Manifest Id: ARX_THREE_SYMBOL_ACQUISITION_MANIFEST_002
Accounting Architecture: PERSISTENT_HTTP_REQUEST_COUNTER (POSIX flock Atomicity)
Attempt Ceiling: 6 HTTP transmissions total
Attempts Reserved: 3
Attempts Completed: 3
Remaining Attempt Budget: 3
Network Errors / Retries / Redirects: Exactly 0
Status: ACQUISITION_COMPLETED
```

### Executed Transmissions & SHA-256 Integrity Receipts

1. **`VOO` (Vanguard S&P 500 ETF):**
   - **Reservation Id:** `RES_001_20261123064455449638`
   - **HTTP Status:** 200 OK
   - **Raw Bytes Received:** 3,759 bytes
   - **Raw SHA-256 Digest:** `e344bc0a1f2c01fa2751dafe721fcc799bb85f7ae7c5f07aab20f6c9989512db`
   - **Integrity Receipt:** [VOO_RES_001_20261123064455449638.meta.json](file:///Users/iDavid/.gemini/antigravity/brain/e53e772b-2682-4ca8-a2bd-008123fee60e/scratch/raw_responses/VOO_RES_001_20261123064455449638.meta.json)
   - **Sessions Acquired:** 24 regular XNYS sessions (`2026-09-25` to `2026-10-28`).
   - **Split Provenance:** Verified `indicators.quote[0]`; zero split events recorded in window; non-dividend-adjusted.

2. **`QQQM` (Invesco NASDAQ 100 ETF):**
   - **Reservation Id:** `RES_002_20261123064455729181`
   - **HTTP Status:** 200 OK
   - **Raw Bytes Received:** 3,796 bytes
   - **Raw SHA-256 Digest:** `39a2c05b7806433128df6a57895c7c62a05678f2e3dbf8d9564672a481969772`
   - **Integrity Receipt:** [QQQM_RES_002_20261123064455729181.meta.json](file:///Users/iDavid/.gemini/antigravity/brain/e53e772b-2682-4ca8-a2bd-008123fee60e/scratch/raw_responses/QQQM_RES_002_20261123064455729181.meta.json)
   - **Sessions Acquired:** 24 regular XNYS sessions (`2026-09-25` to `2026-10-28`).
   - **Split Provenance:** Verified `indicators.quote[0]`; zero split events recorded in window; non-dividend-adjusted.

3. **`JEPI` (JPMorgan Equity Premium Income ETF):**
   - **Reservation Id:** `RES_003_20261123064455964558`
   - **HTTP Status:** 200 OK
   - **Raw Bytes Received:** 3,373 bytes
   - **Raw SHA-256 Digest:** `761ceaa809db7530ad31bbff11e93c335f580112da94f10f5068e64b8a8c3ce1`
   - **Integrity Receipt:** [JEPI_RES_003_20261123064455964558.meta.json](file:///Users/iDavid/.gemini/antigravity/brain/e53e772b-2682-4ca8-a2bd-008123fee60e/scratch/raw_responses/JEPI_RES_003_20261123064455964558.meta.json)
   - **Sessions Acquired:** 20 regular XNYS sessions (`2026-09-25` to `2026-10-22`).
   - **Split Provenance:** Verified `indicators.quote[0]`; zero split events recorded in window; non-dividend-adjusted.

---

## 6. Repeat-Run Determinism Verification

Following the incorporation of acquired ETF bars, the offline evaluation engine was executed in two consecutive independent passes across the complete 19-record universe:
- **Pass 1 Evaluation Digest:** `a9675c9592c474c2b3ad94998922c3c184d4a27305b091e42cdcd9e92eda2895`
- **Pass 2 Evaluation Digest:** `a9675c9592c474c2b3ad94998922c3c184d4a27305b091e42cdcd9e92eda2895`
- **Result:** **100% Bit-for-Bit Deterministic Match**. Zero state leakage or stochastic divergence.

---

## 7. Production Governance & Boundary Verification

- **Production Integrity:** Exactly **0** writes to production databases (`.finance_market_store.db`, `governance.db`, `.finance_platform_history.db`) or application files.
- **Broker & SEC Isolation:** Exactly **0** broker endpoint calls, paper order submissions, or SEC EDGAR requests.
- **Settlement Guard:** Natural settlement scheduler remains disabled (`VCP_NATURAL_SCHEDULER_ENABLED=0`); `HOLD_SCAN_001` remains in effect.
- **Durable Ledger Status:** Persisted ledger at [acquisition_attempt_ledger.json](file:///Users/iDavid/.gemini/antigravity/brain/e53e772b-2682-4ca8-a2bd-008123fee60e/scratch/acquisition_attempt_ledger.json) confirms 3 attempts reserved, 3 completed, and 3 remaining budget. Process continuation fails closed.
