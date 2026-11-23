# ARX Terminal — Pragmatic Forward Paper-Trading Implementation Roadmap

**Document ID:** `PLN-ROADMAP-PT-2026-11-23`  
**Effective Date:** 2026-11-23  
**Status:** Offline capture, settlement, and market-data/corporate-action adapter qualified (PR #322); pilot activation remains strictly disabled under operational holds
**Accountable Owner:** `daakara`  
**Classification:** Canonical Implementation Roadmap  
**Governing Contract:** ARX Paper-Trading Outcome Evaluation Contract V1.0.1 ([`docs/governance/ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1.json`](../governance/ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1.json))  
**Companion Documents:**
- [Integrated Status Addendum (2026-11-23)](../ARX_STATUS_ADDENDUM_2026_11_23.md)
- [Bounded Offline Outcome Pilot Report (2026-11-23)](../governance/ARX_BOUNDED_OFFLINE_OUTCOME_PILOT_REPORT.md)
- [Paper-Trading Readiness Audit Report (2026-11-22)](../governance/ARX_PAPER_TRADING_READINESS_AUDIT_REPORT.md)
- [Operational Holds Register](../governance/ARX_OPERATIONAL_HOLDS_REGISTER.md)
- [Active Work Register (2026-11-17)](ARX_ACTIVE_WORK_REGISTER_2026_11_17.md)

---

## 1. Executive Summary & Objective

The objective of this roadmap is to establish a **bounded forward paper-trading pilot** that:
1. Records incoming candidate recommendations deterministically *before* price outcomes are known.
2. Simulates entry, stop, and target executions consistently under frozen Contract V1.0.1 rules.
3. Settles outcomes automatically against authoritative post-market price tape.

### Core Architecture & Phasing Principle
- **Phase 1 (This Roadmap):** 100% internal simulated execution. Zero broker API order routing, zero live capital commitment, zero model retraining, and zero background daemon activation.
- **Phase 2 (Optional Future Milestone):** Broker paper-order adapter (e.g. Alpaca Paper API), gated behind successful Phase 1 completion and formal Owner unfreeze.

### Context & Epistemic Baseline
The recently concluded **Bounded Offline Recommendation Outcome Pilot (`ARX_BOUNDED_OFFLINE_OUTCOME_PILOT_001`)** established:
- **Baseline Inventory:** 19 total recorded recommendations; 1 synthetic quarantined setup excluded; 18 natural signals ($D_2 = 18$ data eligible).
- **Execution & Resolution:** 10 simulated triggered entries ($D_3 = 10$), 8 untriggered setups expired without entry.
- **Empirical Results:** Mean net return of **+0.54%** across 10 resolved trades (+0.16R reconciled from trades; prior reported -0.65% net and +0.08R reflected legacy transcription errors); historical baseline yielded +0.60% net (+0.17R reconciled; prior -1.78% net and -0.01R reflected CELH/CRSP/NET/ANET transcription errors) with a 60% stop-out rate, while prospective clean setups yielded +0.47% net (+0.15R reconciled, prior +0.18R) with 100% horizon exits. Pure daily resolved (N=8) yielded +1.48% net (+0.45R).
- **Epistemic Constraint:** This offline pilot serves strictly as **historical validation of evaluation mechanics**, not proof of profitability, statistical edge, or production readiness.

---

## 2. Confirmed Capabilities vs. Identified Gaps

| Dimension | Confirmed Capabilities (Reused) | Identified Gaps (Addressed in Roadmap) |
| :--- | :--- | :--- |
| **Outcome Evaluation** | `PaperTradingOutcomeEvaluator` (`analyst_dashboard/governance/paper_trading_outcome_evaluator.py`) fully implements Contract V1.0.1 (25 bps friction, 5 entry sessions, 20 holding sessions, intrabar collision isolation). | Lacks an automated post-market daily batch driver to advance open setups through calendar time. |
| **Decision Capture** | `ProspectiveDecisionCaptureEngine` (`analyst_dashboard/governance/prospective_capture.py`) logs full candidate decisions with deterministic IDs, sampling mode segregation, and write firewalls. | Current production decisions stem exclusively from ad-hoc user queries (`USER_SELECTED`); systematic daily forward scan capturing is not scheduled. |
| **Data Acquisition** | `BoundedHttpClient` (`scratch/bounded_acquisition_client.py`) provides durable on-disk reservations under POSIX `flock`, fail-closed restart safety, redirect/retry prohibitions, and SHA-256 integrity receipts. | Need standard CLI pipeline to pull daily split-adjusted OHLCV bars without manual operator intervention. |
| **Settlement Storage** | Tables `prospective_outcome_links` and `paper_trading_ledger.json` exist. | Settlement daemon is held (`HOLD_SCAN_001=ACTIVE`, `VCP_NATURAL_SCHEDULER_ENABLED=0`); requires idempotent batch replay CLI rather than continuous daemon. |
| **Broker Integration** | Alpaca credentials exist for market data (`ALPACA_IEX`) and security master directory. | **No broker paper-order adapter exists.** Sizing and order submission actions remain strictly barred by `HOLD_EXEC_001`. |

---

## 3. Four Practical Milestones

```mermaid
flowchart LR
    M1["Milestone 1<br>Baseline Preservation<br>& Recomputation"] --> M2["Milestone 2<br>Bounded Forward<br>Cohort Capture"]
    M2 --> M3["Milestone 3<br>Simulated Execution<br>& Auto-Settlement"]
    M3 --> M4["Milestone 4<br>Pilot Scoreboard<br>& Gate Review"]
```

---

### Milestone 1: Preserve and Verify Baseline

**Objective:** Lock, isolate, and cryptographically verify all existing pilot evidence, contracts, schemas, and evaluator versions so forward observations cannot contaminate historical audit trails. Recompute results from first principles to establish correctness.

- **Scope & Reusable Components:**
  - Retain `docs/governance/ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1.json` (SHA-256 `0f13677667d6755ad97a63de9264bff8e3c77fd37642eca0f80bf3b95166c2c1`).
  - Isolate historical inventory ($D_1 = 19$, 18 natural signals) as `COHORT_HISTORICAL_PILOT_001`.
  - Retain raw response files and metadata in `data/raw_feeds/paper_trading_pilot/` with SHA-256 integrity receipts.
  - Implement a deterministic regression fixture that **recomputes results from scratch** using retained signal plans and raw price bars. Do not rely solely on matching an output hash:
    - Recompute entry states, fill prices, exit reasons, returns, and R-multiples for every trade.
    - Assert that individual trade values (`NET` TP1, `TMDX` Stop, `VOO` Neutral, `JEPI` Neutral, `CELH`/`CRSP` intrabar collisions) and statistical aggregates match Contract V1.0.1 rules exactly.
    - Confirm repeat-run output matches serialized SHA-256 `a9675c9592c474c2b3ad94998922c3c184d4a27305b091e42cdcd9e92eda2895`.
- **Dependencies & Accountable Owner:**
  - Dependencies: None (pure offline baseline verification).
  - Accountable Owner: `daakara`.
- **Smallest Reviewable PR:**  
  `PR-PT-01: Add historical pilot first-principles recomputation test fixture and baseline lock`.
- **Quality Gates & Acceptance Criteria:**
  - Automated test `tests/test_paper_trading_baseline_determinism.py` passes 100% with zero network calls.
  - Forward cohort records strictly isolated in durable project directory `data/forward_pilot/` with cohort prefix `COHORT_FORWARD_*`.
  - Zero modifications to production database tables or existing ledger files.
- **Rollback / Disable Procedure:** Revert commit; production runtime is unmutated.
- **Effort Range & Decision:**
  - Effort: 1 engineering day.
  - Key Decision: Confirmation of durable storage directory (`data/forward_pilot/`).

---

### Milestone 2: Capture a Bounded Forward Cohort

**Objective:** Record new candidate setups and eligibility decisions deterministically *before* subsequent price bars exist, enforcing strict separation between systematic opportunities and user on-demand queries.

- **Scope & Reusable Components:**
  - Reuse `ProspectiveDecisionCaptureEngine` (`analyst_dashboard/governance/prospective_capture.py`) and schema `ARX_PROSPECTIVE_DECISION_CAPTURE_SCHEMA_V1.json`.
  - Bounded Forward Universe: 10 liquid US instruments (5 index/sector ETFs: `VOO`, `QQQM`, `IWM`, `XLE`, `XLK` + 5 mega-cap equities: `AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`).
  - Pilot Size & Cohort Ceilings:
    - **Capture Horizon:** Exactly **20 capture sessions** (trading days).
    - **Evaluated Volume:** 20 sessions × 10 symbols = **exactly 200 evaluation records** ($D_1$ denominator).
    - **Separate Entry Cap:** Cap simulated triggered entries at a maximum of **20 triggered trades** ($D_3 \le 20$). If 20 entries trigger before Session 20, further candidate setups are marked `SKIPPED_CAP_REACHED` for simulation purposes while candidate evaluation logging continues.
    - **Execution Model:** Begin with an **operator-triggered systematic ten-symbol batch** per completed trading session (manual CLI invocation after market close 16:15 ET, before next-session open -12h).
    - **Scheduling Decoupling:** Automated background scheduling (daemon, cron, or CI scheduled workflow) is a **later operational choice**, not an initial pilot prerequisite.
    - **Broker/Execution Holds Preserved:** Maintain `HOLD_EXEC_001=ACTIVE` strictly throughout internal simulation. Zero live or broker-paper order placement.
    - **Durable Storage Attestation Required:** Require physical storage persistence attestation (`analyst_dashboard.governance.storage.attest_persistent_storage()`, verifying persistent volume mount distinct from container root overlayfs) prior to any operational pilot activation or forward writes.
  - Decision Recording Standards:
    - Record every evaluated symbol: `ACTIONABLE_SETUP`, `VALID_SETUP`, `INSUFFICIENT_DATA`, `EVIDENCE_INCOMPLETE`.
    - Freeze signal plan: spot price, corridor [min, max], stop floor, take profit targets, confluence score, engine SHA (`7ad4459...`), and timestamp UTC.
    - Sampling Mode: `samplingMode = "SYSTEMATIC"` (`SCHEDULED_UNIVERSE_SCAN`). Only systematic evaluations enter the forward performance denominator.
- **Dependencies & Accountable Owner:**
  - Dependencies: Milestone 1 baseline lock; daily market data snapshot; durable storage attestation.
  - Accountable Owner: `daakara`.
- **Smallest Reviewable PR:**  
  `PR-PT-02: Implement forward cohort candidate capture CLI and frozen universe manifest`.
- **Quality Gates & Acceptance Criteria:**
  - **No Look-Ahead:** Capture timestamp must precede $T+1$ market open by $\ge 12$ hours.
  - **No Cherry-Picking:** All 10 candidates evaluated in each daily cycle are recorded in the capture stream ($D_1$ preserved).
  - **Deterministic ID:** Reuse `compute_decision_id(symbol, cycle_id, engine_sha, decision_schema_version)` from the existing capture engine. Its canonical JSON tuple encoding prevents concatenation ambiguity.
  - **Append-Only:** Capture stream written as immutable JSONL to `data/forward_pilot/capture_stream.jsonl`.
  - **Storage Durability Attested:** Fails closed if running against unmounted or ephemeral storage without persistent volume attestation.
- **Rollback / Disable Procedure:** Disable forward capture script; zero web service disruption.
- **Effort Range & Decision:**
  - Effort: 2–3 engineering days.
  - Key Decision: Operator-triggered systematic batch confirmed; automated scheduling deferred to post-pilot operational phase.

---

### Milestone 3: Simulate Execution & Settle Outcomes

**Objective:** Simulate order fills without broker APIs, and advance settlement automatically against post-market price tape using `PaperTradingOutcomeEvaluator`.

- **Scope & Reusable Components:**
  - Reuse `PaperTradingOutcomeEvaluator` under frozen Contract V1.0.1.
  - Execution Simulation Rules (Exact Contract Parity):
    - **Opening Gap Up Above Corridor Max:** If `Open > Corridor Max`, the setup does **NOT** trigger at open. A limit entry can only be filled if intraday `Low <= Corridor Max`. If `Low > Corridor Max`, no entry triggers on that session. A gap above a limit-entry ceiling **must not automatically receive a fill at that ceiling**.
    - **Opening Gap Down Below Stop Floor:** If `Open <= Stop Floor`, setup is invalidated at open (`PRE_ENTRY_STOP_INVALIDATION_GAP_DOWN`, untriggered).
    - **Open Inside Corridor:** Fills at session `Open` if `Corridor Min <= Open <= Corridor Max`.
    - **Ambiguous Intrabar Collision:** If `Low <= Stop Floor` on the same session as entry, marked `UNRESOLVED_INTRABAR` on daily OHLC; resolved only when time-ordered 1-minute bars exist.
    - **Transaction Cost Friction:** Flat 25 bps deducted from gross return.
  - Settlement Horizon & Multi-Session Lifecycle:
    - **Capture Closes at Session 20; Settlement Continues:** The last recommendation emitted on Session 20 is entitled to its full 5-session entry window (Sessions 21–25) and full 20-session holding horizon (up to Session 44).
    - **Total Timeline:** Up to **44 trading sessions** (~9 calendar weeks) to ensure 100% terminal maturity for all signals.
  - Price Adjustment Provenance:
    - Reading `indicators.quote[0]` alone does not prove split adjustment.
    - Require documented adjustment provenance: inspect provider split events metadata, check corporate actions calendars, and confirm split factors. Any unverified corporate action leaves adjustment status `UNRESOLVED_ADJUSTMENT_PROVENANCE` (fails closed).
  - Manual Offline Settlement CLI (`scripts/settle_forward_cohort.py`):
    - Implemented for retained price batches only; **16:30 ET** remains a proposed later schedule, not an active job.
    - Reuses the frozen evaluator, retaining unresolved daily collisions; prior result-change events are preserved in each atomic state snapshot.
    - Per-cohort checkpoint at `data/forward_pilot/<cohort-id>/settlement_ledger.json` under the shared capture `flock`.
    - Missing required sessions block the batch before ledger advancement. No prices or outcomes are invented.
    - Twenty capacity slots are allocated chronologically, independent of returns; uncertain entries reserve capacity without counting as confirmed fills. Multiple same-symbol setups remain independent simulated recommendations, not a portfolio.
    - Retained adjustment metadata is an input assertion, not independent source authentication. The qualified market-data adapter (`analyst_dashboard/governance/market_data_adapter.py`) enforces cryptographic raw-byte retention, fail-closed SHA-256 verification, and explicit `corporateActionReference` checks (rejecting field-name and empty-split-list inferences).
- **Dependencies & Accountable Owner:**
  - Dependencies: Milestone 2 capture stream; daily price bar ingest.
  - Accountable Owner: `daakara`.
- **Smallest Reviewable PR:**  
  `PR-PT-03: Implement idempotent outcome settlement batch runner, market-data adapter, and event ledger`.
- **Quality Gates & Acceptance Criteria:**
  - **Idempotency:** Executing settlement runner 5 times consecutively produces identical state and zero duplicate outcome events.
  - **Immutability:** Signal inputs (corridor, stop, targets) are never modified during settlement.
  - **Adjustment Basis:** Documented adjustment provenance verified prior to settling any trade; unverified corporate actions fail closed.
- **Rollback / Disable Procedure:** Stop daily settlement batch; state is preserved in append-only JSON ledger.
- **Effort Range & Decision:**
  - Effort: 3–4 engineering days (Completed on PR #322).
  - Key Decision: Qualified Yahoo Finance v8 chart adapter via `MarketDataPayload` with non-dividend-adjusted OHLCV (`quote` array only, `adjclose` strictly ignored) and authentic corporate action verification.

---

### Milestone 4: Operational & Strategy Review

**Objective:** Produce an objective evaluation separating operational software reliability from strategy expectancy, governed by pre-declared continue, investigate, and stop thresholds.

- **Scope & Reusable Components:**
  - Scoreboard Generation (`scripts/generate_forward_pilot_report.py`):
    - Operational Scorecard: Capture completeness %, duplicate rate (0% required), missing session rate %, settlement idempotency pass rate.
    - Strategy Metrics: Trigger rate ($D_3 / D_1$), win rate ($TP1 / D_5$), stop-out rate ($Stop / D_5$), horizon exit rate ($Neutral / D_5$), mean net return, average R-multiple, max adverse excursion (MAE), max favorable excursion (MFE).
  - Explicit Methodological Boundaries:
    - **Zero Model Tuning:** Model weights, thresholds, and filters must not be adjusted based on pilot results (prevents p-hacking).
    - **No Over-Interpretation:** Sample size ($N \le 20$ triggered entries) is recognized as statistically underpowered for Sharpe/drawdown assertions. Small-sample performance prompts review rather than immediate mechanical stops.
    - **No Implied Portfolio P&L:** Individual trade R-multiples reported; portfolio P&L requires explicit sizing, cash reserves, and overlapping-position allocation rules.
  - Review & Threshold Structure:
    - **Hard Stop Criteria (Immediate Operational Halt):**
      - Any detected look-ahead bias (recommendation timestamp not $\ge 12\text{h}$ prior to $T+1$ open).
      - Any duplicate decision ID or settlement record (idempotency failure).
      - Any data corruption or unverified corporate action / stock split.
      - Any attempt to place live broker orders or write to production databases.
    - **Investigation / Review Triggers (Provisional Research Review):**
      - Strategy return and stop-rate figures (e.g. stop-out rate $> 50\%$, negative net expectancy) are treated as **provisional research signals** in a small sample, prompting qualitative review and setup audit rather than a premature automated stop.
      - Intrabar collision rate $> 20\%$.
      - Trigger rate $< 10\%$.
- **Dependencies & Accountable Owner:**
  - Dependencies: All forward signals matured or exited (up to Session 44).
  - Accountable Owner: `daakara`.
- **Smallest Reviewable PR:**  
  `PR-PT-04: Automated pilot scoreboard generator and gate review template`.
- **Quality Gates & Acceptance Criteria:**
  - Scoreboard outputs both human-readable Markdown and machine-readable JSON matching Contract V1.0.1 schema.
  - Clear frontend/report disclaimer distinguishing simulated paper execution from live trading.
- **Rollback / Disable Procedure:** Informational reporting only; zero production impact.
- **Effort Range & Decision:**
  - Effort: 1–2 engineering days.
  - Key Decision: Formal decision by `daakara` whether to continue forward capture, calibrate sizing models, or maintain hold.

---

## 4. Minimum Quality Gates Matrix

| Rule ID | Quality Gate Invariant | Enforcement Mechanism | Failure Action |
| :--- | :--- | :--- | :--- |
| **QG-PT-01** | **Zero Look-Ahead Bias** | Recommendation timestamp must precede $T+1$ market open by $\ge 12\text{h}$. | Immediate operational stop. |
| **QG-PT-02** | **Zero Cherry-Picking** | Every evaluated symbol logged to capture stream ($D_1$ preserved). | Audit failure; report blocked. |
| **QG-PT-03** | **Zero Duplicate Records** | Deterministic IDs under POSIX `flock` and unique SQLite index. | Process terminates fail-closed. |
| **QG-PT-04** | **Fail-Closed Missing Data** | Incomplete bars marked `OUTCOME_DATA_MISSING`; no synthetic fills. | Trade remains open/unresolved. |
| **QG-PT-05** | **Strict Determinism** | Re-running evaluator against identical bars yields identical recomputed results. | Gate failure; release blocked. |
| **QG-PT-06** | **Epistemic Labeling** | UI/Reports explicitly label simulation vs broker paper vs live fills. | Gate failure; presentation blocked. |
| **QG-PT-07** | **No Portfolio Over-Claim** | Prohibits implied portfolio equity curves without position sizing rules. | Report redacts portfolio claims. |

---

## 5. Governance Boundaries & Hold Decoupling

- **Planning Boundary:** This roadmap authorizes design and review only. It does **not** authorize activating live background schedulers, placing broker orders, modifying production databases, or altering frozen Contract V1.0.1.
- **Decoupling from Existing Holds:**
  - `HOLD_SEC_001` (SEC Form 4 Ingestion): **Completely decoupled.** The forward paper-trading pilot uses market price bars and Radar setups, not SEC EDGAR filings.
  - `HOLD_SCAN_001` (Continuous Scanning): **Preserved.** Continuous background polling remains disabled. The operational pilot begins exclusively with an operator-triggered systematic ten-symbol batch per completed session; background daemon/cron scheduling is deferred as a later operational choice.
  - `HOLD_EXEC_001` (Financial Execution): **Strictly Preserved.** No broker order routing is implemented. All executions are simulated internally under Contract V1.0.1 rules.
  - `PHASE_3_HOLD` (Autonomous Trading): **Preserved.** Model weights remain frozen; no online learning.
  - **Storage Attestation Invariant:** Physical persistent volume attestation via `analyst_dashboard.governance.storage.attest_persistent_storage()` is mandatory before any operational candidate capture or settlement execution to prevent ephemeral data loss.

---

## 6. Recommended Minimum Viable Pilot (MVP)

```yaml
Pilot Configuration:
  Universe: 10 Liquid US Assets
    Index & Sector ETFs: ["VOO", "QQQM", "IWM", "XLE", "XLK"]
    Mega-Cap Equities: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
  Capture Window: Exactly 20 consecutive XNYS trading sessions
  Evaluation Volume: 20 sessions × 10 symbols = 200 evaluated records (D1)
  Simulated Entry Cap: Maximum 20 triggered entries (D3 <= 20)
  Settlement Horizon: Up to 44 XNYS trading sessions (allowing session 20 signals full 5d entry + 20d hold)
  Execution Mode: 100% Internal Simulation (HOLD_EXEC_001=ACTIVE strictly preserved, zero broker APIs)
  Trigger Mechanism: Operator-triggered systematic ten-symbol batch (scheduling is a later operational choice)
  Storage Attestation: Mandatory durable volume attestation (attest_persistent_storage()) before operational use
  Settlement: Manual offline batch CLI replay (16:30 ET proposed schedule deferred)
  Transaction Friction: 25 bps flat fee per triggered trade
  Storage Location: data/forward_pilot/ (capture_stream.jsonl, settlement_ledger.json, raw_responses/)
```

---

## 7. Next Three Concrete Actions

1. **Submit `PR-PT-01` (Baseline Preservation):** Add `tests/test_paper_trading_baseline_determinism.py` recomputing all 19 historical pilot records from first principles and locking the baseline.
2. **Submit `PR-PT-02` (Forward Capture CLI):** Implement `scripts/capture_forward_candidates.py` configuring the 10-symbol MVP universe for 20 daily capture sessions under Contract V1.0.1.
3. **Submit `PR-PT-03` (Automated Settlement Runner):** Implement `scripts/settle_forward_cohort.py` providing idempotent, fail-closed daily outcome settlement with append-only event logging.

---

## 8. Ready-to-Run Prompt for First Implementation PR (`PR-PT-01`)

```markdown
Implement PR-PT-01: Historical Baseline First-Principles Recomputation & Isolation Lock for ARX Paper Trading.

Follow docs/planning/ARX_PAPER_TRADING_ROADMAP.md Milestone 1:

1. Create `tests/test_paper_trading_baseline_determinism.py`:
   - Load the frozen 19-record inventory and signal plans from `docs/governance/ARX_BOUNDED_OFFLINE_OUTCOME_PILOT_RECEIPT.json`.
   - Load raw price evidence from retained fixtures and `scratch/acquired_etf_bars.json`.
   - Recompute every trade outcome from scratch using `PaperTradingOutcomeEvaluator.evaluate_signal`.
   - Assert from first principles:
     * D1=19 total, 1 synthetic excluded, D1(natural)=18.
     * Exactly 10 triggered entries (D3=10) and 8 untriggered setups.
     * Recomputed trade outcomes match Contract V1.0.1:
       - NET: SUCCESS (+14.59% gross, +14.34% net, +3.09R)
       - TMDX: FAILURE (-10.55% gross, -10.80% net, -1.00R)
       - VOO: NEUTRAL (-0.55% gross, -0.80% net, -0.14R)
       - JEPI: NEUTRAL (+1.29% gross, +1.04% net, +0.37R)
       - QQQM: ENTRY_NOT_TRIGGERED
       - CELH & CRSP: UNRESOLVED_INTRABAR on pure daily; FAILURE on 1-min intraday bars.
     * Statistical averages recomputed from individual trades match:
       - Historical Net: +0.60%, Reconciled R: +0.17R (prior reported -1.78% net and -0.01R reconciled from verified fills)
       - Prospective Net: +0.47%, Reconciled R: +0.15R (prior reported +0.18R reconciled)
       - Pooled Net: +0.54%, Reconciled R: +0.16R (prior reported -0.65% net and +0.08R reconciled)
       - Pure Daily Net (N=8): +1.48%, Reconciled R: +0.45R (excluding unsequenced CELH and CRSP)
     * Serialized output hash matches SHA-256 `a9675c9592c474c2b3ad94998922c3c184d4a27305b091e42cdcd9e92eda2895`.

2. Establish isolated durable project storage:
   - Create `data/forward_pilot/` with README.md documenting storage layout.
   - Assert in unit test that forward cohort prefix `COHORT_FORWARD_*` cannot overwrite historical records.

3. Strict Execution Isolation:
   - Zero network socket calls, zero database writes, and zero modifications to Contract V1.0.1.

Run pytest to verify 100% pass, and provide git diff for review.
```
