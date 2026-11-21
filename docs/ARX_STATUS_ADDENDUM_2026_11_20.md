# ARX Terminal — Integrated Status Addendum (2026-11-20)

> **Authoritative Operational Reconciliation (Reconciled 2026-11-21, Production Baseline SHA [`2eccc64a2cb3c649ad4d584a942285a6406d1ce0`](https://github.com/daakara/finance/commit/2eccc64a2cb3c649ad4d584a942285a6406d1ce0)):**  
> This document provides operational status across all five ARX Terminal engineering workstreams. It is supplemented by the authoritative [Current Reconciliation Register (2026-11-21)](planning/ARX_CURRENT_RECONCILIATION_REGISTER_2026_11_21.md) and [Comprehensive Backlog Coverage Audit](reports/ARX_BACKLOG_COVERAGE_AUDIT_2026-11-21.md). Day 1 trial completed Nov 20; worker is currently safely recovered in fail-closed `STANDBY` / `DISABLED` role. All immutable receipts, historical evidence, and audit logs are preserved untouched.

---

## 1. Operational Framework: Four-State Separation

In accordance with institutional governance standards, ARX Terminal strictly separates four dimensions of project reality:

1. **Implementation State (Code):** Has code been written, tested, and passed Release Qualification (RQ-01..13)?
2. **Deployment State (Runtime):** Is the release artifact actively serving traffic on Railway or Cloudflare Pages?
3. **Operational Authorization (Governance):** Has the System Owner issued an explicit, scoped grant for live execution?
4. **Operational Holds (Safety):** Are fail-closed architectural boundaries actively blocking unauthorized execution?

A feature may be **fully implemented** and **deployed to production** while remaining **operationally held** and **unauthorized for autonomous execution**.

---

## 2. Workstream Status & Operational Truth

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        ARX WORKSTREAM RECONCILIATION SUMMARY                           │
├────────────────────┬──────────────────┬─────────────────┬──────────────┬───────────────┤
│ Workstream         │ Implementation   │ Deployment      │ Auth Status  │ Hold State    │
├────────────────────┼──────────────────┼─────────────────┼──────────────┼───────────────┤
│ 1. Smart Money     │ COMPLETE (PR#316)│ Railway(STANDBY)│ Day 1 Done   │ HOLD_SEC_001* │
│ 2. Radar           │ COMPLETE (PR#314)│ Cloudflare LIVE │ Discovery Held │ HOLD_SCAN_001 │
│ 3. Confluence      │ DEPLOYED (PR299) │ Production LIVE │ Research OK  │ CONFLUENCE_HLD│
│ 4. Pre-Flight / UX │ DEPLOYED (PR303) │ Production LIVE │ Research OK  │ PHASE_A3_UX_HL│
│ 5. Infrastructure  │ VERIFIED (2eccc6)│ Cloudflare / PG │ STANDBY      │ HOLD_EXEC_001 │
└────────────────────┴──────────────────┴─────────────────┴──────────────┴───────────────┘
* HOLD_SEC_001 scoped: bounded two-issuer ingestion-only trial authorized; general continuous scanning & snapshot publication strictly held.
```

---

### Workstream 1: Smart Money (`ARX-SM-ACT-020`, Issue #165 & PRs #304–#312)

- **Objective:** Establish compliant, point-in-time SEC Form 4 insider transaction surveillance without violating SEC fair-access rate limits, polluting prospective learning datasets, or risking unmonitored egress.
- **Milestone Deployments (PR #304 through PR #312):**
  - **PR #304 (`c6e3c03b`):** Qualified atomic Step 7 commit and read-only point-in-time historical snapshot extraction under repeatable-read isolation.
  - **PR #305 (`308f5392`):** Issuer-bound historical coverage and statutory amendment-effect classification (`FULL_REPLACEMENT`, `PARTIAL_CORRECTION`, `ADDITIONAL_TRANSACTION`) via remarks/footnotes without DDL mutations. 72/72 evidence integrity tests passed across Gates 1–4.
  - **PR #306 (`8b3a40fc`):** Bounded one-shot NVDA executor hardening:
    - Enforced non-overridable ceilings: $\le 4$ requests, $\le 5$ req/s, $\le 3$ unseen documents, $\le 30.0$s total timeout, $\le 15$ min authorization lease.
    - Decoded HTTP Content-Encoding (gzip/deflate) preserving absolute response deadline.
    - Added read-only dry-run checks and unified CLI entrypoints.
    - Supervised single-cycle live NVDA execution completed on Nov 19.
  - **PR #307 (`06f1ffdf`):** Qualified AAPL as the single alternative-issuer canary:
    - Accession `0001140361-26-044773` (accepted `2026-11-18T23:30:10Z`, 3 transactions: S, S, G).
    - Verified atomic persistence, hash-conflict detection, rollback, cleanup, and replay in disposable PostgreSQL.
    - Supervised single-cycle live AAPL canary executed cleanly on Nov 19.
  - **PR #308 (`540075c8`):** Offline qualification of sequential two-issuer execution (AAPL then NVDA):
    - Verified 4 disposable-PostgreSQL scenarios: identical state before/after on no-work, separate issuer rows and PARTIAL receipts on actual work, cross-issuer XML rejection halting without writes, and failure rollback without second-issuer execution.
  - **PR #309 (`16549be1`):** Sequential canary pair wrapper qualification and deterministic cryptographic content preservation:
    - Upgraded and versioned live wrapper [`scripts/execute_sequential_canary_pair.py`](../scripts/execute_sequential_canary_pair.py) (v1.1.0) with canonical argument parser (`--dry-run`, `--live`, `--dsn`, `--auth-issued-at`, `--auth-expires-at`, `--timeout`, `--verbose`).
    - Implemented cryptographic content preservation fingerprinting in [`analyst_dashboard/smart_money/storage/postgres_preservation.py`](../analyst_dashboard/smart_money/storage/postgres_preservation.py) across all 12 protected tables via PostgreSQL `to_jsonb(t)` row serialization and SHA-256 row hashing.
    - Differentiated recovery assurance: structural/read/trigger contracts `VERIFIED_RECOVERABLE` vs. source parity / global roles `NOT_CHECKED`.
    - Added 8 new wrapper regressions in `TestSequentialPairWrapperRegression` (125 total runner tests passed).
  - **PR #311 (`b937706e`):** Crash-safe two-issuer ingestion-only trial qualification:
    - Enforced transaction advisory lock (`pg_advisory_xact_lock`), durable pre-reservation of request budget before network egress, distinct attempted slot counting, and snapshot publication isolation (`scanner_snapshots` and `scanner_active_snapshots` strictly untouched).
    - Qualified by hosted release run 44690336127 (all 13 RQ controls passed, 207 backend tests, 72 SEC evidence tests, 150 canary tests, frontend regression/typecheck).
  - **PR #312 (`868d7afc`):** Bounded ingestion-only trial slot dispatch integration:
    - Implemented `dispatch_trial_slot()` in worker poll; evaluates America/New_York slot eligibility against `TwoIssuerTrialScheduler`, verifies unexpired authorization and exclusive owner lock, enqueues exactly one delivery per trial/slot (`TRIAL_SLOT_FORM4`).
    - Deduplicates terminal and restarted deliveries without DDL, binds processing to original slot, and rejects slot rollover under pipeline lock. Preserves legacy operator execution path.
    - Qualified by hosted release run 44696986156 (all 13 RQ controls passed, 223 backend tests, 72 evidence tests, 150 canary tests, frontend regression/typecheck). Deployed to Railway production.
    - Passed all 13 Release Qualification controls locally and in hosted CI ([Run 44632318474](https://github.com/daakara/finance/actions/runs/44632318474)).
- **Completed One-Shot Candidate & Canary Ingestions:**
  - Qualified and ingested AAPL Form 4 accession `0001140361-26-044952` (accepted `2026-11-19T23:30:21Z`, Jennifer Newstead sale of 2,398 shares at $352.77, transaction date `2026-11-17`).
  - Verified atomic persistence, hash-conflict detection, honest `PARTIAL` reconciliation, and snapshot preservation.
  - Production database holds exactly 16 NVDA Form 4 accessions (100% trailing 90-day backfill baseline) + 6 AAPL Form 4 accessions (including canary `0001140361-26-044773` and accession `0001140361-26-044952`), totaling 140 parsed transactions. Active snapshot `SNAP_SM_1792939989_48f30db6` (`8082f053...`) intact.
  - All supervised one-shot canaries and candidate ingestions under Issue #165 are **COMPLETE**.

- **Authoritative Provenance Resolution (v2 Supersession):**
  - Verified and recorded authoritative superseding provenance receipt v2 (`6c7bf075-0d62-4a95-9727-044515c3e376`) in production database `sm_reconciliation_receipts`:
    - Supersedes `dec95ff5-b6ca-42cf-be6a-6b435c6c4e5f` without mutating the original record.
    - Attributes historical execution strictly to `16549be1acb9fdafef3c6b97f9115b674f9f4842`.
    - Canonical correction hash: `4c3a59604234d0ff417e391774c1cb09425aafc6251cd81c22729c2a8953ae83`.
    - Preserves full envelope: catalog hash (`35e55888...`), XML hash (`93aa1a0a...`), watermark (`2026-11-20T00:00:00+00:00`), material gaps, and `PARTIAL` coverage.
    - Repository readback `get_latest_reconciliation_receipt(covered_tickers=['AAPL'])` returns v2.

- **Two-Issuer Recurring Ingestion Trial Day 1 Armed Configuration:**
  - **Trial ID:** `ARX-TRIAL-2026-NVDA-AAPL-001`.
  - **Implementation vs. Operational Activation:**
    - *Implementation State:* **COMPLETE & DEPLOYED** (PR #311 trial architecture under SHA `b937706e`; PR #312 durable slot dispatch under SHA `868d7afc`).
    - *Operational Activation:* **ARMED & AWAITING NATURAL EXECUTION.** Zero automated recurring slots have executed yet. Autonomous ingestion occurs strictly when wall-clock time reaches scheduled slots during active authorization.
  - **Calendar Window:** `2026-11-20` (Friday) through `2026-11-24` (Tuesday) America/New_York (weekend `2026-11-21`/`2026-11-22` automatically excluded per SEC dissemination calendar).
  - **Daily Execution Slots:** `09:15`, `13:00`, `16:30`, `20:30` EST.
  - **Durable Quotas:** $\le 4$ cycles/day, $\le 12$ cycles/trial; $\le 32$ requests/day, $\le 96$ requests/trial; $\le 8$ requests/cycle ($\le 4$/issuer).
  - **Active Day 1 Authorization Grant:** Single-reading grant `dc58b294b7a744da9730c5a2b97725b1`, issued `2026-11-20T08:08:31Z`, expires `2026-11-21T08:03:31Z` (exactly 23h 55m duration, $\le 23\text{h}55\text{m}$). Bound strictly to deployed release SHA `868d7afcd528006c2323e6b79b5a17f2f20f677c`.
  - **Worker Runtime State:** Online in Railway production container, running in `ARX_SEC_EGRESS_ROLE=OWNER`, holding exclusive session advisory lock `401007311509093371` (`classid=93366790, objid=1926593531`), polling event loop in `HEALTHY` state.
  - **Next Scheduled Slot:** `2026-11-20:SLOT_1:0915` (09:15 EST / 14:15 UTC). Zero manual run requests enqueued.
  - **Teardown Boundary & Stop Semantics:** Runtime hard gate in code is `ARX_CONTINUOUS_AUTH_EXPIRES_AT` (`ARX_TRIAL_SHUTDOWN_AT` is not read by runtime code). The final grant covering Day 3 is capped at `2026-11-25T02:35:00Z` (November 24 at 21:35 EST: Slot 4 20:30 + 60m admission window + 5m bounded run/cleanup). `00:00Z` cutoff is prohibited as premature.
  - **Current Operational Verdict:** **`ARMED` (Awaiting Natural Execution)**.

- **Historical Readiness Audit Findings (Recorded Evidence 2026-11-20T08:08:31Z – 2026-11-20T08:29:08Z):**
  > [!NOTE]
  > The following findings represent historical timestamped evidence established during the preflight readiness audit and Day 1 arming ceremony. They are documented here for audit completeness and are **not claimed as freshly re-queried**.
  - **Evidence Freshness:** Following early morning connection failures during initial verification, subsequent authenticated read-only queries successfully established authoritative database state prior to trial arming. Earlier connection drops were labeled as historical.
  - **Advisory Lock (QG-07):** Verified completely unheld (`pg_locks` returned 0 locks for `401007311509093371`) prior to worker arming. Transitioned to exclusive held state upon container startup in `ARX_SEC_EGRESS_ROLE=OWNER`.
  - **Queue Hygiene (QG-08):** Verified 0 active queue entries (`PENDING`, `CLAIMED`, `RUNNING`, `RETRY_ELIGIBLE`) in `sm_run_requests` prior to arming. Zero operator-injected runs present.
  - **Baseline & Snapshot Lineage (QG-10):** Verified active snapshot `SNAP_SM_1792939989_48f30db6` (`8082f053...`) intact and unchanged. The 16 NVDA baseline accessions and 6 AAPL accessions verified preserved. Trial architecture enforces 0 snapshot table mutations.

- **Merge & Deployment Isolation Verification (PR #310 Merge Verdict: DEFERRED):**
  - **Railway Auto-Deploy Mechanism:** Pushes and merges to `release/production` trigger automated container builds and rolling redeployments on Railway.
  - **Worker Disruption Impact:** Redeploying `smart-money-worker` severs the PostgreSQL TCP connection, immediately releasing exclusive session advisory lock `401007311509093371`.
  - **Grant Binding Invalidation:** The active Day 1 grant `dc58b294...` is cryptographically bound to `release_sha = '868d7afcd528006c2323e6b79b5a17f2f20f677c'`. A new container running a merge commit SHA will fail closed on preflight release SHA mismatch, terminating autonomous execution.
  - **Action:** In accordance with trial isolation governance, **merging PR #310 is DEFERRED** until a controlled post-trial maintenance window. PR #310 is retained on branch `docs/reconcile-arx-status-2026-11-20` in a qualified, passing state. Zero deployment controls are bypassed, authorization is not modified, and the worker remains undisturbed.

- **Hold State: `HOLD_SEC_001=ACTIVE (Scoped Trial Ingestion Authorized)`.** Continuous background market-wide ingestion, daily 10-issuer ingestion, and snapshot publication remain strictly held; bounded ingestion-only execution is authorized exclusively for the two-issuer trial profile.

---

### Workstream 2: Market Opportunity Radar (`ARX-RADAR-018` & Issue #166, #230)

- **Objective:** Eliminate deceptive presentation, maintain Stage 2 screening separation, and provide transparent Smart Money flow attribution.
- **Implementation & Deployment State: DEPLOYED & PRODUCTION VERIFIED.**
  - PR #285 merged: Eliminated repetitive candidate rationale text and introduced contextual Learn Hub deep-links.
  - PR #292 merged: Reconciled Stage 3 population accounting and test fixture market isolation.
  - PR #294 merged: Implemented read-only Smart Money evidence adapter for Radar cards with fail-closed hold disclosures.
  - Live on `https://www.arxterminal.com/radar/`.
- **Radar Readiness Audit & Historical Blockers (Historical Findings Pending Revalidation):**
  > [!NOTE]
  > The following blockers represent historical audit findings from the Radar readiness assessment. They are recorded here as **historical findings pending revalidation** and govern the continued active enforcement of `HOLD_SCAN_001`:
  - **Stale Market Data (439/500 Symbols):** Exactly 439 out of 500 candidate universe symbols exhibited market data beyond acceptable freshness thresholds during screening universe inspection.
  - **Missing Provenance Records (67 Symbols):** Exactly 67 symbols lacked verifiable upstream vendor attribution, origin timestamps, or transformation audit traces in the universe store.
  - **Snapshot Release Lineage Divergence:** The static published Radar universe snapshot serving client traffic was generated from an earlier release lineage that differed from the active deployed backend SHA.
- **Hold State: `HOLD_SCAN_001=ACTIVE`.** In light of these historical findings pending revalidation, natural background scanning and automated universe generation remain strictly disabled; a single static frozen universe (250 evaluated, 100 Stage 2) is served with discovery held.

---

### Workstream 3: Confluence Scoring & Architecture (`ARX-CONFLUENCE-002/003`)

- **Objective:** Guarantee mathematical determinism, full multi-pillar provenance parity, and truthful client presentation across all four analytical pillars.
- **Implementation & Deployment State: DEPLOYED & PRODUCTION VERIFIED.**
  - PR #289 & #291 merged: Form 4 classification, CDNS price parity, ratified engine manifest v2.5.3, and RQ-10/RQ-13 enforcement.
  - PR #295 merged (`b485e441`): Distinguished unverified and held confluence evidence, mapped `pipeline_held` to `UNAVAILABLE`, preserved failing pillar severity colors.
  - PR #296 merged (`2fec4d2c`): Corrected FRED macro contract in `api/routes/regimes.py` and eliminated `$0.00` placeholder purchases in `InstitutionalFeeds.tsx`.
  - PR #299 merged (`ee8ceab7`): Confluence drawer (`WhyInspectModal`) presentation truth:
    - Authoritative composite score displayed in header.
    - Displays raw score before coverage cap (`rawConfluenceScore / 100`).
    - Displays coverage cap limit (`coverageCap / 100`) and verified pillar count (`N/4`).
    - Displays prominent earnings & binary event warning banner (`⚠️ HIGH BINARY GAP RISK`).
    - Replaced conflicting point impact labels with factual backend pillar scores (`30/100`, `UNVERIFIED`, `UNAVAILABLE`, `85/100`), status badges, and 25% base weight governance rules.
- **Hold State: `ARX_CONFLUENCE_INTEGRITY_HOLD=ACTIVE`.** Confluence scoring remains in governed research mode; automated live signal promotion is held pending formal post-closure review.

---

### Workstream 4: Pre-Flight Decision Authority & UX

- **Objective:** Enforce fail-closed clearance safeguards, epistemic mode separation, and eliminate false live clearance.
- **Implementation & Deployment State: DEPLOYED & PRODUCTION VERIFIED.**
  - PR #286 merged: Simplified Pre-Flight Reward vs. Risk presentation under Issue #181.
  - PR #287 merged: Pre-Flight decision authority, evidence freshness, and epistemic safeguards under Issue #197.
  - PR #288 merged: In-memory analysis cache TTL bounds (15m max, 5m live), live freshness string extraction, and atomic OHLCV provenance replacement under Issue #198.
  - PR #297 merged: Sector-appropriate labeling for tech, healthcare, financials, and business catalysts in `CatalystForecastCard.tsx`.
  - PR #298 merged: Synchronized `CatalystForecastCard.tsx` with selected ticker during loading.
  - PR #300 merged (`bb9fdcf4`): Quote provenance & sizing authorization safeguards:
    - Missing sizing strictly blocks clearance with `CANONICAL SIZING PROHIBITED`.
    - Delayed or unverified quotes fail closed with `DELAYED PRICE OBSERVATION` or `UNVERIFIED PRICE PROVENANCE`.
  - PR #302 merged (`7a0f7676`): Canonical second profit target resolution (DEF-TP2-001):
    - Passed canonical ATR-anchored `take_profit_2` from execution ladder into `PreFlightChecklistInputs`, eliminating synthetic client-side 1.12x markup.
  - PR #303 merged (`933295cc`): Catalyst call time disentanglement (DEF-CAT-001):
    - Disentangled earnings results release (16:20 ET) from conference call (17:00 ET) in Catalyst card and Pre-Flight earnings warning.
- **Hold State: `PHASE_A3_UX_HOLD=ACTIVE`.** Progressive disclosure redesign (PR #274) remains held in draft awaiting authentic 5-participant formative usability study.

---

### Workstream 5: Infrastructure & Production Telemetry

- **Production Deployment Baseline:** `release/production` commit SHA [`2eccc64a2cb3c649ad4d584a942285a6406d1ce0`](https://github.com/daakara/finance/commit/2eccc64a2cb3c649ad4d584a942285a6406d1ce0).
- **Cloudflare Pages:** Serving verified production release on `https://www.arxterminal.com/`.
- **Railway Services:**
  - `web` (`analyst-dashboard`): FastAPI 0.110+ online, serving live endpoints.
  - `smart-money-worker`: Recovered to fail-closed `STANDBY` / `DISABLED` posture (`ARX_SEC_EGRESS_ROLE=DISABLED`, `ARX_CONTINUOUS_INGESTION_AUTHORIZED=false`). Session advisory lock `401007311509093371` is unheld (0 held locks, 0 active/retryable queue rows). Day 1 completed; Day 2 preflight pending.
  - `smart-money-backup`: Automated cron (`0 */6 * * *`) online. Routine backups stored on Cloudflare R2; isolated restore receipts required prior to trial dispatch.
  - `postgres`: PostgreSQL 16.x holding full 16 NVDA Form 4 accessions (100% trailing 90-day baseline) + 6 AAPL Form 4 accessions (including canary `0001140361-26-044773` and accession `0001140361-26-044952`), totaling 140 parsed transactions. Active snapshot `SNAP_SM_1792939989_48f30db6` (`8082f053...`) intact.
- **Hold State: `HOLD_EXEC_001=ACTIVE`.** Zero live trade execution code exists in production runtime; all orders are simulated.

---

## 3. Separately Tracked Defects & Actionable Backlog

| Defect ID | Defect Summary | Code Location & Root Cause | Status & Operational Disposition |
| :--- | :--- | :--- | :--- |
| **DEF-TP2-001** | **Target 2 Mismatch (Pre-Flight Modal)** | `frontend/lib/preflightEvaluator.ts` computed synthetic `safeTarget * 1.12` markup because `PreFlightChecklistInputs` omitted `takeProfit2`. Backend computes canonical ATR-based `take_profit_2` in `optimal_execution.py`. | **CLOSED & VERIFIED IN PRODUCTION (PR #302, SHA `7a0f7676`):** Passed canonical `take_profit_2` from execution ladder into `PreFlightChecklistInputs` and eliminated synthetic 12% calculation. |
| **DEF-CAT-001** | **Earnings Call Time Conflation (`16:00 ET Call`)** | `analyst_dashboard/corporate_events.py` (lines 280–281) assigned market-close epoch timestamp (16:00 ET) to `conference_call_time_et`. For NVDA, IR call is at 17:00 ET (results at 16:20 ET). | **CLOSED & VERIFIED IN PRODUCTION (PR #303, SHA `933295cc`):** Confirmed `results_release_time_et: "16:20"` and `conference_call_time_et: "17:00"` in primary-source schedule; disentangled display across Catalyst card and Pre-Flight warning. |
| **DEF-CAT-002** | **Empty 5-Year Revenue & Earnings Simulation Panel** | `analyst_dashboard/analyzers/catalysts.py` (lines 383–388) strips `multi_year_forecast` to `[]` under live-only invariant. `CatalystForecastCard.tsx` renders table headers with blank rows. | **Sprint 1 Active:** Add explicit empty-state disclosure in `CatalystForecastCard.tsx`: *"Forward multi-year projections awaiting verified consensus disclosures under live-only governance."* |
