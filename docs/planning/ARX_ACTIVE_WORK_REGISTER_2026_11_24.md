# ARX Terminal — Active Work Register and Backlog Reconciliation Matrix

**Document ID:** `PLN-REG-WORK-2026-11-24`  
**Effective Date:** 2026-11-24  
**Authoritative Production Baselines:**
- **Cloudflare Pages Frontend:** Deployment `f96c89d2-08e8-410e-a23c-e3c684f9f76c` | Candidate `6419213f56695cbc0a5fa6dc08b10e0496f37afa`
- **Railway Backend (Web API):** Service `56e38249-b78c-4a60-80d7-303ce7451bd8`, deployment `d58b5b15-3f06-4a6e-919c-223110f4ab92` (`SUCCESS` in Railway readback, 2026-11-24); source SHA and in-container file digests **NOT_CHECKED**. The older `bb115266` attribution below is historical, not current runtime proof.
- **Railway Backend (Worker):** Deployment `bba68cfb-f3cd-4d2d-afb0-05415999fe56` | Commit `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`
- **Railway Database:** PostgreSQL 16.x (`finance_smart_money` at `crossover.proxy.rlwy.net:17685`)

**Classification:** Canonical Backlog & Sprint Portfolio Register  
**Supersedes:** `docs/planning/ARX_ACTIVE_WORK_REGISTER_2026_11_17.md`

---

## 1. Executive Summary & Governance Authority

This register establishes the single authoritative inventory of all active, deferred, held, and completed engineering work across the ARX Terminal repository as of November 24, 2026. It formalizes a rigorous reconciliation pass across the last 7 days (2026-11-17 through 2026-11-24), closing completed and evidence-backed subtasks, correcting stale tracking entries, and explicitly delineating all genuinely unfinished workstreams.

### Core Achievements Reconciled in this Register:
1. **Dynamic Asset Discovery & Provider Fault Isolation (PR #323):** Deployed to production Cloudflare Pages (`f96c89d2` @ Candidate `6419213f`). Fully implements edge proxy autocomplete/search (`/api/assets/search`), in-flight submission gating, custom symbol launch suppression when provider matches exist, 404 `asset-not-found-state`, and 422 `insufficient-history-state`.
2. **Epistemic Tri-State Architecture & Decoupled Presentation:** Formally decouples Security Master listing identity (`Listing Identity: Verified`) from SEC filing coverage (`SEC Filing Coverage: Not verified`) and scoring eligibility. Codifies the 15-session baseline modeling gate.
3. **Sourced Fundamentals (PR #318):** Repository inclusion of `offline_reader.py` and `acquire_nvda_fundamentals.py` in successor source `6419213f` is verified. Python runtime deployment and use are **NOT_CHECKED**; Cloudflare frontend publication does not establish either.
4. **Historical Coverage Qualification Evidence (Commit `98ec10f8`):** Reconciled 13-issuer database inventory (55 filings, 152 transactions across all issuers; NVDA 16/48, AAPL 6/18). Preserves `PARTIAL_UNCERTIFIED` status with zero live egress during audit.
5. **BTU Offline Mechanics Validation (Commit `f1745ac6`):** Full offline discovery, XML parsing, and PostgreSQL replay mechanics validated and retained in canary package.
6. **Defect Resolutions:** Verified resolution of `DEF-TP2-001` (PR #302), `DEF-CAT-001` (PR #303), `DEF-CAT-002` (commit `c7fe3052`), and `DEF-SEC-004` (PR #315).

**Governance Authority:** This document provides operational inventory and sprint prioritization; it does **not** authorize lifting any of the six active operational holds or activating Sprint 5 execution.

---

## 2. Governance Lifecycle Taxonomy

To prevent ambiguities across development, testing, and production deployment, this register records five independent lifecycle facts. These are not a mandatory sequence: a service-specific upload can deploy an unmerged branch, and an implemented change may still have failing tests. Qualification applies only to its evidenced source/execution revision; commit lineage alone does not establish deployment or activation.

$$\boxed{\text{Implemented} \longrightarrow \text{Qualified} \longrightarrow \text{Merged} \longrightarrow \text{Deployed} \longrightarrow \text{Operationally Activated}}$$

1. **Implemented:** Code, configuration, and unit/integration tests are written and committed on an engineering branch; all localized automated tests pass within the branch workspace.
2. **Qualified:** The changeset has undergone and passed the formal Release Qualification suite (RQ-01 through RQ-13) or domain qualification harness, producing signed, tamper-evident qualification receipts and manifests.
3. **Merged:** The qualified changeset has been squash-merged or fast-forward merged into an integration or production trunk branch (`main`, `release/production`, etc.) with clean working trees.
4. **Deployed:** Artifacts built from the revision are published and running on live cloud infrastructure (Cloudflare Pages edge network, Railway Web API, or Railway Worker).
5. **Operationally Activated:** Background polling, live external network egress, autonomous scanning, or live capital routing are affirmatively enabled. **Requires formal Owner clearance lifting the governing operational hold.** An artifact may be fully deployed while remaining strictly held in standby/disabled mode.

---

## 3. Verified Production Runtime Baseline

The production environment operates across three distinct subsystem revisions, each bound to specific SHA commits and operational postures:

| Subsystem | Deployed SHA / Artifact | Hosting Environment | Operational Posture & State |
| :--- | :--- | :--- | :--- |
| **Cloudflare Pages Frontend** | Deployment `f96c89d2-08e8-410e-a23c-e3c684f9f76c` (Candidate `6419213f56695cbc0a5fa6dc08b10e0496f37afa`) | Cloudflare Pages (`arxterminal.com`) | **LIVE PRODUCTION.** Serves dynamic company autocomplete, submission gating, 404 unavailable states, decoupled limited-history presentation (`Listing Identity: Verified` / `SEC Filing Coverage: Not verified`), and NVDA baseline. 3/3 live audit checks PASS (Receipt: `ARX-ANT-ANALYSIS-EXPERIENCE-QUALIFICATION-RECEIPT.md`). |
| **Railway Backend (Web API)** | Deployment `d58b5b15-3f06-4a6e-919c-223110f4ab92`; source SHA **NOT_CHECKED** | Railway `web`, service `56e38249-b78c-4a60-80d7-303ce7451bd8` | Railway readback on 2026-11-24 reports **SUCCESS**. Deployment metadata contains no commit hash. PR #318 file digest parity and runtime reader integration are **NOT_CHECKED**; no in-container artifact or matching digest receipt was available in this closeout. |
| **Railway Backend (Worker)** | Deployment `bba68cfb-f3cd-4d2d-afb0-05415999fe56` (Commit `2eccc64a2cb3c649ad4d584a942285a6406d1ce0`) | Railway `smart-money-worker` service | **SUPERVISED STANDBY.** Running trial `ARX-TRIAL-2026-NVDA-AAPL-001` in ingestion-only mode (`ARX_SEC_EGRESS_ROLE=DISABLED` except for authorized slots). Single authorization lease expiring `2026-11-25T02:35:00Z`. Zero active queue rows, 0 lingering advisory locks. |
| **Railway Database** | PostgreSQL 16.x (`crossover.proxy.rlwy.net:17685`) | Railway `postgres` | Holds authoritative 13-issuer database inventory: 55 raw filings / 152 parsed transactions (NVDA 16/48, AAPL 6/18, 11 other issuers 33/86). Parker amendment linked. |
| **Operational Holds** | All 6 Holds Strictly Active | System-wide | `HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`. |

---

## 4. Pull Requests & Engineering Workstream Status

### 4.1 Recently Completed, Merged & Deployed Workstreams

| PR # | Branch / Title | Scope & Artifacts | Lifecycle Status & Evidence |
| :--- | :--- | :--- | :--- |
| **#323** | `feat/dynamic-asset-discovery`<br>`feat(search): dynamic company-name asset discovery and decoupled limited-history presentation` | Dynamic symbol & company name resolution via edge proxy `/api/assets/search`, query submission gating while discovery is in-flight, suppression of custom symbol launch buttons when provider matches exist, 404 `asset-not-found-state`, 422 `insufficient-history-state`, decoupled capability chips (`Listing Identity: Verified` / `SEC Filing Coverage: Not verified`), clarified 15-session baseline modeling gate. Candidate `6419213f`. | **QUALIFIED & DEPLOYED:**<br>13/13 RQ controls PASS (100.33s). Deployed to Cloudflare Pages `f96c89d2`. 3/3 live audit checks PASS. Full 63-file unit suite executed: 61/63 files passed (682/684 tests) with 2 pre-existing failures in `DecisionSurfaceIntegrity` and `Wave1DeclutterPreservation` present since baseline `2eccc64a`. Receipt: `ARX-ANT-ANALYSIS-EXPERIENCE-QUALIFICATION-RECEIPT.md`. |
| **#318** | `fix/fundamental-evidence-diagnostics`<br>`fix(provenance): clarify unverified fundamentals and unknown freshness` | Successor source `6419213f` and `62c19395` both contain `analyst_dashboard/research/sourced_fundamentals/offline_reader.py` (Git blob `fda992bea7da7d1bba141a4232051d316743dbb8`) and `scripts/research/acquire_nvda_fundamentals.py` (blob `a5a36aa4d0e445fdce753bcdd500a0e5aa72ffde`). | **REPOSITORY INCLUSION VERIFIED; RUNTIME NOT_CHECKED.** The reader is an offline research component; the acquisition file is operator tooling, not proof of a running WEB integration. No matching deployed file digests were available. [PR #318](https://github.com/daakara/finance/pull/318) is **OPEN / DRAFT / UNMERGED**, head `e375ad696f9aae0ac08a139249a6e2d6cfcc443e` in authenticated readback. Its older body references another head; qualification is not inferred from that stale text or frontend deployment. |
| **#315** | `fix/backup-retention-bounded-verification-001`<br>`fix(ops): bounded backup retention qualification` | Bounded backup retention qualification with fail-closed R2 integrity and isolated restore qualification. Defers trial retention during live run. Candidate `1cd1a0cf`. | **QUALIFIED & MERGED:**<br>Merged into `release/production` as `2eccc64a`. Deployed to Railway Worker (`bba68cfb`). Resolves `DEF-SEC-004`. |
| **#312** | `release/production`<br>`fix(sec): dispatch durable ingestion-only trial slots` | Supervised slot execution and verification for trial. Commit `868d7afc`. | **QUALIFIED & MERGED.** |
| **#309** | `feat/sec-aapl-nvda-pair-wrapper`<br>`gov(release): record release qualification receipts for PR 309` | Sequential pair wrapper qualification for AAPL/NVDA ingestion trial. Candidate `807e6d57`. | **QUALIFIED:**<br>13/13 RQ controls PASS. |
| **#305** | `fix/sec-ingestion-historical-coverage-amendments`<br>`fix(sec): issuer-bound historical coverage and amendment-effect classification` | Atomic Step 7 commit and read-only PIT historical snapshot extraction under repeatable-read isolation. Commit `f1f8b674`. | **MERGED & DEPLOYED:**<br>Merged into `release/production` as `308f5392`. Deployed to Railway Web API (`bb115266`). |
| **#304** | `fix(sec): qualify atomic ingestion and historical point-in-time snapshots` | Railway backend deployment `c6e3c03b`. | **MERGED & DEPLOYED.** |
| **#303** | `fix(catalysts): disentangle NVDA results release from conference call` | Primary-source schedule for NVDA in `corporate_events.py` (results 16:20 ET, call 17:00 ET). Commit `933295cc`. | **MERGED & DEPLOYED:**<br>Deployed to Cloudflare Pages. Resolves `DEF-CAT-001`. |
| **#302** | `fix(preflight): use the canonical second profit target` | Target 2 alignment in `PreFlightChecklistInputs`. Commit `7a0f7676`. | **MERGED & DEPLOYED:**<br>Resolves `DEF-TP2-001`. |
| **#301** | `docs(governance): ARX-DOCS-COMPREHENSIVE-RECONCILIATION-027` | Operational baseline, holds, active registers, and roadmap. Commit `ece65937`. | **MERGED.** |
| **#300** | `fix(preflight): enforce quote provenance and sizing authorization` | Missing sizing and unverified quotes block live clearance fail-closed. Commit `bb9fdcf4`. | **MERGED & DEPLOYED.** |
| **#299** | `fix(ui): show backend pillar scores, score before cap, coverage cap` | Confluence drawer presentation alignment. Commit `ee8ceab7`. | **MERGED & DEPLOYED.** |

---

### 4.2 Qualified Offline Implementations with Held Operational Activation

| Workstream / PR | Branch / Candidate | Implementation Scope | Offline Qualification Status | Operational Activation Status |
| :--- | :--- | :--- | :--- | :--- |
| **Paper-Trading Pilot** | `feat/arx-paper-trading-pilot`<br>Candidate `42bd2505` | Paper-trading pilot execution harness, virtual portfolio lifecycle, simulated order fills with slippage, equity curve tracking, risk boundary enforcement. | **QUALIFIED (Offline):**<br>13/13 RQ controls PASS (100% green across receipts `42bd2505`, `42518a05`, `487ee39d`). Offline simulation contracts verified. | **HELD (Unapproved):**<br>Operational activation in live forward trading is strictly prohibited under `HOLD_EXEC_001` and `PHASE_3_HOLD`. No broker egress or forward portfolio tracking authorized. |
| **PR #319** | `fix/sec-insider-flow-presentation`<br>Candidate `d3c611c4` | Decouples stored Form 4 disclosure metadata and transaction counts from held scoring; historical coverage and uncertified indicators separated from issuer support. | **QUALIFIED (Offline):**<br>13/13 RQ controls PASS (Receipt `13dbd552`). Candidate `d3c611c4`. | **HELD:**<br>Continuous live SEC ingestion remains disabled under `HOLD_SEC_001`. Scoring activation held under `ARX_CONFLUENCE_INTEGRITY_HOLD`. |
| **BTU Canary Validation** | `fix/sec-insider-flow-presentation`<br>Commit `f1745ac6` | BTU Peabody Energy (`0001570382-26-000015`) offline discovery, XML parsing, and PostgreSQL replay mechanics validation package (`data/canary/btu_canary_20261123/`). | **VERIFIED (Offline):**<br>Complete offline discovery, schema validation, XML parsing, and dual-store (SQLite/PostgreSQL) replay custody verified. | **OFFLINE ONLY:**<br>Retained canary fixture package. Zero live egress or worker scheduling. |

---

### 4.3 Qualification Boundary & Draft Candidates

| PR / Artifact | Branch / Commit | Qualification Boundaries & Evidence Restrictions | Governing Disposition |
| :--- | :--- | :--- | :--- |
| **PR #317** | `codex/sec-expansion-preparation`<br>Qualified source `df87b1748437ff7cc4509718659604df6f1642f3`<br>Execution merge `79d67c9713912cef6f614b931b4c8c545d90a8cf` | [Hosted run 45284105513](https://github.com/daakara/finance/actions/runs/45284105513) is **completed / success**. [Job 138075811183](https://github.com/daakara/finance/actions/runs/45284105513/job/138075811183) checkout and RQ-01/RQ-08 logs bind the source and execution SHAs above; execution merged this source into base `2eccc64a`. RQ-12 reports **12 aggregated controls, 12 passed, 0 failed/blocked/missing, RELEASE_QUALIFIED**. Successful steps include RQ-01 through RQ-13, SEC evidence (72 passed), SEC canary (150 passed), backend regression and frontend regression/typecheck. The earlier run `45011651092` is historical. Qualification is **not inherited** by subsequent documentation/register heads (`98ec10f8`, `fa9e7554`, or this correction). | [PR #317](https://github.com/daakara/finance/pull/317) remains **OPEN / DRAFT / UNMERGED**. Historical coverage remains **PARTIAL_UNCERTIFIED**. This offline qualification does not activate the candidate; the separately authorized AAPL/NVDA production trial is unchanged. |
| **Historical Coverage Report & Reconciliation JSON** | Commit `98ec10f8`<br>`docs/reports/SEC_HISTORICAL_REPLAY_QUALIFICATION_2026_11_22.md`<br>`data/research/nvda_historical_coverage_reconciliation_165.json` | Corrected terminology (transaction code S labeled "Sale"; suppression of unverified "open-market sale" labels); reconciled database inventory (13 issuers, 55 filings, 152 transactions); removed legal speculation; bounded "zero egress" to audit run. Durably pushed to `origin/codex/sec-expansion-preparation`. | **COMPLETED & TRACKED:**<br>Remote artifacts tracked and verified. Status preserved as **`PARTIAL_UNCERTIFIED`**. |

---

### 4.4 Documentation Corrections & Codified Invariants (Commit `f82b394a`)

Commit `f82b394a` resolved critical documentation desynchronizations across the repository:

1. **Frontend Test Suite Disposition:**
   - **`RQ-04` Release Qualification:** 6 targeted regression checks (`scripts/release/rq04_frontend_regression.py`), which **100% PASS (6/6)**.
   - **Full Unit & Component Suite:** 63 test files executed via Vitest. **61/63 test files passed (682 passed, 2 failed)**.
   - **Root Cause of 2 Failures:** The two failures (`DecisionSurfaceIntegrity.test.tsx` and `Wave1DeclutterPreservation.test.tsx`) are pre-existing since baseline `2eccc64a` due to layout decoupling changes. They are strictly tracked as open technical debt and distinguished from passing `RQ-04` qualification.
2. **Indicator Readiness Thresholds:**
   - **15 Sessions:** Baseline modeling gate (strictly enforced runtime threshold for volume, VCP, and ATR; assets with $<15$ sessions return 422 `insufficient-history-state`).
   - **20 Sessions:** Smoothing heuristic (SMA20, RSI14; advisory only).
   - **30 Valid Return Observations:** [Scoring semantics, Section 7](../scoring-semantics.md#7-tail-risk-score-tailriskscore) explicitly requires **30 closed daily returns**, not 30 price sessions. Thirty valid one-session returns normally require at least 31 valid consecutive closing prices; invalid/missing observations must not count toward the minimum. [ARX_INV_013](../governance/ARX_DOMAIN_INVARIANT_REGISTRY.md#arx_inv_013-statistical_confidence_requires_empirical_proof) requires sample-size evidence for statistical-confidence claims but does not itself define the return-construction policy. The former fallback of 65 is eliminated: runtime implementation enforces explicit **UNAVAILABLE** state without a fabricated numerical score, verified in `feat/cornish-fisher-sufficiency-gate` (TASK-OPEN-03).
   - **50 Sessions:** Intermediate trend standard (SMA50; advisory).
   - **200 Sessions:** Institutional regime standard (SMA200; advisory).
3. **Cardinal Separation Principle Codified:**
   $$\boxed{\text{Listing Identity} \ne \text{SEC Filing Completeness} \ne \text{Scoring Eligibility} \ne \text{Execution Clearance}}$$
   - *Listing Identity:* Asset exists in authoritative Security Master / provider catalog.
   - *SEC Filing Completeness:* Required statutory filings (Form 4, 10-K, 10-Q) are ingested, verified by SHA-256 custody, and certified.
   - *Scoring Eligibility:* Asset meets historical depth and data-density requirements to generate quantitative pillar scores.
   - *Execution Clearance:* Pre-Flight checklist passes all risk, sizing, spread, and quote provenance rules under live authorization.

---

## 5. Closed Milestones & Verified Defect Resolutions

### 5.1 Closed Governance Milestones

| Milestone / Issue | Scope & Verification | Final Disposition |
| :--- | :--- | :--- |
| **PR #310 / Issue #228**<br>(Backlog Reconciliation & Historical Status Synchronization) | Finite inventory/reconciliation milestone recorded complete using `AUD-REG-COV-2026-11-21` and `PLN-REG-RECON-2026-11-21`. This is a register disposition, not a GitHub closure or merge action. | **MILESTONE RECORDED COMPLETE.** [PR #310](https://github.com/daakara/finance/pull/310) is **OPEN / UNMERGED** at `ac5ce3ad4c72c6e2f40067b25136819a190036f0` in authenticated readback. Issue #228 closure state **NOT_CHECKED**. No issue/PR state was changed by this documentation closeout. |

### 5.2 Individually Resolved Defect Register

| Defect ID | Title & Summary | Code Evidence & Root Cause | Resolution PR & Artifact | Final Status |
| :--- | :--- | :--- | :--- | :--- |
| **DEF-TP2-001** | Target 2 Mismatch in Pre-Flight Checklist Modal | `frontend/lib/preflightEvaluator.ts:1036` computed synthetic `safeTarget * 1.12` markup because `PreFlightChecklistInputs` omitted `takeProfit2`. Backend computes canonical ATR-based `take_profit_2` in `optimal_execution.py`. | **PR #302** (Commit `7a0f7676`). Bound canonical second profit target into `PreFlightChecklistInputs` and eliminated synthetic 12% calculation. | **RESOLVED & DEPLOYED** |
| **DEF-CAT-001** | Earnings Call Time Conflation (`16:00 ET Call`) | `analyst_dashboard/corporate_events.py:280–281` assigned Yahoo Finance market-close epoch timestamp (16:00 ET) to `conference_call_time_et`. For NVDA, IR call is at 17:00 ET (results at 16:20 ET). | **PR #303** (Commit `933295cc`). Primary-source schedule for NVDA added to `corporate_events.py` with confirmed `results_release_time_et: "16:20"` and `conference_call_time_et: "17:00"`. Disentangled in UI. | **RESOLVED & DEPLOYED** |
| **DEF-CAT-002** | Empty 5-Year Revenue & Earnings Simulation Panel | `analyst_dashboard/analyzers/catalysts.py:383–388` stripped `multi_year_forecast` to `[]` under live-only invariant. `CatalystForecastCard.tsx:316–350` rendered table headers with blank rows. | **Commit `c7fe3052`** (Candidate `d3c611c4`). Added explicit empty-state disclosure in `CatalystForecastCard.tsx`: *"Forward multi-year projections awaiting verified consensus disclosures under live-only governance."* | **RESOLVED & VERIFIED** |
| **DEF-SEC-004** | Backup Retention Bounded Qualification & Recovery Protection | Unbounded backup retention risked timing out or deleting active recovery points. | **PR #315** (Commit `1cd1a0cf` -> `2eccc64a`). Enforces fail-closed R2 integrity, monotonic execution deadlines, and isolated restore qualification. Defers trial retention during live run. | **RESOLVED & DEPLOYED** |

---

## 6. Active Open Engineering Backlog (Genuinely Unfinished Work)

This section catalogs all genuinely unfinished workstreams. Each item includes precise acceptance criteria and remains open until fully verified.

### 6.1 Genuinely Unfinished Open Workstreams

| Item ID | Title | Priority | Current State & Blockers | Concrete Acceptance Criteria for Closure |
| :--- | :--- | :---: | :--- | :--- |
| **TASK-OPEN-01** | **Issue #165:** Final Trial Reconciliation, Post-Lease Shutdown & Bounded Catalog Acquisition Proposal | **P1** | Worker running trial `ARX-TRIAL-2026-NVDA-AAPL-001` in ingestion-only standby (`bba68cfb` @ `2eccc64a`). Lease expires `2026-11-25T02:35:00Z`. Retains `PARTIAL_UNCERTIFIED`. | 1. Await scheduled authorization lease expiry (`2026-11-25T02:35:00Z`) without disturbance.<br>2. Perform post-trial shutdown audit confirming 0 locks, 0 active queue rows, and final egress budget balance.<br>3. Propose bounded post-trial SEC submissions catalog acquisition under separate human authorization to address historical coverage gap. |
| **TASK-OPEN-02** | **Issue #166:** Market Opportunity Radar Live Scanning Activation | **P1** | Currently serves static/published single-universe snapshots (250 evaluated, 100 Stage 2) under `HOLD_SCAN_001`. Background schedulers disabled. Branch `audit/radar-readiness-assessment-166` (`17fe7375`) under offline evaluation. | 1. Complete offline qualification of 500-universe refresh pipeline.<br>2. Verify zero look-ahead bias and fail-closed checkpointing across Alpaca market data streams.<br>3. Obtain formal Owner decree lifting `HOLD_SCAN_001`. |
| **TASK-OPEN-03** | **Cornish–Fisher Minimum-Observation Runtime Enforcement** | **P2** | Fully qualified on branch `feat/cornish-fisher-sufficiency-gate` (undeployed for review; PR created). Verified by 15/15 tests in `tests/test_cornish_fisher_sufficiency_gate.py` (including exchange-calendar session alignment, holiday transitions, and omitted trading session invalidation), 44/44 backend risk/domain tests, 8/8 frontend behavioral tests in `RiskMetricsAndSizingBehavior.test.tsx`, 17/17 governor sizing tests, clean frontend typecheck, and release qualification checks. Sizing safety hardened: arbitrary 2.5% ATR fallback eliminated; ATR fallback permitted only when validly populated and accurately labeled as ATR Vol; otherwise strictly disables sizing. Former 65 fallback eliminated. | 1. Enforce $N \ge 30$ valid finite closed daily return observations via `extract_valid_returns` (disallowing forward-fill and cross-session bridging).<br>2. For $N < 30$, return explicit `UNAVAILABLE` state with `reason="INSUFFICIENT_OBSERVATIONS"`, `actual_observations`, `required_observations=30`, returning `null` for `Modified_VaR_*` and `tailRiskScore` (never 0 or 65).<br>3. Degenerate and non-finite samples fail closed to `UNAVAILABLE`.<br>4. Verified downstream propagation through API, UI (`RiskMetricsCard`, `DayTraderPositionSizer`, `AssetFactorRadar`), self-healing engine (Kupiec test deferred), and trader archetypes (Simons model fails closed). |
| **TASK-OPEN-04** | **Resolution of Two Pre-Existing Frontend Baseline Test Failures** | **P2** | Full unit test suite reports 61/63 files passed (682 passed, 2 failed). Failures are in `DecisionSurfaceIntegrity.test.tsx` and `Wave1DeclutterPreservation.test.tsx` (present since baseline `2eccc64a`). | 1. Update test fixtures and selector queries in `DecisionSurfaceIntegrity.test.tsx` and `Wave1DeclutterPreservation.test.tsx` to match decoupled layout structure.<br>2. Verify full 63/63 test files pass (684/684 green) in local Vitest run. |

---

### 6.2 Security & Architectural Technical Debt Register

| Debt ID | Linked Container | Title & Scope | Technical Problem & Root Cause | Required Action for Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **TD-07** | **Issue #203** (Security) | Dependency Lockfile Integrity & Hash Pinning | `requirements.txt` specifies minimum version bounds without cryptographically hash-pinned transitive lockfiles, creating supply-chain vulnerability. | Generate reproducible, hash-pinned `requirements.lock` from verified Python virtualenv. Add CI step verifying lockfile synchronization. |
| **TD-04** | **Issue #204** (Architecture) | Warmup Threadpool Graceful Cancellation | In-flight analysis warmup tasks lack graceful cancellation tokens on process shutdown or request timeout, risking threadpool exhaustion. | Add cancellation token propagation and timeout cleanup in `analyst_dashboard/warmup.py`. |
| **TD-05** | **Issue #204** (Architecture) | Screener CDN Cache Identity & Timestamp Truth | Screener CDN headers specify `stale-while-revalidate=86400`, but downstream UI timestamps do not explicitly distinguish 24h cached discovery from real-time observations. | Ensure screener API response headers and UI cards display explicit cache age and observation timestamps. |
| **TD-06** | **Issue #204** (Architecture) | SQLite Scanner Store Recovery Under Parallel Mounts | Scanner store uses atomic table creation, but behavior under concurrent multi-worker startup and read-only mounts requires formal fault testing. | Implement fault-injection test covering concurrent multi-process initialization and read-only volume recovery. |
| **TD-08** | **Issue #204** (Architecture) | Fabricated Options Expiration Fallback | `frontend/app/page.tsx:100` falls back to fabricated `"2026-04-01"` date if options chain expiration date is absent from provider. | Replace hardcoded fallback with explicit `null` / omission badge (`"Expiration: Undated"`). |
| **TD-09** | **Issue #204** (Architecture) | Research JSON Artifacts Repository Benchmarking | Multi-megabyte research artifacts in `docs/research/` increase git clone depth and CI resource usage. | Benchmark clone size and CI runtime impacts across research files before deciding whether to offload to external artifact storage. |

---

### 6.3 Held Governance & Research Prototypes

| PR / Issue | Working Title | Original Research Scope | Holding Reason & Required Clearance Action |
| :--- | :--- | :--- | :--- |
| **PR #274 / Issue #66** | `feat(ux): ARX-UX-R3-001 progressive disclosure & decision-first layout` | Progressive disclosure redesign, collapsible analysis cards, decision-first layout hierarchy. | **HELD UNDER `PHASE_A3_UX_HOLD`:** Requires formal human usability testing ($N=5$ participants, min 2 mobile, 2 desktop; 0% safety defects) before release admission. |
| **PR #146** | `fix(observability): OBS-01 Sentry PII and breadcrumb hygiene` | Sentry PII scrubbing, canonical error taxonomy engine (`error_taxonomy.py`), query redaction precision. | **DRAFT / REBASE REQUIRED:** Has merge conflicts with current main. High-value error taxonomy components scheduled for selective extraction under Issue #204. |
| **PR #83** | `feat(hva01): Wave 4C-1 dual-horizon candle geometry` | High-volatility candle geometry research and multi-timeframe candle overlap modeling. | **HELD RESEARCH:** Preserved as empirical research evidence; autonomous execution blocked under `PHASE_3_HOLD`. |
| **PR #78** | `fix(hva01): Wave 4B-S restore high-conviction screening` | Historical screening heuristics and multi-factor filtering rules. | **SUPERSEDED RESEARCH:** Preserved for reference; operational screening superseded by dynamic Radar architecture. |
| **PR #75** | `research(arx): INSTR-001 empirical research instrument` | Empirical instrument and synthetic observer telemetry validation pipelines. | **HELD RESEARCH:** Preserved for future Phase A3 calibration; zero live activation. |

---

## 7. Operational Holds & Release Gates Summary

All six operational holds remain in active, fail-closed enforcement across the repository:

1. **`HOLD_SEC_001` (SEC Form 4 Continuous Ingestion):** Railway Worker remains in strict `STANDBY` mode (`ARX_SEC_EGRESS_ROLE=DISABLED`). Supervised trial execution (`ARX-TRIAL-2026-NVDA-AAPL-001`) operates under non-renewable lease with zero autonomous polling.
2. **`HOLD_SCAN_001` (Market Opportunity Radar):** Background schedulers disabled; serves static/published single-universe snapshots (250 evaluated, 100 Stage 2). Live market data streaming blocked.
3. **`HOLD_EXEC_001` (Live Execution & Capital Commitment):** Zero live broker API credentials; all orders route to simulated execution with fail-closed sizing (`INV-28`).
4. **`PHASE_3_HOLD` (Autonomous Trading & Sprint 5 Activation):** Locked (`sprint5_execution_authorized = false`).
5. **`ARX_CONFLUENCE_INTEGRITY_HOLD` (Automated Scoring Promotion):** Held pending complete multi-pillar decision trace review and full SEC coverage certification.
6. **`PHASE_A3_UX_HOLD` (Progressive Disclosure Redesign PR #274):** Held pending authentic 5-participant real-human formative evaluation.

---

## 8. Prioritized Next Actions & Execution Queue

| Priority Rank | Workstream / Task | Immediate Next Action | Governing Gate / Trigger |
| :---: | :--- | :--- | :--- |
| **1** | **Issue #165:** Trial Monitoring & Post-Expiry Shutdown | Monitor running trial passively until lease expiry (`2026-11-25T02:35:00Z`). Execute post-trial shutdown audit confirming 0 locks and 0 queue rows. | Scheduled lease expiry |
| **2** | **Frontend Unit Test Suite Polish:** Resolve 2 Regressions | Update test selectors in `DecisionSurfaceIntegrity.test.tsx` and `Wave1DeclutterPreservation.test.tsx` to match decoupled layout. Restore 63/63 green. | Pre-release hygiene |
| **3** | **Security Hardening (Issue #203 / TD-07):** Hash-Pinned Lockfile | Generate cryptographically pinned `requirements.lock` from clean Python virtualenv on an isolated branch. | Sprint P4 Security |
| **4** | **Architectural Cleanup (Issue #204 / TD-08):** Options Fallback | Remove fabricated `"2026-04-01"` fallback in `frontend/app/page.tsx:100` and replace with explicit omission indicator. | Sprint P4 Architecture |
| **5** | **Post-Trial Bounded Catalog Acquisition Proposal** | Prepare bounded, rate-limited submissions catalog acquisition proposal for NVDA post-trial shutdown. | Human Owner Authorization |
