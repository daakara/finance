# ARX Terminal — Operational Holds Register & Governance Enforcement Matrix

**Document ID:** `GOV-REG-HOLDS-2026-11-14`  
**Effective Date:** 2026-11-14  
**Authoritative Baseline:** `release/production` @ SHA `6ea232838a53e482753d5705ba533ad9ce041e44`  
**Classification:** Canonical Governance Register  
**Supersedes:** Historical hold tables in dated sprint registers

---

## 1. Executive Summary & Purpose

The ARX Terminal operates under an uncompromising fail-closed governance model. To protect system stability, capital preservation, regulatory compliance, and analytical integrity, specific platform capabilities are placed under **Operational Holds**.

An Operational Hold is a hard, audited constraint enforced at the architectural, environmental, and continuous integration levels. **No documentation change, PR merge, conditional approval, or conversational instruction can lift an operational hold.** Lifting any hold requires fulfilling its explicit, non-negotiable exit criteria and receiving formal, cryptographic or signed Owner authorization.

As of November 14, 2026, exactly **six (6) operational holds** are active across the ARX Terminal platform.

---

## 2. Canonical Operational Holds Matrix

| Hold ID | Hold Name & Scope | Invariant / Policy Basis | Live Enforcement Mechanism | Entry Trigger | Mandatory Exit Criteria | Current State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`HOLD_SEC_001`** | **SEC Form 4 Continuous Ingestion Hold**<br>Prohibits continuous background polling, automatic egress, or unauthorized EDGAR requests. | `INV-05`, `INV-18`<br>`SM_DATA_GOV_001` | Environment variable `ARX_SEC_EGRESS_ROLE=DISABLED`. Railway worker runs in `STANDBY` mode. Zero active DB advisory locks. Egress blocked. | Protection against SEC fair-access rate limit violations (10 req/s), data pollution, and unmonitored egress. | 1. Bounded, idempotent operator authorization.<br>2. Full calendar & dissemination date gating.<br>3. Independent verification of worker shutdown after cycle.<br>4. Explicit OWNER GO. | **ACTIVE**<br>*(One-shot supervised qualification cycle scheduled for Mon Nov 16 at 22:30 EST; recurring ingestion strictly held)* |
| **`HOLD_SCAN_001`** | **Market Radar Continuous Scanning Hold**<br>Prohibits unthrottled continuous market-wide scanning and background natural scheduling. | `INV-09`<br>`RADAR_GOV_001` | Natural cron/background schedulers disabled. Scanner serves static frozen/published single-universe snapshots (250 evaluated, 100 Stage 2). | Provider rate-limit exhaustion, cache fencing, and data freshness guarantees. | 1. Verified single-universe snapshot caching.<br>2. Multi-tier rate limit budget compliance.<br>3. End-to-end data provenance verification.<br>4. Owner release authorization. | **ACTIVE**<br>*(Radar Recommendation Integrity PR #279 live; natural scheduling held)* |
| **`HOLD_EXEC_001`** | **Financial Execution & Live Trading Hold**<br>Absolute prohibition on live capital commitment, broker API connectivity, or automated order routing. | `INV-01`, `INV-02`<br>`RISK_GOV_001` | Zero broker API keys wired in production runtime. All trade paths route to simulated order models with fail-closed sizing (`INV-28`). | Capital preservation, regulatory compliance, risk-management mandate. | 1. Multi-jurisdiction regulatory clearance.<br>2. Full broker sandbox certification.<br>3. Institutional risk audit & circuit breakers.<br>4. Unanimous Owner authorization. | **ACTIVE**<br>*(Permanent fail-closed invariant)* |
| **`PHASE_3_HOLD`** | **Autonomous Trading & Sprint 5 Activation Hold**<br>Prohibits activation of Phase 3 autonomous trading features, online learning retraining, or Sprint 5 scope. | `INV-14`<br>`SPRINT5_LOCK` | Flag `sprint5_execution_authorized = false`. Static model weights in production. Prospective parameter mutation blocked. | Sprint 4 administrative closure; governance gate requiring clean settled prospective cohort evidence before model calibration. | 1. Audited, clean settled prospective decision cohort.<br>2. Formal Product Owner sign-off on Sprint 5 initiation.<br>3. Independent eval gate verification. | **ACTIVE**<br>*(Sprint 4 closed; Sprint 5 `NOT_AUTHORIZED`)* |
| **`ARX_CONFLUENCE_INTEGRITY_HOLD`** | **Confluence Scoring Operational Hold**<br>Holds automated scoring promotion pending complete multi-pillar provenance parity and decision trace integrity. | `INV-11`, `INV-12`<br>`CONFLUENCE_GOV_001` | Confluence scores served in research mode with explicit provenance badges; PR #272/#277 verified schema parity and decision trace integrity; technical closure recorded. | Remediation of provenance tracking discrepancies and sub-threshold conviction signaling. | 1. Technical closure verification (PR #272, #277).<br>2. Latency SLA verification (PR #275).<br>3. Complete post-closure operational review by Owner. | **ACTIVE**<br>*(Technical remediation deployed; operational hold maintained)* |
| **`PHASE_A3_UX_HOLD`** | **Phase A3 Formative Human UX Evaluation Hold**<br>Prohibits merging or releasing PR #274 (progressive disclosure & decision-first layout redesign) prior to real-user validation. | `UX_GOV_001`<br>`RQ-01`, `RQ-02` | PR #274 held in DRAFT status. Release qualification gate blocks admission. | UX governance mandate requiring empirical human validation before altering investor decision-making workflows. | 1. Completion of 5-participant formative study with real human users.<br>2. Authentic qualitative and quantitative evaluation worksheets.<br>3. Zero fabricated observations.<br>4. Formal UX Owner acceptance. | **ACTIVE**<br>*(Study pack prepared; real-human evaluation pending)* |

---

## 3. Enforcement Infrastructure & Telemetry

### 3.1 Railway Production Container Enforcement
- **Worker Configuration:** The production Smart Money worker (`smart-money-worker` on Railway project `tranquil-radiance`) is permanently pinned to `ARX_SEC_EGRESS_ROLE=DISABLED`.
- **Runtime State:** The worker process operates in `STANDBY` mode, rejecting any unauthenticated or scheduled polling requests.
- **Advisory Locks:** Database connection monitoring verifies exactly 0 active advisory locks in `pg_locks` during standby.
- **Fail-Closed Verification:** Automated tests in `tests/test_sec002_remediation.py` and `tests/test_worker_queue_remediation.py` verify that any attempt to initiate egress without an active, unexpired, cryptographic Owner lease immediately raises an unhandled `SecurityInvariantViolation` and terminates the container.

### 3.2 Cloudflare Pages Presentation Enforcement
- **Deployment Policy:** Production deployments to `www.arxterminal.com` are decoupled from Git merges (`production_deployments_enabled=false`).
- **Bundle Verification:** Every release artifact is checksummed. Current production bundle `page-7644a93d0a184821.js` strictly enforces:
  - Sub-threshold Confluence scores (29–45/100) rendered in defensive rose styling (`text-rose-400 bg-rose-500/10 border-rose-500/20`).
  - Actionability status `READY TO BUY` demoted to `NEAR SCREENING ZONE`.
  - Non-actionable discovery disclaimers prominently rendered.
- **Simulated Execution:** The client-side sizing calculator strictly requires `telemetry_verified` broker equity. LocalStorage overrides are quarantined.

### 3.3 Release Qualification (RQ) Gate Enforcement
The release qualification suite (`run_release_ci_suite.py`) executes 12 mandatory gates (RQ-01..12) on every PR targeting `release/production`:
- **RQ-01:** Git tree cleanliness & branch parity.
- **RQ-02:** Release qualification admission & integrity.
- **RQ-03:** Domain invariant audit (all 28 invariants verified).
- **RQ-04:** Frontend type-check & test coverage.
- **RQ-05:** Backend unit & integration test coverage (100% pass required).
- **RQ-06:** Security vulnerability audit & Bandit scan.
- **RQ-07:** Performance & latency SLA verification (<150ms).
- **RQ-08:** Schema migration & database parity.
- **RQ-09:** Cloudflare Pages build & bundle integrity.
- **RQ-10:** Operational hold preservation check.
- **RQ-11:** Adversarial boundary attack tests (11/11 controls).
- **RQ-12:** Gate H tamper-resistance verification (12/12 controls).

---

## 4. Change Management & Audit Logging

Any modification to this register requires:
1. An explicit, written directive from the System Owner (`daakara`).
2. Inclusion in a dedicated governance Pull Request targeting `release/production`.
3. Execution and clean pass of the complete Release Qualification Suite.
4. An immutable audit record posted to the corresponding GitHub Master Tracker.
