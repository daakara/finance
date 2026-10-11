# ARX Terminal — Smart Money: Final Pre-Arm Readiness Gate V1

**Gate Identifier:** `SMART_MONEY_FINAL_PRE_ARM_READINESS_V1`  
**Governing Epoch:** `EPOCH_001`  
**Entry Branch:** `arx/sigstore-workload-identity-qualification`  
**Current HEAD:** `da93706b54ccade17ece4965ded747511b4fe008`  
**Authority:** Independent Pre-Arm Governance Auditor + Clock Authority Auditor + SEC Operational Preflight Auditor + Candidate 004 Firewall Authority  
**Date:** October 11, 2026  
**Operating Mode:** `FAIL_CLOSED` + `NON_NATURAL_PREFLIGHT_ONLY` + `SOURCE_FIRST` + `CLOCK_REFERENCE_REQUIRED` + `NO_NATURAL_CAPTURE` + `NO_SYNTHETIC_CONTAMINATION` + `NO_PRODUCTION_AUTHORITY`

---

## 1. Executive Summary & Gate Decision

The purpose of this gate is to determine whether Smart Money is eligible to transition from:

$$\text{NATURAL\_OBSERVER\_ARMING} = \mathbf{NOT\_AUTHORIZED}$$

to:

$$\text{NATURAL\_OBSERVER\_ARMING} = \mathbf{AUTHORIZED\_FOR\_FIRST\_NATURAL\_CAPTURE}$$

Following comprehensive verification across 12 operational dimensions (Candidate 004 firewall, natural scheduler identity, natural workload policy, live independent NTP clock authority, SEC calendar eligibility, active filing window, SEC endpoint accessibility, Sigstore infrastructure, TUF trust-root validity, local evidence durability, R2 closure policy, and natural denominator purity), the gate reaches a definitive, fail-closed finding:

$$\mathbf{NATURAL\_OBSERVER\_ARMING = NOT\_AUTHORIZED}$$

### Primary Blocking Predicates
1. **`SEC_BUSINESS_DAY = NO`**: Today is **Sunday, October 11, 2026**. Under Section 7, weekend days trigger unconditional preflight blocking.
2. **`WITHIN_AUTHORIZED_SEC_WINDOW = NO`**: EDGAR filing dissemination is closed on non-business days.
3. **`NATURAL_PRODUCTION_SCHEDULER_IDENTITY = NOT_ESTABLISHED`**: The dedicated production scheduler workflow (e.g. `smart_money_rooting.yml`) has not yet been authored or established on `main`.
4. **`NATURAL_SCHEDULER_DRY_RUN = NOT_EXECUTED`**: Blocked by the absence of the natural production scheduler workflow and non-business day calendar.

### Key Operational Milestones Certified
- **Host Clock Synchronized**: Fresh live NTP measurement across 4 stratum-1/2/3 servers certified a total uncertainty bound of **$62.48\text{ ms}$** ($\le 100.0\text{ ms}$, `SYNCHRONIZED`), fully resolving prior clock drift.
- **SEC Endpoint Access Operational**: Live HTTP 200 responses verified on official SEC metadata (`CIK0000320193.json`) and content endpoints under governed Fair Access User-Agent and rate limits.
- **Sigstore & TUF Ready**: Public Rekor and Fulcio APIs live and verified; TUF trust-root active through November 20, 2026.
- **Candidate 004 Firewall Intact**: Strict 0 diff vs `63131b9` across all protected files.

---

## 2. Repository & Candidate 004 Firewall Audit

```
=== BRANCH ===
arx/sigstore-workload-identity-qualification
=== HEAD ===
da93706b54ccade17ece4965ded747511b4fe008
=== STATUS ===
clean (0 unstaged, 0 untracked)
=== UPSTREAM TRACKING ===
origin/arx/sigstore-workload-identity-qualification (in sync)
=== CANDIDATE 004 FIREWALL DIFF (vs 63131b9) ===
shadow_evidence.db: 0 diff
analyst_dashboard/vcp/**: 0 diff
natural_trigger.py: 0 diff
```

- **`WORKTREE_CLEAN`**: `YES`
- **`PROTECTED_FILES_CHANGED`**: `0`
- **`CANDIDATE_004_INTERFERENCE`**: `NONE`

---

## 3. Natural Scheduler Identity Discovery

The qualification workflow (`.github/workflows/smart_money_sigstore_oidc_qualification.yml`) was authored and executed strictly under `qualification_scope: TEST_ONLY` on a dedicated branch. It cannot and must not be conflated with the production natural scheduler.

Audit findings:
- **`NATURAL_SCHEDULER_RUNTIME`**: `GITHUB_ACTIONS`
- **`NATURAL_SCHEDULER_WORKFLOW`**: `NOT_ESTABLISHED` (workflow file not yet authored on `main`)
- **`NATURAL_SCHEDULER_TRIGGER`**: `NOT_ESTABLISHED` (pending schedule/cron definition)
- **`NATURAL_SCHEDULER_REF`**: `NOT_ESTABLISHED` (requires `refs/heads/main`)
- **`NATURAL_SCHEDULER_REPOSITORY`**: `daakara/finance`
- **`NATURAL_SCHEDULER_REPOSITORY_ID`**: `1087399232`
- **`NATURAL_SCHEDULER_OWNER_ID`**: `53128824`
- **`NATURAL_PRODUCTION_SCHEDULER_IDENTITY`**: `NOT_ESTABLISHED`

Under Section 3 governance rules: *If no natural scheduler exists, `NATURAL_PRODUCTION_SCHEDULER_IDENTITY = NOT_ESTABLISHED`. STOP.*

---

## 4. Natural Workload OIDC Policy Governance

The ratified qualification policy `SMART_MONEY_SIGSTORE_WORKLOAD_IDENTITY_POLICY_V1` authorized only `Smart Money Sigstore OIDC Qualification` on `refs/heads/arx/sigstore-workload-identity-qualification`. 

A distinct production policy `SMART_MONEY_NATURAL_ROOTING_WORKLOAD_IDENTITY_POLICY_V1` must be established once the natural scheduler is authored on `main`. It must enforce exact pinning without wildcards:
- **`authorized_issuer`**: `https://token.actions.githubusercontent.com`
- **`authorized_repository_id`**: `1087399232`
- **`authorized_repository_owner_id`**: `53128824`
- **`authorized_ref`**: `refs/heads/main`
- **`authorized_event`**: `schedule` / `workflow_dispatch`
- **`status`**: `NOT_RATIFIED` (blocked by scheduler workflow establishment)

---

## 5. Natural Scheduler Dry Qualification

- **`qualification_scope`**: `PRE_ARM_DRY_RUN`
- **`natural_accession`**: `false`
- **`execution_status`**: `NOT_EXECUTED` (blocked by absence of dedicated production scheduler workflow)
- **`NATURAL_SCHEDULER_DRY_RUN`**: `NOT_EXECUTED`

---

## 6. Fresh Independent Clock Qualification

Live UDP SNTP measurements were captured against 4 independent reference servers immediately prior to the arming decision:

```
Captured at: 2026-10-11T18:43:45.487657+00:00
Reference servers count: 4
- time.google.com (stratum 1):     offset = -19.544 ms, rtt = 46.936 ms, uncertainty = 23.468 ms
- time.cloudflare.com (stratum 3): offset = -24.201 ms, rtt = 30.939 ms, uncertainty = 15.469 ms
- time.windows.com (stratum 4):    offset = -24.391 ms, rtt = 32.849 ms, uncertainty = 16.424 ms
- pool.ntp.org (stratum 3):        offset = -23.816 ms, rtt = 37.235 ms, uncertainty = 18.618 ms
```

### Derivation
$$\text{Observed Offset (Median)} = -24.008\text{ ms}$$
$$\text{Measurement Uncertainty } (\max(RTT/2)) = 23.468\text{ ms}$$
$$\text{Network Asymmetry Allowance} = 10.000\text{ ms}$$
$$\text{Other Error Allowance} = 5.000\text{ ms}$$
$$B_{total} = |-24.008| + 23.468 + 10.000 + 5.000 = \mathbf{62.477\text{ ms}}$$

### Threshold Classification
- $B_{total} \le 100.0\text{ ms} \implies \mathbf{SYNCHRONIZED}$
- **`CLOCK_MEASURED_AT`**: `2026-10-11T18:43:45.487657+00:00`
- **`CLOCK_MEASUREMENT_AGE_SECONDS`**: $< 60.0\text{ s}$ (Fresh)
- **`CLOCK_AUTHORITY_STATUS`**: $\mathbf{PASS}$
- **`Raw Evidence SHA-256`**: `524f8006fd9b681cb769b7e6c48282f03227d7f70b51b471aed7666901df07c0`

---

## 7. SEC Calendar & Filing Window Audit

Live evaluation against the authoritative SEC operational calendar:
- **`Captured at UTC`**: `2026-10-11T18:44:03.973320+00:00`
- **`Current US Eastern Time`**: `14:44:03 EDT`
- **`Day of Week`**: `Sunday`
- **`SEC_BUSINESS_DAY`**: $\mathbf{NO}$
- **`WITHIN_AUTHORIZED_SEC_WINDOW`**: $\mathbf{NO}$
- **`Next Eligible Operating Session`**: `AWAIT_NEXT_SEC_BUSINESS_DAY (Current day is Sunday)`
- **`Evidence Digest`**: `c7ca769911bdb4db0299cd39ff30ed21a5079d2f309ea26bc38bf4a81e8c0f46`

Under Section 7 governance rules: *If holiday, weekend, or known SEC closure, then `SEC_PREFLIGHT = BLOCKED`, `NATURAL_OBSERVER_ARMING = NOT_AUTHORIZED`. STOP.*

---

## 8. Fresh SEC Access & Fair Access Preflight

Non-destructive HTTP reachability probes to official SEC endpoints:
- **Metadata Probe (`https://data.sec.gov/submissions/CIK0000320193.json`)**: HTTP `200`
- **Content Probe (`https://www.sec.gov/files/company_tickers.json`)**: HTTP `200`
- **Live SEC Date Header**: `Sun, 11 Oct 2026 18:44:04 GMT`
- **User-Agent Compliance**: `ArxTerminal/1.0 (akara@finance.app)` ($\ge 10$ chars, valid email) $\implies \mathbf{PASS}$
- **Rate Policy Compliance**: $5.0\text{ RPS}$ ($\le 10.0\text{ RPS}$ ceiling) $\implies \mathbf{PASS}$
- **`SEC_ENDPOINT_ACCESS`**: $\mathbf{PASS}$

---

## 9. Rooting Infrastructure Freshness

- **Public Rekor API (`https://rekor.sigstore.dev/api/v1/log`)**: Reachable, HTTP `200` $\implies \mathbf{PASS}$
- **Public Fulcio API (`https://fulcio.sigstore.dev`)**: Reachable $\implies \mathbf{PASS}$
- **Sigstore TUF Root Expiration**: `2026-11-20T13:58:18Z` (Valid) $\implies \mathbf{PASS}$
- **GitHub Actions Ambient OIDC**: Qualified under run `38164305159` $\implies \mathbf{PASS}$
- **`SIGSTORE_RUNTIME_READINESS`**: $\mathbf{PASS}$
- **`TUF_RUNTIME_READINESS`**: $\mathbf{PASS}$
- **`OIDC_RUNTIME_READINESS`**: $\mathbf{PASS}$

---

## 10. Evidence Durability, R2 Delay & State Machine

- **`RAW_EVIDENCE_DURABILITY`**: $\mathbf{PASS}$ (dual-store structure, deterministic manifest `3315dd...`, 0 leaked secrets)
- **`NATURAL_PRODUCTION_DURABILITY_OBSERVED`**: `NO` (awaiting first natural capture)
- **`R2_CLOSURE_MAX_DELAY_SECONDS`**: `300.0`
- **`R2_CLOSURE_DELAY_POLICY_STATUS`**: `PROVISIONAL`
- **`OBSERVER_STATE_MACHINE`**: Current state is strictly `PRE_OBSERVATION_READY` $\implies \mathbf{PASS}$

---

## 11. Natural Denominator Invariant

- **`NATURAL_ACCESSIONS_CAPTURED`**: `0`
- **`NATURAL_DENOMINATOR`**: `0`
- **`SYNTHETIC_RECORDS_IN_NATURAL_DENOMINATOR`**: `0`
- **`HISTORICAL_RECORDS_IN_NATURAL_DENOMINATOR`**: `0`
- **`NATURAL_DENOMINATOR_CLEAN`**: $\mathbf{YES}$

---

## 12. Final Assumption Audit Matrix

| Audit Dimension | Required Standard | Observed State | Classification |
|---|---|---|---|
| **Live R1/R2 Chain** | Sealed on public Rekor | Sealed in evidence v2 | **SEALED** |
| **Unattended Signing** | Qualified via remote OIDC | GHA Run 38164305159 | **PASS** |
| **Natural Scheduler** | Established on main | Workflow not yet authored | **NOT_ESTABLISHED** |
| **Scheduler Dry Run** | Test-only pass | Not executed | **NOT_EXECUTED** |
| **Clock Authority** | Synchronized ($\le 100\text{ ms}$) | $62.48\text{ ms}$ bound | **PASS** |
| **SEC Business Day** | Monday–Friday | Sunday, Oct 11, 2026 | **NO (FAIL)** |
| **SEC Active Window** | 06:00–22:00 ET business day | Closed (Weekend) | **NO (FAIL)** |
| **SEC Endpoints** | HTTP 200 compliant | HTTP 200 on both | **PASS** |
| **Sigstore Readiness** | Reachable & responsive | Reachable | **PASS** |
| **TUF Readiness** | Unexpired trust root | Valid thru Nov 20, 2026 | **PASS** |
| **Evidence Durability** | Dual-store verified | Fully verified | **PASS** |
| **Natural Denominator** | Strictly 0, clean | Denominator = 0 | **YES** |
| **Candidate 004** | Strict 0 diff vs 63131b9 | 0 diff | **NONE** |

---

## 13. Required Final Parameter Block

```
GATE =
  SMART_MONEY_FINAL_PRE_ARM_READINESS_V1

CURRENT_HEAD =
  da93706b54ccade17ece4965ded747511b4fe008

NATURAL_SCHEDULER_RUNTIME =
  GITHUB_ACTIONS
NATURAL_SCHEDULER_WORKFLOW =
  NOT_ESTABLISHED
NATURAL_SCHEDULER_TRIGGER =
  NOT_ESTABLISHED
NATURAL_SCHEDULER_REF =
  NOT_ESTABLISHED

NATURAL_PRODUCTION_SCHEDULER_IDENTITY =
  NOT_ESTABLISHED

NATURAL_WORKLOAD_POLICY =
  NOT_RATIFIED

NATURAL_SCHEDULER_DRY_RUN =
  NOT_EXECUTED

CLOCK_MEASURED_AT =
  2026-10-11T18:43:45.487657+00:00
CLOCK_MEASUREMENT_AGE_SECONDS =
  <60.0
OBSERVED_OFFSET_MS =
  -24.008
TOTAL_CLOCK_UNCERTAINTY_BOUND_MS =
  62.477

CLOCK_AUTHORITY_STATUS =
  PASS

SEC_BUSINESS_DAY =
  NO

CURRENT_ET_TIME =
  14:44:03 EDT

WITHIN_AUTHORIZED_SEC_WINDOW =
  NO

SEC_ENDPOINT_ACCESS =
  PASS

SEC_RATE_POLICY =
  PASS

SIGSTORE_RUNTIME_READINESS =
  PASS

TUF_RUNTIME_READINESS =
  PASS

OIDC_RUNTIME_READINESS =
  PASS

RAW_EVIDENCE_DURABILITY =
  PASS

R2_CLOSURE_MAX_DELAY_SECONDS =
  300
R2_CLOSURE_DELAY_POLICY_STATUS =
  PROVISIONAL

OBSERVER_STATE_MACHINE =
  PASS

NATURAL_ACCESSIONS_CAPTURED =
  0

NATURAL_DENOMINATOR =
  0

NATURAL_DENOMINATOR_CLEAN =
  YES

PROTECTED_FILES_CHANGED =
  0

CANDIDATE_004_INTERFERENCE =
  NONE

NATURAL_OBSERVER_ARMING =
  NOT_AUTHORIZED

EMPIRICAL_SCANNER_QUALITY =
  INSUFFICIENT_EVIDENCE

MODEL_TUNING =
  FROZEN

LEARNING_CLAIM =
  NOT_AUTHORIZED

PRODUCTION_READINESS =
  NOT_AUTHORIZED

MAIN_MERGE =
  NO

PRODUCTION_DEPLOYMENT =
  NO
```

---

## 14. Stop & Next Actions

```
NATURAL_OBSERVER_ARMING =
  NOT_AUTHORIZED

BLOCKING_PREDICATES:
  1. SEC_BUSINESS_DAY = NO (Today is Sunday, October 11, 2026)
  2. WITHIN_AUTHORIZED_SEC_WINDOW = NO (EDGAR filing acceptance closed)
  3. NATURAL_PRODUCTION_SCHEDULER_IDENTITY = NOT_ESTABLISHED
  4. NATURAL_SCHEDULER_DRY_RUN = NOT_EXECUTED

NEXT_ACTION =
  AWAIT_NEXT_SEC_BUSINESS_DAY_AND_PROVISION_NATURAL_SCHEDULER_WORKFLOW
```

*Under fail-closed governance, zero natural SEC filings may be observed or admitted while the market and EDGAR filing windows are closed, and zero synthetic accessions may contaminate the natural prospective denominator.*
