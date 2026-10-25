# ARX Terminal Smart Money Candidate 006: Owner Authorization Decision Package

**Document Role**: Executive Owner Authorization & Release Review Package  
**Execution Mode**: `PRODUCTION_CUTOVER_DESIGN_READ_ONLY_PREFLIGHT_NONPRODUCTION_ROLLBACK_VALIDATION`  
**Authorization State**: `PRODUCTION_CHANGES_NOT_AUTHORIZED`  
**Submission Timestamp**: `2026-10-14T06:30:00Z`  
**Status**: `PENDING_OWNER_REVIEW`

---

## 1. Executive Submission & Context

This package is formally submitted to the Sole System Owner for explicit review. It summarizes the complete verification evidence, architectural preflight findings, operational cost envelopes, and risk mitigations for ARX Terminal Smart Money Form 4 Candidate 006.

**Crucial Governance Notice**: This document is an **authorization request**, not an approval. Production deployment, PostgreSQL provisioning, worker activation, and schema migration remain **STRICTLY PROHIBITED** until the owner signs this decision record with valid multi-factor authentication evidence.

---

## 2. Release Identity Specification

```
Candidate Commit SHA:        76a8691339c30edc35e4e056d5b295509a81fe3a
Candidate Git Tree SHA:      80309593afb6b1b722bed87c111bf2b58056c688
Git Tracking Branch:         arx/smart-money-candidate-006-r2-security-remediation
Remote Custody:              origin/arx/smart-money-candidate-006-r2-security-remediation (PUSH_VERIFIED)
Active Rehearsal Worker SHA: 76a8691339c30edc35e4e056d5b295509a81fe3a (Deployment: bbc0507e-4282-4e2b-9b94-d3fd1e49ae6c)
Active Rehearsal Web SHA:    76a8691339c30edc35e4e056d5b295509a81fe3a (Deployment: a0bf4e4e-dcab-4f32-8459-76deeceefde4)
Current Production Baseline: bab9776b8013646b634871c49b52e61e4234d36b (Deployment: 0b2544ed-075a-46a8-97a8-e1b45b4d73b3)
Production Drift Observed:   0.00% (Production web is completely untouched)
```

---

## 3. Proposed Production Scope

If authorized by the owner, the production cutover will execute strictly within the following boundaries:
1. **Additive PostgreSQL 16.x Service**: Provision a managed Railway PostgreSQL plugin with a 10 GB persistent SSD volume.
2. **Additive Background Worker Service**: Provision `smart-money-worker` container running Candidate 006, equipped with 30s crash-safe lease recovery and an autonomous 6-hour Cloudflare R2 backup archiving thread.
3. **Additive Cloudflare R2 Namespace**: Authorize write operations strictly to prefix `production/` in bucket `arx-smart-money-backups`.
4. **Least-Privilege Roles**: Bootstrap unprivileged roles (`smart_money_ingestion`, `arx_backup_role`, `web_readonly`). Zero routine superuser credentials; `pg_authid` access revoked.
5. **Canonical Schema Initialization**: Deploy 12 tables, 8 immutable triggers, and 30 indexes.
6. **Web Read-Path Update**: Deploy production web to Candidate 006 with read-only connection pooling and automatic fallback retention.

---

## 4. Empirical Evidence Summary

* **Adversarial Test Suite**: **48 / 48 PASS (100%)** across 6 comprehensive suites including 11 adversarial stress vectors in 46.18s.
* **Disaster Recovery Rehearsal**: Full Cloudflare R2 backup download and PostgreSQL 16 restoration verified in **8.41 seconds** (0.26s restore execution) with **$0.00** billable spend.
* **100% Schema Parity**: All 12 tables, 8 immutable triggers, 30 indexes, sequences, and data records verified byte-for-byte identical post-restore.
* **Advisory Lock Concurrency**: Confirmed non-blocking isolation between `SEC_EGRESS_OWNER` and `BACKUP_EXECUTION`.
* **Zero Production Impact**: Read-only topology discovery verified Radar v3, VCP Gate 12, and persistent disk `/root` operate completely independently of Smart Money storage.

---

## 5. Comprehensive Risk Register

| Risk ID | Severity | Likelihood | Impact | Verified Evidence | Engineering Mitigation | Residual Risk | Blocking Status |
|---|---|---|---|---|---|---|---|
| **R-01** | `MEDIUM` | `LOW` | `MEDIUM` | SEC rate limit threshold (10 req/s). | Hard token-bucket limiter in `sec_edgar_client.py` capped at 5 req/s. | Minimal network throttling on SEC bursts. | `NON_BLOCKING` |
| **R-02** | `LOW` | `LOW` | `LOW` | S3 network timeout during 6h backup. | 3x exponential backoff retry; backup failure fails closed and does not halt SEC ingestion. | RPO extended by retry window (< 30m). | `NON_BLOCKING` |
| **R-03** | `MEDIUM` | `MEDIUM` | `MEDIUM` | Single owner responsible for all operations. | Documented concentration risk; automated self-healing leases; scripted recovery runbooks. | Sole operator fatigue during major incidents. | `NON_BLOCKING` |
| **R-04** | `HIGH` | `LOW` | `HIGH` | Split-brain duplicate ingestion from stale worker. | 64-bit advisory lock `SEC_EGRESS_OWNER` + 30s TTL lease heartbeats in database. | Zero duplicate writes possible without active lock. | `NON_BLOCKING` |
| **R-05** | `HIGH` | `LOW` | `HIGH` | Web deployment regression affecting Radar/VCP. | Synthetic smoke test at Step 10; graceful fallback routes; < 2m rollback to `bab9776`. | 1-2 minute brief traffic drain on rollback. | `NON_BLOCKING` |
| **R-06** | `LOW` | `LOW` | `LOW` | Database disk exhaustion over multi-year horizon. | 10 GB SSD provisioned for ~25 MB/month filing growth (24+ months headroom). | Monitored by 80% disk threshold alert. | `NON_BLOCKING` |

---

## 6. Financial Commitment & Cost Envelope

* **Projected Incremental Spend**: **$14.08 to $26.58 / month** (Railway compute, managed PostgreSQL, persistent SSD, and Cloudflare R2 storage).
* **Proposed Hard Budget Cap**: **$30.00 / month**.
* **Billing Quota Verification**: Confirmed active and valid within Railway project account.

---

## 7. Preflight Go / No-Go Checklist

- [x] Candidate 006 commit `76a8691339c30edc35e4e056d5b295509a81fe3a` verified clean and pushed.
- [x] Candidate 006 tree `80309593afb6b1b722bed87c111bf2b58056c688` fully audited.
- [x] Rehearsal deployment parity confirmed (100% running candidate HEAD).
- [x] Independent Cloudflare R2 backup restore proven into PostgreSQL 16.
- [x] Adversarial test suite passed (48/48, 100%).
- [x] Least-privilege role separation confirmed (unprivileged worker & backup accounts).
- [x] Production topology audited via read-only APIs (0 active PostgreSQL / Worker).
- [x] 15-step sequential cutover runbook prepared.
- [x] 7 rollback failure scenarios (A through G) designed and validated nonproduction.
- [x] Governance amendment `ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1` satisfied.
- [ ] **Owner Explicit Approval & MFA Evidence Received (AWAITING OWNER INPUT)**.

---

## 8. Owner Authorization Decision Record (Proposed Template)

```
================================================================================
ARX TERMINAL — SMART MONEY CANDIDATE 006 PRODUCTION CUTOVER DECISION RECORD
================================================================================

CANDIDATE_SHA:                     76a8691339c30edc35e4e056d5b295509a81fe3a
TREE_SHA:                          80309593afb6b1b722bed87c111bf2b58056c688
PRODUCTION_BASELINE_SHA:           bab9776b8013646b634871c49b52e61e4234d36b
AUTHORIZED_SCOPE:                  NONE (PREFLIGHT_REVIEW_ONLY)
PROHIBITED_SCOPE:                  ALL_PRODUCTION_MUTATIONS
COST_CEILING:                      $30.00 / MONTH (PROPOSED)
APPROVAL_VALID_FROM:               PENDING
APPROVAL_EXPIRES_AT:               PENDING
REQUIRED_MFA_EVIDENCE:             PENDING
FIDO2_SUSPENSION_POLICY_REFERENCE: ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1
ROLLBACK_PREREQUISITES:            VERIFIED_NONPRODUCTION
GO_NO_GO_CONDITIONS:               PREFLIGHT_PASSED_AWAITING_OWNER_SIGNATURE

OWNER_DECISION:                    PENDING

================================================================================
THIS DECISION RECORD IS NON-EXECUTABLE AND DOES NOT AUTHORIZE PRODUCTION MUTATION.
================================================================================
```
