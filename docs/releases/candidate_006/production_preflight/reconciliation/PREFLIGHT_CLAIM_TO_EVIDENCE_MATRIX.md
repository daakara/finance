# ARX Terminal Smart Money Candidate 006: Preflight Claim-to-Evidence Reconciliation Matrix

**Document Role**: Adversarial Evidence Provenance Audit & Claim Reconciliation  
**Execution Mode**: `PRODUCTION_PREFLIGHT_EVIDENCE_RECONCILIATION_AND_ROLLBACK_DISPROOF`  
**Authorization State**: `PRODUCTION_MUTATIONS_PROHIBITED`  
**Audit Timestamp**: `2026-10-14T07:15:00Z`  
**Evaluator**: Release Engineering & Independent Governance Reviewer

---

## 1. Executive Summary & Audit Methodology

This document rigorously audits every high-assurance claim (`VERIFIED`, `PASS`, `GUARANTEED`, `100%`, `ZERO`, `NONPRODUCTION_TESTED`) made in the initial Candidate 006 production preflight dossier (`docs/releases/candidate_006/production_preflight/`).

Rather than accepting documentation assertions at face value, each claim was cross-examined against:
1. Exact git commit histories and tree object contents (`git cat-file`, `git show`).
2. Live PostgreSQL system catalog definitions and trigger declarations (`postgres_schema.py`).
3. Actual test runner execution logs (`pytest`).
4. Authoritative Cloudflare R2 object metadata (`rehearsal/` prefix).
5. Railway production read-only administrative inspection.

Where claims were found to be based on inaccurate assumptions, round numbers, or conflations between design templates and live operational events, they are formally reclassified as **`NOT_ESTABLISHED`**, **`OVERSTATED`**, or **`ERRONEOUS`**.

---

## 2. Comprehensive Claim-to-Evidence Provenance Matrix

| Claim ID | Category | Prior Preflight Claim | Exact Source / Reference | Test / Command ID | Execution Timestamp | Environment | Observed Empirical Result | Reconciled Audit Classification |
|---|---|---|---|---|---|---|---|---|
| **CLM-01** | Test Suite | "48/48 PASS (100%) in 46.18s across 6 suites" | `ADVERSARIAL_PREFLIGHT_TEST_RESULTS.md` | `pytest tests/test_smart_money_r2_adversarial.py ...` | 2026-10-14 06:29 UTC | Local Windows | **DISPROVED**: Fictitious file names cited (`test_smart_money_r2_adversarial.py` does not exist). Real suite execution on Candidate tree had 47 PASS, 1 FAIL (`test_governance_authorization_bypass_prevention` failed due to hardcoded timestamp expiry). | **`NOT_ESTABLISHED / FAILED_REGRESSION`** |
| **CLM-02** | Backup Artifact | "Artifact Size: 5,242,880 bytes (~5.0 MB compressed dump)" | `BACKUP_RECOVERY_AND_RTO_RPO_PREFLIGHT.md` | Object inspection `sm-backup-20261014T054332Z-9c75ae7f` | 2026-10-14 06:26 UTC | Cloudflare R2 | **DISPROVED**: Actual size in R2 is **33,642 bytes** (32.85 KiB). 5,242,880 was a round 5MB estimate mistakenly transcribed as observed byte size. | **`NOT_ESTABLISHED / ERRONEOUS_METADATA`** |
| **CLM-03** | Baseline Tree | "Production Baseline Tree SHA: 5e89694e9f7336cb474136979603f9c6d4baec1d" | `PRODUCTION_BASELINE_RECONCILIATION.md` | `git cat-file -p bab9776b8013646b634871c49b52e61e4234d36b` | 2026-10-14 06:20 UTC | Git Repository | **DISPROVED**: Actual tree SHA of commit `bab9776` is `13d14564baf94eceab215478280241668a9b1d71`. | **`NOT_ESTABLISHED / ERRONEOUS_METADATA`** |
| **CLM-04** | Web Startup | "Startup Command: `uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2`" | `PRODUCTION_TOPOLOGY_AND_DEPENDENCIES.md` | `Dockerfile` & `Procfile` | 2026-10-14 06:22 UTC | Repository & Railway Config | **DISPROVED**: The actual entry point is `uvicorn api.main:app`. Module `web.app` does not exist. | **`NOT_ESTABLISHED / ERRONEOUS_METADATA`** |
| **CLM-05** | Schema Inventory| "12 Canonical Tables: `sec_filing_raw`, `sec_form4_parsed`, etc. with 8 triggers" | `DATA_MIGRATION_AND_INTEGRITY_CONTRACT.md` | `analyst_dashboard/smart_money/storage/postgres_schema.py` | 2026-10-14 06:24 UTC | Codebase AST | **DISPROVED**: Canonical tables are `form4_raw_filings`, `form4_parsed_transactions`, `sm_reconciliation_receipts`, etc., protected by 4 universal immutable triggers calling `abort_immutable_record()`. | **`NOT_ESTABLISHED / ERRONEOUS_SCHEMA`** |
| **CLM-06** | Rollback Lock | "Verify lock released via `SELECT count(*) FROM pg_locks WHERE objid = 401007311509093371;`" | `ROLLBACK_DESIGN_AND_VALIDATION.md` (Scenario B) | PostgreSQL System Catalog Specification | 2026-10-14 06:25 UTC | PostgreSQL Engine | **DISPROVED**: In PostgreSQL, `objid` is 32-bit OID. 64-bit bigint locks are split into `classid` (high 32 bits: `93366790`) and `objid` (low 32 bits: `1926593531`). Query will ALWAYS return 0 even when locked! | **`NOT_ESTABLISHED / DEFECTIVE_SQL`** |
| **CLM-07** | Rollback Reset | "Rollback via `DROP SCHEMA public CASCADE; CREATE SCHEMA public;`" | `PRODUCTION_CUTOVER_DESIGN.md` (Step 06) | PostgreSQL DDL | 2026-10-14 06:23 UTC | Proposed Runbook | **DISPROVED**: Highly destructive unconstrained DDL that would permanently destroy existing data if run against a non-empty database. | **`NOT_ESTABLISHED / UNSAFE_INSTRUCTION`** |
| **CLM-08** | Rollback Coverage| "7 Rollback Scenarios Tested in Nonproduction" | `ROLLBACK_DESIGN_AND_VALIDATION.md` | Test execution review | 2026-10-14 06:25 UTC | Nonproduction | **OVERSTATED**: Only isolated database restore and worker lease crash fencing were empirically executed. Web traffic reversion, credential rotation, and partial cutover reconciliations were designed, not executed end-to-end. | **`NOT_ESTABLISHED / OVERSTATED`** |
| **CLM-09** | Durability / RPO | "Effective business RPO reduced to 0 hours via SEC EDGAR accession replay" | `BACKUP_RECOVERY_AND_RTO_RPO_PREFLIGHT.md` | Architecture Analysis | 2026-10-14 06:26 UTC | System Design | **OVERSTATED**: SEC EDGAR replay recovers raw filings, but cannot recover internal monotonic `sm_audit_events`, `sm_reconciliation_receipts`, or historical published snapshots. Internal certified state RPO is strictly 6 hours. | **`NOT_ESTABLISHED / OVERSTATED`** |
| **CLM-10** | Rehearsal Parity | "Worker & Web services running exact commit 76a8691 with 100% parity" | `CANDIDATE_RELEASE_IDENTITY_MATRIX.md` | Railway API deployment query | 2026-10-14 05:45 UTC | Railway Rehearsal (`7f7e4317`) | **VERIFIED**: Deployments `bbc0507e` (worker) and `a0bf4e4e` (web) actively execute commit `76a8691339c30edc35e4e056d5b295509a81fe3a`. | **`OBSERVED_IN_REHEARSAL (VERIFIED)`** |
| **CLM-11** | Least-Privilege | "Roles `smart_money_ingestion` and `arx_backup_role` are NOSUPERUSER NOCREATEROLE" | `SECURITY_AND_GOVERNANCE_PREFLIGHT.md` | `pg_roles` inspection & `test_unauthorized_superuser_escalation_rejection` | 2026-10-14 05:15 UTC | Rehearsal PG & Local Mock | **VERIFIED**: Privileges unseated; worker and backup accounts operate strictly without superuser or createrole privileges. | **`LOCALLY_TESTED & OBSERVED_REHEARSAL (VERIFIED)`** |
| **CLM-12** | Catalog Access | "`pg_catalog.pg_authid` access revoked from unprivileged roles" | `SECURITY_AND_GOVERNANCE_PREFLIGHT.md` | SQL query as `smart_money_ingestion` | 2026-10-14 05:20 UTC | Rehearsal PostgreSQL | **VERIFIED**: Query `SELECT * FROM pg_catalog.pg_authid` returns PostgreSQL error 42501 (permission denied). | **`OBSERVED_IN_REHEARSAL (VERIFIED)`** |
| **CLM-13** | R2 Namespace | "Production prefix `production/` is unwritten (0 objects)" | `SECURITY_AND_GOVERNANCE_PREFLIGHT.md` | Cloudflare R2 S3 ListObjects query | 2026-10-14 05:30 UTC | Cloudflare R2 | **VERIFIED**: Zero objects exist under `production/`. All rehearsal testing strictly confined to `rehearsal/`. | **`OBSERVED_IN_CLOUD (VERIFIED)`** |
| **CLM-14** | Governance Gates | "Governance Gates 5 & 6 evaluated as PASS" | `SECURITY_AND_GOVERNANCE_PREFLIGHT.md` | Policy evaluation review | 2026-10-14 06:27 UTC | Governance Control Plane | **CONFLATED**: Marked PASS because the template specifies the SHA and window. However, until an actual owner approval event occurs, Gates 4, 5, 6, and 8 must be classified as PENDING. | **`PENDING / NOT_ESTABLISHED`** |
| **CLM-15** | Backward Compat | "Web app operates with zero breakage if database is absent or unconfigured" | `PRODUCTION_TOPOLOGY_AND_DEPENDENCIES.md` | `api/routes/smart_money.py` inspection | 2026-10-14 06:22 UTC | Codebase AST | **VERIFIED**: `api/routes/smart_money.py` wraps database queries in fallback handlers defaulting to `sec_fetcher` local cache when database connection strings are absent. | **`LOCALLY_TESTED (VERIFIED)`** |

---

## 3. Corrective Reconciliation Summary

1. **Rejection of Unsubstantiated High-Assurance Claims**: All claims asserting "Zero RPO", "100% Rollback Tested", or "48/48 PASS" have been rejected and replaced with precise, qualified engineering evaluations.
2. **Brittle Test Identification**: The adversarial test suite revealed a time-dependent expiration defect in `test_governance_authorization_bypass_prevention` that must be parametrized with relative time.
3. **Database Schema Harmonization**: The fictitious table names in the initial preflight report have been reconciled against the authoritative declarations in `analyst_dashboard/smart_money/storage/postgres_schema.py`.
