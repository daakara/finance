# ARX Terminal — Production Shadow Release Note
## Radar Sprint 3 Production Shadow: Candidate Generation 004 Controlled Release

**RELEASE_NOTE_STATUS** = PRODUCTION_RECONCILED  
**CANDIDATE_GENERATION_ID** = CANDIDATE_GENERATION_004  
**CANDIDATE_FUNCTIONAL_SHA** = fafba93ccd60f40792bb3f5ea6702804eeea40cf  
**CANDIDATE_FREEZE_SHA256** = 297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90  
**CANDIDATE_SEMANTIC_CLOSURE_HASH** = 53b6372349cfdb1046a4ddbd678f628bba28c7778cef2fe381e02f9097f7a9e6  
**DATABASE_SCHEMA_VERSION_TARGET** = 4.0.0  
**MIGRATION_ID** = MIGRATION_20261010_004_NATURAL_EVIDENCE_EPOCH_AND_HISTORICAL_ISOLATION  
**DDL_HASH** = ab82282210aef8c8873da0ea4a31fe2c865b0c2a63e1bbdce86320e9e67dde0b  
**PRE_PUSH_RECONCILIATION_SHA256** = 895c2e567e25a53d0099363960aab36f38a8c87999db3f9d1d2e27a92a9d4af4  
**PROCESS_RACE_RECONCILIATION_SHA256** = c4b5d6ffe87bb83b3c25205f4ca0c151ff21f02a809fdb73824a880690986fc5  
**PRODUCTION_BASELINE_SHA** = 793df4844cfe0879ad8e6143a256dd214cf0b908  
**PUSHED_HEAD** = 8fd4622fc200baa16c371372feb96bf4f2d3d6e8  
**PRODUCTION_RUNTIME_SHA** = 8fd4622fc200baa16c371372feb96bf4f2d3d6e8  
**CANDIDATE_FREEZE_STATUS** = FROZEN_PRE_DEPLOY  
**NATURAL_EVIDENCE_EPOCH_ID** = SPRINT3_CANDIDATE004_EPOCH_001  
**PRODUCTION_EPOCH_ACTIVATED** = YES  
**FIRST_NATURAL_CAPTURE_EVIDENCE_AUTHORIZED** = YES  
**FIRST_NATURAL_CAPTURE_STATUS** = AWAITING_NATURAL_PRODUCTION_OCCURRENCE  
**PRODUCTION_RELEASE_STATUS** = DEPLOYED_VERIFIED_EPOCH_ACTIVE_WAITING_FOR_NATURAL_CAPTURE  
**FINAL_OUTCOME** = OUTCOME_B  
**EXTERNAL_VALIDATION_STATUS** = DEFERRED  
**GATE_12_STATUS** = DEFERRED / NOT_SATISFIED  
**EMPIRICAL_SCANNER_QUALITY** = INSUFFICIENT_EVIDENCE  
**MODEL_TUNING** = FROZEN  
**LEARNING_CLAIM** = NOT_AUTHORIZED  

---

### 1. Release Scope & Objectives

This canonical release note binds Candidate Generation 004 for Radar Sprint 3 Production Shadow.
Candidate 004 remediates Candidate 002 (provenance and boot-warmup defects) and Candidate 003 (denominator mutation and epoch boundary defects).

Key Architectural Components:
1. **Schema V4 Migration**: Introduces `natural_evidence_epochs`, `epoch_activation_receipts`, `evidence_epoch_memberships`, `migration_source_manifests`, `migration_unit_dispositions`, and `historical_reconciliation_records`.
2. **Historical Isolation**: 100% of historical pre-epoch shadow observations are quarantined with disposition `PRE_EPOCH_INELIGIBLE`.
3. **Relational Natural Denominator**: The natural production denominator is strictly derived via relational join on active epoch prospective memberships.
4. **Invocation Provenance & Caller Integrity**: Separates `SCHEDULED_PRODUCTION` (natural) from `BOOT_WARMUP` and `MANUAL_OPERATOR` (non-natural). Callers cannot self-declare natural status.
5. **Epoch Activation Governance**: Epoch activation requires an immutable cryptographic receipt and does not automatically authorize natural evidence admission.
6. **Strict Non-Actioning**: Preserves fail-closed non-actioning boundaries (zero order execution, zero portfolio mutation).

---

### 2. Pre-Deploy Binding Authorities

- **Candidate Functional SHA**: `fafba93ccd60f40792bb3f5ea6702804eeea40cf`
- **Candidate Freeze Artifact**: `docs/domain/vcp/sprint_3/candidates/CANDIDATE_GENERATION_004_FREEZE.json` (`297ef5ab6399f57ef541d4413dc9e9fde15a143f8c1d18e1608173a4165f8e90`)
- **Pre-Push Reconciliation Artifact**: `evidence/release-preparation/2026-10-10-radar-sprint-3-shadow-candidate-004/pre_push_reconciliation.json` (`895c2e567e25a53d0099363960aab36f38a8c87999db3f9d1d2e27a92a9d4af4`)
- **Process Race Reconciliation Artifact**: `evidence/release-preparation/2026-10-10-radar-sprint-3-shadow-candidate-004/process_epoch_race_reconciliation.json` (`c4b5d6ffe87bb83b3c25205f4ca0c151ff21f02a809fdb73824a880690986fc5`)
- **DDL Hash**: `ab82282210aef8c8873da0ea4a31fe2c865b0c2a63e1bbdce86320e9e67dde0b`
- **Target Schema Version**: `4.0.0`
- **Migration Identity**: `MIGRATION_20261010_004_NATURAL_EVIDENCE_EPOCH_AND_HISTORICAL_ISOLATION`

---

### 3. Production Deployment Targets

- **Target Backend**: Railway (`tranquil-radiance`, `web` service, production environment)
- **Target Frontend**: Cloudflare Pages (`finance-xp8.pages.dev`, `arxterminal.com`)
- **Storage Topology**: Single replica persistent mount `/root` (`/root/analyst_dashboard/data/shadow_evidence.db`) in SQLite WAL mode.

---

### 4. Production Deployment & Verification Attestation

- **Pushed Commit HEAD**: `8fd4622fc200baa16c371372feb96bf4f2d3d6e8`
- **Production Runtime SHA**: `8fd4622fc200baa16c371372feb96bf4f2d3d6e8` (Contains C004 functional SHA `fafba93ccd60f40792bb3f5ea6702804eeea40cf`)
- **Production Runtime Identity**: `VERIFIED`
- **Railway Deployment ID**: `7edd156a-c092-4d24-98e6-f7bef77d84d0`
- **Railway Deployment Status**: `SUCCESS / ONLINE` (`/health` returns `{"status":"online"}`, storage persistence `VERIFIED`)
- **Cloudflare Pages Deployment ID**: `production-cloudflare-pages` (`cf-pages-8fd4622`)
- **Cloudflare Deployment Status**: `SUCCESS / ONLINE` (HTTP 200 on `finance-xp8.pages.dev` and `arxterminal.com`)
- **GitHub Actions Run ID**: `38046790101`
- **GitHub Actions Status**: `FAILURE` (Classification: `PREEXISTING_FAILURE` on baseline commit `793df4844cfe0879ad8e6143a256dd214cf0b908`, Run ID `38030363811`)
- **Production Replica Count**: 1
- **Application Process Count**: 2 (uvicorn workers)
- **Production Volume Mount**: `/root`
- **Production SQLite Path**: `/root/analyst_dashboard/data/shadow_evidence.db`
- **Production Storage Topology**: `SINGLE_REPLICA_ONLY` (`PASS`)
- **Production Database Generation ID**: `GEN_20261010_RADAR_SPRINT_3_SHADOW_001`
- **Unverified Database Restore Detected**: `NO`
- **Production Schema Version**: `4.0.0`
- **Migration Completion Receipt ID**: `RECEIPT_MIGRATION_20261010_004_NATURAL_EVIDENCE_EPOCH_AND_HISTORICAL_ISOLATION_313`
- **Pre-Migration Source Count**: `313`
- **Pre-Migration Source Population Hash**: `9650c0fc7dc17f4b7e1b51f572fd190f495126cbb53172e2c400614e8ab2cf15`
- **Post-Migration Source Count**: `313`
- **Post-Migration Source Population Hash**: `9650c0fc7dc17f4b7e1b51f572fd190f495126cbb53172e2c400614e8ab2cf15`
- **Unclassified Migration Units**: `0`
- **Source Units Without Disposition**: `0`
- **Dispositions Without Source Unit**: `0`
- **Duplicate Dispositions**: `0`
- **Production Scheduler Provider**: `RAILWAY_CRON_OR_SCHEDULER_SERVICE` (`NaturalVCPTriggerService`)
- **Scheduler Job ID**: `job-vcp-daily-eod`
- **Scheduler Workload Principal ID**: `scheduler:arx-daily-cadence`
- **Operator Principal ID**: `operator:manual-trigger`
- **Bootstrap Principal ID**: `bootstrap:container-warmup`
- **Scheduler Principal Distinct from Operator**: `YES`
- **Scheduler Principal Distinct from Bootstrap**: `YES`
- **Scheduler Secret Distinct from Operator**: `YES`
- **Caller Can Self-Declare Natural**: `NO`
- **Natural Trigger Reachability**: `PASS`
- **Production Provenance Forgery Rejected**: `PASS`
- **Production Provenance Conflict Fail-Closed**: `PASS`
- **Production Restart Durability Pre-Activation**: `PASS`
- **Epoch Activation Readiness**: `PASS`
- **Activation Receipt ID**: `rcpt-act-SPRINT3_CANDIDATE004_EPOCH_001-1`
- **Activation Receipt Content Hash**: `15d6683d17c8bb9ad06980b6fe675c087d1221aab51fd6edf038e797bc9d2e13`
- **Activation Sequence**: 1
- **Activated At UTC**: `2026-10-10T11:15:00+00:00`
- **Post-Activation Integrity Gate**: `PASS`
- **Reauthorization Receipt ID**: `rcpt-reauth-SPRINT3_CANDIDATE004_EPOCH_001-1`
- **Natural Evidence Re-Authorization**: `PASS`
- **First Natural Capture Evidence Authorized**: `YES`
- **First Natural Capture Status**: `AWAITING_NATURAL_PRODUCTION_OCCURRENCE` (Section 22 Outcome B)
- **Current Natural Denominator**: `0`
- **Denominator Reconstruction Delta**: `0`
- **Contamination Check**: `0` (Zero synthetic, replay, boot, admin, or historical observations admitted into natural denominator)
