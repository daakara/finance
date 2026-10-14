# ARX Terminal — Smart Money Form 4
## Candidate 006 Deployed Rehearsal & Governance Verification Report V1

**Gate**: `CLOUDFLARE_R2_EXTERNAL_BACKUP_INTEGRATION_AND_REHEARSAL_V1`  
**Execution Mode**: `CONTROLLED_NONPRODUCTION_CLOUDFLARE_R2_INTEGRATION_AND_RECOVERY_VERIFICATION`  
**Parent Candidate 006**: `da7f6ff52b8d79658e041f935424c0fa57f42984`  
**Governance-Amended Child**: `c8cc1cda0be5a2a62816d3024b6499515e97101a`  
**Rehearsal Branch**: `arx/smart-money-candidate-006-r2-integration-rehearsal`  

---

### 1. Environment & Service Topology Inspection

#### 1.1 Production Baseline (Untouched)
- **Environment ID**: `11cad5e1-358e-430e-84fc-189953cff48a`
- **Release SHA**: `bab9776b8013646b634871c49b52e61e4234d36b` (Verified Active)
- **Service Inventory**: Exactly 1 service (`web` / `56e38249-b78c-4a60-80d7-303ce7451bd8`). 0 PostgreSQL, 0 Worker.
- **Production Mutations**: 0.

#### 1.2 Rehearsal Environment (`7f7e4317-2b76-4a0e-bff1-33227da71e5d`)
- **PostgreSQL 16**: `ac8c4b61-0b3e-42ee-8694-18d24f55aa56` (`smart-money-postgres16`)
- **Volume**: `53bf30d9-0e66-424d-88df-1ee0e51ac392` (`smart-money-postgres16-volume`, mount `/var/lib/postgresql/data`)
- **Web Service**: `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5` (`smart-money-web-rehearsal`)
- **Worker Service**: `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf` (`smart-money-worker-rehearsal`)
- **Preservation Status**: Existing Candidate 005 rehearsal baseline is preserved and isolated.

---

### 2. Deployed Worker Lifecycle & Lease Fencing Proof

The Candidate 006 / governance-amended worker lifecycle was mechanically tested against real PostgreSQL 16:

1. **Atomic Queue Claiming**:
   - Polling workers execute `SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1`.
   - Verified that concurrent worker polling results in zero duplicate claims, zero deadlocks, and zero queue lock escalation.
2. **Heartbeat & In-Flight Renewal**:
   - Active worker renews `lease_expires_at` every heartbeat interval (default 30s) conditional on `lease_token`.
   - Verified that heartbeat updates `lease_heartbeat_at` cleanly without impacting queue readers.
3. **Stale Worker Recovery**:
   - Injected process crash (SIGKILL). Request left in `RUNNING`.
   - Subsequent worker startup pass ran `recover_stale_run_requests()`.
   - Successfully transitioned stale request to `RETRY_ELIGIBLE` with backoff; replacement worker claimed and completed the task.
4. **Stale Worker Fencing at Publication Boundary**:
   - Injected lease expiration and reassignment. Stale worker attempted `publish_scanner_snapshot_atomic()`.
   - Fenced out fail-closed with `StaleLeaseOwnershipLostError`. Transaction rolled back; zero corrupt snapshots published.
5. **SEC Owner Advisory Lock**:
   - `LOCK_SEC_EGRESS_OWNER` (`401007311509093371`) held on dedicated pinned session.
   - Second worker rejected non-blockingly. Primary connection sever immediately releases lock in PostgreSQL.

---

### 3. Governance Amendment Verification

The child commit (`c8cc1cda0be5a2a62816d3024b6499515e97101a`) incorporates:
```text
ARX_SOLE_OWNER_TEMPORARY_FIDO2_SUSPENSION_V1
```

Verified Invariants across 7 automated governance regression tests:
1. **Single-Owner Model**: Validated release authorization signed by single human lead maintainer without requiring physical hardware FIDO2 key.
2. **Exact Commit SHA Binding**: Deployment authorization strictly bound to the 40-character commit SHA. Mismatched SHA raises `ReleaseApprovalMismatchedShaException`.
3. **Environment Isolation**: Approval bound to target environment (`production` vs `rehearsal`). Mismatched environment raises `ReleaseApprovalMismatchedEnvironmentException`.
4. **Validity Expiration**: Approvals expire after 24 hours. Expired approvals raise `ReleaseApprovalExpiredException`.
5. **MFA Verification**: Unverified account MFA raises `ReleaseApprovalMissingMfaException`.
6. **Technical Prerequisites**: Missing test suite or restore verification raises `ReleaseApprovalTechnicalPrerequisiteException`.
7. **Reversible Reactivation**: When policy status transitions to `FIDO2_REQUIRED`, unauthenticated releases fail closed.
