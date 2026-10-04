# ARX TERMINAL — SAAS FOUNDATION PHASE 1G — MISSING HISTORICAL EVIDENCE GOVERNANCE ADJUDICATION

## 1. Frozen Technical State

This governance adjudication proceeds strictly from the reconciled production baseline:

```ini
PREDECESSOR_GATE =
  HOLD_ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_RELEASE_RECONCILIATION

RUNTIME_PRODUCTION_HEALTH =
  VERIFIED

CURRENT_DATABASE_INTEGRITY =
  VERIFIED

DATABASE_VOLUME_PERSISTENT =
  YES

PRODUCTION_DATABASE_PATH =
  /root/.finance_platform_history.db

DATABASE_STORAGE_CLASS =
  PERSISTENT_DISK

WS_DEFAULT_PRIVATE_PERSISTENCE_PROHIBITED =
  VERIFIED_LIVE

CROSS_WORKSPACE_UNAUTHORIZED_ACCESS =
  REJECTED_LIVE

PRIVATE_CACHE_BOUNDARY =
  VERIFIED_LIVE

PUBLIC_ROUTE_CONTEXT_INDEPENDENCE =
  VERIFIED_LIVE

INV_SAAS_01 =
  PRESERVED

INV_SAAS_02 =
  PRESERVED

INV_SAAS_03 =
  ENFORCED

INV_SAAS_04 =
  PRESERVED

INV_SAAS_05 =
  PRESERVED

INV_SAAS_06 =
  ENFORCED

INV_SAAS_07 =
  ENFORCED

CONFIRMED_DATA_LOSS =
  NO

CONFIRMED_PRODUCTION_DEFECT =
  NO
```

---

## 2. Immutable Historical Evidence Gaps

The following evidence gaps cannot be reconstructed after the fact without fabricating historical reality:

```ini
PRE_DEPLOYMENT_BACKUP =
  NOT_ESTABLISHED

PRE_MIGRATION_PORTFOLIO_ROWS =
  NOT_ESTABLISHED

PRE_MIGRATION_JOURNAL_ROWS =
  NOT_ESTABLISHED

PRE_MIGRATION_COCKPIT_ACTION_ROWS =
  NOT_ESTABLISHED

PRE_MIGRATION_USER_PROFILE_ROWS =
  NOT_ESTABLISHED

HISTORICAL_ZERO_ROW_LOSS =
  NOT_FULLY_ADJUDICABLE
```

Per Section 2 of the governing protocol, these limitations are treated as permanent historical facts. No synthetic or retroactively dated artifacts have been created.

---

## 3. Evidence Hierarchy

All claims and findings are evaluated in accordance with the established evidence hierarchy:

1. **Contemporaneous Pre-Deployment Artifacts**: Cryptographic hashes, database snapshots, or row audit logs captured strictly before migration execution. *(Result: Absent for Phase 1G).*
2. **Immutable Migration/Runtime Artifacts**: Recorded migration execution traces and system initializations in runtime logs.
3. **Current Production Database Integrity**: Direct runtime and schema verification on the active persistent storage volume (`/root/.finance_platform_history.db` on Railway).
4. **Production Behavioral Evidence**: Live HTTP/API probes proving rejection of unauthorized access (HTTP 403), enforcement of `ws_default` non-persistence (HTTP 403), and retention of private cache headers.
5. **Source/Test Evidence**: Automated test suites and idempotent expand-only schema definitions (`apply_workspace_tenancy_migration`).
6. **Inference**: Strictly prohibited from masquerading as higher-order historical proof.

---

## 4. Bounded Historical-Artifact Search

A comprehensive search of repository artifacts, git history, and local workspace ledgers was conducted to identify any pre-existing contemporaneous evidence:

### Discovered Artifact Evaluation
- **Artifact**: `pre_migration_workspace_fingerprint.json`
- **Created At**: Prior commit stage (git tree pre-migration state).
- **Provenance**: Local git staging tree audit.
- **Predates Migration**: YES (for file worktree state).
- **Independent of Reconciliation**: YES.
- **Contents**: 778 repository markdown and documentation files with file sizes and SHA-256 hashes. Contains zero SQLite database hashes, zero database file sizes, and zero table row counts.
- **Evidentiary Value for Database/Row Counts**: `NONE`.
- **Classification**: `PRE_MIGRATION_FINGERPRINT = INSUFFICIENT_FOR_HISTORICAL_ZERO_ROW_LOSS`.

No other independent pre-deployment database snapshots or row count logs exist.

---

## 5. Backup-Control Classification

### Governance Question
*Was a verified pre-deployment backup an absolute release-safety prerequisite, or was it a procedural safeguard whose absence can be accepted after-the-fact when production durability and integrity are independently verified?*

### Classification & Rationale
```ini
PRE_DEPLOYMENT_BACKUP_REQUIREMENT =
  PROCEDURAL_CONTROL
```

**Rationale**:
A pre-deployment backup functions as a rollback contingency mechanism in the event that a destructive schema change or migration script corrupts persistent data during deployment. In Phase 1G:
- The migration was strictly additive and expand-only (`CREATE TABLE IF NOT EXISTS`, `ALTER TABLE ADD COLUMN workspace_id TEXT` nullable).
- The migration executed successfully without throwing SQLite exceptions or operational locks.
- The live database resides on a persistent ext4 volume (`web-volume` mounted at `/root` on Railway) where physical storage persistence is verified via device ID distinctness (`mount_dev != root_dev`).
- The running application has been verified live across multiple endpoints with zero errors.

While the omission of an external pre-deployment backup snapshot constituted a procedural safeguard breach, the application is already deployed, operational, and stable. Fabricating a backup post-deployment and backdating it would violate governance integrity. Therefore, this breach is adjudicated as a procedural control lapse that is acceptable under qualified operational release.

---

## 6. Historical Data-Preservation Evidence Classification

### Governance Question
*Is mathematical proof of zero historical row loss required for production acceptance, or is absence of confirmed loss plus verified current integrity sufficient for qualified operational acceptance?*

### Classification & Rationale
```ini
HISTORICAL_ZERO_ROW_LOSS_REQUIREMENT =
  QUALIFIABLE_HISTORICAL_EVIDENCE_GAP
```

**Rationale**:
Because pre-migration row counts were not recorded prior to deployment, mathematical proof of $\Delta \text{rows} = 0$ is permanently impossible to demonstrate. However:
- The DDL applied in Phase 1G executed solely non-destructive operations (no `DROP TABLE`, no `DROP COLUMN`, no `TRUNCATE`, no destructive `DELETE`).
- Legacy `user_id` columns, constraints, and indexes were completely preserved.
- Live user queries return expected user data with zero data corruption.
- There is zero confirmed data loss (`CONFIRMED_DATA_LOSS = NO`).

Demanding mathematical proof of historical row preservation when the baseline was not recorded would permanently deadlock the system. Consequently, this gap is classified as a qualifiable historical evidence gap.

---

## 7. Current Production Integrity Attestation

Without mutating production data, current production state on Railway (`https://web-production-e370b.up.railway.app`) is attested as follows:

```ini
CURRENT_DATABASE_INTEGRITY =
  VERIFIED

CURRENT_SCHEMA_INTEGRITY =
  VERIFIED

CURRENT_WS_DEFAULT_PRIVATE_ROWS =
  0

CURRENT_GUESSED_WORKSPACE_ASSIGNMENTS =
  0

CURRENT_INVALID_MEMBERSHIPS =
  0

CURRENT_CROSS_WORKSPACE_LEAKAGE =
  0_CONFIRMED

CONFIRMED_DATA_LOSS =
  NO

CONFIRMED_PRODUCTION_DEFECT =
  NO
```

---

## 8. Release Identity Lineage

```ini
INTEGRATION_RUNTIME_SHA =
  756674fb9d9e9dd4bd9709c6d00408f34dc02bcd

CURRENT_MAIN_SHA =
  725dcfffa2922bf625e888fc9fa3203245f90996

CURRENT_MAIN_RELATION_TO_RUNTIME =
  DOCUMENTATION_ONLY_SUCCESSOR

PRODUCTION_RELEASE_IDENTITY =
  EXACT_INTEGRATION_RUNTIME_SHA_WITH_DOCUMENTATION_ONLY_MAIN_SUCCESSOR
```

---

## 9. Formal Governance Policy Decision

```ini
PRE_DEPLOYMENT_BACKUP_REQUIREMENT =
  PROCEDURAL_CONTROL

HISTORICAL_ZERO_ROW_LOSS_REQUIREMENT =
  QUALIFIABLE_HISTORICAL_EVIDENCE_GAP

QUALIFIED_PRODUCTION_PASS_ALLOWED =
  YES
```

An unqualified production PASS (`PASS_ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_RELEASE`) is strictly prohibited due to the absence of contemporaneous pre-deployment evidence. However, because technical runtime health, storage durability, and security boundaries are 100% verified, a qualified production acceptance is formally authorized:

```ini
GATE =
  PASS_ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_RELEASE_QUALIFIED

PHASE_1G_PRODUCTION =
  VERIFIED_WITH_HISTORICAL_EVIDENCE_LIMITATIONS

RUNTIME_PRODUCTION_HEALTH =
  VERIFIED

CURRENT_DATABASE_INTEGRITY =
  VERIFIED

DATABASE_VOLUME_PERSISTENT =
  YES

PRE_DEPLOYMENT_BACKUP =
  NOT_ESTABLISHED

HISTORICAL_ZERO_ROW_LOSS =
  NOT_FULLY_ADJUDICABLE

CONFIRMED_DATA_LOSS =
  NO

CONFIRMED_PRODUCTION_DEFECT =
  NO

HISTORICAL_EVIDENCE_LIMITATION_ACCEPTED =
  YES

PRODUCTION_RELEASE_IDENTITY =
  EXACT_INTEGRATION_RUNTIME_SHA_WITH_DOCUMENTATION_ONLY_MAIN_SUCCESSOR

NEXT_AUTHORIZED_ACTION =
  ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_OBSERVATION_GATE

AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```

---

## 10. Prospective Mandatory Controls (Preventing Recurrence)

To prevent similar evidence gaps in future migrations (specifically the upcoming contract phase and Phase 2 migrations), the following mandatory pre-flight controls are enacted prospectively:

```ini
FUTURE_PRE_DEPLOY_BACKUP_REQUIRED =
  YES

FUTURE_PRE_MIGRATION_DB_FINGERPRINT_REQUIRED =
  YES

FUTURE_PRE_MIGRATION_TABLE_COUNTS_REQUIRED =
  YES

FUTURE_BACKUP_SHA256_REQUIRED =
  YES

FUTURE_POST_MIGRATION_TABLE_COUNTS_REQUIRED =
  YES

FUTURE_DATA_PRESERVATION_DIFF_REQUIRED =
  YES
```

Before any future database migration is executed in production:
1. An external SQLite snapshot (`.db.bak`) must be dumped and its SHA-256 recorded.
2. Exact pre-migration row counts for all tables must be exported to a versioned json ledger.
3. Post-migration row counts must be compared to verify $\Delta \text{rows} \ge 0$.
4. Failure to produce contemporaneous pre-flight artifacts will fail the pre-release gate closed.

---

## 11. Next Authorized Action

The next authorized action is:

```ini
NEXT_AUTHORIZED_ACTION =
  ARX_SAAS_FOUNDATION_PHASE_1G_PRODUCTION_OBSERVATION_GATE
```

Automatic execution of the observation successor is NOT authorized.
