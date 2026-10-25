# ARX TERMINAL — SMART MONEY
## EXTERNAL AUTHORITY ARCHITECTURE CONTRACT
### ARTIFACT 5: CHANGE CONTROL AND REQUALIFICATION POLICY

**Document Identity**: `urn:arx:contract:smart_money:change_control:v1`  
**Protocol Version**: `1.0.0`  
**Governance Scope**: System Evolution, Requalification Gates, Protocol Versioning, and Decommissioning  
**Security Classification**: CONFIDENTIAL — ARCHITECTURAL GOVERNANCE FREEZE  
**Status**: CANONICAL AND FROZEN  

---

### 1. Purpose & Core Governance Law

This contract governs the change management, requalification criteria, protocol versioning, schema migration, and decommissioning protocols for ARX Terminal Smart Money.

The governing invariant is:
$$\text{Any Architectural, Cryptographic, or Infrastructure Mutation Invalidates Active Qualification}$$

No production or qualification authorization can survive an unverified change to code, dependencies, cloud policies, schemas, or cryptographic keys.

---

### 2. Comprehensive Requalification Trigger Classification

The architecture explicitly classifies all nineteen recognized modification events:

```
+---------------------------------------------------------------------------------------------------------------------+
|                                            REQUALIFICATION TRIGGER MATRIX                                           |
+----+----------------------------------------+-------------------------------+---------------------------------------+
| ID | Trigger Event Description              | Requalification Tier          | Mandatory Verification Scope          |
+----+----------------------------------------+-------------------------------+---------------------------------------+
| C1 | release_sha_change                     | FULL_A_E_REQUALIFICATION      | Complete pipeline re-qualification    |
| C2 | workflow_change (.github/workflows)   | FULL_A_E_REQUALIFICATION      | CI execution and OIDC claim re-audit  |
| C3 | github_action_dependency_change        | FULL_A_E_REQUALIFICATION      | Supply-chain integrity re-audit       |
| C4 | dependency_lock_change (poetry.lock)   | FULL_A_E_REQUALIFICATION      | Software supply-chain re-audit        |
| C5 | runtime_config_change                  | TARGETED_REQUALIFICATION      | Configuration delta regression check  |
| C6 | iam_policy_change                      | FULL_A_D_REQUALIFICATION      | Boundary and permission audit         |
| C7 | issuer_key_rotation                    | TARGETED_REQUALIFICATION      | Cryptographic verification & Rekor    |
| C8 | gate3_scope_change                     | FULL_A_E_REQUALIFICATION      | Authorization contract & CAS re-test  |
| C9 | state_schema_change                    | FULL_A_E_REQUALIFICATION      | DDL migration & atomicity verification|
| C10| transaction_protocol_change            | FULL_A_E_REQUALIFICATION      | Distributed CAS & fencing re-test     |
| C11| evidence_metadata_schema_change        | FULL_A_D_REQUALIFICATION      | Custody tuple & replay re-test        |
| C12| retention_policy_change                | FULL_A_D_REQUALIFICATION      | WORM Compliance Lock audit            |
| C13| state_backend_provider_change          | FULL_A_E_REQUALIFICATION      | Full distributed authority validation |
| C14| evidence_store_provider_change         | FULL_A_E_REQUALIFICATION      | WORM storage & replay validation      |
| C15| region_change                          | FULL_A_E_REQUALIFICATION      | Cross-region failover & latency check |
| C16| account_change                         | FULL_A_E_REQUALIFICATION      | Multi-account IAM boundary validation |
| C17| disaster_recovery_event                | FULL_A_E_REQUALIFICATION      | Rehydration & state integrity drill   |
| C18| break_glass_event                      | FULL_A_E_REQUALIFICATION      | Post-mortem audit & full requalify    |
| C19| protocol_version_change                | FULL_A_E_REQUALIFICATION      | Version compatibility & state boundary|
+----+----------------------------------------+-------------------------------+---------------------------------------+
```

$$\mathtt{REQUALIFICATION\_TRIGGERS\_UNCLASSIFIED = 0}$$

---

### 3. Qualification Validity & Lifecycle

```yaml
QUALIFICATION_BOUND_TO_RELEASE_SHA: YES
QUALIFICATION_MAX_LIFESPAN_DAYS: 90
QUALIFICATION_EXPIRY_POLICY: 90_DAYS_OR_ON_INVALIDATION
QUALIFICATION_INVALIDATION_EVENTS:
  - GIT_RELEASE_SHA_MUTATION
  - GITHUB_WORKFLOW_MODIFICATION
  - IAM_POLICY_MODIFICATION
  - DATABASE_SCHEMA_MIGRATION
  - CRYPTOGRAPHIC_KEY_ROTATION
  - SECURITY_BREACH_OR_SUSPECTED_COMPROMISE
  - STORAGE_CONFIGURATION_ALTERATION
```

A qualification verdict is strictly cryptographically coupled to the Git commit SHA evaluated during Run A and Run B. Any commit to `main`, even a documentation fix, produces a new SHA and requires requalification.

---

### 4. Protocol Versioning & Migration Safety

The architecture defines four explicit protocol identifiers:

```yaml
AUTHORITY_PROTOCOL_VERSION: 1.0.0
STATE_PROTOCOL_VERSION: 1.0.0
EVIDENCE_PROTOCOL_VERSION: 1.0.0
QUALIFICATION_PROTOCOL_VERSION: 1.0.0
```

#### 4.1 Version Compatibility Invariants
1. **SemVer Compliance**: MAJOR versions break backwards compatibility; MINOR versions add backward-compatible functionality; PATCH versions supply backward-compatible defect repairs.
2. **Mixed-Version Runner Prohibition**: Runners executing protocol version $P_A$ are strictly prohibited from interacting with state stores operating under protocol version $P_B$ where $P_A \neq P_B$. Any version mismatch raises `ProtocolVersionMismatchError` and aborts.
3. **Downgrade Prohibition**:
   $$\text{If } \mathtt{runner\_protocol\_version} < \mathtt{backend\_protocol\_version} \implies \mathtt{TERMINAL\_HOLD}$$
   Downgrade attacks are rejected deterministically.

---

### 5. External Schema Migration Governance

When database DDL or evidence metadata schemas must be migrated:

1. **Forward-Only Migration**: Schema migrations advance monotonically. Rollbacks are performed via forward-compensating migrations, never by rewinding DDL.
2. **Atomic Manifest Logging**: Every migration is recorded in an immutable `schema_migrations` table with fields `(migration_id, schema_version, ddl_hash, applied_by, applied_at, witness_sha)`.
3. **Anti-Second-Genesis Invariant**: Schema migration tools must verify the existence of the active epoch head before executing. Migrations cannot execute against an uninitialized or deleted epoch.

---

### 6. Authority Transfer & Decommissioning

If the physical cloud backend, storage bucket, or account must be retired or replaced:

1. **Lineage Preservation**: The final state head of the retiring system ($HEAD_{final}$) is frozen with status `DECOMMISSIONED`.
2. **Transfer Attestation Certificate**: A cryptographic transfer manifest is signed by `RECOVERY_APPROVER` quorum linking $HEAD_{final}$ of the legacy backend to $HEAD_{genesis}$ of the successor backend.
3. **Zero Data Loss Guarantee**: All WORM evidence objects are replicated to the successor bucket with identical SHA-256 hashes before the legacy store is archived.
4. **Decommissioning Ceremony**: Legacy storage is placed in archival read-only mode for the remaining duration of the compliance retention epoch.
