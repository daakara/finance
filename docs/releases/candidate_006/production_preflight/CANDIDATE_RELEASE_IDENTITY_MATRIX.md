# ARX Terminal Smart Money Candidate 006: Candidate Release Identity Matrix

**Execution Mode**: `PRODUCTION_CUTOVER_DESIGN_READ_ONLY_PREFLIGHT_NONPRODUCTION_ROLLBACK_VALIDATION`  
**Authorization State**: `PRODUCTION_CHANGES_NOT_AUTHORIZED`  
**Evaluation Timestamp**: `2026-10-14T06:21:00Z`  
**Evidence Classification**: `OBSERVED` / `TESTED`

---

## 1. Release Identity & Deployment Matrix

The table below provides an exact cross-sectional mapping of every release candidate version, git reference, infrastructure deployment instance, and active runtime state across Rehearsal and Production environments.

| Entity / Target | Git Commit SHA | Git Tree SHA | Origin Branch | Deployed Service ID | Railway Environment ID | Deployment ID | Runtime Commit Verified | Lifecycle Status | Evidence Class |
|---|---|---|---|---|---|---|---|---|---|
| **Candidate 005 (Frozen Base)** | `09f183de52e275ae543615747dff2f9ae00ef9ca` | `c4ecd94ce894032a987b0a51440a876eab69afba` | `arx/smart-money-postgres-runtime-integration-candidate-v1` | N/A (Superseded) | N/A | N/A | N/A | `FROZEN_BASE` | `OBSERVED` |
| **Candidate 006 (Base)** | `da7f6ff52b8d79658e041f935424c0fa57f42984` | `288b164be323f61b3d45c5070e3f4bd6b7354e13` | `arx/smart-money-candidate-006-worker-recovery-backup` | N/A (Superseded) | N/A | N/A | N/A | `BASE_CANDIDATE` | `OBSERVED` |
| **Candidate 006 (Governance)** | `c8cc1cda0be5a2a62816d3024b6499515e97101a` | `d65e57a5865d30479d1058c7dab50d3608e26e7b` | `arx/smart-money-candidate-006-governance-amendment` | N/A (Superseded) | N/A | N/A | N/A | `GOVERNANCE_AMENDED` | `OBSERVED` |
| **Candidate 006 (R2 Rehearsal)** | `dc6e5b79c8fb42e6c9029e79d2a4463afac016f4` | `0ffd2e60c32a636ca1eca58216fa952085fcb285` | `arx/smart-money-candidate-006-r2-integration-rehearsal` | N/A (Superseded) | N/A | N/A | N/A | `REHEARSAL_VERIFIED` | `OBSERVED` |
| **Candidate 006 (Sec Remediation)** | `9cad087868386d40dc24970be80776abb3f8756a` | `82f9903306ed3a0027549d55454d563143bbeb7b` | `arx/smart-money-candidate-006-r2-security-remediation` | N/A (Parent) | N/A | N/A | N/A | `SECURITY_REMEDIATED` | `OBSERVED` |
| **Candidate 006 (Remediation Closure)** | `76a8691339c30edc35e4e056d5b295509a81fe3a` | `80309593afb6b1b722bed87c111bf2b58056c688` | `arx/smart-money-candidate-006-r2-security-remediation` | **TARGET_HEAD** | N/A | N/A | N/A | `PREFLIGHT_TARGET` | `OBSERVED` |
| **Rehearsal Worker Service** | `76a8691339c30edc35e4e056d5b295509a81fe3a` | `80309593afb6b1b722bed87c111bf2b58056c688` | `arx/smart-money-candidate-006-r2-security-remediation` | `929874a7-bcd9-41b3-9bc7-d5cd7720f7bf` | `7f7e4317-2b76-4a0e-bff1-33227da71e5d` | `bbc0507e-4282-4e2b-9b94-d3fd1e49ae6c` | `76a8691339c30edc35e4e056d5b295509a81fe3a` | `DEPLOYED_ACTIVE` | `OBSERVED` / `TESTED` |
| **Rehearsal Web Service** | `76a8691339c30edc35e4e056d5b295509a81fe3a` | `80309593afb6b1b722bed87c111bf2b58056c688` | `arx/smart-money-candidate-006-r2-security-remediation` | `6b53ad3f-0d59-4a7f-83dc-8731efe48cf5` | `7f7e4317-2b76-4a0e-bff1-33227da71e5d` | `a0bf4e4e-dcab-4f32-8459-76deeceefde4` | `76a8691339c30edc35e4e056d5b295509a81fe3a` | `DEPLOYED_ACTIVE` | `OBSERVED` / `TESTED` |
| **Rehearsal PostgreSQL Service**| N/A (Docker pg 16.15) | N/A | N/A | `ac8c4b61-0b3e-42ee-8694-18d24f55aa56` | `7f7e4317-2b76-4a0e-bff1-33227da71e5d` | `84c4ad8d-b0df-4b7f-b258-0027f6cfb160` | N/A | `DEPLOYED_ACTIVE` | `OBSERVED` / `TESTED` |
| **Production Web Service** | `bab9776b8013646b634871c49b52e61e4234d36b` | `13d14564baf94eceab215478280241668a9b1d71` | `main` | `56e38249-b78c-4a60-80d7-303ce7451bd8` | `11cad5e1-358e-430e-84fc-189953cff48a` | `0b2544ed-075a-46a8-97a8-e1b45b4d73b3` | `bab9776b8013646b634871c49b52e61e4234d36b` | `PRODUCTION_ACTIVE` [Errata: Tree corrected from 5e89694] | `OBSERVED` |
| **Production Worker Service** | **NONE** | **NONE** | N/A | **NONE (0 instances)** | `11cad5e1-358e-430e-84fc-189953cff48a` | N/A | N/A | `NOT_PROVISIONED` | `OBSERVED` |
| **Production PostgreSQL Service**| **NONE** | **NONE** | N/A | **NONE (0 instances)** | `11cad5e1-358e-430e-84fc-189953cff48a` | N/A | N/A | `NOT_PROVISIONED` | `OBSERVED` |

---

## 2. Remote Custody and Branch State

* **Remote Tracking Branch**: `origin/arx/smart-money-candidate-006-r2-security-remediation`
* **Local Tracking**: Up to date with origin (`PUSH_VERIFIED`).
* **Active Head SHA**: `76a8691339c30edc35e4e056d5b295509a81fe3a`
* **Tree State**: Clean working tree.

---

## 3. Discrepancy & Drift Verdict

1. **Rehearsal Parity**: Worker deployment `bbc0507e-4282-4e2b-9b94-d3fd1e49ae6c` and Web deployment `a0bf4e4e-dcab-4f32-8459-76deeceefde4` both run exact commit `76a8691339c30edc35e4e056d5b295509a81fe3a`. Drift: **ZERO (0%)**.
2. **Production Baseline**: Production web runs commit `bab9776b8013646b634871c49b52e61e4234d36b`. Zero modifications or deployments have been dispatched to production. Drift: **ZERO (0%)**.
3. **Proposed Release Target**: Commit `76a8691339c30edc35e4e056d5b295509a81fe3a` (Tree: `80309593afb6b1b722bed87c111bf2b58056c688`) is the sole proposed release candidate.
