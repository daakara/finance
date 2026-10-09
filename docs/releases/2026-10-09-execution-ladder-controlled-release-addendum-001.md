# ARX Terminal — Execution Ladder Controlled Release Evidence Remediation Addendum 001

## 1. Traceable Provenance & Supersession Baseline
- **Original Evidence Package Path**: `evidence/release-preparation/2026-10-09-execution-ladder-controlled-release/`
- **Original Evidence Package Commit SHA-1**: `4b5206228b7f6762e10125b5b1e4943f4c2d5a86`
- **Superseding Evidence Package Path**: `evidence/release-preparation/2026-10-09-execution-ladder-controlled-release-supersession-001/`
- **Certified Implementation Commit SHA-1**: `74baf306cfe2b2b53da8269990e6f7363c2fe42d`
- **Origin Main Commit SHA-1**: `5a90b918b0975151b74e936b3fbfa536b575edd7`
- **Remediation Timestamp UTC**: `2026-10-09T20:36:49Z`

---

## 2. Provenance Gate Failure Analysis & Defect Remediation

The original release preparation evidence package committed at commit SHA-1 `4b5206228b7f6762e10125b5b1e4943f4c2d5a86` failed independent provenance certification due to three critical structural and semantic defects:

### A. Duplicate Evidence Record Identifiers
- **Original Defect**: The master index `evidence-index.yaml` in the original evidence package contained duplicate evidence identifiers:
  - Evidence record identifier `ARX-EL-RP-P2-I03-20261009T201500Z-49DE57` was assigned to both Record #22 (`baseline/manifest-failure-output.txt`) and Record #23 (`baseline/is-actionable-failure-output.txt`).
  - Evidence record identifier `ARX-EL-RP-P2-I04-20261009T201500Z-8CD3AA` was assigned to both Record #24 (`candidate/manifest-failure-output.txt`) and Record #25 (`candidate/is-actionable-failure-output.txt`).
- **Remediation**: In the superseding evidence package, every evidence artifact is assigned a globally unique evidence identifier matching regex `^[A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+$`:
  - Baseline manifest failure output: `P2-E03-A`
  - Baseline is_actionable failure output: `P2-E03-B`
  - Candidate manifest failure output: `P2-E04-A`
  - Candidate is_actionable failure output: `P2-E04-B`
  Total duplicate evidence identifier count across the superseding package is strictly 0.

### B. Self-Referential Evidence Index Sealing Defect
- **Original Defect**: The original master index `evidence-index.yaml` indexed itself at Record #6 with an internal artifact SHA-256 digest (`661e835d2ad793593f7cc1b592c14909cad5f698f18f50cc4e013f739863d8df`). Because mutating file content alters its cryptographic hash, the committed file's actual artifact SHA-256 digest (`8810e49f47a7b3899c6a2316325fea5eb7507623bf4dcdce7fe91382831bccf2`) diverged, causing an unsealable digest failure.
- **Remediation**: The superseding index `evidence-index.yaml` excludes itself entirely from internal indexing. Sealing is externalized to an immutable companion artifact `evidence-seal.yaml`, which records the exact artifact SHA-256 digest of the finalized index bytes without self-hashing.

### C. Reconciled Production Denominators
- **Original Claim**: Original release notes asserted that 5 persisted database rows across releases evaluated to 4 unique canonical plans (`CUMULATIVE_DENOMINATOR = 4`).
- **Authoritative Database Query**: A direct, read-only query of the live production Railway SQLite database (`/root/analyst_dashboard/data/governance.db`) completed at `DENOMINATOR_AS_OF_UTC = 2026-10-09T20:36:49Z` revealed 9 physical rows in `execution_ladder_prospective_plans`.
- **Reconstructed Denominators**:
  - `PHYSICAL_ROW_COUNT = 9`
  - `HISTORICAL_01683A3_PHYSICAL_ROW_COUNT = 4`
  - `HISTORICAL_01683A3_CANONICAL_IDENTITY_COUNT = 3`
  - `CURRENT_5A90B91_PHYSICAL_ROW_COUNT = 5`
  - `CURRENT_5A90B91_CANONICAL_IDENTITY_COUNT = 3`
  - `CUMULATIVE_CANONICAL_IDENTITY_COUNT = 6`
  - `DENOMINATOR_AS_OF_UTC = 2026-10-09T20:36:49Z`

---

## 3. Mandatory Denominator Definitions & Boundary Discipline

To prevent semantic conflation between historical baselines and cumulative runtime capture:
- **`HISTORICAL_DENOMINATOR_DEFINITION`**: `canonical-identity count for the frozen historical cohort` (persisted under release commit SHA-1 `01683a39a19f3f74720f798459cec717698e2ab2`). Reconfirmed as 3 distinct canonical plans derived from 4 physical rows.
- **`CUMULATIVE_DENOMINATOR_DEFINITION`**: `canonical-identity count for the defined production population at DENOMINATOR_AS_OF_UTC` (`2026-10-09T20:36:49Z`). Reconstructed as 6 distinct canonical plans derived from 9 physical rows across release commit SHA-1 `01683a39a19f3f74720f798459cec717698e2ab2` and release commit SHA-1 `5a90b918b0975151b74e936b3fbfa536b575edd7`.

---

## 4. Production Cloud Provider Baseline Authority

### Railway Deployment Authority
Authoritative provider records directly queried via the authenticated Railway CLI establish:
- **Project Identity**: `tranquil-radiance` (Project ID: `a339a92a-7236-4ac0-9a3a-d7ad440f9690`)
- **Environment Identity**: `production` (Environment ID: `11cad5e1-358e-430e-84fc-189953cff48a`)
- **Superseded Deployment**:
  - `RAILWAY_SUPERSEDED_DEPLOYMENT_ID = 0685134c-134c-4cce-8e0a-850299e18c34`
  - Deployed commit SHA-1 associated with superseded deployment: `d97801e783620294454d1989164c907534ed4358`
  - Status: `REMOVED`
- **Current Active Production Deployment**:
  - `RAILWAY_CURRENT_DEPLOYMENT_ID = 66617940-e0a7-45a9-8394-fda0529367cb`
  - Deployed commit SHA-1 associated with current deployment: `5a90b918b0975151b74e936b3fbfa536b575edd7` (`origin/main`)
  - Status: `SUCCESS`
  - Deployed commit message: `docs(release): record price authority semantic remediation release`
  - Provider query timestamp: `2026-10-09T20:36:49Z`

### Cloudflare Deployment Authority
- **`CLOUDFLARE_PRODUCTION_BASELINE_STATUS`**: `BLOCKED`
- **Block Reason**: Cloudflare API credentials and Wrangler CLI are unavailable in the audit environment. Live dashboard verification cannot be conducted directly.

---

## 5. Exception Governance & Release Guardrails
- **`EXC_001_OWNER_ACCEPTANCE_STATUS`**: `PENDING`
- **`EXC_002_OWNER_ACCEPTANCE_STATUS`**: `PENDING`
Both pre-existing baseline exceptions remain visible and documented. Neither exception is closed or inferred as approved without explicit, attributable Product Owner sign-off.

---

## 6. VCP Epoch 002 Boundary Preservation
- **`VCP_EPOCH_002_STATUS`**: `PAUSED_PENDING_EXTERNAL_CUSTODIAN`
- **`GATE_12_STATUS`**: `NOT_SATISFIED`
- **`PRIVATE_CASE_SELECTION_STATUS`**: `BLOCKED`
- **`EXTERNAL_ADJUDICATION_STATUS`**: `BLOCKED`
- **`HOLDOUT_COMMITMENT_STATUS`**: `NOT_CREATED`
- **`MODEL_TUNING_STATUS`**: `FROZEN`
Custodian revocation record `CUSTODIAN_REVOCATION_001.json` is validated and all 81 precommitment verification tests pass. No holdout authorization is granted or implied.
