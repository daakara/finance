# ARX Terminal — Production Release Note
## Execution Ladder V5 Controlled Production Release

**RELEASE_SHA** = ad3169d755123fc7bb2fc520ae7f0f76851918a4  
**RELEASE_DATE** = 2026-10-10  
**DEPLOYED_GIT_HEAD** = a0d8f53e1bb7ac42b7ebbba9ec0b27312b07d94f  
**CERTIFIED_IMPLEMENTATION_COMMIT_SHA1** = ad3169d755123fc7bb2fc520ae7f0f76851918a4  
**FINAL_RELEASE_EVIDENCE_COMMIT_SHA1** = a0d8f53e1bb7ac42b7ebbba9ec0b27312b07d94f  
**V5_SOURCE_SNAPSHOT_COMMIT_SHA1** = cbbce7ea08bb259a7fd5207b54cab5ae78bf0c3a  
**PRODUCTION_DEPLOYMENT_TIMESTAMP_UTC** = 2026-10-10T05:02:43Z  

---

### 1. Executive Summary & Release Scope

This canonical production release note records the controlled release of the ARX Terminal Execution Ladder V5 subsystem, including:
1. **Execution Ladder Contract & Identity Remediation**: Enforces 11-field mathematical equivalence and atomic admission in execution ladder capture (`37a665c`, `74baf30`).
2. **Resolution of Product-Owner Rejected Exceptions**: Full root-cause resolution for EXC-001 (manifest authority conflation) and EXC-002 (`is_actionable` contract truncation) in commit `cd09224`.
3. **Epoch 4 V5 Non-Circular Manifest Succession**: Activation of `EPOCH_4_MANIFEST_V5.json` (`ad3169d`) superseding V4, bound to frozen source snapshot `cbbce7e`.
4. **Frozen Evidence Restoration**: Restoration of `evidence/release-preparation/2026-10-09-execution-ladder-controlled-release` matching historical tree `1cee3c9033e904dc0e80eb301eefc45562f2b00f`.
5. **Multi-Platform Deployment Integration**: Verified deployment to Railway production backend and Cloudflare Pages edge delivery.

---

### 2. Immutable Verification & Provenance Anchors

- **Functional Authority**: `ad3169d755123fc7bb2fc520ae7f0f76851918a4`
- **Runtime Deployed Head**: `a0d8f53e1bb7ac42b7ebbba9ec0b27312b07d94f`
- **Ancestry Relationship**: `git merge-base --is-ancestor ad3169d a0d8f53` = `YES`
- **Post-Certification Implementation Diff**: `0` files (pure docs & evidence closure)
- **V5 Activation Binding Artifact**: `evidence/release-closure/2026-10-10-execution-ladder-v5-final/v5-activation-binding.json`
  - SHA-256: `aaa741cc785fdc8145556b34b14d153cbcd9ed3d58671f15b711e2efb12237d3`
  - Git Blob SHA-1: `fbb19da12efc9c2f2ee5438d9954b88a1932fb04`

---

### 3. Production Deployment Status

- **Railway Backend**:
  - Service: `web` (`tranquil-radiance`, environment `production`)
  - Deployment ID: `2a3df87b-95ef-436f-bfee-65e39e10ba45`
  - Status: `SUCCESS`
  - Live Endpoint: `https://web-production-470560.up.railway.app/health` (`200 OK`)
  - Runtime Logging Attestation: `backend_release_sha="a0d8f53e1bb7ac42b7ebbba9ec0b27312b07d94f"`
- **Cloudflare Pages Frontend**:
  - Project: `arx-frontend`
  - Domains: `https://www.arxterminal.com`, `https://finance-xp8.pages.dev`
  - Status: `SUCCESS` (Verified ETag change and static chunk delivery)
- **GitHub Actions Release Workflow**:
  - Run ID: `38026159819`
  - Status: Evaluated. Downstream deployment job skipped due to pre-existing flake8 lint errors in research scripts and pre-existing epoch compatibility tests. Direct continuous integration deployment on Railway and Cloudflare Pages succeeded independently.

---

### 4. Governed Release Verification Results

- **Release Test Suite**: 520 collected / 520 passed / 0 failed in 32.61s.
- **Dual-Price Contract**: `PASS`
- **Execution Ladder Identity**: `PASS`
- **11-Field Equivalence**: `PASS`
- **Atomic Admission**: `PASS`
- **Epoch 4 V5 Routing**: `PASS`
- **Fail-Closed Behavior**: `PASS`
- **EXC-001 Remediation**: `PASS`
- **EXC-002 Remediation**: `PASS`

---

### 5. Known Limitations & Preserved Governance Guardrails

- **Zero Unauthorized Domain Mutation**: Scanner methodology, target/entry formulas, risk formulas, and model parameters are unchanged.
- **Model Tuning Status**: `FROZEN`
- **VCP Epoch 002 Status**: `PAUSED_PENDING_EXTERNAL_CUSTODIAN`
- **Gate 12 Status**: `NOT_SATISFIED`
- **Private Case Selection**: `BLOCKED`
- **External Adjudication**: `BLOCKED`
- **Radar Sprint 3 Authorization**: `NO` (Deferred to separate engineering entry gate)
