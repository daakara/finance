# ARX TERMINAL — RC5 MANIFEST DEPENDENCY INVENTORY

**Date:** 2026-10-23T11:21:00Z (Local: 2026-10-23T13:21:00+02:00)  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Authorization Token:** `ARX_RC5_MANIFEST_SUCCESSION_READ_ONLY_PREFLIGHT_AUTHORIZED`  
**Repository:** `daakara/finance`  

---

## 1. Manifest Discovery and Ecosystem Taxonomy

The repository contains two distinct categories of manifests: **Decision-Engine Freeze Authorities** (cryptographic source-code locking of algorithmic models) and **Operational / Application Manifests** (observation governance and client configuration).

```
daakara/finance Manifest Ecosystem
│
├── Decision-Engine Freeze Authorities
│   ├── FROZEN_ENGINE_MANIFEST_V2_5_1.json  [ACTIVE PRODUCTION AUTHORITY]
│   ├── FROZEN_ENGINE_MANIFEST_V2_5_0.json  [HISTORICAL ARCHIVAL - EPOCH 3]
│   └── FROZEN_ENGINE_MANIFEST_V2_4_0.json  [HISTORICAL ARCHIVAL - STRATEGY 2.4.0]
│
├── Observation Governance Authorities
│   ├── EPOCH_4_MANIFEST_V5.json            [ACTIVE PRODUCTION AUTHORITY - EPOCH 4]
│   ├── EPOCH_3_MANIFEST.json               [HISTORICAL ARCHIVAL - EPOCH 3]
│   ├── EPOCH_2_MANIFEST.json               [HISTORICAL ARCHIVAL - EPOCH 2]
│   └── EPOCH_1_MANIFEST.json               [HISTORICAL ARCHIVAL - EPOCH 1]
│
└── Application Runtime Manifests
    └── frontend/public/manifest.json       [PWA WEB APPLICATION METADATA]
```

---

## 2. Detailed Authority Specifications

### A. `FROZEN_ENGINE_MANIFEST_V2_5_1.json` (Active Engine Authority)
- **File Path:** `FROZEN_ENGINE_MANIFEST_V2_5_1.json`
- **File SHA-256:** `bb51df94f6d328a721089cea7d6004b4dcac1c2b1eeea55d763b5f8f6a58c5f0`
- **Version:** `2.5.1`
- **Frozen Timestamp:** `2026-10-16T06:00:00Z`
- **Provenance Commit:** `cd0922471767775636957df74406a2d5efb8f519`
- **Ratification Record:** Release Addendum 002 (Binding PO Rejection of EXC-002)
- **Supersedes:** `FROZEN_ENGINE_MANIFEST_V2_5_0.json`
- **Authority Scope:** `CURRENT_PRODUCTION_AUTHORITY`
- **Tracked Assets:**
  1. `analyst_dashboard/analyzers/optimal_execution.py` (`84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576`)
  2. `analyst_dashboard/analyzers/confluence_engine.py` (`147c010eb6068c3470fa3269f08610c0f436ea117b99b3b5217e7020b911e2b1`)
  3. `analyst_dashboard/analyzers/decision_hierarchy.py` (`823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3`)
- **Direct Downstream Enforcement Gates:**
  - `tests/test_f13a_epoch_compatibility.py` -> `test_frozen_engine_manifest_hashes_unmodified`
  - `tests/test_f13a_epoch_compatibility.py` -> `test_current_production_engine_compliance`
  - `tests/test_measurement_adversarial_suite.py` -> `test_production_engine_freeze_manifest_compliance`
  - GitHub Actions CI Job: `Backend Test Suite - Python 3.11` & `Python 3.12` -> `Run Tier 1 Core Invariant Gate`

### B. `EPOCH_4_MANIFEST_V5.json` (Active Observation Governance Authority)
- **File Path:** `EPOCH_4_MANIFEST_V5.json`
- **File SHA-256:** `ca7b860da5aa72225d5ee2d7212d4a7d4b22dfb4de299d8dfc3f08d451fd40c4`
- **Version:** `5.0.0` (Epoch 4)
- **Authority Scope:** `CURRENT_PRODUCTION_AUTHORITY`
- **Tracked Assets:** 10 executable governance files, including `api/routes/analytics.py`.
- **RC5 Impact:** PR #53 exposes `corporateEvent` in `/api/analytics/{ticker}`, changing `api/routes/analytics.py` SHA-256 from `e1642878...` to `361ae1da...`.
- **Direct Downstream Enforcement Gates:**
  - `tests/test_epoch_manifest_integrity.py`
  - `tests/test_live_dual_price_contract.py`

### C. Archival Manifests (`V2_4_0`, `V2_5_0`, `EPOCH_1-3`)
- **Authority Scope:** `HISTORICAL_AUDIT_RECORD`
- **Enforcement:** Enforced as immutable historical checkpoints by `test_frozen_engine_historical_manifest_integrity` and `test_epoch_1_manifest_hashes_unmodified`. These manifests are completely unaffected by RC5 and must never be edited.

### D. `manifest.json` (Web PWA Manifest)
- **Path:** `frontend/public/manifest.json`
- **Role:** Web application presentation metadata.
- **Enforcement:** Static export and Next.js frontend build. Zero relation to financial or algorithmic decision-engine freeze authorities.

---

## 3. Downstream Consumer and Gate Mapping

| Gate Identifier | Target Invariant | Source File / Test | Failure Impact |
|---|---|---|---|
| **Tier 1 Manifest Check** | Bitwise match of 3 engine SHAs | `test_f13a_epoch_compatibility.py` | CI Hard Block |
| **Tier 1 Production Compliance** | Direct SHA assertion on engines | `test_f13a_epoch_compatibility.py` | CI Hard Block |
| **Adversarial Freeze Compliance** | Production engine freeze integrity | `test_measurement_adversarial_suite.py` | CI Hard Block |
| **Matrix Fail-Fast Trigger** | Fast-fail on first test job error | `.github/workflows/ci.yml` | Cancels Python 3.12, Skips SSG |

---

## 4. Manifest Succession Blast Radius

Any succession of `FROZEN_ENGINE_MANIFEST_V2_5_1.json` will touch:
1. Creation of new manifest `FROZEN_ENGINE_MANIFEST_V2_5_2.json` (or `V2_6_0.json`).
2. Update of active manifest pointer and SHA assertions in `tests/test_f13a_epoch_compatibility.py`.
3. Update of active manifest SHA assertions in `tests/test_measurement_adversarial_suite.py`.
4. Update of observation governance manifest `EPOCH_4_MANIFEST_V5.json` (or succession to `EPOCH_4_MANIFEST_V6.json` / `EPOCH_5_MANIFEST_V1.json`) to reflect `api/routes/analytics.py`.

No modifications to GitHub Actions workflow files or repository rulesets are required.
