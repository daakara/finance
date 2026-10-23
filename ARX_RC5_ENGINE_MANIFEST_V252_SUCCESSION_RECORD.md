# ARX TERMINAL — ENGINE MANIFEST V2.5.2 SUCCESSION RECORD

**Date:** 2026-10-23T11:47:00Z (Local: 2026-10-23T13:47:00+02:00)  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Authorization Token:** `ARX_RC5_MANIFEST_SUCCESSION_MUTATION_AUTHORIZED`  
**Target Manifest File:** `FROZEN_ENGINE_MANIFEST_V2_5_2.json`  
**Release Authority Binding:** `SPRINT3_RC5_CORPORATE_EVENTS_RATIFICATION`  
**Ratified Record:** `ARX_RC5_ENGINE_MANIFEST_V252_SUCCESSION_RECORD.md`  

---

## 1. Lineage & Chain-of-Custody

```
FROZEN_ENGINE_MANIFEST_V2_4_0.json (Strategy 2.4.0 Baseline)
  │  provenanceCommit: 4e3686296aad24e2210ef580bbc9116054d84fd1
  ▼
FROZEN_ENGINE_MANIFEST_V2_5_0.json (Pending Epoch 3 Freeze)
  │  provenanceCommit: PENDING_EPOCH_3_FREEZE
  ▼
FROZEN_ENGINE_MANIFEST_V2_5_1.json (Release Addendum 002)
  │  provenanceCommit: cd0922471767775636957df74406a2d5efb8f519
  │  releaseAuthorityBinding: RELEASE_ADDENDUM_002
  ▼
FROZEN_ENGINE_MANIFEST_V2_5_2.json (Sprint 3 Corporate Events RC5)
     provenanceCommit: 7980d3a6e5dde0fe172d215c729ec213b9a164b1
     releaseAuthorityBinding: SPRINT3_RC5_CORPORATE_EVENTS_RATIFICATION
     supersedesManifest: FROZEN_ENGINE_MANIFEST_V2_5_1.json
```

---

## 2. Certified Engine Hashes

| Governed Component | File Path | Certified SHA-256 | Provenance / Authority |
|---|---|---|---|
| **Optimal Execution** | `analyst_dashboard/analyzers/optimal_execution.py` | `84d1a8b855a5368466b53105fdd58df15c3d60aaab77024286f3fcaaaad95576` | Ratified Release Addendum 002 Section 7 (PO Rejection of EXC-002) |
| **Confluence Engine** | `analyst_dashboard/analyzers/confluence_engine.py` | `0f77d3b0b15f5d60a6de52ec1a0cd6b385ce1615865839d684d53aa655dadc5a` | Sprint 3 Corporate Events RC5 Runway Risk Integration |
| **Decision Hierarchy** | `analyst_dashboard/analyzers/decision_hierarchy.py` | `823f37f81c7f601885213d968934c1118285fbf6c875b2993e091984e9b8ebe3` | Synthesis E Wave 3 Decision Integrity (PRD Addendum 001) |

---

## 3. Scope of Ratified Change

The only algorithmic modification certified by this successor is the addition of the Catalyst Runway Risk Adjustment in `confluence_engine.py`:
1. Deducts `25.0` points for acute earnings proximity ($\le 1$ day).
2. Deducts `8.0` points for cautionary earnings proximity ($2 \le \text{days} \le 7$).
3. Bypasses adjustment (0.0 score change) for completed/elapsed events or when corporate event data is absent.
4. Clamps final score strictly within $[0.0, 100.0]$.
