# ARX TERMINAL — RC5 MANIFEST SUCCESSION NECESSITY MATRIX

**Date:** 2026-10-23T11:24:00Z (Local: 2026-10-23T13:24:00+02:00)  
**Governance Framework:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  
**Authorization Token:** `ARX_RC5_MANIFEST_SUCCESSION_READ_ONLY_PREFLIGHT_AUTHORIZED`  

---

## 1. Five Mandatory Criteria for Successor Justification

In accordance with ARX Closure Model v2, a manifest succession is justified if and only if all five criteria are satisfied:
1. **Genuine Protected Input Change:** A governed source file's byte contents have changed.
2. **Approved & Understood Delta:** The modification is directly traceable to an authorized specification.
3. **Semantic Invariant Preservation:** Mathematical, bounds, and backward-compatibility invariants hold.
4. **Reconcilable Hash Graph:** All downstream test assertions and dependent manifests can be updated atomically.
5. **Human Ratification Gate:** The Product Owner explicitly ratifies the succession.

---

## 2. Comprehensive Manifest Decision Matrix

| Candidate Manifest | Target Authority | Classification | Justification & Findings |
|---|---|---|---|
| `FROZEN_ENGINE_MANIFEST_V2_5_1.json` | Active Production Engine Authority | **SUCCESSION_REQUIRED** | `confluence_engine.py` was intentionally modified (+16 lines) to introduce RC5 earnings risk modifiers. SHA-256 shifted from `147c...` to `0f77...`. A successor manifest (`V2_5_2.json` or `V2_6_0.json`) is strictly required to restore CI without bypassing cryptographic gates. |
| `EPOCH_4_MANIFEST_V5.json` | Active Observation Governance Authority | **SUCCESSION_REQUIRED** | Tracks `api/routes/analytics.py` among 10 executable governance files. RC5 added `corporateEvent` to the analytics API route, changing SHA-256 from `e164...` to `361a...`. Successor `EPOCH_4_MANIFEST_V6.json` (or Epoch 5) is required. |
| `FROZEN_ENGINE_MANIFEST_V2_5_0.json` | Historical Archival (Epoch 3) | **SUCCESSION_NOT_REQUIRED** | Immutable historical milestone (`PENDING_EPOCH_3_FREEZE`). Preserved for backward-compatibility replay tests. Must not be modified. |
| `FROZEN_ENGINE_MANIFEST_V2_4_0.json` | Historical Archival (Strategy 2.4.0) | **SUCCESSION_NOT_REQUIRED** | Immutable historical milestone (commit `4e36862...`). Preserved for baseline compliance tests. Must not be modified. |
| `EPOCH_1_MANIFEST.json` | Historical Governance (Epoch 1) | **SUCCESSION_NOT_REQUIRED** | Immutable audit artifact for Epoch 1. Must remain bitwise unchanged. |
| `EPOCH_2_MANIFEST.json` | Historical Governance (Epoch 2) | **SUCCESSION_NOT_REQUIRED** | Immutable audit artifact for Epoch 2. Must remain bitwise unchanged. |
| `EPOCH_3_MANIFEST.json` | Historical Governance (Epoch 3) | **SUCCESSION_NOT_REQUIRED** | Immutable audit artifact for Epoch 3. Must remain bitwise unchanged. |
| `frontend/public/manifest.json` | Frontend Web App / PWA Metadata | **SUCCESSION_NOT_REQUIRED** | Static browser configuration asset (icons, display mode). Not subject to backend decision-engine governance. |

---

## 3. Evaluation of Criteria per Required Succession

### Manifest A: `FROZEN_ENGINE_MANIFEST_V2_5_1.json` -> Successor `V2_5_2.json`
- **Criterion 1 (Input Change):** SATISFIED (`confluence_engine.py` SHA-256 changed).
- **Criterion 2 (Approved Scope):** SATISFIED (matches Corporate Events RC5 specification).
- **Criterion 3 (Semantic Safety):** SATISFIED (clamped [0.0, 100.0], zero-drift on missing events).
- **Criterion 4 (Hash Reconciliation):** SATISFIED (clean mapping in `test_f13a` and `test_adversarial`).
- **Criterion 5 (Human Ratification):** PENDING (requires separate user authorization).

### Manifest B: `EPOCH_4_MANIFEST_V5.json` -> Successor `V5_1` / `V6`
- **Criterion 1 (Input Change):** SATISFIED (`api/routes/analytics.py` SHA-256 changed).
- **Criterion 2 (Approved Scope):** SATISFIED (additive `corporateEvent` response contract).
- **Criterion 3 (Semantic Safety):** SATISFIED (backward-compatible JSON payload).
- **Criterion 4 (Hash Reconciliation):** SATISFIED (deterministic SHA-256 update).
- **Criterion 5 (Human Ratification):** PENDING (requires separate user authorization).
