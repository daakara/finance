# ARX TERMINAL — CANONICAL RESEARCH OWNER DECISION REGISTER

**Document Identifier:** `ARX_OWNER_DECISION`  
**Execution Classification:** `ARX_ALCON_V5_CONTROLLED_PRODUCTION_RELEASE`  
**Governance Framework:** ARX Closure Model v2 + Governance Efficiency v1.1  
**Sole Approval Authority:** Research Owner exclusively (David Akara)  
**Owner Decision Authorization:** `APPROVED`  
**Selected Policy Direction:**  
- **Valuation Policy:** Option B (Scenario-Derived Terminal Net Debt from Ending Gross Debt minus Ending Cash)  
- **Integration Direction:** Path 1 (Preserve Production UX and Regularize Mainline Lineage)  
**Deployed Production SHA:** `64b5f892bf8b20f55b810599393bbbeb4d6aba91`  
**Active Production Cloudflare Deployment:** `1ce68307-6cec-40ad-95a1-44b062d4a4da`  
**Serving Domains:** `https://www.arxterminal.com`, `https://1ce68307.finance-xp8.pages.dev`  
**Preserved Railway Backend Baseline:** Deployment `059f9d79-6848-4095-aa61-5936ac3ee650` (Commit `a864562`, HTTP 200 OK `{"status":"online"}`)  
**Consolidated Verdict:** **`ARX_ALCON_V5_PRODUCTION_RELEASE_VERIFIED`**  
**Timestamp:** `2026-10-25T08:50:00Z`  
**Primary Supporting Evidence:**
- [`ARX_GATE_EVIDENCE.md`](file:///c:/Users/akara/Documents/Projects/finance-release-prod/ARX_GATE_EVIDENCE.md)
- [`ARX_EVIDENCE_MANIFEST.json`](file:///c:/Users/akara/Documents/Projects/finance-release-prod/ARX_EVIDENCE_MANIFEST.json)
- Live Browser Audit: [`screenshots/production_release_v5/production_acceptance_results.json`](file:///C:/Users/akara/.gemini/antigravity/brain/48da04d0-f5bf-4544-8766-b2bf86390360/screenshots/production_release_v5/production_acceptance_results.json)
- Canonical V5 Ledger: [`ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER_V5.json`](file:///c:/Users/akara/Documents/Projects/finance-release-prod/ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER_V5.json)
- Preserved V4 Ledger: [`ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json`](file:///c:/Users/akara/Documents/Projects/finance-release-prod/ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json)

---

## 1. Executive Summary & Authoritative Determination

1. **Owner Option B Ratified & Implemented:**
   - Under Owner Decision Option B, terminal net debt is derived strictly from each scenario's ending gross debt minus ending cash ($\text{terminal\_net\_debt} = \text{ending\_gross\_debt} - \text{ending\_cash}$).
   - Historical Model v4.0 is preserved as an immutable baseline (`ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER.json`, SHA-256: `d28270a18984de2b851ab5ce24dfcbe822d6bc55479024b708e418e21acdced4`) without retroactive relabeling.
   - Successor Model v5.0 is certified in `ARX_ALC_CANONICAL_SCENARIO_OUTPUT_LEDGER_V5.json` (SHA-256: `c4db315f037416a10c01ebf0eb1ccfda52a5ed66fa9396e6f2b8aa9d38256c38`).
   - Reference scenario targets: Base DCF Primary: **$95.00** (7.64% IRR, -$3,687.6M net cash); Base Multiple: **$93.20** (7.23% IRR); Conservative: **$76.36** (3.11% IRR); Optimistic: **$129.97** (14.50% IRR); Conservative Dynamic: **$76.70** (3.20% IRR); Optimistic Dynamic: **$129.06** (14.34% IRR).
   - Hurdle accumulation ceilings: $66.31 (8.0%), $64.82 (8.5%), $60.56 (10.0%).

2. **Security Identity Reconciled:**
   - Authoritative SEC Filer CIK established as `0001167379` from primary SEC EDGAR 20-F filings (`0001167379-26-000014`).
   - Fully reconciled across all ledgers, exports, catalog, maintenance registration, and frontend rendering.

3. **Lineage Reconciliation & Guarded Merge:**
   - Production UX lineage (`9b00142` & `8a15f1a`) regularized into mainline `main` via PR #86 (`ae28f73335d94c3d41072def6bdaaa6fdc6d91c6`).
   - Alcon PR #85 synchronized with `main`, verified across all 64 Python valuation/integrity tests and 36 frontend architectural suites, and squash-merged into `main` (`136ecb301bca1ba4acfa6b0540802ba7ef86e6e2`).
   - Canonical controls and Design Parity Gate v1.1 merged via PR #87 (`64b5f892bf8b20f55b810599393bbbeb4d6aba91`).

4. **Production Release Verification:**
   - Built in isolated release worktree `finance-release-prod` with 149/149 static routes prerendered.
   - Deployed exact-artifact to Cloudflare Pages production (`1ce68307-6cec-40ad-95a1-44b062d4a4da`).
   - Automatic deployments remain disabled (`production_deployments_enabled: false`).
   - Preserved Railway production backend `059f9d79-6848-4095-aa61-5936ac3ee650` untouched.
   - Real-browser acceptance on `https://www.arxterminal.com` confirmed 0 console errors, 0 hydration errors, four-pillar confluence intact, pre-flight readiness bar mounted beneath candlestick chart, and complete Alcon V5 dossier accessible.

**Final Machine-Readable Determination:** **`ARX_ALCON_V5_PRODUCTION_RELEASE_VERIFIED`**

---

## 2. Preserved Governance Invariants

```ini
MODEL_TUNING = FROZEN
LIVE_ORDER_EXECUTION = PROHIBITED
UNRELATED_BACKEND_DEPLOYMENT = PROHIBITED
HUMAN_RESEARCH = NOT_AUTHORIZED
HVA_ALGORITHM_ACCEPTANCE = NOT_AUTHORIZED
RESEARCH_EVIDENCE_MANIPULATION = PROHIBITED
UNQUALIFIED_PRODUCTION_CHANGE = PROHIBITED

ALCON_V5_NET_DEBT_POLICY = SCENARIO_DERIVED_FROM_CASH_FLOWS
ALCON_V4_HISTORICAL_BASELINE = IMMUTABLE_FROZEN
SEC_FILER_CIK_ALC = 0001167379
CLOUDFLARE_PAGES_AUTO_DEPLOY = DISABLED
PRODUCTION_DEPLOYMENT_METHOD = WRANGLER_STATIC_OUTPUT_EXACT_SHA
RAILWAY_BACKEND_DEPLOYMENT = 059f9d79-6848-4095-aa61-5936ac3ee650 (PRESERVED)
```