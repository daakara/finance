# ARX RC5 — Engine Manifest v2.5.3 Succession Record

**Decision:** OWNER RATIFIED — scope-limited authority succession (ARX-FINANCIAL-AUTHORITY-039)  
**Owner authorization:** Explicit user instruction in ChatGPT, 2026-11-15: “RATIFY v2.5.3. Authorize the scoped manifest-succession governance PR under the exact supplied engine digests, with zero numerical changes and all existing operational holds preserved.”  
**Authorization token:** `ARX_RC5_MANIFEST_SUCCESSION_V253_AUTHORIZED`  
**Authority binding:** `SPRINT4_RC5_CONFLUENCE_PROVENANCE_AND_CAPS_RATIFICATION`  
**Code identity / deployed baseline:** `cc0c72c23cd874148a6db49730153d9b644617f0`  
**Successor:** `FROZEN_ENGINE_MANIFEST_V2_5_3.json`  
**Predecessor (retained, unmodified):** `FROZEN_ENGINE_MANIFEST_V2_5_2.json`

## Decision scope

Ratify existing production-code identity without changing any numerical calculations, engine source, thresholds, weights or scoring. Ratification does **not** authorize live trading, recurring SEC polling, autonomous scanning, experimental scoring activation, or operational hold removal.

| Engine | SHA-256 of ratified source |
|---|---|
| `analyst_dashboard/analyzers/confluence_engine.py` | `544851fadf3e8c5e0ae3a4390cb7c94ef223c1d4aacf9e7c735ee1eda8404ce2` |
| `analyst_dashboard/analyzers/optimal_execution.py` | `df2dc6050bf6fe91cae442e1872a3eccc40520b990f4efb4af63f79311a31dec` |
| `analyst_dashboard/analyzers/decision_hierarchy.py` | `b0d41de56c34a46000496e2cf80afdfd76a3ff19ffc18fbdd80f430050c375b3` |

## Existing behaviors covered

- Confluence: fundamental provenance gates; synthetic-factor suppression; hierarchical coverage caps 50/72/84/96; ETF exemption; liquidity hazard sizing suppression.
- Optimal execution: unclamped geometric reward-to-risk; raw price geometry; invalid or missing geometry fails closed.
- Decision hierarchy: verified evidence precedence; actionability requires at least three verified pillars; liquidity-risk sizing invalidation.

The above is a declaration of financial-model authority for the **already deployed** source hashes, not a numerical change to an analytical engine.

## Governance enforcement and provenance

RQ-10 must fail closed for unratified candidate manifests in production and verify all three engine hashes against this ratified manifest. RQ-13 continues to own runtime financial semantics, not ratification metadata. The experiment ledger must bind v2.5.3 manifest identity, authority binding and provenance commit. A mere edit to `isProductionRatified` in the candidate file must never authorize a release.

Retain `CANDIDATE_ENGINE_MANIFEST_SPRINT_4.json` as an explicitly **unratified historical reference**, excluded from production authority; no need to delete a provenance artifact used by historical tests.

## Preservation and release control

The following remain active: `HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`, `HOLD_CF_AUTO`. Cloudflare automatic production deployments remain disabled. Issue #165 supervised SEC qualification has independent restrictions. A governance PR passing CI is not authorization to bypass required manual deployment promotion.

## Rollback

Rollback restores the preceding release's governance binding and production code as a **coherent pair**. Do not relabel an unchanged Sprint 4 engine as v2.5.2, whose historical engine hashes are different. Preserve immutable predecessor records and immutable acceptance evidence.

## Evidence limitation

This ADR captures the owner's explicit decision provided in the conversation; it is not a cryptographically signed owner identity attestation. The GitHub PR review and release process must retain this approval record and enforce existing branch permissions.
