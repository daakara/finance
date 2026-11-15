# ARX RC5 Engine Manifest v2.5.3 — Owner-Ratified Succession Record

**Directive:** ARX-FINANCIAL-AUTHORITY-039  
**Date:** 2026-11-15  
**Status:** OWNER RATIFICATION COMMUNICATED; IMPLEMENTATION/RELEASE QUALIFICATION PENDING  
**Authorization token:** `ARX_RC5_MANIFEST_SUCCESSION_V253_AUTHORIZED`  
**Authority binding:** `SPRINT4_RC5_CONFLUENCE_PROVENANCE_AND_CAPS_RATIFICATION`  
**Baseline / audited production source commit:** `cc0c72c23cd874148a6db49730153d9b644617f0`  
**Predecessor:** `FROZEN_ENGINE_MANIFEST_V2_5_2.json` (immutable historical authority)  
**Successor:** `FROZEN_ENGINE_MANIFEST_V2_5_3.json`

## Recorded System Owner decision

The System Owner explicitly authorized the v2.5.3 succession on 2026-11-15 in the approval conversation, with the scope limited to (1) the exact three engine digests specified below, (2) a governance manifest, (3) RQ-10 ratification controls, and (4) experiment-ledger authority registration. **No numerical, analytical, execution, scanning, SEC-egress, or operational configuration changes were authorized.**

This document records that authorization; the token and text alone do not constitute cryptographic proof of owner identity. GitHub PR review/merge provenance and repository protections must provide independent access-control evidence. Do not present this text as a detached digital signature.

## Exact authorized engine digests (SHA-256)

| Engine | Source | SHA-256 |
| --- | --- | --- |
| Confluence | `analyst_dashboard/analyzers/confluence_engine.py` | `544851fadf3e8c5e0ae3a4390cb7c94ef223c1d4aacf9e7c735ee1eda8404ce2` |
| Optimal execution | `analyst_dashboard/analyzers/optimal_execution.py` | `df2dc6050bf6fe91cae442e1872a3eccc40520b990f4efb4af63f79311a31dec` |
| Decision hierarchy | `analyst_dashboard/analyzers/decision_hierarchy.py` | `b0d41de56c34a46000496e2cf80afdfd76a3ff19ffc18fbdd80f430050c375b3` |

Digests bind to source content at `cc0c72c23cd874148a6db49730153d9b644617f0`; the governance PR must not modify these three engine files.

## Model behavior being ratified (already deployed; not changed by this PR)

- **Confluence:** unverified/synthetic/failed/stale/unavailable fundamental filings receive zero fundamental conviction; verified-factor coverage caps of 50 / 72 / 84 / 96; ETFs receive neutral corporate governance score 50; liquidity/execution hazards suppress executable sizing.
- **Optimal execution:** report unclamped geometric reward/risk, fail closed for invalid entry/stop/take-profit geometry, and normalize OHLCV casing.
- **Decision hierarchy:** require at least three verified pillars before `READY_TO_BUY`; breached stops or severe hazards invalidate or restrict sizing.

These disclosures document already-deployed behavior and are not a claim of demonstrated empirical profitability.

## Ratification controls

- RQ-10 must validate the **frozen v2.5.3 production manifest**, exact authority binding, predecessor, authorized source commit, ADR record, ledger identity, and SHA-256 for **all three** engine files.
- Reject missing, malformed, candidate-only, unratified, mismatched, or unauthorized substitutions. A Boolean `isProductionRatified=true` alone is insufficient.
- Preserve historical v2.5.2 manifest unchanged. Candidate Sprint 4 manifest stays a historical *unratified* artifact; it must never be silently rewritten as ratified.
- Run negative/adversarial tests and release qualification at the exact PR head before merge. The correct number for any new test must not overwrite preexisting NEG-26/27 identity attacks.

## Unchanged operational restrictions

`HOLD_SEC_001`, `HOLD_SCAN_001`, `HOLD_EXEC_001`, `PHASE_3_HOLD`, `ARX_CONFLUENCE_INTEGRITY_HOLD`, `PHASE_A3_UX_HOLD`, and Cloudflare `HOLD_CF_AUTO` remain effective. No broker execution, natural scanning, recurring SEC ingestion, autonomous promotion, prospective epoch activation, or scoring feature expansion is authorized. Issue #165's bounded, separately authorized pilot is unchanged.

## Owner authorization scope / release record

**OWNER DECISION:** RATIFY v2.5.3 — ARX-FINANCIAL-AUTHORITY-039, governance-only, exact digests above, ledger registration and RQ-10 enforcement; zero numerical and operational alterations.  
**Release engineering:** implement through one scoped PR, obtain hosted CI and independent approval/provenance checks, then deploy via existing manual promotion controls.  
**Implementation status:** Pending qualification; this document is not evidence that the PR has merged or deployed.
