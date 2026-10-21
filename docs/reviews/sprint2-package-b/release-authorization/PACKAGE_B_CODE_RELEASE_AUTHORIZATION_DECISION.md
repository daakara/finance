# ARX TERMINAL — SPRINT 2 PACKAGE B
# CODE RELEASE AUTHORIZATION DECISION & HANDOFF

## 1. Governing Authority & Decision

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Sprint 2 Operational Gate**: `G-S2-ACT`.
- **Deferred Authority**: `G-VCP-012 = ACTIVATION_DEFERRED` (Independently governed; production historical ingestion does NOT require Gate 12).
- **Formal Decision**:
```
PACKAGE_B_CODE_RELEASE_READY_FOR_EXPLICIT_APPROVAL
```

---

## 2. Decision Rationale & Verified Evidence

1. **Source Authority Cleanly Reconstructed**:
   - Release branch `fix/sprint2-package-b-canonical-ingestion-release` established in dedicated clean worktree `C:\Users\akara\Documents\Projects\finance-sprint2-package-b-canonical-release`.
   - Exact base commit: `origin/main` (`f6c5755126059b2f227e28ce90ddaad43997884c`), matching live Railway deployment `4aa235d8-b571-4f67-bb41-2cdfa103fa00`.
   - Complete isolation: All 218 Smart Money Candidate 006 files remain safely preserved in `finance` worktree with zero bleed.
2. **Exact Release Candidate Artifacts & Hashes Verified**:
   - `scripts/ingest_alpaca_historical.py`: `7DA7CFC6BE8717C86EC15568F6D16E7173A44A7F33084653E889800F80C273C7` (1,579 lines)
   - `analyst_dashboard/data/guarded_recovery.py`: `8BE7D6A69A3D4316C3CF57793C8921C4E36E5E8A5BF6A78DA91281D6C4A6299D` (377 lines)
   - `scripts/guarded_market_recovery.py`: `73E940354777D51F5DBEEB51A86BFF441828BE9138AA3E26FFFCD8614398A7BF` (73 lines)
3. **Manifest Selection & Ambiguity Elimination**:
   - Canonical Pilot Manifest V3 (`PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json`, `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF`, 9,972 bytes): Status `PROPOSED_NOT_AUTHORIZED`. Reconciles 288 NYSE trading sessions (`2025-08-27` to `2026-10-19`) dynamically via exchange calendar. Sole candidate for eventual pilot ingestion. Unsigned during PR preparation.
   - Predecessor Manifest V2 (`PILOT_PROPOSED_MANIFEST.json`, `8923DAD988A10AE07A2B7BDA1044127130BF9C90539424D98F25676A9A59F07B`, 13,833 bytes): Formally marked `SUPERSEDED / NOT FOR EXECUTION`. Retained solely for regression digest verification.
4. **Code-Only Startup Safety Formally Established**:
   - Inspection of `railway.json`, `Procfile`, `Dockerfile`, and `api/main.py` confirms zero ingestion adapter calls, zero DB writes, and zero scanner activations on boot.
5. **Full Rehearsal & Regression Parity Verified**:
   - **253/253 automated tests pass** (100% pass rate, zero failures, zero regressions) across 94 release worktree rehearsal tests and 159 cross-boundary regression tests.
6. **Ingestion Holding Record (AAPL Out-of-Scope Production Row)**:
   - `AAPL_OUTSIDE_SCOPE_UNPROVENANCED_TAIL = BLOCKING_FOR_INGESTION`.
   - Production readback has AAPL with 289 rows through `2026-10-20`. V3 manifest ends on `2026-10-19`. Current adapter rejects this condition fail-closed (`BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE`) with zero database mutation. Manifest V3 is not expanded; production row is not altered; V3 digest is preserved. Blocks production ingestion, not code-only release.

---

## 3. Two-Stage Execution Ordering & Strict Boundaries

```
STAGE 1: CODE RELEASE (CURRENT HANDOFF)
Operator Reviews PR ---> Squash/Merge to main ---> Railway Deploys Container ---> Verify File Hashes
                                                                                       |
                                                                                       v
STAGE 2: PILOT INGESTION (SEPARATELY GOVERNED - CURRENTLY NOT AUTHORIZED)
Operator Issues Explicit Execution Approval ---> Run Pre-Apply Backup ---> Execute Bounded 4-Symbol Ingestion
```

### Absolute Stop Conditions Maintained
- **Zero Production Mutation**: No database writes or schema modifications executed.
- **Zero Push / Merge / Deploy**: Code remains staged in local worktree; no remote git actions taken.
- **Zero Ingestion Execution**: Historical Alpaca SIP ingestion script was not run against production.
- **Universe Publication & Scanner Activation**: Remain strictly `NOT_AUTHORIZED`.
- **Handoff**: Control returned to the authorized operator for explicit review and deployment decision.
