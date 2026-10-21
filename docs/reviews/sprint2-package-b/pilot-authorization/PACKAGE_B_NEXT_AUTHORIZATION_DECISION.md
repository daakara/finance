# ARX TERMINAL — SPRINT 2 PACKAGE B
# FORMAL RELEASE CANDIDATE & AUTHORIZATION DECISION

## 1. Authority & Governance

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Operational Authority Gate**: `G-S2-ACT`.
- **Deferred Authority**: `G-VCP-012` (separately governed).
- **Previous Disposition**: `PACKAGE_B_PILOT_PREFLIGHT_CONDITIONAL`.

---

## 2. Formal Decision

**Disposition**:
```
PACKAGE_B_CODE_RELEASE_CANDIDATE_READY_FOR_APPROVAL
```

### Rationale

1. **Workstream Convergence Complete**: Workstream A and Workstream B have been fully reconciled. The minimal necessary code changes have been synthesized into a unified, verified release candidate without resetting or overwriting worktree branches.
2. **Authoritative Release Candidate Established**:
   - `scripts/ingest_alpaca_historical.py` (`CCF4A96197B0AF689B119C99C8E6CE7598D74DC619835945D423657F9A1D137D`, 1,567 lines)
   - `analyst_dashboard/data/guarded_recovery.py` (`8BE7D6A69A3D4316C3CF57793C8921C4E36E5E8A5BF6A78DA91281D6C4A6299D`, 377 lines)
   - `scripts/guarded_market_recovery.py` (`73E940354777D51F5DBEEB51A86BFF441828BE9138AA3E26FFFCD8614398A7BF`, 73 lines)
3. **Historical Boundary Defect Resolved (Policy B)**: The safe historical window `[2025-08-27, 2026-10-19]` cleanly replaces all 288 sessions, eliminating the outside-scope unprovenanced/Polygon failure mode (`BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE`) and preventing downstream consumer contamination.
4. **Canonical Proposed Manifest Frozen**:
   - File: `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json`
   - Byte length: `9,972` bytes
   - SHA-256: `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF`
   - Both predecessor manifests (`b75af07e...` and `8923dad9...`) are superseded and invalidated.
5. **Full Test Parity Verified**: 228 automated tests pass (100% pass rate, 0 regressions).

---

## 3. Mandatory Two-Stage Release Sequence

Execution must proceed through two distinct, strictly separated gates:

```
STAGE 1: CODE RELEASE AUTHORIZATION & DEPLOYMENT
[Review Release Candidate] ---> [Deploy to Railway Web Service] ---> [Verify Container File Hashes]
                                                                                |
                                                                                v
STAGE 2: PILOT INGESTION AUTHORIZATION
[Operator G-S2-ACT Execution Approval] ---> [Quiescence & Backup] ---> [Run Pilot Ingestion]
```

1. **Stage 1 — Code Release Gate (PENDING OPERATOR APPROVAL)**:
   - Deploy the release candidate files to Railway production service `web`.
   - Verify on the container that `/app/scripts/ingest_alpaca_historical.py` matches SHA-256 `CCF4A961...` and `/app/analyst_dashboard/data/guarded_recovery.py` matches SHA-256 `8BE7D6A6...`.
2. **Stage 2 — Pilot Execution Gate (NOT AUTHORIZED)**:
   - Only after Stage 1 is verified may the operator grant separate authorization to execute the 4-security pilot against `PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json`.
   - Ingestion execution remains strictly forbidden until Stage 1 deployment verification passes.

---

## 4. Strict Prohibitions

- **No Production Mutation**: Zero database writes, modifications, or restorations authorized.
- **No Git Mutation**: No branches merged, pushed, or reset.
- **No Universe Publication**: Package C publication remains locked.
- **No Scanner Activation**: Live radar scanners remain offline.
- **No Gate 12 Release**: Gate 12 remains locked.
