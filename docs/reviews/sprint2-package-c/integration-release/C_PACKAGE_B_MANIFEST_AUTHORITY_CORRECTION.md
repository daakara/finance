# ARX TERMINAL — SPRINT 2 TRACK C
# PACKAGE B MANIFEST AUTHORITY CORRECTION & SINGLE-PRODUCER CONTRACT

## 1. Executive Summary & Review Coordinates

- **Governing Framework:** ARX Closure Model v2 (*Decision-First, Minimum Sufficient Evidence*).
- **Track:** Sprint 2 Track C (Universe Publication Architecture & Consumer Integration).
- **Worktree:** `C:\Users\akara\Documents\Projects\finance-sprint-2-package-c` (`feat/sprint-2-package-c`).
- **Canonical Producer Source:** `C:\Users\akara\Documents\Projects\finance-sprint2-package-b-canonical-release` (`fix/sprint2-package-b-canonical-ingestion-release`, commit `5e86b5b5c22dcb28cfa350e25562eeb04eac55ee`).
- **Prior Review Disposition:** `PACKAGE_C_V3_INTEGRATION_HOLD`.
- **Target Disposition:** `PACKAGE_C_RELEASE_CANDIDATE_READY_FOR_EXPLICIT_REVIEW`.

This document records the forensic reconciliation, invalidation of dual predecessor manifests, and establishment of strict, single active producer authority in Package C.

---

## 2. Forensic Discovery of Predecessor Manifests

During integration testing, Package C previously exhibited an ambiguous dual-manifest condition where two disparate manifests claimed Package B authority. Under the ARX Closure Model v2, dual authority is an invariant violation. A forensic inspection of the competing workstreams established the lineage:

| Manifest Identifier | Originating Workstream | File Path | Byte Size | SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Predecessor 1** | Workstream A (`finance-sprint2-historical-sip-release`) | `evidence/coverage_expansion/package_b_four_security_pilot_manifest.json` | 3,489 bytes | `65F5D9AB185056FD8C092AD1601A6DB195494063B5CCFCE562A11AAD8AA063CD` | **FORMALLY SUPERSEDED & INVALIDATED** |
| **Predecessor 2** | Workstream B (`finance`) | `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST.json` | 13,833 bytes | `8923DAD988A10AE07A2B7BDA1044127130BF9C90539424D98F25676A9A59F07B` | **FORMALLY SUPERSEDED & INVALIDATED** (Boundary defect: 2025-09-15 start date) |
| **Canonical V3** | Package B Canonical Release | `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` | 9,972 bytes | `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` | **AUTHORITATIVE SOLE ACTIVE PRODUCER CONTRACT** |

### Root Cause Analysis of Predecessors

1. **Predecessor 1 (`65F5D9AB...`)**:
   - Synthesized during early Package C offline mocking as a provisional contract shim.
   - Omitted full operational sections: missing `target_store`, `recovery_policy`, explicit retry matrices, and provider configuration fields.
   - Declared provisional manifest ID `MANIFEST_PACKAGE_B_FOUR_SECURITY_PILOT_20261021_V3_RECONCILED`.
   - Never accepted by Package B canonical release lead.
2. **Predecessor 2 (`8923DAD9...`)**:
   - Developed in Workstream B with start date `2025-09-15`.
   - Formally invalidated by Package B Option B Full-History ruling because a start date of `2025-09-15` left 12 unprovenanced sessions (2025-08-27 through 2025-09-12) unmanaged in the production SQLite store, creating an unacceptable provenance boundary contamination defect.

---

## 3. Authoritative Canonical V3 Manifest Specification

Package B's official code release artifact establishes the exact producer contract:

- **Source File Path:** `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json`
- **Canonical Manifest ID:** `MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL`
- **Exact Byte Size:** `9,972 bytes`
- **Bit-for-Bit SHA-256 Digest:** `0f54c702f0a4ede41f4e54bb2635464919a84270bd9532a4830ad2a0a2b7f4bf`
- **Acquisition Window:** `2025-08-27` to `2026-10-19` (419 calendar days, 288 expected NYSE trading sessions, 280 minimum acceptable bars).
- **Feed Specification:** Alpaca Market Data API v2, `tape_feed="sip"` (CTA/UTP Consolidated Tape), `adjustment_convention="split"`.
- **Governance Status:**
  - `authorization_state="PROPOSED_NOT_AUTHORIZED"`
  - `is_authorized=false`
  - `authorized_by=null`
  - `operational_authority="G-S2-ACT"`
  - `deferred_gate_reference="G-VCP-012"`
- **Target Ingestion Cohort (4 Pilot Candidates):**
  1. `AAPL` (`ARX_SEC_0B4C728FE0BE`, `ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17`, `XNAS`, Action: `REPLACE`)
  2. `GOOG` (`ARX_SEC_A11A24D442E2`, `ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17`, `XNAS`, Class C, Action: `REPLACE`)
  3. `GOOGL` (`ARX_SEC_B7EF9ED7A35A`, `ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17`, `XNAS`, Class A, Action: `REPLACE`)
  4. `ABBV` (`ARX_SEC_BFA51AC4C0B5`, `ASSOC_ARX_SEC_BFA51AC4C0B5_XNYS_ABBV_2026-10-19`, `XNYS`, Action: `BACKFILL`)

---

## 4. Remediation Actions Executed in Package C

1. **Elimination of Superfluous / Superseded Manifest Files:**
   - Deleted `evidence/coverage_expansion/package_b_four_security_pilot_manifest.json` from Package C.
   - Copied the bit-for-bit canonical manifest to `tests/fixtures/package_b_canonical_v3/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` (9,972 bytes, SHA-256 `0F54C702...`) as a clearly labeled, immutable test fixture.
2. **Elimination of `allow_reconciled_digest` Bypass:**
   - Removed all fallback arguments and soft warning flags from `PackageBV3Consumer.load_manifest`.
   - Hardcoded `SUPERSEDED_PREDECESSOR_DIGESTS = {"65f5d9ab...", "8923dad9..."}`. Any call attempting to load or request a superseded predecessor immediately triggers `PolicyViolationError`.
3. **Strict Digest Matching:**
   - Package C consumer requires computed SHA-256 to equal `CANONICAL_PACKAGE_B_V3_DIGEST` (`0f54c702...`). Any byte modification, tampering, or foreign manifest raises `PolicyViolationError` ("Package B V3 manifest digest mismatch").
4. **Single Active Producer Authority:**
   - Package C code recognizes exactly one active manifest: `MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL`.
