# ARX TERMINAL — SPRINT 2 PACKAGE B
# CANONICAL MANIFEST V3 COMPATIBILITY REPORT

## 1. Governing Authority & Scope

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Operational Gate**: `G-S2-ACT`.
- **Subject**: Four-Security Historical SIP Ingestion Pilot Manifest V3 Integrity.
- **Manifest File**: `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json`.
- **Authorization State**: `PROPOSED_NOT_AUTHORIZED`.

---

## 2. Cryptographic Integrity & File Metrics

| Metric | Authoritative V3 Specification | Verified Candidate Value | Status |
| :--- | :--- | :--- | :--- |
| **File Path** | `.../PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` | Exact match | Verified |
| **Byte Length** | `9,972` bytes | `9,972` bytes | Exact Match |
| **SHA-256 Digest** | `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` | `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` | Exact Match |
| **Schema Standard** | Draft 2020-12 JSON Schema | Conforms | Verified |
| **Governance Status** | `PROPOSED_NOT_AUTHORIZED` | Hardcoded `is_authorized: false` | Enforced |

---

## 3. Candidate Cohort & Action Mapping

| Candidate Symbol | Durable Security ID | Listing Association ID | MIC | Class | Mandated Action | Action Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | `ARX_SEC_0B4C728FE0BE` | `ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17` | `XNAS` | None | `REPLACE` | Replaces 288 unprovenanced bars with qualified Alpaca SIP split-adjusted data. |
| **GOOG** | `ARX_SEC_A11A24D442E2` | `ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17` | `XNAS` | `CLASS_C` | `REPLACE` | Eliminates 287 bars derived from `MASSIVE_POLYGON` to enforce lineage purity. |
| **GOOGL** | `ARX_SEC_B7EF9ED7A35A` | `ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17` | `XNAS` | `CLASS_A` | `REPLACE` | Replaces 288 unprovenanced bars while preserving voting isolation from GOOG. |
| **ABBV** | `ARX_SEC_BFA51AC4C0B5` | `ASSOC_ARX_SEC_BFA51AC4C0B5_XNYS_ABBV_2026-10-19` | `XNYS` | None | `BACKFILL` | Empty-state initial backfill across identical 288-session historical window. |

---

## 4. Dynamic Calendar Session Boundary Verification

Rather than relying on a hardcoded session constant, the candidate adapter computes expected sessions dynamically via `get_expected_nyse_sessions`:

- **Requested Start Date**: `2025-08-27`
- **Requested End Date**: `2026-10-19`
- **Exchange Calendar Authority**: `exchange_calendars:XNYS` / `pandas_market_calendars`
- **Resolved Sessions Count**: **`288` trading sessions**
- **First 3 Sessions**: `2025-08-27`, `2025-08-28`, `2025-08-29`
- **Last 3 Sessions**: `2026-10-15`, `2026-10-16`, `2026-10-19`

---

## 5. History-Boundary Fail-Closed Protection

The pre-apply validation logic in `scripts/ingest_alpaca_historical.py` (lines 1173–1211) performs bidirectional boundary safety scanning:

1. **Before Start Date (`time < sym_start`)**:
   - Detects any legacy unprovenanced or foreign provider bars preceding `2025-08-27`.
   - Fails closed with `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE` if found.
2. **After End Date (`time > end_date`)**:
   - Detects any unprovenanced or non-Alpaca bars extending beyond `2026-10-19`.
   - Fails closed with `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE` if found.
3. **No Silent Inclusion**:
   - Out-of-window bars cannot silently blend into certified Alpaca SIP time series.
   - The write window is never automatically expanded without explicit manifest authorization.

---

## 6. Predecessor Supersession & Invalidation Ledger

1. **Predecessor 1 (`package_b_four_security_pilot_manifest.json`)**:
   - Digest: `65F5D9AB185056FD8C092AD1601A6DB195494063B5CCFCE562A11AAD8AA063CD`
   - Status: `INVALIDATED_SUPERSEDED` (contained minimal bindings without complete governance assertions).
2. **Predecessor 2 (`PILOT_PROPOSED_MANIFEST.json`)**:
   - Digest: `8923DAD988A10AE07A2B7BDA1044127130BF9C90539424D98F25676A9A59F07B`
   - Status: `SUPERSEDED / NOT FOR EXECUTION` (frozen predecessor with boundary defect `2025-09-15`; retained solely for regression and digest verification; adapter never executes or substitutes V2).

---

## 7. Ingestion Hold Record: AAPL Tail Out-of-Scope

```
AAPL_OUTSIDE_SCOPE_UNPROVENANCED_TAIL = BLOCKING_FOR_INGESTION
```

1. **Baseline Fact**: Production readback reports AAPL with 289 rows through `2026-10-20` (0 provenance rows). Canonical V3 manifest specifies window `2025-08-27` to `2026-10-19`.
2. **Fail-Closed Behavior**: The adapter pre-apply validator flags `outside_unprovenanced > 0` and halts with `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE`.
3. **No Automatic Expansion**: V3 manifest is NOT automatically expanded; the production row is NOT deleted or quarantined; V3 digest (`0F54C702...`) remains frozen.
4. **Gate Separation**: Governed by `G-S2-ACT`; `G-VCP-012 = ACTIVATION_DEFERRED` is independent.
5. **Release Classification**: Blocks production ingestion, NOT code release.
6. **Signature Status**: V3 manifest remains `PROPOSED_NOT_AUTHORIZED` and is NOT signed during PR preparation.
