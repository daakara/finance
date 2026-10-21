# ARX TERMINAL — SPRINT 2 PACKAGE B
# FINAL PILOT MANIFEST RECONCILIATION & SCOPE BOUNDARY REPORT

## 1. Governing Authority & Scope

- **Authority**: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence.
- **Sprint 2 Operational Gate**: `G-S2-ACT`.
- **Subject**: Four-Security Historical SIP Ingestion Pilot Manifest Reconciliation.
- **Production Execution Status**: `NOT_AUTHORIZED`.

---

## 2. Production Store Baseline & Historical Boundary Analysis

Read-only inspection of the production database (`/root/.finance_market_store.db`, 5,427,200 bytes) establishes the following baseline across the four pilot securities:

| Symbol | Stored Candles | Earliest Date | Latest Date | Provenance Count | Provenance Provider | Current Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | 288 | `2025-08-27` | `2026-10-16` | 0 | None | `PRESENT_FRESH_UNPROVENANCED` |
| **GOOG** | 287 | `2025-08-27` | `2026-10-16` | 287 | `MASSIVE_POLYGON` | `PRESENT_STALE_PROV_CERT` |
| **GOOGL** | 288 | `2025-08-27` | `2026-10-16` | 0 | None | `PRESENT_FRESH_UNPROVENANCED` |
| **ABBV** | 0 | N/A | N/A | 0 | None | `MISSING` |

### Evaluation of Ingestion Boundary Policies

#### Policy A: Bounded Historical Interval (`2025-09-15` to `2026-10-19`)
- **Mechanism**: Restricts query window to 276 sessions starting `2025-09-15`.
- **Failure Mode**: The 12 trading sessions between `2025-08-27` and `2025-09-12` remain in `asset_ohlcv_daily`.
- **Safety Invariant Violation**: The ingestion adapter scans records outside the requested scope. For AAPL and GOOGL, `outside_unprovenanced` is non-zero (12 rows). For GOOG, `incompatible_outside` detects non-Alpaca provider (`MASSIVE_POLYGON`). The adapter **fails closed** with `BLOCKED_INCOMPATIBLE_OUTSIDE_SCOPE`.
- **Downstream Consumer Corruption**: `MarketDatabaseEngine.get_daily_candles` does not filter by provider. If unprovenanced or Polygon prefix rows remain before `2025-09-15`, consumers read a contaminated, blended time series.
- **Disposition**: **REJECTED (UNSAFE & DEFECTIVE)**.

#### Policy B: Expanded Explicitly Authorized Replacement (`2025-08-27` to `2026-10-19`)
- **Mechanism**: Expands query window to 288 sessions, matching the earliest stored session in production (`2025-08-27`).
- **Safety Invariant Compliance**: Bounded replacement covers all existing records. `outside_candles` is exactly 0. `outside_prov` is exactly 0. No unprovenanced or foreign records remain.
- **Downstream Lineage Purity**: 100% of stored candles for AAPL, GOOG, GOOGL, and ABBV become pure Alpaca SIP split-adjusted bars with 1-to-1 matching provenance records in `asset_ohlcv_provenance`.
- **Disposition**: **ACCEPTED (MANDATED SAFE POLICY)**.

---

## 3. Manifest Forensic Comparison & Supersession

| Manifest Property | Predecessor 1 (Workstream A) | Predecessor 2 (Workstream B) | Canonical Reconciled V3 |
| :--- | :--- | :--- | :--- |
| **Manifest ID** | `MANIFEST_PACKAGE_B_FOUR_SECURITY_PILOT_20261021_V3_RECONCILED` | `MANIFEST-PILOT-EXPANSION-150-B-001` | `MANIFEST-PILOT-EXPANSION-150-B-003-CANONICAL` |
| **File Path** | `evidence/.../package_b_four_security_pilot_manifest.json` | `docs/.../PILOT_PROPOSED_MANIFEST.json` | `docs/.../PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json` |
| **Byte Length** | `3,489` bytes | `13,833` bytes | `9,972` bytes |
| **SHA-256 Digest** | `65F5D9AB185056FD8C092AD1601A6DB195494063B5CCFCE562A11AAD8AA063CD` | `8923DAD988A10AE07A2B7BDA1044127130BF9C90539424D98F25676A9A59F07B` | `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF` |
| **Window Start Date** | `2025-08-27` | `2025-09-15` (DEFECTIVE) | `2025-08-27` (CORRECTED) |
| **Window End Date** | `2026-10-19` | `2026-10-19` | `2026-10-19` |
| **Expected NYSE Sessions** | 288 | 276 | 288 |
| **AAPL Action** | `REPLACE` | `BACKFILL` | `REPLACE` |
| **GOOG Action** | `REPLACE` | `BACKFILL` | `REPLACE` |
| **GOOGL Action** | `REPLACE` | `BACKFILL` | `REPLACE` |
| **ABBV Action** | `BACKFILL` | `BACKFILL` | `BACKFILL` |
| **Status** | `INVALIDATED_SUPERSEDED` | `INVALIDATED_SUPERSEDED` | **`CANONICAL_AWAITING_EXPLICIT_AUTHORIZATION`** |

Both predecessor candidates are formally superseded and invalidated.

---

## 4. Reconciled Pilot Manifest Authoritative Coordinates

- **File**: `docs/reviews/sprint2-package-b/pilot-authorization/PILOT_PROPOSED_MANIFEST_CANONICAL_V3.json`
- **Byte Length**: `9,972` bytes
- **SHA-256**: `0F54C702F0A4EDE41F4E54BB2635464919A84270BD9532A4830AD2A0A2B7F4BF`
- **Canonical Pilot Candidates**:
  1. `AAPL`: `ARX_SEC_0B4C728FE0BE` | `ASSOC_ARX_SEC_0B4C728FE0BE_XNAS_AAPL_2026-10-17` | `XNAS` | Action: `REPLACE`
  2. `GOOG`: `ARX_SEC_A11A24D442E2` | `ASSOC_ARX_SEC_A11A24D442E2_XNAS_GOOG_2026-10-17` | `XNAS` | Class: `CLASS_C` | Action: `REPLACE`
  3. `GOOGL`: `ARX_SEC_B7EF9ED7A35A` | `ASSOC_ARX_SEC_B7EF9ED7A35A_XNAS_GOOGL_2026-10-17` | `XNAS` | Class: `CLASS_A` | Action: `REPLACE`
  4. `ABBV`: `ARX_SEC_BFA51AC4C0B5` | `ASSOC_ARX_SEC_BFA51AC4C0B5_XNYS_ABBV_2026-10-19` | `XNYS` | Action: `BACKFILL`
