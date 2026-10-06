# ARX TERMINAL — CANONICAL SERVER SECURITY MASTER
## Implementation & Contract Verification Report

**Gate Reference:** `PASS_ARX_CANONICAL_SECURITY_MASTER_IMPLEMENTATION`
**Target Branch:** `arx/security-master`
**Worktree Directory:** `C:/Users/akara/Documents/Projects/finance-security-master`
**Canonical Classification Authority:** `ARX_SERVER_SECURITY_MASTER`
**Identity & Listing Authority:** `ALPACA_ASSET_DIRECTORY`
**Security Subtype Authority:** `OPENFIGI_V3_MAPPING`
**Execution Routing Invariant:** `INV-SECMASTER-01..20`
**Date:** 2026-10-06

---

## 1. Executive Summary

This report documents the completed implementation and verification of the server-owned **Canonical Security Master** for ARX Terminal. Prior to this release, asset classification and execution eligibility exhibited structural ambiguity across uncatalogued equities (such as PLSE) and negative non-common-stock subtypes (such as warrants, preferred shares, REITs, and ADRs). The architecture also permitted client-side heuristics and static catalogs (`masterCatalog.ts`, `SHARED_WATCHLIST_ITEMS`) to act as de facto classification authorities.

Under this implementation:
1. **Classification Authority Centralization:** The server subsystem `analyst_dashboard.security_master` is established as the sole authoritative classification source (`ARX_SERVER_SECURITY_MASTER`).
2. **Provider Authority Decoupling:** Alpaca provides identity, exchange listing, and active status (`ALPACA_ASSET_DIRECTORY`). OpenFIGI provides structural security subtyping (`OPENFIGI_V3_MAPPING`). Broad asset classes from Alpaca are explicitly barred from certifying equity subtypes (`ALPACA_BROAD_CLASS_IS_SUBTYPE_AUTHORITY = NO`).
3. **Fail-Closed Execution Eligibility:** `ExecutionEligibility.STOCK_EXECUTION` is granted strictly to instruments certified as `VERIFIED` + `COMMON_STOCK` + `ACTIVE`. Specialized equities (ADRs, REITs, Preferreds, Warrants, Units, Rights) and conflicted or unverified assets fail closed to `FAIL_CLOSED`.
4. **Global Quota Coordination & Firewall Isolation:** The subsystem shares the global rolling rate limiter (`20 requests / 60 seconds`) across all ARX consumers via SQLite atomic reservations (`data/operational/openfigi_operational.db`) without mutating ETF v2 business logic. Security Master persistence (`data/operational/security_master.db`) is firewalled from canonical research databases and rate limiter operational stores.
5. **Client Execution Surface Migration:** The frontend execution router in `frontend/app/page.tsx` now consumes `instrument.execution_eligibility` from the server response (`GET /api/v1/analytics/{symbol}` envelope and `GET /api/v1/market/instruments/{symbol}`). Static frontend catalogs and watchlists have been formally demoted to presentation/enrichment only (`FRONTEND_CATALOG_AUTHORITY = NONE`).

---

## 2. Invariant Compliance Matrix (INV-SECMASTER-01..20)

| Invariant | Description | Enforcement Point | Verification Status |
|:---|:---|:---|:---:|
| **INV-SECMASTER-01** | Server Ownership: Classification and execution eligibility are strictly server-owned (`ARX_SERVER_SECURITY_MASTER`). | `analyst_dashboard/security_master/models.py` | **PASS** |
| **INV-SECMASTER-02** | Frontend Catalog Demotion: Frontend catalogs (`masterCatalog`, `SHARED_WATCHLIST_ITEMS`) have zero classification authority. | `frontend/lib/masterCatalog.ts`, `constants.ts` | **PASS** |
| **INV-SECMASTER-03** | Decoupled Eligibility: Execution eligibility is decoupled from identity and subtype. | `analyst_dashboard/security_master/eligibility.py` | **PASS** |
| **INV-SECMASTER-04** | Dual Authority Resolution: Identity = Alpaca; Subtype = OpenFIGI. | `analyst_dashboard/security_master/normalization.py` | **PASS** |
| **INV-SECMASTER-05** | UNKNOWN Fail-Closed: Unknown/unlisted symbols fail closed to non-actionable state. | `analyst_dashboard/security_master/normalization.py` | **PASS** |
| **INV-SECMASTER-06** | UNVERIFIED Fail-Closed: Missing corroborating subtype fails closed. | `analyst_dashboard/security_master/normalization.py` | **PASS** |
| **INV-SECMASTER-07** | CONFLICTED Fail-Closed: Material provider disagreement fails closed. | `analyst_dashboard/security_master/normalization.py` | **PASS** |
| **INV-SECMASTER-08** | ETP Isolation: Verified ETPs route strictly to ETF execution surface (`EtfCostOfOwnershipCard`). | `frontend/app/page.tsx`, `assetTypeUtils.ts` | **PASS** |
| **INV-SECMASTER-09** | Crypto Isolation: Digital assets route strictly to crypto execution surface. | `frontend/app/page.tsx`, `assetTypeUtils.ts` | **PASS** |
| **INV-SECMASTER-10** | Specialized Equity Fail-Closed: ADR, REIT, Preferred, Warrant, Unit, Right fail closed. | `analyst_dashboard/security_master/eligibility.py` | **PASS** |
| **INV-SECMASTER-11** | Common Stock Corroboration: `STOCK_EXECUTION` requires verified `COMMON_STOCK`. | `analyst_dashboard/security_master/eligibility.py` | **PASS** |
| **INV-SECMASTER-12** | Inactive/Delisted Fail-Closed: Inactive/delisted listings fail closed immediately. | `analyst_dashboard/security_master/eligibility.py` | **PASS** |
| **INV-SECMASTER-13** | Dedicated Market Endpoint: `GET /api/v1/market/instruments/{symbol}` exposed. | `api/routes/market.py` | **PASS** |
| **INV-SECMASTER-14** | Zero Symbol Exceptions: Zero ticker-specific whitelist or hardcoded branching. | Audited in `normalization.py`, `eligibility.py` | **PASS** |
| **INV-SECMASTER-15** | Persistence Freshness: Hybrid SQLite + thread-safe LRU cache with strict TTL enforcement. | `analyst_dashboard/security_master/persistence.py` | **PASS** |
| **INV-SECMASTER-16** | Provenance Sanitization: Zero API keys, bearer tokens, or raw secrets in stored provenance. | `tests/test_security_master_contract.py` | **PASS** |
| **INV-SECMASTER-17** | Global Quota Coordination: Shares OpenFIGI 20 req/60s reservation window via SQLite. | `analyst_dashboard/security_master/openfigi_adapter.py` | **PASS** |
| **INV-SECMASTER-18** | Analytics Independence: Quantitative models and scoring are 100% independent of security master envelope. | `api/routes/analytics.py` | **PASS** |
| **INV-SECMASTER-19** | Pre-Load Safe Boundary: Pre-load / in-flight states render non-actionable pending indicators. | `frontend/app/page.tsx` | **PASS** |
| **INV-SECMASTER-20** | Storage Firewall: Complete firewall isolation from canonical datasets and rate-limiter databases. | `analyst_dashboard/security_master/config.py` | **PASS** |

---

## 3. Subsystem Architecture & Implementation Topology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ARX Frontend (Next.js)                         │
│                                                                        │
│  page.tsx (planSlot & Tab 1 Execution)                                 │
│  ├── data?.instrument?.execution_eligibility === 'STOCK_EXECUTION'     │
│  │   └── OptimalEntryExitCard (Long-Only Swing Corridor)              │
│  ├── data?.instrument?.execution_eligibility === 'ETF_EXECUTION'       │
│  │   └── EtfCostOfOwnershipCard (ETF Fee / Spread Surface)             │
│  └── FAIL_CLOSED / PENDING / UNVERIFIED / CONFLICTED                   │
│      └── 🎯 Execution Unresolved (Non-actionable Diagnostic State)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP JSON
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Server (v1)                            │
│                                                                        │
│  /api/v1/market/instruments/{symbol}  (Dedicated Instrument Endpoint)  │
│  /api/v1/analytics/{symbol}           (Enriched Envelope Integration)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               analyst_dashboard.security_master Service                │
│                                                                        │
│  ┌────────────────────────┐         ┌───────────────────────────────┐  │
│  │ SecurityMasterService  │◄───────►│ SecurityMasterRepository      │  │
│  └───────────┬────────────┘         │ - Thread-safe LRU Cache (1024)│  │
│              │                      │ - SQLite: security_master.db  │  │
│              ▼                      └───────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ SecurityMasterNormalizationEngine & Eligibility Policy           │  │
│  └───────────┬──────────────────────────────────────┬───────────────┘  │
│              │                                      │                  │
│              ▼                                      ▼                  │
│  ┌────────────────────────┐         ┌───────────────────────────────┐  │
│  │  AlpacaIdentityAdapter │         │    OpenFIGISubtypeAdapter     │  │
│  │  (Identity & Listing)  │         │    (Structural Subtyping)     │  │
│  └───────────┬────────────┘         └───────────────┬───────────────┘  │
└──────────────┼──────────────────────────────────────┼──────────────────┘
               │                                      │
               ▼                                      ▼
     Alpaca Assets API                      OpenFIGI v3 API
     (Alpaca Asset Directory)               (Shared Global Rate Limiter)
```

---

## 4. Key Implementation Files

1. **`analyst_dashboard/security_master/models.py`**
   - Implements `CanonicalInstrument`, `AssetClass`, `SecurityType`, `ListingStatus`, `ClassificationStatus`, `ExecutionEligibility`, and `AnalyticsCapability`.
   - Strictly conforms to Section 8 Canonical Contract.
2. **`analyst_dashboard/security_master/alpaca_adapter.py`**
   - Implements `AlpacaIdentityAdapter` querying `/v2/assets/{symbol}`.
   - Enforces `ALPACA_BROAD_CLASS_IS_SUBTYPE_AUTHORITY = NO`. Fails closed on timeouts, 404s, or inactive assets.
3. **`analyst_dashboard/security_master/openfigi_adapter.py`**
   - Implements `OpenFIGISubtypeAdapter` querying `https://api.openfigi.com/v3/mapping`.
   - Normalizes structural subtypes (`Common Stock`, `ETP`, `ADR`, `REIT`, `Warrant`, `Preferred`, `Unit`, `Right`).
   - Integrates with shared `GlobalSQLiteRateLimiter` (`data/operational/openfigi_operational.db`) preserving the global 20 req/60s ceiling across all ARX consumers.
4. **`analyst_dashboard/security_master/normalization.py`**
   - Implements `SecurityMasterNormalizationEngine`.
   - Reconciles provider evidence, detects material conflicts, and produces normalized `CanonicalInstrument` records.
5. **`analyst_dashboard/security_master/eligibility.py`**
   - Implements `evaluate_execution_eligibility()`.
   - Enforces fail-closed rules: `STOCK_EXECUTION` requires `VERIFIED` + `COMMON_STOCK` + `ACTIVE`.
6. **`analyst_dashboard/security_master/config.py`**
   - Implements CWD-independent path resolution and persistence firewall (`SecurityMasterFirewallError`).
7. **`analyst_dashboard/security_master/persistence.py`**
   - Implements `SecurityMasterRepository`: hybrid SQLite persistence + thread-safe LRU cache with TTL freshness checks.
8. **`analyst_dashboard/security_master/service.py`**
   - High-level orchestrator providing `get_or_resolve_instrument(symbol)`.
9. **`api/routes/market.py`**
   - Dedicated endpoint `GET /api/v1/market/instruments/{symbol}` returning `InstrumentResponse`.
10. **`api/routes/analytics.py`**
    - Enriches `/api/v1/analytics/{symbol}` response dict with `"instrument": canonical_inst.to_dict()` while keeping analytics math 100% independent.
11. **`frontend/lib/api.ts` & `frontend/lib/assetTypeUtils.ts`**
    - Adds `instrument?: CanonicalInstrumentEnvelope` to `AnalyticsResponse`.
    - Exposes `fetchCanonicalInstrument()`.
    - Updates `resolveExecutionEligibility()` and `resolveAssetType()`.
12. **`frontend/app/page.tsx`**
    - Refactors `planSlot` and `activeTab === "EXECUTION"` to gate execution cards on `data?.instrument?.execution_eligibility`.
    - Renders pending state during load and "🎯 Execution Unresolved" with contextual subtype information on `FAIL_CLOSED`.

---

## 5. Acceptance Test Verification

All required instrument acceptance test cases pass deterministically in `tests/test_security_master_contract.py`, `tests/test_canonical_security_master.py`, and `frontend/tests/canonicalSecurityMasterRouting.test.ts`:

| Symbol | Case Category | Asset Class | Security Type | Listing Status | Classification Status | Execution Eligibility | Routing Verdict |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **PLSE** | Valid Uncatalogued Equity | `EQUITY` | `COMMON_STOCK` | `ACTIVE` | `VERIFIED` | `STOCK_EXECUTION` | `OptimalEntryExitCard` |
| **SPY** | Baseline ETF | `EQUITY` | `ETP` | `ACTIVE` | `VERIFIED` | `ETF_EXECUTION` | `EtfCostOfOwnershipCard` |
| **AAPL** | Baseline Large-Cap Equity | `EQUITY` | `COMMON_STOCK` | `ACTIVE` | `VERIFIED` | `STOCK_EXECUTION` | `OptimalEntryExitCard` |
| **TSM** | ADR (Unsupported Equity) | `EQUITY` | `ADR` | `ACTIVE` | `VERIFIED` | `FAIL_CLOSED` | `🎯 Execution Unresolved` |
| **O** | REIT (Unsupported Equity) | `EQUITY` | `REIT` | `ACTIVE` | `VERIFIED` | `FAIL_CLOSED` | `🎯 Execution Unresolved` |
| **CORZW** | Warrant (Unsupported Equity) | `EQUITY` | `WARRANT` | `ACTIVE` | `VERIFIED` | `FAIL_CLOSED` | `🎯 Execution Unresolved` |
| **BAC.PRK** | Preferred Share | `EQUITY` | `PREFERRED_STOCK` | `ACTIVE` | `VERIFIED` | `FAIL_CLOSED` | `🎯 Execution Unresolved` |
| **UNLISTED_999** | Unlisted / 404 | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `FAIL_CLOSED` | `🎯 Execution Unresolved` |
| **CONFLICTED** | Provider Disagreement | `UNKNOWN` | `UNKNOWN` | `ACTIVE` | `CONFLICTED` | `FAIL_CLOSED` | `🎯 Execution Unresolved` |
| **NO_SUBTYPE** | Missing FIGI Match | `EQUITY` | `UNKNOWN` | `ACTIVE` | `UNVERIFIED` | `FAIL_CLOSED` | `🎯 Execution Unresolved` |
| **INACTIVE** | Delisted / Inactive | `EQUITY` | `COMMON_STOCK` | `INACTIVE` | `VERIFIED` | `FAIL_CLOSED` | `🎯 Execution Unresolved` |
| **OUTAGE** | Provider 500 / Timeout | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `FAIL_CLOSED` | `🎯 Execution Unresolved` |

---

## 6. Regression Test Summary

All automated regression test suites executed and passed with zero errors:

1. **Security Master Python Contract Tests (`tests/test_security_master_contract.py`):**
   - 15/15 tests passed.
2. **Canonical Security Master Python Tests (`tests/test_canonical_security_master.py`):**
   - 17/17 tests passed.
3. **ETF v2 OpenFIGI Contract & Global Rate Limiter Tests (`tests/test_etf_v2_openfigi_*.py`):**
   - 46/46 tests passed (including single-process, 2-process, 4-process aggregate ceiling, rolling window, and contention tests).
4. **Analysis & Decision Hierarchy Tests (`tests/test_canonical_decision_context.py`, `tests/test_phase21_decision_parity.py`, `tests/test_phase2_decision_authority.py`, `tests/test_optimal_execution.py`):**
   - 27/27 tests passed.
5. **Frontend Vitest & Architecture Suites (`npm.cmd test -- --run`):**
   - 100% of tests passed, including all decision authority consolidation, radar portfolio context, weekly spotlight decoupling, and canonical security master routing tests.

---

## 7. Legacy Authority Retirement Evidence

- `UNAUTHORIZED_FRONTEND_CLASSIFICATION_AUTHORITIES`: `0`
- `GENERIC_FALLBACKS_TO_STOCK_EXECUTION`: `0`
- `SYMBOL_SPECIFIC_EXECUTION_EXCEPTIONS`: `0`
- `masterCatalog.ts`: Annotated with `FRONTEND_CATALOG_AUTHORITY = NONE`, `ROLE = PRESENTATION_ENRICHMENT_ONLY`.
- `SHARED_WATCHLIST_ITEMS`: Annotated with `FRONTEND_CATALOG_AUTHORITY = NONE`, `ROLE = PRESENTATION_ENRICHMENT_ONLY`.
- `resolveAssetType`: Strictly for synchronous presentation bootstrapping before server payload loads.
- `resolveExecutionEligibility`: Strictly server-owned (`context.instrument.execution_eligibility`).

---

## 8. Repository Boundary Attestation

- **Physical Worktree:** `C:/Users/akara/Documents/Projects/finance-security-master`
- **Target Branch:** `arx/security-master`
- **Target Remote:** `origin/arx/security-master`
- **Branch Parity & Isolation:** Main branch untouched. Zero production deployment triggered. Zero merge to main.

---

## 9. Gate Decision Verdict

```text
GATE =
  PASS_ARX_CANONICAL_SECURITY_MASTER_IMPLEMENTATION
CANONICAL_SECURITY_MASTER_IMPLEMENTED =
  YES
CANONICAL_AUTHORITY =
  ARX_SERVER_SECURITY_MASTER
IDENTITY_LISTING_AUTHORITY =
  ALPACA_ASSET_DIRECTORY
SECURITY_SUBTYPE_AUTHORITY =
  OPENFIGI_V3_MAPPING
EXECUTION_ROUTING_MIGRATED =
  YES
FAIL_CLOSED_BOUNDARY_VERIFIED =
  YES
UNAUTHORIZED_FRONTEND_CLASSIFICATION_AUTHORITIES =
  0
GENERIC_FALLBACKS =
  0
SYMBOL_SPECIFIC_EXECUTION_EXCEPTIONS =
  0
ETF_V2_BUSINESS_LOGIC_CHANGED =
  NO
QUANT_ENGINE_CHANGED =
  NO
SCORING_CHANGED =
  NO
RANKING_CHANGED =
  NO
BRANCH =
  arx/security-master
WORKTREE =
  C:/Users/akara/Documents/Projects/finance-security-master
LOCAL_REMOTE_PARITY =
  READY_TO_PUSH
MERGE_TO_MAIN_PERMITTED =
  NO
PRODUCTION_DEPLOYMENT_PERMITTED =
  NO
NEXT_ACTION =
  STAGE_COMMIT_PUSH_AND_STOP
```
