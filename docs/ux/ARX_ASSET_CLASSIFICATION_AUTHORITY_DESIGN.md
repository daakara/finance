# ARX TERMINAL — ASSET CLASSIFICATION & EXECUTION ELIGIBILITY
## ARCHITECTURAL DESIGN & POLICY RECONCILIATION SPECIFICATION

```ini
DOCUMENT_CLASS =
  CANONICAL_UX_AND_DATA_CONTRACT_SPECIFICATION
SUBSYSTEM =
  ASSET_CLASSIFICATION_AND_EXECUTION_ROUTING
STATUS =
  APPROVED_POLICY_FROZEN
DATE =
  2026-10-06
GATE =
  PASS_ARX_COMMON_STOCK_EXECUTION_POLICY_RECONCILIATION
PREDECESSOR_GATE =
  PASS_ARX_ANALYSIS_CORRIDOR_TRIGGER_SEMANTICS_RELEASE_VERIFICATION
POLICY_ARTIFACT_HIERARCHY =
  CANONICAL_ARCHITECTURE: docs/architecture/ARX_CANONICAL_SECURITY_MASTER_DESIGN.md
  PROVIDER_EVIDENCE: docs/architecture/ARX_SECURITY_MASTER_PROVIDER_ELIGIBILITY_EVIDENCE.md
  UX_POLICY_COMPANION: docs/ux/ARX_ASSET_CLASSIFICATION_AUTHORITY_DESIGN.md
```

---

## 1. Executive Summary & Purpose

This design document establishes the canonical authority model for **Asset Classification** and **Execution Eligibility** across ARX Terminal surfaces (Analysis, Radar, Setups, Portfolio, Workstation).

### Problem Statement & Policy Contradiction
Prior frontend implementations exhibited a critical semantic tension between two competing models:
1. **The Frozen Epistemic Boundary**: Instruments with `UNKNOWN` classification must strictly fail closed to protect users and systems from applying invalid equity models to unsupported asset classes (e.g. warrants, units, rights, preferreds).
2. **The Uncatalogued Equities Dilemma (The PLSE Case)**: Valid US exchange-listed common equities (such as `PLSE` - Pulse Biosciences, Inc., `IREN`, `MU`) that were absent from the client-side static 43-symbol [`MASTER_ASSET_CATALOG`](file:///c:/Users/akara/Documents/Projects/finance/frontend/lib/masterCatalog.ts) were classified as `UNKNOWN`, causing the UI to suppress the execution plan and display `"🎯 Execution Unresolved"` despite successful backend quantitative analytics.

A temporary or proposed heuristic — treating any instrument that is "not an ETF and not crypto" as eligible for the equity execution surface — creates material risk of execution misrouting: execution behavior is not validated for these security subtypes under the current execution capability contract, and therefore routing must fail closed.

This specification adjudicates that policy contradiction, audits the execution surface assumptions, separates the four core concepts of financial instrument handling, and freezes **Option A (COMMON_STOCK_ONLY with FAIL_CLOSED)** as the authoritative architecture.

---

## 2. Separation of the Four Core Concepts

To eliminate domain conflation, the ARX architecture formally decouples four distinct concepts:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. ASSET IDENTITY & LISTING STATUS                                     │
│    • Ticker symbol, CIK, FIGI, Issuer Name, Primary Exchange, Active  │
│    • Authority: Alpaca Asset Directory (Listing/Status) + OpenFIGI    │
│    • Corroboration: SEC EDGAR / Exchange Directory (Optional Future)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. SECURITY SUBTYPE                                                    │
│    • Common Stock, ETF, Crypto, Preferred, Warrant, Unit, Right, etc.  │
│    • Authority: OpenFIGI V3 Mapping (Normalized by Server SecMaster)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. EXECUTION ELIGIBILITY                                               │
│    • STOCK_EXECUTION, ETF_EXECUTION, CRYPTO_EXECUTION, UNSUPPORTED     │
│    • Authority: Canonical Precondition Rules (Subtype + Valid Tape)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. EXECUTION SURFACE                                                   │
│    • OptimalEntryExitCard, EtfCostOfOwnershipCard, ExecutionUnresolved │
│    • Authority: UI Routing Layer (Downstream consumer of Eligibility)  │
└────────────────────────────────────────────────────────────────────────┘
```

### Authoritative Mapping Ledger

```ini
CANONICAL_AUTHORITY =
  ARX_SERVER_SECURITY_MASTER
IDENTITY_LISTING_AUTHORITY =
  ALPACA_ASSET_DIRECTORY
SECURITY_SUBTYPE_AUTHORITY =
  OPENFIGI_V3_MAPPING
SEC_EDGAR_ROLE =
  OPTIONAL_FUTURE_CORROBORATION_UNLESS_SEPARATELY_EVIDENCED
PRIMARY_EXCHANGE_DIRECTORY_ROLE =
  OPTIONAL_FUTURE_CORROBORATION_UNLESS_SEPARATELY_EVIDENCED
EXECUTION_ELIGIBILITY_AUTHORITY =
  CANONICAL_PRECONDITION_RULES_ENGINE
EXECUTION_SURFACE_AUTHORITY =
  FRONTEND_ROUTING_LAYER (Consumes Execution Eligibility strictly)
```

No concept may be substituted for another. Specifically:
- `Analytics Success ≠ Security Subtype`
- `Curated Catalog Membership ≠ Execution Eligibility`
- `Non-ETF/Non-Crypto Negation ≠ Common Stock Identity`

---

## 3. OptimalEntryExitCard Capability & Assumption Audit

An exhaustive audit of [`OptimalEntryExitCard`](file:///c:/Users/akara/Documents/Projects/finance/frontend/components/OptimalEntryExitCard.tsx), [`OptimalExecutionEngine`](file:///c:/Users/akara/Documents/Projects/finance/analyst_dashboard/analyzers/optimal_execution.py), and dependent components ([`PositionSizerModal`](file:///c:/Users/akara/Documents/Projects/finance/frontend/components/PositionSizerModal.tsx), [`governorSizingEngine`](file:///c:/Users/akara/Documents/Projects/finance/frontend/lib/simulation/governorSizingEngine.ts), [`PreFlightChecklistModal`](file:///c:/Users/akara/Documents/Projects/finance/frontend/components/PreFlightChecklistModal.tsx)) was conducted to evaluate whether they assume characteristics specific to operating common stock.

### Detailed Assumption Audit Matrix

| Execution Assumption | Source Component | Common Stock Specific? | Safe for ADR? | Safe for Preferred? | Safe for Warrant? | Safe for Unit? | Safe for Right? | Safe for UNKNOWN? |
|---|---|---|---|---|---|---|---|---|
| **Minervini Stage 2 Trend Template** | `optimal_execution.py` (L120-170) | **YES** | PARTIAL (Sponsored) | **NO** | **NO** | **NO** | **NO** | **NO** |
| **VCP Volatility Contraction** | `optimal_execution.py` (L180-210) | **YES** | PARTIAL (Sponsored) | **NO** | **NO** | **NO** | **NO** | **NO** |
| **20 EMA Trend Pullback Entry** | `optimal_execution.py` (L215-240) | **YES** | PARTIAL (Sponsored) | **NO** | **NO** | **NO** | **NO** | **NO** |
| **ATR-14 Risk Floor & Geometry** | `optimal_execution.py` (L85-97) | **YES** | **YES** | **NO** (Par ceiling) | **NO** (Decay/Delta) | **NO** | **NO** | **NO** |
| **Unbounded Upside 2:1 / 3:1 RR** | `optimal_execution.py` (L245-280) | **YES** | **YES** | **NO** (Fixed Par) | **NO** (Strike/Expiry) | **NO** | **NO** | **NO** |
| **Amihud Liquidity / $5M Dollar Vol** | `liquidity_guard.py` | **YES** | **YES** | **NO** (Spread wider) | **NO** (Extreme thin) | **NO** | **NO** | **NO** |
| **Linear Dollar Risk Position Sizing** | `governorSizingEngine.ts` | **YES** | **YES** | **NO** | **NO** (100x leverage) | **NO** | **NO** | **NO** |
| **100% Cash Buying Power Floor** | `DayTraderPositionSizer.tsx` | **YES** | **YES** | **NO** | **NO** | **NO** | **NO** | **NO** |
| **Corporate Fundamental Checklist** | `PreFlightChecklistModal.tsx` | **YES** | **YES** | **NO** (Fixed income) | **NO** (Derivative) | **NO** | **NO** | **NO** |
| **Direct Order Drafting (`LMT: $X`)** | `orderClipboard.ts` | **YES** | **YES** | **NO** | **NO** | **NO** | **NO** | **NO** |
| **5-Day Multi-Session Outcome Fill** | `paper_trading_outcome_evaluator.py` | **YES** | **YES** | **NO** | **NO** (Time decay) | **NO** | **NO** | **NO** |

### Capability Audit Conclusion
Every primary calculation in the execution ladder assumes:
1. **Operating equity capital structure** (common shares with voting rights, revenue, gross margin, institutional float).
2. **Continuous, unbounded price discovery** with linear risk/reward (no fixed par redemption, no callable provisions, no expiration dates, no strike conversions).
3. **Linear, non-leveraged position sizing** (position sizing algorithms are not validated for leveraged or derivative structures; execution behavior is not authorized for warrants, and routing fails closed).

```ini
MATERIAL_COMMON_STOCK_ASSUMPTIONS_PRESENT =
  YES
GENERIC_FALLBACK_SAFETY =
  FAIL
```

Because material assumptions strictly demand common-stock characteristics, a generic fallback permitting UNKNOWN instruments into `OptimalEntryExitCard` is **architecturally unsafe and rejected**.

---

## 4. Evaluation of Policy Options

### 4.1 Option B: Generic Long-Only Execution Candidate (Rejected)
- **Proposed Logic**: If an instrument is not verified as an ETF, not crypto, and not on an explicit blacklist of unsupported subtypes, route it to `OptimalEntryExitCard`.
- **Fatal Flaws**:
  1. *Negative Enumeration Vulnerability*: US markets list over 4,000 non-standard instruments (warrants, SPAC units, preferreds, rights, contingent value rights, liquidation trusts). Maintaining a client-side blacklist of all specialized instruments is impossible and violates fail-closed security.
  2. *Derivative Leverage Mis-Sizing*: An uncatalogued warrant (e.g. `LUNR.WS` or `ASTSW`) trading at \$1.50 would be sized by `PositionSizerModal` as an operating stock, allocating thousands of units and exposing the trader to massive delta-driven loss upon expiration.
  3. *Semantic Corruption*: Calling an UNKNOWN instrument a "Stock" or routing it to a stock execution card undermines the platform's institutional credibility.
- **Verdict**: **REJECTED (FAIL)**.

### 4.2 Option A: Common-Stock-Only Candidate (Approved)
- **Mandatory Policy**:
  ```ini
  EXECUTION_POLICY =
    COMMON_STOCK_ONLY
  UNKNOWN_ROUTING =
    FAIL_CLOSED
  AUTHORITATIVE_SECURITY_TYPE_REQUIRED =
    YES
  SERVER_OWNED_CLASSIFICATION_AUTHORITY_REQUIRED =
    YES
  ```
- **Rationale**:
  - `OptimalEntryExitCard` represents the execution surface for operating common equities and sponsored exchange-listed ADRs only.
  - Instruments whose security subtype cannot be verified fail closed to `"🎯 Execution Unresolved"`.
  - The "PLSE regression" was caused not by an overly strict fail-closed contract, but by an **authority deficit**: the client was relying on a static 43-symbol dictionary instead of a real, server-owned security master.
  - The correct architectural resolution is to provide the server with a live classification authority (e.g. provider `quoteType` / SEC filing metadata) so that PLSE and all 8,000+ real common stocks are authoritatively recognized as `Common Stock`.
- **Verdict**: **APPROVED (PASS)**.

---

## 5. Residual Risk Analysis: Uncatalogued Specialized Securities

To prove that generic fallback is unsafe, five representative uncatalogued specialized securities were evaluated against current and proposed routing:

| Security | Authoritative Type | In Static MasterCatalog? | Generic Fallback Route | Expected Safe Route | Misrouting Risk? |
|---|---|---|---|---|---|
| `LUNR.WS` (Intuitive Machines Warrants) | Warrant | NO | Stock Execution (`OptimalEntryExitCard`) | Fail-Closed (`Execution Unresolved`) | **YES (Not Authorized by Capability Contract; Fails Closed)** |
| `AAC.U` (Ares Acquisition Units) | Unit (SPAC) | NO | Stock Execution (`OptimalEntryExitCard`) | Fail-Closed (`Execution Unresolved`) | **YES (Not Authorized by Capability Contract; Fails Closed)** |
| `BMA.RT` (Banco Macro Rights) | Subscription Right | NO | Stock Execution (`OptimalEntryExitCard`) | Fail-Closed (`Execution Unresolved`) | **YES (Not Authorized by Capability Contract; Fails Closed)** |
| `BAC.PR.L` (Bank of America Preferred) | Preferred Stock | NO | Stock Execution (`OptimalEntryExitCard`) | Fail-Closed (`Execution Unresolved`) | **YES (Not Authorized by Capability Contract; Fails Closed)** |
| `HYG` (High Yield Corporate Bond ETF) | Bond ETF | NO (if uncatalogued) | Stock Execution (`OptimalEntryExitCard`) | ETF Execution (`EtfCostOfOwnershipCard`) | **YES (Requires ETF Execution Routing)** |

```ini
GENERIC_FALLBACK_RISK =
  UNACCEPTABLE
UNSUPPORTED_SPECIALIZED_SECURITY_LEAK =
  CONFIRMED
CONCLUSION =
  FAIL_CLOSED_IS_MANDATORY
```

---

## 6. Separation of Analytics Capability from Execution Eligibility

A fundamental principle ratified by this gate is that backend mathematical success does not confer asset classification or execution eligibility:

```
┌─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│ Instrument & Analytics State    │ Authoritative Identity          │ Execution Eligibility           │
├─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Valid Common Stock + Analytics OK│ Common Stock                    │ STOCK_EXECUTION (Eligible)      │
│ Valid Common Stock + Analytics Err│ Common Stock                   │ INELIGIBLE (Data/History Error) │
│ Warrant + Analytics OK          │ Warrant                         │ UNSUPPORTED (Blocked)           │
│ ETF + Analytics OK              │ ETF                             │ ETF_EXECUTION (Cockpit Only)    │
│ Crypto + Analytics OK           │ Crypto                          │ CRYPTO_EXECUTION (Crypto Only)  │
│ UNKNOWN + Analytics OK          │ UNKNOWN                         │ FAIL_CLOSED (Execution Unresolved)│
└─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```

The quant engine's ability to calculate moving averages or ATR on a price series does not transform a warrant or an unknown instrument into a common stock.

---

## 7. The PLSE Case Reconstruction

The regression regarding `PLSE` (Pulse Biosciences, Inc.) was audited:
- **Authoritative Entity**: Operating medical technology company listed on NASDAQ under symbol `PLSE`.
- **Security Subtype**: US Exchange-Listed Common Stock.
- **Backend Analytics**: Fully capable, returns valid price candles, Stage 2 evaluation, and ATR levels.
- **Root Cause of Unresolved Status**: The frontend [`assetTypeUtils.ts`](file:///c:/Users/akara/Documents/Projects/finance/frontend/lib/assetTypeUtils.ts) relied strictly on static dictionary [`masterCatalog.ts`](file:///c:/Users/akara/Documents/Projects/finance/frontend/lib/masterCatalog.ts), which contains only 43 hand-curated tickers.
- **Architectural Remedy**:
  1. Do NOT hardcode `PLSE` as a one-off exception in `masterCatalog.ts`.
  2. Do NOT degrade the fail-closed boundary to allow all UNKNOWN instruments into stock execution.
  3. Implement server-owned security subtype classification in the analytics response payload (e.g. `securityType: "EQUITY" | "ETF" | "CRYPTO" | "UNKNOWN"` derived from authoritative provider metadata).
  4. Update `frontend/lib/assetTypeUtils.ts` to consume the server-certified `securityType`.

---

## 8. Frozen Invariants

```ini
INV-ASSET-POLICY-01 =
  Asset classification (identity/subtype) and execution eligibility are strictly separate concepts.

INV-ASSET-POLICY-02 =
  UNKNOWN classification is never silently reclassified or defaulted to common stock.

INV-ASSET-POLICY-03 =
  Execution eligibility must be represented explicitly through structured eligibility states, not inferred from security type alone.

INV-ASSET-POLICY-04 =
  Unsupported specialized securities (warrants, units, rights, preferreds) must strictly fail closed and cannot enter OptimalEntryExitCard.

INV-ASSET-POLICY-05 =
  Successful calculation of backend analytics or quantitative indicators cannot grant execution eligibility.

INV-ASSET-POLICY-06 =
  ETF and Crypto routing contracts remain fully isolated from equity execution routing.

INV-ASSET-POLICY-07 =
  Membership in a client-side curated catalog (masterCatalog.ts) is a presentation convenience and cannot act as authoritative security master.

INV-ASSET-POLICY-08 =
  PLSE and uncatalogued equities must be supported through authoritative server-side classification, not through special-case exemptions or weakened fail-closed gates.
```

---

## 9. Acceptance Matrix Evaluation

| Criterion ID | Requirement | Evaluation & Evidence | Status |
|---|---|---|---|
| **ASSET-POLICY-01** | Asset identity authority reconstructed | Mapped to Alpaca Asset Directory (listing/activity) and OpenFIGI v3 (subtype); SEC EDGAR / primary exchange retained as optional future corroboration. | **PASS** |
| **ASSET-POLICY-02** | Execution eligibility authority reconstructed | Defined as precondition rule engine consuming verified subtype. | **PASS** |
| **ASSET-POLICY-03** | Execution surface capability audited | Full 11-point audit completed across `OptimalExecutionEngine` & UI. | **PASS** |
| **ASSET-POLICY-04** | Common-stock assumptions identified | Stage 2, VCP, EPS growth, float, linear sizing confirmed equity-only. | **PASS** |
| **ASSET-POLICY-05** | Generic execution safety evaluated | Generic fallback proven unsafe (`GENERIC_FALLBACK_SAFETY = FAIL`). | **PASS** |
| **ASSET-POLICY-06** | Uncatalogued specialized-security risk tested | Warrants, units, rights, preferreds audited; unvalidated execution routing confirmed; fails closed. | **PASS** |
| **ASSET-POLICY-07** | Analytics capability separated from eligibility | Matrix codified: Analytics Success ≠ Eligibility. | **PASS** |
| **ASSET-POLICY-08** | PLSE reproduced without exception logic | Root cause identified as static client dictionary deficit. | **PASS** |
| **ASSET-POLICY-09** | One explicit execution policy selected | **OPTION A (COMMON_STOCK_ONLY)** formally selected and ratified. | **PASS** |
| **ASSET-POLICY-10** | UNKNOWN semantics reconciled | UNKNOWN strictly fails closed to `"🎯 Execution Unresolved"`. | **PASS** |
| **ASSET-POLICY-11** | ETF semantics preserved | ETF routes strictly to `EtfCostOfOwnershipCard` / ETF Cockpit. | **PASS** |
| **ASSET-POLICY-12** | Crypto semantics preserved | Server Security Master is canonical authority (`CRYPTO_CANONICAL_AUTHORITY = SERVER_SECURITY_MASTER`); client `-USD` suffix heuristic is retained strictly as `TEMPORARY_COMPATIBILITY_BEHAVIOR` with zero canonical authority (`CRYPTO_SUFFIX_HEURISTIC_AUTHORITY = NONE`). | **PASS** |
| **ASSET-POLICY-13** | Curated metadata remains non-authoritative | `masterCatalog.ts` classified strictly as non-authoritative fallback. | **PASS** |
| **ASSET-POLICY-14** | No quant logic changed | Zero modifications to mathematical formulas or quant engines. | **PASS** |
| **ASSET-POLICY-15** | No scoring/ranking/actionability changed | Confluence scores, Radar rankings, and actionability logic untouched. | **PASS** |

---

## 10. Formal Gate Verdict

```ini
GATE =
  PASS_ARX_COMMON_STOCK_EXECUTION_POLICY_RECONCILIATION
EXECUTION_POLICY =
  COMMON_STOCK_ONLY
UNKNOWN_ROUTING =
  FAIL_CLOSED
AUTHORITATIVE_SECURITY_TYPE_REQUIRED =
  YES
SERVER_OWNED_CLASSIFICATION_AUTHORITY_REQUIRED =
  YES
PLSE_CURRENT_RUNTIME_BEHAVIOR =
  TEMPORARY_COMPATIBILITY_BEHAVIOR
ETF_ROUTING =
  PRESERVED
CRYPTO_ROUTING =
  PRESERVED
CRYPTO_CANONICAL_AUTHORITY =
  SERVER_SECURITY_MASTER
CRYPTO_SUFFIX_HEURISTIC_AUTHORITY =
  NONE
CRYPTO_SUFFIX_HEURISTIC_RUNTIME_STATE =
  TEMPORARY_COMPATIBILITY_BEHAVIOR
QUANT_ENGINE_CHANGED =
  NO
SCORING_CHANGED =
  NO
RANKING_CHANGED =
  NO
ACTIONABILITY_CHANGED =
  NO
NEXT_AUTHORIZED_ACTION =
  COMPLETE_SECURITY_MASTER_DESIGN_FREEZE_AND_REPOSITORY_FREEZE
RUNTIME_IMPLEMENTATION =
  NOT_AUTHORIZED
AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```

*Epistemic Boundary Ratification*:
The execution surface represented by `OptimalEntryExitCard` is formally frozen as **COMMON_STOCK_ONLY**. The platform strictly prohibits defaulting unverified or UNKNOWN instruments into stock execution. Progression to supporting valid uncatalogued equities such as PLSE must proceed through a dedicated implementation gate that establishes authoritative server-side security-type classification, preserving all fail-closed boundaries. Automatic successor execution is halted.
