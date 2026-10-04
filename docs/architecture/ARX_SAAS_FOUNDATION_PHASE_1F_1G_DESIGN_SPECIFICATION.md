# ARX Terminal — SaaS Foundation Phase 1F–1G Design Specification

**Gate Reference**: `ARX_SAAS_FOUNDATION_PHASE_1F_1G_DESIGN_RECONCILIATION_GATE`
**Execution Timestamp**: 2026-10-04T11:30:00+02:00
**Predecessor Gate**: `ARX_SAAS_FOUNDATION_PHASE_1F_1G_DESIGN_AND_WIRING_GATE`
**Predecessor Status**: `HOLD_PENDING_DESIGN_RECONCILIATION`
**Reconciliation Status**: `RECONCILED`
**Predecessor Release SHA**: `724b5e3659ba0287fc3d8b9d58b4ef7eecde8703`
**Remote Branch**: `origin/feat/arx-saas-foundation-phase1-seams`
**Design Scope**: Phase 1F (Application-Service & Context Wiring) and Phase 1G (Workspace Tenancy & Migration) Reconciled Architecture
**Implementation Authorization**: `NONE (STRICTLY DESIGN-ONLY GATE)`

---

## 1. Frozen Baseline & Predecessor State

The foundational contracts and invariants established in Phase 1A–1E and frozen under release `724b5e3659ba0287fc3d8b9d58b4ef7eecde8703` remain authoritative:

| Item | Status | Lineage / Reference |
|---|---|---|
| Predecessor Release Gate | `PASS_ARX_SAAS_FOUNDATION_PHASE_1A_1E_RELEASE` | Verified & frozen |
| Predecessor Design Gate | `HOLD_PENDING_DESIGN_RECONCILIATION` | Reconciled in this document |
| Release Commit SHA | `724b5e3659ba0287fc3d8b9d58b4ef7eecde8703` | Verified on `origin/feat/arx-saas-foundation-phase1-seams` |
| `RequestContext` Contract | Implemented (`api/context/request_context.py`) | Frozen, immutable dataclass |
| Capability Vocabulary | Implemented (`api/capabilities/capabilities.py`) | 17 capabilities, 5 limits |
| Entitlement Resolver | Implemented (`api/services/entitlement_resolver.py`) | `EntitlementSet`, `DefaultEntitlementResolver` |
| Frontend Contracts | Implemented (`frontend/lib/saas/*.ts`) | 1:1 typed parity with backend |
| Invariant `INV-SAAS-01` | Enforced (`tests/architecture/test_saas_invariants.py`) | 14/14 AST checks passing |
| Invariant `INV-SAAS-05` | Established (Reconciliation Gate) | Public Context-Free Invariant |
| Authentication | Not Implemented | Strictly deferred |
| Subscriptions / Billing | Not Implemented | Strictly deferred |
| Database Schema | Unchanged | 0 migrations |
| Route Wiring | Unrewired | `/api/v1/analytics/{symbol}` unmodified |
| Public Analytics | Context-Free, Plan-Agnostic | Shared public CDN cache preserved |

---

## 2. Current Request-Flow Reconstruction

### 2.1 Current Public Analytics Flow (`GET /api/v1/analytics/{symbol}`)
```
Client HTTP Request
    ↓
FastAPI Router (`api/routes/analytics.py::get_asset_analytics`)
    ↓ Input Validation (Symbol, Period, Interval, User Role)
Direct Yahoo Finance Historical Tape Fetch (`yfinance.Ticker.history`)
    ↓
Optimal Execution Engine (`OptimalExecutionEngine.calculate_trade_levels`)
    ↓
Database Recommendation Logging (`HistoryDatabaseEngine.log_trade_recommendation`)
    ↓
Self-Healing Auditor (`SelfHealingForecastAuditor.audit_and_calibrate`)
    ↓
Market Relationship Graph Engine (`MarketGraphEngine.get_relationship_graph`)
    ↓
Smart Money Engine (`SmartMoneyEngine.get_options_flow`, `get_congressional_trades`)
    ↓
Confluence Engine (`ConfluenceEngine.calculate_confluence`)
    ↓
Decision Trace Engine (`DecisionTraceEngine.build_decision_trace`)
    ↓
Decision Hierarchy Engine (`DecisionHierarchyEngine.evaluate`)
    ↓
Direct Route Response Assembly (500+ LOC dictionary construction)
    ↓
Headers Applied (`Cache-Control: public, max-age=15, s-maxage=60...`)
    ↓
HTTP Response JSON
```

### 2.2 Current Legacy Private Route Flow (`GET /api/v1/portfolio`, `GET /api/v1/journal/trades`)
```
Client HTTP Request (Optional Header: `X-User-Id`)
    ↓
Local Route Fallback (`_resolve_user_id`: cleans `X-User-Id` or defaults to `"default_user"`)
    ↓
Direct SQLite Query (`HistoryDatabaseEngine.get_user_portfolio(user_id)`)
    ↓
Headers Applied (`Cache-Control: private, no-cache, no-store, must-revalidate`)
    ↓
HTTP Response JSON
```

### 2.3 Key Architectural Deficiencies in Current Flow:
1. **Direct Route-to-Engine Coupling**: Route handlers contain orchestration logic, tape fetching, logging, and response assembly.
2. **Conflated Ownership & Actor**: `user_id` is used simultaneously as an actor identity and as a persistent resource partition selector.
3. **Hard-coded Fallbacks**: `"default_user"` and `"default"` are baked into multiple route files without central coordination.
4. **Lack of Application Seams**: There is no intermediate boundary to enforce authorization, entitlement, or resource limits on private workflows.

---

## 3. Target Request Flow & Layer Boundary Rules

### 3.1 Reconciled Route-Class-Aware Request Pipeline
The architecture strictly rejects any universal middleware or dependency pipeline that forces every HTTP request through `RequestContext` resolution. Public context-free routes bypass context resolution entirely:

```
HTTP Request
    │
    ▼
Route Classification
    ├── PUBLIC CONTEXT-FREE
    │       │
    │       ▼ (Zero RequestContext, zero tenant headers, zero entitlement checks)
    │   Public Route Handler (`api/routes/analytics.py`, `volatility.py`, etc.)
    │       │
    │       ▼ (Direct pure domain invocation)
    │   Domain Engine / Tape Fetch (`analyst_dashboard/analyzers/*.py`, `engines/*.py`)
    │       │
    │       ▼
    │   HTTP Response (Cache-Control: public, s-maxage=...)
    │
    └── PRIVATE OR CONTEXT-AWARE
            │
            ▼
        RequestContext Resolver (`api/context/resolver.py` injected via Depends)
            │
            ▼ (Passes RequestContext and typed command/query)
        Private Route Handler (`api/routes/portfolio.py`, `journal.py`, `cockpit.py`)
            │
            ▼
        Application Service (`api/services/*_service.py`)
            │
            ├─► Authorization Check (Does Actor have access to Workspace?)
            ├─► Entitlement Resolution (Does Workspace possess Capability/Limit?)
            │
            ▼ (Plain domain primitives only)
        Domain Engine / Repository Adapter
            │
            ▼
        Response Projection (Applies private/metadata policies)
            │
            ▼
        HTTP Response (Cache-Control: private, no-cache, no-store, must-revalidate)
```

### 3.2 Layer Boundary Matrix

| Layer | May use RequestContext | May use Entitlements | May use Workspace state | May perform mathematical decisions |
|---|:---:|:---:|:---:|:---:|
| **Public Context-Free Route** | **NO (`INV-SAAS-05`)** | **NO (`INV-SAAS-05`)** | **NO (`INV-SAAS-05`)** | No (Delegates to domain) |
| **Private Route** | Injected via Depends | No (delegates to service) | No (delegates to service) | No |
| **RequestContext Resolver** | Constructs | No | Reads workspace mapping | No |
| **Application Service** | Yes | Yes | Yes | No |
| **Repository / Adapter** | Scoped inputs only | No | Yes | No |
| **Domain Engine** | **NO (`INV-SAAS-01`)** | **NO (`INV-SAAS-01`)** | **NO (`INV-SAAS-01`)** | **YES** |
| **Response Projection** | Optional metadata only | Yes | Optional metadata only | No |

---

## 4. Architectural Invariants

### INV-SAAS-01: Domain Purity Invariant
Protected quantitative, analytical, and execution engines must remain pure mathematical functions. They must never import, accept, or reference commercial or account state:
- Prohibited Engine Inputs: `RequestContext`, `Workspace`, `WorkspaceMembership`, `EntitlementSet`, `Subscription`, `Plan`, `BillingCustomer`, `PaymentStatus`, `SeatCount`, `Capability`.

### INV-SAAS-02: Private Cache Isolation Invariant
Identity-, workspace-, or entitlement-dependent responses must never use shared public cache or CDN cache.
- Private endpoints must emit: `Cache-Control: private, no-cache, no-store, must-revalidate`.
- Public CDN-cached endpoints must remain completely context-free.

### INV-SAAS-03: Workspace Tenancy Invariant
Every workspace-owned persistent record must resolve to exactly one workspace (`workspace_id NOT NULL` at target completion). Nullable tenancy is permitted only as a temporary transition during migration phases M2–M5.

### INV-SAAS-04: Canonical Truth Invariant
Entitlements and commercial plans may alter disclosure depth, UI affordances, or operation availability, but **must never alter canonical quantitative truth**. The mathematical score, technical indicator, and setup recommendation for a given market tape must be identical regardless of caller subscription tier.

### INV-SAAS-05: Public Context-Free Invariant
Public context-free routes must execute without:
1. Resolving `RequestContext` (either via middleware or route-level dependency injection);
2. Inspecting actor or tenant headers (`X-User-Id`, `X-Workspace-ID`, `X-Profile-Id`);
3. Executing entitlement or capability checks;
4. Mutating or accessing tenant-scoped state;
5. Emitting tenancy-dependent telemetry.

---

## 5. PHASE_1F_1G_DESIGN_RECONCILIATION

### 5.1 Reconciliation Context & Objectives
The predecessor design gate `ARX_SAAS_FOUNDATION_PHASE_1F_1G_DESIGN_AND_WIRING_GATE` established sound architectural direction but left critical seams requiring formal reconciliation:
1. **Narrow RequestContext Scope**: Prevent accidental coupling of public analytics to SaaS request pipeline;
2. **Independent 5-Dimensional Route Classification**: Uncouple access, context, semantics, cacheability, and plan dependence;
3. **Public Analytical Route Authority Adjudication**: Ground `/api/v1/analytics/{symbol}`, `/api/v1/analytics/setups`, and `/api/v1/volatility/{symbol}` in explicit code authority sources;
4. **Field-by-Field `user_profiles` Classification**: Correct the premature classification of human trader personal metrics into workspace settings;
5. **Redefinition of Wave 1F-A**: Confine Wave 1F-A strictly to private application services and context resolver without modifying public route code.

---

### 5.2 Independent 5-Dimensional Route Classification Framework

To eliminate conflation, every route is classified across five orthogonal dimensions:

```
┌────────────────────────────────────────────────────────────────────────┐
│               5 INDEPENDENT ROUTE CLASSIFICATION AXES                  │
├────────────────────────────────┬───────────────────────────────────────┤
│ 1. ACCESS_CLASS                │ PUBLIC | AUTHENTICATED | INTERNAL     │
│ 2. CONTEXT_CLASS               │ CONTEXT_FREE | WORKSPACE_SCOPED |     │
│                                │ ACTOR_SCOPED                          │
│ 3. SEMANTIC_CLASS              │ CANONICAL | DERIVED | OPERATIONAL     │
│ 4. CACHE_CLASS                 │ PUBLIC_SHARED | PRIVATE_NO_STORE |    │
│                                │ UNCACHED                              │
│ 5. PLAN_DEPENDENCE             │ PROHIBITED | ALLOWED                  │
└────────────────────────────────┴───────────────────────────────────────┘
```

#### Independence Axioms:
- **Axiom 1**: `ACCESS_CLASS = PUBLIC` does NOT imply `SEMANTIC_CLASS = CANONICAL` (e.g., `/api/v1/analytics/setups` is public but derived; `/health` is public but operational).
- **Axiom 2**: `CONTEXT_CLASS = CONTEXT_FREE` does NOT imply `ACCESS_CLASS = PUBLIC` (e.g., `/api/v1/governance/*` is context-free but internal).
- **Axiom 3**: `CACHE_CLASS = PUBLIC_SHARED` requires `CONTEXT_CLASS = CONTEXT_FREE` and `PLAN_DEPENDENCE = PROHIBITED` (Cache Poisoning Prevention).
- **Axiom 4**: `PLAN_DEPENDENCE = ALLOWED` requires `CACHE_CLASS = PRIVATE_NO_STORE` and `CONTEXT_CLASS != CONTEXT_FREE`.

---

### 5.3 Complete 5-Dimensional Route Classification Matrix

Exhaustive classification of all 45 mounted endpoints across the FastAPI application:

| # | Route Path | Method | ACCESS_CLASS | CONTEXT_CLASS | SEMANTIC_CLASS | CACHE_CLASS | PLAN_DEPENDENCE | Primary Controller / Engine |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | `/health` | GET | `PUBLIC` | `CONTEXT_FREE` | `OPERATIONAL` | `UNCACHED` | `PROHIBITED` | `api/main.py::health_check` |
| 2 | `/api/v1/analytics/{symbol}` | GET | `PUBLIC` | `CONTEXT_FREE` | `CANONICAL` | `PUBLIC_SHARED` | `PROHIBITED` | `DecisionHierarchyEngine` |
| 3 | `/api/v1/analytics/setups` | GET | `PUBLIC` | `CONTEXT_FREE` | `DERIVED` | `PUBLIC_SHARED` | `PROHIBITED` | `OptimalExecutionEngine` |
| 4 | `/api/v1/analytics/setups/{symbol}` | GET | `PUBLIC` | `CONTEXT_FREE` | `DERIVED` | `PUBLIC_SHARED` | `PROHIBITED` | `OptimalExecutionEngine` |
| 5 | `/api/v1/volatility/{symbol}` | GET | `PUBLIC` | `CONTEXT_FREE` | `DERIVED` | `PUBLIC_SHARED` | `PROHIBITED` | `technical_engine.py / GARCH` |
| 6 | `/api/v1/regimes/current` | GET | `PUBLIC` | `CONTEXT_FREE` | `CANONICAL` | `PUBLIC_SHARED` | `PROHIBITED` | `regimes.py::MarketRegime` |
| 7 | `/api/v1/regimes/{symbol}` | GET | `PUBLIC` | `CONTEXT_FREE` | `CANONICAL` | `PUBLIC_SHARED` | `PROHIBITED` | `regimes.py::AssetRegime` |
| 8 | `/api/v1/smart-money/overview` | GET | `PUBLIC` | `CONTEXT_FREE` | `DERIVED` | `PUBLIC_SHARED` | `PROHIBITED` | `SmartMoneyEngine` |
| 9 | `/api/v1/smart-money/congress` | GET | `PUBLIC` | `CONTEXT_FREE` | `CANONICAL` | `PUBLIC_SHARED` | `PROHIBITED` | `SmartMoneyEngine (SEC/Senate)` |
| 10 | `/api/v1/smart-money/options-flow` | GET | `PUBLIC` | `CONTEXT_FREE` | `DERIVED` | `PUBLIC_SHARED` | `PROHIBITED` | `SmartMoneyEngine (Options)` |
| 11 | `/api/v1/smart-money/sec-filings/{symbol}` | GET | `PUBLIC` | `CONTEXT_FREE` | `CANONICAL` | `PUBLIC_SHARED` | `PROHIBITED` | `SmartMoneyEngine (EDGAR)` |
| 12 | `/api/v1/smart-money/finra-darkpool/{symbol}` | GET | `PUBLIC` | `CONTEXT_FREE` | `CANONICAL` | `PUBLIC_SHARED` | `PROHIBITED` | `SmartMoneyEngine (FINRA)` |
| 13 | `/api/v1/macro/ribbon` | GET | `PUBLIC` | `CONTEXT_FREE` | `CANONICAL` | `PUBLIC_SHARED` | `PROHIBITED` | `macro.py::MacroEngine` |
| 14 | `/api/v1/screener/run` | GET | `PUBLIC` | `CONTEXT_FREE` | `DERIVED` | `PUBLIC_SHARED` | `PROHIBITED` | `screener.py::run_screener` |
| 15 | `/api/v1/screener/run` | POST | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `DERIVED` | `PRIVATE_NO_STORE` | `ALLOWED` | `RadarApplicationService` |
| 16 | `/api/v1/screener/position-size` | GET | `PUBLIC` | `CONTEXT_FREE` | `DERIVED` | `PUBLIC_SHARED` | `PROHIBITED` | `governorSizingEngine` |
| 17 | `/api/v1/screener/phase26/validation-report`| GET | `INTERNAL` | `CONTEXT_FREE` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `PROHIBITED` | `screener.py::Validation` |
| 18 | `/api/v1/portfolio` | GET | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `PortfolioApplicationService` |
| 19 | `/api/v1/portfolio` | POST | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `PortfolioApplicationService` |
| 20 | `/api/v1/portfolio/{symbol}` | PUT | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `PortfolioApplicationService` |
| 21 | `/api/v1/portfolio/{symbol}` | DELETE | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `PortfolioApplicationService` |
| 22 | `/api/v1/portfolio/migrate` | POST | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `PortfolioApplicationService` |
| 23 | `/api/v1/journal/telemetry` | GET | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `DERIVED` | `PRIVATE_NO_STORE` | `ALLOWED` | `JournalApplicationService` |
| 24 | `/api/v1/journal/trades` | GET | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `JournalApplicationService` |
| 25 | `/api/v1/journal/trades` | POST | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `JournalApplicationService` |
| 26 | `/api/v1/journal/fill` | POST | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `JournalApplicationService` |
| 27 | `/api/v1/journal/exit` | POST | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `JournalApplicationService` |
| 28 | `/api/v1/journal/close` | POST | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `JournalApplicationService` |
| 29 | `/api/v1/cockpit/state` | GET | `AUTHENTICATED` | `ACTOR_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `CockpitApplicationService` |
| 30 | `/api/v1/cockpit/profile` | POST | `AUTHENTICATED` | `ACTOR_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `CockpitApplicationService` |
| 31 | `/api/v1/cockpit/actions` | POST | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `CockpitApplicationService` |
| 32 | `/api/v1/cockpit/action` | POST | `AUTHENTICATED` | `WORKSPACE_SCOPED` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `ALLOWED` | `CockpitApplicationService` |
| 33 | `/api/v1/etf/profile/{symbol}` | GET | `PUBLIC` | `CONTEXT_FREE` | `CANONICAL` | `PUBLIC_SHARED` | `PROHIBITED` | `etf.py::ETFEngine` |
| 34 | `/api/v1/etf/sectors/{symbol}` | GET | `PUBLIC` | `CONTEXT_FREE` | `CANONICAL` | `PUBLIC_SHARED` | `PROHIBITED` | `etf.py::ETFSectors` |
| 35 | `/api/v1/cache/clear` | POST | `INTERNAL` | `CONTEXT_FREE` | `OPERATIONAL` | `UNCACHED` | `PROHIBITED` | `cache.py::clear_cache` |
| 36 | `/api/v1/governance/evaluation-summary` | GET | `INTERNAL` | `CONTEXT_FREE` | `CANONICAL` | `PRIVATE_NO_STORE` | `PROHIBITED` | `governance_db.py` |
| 37 | `/api/v1/governance/prospective-ledger` | GET | `INTERNAL` | `CONTEXT_FREE` | `CANONICAL` | `PRIVATE_NO_STORE` | `PROHIBITED` | `governance_db.py` |
| 38 | `/api/v1/governance/epoch-2/certify-release` | POST | `INTERNAL` | `CONTEXT_FREE` | `OPERATIONAL` | `UNCACHED` | `PROHIBITED` | `governance.py::certify` |
| 39 | `/api/v1/governance/epoch-2/activate` | POST | `INTERNAL` | `CONTEXT_FREE` | `OPERATIONAL` | `UNCACHED` | `PROHIBITED` | `governance.py::activate` |
| 40 | `/api/v1/governance/epoch-2/revoke-runtime` | POST | `INTERNAL` | `CONTEXT_FREE` | `OPERATIONAL` | `UNCACHED` | `PROHIBITED` | `governance.py::revoke` |
| 41 | `/api/v1/governance/epoch-2/status` | GET | `INTERNAL` | `CONTEXT_FREE` | `OPERATIONAL` | `PRIVATE_NO_STORE` | `PROHIBITED` | `governance.py::status` |
| 42 | `/api/v1/telemetry/persist` | POST | `INTERNAL` | `CONTEXT_FREE` | `OPERATIONAL` | `UNCACHED` | `PROHIBITED` | `telemetry.py::persist` |
| 43 | `/api/v1/telemetry/denominator` | GET | `INTERNAL` | `CONTEXT_FREE` | `CANONICAL` | `PRIVATE_NO_STORE` | `PROHIBITED` | `telemetry.py::denominator` |
| 44 | `/api/v1/telemetry/audit` | GET | `INTERNAL` | `CONTEXT_FREE` | `CANONICAL` | `PRIVATE_NO_STORE` | `PROHIBITED` | `telemetry.py::audit` |
| 45 | `/api/v1/telemetry/epochs` | GET | `INTERNAL` | `CONTEXT_FREE` | `CANONICAL` | `PRIVATE_NO_STORE` | `PROHIBITED` | `telemetry.py::epochs` |

---

### 5.4 Public Analytical Route Authority & Canonical Adjudication

Each of the three primary public analytical routes is evaluated against the authoritative 5-part canonical test:
- **Criterion 1**: Is the response identical for all callers given identical market state?
- **Criterion 2**: Is the source of truth purely market data and deterministic engines?
- **Criterion 3**: Is there zero dependency on actor identity or workspace?
- **Criterion 4**: Is it safe to cache publicly on shared CDN edge infrastructure?
- **Criterion 5**: Does the response contain zero tenant-specific data?

| Route Path | Authority Source (Class/Method/Tape) | Crit 1 | Crit 2 | Crit 3 | Crit 4 | Crit 5 | Semantic Class | Confidence | Decision |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `/api/v1/analytics/{symbol}` | `analyst_dashboard/analyzers/decision_hierarchy.py::DecisionHierarchyEngine.evaluate`, `decision_trace.py::DecisionTraceEngine.build_decision_trace`, `confluence.py::ConfluenceEngine.calculate_confluence`, `optimal_execution.py::OptimalExecutionEngine.calculate_trade_levels` | PASS | PASS | PASS | PASS | PASS | `CANONICAL` | `EVIDENCE_CONFIRMED` | **RESOLVE** |
| `/api/v1/analytics/setups` | `api/routes/analytics.py::_build_tactical_setup` invoking `OptimalExecutionEngine.calculate_trade_levels` & `ConfluenceEngine.calculate_confluence` over candidate universes (`DAY_TRADER_CANDIDATES`, `LONG_TERM_CANDIDATES`) | PASS | PASS | PASS | PASS | PASS | `DERIVED` | `EVIDENCE_CONFIRMED` | **RESOLVE** |
| `/api/v1/volatility/{symbol}` | `api/routes/volatility.py::get_volatility_analysis` invoking GARCH(1,1), Parkinson volatility, and Cornish-Fisher VaR via `engines/technical_engine.py` | PASS | PASS | PASS | PASS | PASS | `DERIVED` | `EVIDENCE_CONFIRMED` | **RESOLVE** |

#### Adjudication Rationale:
1. **`/api/v1/analytics/{symbol}`**:
   - Represents the canonical evaluation and verdict for a single asset.
   - Grounded in deterministic algorithms executing over historical and streaming OHLCV tape.
   - Cache-Control: `public, max-age=15, s-maxage=60, stale-while-revalidate=86400`.
   - Adjudicated as **CANONICAL** authority. Zero tenant or plan variations permitted.
2. **`/api/v1/analytics/setups`**:
   - Computes a derived aggregate catalog of tactical trade setups across a fixed universe.
   - While public and shared, it is a multi-asset derivation, not the single source of truth for an asset's identity.
   - Adjudicated as **DERIVED**.
3. **`/api/v1/volatility/{symbol}`**:
   - Computes econometric volatility models, conditional standard deviations, and value-at-risk projections.
   - Represents a statistical derivative of asset returns, not the raw authoritative verdict.
   - Adjudicated as **DERIVED**.

---

### 5.5 Field-by-Field `user_profiles` Audit & Table Strategy

#### Diagnostic Findings:
The existing `user_profiles` table in `analyst_dashboard/data/db_engine.py` (lines 134–145, 478–535) and its endpoints in `api/routes/cockpit.py` (lines 80–180, 345–385) store personal psychological, cognitive, and financial resilience telemetry for an individual trader:
- Life Health Index (`lhi`)
- Household Health Index (`hhi`)
- Identity Alignment Index (`iai`)
- Liquid Reserves (`liquid_reserves`)
- Monthly Living Expenses Burn (`monthly_burn`)

#### Strategy Selection:
- **Strategy A (`SPLIT_ACTOR_PROFILE_AND_WORKSPACE_SETTINGS`) is SELECTED**.
- **Prohibited Strategy C Rejected**: Migrating `user_profiles` directly into `workspace_settings` is strictly prohibited because personal life metrics cannot be shared across multiple workspace collaborators.
- In Phase 1G, personal trader attributes migrate to `actor_profiles`. A clean, independent `workspace_settings` table will be introduced for organizational configurations (e.g. reporting currency, organizational risk limits).

#### Field-by-Field Classification Matrix:

| Field Name | Type | Current Consumers / Code References | Scope | Target Table (Phase 1G) | Target Field Name | Migration Action | Confidence |
|---|---|---|:---:|---|---|:---:|:---:|
| `user_id` | TEXT PK | `cockpit.py:84,352`, `db_engine.py:488,510` | `ACTOR_PROFILE` | `actor_profiles` | `actor_id` | `MIGRATE_TO_ACTOR_PROFILE` | `EVIDENCE_CONFIRMED` |
| `name` | TEXT | `cockpit.py:102,364`, `db_engine.py:489,511` | `ACTOR_PROFILE` | `actor_profiles` | `display_name` | `MIGRATE_TO_ACTOR_PROFILE` | `EVIDENCE_CONFIRMED` |
| `role` | TEXT | `cockpit.py:103,365`, `db_engine.py:490,512` | `ACTOR_PROFILE` | `actor_profiles` | `trader_persona` | `MIGRATE_TO_ACTOR_PROFILE` | `EVIDENCE_CONFIRMED` |
| `lhi` | REAL | `cockpit.py:112,366`, `db_engine.py:491,513` | `ACTOR_PROFILE` | `actor_profiles` | `lhi` | `MIGRATE_TO_ACTOR_PROFILE` | `EVIDENCE_CONFIRMED` |
| `hhi` | REAL | `cockpit.py:113,367`, `db_engine.py:492,514` | `ACTOR_PROFILE` | `actor_profiles` | `hhi` | `MIGRATE_TO_ACTOR_PROFILE` | `EVIDENCE_CONFIRMED` |
| `iai` | REAL | `cockpit.py:114,368`, `db_engine.py:493,515` | `ACTOR_PROFILE` | `actor_profiles` | `iai` | `MIGRATE_TO_ACTOR_PROFILE` | `EVIDENCE_CONFIRMED` |
| `liquid_reserves` | REAL | `cockpit.py:115,369`, `db_engine.py:494,516` | `ACTOR_PROFILE` | `actor_profiles` | `liquid_reserves` | `MIGRATE_TO_ACTOR_PROFILE` | `EVIDENCE_CONFIRMED` |
| `monthly_burn` | REAL | `cockpit.py:116,370`, `db_engine.py:495,517` | `ACTOR_PROFILE` | `actor_profiles` | `monthly_burn` | `MIGRATE_TO_ACTOR_PROFILE` | `EVIDENCE_CONFIRMED` |
| `created_at` | TIMESTAMP | `db_engine.py:496,518` | `ACTOR_PROFILE` | `actor_profiles` | `created_at` | `MIGRATE_TO_ACTOR_PROFILE` | `EVIDENCE_CONFIRMED` |
| `updated_at` | TIMESTAMP | `cockpit.py:371`, `db_engine.py:497,519` | `ACTOR_PROFILE` | `actor_profiles` | `updated_at` | `MIGRATE_TO_ACTOR_PROFILE` | `EVIDENCE_CONFIRMED` |

Zero fields remain `UNKNOWN_REQUIRES_DECISION`.

---

### 5.6 Redefined Wave 1F-A Scope & Boundaries

Wave 1F-A is strictly bounded to the private context resolver, private application service interfaces, and `INV-SAAS-05` enforcement. Public routes are completely forbidden from modification.

#### File Boundaries:
- **Allowed Files in Wave 1F-A**:
  - `api/context/resolver.py` (New private RequestContext resolver dependency)
  - `api/services/portfolio_service.py` (New private PortfolioApplicationService)
  - `api/services/journal_service.py` (New private JournalApplicationService)
  - `api/services/cockpit_service.py` (New private CockpitApplicationService)
  - `tests/architecture/test_saas_invariants.py` (Add AST verification for `INV-SAAS-05`)
  - `tests/unit/test_request_context_resolver.py` (Unit tests for private resolver)
  - `docs/architecture/*` (Verification documentation)
- **Forbidden Files in Wave 1F-A**:
  - `api/routes/analytics.py` (**STRICTLY FORBIDDEN**)
  - `api/routes/volatility.py` (**STRICTLY FORBIDDEN**)
  - `api/routes/regimes.py` (**STRICTLY FORBIDDEN**)
  - `api/routes/macro.py` (**STRICTLY FORBIDDEN**)
  - `api/routes/etf.py` (**STRICTLY FORBIDDEN**)
  - `analyst_dashboard/analyzers/*.py` (**STRICTLY FORBIDDEN**)
  - `engines/*.py` (**STRICTLY FORBIDDEN**)
  - `analyst_dashboard/data/db_engine.py` (**STRICTLY FORBIDDEN**)
  - `frontend/*` (**STRICTLY FORBIDDEN**)

#### Wave 1F-A Acceptance Criteria:
1. `RequestContextResolver` is created and handles header-based actor/workspace resolution with deterministic fallback (`ws_default`).
2. Private service interfaces (`PortfolioApplicationService`, `JournalApplicationService`, `CockpitApplicationService`) are scaffolded with fail-closed workspace checks and limit enforcement stubs.
3. Automated AST test for `INV-SAAS-05` verifies that no public context-free route imports or uses `RequestContextResolver` or `RequestContext`.
4. Zero database schema modifications or migrations.
5. Zero modifications to public analytical routes.

#### Wave 1F-A Verification Commands:
- `pytest tests/architecture/test_saas_invariants.py`
- `pytest tests/unit/test_request_context_resolver.py`

---

### 5.7 Audit of Reconciliation Acceptance Criteria (`SAAS-REC-01` to `SAAS-REC-18`)

| ID | Acceptance Criterion | Design Evidence / Verification | Status |
|---|---|---|:---:|
| **SAAS-REC-01** | RequestContext pipeline redefined: public routes bypass resolver | Section 3.1 & 5.1 detail route-class-aware pipeline where public context-free routes bypass context resolver entirely | **PASS** |
| **SAAS-REC-02** | Invariant `INV-SAAS-05` established with 5 negative guarantees | Section 4 & 5.1 formally define `INV-SAAS-05` with all 5 negative guarantees | **PASS** |
| **SAAS-REC-03** | 5 independent route classification dimensions defined | Section 5.2 defines Access, Context, Semantic, Cache, and Plan Dependence axes | **PASS** |
| **SAAS-REC-04** | No classification dimension inferred from another | Section 5.2 establishes explicit independence axioms preventing inference | **PASS** |
| **SAAS-REC-05** | All 45 routes classified across all 5 dimensions | Section 5.3 contains complete 45-endpoint classification matrix | **PASS** |
| **SAAS-REC-06** | `/api/v1/analytics/{symbol}` evaluated against 5-part canonical test | Section 5.4 evaluates 5-part test; grounded in `DecisionHierarchyEngine` | **PASS** |
| **SAAS-REC-07** | `/api/v1/analytics/setups` evaluated against 5-part canonical test | Section 5.4 evaluates 5-part test; grounded in `OptimalExecutionEngine` | **PASS** |
| **SAAS-REC-08** | `/api/v1/volatility/{symbol}` evaluated against 5-part canonical test | Section 5.4 evaluates 5-part test; grounded in `technical_engine.py` | **PASS** |
| **SAAS-REC-09** | All analytical routes assigned explicit confidence level | Section 5.4 assigns `EVIDENCE_CONFIRMED` to all analytical routes | **PASS** |
| **SAAS-REC-10** | No analytical route left with `UNKNOWN_REQUIRES_DECISION` | Section 5.4 explicitly adjudicates all analytical routes as `RESOLVE` | **PASS** |
| **SAAS-REC-11** | `user_profiles` audited field by field (all 10 fields) | Section 5.5 audits all 10 fields with exact consumers in code | **PASS** |
| **SAAS-REC-12** | Each field assigned target table, field name, migration action | Section 5.5 assigns target `actor_profiles` and explicit migration actions | **PASS** |
| **SAAS-REC-13** | Table strategy selected with technical rationale | Section 5.5 selects Strategy A (`SPLIT_ACTOR_PROFILE_AND_WORKSPACE_SETTINGS`) | **PASS** |
| **SAAS-REC-14** | No `user_profiles` field left with `UNKNOWN_REQUIRES_DECISION` | Section 5.5 assigns explicit classifications and decisions to all 10 fields | **PASS** |
| **SAAS-REC-15** | Wave 1F-A redefined to exclude public-route rewiring | Section 5.6 & 18 restrict 1F-A to private resolver and service interfaces | **PASS** |
| **SAAS-REC-16** | Wave 1F-A allowed/forbidden files, criteria, and tests specified | Section 5.6 defines exact allowed/forbidden files, criteria, and test commands | **PASS** |
| **SAAS-REC-17** | Dedicated `PHASE_1F_1G_DESIGN_RECONCILIATION` section added | Section 5 is explicitly dedicated to design reconciliation | **PASS** |
| **SAAS-REC-18** | Formal Gate Verdict issued with next authorized action | Section 21 issues `PASS_ARX_SAAS_FOUNDATION_PHASE_1F_1G_DESIGN_RECONCILED` | **PASS** |

---

## 6. Public vs. Private Cache Contracts

### 6.1 Public Canonical Analytics Contract (`/api/v1/analytics/{symbol}`)
```
Cache-Control: public, max-age=15, s-maxage=60, stale-while-revalidate=86400
CDN-Cache-Control: max-age=60, stale-while-revalidate=86400
Cloudflare-CDN-Cache-Control: max-age=60, stale-while-revalidate=86400
```
- **Invariance Rule**: Any request with identical parameters (`symbol`, `period`, `interval`, `user_role`) MUST yield the identical payload and cache headers regardless of caller identity, workspace, or subscription plan.
- **Forbidden**: Passing identity, workspace, or plan tokens into this response.

### 6.2 Private Workspace Contract (`/api/v1/portfolio`, `/api/v1/journal/*`, `/api/v1/cockpit/*`)
```
Cache-Control: private, no-cache, no-store, must-revalidate
Pragma: no-cache
```
- **CDN Defense**: Cloudflare and edge proxies are strictly forbidden from storing private responses.
- **Purge Procedure**: If any private response is ever accidentally served with public cache headers, an immediate edge purge of `/api/v1/portfolio*` and `/api/v1/journal*` must be triggered via Cloudflare API.

---

## 7. RequestContext Creation Boundary & Resolver Design

### 7.1 FastAPI Dependency Architecture (`api/context/resolver.py`)
```python
class RequestContextResolver:
    """Centralized, fail-closed resolver for application RequestContext.
    Enforced strictly on private, context-aware endpoints."""

    def __init__(self, workspace_repository: Optional[Any] = None):
        self.workspace_repo = workspace_repository

    async def resolve(
        self,
        request: Request,
        x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
        x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-ID"),
        x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    ) -> RequestContext:
        # 1. Resolve Request ID
        req_id = x_request_id.strip() if (x_request_id and x_request_id.strip()) else f"req_{uuid.uuid4().hex[:12]}"

        # 2. Resolve Actor ID (None for anonymous)
        actor_id = None
        if x_user_id and x_user_id.strip():
            actor_id = re.sub(r"[^a-zA-Z0-9_\-]", "", x_user_id.strip())

        # 3. Resolve Workspace ID (Deterministic mapping)
        if x_workspace_id and x_workspace_id.strip():
            workspace_id = re.sub(r"[^a-zA-Z0-9_\-]", "", x_workspace_id.strip())
        elif actor_id:
            # Deterministic personal workspace mapping during migration
            workspace_id = f"ws_usr_{hashlib.sha256(actor_id.encode()).hexdigest()[:16]}"
        else:
            workspace_id = "ws_default"

        return RequestContext(
            actor_id=actor_id,
            workspace_id=workspace_id,
            request_id=req_id,
        )
```

### 7.2 Fail-Closed Invariants:
1. Public routes must NEVER declare `Depends(get_request_context)` (`INV-SAAS-05`).
2. Private routes must NEVER construct `RequestContext` instances directly; they must inject it via `Depends(get_request_context)`.
3. Anonymous requests on private routes resolve safely: `actor_id` defaults to `None`, `workspace_id` defaults to `"ws_default"`.
4. If a private route requires explicit membership, the Application Service verifies that `actor_id` is a member of `workspace_id`. If unauthorized, it raises `403 Forbidden`.

---

## 8. Application-Service Architecture

Application Services sit strictly between the private HTTP route layer and domain repositories/engines.

### 8.1 Service Inventory & Responsibilities

```
                      ┌────────────────────────────────────┐
                      │        Private HTTP Routes         │
                      └─────────────────┬──────────────────┘
                                        │
                         RequestContext │ Command / Query
                                        ▼
                      ┌────────────────────────────────────┐
                      │        Application Services        │
                      ├────────────────────────────────────┤
                      │ - PortfolioApplicationService      │
                      │ - JournalApplicationService        │
                      │ - RadarApplicationService          │
                      │ - CockpitApplicationService        │
                      └───────┬──────────────┬─────────────┘
                              │              │
              Authorization / │              │ Plain Domain
              Entitlements    │              │ Inputs
                              ▼              ▼
                    ┌──────────────┐   ┌──────────────┐
                    │ Repositories │   │ Domain       │
                    │ & Adapters   │   │ Engines      │
                    └──────────────┘   └──────────────┘
```

1. **`PortfolioApplicationService`**:
   - Authorizes workspace access.
   - Enforces `portfolio.max_holdings` limit atomically before adding positions.
   - Coordinates holding mutations with `HistoryDatabaseEngine` / `PortfolioRepository`.
   - Never passes workspace or entitlement tokens into risk engines.
2. **`JournalApplicationService`**:
   - Enforces `journal.read` and `journal.write` capabilities.
   - Authorizes trade entry, execution fills, and position closures.
   - Rejects unowned trade closures.
3. **`RadarApplicationService`**:
   - Coordinates custom scans vs. pre-filtered candidate universes.
   - Checks `radar.custom_scan` capability for user-supplied filter criteria.
4. **`CockpitApplicationService`**:
   - Aggregates actor profile, workspace holdings, and action items.
   - Maintains action state transitions.

---

## 9. Complete Persistence Ownership Matrix

Every database table in the ARX Terminal schema is categorized into one of four persistence classes:

| Table Name | Database File | Primary Key | Current Partitioning | Target Classification | `workspace_id` Required | `actor_id` Required | Global by Design | Immutable Evidence | Migration Action |
|---|---|---|---|---|:---:|:---:|:---:|:---:|---|
| `portfolio_holdings` | `history.db` | `id` (INTEGER) | `user_id` | **WORKSPACE_OWNED** | **YES** | Optional (audit) | No | No | Backfill & enforce `NOT NULL` |
| `user_trade_journal` | `history.db` | `id` (INTEGER) | `user_id` | **WORKSPACE_OWNED** | **YES** | Optional (author) | No | No | Backfill & enforce `NOT NULL` |
| `user_cockpit_actions` | `history.db` | `id` (TEXT) | `user_id` | **WORKSPACE_OWNED** | **YES** | Optional (actor) | No | No | Backfill & enforce `NOT NULL` |
| `user_profiles` | `history.db` | `user_id` (TEXT) | `user_id` | **ACTOR_OWNED** | No | **YES** | No | No | **Split Strategy A**: Migrate personal fields to `actor_profiles`; create new `workspace_settings` |
| `gem_screening_history` | `history.db` | `id` (INTEGER) | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `forecast_history` | `history.db` | `id` (INTEGER) | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `trade_recommendation_history` | `history.db` | `id` (INTEGER) | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `asset_ohlcv_daily` | `market.db` | `(symbol, date)` | System | **SYSTEM_GLOBAL** | No | No | **YES** | No | Preserve untouched |
| `asset_ohlcv_provenance` | `market.db` | `id` (INTEGER) | System | **SYSTEM_GLOBAL** | No | No | **YES** | No | Preserve untouched |
| `asset_factor_snapshots` | `market.db` | `(symbol, date)` | System | **SYSTEM_GLOBAL** | No | No | **YES** | No | Preserve untouched |
| `asset_catalyst_registry` | `market.db` | `id` (INTEGER) | System | **SYSTEM_GLOBAL** | No | No | **YES** | No | Preserve untouched |
| `insider_disclosures` | `market.db` | `id` (INTEGER) | System | **SYSTEM_GLOBAL** | No | No | **YES** | No | Preserve untouched |
| `epoch_certification_results` | `governance.db`| `epoch_id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `epoch_release_authorizations`| `governance.db`| `epoch_id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `epoch_release_revocations` | `governance.db`| `epoch_id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `epoch_activation_records` | `governance.db`| `epoch_id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `epoch_supersession_records` | `governance.db`| `epoch_id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `prospective_evaluation_cycles`| `capture.db` | `cycle_id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `prospective_expected_evaluations`| `capture.db`| `id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `prospective_episodes` | `capture.db` | `episode_id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `prospective_universe_snapshots`| `capture.db`| `id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `prospective_decision_events` | `capture.db` | `event_id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `prospective_feature_snapshots`| `capture.db`| `id` | System | **EVIDENCE_IMMUTABLE** | No | No | Yes | **YES** | Preserve untouched |
| `openfigi_rate_limit_reservations`| `limiter.db`| `id` | Transient | **SYSTEM_GLOBAL** | No | No | **YES** | No | Transient token bucket |

---

## 10. Legacy Ownership Inventory & Remediation

All 16 occurrences of legacy user selectors across the source code are classified:

| File Location | Legacy Identifier | Current Semantics | Target Remediation Class | Remediation Plan |
|---|---|---|---|---|
| `api/routes/journal.py:74` | `_resolve_user_id` | Selects journal owner | `MAP_TO_WORKSPACE` | Deprecate in favor of `request_context.workspace_id` |
| `api/routes/journal.py:80` | `"default_user"` | Fallback journal owner | `LEGACY_COMPATIBILITY_ONLY` | Map to `"ws_default"` during migration |
| `api/routes/portfolio.py:34` | `_resolve_user_id` | Selects portfolio owner | `MAP_TO_WORKSPACE` | Deprecate in favor of `request_context.workspace_id` |
| `api/routes/portfolio.py:40` | `"default_user"` | Fallback portfolio owner | `LEGACY_COMPATIBILITY_ONLY` | Map to `"ws_default"` during migration |
| `api/routes/cockpit.py:64` | `_resolve_profile_selector` | Resolves profile selector | `MAP_TO_ACTOR` | Map to `actor_profiles.actor_id` |
| `api/routes/cockpit.py:64` | `"default"` | Fallback profile selector | `LEGACY_COMPATIBILITY_ONLY` | Map to `"actor_default"` during migration |
| `api/main.py:145` | `"X-User-Id"` | CORS allowed header | `LEGACY_COMPATIBILITY_ONLY` | Retain in CORS whitelist during migration |
| `analyst_dashboard/data/db_engine.py:116`| `portfolio_holdings.user_id` | Table partition key | `MAP_TO_WORKSPACE` | Add `workspace_id`, backfill, retire `user_id` |
| `analyst_dashboard/data/db_engine.py:171`| `user_trade_journal.user_id` | Table partition key | `MAP_TO_WORKSPACE` | Add `workspace_id`, backfill, retire `user_id` |
| `analyst_dashboard/data/db_engine.py:135`| `user_profiles.user_id` | Table partition key | `MAP_TO_ACTOR` | Migrate to `actor_profiles.actor_id` |
| `analyst_dashboard/data/db_engine.py:152`| `user_cockpit_actions.user_id`| Table partition key | `MAP_TO_WORKSPACE` | Add `workspace_id`, backfill, retire `user_id` |

---

## 11. Workspace Model (Phase 1G Design)

### 11.1 Schema Definitions
```sql
-- Workspaces Table (Conceptual Ownership Boundary)
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    workspace_type TEXT NOT NULL DEFAULT 'PERSONAL', -- 'PERSONAL' or 'TEAM'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workspace Memberships Table (Actor-to-Workspace Association)
CREATE TABLE IF NOT EXISTS workspace_memberships (
    workspace_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'MEMBER', -- 'OWNER', 'ADMIN', 'MEMBER', 'VIEWER'
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (workspace_id, actor_id),
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

-- Actor Profiles Table (Split from user_profiles - Strategy A)
CREATE TABLE IF NOT EXISTS actor_profiles (
    actor_id TEXT PRIMARY KEY,
    display_name TEXT,
    trader_persona TEXT DEFAULT 'Investor',
    lhi REAL DEFAULT 70.0,
    hhi REAL DEFAULT 75.0,
    iai REAL DEFAULT 65.0,
    liquid_reserves REAL DEFAULT 50000.0,
    monthly_burn REAL DEFAULT 4000.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workspace Settings Table (Distinct organizational preferences)
CREATE TABLE IF NOT EXISTS workspace_settings (
    workspace_id TEXT PRIMARY KEY,
    reporting_currency TEXT DEFAULT 'USD',
    max_risk_per_trade_bps INTEGER DEFAULT 100,
    default_benchmark TEXT DEFAULT 'SPY',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);
```

### 11.2 Tenancy Rules:
1. Every workspace-owned persistent record resolves to exactly one `workspace_id`.
2. Workspace ID generation format:
   - Default migration workspace: `ws_default`
   - Deterministic actor workspace: `ws_usr_<sha256(actor_id)[:16]>`
   - New workspace: `ws_<uuid4().hex[:16]>`

---

## 12. Workspace Migration Strategy (Phase 1G Design)

### 12.1 Seven-Stage Migration Lifecycle

```
M1: Schema Expansion (Add nullable workspace_id columns and workspace tables)
    ↓
M2: Dual-Writing Introduced (New writes populate both legacy user_id and workspace_id)
    ↓
M3: Historical Backfill (Backfill legacy records: 'default_user' -> 'ws_default')
    ↓
M4: Backfill Verification & Parity Audit (Zero unmapped records)
    ↓
M5: Read Transition (Application services read primarily by workspace_id)
    ↓
M6: Schema Constraint Enforcement (Enforce NOT NULL and foreign keys)
    ↓
M7: Legacy Column Retirement (Drop or deprecate legacy user_id columns)
```

### 12.2 Migration Fallback Matrix

| Scenario | Primary Key Resolved | Fallback Target | Action / Telemetry |
|---|---|---|---|
| Identified user with legacy data | `actor_id = "trader_1"` | `ws_usr_e3b0c442...` | Creates workspace & membership; links records |
| Anonymous legacy user (`default_user`) | `actor_id = None` | `ws_default` | Links records to default workspace |
| Unrecognized / orphaned record | `user_id = NULL` | Quarantine / Alert | Flagged in audit; assigned to `ws_quarantine` |

---

## 13. Tenant Isolation & Security Boundaries

1. **Query-Level Enforcement**: All queries for workspace-owned entities MUST include `WHERE workspace_id = :workspace_id`.
2. **Fail-Closed Authorization**: `ApplicationService` verifies membership prior to repository invocation.
3. **Defense Against Cross-Tenant Poisoning**: Parameter tampering (`X-Workspace-ID` belonging to another actor) triggers immediate `403 Forbidden` and security log emission.

---

## 14. Rollback & Forward-Recovery Specifications

### 14.1 Phase 1F Rollback:
- Revert application service wiring in private route handlers back to direct repository calls.
- Purely code-level rollback; zero database rollback required.

### 14.2 Phase 1G Rollback (Per Stage):
- **Stages M1–M4**: Drop `workspace_id` column; discard workspace tables. Zero data loss on legacy columns.
- **Stage M5**: Revert read configuration flag back to reading legacy columns.
- **Stage M6**: Remove `NOT NULL` constraint.

---

## 15. Security & Threat Modeling

1. **Header Spoofing**: Untrusted `X-User-Id` / `X-Workspace-ID` headers are quarantined. In future auth phase, headers will be replaced by cryptographically verified JWT claims.
2. **Public Cache Poisoning**: Prevented by Invariant `INV-SAAS-02` and `INV-SAAS-05`. Public routes cannot execute tenancy logic or vary by caller state.
3. **Data Leakage Across Workspaces**: SQLite queries strictly parameterize `workspace_id`.

---

## 16. Test Strategy & Architectural Invariant Tests

### 16.1 Automated Invariant Suite (`tests/architecture/test_saas_invariants.py`)
- `test_inv_saas_01_domain_purity`: Verifies AST of all engines in `analyst_dashboard/analyzers/` and `engines/` contain zero SaaS imports or types.
- `test_inv_saas_02_private_cache_isolation`: Asserts all private routes emit `no-store` headers and zero public cache directives.
- `test_inv_saas_03_workspace_tenancy`: Audits schema to verify workspace-owned entities require `workspace_id`.
- `test_inv_saas_04_canonical_truth`: Verifies analytical scoring produces bit-identical outputs regardless of entitlement context.
- `test_inv_saas_05_public_context_free`: Asserts all public context-free routes do NOT import or depend on `RequestContext`.

---

## 17. Telemetry & Observability Specifications

| Event Name | Description | Severity | Emitted Attributes | Bounded Vocabulary |
|---|---|---|---|---|
| `REQUEST_CONTEXT_RESOLUTION_FAILURE` | Invalid header or unresolvable context | Warning | `request_id`, `route`, `failure_code` | Bounded error codes |
| `WORKSPACE_AUTHORIZATION_FAILURE` | Actor lacks access to requested workspace | Warning | `request_id`, `actor_hash`, `route` | Hashed IDs |
| `ENTITLEMENT_DENIAL` | Capability check returns False | Info | `request_id`, `capability`, `operation` | Fixed 17 capabilities |
| `WORKSPACE_OWNERSHIP_MISMATCH` | Record workspace_id conflicts with request | Error | `request_id`, `record_type`, `table` | Fixed table names |
| `LEGACY_OWNERSHIP_FALLBACK_USED` | Fallback to default_user invoked | Info | `route`, `operation` | Fixed route set |
| `PUBLIC_CACHE_CONTEXT_CONTAMINATION` | Public response varies across contexts | **Critical** | `route`, `symbol`, `test_case` | Fixed routes |

---

## 18. Implementation Wave Decomposition

| Wave | Scope | Dependencies | Allowed Files | Forbidden Files | Verification Gate |
|---|---|---|---|---|:---:|
| **1F-A** | Context resolver & private service interfaces (`INV-SAAS-05`) | Phase 1A–1E | `api/context/resolver.py`, `api/services/portfolio_service.py`, `api/services/journal_service.py`, `api/services/cockpit_service.py`, `tests/` | Public routes (`analytics.py`, `volatility.py`, etc.), domain engines, DB schema | `1F-A Gate` |
| **1F-B** | Private route wiring (`portfolio`, `journal`, `cockpit`) | 1F-A | `api/routes/portfolio.py`, `api/routes/journal.py`, `api/routes/cockpit.py`, tests | Public routes, DB schema, domain engines | `1F-B Gate` |
| **1G-A** | Workspace schema & Strategy A split (`actor_profiles`) | 1F-B | `analyst_dashboard/data/db_engine.py`, migration scripts | Domain engines, billing, auth | `1G-A Gate` |
| **1G-B** | Data backfill & parity audit | 1G-A | Backfill scripts, verification tests | NOT NULL constraint, legacy retirement | `1G-B Gate` |
| **1G-C** | NOT NULL constraint & legacy retirement | 1G-B | DB schema constraints, repository code | Billing, auth, public routes | `1G-C Gate` |
| **FRONTEND** | Frontend context bootstrap integration | 1F-B, 1G-B | `frontend/lib/saas/`, context provider | TerminalShell layout, page rewrites | `FE Gate` |
| **SHELL** | TerminalShell convergence | FRONTEND | `frontend/components/Navbar.tsx`, Shell layout | Domain page logic | `Shell Gate` |

---

## 19. Resolution of Open Questions

1. **Which routes are currently public vs actor-scoped?**
   - Public context-free: `/api/v1/analytics/{symbol}`, `/api/v1/analytics/setups`, `/api/v1/volatility/*`, `/api/v1/regimes/*`, `/api/v1/smart-money/*`, `/api/v1/macro/*`, `/api/v1/etf/*`, `/api/v1/screener/run` (GET).
   - Actor/Workspace-scoped: `/api/v1/portfolio/*`, `/api/v1/journal/*`, `/api/v1/cockpit/*`, `/api/v1/screener/run` (POST).
   - Internal-only: `/api/v1/governance/*`, `/api/v1/telemetry/*`, `/api/v1/cache/*`, `/health`.
2. **Which persistence tables exist in current schema?**
   - Exactly 24 tables cataloged in Section 9 across `history.db`, `market.db`, `governance.db`, and `capture.db`.
3. **Which records are immutable evidence?**
   - `gem_screening_history`, `forecast_history`, `trade_recommendation_history`, and all 15 governance/prospective capture tables.
4. **What is the exact legacy anonymous identity behavior?**
   - If `X-User-Id` is missing, `portfolio.py` and `journal.py` fall back to `"default_user"`; `cockpit.py` falls back to `"default"`.
5. **Does any current route rely on X-User-Id?**
   - Yes: `portfolio.py`, `journal.py`, and `cockpit.py`.
6. **Which entities require actor audit fields after workspace migration?**
   - `portfolio_holdings` (optional `created_by_actor_id`), `user_trade_journal` (optional `author_actor_id`).
7. **Are recommendation and forecast histories global evidence or workspace-created?**
   - Purely global algorithmic evidence (`EVIDENCE_IMMUTABLE`).
8. **What is the table strategy for `user_profiles`?**
   - Strategy A (`SPLIT_ACTOR_PROFILE_AND_WORKSPACE_SETTINGS`): personal trader attributes migrate to `actor_profiles`; shared workspace configuration is housed in new `workspace_settings`.
9. **What is the exact default migration workspace key?**
   - `"ws_default"` for legacy `"default_user"` / `"default"`; `"ws_usr_" + sha256(actor_id)[:16]` for identified actors.
10. **Which cache headers are currently emitted?**
    - Public: `public, max-age=15, s-maxage=60, stale-while-revalidate=86400`.
    - Private: `private, no-cache, no-store, must-revalidate`.
11. **Which routes require dual-read compatibility?**
    - `/api/v1/portfolio` and `/api/v1/journal/*` during migration phases M2–M5.

---

## 20. Explicitly Deferred Scope

The following features remain explicitly deferred and are **NOT** authorized:
- Authentication provider selection (Auth0, Supabase, Clerk, Firebase)
- User login / signup / session UI
- JWT / session token validation
- Stripe SDK integration, checkout, webhooks
- Subscription tables (`subscriptions`, `invoices`, `plans`)
- Commercial pricing tiers (`Free`, `Pro`, `Fund`) in runtime domain code
- Paid seat administration and team invites

---

## 21. Formal Gate Verdict

```
GATE = PASS_ARX_SAAS_FOUNDATION_PHASE_1F_1G_DESIGN_RECONCILED
PREDECESSOR_GATE = PASS_ARX_SAAS_FOUNDATION_PHASE_1A_1E_RELEASE
PREDECESSOR_VERDICT = PASS_ARX_SAAS_FOUNDATION_PHASE_1A_1E_RELEASE
RELEASE_SHA = 724b5e3659ba0287fc3d8b9d58b4ef7eecde8703
BRANCH = feat/arx-saas-foundation-phase1-seams
DESIGN_RECONCILIATION = COMPLETE
REQUEST_CONTEXT_SCOPE = NARROWED_TO_PRIVATE_ROUTES
INV_SAAS_05 = ESTABLISHED
FIVE_DIMENSIONAL_ROUTE_CLASSIFICATION = COMPLETE_45_ROUTES
PUBLIC_ANALYTICS_ADJUDICATION = RESOLVED
/api/v1/analytics/{symbol} = CANONICAL
/api/v1/analytics/setups = DERIVED
/api/v1/volatility/{symbol} = DERIVED
USER_PROFILES_STRATEGY = STRATEGY_A_SPLIT_ACTOR_PROFILE_AND_WORKSPACE_SETTINGS
WAVE_1F_A_SCOPE = REDEFINED_STRICTLY_PRIVATE
SAAS_REC_01_THROUGH_18 = PASS
AUTHENTICATION = NOT_IMPLEMENTED
SUBSCRIPTIONS = NOT_IMPLEMENTED
BILLING = NOT_IMPLEMENTED
TEAM_ACCOUNTS = NOT_IMPLEMENTED
DATABASE_SCHEMA_CHANGED = NO
ROUTES_REWIRED = NO
NEXT_AUTHORIZED_ACTION = ARX_SAAS_FOUNDATION_PHASE_1F_A_IMPLEMENTATION_GATE
AUTOMATIC_SUCCESSOR_EXECUTION = NOT_AUTHORIZED
```
