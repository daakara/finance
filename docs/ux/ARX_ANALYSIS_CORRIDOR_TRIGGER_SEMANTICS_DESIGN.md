# ARX TERMINAL — ANALYSIS
## CORRIDOR & TRIGGER SEMANTICS — AUTHORITY RECONCILIATION GATE

```ini
GATE =
  PASS_ARX_ANALYSIS_CORRIDOR_TRIGGER_SEMANTICS_RECONCILIATION
BACKLOG_ITEM =
  ARX_UX_RADAR_ANALYSIS_PORTFOLIO_DECISION_LIFECYCLE
SUBITEM =
  ANALYSIS_CORRIDOR_TRIGGER_SEMANTICS
DESIGN_STATE =
  CLOSED_VERIFIED_AND_FROZEN
CORRIDOR_AUTHORITY =
  ESTABLISHED (OptimalExecutionEngine.optimal_entry_min / optimal_entry_max)
TRIGGER_MODEL =
  STATE_BASED_READINESS
CANONICAL_NUMERIC_TRIGGER =
  NOT_ESTABLISHED
TRIGGER_PRICE_AUTHORITY =
  NONE
TRIGGER_STATE_AUTHORITY =
  ESTABLISHED (DecisionHierarchyEngine.resolve_decision_state + OptimalExecutionEngine.execution_status)
ACTIONABILITY_AUTHORITY =
  ESTABLISHED (DecisionHierarchyEngine + OptimalExecutionEngine)
SUPPORTED_DIRECTIONALITY =
  LONG_ONLY
SHORT_SEMANTICS =
  NOT_APPLICABLE
SHORT_UI_STATE =
  PRESENT_BUT_UNSUPPORTED
SHORT_UI_RECOMMENDATION =
  DISABLE
FRONTEND_COMPETING_DOMAIN_LOGIC =
  IDENTIFIED
CORRIDOR_DRIFT =
  YES (HISTORICAL_IN_CODEBASE) -> RESOLVED_IN_SPECIFICATION
QUANT_ENGINE_CHANGED =
  NO
SCORING_CHANGED =
  NO
RANKING_CHANGED =
  NO
PAPER_TRADING_CONTRACT_CHANGED =
  NO
IMPLEMENTATION_AUTHORIZED =
  NO
NEXT_AUTHORIZED_ACTION =
  ARX_ANALYSIS_CORRIDOR_TRIGGER_SEMANTICS_IMPLEMENTATION_GATE
AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```

---

## 0. Executive Purpose & Scope

This specification establishes the single, authoritative, reconciled semantic model for **Analysis Corridor, Trigger, Actionability, and Directionality Semantics** across ARX Terminal.

Prior code audits identified multiple competing definitions of "corridor" and "trigger":
1. **Entry Corridor Drift**: Backend `OptimalExecutionEngine` calculates `optimal_entry_min` and `optimal_entry_max`, while frontend utility `decisionHierarchyUtils.ts` independently re-derived conflicting corridor bounds from `stopLoss * 1.02` and `sma50`.
2. **Trigger Concept Conflation**: Different platform surfaces alternately referred to "trigger" as (a) an intra-corridor candlestick absorption confirmation (`is_stabilized`), (b) an opening-print / pullback condition in paper trading outcome evaluation, (c) a Stage 4 breakout pivot level (`breakout_pivot`), (d) an order entry limit price (`entryPivot`), or (e) an actionability gate (`isActionable`).
3. **Distance Calculation Ambiguity**: Inverted signs in `ExecutionCorridor.tsx` rendered confusing strings like `-3.2% below pivot`.
4. **Unsupported Directionality**: The backend quantitative engine is strictly long-only, while `DayTraderPositionSizer.tsx` exposed an ungrounded client-side `SHORT` toggle that manufactured synthetic short targets.

This reconciliation gate resolves all competing meanings, freezes the canonical backend/frontend contracts, enforces long-only invariants, and prohibits premature mutations.

---

## 1. Scope, Frozen Authorities, and Decision Path

### 1.1 Complete Production Decision Path
The production recommendation and actionability lifecycle traverses eight sequential layers:

```
1. Market Data Assessment
   ├─ MarketPriceState (analyst_dashboard/data/market_price_state.py)
   │    └─ Ingests liveSpotPrice (Alpaca WebSocket/REST) & analysisReferencePrice (completed daily session)
   └─ LiquidityGuard (analyst_dashboard/analyzers/liquidity_guard.py)
        └─ Audits trading volume, spread, and liquidity defense
   ↓
2. Entry Corridor Calculation
   └─ OptimalExecutionEngine.calculate_trade_levels() (analyst_dashboard/analyzers/optimal_execution.py)
        └─ Computes: optimal_entry_min, optimal_entry_max
   ↓
3. Stop / Target Geometry
   └─ OptimalExecutionEngine.calculate_trade_levels()
        ├─ Computes: stop_loss, stop_loss_pct (anchored below swing support / ATR)
        ├─ Computes: take_profit_1, take_profit_2 (enforcing institutional R:R >= 1.85:1)
        └─ Evaluates: is_stabilized (buyer absorption candle) & execution_status
   ↓
4. Multi-Factor Confluence
   └─ ConfluenceEngine.calculate_confluence() (analyst_dashboard/analyzers/confluence_engine.py)
        └─ Emits: confluenceScore (0 - 100) across 5 institutional pillars
   ↓
5. Decision State Resolution
   └─ DecisionHierarchyEngine.resolve_decision_state() (analyst_dashboard/analyzers/decision_hierarchy.py)
        └─ Resolves 6-state mutually exclusive precedence:
           UNVERIFIED (P1) → INSUFFICIENT_DATA (P2) → STALE_DATA (P3) →
           EVIDENCE_INCOMPLETE (P4) → VALID_SETUP (P5) → ACTIONABLE_SETUP (P6)
   ↓
6. Actionability Gating
   ├─ DecisionTraceEngine.build_decision_trace() (analyst_dashboard/analyzers/decision_trace.py)
   ├─ OptimalExecutionEngine._enforce_execution_invariants() (plan["is_actionable"])
   └─ frontend/types/decisionContract.ts (isDecisionActionable)
        └─ Gating Rule: State must be ACTIONABLE_SETUP AND executionStatus in ACTIONABLE_EXECUTION_STATUSES
   ↓
7. Frontend Adaptation & API Serialization
   ├─ /analytics/{symbol} (api/routes/analytics.py)
   ├─ api.ts (frontend/lib/api.ts)
   └─ insightGenerator.ts (frontend/lib/insightGenerator.ts)
   ↓
8. Displayed Semantic Labels & CTA Behavior
   ├─ Analysis Page (frontend/app/page.tsx) & Terminal Views (Standard, Guided, Advanced)
   ├─ OptimalEntryExitCard (frontend/components/OptimalEntryExitCard.tsx)
   └─ CTA: "Size Position & Calculate Risk" (Enabled ONLY when isActionable === True)
```

### 1.2 Result-Affecting Field Registry

| Field | Source File | Source Symbol | Authority Layer | Result Effect | Competing Derivation? |
|---|---|---|---|---|---|
| `optimal_entry_min` | `analyst_dashboard/analyzers/optimal_execution.py` | `OptimalExecutionEngine.calculate_trade_levels` | `BACKEND` | Lower accumulation bound | **YES** (`stopLoss * 1.02` in `decisionHierarchyUtils.ts`) |
| `optimal_entry_max` | `analyst_dashboard/analyzers/optimal_execution.py` | `OptimalExecutionEngine.calculate_trade_levels` | `BACKEND` | Upper accumulation bound / Breakout ceiling | **YES** (`sma50` in `decisionHierarchyUtils.ts`; scalar `entryPivot` in `setups/page.tsx`) |
| `stop_loss` | `analyst_dashboard/analyzers/optimal_execution.py` | `OptimalExecutionEngine.calculate_trade_levels` | `BACKEND` | Setup invalidation floor | **NO** (Universally authoritative) |
| `take_profit_1` | `analyst_dashboard/analyzers/optimal_execution.py` | `OptimalExecutionEngine.calculate_trade_levels` | `BACKEND` | Primary target ($R:R \ge 1.85:1$) | **NO** (Universally authoritative) |
| `take_profit_2` | `analyst_dashboard/analyzers/optimal_execution.py` | `OptimalExecutionEngine.calculate_trade_levels` | `BACKEND` | Secondary extended target | **NO** (Universally authoritative) |
| `risk_reward_ratio` | `analyst_dashboard/analyzers/optimal_execution.py` | `OptimalExecutionEngine.calculate_trade_levels` | `BACKEND` | Blended institutional R:R ratio | **NO** (Universally authoritative) |
| `is_stabilized` | `analyst_dashboard/analyzers/optimal_execution.py` | `OptimalExecutionEngine.calculate_trade_levels` | `BACKEND` | Buyer absorption candle condition | **NO** (Evaluated internally) |
| `execution_status` | `analyst_dashboard/analyzers/optimal_execution.py` | `OptimalExecutionEngine.calculate_trade_levels` | `BACKEND` | Tactical buy zone posture | **YES** (`radar/page.tsx` invents `NEAR_PIVOT`, `VOLUME_DRYUP`, `AWAITING_TRIGGER`) |
| `breakout_pivot` | `analyst_dashboard/analyzers/optimal_execution.py` | `OptimalExecutionEngine.calculate_trade_levels` | `BACKEND` | Stage 4 50-SMA target anchor | **YES** (Conflated with trigger price in UI) |
| `confluence_score` | `analyst_dashboard/analyzers/confluence_engine.py` | `ConfluenceEngine.calculate_confluence` | `BACKEND` | Multi-factor conviction score (0-100) | **NO** (Universally authoritative) |
| `decision_state` | `analyst_dashboard/analyzers/decision_hierarchy.py` | `DecisionHierarchyEngine.resolve_decision_state` | `BACKEND` | 6-state institutional precedence | **NO** (Universally authoritative) |
| `is_actionable` | `frontend/types/decisionContract.ts` | `isDecisionActionable` | `SHARED_DOMAIN` | Capital sizing CTA gate | **YES** (`deriveUnmetConditions` checks local $\ge 70$ score) |

```ini
QUANT_ENGINE_CHANGE_AUTHORIZED =
  NO
SCORING_CHANGE_AUTHORIZED =
  NO
RANKING_CHANGE_AUTHORIZED =
  NO
OUTCOME_CONTRACT_CHANGE_AUTHORIZED =
  NO
```

---

## 2. Canonical Corridor, Trigger, and State Authority

### 2.1 Entry Corridor Authority Re-Attestation

```ini
ENTRY_CORRIDOR_MIN_AUTHORITY =
  OptimalExecutionEngine.optimal_entry_min (analyst_dashboard/analyzers/optimal_execution.py:204, 327)
ENTRY_CORRIDOR_MAX_AUTHORITY =
  OptimalExecutionEngine.optimal_entry_max (analyst_dashboard/analyzers/optimal_execution.py:205, 328)
ENTRY_CORRIDOR_AUTHORITY =
  SINGLE_CANONICAL
CORRIDOR_DRIFT =
  YES
```

#### Competing Derivations Identified
1. **`frontend/lib/decisionHierarchyUtils.ts` (Lines 24-27)**:
   ```typescript
   const minEntry = kl.stopLoss ? kl.stopLoss * 1.02 : price * 0.95;
   const maxEntry = kl.sma50 ? kl.sma50 : price * 1.02;
   const lower = Math.min(minEntry, maxEntry);
   const upper = Math.max(minEntry, maxEntry);
   const isInCorridor = price >= lower && price <= upper;
   ```
   *Defect*: Directly manufactures synthetic bounds, ignoring `optimal_entry_min` and `optimal_entry_max`.
2. **`api/routes/analytics.py` (Line 255)** & **`frontend/app/setups/page.tsx` (Line 219)**:
   ```python
   entry_pivot = plan.get("optimal_entry_max") or plan.get("breakout_pivot") or cur_price
   ```
   *Defect*: Collapses the two-sided accumulation corridor into a single scalar limit price.

### 2.2 Trigger Vocabulary and Authority Audit

| Concept ID | Source File | Source Symbol | Semantic Meaning | Used For | Numeric Value? | Canonical for Analysis? |
|---|---|---|---|---|---|---|
| `TRG-01` | N/A | `trigger_price` | Explicit numeric price to trigger entry | None (Does not exist in backend) | **NO** | **NO** |
| `TRG-02` | `optimal_execution.py` (L298) | `is_stabilized` | Candlestick buyer absorption condition: close in upper 45%, or close >= open/prev/EMA20 | Confirmation gate inside buy zone | **NO** (Boolean) | **YES** (Internal gate) |
| `TRG-03` | `optimal_execution.py` (L174, L229) | `breakout_pivot` | 50-day SMA ceiling used to re-anchor targets in Stage 4 corrections | Target calculation anchor | **YES** | **NO** (Not a trigger) |
| `TRG-04` | `analytics.py` (L255), `setups/page.tsx` (L219) | `entryPivot` | Upper bound of corridor (`optimal_entry_max`) | Tactical limit price (LMT) | **YES** | **NO** (Scalar alias of corridor max) |
| `TRG-05` | `paper_trading_outcome_evaluator.py` (L161-236) | Opening/pullback trigger | Price entering corridor on T+1 to T+5 sessions | Paper trading execution fill | **YES** | **NO** (Outcome model only) |
| `TRG-06` | `decision_hierarchy.py` (L158), `decisionContract.ts` (L53) | `isActionable` | Gating boolean: full evidence + Stage 2 + in buy zone + stabilized + R:R >= 2.0 | Position sizing authorization | **NO** (Boolean) | **YES** (Actionability authority) |
| `TRG-07` | `optimal_execution.py` (L308) | In-zone condition | `optimal_entry_min <= eval_price <= optimal_entry_max` | Zone presence classification | **NO** (Spatial condition) | **YES** |
| `TRG-08` | `decision_hierarchy.py` (L24) | `DecisionState` | 6-tier institutional state (`UNVERIFIED` to `ACTIONABLE_SETUP`) | Readiness classification | **NO** (Enum) | **YES** |

```ini
CANONICAL_NUMERIC_TRIGGER =
  NOT_ESTABLISHED
TRIGGER_PRICE_AUTHORITY =
  NONE
```

### 2.3 Trigger Price Versus Trigger State Classification

```ini
TRIGGER_STATE =
  YES
TRIGGER_PRICE =
  NO
ANALYSIS_TRIGGER_MODEL =
  STATE_BASED_READINESS
TRIGGER_STATE_AUTHORITY =
  analyst_dashboard.analyzers.decision_hierarchy.DecisionHierarchyEngine + OptimalExecutionEngine.execution_status
UI_TRIGGER_LABEL_AUTHORIZED =
  YES (FOR_READINESS_STATE_ONLY_E.G._"Awaiting Confirmation Trigger")
NUMERIC_TRIGGER_DISPLAY_AUTHORIZED =
  NO
```

*Policy*: The platform strictly prohibits inventing a numeric trigger price from corridor midpoint, corridor ceiling, stop multiplier, or moving average. ARX uses a **State-Based Readiness Model**.

### 2.4 Stabilization, Breakout, and Pivot Semantics

```ini
IS_STABILIZED_AUTHORITY =
  OptimalExecutionEngine.calculate_trade_levels (analyst_dashboard/analyzers/optimal_execution.py:281-298)
IS_STABILIZED_INPUTS =
  range_pos >= 0.45 OR last_close >= last_open OR last_close >= prev_close OR last_close >= ema_20
IS_STABILIZED_RESULT_EFFECT =
  If True -> execution_status = "IN_BUY_ZONE"
  If False -> execution_status = "IN_BUY_ZONE_AWAITING_TRIGGER"
IS_STABILIZED_EQUALS_TRIGGER =
  NO (It is a supporting buyer absorption gate, not an independent trigger price)

BREAKOUT_PIVOT_AUTHORITY =
  OptimalExecutionEngine.calculate_trade_levels (analyst_dashboard/analyzers/optimal_execution.py:174, 229)
BREAKOUT_PIVOT_ROLE =
  50-day SMA ceiling used strictly in Stage 4 corrections to re-anchor take-profit targets (TP1/TP2)
BREAKOUT_PIVOT_REQUIRED_FOR_ACTIONABILITY =
  NO
BREAKOUT_PIVOT_EQUALS_TRIGGER_PRICE =
  NO
```

### 2.5 Actionability, Decision State, and Session Semantics

```ini
ACTIONABILITY_AUTHORITY =
  DecisionHierarchyEngine.resolve_decision_state (analyst_dashboard/analyzers/decision_hierarchy.py:131)
  & OptimalExecutionEngine._enforce_execution_invariants (analyst_dashboard/analyzers/optimal_execution.py:552)
ACTIONABILITY_PREDICATE =
  confluence_score >= 75.0
  AND eval_price in [optimal_entry_min, optimal_entry_max]
  AND is_stabilized == True
  AND stage_phase is eligible (Stage 2 for Swing; Stage 2 or Intraday Momentum for Day)
  AND risk_reward_ratio >= 2.0
  AND stop_loss is not None
  AND optimal_entry_max is not None
ACTIONABILITY_DEPENDS_ON_CORRIDOR =
  YES
ACTIONABILITY_DEPENDS_ON_TRIGGER_PRICE =
  NO
ACTIONABILITY_DEPENDS_ON_STABILIZATION =
  YES
ACTIONABILITY_DEPENDS_ON_SESSION =
  NO
```

#### Actual Decision Hierarchy States (Precedence Order)
1. **`UNVERIFIED` (Precedence 1 - Highest)**: `current_price <= 0` or `candle_count == 0` or unverified asset. `isActionable = False`.
2. **`INSUFFICIENT_DATA` (Precedence 2)**: `candle_count < 50` sessions. `isActionable = False`.
3. **`STALE_DATA` (Precedence 3)**: Market tape $> 4$ calendar days old. `isActionable = False`.
4. **`EVIDENCE_INCOMPLETE` (Precedence 4)**: Audited SEC EDGAR 10-K/10-Q fundamentals missing. `isActionable = False`.
5. **`VALID_SETUP` (Precedence 5)**: Sound verified data, but outside buy zone, unstabilized, or awaiting breakout. `isActionable = False`.
6. **`ACTIONABLE_SETUP` (Precedence 6 - Lowest Precedence / Highest Criteria)**: Full evidence + Stage 2 + in buy zone + stabilized + $R:R \ge 2.0$. `isActionable = True`.

#### Actual Execution Statuses
- **`IN_BUY_ZONE`**: Inside corridor with confirmed stabilization (`ACTIONABLE`).
- **`READY_TO_BUY`**: Equivalent actionable confirmation state (`ACTIONABLE`).
- **`WAITING_PULLBACK`**: Price above corridor ceiling or awaiting reaction low (`NON_ACTIONABLE`).
- **`IN_BUY_ZONE_AWAITING_TRIGGER`**: Inside corridor but unconfirmed absorption (`NON_ACTIONABLE`).
- **`APPROACHING_TARGET`**: Price extended past corridor toward TP1 (`NON_ACTIONABLE`).
- **`STOPPED_OUT`**: Price breached stop loss floor (`NON_ACTIONABLE`).
- **`INSUFFICIENT_HISTORY`**: Insufficient sessions to compute execution levels (`NON_ACTIONABLE`).
- **`UNVERIFIED_ASSET`**: Missing exchange record (`NON_ACTIONABLE`).
- **`STALE_MARKET_DATA`**: Stale tape (`NON_ACTIONABLE`).

#### Market Session Role
```ini
MARKET_SESSION_AFFECTS_ANALYSIS_STATE =
  NO
MARKET_SESSION_AFFECTS_ACTIONABILITY =
  NO
MARKET_SESSION_AFFECTS_EXECUTION =
  YES (PassiveCaptureHook.record_natural_recommendation requires market_session == "REGULAR_SESSION")
MARKET_SESSION_AFFECTS_PRESENTATION =
  YES (Badge display of session state: REGULAR_SESSION, PRE_MARKET, AFTER_HOURS, CLOSED)
```

---

## 3. Paper-Trading Separation and Directionality

### 3.1 Paper-Trading Execution Separation

```ini
PAPER_TRADING_EXECUTION_TRIGGER =
  PaperTradingOutcomeEvaluator.evaluate_signal (analyst_dashboard/governance/paper_trading_outcome_evaluator.py:160-240)
ANALYSIS_TRIGGER_AUTHORITY =
  OptimalExecutionEngine.is_stabilized + DecisionHierarchyEngine.resolve_decision_state
SEMANTICALLY_IDENTICAL =
  NO
PAPER_TRIGGER_REUSED_AS_ANALYSIS_TRIGGER =
  NO
```

*Separation Boundary*: Paper-trading evaluates multi-day fill conditions across sessions $T+1$ through $T+5$ (opening gap, rally into corridor, pullback into corridor). Analysis evaluates point-in-time recommendation readiness at timestamp $T$. These distinct concepts must never be merged.

### 3.2 Directionality Semantics

```ini
ENGINE_DIRECTIONALITY =
  LONG_ONLY
ANALYSIS_DIRECTIONALITY =
  LONG_ONLY
DAY_TRADER_UI_DIRECTIONALITY =
  DUAL_LONG_AND_SHORT (Unsupported client toggle)
SHORT_ANALYSIS_SEMANTICS =
  NOT_APPLICABLE
SHORT_UI_STATE =
  PRESENT_BUT_UNSUPPORTED
SHORT_UI_RECOMMENDATION =
  DISABLE
RATIONALE =
  The backend quant engines (OptimalExecutionEngine, DecisionHierarchyEngine, ConfluenceEngine) strictly model Stage 2 long accumulation. Zero quantitative short models, short confluence weights, or short risk floors exist in the backend. Exposing an active SELL/SHORT toggle in DayTraderPositionSizer manufactures unsupported short levels directly in the client.
```

---

## 4. Frontend Authority and Presentation Contract

### 4.1 Frontend Calculations Inventory

| Frontend Calculation | Component / Location | Role | Can Change Domain Semantics? |
|---|---|---|---|
| Synthetic Corridor Bounds (`stopLoss * 1.02`, `sma50`) | `frontend/lib/decisionHierarchyUtils.ts` (L24-27) | `COMPETING_DOMAIN_LOGIC` | **YES** (Overwrites backend corridor) |
| Synthetic Radar Status (`NEAR_PIVOT`, `VOLUME_DRYUP`, `AWAITING_TRIGGER`) | `frontend/app/radar/page.tsx` (L108-115, L267-272) | `COMPETING_DOMAIN_LOGIC` | **YES** (Presents client-invented states) |
| Synthetic Short Targets & Stop Calculation | `frontend/components/DayTraderPositionSizer.tsx` (L95-109) | `COMPETING_DOMAIN_LOGIC` | **YES** (Invents unverified short levels) |
| Inverted Distance String (`-X% below pivot`) | `frontend/components/workstation/ExecutionCorridor.tsx` (L178) | `PRESENTATION_ONLY` | **NO** (Causes semantic confusion) |
| Scalar Entry Pivot Limit (`entryPivot = optimal_entry_max`) | `frontend/app/setups/page.tsx` (L219, L669) | `PRESENTATION_ONLY` | **NO** (Scalar limit approximation) |

### 4.2 Backend / Shared-Domain / Frontend Contract

```
┌────────────────────────────────────────────────────────────────────────┐
│                        BACKEND / SHARED DOMAIN                         │
│  • Owns canonical corridor bounds: optimal_entry_min, optimal_entry_max│
│  • Owns stop loss and profit target geometry: stop_loss, TP1, TP2      │
│  • Owns buyer absorption confirmation: is_stabilized                   │
│  • Owns execution status: execution_status (9 canonical statuses)      │
│  • Owns 6-state decision hierarchy: DecisionHierarchyEngine            │
│  • Owns actionability gate: isActionable (Fail-closed)                 │
│  • Owns directionality: LONG_ONLY                                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Canonical API Payload
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND PRESENTATION                         │
│  • Renders canonical values without modification                       │
│  • Strictly prohibited from locally computing corridor bounds          │
│  • Strictly prohibited from manufacturing numeric trigger prices       │
│  • Strictly prohibited from synthesizing SHORT trade levels            │
│  • Derives presentation-only labels and accessible badges              │
│  • Enforces CTA enablement: Size Trade active ONLY when isActionable   │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Distance and Corridor Display Semantics

```ini
DISTANCE_CALCULATION_AUTHORITY =
  OptimalExecutionEngine.calculate_trade_levels (analyst_dashboard/analyzers/optimal_execution.py:318-320)
DISTANCE_SIGN_CONVENTION =
  PRICE_MOVEMENT_REQUIRED (+ means price must rise to reach level; - means price must fall)
DISPLAY_WORDING =
  STANDARDIZED (Without double negatives)
DOUBLE_NEGATIVE_OR_DIRECTIONAL_AMBIGUITY =
  YES (PRESENT_IN_CURRENT_CODE: ExecutionCorridor.tsx renders "-3.2% below pivot")
CORRIDOR_DISPLAY_STATES =
  BELOW_ENTRY_CORRIDOR, IN_ENTRY_CORRIDOR, ABOVE_ENTRY_CORRIDOR (EXTENDED_BEYOND_ENTRY)
DERIVED_FROM_CANONICAL_CORRIDOR_ONLY =
  YES
```

---

## 5. Canonical Semantic Matrix and Invariants

### 5.1 Canonical Semantic Matrix

| Concept | Canonical Authority | Numeric Value? | Result-Affecting? | Frontend May Derive? | Analysis Display Allowed? |
|---|---|---|---|---|---|
| **Entry corridor min** | `OptimalExecutionEngine.optimal_entry_min` | **YES** | **YES** | **NO** | **YES** |
| **Entry corridor max** | `OptimalExecutionEngine.optimal_entry_max` | **YES** | **YES** | **NO** | **YES** |
| **Trigger price** | `NONE` (Explicitly absent) | **NO** | **NO** | **NO** | **NO** (Numeric label prohibited) |
| **Trigger readiness** | `DecisionHierarchyEngine` + `is_stabilized` | **NO** (State) | **YES** | **NO** | **YES** ("Awaiting Trigger") |
| **Stabilized** | `OptimalExecutionEngine.is_stabilized` | **NO** (Boolean) | **YES** | **NO** | **YES** (Absorption confirmation) |
| **Breakout pivot** | `OptimalExecutionEngine.breakout_pivot` | **YES** (Stage 4) | **YES** (Target anchor) | **NO** | **YES** (Stage 4 target ceiling) |
| **Actionable** | `DecisionHierarchyEngine` & `OptimalExecutionEngine` | **NO** (Boolean) | **YES** | **NO** | **YES** (Actionable Setup badge) |
| **Decision state** | `DecisionHierarchyEngine.resolve_decision_state` | **NO** (Enum) | **YES** | **NO** | **YES** (6 canonical states) |
| **Directionality** | `OptimalExecutionEngine` (Long-only) | **NO** (Enum) | **YES** | **NO** | **YES** (Long-only displayed) |
| **Distance to corridor**| `OptimalExecutionEngine.distance_to_entry_pct` | **YES** | **NO** (Presentation) | **YES** (If preserving sign) | **YES** |

### 5.2 Frozen Semantic Invariants

```ini
INV-ANALYSIS-01 =
  Analysis entry corridor comes from one canonical authority (OptimalExecutionEngine).

INV-ANALYSIS-02 =
  Frontend does not redefine entry-corridor geometry.

INV-ANALYSIS-03 =
  Analysis does not invent a numeric trigger when none exists canonically.

INV-ANALYSIS-04 =
  Paper-trading execution semantics do not redefine Analysis trigger semantics.

INV-ANALYSIS-05 =
  Stabilization is not equated with trigger unless canonical code explicitly does so.

INV-ANALYSIS-06 =
  Actionability is rendered from canonical authority.

INV-ANALYSIS-07 =
  Unsupported SHORT semantics are not presented as quantitatively valid.

INV-ANALYSIS-08 =
  Distance wording agrees with numerical sign and natural-language direction.

INV-ANALYSIS-09 =
  Market-session fallback cannot create Analysis semantics absent from the backend.

INV-ANALYSIS-10 =
  Frontend presentation derivations cannot change recommendation state, score, rank, stop, target, or corridor.
```

---

## 6. Acceptance Matrix

| Item | Criterion | Status | Code & Evidence Attestation |
|---|---|---|---|
| **ANALYSIS-SEM-01** | Canonical corridor authority established | **PASS** | `OptimalExecutionEngine.optimal_entry_min` / `max` (`optimal_execution.py:204, 205`). |
| **ANALYSIS-SEM-02** | Competing corridor calculations identified | **PASS** | Identified `decisionHierarchyUtils.ts:24-27` (`stopLoss * 1.02`, `sma50`) and `analytics.py:255`. |
| **ANALYSIS-SEM-03** | All trigger-like concepts inventoried | **PASS** | Section 2.2 catalogs 8 distinct concepts (`TRG-01` through `TRG-08`). |
| **ANALYSIS-SEM-04** | Numeric trigger authority established or absent | **PASS** | Confirmed `CANONICAL_NUMERIC_TRIGGER = NOT_ESTABLISHED`; `TRIGGER_PRICE_AUTHORITY = NONE`. |
| **ANALYSIS-SEM-05** | Trigger state authority established | **PASS** | `DecisionHierarchyEngine.resolve_decision_state` + `OptimalExecutionEngine.execution_status`. |
| **ANALYSIS-SEM-06** | Stabilization semantics reconciled | **PASS** | `is_stabilized` verified as intra-corridor absorption candle, not independent trigger (`optimal_execution.py:298`). |
| **ANALYSIS-SEM-07** | Breakout/pivot semantics reconciled | **PASS** | `breakout_pivot` verified as Stage 4 target anchor ceiling, not universal trigger (`optimal_execution.py:174, 229`). |
| **ANALYSIS-SEM-08** | Paper-trading trigger separated | **PASS** | `PaperTradingOutcomeEvaluator.evaluate_signal` isolated to outcome fill simulation (`paper_trading_outcome_evaluator.py:160`). |
| **ANALYSIS-SEM-09** | Actionability authority frozen | **PASS** | Precedence 6 in `DecisionHierarchyEngine` (`decision_hierarchy.py:131`) and `decisionContract.ts:53`. |
| **ANALYSIS-SEM-10** | Actual decision-state vocabulary frozen | **PASS** | 6 states (`UNVERIFIED` to `ACTIONABLE_SETUP`) and 9 execution statuses frozen. |
| **ANALYSIS-SEM-11** | Market-session role established | **PASS** | Verified pure for Analysis recommendation state; gates passive capture ledger logging (`analytics.py:1227`). |
| **ANALYSIS-SEM-12** | LONG-only authority re-attested | **PASS** | `OptimalExecutionEngine` verified strictly long-only accumulation. |
| **ANALYSIS-SEM-13** | Unsupported SHORT UI classified | **PASS** | `DayTraderPositionSizer.tsx:205` classified as `PRESENT_BUT_UNSUPPORTED`; recommendation is `DISABLE`. |
| **ANALYSIS-SEM-14** | Distance semantics reconciled | **PASS** | Standardized to Price Movement Required; identified inverted sign in `ExecutionCorridor.tsx:178`. |
| **ANALYSIS-SEM-15** | Frontend competing logic inventoried | **PASS** | Section 4.1 inventories 5 frontend components with competing logic. |
| **ANALYSIS-SEM-16** | Backend/frontend authority contract frozen | **PASS** | Strict unidirectional flow defined in Section 4.2. |
| **ANALYSIS-SEM-17** | No invented trigger/reversion/session rules | **PASS** | Confirmed zero synthesized numeric trigger or artificial reversion rules. |
| **ANALYSIS-SEM-18** | Quant engine unchanged | **PASS** | Zero mutations to math, ATR, EMA, SMA, or execution logic (`QUANT_ENGINE_CHANGED = NO`). |
| **ANALYSIS-SEM-19** | Ranking unchanged | **PASS** | Zero mutations to ConfluenceEngine or Radar sorting (`RANKING_CHANGED = NO`). |
| **ANALYSIS-SEM-20** | Paper-trading outcome contract unchanged | **PASS** | Zero modifications to `ARX_PAPER_TRADING_OUTCOME_CONTRACT_V1`. |

---

## 7. Formal Gate Verdict and Implementation Boundary

### 7.1 PASS Verdict

```ini
GATE =
  PASS_ARX_ANALYSIS_CORRIDOR_TRIGGER_SEMANTICS_RECONCILIATION
DESIGN_STATE =
  CLOSED_VERIFIED_AND_FROZEN
CORRIDOR_AUTHORITY =
  ESTABLISHED
TRIGGER_MODEL =
  STATE_BASED_READINESS
TRIGGER_AUTHORITY =
  NONE (Numeric) / ESTABLISHED (State-Based Readiness)
ACTIONABILITY_AUTHORITY =
  ESTABLISHED
DIRECTIONALITY =
  LONG_ONLY
SHORT_SEMANTICS =
  NOT_APPLICABLE
FRONTEND_COMPETING_DOMAIN_LOGIC =
  IDENTIFIED
QUANT_ENGINE_CHANGED =
  NO
RANKING_CHANGED =
  NO
PAPER_TRADING_CONTRACT_CHANGED =
  NO
NEXT_AUTHORIZED_ACTION =
  ARX_ANALYSIS_CORRIDOR_TRIGGER_SEMANTICS_IMPLEMENTATION_GATE
AUTOMATIC_SUCCESSOR_EXECUTION =
  NOT_AUTHORIZED
```

### 7.2 Mandatory Implementation Prohibitions Enforced

Until a subsequent implementation gate explicitly authorizes work, the following actions remain **STRICTLY PROHIBITED**:
- Zero automatic UI remediation without an authorized implementation gate.
- Zero invention of a numeric trigger price.
- Zero invention of synthetic trigger-state transitions.
- Zero promotion of paper-trading fill logic into Analysis recommendations.
- Zero invention of session-dependent Analysis recommendation semantics.
- Zero creation of synthetic SHORT quantitative semantics.
- Zero alteration of corridor calculations, stop/target geometry, actionability, scoring, or ranking.
- Zero alteration of the quant engine or paper-trading outcome contracts.
- Execution stops immediately upon formal gate adjudication.

---

## 8. Bounded Implementation & Parity Verification Report

### 8.1 Gate Identification & Purpose

```ini
GATE =
  PASS_ARX_ANALYSIS_CORRIDOR_TRIGGER_SEMANTICS_IMPLEMENTATION
PREDECESSOR_GATE =
  PASS_ARX_ANALYSIS_CORRIDOR_TRIGGER_SEMANTICS_RECONCILIATION
SCOPE =
  FRONTEND_BOUNDED_IMPLEMENTATION_AND_PARITY
QUANT_ENGINE_CHANGED =
  NO
SCORING_CHANGED =
  NO
RANKING_CHANGED =
  NO
PAPER_TRADING_CONTRACT_CHANGED =
  NO
```

### 8.2 Inventory of Modified Implementation Files

| File | Subsystem | Nature of Change |
|---|---|---|
| `frontend/types/insight.ts` | Shared Types | Added canonical `entryMin?: number` and `entryMax?: number` to `keyLevels` interface. |
| `frontend/lib/insightGenerator.ts` | Data Layer | Populated `entryMin` and `entryMax` in `keyLevels` directly from canonical backend `optimalExecution.optimal_entry_min` / `optimal_entry_max`. |
| `frontend/lib/decisionHierarchyUtils.ts` | Domain Presentation | Removed `stopLoss * 1.02` and `sma50` synthetic corridor derivations; bound corridor condition directly to `kl.entryMin` and `kl.entryMax` with `watchZone` fallback. Emits fail-closed `"Accumulation Corridor Unavailable"` when bounds are absent. |
| `frontend/app/radar/page.tsx` | Radar Surface | Removed client-invented statuses (`'NEAR_PIVOT' \| 'VOLUME_DRYUP' \| 'AWAITING_TRIGGER'`); imported canonical `ExecutionStatus`; bound `RadarAsset.executionStatus` to canonical `ExecutionStatus`; implemented `parseCanonicalExecutionStatus` and `formatRadarExecutionStatus`. |
| `frontend/components/DayTraderPositionSizer.tsx` | Workstation Tool | Disabled `SELL / SHORT` radio input (`disabled={true}`, `aria-disabled="true"`, informative tooltip); locked active `tradeDirection` to `"LONG"`; eliminated synthetic short stop loss and profit target calculations. |
| `frontend/components/workstation/ExecutionCorridor.tsx` | Workstation Presentation | Normalized distance wording to eliminate double negatives: replaced inverted `${((spotPrice - entryLow) / entryLow * 100).toFixed(1)}% below pivot` with `${Math.abs(...).toFixed(1)}% below corridor`. |
| `frontend/components/__tests__/AnalysisDecisionHierarchy.test.tsx` | Test Suite | Added comprehensive invariant regression tests enforcing INV-ANALYSIS-01 through INV-ANALYSIS-08. |

### 8.3 Competing Domain Logic Removed

1. **Synthetic Corridor Calculations**:
   - `decisionHierarchyUtils.ts` previously synthesized corridor bounds as `kl.stopLoss * 1.02` up to `kl.sma50` whenever `kl.watchZone` was unparseable. This completely overrode the backend's `OptimalExecutionEngine` corridor.
   - **Resolution**: Fully removed. The component now reads `kl.entryMin` and `kl.entryMax` (populated directly from `optimal_entry_min` / `optimal_entry_max`), falling back safely to the canonical formatted `watchZone` string. If both are unpopulated, it renders a fail-closed status `"Accumulation Corridor Unavailable"`. Zero synthetic geometry is created.

2. **Synthetic Radar Execution Statuses**:
   - `radar/page.tsx` previously defined an ad-hoc union `'NEAR_PIVOT' | 'VOLUME_DRYUP' | 'AWAITING_TRIGGER' | ...` and mapped assets into these synthetic states via heuristic client string parsing.
   - **Resolution**: Ad-hoc union eliminated. `radar/page.tsx` imports canonical `ExecutionStatus` from `types/decisionContract.ts`. A deterministic mapper `parseCanonicalExecutionStatus` maps backend execution statuses directly, maintaining 100% vocabulary parity with Analysis.

3. **Synthetic SHORT Geometry in DayTraderPositionSizer**:
   - `DayTraderPositionSizer.tsx` previously allowed switching to `"SHORT"`, which then mathematically inverted risk math using `spotPrice + (riskPerShare)` and computed synthetic downside targets (`target15 = spot - risk * 1.5`, etc.).
   - **Resolution**: The `SELL / SHORT` option is explicitly disabled in the UI with a descriptive tooltip explaining that ARX quantitatively models Long accumulation only. The execution state is hardcoded to `"LONG"`, eliminating all synthetic short stop and target generation.

4. **Inverted Distance Wording**:
   - `ExecutionCorridor.tsx` previously computed `(spotPrice - entryLow) / entryLow * 100` when price was below the corridor, producing a negative number (e.g. `-3.2%`), which was displayed as `"-3.2% below pivot"`.
   - **Resolution**: Normalized to `Math.abs(...)` and formatted as `X.X% below corridor`, eliminating directional ambiguity and double negatives.

### 8.4 Disposition of `entryPivot`

An audit of all references to `entryPivot` across the codebase (`setups/page.tsx`, `governorSizingEngine.ts`, `orderClipboard.ts`) was completed:
- `entryPivot` represents a scalar limit-order planning ceiling (`optimal_entry_max` / `LMT: $X`) used for order drafting and conservative share-count ceilings.
- It does **not** act as an independent trigger price, does not trigger automated entries, and does not compete with the two-sided accumulation corridor.
- Its role is ratified as a limit-order planning parameter.

### 8.5 Verification & Test Execution Results

```ini
FRONTEND_VITEST_SUITE =
  PASS (18 test files, 161 tests passing)
FRONTEND_TYPECHECK_SUITE =
  PASS (cmd.exe /c npx tsc --noEmit — 0 errors)
BACKEND_QUANT_EXECUTION_SUITE =
  PASS (tests/test_optimal_execution.py & test_phase2_decision_authority.py — 12 tests passing)
GOVERNANCE_INVARIANTS_SUITE =
  PASS (tests/governance/test_counterfactual_phase_a.py & test_counterfactual_phase_b.py — 32 tests passing)
STATIC_COMPETING_LOGIC_AUDIT =
  PASS (0 instances of unauthorized competing domain logic in production paths)
WHITESPACE_AND_FORMATTING =
  PASS (git diff --check — 0 errors)
```

### 8.6 Invariant Adherence Attestation

| Invariant | Description | Verification Method | Status |
|---|---|---|---|
| **INV-ANALYSIS-01** | Canonical corridor authority (`OptimalExecutionEngine`) | Unit test in `AnalysisDecisionHierarchy.test.tsx` | **PASS** |
| **INV-ANALYSIS-02** | Zero frontend corridor recalculation | Unit test & static code audit | **PASS** |
| **INV-ANALYSIS-03** | Zero synthetic numeric trigger price | Unit test verifying absence of trigger price | **PASS** |
| **INV-ANALYSIS-04** | Paper-trading fill decoupled from Analysis trigger | Governance separation verified | **PASS** |
| **INV-ANALYSIS-05** | Stabilization not equated to trigger | Verified in `DecisionHierarchyEngine` | **PASS** |
| **INV-ANALYSIS-06** | Actionability rendered strictly from canonical authority | Unit test asserting fail-closed gate | **PASS** |
| **INV-ANALYSIS-07** | Unsupported SHORT UI disabled | Unit test asserting `disabled={true}` on short button | **PASS** |
| **INV-ANALYSIS-08** | Distance wording free of double negatives | Unit test verifying non-negative corridor distance string | **PASS** |
| **INV-ANALYSIS-09** | Session fallback cannot create Analysis semantics | Verified in `weeklyConfluenceSpotlightDecoupling.test.ts` | **PASS** |
| **INV-ANALYSIS-10** | Zero alteration of backend quant, scoring, or ranking | Verified across all test suites | **PASS** |

### 8.7 Formal Implementation Verdict

```ini
GATE =
  PASS_ARX_ANALYSIS_CORRIDOR_TRIGGER_SEMANTICS_IMPLEMENTATION
VERDICT =
  PASS
CORRIDOR_SEMANTICS =
  CANONICAL_OPTIMAL_EXECUTION_BOUND
TRIGGER_SEMANTICS =
  STATE_BASED_READINESS_WITHOUT_NUMERIC_TRIGGER
SHORT_SEMANTICS =
  DISABLED_IN_UI_LONG_ONLY_AUTHORITATIVE
QUANT_ENGINE_CHANGED =
  NO
SCORING_CHANGED =
  NO
RANKING_CHANGED =
  NO
PAPER_TRADING_CONTRACT_CHANGED =
  NO
UNAUTHORIZED_COMPETING_DOMAIN_LOGIC =
  0
NEXT_AUTHORIZED_ACTION =
  AWAIT_USER_DIRECTION
```
