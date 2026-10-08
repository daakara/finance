# PRD Addendum 002: Decision Readiness, Next-Action Clarity & State Harmonization

**Product**: ARX Terminal  
**Document ID**: `PRD-ADDENDUM-002-DECISION-READINESS`  
**Parent Document**: `docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`  
**Related Documents**:  
- `docs/prd/PRD_ADDENDUM_001_DECISION_INTEGRITY.md`  
- `docs/releases/2026-10-08_a05331f_decision-surface-integrity-remediation.md` (QA-ESC-011)  
- `docs/governance/ARX_QA_ESCAPE_REGISTRY.md`  
**Lifecycle Status**: `RATIFIED`  
**Milestone**: Synthesis E — Wave 4 Specification  
**Classification**: Product, UX Architecture & Domain Engineering Contract  
**Created**: 2026-10-07  
**Ratified**: 2026-10-07 (Local Draft `0d0f9b1`)  
**Reconciled & Re-Ratified**: 2026-10-08 (On Canonical Baseline `6e10051` post QA-ESC-011)  

---

## 1. Executive Intent & Primary Objective

The primary objective of Synthesis E Wave 4 is to ensure that after ARX evaluates an asset and renders a verdict, the user can immediately and unambiguously answer five operational questions:

1. **What blocks action?** (Identify the exact condition preventing immediate capital allocation).
2. **What condition must occur next?** (Identify the observable event that advances the setup toward actionability).
3. **What should the user do now?** (Present a single, high-leverage primary action backed by a genuine platform capability).
4. **What must the user NOT do?** (Deliver unambiguous, evidence-grounded protective negative guidance, such as avoiding chasing extended setups).
5. **Has the setup become execution-ready?** (Provide a deterministic 3-gate readiness progression that clears only when all execution criteria are satisfied).

Wave 4 establishes complete semantic and visual harmonization across all primary workstation surfaces:
- **Radar** (`/radar` — Universe discovery and screening)
- **Terminal Cockpit** (`/` — Asset deep-dive and institutional decision synthesis)
- **Execution Cards & Modals** (`OptimalEntryExitCard`, `PreFlightChecklistModal`, `PositionSizerModal`, `AlertTriggerModal`)

---

## 2. Decision-Lifecycle Ownership Model & Acyclic Authority DAG

To eliminate architectural confusion, circular dependencies, and duplicate decision authorities between decision classification, setup qualification, risk clearance, and execution sizing, Wave 4 ratifies an explicit, strictly acyclic authority Directed Acyclic Graph (DAG):

```text
                  MARKET DATA & GEOMETRY
                 (Price, Corridor, Stage)
                       /          \
                      /            \
                     v              v
         CANONICAL_DECISION_STATE   MARKET_LOCATION_STATE
           (DecisionHierarchy)     (ExecutionStatus: in-zone, pullback)
                     \              /
                      \            /
                       v          v
                     ACTIONABILITY
                   (isDecisionActionable)
                            ↓
                    DECISION_READINESS
                 (3 Ordered Market Gates)
                            ↓
                        PRE_FLIGHT
                 (5 Risk Hygiene Checks)
                            ↓
                      EXECUTION_PLAN
             (Sizing, Governor, Order Brackets)
                            ↓
                    SHARED_PRESENTATION
              (Single Consumer Formatting Authority)
```

$$\text{MARKET\_DATA} \longrightarrow \{\text{CANONICAL\_DECISION\_STATE}, \text{MARKET\_LOCATION\_STATE}\} \longrightarrow \text{ACTIONABILITY} \longrightarrow \text{DECISION\_READINESS} \longrightarrow \text{PRE\_FLIGHT} \longrightarrow \text{EXECUTION\_PLAN} \longrightarrow \text{SHARED\_PRESENTATION}$$

### 2.1 Decoupling of Market Location vs Order Parameterization
To prevent circular dependency between `ACTIONABILITY` and `EXECUTION_PLAN`:
1. **`MARKET_LOCATION_STATE`** (`executionStatus`): Upstream geometric classification computed directly from price and corridor levels (`IN_BUY_ZONE`, `WAITING_PULLBACK`, `EXTENDED_ABOVE_BUY_ZONE`, etc.). This is an **input** to `isDecisionActionable()`.
2. **`ORDER_PARAMETERIZATION`** (`EXECUTION_PLAN`): Downstream order execution computation (share sizing, dollar risk, volatility stops, multi-stage profit targets, ratchet rules, and order staging). This is **downstream** of `ACTIONABILITY` and `PRE_FLIGHT`.

### 2.2 Distinct Ownership Definitions

| Lifecycle Concept | Core Question | Authoritative Engine / Owner | Scope of Authority |
| :--- | :--- | :--- | :--- |
| **`CANONICAL_DECISION_STATE`** | *"Is the asset or setup structurally acceptable?"* | `DecisionHierarchyEngine` (`analyst_dashboard/analyzers/decision_hierarchy.py`) | Evaluates data completeness, tape freshness, asset verification, instrument applicability, and structural Minervini stage discipline. Produces `decisionState` and `decisionStateLabel`. |
| **`MARKET_LOCATION_STATE`** | *"Where is price positioned relative to the accumulation corridor?"* | Pure geometric evaluator (`OptimalExecutionPlan` levels) | Evaluates spot price against entry min/max. Emits `executionStatus`. |
| **`ACTIONABILITY`** | *"Is capital action authorized right now?"* | `isDecisionActionable()` (`frontend/types/decisionContract.ts`) | Strict binary gate: `true` only if `decisionState == ACTIONABLE_SETUP` AND `executionStatus in ('IN_BUY_ZONE', 'READY_TO_BUY')`. Fails closed. |
| **`DECISION_READINESS`** | *"What specific market/setup condition is currently blocking action?"* | `resolveDecisionReadiness()` (`frontend/lib/decisionReadiness.ts`) | Evaluates the 3 ordered market gates (Location $\to$ Trigger $\to$ Risk Floor) to identify the single `ACTIVE_BLOCKING_GATE` and downstream pending dependencies. |
| **`PRE_FLIGHT`** | *"Once action is otherwise ready, is execution safe and permissible?"* | `PreFlightChecklistModal` (`frontend/components/PreFlightChecklistModal.tsx`) | 5-point execution hygiene sanity check evaluated immediately before order staging/sizing (R:R $\ge 2.0$, trend slippage $< 2\%$, capital risk/short float, earnings blackout, macro VIX $< 26.0$). Cannot grant clearance if `isActionable == false`. |
| **`EXECUTION_PLAN`** | *"How should the position be entered, sized, protected, and exited?"* | `governorSizingEngine` (`frontend/lib/simulation/governorSizingEngine.ts`) | Computes bracket orders, volatility stops (-1.5x ATR), multi-stage profit targets (TP1/TP2), ratchet rules, and liquidity defense sizing. Active only when authorized. |
| **`SHARED_PRESENTATION`** | *"How is canonical state consistently formatted across surfaces?"* | `decisionPresentation.ts` (`frontend/lib/decisionPresentation.ts`) | Single consumer formatting authority. Formats labels, badges, and colors for Radar, Cockpit, and Cards. **Never evaluates decisions or overrides upstream states.** |

### 2.3 Boundary Invariant
No presentation component or subordinate modal may override, bypass, or contradict an upstream lifecycle authority.

---

## 3. Preservation of QA-ESC-011 as a Frozen Invariant

Wave 4 explicitly commits to preserving the production-verified decision-surface remediation established in QA-ESC-011 (`a05331f`, `6e10051`):

```text
BACKEND_DECISION_FIELD =
  decisionStateLabel

GENERIC_WAIT_FOR_TRIGGER_FALLBACK =
  PROHIBITED

ACTIONABILITY_FALSE_TO_WAIT_FOR_TRIGGER_MAPPING =
  PROHIBITED

PRIMARY_SECONDARY_VERDICT_DUPLICATION =
  PROHIBITED

ACTIONABILITY_DOMAIN =
  ACTIONABLE / NOT ACTIONABLE

QA_ESC_011_REOPENED =
  NO
```

### 3.1 Strict Anti-Collapse Rules
1. **No Competing Authority**: `decisionPresentation.ts` must purely format canonical upstream state (`decisionState`, `decisionStateLabel`, `executionStatus`, `isActionable`). It must never invent alternative decision classifications or re-evaluate Bayesian scores.
2. **Actionability Independence**: `isActionable === false` must render `[NOT ACTIONABLE]` (or explicit gate blocker), **NEVER** generic `"WAIT FOR TRIGGER"`. States such as `AVOID`, `HOLD`/`OWNED`, `UNVERIFIED`, `INSUFFICIENT_HISTORY`, and `EVIDENCE_INCOMPLETE` must never display `"WAIT FOR TRIGGER"`.
3. **Zero Headline/Badge Duplication**: The primary analytical verdict headline (e.g. `"Valid Setup — Awaiting Trigger"`) must never be duplicated verbatim in adjacent secondary badges.
4. **Fail-Closed Fallback**: Any missing or unassessed decision state must fail closed to `"Setup Evaluation Pending"` (Neutral/Slate), **NEVER** `"Wait for Trigger"`.

---

## 4. The 3-Gate Decision Readiness Model

Decision Readiness models the path from setup discovery to trade readiness across three ordered, sequential gates:

```text
GATE 1: LOCATION / GEOMETRY
  (Is price inside the accumulation corridor?)
         ↓
GATE 2: DYNAMIC TRIGGER
  (Has breakout volume or confirmation candle triggered?)
         ↓
GATE 3: RISK / GOVERNANCE CLEARANCE
  (Is prospective R:R ≥ 2.0:1 under calm macro volatility?)
```

### 4.1 Gate Specifications

#### Gate 1: Location / Geometry (`GATE_1_LOCATION`)
- **GATE_ID**: `GATE_1_LOCATION`
- **DISPLAY_NAME**: `1. Corridor Location & Geometry`
- **AUTHORITATIVE_INPUTS**: `current_price`, `optimal_entry_min`, `optimal_entry_max`, `stage_phase`, `setup_pattern`.
- **PASS_CONDITION**: `current_price >= optimal_entry_min` AND `current_price <= (optimal_entry_max * 1.02)` AND `optimal_entry_min > 0` AND not Stage 4.
- **FAIL_CONDITION**:
  - *Extended / Chase*: `current_price > (optimal_entry_max * 1.02)`
  - *Below Zone / Awaiting Pullback*: `current_price < optimal_entry_min`
  - *Stage 4 Correction*: Setup pattern or stage indicates Stage 4 markdown.
- **UNAVAILABLE_CONDITION**: `optimal_entry_min == null` OR `optimal_entry_max == null` OR `current_price <= 0` OR candle count $< 50$.
- **DEPENDENCIES**: None (Root evaluation gate).
- **USER_VISIBLE_EXPLANATIONS**:
  - *Passed*: `"Price ($[spot]) is positioned inside the institutional accumulation corridor ($[min]–$[max])."`
  - *Blocking (Extended)*: `"Price ($[spot]) has extended >2% past the accumulation corridor ($[min]–$[max]). Do not chase above resistance."`
  - *Blocking (Below Zone)*: `"Price ($[spot]) is below the accumulation corridor ($[min]–$[max]). Awaiting base stabilization."`
  - *Blocking (Stage 4)*: `"Asset is in Stage 4 distribution. Structural base required before accumulation corridor applies."`
  - *Unavailable*: `"Optimal entry corridor uncalculated or candle history insufficient (< 50 sessions)."`

#### Gate 2: Dynamic Trigger (`GATE_2_TRIGGER`)
- **GATE_ID**: `GATE_2_TRIGGER`
- **DISPLAY_NAME**: `2. Dynamic Trigger & Volume Confirmation`
- **AUTHORITATIVE_INPUTS**: `is_confirmed` (from execution plan / scanner), `vcp_contraction_status`, `breakout_pivot`, `setup_pattern`.
- **PASS_CONDITION**: `is_confirmed == true`.
- **FAIL_CONDITION**: `is_confirmed == false` (reversal confirmation candle unformed or breakout volume unconfirmed).
- **UNAVAILABLE_CONDITION**: Trigger telemetry uncalculated or missing.
- **DEPENDENCIES**: `GATE_1_LOCATION == PASSED`. If Gate 1 is `BLOCKING` or `UNAVAILABLE`, Gate 2 evaluates to `PENDING_DEPENDENCY`.
- **USER_VISIBLE_EXPLANATIONS**:
  - *Passed*: `"Breakout volume expansion or reversal confirmation candle verified."`
  - *Blocking*: `"Price is inside the accumulation corridor, but confirmation trigger is pending. Awaiting volume expansion / pivot reclaim."`
  - *Pending Dependency*: `"Waiting for Gate 1 (Location) to clear before evaluating trigger confirmation."`
  - *Unavailable*: `"Trigger telemetry unavailable."`

#### Gate 3: Risk / Governance Clearance (`GATE_3_RISK_CLEARANCE`)
- **GATE_ID**: `GATE_3_RISK_CLEARANCE`
- **DISPLAY_NAME**: `3. Risk Floor & Macro Clearance`
- **AUTHORITATIVE_INPUTS**: `risk_reward_ratio`, `stop_loss`, `take_profit_1`, `vix`. *(Note: `macroRegime` string is removed from authoritative inputs as it is purely informational and not evaluated in numeric clearance).*
- **PASS_CONDITION**: `risk_reward_ratio >= 2.0` AND `stop_loss > 0` AND `take_profit_1 > 0` AND `vix != null` AND `vix < 26.0`.
- **FAIL_CONDITION**: `risk_reward_ratio < 2.0` OR `vix >= 26.0`.
- **UNAVAILABLE_CONDITION (FAIL CLOSED)**:
  - `vix == null` OR `isNaN(vix)` OR VIX upstream unavailable.
  - `stop_loss == null` OR `take_profit_1 == null` OR `risk_reward_ratio == null`.
  - **Rule**: Missing VIX is **NEVER** treated as passed. It strictly evaluates to `UNAVAILABLE` (fail closed).
- **DEPENDENCIES**: `GATE_1_LOCATION == PASSED` AND `GATE_2_TRIGGER == PASSED`. If either prerequisite is unmet, Gate 3 evaluates to `PENDING_DEPENDENCY`.
- **USER_VISIBLE_EXPLANATIONS**:
  - *Passed*: `"Setup clears the 2:1 institutional risk/reward floor ([rr]:1) under calm macro volatility (VIX [vix] < 26.0)."`
  - *Blocking (R:R)*: `"Prospective Risk/Reward ([rr]:1) is below the institutional 2:1 minimum floor. Capital efficiency inadequate."`
  - *Blocking (Macro)*: `"Broad market volatility is elevated (VIX [vix] ≥ 26.0). New equity risk deployment suspended."`
  - *Pending Dependency*: `"Waiting for Gate 1 (Location) and Gate 2 (Trigger) to clear before clearing execution risk."`
  - *Unavailable (Missing VIX)*: `"Macro volatility reading unavailable. Risk clearance requires live verified VIX tape (fail-closed)."`
  - *Unavailable (Missing Levels)*: `"Execution levels or risk/reward ratio uncalculated."`

---

## 5. Blocking-Gate & Dependency Cascade Semantics

To prevent contradictory diagnostic messaging (e.g. showing 3 simultaneous failures when an asset has simply not pulled back yet), Wave 4 enforces deterministic dependency cascade semantics:

### 5.1 Gate Evaluation States
Every gate in the 3-gate ladder resolves to exactly one of four mutually exclusive states:
```text
1. PASSED:             The gate's conditions are fully satisfied.
2. BLOCKING:           The gate's conditions are NOT satisfied AND all prerequisite gates have PASSED.
3. PENDING_DEPENDENCY: An upstream prerequisite gate has NOT passed; this gate cannot be evaluated yet.
4. UNAVAILABLE:        The authoritative inputs for this gate are missing, degraded, or uncalculated.
```

### 5.2 Definition of Active Blocking Gate
$$\text{ACTIVE\_BLOCKING\_GATE} = \text{earliest gate } i \in \{1, 2, 3\} \text{ where } \text{State}(i) \neq \text{PASSED}$$
If all gates pass:
$$\text{ACTIVE\_BLOCKING\_GATE} = \text{NONE (Setup is EXECUTION\_READY)}$$

### 5.3 Deterministic Cascade Matrix

| Gate 1 Condition | Gate 2 Condition | Gate 3 Condition | Gate 1 State | Gate 2 State | Gate 3 State | Active Blocker |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fail** (Extended / Pullback) | Any | Any | **`BLOCKING`** | `PENDING_DEPENDENCY` | `PENDING_DEPENDENCY` | **Gate 1** |
| **Pass** (In Corridor) | **Fail** (No Volume) | Any | `PASSED` | **`BLOCKING`** | `PENDING_DEPENDENCY` | **Gate 2** |
| **Pass** (In Corridor) | **Pass** (Triggered) | **Fail** (R:R < 2.0) | `PASSED` | `PASSED` | **`BLOCKING`** | **Gate 3** |
| **Pass** (In Corridor) | **Pass** (Triggered) | **Pass** (R:R $\ge$ 2.0, VIX < 26) | `PASSED` | `PASSED` | `PASSED` | **`NONE`** |
| **Unavailable** (No data) | Any | Any | **`UNAVAILABLE`** | `PENDING_DEPENDENCY` | `PENDING_DEPENDENCY` | **Gate 1 (Data)** |
| **Pass** (In Corridor) | **Pass** (Triggered) | **Unavailable** (VIX null) | `PASSED` | `PASSED` | **`UNAVAILABLE`** | **Gate 3 (VIX)** |

---

## 6. Operational CTA Capability Reconciliation

Wave 4 strictly enforces the Core Operational Action Invariant:
```text
USER_VISIBLE_OPERATIONAL_ACTION => REAL_BACKING_CAPABILITY
```
$$\forall \text{ action } a \in \text{VisibleActions}, \quad \text{ExistsBackingCapability}(a) = \text{true}$$

No button, link, or CTA may be presented to the user unless a genuine, tested, production-verified backing implementation exists. Mock toasts, placeholder handlers, and fake cloud-sync confirmations are strictly prohibited.

### 6.1 Audit of Existing Operational Capabilities

| Proposed Action | Real Handler | Component | Persistence Layer | Production Verified | Safe for Wave 4 CTA | Operational Limits & Grounding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Size Position`** | `setIsSizerOpen(true)` | `PositionSizerModal.tsx` | In-memory | **YES** | **YES** | Active only when authorized (`canSizeTrade === true`). |
| **`Pre-Flight Checklist`** | `setIsChecklistOpen(true)` | `PreFlightChecklistModal.tsx` | In-memory interactive | **YES** | **YES** | 5-point execution sanity checklist. |
| **`Log to Portfolio`** | `handleLogToPortfolio` | `addPortfolioPosition()` (`portfolio.ts`) | `localStorage` (`ARX_PORTFOLIO_POSITIONS`) | **YES** | **YES** | Persists entry plan to local paper portfolio. |
| **`Set Pullback Alert`** | `setIsAlertOpen(true)` (`!isStage4`) | `AlertTriggerModal.tsx` / `AlertManager` | `localStorage` (`FINANCE_EXECUTION_ALERTS_V1`) | **YES** | **YES** | Backed by `notifyOnBuyZone` in `alertManager.ts`. Alerts when price enters accumulation corridor. |
| **`Set 50-SMA Pivot Alert`** | `setIsAlertOpen(true)` (`isStage4 === true`) | `AlertTriggerModal.tsx` / `AlertManager` | `localStorage` (`FINANCE_EXECUTION_ALERTS_V1`) | **YES** | **YES** | Backed by `breakoutPivotPrice` in `alertManager.ts`. **Available exclusively for Stage 4 setups.** |
| **`Set Breakout Alert`** | *N/A (No non-Stage-4 breakout pivot rule in `AlertManager`)* | *N/A* | *N/A* | **ABSENT** | **PROHIBITED** | `AlertManager` has no independent breakout pivot rule for non-Stage-4 assets. **Do not create fake button.** For `IN_BUY_ZONE_AWAITING_TRIGGER`, CTA is replaced with `Set Buy Zone Alert` or `Explore Radar Setups`. |
| **`Explore Radar Setups`** | `router.push('/radar')` | Next.js Router | URL navigation | **YES** | **YES** | Navigates to Radar screener to find alternatives. |
| **`Open Watchlist`** | `useUIStore.getState().openWatchlist()` | `WatchlistDrawer.tsx` | Static list + localStorage quotes | **YES** | **YES (Navigation Only)** | Opens Watchlist Drawer. **Dynamic symbol insertion (`ADD_TO_WATCHLIST`) is ABSENT and prohibited.** |
| **`Search Another Ticker`** | `input.focus()` / search modal | `Navbar.tsx` | In-memory | **YES** | **YES** | Focuses search bar for new ticker lookup. |
| **`Review Execution Ladder`** | `#execution-levels.scrollIntoView()` | Native DOM scroll | None (in-page navigation) | **YES** | **YES** | In-page smooth scroll to execution table. |
| **`Inspect Evidence Dossier`** | `<details>.open = true` / modal | `InsightProvenanceModal.tsx` | None (in-page disclosure) | **YES** | **YES** | In-page disclosure of statutory filing provenance. |
| **`Await Market Refresh`** | *N/A (Informational status)* | *N/A* | *N/A* | **ABSENT AS CTA** | **PROHIBITED AS CTA** | **Not an operational action.** Rendered as an advisory status badge / banner (`[TAPE STALE: LIVE TRADING LOCKED]`), **never as a clickable button.** |

---

## 7. Ratified Next-Action Matrix

Every canonical state maps to a primary and secondary operational action backed strictly by the verified capabilities audited above:

| Canonical State | Active Blocker / Situation | Primary Action (CTA) | Secondary Action | Backing Implementation & Handler | Failure & Guardrail Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ACTIONABLE_SETUP`** | None (All 3 gates passed) | **`Size Position`** | `Pre-Flight Checklist` / `Log to Portfolio` | `setIsSizerOpen(true)` $\to$ `PositionSizerModal` | If inputs missing, button disabled with tooltip explaining missing level. |
| **`VALID_SETUP`** *(Waiting Pullback)* | Gate 1 (Price extended >2% above zone) | **`Set Pullback Alert`** | `Explore Radar Setups` | `setIsAlertOpen(true)` $\to$ `AlertTriggerModal` (pre-checks Buy Zone) | If notification permission denied, user alerted via UI inline banner and Web Audio chime. |
| **`VALID_SETUP`** *(In Buy Zone, Awaiting Trigger)* | Gate 2 (Awaiting volume / confirmation) | **`Set Buy Zone Alert`** | `Pre-Flight Checklist` | `setIsAlertOpen(true)` $\to$ `AlertTriggerModal` (tracks zone entry & invalidation) | Modal pre-selects stop-loss and buy-zone alert rules. |
| **`VALID_SETUP`** *(Stage 4 Correction / Basing)* | Gate 1 (Stage 4 distribution) | **`Explore Radar Setups`** | `Set 50-SMA Pivot Alert` | `router.push('/radar')` (Primary) / `setIsAlertOpen(true)` (Secondary) | Immediate client route transition. |
| **`VALID_SETUP`** *(Sub-2:1 R:R)* | Gate 3 (Risk/Reward < 2.0:1) | **`Review Execution Ladder`** | `Explore Radar Setups` | In-page scroll to execution levels table | Highlights insufficient reward/risk asymmetry. |
| **`EVIDENCE_INCOMPLETE`** | Incomplete statutory filing | **`Inspect Evidence Dossier`**| `Explore Radar Setups` | In-page toggle `InsightProvenanceModal` | Displays instrument-specific filing reason (e.g. ETF profile vs 10-K). |
| **`INSUFFICIENT_DATA`** | < 50 daily trading sessions | **`Explore Radar Setups`** | `Search Another Ticker` | `router.push('/radar')` | Informs user that minimum 50 sessions required for trend analysis. |
| **`STALE_DATA`** | Historical tape > 4 days | **`Explore Radar Setups`** | *None (Status banner only)* | `router.push('/radar')` | Displays `[TAPE STALE: LIVE TRADING LOCKED]` advisory banner. |
| **`UNVERIFIED`** | Unverified ticker identity | **`Explore Radar Setups`** | `Search Another Ticker` | `router.push('/radar')` | Sizing and triggers locked. |

### 7.1 Prohibition on Informational Modals as Operational CTAs
Under Wave 4 invariant `AC-W4-006`, no non-actionable primary CTA may open an informational explanation modal (such as "Why Score") as its primary action. Primary CTAs must be operational (e.g. setting an alert, sizing a trade, or navigating to alternative setups).

---

## 8. Negative Guidance Contract & Canonical 2:1 R:R Authority Audit

### 8.1 Core Epistemic Governance
$$\text{CLAIM\_SET} \subseteq \text{EVIDENCE\_SET}$$

Wave 4 authorizes the display of protective negative guidance (e.g. *"DO NOT CHASE"*, *"WAIT FOR PULLBACK"*, *"DO NOT ALLOCATE YET"*). Every negative guidance assertion must derive directly from authentic runtime evidence and canonical domain state.

### 8.2 Audit of the 2:1 Risk/Reward Threshold
```text
CANONICAL_RR_THRESHOLD = 2.0:1 (At least double reward relative to risk)
AUTHORITY =
  1. analyst_dashboard/analyzers/decision_hierarchy.py (line 236: rr >= 2.0; line 266: "Risk/Reward ratio ({rr:.1f}:1) is below the institutional 2.0:1 minimum threshold.")
  2. analyst_dashboard/analyzers/optimal_execution.py (lines 280-310: models target floor >= 2.0)
  3. docs/prd/PRD_ADDENDUM_001_DECISION_INTEGRITY.md (line 90: safeRR >= 2.0)
  4. frontend/components/PreFlightChecklistModal.tsx (lines 130-132: safeRR >= 2.0)

WAVE_4_COPY_MAY_REFERENCE_THRESHOLD = YES (Fully authorized as a canonical domain rule)
```

### 8.3 Canonical Negative Guidance Rules

| Triggering Condition | Domain Evidence Input | Canonical Negative Guidance Banner | Rationale & Protection |
| :--- | :--- | :--- | :--- |
| **Price Extended** | `currentPrice > optimalEntryMax * 1.02` | **`DO NOT CHASE: Price has extended past the accumulation corridor.`** | Entering after a breakout degrades the risk/reward ratio below the 2:1 institutional floor and increases drawdown probability. |
| **Awaiting Trigger** | `in_buy_zone && !is_confirmed` | **`DO NOT PRE-EMPT: Price is in buy zone, but confirmation trigger is pending.`** | Entering before volume breakout or reversal confirmation exposes capital to false bounces and continued base drift. |
| **Stage 4 Markdown** | `isStage4 == true` | **`CAPITAL DEFENSE: Asset is in Stage 4 correction below 50-day SMA.`** | Catching falling knives in distribution violates Minervini stage discipline. Wait for structural base and breakout pivot reclaim. |
| **Sub-2:1 R:R** | `riskRewardRatio < 2.0` | **`INADEQUATE ASYMMETRY: Prospective reward is below the 2:1 institutional floor.`** | Trade does not offer sufficient mathematical payoff to justify risk. |
| **High Macro Volatility**| `vix >= 26.0` | **`MACRO CAUTION: Market volatility (VIX ≥ 26.0) is elevated.`** | Systemic volatility suppresses breakout follow-through across all equities. |
| **Missing Macro VIX** | `vix == null` | **`MACRO DATA DEGRADED: Live market volatility unavailable.`** | Fails closed; risk clearance suspended until authentic VIX is acquired. |

---

## 9. Canonical State Nomenclature & Shared Presentation Authority

To eliminate desynchronization where Radar, Terminal Cockpit, and Execution Cards render conflicting state labels or color badges for the same underlying asset, Wave 4 mandates a single shared presentation resolver: `frontend/lib/decisionPresentation.ts`.

### 9.1 State Presentation Glossary

| Domain State (`decisionState`) | Execution Status (`executionStatus`) | Shared Presentation Badge | Meaning & Context | Actionability | Active Blocker | Primary Next Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`ACTIONABLE_SETUP`** | `IN_BUY_ZONE` / `READY_TO_BUY` | `[ACTIONABLE SETUP]` (Emerald) | All criteria met: in buy zone, trigger confirmed, R:R $\ge 2.0$, trend advancing. | `true` | `NONE` | `Size Position` |
| **`VALID_SETUP`** | `WAITING_PULLBACK` | `[AWAITING PULLBACK]` (Amber) | Setup structure is valid, but price is extended >2% above entry corridor. | `false` | Gate 1 (Location) | `Set Pullback Alert` |
| **`VALID_SETUP`** | `IN_BUY_ZONE_AWAITING_TRIGGER` | `[AWAITING TRIGGER]` (Cyan) | Price is in accumulation zone; volume expansion or pivot confirmation pending. | `false` | Gate 2 (Trigger) | `Set Buy Zone Alert` |
| **`VALID_SETUP`** | `APPROACHING_TARGET` | `[APPROACHING TARGET]` (Purple) | Price has reached or is nearing TP1/TP2 profit objectives. | `false` | Execution Phase | `Review Ratchet Rule` |
| **`VALID_SETUP`** | `STAGE_4_CORRECTION` | `[STAGE 4 DEFENSE]` (Rose) | Structural markdown below 50-day SMA; waiting for base floor formation. | `false` | Gate 1 (Location) | `Explore Radar Setups` |
| **`EVIDENCE_INCOMPLETE`** | Any | `[EVIDENCE INCOMPLETE]` (Slate) | Statutory filing missing (e.g. 10-K missing for equity or ETF fund profile pending). | `false` | Gate 1 (Evidence) | `Inspect Evidence Dossier` |
| **`INSUFFICIENT_DATA`** | `INSUFFICIENT_HISTORY` | `[INSUFFICIENT HISTORY]` (Slate) | < 50 daily trading sessions; insufficient depth for trend modeling. | `false` | Gate 1 (Data) | `Explore Radar Setups` |
| **`STALE_DATA`** | `STALE_MARKET_DATA` | `[STALE TAPE]` (Slate) | Historical market data > 4 days old; live trading triggers suspended. | `false` | Gate 1 (Data) | `Explore Radar Setups` |
| **`UNVERIFIED`** | `UNVERIFIED_ASSET` | `[UNVERIFIED ASSET]` (Slate) | Security identity or listing state cannot be verified safely. | `false` | Gate 1 (Security) | `Explore Radar Setups` |

---

## 10. Pre-Flight Ownership Reconciliation

Wave 4 de-duplicates overlapping checks between Decision Readiness and Pre-Flight while preserving the Wave 3 epistemic contract (`PRD_ADDENDUM_001_DECISION_INTEGRITY.md`):

### 10.1 De-Duplication & Classification Matrix

| Check ID | Original Pre-Flight Check | Wave 4 Classification | Role & Precedence | Wave 3 Invariant Status |
| :--- | :--- | :--- | :--- | :--- |
| **`CHK-1`** | **Reward vs Risk Balance** ($\ge 2:1$) | `KEEP_IN_PREFLIGHT` & `REFERENCE_IN_GATE_3` | Pre-Flight validates execution parameters ($R:R \ge 2.0$, stop loss, target 1) immediately before sizing modal opens. Gate 3 in Decision Readiness references this same ratio as a macro prerequisite. | **PRESERVED** |
| **`CHK-2`** | **Trend & Corridor Proximity** | `KEEP_IN_PREFLIGHT` (As execution slippage check) | Decision Readiness Gate 1 owns setup qualification (identifies whether price is inside or outside corridor). Pre-Flight Check 2 acts as a final slippage check ensuring price has not slipped $>2\%$ between analysis and execution. | **PRESERVED** |
| **`CHK-3`** | **Capital Health & Squeeze Risk** | `KEEP_IN_PREFLIGHT` | Evaluates short float $\le 12\%$, quality score $\ge 60$, no turnaround verdict. Stays exclusively in Pre-Flight as fundamental account risk hygiene. | **PRESERVED** (`CLAIM_SET ⊆ EVIDENCE_SET`, no unproven institutional accumulation claims) |
| **`CHK-4`** | **Catalyst Hazard Buffer** | `KEEP_IN_PREFLIGHT` | Binary earnings/FDA event blackout within $\le 7$ days. Stays exclusively in Pre-Flight as an execution timing rule. | **PRESERVED** |
| **`CHK-5`** | **Macro Regime Guard** (VIX $< 26.0$) | `KEEP_IN_PREFLIGHT` & `REFERENCE_IN_GATE_3` | Evaluates broad market volatility from canonical ribbon. Referenced in Gate 3 and verified in Pre-Flight. Fail-closed if missing. | **PRESERVED** (Single authority `GET /api/v1/macro/ribbon`, fail-closed if missing) |

---

## 11. Mobile Readiness & Responsive Budget Contract

The Wave 4 audit revealed that mobile decision traversal was degraded due to repetitive, vertically stacked cards. Wave 4 ratifies measurable mobile viewport constraints:

### 11.1 Viewport Scope
- Primary Mobile Baseline: **390 × 844 px** (iPhone 12/13/14 Pro).
- Tablet & Desktop Baselines: **768 × 1024 px**, **1024 × 768 px**, **1280 × 800 px**, **1440 × 900 px**.

### 11.2 Mobile Scroll-Depth Budget (390 × 844)
1. **Critical Decision Payload**:
   At 390 × 844, the user MUST be able to view all five critical decision elements within the **first 1.5 viewport heights ($\le 1266\text{ px}$ of cumulative vertical scroll depth)**:
   - Verdict Badge & State
   - Primary Qualification Reason
   - Active Blocking Gate Badge
   - Primary Operational CTA Button
   - Negative Guidance Banner (if applicable)
2. **Horizontal Overflow**:
   $$\text{Horizontal Overflow} = 0\text{ px} \quad (\text{document.documentElement.scrollWidth} \le 390\text{ px})$$
3. **Progressive Disclosure**:
   On mobile screens ($< 640\text{px}$), the 3 Decision Readiness Gates are displayed as a compact horizontal stepper showing the active blocker by default. Detailed explanation text for downstream gates is placed inside a collapsible progressive disclosure drawer or tap-to-expand card, preventing vertical bloat.

### 11.3 Test Capability Classification
- **`TRUE_BROWSER_E2E_AVAILABLE`**: **YES** (via Puppeteer with real Chromium runtime). Supports viewport setting (1440×900, 1280×800, 1024×768, 768×1024, 390×844), DOM coordinate inspection (`getBoundingClientRect().top <= 1266px`), scrollWidth validation (`scrollWidth <= viewport width`), and touch target sizing ($\ge 44\times 44\text{px}$).
- **`WEBKIT_AVAILABLE`**: **NO** (Playwright WebKit is not installed in `package.json`; Puppeteer runs Chromium).
- **`PHYSICAL_IOS_REQUIRED`**: **YES** for native iOS Safari WebKit gesture quirks. Layout bounds and touch metrics are verified in Chromium mobile emulation.

---

## 12. Accessibility Contract

Wave 4 enforces strict web accessibility (WCAG 2.1 AA) standards:

1. **Non-Color-Only Communication**:
   No gate or decision state may be communicated solely by color (e.g. red/green/amber). Every indicator must include:
   - An explicit text label (e.g. `[PASSED]`, `[BLOCKING]`, `[PENDING]`, `[UNAVAILABLE]`).
   - A distinct graphical icon or symbol (e.g. `✓`, `🛑`, `⏳`, `—`).
2. **Touch Targets**:
   All interactive buttons, CTA links, and stepper tabs must provide a minimum hit area of **44 × 44 px** at mobile viewports (`min-h-[44px]`, `min-w-[44px]`).
3. **Semantic Hierarchy**:
   Proper heading nesting (`h1` for page title, `h2` for primary workstation modules, `h3` for cards and gates).
4. **Focus Management & Keyboard Navigation**:
   - Modals (`AlertTriggerModal`, `PositionSizerModal`, `PreFlightChecklistModal`, `WatchlistDrawer`) must trap focus when open and return focus to the triggering element upon closure.
   - Modals must support instant dismissal via the `Escape` key.
   - All interactive elements must be reachable via `Tab` with visible focus rings (`focus-visible:ring-2 focus-visible:ring-cyan-400`).
5. **Screen Reader Announcements**:
   - The active blocker and primary CTA must provide descriptive `aria-label` attributes.
   - State transitions and alert save confirmations must announce via an `aria-live="polite"` region.

---

## 13. Wave 1–3 and QA-ESC-011 Invariant Preservation

Wave 4 explicitly commits to preserving all prior ratified baselines:

```text
WAVE_1 =
  Declutter, single primary landmark, unified information hierarchy (FROZEN)

WAVE_2 =
  First-viewport composition, cockpit dock, chart ResizeObserver refit, zero vertical shift (FROZEN)

WAVE_3 =
  Epistemic integrity (CLAIM_SET ⊆ EVIDENCE_SET), canonical VIX authority,
  instrument-specific applicability (ETFs vs Equities), Smart Money archive semantics (FROZEN)

QA_ESC_011 =
  Decision-surface integrity: decisionStateLabel consumed canonically, zero duplicate headlines,
  actionability decoupled from wait-for-trigger, fail-closed fallback (FROZEN)
```

No Wave 4 change may alter, weaken, or reopen these frozen baseline contracts.

---

## 14. Acceptance Criteria

- **`AC-W4-001`**: Ordered Decision Readiness Gates evaluate sequentially: Gate 1 (Location) $\to$ Gate 2 (Trigger) $\to$ Gate 3 (Risk Clearance).
- **`AC-W4-002`**: Exactly one Active Blocker is identified whenever a setup is not execution-ready; when all 3 gates pass, Active Blocker is `NONE`.
- **`AC-W4-003`**: Downstream gates correctly reflect `PENDING_DEPENDENCY` whenever an upstream prerequisite gate is blocking or unavailable. Downstream gates never display as failed when prerequisites are unmet.
- **`AC-W4-004`**: Protective negative guidance copy is displayed only when canonical evidence warrants it (`CLAIM_SET ⊆ EVIDENCE_SET`).
- **`AC-W4-005`**: Every user-visible operational CTA maps strictly to a genuine, production-verified backing capability. No mock toasts, no placeholder buttons.
- **`AC-W4-006`**: No non-actionable primary CTA opens an informational explanation modal (such as "Why Score") as its primary action.
- **`AC-W4-007`**: Radar, Terminal Cockpit, and Execution Cards consume the unified shared state nomenclature authority (`frontend/lib/decisionPresentation.ts`).
- **`AC-W4-008`**: Pre-Flight checklist and Decision Readiness boundaries are strictly demarcated without duplicate formulas or epistemic degradation.
- **`AC-W4-009`**: Mobile critical-decision payload (Verdict, Reason, Active Blocker, Next Action, Negative Guidance) renders within the 1.5 viewport height budget ($\le 1266\text{ px}$) at 390 × 844 px.
- **`AC-W4-010`**: Accessibility standards are satisfied: $\ge 44\text{px}$ touch targets, non-color-only state indicators, focus trap, and ARIA announcements.
- **`AC-W4-011`**: Wave 1 declutter and single-landmark layout contracts remain 100% satisfied.
- **`AC-W4-012`**: Wave 2 first-viewport composition and chart ResizeObserver contracts remain 100% satisfied.
- **`AC-W4-013`**: Wave 3 epistemic integrity, canonical VIX, and instrument applicability contracts remain 100% satisfied.
- **`AC-W4-014`**: QA-ESC-011 decision-surface integrity is 100% preserved (zero generic "Wait for Trigger" fallbacks, zero actionability collapse, zero duplicate secondary badges).

---

## 15. Regression & Test Contract

Prior to implementation authorization, the test matrix is formally defined:

1. **QA-ESC-011 Permanent Regression Suite**:
   - `frontend/components/__tests__/DecisionSurfaceIntegrity.test.tsx` (12/12 passing)
   - `frontend/tests/qaEscapeSemanticInvariants.test.ts` (8/8 passing)
   - `frontend/tests/decisionContract.test.ts` (Passing)
   - Assert representative states (`VALID_SETUP + WAITING_PULLBACK`, `VALID_SETUP + IN_BUY_ZONE_AWAITING_TRIGGER`, `AVOID`, `HOLD/OWNED`, `UNVERIFIED`, `ACTIONABLE_SETUP`) never render duplicate `"Wait for Trigger"` or collapse `!isActionable` to trigger states.
2. **State & Readiness Logic Tests** (`frontend/tests/decisionReadiness.test.ts`):
   - Assert Gate 1 blocks when price is extended $>2\%$ or below buy zone.
   - Assert Gate 2 reports `PENDING_DEPENDENCY` when Gate 1 is blocking.
   - Assert Gate 3 reports `PENDING_DEPENDENCY` when Gate 2 is blocking.
   - Assert Gate 3 reports `UNAVAILABLE` (fail closed) when VIX is `null` or uncalculated.
   - Assert all 3 gates pass when in buy zone, trigger confirmed, $R:R \ge 2.0$, and $VIX < 26.0$.
   - Assert Active Blocker is `NONE` when all 3 gates pass.
3. **State Harmonization Tests** (`frontend/tests/decisionPresentation.test.ts`):
   - Assert Radar, Terminal, and Execution cards resolve identical presentation labels and badges for every canonical domain state.
4. **Operational CTA Binding Tests** (`frontend/tests/operationalActions.test.ts`):
   - Assert `ACTIONABLE` state yields `SIZE_POSITION` primary CTA.
   - Assert `WAITING_PULLBACK` yields `SET_PULLBACK_ALERT` primary CTA.
   - Assert `AWAITING_TRIGGER` yields `SET_BUY_ZONE_ALERT` primary CTA.
   - Assert Stage 4 yields `SET_50_SMA_PIVOT_ALERT` secondary CTA.
   - Assert `AlertManager.saveAlertRule` is invoked on alert save.
   - Assert Watchlist actions trigger drawer navigation only and never emit false cloud-persistence toasts.
5. **Epistemic & Negative Guidance Tests** (`frontend/tests/negativeGuidance.test.ts`):
   - Assert negative guidance banners appear if and only if triggering evidence is present.
   - Assert 2:1 R:R threshold copy is emitted only when $R:R < 2.0$.
6. **Responsive & Viewport Tests** (`frontend/tests/e2e/decision-readiness-viewports.spec.ts`):
   - Test 1440×900, 1280×800, 1024×768, 768×1024, and 390×844.
   - Assert zero horizontal scrollbar across all viewports.
   - Assert critical decision elements render within $1266\text{px}$ scroll depth at 390×844.
7. **Wave 1–3 Regression Suite**:
   - Run existing suites (`Wave1DeclutterPreservation.test.tsx`, `Wave2FirstViewportComposition.test.tsx`, `wave3DecisionIntegrity.test.ts`) and assert 100% passing.

---

## 16. Domain Boundary Matrix

| Proposed Wave 4 Field / Component | Classification | Implementation Allowance |
| :--- | :--- | :--- |
| Readiness Stepper UI & Badges | `PRESENTATION` | Authorized |
| Readiness Copy & Explanations | `UX_COPY` | Authorized |
| `resolveDecisionReadiness()` helper | `FRONTEND_STATE` | Authorized (Pure frontend resolver) |
| `decisionPresentation.ts` resolver | `FRONTEND_DATA_BINDING` | Authorized (Consumes existing contract) |
| Stepper clicks & Modal triggers | `INTERACTION` | Authorized |
| New REST / WebSocket endpoints | `API_CONTRACT` | **UNAUTHORIZED** (Frozen) |
| Python quantitative analyzers | `QUANT_MODEL` / `DOMAIN_LOGIC` | **UNAUTHORIZED** (Frozen) |
| Database migrations / schemas | `DATABASE` | **UNAUTHORIZED** (Frozen) |
| Matomo interaction analytics | `OBSERVABILITY` | Authorized |
