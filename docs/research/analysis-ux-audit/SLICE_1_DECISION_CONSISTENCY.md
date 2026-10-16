# ARX Terminal — Workstream D, Slice 1 Implementation Note
## Decision Text Truthfulness & Blocked-Action Consistency

- **Backlog Reference**: `BACKLOG-ARX-ANALYSIS-001` (Workstream D, Slice 1)
- **Mode**: `ISOLATE → TRACE → IMPLEMENT MINIMAL FIX → TEST → LOCAL COMMIT → STOP`
- **Date**: 2026-10-16
- **Status**: Completed Locally (Zero Remote Mutations)

---

### 1. Context & Worktree Isolation

- **Repository**: `daakara/finance`
- **Worktree**: `C:\Users\akara\Documents\Projects\finance-analysis-decision-consistency`
- **Branch**: `fix/analysis-decision-consistency`
- **Base Commit**: `2eb032b2aec18ac4fea0f654eee8fb6c028466fe` (`origin/main`, PR #5 merged)
- **Base Tree**: `483bee72e2f8df63adecf15891b6e0091d52a738`
- **Isolation Boundaries**:
  - `APPLICATION_CHANGES`: Minimal frontend correction only
  - `BACKEND_POLICY_CHANGES`: None (Zero edits to Python engines, backend scoring, mathematical boundaries, thresholds, stops, or eligibility)
  - `TOLERANCE_CHANGES`: None (The canonical 2% extension tolerance is strictly preserved)
  - `REMOTE_OPERATIONS`: None (No push, no PR, no remote deployment)

---

### 2. Core Defect Diagnosis & Root Causes

During the ARX Terminal user experience audit (`LNTH` asset inspection), two critical consistency defects were identified:

1. **Spatial Inversion Defect**:
   - Spot price ($341.91) exceeded the displayed optimal accumulation corridor ceiling ($325.07–$339.38) by +$2.53 (+0.75%).
   - While within the 2% extension tolerance window ($346.17 ceiling), the UI erroneously emitted the statement:
     `"Price ($341.91) is positioned inside the institutional accumulation corridor ($325.07–$339.38)"`.
   - Describing an extended price as "inside" the corridor created a direct spatial falsehood that contradicted the visual chart and corridor numbers.

2. **Action Contradiction Defect**:
   - The primary decision state was canonical `Wait` (Gate 2 confirmation trigger was pending).
   - Simultaneously, the tactical guidance card rendered:
     `"Scale in with 30% initial size or place limit orders near $325.07..."`.
   - Emitting actionable scale-in instructions while the canonical verdict and readiness gates were explicitly blocking generated an irreconcilable contradiction for users.

3. **Inverted Bounds Vulnerability**:
   - If raw corridor bounds were inverted (`entryMin > entryMax`), legacy normalization could silently invert bounds or generate nonsensical pass states rather than failing closed.

---

### 3. Architecture & Surgical Modifications

#### 3.1 Pure Location Helper: `frontend/lib/corridorLocation.ts` (New File)
- Implemented `evaluateCorridorLocation(price, rawMin, rawMax, toleranceRatio = 0.02)` as a pure mathematical evaluator decoupled from policy gates.
- Strictly separates geometric spatial position on the real line (`BELOW`, `INSIDE`, `ABOVE`) from tolerance qualification (`isWithinTolerance`).
- Inclusive boundary conventions:
  - `price === entryMin` -> `INSIDE`
  - `price === entryMax` -> `INSIDE`
  - `price > entryMax` -> strictly `ABOVE`, `isInside: false`
- Fail-Closed Invariants:
  - Missing, non-finite, non-positive, or inverted (`rawMin > rawMax`) inputs return `isValid: false`, `spatialRelation: "UNAVAILABLE"`.

#### 3.2 Decision Readiness: `frontend/lib/decisionReadiness.ts` (Modified)
- Consumes `evaluateCorridorLocation`.
- **Gate 1**: Preserves existing 2% tolerance decision behavior (`gate1State = "PASSED"` when `isWithinTolerance`), but emits truthful explanation:
  `"Price ($341.91) is above the entry corridor ($325.07–$339.38). Existing extension tolerance applies."`
- **Gate 2**: If price is above corridor within tolerance, explanation truthfully reports:
  `"Price is above the entry corridor (within extension tolerance), but confirmation trigger is pending. Awaiting volume expansion / pivot reclaim."`
- **Negative Guidance & Action Reasons**: Synchronized to avoid false "in buy zone" claims when price is extended.
- **Pure Tactical Hint Resolver (`resolveTacticalExecutionHint`)**:
  - Exported pure function to evaluate tactical hints for the trade plan card.
  - Strictly requires `isActionSatisfied = Boolean(isActionable) && readinessResult.isExecutionReady`.
  - When `isActionSatisfied` is false, actionable scale-in instructions are suppressed, replacing them with truthful conditional guidance (e.g. `"Spot ($341.91) is above the entry corridor... Existing extension tolerance applies, but entry remains conditional. Wait for required trigger confirmation."`).

#### 3.3 Decision Hierarchy Utils: `frontend/lib/decisionHierarchyUtils.ts` (Modified)
- `deriveUnmetConditions` consumes `evaluateCorridorLocation`.
- Inverted bounds fail closed (`status: "UNMET"`, descriptive reason).
- Pullback condition preserves `status: "UNMET"` when `corridorLoc.spatialRelation === "ABOVE"`.

#### 3.4 Presentation Component: `frontend/components/OptimalEntryExitCard.tsx` (Modified)
- `inZone` calculation strictly requires `corridorLoc.isValid && corridorLoc.isInside`, guaranteeing that prices above the ceiling or below the floor can never be flagged as `inZone`.
- Delegates tactical execution hints to `resolveTacticalExecutionHint`.

---

### 4. Verification Evidence & Acceptance Matrix

A dedicated regression test suite (`frontend/tests/corridorDecisionConsistency.test.ts`) was executed alongside existing architecture and unit suites.

#### 4.1 Corridor Location & Geometry Invariants (9 Cases)
| # | Case | Spot / Bounds | Expected Spatial Relation | Gate 1 State | Verified Text Guarantee | Status |
|---|------|---------------|---------------------------|--------------|-------------------------|--------|
| 1 | Below lower floor | $95 / $98–$102 | `BELOW` | `BLOCKING` | "below the accumulation corridor" | **PASS** |
| 2 | Exact lower floor | $98 / $98–$102 | `INSIDE` | `PASSED` | "positioned inside the institutional accumulation corridor" | **PASS** |
| 3 | Inside corridor | $100 / $98–$102 | `INSIDE` | `PASSED` | "positioned inside the institutional accumulation corridor" | **PASS** |
| 4 | Exact upper ceiling | $102 / $98–$102 | `INSIDE` | `PASSED` | "positioned inside the institutional accumulation corridor" | **PASS** |
| 5 | Just above ceiling | $102.01 / $98–$102 | `ABOVE` | `PASSED` (tolerance) | "above the entry corridor", `isInside: false` | **PASS** |
| 6 | Extended in 2% tol | $103.50 / $98–$102 | `ABOVE` | `PASSED` (tolerance) | "above the entry corridor", mentions tolerance | **PASS** |
| 7 | Extended > 2% | $105 / $98–$102 | `ABOVE` | `BLOCKING` | "extended >2% past the accumulation corridor" | **PASS** |
| 8 | Missing / non-finite | $null, NaN / $98–$102 | `UNAVAILABLE` | `UNAVAILABLE` | Fail-closed, `isValid: false` | **PASS** |
| 9 | Inverted bounds | $100 / $102–$98 | `UNAVAILABLE` | `UNAVAILABLE` | "Corridor bounds are invalid (inverted bounds: min > max)" | **PASS** |

#### 4.2 Decision Guidance & Blocked-Action Consistency (6 Cases)
| # | Scenario | Inputs | Gate Progression | Tactical Guidance Behavior | Status |
|---|----------|--------|------------------|-----------------------------|--------|
| 1 | Inside corridor + trigger blocked | Spot $100, isConfirmed: false | G1: PASS, G2: BLOCK | Scale-in advice suppressed; emits "⏳ Trigger Pending" | **PASS** |
| 2 | Above corridor in tol + trigger blocked | Spot $103.50, isConfirmed: false | G1: PASS, G2: BLOCK | Scale-in advice suppressed; emits "⚠️ Above Corridor Ceiling" | **PASS** |
| 3 | Canonical Wait authoritative | isActionable: false, G1: PASS | Read: NOT READY | Canonical Wait posture respected; scale-in suppressed | **PASS** |
| 4 | Missing trigger confirmation | isConfirmed: null | G1: PASS, G2: UNAVAIL | Gate 2 fails closed; scale-in suppressed | **PASS** |
| 5 | Fully satisfied canonical state | isActionable: true, G1-3: PASS | Read: READY | Tactical scale-in / position advice unlocked | **PASS** |
| 6 | Stage 4 distribution | Stage 4 Markdown | G1: BLOCK, G2-3: PEND | Gate 1 blocks; all tactical hints suppressed | **PASS** |

#### 4.3 LNTH Live Fixture Audit ($341.91 vs $325.07–$339.38)
- `corridorLoc.spatialRelation`: `ABOVE` (`isInside: false`)
- `corridorLoc.isWithinTolerance`: `true` (+$2.53 vs +$6.79 2% limit)
- Gate 1 Explanation: `"Price ($341.91) is above the entry corridor ($325.07–$339.38). Existing extension tolerance applies."` (Zero "positioned inside" claims)
- Gate 2 Explanation: `"Price is above the entry corridor (within extension tolerance), but confirmation trigger is pending. Awaiting volume expansion / pivot reclaim."`
- Tactical Hint: `"⚠️ Above Corridor Ceiling: Spot ($341.91) is above the entry corridor ($325.07–$339.38). Existing extension tolerance applies, but entry remains conditional. Wait for required trigger confirmation."`
- Contradictory `"Scale in with 30% initial size"` guidance: **Eliminated**.
- Unmet Conditions: Correctly reports `Pullback into Accumulation Corridor` as `status: "UNMET"`.

#### 4.4 Automated Test Suite Execution Summary
- `npx tsx tests/corridorDecisionConsistency.test.ts`: **PASSED** (100% of invariants verified)
- `npx tsx tests/decisionReadiness.test.ts`: **PASSED** (100% coverage)
- `npm run type-check` (`tsc --noEmit`): **PASSED** (0 errors)
- `npm run test:arch`: **PASSED** (22/22 suites passed)

---

### 5. Scope Boundary Compliance & Limitations

This slice strictly adheres to the bounded authorization for Slice 1:
- **Preserved Invariants**:
  - The 2% tolerance ratio and its pass criteria are retained without modification.
  - No backend scoring, weights, stops, targets, or eligibility rules were altered.
- **Excluded Backlog Items**:
  - Radar filters and sorting redesign.
  - Ownership structure data collection.
  - Calculation formula disclosures.
  - Data freshness subsystem overhaul.
  - Mobile responsiveness layout adjustments.
