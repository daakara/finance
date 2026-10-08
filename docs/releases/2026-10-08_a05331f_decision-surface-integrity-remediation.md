# ARX Terminal — Production Release Notes

## Decision-Surface Integrity Remediation (QA-ESC-011)

### Release Identity

```ini
RELEASE_DATE =
  2026-10-08
RELEASE_SHA =
  a05331f0accf17c004c0c2efb1ff8e4af59ed9a1
PREVIOUS_RUNTIME_SHA =
  36d75a6ff78ee9d319514ece2decbeccd2653bfd
BRANCH =
  main
DEPLOYMENT_TRIGGER =
  GitHub push -> automatic Railway & Cloudflare deployment
DEPLOYMENT_STATUS =
  DEPLOYED
PRODUCTION_VERIFICATION =
  VERIFIED
```

---

### Release Classification & Behavior Invariants

```ini
RELEASE_CLASSIFICATION =
  PRODUCT_BUG_FIX
  QA_HARDENING
  SEMANTIC_INTEGRITY
  UX_CORRECTION

PRODUCTION_APPLICATION_BEHAVIOR_CHANGE =
  YES

QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE

MODEL_TUNING =
  NO

PASSIVE_CAPTURE_SEMANTIC_CHANGE =
  NO

TARGET_REACHED_REGRESSION =
  NO
```

This release remediates QA-ESC-011, restoring decision-surface semantic integrity across the ARX Analytical Verdict card. Zero modifications were made to quantitative models, ranking algorithms, sizing equations, execution-ladder math, or passive capture stores.

---

### QA Escape Context: QA-ESC-011

* **QA_ESCAPE_ID**: `QA-ESC-011`
* **DATE**: `2026-10-08`
* **SEVERITY**: `P1_DECISION_SURFACE_INTEGRITY_DEFECT`
* **PRODUCTION_SYMPTOM**:
  On the Analytical Verdict surface (observed on NAUT production screenshot), the exact phrase `"Wait for Trigger"` rendered three times simultaneously:
  1. Primary Headline: `"Wait for Trigger"`
  2. Amber Badge: `"WAIT FOR TRIGGER"`
  3. Secondary Grey Badge: `"Wait for Trigger"`
  Furthermore, on non-actionable setups (`AVOID`, `HOLD`/`OWNED`, `UNVERIFIED`), the actionability badge collapsed to `"WAIT FOR TRIGGER"`, presenting contradictory operational instructions.
* **ROOT_CAUSE**:
  1. *Backend/Frontend Contract Mismatch*: Backend `DecisionHierarchyEngine` emitted `decisionStateLabel`, while frontend `assessmentEngine.ts` read `decisionTrace?.stateLabel`, causing fallback to generic `"Wait for Trigger"`.
  2. *Unsafe Missing-State Fallback*: `assessmentEngine.ts` line 216 defaulted undefined `decisionTrace` to `"Wait for Trigger"` rather than failing closed to an explicit neutral/unassessed condition (`"Setup Evaluation Pending"`).
  3. *Actionability Presentation Collapse*: Terminal views (`StandardTerminalView`, `GuidedTerminalView`, `AdvancedTerminalView`) hardcoded `{isActionable ? "ACTIONABLE" : "WAIT FOR TRIGGER"}`, falsely equating `!isActionable` with awaiting a trade trigger.
  4. *Redundant Secondary Badge*: Terminal views rendered `{insight.verdictLabel}` as headline and `{insight.terminalState.uiStateLabel}` as secondary grey badge, repeating identical text when `verdictLabel` was mapped to `uiStateLabel`.

---

### User Visible Change

#### Before
```text
[Wait for Trigger]              (Headline)
[WAIT FOR TRIGGER]              (Amber Badge)
[Wait for Trigger]              (Grey Badge)
```

#### After
```text
Valid Setup — Awaiting Trigger   (Canonical Analytical Verdict)
[NOT ACTIONABLE]                (Actionability State Badge)
[Partial: Price Tape Only]      (Independent Evidence Quality Badge)
(Redundant duplicate grey badge eliminated)
```

On non-actionable states (`AVOID`, `HOLD`, `UNVERIFIED`), the operational badge cleanly reads `NOT ACTIONABLE` instead of contradictory `WAIT FOR TRIGGER`.

---

### Verification & Acceptance Evidence

#### 1. Live Rendered Production Acceptance (NAUT LONG_TERM)
* **Domain / URL**: `https://www.arxterminal.com/?ticker=NAUT`
* **ANALYTICAL_VERDICT**: `Valid Setup — Awaiting Trigger`
* **ACTIONABILITY_BADGE**: `NOT ACTIONABLE`
* **EVIDENCE_QUALITY**: `Partial: Price Tape Only`
* **REDUNDANT_SECONDARY_BADGE**: `ABSENT`
* **DUPLICATE_WAIT_FOR_TRIGGER_COUNT**: `0`
* **LIVE_SEMANTIC_COHERENCE**: `PASS`
* **Screenshots Captured**:
  * Desktop (1280×800): `screenshots/naut_decision_surface_remediated_1280x800.png`
  * Mobile (390×844): `screenshots/naut_decision_surface_remediated_390x844.png`

#### 2. Quantitative Non-Regression
* **Execution Status**: `WAITING_PULLBACK`
* **Market Location**: `BETWEEN_TP1_AND_TP2`
* **Planned Entry**: `1.46`
* **Structural Invalidation**: `1.23`
* **Take Profit 1**: `1.89`
* **Take Profit 2**: `2.17`
* **Target Reached Regression**: `NO`

#### 3. Test Suites Executed Against Candidate Commit
* **Frontend Typecheck (`tsc --noEmit`)**: PASS (0 errors)
* **Decision Surface Integrity Suite (`DecisionSurfaceIntegrity.test.tsx`)**: PASS (12/12 passed)
* **Analysis Decision Hierarchy Suite (`AnalysisDecisionHierarchy.test.tsx`)**: PASS (23/23 passed)
* **Full Vitest Suite**: PASS (217/217 passed across 23 files)
* **Decision Contract Suite (`decisionContract.test.ts`)**: PASS (5/5 passed)
* **Permanent QA Invariants (`qaEscapeSemanticInvariants.test.ts`)**: PASS (8/8 passed)
* **Python QA Escape Invariants (`test_qa_escape_invariants.py`)**: PASS (8/8 passed)
* **Passive Capture Engine (`test_execution_ladder_passive_capture.py`)**: PASS (42/42 passed)

---

### Passive Capture & Prospective Denominator

```ini
PRE_RELEASE_PROSPECTIVE_DENOMINATOR =
  0
POST_DEPLOY_PROSPECTIVE_DENOMINATOR =
  0
DENOMINATOR_DELTA =
  0
NATURAL_CAPTURE_DETECTED_DURING_RELEASE =
  NO
```

No synthetic records were injected during verification. The prospective denominator remains clean at 0.
