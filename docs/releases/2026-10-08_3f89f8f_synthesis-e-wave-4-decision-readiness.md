# ARX Terminal — Production Release Notes

## Synthesis E Wave 4: Decision Readiness, Next-Action Clarity & State Harmonization

### Release Identity

```ini
RELEASE_DATE =
  2026-10-08
RELEASE_SHA =
  3f89f8f412b031018f3c8c4e5a827264696ee272
FUNCTIONAL_RELEASE_SHA =
  3f89f8f412b031018f3c8c4e5a827264696ee272
PREVIOUS_PRODUCTION_SHA =
  6e10051535281deaa8115dbde6e9083cc7818446
PRD_BASELINE_SHA =
  7776ead4770ec1ba6c14dd30017e5cdf8a850397
BRANCH =
  main
DEPLOYMENT_TRIGGER =
  GitHub push -> automatic Railway & Cloudflare Pages deployment
DEPLOYMENT_STATUS =
  DEPLOYED
PRODUCTION_VERIFICATION =
  VERIFIED
```

---

### Release Classification & Behavior Invariants

```ini
RELEASE_CLASSIFICATION =
  UX_PRODUCT_FEATURE
  DECISION_READINESS
  SEMANTIC_HARMONIZATION
  QA_HARDENING

PRODUCTION_APPLICATION_BEHAVIOR_CHANGE =
  YES

QUANTITATIVE_BEHAVIOR_CHANGE =
  NONE

MODEL_TUNING =
  NO

PROSPECTIVE_CAPTURE_SEMANTIC_CHANGE =
  NO

DATABASE_CHANGE =
  NONE

HUMAN_VALIDATION_EXECUTED =
  NO

CONTRAST_RUNTIME_MEASUREMENT =
  NOT_EXECUTED

WEBKIT_E2E =
  NOT_EXECUTED

PHYSICAL_IOS =
  NOT_EXECUTED
```

This release delivers Synthesis E Wave 4: Decision Readiness Progression, Next-Action Clarity & State Harmonization across the execution surface. Zero modifications were made to quantitative models, ranking algorithms, position sizing mathematics, execution-ladder equations, database schemas, or prospective capture stores.

---

### Major Behaviors Delivered

1. **3-Gate Decision Readiness Progression**:
   - `Gate 1: Corridor Location & Geometry` — Evaluates accumulation corridor entry boundaries and invalidation structural floors.
   - `Gate 2: Dynamic Trigger & Volume Confirmation` — Evaluates confirmation triggers, volume spikes, and contraction breakouts.
   - `Gate 3: Risk Floor & Macro Clearance` — Evaluates VIX macro climate (< 26), market regime, and statutory stop viability.

2. **Single Active Blocker Invariant**:
   - Exactly one blocking gate is identified as active at any time (`EXACTLY_ONE_ACTIVE_BLOCKER = YES`).
   - Prevents overwhelming users with contradictory or multiple simultaneous blocker states.

3. **Dependency Cascade**:
   - Downstream gates evaluate to `PENDING_DEPENDENCY` when an upstream precondition is `BLOCKING`.
   - Eliminates fabricated failures on dependent downstream checks (e.g., when price is extended above corridor, Gate 2 & Gate 3 remain cleanly pending).

4. **Operational Next Actions**:
   - Primary operational call-to-actions (`Set Pullback Alert`, `Size Position`, `Set Buy Zone Alert`, `Explore Radar`) wired to genuine client-side handlers and modals.
   - Zero fake CTAs, zero dummy links, and zero speculative trigger claims.

5. **Evidence-Grounded Negative Guidance**:
   - Protective guidance banners (`DO NOT CHASE`, `DO NOT PREEMPT`, `CAPITAL DEFENSE`, `MACRO CAUTION`) strictly enforced with `CLAIM_SET ⊆ EVIDENCE_SET`.
   - In live verification for extended price, renders evidence-grounded `"DO NOT CHASE: Price has extended past the accumulation corridor. Wait for a pullback to the buy zone."`

6. **Shared Decision Presentation & Semantic Harmonization**:
   - Preserves QA-ESC-011 frozen invariants across all views (`StandardTerminalView`, `AdaptiveTerminalView`).
   - Clean domain separation: `decisionStateLabel` presented truthfully with zero generic `"WAIT FOR TRIGGER"` collapse.
   - Actionability badge strictly restricted to `ACTIONABLE` vs `NOT ACTIONABLE`.

7. **Mobile Progressive Disclosure**:
   - Responsive layout verified across 5 standard viewports (1440×900, 1280×800, 1024×768, 768×1024, 390×844).
   - Mobile 390×844 critical payload bottom verified at 1064px ($\le 1266\text{px}$ threshold).
   - Horizontal overflow = 0.
   - Collapsible precondition disclosure on mobile screens.

---

### Verification & Acceptance Evidence

#### 1. Live Rendered Production Acceptance (NAUT LONG_TERM)
* **Domain / URL**: `https://www.arxterminal.com/?symbol=NAUT`
* **ANALYTICAL_VERDICT**: `Valid Setup — Awaiting Trigger`
* **ACTIONABILITY_BADGE**: `NOT ACTIONABLE`
* **DECISION_READINESS_CARD**: Present (`[data-testid="decision-readiness-card"]`)
* **GATE_1_STATE**: `BLOCKING` (Price $1.96 extended above $1.27–$1.46 corridor)
* **GATE_2_STATE**: `PENDING` (Pending upstream corridor resolution)
* **GATE_3_STATE**: `PENDING` (Pending upstream corridor resolution)
* **ACTIVE_BLOCKER**: `Wait for pullback into optimal accumulation corridor without chasing.`
* **PRIMARY_CTA**: `Set Pullback Alert` (triggers functional live Alert modal)
* **PRE_FLIGHT_CTA**: `✈️ Pre-Flight` (triggers functional live Pre-Flight Checklist modal)

#### 2. Responsive & Viewport Acceptance
* `1440×900`: `horizontalOverflow = 0`
* `1280×800`: `horizontalOverflow = 0`
* `1024×768`: `horizontalOverflow = 0`
* `768×1024`: `horizontalOverflow = 0`
* `390×844`: `horizontalOverflow = 0`, `criticalPayloadBottom = 1064px` ($\le 1266\text{px}$)

#### 3. Epistemic & Evidence Boundaries
* `QUANTITATIVE_BEHAVIOR_CHANGE`: `NONE`
* `MODEL_TUNING`: `NO`
* `PROSPECTIVE_CAPTURE_SEMANTIC_CHANGE`: `NO`
* `DATABASE_CHANGE`: `NONE`
* `HUMAN_VALIDATION_EXECUTED`: `NO`
* `CONTRAST_RUNTIME_MEASUREMENT`: `NOT_EXECUTED`
* `WEBKIT_E2E`: `NOT_EXECUTED`
* `PHYSICAL_IOS`: `NOT_EXECUTED`
