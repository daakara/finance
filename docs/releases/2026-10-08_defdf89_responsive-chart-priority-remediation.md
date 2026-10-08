# ARX Terminal — Production Release Notes

## Responsive Chart Priority & Data Authority Remediation

### Release Identity

```ini
RELEASE_DATE =
  2026-10-08
RELEASE_SHA =
  defdf8900ab64d14705b57ecdd0c8ed902526195
FUNCTIONAL_RELEASE_SHA =
  defdf8900ab64d14705b57ecdd0c8ed902526195
PREVIOUS_RUNTIME_SHA =
  9c02b2f942c5d272dd8060d1c82322d5090099e9
PREVIOUS_FUNCTIONAL_RELEASE_SHA =
  3f89f8f412b031018f3c8c4e5a827264696ee272
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
  RESPONSIVE_UX_REMEDIATION
  DATA_AUTHORITY_HARDENING
  QA_ESCAPE_PREVENTION

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

WEBKIT_E2E =
  NOT_EXECUTED

PHYSICAL_IOS =
  NOT_EXECUTED
```

This release delivers the Responsive Chart Priority & Data Authority Remediation for ARX Terminal. Zero modifications were made to quantitative models, ranking algorithms, position sizing mathematics, execution-ladder equations, database schemas, or prospective capture stores.

---

### Executive Summary & Remediation Context

During post-Wave-4 inspection, two responsive layout discrepancies and one data-authority consideration were addressed:
1. **Desktop Workspace Alignment Shift (+8px)**: On 1440×900 and 1280×800 desktop viewports, an unintended +8px vertical delta separated the top of the Analytical Verdict card and the Price Chart container.
2. **Mobile/Tablet Viewport Priority Drift**: Below the `xl` breakpoint, the Price Chart was subordinated below the Execution / Decision Readiness surfaces rather than maintaining chart context immediately after the decision verdict.
3. **Mobile Data Authority & Fallback Hardening**: The mobile chart preview affordance was hardened to bind strictly to canonical market and quant authority, failing closed to `UNAVAILABLE` rather than admitting synthetic or generic fallback trading levels.

---

### Root Cause Analysis & Architectural Remediation

#### 1. Desktop Chart/Verdict Zero-Delta Restoration
* **Root Cause**: The right-column wrapper in `StandardTerminalView.tsx` previously utilized Tailwind's `space-y-2` utility class. The introduction of the `<div data-testid="mobile-chart-preview" className="md:hidden ...">` element as the first child caused Tailwind's child combinator (`> :not([hidden]) ~ :not([hidden])`) to match. Because `md:hidden` applies `display: none` via CSS media query rather than the HTML boolean attribute `[hidden]`, the browser applied `margin-top: 0.5rem` (8px) onto `#market-workspace-chart` at desktop viewports.
* **Remediation**: Removed `space-y-2` from the outer right-column wrapper and localized margin spacing (`mb-2`) directly onto the mobile preview affordance. This restored exact parallel top alignment (`VERDICT_TOP == CHART_TOP == 152px`, `DELTA = 0px`) across all desktop and laptop resolutions.

#### 2. Mobile Chart Priority & Collapsible Preview Affordance
* **Layout Hierarchy**: Below `md` breakpoints (<768px), standard terminal view renders a compact 56–64px `mobile-chart-preview` affordance between the Analytical Verdict and the trade plan.
* **Progressive Disclosure**: Provides an accessible interactive toggle (`aria-expanded`, `aria-controls="market-workspace-chart"`, `min-h-[44px] min-w-[44px]` touch target) allowing mobile operators to expand the full interactive candlestick chart on demand without displacing the critical decision payload.
* **Tablet Landscape (1024×768)**: Corrected column grid to `lg:grid-cols-12` (`lg:col-span-5` / `lg:col-span-7`), restoring side-by-side parallel layout at 1024px.
* **Tablet Portrait (768×1024)**: Preserved natural stacked order: Verdict (152px) → Full Price Chart (730px) → Conditional Trade Plan (1318px).

#### 3. Data Authority & Fail-Closed Guardrails
* **Canonical Authority**: Spot price is bound strictly to `insight.price` / `kl.currentPrice`; watch zone corridor is bound strictly to `kl.watchZone` / `kl.entryMin` / `kl.entryMax`.
* **Zero Synthetic Fallbacks**: Missing spot prices and uncalculated watch zones strictly render `UNAVAILABLE`. Prohibits fabricated `$0.00` prices, generic fallback ranges, or speculative corridors.
* **Deterministic Test Proof**: Formal unit suite `frontend/components/__tests__/ResponsiveChartPriorityRemediation.test.tsx` validates 4 discrete authority scenarios (full data, spot only, zone only, missing both).

---

### Verification Evidence & Measurement Matrix

#### 1. Live Rendered Desktop Geometry (Production `https://www.arxterminal.com`)
* `1440×900 Desktop`: `WORKSPACE_TOP = 152px`, `VERDICT_TOP = 152px`, `CHART_TOP = 152px`, `DELTA = 0px`, `OVERFLOW = 0px` (PASS)
* `1280×800 Laptop`: `WORKSPACE_TOP = 152px`, `VERDICT_TOP = 152px`, `CHART_TOP = 152px`, `DELTA = 0px`, `OVERFLOW = 0px` (PASS)
* `1024×768 Tablet Landscape`: `WORKSPACE_TOP = 152px`, `VERDICT_TOP = 152px`, `CHART_TOP = 152px`, `DELTA = 0px`, `OVERFLOW = 0px` (PASS)
* `Cold State (Unseeded Tour)`: `VERDICT_TOP == CHART_TOP`, `DELTA = 0px` (Zero layout shift from onboarding shell)

#### 2. Live Mobile & Tablet Responsive Contract
* `768×1024 Tablet Portrait`: Stacked order verified (Verdict 152px → Chart 730px → Plan 1318px), `OVERFLOW = 0px` (PASS)
* `390×844 Mobile Viewport`:
  - `CRITICAL_PAYLOAD_BOTTOM`: 1173px ($\le 1266\text{px}$ threshold -> PASS)
  - `EXPANDED_CHART_CANVAS_HEIGHT`: 665px ($\ge 220\text{px}$ local QA threshold -> PASS)
  - `TOUCH_TARGET_MINIMUM`: 201×44px ($\ge 44\times 44\text{px}$ touch target requirement -> PASS)
  - `HORIZONTAL_OVERFLOW`: 0px (PASS)

#### 3. Quality Gates & Test Suites
* `TypeScript Typecheck`: 0 errors
* `Unit Tests (Vitest)`: 25 test files passed, 231 tests passed (0 failures)
* `Architectural & Invariant Suites (`test:arch`)`: 21 suites passed (100% coverage)
* `Production Build`: Next.js 14.2.35 static export completed with 145/145 pages generated and 0 errors.

#### 4. QA-ESC-011 & Wave 4 Invariant Preservation
* `QA-ESC-011 Invariants`: Zero generic `"WAIT FOR TRIGGER"` collapse, truthful domain labels preserved (`NOT ACTIONABLE`).
* `Wave 4 Decision Readiness`: Decision Readiness Card, Active Blocker, Dependency Cascade, and Operational CTAs strictly preserved.
