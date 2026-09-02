# Phase 13: Product / UX Readiness Audit Report
**ARX Quantitative Decision Platform**  
*Audit Mode: READ-ONLY AUDIT (Phase 13.0 Discovery)*  
*Audit Date: September 2, 2026*  
*Baseline Test Suite: 191/191 Passed (38.32s)*  
*Status: COMPLETE · HARD STOP AT DISCOVERY GATE · REMEDIATION PENDING USER REVIEW*

---

## 1. Executive Summary

Phase 12 proved that the quantitative engine produces mathematically truthful numbers from empirical data. 

**Phase 13 audits whether a real investor can understand, act upon, and trust those numbers without being misled, confused, or stalled by cognitive dead ends.**

### Key Verdict: CRITICAL PRODUCT/UX REMEDIATION REQUIRED (5 P0s, 6 P1s, 3 P2s)
While the quantitative core is solid, the product currently suffers from **epistemic leakage at the presentation layer**:
1. **Posture Semantics & Epistemic Leakage (P0)**: Assets with zero price history or missing fundamentals can still achieve an `"ACQUIRE"` posture or a `"WATCH"` posture with a fabricated headline ("Price remains below 50-day average ($X)").
2. **Guided Lens Disinformation (P0)**: In Guided View, the 4 explanatory pills hardcode `"Healthy"` and `"Price holding firmly above rising 20 EMA and 50 SMA"` even when data is `UNAVAILABLE` or uncataloged (`SYM`), contradicting the Advanced View.
3. **Absence of Legal & Risk Safe Harbors (P0)**: The platform uses definitive trading directives ("ACQUIRE", "Actionable Setup", "In Buy Zone") with **zero financial disclaimers or risk disclosures**.
4. **Comparison Tool Data Fabrication (P0)**: The `/compare` tool fabricates synthetic metrics (`ROIC: 24.0%`, `Fwd P/E: 25.0x`, `Market Cap: $(price * 0.45)B`) when uncataloged tickers are compared.
5. **Position Sizer Explosive Leverage (P0)**: When price has breached the stop loss, the position sizer clamps risk per share to \$0.01, recommending an absurd 25,000+ share purchase on an invalidated setup.
6. **Cross-Screen Data Decoupling (P1)**: The Screener, Strategy pages, and SEO landing pages maintain disconnected hardcoded price dictionaries (e.g. Comfort Systems `FIX` at \$346.20 in screener vs \$1,560.13 in terminal; `NVDA` at \$128.50 in strategy vs \$224.41 in terminal).

---

## 2. Audit Matrix Across the 12 Readiness Gates

| Gate | Focus | Evaluation Method | Result | Primary Finding |
| :--- | :--- | :--- | :---: | :--- |
| **13.1 First-Use Comprehension** | 30-Second Test & Home UX | Home page intent hero & hierarchy | **PASS** | Clear objective cards ("Find an investment", "Understand a stock", "Check my portfolio"). |
| **13.2 Discovery → Decision** | Canonical User Journeys | Step-through across 5 journeys | **FAIL (P1)** | Screener and Strategy pages contain decoupled prices and drop breadcrumbs (`FINDING-13-01`, `FINDING-13-02`, `FINDING-13-12`). |
| **13.3 Decision Clarity** | "What does ARX think and why?" | Posture semantics & explanation trace | **FAIL (P0)** | Posture derivation leaks missing data into `ACQUIRE` and `WATCH` with fake 50D SMA triggers (`FINDING-13-03`). |
| **13.4 Evidence Transparency** | Fact vs calculation vs missing data | Audit unavailable-state copy | **FAIL (P0)** | Guided View `whyPills` asserts "Healthy" for missing domains (`FINDING-13-04`). Standard View displays targets on missing trend (`FINDING-13-05`). |
| **13.5 Action Clarity** | Next step, trigger & invalidation | Terminal CTA & Position Sizer | **FAIL (P0)** | Sizer explodes leverage on breached setups (`FINDING-13-14`). Sizer button shown for `AVOID` (`FINDING-13-13`). |
| **13.6 Lens Usefulness** | Guided vs Standard vs Advanced | Cross-lens comparison on same asset | **FAIL (P0)** | Severe Lens Desync: Guided says "Healthy" while Advanced says "N/A" (`FINDING-13-04`). |
| **13.7 Mobile UX** | 375px, 390px, 430px viewports | Inspection of CSS grids & overflow | **PASS** | Fixed bottom dock with 44px tap targets; `pb-28` clearance; responsive card grids. |
| **13.8 Degraded States** | Ingestion failure & uncataloged assets | Error catching & fallback paths | **FAIL (P1)** | Ingestion errors fail silently and fall back to mock \$100 price (`FINDING-13-06`). |
| **13.9 Accessibility** | Keyboard, ARIA, dialogs, focus | Dialog markup & focus flow | **FAIL (P2)** | Modals lack focus traps and focus restoration upon closing (`FINDING-13-07`). |
| **13.10 Trust & Disclosure** | Regulatory language & freshness | Sweeps for financial disclaimers | **FAIL (P0)** | Zero financial disclaimers or non-fiduciary disclosure copy exists across the app (`FINDING-13-08`). |
| **13.11 Performance** | Load latency & transitions | Bundle traces & client cache | **PASS** | 99 SSG pages; <10ms client memory cache; 268 kB first load JS. |
| **13.12 Real-World Utility** | Defensible decision support | 30-Second Test on real scenarios | **FAIL (P1)** | User misled if starting from Screener or Guided View on uncataloged tickers. |

---

## 3. The "30-Second Test" Across the 5 Canonical Journeys

### Journey 1: Discovery (`Home → Find Investment → Screener → Candidate → Terminal`)
- **What am I looking at?** Screener provides clear goals ("In Buy Zone", "High R:R", "GARP").
- **What does the system think?** Confluence score and setup pattern clearly displayed.
- **Cognitive Friction Point**: When user clicks `FIX` in the Screener, the card displays price `$346.20` and ROIC `30.5%`. When they arrive at the Terminal, the price is `$1,560.13` and ROIC is `28.5%`. A $1,214 price jump destroys user confidence immediately.
- **Recovery Issue**: Clicking the ticker link at the top of the screener card drops `fromGoal`, losing the "Back to Candidates" breadcrumb.

### Journey 2: Direct Search (`Search → Ticker → Terminal`)
- **What am I looking at?** Immediate display of symbol, price, and decision posture.
- **What does the system think?** Setup Score and verdict label prominent.
- **Cognitive Friction Point**: If a user searches for an uncataloged ticker (`SYM`), Advanced View correctly shows `"N/A"`, but switching to Guided View displays four green pills asserting `"Company Health: Healthy"` and `"Price is holding firmly above rising 20 EMA and 50 SMA"`. The user receives conflicting guidance depending on which lens they look through.
- **Degraded State Failure**: If an invalid ticker or network error occurs, the terminal renders a fake asset priced at `$100.00` without any error message or retry action.

### Journey 3: Portfolio Review (`Portfolio → Holding → Terminal`)
- **What am I looking at?** Clear position list, Cornish-Fisher VaR (95%), and equity balance.
- **What does the system think?** Real-time unrealized P&L and risk weightings visible.
- **Cognitive Friction Point**: When the user clicks their owned position in the portfolio table (`portfolio/page.tsx:L496`), the link is `/?symbol=${pos.symbol}` without `&ownership=OWNED`. The terminal opens with `ownership: "NOT_OWNED"`, prompting the user to `"ACQUIRE"` a stock they already own!

### Journey 4: Comparison (`Compare → Symbol A vs Symbol B → Terminal`)
- **What am I looking at?** Side-by-side fundamental and technical comparison.
- **Cognitive Friction Point**: When an uncataloged ticker is entered, the comparison tool silently generates fake metrics (`ROIC: 24.0%`, `Gross Margin: 55.0%`, `Fwd P/E: 25.0x`, `PEG: 1.15`, `FCF Yield: 3.8%`, `Market Cap: $(price * 0.45)B`). The comparison is based on fabricated numbers masquerading as institutional analysis.

### Journey 5: Strategy / Archetype (`Strategy → Candidate List → Stock Terminal`)
- **What am I looking at?** Minervini VCP, Greenblatt Magic Formula, and Peter Lynch GARP archetypes.
- **Cognitive Friction Point**: The strategy candidate lists hardcode stale prices from 2023–2024 (`NVDA: $128.50`, `PLTR: $31.20`). The stop loss and entry ranges are anchored to stale prices. Clicking through to `/stock/nvda` or `/?symbol=NVDA` reveals current prices over $220, rendering the strategy thesis numbers obsolete.

---

## 4. Deep-Dive: Posture Semantics & Epistemic Honesty

The user specifically challenged:
> `H-01 Zero Candles → overallEligibility: LIMITED → posture: WATCH`  
> *“Missing evidence should not accidentally imply an actionable monitoring recommendation. RESEARCH may be more epistemically honest than WATCH when the system cannot actually assess the security.”*

### Code Audit of `assessmentEngine.ts` (Lines 108–175)
Our audit confirmed the user's critique and uncovered an even deeper vulnerability:

```typescript
// 1. Overall Data Eligibility
if (agreement.evaluated === 0) {
  overallEligibility = "INELIGIBLE";
} else if (agreement.unavailable > 0) {
  overallEligibility = "LIMITED";
}

// 2. Evidence Assessment
if (overallEligibility === "INELIGIBLE") {
  assessment = "INSUFFICIENT_EVIDENCE";
} else if (agreement.favorable >= 3 && agreement.unfavorable === 0) {
  assessment = "FAVORABLE"; // <-- LEAK: Triggers even if 1 domain is UNAVAILABLE!
}

// 4. Contextual Posture
if (overallEligibility === "INELIGIBLE") {
  posture = "RESEARCH";
} else if (ownershipState !== "OWNED") {
  if (assessment === "FAVORABLE" && !isInvalidationBreached) {
    posture = "ACQUIRE"; // <-- LEAK: Missing trend can get posture ACQUIRE!
  } else if (assessment === "UNFAVORABLE") {
    posture = "AVOID";
  } else {
    posture = "WATCH"; // <-- LEAK: Defaults to WATCH with fabricated 50D SMA target!
    headlineExplanation = `Price remains below 50-day average ($${reclaimTarget.toFixed(2)}); awaiting base confirmation.`;
  }
}
```

### The Three Epistemic Violations:
1. **False "ACQUIRE" on Partial Data**:
   If an asset has `trend = UNAVAILABLE` (0 candles) but `health`, `smart_money`, and `macro` are favorable ($3 \ge 3$), `assessment` becomes `"FAVORABLE"`. The system assigns `posture: "ACQUIRE"` to an asset with literally zero price action history!
2. **False "WATCH" Recommendation**:
   When technical data is unavailable, defaulting to `"WATCH"` implies that ARX has identified a base and is monitoring for an entry trigger. But ARX cannot monitor a trigger without price history!
3. **Fabricated Headline Explanation**:
   When falling back to `"WATCH"`, line 173 prints:
   `headlineExplanation: "Price remains below 50-day average ($111.50); awaiting base confirmation."`
   The 50-day average `$111.50` is fabricated via `currentPrice * 1.115` (`assessmentEngine.ts:L131`), explicitly telling the user the price is below an indicator that does not exist!

### Canonical Posture Semantic Contract (Proposed for Phase 13.1)

| Posture | Semantic Intent | Strict Entry Invariant | Disallowed Conditions |
| :--- | :--- | :--- | :--- |
| **`RESEARCH`** | Evidence incomplete; manual investigation required. | Any core domain (`trend` or `health`) is `UNAVAILABLE`, OR `overallEligibility === "LIMITED" / "INELIGIBLE"`. | Must NEVER be used when all core evidence is available and actionable. |
| **`WATCH`** | Valid setup identified; actively monitoring for technical trigger. | `trend` and `health` both `AVAILABLE`, setup in consolidation or awaiting volume pivot. | Must NEVER be assigned if `trend` is `UNAVAILABLE` or candles $< 50$. |
| **`ACQUIRE`** | High-conviction multi-factor confluence in optimal buy zone. | `overallEligibility === "ELIGIBLE"`, all 4 domains evaluated, $\ge 3$ favorable, price in buy zone. | Must NEVER be assigned if ANY domain is `UNAVAILABLE` or `LIMITED`. |
| **`HOLD`** | Owned position; multi-factor thesis remains intact. | Position is `OWNED`, setup valid, price above stop loss. | Must NEVER be assigned to `NOT_OWNED` positions. |
| **`TRIM`** | Owned position; technical or fundamental deterioration. | Position is `OWNED`, $\ge 2$ unfavorable factors, risk mitigation warranted. | Must NEVER be assigned to `NOT_OWNED` positions. |
| **`EXIT_REVIEW`** | Owned position; price breached stop loss invalidation floor. | Position is `OWNED`, `currentPrice < stopLoss`. | Must take precedence over all factor scores. |
| **`AVOID`** | Unowned position; poor fundamentals or Stage 4 markdown. | Position is `NOT_OWNED`, $\ge 2$ unfavorable factors or Stage 4 downtrend. | Must NEVER be accompanied by a "Position Sizer" CTA. |

---

## 5. Detailed Forensic Findings Log

### `FINDING-13-01` (Severity: P1 · Journey 1: Discovery)
- **Component**: `frontend/app/screener/page.tsx:L126`
- **Observed Behavior**: Local `BASE_PRICES` dictionary has `FIX: { price: 346.20, roic: 30.5 }`.
- **Expected Behavior**: Screener should read from `MASTER_ASSET_CATALOG` (`FIX: price 1560.13, roic 28.5%`).
- **Cognitive Consequence**: User sees FIX at \$346 in the screener, clicks it, and lands on a terminal showing \$1,560.13. Discredits platform pricing accuracy.
- **Recommended Remediation**: Remove local `BASE_PRICES` dictionary; unify screener rows directly with `MASTER_ASSET_CATALOG` and `SpotPriceRegistry`.

### `FINDING-13-02` (Severity: P2 · Journey 1: Discovery)
- **Component**: `frontend/app/screener/page.tsx:L1030`
- **Observed Behavior**: Ticker symbol link in card header points to `/?symbol=${gem.symbol}` without `fromGoal` or `fromCount`.
- **Expected Behavior**: Should point to `/?symbol=${gem.symbol}&fromGoal=${selectedFilter}&fromCount=${displayGems.length}`.
- **Cognitive Consequence**: Clicking the ticker name instead of the bottom button drops the back-navigation breadcrumb in the Terminal.
- **Recommended Remediation**: Update card header `Link` to include `fromGoal` and `fromCount` parameters.

### `FINDING-13-03` (Severity: P0 · Gate 13.3: Decision Clarity & Posture Semantics)
- **Component**: `frontend/lib/assessmentEngine.ts:L119-175`
- **Observed Behavior**: Assets with missing trend or health can achieve `posture: "ACQUIRE"` if other factors are favorable, or default to `posture: "WATCH"` with headline citing a fabricated 50-day average.
- **Expected Behavior**: If `overallEligibility !== "ELIGIBLE"` or if core domains (`trend` / `health`) are `UNAVAILABLE`, posture must be `RESEARCH` with explanation: *"ARX cannot reliably assess this asset due to incomplete historical price action or missing financial filings."*
- **Cognitive Consequence**: Users are directed to buy or actively monitor assets where ARX has zero technical evidence.
- **Recommended Remediation**: Require `overallEligibility === "ELIGIBLE"` for `ACQUIRE` and `WATCH`. If core domains are missing, force `posture = "RESEARCH"`.

### `FINDING-13-04` (Severity: P0 · Gate 13.4: Evidence Transparency & Gate 13.6: Lens Parity)
- **Component**: `frontend/lib/insightGenerator.ts:L298-324`
- **Observed Behavior**: Guided View `whyPills` hardcodes `Company Health: Healthy` and `Price Trend: Healthy ("Price holding firmly above rising 20 EMA and 50 SMA")` even when data is `UNAVAILABLE`.
- **Expected Behavior**: Guided View pills must reflect actual domain availability: `Company Health: Data Unavailable`, `Price Trend: Insufficient History`.
- **Cognitive Consequence**: A beginner using Guided View is told the stock has healthy financials and a strong uptrend, while an advanced user sees "N/A".
- **Recommended Remediation**: Dynamically construct `whyPills` from `domains.trend` and `domains.health` status, using `sentiment: "neutral"` and `status: "Unavailable"` when missing.

### `FINDING-13-05` (Severity: P1 · Gate 13.4: Evidence Transparency & Gate 13.5: Action Clarity)
- **Component**: `frontend/lib/insightGenerator.ts:L353-364`
- **Observed Behavior**: Key levels generate synthetic `target1`, `target2`, and an attractive `2.91 : 1.0` Profit/Risk ratio even when `isTrendAvailable === false`.
- **Expected Behavior**: Targets and R:R ratio should be `undefined` / `"N/A"` when trend evidence is incomplete.
- **Cognitive Consequence**: Users see an attractive 2.91:1 R:R ratio and are tempted to trade an asset with no verifiable technical base.
- **Recommended Remediation**: If `!isTrendAvailable`, set `target1 = undefined`, `target2 = undefined`, `profitRiskRatio = undefined`.

### `FINDING-13-06` (Severity: P1 · Gate 13.8: Degraded States)
- **Component**: `frontend/app/page.tsx:L174-176, L221`
- **Observed Behavior**: Catch block logs error to console; `data` remains null; `AdaptiveTerminal` receives `currentPrice={data?.currentPrice || 100}`.
- **Expected Behavior**: Set `error` state; render an informative error banner with ticker name and "Retry Analysis" button.
- **Cognitive Consequence**: Entering an invalid ticker silently opens a terminal for a fictional \$100 asset.
- **Recommended Remediation**: Add `error` state in `page.tsx`; display clean error screen when ticker data cannot be resolved.

### `FINDING-13-07` (Severity: P2 · Gate 13.9: Accessibility)
- **Component**: `frontend/components/WhyInspectModal.tsx`, `PositionSizerModal.tsx`
- **Observed Behavior**: Modals lack focus traps (Tab key escapes dialog into background) and do not return focus to trigger button on close.
- **Expected Behavior**: Focus should be trapped within the dialog while open and restored to the activating button on close.
- **Cognitive Consequence**: Screen reader and keyboard users lose context when opening and closing modals.
- **Recommended Remediation**: Implement lightweight `focus-trap` or standard React keyboard focus management in both modals.

### `FINDING-13-08` (Severity: P0 · Gate 13.10: Trust & Disclosure)
- **Component**: `frontend/app/layout.tsx`, `frontend/components/Navbar.tsx`, `frontend/components/terminal/`
- **Observed Behavior**: Zero instances of financial disclaimer, non-fiduciary safe harbor notice, or risk warning exist in the application.
- **Expected Behavior**: Persistent global footer and modal disclaimers: *"ARX is an informational quantitative research platform, not a registered investment adviser or broker-dealer. Analysis is for educational purposes only and does not constitute personalized investment advice."*
- **Cognitive Consequence**: Legal and regulatory exposure; retail users could mistake automated quantitative scoring for fiduciary investment advice.
- **Recommended Remediation**: Add a global `FinancialDisclaimer` component to `layout.tsx` and persistent compact risk notes in Terminal views.

### `FINDING-13-09` (Severity: P2 · Gate 13.6: Cross-Screen Consistency)
- **Component**: `frontend/app/stock/[ticker]/page.tsx:L125` vs `frontend/lib/insightGenerator.ts:L357`
- **Observed Behavior**: Static stock page calculates Stop Loss as `spotPrice - 1.25 * atr14`, while Terminal uses `safePrice * 0.93`.
- **Expected Behavior**: Single canonical source of truth for stop loss calculations across static and dynamic views.
- **Cognitive Consequence**: Numerical discrepancy between the SEO landing page and the live terminal.
- **Recommended Remediation**: Unify stop loss calculation in `stock/[ticker]/page.tsx` to match `insightGenerator.ts`.

### `FINDING-13-10` (Severity: P1 · Journey 3: Portfolio Review)
- **Component**: `frontend/app/portfolio/page.tsx:L496`
- **Observed Behavior**: Link in portfolio position table is `<Link href={`/?symbol=${pos.symbol}`}>`.
- **Expected Behavior**: Link should be `<Link href={`/?symbol=${pos.symbol}&ownership=OWNED`}>`.
- **Cognitive Consequence**: Clicking an owned holding opens the terminal in `NOT_OWNED` mode, advising the user to `ACQUIRE` or `WATCH` rather than `HOLD` or `TRIM`.
- **Recommended Remediation**: Append `&ownership=OWNED` to all portfolio table terminal links.

### `FINDING-13-11` (Severity: P0 · Journey 4: Comparison)
- **Component**: `frontend/app/compare/page.tsx:L224-243`
- **Observed Behavior**: Uncataloged comparison tickers receive hardcoded fallback fundamentals (`ROIC: 24.0%`, `Gross Margin: 55.0%`, `Fwd P/E: 25.0x`, `PEG: 1.15`, `FCF Yield: 3.8%`, `Market Cap: $(price * 0.45)B`).
- **Expected Behavior**: Uncataloged tickers must display `"N/A"` for unverified fundamental metrics.
- **Cognitive Consequence**: Comparing an uncataloged stock creates fictitious parity with verified catalog assets.
- **Recommended Remediation**: Set uncataloged fields to `"N/A"` and raw scores to `0` or `null` in `compare/page.tsx`.

### `FINDING-13-12` (Severity: P1 · Journey 5: Strategy Archetypes)
- **Component**: `frontend/app/strategy/[type]/page.tsx:L50-95`
- **Observed Behavior**: `STRATEGY_DATABASE` hardcodes static prices from 2023–2024 (`NVDA: $128.50`, `PLTR: $31.20`).
- **Expected Behavior**: Strategy candidates should display live prices from `SpotPriceRegistry` or `MASTER_ASSET_CATALOG`.
- **Cognitive Consequence**: Severe price disorientation when moving from Strategy pages to the Terminal.
- **Recommended Remediation**: Dynamically hydrate candidate prices from `getMasterBaselinePrice` or `SpotPriceRegistry`.

### `FINDING-13-13` (Severity: P1 · Gate 13.5: Action Clarity)
- **Component**: `frontend/components/terminal/StandardTerminalView.tsx:L145-150`
- **Observed Behavior**: Primary action button unconditionally renders `⚖️ Institutional Position Sizer` even when posture is `AVOID`, `EXIT_REVIEW`, or `RESEARCH`.
- **Expected Behavior**: Primary CTA should be context-sensitive:
  - `ACQUIRE` $\rightarrow$ `⚖️ Institutional Position Sizer`
  - `AVOID` $\rightarrow$ `⚠️ View Alternative Candidates`
  - `EXIT_REVIEW` $\rightarrow$ `🚨 Review Exit Checklist`
  - `RESEARCH` $\rightarrow$ `📚 Research SEC Filings`
- **Cognitive Consequence**: User is invited to size a position on a stock ARX recommends avoiding or exiting.
- **Recommended Remediation**: Map primary CTA button label and action to `insight.terminalState.posture`.

### `FINDING-13-14` (Severity: P0 · Gate 13.5: Position Sizer Safety)
- **Component**: `frontend/components/PositionSizerModal.tsx:L53`
- **Observed Behavior**: When `safeEntry <= safeStop` (breached setup), `safeEntry - safeStop` is $\le 0$, clamped to `0.01`. `rawShares = maxDollarRisk / 0.01`, producing a 25,000+ share order recommendation.
- **Expected Behavior**: Modal must detect when `safeEntry <= safeStop`, disable position calculation, and display: *"Setup Invalidated: Price is at or below stop loss floor. Position sizing disabled."*
- **Cognitive Consequence**: Extreme financial hazard: An investor could enter an order with 100x unintended leverage.
- **Recommended Remediation**: Add validation guard in `PositionSizerModal.tsx`: if `safeEntry <= safeStop`, render invalidation banner and disable calculation.

---

## 6. Phase 13 Hard Gate & Next Steps

```text
Discovery Audit Result: 14 Findings Recorded
  - Priority 0 (Blockers): 5
  - Priority 1 (High):      6
  - Priority 2 (Medium):    3
  - Priority 3 (Low):       0

Production Code Changes: ZERO (Read-Only protocol respected)
Baseline Regression Suite: 191/191 tests PASS (Clean baseline maintained)
```

### Proposed Next Phase: Phase 13.1 — Product / UX Remediation Sprint
Execution order prioritized by cognitive and financial risk:
1. **Batch 1 (P0 Safety & Integrity)**:
   - Fix `PositionSizerModal.tsx` explosive leverage on breached setups (`FINDING-13-14`).
   - Fix `assessmentEngine.ts` posture leaks to enforce strict `RESEARCH` gating and eliminate fake 50D SMA triggers (`FINDING-13-03`).
   - Fix `insightGenerator.ts` Guided View `whyPills` hardcoded disinformation (`FINDING-13-04`).
   - Add persistent `FinancialDisclaimer` component and safe harbor notices (`FINDING-13-08`).
   - Clean up synthetic fundamentals in `compare/page.tsx` (`FINDING-13-11`).
2. **Batch 2 (P1 Journey & Context Integrity)**:
   - Pass `&ownership=OWNED` in portfolio table links (`FINDING-13-10`).
   - Make Terminal primary CTA context-sensitive to posture (`FINDING-13-13`).
   - Guard targets and R:R ratios when trend is incomplete (`FINDING-13-05`).
   - Unify Screener and Strategy candidate prices with `masterCatalog.ts` (`FINDING-13-01`, `FINDING-13-12`).
   - Add graceful error screen on ticker ingestion failure (`FINDING-13-06`).
3. **Batch 3 (P2 Usability & Accessibility)**:
   - Preserve Screener breadcrumb in card header ticker link (`FINDING-13-02`).
   - Add modal focus trap and restoration (`FINDING-13-07`).
   - Harmonize static stock page stop loss formula (`FINDING-13-09`).
