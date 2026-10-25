# ARX Terminal — Worktree D: Analysis UX, Decision Consistency & Calculation Audit Report

**Document ID**: `AUDIT-ARX-ANALYSIS-UX-CALC-001`  
**Classification**: Formal Research & Governance Audit Report  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-analysis-ux-audit`  
**Branch**: `research/analysis-ux-decision-audit`  
**Baseline Mainline Commit**: `2eb032b2aec18ac4fea0f654eee8fb6c028466fe` (`origin/main`)  
**Baseline Mainline Tree**: `483bee72e2f8df63adecf15891b6e0091d52a738`  
**Audit Date**: 2026-10-16T14:15:00+02:00  
**Operating Mode**: `ISOLATE → RECONSTRUCT → REPRODUCE → TRACE AUTHORITIES → DOCUMENT FINDINGS → REGISTER BACKLOG → LOCAL COMMIT → STOP`  
**Execution Authority**: Product Owner Mandate — Worktree D UX & Calculation Audit  

---

## 1. Executive Summary

This audit investigates the Analysis page (`frontend/app/page.tsx`, `AdaptiveTerminal.tsx`, `StandardTerminalView.tsx`, `GuidedTerminalView.tsx`, `OptimalEntryExitCard.tsx`, `WeeklyConfluenceSpotlight.tsx`, and associated utility modules) of the ARX Terminal following user-submitted observations and forensic review of Product Owner screenshots (`IMG_8590` through `IMG_8599`).

The investigation was conducted strictly under **read-only diagnostic isolation** in dedicated worktree `finance-analysis-ux-audit` without mutating application logic, scoring algorithms, database tables, or remote git branches.

### Core Audit Outcomes:
1. **Critical Decision Inversion Confirmed (Finding 1 / DEFECT)**: At the top of the page (`StandardTerminalView.tsx` via `decisionHierarchyUtils.ts`), a stock trading at $341.91 against an accumulation corridor of $325.07–$339.38 is correctly flagged as **UNMET** (*"Pullback into Accumulation Corridor: Price ($341.91) is extended above preferred entry corridor... Wait for low-volume pullback"*). Concurrently, at the bottom of the page (`OptimalEntryExitCard.tsx` via `decisionReadiness.ts`), the identical price is evaluated by Gate 1 as **PASSED** (*"Price ($341.91) is positioned inside the institutional accumulation corridor"*), accompanied by a tactical hint advising the user to *"Scale in with 30% initial size"*.
2. **Dual-Denominator Stop-Loss Percentages Confirmed (Finding 2 / DEFECT)**: The structural invalidation stop of $315.32 is simultaneously presented as **-7.09%** (*"from base"*) in the execution card and **-7.8%** in the narrative insight generator. Forensic tracing proves these derive from two different denominators ($339.38 planned entry vs $342.00 reference close) without explicit labeling.
3. **Blended Risk/Reward Ratio Unpacked (Finding 3 / ARITHMETIC OPAQUENESS)**: The displayed 2.35:1 R:R ratio is mathematically verified as deriving from `analyst_dashboard/analyzers/optimal_execution.py:751-756`. It represents a **50/50 blended exit model** across Take Profit 1 ($383.89) and Take Profit 2 ($407.95) evaluated against Planned Entry ($339.38). When evaluated against the current spot price ($341.91) to Target 1, the true immediate R:R is only **1.58:1**. The UI provides zero indication of this blended model or entry assumption.
4. **Confluence Score & Weighting Decoded (Finding 4 / NOMENCLATURE DRIFT)**: "Setup Score", "Confluence Score", and "Multi-Model Confluence" are verified to be the identical underlying metric (`confluence.confluenceScore` from `confluence_engine.py`). The score of 76/100 is an equal-weighted average of 4 pillars (25% each: Technical 85, Fundamental 80, Smart Money 85, Macro 55). The "3/4 Strong Alignment" badge reflects the three pillars $\ge 70$. The 70 threshold is an institutional execution eligibility gate, not an empirical probability of trade success.
5. **Superficial Ownership Context (Finding 5 / SCOPE LIMITATION)**: Switching ownership to "I Own It" merely toggles an ephemeral query parameter without capturing cost basis, position size, or entry dates, leaving risk calculations unanchored to actual portfolio holdings.
6. **Unprofessional Copy & High Redundancy (Finding 6 / COPY HYGIENE)**: The page suffers from repetitive explanations of accumulation zones and informal, colloquial phrasing (e.g., *"No recent big boss insider purchases"*, *"Rock-Solid balance sheet"*, *"Canonical Verdict"*) that erode institutional credibility.
7. **Mobile Usability, Duplicated Buttons & Truncation (Finding 7 / UX DEFECT)**: Truncated walkthrough labels (`GuidedTerminalView.tsx:336`), duplicate "View Full Setups" buttons rendered simultaneously in collapsed spotlight ribbons (`WeeklyConfluenceSpotlight.tsx:809, 865`), and fixed navigation collisions on screens $<768\text{px}$.

---

## 2. Direct Answers to the 7 Core Product Owner Questions

### Q1: Which repeated text can be consolidated?
- **Redundant Setup Narratives**: The description of accumulation corridors, low-volume pullbacks, and institutional bases appears across four separate UI sections (`PageIntro`, `StandardTerminalView`, `OptimalEntryExitCard`, and `WeeklyConfluenceSpotlight`).
- **Volume Dry-Up Boilerplate**: Explanations stating that *"Volume is drying up along the right side of the base indicating supply exhaustion"* are rendered nearly verbatim in both the narrative thesis and the technical pillar card.
- **Duplicate Action Triggers**: In `WeeklyConfluenceSpotlight.tsx`, when collapsed, both the header button (line 809) and the compact ribbon link (line 865) render *"View Full Setups ▼"* within 100px of each other.
- **Remediation**: Consolidate setup narratives into a single authoritative **Context & Rationale** container, eliminate duplicate header/ribbon buttons, cutting 60% of redundant textual copy.

### Q2: Do all sections communicate the same decision state?
- **NO. Direct State Contradiction**:
  - `StandardTerminalView`: Reports **BLOCKING / UNMET** condition (*"Price ($341.91) is extended above preferred entry corridor... Wait for low-volume pullback"*).
  - `OptimalEntryExitCard`: Reports **PASSED** on Gate 1 (*"Price ($341.91) is positioned inside the institutional accumulation corridor"*) and provides an immediate buy-entry hint (*"Scale in with 30% initial size"*).
- **Cause**: Dual implementation of corridor checks. `decisionHierarchyUtils.ts` uses strict bounds `[entryMin, entryMax]`, whereas `decisionReadiness.ts` applies a `1.02x` buffer (`currentPrice <= entryMax * 1.02`) and erroneously outputs text claiming price is "inside" the corridor when it is in fact extended by +0.74%.

### Q3: Are score, threshold, price, stop, and risk/reward calculations consistent?
- **Score**: Consistent mathematically (76/100 across components), but inconsistently labeled as "Setup Score", "Confluence Score", and "Multi-Model Confluence".
- **Threshold**: The 70/100 threshold is consistently applied in code (`decisionHierarchyUtils.ts:98`, `fail-close-governance.ts:261`), but lacks UI explanation and risks being misinterpreted as win-rate probability.
- **Price & Stop**: Structural invalidation price is consistent at $315.32, but stop-loss percentages conflict (-7.09% vs -7.80%) due to unstated denominator shifts.
- **Risk/Reward**: Displayed as 2.35:1 based on an uncommunicated 50/50 TP1/TP2 exit blending from planned corridor entry ($339.38), conflicting with spot-to-TP1 arithmetic (1.58:1).

### Q4: How are dynamic data, historical context, and demonstration data presented?
- **Current Dynamic Data Reality**:
  - **Zero Polling or Streaming**: The Analysis page (`frontend/app/page.tsx:210`) issues a single `fetchAssetAnalytics()` request on mount. There are NO WebSockets, NO Server-Sent Events (SSE), and NO periodic `setInterval()` background refreshes.
  - **Client Clock Masquerading as Feed Timestamp**: In `page.tsx:66`, `lastUpdatedTime` is populated with `new Date().toLocaleTimeString()` whenever local React state updates. This reflects client browser time, NOT the exchange quote timestamp.
  - **Demonstration Mode Trigger**: In `page.tsx:664-667`, navigating to the Analysis root without a `?symbol=` parameter defaults to `selectedSymbol = "AAPL"` and sets `isDemo = !hasExplicitSymbol`. This triggers the "Demonstration Mode / Demonstration Asset" banner on AAPL, even though the backend data returned is authentic market data.
- **Remediation**: Standardize a dedicated **Data Freshness & Provenance Banner** specifying data source, true upstream quotation timestamp, market session phase, and simulated vs live market status.

### Q5: Does ownership change the guidance appropriately?
- **NO**: Toggling between "Unspecified", "Considering", and "I Own It" (`AdaptiveTerminal.tsx:203`) only alters cosmetic label text. It does not collect user cost basis, shares held, or entry timestamps.
- **Flaw**: Because actual basis is unknown, the terminal cannot compute personalized drawdown, unrealized P&L, or trailing stops. Market-wide structural levels ($315.32) are appropriate for both owners and non-owners, but position-management guidance remains incomplete and misleading.

### Q6: Should purchase-readiness and confluence be filters, sections, or pages?
- **Recommendation**:
  - **Radar**: Add a canonical **Actionability Filter** (`READY_TO_BUY` vs `WAITING_PULLBACK` vs `INVALIDATED`) alongside a numerical Confluence Score slider ($\ge 70$).
  - **Analysis Page**: Keep as a deep-dive single-asset page with a structured 6-tier section hierarchy.
  - **Avoid Separate Pages**: Do not fragment the experience into disconnected "Readiness" or "Confluence" pages; preserve deep linking and saved views.

### Q7: What mobile accessibility issues require remediation?
- **Truncated Walkthrough Labels**: In `GuidedTerminalView.tsx:336`, step titles inside the 6-step walkthrough use `truncate`, which clips strings like *"3. Why does ARX like/caution it?"* into an illegible ellipsis on mobile viewports.
- **Hidden Precondition Descriptions**: In `StandardTerminalView.tsx:149`, precondition descriptions are wrapped in `hidden sm:block`, hiding critical execution rationale entirely on screens $<640\text{px}$.
- **Duplicate "View Full Setups" Buttons**: In `WeeklyConfluenceSpotlight.tsx:809, 865`, two buttons rendering "View Full Setups" appear within the same collapsed ribbon view.
- **Bottom Navigation Collision**: The sticky bottom action bar overlaps primary interactive buttons and disclosures on viewports $<768\text{px}$.
- **Touch Target Deficits**: Walkthrough step buttons measure $36\text{px}$ in height (`min-h-[36px]`), failing the $44\times 44\text{px}$ touch target requirement.

---

## 3. Evidence Classification Taxonomy

In compliance with prompt Section 5, all findings are categorized into five rigorous evidentiary classes:

### Class 1: Confirmed Current Defects (Code-Level Bugs)
- **`DEF-01`**: Inverted Corridor Gate 1 logic in `frontend/lib/decisionReadiness.ts:99-108`. Evaluates to `PASSED` when $Spot > Max$, and emits a false string claiming spot is inside corridor.
- **`DEF-02`**: Unsynchronized stop-loss percentage denominators (-7.09% vs -7.80%) in `OptimalEntryExitCard.tsx:733` vs `insightGenerator.ts:486`.
- **`DEF-03`**: Truncated walkthrough titles via unconstrained `truncate` in `GuidedTerminalView.tsx:336`.
- **`DEF-04`**: Duplicate "View Full Setups" buttons rendered concurrently in `WeeklyConfluenceSpotlight.tsx:809, 865`.

### Class 2: Confirmed Screenshot Inconsistencies (PO Image Verifications)
- **`IMG_8590` vs `IMG_8592`**: Top header reports Pullback UNMET; bottom card reports Gate 1 PASSED and advises scaling in 30%.
- **`IMG_8594` vs `IMG_8596`**: Stop loss dollar level ($315.32) paired with conflicting percentages (-7.09% vs -7.8%).
- **`IMG_8592`**: 2.35:1 R:R displayed alongside spot price $341.91 (which yields 1.58:1 to TP1).
- **`IMG_8595`, `IMG_8597`**: Informal copy *"big boss insider purchases"* and *"Rock-Solid balance sheet"*.
- **`IMG_8598`**: "Demonstration Mode / Demonstration Asset" rendered because URL lacked explicit `?symbol=AAPL`.
- **`IMG_8599`**: "View Full Setups" rendered twice in spotlight ribbon.

### Class 3: Plausible Implementation Risks (Latent Architectural Hazards)
- **Clock Masquerading as Data Freshness**: `lastUpdatedTime` generated by client `toLocaleTimeString()` risks misleading traders into believing stale cached quotes are real-time live tape feeds.
- **Missing Factor Degradation in Confluence Engine**: If an external provider (e.g. SEC EDGAR API) times out, the confluence engine could default missing pillars to 0 or uncalibrated states without explicit user notification.
- **Single-Request Lifecycle**: Lack of WebSocket or polling means long-lived tabs display stale price data while claiming an active market session.

### Class 4: Product Design Proposals (Strategic Backlog Items)
- **`PROP-01`**: 6-Tier Information Architecture consolidation (Header $\to$ Decision $\to$ Visual Levels $\to$ Confluence $\to$ Sizing $\to$ Deep Inspector).
- **`PROP-02`**: Radar Actionability filter (`READY_TO_BUY`, `WAITING_PULLBACK`, `INVALIDATED`) and Confluence slider ($\ge 70$).
- **`PROP-03`**: Ownership flow enhancement: collecting user cost basis, shares held, and entry date for personalized risk metrics.
- **`PROP-04`**: Deconstructed 4-pillar scorecard with 25% weight indicators and transparent sub-feature scores.

### Class 5: Not Reproduced / Insufficient Evidence
- **User-Supplied Text Example**: *"Elevate Multi-Model Confluence — UNMET: Current score 56/100 is below the 70/100 threshold required for high-conviction execution."* (Confirmed arithmetic $70 - 56 = 14$, but not present in the supplied AAPL screenshots).
- **Physical Device Touch Latency**: Touch target tap collisions were verified via CSS layout inspection (`min-h-[36px]` vs `44px`), not physical capacitive touch hardware testing.

---

## 4. Inconsistency Matrix (Detailed Forensic Findings)

```
Trigger → Observed Output → Expected Contract Behavior → Source Location → User Impact → Smallest Remediation Scope
```

### Finding 1: Corridor Gate 1 Contradiction & Split Decision Verdicts
- **Trigger**: Spot price ($341.91) is higher than accumulation corridor upper bound ($339.38) by +0.74%, within a 2.0% extension buffer.
- **Observed Output**:
  - Header verdict: `"Pullback into Accumulation Corridor: Price ($341.91) is extended above preferred entry corridor ($325.07–$339.38). Wait for low-volume pullback."` [State: UNMET / BLOCKING]
  - Execution Card Gate 1: `"Gate 1: PASSED — Price ($341.91) is positioned inside the institutional accumulation corridor ($325.07–$339.38)"` [State: PASSED]
  - Execution Card Hint: `"⚠️ Near Zone Ceiling: Spot ($341.91) is near the upper bound of the buy zone. Scale in with 30% initial size..."`
- **Expected Contract Behavior**: All UI components must consume a single source of truth for corridor status. If $Price > EntryMax$, the status must be `EXTENDED_ABOVE` (or `WAITING_PULLBACK`). The system must NEVER claim a price is "inside" a corridor when it exceeds the ceiling.
- **Source Location**:
  - `frontend/lib/decisionHierarchyUtils.ts:39-56` (`isInCorridor = price >= canonicalLower && price <= canonicalUpper`)
  - `frontend/lib/decisionReadiness.ts:99-108` (`gate1State = currentPrice > entryMax * 1.02 ? "BLOCKING" : "PASSED"`)
  - `frontend/components/OptimalEntryExitCard.tsx:258-270`
- **User Impact**: Total confusion. User receives conflicting guidance on whether to hold off or enter a 30% position immediately.
- **Smallest Remediation Scope**: Remove the `1.02x` buffer logic that falsely reports "inside corridor" in `decisionReadiness.ts`; export and enforce a unified helper `evaluateCorridorMembership(spot, min, max)` that returns `{ state: 'INSIDE' | 'EXTENDED' | 'BELOW', diffPct, message }` across both components.

### Finding 2: Dual Stop-Loss Percentage Denominators (-7.09% vs -7.80%)
- **Trigger**: Structural floor price is $315.32. Planned entry ceiling is $339.38. Prior session close is $342.00. Current spot is $341.91.
- **Observed Output**:
  - Optimal Execution Card: `"Stop Loss: $315.32 (-7.09% from base)"`
  - Insight Summary Card: `"Stop Loss $315.32 (-7.8%)"`
- **Expected Contract Behavior**: Stop loss percentages must explicitly indicate the reference price used in the denominator.
- **Source Location**:
  - `analyst_dashboard/analyzers/optimal_execution.py:702` & `OptimalEntryExitCard.tsx:733`: Formula is `((315.32 - 339.38) / 339.38) * 100 = -7.089%`. Denominator = Planned Entry ($339.38).
  - `frontend/lib/insightGenerator.ts:486, 539, 585`: Formula is `((315.32 - 342.00) / 342.00) * 100 = -7.801%`. Denominator = Prior Session Close ($342.00).
- **User Impact**: User suspects a calculation bug or data corruption because two different percentages are shown for the same dollar stop price on the same screen.
- **Smallest Remediation Scope**: Standardize stop-loss percentage display to state: `"-$26.59 (-7.78% from spot $341.91)"` and `"-$24.06 (-7.09% from planned base $339.38)"`.

### Finding 3: Opaque Multi-Target Blended Risk/Reward Model
- **Trigger**: Execution card displays `Risk/Reward Ratio: 2.35:1`. Spot is $341.91, TP1 is $383.89, TP2 is $407.95, Stop is $315.32.
- **Observed Output**: User manually calculating $(383.89 - 341.91) / (341.91 - 315.32) = 41.98 / 26.59 = 1.58:1$ finds a glaring discrepancy with the displayed 2.35:1.
- **Expected Contract Behavior**: The UI must disclose the mathematical basis of the R:R metric (blended 50% TP1 / 50% TP2 exit from planned base $339.38).
- **Source Location**: `analyst_dashboard/analyzers/optimal_execution.py:751-756`
  ```python
  tp1_gain = 383.89 - 339.38 = 44.51
  tp2_gain = 407.95 - 339.38 = 68.57
  blended_reward = 0.50 * 44.51 + 0.50 * 68.57 = 56.54
  execution_risk = 339.38 - 315.32 = 24.06
  blended_rr = 56.54 / 24.06 = 2.3499... -> 2.35
  ```
- **User Impact**: Erosion of analytical confidence; institutional traders reject the system if risk calculations appear fabricated.
- **Smallest Remediation Scope**: Add an inline breakdown disclosure popover or label: `"2.35:1 (Blended 50/50 TP1/TP2 from $339.38 base; 1.58:1 Spot to TP1)"`.

### Finding 4: Confluence Score Semantic Drift & Missing Weighting Transparency
- **Trigger**: Engine outputs `confluenceScore = 76` with sub-scores: Technical = 85, Fundamental = 80, Smart Money = 85, Macro = 55.
- **Observed Output**: Displayed variously as `"Setup Score: 76"`, `"Confluence Score: 76/100"`, and `"Elevate Multi-Model Confluence: 76"`, paired with badge `"3/4 Strong Alignment"`.
- **Expected Contract Behavior**: Unified nomenclature ("Confluence Score") and explicit rendering of the 4 equal-weighted pillars ($25\% \times 85 + 25\% \times 80 + 25\% \times 85 + 25\% \times 55 = 76.25 \to 76$).
- **Source Location**: `confluence_engine.py:100-220`, `decisionHierarchyUtils.ts:98`, `WeeklyConfluenceSpotlight.tsx:971`.
- **User Impact**: Users misinterpret 3/4 as an incomplete score or 75%, and view the 70 threshold as an empirical statistical probability of winning.
- **Smallest Remediation Scope**: Standardize naming to "Confluence Score"; display the four 25% component bars; add tooltip clarifying that 70 is an institutional quality gate, not a win probability.

### Finding 5: Superficial Ownership Flow & Lack of Position-Level Context
- **Trigger**: User selects `"I Own It"` in `AdaptiveTerminal.tsx:203`.
- **Observed Output**: UI updates query param `?ownership=own` and switches wording to "holding", but still displays entry recommendations and theoretical setup percentages.
- **Expected Contract Behavior**: If "I Own It" is selected, the UI should prompt for cost basis and shares held to provide personalized risk/return metrics, or explicitly clarify that all levels shown are market-derived reference structures.
- **Source Location**: `frontend/components/AdaptiveTerminal.tsx:203-240`.
- **User Impact**: Position holders cannot assess their specific portfolio exposure or trailing-stop risk.
- **Smallest Remediation Scope**: Add a modal/form for cost basis input; adjust narrative when basis is entered to display actual P&L and basis-relative risk.

### Finding 6: Redundant Explanations & Unprofessional Institutional Wording
- **Trigger**: Reading thesis paragraphs and pillar explanations.
- **Observed Output**: Colloquial expressions:
  - `"Rock-Solid balance sheet"` (`confluence_engine.py:107`)
  - `"No recent big boss insider purchases filed with the SEC this month"` (`confluence_engine.py:180`)
  - `"Canonical Verdict"` (`decisionHierarchyUtils.ts`)
- **Expected Contract Behavior**: Professional institutional financial terminology.
- **Source Location**: `analyst_dashboard/analyzers/confluence_engine.py:107, 180`, `insightGenerator.ts`.
- **User Impact**: Diminished institutional credibility and cluttered interface.
- **Smallest Remediation Scope**: Replace with institutional phrasing: *"Pristine balance sheet (low leverage)"*, *"No Form 4 insider acquisitions recorded in the past 30 days"*, *"System Decision State"*.

### Finding 7: Mobile Viewport Collision, Truncation & Duplicate Spotlight Controls
- **Trigger**: Viewing terminal on mobile screen ($<768\text{px}$).
- **Observed Output**:
  - Fixed bottom navigation bar covers execution buttons (`AdaptiveTerminal.tsx:350-420`).
  - Step titles truncated via `truncate` (`GuidedTerminalView.tsx:336`).
  - Precondition descriptions completely hidden on screens $<640\text{px}$ (`StandardTerminalView.tsx:149`).
  - Duplicate "View Full Setups" buttons rendered concurrently (`WeeklyConfluenceSpotlight.tsx:809, 865`).
  - Interactive touch targets measure $36\text{px}$ in height (`min-h-[36px]`).
- **Expected Contract Behavior**: Clean layout clearance with bottom padding (`pb-24`); responsive text wrapping without truncation; $\ge 44\text{px}$ touch targets; zero duplicate buttons.
- **Source Location**: `AdaptiveTerminal.tsx:350-420`, `GuidedTerminalView.tsx:336`, `WeeklyConfluenceSpotlight.tsx:809, 865`.
- **User Impact**: Inoperable buttons on mobile devices; truncated critical text; confusing repetitive controls.
- **Smallest Remediation Scope**: Add responsive bottom clearance padding; remove unneeded `truncate` on step titles; consolidate spotlight ribbon button to single trigger; increase min-height to 44px.

---

## 5. Radar Filter and Confluence Architecture Proposal (Section 10)

To resolve the relationship between Radar filtering and Analysis page deep-dives without creating disconnected routes:

### 5.1 Architecture & Hypotheses Evaluated
- **Radar Remains Canonical Sieve**: The existing Radar screen (`/radar`) is enhanced with an **Actionability Filter** dimension, rather than creating a new "Readiness" page.
- **Confluence Remains Orthogonal Sieve**: Score $\ge 70$ is a filter dimension on Radar and an explanatory breakdown on Analysis, NOT a separate screen.
- **Deep Linking via URL Serialization**: All filter states (`actionability=READY_TO_BUY&confluence_min=70`) serialize directly to query parameters for 1-click sharing and saved views.

### 5.2 Authoritative Predicates for "Meets Execution Criteria" Filter
To appear under `Meets Execution Criteria`, a setup must satisfy 4 independent gates:
1. **Market/Setup Structure**: Spot Price must sit strictly inside the Accumulation Corridor:
   $$\text{SpotPrice} \ge \text{EntryMin} \quad \text{AND} \quad \text{SpotPrice} \le \text{EntryMax}$$
2. **Data Availability & Freshness**: Upstream quote must be fresh ($\le 15\text{m}$ if market open, or confirmed previous session close). No synthetic fallback data.
3. **Multi-Model Confluence Quality Gate**: Confluence composite score must satisfy institutional threshold:
   $$\text{confluenceScore} \ge 70$$
4. **Capital Protection Invalidation Floor**: Structural stop must be defined, non-zero, and positive:
   $$\text{structuralInvalidation} > 0 \quad \text{AND} \quad \text{structuralInvalidation} < \text{SpotPrice}$$

### 5.3 Filter Lifecycle States Defined
- **EMPTY**: 0 assets satisfy active filter criteria $\implies$ Render institutional empty state: *"No assets currently satisfy all 4 execution gates. Monitoring 500 universe candidates for pullback entry."*
- **UNAVAILABLE**: Upstream quote or scanner store unreachable $\implies$ Render non-blocking banner: *"Market data feed connecting. Screener displaying verified historical cache."*
- **STALE**: Data timestamp $>4$ market sessions old $\implies$ Amber status badge: *"Tactical setups stale (>4 days). Awaiting scheduled scanner run."*
- **PARTIAL COVERAGE**: Asset missing secondary feeds (e.g. SEC insider filing or options volume) $\implies$ Confluence score calculated with asterisk: `76* (Partial Coverage: 3 of 4 pillars active)`.

---

## 6. Scope, Workstream Coordination & Next Steps

This report and its companion deliverable, [`EVIDENCE_ADDENDUM.md`](file:///docs/research/analysis-ux-audit/EVIDENCE_ADDENDUM.md), constitute the complete forensic research and governance baseline for Workstream D.

### 6.1 Parallel Workstream Status Summary
The authoritative status of all parallel active workstreams is recorded below:

| Workstream | Worktree Location | Active Branch / Reference | Authoritative Status |
| :--- | :--- | :--- | :--- |
| **Workstream A** | `finance-vcp-c004-gate3c-integration` | `feat/vcp-candidate-004-gate3c-wiring` (PR #5 merged to `origin/main` at `2eb032b`) | **`production verified / evidence closed`** |
| **Workstream B** | `finance-data-gov-discovery` | `research/data-gov-discovery` (commit `0d7cd81`) | **`discovery complete / bounded evidence closed`** |
| **Workstream C** | `finance-vcp-universe-inputs` / `finance-universe-validation-core` | `fix/universe-input-validation-core` (base `787c197`) | **`validation-core implementation authorized; completion is not yet reported`** |
| **Workstream D** | `finance-analysis-ux-audit` | `research/analysis-ux-decision-audit` (commit `32fc40a`) | **`audit complete / evidence addendum delivered; engineering handoff prepared`** |

### 6.2 Governance Guarantees & Staged Deliverables
- **No production code has been modified.**
- Forensic proof, boundary partitions, and arithmetic traces for the five core audit items are formally detailed in [`EVIDENCE_ADDENDUM.md`](file:///docs/research/analysis-ux-audit/EVIDENCE_ADDENDUM.md).
- Backlog items `UX-ANALYSIS-001` through `006`, `UX-RADAR-001`, and `UX-MOBILE-001` are registered in `docs/research/analysis-ux-audit/BACKLOG_HANDOFF.md` with status `PREPARED`.
- Authoritative backlog updates in `docs/planning/ENGINEERING_BACKLOG.md` remain `PENDING_CONTROLLED_INTEGRATION`.

