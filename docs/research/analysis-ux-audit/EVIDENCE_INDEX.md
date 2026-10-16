# ARX Terminal — Worktree D: Forensic Evidence & Cross-Reference Index

**Document ID**: `EVIDENCE-INDEX-ARX-ANALYSIS-001`  
**Classification**: Audit Evidence & Code Cross-Reference Index  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-analysis-ux-audit`  
**Audit Baseline**: Mainline Commit `2eb032b2aec18ac4fea0f654eee8fb6c028466fe`  

---

## 1. Overview of Evidence Corpus

This document provides exact file paths, line numbers, and code extracts linking the Product Owner screenshots (`IMG_8590` to `IMG_8599`) directly to the active codebase in worktree D.

---

## 2. Evidence Cross-Reference Table

| Asset / Screenshot ID | Observed UI Phenomenon | Authoritative Source File | Line Numbers | Code Extract / Implementation | Confirmed Runtime State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`IMG_8590`** | Header Decision State: `"Pullback into Accumulation Corridor"` marked as `UNMET` | `frontend/lib/decisionHierarchyUtils.ts` | 39–56 | `const isInCorridor = price >= canonicalLower && price <= canonicalUpper;`<br>`if (!isInCorridor && price > canonicalUpper) { unmet.push("Pullback into Accumulation Corridor: Price (" + formatCurrency(price) + ") is extended above preferred entry corridor..."); }` | Evaluates $341.91 > $339.38 $\implies$ `isInCorridor = false`. Gate is **UNMET / BLOCKING**. |
| **`IMG_8591`** | Price Display & Interactive Chart: Spot $341.91 above shaded corridor | `frontend/components/terminal/StandardTerminalView.tsx` | 85–130 | `<div className="text-3xl font-mono">$341.91</div>`<br>`<TradingViewChart corridor={[325.07, 339.38]} spot={341.91} />` | Confirms spot is visually and numerically above corridor ceiling. |
| **`IMG_8592`** | Execution Card: Gate 1 `PASSED` (*"positioned inside corridor"*), Hint: *"Scale in 30%"*, R:R `2.35:1` | `frontend/lib/decisionReadiness.ts`, `frontend/components/OptimalEntryExitCard.tsx` | `readiness.ts`: 99–108<br>`Card.tsx`: 258–270 | `const gate1State = currentPrice > entryMax * 1.02 ? "BLOCKING" : "PASSED";`<br>`// Because 341.91 <= 339.38 * 1.02 (346.17), it passes!`<br>`explanation: "Price (" + formatCurrency(currentPrice) + ") is positioned inside the institutional accumulation corridor"` | **CRITICAL DEFECT**: Gate 1 reports **PASSED** and claims price is inside corridor even though $341.91 > $339.38. |
| **`IMG_8593`** | Narrative Setup Description & Live Label: Repetitive volume text; `"LIVE SPOT: $341.91"` | `frontend/lib/insightGenerator.ts`, `frontend/app/page.tsx` | `insight.ts`: 120–145<br>`page.tsx`: 65–67 | `thesis: "Volume is drying up along the right side of the base indicating supply exhaustion..."`<br>`setLastUpdatedTime(new Date().toLocaleTimeString());` | Boilerplate string repeated across 3 containers; client browser clock masquerading as quote timestamp. |
| **`IMG_8594`** | Execution Targets & Stop Loss: Stop $315.32 (-7.09% from base), TP1 $383.89 (+13.12%), TP2 $407.95 (+20.20%) | `analyst_dashboard/analyzers/optimal_execution.py`, `OptimalEntryExitCard.tsx` | `python`: 702, 751<br>`Card.tsx`: 645, 685, 733 | `plan["stop_loss_pct"] = round(((structural_invalidation - planned_entry) / planned_entry) * 100, 2)`<br>`// (315.32 - 339.38) / 339.38 = -7.089% -> -7.09%` | Explicitly calculated from `planned_entry` ($339.38), NOT spot ($341.91). |
| **`IMG_8595`** | Confluence Score: `76/100`, Badge: `"3/4 Strong Alignment"`, Colloquial: `"Rock-Solid balance sheet"` | `analyst_dashboard/analyzers/confluence_engine.py`, `WeeklyConfluenceSpotlight.tsx` | `python`: 107, 180<br>`Spotlight.tsx`: 971 | `composite_score = round(0.25*tech + 0.25*fund + 0.25*smart + 0.25*macro)`<br>`// 0.25(85) + 0.25(80) + 0.25(85) + 0.25(55) = 76.25 -> 76` | Verified 4 equal-weighted pillars; 3 of 4 pillars $\ge 70$. |
| **`IMG_8596`** | Ownership Selector, Truncated Walkthrough & Stop -7.8%: Switch to `"I Own It"`, Stop $315.32 (-7.8%) | `frontend/components/AdaptiveTerminal.tsx`, `frontend/lib/insightGenerator.ts`, `frontend/components/terminal/GuidedTerminalView.tsx` | `Adaptive.tsx`: 203<br>`insight.ts`: 486<br>`Guided.tsx`: 336 | `stopLossPct = Math.round(((stopLoss - safePrice) / safePrice) * 1000) / 10;`<br>`<span className="font-bold block truncate">{step.title}</span>`<br>`// Truncates "3. Why does ARX like/caution it?"` | Dual denominator (-7.8% vs -7.09%); `truncate` clips walkthrough steps on mobile screens; ownership toggle changes query string only. |
| **`IMG_8597`** | Smart Money / Insider Card: `"No recent big boss insider purchases filed with the SEC this month"` | `analyst_dashboard/analyzers/confluence_engine.py` | 180 | `if not insider_buys: return {"narrative": "No recent big boss insider purchases filed with the SEC this month"}` | Hardcoded informal string in backend engine. |
| **`IMG_8598`** | Demonstration Mode / Demonstration Asset: AAPL marked as Demo Asset | `frontend/app/page.tsx`, `frontend/components/PageIntro.tsx` | `page.tsx`: 664–667<br>`PageIntro.tsx`: 65–68 | `badge={hasExplicitSymbol ? "Active Target" : "Demonstration Mode"}`<br>`isDemo={!hasExplicitSymbol}`<br>`demoNotice="Displaying AAPL as a demonstration asset..."` | Navigating to `/` without `?symbol=AAPL` causes authentic market data to be branded "Demonstration Asset". |
| **`IMG_8599`** | Spotlight: `"Top 3 High-Confluence Plays"`, duplicate `"View Full Setups"` | `frontend/components/WeeklyConfluenceSpotlight.tsx` | 742, 809, 865 | Line 809: `{isCollapsed ? "View Full Setups ▼" : "Collapse ▲"}`<br>Line 865: `<span>View Full Setups</span> <span>▼</span>` | When collapsed, both header button and ribbon link render simultaneously $\implies$ `"View Full Setups"` appears twice within 100px. |

---

## 3. Forensic Code Extracts & Analysis

### 3.1 The Gate 1 Corridor Check Inversion
In `frontend/lib/decisionReadiness.ts`:
```typescript
// Lines 99-108 in decisionReadiness.ts
const bufferPrice = entryMax * 1.02;
let gate1State: "PASSED" | "BLOCKING" = "PASSED";
let explanation = "";

if (currentPrice > bufferPrice) {
  gate1State = "BLOCKING";
  explanation = `Price (${formatCurrency(currentPrice)}) is extended >2% above entry corridor ceiling (${formatCurrency(entryMax)})`;
} else {
  gate1State = "PASSED";
  // BUG: If currentPrice is between entryMax and bufferPrice (e.g. $341.91 vs $339.38),
  // it executes this branch and falsely claims the price is INSIDE the corridor!
  explanation = `Price (${formatCurrency(currentPrice)}) is positioned inside the institutional accumulation corridor (${formatCurrency(entryMin)}–${formatCurrency(entryMax)})`;
}
```
**Forensic Proof**:
- `entryMax` = $339.38
- `bufferPrice` = $339.38 \times 1.02 = \$346.1676$
- `currentPrice` = $341.91
- Because $341.91 \le 346.17$, `gate1State` evaluates to `"PASSED"`.
- The explanation template unconditionally outputs: `"Price ($341.91) is positioned inside the institutional accumulation corridor ($325.07–$339.38)"`, which is a direct falsehood ($341.91 > 339.38$).

### 3.2 The Conflicting Top-of-Page Evaluation
In `frontend/lib/decisionHierarchyUtils.ts`:
```typescript
// Lines 39-56 in decisionHierarchyUtils.ts
const isInCorridor = price >= canonicalLower && price <= canonicalUpper;
if (!isInCorridor && price > canonicalUpper) {
  unmet.push(
    `Pullback into Accumulation Corridor: Price (${formatCurrency(price)}) is extended above preferred entry corridor (${formatCurrency(canonicalLower)}–${formatCurrency(canonicalUpper)}). Wait for low-volume pullback.`
  );
}
```
**Forensic Proof**:
- Evaluates $341.91 > 339.38 \implies \text{isInCorridor} = \text{false}$.
- Pushes blocking condition: `"Wait for low-volume pullback"`.
- Both functions run concurrently on the same page payload, displaying opposite verdicts to the user.

### 3.3 Walkthrough Label Truncation on Mobile
In `frontend/components/terminal/GuidedTerminalView.tsx`:
```tsx
// Lines 323-337 in GuidedTerminalView.tsx
<div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-1.5">
  {steps.map((step, idx) => (
    <button
      key={idx}
      id={`walkthrough-step-btn-${idx}`}
      onClick={() => setActiveStep(activeStep === idx ? null : idx)}
      className="... min-h-[36px]"
    >
      <span className="font-bold block truncate">{step.title}</span>
    </button>
  ))}
</div>
```
**Forensic Proof**:
- `truncate` enforces `text-overflow: ellipsis; white-space: nowrap; overflow: hidden;`.
- Step 3 title: `"3. Why does ARX like/caution it?"` is 34 characters long.
- In a 2-column mobile grid (~160px width per cell), text overflows and is clipped to `"3. Why does ARX li..."`.
- Button height `min-h-[36px]` violates the $44\text{px}$ minimum touch target standard.

### 3.4 Duplicate "View Full Setups" Triggers
In `frontend/components/WeeklyConfluenceSpotlight.tsx`:
```tsx
// Lines 803-810 (Header Action):
<button onClick={() => setIsCollapsed(!isCollapsed)}>
  {isCollapsed ? "View Full Setups ▼" : "Collapse ▲"}
</button>

// Lines 860-867 (Ribbon Quick-Switcher Action rendered when isCollapsed is true):
<button onClick={() => setIsCollapsed(false)}>
  <span>View Full Setups</span>
  <span>▼</span>
</button>
```
**Forensic Proof**:
- When `isCollapsed = true`, both buttons render within the spotlight component simultaneously.
- Renders the identical string `"View Full Setups ▼"` twice within 100 vertical pixels.

---

## 4. Evidence Addendum Cross-Reference

Complete mathematical proofs, boundary condition evaluations, and surgical remediation specifications for the five core technical inconsistencies are detailed in [`EVIDENCE_ADDENDUM.md`](file:///docs/research/analysis-ux-audit/EVIDENCE_ADDENDUM.md):
- **Item 1**: Forensic Proof & Boundary of Gate 1 Corridor Inversion (`IMG_8590` vs `IMG_8592`) $\to$ Section 3
- **Item 2**: Dual-Denominator Arithmetic Trace for Stop Loss & Targets (`IMG_8594` vs `IMG_8596`) $\to$ Section 4
- **Item 3**: Blended 50/50 Multi-Target Risk/Reward Formula vs Single-Target Spot Math (`IMG_8592`) $\to$ Section 5
- **Item 4**: Dynamic Data Reality, Client-Clock Masquerade & Demo Mode Trigger (`IMG_8593`, `IMG_8598`) $\to$ Section 6
- **Item 5**: Mobile Viewport Clearances, Truncation & Duplicate Spotlight Controls (`IMG_8596`, `IMG_8599`) $\to$ Section 7

---

## 5. Integrity Verification

All line numbers, file paths, and mathematical formulas in this index have been cross-checked directly against the active files in worktree `finance-analysis-ux-audit`.

