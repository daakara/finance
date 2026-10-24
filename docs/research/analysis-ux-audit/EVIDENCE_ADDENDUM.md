# ARX Terminal — Worktree D: Analysis UX & Calculation Evidence Addendum

**Document ID**: `ADDENDUM-EVIDENCE-ARX-ANALYSIS-001`  
**Classification**: Formal Forensic Evidence Addendum & Mathematical Provenance  
**Worktree**: `C:\Users\akara\Documents\Projects\finance-analysis-ux-audit`  
**Branch**: `research/analysis-ux-decision-audit`  
**Baseline Mainline Commit**: `2eb032b2aec18ac4fea0f654eee8fb6c028466fe` (`origin/main`, PR #5)  
**Baseline Mainline Tree**: `483bee72e2f8df63adecf15891b6e0091d52a738`  
**Previous Audit Commit**: `32fc40ad8944e2fe64450238bd32eef5a9760a45`  
**Audit & Addendum Date**: 2026-10-16T14:40:00+02:00  
**Operating Mode**: `EVALUATE EVIDENCE → PROVE ARITHMETIC → MAP CODE LOCATIONS → SPECIFY SURGICAL REMEDIATION → DOCUMENT ADDENDUM → LOCAL COMMIT → STOP`  
**Mandate Authority**: Product Owner Worktree D Evidence-Addendum Authorization  

---

## 1. Executive Summary & Purpose

This Evidence Addendum supplements and completes the forensic analysis delivered in `AUDIT_REPORT.md` (commit `32fc40a`). It provides definitive, line-by-line mathematical proofs, boundary condition evaluations, exact code citations, and surgical remediation specifications for the five core technical inconsistencies identified in Product Owner screenshots `IMG_8590` through `IMG_8599`.

All investigations were performed strictly under read-only diagnostic isolation within worktree `finance-analysis-ux-audit`. No application source code, API handlers, pricing algorithms, or database records have been mutated.

---

## 2. Parallel Workstream Governance & Coordination Status

To maintain synchronized, immutable accounting across all active finance engineering workstreams, the authoritative status of each parallel branch is formally recorded below:

| Workstream | Worktree Path | Active Branch / Reference | Authoritative Status | Scope & Boundary Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Workstream A** | `C:\Users\akara\Documents\Projects\finance-vcp-c004-gate3c-integration` | `feat/vcp-candidate-004-gate3c-wiring` (PR #5 merged to `origin/main` at `2eb032b`) | **`production verified / evidence closed`** | Candidate 004 universe wiring and inactive-read safety contracts verified remotely and merged into mainline. |
| **Workstream B** | `C:\Users\akara\Documents\Projects\finance-data-gov-discovery` | `research/data-gov-discovery` (commit `0d7cd81`) | **`discovery complete / bounded evidence closed`** | Data.gov and primary-source agency qualification completed; recommendations delivered; no production connectors authorized. |
| **Workstream C** | `C:\Users\akara\Documents\Projects\finance-vcp-universe-inputs` / `finance-universe-validation-core` | `fix/universe-input-validation-core` (base `787c197`) | **`validation-core implementation authorized; completion is not yet reported`** | Minimal policy-independent validation core authorized; offline implementation in progress; production universe publication blocked. |
| **Workstream D** | `C:\Users\akara\Documents\Projects\finance-analysis-ux-audit` | `research/analysis-ux-decision-audit` (commit `32fc40a`) | **`audit complete / evidence addendum delivered; engineering handoff prepared`** | Forensic UX, calculation trace, content inventory, and evidence addendum completed; staging backlog registered. |

---

## 3. Item 1: Forensic Proof & Boundary of Gate 1 Corridor Inversion (`IMG_8590` vs `IMG_8592`)

### 3.1 Observed Contradiction
In the Product Owner screenshot series for AAPL (Spot Price: **$341.91**, Accumulation Corridor: **$325.07 – $339.38**):
- **Top of Page (`IMG_8590`, `StandardTerminalView.tsx` via `decisionHierarchyUtils.ts`)**:
  - Precondition Title: `"Pullback into Accumulation Corridor"`
  - Status Badge: `UNMET` (Rose badge)
  - Diagnostic Text: `"Price ($341.91) is extended above preferred entry corridor ($325.07–$339.38). Wait for low-volume pullback."`
  - Guidance: Do not buy; wait for price to return to corridor.
- **Bottom of Page (`IMG_8592`, `OptimalEntryExitCard.tsx` via `decisionReadiness.ts`)**:
  - Checklist Item: `"Gate 1: PASSED"` (Emerald badge)
  - Diagnostic Text: `"Price ($341.91) is positioned inside the institutional accumulation corridor ($325.07–$339.38)"`
  - Tactical Action Hint: `"⚠️ Near Zone Ceiling: Spot ($341.91) is near the upper bound of the buy zone. Scale in with 30% initial size..."`
  - Guidance: Buy immediately with 30% initial tranche.

### 3.2 Exact Code Tracing
The contradiction arises from two separate, unsynchronized corridor evaluation functions running in parallel on the identical data payload:

```
[Payload: spot=$341.91, min=$325.07, max=$339.38]
       │
       ├──► decisionHierarchyUtils.ts:39-56 ──► Strict [min, max] ────────► UNMET ("Wait for pullback")
       │
       └──► decisionReadiness.ts:99-108 ──────► 1.02x Buffer Logic ───────► PASSED ("Positioned inside corridor")
```

1. **Top-of-Page Implementation (`frontend/lib/decisionHierarchyUtils.ts:39–56`)**:
```typescript
const canonicalLower = Math.min(lower, upper);
const canonicalUpper = Math.max(lower, upper);
const isInCorridor = price >= canonicalLower && price <= canonicalUpper;

if (isInCorridor) {
  conditions.push({
    id: "corridor",
    category: "CORRIDOR",
    title: "Price Inside Accumulation Corridor",
    description: `Current spot ($${price.toFixed(2)}) is positioned within the preferred risk corridor (${kl.watchZone || `$${canonicalLower.toFixed(2)} – $${canonicalUpper.toFixed(2)}`}).`,
    status: "MET",
  });
} else if (price > canonicalUpper) {
  conditions.push({
    id: "corridor",
    category: "CORRIDOR",
    title: "Pullback into Accumulation Corridor",
    description: `Price ($${price.toFixed(2)}) is extended above preferred entry corridor (${kl.watchZone || `$${canonicalLower.toFixed(2)} – $${canonicalUpper.toFixed(2)}`}). Wait for low-volume pullback.`,
    status: "UNMET",
  });
} else {
  // Below corridor floor...
}
```

2. **Bottom-of-Page Implementation (`frontend/lib/decisionReadiness.ts:99–108`)**:
```typescript
if (isStage4) {
  gate1State = "BLOCKING";
  gate1Explanation = "Asset is in Stage 4 distribution. Structural base required before accumulation corridor applies.";
} else if (currentPrice! > entryMax * 1.02) {
  gate1State = "BLOCKING";
  gate1Explanation = `Price ($${currentPrice!.toFixed(2)}) has extended >2% past the accumulation corridor ($${entryMin.toFixed(2)}–$${entryMax.toFixed(2)}). Do not chase above resistance.`;
} else if (currentPrice! < entryMin) {
  gate1State = "BLOCKING";
  gate1Explanation = `Price ($${currentPrice!.toFixed(2)}) is below the accumulation corridor ($${entryMin.toFixed(2)}–$${entryMax.toFixed(2)}). Awaiting base stabilization.`;
} else {
  gate1State = "PASSED";
  // DEFECT: Emits factually false text when entryMax < currentPrice <= entryMax * 1.02!
  gate1Explanation = `Price ($${currentPrice!.toFixed(2)}) is positioned inside the institutional accumulation corridor ($${entryMin.toFixed(2)}–$${entryMax.toFixed(2)}).`;
}
```

### 3.3 Complete Boundary Condition Partitioning
The behavior across the full real line $\mathbb{R}^+$ of spot prices is formally partitioned into four mathematical regimes:

| Regime | Mathematical Condition | Numerical Range (AAPL) | `decisionHierarchyUtils.ts` | `decisionReadiness.ts` | UI Consistency | Defect State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Floor Violation** | $\text{Spot} < \text{EntryMin}$ | $\text{Spot} < \$325.07$ | `UNMET` (*"Reclaim Base Floor"*) | `BLOCKING` (*"Below corridor"*) | **Consistent** (Both block) | None |
| **2. True Corridor** | $\text{EntryMin} \le \text{Spot} \le \text{EntryMax}$ | $\$325.07 \le \text{Spot} \le \$339.38$ | `MET` (*"Price Inside Corridor"*) | `PASSED` (*"Positioned inside corridor"*) | **Consistent** (Both pass) | None |
| **3. Inversion Buffer** | $\mathbf{\text{EntryMax} < \text{Spot} \le \text{EntryMax} \times 1.02}$ | $\mathbf{\$339.38 < \text{Spot} \le \$346.17}$ | **`UNMET` (*"Extended above corridor; Wait"*)** | **`PASSED` (*"Positioned inside corridor"*)** | **DIRECT CONTRADICTION** | **CRITICAL DEFECT (`DEF-01`)** |
| **4. Hard Extension** | $\text{Spot} > \text{EntryMax} \times 1.02$ | $\text{Spot} > \$346.17$ | `UNMET` (*"Extended above corridor; Wait"*) | `BLOCKING` (*"Extended >2% past corridor"*) | **Consistent** (Both block) | None |

### 3.4 Forensic Proof of Bug in Regime 3
For the audited AAPL asset state:
- $\text{EntryMax} = \$339.38$
- $1.02 \times \text{EntryMax} = 1.02 \times \$339.38 = \mathbf{\$346.1676}$
- $\text{Spot} = \mathbf{\$341.91}$
- Extension magnitude: $\frac{\$341.91 - \$339.38}{\$339.38} \times 100 = \mathbf{+0.7454\%}$
- Evaluation in `decisionHierarchyUtils.ts`: $\$341.91 > \$339.38 \implies \text{isInCorridor} = \text{false} \implies \mathbf{\text{UNMET}}$.
- Evaluation in `decisionReadiness.ts`:
  - `currentPrice > entryMax * 1.02` $\iff \$341.91 > \$346.1676 \implies \mathbf{\text{False}}$.
  - `currentPrice < entryMin` $\iff \$341.91 < \$325.07 \implies \mathbf{\text{False}}$.
  - Fall-through to `else` branch: sets `gate1State = "PASSED"` and generates string:
    `"Price ($341.91) is positioned inside the institutional accumulation corridor ($325.07–$339.38)"`.

**Defect Summary**: The code emits an affirmative assertion that $\$341.91$ is inside $[\$325.07, \$339.38]$, which is a mathematical impossibility, while the top of the page instructs the user to wait for a pullback.

### 3.5 Surgical Remediation Specification
1. In `frontend/lib/decisionHierarchyUtils.ts`, export a canonical corridor state evaluator:
```typescript
export interface CorridorEvaluation {
  state: "INSIDE_CORRIDOR" | "EXTENDED_MARGINAL" | "EXTENDED_EXCESSIVE" | "BELOW_FLOOR";
  isActionable: boolean;
  spotPrice: number;
  entryMin: number;
  entryMax: number;
  extensionPct: number;
  statusBadge: "MET" | "UNMET";
  canonicalMessage: string;
}

export function evaluateCorridorMembership(spot: number, min: number, max: number): CorridorEvaluation {
  const lower = Math.min(min, max);
  const upper = Math.max(min, max);
  const extensionPct = ((spot - upper) / upper) * 100;

  if (spot < lower) {
    return {
      state: "BELOW_FLOOR",
      isActionable: false,
      spotPrice: spot,
      entryMin: lower,
      entryMax: upper,
      extensionPct,
      statusBadge: "UNMET",
      canonicalMessage: `Price ($${spot.toFixed(2)}) is testing lower bounds below corridor ($${lower.toFixed(2)}–$${upper.toFixed(2)}). Awaiting base stabilization.`,
    };
  }

  if (spot <= upper) {
    return {
      state: "INSIDE_CORRIDOR",
      isActionable: true,
      spotPrice: spot,
      entryMin: lower,
      entryMax: upper,
      extensionPct: 0,
      statusBadge: "MET",
      canonicalMessage: `Price ($${spot.toFixed(2)}) is positioned within the preferred accumulation corridor ($${lower.toFixed(2)}–$${upper.toFixed(2)}).`,
    };
  }

  if (spot <= upper * 1.02) {
    return {
      state: "EXTENDED_MARGINAL",
      isActionable: false,
      spotPrice: spot,
      entryMin: lower,
      entryMax: upper,
      extensionPct,
      statusBadge: "UNMET",
      canonicalMessage: `Price ($${spot.toFixed(2)}) is extended +${extensionPct.toFixed(2)}% above corridor ceiling ($${upper.toFixed(2)}). Pending pullback into buy zone.`,
    };
  }

  return {
    state: "EXTENDED_EXCESSIVE",
    isActionable: false,
    spotPrice: spot,
    entryMin: lower,
    entryMax: upper,
    extensionPct,
    statusBadge: "UNMET",
    canonicalMessage: `Price ($${spot.toFixed(2)}) has extended +${extensionPct.toFixed(2)}% past corridor ceiling ($${upper.toFixed(2)}). Do not chase above resistance.`,
  };
}
```
2. Refactor `decisionReadiness.ts` to consume `evaluateCorridorMembership(...)`. Gate 1 MUST evaluate to `BLOCKING` whenever `state !== "INSIDE_CORRIDOR"`, eliminating contradictory scaling hints.

---

## 4. Item 2: Dual-Denominator Arithmetic Trace for Stop Loss & Targets (`IMG_8594` vs `IMG_8596`)

### 4.1 Observed Inconsistency
In the Product Owner screenshot series:
- **`IMG_8594` (`OptimalEntryExitCard.tsx:733`)**:
  `"Stop Loss: $315.32 (-7.09% from base)"`
- **`IMG_8596` (`insightGenerator.ts:486`)**:
  `"Capital Preservation: Invalidation Stop Loss $315.32 (-7.8%)"`

Both UI components reference the identical dollar level ($315.32) for the identical asset (AAPL) at the identical point in time, but display conflicting risk percentages: **`-7.09%` vs `-7.8%`**.

### 4.2 Mathematical Derivations of the Three Bases

```
┌────────────────────────────────────────────────────────────────────────┐
│ Structural Invalidation Price (Stop Floor) = $315.32                  │
├────────────────────────────────┬───────────────────────┬───────────────┤
│ Reference Base Price           │ Dollar Difference     │ Percentage    │
├────────────────────────────────┼───────────────────────┼───────────────┤
│ Planned Entry ($339.38)        │ -$24.06               │ -7.09%        │
│ Prior Session Close ($342.00)  │ -$26.68               │ -7.80%        │
│ Live Spot Price ($341.91)      │ -$26.59               │ -7.78%        │
└────────────────────────────────┴───────────────────────┴───────────────┘
```

#### Basis A: Planned Entry Ceiling ($339.38)
- Source code: `analyst_dashboard/analyzers/optimal_execution.py:702`
  ```python
  plan["stop_loss_pct"] = round(((structural_invalidation - planned_entry) / planned_entry) * 100, 2)
  ```
- Calculation:
  $$\Delta_{\text{entry}} = \$315.32 - \$339.38 = -\$24.06$$
  $$\%_{\text{entry}} = \frac{-\$24.06}{\$339.38} \times 100 = -7.089398...\% \xrightarrow{\text{round 2 dec}} \mathbf{-7.09\%}$$
- Displayed in `OptimalEntryExitCard.tsx:733` as: `(${stop_loss_pct}% from base)`.
- Rationale: Represents planned execution risk assuming an order fills exactly at the corridor ceiling.

#### Basis B: Prior Session Close / Reference Close ($342.00)
- Source code: `frontend/lib/insightGenerator.ts:33, 486, 539, 585`
  ```typescript
  const safePrice = isPriceValid ? currentPrice : 0;
  // ...
  riskStop: `$${stopLoss.toFixed(2)} (${(((stopLoss - safePrice) / safePrice) * 100).toFixed(1)}%)`
  ```
- Calculation:
  $$\Delta_{\text{ref}} = \$315.32 - \$342.00 = -\$26.68$$
  $$\%_{\text{ref}} = \frac{-\$26.68}{\$342.00} \times 100 = -7.801169...\% \xrightarrow{\text{round 1 dec}} \mathbf{-7.8\%}$$
- Rationale: The narrative generator received the reference close ($342.00) as `currentPrice`, computing drawdown from the daily close.

#### Basis C: Live Spot Price ($341.91)
- Calculation:
  $$\Delta_{\text{spot}} = \$315.32 - \$341.91 = -\$26.59$$
  $$\%_{\text{spot}} = \frac{-\$26.59}{\$341.91} \times 100 = -7.776899...\% \xrightarrow{\text{round 2 dec}} \mathbf{-7.78\%} \xrightarrow{\text{round 1 dec}} \mathbf{-7.8\%}$$
- Rationale: True immediate downside risk for a trader buying at the current tape price.

### 4.3 Target Return Arithmetic Traces
The same denominator ambiguity affects Take Profit 1 ($383.89) and Take Profit 2 ($407.95):

1. **Take Profit 1 ($383.89)**:
   - Relative to Planned Entry ($339.38):
     $$\frac{\$383.89 - \$339.38}{\$339.38} \times 100 = \frac{+\$44.51}{\$339.38} \times 100 = \mathbf{+13.115\% \xrightarrow{\text{round 2 dec}} +13.12\%}$$
     (Matches `OptimalEntryExitCard.tsx:645`).
   - Relative to Live Spot ($341.91):
     $$\frac{\$383.89 - \$341.91}{\$341.91} \times 100 = \frac{+\$41.98}{\$341.91} \times 100 = \mathbf{+12.278\% \xrightarrow{\text{round 2 dec}} +12.28\%}$$

2. **Take Profit 2 ($407.95)**:
   - Relative to Planned Entry ($339.38):
     $$\frac{\$407.95 - \$339.38}{\$339.38} \times 100 = \frac{+\$68.57}{\$339.38} \times 100 = \mathbf{+20.204\% \xrightarrow{\text{round 2 dec}} +20.20\%}$$
     (Matches `OptimalEntryExitCard.tsx:685`).
   - Relative to Live Spot ($341.91):
     $$\frac{\$407.95 - \$341.91}{\$341.91} \times 100 = \frac{+\$66.04}{\$341.91} \times 100 = \mathbf{+19.315\% \xrightarrow{\text{round 2 dec}} +19.32\%}$$

### 4.4 Remediation & UI Standardization
The UI must explicitly label the reference anchor for all percentage changes:
- In `OptimalEntryExitCard.tsx`:
  `"Stop Loss: $315.32 | -$26.59 (-7.78% from spot $341.91) | -$24.06 (-7.09% from entry $339.38)"`
- In `insightGenerator.ts`: Standardize `safePrice` to always reflect live spot, and round to 2 decimal places with explicit basis tagging: `"$315.32 (-7.78% from spot)"`.

---

## 5. Item 3: Blended 50/50 Multi-Target Risk/Reward Formula vs Single-Target Spot Math (`IMG_8592`)

### 5.1 Observed Phenomenon
In `IMG_8592`, the Execution Card prominently reports:
$$\mathbf{\text{Risk/Reward Ratio: 2.35:1}}$$
Directly adjacent to:
- Spot: `$341.91`
- Entry Zone: `$325.07 – $339.38`
- Take Profit 1: `$383.89`
- Take Profit 2: `$407.95`
- Stop Loss: `$315.32`

### 5.2 The Trader Discrepancy
If an institutional analyst performs standard single-target spot arithmetic from live tape to TP1:
$$\text{Reward}_{\text{Spot}\to\text{TP1}} = \$383.89 - \$341.91 = \$41.98$$
$$\text{Risk}_{\text{Spot}\to\text{Stop}} = \$341.91 - \$315.32 = \$26.59$$
$$\text{R:R}_{\text{Spot}\to\text{TP1}} = \frac{\$41.98}{\$26.59} = \mathbf{1.5788... \approx 1.58:1}$$

Comparing $1.58:1$ to the displayed $2.35:1$ gives the immediate impression of a 49% calculation error or arbitrary marketing inflation.

### 5.3 Authoritative Python Engine Implementation
The backend code in `analyst_dashboard/analyzers/optimal_execution.py:751–756` implements a multi-target blended scale-out model:

```python
tp1_gain = max(min_tick, plan["take_profit_1"] - plan["planned_entry"])
tp2_gain = max(min_tick, plan["take_profit_2"] - plan["planned_entry"])
blended_reward = 0.50 * tp1_gain + 0.50 * tp2_gain
execution_risk = plan["planned_entry"] - plan["structural_invalidation"]
if execution_risk > 0:
    blended_rr = round(blended_reward / execution_risk, 2)
```

### 5.4 Exact Mathematical Derivation of 2.35:1
1. **Planned Entry Anchor**: Clamped to corridor upper ceiling:
   $$\text{planned\_entry} = \$339.38$$
2. **Gain to TP1 from Planned Entry**:
   $$\text{tp1\_gain} = \$383.89 - \$339.38 = \mathbf{\$44.51}$$
3. **Gain to TP2 from Planned Entry**:
   $$\text{tp2\_gain} = \$407.95 - \$339.38 = \mathbf{\$68.57}$$
4. **Blended Reward (50% TP1 / 50% TP2 Exit Allocation)**:
   $$\text{blended\_reward} = (0.50 \times \$44.51) + (0.50 \times \$68.57) = \$22.255 + \$34.285 = \mathbf{\$56.54}$$
5. **Execution Risk from Planned Entry**:
   $$\text{execution\_risk} = \$339.38 - \$315.32 = \mathbf{\$24.06}$$
6. **Blended Risk/Reward Ratio**:
   $$\text{blended\_rr} = \frac{\$56.54}{\$24.06} = 2.349958437... \xrightarrow{\text{round to 2 dec}} \mathbf{2.35:1}$$

### 5.5 Comparison Matrix Across All Analytical Models

| Model Name | Entry Basis | Exit Allocation | Reward ($) | Risk ($) | Calculated R:R | Institutional Utility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Engine Blended (Current)** | Planned Base ($339.38) | 50% TP1 / 50% TP2 | $56.54 | $24.06 | **2.35:1** | Structural setup potential from buy zone ceiling |
| **Spot to TP1 (Immediate)** | Spot Tape ($341.91) | 100% TP1 | $41.98 | $26.59 | **1.58:1** | Real-world conservative target from current tape |
| **Spot to TP2 (Max Hold)** | Spot Tape ($341.91) | 100% TP2 | $66.04 | $26.59 | **2.48:1** | Full position hold to secondary extension |
| **Spot Blended (50/50)** | Spot Tape ($341.91) | 50% TP1 / 50% TP2 | $54.01 | $26.59 | **2.03:1** | Staged exit evaluated from current live tape |
| **Optimal Base Floor** | Corridor Min ($325.07) | 50% TP1 / 50% TP2 | $70.85 | $9.75 | **7.27:1** | Theoretical upper bound for entry at support floor |

### 5.6 Friction & Transparency Assessment
- **Friction Assumptions**: The engine assumes zero execution friction ($\text{slippage} = \$0.00$, $\text{broker commission} = \$0.00$, SEC regulatory fees omitted).
- **UI Transparency Deficit**: The terminal presents `2.35:1` as an unvarnished number without disclosing:
  1. That entry is assumed at `$339.38` (planned base), NOT current spot `$341.91`.
  2. That exit assumes a 50/50 split across TP1 and TP2.
- **Remediation Specification**: Add an explanatory popover and dual metric display:
  `"2.35:1 (Setup Potential: Blended 50/50 from $339.38 base) | 1.58:1 (Spot to TP1 from $341.91)"`.

---

## 6. Item 4: Dynamic Data Reality, Client-Clock Masquerade & Demo Mode Trigger (`IMG_8593`, `IMG_8598`)

### 6.1 Dynamic Data Reality & Feed Architecture
Inspection of `frontend/app/page.tsx:200–230` reveals the complete data retrieval architecture of the Analysis page:
- **Zero WebSocket Connections**: There are NO active socket listeners (`new WebSocket` or Socket.io) in the analysis bundle.
- **Zero Server-Sent Events (SSE)**: No `EventSource` connections exist.
- **Zero Polling Loops**: There is NO `setInterval(...)` or recurring background worker fetching fresh price quotes.
- **Single Mount-Time HTTP Fetch**:
  ```typescript
  // frontend/app/page.tsx:210
  const res = await fetchAssetAnalytics(selectedSymbol, period, apiInterval, userRole, knownPrice, knownChange);
  ```
  The page fetches data once when mounted or when `selectedSymbol` changes. If a user leaves the browser tab open for six hours during market hours, the data remains entirely static.

### 6.2 Client-Clock Masquerading as Exchange Tape Timestamp
In `frontend/app/page.tsx:64–68`:
```typescript
const [lastUpdatedTime, setLastUpdatedTime] = useState<string>("");

useEffect(() => {
  setLastUpdatedTime(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
}, [data]);
```
- **Forensic Finding**: `lastUpdatedTime` is populated using the **client browser's local operating system clock** (`new Date().toLocaleTimeString()`).
- **Hazard**: In `IMG_8593`, this renders as `"LIVE SPOT: $341.91"` next to the browser time. This misleads traders into believing they are viewing a live streaming tape tick, when in reality:
  - The quote may be 20-minute delayed or from yesterday's session close.
  - If the client OS clock is inaccurate by 15 minutes, the displayed time will be wrong by 15 minutes.
  - Long-lived tabs show the initial load time rather than the true quote age.

### 6.3 Demo Mode Triggering on Live Assets (`IMG_8598`)
In `frontend/app/page.tsx:44–46` and `664–667`:
```typescript
// Lines 44-46
const searchParams = useSearchParams();
const urlSymbol = searchParams.get("symbol") || searchParams.get("ticker");
const [hasExplicitSymbol, setHasExplicitSymbol] = useState<boolean>(!!urlSymbol);
const [selectedSymbol, setSelectedSymbol] = useState<string>(urlSymbol ? urlSymbol.toUpperCase() : "AAPL");

// Lines 664-667
badge={hasExplicitSymbol ? "Active Target" : "Demonstration Mode"}
symbol={hasExplicitSymbol ? selectedSymbol : null}
isDemo={!hasExplicitSymbol}
demoNotice={!hasExplicitSymbol ? `Displaying ${selectedSymbol} as a demonstration asset. Search any ticker to analyze your target.` : undefined}
```

- **Forensic Finding**:
  - When a user navigates directly to `/` or `/analysis` without an explicit query string `?symbol=AAPL`, `urlSymbol` is `null`.
  - `hasExplicitSymbol` evaluates to `false`.
  - The component defaults `selectedSymbol` to `"AAPL"`, executes a live backend query, and receives authentic production telemetry.
  - However, because `hasExplicitSymbol` is `false`, it passes `isDemo = true` to `PageIntro.tsx`.
- **User Impact (`IMG_8598`)**: The user sees:
  `"Demonstration Mode — Displaying AAPL as a demonstration asset. Search any ticker to analyze your target."`
  This degrades authentic market data into an apparent "toy" or simulation, destroying analytical credibility.

### 6.4 Remediation Specification
1. **True Upstream Provenance**: Pass the exchange quote timestamp from the backend API response (`data.quoteTimestamp` or `data.asOfUtc`).
2. **Freshness Banner Component**:
   ```tsx
   <DataProvenanceBanner
     exchangeTimestamp={data.quoteTimestamp}
     marketSession={data.marketSession} // 'OPEN' | 'CLOSED' | 'POST_MARKET'
     dataSource={data.dataSource} // 'POLYGON_FEED' | 'EOD_CACHE' | 'SYNTHETIC_MOCK'
     isDelayed={data.is15MinDelayed}
   />
   ```
3. **Decouple Demo Mode**: Default `selectedSymbol = "AAPL"` should retain `isDemo = false`. `isDemo` must only trigger when `data.isSynthetic === true` or in dedicated offline sandbox test suites.

---

## 7. Item 5: Mobile Viewport Clearances, Truncation & Duplicate Spotlight Controls (`IMG_8596`, `IMG_8599`)

### 7.1 Walkthrough Step Title Truncation (`IMG_8596`)
In `frontend/components/terminal/GuidedTerminalView.tsx:325–340`:
```tsx
<div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-1.5">
  {steps.map((step, idx) => (
    <button
      key={idx}
      id={`walkthrough-step-btn-${idx}`}
      onClick={() => setActiveStep(activeStep === idx ? null : idx)}
      className="p-2.5 rounded-lg text-left text-xs font-mono border transition-all cursor-pointer min-h-[36px] ..."
    >
      <span className="font-bold block truncate">{step.title}</span>
    </button>
  ))}
</div>
```
- **Forensic Proof**:
  - `truncate` applies CSS: `overflow: hidden; text-overflow: ellipsis; white-space: nowrap;`.
  - In a 2-column mobile layout ($<640\text{px}$), column width is approximately $160\text{px}$.
  - Step 3 title: `"3. Why does ARX like/caution it?"` (34 characters) requires $\approx 240\text{px}$ in `font-mono text-xs`.
  - It is aggressively clipped to `"3. Why does ARX li..."`, hiding the topic from the user.

### 7.2 Hidden Precondition Descriptions on Mobile
In `frontend/components/terminal/StandardTerminalView.tsx:149`:
```tsx
<p className="text-xs text-slate-400 leading-relaxed font-sans hidden sm:block">
  {cond.description}
</p>
```
- **Forensic Proof**: On mobile screens ($<640\text{px}$), `hidden sm:block` completely suppresses the precondition description.
- **Impact**: When Gate 1 evaluates to `UNMET`, the user sees the title `"Pullback into Accumulation Corridor"` and badge `UNMET`, but the explanation—*"Price ($341.91) is extended above preferred entry corridor ($325.07–$339.38). Wait for low-volume pullback"*—is completely hidden.

### 7.3 Duplicate "View Full Setups" Buttons (`IMG_8599`)
In `frontend/components/WeeklyConfluenceSpotlight.tsx:800–870`:
- **Line 809 (Header Button)**:
  ```tsx
  <button
    type="button"
    onClick={() => setIsCollapsed(!isCollapsed)}
    className="text-xs px-2.5 py-2 sm:py-1 min-h-[36px] sm:min-h-0 rounded-md font-mono text-slate-400 ..."
  >
    {isCollapsed ? "View Full Setups ▼" : "Collapse ▲"}
  </button>
  ```
- **Line 865 (Quick-Switcher Ribbon Button)**:
  ```tsx
  {isCollapsed && (
    <div className="flex flex-wrap items-center justify-between gap-2 ...">
      {/* ... candidate tickers ... */}
      <button
        type="button"
        onClick={() => setIsCollapsed(false)}
        className="text-[11px] font-mono text-cyan-400 ... min-h-[36px] sm:min-h-0"
      >
        <span>View Full Setups</span>
        <span>▼</span>
      </button>
    </div>
  )}
  ```
- **Forensic Proof**: When `isCollapsed === true`, both buttons render concurrently within 80 vertical pixels of each other. Both execute the identical action (`setIsCollapsed(false)`).

### 7.4 Sub-44px Touch Target Deficits
- `GuidedTerminalView.tsx:331`: `className="... min-h-[36px]"`
- `WeeklyConfluenceSpotlight.tsx:805`: `className="... min-h-[36px] sm:min-h-0"`
- `WeeklyConfluenceSpotlight.tsx:867`: `className="... min-h-[36px] sm:min-h-0"`
- **Forensic Finding**: WCAG 2.1 Success Criterion 2.5.5 (Target Size) mandates a minimum interactive target of $44\times 44\text{px}$. The codebase explicitly sets mobile heights to $36\text{px}$, causing tap collisions and mis-clicks on mobile devices.

### 7.5 Sticky Bottom Navigation Clearance Collision
In `frontend/components/AdaptiveTerminal.tsx:350–420`, the terminal features a sticky bottom action dock. The root container lacks adequate bottom padding (`pb-12`), causing the fixed dock to cover primary buttons and disclosures on viewports $<768\text{px}$.

### 7.6 Surgical Remediation Specification
1. **GuidedTerminalView**: Replace `truncate` with `line-clamp-2 break-words leading-tight whitespace-normal`. Change `min-h-[36px]` to `min-h-[44px]`.
2. **StandardTerminalView**: Remove `hidden sm:block` from line 149, or wrap descriptions in a responsive `<details>` disclosure on mobile.
3. **WeeklyConfluenceSpotlight**: Remove the redundant ribbon button at line 865 when collapsed, retaining the header button on line 809 as the single toggle.
4. **Touch Targets**: Enforce `min-h-[44px] min-w-[44px]` across all mobile buttons.
5. **Bottom Padding**: Add `pb-28 sm:pb-8` to the terminal layout container.

---

## 8. Backlog & Traceability Matrix

The five forensic items documented in this Addendum map directly to the staged engineering backlog items in `BACKLOG_HANDOFF.md`:

| Addendum Item | Primary Defect / Issue | Governing File(s) & Lines | Backlog ID | Proposed Priority |
| :--- | :--- | :--- | :--- | :--- |
| **Item 1** | Gate 1 Corridor Check Inversion ($341.91 \le 339.38 \times 1.02$) | `frontend/lib/decisionReadiness.ts:99-108`<br>`frontend/lib/decisionHierarchyUtils.ts:39-56` | **`UX-ANALYSIS-001`** | **P1 (High Impact)** |
| **Item 2** | Dual Stop-Loss Denominators (-7.09% vs -7.80%) | `analyst_dashboard/analyzers/optimal_execution.py:702`<br>`frontend/lib/insightGenerator.ts:486, 539, 585` | **`UX-ANALYSIS-002`** | **P1 (High Impact)** |
| **Item 3** | Blended 50/50 Multi-Target R:R (2.35:1 vs 1.58:1) | `analyst_dashboard/analyzers/optimal_execution.py:751-756`<br>`frontend/components/OptimalEntryExitCard.tsx:258-270` | **`UX-ANALYSIS-002`** | **P1 (High Impact)** |
| **Item 4** | Static Data Lifecycle, Client-Clock Masquerade & Demo Mode Trigger | `frontend/app/page.tsx:64-68, 210, 664-667`<br>`frontend/components/PageIntro.tsx:65-68` | **`UX-ANALYSIS-003`** | **P1 (High Impact)** |
| **Item 5** | Mobile Viewport Clipping, Hidden Descriptions, Duplicate Spotlight Buttons | `frontend/components/terminal/GuidedTerminalView.tsx:336`<br>`frontend/components/terminal/StandardTerminalView.tsx:149`<br>`frontend/components/WeeklyConfluenceSpotlight.tsx:809, 865` | **`UX-MOBILE-001`** | **P2 (Standard)** |

---

## 9. Governance & Verification Attestation

The undersigned agent certifies under strict read-only audit isolation that:
1. All five outstanding items have been fully investigated, mathematically proven, and cited to exact source lines.
2. No application source files, test suites, or configuration records have been modified.
3. No remote git commands (`git push`, `git fetch`, `gh pr`) have been executed.
4. Existing research history in `docs/research/analysis-ux-audit/` is preserved.
5. All deliverables remain staged for controlled implementation in accordance with ARX engineering release gates.

**Verification Sign-Off**:  
`ARX Systems Forensic Engineering — Worktree D Audit Team`  
`Commit Reference: Follow-up forward commit on research/analysis-ux-decision-audit`
