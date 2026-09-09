# Visual Hierarchy Recommendations
## ARX Terminal: 3-Tier Semantic Hierarchy & Metric Governance Specification (Horizon 14.3)

**Document ID**: `HIERARCHY-SPEC-ARX-H14.3-M1`  
**Classification**: Visual Hierarchy & Metric Classification Architecture  
**Governing Standard**: Institutional Financial Terminal Decision Velocity Standard  
**Target Systems**: All Flagship Hubs (`/radar`, `/setups`, `/portfolio`, `/journal`, `/performance`, `/research`)  
**Date**: 2026-09-09T18:12:00+02:00  

---

## 1. Executive Summary & The Problem of Cognitive Noise

In high-stakes financial operations, cognitive overload is fatal. A quantitative terminal that presents 20 metrics with equal visual weight forces the operator to manually synthesize relevance, introducing decision latency and emotional fatigue.

ARX Terminal eliminates cognitive noise by enforcing two strict architectural standards:
1. **The 3-Tier Semantic Hierarchy**: Every flagship screen organizes its information into three distinct perceptual layers: **Level 0 (Decision)** $\to$ **Level 1 (Rationale)** $\to$ **Level 2 (Proof)**.
2. **The Metric Classification & -20% Pruning Rule**: Every data point displayed must be formally classified into one of four functional categories (**Decision**, **Risk**, **Performance**, or **Diagnostic**). Metrics that fail to directly support an institutional action are systematically pruned.

---

## 2. The 3-Tier Semantic Hierarchy Framework

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 0: THE DECISION LAYER (Visual Weight: ≥ 40%)                                     │
│ Time-to-Absorb: ≤ 10 seconds (Execution) / ≤ 30 seconds (Evaluation)                   │
│ Objective: Instant answer to the singular screen question.                            │
│ Primary Artifact: Asymmetric Decision Hero / Asymmetric Execution Ticket               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ LEVEL 1: THE RATIONALE LAYER (Visual Weight: 35%)                                      │
│ Time-to-Absorb: ≤ 30 seconds                                                           │
│ Objective: Why this decision is warranted and what constraints govern it.              │
│ Primary Artifacts: Confluence Streams, Risk Heat Maps, Attribution Pillars, Calibration│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ LEVEL 2: THE PROOF LAYER (Visual Weight: 25%)                                          │
│ Time-to-Absorb: On-Demand Deep-Dive (Audit & Verification)                             │
│ Objective: Verifiable quantitative backing, empirical ledgers, and raw formulas.       │
│ Primary Artifacts: Dense Data Ledgers, Monte Carlo Scatters, SEC EDGAR Disclosures    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Level 0: The Decision Layer ("What Action Must Be Taken?")
- **Definition**: The single dominant visual anchor on the page. It must occupy at least 40% of the initial above-the-fold visual weight.
- **Cognitive Velocity**:
  - For operational screens (`/radar`, `/setups`, `/portfolio`): **$\le 10$ seconds** to identify what deserves attention, what to trade, and what risk is carried.
  - For evaluative screens (`/journal`, `/performance`, `/research`): **$\le 30$ seconds** to identify whether ARX is preserving capital, rule discipline integrity, and thesis validity.
- **Architectural Rules**:
  - **Zero Symmetrical 4-Card Farms**: No screen may begin with a generic grid of 4 equal-sized cards (`grid-cols-4`).
  - **Asymmetric Composition**: The Level 0 container must feature a clear hero-and-rail asymmetry (e.g. 70% primary metric display + 30% utility context).
  - **Unambiguous Primary Action**: If an operational action is possible (e.g. copying an order string, exiting a breached position), exactly one high-contrast primary CTA must be present.

### 2.2 Level 1: The Rationale Layer ("Why Should This Action Be Taken?")
- **Definition**: The explanatory layer providing direct causal drivers, confluence criteria, and behavioral guardrails that justify the Level 0 decision.
- **Cognitive Velocity**: **$\le 30$ seconds** to inspect and validate.
- **Architectural Rules**:
  - **Scannable Density**: Structured as prioritized streams, comparative charts, or 2D matrices rather than fragmented text cards.
  - **High-Contrast Semantic Status**: Clear differentiation between states (e.g. `IN BUY ZONE` in vivid emerald vs `NEAR PIVOT` in amber).
  - **Explicit Behavioral Constraints**: Behavioral Governor sizing clamp factors (-25%, -50%) and rationale must be visible alongside technical parameters.

### 2.3 Level 2: The Proof Layer ("What Empirical Data Backs This?")
- **Definition**: The cryptographic, empirical, and mathematical foundation supporting Levels 0 and 1.
- **Cognitive Velocity**: On-demand auditability; secondary drill-down.
- **Architectural Rules**:
  - **Tabular Ledger Density**: Rendered using dense, compact institutional tables (`DataLedgerTable`) with row heights $\le 36\text{px}$, monospace tabular figures, and sorting.
  - **Zero Omission of Math**: Retains full quantitative rigor (Cornish-Fisher VaR 95%, Sortino Skew, Monte Carlo path counts, 31-trade canonical audit ledger).
  - **Progressive Disclosure**: Detailed formula parameters or trade logs can be collapsed or tabbed to prevent displacing Levels 0 and 1.

---

## 3. Flagship Hub 3-Tier Mapping Matrix

| Hub | Singular Core Question | Level 0: Decision (Hero) | Level 1: Rationale (Context) | Level 2: Proof (Verification) |
|-----|------------------------|--------------------------|------------------------------|-------------------------------|
| **`/radar`** | *"What deserves attention today?"* | **#1 Attention Leader Hero**: GOOGL (94 Confluence Score, Stage 2 VCP Breakout, -1.8% to pivot) with direct CTA to `/setups`. | **Confluence Stream**: High-contrast split stream: `IN BUY ZONE` (Emerald) vs `NEAR PIVOT` (Amber), RS $\ge 80$, Vol Dry-Up %. | **Factor Breakdown Drawer**: 21>50>200 EMA alignment, Greenblatt Magic Formula ROIC, 13F whale net inflow. |
| **`/setups`** | *"What should I trade right now?"* | **Asymmetric Execution Ticket**: Buy Limit `$182.40`, Stop Loss `$176.10` (-3.45%), TP1 `$195.00`, Governed Size: 13 Shs ($2,371) + Single Authorize Button. | **Governor Dynamic Sizing & Confluence**: Unclamped ($100 / 17 Shs) vs Governed ($75 / 13 Shs), -25% Drawdown Clamp rationale, VCP check. | **Quant Risk Panel**: 1,000 Monte Carlo paths, Cornish-Fisher VaR 95% (-$86), Sortino Skew (+2.84), Half-Kelly (0.25x). |
| **`/portfolio`** | *"What can hurt me?"* | **Capital at Risk Hero**: Total Risk at Stop Floors: `-$1,420 (5.68%)` + **Active Exit Rule Triggers Banner** (breached stops / profit taking). | **2D Risk Heat Map & Concentration**: Holdings sized by portfolio weight, colored by distance to stop; Tech sector cap warning. | **Open Quant Holdings Ledger**: Dense table with live quotes, entry price, stop-loss protection floors, and stress simulator. |
| **`/journal`** | *"Did I follow my rules?"* | **Discipline Status Hero**: `94.2% Discipline Score (Grade A)` + `CALM & OBJECTIVE` status (0 active losses, full sizing authorized). | **Brier Calibration Curve & Anti-Tilt Matrix**: 5-bucket probability curve ($0.18 \le 0.25$) + 4 diagnostic anti-tilt pills. | **Execution Discipline Ledger**: Audited table with planned vs realized R-multiples, rule verification stamps, and trade reflections. |
| **`/performance`**| *"Is ARX improving my results?"* | **Proof of Edge Hero**: `+$6,140 Capital Preserved` + Drawdown Protected (`-8.4% with ARX vs -19.2% Naive`). | **Counterfactual Trajectory & Pillars**: Dual equity curve + 3 Defense Pillars (Drawdown Defense, Execution Window, Capital Floor). | **31-Trade Governor Audit Ledger**: Complete audited trade log with unclamped vs governed loss deltas and replay verification hash. |
| **`/research`** | *"Why does this opportunity exist?"* | **Research Dossier Hero**: High-conviction thesis for GOOGL + Primary Catalyst Window (`Q3 Cloud Margin Expansion · 2-4 Wks`). | **Catalyst Stream & Smart Money Inflow**: Ranked catalyst feed + 13F Whale Clusters (Duquesne +$45M) + Congressional STOCK Act trades. | **Fundamental Forensics**: ROIC (31.4%), Net Debt/EBITDA (0.2x Net Cash), FCF yield (5.2%), direct SEC EDGAR links. |

---

## 4. Institutional Metric Classification System

To prevent ad-hoc metric dumping, every metric displayed across the ARX Terminal platform MUST belong to exactly one of the four institutional classes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    METRIC CLASSIFICATION TAXONOMY                       │
├───────────────────────┬─────────────────────────────────────────────────┤
│ 1. DECISION METRICS   │ Directly drives action: Buy, Sell, Size, Exit   │
├───────────────────────┼─────────────────────────────────────────────────┤
│ 2. RISK METRICS       │ Defines boundaries: Invalidation, Loss, Tilt    │
├───────────────────────┼─────────────────────────────────────────────────┤
│ 3. PERFORMANCE METRICS│ Evaluates edge: Returns, Alpha, Realized R, P&L │
├───────────────────────┼─────────────────────────────────────────────────┤
│ 4. DIAGNOSTIC METRICS │ Provides sanity context: Volume, RS, Ratios     │
└───────────────────────┴─────────────────────────────────────────────────┘
```

### 4.1 Class 1: Decision Metrics
- **Purpose**: Directly dictates or authorizes an execution action.
- **Allowed Locations**: Level 0 Hero, Primary Order Ticket, Exit Alert Banner.
- **Canonical Inventory**:
  - `Entry Pivot Price` ($)
  - `Stop Loss Floor Price` ($)
  - `Profit Target 1 & 2 Prices` ($)
  - `Recommended Governed Position Size` (Shares & Capital $)
  - `Governor Sizing Clamp Factor` (% reduction: 0%, -25%, -50%, -75%)
  - `Exit Rule Trigger State` (`STOP_BREACHED`, `TP1_HIT`, `TRAILING_STOP`)
  - `Active Confluence Score` (0–100)
  - `Discipline Adherence Score` (%)
  - `Capital Preserved Total` ($)

### 4.2 Class 2: Risk Metrics
- **Purpose**: Quantifies vulnerability, invalidation boundaries, and downside potential.
- **Allowed Locations**: Level 0 Risk Hero, Level 1 Heat Map, Level 2 Quant Drawer.
- **Canonical Inventory**:
  - `Total Capital at Risk at Stops` ($ and % of portfolio)
  - `Stop Loss Distance` (% from entry)
  - `Maximum Account Drawdown` (%)
  - `Cornish-Fisher Value at Risk (VaR 95% / 99%)` ($)
  - `Brier Probabilistic Calibration Score` (Benchmark $\le 0.25$)
  - `Active Loss Streak Count` (Integer: 0, 1, 2, 3)
  - `Position & Sector Exposure Concentration` (%)
  - `Risk of Ruin Probability` (%)

### 4.3 Class 3: Performance Metrics
- **Purpose**: Measures realized quantitative alpha, expectancy, and historical edge.
- **Allowed Locations**: Level 0 Proof Hero, Level 1 Attribution Pillars, Level 2 Ledgers.
- **Canonical Inventory**:
  - `Governed vs Naive Sharpe Ratio` (e.g. 2.41 vs 1.62)
  - `Governed vs Naive Profit Factor` (e.g. 2.14 vs 1.52)
  - `Realized R-Multiple per Trade` (e.g. +2.1R, -1.0R)
  - `Cumulative Realized & Unrealized P&L` ($)
  - `Win Rate on Governed Setups` (%)
  - `Counterfactual Alpha Spread` ($)

### 4.4 Class 4: Diagnostic Metrics
- **Purpose**: Secondary technical and fundamental indicators providing contextual confirmation.
- **Allowed Locations**: Level 1 Scannable Stream, Level 2 Technical Deep-Dive.
- **Canonical Inventory**:
  - `Relative Strength (RS) Rating` (1–99)
  - `Volume Dry-Up Contraction %` (e.g. -48%)
  - `VCP Contraction Stages` (2T, 3T, 4T)
  - `Return on Invested Capital (ROIC)` (%)
  - `Net Debt / EBITDA Ratio` (Leverage)
  - `Sortino Skewness & Kurtosis`
  - `13F Institutional Inflow Delta` ($)

---

## 5. The -20% Anti-Slop Pruning Rule

### 5.1 Pruning Criteria
A data point, container, or UI element MUST be pruned if it meets any of the following conditions:
1. **The Non-Actionable Test**: The metric cannot alter an operator's action (it is interesting trivia rather than a decision driver).
2. **The Redundancy Test**: The data is already presented with higher clarity in an adjacent container or persistent shell (e.g. triplicated market regime).
3. **The Cookie-Cutter Test**: The element exists solely to complete an artificial symmetrical layout grid (e.g. filling out a 4-card row).
4. **The Retail Toy Test**: The element caters to micro-retail gamification rather than institutional capital preservation (e.g. $50 wallet presets, celebratory confetti emojis).

### 5.2 Formal Pruning Ledger Across Flagship Hubs

| Hub | Pruned UI Element / Metric | Current Location | Rationale for Pruning | Replaced By |
|-----|----------------------------|------------------|-----------------------|-------------|
| `/radar` | 40-card repetitive 3-column grid | `radar/page.tsx:455-527` | Creates severe scanning fatigue; violates 10s rule | Asymmetric #1 Leader Hero + Dense Confluence Stream |
| `/radar` | Repeated "Stage 2 Uptrend" label | `radar/page.tsx:473` | Printed 40 times identically; zero differential signal | Implicit via VCP badge |
| `/radar` | Standalone Page Regime Header | `radar/page.tsx:337-358` | Triplicated across ribbon, shell, and page | Unified `MarketCommandRibbon` |
| `/setups` | Duplicate "Copy Broker String" button | `setups/page.tsx:287-292` | Duplicates line 280's copy handler | Single unified "Authorize Order" CTA |
| `/setups` | Duplicate Guided Mode Governor text | `setups/page.tsx:195-204` | Duplicated 40px away on lines 243-277 | Consolidated in Governor Risk Panel |
| `/setups` | Symmetrical 4-box parameter grid | `setups/page.tsx:114-142` | Equal-weight boxes bury risk floor | Asymmetric Execution Ticket Ladder |
| `/portfolio` | Generic 4-card summary row | `portfolio/page.tsx:298-340` | Textbook card-farm anti-pattern | Asymmetric Capital at Risk Hero |
| `/portfolio` | Retail wallet presets ($50, $100) | `portfolio/page.tsx:342-438` | Toy-like retail artifacts | Standard institutional scaling inputs |
| `/journal` | Generic 4-card discipline row | `journal/page.tsx:27-48` | Card-farm anti-pattern; unrendered LaTeX | Asymmetric Discipline Hero + Brier Curve |
| `/journal` | Raw LaTeX string `$\le 0.25$` | `journal/page.tsx:36` | Unrendered syntax defect | Clean Unicode `≤ 0.25` |
| `/performance`| 6 boxed mini-milestone cards | `performance/page.tsx:252-261` | Fragmented boxes clutter the canvas | Continuous Counterfactual Equity Curve |
| `/research` | 3 isolated static placeholder cards | `research/page.tsx:31-70` | Superficial stub lacking depth | Asymmetric Research Dossier & 13F stream |

---

## 6. Typographic & Color System Governance

### 6.1 Typographic Bifurcation: Monospace vs Sans-Serif
Institutional workstations maintain strict typographic boundaries to eliminate eye strain:
1. **`font-sans` (System Sans / Inter / Geist)**:
   - Reserved strictly for narrative text, investment theses, catalyst descriptions, button text, table headers, section titles, and modal instructions.
   - **Forbidden**: Applying `font-mono` to container roots (e.g. `<main>`, `<section>`).
2. **`font-mono` + `tabular-nums` (Geist Mono / JetBrains Mono / System Mono)**:
   - Reserved strictly for numerical data: dollar prices (`$182.40`), share quantities (`13 Shs`), percentages (`+10.8%`), R-multiples (`+2.06R`), order parameters (`LMT`, `STP`, `TGT`), ticker symbols (`GOOGL`), and mathematical formulas.
   - Every column of numbers in tables must align right and use tabular figures to enable vertical column scanning.

### 6.2 Semantic Color Discipline: The 5-Color Invariant
Colors carry unambiguous functional meaning. Zero decorative rainbow accents are permitted:

```
┌───────────┬───────────────────┬────────────────────────────────────────────────────────┐
│ Color     │ Tailwind Tokens   │ Exclusive Functional Purpose                           │
├───────────┼───────────────────┼────────────────────────────────────────────────────────┤
│ Emerald   │ emerald-400 / 500 │ Positive Returns, Capital Preserved, IN_BUY_ZONE, Pass │
│ Rose      │ rose-400 / 500    │ Stop Loss Floor, Drawdown, Tilt Alert, Invalidation    │
│ Cyan      │ cyan-400 / 500    │ Active Tool Focus, Selection Cursor, Active Nav Tab    │
│ Amber     │ amber-400 / 500   │ NEAR_PIVOT (-2% to 0%), Caution, Capital Floor Buffer  │
│ Purple    │ purple-400 / 500  │ Smart Money 13F Inflow, Quant Analytics, Target 2      │
└───────────┴───────────────────┴────────────────────────────────────────────────────────┘
```
- **The Anti-Cyan Strict Rule**: Under no circumstances may Cyan be used for bullish performance, setup confluence scores, or profit indicators.
