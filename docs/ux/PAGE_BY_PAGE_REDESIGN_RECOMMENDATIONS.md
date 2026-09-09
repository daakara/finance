# Page-by-Page Redesign Recommendations
## ARX Terminal: Comprehensive Architectural Redesign Blueprints for the 6 Flagship Hubs (Horizon 14.3)

**Document ID**: `PAGES-REDESIGN-ARX-H14.3-M1`  
**Classification**: Flagship Screen Architecture & Redesign Blueprints  
**Governing PRD**: `docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`  
**Core Framework**: 3-Tier Semantic Hierarchy & -20% Anti-Slop Rule  
**Target Hubs**: `frontend/app/` (`/radar`, `/setups`, `/portfolio`, `/journal`, `/performance`, `/research`)  
**Date**: 2026-09-09T18:18:00+02:00  

---

## 1. Executive Summary & Core Redesign Mandate

Each of the 6 flagship hubs in ARX Terminal exists to answer **exactly one operational question** for the trader or portfolio manager. Under the Horizon 14.3 mandate, each page is reconstructed around an asymmetric Level 0 Decision Hero, supported by high-density Level 1 Rationale and empirical Level 2 Proof, while pruning 20% of non-essential UI clutter.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ THE 6 SINGULAR OPERATIONAL QUESTIONS                                                   │
├─────────────────┬──────────────────────────────────────────┬───────────────────────────┤
│ Hub Route       │ Singular Operational Question            │ Decision Velocity Target  │
├─────────────────┼──────────────────────────────────────────┼───────────────────────────┤
│ 1. /radar       │ "What deserves attention today?"         │ ≤ 10 Seconds              │
│ 2. /setups      │ "What should I trade right now?"         │ ≤ 10 Seconds              │
│ 3. /portfolio   │ "What can hurt me?"                      │ ≤ 10 Seconds              │
│ 4. /journal     │ "Did I follow my rules?"                 │ ≤ 30 Seconds              │
│ 5. /performance │ "Is ARX actually improving my results?"  │ ≤ 30 Seconds              │
│ 6. /research    │ "Why does this opportunity exist?"       │ ≤ 30 Seconds              │
└─────────────────┴──────────────────────────────────────────┴───────────────────────────┘
```

---

## 2. Hub 1: `/radar` ("What Deserves Attention Today?")

### 2.1 Current Anti-Patterns & Deficiencies
- **40-Card Uniform Grid**: Lines 455–527 render 40 identical cards in a 3-column grid (`grid-cols-3`). Every asset competes with equal visual weight, causing optical fatigue.
- **Zero Urgency Distinction**: Status badges for `IN_BUY_ZONE`, `NEAR_PIVOT`, `VOLUME_DRYUP`, and `PULLBACK_SUPPORT` all use the identical emerald class (`bg-emerald-950/80 border-emerald-800/60 text-emerald-400`).
- **Repetitive Boilerplate**: The string "Stage 2 Uptrend" is printed 40 times in identical typography.
- **Anti-Cyan Invariant Violation**: Line 492 renders RS Rating in `text-cyan-400`.
- **Vertical Overhead**: Lines 337–358 mount a standalone Market Regime banner duplicating `MarketCommandRibbon`.

### 2.2 The -20% Anti-Slop Cut List
1. **Cut**: Standalone page-level Market Regime header (lines 337–358).
2. **Cut**: Generic 40-card uniform grid (`grid-cols-3`).
3. **Cut**: Repetitive "Stage 2 Uptrend" label on every card.
4. **Prune**: Three-box metric footers on individual cards; consolidate into a dense stream row.

### 2.3 3-Tier Semantic Hierarchy Architecture
- **Level 0 (Decision — ≥ 40% Visual Weight)**:
  - **Asymmetric #1 High-Confluence Attention Leader**:
    - Featured asset: `GOOGL` (Score 94, Stage 2 VCP 4T, -1.8% to pivot).
    - Prominent status indicator: `IN ACTIONABLE BREAKOUT ZONE ($182.40 – $184.20)`.
    - Key metrics rail: RS 96/99, Vol Dry-Up -48%, Insider Inflow $1.2M.
    - Direct primary action CTA: *"ARM EXECUTION TICKET IN /SETUPS →"*.
- **Level 1 (Rationale — 35% Visual Weight)**:
  - **Prioritized Confluence Stream**:
    - High-contrast visual grouping:
      - `[IN BUY ZONE]` (Vivid Emerald `bg-emerald-500/20 text-emerald-300 border-emerald-500`): Active buy corridors.
      - `[NEAR PIVOT]` (Amber Radar `bg-amber-500/10 text-amber-400 border-amber-500/40`): Distance to trigger (-0.5% to -2.0%).
      - `[WATCHING]` (Slate `bg-slate-800/40 text-slate-400`): Consolidating bases.
    - Scannable stream columns: Ticker & Name, Confluence Score, RS Rating, VCP Contraction Stage, Volume Dry-Up %, Converged Models, Catalyst Summary.
- **Level 2 (Proof — 25% Visual Weight)**:
  - **Multi-Factor Criteria Drawer**:
    - Moving Average alignment (21 > 50 > 200 EMA verified).
    - SEC Form 4 Director Purchase verification.
    - Greenblatt Magic Formula ROIC (31.4%) & Earnings Yield.

### 2.4 Metric Classification
- **Decision Metrics**: Confluence Score (94), Execution Status (`IN_BUY_ZONE`), Pivot Trigger Price ($182.40).
- **Risk Metrics**: Volume Dry-Up % (-48%), VCP Contraction Swings (4T), Distance to Pivot (-1.8%).
- **Performance Metrics**: Relative Strength Rating (96/99), 1-Month Price Momentum (+14.2%).
- **Diagnostic Metrics**: Multi-Model Convergence Count (3 Models: VCP + Smart Money + Value).

### 2.5 10-Second Decision Velocity Walkthrough
1. **0–3s**: The user's eye lands on the top-left Asymmetric Leader Hero (`GOOGL`, 94 Confluence, In Buy Zone).
2. **3–7s**: The user scans the Level 1 stream filtered to `IN BUY ZONE` and spots the 3 actionable names today.
3. **7–10s**: The user clicks *"Arm Execution Ticket"* to transition directly to `/setups?ticker=GOOGL`.

---

## 3. Hub 2: `/setups` ("What Should I Trade Right Now?")

### 3.1 Current Anti-Patterns & Deficiencies
- **Duplicate Copy Buttons**: Lines 280–292 render two stacked buttons ("Authorize Order" and "Copy Broker Order String") that execute identical clipboard operations.
- **Symmetric 4-Box Parameter Grid**: Entry Pivot, Stop Loss, Target 1, Target 2 are styled as 4 identical boxes without clear risk/reward asymmetry.
- **Duplicated Guided Mode Content**: Lines 195–204 and lines 243–277 repeat the exact same text `sizing.cleanRoomRationale`.
- **Anti-Cyan Violations**: Line 98 renders confluence score in cyan; line 159 renders shares in cyan.

### 3.2 The -20% Anti-Slop Cut List
1. **Cut**: Line 287's duplicate "Copy Broker Order String" button.
2. **Cut**: Duplicated Governor sizing explanation in left panel.
3. **Consolidate**: Replace the 4 identical parameter boxes with an Asymmetric Execution Ticket Ladder.
4. **Prune**: Redundant setup summary cards above the ticket; replace with a sleek setup selector bar.

### 3.3 3-Tier Semantic Hierarchy Architecture
- **Level 0 (Decision — ≥ 40% Visual Weight)**:
  - **Asymmetric Execution Ticket & Order Ladder**:
    - Large high-contrast execution coordinates:
      - **Entry Pivot**: `$182.40` (Slate-100 font-mono)
      - **Stop Loss Floor**: `$176.10 (-3.45%)` (Rose-400 font-mono)
      - **Target 1**: `$195.00 (+2.06R)` (Emerald-400 font-mono)
      - **Target 2**: `$207.00 (+3.80R)` (Purple-400 font-mono)
    - **Governed Position Sizing Pill**: `13 Shares ($2,371 Capital)` with active Governor clamp indicator.
    - **Single Primary Action Button**:
      ```tsx
      <button onClick={handleCopyOrder} className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-mono font-bold rounded-lg ...">
        {copiedOrder ? "✓ ORDER COPIED TO CLIPBOARD" : "AUTHORIZE ORDER: 13 SHARES ($2,371) [COPY STRING]"}
      </button>
      ```
- **Level 1 (Rationale — 35% Visual Weight)**:
  - **Behavioral Governor Dynamic Risk Allocation**:
    - Visual risk breakdown: Unclamped Naive Risk ($100 / 17 Shs) vs Governed Risk ($75 / 13 Shs).
    - Explicit clamp factor: `-25% Drawdown Defense Clamp`.
    - Clean room rationale: *"Risk halved after 2 consecutive losses to preserve emotional capital."*
    - "Why Take This Trade?" 3-point checklist (Stage 2 verified, VCP base, R:R $\ge$ 1.85:1).
- **Level 2 (Proof — 25% Visual Weight)**:
  - **Quant Risk Diagnostics (Active in `QUANT` Mode)**:
    - 1,000-path Monte Carlo distribution graph.
    - Cornish-Fisher 95% VaR: `-$86.00`.
    - Sortino Skew: `+2.84` (positive upside asymmetry).
    - Half-Kelly Sizing: `0.25x`.
    - Risk of Ruin: `<0.05%`.

### 3.4 Metric Classification
- **Decision Metrics**: Recommended Shares (13), Capital Allocated ($2,371), Entry Price ($182.40), Stop Loss ($176.10), Target 1 ($195.00).
- **Risk Metrics**: Stop Distance (-3.45%), Dollar Risk ($75.00), Governor Clamp % (-25%), Cornish-Fisher VaR 95% (-$86).
- **Performance Metrics**: Reward-to-Risk Ratio (+2.06R TP1, +3.80R TP2), Expected Value (+$342.50).
- **Diagnostic Metrics**: Sortino Skew (+2.84), Half-Kelly multiplier (0.25x), Monte Carlo paths (1,000).

### 3.5 10-Second Decision Velocity Walkthrough
1. **0–3s**: The user sees the high-contrast Execution Ladder: Entry $182.40 | Stop $176.10 | Target $195.00.
2. **3–6s**: The user verifies Governed Sizing: 13 Shares ($2,371), clamped -25% for drawdown protection.
3. **6–10s**: The user clicks the single "Authorize Order" button to copy the order string directly into broker software.

---

## 4. Hub 3: `/portfolio` ("What Can Hurt Me?")

### 4.1 Current Anti-Patterns & Deficiencies
- **Textbook 4-Card Farm**: Lines 298–339 render Net Worth, Stock Holdings, Cash, and Unrealized P&L in a generic `grid-cols-4` row.
- **Global Monospace Abuse**: Line 245 `<main className="... font-mono ...">` forces all prose, labels, and table headers into monospace.
- **Retail Wallet Presets**: Lines 342–438 provide quick preset buttons for `$50`, `$100`, `$500`, evoking toy-like retail investing apps.
- **Missing Risk-First Heat Map**: There is no 2D position exposure visualizer or dedicated Exit Rule Triggers panel.

### 4.2 The -20% Anti-Slop Cut List
1. **Cut**: Generic 4-card KPI row (lines 298–339).
2. **Cut**: Retail wallet presets ($50, $100).
3. **Cut**: Global `font-mono` on `<main>`.
4. **Demote**: Move `MacroStressTestSimulator` into an on-demand collapsible drawer.

### 4.3 3-Tier Semantic Hierarchy Architecture
- **Level 0 (Decision — ≥ 40% Visual Weight)**:
  - **Asymmetric Capital at Risk Hero**:
    - Dominant primary metric: **Total Capital at Risk at Stop Floors**: `-$1,420 (5.68% of Equity)`.
    - Right rail utility cluster: Total Equity ($25,000), Stock Value ($18,420), Cash ($6,580), Unrealized P&L (+$1,240 / +5.2%).
    - **Active Exit Rule Triggers Alert Banner**:
      - `[EXIT TRIGGER: SEDG]` Stop Loss Floor Breached at $31.16 → Action: Liquidate position.
      - `[PROFIT TARGET: NVDA]` Target 1 Hit at $137.90 → Action: Sell 50%, trail stop to breakeven.
- **Level 1 (Rationale — 35% Visual Weight)**:
  - **2D Risk-First Portfolio Heat Map**:
    - Holdings displayed as tiles sized by portfolio weight and color-coded by Distance to Stop:
      - Deep Rose: $< 2\%$ to Stop (High Vulnerability).
      - Amber: $2\% - 5\%$ to Stop (Caution).
      - Muted Slate/Emerald: $> 5\%$ to Stop (Safe Buffer).
    - Sector Concentration Warning Bar: Flags if any sector exceeds 40% (e.g. Technology at 38.4%).
- **Level 2 (Proof — 25% Visual Weight)**:
  - **Open Quant Holdings Table (`DataLedgerTable`)**:
    - 8 dense columns: Asset, Shares, Cost Basis, Current Price, Market Value, Stop Floor, Unrealized P&L, Actions.
    - Tabular monospace numbers with right-alignment.
    - Collapsible Macro Stress Test Simulator (2008 GFC, 2020 COVID).

### 4.4 Metric Classification
- **Decision Metrics**: Total Capital at Risk at Stops ($1,420 / 5.68%), Active Exit Triggers Count (2), Largest Holding Weight (18.2%).
- **Risk Metrics**: Distance to Stop %, Sector Concentration %, Cash Reserves % (26.3%), Portfolio Beta (1.14).
- **Performance Metrics**: Total Unrealized P&L (+$1,240), Total Account Net Worth ($25,000).
- **Diagnostic Metrics**: Shares Count, Entry Price, Market Price, Exchange Source.

### 4.5 10-Second Decision Velocity Walkthrough
1. **0–3s**: The user immediately reads the Level 0 Hero: Total risk if all stops trigger is -$1,420 (5.68%).
2. **3–6s**: The user checks the Exit Rule Triggers banner and sees 1 active stop breach (`SEDG`).
3. **6–10s**: The user scans the 2D heat map, confirming position sizing is safe across all other holdings.

---

## 5. Hub 4: `/journal` ("Did I Follow My Rules?")

### 5.1 Current Anti-Patterns & Deficiencies
- **Severe Under-Implementation**: Only 98 lines of code; renders a generic 4-card KPI row followed by a static 4-row table.
- **Card Farm Anti-Pattern**: Lines 27–48 render Rule Adherence, Brier Calibration, Anti-Tilt, and Loss Containment in a generic `grid-cols-4` box grid.
- **Raw LaTeX Syntax Defect**: Line 36 renders unformatted LaTeX `Target $\le 0.25$`.
- **Missing Core Visualizations**: Lacks the Brier calibration curve and anti-tilt pattern grid mandated by Requirement R3.

### 5.2 The -20% Anti-Slop Cut List
1. **Cut**: Generic 4-card KPI row (lines 27–48).
2. **Cut**: Raw unrendered LaTeX string `$\le 0.25$`.
3. **Consolidate**: Transform static numbers into an Asymmetric Discipline Hero + Brier Calibration Curve.

### 5.3 3-Tier Semantic Hierarchy Architecture
- **Level 0 (Decision — ≥ 40% Visual Weight)**:
  - **Asymmetric Discipline Status Hero**:
    - Dominant metric: `94.2% Discipline Adherence Score (Grade A)`.
    - Live behavioral state pill: `CALM & OBJECTIVE · 0 Active Losses · Full Sizing Authorized`.
    - Operational directive: *"Discipline intact. You are operating within verified statistical edge."*
- **Level 1 (Rationale — 35% Visual Weight)**:
  - **Brier Probabilistic Calibration Curve**:
    - Visual chart comparing Subjective Conviction Probability (50%–90%) vs Empirical Win Rate.
    - Displays Brier score `0.18 ≤ 0.25` (Certified Well-Calibrated).
  - **4-Quadrant Anti-Tilt Behavioral Matrix**:
    - Loss Streak: `0 Active Losses` (Clamp Threshold: 2 losses).
    - Time-of-Day Discipline: `100% Adherence` (Zero trades in afternoon chop).
    - Loss Containment: `100% Contained ≤ 1.0R` (Zero stop-loss blowouts).
    - Revenge Trading Risk: `Nominal` (Zero immediate re-entries).
- **Level 2 (Proof — 25% Visual Weight)**:
  - **Execution Discipline Ledger**:
    - Comprehensive audit table of recent executions with Trade ID, Date, Ticker, Setup Archetype, Planned vs Actual R-Multiple, Rule Verification Stamp (`VERIFIED` vs `VIOLATION`), and P&L.

### 5.4 Metric Classification
- **Decision Metrics**: Discipline Adherence Score (94.2%), Behavioral Governance State (`CALM`), Sizing Authorization %.
- **Risk Metrics**: Brier Calibration Score (0.18), Active Loss Streak (0), Max Loss R-Multiple Containment ($\le 1.0R$).
- **Performance Metrics**: Realized R-Multiple (+1.82R average), Win Rate on Planned Setups (62.5%), Realized P&L (+$4,820).
- **Diagnostic Metrics**: Trade ID, Execution Timestamp, Conviction Probability.

### 5.5 30-Second Decision Velocity Walkthrough
1. **0–10s**: The user reads the Level 0 Hero: 94.2% discipline adherence, status CALM, zero active loss streak.
2. **10–20s**: The user reviews the Brier calibration curve and confirms their subjective conviction is mathematically calibrated ($0.18 \le 0.25$).
3. **20–30s**: The user scans the 4-quadrant anti-tilt matrix to verify zero behavioral leaks.

---

## 6. Hub 5: `/performance` ("Is ARX Helping Me?")

### 6.1 Current Anti-Patterns & Deficiencies
- **Fragmented Milestone Cards**: Lines 252–261 render 6 boxed mini-milestone cards in `grid-cols-6` that clutter the attribution tab.
- **Nested 4-Card Boxes**: Lines 126–148 render Max Drawdown, Sharpe, Profit Factor, Risk of Ruin in 4 equal boxes inside the hero.
- **Information Splitting**: 3 separate tabs hide critical proof points from the initial 30-second view.

### 6.2 The -20% Anti-Slop Cut List
1. **Cut**: The 6 tiny milestone cards (lines 252–261).
2. **Consolidate**: Replace fragmented cards with a continuous visual Counterfactual Equity Curve.
3. **Streamline**: Make the 3 tabs secondary drill-downs while keeping core proof metrics visible on the main canvas.

### 6.3 3-Tier Semantic Hierarchy Architecture
- **Level 0 (Decision — ≥ 40% Visual Weight)**:
  - **Asymmetric Proof of Edge Hero**:
    - Dominant primary headline: `+$6,140 Capital Preserved` (Verified Governor Alpha).
    - Drawdown Mitigation Delta: `Max Drawdown: -8.4% with ARX vs -19.2% Naive Unconstrained` (`+10.8% Protected`).
    - Core attribution statement: *"ARX saved +$6,140 across 31 interventions by clamping sizing during drawdown streaks, late-day chop, and capital floor tests."*
- **Level 1 (Rationale — 35% Visual Weight)**:
  - **Counterfactual Equity Trajectory Chart**:
    - Dual visual equity curve: Governed Equity ($64,800) vs Unconstrained Equity ($58,200).
    - Shaded green alpha region illustrating preserved capital.
  - **3 Defense Category Breakdown**:
    1. Drawdown Defense: `+$3,420` saved (halving risk after 2 losses).
    2. Execution Window: `+$1,850` saved (dampening risk during afternoon fatigue).
    3. Capital Floor: `+$870` saved (shielding cash runway).
  - Key Comparative Ratios: Sharpe 2.41 vs 1.62 | Profit Factor 2.14 vs 1.52 | Risk of Ruin <0.1% vs 4.2%.
- **Level 2 (Proof — 25% Visual Weight)**:
  - **Audited Governor Ledger (31 Verified Trades)**:
    - Complete trade-by-trade audit table with ID, Timestamp, Ticker, Setup, Unclamped Risk, Governed Risk, Clamp %, Outcome, and Capital Preserved $.
    - Data Provenance Toggle (`INV-OI119-P`): Benchmark Mode (31 verified trades) vs Live Trader Account.

### 6.4 Metric Classification
- **Decision Metrics**: Capital Preserved Total (+$6,140), Drawdown Delta (+10.8% protection), Net Alpha Spread (+$6,600).
- **Risk Metrics**: Governed Max Drawdown (-8.4%) vs Unclamped (-19.2%), Risk of Ruin (<0.1% vs 4.2%).
- **Performance Metrics**: Governed Sharpe (2.41 vs 1.62), Profit Factor (2.14 vs 1.52), Win Rate (58.3% vs 51.2%).
- **Diagnostic Metrics**: Governor Interventions Count (31), Category dollar splits, Trade clamp logs.

### 6.5 30-Second Decision Velocity Walkthrough
1. **0–10s**: The user reads the hero: +$6,140 in verified capital preserved; drawdown cut from -19.2% to -8.4%.
2. **10–20s**: The user inspects the counterfactual equity curve and the 3 defense pillars to understand where alpha came from.
3. **20–30s**: The user spot-checks the 31-trade audit ledger to verify exact dollar calculations.

---

## 7. Hub 6: `/research` ("Why Does This Work?")

### 7.1 Current Anti-Patterns & Deficiencies
- **Superficial Stub**: Only 76 lines long, rendering 3 static placeholder cards for GOOGL, NVDA, ANET.
- **Missing Catalyst Hierarchy**: No ranked stream of upcoming catalysts or filing dates.
- **Missing Smart Money Forensics**: Lacks the institutional 13F whale cluster analysis and Congressional STOCK Act breakdown specified in R3.

### 7.2 The -20% Anti-Slop Cut List
1. **Cut**: Generic 3-card card farm (lines 31–70).
2. **Cut**: Isolated single-line catalyst strings without conviction or timing metadata.
3. **Replace**: Build a full-scale institutional research workstation.

### 7.3 3-Tier Semantic Hierarchy Architecture
- **Level 0 (Decision — ≥ 40% Visual Weight)**:
  - **Asymmetric Research Dossier Hero**:
    - High-conviction featured asset: `GOOGL`.
    - Core Investment Thesis: High-density narrative explaining cloud margin expansion, search monetization durability, and institutional accumulation.
    - Catalyst Window: `Q3 Cloud Margin Expansion · 2–4 Weeks · Tier 1 High Conviction`.
    - Smart Money Alignment Badge: `Whale Accumulation + Congressional Inflow + Insider Buying Floor`.
- **Level 1 (Rationale — 35% Visual Weight)**:
  - **Ranked Catalyst Priority Stream**:
    - Prioritized feed of active catalysts across the universe (SEC Form 4 cluster buys, earnings revisions, product milestones).
    - Impact weighting badges (`HIGH IMPACT: Form 4 Director Purchase $1.2M at $178 Floor`).
  - **13F Whale Clusters & Congressional Disclosures**:
    - 13F Institutional Inflows: Duquesne Capital (+$45M), Renaissance Technologies, Berkshire Hathaway.
    - Congressional STOCK Act Forensics: Commerce & Science Committee members accumulating.
- **Level 2 (Proof — 25% Visual Weight)**:
  - **Deep-Dive Fundamental Forensics**:
    - Return on Invested Capital (ROIC: 31.4% vs industry 14.2%).
    - Balance Sheet Armor: Net Debt / EBITDA (0.2x Net Cash), FCF Yield (5.2%).
    - Greenblatt Magic Formula Decile & Peter Lynch PEG Ratio.
    - Direct verified links to SEC EDGAR filings and source disclosures.

### 7.4 Metric Classification
- **Decision Metrics**: Catalyst Impact Score (92/100), Conviction Tier (`TIER 1`), Overall Thesis Verdict.
- **Risk Metrics**: Net Debt / EBITDA (0.2x), FCF Burn / Runway, Filing Lag Days (STOCK Act).
- **Performance Metrics**: ROIC (31.4%), FCF Yield (5.2%), 3-Year Operating Margin Trend (+420 bps).
- **Diagnostic Metrics**: 13F Net Institutional Change, SEC Form 4 Transaction Value, Committee Jurisdiction Match.

### 7.5 30-Second Decision Velocity Walkthrough
1. **0–10s**: The user reads the Level 0 Research Dossier Hero and understands the core thesis and timing window for `GOOGL`.
2. **10–20s**: The user checks the 13F whale clusters and Congressional filings to confirm institutional backing.
3. **20–30s**: The user reviews the fundamental forensics (ROIC 31.4%, net cash) and verifies the SEC EDGAR source link.
