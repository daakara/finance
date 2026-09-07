# ARX Terminal Product Requirements Document (PRD)

**Product**: ARX Terminal  
**Version**: `vNext` Institutional Decision Intelligence Workstation  
**Document ID**: `PRD-ARX-DECISION-WORKSTATION-VNEXT`  
**Status**: `APPROVED_FOR_IMPLEMENTATION`  
**Owner**: Product Management, Quantitative Engineering, UI/UX Architecture  
**Target Milestone**: Phase 1 Modernization (Sprint 1: 65/35 Viewport Anchoring)  

---

## 1. Executive Summary

ARX Terminal is evolving from a feature-dense quantitative dashboard into an **Institutional Decision Intelligence Workstation**. 

The fundamental goal is not to reduce analytical depth, but to compress the time and cognitive friction required to reach an informed, defensible investment conviction.

### 1.1. Product Vision Statement
> **"ARX Terminal is an Institutional Decision Intelligence Workstation that reduces time-to-conviction by transforming raw market data, Bayesian quantitative signals, and governance-aware research into structured, explainable, and continuously monitored investment decisions."**

### 1.2. Core Product Principles
1. **Preserve Quantitative Rigor**: Never dumb down mathematical models, Bayesian probabilities, or volatility geometry.
2. **Answers Precede Evidence**: Show conclusions and actionability before unfolding deep mathematical proofs.
3. **Layered Complexity, Not Deleted Complexity**: Progressive disclosure isolates depth without removing it.
4. **Ergonomic Velocity**: Every visual element must directly accelerate conviction formation.
5. **Decoupled Governance**: The presentation layer reflects server-authoritative models without mutating state or invariants.

---

## 2. Problem Statement

### Current State
Users are currently presented with:
- Multi-timeframe charts
- 4-stream confluence scores
- Model governance warnings
- Historical validation indicators
- Smart money flows
- Macroeconomic indices
- Execution trade corridors
- Factor radars and congressional trade trackers

All simultaneously rendered on a single scrolling page with equalized visual weight (identical dark cards, borders, and typography).

### Root Consequences
- **High Cognitive Load**: Attention fragmentation across 15+ competing modules.
- **Visual Weight Collapse**: Crucial signals (Setup Score, Buy Corridor) compete with secondary diagnostics.
- **Buried Price Anchors**: The interactive price chart appears halfway down the page.
- **Workflow Friction**: Professional traders and analysts must spend valuable time scanning for information rather than evaluating thesis validity.

### Desired State
Users encounter information strictly sequenced by the institutional decision-making lifecycle:

$$\text{Orientation} \longrightarrow \text{Market Structure} \longrightarrow \text{Conviction} \longrightarrow \text{Explanation} \longrightarrow \text{Deep Validation} \longrightarrow \text{Change Intelligence}$$

---

## 3. Strategic Positioning

### 3.1. Category Definition
- **What ARX is NOT**: A retail brokerage app (Robinhood, eToro), a social trading hub, or a simplistic fundamental screener (FinViz).
- **What ARX IS**: An **Institutional Decision Intelligence Workstation** (analogous to Koyfin, TrendSpider, and FactSet, powered by proprietary Bayesian quantitative engines).

### 3.2. Competitive Value Matrix
| Platform Category | Representative Tools | Primary Value Proposition | Weakness Solved by ARX |
| :--- | :--- | :--- | :--- |
| **Retail Brokerage** | Robinhood, Trading212 | Order execution velocity | Zero analytical depth; gamified |
| **Legacy Terminal** | Bloomberg, FactSet | Unbounded data breadth | Prohibitive cognitive drag; $25k/yr cost |
| **Modern Charting** | TradingView, TrendSpider | Technical indicators | Disconnected from fundamentals & macro |
| **ARX Terminal** | **ARX vNext** | **Compressed Time-to-Conviction (TTC)** | **Synthesizes technical, fundamental, and regime confluence into immediate actionability** |

---

## 4. User Personas & Journey Topology

### Persona 1: Sarah Chen — Tactical Active Trader (Primary)
- **Profile**: Independent swing/momentum trader. Daily active frequency.
- **Decision Window**: Seconds to Minutes.
- **Core Needs**: Instant actionability assessment, entry corridor proximity, risk/reward ratio, and hard stop floor.
- **Target TTC (Time-to-Conviction)**: **$< 10\text{ seconds}$**.
- **Primary Workspace**: Standard Mode (Chart + Execution Corridor).

### Persona 2: Michael Roberts — Fundamental Investor (Primary)
- **Profile**: Private long-term allocator. Holding period: Months to Years.
- **Decision Window**: Hours to Days.
- **Core Needs**: Company health, capital accumulation, institutional sponsorship, and regime alignment.
- **Target TTC**: **$< 60\text{ seconds}$**.
- **Primary Workspace**: Guided / Standard Mode (Conviction Matrix + Factor Breakdown).

### Persona 3: Jennifer Park — Wealth Advisor (Primary)
- **Profile**: Fiduciary wealth manager managing client portfolios.
- **Decision Window**: Pre-client meetings & investment committees.
- **Core Needs**: Defensible investment rationale, client-ready explanations, and compliance-safe one-pagers.
- **Target TTC**: **$< 3\text{ minutes}$**.
- **Primary Workspace**: Guided Mode + "Export Institutional Due Diligence Brief".

### Persona 4: David — Quantitative Research Analyst (Primary)
- **Profile**: Professional quant researcher auditing signal edge.
- **Decision Window**: In-depth analytical sessions.
- **Core Needs**: Transparent factor attribution, SEC Form 4 insider distributions, regression intercepts, and model governance audit trails.
- **Target TTC**: Unconstrained depth without UI friction.
- **Primary Workspace**: Quant Mode (All modules expanded; zero hand-holding).

### Persona 5: Robert — CIO / Investment Committee Member (Primary Enterprise)
- **Profile**: Portfolio Manager / Chief Investment Officer allocating institutional capital ($10M–$500M AUM).
- **Decision Window**: Weekly allocation reviews & quarterly rebalancing.
- **Core Questions**: *"Why do we own this? What changed since last committee? Can this thesis survive board scrutiny?"*
- **Target TTC**: **$< 5\text{ minutes}$**.
- **Primary Workspace**: Stage 3 (Conviction Summary) + Stage 6 (Change Intelligence Delta Engine).

---

## 5. Information Architecture & The 6-Stage Conviction Lifecycle

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: ORIENTATION (Header Strip — 120px)                                                            │
│ Ticker · Spot Price · Setup Score · Real-Time Execution State · 20D ADV Liquidity Grade                │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 2: MARKET UNDERSTANDING (65/35 Primary Workspace — 650px)                                        │
│ [ 65% Price Chart: Candlesticks + ATR Bands ]    │   [ 35% Execution Corridor: Entry, Stop, T1, R/R ]  │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 3: CONVICTION MATRIX (Compact Status Bar — 120px)                                                │
│ Company Health ●  │  Smart Money ●  │  Market Regime ●  │  Structure ●  │  Validation Depth ●          │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 4: EXPLANATION LAYER (Deterministic Mathematical Synthesis)                                      │
│ "Why ARX Reached This Conclusion": 3 Quant Drivers  │  [ Export Due Diligence Brief (PDF) ↗ ]          │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 5: DEEP RESEARCH & AUDIT LAYER (Progressive Accordions / Quant Mode)                             │
│ [▶ Confluence Breakdown]  [▶ SEC Form 4 Insiders]  [▶ Macro (FRED)]  [▶ Model Governance]              │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 6: CHANGE INTELLIGENCE ENGINE (Continuous Monitoring)                                            │
│ State-Delta Ledger: What changed since last review? (Setup shift, volume spike, regime transition)     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.1. Stage 1: Orientation
- **Visual Weight**: 8 / 10.
- **Contents**: Ticker, Company Name, Spot Price, 24h Change (%), Setup Score (0–100), Execution State (`IN_BUY_ZONE`, `APPROACHING_TARGET`, `WAITING_PULLBACK`, `STOPPED_OUT`), and Liquidity Grade (`HIGH`, `MODERATE`, `RISK`, `UNKNOWN`).

### 5.2. Stage 2: Market Understanding (Above the Fold)
- **Visual Weight**: 10 / 10 (Primary Anchor).
- **Layout**: 65% Chart / 35% Execution Corridor grid.
- **Left Panel (65%)**: Interactive TradingView candlestick chart with ATR volatility bands, dynamic volume profiles, and support/resistance corridors.
- **Right Panel (35%)**: `OptimalEntryExitCard` showing Entry Zone, Stop Loss floor, Take Profit 1 & 2 targets, calculated Risk/Reward ratio, and Order Participation rate advisory (< 1.0% ADV).

### 5.3. Stage 3: Conviction Matrix
- **Visual Weight**: 7 / 10.
- **Layout**: 5-pill compact horizontal status strip.
- **Dimensions**:
  1. *Company Health* (Strong / Stable / Deteriorating)
  2. *Smart Money Flow* (Accumulation / Neutral / Distribution)
  3. *Market Regime* (Bull Supportive / Neutral / Bear Defensive)
  4. *Technical Structure* (Stage 2 Advancing / Basing / Stage 4 Decline)
  5. *Validation Depth* (Verified Historical / Moderate / Limited History)

### 5.4. Stage 4: Explanation Layer ("Why ARX Thinks This")
- **Visual Weight**: 8 / 10.
- **Function**: Translates Bayesian quantitative matrices into 3 human-readable, deterministic bullet points.
- **Action Buttons**: `View Confluence Breakdown ↗` (opens trace modal) and `Export Due Diligence Brief` (generates 1-page institutional PDF).

### 5.5. Stage 5: Deep Research & Audit Layer
- **Visual Weight**: 5 / 10 (Folded by default in Guided/Standard; Expanded in Quant).
- **Accordion Components**:
  - *Accordion 1*: 4-Stream Bayesian Confluence Weighting
  - *Accordion 2*: SEC Form 4 Insider Tracking & Congressional Filings
  - *Accordion 3*: FRED Macroeconomic Regime & Yield Curve Spreads
  - *Accordion 4*: Asset Factor Radar (Quality, Value, Growth, Momentum)
  - *Accordion 5*: Phase 25/26 Governance & Dual Cryptographic Hashes

### 5.6. Stage 6: Change Intelligence Layer
- **Visual Weight**: High Contextual (Renders on repeat visits).
- **Function**: Snapshot diff engine comparing the asset's current state against the user's last acknowledged review.
- **Sample Output**:
  ```text
  CHANGES SINCE LAST REVIEW (2026-09-01):
  ▲ Setup Score: 63 → 71 [Entered Buy Zone]
  ▲ Institutional Volume: +2.4σ accumulation detected
  ▬ Market Regime: Supportive (BULL)
  ▼ Validation Depth: Still thin (3/20 required forward sessions observed)
  ```

---

## 6. Experience Modes Architecture

ARX provides three specialized operational modes governed by the global `[Guided | Standard | Quant]` segmented controller:

| Mode | Target User | Visible Stages | Accordion Behavior | Design Philosophy |
| :--- | :--- | :---: | :---: | :--- |
| **Guided** | Jennifer (Advisor), Michael (Investor) | Stages 1, 2, 3, 4 | Deep diagnostics hidden | **Narrative-First**: Answers and investment thesis precede technical complexity. |
| **Standard** | Sarah (Trader), Active Operators | Stages 1, 2, 3, 4, 6 | Accordions collapsed (on-demand) | **Decision Ergonomics**: Chart, execution corridor, and conviction status optimized for speed. |
| **Quant** | David (Analyst), Quants | All Stages (1–6) | Accordions **expanded by default** | **Total Transparency**: Maximum data density; full institutional proof visible without clicks. |

---

## 7. Functional Requirements

### 7.1. Scope Fence (What vNext Will NOT Do)
- **WON'T Execute Live Trades**: No direct broker routing or automated execution. ARX is strictly a decision intelligence engine.
- **WON'T Issue Retail Fiduciary Advice**: No retail "BUY NOW" or "SELL" commands. All states are mathematical classifications.
- **WON'T Build Custom Chart Canvas from Scratch**: Leverages lightweight TradingView canvas overlays.
- **WON'T Mutate Model Invariants**: The UI layer will never recalculate server-side model scores, corridors, or liquidity metrics.
- **WON'T Clutter Mobile Viewports**: Mobile will strictly support Stages 1–4; deep Stage 5 accordions will be deferred to desktop.

### 7.2. Core Functional Requirements & Acceptance Criteria

#### FR-1: Global Experience Mode Switcher (P0)
- **AC-1.1**: The mode switcher must appear as a prominent segmented control (`Guided | Standard | Quant`) in the global navbar.
- **AC-1.2**: Mode selection must sync bidirectionally with the URL query parameter (`?mode=standard`) and persist in `localStorage`.
- **AC-1.3**: Switching modes must reconfigure viewport layout without triggering a full page reload or chart re-mount.

#### FR-2: Collapsible Slide-Over Watchlist (P0)
- **AC-2.1**: The watchlist sidebar must be collapsed by default on initial visit, granting 100% width to the primary workspace.
- **AC-2.2**: The watchlist must toggle via keyboard shortcut (`[` or `Ctrl+B`) or floating edge button.
- **AC-2.3**: Watchlist open/closed state must persist in `localStorage`.

#### FR-3: 65 / 35 Chart & Execution Workspace (P0)
- **AC-3.1**: Price Chart and Execution Corridor must render side-by-side above the fold on viewports $\ge 1024\text{px}$.
- **AC-3.2**: On tablet/mobile ($< 1024\text{px}$), the layout must stack vertically (Chart full-width $\to$ Execution Corridor full-width).
- **AC-3.3**: The Execution Corridor must dynamically display the real-time state badge (`IN_BUY_ZONE`, `WAITING_PULLBACK`, etc.) and the 1.0% ADV participation heuristic pill.

#### FR-4: Compact Conviction Matrix (P1)
- **AC-4.1**: Render a 5-column horizontal status strip (`Health`, `Money`, `Regime`, `Structure`, `Validation`) directly beneath Stage 2.
- **AC-4.2**: Each pill must display an interactive tooltip explaining the quantitative input on hover.

#### FR-5: Deterministic "Why ARX Thinks This" Layer (P1)
- **AC-5.1**: Render the top 3 drivers generated deterministically by `insightGenerator.ts`.
- **AC-5.2**: Provide a prominent CTA to inspect the full mathematical Bayesian confluence trace.

#### FR-6: Contextual Institutional Tooltips (`ⓘ`) (P1)
- **AC-6.1**: All technical terms (`Domain Confidence`, `Amihud ILLIQ`, `Bayesian Confluence`) must include an `ⓘ` icon that triggers an explanatory popover explaining *what it is* and *how to interpret high vs. low*.

#### FR-7: Lazy-Hydrated Research Accordions (P1)
- **AC-7.1**: Secondary modules (Form 4, FRED macro, factor radar) must be wrapped in collapsible accordions, collapsed by default in Guided and Standard modes.
- **AC-7.2**: Accordion content must use Next.js lazy-loading/dynamic imports so unexpanded data is not fetched during initial SSR.

#### FR-8: Institutional Due Diligence Brief Export (P2)
- **AC-8.1**: Provide an "Export Due Diligence Brief" button in Stage 4 that renders a clean, printable 1-page PDF summarizing Ticker, Price, Setup Score, 5 Conviction Pills, 3 Drivers, Execution Levels, and FINRA disclaimer.

#### FR-9: Stage 6 Change Intelligence Engine (P2)
- **AC-9.1**: When a user inspects a ticker previously visited, compute the delta between the current state and the last acknowledged snapshot.
- **AC-9.2**: Render a prominent Delta Banner highlighting score shifts ($\Delta \ge \pm 3$), execution state changes, or volume spikes.

### 7.3. Edge Cases & Resilience Behaviors
- **EC-1: Asset with Thin / Missing Historical Data (< 20 sessions)**:
  - Display `UNKNOWN_LIQUIDITY` with neutral slate styling.
  - Setup score renders with `Domain Confidence: Limited` banner.
  - Position sizing modal displays clear cautionary guidance.
- **EC-2: Non-Trading Sessions (Weekends / Market Holidays)**:
  - Header displays pinned settlement banner: `[Session Closed / Friday Settlement Pinned]`.
  - Change Intelligence engine does not flag weekend gaps as stale data defects.
- **EC-3: Offline / SQLite Store Fallback**:
  - `DataSourceBadge` displays amber indicator (`Cached Store`).
  - All frozen decision parameters remain accessible; charts display last verified settlement bar.

---

## 8. Non-Functional Requirements & Compliance

### 8.1. Performance & Loading SLAs
- **Initial Load Time**: $< 2.0\text{ seconds}$ on standard broadband.
- **Largest Contentful Paint (LCP)**: $< 2.5\text{ seconds}$.
- **Cumulative Layout Shift (CLS)**: $< 0.05$ (Chart and execution containers must have fixed min-heights to prevent layout jumping during hydration).
- **Mode Switch Latency**: $< 100\text{ms}$ client-side transition.

### 8.2. Security, Privacy & Fiduciary Compliance
- **Fiduciary Disclaimer**: All views and PDF exports must include binding compliance text:
  > *"ARX Terminal provides quantitative decision intelligence and mathematical modeling for informational and research purposes only. It does not provide personalized investment, tax, or legal advice."*
- **Client Storage Hygiene**: User watchlists, portfolio allocations, and custom mode preferences stored in `localStorage` must be strictly local and never leaked to external telemetry.

---

## 9. Success Metrics & North Star KPI

### 9.1. North Star Metric: Time-to-Conviction (TTC)
> **Definition**: *The elapsed time from ticker search / page load to an informed, defensible capital allocation decision.*

| Persona | Baseline TTC | Target vNext TTC | Reduction Target |
| :--- | :---: | :---: | :---: |
| **Tactical Trader (Sarah)** | $45\text{s}$ | **$< 10\text{s}$** | **$78\%$ faster** |
| **Fundamental Investor (Michael)** | $120\text{s}$ | **$< 60\text{s}$** | **$50\%$ faster** |
| **Financial Advisor (Jennifer)** | $8\text{ min}$ | **$< 3\text{ min}$** | **$62\%$ faster** |
| **CIO / Committee (Robert)** | $20\text{ min}$ | **$< 5\text{ min}$** | **$75\%$ faster** |

### 9.2. Secondary Product Metrics
- **Cognitive Load Reduction (NASA-TLX)**: $\ge 30\%$ decrease in reported cognitive effort.
- **System Usability Scale (SUS)**: Target score $\ge 80 / 100$.
- **Stage 6 Retention / Revisit Velocity**: $\ge 35\%$ increase in weekly return visits via Change Intelligence alerts.
- **Advisor Export Rate**: $\ge 15\%$ of Stage 4 visits trigger a Due Diligence Brief download.

---

## 10. Prioritized Implementation Roadmap

```
PHASE 1: Immediate Visible ROI & Core Ergonomics (P0 — Sprint 1–2)
  ├── 1. Market Command Ribbon: Persistent 36px macro regime strip (SPY, QQQ, VIX, 10Y, Regime).
  ├── 2. Discovery Workspace: Zero-state landing with curated institutional strategy baskets.
  ├── 3. 65/35 Primary Workspace: Side-by-side Chart and Execution Corridor above the fold.
  ├── 4. Watchlist Slide-Over Drawer: Collapsible sidebar (default collapsed; hotkey toggle).
  └── 5. Stage 3 Conviction Matrix: 5-pill compact horizontal status strip.

PHASE 2: Strategic Market Differentiators (P1 — Sprint 3–4)
  ├── 1. Stage 6 Change Intelligence Engine: Local snapshot ledger & automated Delta Banner.
  ├── 2. Institutional Due Diligence Brief Export: 1-page printable/PDF compliance summary.
  ├── 3. Stage 5 Research Accordions: Lazy-hydrated Radix UI accordions for deep auditability.
  └── 4. Multi-Tier Mode Specialization: [Guided | Standard | Quant] presentation densities.

PHASE 3: Enterprise Stickiness & Platform Moat (P2 — Sprint 5+)
  ├── 1. Workspace Personalization: Draggable viewport split (50/50 to 80/20) with 65/35 snap.
  ├── 2. Desk Presets: Saved configurations (Default Desk, Research Desk, Execution Focus).
  ├── 3. Deep-Link Layout URLs: Base64-encoded URL parameters for desk sharing.
  └── 4. Multi-Monitor Canvas Detach: Pop-out TradingView chart window with active telemetry.
```

---

## 11. Open Product Questions (OPQ)

### OPQ-1: Change Intelligence Baseline Trigger
- **Question**: Should the Stage 6 Change Intelligence engine compute deltas against **A) Last Visited Timestamp** or **B) Last Acknowledged Snapshot**?
- **Recommendation**: **Option B (Last Acknowledged Snapshot)**.
- **Rationale**: Institutional operators frequently open a tab briefly without completing a review. Acknowledged snapshots prevent missed alerts.

### OPQ-2: Scope of Quant Mode Datasets
- **Question**: Should Quant Mode remain strictly a visual density mode, or should it expose additional raw datasets?
- **Recommendation**: **Visual density mode only in vNext**. Additional raw datasets (e.g. raw tick feeds, custom Python regression sandboxes) are deferred to Phase 4+.

### OPQ-3: Export Formats for Advisory Due Diligence
- **Question**: Should the due diligence export support PowerPoint (`.pptx`) in addition to PDF?
- **Recommendation**: **PDF only in vNext**. PowerPoint export will be considered in future enterprise tiers.

---

## 12. Future Considerations (vNext+1)

1. **Portfolio-Level Conviction Aggregation**: Rolling up individual asset conviction scores into an aggregate portfolio-level health score.
2. **LP / Committee Presentation Mode**: Fullscreen, sanitized slideshow mode designed for projection in investment committee meetings.
3. **Multi-Asset Spread Corridors**: Extending the 65/35 execution geometry to pairs trading and sector rotation spreads.

---

*Certified as Authoritative Master PRD for ARX Terminal vNext.*  
*Approved by Antigravity Quantitative Architecture & Product Strategy.*
