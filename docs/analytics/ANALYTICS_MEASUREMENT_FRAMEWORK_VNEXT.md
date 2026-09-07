# ARX Terminal vNext: Analytics & Success Measurement Framework
## Telemetry Architecture, Event Taxonomy, and Time-to-Conviction (TTC) Specification

**Document ID**: `ANALYTICS-SPEC-ARX-WORKSTATION-VNEXT`  
**Version**: `1.0.0-PROD-SPEC`  
**Status**: `APPROVED_FOR_IMPLEMENTATION`  
**Classification**: Enterprise Product Analytics & Instrumentation Specification  
**Governing Documents**:  
- [`docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/prd/PRD_INSTITUTIONAL_DECISION_WORKSTATION_VNEXT.md)  
- [`docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md`](file:///c:/Users/akara/Documents/Projects/finance/docs/ux/UX_DESIGN_SPEC_INSTITUTIONAL_WORKSTATION.md)  

---

## 1. Executive Purpose & Measurement Thesis

### 1.1 The "Measurement-First" Imperative
The central value proposition of the ARX Terminal redesign is:

$$\textbf{Reduce Time-to-Conviction (TTC) while preserving institutional quantitative rigor.}$$

If TTC and decision efficiency are not instrumented with millisecond precision before frontend code is deployed, the product team cannot objectively prove that the workstation reduced cognitive load or compressed decision latency. 

This framework defines:
1. The **mathematical formulation of Time-to-Conviction (TTC)** across all user personas.
2. An exhaustive, client-side **Decision Event Taxonomy** capturing conviction formation.
3. The **Stage Completion Funnel** measuring progression across the 6-Stage Conviction Lifecycle.
4. The **Change Intelligence Retention Dashboard** tracking delta engagement and thesis revisit velocity.
5. Strict **Privacy and Fiduciary Invariants** ensuring zero portfolio or PII leakage.

---

## 2. North Star KPI: Time-to-Conviction (TTC)

### 2.1 Mathematical Definition
Time-to-Conviction is defined as the elapsed time between an asset session initiation and the moment the user registers an explicit or high-confidence capital decision action:

$$\text{TTC} = t_{\text{decision\_event}} - t_{\text{ticker\_opened}}$$

Where:
- $t_{\text{ticker\_opened}}$ is the monotonic high-resolution client timestamp (`performance.now()`) recorded when a ticker workspace successfully mounts and renders Stage 1.
- $t_{\text{decision\_event}}$ is the timestamp when a terminal conviction action is triggered (e.g. sizing a trade, exporting a committee brief, or marking the thesis approved/rejected).

```
   t_ticker_opened                                                        t_decision_event
          │                                                                      │
          ▼                                                                      ▼
  ┌───────────────┐      ┌───────────────┐      ┌───────────────┐        ┌───────────────┐
  │ Stage 1: Info │ ───► │ Stage 2: 65/35│ ───► │ Stage 3 & 4   │ ─────► │ Decision Action│
  └───────────────┘      └───────────────┘      └───────────────┘        └───────────────┘
          │◄──────────────────────── TIME-TO-CONVICTION (TTC) ──────────────────►│
```

### 2.2 Persona Benchmark Targets

| User Persona | Role & Mindset | Legacy Baseline | Target vNext TTC | Reduction Target |
| :--- | :--- | :---: | :---: | :---: |
| **Sarah Chen** | Tactical Active Trader | $45.0\text{s}$ | **$< 10.0\text{s}$** | **$-78\%$** |
| **Michael Roberts** | Fundamental Allocator | $120.0\text{s}$ | **$< 60.0\text{s}$** | **$-50\%$** |
| **Jennifer Park** | Wealth Advisor | $8.0\text{ min}$ | **$< 180.0\text{s}$ ($3\text{m}$)** | **$-62\%$** |
| **Robert Vance** | CIO / Investment Committee | $20.0\text{ min}$ | **$< 300.0\text{s}$ ($5\text{m}$)** | **$-75\%$** |

### 2.3 Diagnostic Lead Indicator: Time-to-First-Material-Insight (TTFMI)
While TTC measures the total elapsed time to terminal conviction, users must first absorb a material insight before forming an investment decision:

$$\text{TTFMI} \longrightarrow \text{Drives} \longrightarrow \text{TTC}$$

$$\text{TTFMI} = t_{\text{first\_material\_insight}} - t_{\text{ticker\_opened}}$$

Where $t_{\text{first\_material\_insight}}$ is recorded upon the user's first qualifying interaction with a high-density diagnostic component:
- Clicking or scrubbing a **Stage 6 Delta Banner item**
- Clicking an individual **Stage 3 Conviction Pill** (`HEALTH`, `FLOW`, `REGIME`)
- Expanding the **Stage 4 "Why ARX Thinks This"** mathematical driver card
- Hovering or inspecting levels in the **Stage 2 Optimal Execution Corridor**

#### TTFMI Benchmark Targets:
- **Trader (Sarah)**: **$< 4.0\text{s}$**
- **Investor (Michael)**: **$< 8.0\text{s}$**
- **Advisor (Jennifer)**: **$< 15.0\text{s}$**
- **Diagnostic Rule**: If $\text{TTFMI}$ is high, the viewport hierarchy is noisy. If $\text{TTFMI}$ is low but $\text{TTC}$ remains elevated, the explanation layer lacks decisiveness.

---

## 3. Decision Event Taxonomy

A **Decision Event** signals that the user has formed sufficient conviction to take action. Events are divided into *High-Confidence Implicit Actions* and *Explicit Workflow Actions*.

```
                                 DECISION EVENT TAXONOMY
                                            │
                  ┌─────────────────────────┴─────────────────────────┐
                  ▼                                                   ▼
     [ HIGH-CONFIDENCE IMPLICIT EVENTS ]             [ EXPLICIT WORKFLOW DECISIONS ]
     • position_sizer_opened                         • thesis_marked_approved
     • candidate_added_to_watchlist                  • thesis_marked_passed
     • due_diligence_pdf_exported                    • thesis_marked_investigate
     • invalidation_alert_created                    • position_allocation_committed
```

### 3.1 High-Confidence Implicit Decision Events
These actions are natural outputs of reaching an investment conclusion:
1. `position_sizer_opened`: Trader has validated the setup, entry corridor, and stop floor; opening the risk calculator is a direct execution intent signal.
2. `candidate_added_to_watchlist`: Investor or trader has confirmed positive confluence and flags the asset for execution timing.
3. `due_diligence_pdf_exported`: Advisor has completed review and is downloading the client-ready 1-page institutional brief.
4. `invalidation_alert_created`: Operator has pinned a price or regime alert at the mathematical stop floor.

### 3.2 Explicit Workflow Decision Events
vNext introduces explicit thesis triage buttons within Stage 4 (`Why ARX Thinks This`):
- `thesis_marked_approved`: Asset meets mandate criteria for capital deployment.
- `thesis_marked_passed`: Asset evaluated and formally rejected (records negative conviction).
- `thesis_marked_investigate`: Sent to junior analyst queue or flagged for committee review.

---

## 4. Core Telemetry Instrumentation & Event Schema

### 4.1 Client Telemetry Envelope
All events emitted by the frontend conform to a strict, typed schema:

```typescript
interface TelemetryEnvelope {
  eventId: string;                 // UUID v4
  timestamp: string;               // ISO 8601 UTC
  sessionElapsedMs: number;        // Monotonic ms since session start
  sessionId: string;               // Ephemeral session identifier
  workspaceMode: 'GUIDED' | 'STANDARD' | 'QUANT';
  activeWorkspace: 'DISCOVERY' | 'TICKER_DECISION';
  ticker?: string;                 // Symbol being evaluated (if applicable)
  assetClass?: 'EQUITY' | 'ETF' | 'CRYPTO';
  marketCapTier?: 'MEGA' | 'LARGE' | 'MID' | 'SMALL';
  eventCategory: 'DISCOVERY' | 'NAVIGATION' | 'STAGE_VIEW' | 'INTERACTION' | 'DECISION' | 'DELTA';
  eventName: string;
  payload: Record<string, unknown>;
}
```

### 4.2 Comprehensive Event Catalog

| Event Name | Category | Trigger Condition | Key Payload Parameters |
| :--- | :--- | :--- | :--- |
| `command_ribbon_viewed` | DISCOVERY | Market Command Ribbon mounts | `{ spxReturn, vixLevel, regimeState }` |
| `discovery_workspace_viewed`| DISCOVERY | User lands on `/` without active symbol | `{ strategyBasketCount, topScore }` |
| `strategy_basket_clicked` | DISCOVERY | Click on `Momentum`, `VCP`, or `Accumulation` | `{ basketType, candidateCount }` |
| `candidate_card_selected` | DISCOVERY | Click on candidate card to deep-dive | `{ ticker, setupScore, rankIndex }` |
| `watchlist_toggled` | NAVIGATION | Slide-over drawer opened or closed | `{ newState: 'OPEN' \| 'CLOSED', trigger: 'HOTKEY' \| 'BUTTON' }` |
| `mode_switched` | NAVIGATION | Global mode segmented control clicked | `{ previousMode, newMode, source: 'USER' \| 'URL' }` |
| `ticker_opened` | TICKER_DECISION | Ticker Workspace mounts (TTC Timer Start)| `{ ticker, spotPrice, setupScore, executionState }` |
| `price_chart_viewed` | STAGE_VIEW | Stage 2 Chart viewport visible for $\ge 1.0\text{s}$ | `{ timeframe: '1D' \| '1W', indicatorsActive: string[] }` |
| `corridor_interacted` | INTERACTION | Hover or click on execution corridor levels| `{ levelType: 'ENTRY' \| 'STOP' \| 'T1' \| 'T2' }` |
| `conviction_matrix_viewed` | STAGE_VIEW | Stage 3 Conviction Bar visible for $\ge 0.5\text{s}$ | `{ healthGrade, moneyFlow, regime, structure, validationDepth }` |
| `conviction_pill_hovered` | INTERACTION | Hover on individual conviction pill tooltip| `{ pillDimension: 'HEALTH' \| 'FLOW' \| 'REGIME' \| 'STRUCTURE' }` |
| `why_arx_viewed` | STAGE_VIEW | Stage 4 Explanation Card visible | `{ driverCount, topDriverType }` |
| `confluence_modal_opened` | INTERACTION | Click on `View Confluence Breakdown ↗` | `{ ticker, confluenceScore }` |
| `accordion_toggled` | INTERACTION | Click on Stage 5 Research accordion | `{ accordionId: 'FACTORS' \| 'SEC_FORM4' \| 'FRED_MACRO', newState: 'OPEN' \| 'CLOSED' }` |
| `delta_banner_viewed` | DELTA | Stage 6 Delta Banner rendered | `{ ticker, deltaScore, stateChanged: boolean, flowZDelta }` |
| `change_acknowledged` | DELTA | Click on `[ Acknowledge & Update Baseline ]` | `{ ticker, durationSinceLastVisitMs, deltaMagnitude }` |
| `due_diligence_exported` | DECISION | Click on `Export Due Diligence Brief (PDF)` | `{ ticker, exportFormat: 'PDF', durationSinceOpenMs }` |
| `position_sizer_opened` | DECISION | Click on `Size Position` modal | `{ ticker, suggestedADVShare, durationSinceOpenMs }` |
| `decision_marked` | DECISION | Click on `Approved`, `Passed`, or `Investigate` | `{ ticker, decisionType: 'APPROVED' \| 'PASSED' \| 'INVESTIGATE', ttcMs }` |

---

## 5. Time-to-Conviction (TTC) Analytics & Dimensional Cuts

To ensure granular operational visibility, TTC must be sliced across four essential dimensions:

```
                                  TTC DIMENSIONAL CUTS
                                            │
         ┌──────────────────┬───────────────┴───────────────┬──────────────────┐
         ▼                  ▼                               ▼                  ▼
  [ BY PERSONA ]     [ BY WORKSPACE MODE ]          [ BY ASSET CLASS ]   [ BY MARKET REGIME ]
  • Trader (<10s)    • Guided                       • Mega-Cap ($100B+)  • Risk-On
  • Investor (<60s)  • Standard                     • Mid/Small Cap      • Neutral / Chop
  • Advisor (<180s)  • Quant                        • Volatile Growth    • Defensive / Bear
  • CIO (<300s)
```

### 5.1 Metric 1: Median Time-to-Conviction ($TTC_{50}$)
The primary metric tracked across all executive reporting is the **Median TTC ($TTC_{50}$)** rather than the arithmetic mean, eliminating skew from idle browser tabs:

$$TTC_{50} = \text{Median}\left(\{ t_{\text{decision}} - t_{\text{open}} \mid \Delta t \le 1800\text{s} \}\right)$$

*(Sessions exceeding 30 minutes without interaction are flagged as abandoned and excluded from active TTC scoring).*

### 5.2 Metric 2: TTC Efficiency by Experience Mode
Hypothesis testing to validate mode-specific UX optimizations:
- **Guided Mode Target**: Does plain-English explanation accelerate non-technical investor conviction?
- **Standard Mode Target**: Does the 65/35 layout deliver $<10\text{s}$ conviction for active momentum operators?
- **Quant Mode Target**: Does zero-click expanded auditability reduce research time for institutional quants?

### 5.3 Metric 3: TTC by Market Capitalization Tier
- **Mega/Large Cap**: High liquidity, deep analyst coverage; expected faster conviction.
- **Small/Micro Cap**: Thinner validation history; tests whether the `UNKNOWN_LIQUIDITY` and cautionary states prompt appropriate due diligence without stall.

---

## 6. Stage Completion Funnel & Cognitive Drop-off Analysis

```
STAGE 1: ORIENTATION (Header Strip)
  │  (100% of ticker page loads)
  ▼
STAGE 2: MARKET UNDERSTANDING (65/35 Chart + Corridor)
  │  (Target: ≥ 92% scroll/view-through | Median Time: 4.2s)
  ▼
STAGE 3: CONVICTION MATRIX (5-Pill Status Strip)
  │  (Target: ≥ 78% view-through | Median Time: 2.1s)
  ▼
STAGE 4: EXPLANATION LAYER ("Why ARX Thinks This")
  │  (Target: ≥ 64% view-through | Median Time: 3.8s)
  ▼
STAGE 5: DEEP RESEARCH & AUDIT (Progressive Accordions)
  │  (Target: 25% expansion in Guided/Standard; 100% in Quant)
  ▼
CONVICTION ACTION (Trade Sized · Watchlist Added · PDF Brief Exported)
     (Target: ≥ 35% of all qualified ticker sessions reach a terminal decision)
```

### 6.1 Diagnostic Drop-Off Indicators
- **Stage 2 $\to$ Bounce Drop-off**: If users abandon before engaging with Stage 2, indicates chart loading latency or visual noise.
- **Stage 3 Hover Ratio**: Percentage of users who hover over the `ⓘ` contextual tooltips; measures metric interpretability.
- **Stage 5 Accordion Stall**: If an operator spends $> 3\text{ minutes}$ in Stage 5 without taking action, it flags that the explanation in Stage 4 was insufficiently clear or contradictory.

---

## 7. Change Intelligence Metrics (The Retention Engine)

Stage 6 is ARX's primary proprietary retention moat. Its success is measured by how effectively it eliminates the **Re-Read Tax**.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CHANGE INTELLIGENCE RETENTION DASHBOARD METRICS                                                        │
├────────────────────────────────┬──────────────────────────┬───────────────────────────────────────────┤
│ Metric Name                    │ Target SLA               │ Measurement Formula                       │
├────────────────────────────────┼──────────────────────────┼───────────────────────────────────────────┤
│ Delta Engagement Rate          │ $\ge 70.0\%$             │ $\frac{\text{Sessions with Delta Banner Interaction}}{\text{Total Returning Visits with Detected Delta}}$ │
├────────────────────────────────┼──────────────────────────┼───────────────────────────────────────────┤
│ Change-Acknowledgement Rate    │ $\ge 50.0\%$             │ $\frac{\text{Sessions Clicking 'Acknowledge Baseline'}}{\text{Total Delta Banners Rendered}}$            │
├────────────────────────────────┼──────────────────────────┼───────────────────────────────────────────┤
│ Time-to-Change Detection       │ $< 15.0\text{ seconds}$   │ $t_{\text{delta\_banner\_interaction}} - t_{\text{ticker\_opened}}$                                      │
├────────────────────────────────┼──────────────────────────┼───────────────────────────────────────────┤
│ Delta-Driven Session Ratio     │ $\ge 30.0\%$ (Phase 2)   │ $\frac{\text{User Sessions Initiated from Delta Alert}}{\text{Total Active User Sessions}}$              │
│                                │ $\ge 50.0\%$ (Phase 3)   │                                           │
└────────────────────────────────┴──────────────────────────┴───────────────────────────────────────────┘
```

---

## 8. Enterprise Advisor & Investment Committee Metrics

These metrics validate enterprise stickiness and executive buy-in:

### 8.1 Due Diligence Brief Export Volume
- **Target**: $\ge 15\%$ of financial advisor (Jennifer) and CIO (Robert) sessions generate a 1-page Due Diligence Brief download.
- **Export Distribution**: Track ratio of PDF generation to total Stage 4 impressions:
  $$\text{Export Ratio} = \frac{\text{Count}(\text{due\_diligence\_exported})}{\text{Count}(\text{why\_arx\_viewed})}$$

### 8.2 Investment Committee Brief View-Through
- **Committee Engagement Index**: Track shared URL opens and multi-seat reviews of candidate briefs ahead of weekly allocation meetings.
- **Revisit Velocity Before Committee**: Number of times an approved ticker is reviewed in the 48 hours preceding regular Monday morning investment committee meetings.

---

## 9. Privacy, Fiduciary Compliance & Telemetry Invariants

> [!IMPORTANT]
> **Strict Non-PII & Financial Privacy Guarantees**
> 1. **Zero Financial Data Exfiltration**: Position size values, dollar allocations, cash reserves, and stop prices configured in the Position Sizer remain strictly local (`localStorage`). Telemetry envelopes record **ONLY** the execution of the sizing action, never the monetary quantity.
> 2. **No Watchlist or Portfolio Scraping**: Tickers contained within a user's personal portfolio or watchlist are never transmitted to external analytics providers.
> 3. **Client-Side Aggregation**: All time-to-conviction timers run client-side using the monotonic `window.performance.now()` API, preventing network jitter from corrupting timing calculations.
> 4. **Anonymized Machine Identifiers**: Sessions use cryptographic UUID v4 tokens with zero linkage to IP addresses, emails, or personal identification.

---

## 10. The Executive KPI Dashboard

The product leadership dashboard will display **five real-time indices**:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ARX TERMINAL vNEXT: EXECUTIVE PRODUCT COCKPIT                                                          │
├──────────────────────┬──────────────────────┬──────────────────────┬──────────────────┬────────────────┤
│ 1. DECISION VELOCITY │ 2. PLATFORM ADOPTION │ 3. CONVICTION FORM.  │ 4. RETENTION     │ 5. ENTERPRISE  │
├──────────────────────┼──────────────────────┼──────────────────────┼──────────────────┼────────────────┤
│ Median TTC (Trader)  │ Daily Active Users   │ Decision Completion  │ Delta Banner     │ Due Diligence  │
│   7.4s (Target: <10s)│   +42% MoM           │ Rate                 │ Usage            │ PDF Exports    │
│                      │                      │   41.2% (Target >35%)│   74.8% (SLA >70)│   18.4% of Desk│
│ Median TTC (Invest.) │ Weekly Revisit Rate  │                      │                  │                │
│   48.2s (Target <60s)│   68.5%              │ False Conviction /   │ Acknowledge Rate │ Shared Desk    │
│                      │                      │ Invalidation Rate    │   56.1% (SLA >50)│ URL Links      │
│ TTFMI                │ Discovery Landing    │   < 8.5%             │                  │   1,420/month  │
│   5.8s (Target: <8s) │ Conversion: 38.2%    │                      │                  │                │
└──────────────────────┴──────────────────────┴──────────────────────┴──────────────────┴────────────────┘
```

---

*Certified as Authoritative Analytics & Success Measurement Specification for ARX Terminal vNext.*  
*Antigravity Principal Product Management & Quantitative Systems Architecture.*
