# Semantic-Zoom Coverage Matrix
**Platform**: ARX Unified Intelligence Operating System  
**Roadmap Section**: H14.2 (Semantic Zoom), H15.1 (Four Core Hubs), & H17.1 (Trading Workstation Hubs)  
**Date**: September 10, 2026  
**Status**: Verified Complete (Architectural & Component Integrity)  

---

## 1. Architectural Principles of Semantic Zoom

Semantic Zoom in ARX replaces traditional multi-page navigation hops with progressive in-place contextual disclosure:
- **Level 0 / Level 1 (Executive 30-Second View)**: Ultra-clean, low-cognitive-load executive summary designed for rapid assessment of attention priorities.
- **Level 1 / Level 2 (Contextual Explanation Drawer)**: Diagnostic factor decomposition, causal lineage, and constraint boundaries explaining *why* the metric or recommendation was generated.
- **Level 2 / Level 3 (Specialist Workbench / Tool Handoff)**: Deep interactive simulation and editing environment with bidirectional context preservation and clean return navigation.

```
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 0 / 1: EXECUTIVE / STREAM OVERVIEW                               │
│ • Primary Metric / Decision Headline / Dense Stream                    │
│ • Single High-Priority Action Card / Hero Attention Asset              │
│ • Status / Invariant Pill Badges                                       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ [Expand Context / Inspect]
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1 / 2: CONTEXTUAL EXPLANATION DRAWER / ON-DEMAND DISCOVERY       │
│ • Factor Contribution Breakdown (Attribution)                          │
│ • On-Demand Exchange Tape Evaluation (for non-prescreened symbols)     │
│ • Invariant Lineage & Data Provenance                                  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ [Launch Specialist Workbench / Execution]
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 2 / 3: SPECIALIST WORKBENCH / EXECUTION HUB                      │
│ • Full Interactive Canvas (/workbench/* or /setups)                   │
│ • Multi-Parameter Sliders & Governed Sizing Ladders                    │
│ • Bidirectional Context & "← Return to Hub" Navigation                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Executive Hubs Semantic Zoom Matrix

| Hub Route & Title | Level 0 Summary Elements | Level 1 Context Drawer Content | Level 2 / 3 Workbench Target & Route | Context State Preserved Across Zoom Levels | Return Navigation & Wayfinding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`/today`**<br>Execution & Capacity Governance | • Global Triad (LHI, HHI, IAI from persisted store)<br>• Primary Next Best Action Card<br>• Active Capacity & Solvency Constraints | • Diagnostic Lineage (Reserves, Burn, Solvency ratio)<br>• Secondary Prioritized Action List<br>• Calendar Collision Risk Flags | **168-Hour Allocator Workbench**<br>`/workbench/allocator` | • Active Triad state (`lhi`, `hhi`, `iai`)<br>• Selected action ID (`nextBestAction.id`)<br>• Active constraints list | Explicit header link:<br>`<Link href="/today">← Return to /today</Link>` |
| **`/future`**<br>Projections & Trajectories | • Target Net Worth & Runway Projections<br>• Monte Carlo Forecast Percentiles<br>• Scenario Branch Point Badges | • Parametric Sensitivity Breakdown (Savings rate vs Return vs Inflation)<br>• Historical Shock Stress Tests (2008, 2020, 2022)<br>• Branch Point Trade-Off Analysis | **Monte Carlo Simulation Workbench**<br>`/workbench/simulation` | • Simulation Horizon (years)<br>• Target Percentile filter<br>• Net Worth baseline configuration | Explicit header link:<br>`<Link href="/future">← Return to /future</Link>` |
| **`/progress`**<br>Velocity & Milestone Attainment | • Life Velocity Index<br>• Milestone Completion Progress Bars<br>• Allocation Drift Warning | • Attribution Breakdown (Discipline alpha vs Market returns vs Savings velocity)<br>• Longitudinal Habit Streaks<br>• Calibration & Bias Ledger | **Reflective Journal Workbench**<br>`/workbench/journal` | • Active Time Filter (Quarterly / YTD)<br>• Selected Milestone ID<br>• Rule adherence scores | Explicit header link:<br>`<Link href="/progress">← Return to /progress</Link>` |
| **`/household`**<br>Capital Structure & Entities | • Consolidated Liquid Reserves<br>• Multi-Entity Breakdown (Personal, LLC, Trust)<br>• Debt Service Runway | • Entity Flow Breakdown (Capital transfers, LLC tax pass-through)<br>• Cross-Collateralization Risk Flags<br>• Fixed vs Variable Burn Ratios | **Life Graph Specialist Workbench**<br>`/workbench/life-graph` | • Selected Entity ID<br>• Active Currency view<br>• Balance sheet filter parameters | Explicit header link:<br>`<Link href="/household">← Return to /household</Link>` |

---

## 3. Trading Workstation Semantic Zoom Matrix

| Hub Route & Title | Level 1 Dense Stream Elements | Level 2 Context & Discovery Drawer | Level 3 Deep Tooling & Execution Handoff | Context State Preserved Across Zoom Levels | Return Navigation & Wayfinding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`/radar`**<br>Tactical Confluence Scanner | • Attention Hero Card<br>• Dense Confluence Stream Table<br>• Match count indicator (`X of Y`) | • Search Status Banner for 0 pre-screened matches<br>• On-Demand Tape Scanner (`handleOnDemandScan`)<br>• Interactive empty state with diagnostics | **Tactical Execution & SEC Research**<br>`/setups?ticker={ticker}`<br>`/research?ticker={ticker}` | • Selected ticker symbol<br>• Confluence score & catalyst data<br>• Filter category context | Direct deep link CTA:<br>`Inspect in /setups →`<br>`Clear filter / Return` |
| **`/setups`**<br>Asymmetric Execution Hub | • Selected Asset Tactical Card<br>• Governed Position Sizing Calculator<br>• Null-safe price metrics (`formatPrice`) | • Available Setups Catalog Grid<br>• Suppression Criteria Diagnostic (e.g. Stage 4 decline)<br>• R-multiple & Stop Distance Breakdown | **Order Execution & Research**<br>`/setups?ticker={ticker}`<br>`/research?ticker={ticker}` | • Account equity & risk budget<br>• Chosen entry pivot & stop loss<br>• Sizing clamp percentage | Clean catalog navigation:<br>`View Available Setups` (clears URL parameter via `router.replace('/setups')`) |

---

## 4. Context Preservation & State Hydration Verification

1. **State Isolation**: When a user transitions from Level 0 to Level 1 inside `<SemanticZoom>`, the UI maintains all current page filters, active selections, and scroll positions without triggering full page reloads.
2. **Workbench Deep-Linking**: When launching a Level 2 / 3 Specialist Workbench, the target route receives context from the unified CQRS store (`getUnifiedCockpitState()`):
   - The Triad banner renders identically across both the hub and the workbench.
   - Constraint states and next actions remain synchronized.
3. **Return Navigation Guarantee**: Every workbench route in `frontend/app/workbench/*` incorporates an invariant-protected top-bar return link to its originating hub.

---

## 5. Honest Evaluation of the "30-Second Executive View"

The product roadmap sets a goal of a **30-second executive scan** for Level 0 views. 
- **What is verified**: Information architecture, typography scale, progressive disclosure, and reduction of visual clutter (eliminated multi-level card farms) are structurally verified.
- **What is unverified**: No empirical eye-tracking or timed task-completion usability studies with representative human users have been performed. In accordance with strict evidence-based auditing standards, this product goal is classified as **Architecturally Supported but Empirically Unverified**.
