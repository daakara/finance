# ARX Terminal vNext: Sprint 3 Engineering Execution Package
## Change Intelligence Engine, Materiality Governance & Attention Allocation

**Document ID**: ENG-PKG-ARX-VNEXT-S3  
**Version**: 3.0.0-PROD  
**Sprint**: Sprint 3 (Stage 6: Change Intelligence, Materiality Engine, Attention Feed)  
**Date**: 2026-09-07  
**Preceding Milestones**: Sprint 1 Closed (VAL-REP-ARX-VNEXT-S1), Sprint 2 Closed (VAL-REP-ARX-VNEXT-S2)  
**Core Invariant**: Materiality Gate (A change may only appear in Stage 6 if it survives Materiality Evaluation. Snapshot differences alone are insufficient grounds for user interruption.)  

---

## 1. Mission

> **Sprint 3 transforms ARX from a Decision Intelligence Workstation into a Change Intelligence Workstation by eliminating the re-read tax and surfacing only materially relevant changes since the user's last acknowledged thesis baseline.**

Sprint 1 succeeded or failed on **layout**.  
Sprint 2 succeeded or failed on **explainability**.  
Sprint 3 succeeds or fails on **trust**.

If users trust the delta system, ARX becomes habit-forming. If they do not, Stage 6 becomes another ignored notification center.

---

## 2. The User Problem: The Re-Read Tax

Today, institutional operators suffer from an invisible cognitive tax:
\text{Return User} \longrightarrow \text{Open Ticker} \longrightarrow \text{Re-Read Everything Again} \longrightarrow \text{Mentally Reconstruct Changes}
- **Baseline Re-Read Time**: **.8\text{ seconds}$** spent re-evaluating unchanged metrics.
- **Mental Fatigue**: Operators cannot easily distinguish whether a thesis shifted by 2 points (statistical noise) or whether price crossed from WAITING_PULLBACK into IN_BUY_ZONE (critical execution trigger).
- **The Solution**: Target workflow is **Return $\to$ Read Delta in $< 10\text{s} \to$ Done**.

---

## 3. Snapshot Schema (Lean & Deterministic)

To prevent Snapshot Explosion, ARX stores only the decision-authoritative state vector.

`	ypescript
export interface TickerSnapshot {
  ticker: string;
  snapshotId: string;
  timestamp: string; // ISO 8601
  modelVer: string;
  setupScore: number; // 0 - 100
  domainConfidence: HIGH | MODERATE | LIMITED;
  executionState: IN_BUY_ZONE | APPROACHING_TARGET | WAITING_PULLBACK | STOPPED_OUT | NEUTRAL;
  marketRegime: RISK_ON | NEUTRAL | DEFENSIVE;
  liquidityTier: HIGH | MODERATE | RISK;
  spotPrice: number;
  entryZone: { low: number; high: number };
  stopLossFloor: number;
  takeProfit1: number;
  takeProfit2: number;
  flowZScore: number;
  validationTier: THIN | DEVELOPING | ESTABLISHED;
  topDriverIds: string[];
}
`

---

## 4. Storage Architecture (Client-Owned IndexedDB)

Following ADR-002 and ADR-004, all snapshot intelligence is client-owned, private, and zero-PII.

`	ypescript
export interface TickerSnapshotRecord {
  ticker: string;
  baselineSnapshot: TickerSnapshot;
  latestSnapshot: TickerSnapshot;
  deltaSummary?: DeltaReport;
  acknowledgedAt: string | null;
  history?: RollingSnapshotEntry[]; // Capped at exactly 30 entries for sparkline / timeline
}
`

- **Database Name**: rx_change_intelligence_db (Version 1)
- **Object Store**: 	icker_snapshots (KeyPath: 	icker)
- **Retention Rule**: Strictly bounds storage to (N)$ tickers, not (\text{Mutations})$.

---

## 5. The Materiality Engine (First-Class Architectural Component)

`
┌──────────────────────────────────────────────────────────────────────────────────┐
│ THE 5-LAYER MATERIALITY PIPELINE                                                 │
│ Raw Snapshot Change                                                              │
│         ↓                                                                        │
│ [Layer 1] Raw Difference Detection (Absolute & percentage deltas)                │
│         ↓                                                                        │
│ [Layer 2] Materiality Rules (Domain-specific variance thresholds)                │
│         ↓                                                                        │
│ [Layer 3] Severity Classification (NONE, INFO, MATERIAL, CRITICAL)               │
│         ↓                                                                        │
│ [Layer 4] User Impact Evaluation (Filters non-actionable changes)                 │
│         ↓                                                                        │
│ [Layer 5] Attention Signal Generation (Delta Banner & Portfolio Feed)            │
└──────────────────────────────────────────────────────────────────────────────────┘
`

> [!IMPORTANT]
> **Core Rule: Hash Difference $\ne$ Material Change.**  
> Changes in decimal precision (e.g. Target .004 \to 22.011$) alter hashes but alter zero meaning. The Materiality Engine must evaluate decision semantics before generating any user signal.

---

## 6. Delta Classification Matrix

`
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ MATERIALITY THRESHOLD MATRIX                                                                           │
├─────────────────────┬──────────────┬──────────────┬──────────────────────────┬─────────────────────────┤
│ METRIC              │ L0: IGNORE   │ L1: INFO     │ L2: MATERIAL (BANNER)    │ L3/L4: CRITICAL (FEED)  │
├─────────────────────┼──────────────┼──────────────┼──────────────────────────┼─────────────────────────┤
│ Setup Score         │ Δ <= 2 pts   │ Δ 3 - 5 pts  │ Δ 6 - 9 pts              │ Δ >= 10 pts             │
│ Institutional Flow  │ < 1.0σ       │ 1.0σ - 1.5σ  │ 1.5σ - 2.5σ              │ >= 2.5σ (Block Surge)   │
│ Execution State     │ No Change    │ —            │ Approaching Target       │ State Transition        │
│ Market Regime       │ No Change    │ —            │ —                        │ Any Transition          │
│ Liquidity Tier      │ No Change    │ —            │ High -> Moderate         │ Any Tier -> Risk        │
│ Validation Depth    │ Same Tier    │ —            │ Tier Upgrade / Downgrade │ —                       │
└─────────────────────┴──────────────┴──────────────┴──────────────────────────┴─────────────────────────┘
`

---

## 7. Attention Feed Logic

Only events of severity **L3** or **L4** are elevated to the cross-ticker **Portfolio Attention Feed**:
1. Execution State Transition (e.g., *CPRX entered Actionable Buy Zone*)
2. Market Regime Rotation (e.g., *Macro Regime rotated from RISK_ON to DEFENSIVE*)
3. Setup Score Breakout (e.g., *NVDA Setup Score jumped +14 pts*)
4. Stop Loss Invalidation (e.g., *Price violated Stop Loss Floor*)

---

## 8. Delta Banner Contract (Stage 6 Presentation)

Mounted immediately above the Stage 1 Command Strip when an unacknowledged material delta (, L3, L4$) exists:
- **Elapsed Time Context**: *Since your last acknowledged thesis 3 days ago (Sept 4):*
- **Synthesized Changes**:
  - Setup Score:  \to 82$ ($+11$ pts, Emerald)
  - Execution State: WAITING_PULLBACK $\to$ IN_BUY_ZONE
- **1-Click Primary Action**: [Acknowledge & Set New Baseline] (instantly clears banner and updates baseline).
- **Secondary Action**: [View 30-Day Timeline Diagnostics]

---

## 9. Acknowledgement Workflow

`	ypescript
export async function acknowledgeThesisDelta(ticker: string): Promise<void> {
  const record = await getTickerSnapshot(ticker);
  if (!record) return;

  const now = new Date().toISOString();
  record.baselineSnapshot = { ...record.latestSnapshot, timestamp: now };
  record.acknowledgedAt = now;
  record.deltaSummary = undefined; // Cleared

  await saveTickerSnapshot(record);
  trackTelemetryEvent(DECISION, delta_acknowledged, { ticker });
}
`

---

## 10. Telemetry & KPI Definitions

### 10.1 Telemetry Envelopes Added
- eturning_user_detected (	icker, daysSinceVisit, hasBaseline)
- delta_generated (	icker, maxSeverity, changeCount)
- delta_banner_displayed (	icker, maxSeverity, 	imeSinceLastReviewMs)
- delta_banner_expanded (	icker, expandedField)
- delta_acknowledged (	icker, 	imeToConfirmMs, severity)
- ttention_item_opened (	icker, severity, source)

### 10.2 Four Non-Negotiable KPIs
1. **Re-Read Elimination Rate (RER)**: Target $\ge 70\% - 80\%$
   \text{RER} = \frac{64.8\text{s} - \text{Measured Re-Read Time}}{64.8\text{s}}
2. **Time To Thesis Confirmation (TTTC)**: Target $< 15.0\text{ seconds}$
3. **Delta Precision Rate (DPR)**: Target $> 80\%$
   \text{DPR} = \frac{\text{Acknowledged Deltas}}{\text{Viewed Deltas}}
4. **False Positive Rate (FPR)**: Target $< 5\%$ (Ideal $< 2\%$)

---

## 11. Acceptance Test Suites & The Delta Trust Test

> [!CAUTION]
> **The Delta Trust Test (Mandatory Release Gate)**:  
> Given a user reviews CPRX and acknowledges the baseline snapshot;  
> When CPRX reloads with sub-threshold fluctuations (Score  \to 72$, Flow .03 \to 1.06$, decimal price fluctuations);  
> **Then exactly ZERO Delta Banners are displayed, and ZERO Attention Feed items are created.**  
> *Result must be 100% Zero False Positives.*

---

## 12. Implementation Phasing Plan

- **Phase 3A: IndexedDB Storage & Snapshot Engine** (storage/idb-snapshots.ts, 	ypes/change-intelligence.ts)
- **Phase 3B: First-Class Materiality Engine** (engine/materiality-engine.ts, engine/delta-classifier.ts)
- **Phase 3C: Stage 6 Delta Banner & Attention Feed** (components/delta/DeltaBanner.tsx, components/delta/AttentionFeed.tsx)
- **Phase 3D: Test Automation & Delta Trust Harness** (scripts/verify-sprint-3.mjs, Playwright E2E)
