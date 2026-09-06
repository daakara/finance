# Implementation Plan: Continuous Prediction Evaluation System

**Date**: September 6, 2026
**Auditor / Architect**: Quantitative Finance Guardian (`quant-guardian`) & Systems Architect
**Objective**: Transition ArxTerminal from one-off forensic audits to a fully automated, continuous prediction evaluation and drift-detection system.

---

## 1. Problem Statement & Root Cause

### The Problem
Historically, evaluating whether ArxTerminal's recommendations produce genuine predictive value required manual, retrospective engineering audits. Retrospective audits introduce two severe risks:
1. **Hindsight Bias & Inadvertent Contamination**: Reconstructing historical state from past data risks look-ahead leakage.
2. **Evaluation Latency**: When performance degrades or market regimes shift, manual audits take days to detect deteriorating expectancy.

### Root Cause
1. **Lack of Automated Forward Ingestion**: Although `ExperimentLedger` was created in Phase 25, daily candle harvesting and excursion tracking still require manual script execution.
2. **Missing Real-Time Calibration Metrics**: Confidence scores (confluence) and expected returns are not automatically tracked against realized Brier scores or Sharpe ratios.
3. **No Automated Drift Gate**: There is no automated circuit breaker that alerts operators if live trade expectancy deviates by $>2\sigma$ from the $+4.29\%$ out-of-sample benchmark.

---

## 2. Proposed Architecture: The Continuous Prediction Evaluation Engine

```
                                  CONTINUOUS EVALUATION TOPOLOGY
 ┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
 │ Live Screener Engine │ ---> │  Immutable Ledgers      │ ---> │ Daily Forward Harvester │
 │ (v2.4.0-phase24)     │      │  (Dual-Hash Snapshots)  │      │ (Cron: 21:30 UTC Daily) │
 └──────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                             │
                                                                             ▼
 ┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
 │ Operator Alert Gate  │ <--- │ Drift & Calibration     │ <--- │ SQLite Market Store     │
 │ (Expectancy Circuit) │      │ Analytics Service       │      │ (EOD Adjusted OHLCV)    │
 └──────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

### Key Architectural Pillars
1. **Immutable Prediction Registration**:
   Every time an asset transitions to `ACTIONABLE_SETUP` or `IN_BUY_ZONE`, an immutable record is registered with a `decisionSnapshotHash` and `inputsSnapshotHash`.
2. **Automated Forward Harvester**:
   A lightweight background daemon runs daily at 21:30 UTC (post-US market close) to fetch daily candles, record maximum favorable excursion (MFE) and maximum adverse excursion (MAE), and resolve outcomes fail-closed.
3. **Continuous Statistical Drift Engine**:
   Computes rolling 30-day expectancy, Brier calibration error, and profit factor against the frozen baseline.

---

## 3. Files & Components Affected

| Component | File Path | Proposed Action | Purpose |
| :--- | :--- | :--- | :--- |
| **Ledger Engine** | `analyst_dashboard/governance/experiment_ledger.py` | [ENHANCE] | Add automated calibration metrics and Brier score tracking |
| **Daemon Scheduler** | `analyst_dashboard/governance/daily_harvester_cron.py` | [NEW] | Automated daily EOD harvester daemon |
| **API Analytics** | `api/routes/evaluation.py` | [NEW] | Expose `/api/v1/evaluation/metrics` for live dashboard |
| **Storage Engine** | `analyst_dashboard/data/market_db.py` | [ENHANCE] | Enable SQLite WAL mode (`journal_mode=WAL`) |
| **Frontend UI** | `components/dashboard/PredictionDriftChart.tsx` | [NEW] | Visual calibration and drift tracking component |

---

## 4. Data Changes & Schema Specification

### Persistent Ledger Schema (`paper_trading_ledger.json`)
The ledger schema will be expanded to incorporate formal calibration metrics without breaking backward compatibility:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "version": "1.1.0",
  "engineCommit": "4e36862",
  "engineTag": "v2.4.0-phase24-freeze",
  "evaluationMetrics": {
    "rollingExpectancy": 4.29,
    "rollingProfitFactor": 2.51,
    "brierCalibrationScore": 0.182,
    "lastEvaluatedSession": "2026-09-04"
  },
  "signals": [
    {
      "signalId": "ANET_2026-09-04",
      "symbol": "ANET",
      "signalDate": "2026-09-04",
      "entryPrice": 191.44,
      "stopLoss": 179.00,
      "takeProfit1": 218.36,
      "takeProfit2": 222.01,
      "confluenceScore": 84.6,
      "decisionSnapshotHash": "2cda1c986bf49e16...",
      "inputsSnapshotHash": "e89f9d5c2a2091d2...",
      "forwardTracking": {
        "sessionsObserved": 0,
        "currentPrice": 191.44,
        "maxFavorableExcursionPct": 0.0,
        "maxAdverseExcursionPct": 0.0,
        "tp1Hit": false,
        "stopHit": false,
        "resolvedOutcome": null,
        "resolutionTimestamp": null
      }
    }
  ]
}
```

---

## 5. API Changes

### New Endpoint: `GET /api/v1/evaluation/summary`
Returns the real-time governance scorecard:
```json
{
  "activeSignalsCount": 9,
  "resolvedSignalsCount": 0,
  "liveWinRate": null,
  "historicalOosWinRate": 0.464,
  "liveExpectancy": null,
  "benchmarkExpectancy": 4.29,
  "profitFactor": 2.51,
  "driftStatus": "HEALTHY_BASELINE",
  "lastHarvestTimestamp": "2026-09-04T17:32:00Z"
}
```

---

## 6. Testing & Validation Plan

1. **Automated Mock Harvester Test**:
   - Seed dummy candles with simulated upside (+15%) and verify that `tp1Hit` triggers and outcome transitions to `TP1_WIN`.
   - Seed dummy candles with simulated downside (-7%) and verify that `stopHit` triggers and outcome transitions to `STOP_LOSS`.
2. **Snapshot Hash Anti-Tampering Test**:
   - Verify that modifying any signal price or confluence raises `GOVERNANCE_INTEGRITY_FAILURE`.
3. **Concurrent Lock Stress Test**:
   - Run 10 concurrent threads reading and writing to SQLite under WAL mode to verify 0 lock collisions.

---

## 7. Migration & Rollout Strategy

- **Phase 1 (Current - September 2026)**: Maintain 100% frozen baseline. All 9 live signals tracked passively.
- **Phase 2 (Sessions 1 to 20: Sept 7 - Oct 2, 2026)**: Daily automated forward harvesting. Zero code modifications to trading logic.
- **Phase 3 (Post-Session 20 Review: October 2026)**: Formal Phase 26 Milestone review comparing live realized expectancy to the $+4.29\%$ OOS benchmark.
