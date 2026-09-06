# ArxTerminal: Prediction Governance & Model Registry Protocol

**Governance Standard**: `ARX-GOV-2026.1`
**Effective Date**: September 6, 2026
**Auditor**: Quantitative Governance Agent (`governance-agent`) & Project Historian (`project-historian`)
**Status**: MEASUREMENT FRAMEWORK v2.0 — ADVERSARIAL TEST SUITE PASSING — OBSERVATION PHASE

---

## 1. The Four-Tier Data Partition Invariant

To permanently eradicate data snooping, in-sample parameter tuning, and retrospective contamination, all data must be partitioned into four strictly insulated tiers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          FOUR-TIER DATA PARTITION TAXONOMY                             │
├─────────────────────┬──────────────────────────────────────┬───────────────────────────┤
│ Tier                │ Permitted Usage                      │ Governance Invariant      │
├─────────────────────┼──────────────────────────────────────┼───────────────────────────┤
│ 1. Development Data │ Strategy exploration, feature        │ Can NEVER be cited as proof│
│                     │ engineering, heuristic prototyping.  │ of predictive edge.       │
├─────────────────────┼──────────────────────────────────────┼───────────────────────────┤
│ 2. Validation Data  │ Hyperparameter tuning, threshold     │ Must be partitioned by    │
│                     │ selection, corridor calibration.     │ time or cross-validation. │
├─────────────────────┼──────────────────────────────────────┼───────────────────────────┤
│ 3. Holdout / OOS    │ Final one-shot pre-release check.    │ Once inspected, it becomes│
│                     │ Genuinely unseen historical window.  │ burned development data.  │
├─────────────────────┼──────────────────────────────────────┼───────────────────────────┤
│ 4. Prospective Live │ Real-time production observations.   │ The gold standard. Locked │
│                     │ Generated after git commit freeze.   │ at T_0 before T_1 occurs. │
└─────────────────────┴──────────────────────────────────────┴───────────────────────────┘
```

### The "Burned Data" Rule
If an engineer or auditor inspects strategy performance on a historical dataset (e.g. discovering that unconfirmed pullbacks produce a $76\%$ stop rate), **that historical dataset is permanently burned**. Any rule modified to fix those observed failures (e.g. adding a confirmation candle) **CANNOT** be evaluated on that same dataset and labeled "out-of-sample."

---

## 2. Model & Strategy Version Registry

Every recommendation emitted by ArxTerminal must carry an immutable, machine-readable provenance block linking it to the exact software state:

```json
{
  "provenance": {
    "gitCommit": "4e3686296aad24e2210ef580bbc9116054d84fd1",
    "gitTag": "v2.4.0-phase24-freeze",
    "strategyVersion": "2.4.0",
    "featureSchemaVersion": "1.2.0",
    "dataSourceVersion": "sqlite_market_store_v1",
    "decisionEngine": "DecisionHierarchyEngine_v1",
    "confluenceEngine": "ConfluenceEngine_v2_continuous"
  }
}
```

### Versioning Questions Answerable by Design
The registry enables answering:
- *"Did strategy version 2.4.0 outperform version 2.3.0 on identical market regimes?"*
- *"Was a specific stop-loss hit caused by a code change or market volatility?"*
- *"Which git commit was running when signal ANET_2026-09-04 was emitted?"*

---

## 3. Prediction Immutability & Anti-Tampering Standard

1. **Write-Once, Read-Many (WORM)**:
   A signal once committed to `paper_trading_ledger.json` cannot be edited, re-weighted, or deleted.
2. **Dual Cryptographic Hashes**:
   - `decisionSnapshotHash`: SHA-256 digest of `{symbol, spot, entry_min, entry_max, stop, tp1, tp2, confluence}`.
   - `inputsSnapshotHash`: SHA-256 digest of `{raw_candles_hash, fundamentals_snapshot, catalyst_state, regime}`.
3. **Fail-Closed Tamper Guard**:
   Any automated script, API endpoint, or evaluation engine that detects a hash mismatch immediately halts execution and emits a `GOVERNANCE_INTEGRITY_FAILURE` alert.

---

## 4. Definition of "Learning" vs. "Adaptation" vs. "Iteration"

To eliminate marketing confusion, ArxTerminal formally adopts the following technical definitions:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM EVOLUTION DEFINITIONS                              │
├─────────────────┬─────────────────────────────────────────────────┬────────────────────┤
│ Term            │ Technical Definition                            │ ArxTerminal Status │
├─────────────────┼─────────────────────────────────────────────────┼────────────────────┤
│ **Learning**    │ Autonomous, machine-executed parameter or       │ **NON-EXISTENT**   │
│                 │ weight updates driven directly by observed      │ (Class C Engine)   │
│                 │ trade outcomes without human code changes.      │                    │
├─────────────────┼─────────────────────────────────────────────────┼────────────────────┤
│ **Adaptation**  │ Dynamic, rule-encoded parameter adjustments     │ **MINIMAL**        │
│                 │ based on real-time detected market regimes      │ (Fixed corridors)  │
│                 │ (e.g. ATR-scaled stop widths).                  │                    │
├─────────────────┼─────────────────────────────────────────────────┼────────────────────┤
│ **Iteration**   │ Human software developers inspecting trade      │ **SOLE ACTIVE      │
│                 │ postmortems, modifying Python source code, and   │  MECHANISM**       │
│                 │ deploying new git commits.                      │                    │
└─────────────────┴─────────────────────────────────────────────────┴────────────────────┘
```

### What Would Be Required to Legitimately Claim "Learning"?
For ArxTerminal to legitimately claim that it is learning, the system would require:
1. An automated trade outcome listener feeding realized P&L back into an online optimizer.
2. A formal mathematical learning algorithm (e.g. Bayesian updating, Reinforcement Learning policy gradient, or online logistic regression).
3. Automated out-of-sample holdout validation before any parameter update takes effect.
4. Automated rollback mechanisms if newly learned weights degrade expectancy.
5. Strict safety boundaries preventing runaway parameter drift.

---

## 5. Promotion & Release Gates (6-Gate Safety Checklist)

No strategy version or parameter update may be deployed to production without passing all 6 release gates:

- [ ] **Gate 1: Invariant Contract Compliance**: 100% pass on spatial monotonicity ($stop < entry_{min} \le price \le entry_{max} < tp_1 < tp_2$).
- [ ] **Gate 2: Point-in-Time Temporal Audit**: Zero look-ahead leakage; fundamentals verified as legally accessible at historical bar.
- [ ] **Gate 3: Shadow-Mode Verification**: Shadow candidate tested against live production for $\ge 30$ trades with positive net alpha.
- [ ] **Gate 4: Correlation & Clustering Check**: Effective sample size $N_{eff} \ge 20$ independent macro decision events.
- [ ] **Gate 5: Full Friction Accounting**: Positive expectancy maintained after deducting 30 bps round-trip transaction drag.
- [ ] **Gate 6: Rollback Strategy Documented**: One-command git rollback procedure verified.

---

## 6. Rollback Policy & Automated Emergency Halt

If a deployed strategy version experiences **Expectancy Drawdown Drift** exceeding $2\sigma$ from its pre-release validation benchmark across $\ge 15$ resolved trades:
1. `canSizeTrade` automatically locks to `False` across all assets.
2. The engine reverts from `ACTIONABLE_SETUP` to `EVIDENCE_INCOMPLETE` advisory mode.
3. The production engine rolls back to the previous verified git commit.

---

## 7. Adversarial Measurement & Comparative Baseline Architecture

The measurement infrastructure (`analyst_dashboard/governance/`) enforces six verified architectural standards:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      ADVERSARIAL MEASUREMENT INVARIANTS SUMMARY                        │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ Module / Standard        │ Implementation              │ Enforcement Guarantee         │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ Production Freeze        │ `FROZEN_ENGINE_             │ • SHA-256 manifest of 3       │
│ Enforcement              │  MANIFEST.json`             │   production engines verified │
│                          │                             │   in CI test suite            │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ Cohort Contamination     │ `ProvenanceCohort` in       │ • Strictly isolates           │
│ Firewall                 │ `experiment_ledger.py`      │   PROSPECTIVE_CLEAN cohort    │
│                          │                             │ • Disallows contaminated data │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ BaselineEngine           │ `baseline_engine.py`        │ • 1,000 Monte Carlo runs      │
│                          │                             │   with matched risk geometry   │
│                          │                             │ • Sector-matched sampling     │
│                          │                             │ • Frozen 30D Momentum decile  │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ Pre-Registered Collision │ `experiment_ledger.py`      │ • If High >= TP1 and Low <= SL│
│ Convention               │                             │   in same bar, resolves to SL │
│                          │                             │ • Flag: `intrabarCollision`   │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 30 bps Friction Grid     │ `compute_governance_        │ • Linear additive deduction   │
│                          │  scorecard`                 │ • Grid: 0, 15, 30, 50, 100 bps│
│                          │                             │ • F_breakeven explicitly logged│
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ Cluster & Portfolio      │ `clusteringAndDependence` & │ • Session/sector clustering   │
│ Diagnostics              │ `portfolioAggregation`      │ • Concurrent exposure bounds  │
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

1. **Dual Cryptographic Freeze**: The 9 live launch positions in `paper_trading_ledger.json` remain completely frozen under SHA-256 decision and input hashes.
2. **Deterministic Reproducibility**: Monte Carlo simulation uses a fixed pseudo-random seed (`seed=42`) ensuring identical percentile rank outputs across runs.
3. **Automated Manifest Enforcement**: `test_production_engine_freeze_manifest_compliance` checks the SHA-256 hashes of `optimal_execution.py`, `confluence_engine.py`, and `decision_hierarchy.py` on every run.
4. **Permanent Historical Insulation**: The 28 historical Phase 24 setups are permanently classified as `HISTORICAL_CONTAMINATED` and cannot enter prospective performance evaluations.
5. **Observation-Only Protocol**: With the measurement framework adversarially tested and frozen, no further code modifications or parameter adjustments may be made to the strategy until prospective evidence accumulates across $N \ge 60$ resolved trades.
