# ArxTerminal: Clean Prediction Evaluation Specification

**Specification Date**: September 6, 2026
**Document Version**: 2.0-CLEAN
**Status**: MEASUREMENT FRAMEWORK v2.0 — ADVERSARIAL TEST SUITE PASSING — OBSERVATION PHASE
**Objective**: Establish the definitive, leak-free mathematical and operational framework for evaluating ArxTerminal's predictive capability going forward.

---

## 1. The Core Temporal Principle

To guarantee zero hindsight contamination and zero look-ahead leakage, all evaluations must satisfy the **Temporal Availability Invariant**:

$$\forall t \in \text{Timeline}, \quad \text{Model Inputs}_t \subset \mathcal{I}_t$$

where $\mathcal{I}_t$ is the set of information legally and publicly accessible to an investor at timestamp $t$.

```
                       TEMPORAL INFORMATION SEPARATION
  Economic Period T        Filing / Dissemination Date T_pub       Database Ingestion T_ingest
 ──────────────────────┬───────────────────────────────────────┬───────────────────────────────►
  (e.g. Q4 ends Dec 31)│ (e.g. 10-K filed Feb 18 at 16:05 EST) │ (e.g. SQLite ETL at 18:00 EST)
                       │                                       │
                       └───────────────► EARLIEST VALID ◄──────┘
                                         AVAILABILITY POINT
```

### Strict Temporal Rules
1. **Rule 1 — No Filing-Date Pre-Dating**: No financial metric (revenue, operating cash flow, net income, book value) may enter model inputs before the SEC acceptance timestamp (`acceptanceDateTime` in SEC EDGAR submissions).
2. **Rule 2 — Point-in-Time Fundamental Tables**: All fundamental tables must carry an explicit `effective_start_date` and `effective_end_date`. A query at cutoff $t$ must execute:
   ```sql
   SELECT * FROM asset_fundamentals_pit
   WHERE symbol = ? AND filing_acceptance_date <= ?
   ORDER BY filing_acceptance_date DESC LIMIT 1;
   ```
3. **Rule 3 — Missing Data Policy**: If an asset lacks audited SEC filings prior to cutoff $t$, the fundamental score is marked as `UNAVAILABLE` and the asset is gated into `EVIDENCE_INCOMPLETE`. It is never backfilled using future statements.

---

## 2. Definitive Status of Historical Backtesting

> **Formal Declaration**:
> **A clean historical fundamental backtest cannot currently be performed with the available database.**
> The current SQLite market store (`~/.finance_market_store.db`) contains only static September 2026 factor snapshots. Until point-in-time XBRL filings are ingested with historical acceptance dates, any backtest claiming to evaluate fundamental confluence across earlier historical dates is contaminated and invalid.

---

## 3. Pure Prospective Forward Evaluation Design

Because clean historical reconstruction is bounded by current database limitations, **the primary standard for evaluating ArxTerminal is pure prospective forward observation**:

```
                       PROSPECTIVE FORWARD EVALUATION TOPOLOGY
 ┌────────────────────────┐
 │ Generation Time T_0    │ ---> Immutable Snapshot (P_0, Stop, TP1, InputsHash, DecisionHash)
 └────────────────────────┘      Stored in paper_trading_ledger.json
             │
             ▼
 ┌────────────────────────┐
 │ Market Sessions T_1..20│ ---> Daily EOD Ingestion (trade_date > T_0)
 └────────────────────────┘      Evaluate Low <= Stop, High >= TP1, Close_20
             │
             ▼
 ┌────────────────────────┐
 │ Outcome Resolution     │ ---> Precedence: Intraday touch arbitration fail-closed to Stop.
 └────────────────────────┘      Record Realized Return, MFE, MAE, Sessions Observed.
```

### Invariants of Prospective Evaluation
- **Zero Retrospective Mutation**: Once recorded, `entryPrice`, `stopLoss`, `takeProfit1`, and `confluenceScore` cannot be modified under any condition.
- **Dual-Hash Anti-Tamper Verification**: Automated harvesters verify SHA-256 hashes prior to updating forward observations. If a hash mismatch occurs, the execution raises `GOVERNANCE_INTEGRITY_FAILURE` and halts.

---

## 4. Shadow-Testing Framework (Pre-Deployment Testing)

Before any proposed strategy improvement (such as a Trailing Breakeven Stop or ATR corridor adjustment) is considered for production, it must be evaluated in **Shadow Mode**:

```
                               SHADOW TESTING TOPOLOGY
                             Point-in-Time Tape at T_0
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
       ┌─────────────────────────┐               ┌─────────────────────────┐
       │   Production Engine     │               │    Shadow Candidate     │
       │ (v2.4.0-phase24-freeze) │               │   (e.g. Trailing Stop)  │
       └─────────────────────────┘               └─────────────────────────┘
                    │                                         │
                    ▼                                         ▼
            Live Order State                          Shadow Tracking Slot
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                         Comparative Performance Evaluation
                         (Expectancy, Win Rate, MFE Truncation)
```

### Shadow Evaluation Metrics
To justify promoting a shadow candidate to production, the candidate must demonstrate:
1. **Net Expectancy Superiority**: $\text{Expectancy}_{\text{shadow}} > \text{Expectancy}_{\text{prod}} + 0.50\%$ after accounting for slippage.
2. **No Premature Winner Truncation**: Gross winning returns must not be reduced by $>10\%$ due to premature breakeven exits.
3. **Minimum Observation Window**: At least 30 prospective trades evaluated in parallel.

---

## 5. Minimum Evidence Required Before Claiming Economically Meaningful Edge

To prevent premature claims of mathematical superiority, ArxTerminal must satisfy the following **8 Multi-Dimensional Verification Gates**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STATISTICAL PROMOTION GATES FOR PREDICTIVE EDGE                 │
├──────────────────────────┬─────────────────────────────┬───────────────────────────────┤
│ Gate                     │ Requirement                 │ Methodological Rationale      │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 1. Minimum Resolved (N)  │ N >= 60 Resolved Trades     │ Observation milestone to bound│
│    Observation Milestone │ (Diagnostic layer; not a    │ win rate standard error <=6.4%│
│                          │ binary truth machine)       │ across multiple market regimes│
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 2. Independent Dates     │ Fired across >= 20 distinct │ Identifies calendar clustering│
│                          │ calendar trading sessions   │ and macro beta dependence.    │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 3. Market Regimes        │ Tested in both Bull (SPY    │ Tests edge stability outside  │
│                          │ > 50 SMA) and Pullback tape │ of trending market regimes.   │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 4. Net Expectancy &      │ Primary Net Expectancy      │ Standardized at 30 bps round- │
│    Friction Sensitivity  │ > +1.00% at 30 bps; compute │ trip drag; calculates full    │
│    Grid                  │ 0, 15, 30, 50, 100 bps +    │ sensitivity grid and breakeven│
│                          │ Breakeven friction (F_break)│ friction where edge hits 0.   │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 5. Profit Factor Hurdle  │ Profit Factor >= 1.50 : 1   │ Baseline survival threshold.  │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 6. Multi-Benchmark Suite │ Outperform 5 Baselines:     │ Decouples alpha from beta,    │
│    Outperformance        │ 1. Cap-Weight (SPY)         │ sector selection, and momentum│
│                          │ 2. Equal-Weight (RSP)       │ tailwinds. Reports mean/median│
│                          │ 3. Sector ETF (e.g. XLK)    │ excess returns and hit rates. │
│                          │ 4. Universe 30d Momentum    │ Evaluates Monte Carlo random  │
│                          │ 5. Monte Carlo Random Entry │ percentile rank (1,000 runs). │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 7. Signal Quality vs     │ Directional Accuracy > 50%  │ Decouples raw asset direction │
│    Trade Construction &  │ at T+1, T+5, T+10, T+20;    │ from stop geometry. PSFE is an│
│    PSFE Exploratory Gate │ Multi-Horizon Capture Ratios│ exploratory shadow metric, NOT│
│                          │ (5d, 10d, 20d); PSFE Tracked│ an assertion of bad stops.    │
├──────────────────────────┼─────────────────────────────┼───────────────────────────────┤
│ 8. Confluence Ranking    │ Ordinal monotonicity across │ Confluence is an ordinal rank.│
│    Power & Spearman      │ buckets (<75, 75-79, 80-84, │ Evaluates Spearman rho (vs    │
│    Correlation           │ 85+); Positive Q4 - Q1 spread│ return & MFE) and top vs      │
│                          │ on Win Rate & Mean Return.  │ bottom quartile discrimination│
└──────────────────────────┴─────────────────────────────┴───────────────────────────────┘
```

### Detailed Methodological Invariants

#### A. Primary Friction Standard (30 bps) & Linear Additive Arithmetic
- **Primary Evaluation Assumption**: Realized trade percentage returns are adjusted via an **additive deduction of 30 bps round-trip transaction drag** ($0.30\%$):
  $$R_{\text{net}} = R_{\text{gross}} - 0.30\%$$
- **Arithmetic Consistency**: Linear basis-point deduction is applied consistently across all strategy trades and comparative baselines (`BaselineEngine`).
- **Friction Sensitivity Grid**: The scorecard computes net expectancy across 5 friction tiers:
  - $0\text{ bps}$ (Gross theoretical ceiling)
  - $15\text{ bps}$ (Liquid large-cap assumption)
  - $30\text{ bps}$ (**Primary Evaluation Assumption**)
  - $50\text{ bps}$ (Stressed execution / mid-cap assumption)
  - $100\text{ bps}$ (Severe execution penalty)
- **Breakeven Friction ($F_{\text{breakeven}}$)**:
  $$F_{\text{breakeven}} = \text{Gross Expectancy} \times 100 \quad (\text{in bps})$$
  This pinpoints the exact execution cost at which the strategy's mathematical expectancy dissolves to zero.

#### B. Pre-Registered Execution Convention: Fail-Closed Intrabar Touch Precedence
- When high $\ge \text{TP1}$ and low $\le \text{Stop}$ on the same daily bar:
  - The outcome resolves to **`STOP_LOSS`** and flags `intrabarCollision = True`.
  - **Methodological Status**: This is a **pre-registered execution convention** (pessimistic fill policy), not an empirical proof of intra-day tick path. It is applied identically across both ArxTerminal and the baseline engines, and cannot be modified post-hoc.

#### C. Cohort Contamination Firewall & Provenance Classification
To ensure that the 28 historically contaminated Phase 24 setups cannot enter prospective evaluations:
- Every prediction carries a rigid provenance tag:
  - `PROSPECTIVE_CLEAN`: Emitted $\ge 2026\text{-}09\text{-}04$, running frozen commit (`4e36862`), with valid dual SHA-256 snapshot hashes.
  - `HISTORICAL_CONTAMINATED`: Emitted prior to the September 4 freeze (e.g. the 28 Phase 24 events). Permanently designated as research-invalid for proving predictive edge.
  - `HISTORICAL_UNKNOWN` / `EXCLUDED`: Tampered hashes, missing dates, or unknown commits.
- **Scorecard Firewall Enforcement**: `compute_governance_scorecard` automatically insulates the clean cohort. If an analyst attempts to run an evaluation on mixed or contaminated cohorts, the system sets `cleanEvaluationEligible = False` and logs a loud contamination warning.

#### D. Multi-Benchmark Evaluation Suite & Survivorship Invariant
ArxTerminal is evaluated against five distinct baselines:
1. **Cap-Weighted Market (SPY)**: Tracks mean excess return, median excess return, and hit rate (% Arx > SPY at $T+20$).
2. **Equal-Weighted Market (RSP)**: Isolates broad market breadth from mega-cap tech dominance.
3. **Sector ETF (e.g. XLK)**: Isolates industry-specific beta from stock-picking alpha.
4. **Universe 30D Momentum Baseline**:
   $$R_{\text{mom}, 30} = \frac{\text{Close}(T_0)}{\text{Close}(T_{-30})} - 1$$
   Top decile (10%) of the universe selected at $T_0$ and tracked forward under matched risk geometry.
5. **Monte Carlo Random Entry Baseline**:
   - $1,000+$ matched random simulations drawn from the traded universe at $T_0$ with fixed seed (`42`).
   - Supports **Sector-Matched Sampling** (`match_sector=True`) to avoid comparing tech-heavy signals against arbitrary unconstrained market beta.
   - **Survivorship Demarcation & Caveat**: For prospective cohorts, the baseline samples from the active universe quoted at $T_0$. However, 'actively quoted at $T_0$' is an active-universe snapshot, not a reconstructed point-in-time historical universe (delistings, bankruptcies, suspensions, historical liquidity cutoffs). The measurement system makes no claim of having solved historical survivorship bias.

#### E. Cluster Structure, Dependence & Portfolio Aggregation Diagnostics
- **Cluster Diagnostics**: The scorecard explicitly measures and reports:
  - Calendar clustering ($\text{Max trades} / \text{session}$, session distribution)
  - Sector concentration ($\text{Max sector } \%$)
  - Market regime distribution ($\text{Bull}$ vs. $\text{Pullback}$)
  - **Diagnostic Caveat**: Nominal $N$ must not be interpreted as $N$ independent observations if trades cluster into a single macro wave or sector rally.
- **Portfolio Aggregation Layer**: Measures maximum concurrent open exposures and top-sector capital concentration. Trade-level positive expectancy does not guarantee bounded portfolio drawdown or Sharpe ratio under correlated simultaneous tech positions.

#### F. Automated Production-Freeze Manifest
- The exact source files of the three production decision engines:
  - `analyst_dashboard/analyzers/optimal_execution.py`
  - `analyst_dashboard/analyzers/confluence_engine.py`
  - `analyst_dashboard/analyzers/decision_hierarchy.py`
  are cryptographically frozen in [`FROZEN_ENGINE_MANIFEST.json`](file:///c:/Users/akara/Documents/Projects/finance/FROZEN_ENGINE_MANIFEST.json).
- The automated test `test_production_engine_freeze_manifest_compliance` runs on every CI/test execution, instantly failing if any production engine code is altered during the observation window.

#### G. Trade Construction vs. Signal Quality & PSFE Framing
- **Multi-Horizon Capture Ratios**: Capture is computed at multiple horizons:
  $$\text{Capture}_{5d} = \frac{\text{Realized}}{\text{MFE}_{5d}}, \quad \text{Capture}_{10d} = \frac{\text{Realized}}{\text{MFE}_{10d}}, \quad \text{Capture}_{20d} = \frac{\text{Realized}}{\text{MFE}_{20d}}$$
- **Post-Stop Favorable Excursion (PSFE)**:
  - Measures the percentage of stopped-out trades that subsequently traded through TP1 or achieved positive excursion prior to $T+20$.
  - **Exploratory Status**: PSFE is an **exploratory research metric** for shadow exit calibration (e.g. testing wider stops). It is **NOT** an assertion that the protective stop was "bad" or should have been ignored.

#### H. Prediction Ranking Power (Spearman & Quartile Discrimination)
Confluence Score is an **ordinal ranking metric**, not a calibrated probability. It is evaluated via:
1. **Spearman Rank Correlation ($\rho$)**: Evaluates rank-order consistency between Confluence Score and Realized Return, as well as Confluence Score and MFE.
2. **Top-vs-Bottom Quartile Discrimination ($Q4 - Q1$)**:
   - Compares the top quartile ($Q4$) against the bottom quartile ($Q1$).
   - A valid predictive ranking requires positive win rate spread ($\Delta\text{WR} > 0$) and positive mean return spread ($\Delta R > 0$).

### Diagnostic Statistical Inference Protocol (No Rigid $p < 0.05$ Gate)
- **Sample Reality**: Over a prospective cohort of $N \approx 60$ actionable trades, a genuinely profitable trading strategy can possess robust positive expectancy without reaching conventional two-tailed significance ($p < 0.05$). Conversely, data-snooped backtests routinely achieve $p < 0.01$ purely by chance.
- **Protocol**: Standard errors, 95% Confidence Intervals (Expectancy CI, Wilson Win Rate CI), and Spearman p-values are reported continuously as a **diagnostic uncertainty layer**. Statistical significance is **not** used as a single binary pass/fail guillotine.

---

## 6. The 30/60/90-Day Evaluation Framework

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          30 / 60 / 90-DAY EVALUATION ROADMAP                           │
├───────────────────────────┬────────────────────────────────────────────────────────────┤
│ Timeframe                 │ Primary Operational & Scientific Milestones                │
├───────────────────────────┼────────────────────────────────────────────────────────────┤
│ **First 30 Days**         │ • Maintain 100% frozen baseline (v2.4.0-phase24-freeze).   │
│ (Sessions 1 to 20)        │ • Ingest daily EOD candles for 9 live launch positions.    │
│                           │ • Track directional returns at T+1, T+5, T+10, T+20.       │
│                           │ • Measure benchmark relative returns vs SPY, RSP, Sector.  │
│                           │ • Zero code changes to production decision engine.         │
│                           │ • Audit data pipeline reliability and SQLite WAL locks.    │
├───────────────────────────┼────────────────────────────────────────────────────────────┤
│ **60 Days**               │ • Accumulate second prospective trade cohort (~15-25 total)│
│ (Sessions 21 to 45)       │ • Audit Post-Stop Favorable Excursion (PSFE) & capture.    │
│                           │ • Evaluate empirical MFE/MAE multi-horizon distribution.   │
│                           │ • Run Confluence Monotonicity bucket evaluation.           │
│                           │ • Compute Spearman rank correlation and quartile spread.   │
│                           │ • Publish Interim Prospective Performance Scorecard.       │
├───────────────────────────┼────────────────────────────────────────────────────────────┤
│ **90 Days**               │ • Evaluate cumulative sample across N >= 50-60 setups.     │
│ (Sessions 46 to 65)       │ • Benchmark vs RSP, Sector, 30D Momentum, and Monte Carlo. │
│                           │ • Assess cross-regime stability (Bull vs Pullback tape).   │
│                           │ • Compute 95% confidence intervals and standard errors.    │
│                           │ • Publish Final 90-Day Prospective Edge Verdict.           │
└───────────────────────────┴────────────────────────────────────────────────────────────┘
```

**Explicit Scientific Caveat**: Depending on market volatility and the frequency of setups meeting the strict Confluence $\ge 75$ hurdle, even 90 days may yield an insufficient sample size for definitive statistical proof. If $N < 60$ or distinct dates $< 20$ at day 90, the evaluation must remain formally classified as **Preliminary / Inconclusive**.
