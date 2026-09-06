# Adversarial Audit: Methodological Disproof & Evidence Stress-Test of ArxTerminal Post-Launch Claims

**Audit Date**: September 6, 2026
**Auditor**: Independent Adversarial Methods Auditor (`critique-methods-auditor`) & Quantitative Risk Guardian (`quant-guardian`)
**Target Document**: `ARXTERMINAL_POST_LAUNCH_PREDICTION_AUDIT.md`
**Primary Question**: *"Can we trust the evidence telling us that ArxTerminal is good?"*
**Top-Level Verdict**: **FUNDAMENTALLY FLAWED & MATERIALLY OVERSTATED (Confidence: HIGH)**

---

## 1. Executive Summary & Top-Level Verdict

The initial post-launch audit succeeded on one primary dimension: it accurately verified that **ArxTerminal has zero live completed trading sessions**, meaning its live prospective predictive edge cannot yet be evaluated from production tape.

However, when subjected to independent adversarial cross-examination, the report’s foundational historical claim:
> *"The frozen engine possesses a genuine mathematical edge out-of-sample (+4.29% expectancy, 2.51 profit factor, 3.3x Buy & Hold)"*

**COLLAPSES UNDER SCRUTINY.**

The reported $+4.29\%$ expectancy and $2.51$ profit factor cannot be accepted as proof of a durable trading edge. They are distorted by at least **four critical methodological vulnerabilities**:
1. **Critical Look-Ahead Contamination in Fundamentals**: The backtest used static September 2026 factor snapshots applied retrospectively to earlier 2026 historical cutoffs.
2. **Severe Cross-Sectional Clustering (Non-Independence)**: 78.6% of the 28 setups fired on just 4 calendar dates, reducing the effective sample size from $N=28$ to approximately $N_{eff} \approx 4\text{–}5$.
3. **Pervasive Universe Selection & Survivorship Bias**: The 81-asset tracking universe was curated in late 2026 from the best-performing market momentum leaders.
4. **Gross Miscalibration Labeled as "Well-Calibrated"**: An 87% confidence score yielding a 50% empirical win rate was erroneously classified as well-calibrated, masking a -37 percentage point over-confidence error.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        ADVERSARIAL CLAIM DISPROOF SUMMARY                              │
├────────────────────────────────┬───────────────────────────┬───────────────────────────┤
│ Prior Audit Claim              │ Adversarial Stress-Test   │ Forensic Status           │
├────────────────────────────────┼───────────────────────────┼───────────────────────────┤
│ "340 out-of-sample events"     │ N = 28 actionable setups  │ MISLEADING PRESENTATION   │
│ "Genuine mathematical edge"    │ N=28, 95% CI: [28%, 66%]  │ STATISTICALLY UNFOUNDED   │
│ "Untouched out-of-sample data" │ Static Sept 2026 factors  │ CONTAMINATED (LOOK-AHEAD) │
│ "28 independent trade setups"  │ 22/28 on 4 calendar dates │ HIGHLY CORRELATED CLUSTERS│
│ "Confidence is well-calibrated"│ 87% conf -> 50% win rate  │ SEVERE OVER-CONFIDENCE    │
│ "Trailing stop is root cause"  │ Untested shadow hypothesis│ PREMATURE ATTRIBUTION     │
│ "Institutional asymmetry"      │ Marketing buzzword        │ PURGED FROM AUDIT         │
└────────────────────────────────┴───────────────────────────┴───────────────────────────┘
```

---

## 2. Forensic Disproof of the Historical Benchmark

### Disproof 1: The N=340 vs. N=28 Masking
- **The Finding**: The initial audit prominently featured "$N=340$ events" to imply statistical depth.
- **The Reality**: The decision engine produced only **$N=28$ actionable trade setups** across the entire historical evaluation. 312 of the 340 events were merely assets scanned that failed the entry criteria.
- **The Consequence**: Statistical power for evaluating trade outcome expectancy, win rate, and profit factor is strictly bounded by $N=28$, not 340.

### Disproof 2: Fatal Look-Ahead Bias in the Fundamental Pillar
- **The Finding**: In `scratch/phase24_out_of_sample_validation.py` (lines 39–41, 72, 97–100), the historical cutoff engine executes:
  ```python
  cur.execute("SELECT * FROM asset_factor_snapshots;")
  factor_rows = {r["symbol"]: dict(r) for r in cur.fetchall()}
  ```
- **The Forensic Evidence**: Inspection of `~/.finance_market_store.db` reveals:
  - Table `asset_factor_snapshots` contains **exactly 73 rows** (one static row per symbol).
  - All records carry timestamps between **September 1, 2026 and September 6, 2026**.
  - **There is no point-in-time historical fundamentals table in the database.**
- **The Smoking Gun**: When the engine evaluated historical cutoffs at $T-160$ (January 2026), $T-140$ (February 2026), and $T-120$ (March 2026), it filtered assets using the **Piotroski F-Score, Quality Score, and Growth Score from September 2026**!
- **The Impact**: The engine benefited from massive look-ahead survival bias. It only bought assets in January–May 2026 that were known to have pristine balance sheets and high margins in September 2026. Any stock that deteriorated, reported disastrous earnings, or collapsed between January and September was pre-filtered out of historical setups by future fundamental scores.

### Disproof 3: Cross-Sectional Clustering & Non-Independence
- **The Finding**: The 28 setups were treated as 28 independent, identically distributed ($i.i.d.$) Bernoulli trials.
- **The Forensic Evidence**: Extracting the exact cutoff dates of all 28 setups reveals severe date clustering:
  ```
  Cutoff Date   │ Firing Count │ Symbols Fired
 ───────────────┼──────────────┼──────────────────────────────────────────────────
   2026-03-13   │      8       │ ACLS, DDOG, ETN, FIX, KLAC, POWI, PWR, SMCI
   2026-02-12   │      6       │ ACLS, ANET, KLAC, MPWR, ULTA, VRTX
   2026-04-13   │      4       │ CDNS, MSTR, PANW, RKLB
   2026-01-14   │      4       │ CIEN, DECK, IONQ, PWR
   2026-05-11   │      2       │ DUOL, GEV
   2026-05-24   │      1       │ BTC-USD
   2026-01-12   │      1       │ CRSP
   2026-05-07   │      1       │ DHLGY
   2026-03-16   │      1       │ NVDA
  ```
  - **22 out of 28 setups (78.6%) occurred on just 4 trading sessions.**
  - On **2026-03-13**, 8 cyclical tech and semiconductor names fired on the exact same day.
  - On **2026-04-13**, 4 high-beta momentum assets (`CDNS`, `MSTR`, `PANW`, `RKLB`) fired simultaneously and all 4 hit $TP_1$. That was not 4 independent predictive insights; it was **one single macro momentum surge** lifting all 4 boats.
  - On **2026-02-12**, 6 setups fired simultaneously and 5 of them stopped out. That was **one single market drawdown** hitting 5 correlated positions.
- **The Impact**: The effective sample size ($N_{eff}$) is not 28; it is approximately **4 to 5 market events**. An $N_{eff} \approx 4$ has zero statistical significance.

### Disproof 4: Strategy Provenance & In-Sample Developer Bleed
- **The Timeline**:
  - **Phase 21 (September 2–3, 2026)**: Auditor identified that frontend and backend decision states were out of parity.
  - **Phase 22 (September 4, 2026)**: Auditor discovered that the screener was generating false positives by buying declining stocks before base stabilization.
  - **Phase 23 (September 4, 2026)**: Developer added the **pullback confirmation candle gate** (session close in upper 50% of range or green hammer off 20 EMA) specifically to eliminate the losing trades seen during Phase 22.
  - **Phase 24 (September 4, 2026)**: That exact confirmation rule was tested against $T-160$ to $T-80$ using the exact same SQLite database.
- **The Reality**: The confirmation candle rule was **empirically reverse-engineered to fix the failures observed in this specific 2026 dataset**. Calling $T-160$ to $T-80$ "untouched out-of-sample data" ignores that the developer tuned the heuristics while actively studying this specific historical price action.

### Disproof 5: Universe Selection Bias
- **The Universe**: The 81 assets in `DAY_TRADER_CANDIDATES` and `LONG_TERM_CANDIDATES` include `NVDA`, `PLTR`, `ARM`, `SMCI`, `MSTR`, `ISRG`, `VRTX`, `LLY`, `DECK`, `GEV`.
- **The Bias**: These symbols represent the strongest, most celebrated momentum winners of the 2024–2026 US market cycle. Selecting these winners retrospectively and then backtesting whether a trend-following pullback strategy succeeded on them creates severe selection bias.

---

## 3. Disproof of Confidence Calibration Claims

The initial audit claimed:
> *"80.0% - 84.9% band (Avg Score: 82.4%): Win Rate 50.0% -> WELL-CALIBRATED"*
> *"85.0% - 90.0% band (Avg Score: 87.1%): Win Rate 50.0% -> WELL-CALIBRATED"*

**This claim is mathematically untenable.**

### Calibration vs. Expectancy
- **Calibration** measures whether predicted probability matches observed frequency:
  $$E[Y | \hat{p} = p] = p$$
  If an algorithm assigns an **$87.1\%$ confidence score**, a well-calibrated system must produce a win rate of approximately **$87.1\%$**.
- **The Empirical Reality**:
  Across the $85.0\%-90.0\%$ band ($N=6$), the empirical win rate was **$50.0\%$** (3 wins, 2 stops, 1 expired).
  $$\text{Calibration Error} = \hat{p} - \bar{y} = 0.871 - 0.500 = +0.371 \quad (+37.1\%)$$
- **The Verdict**: The system is **massively over-confident**. It overstates the likelihood of success by **37 percentage points**. Calling a 50% coin-flip win rate "well-calibrated" for an 87% confidence prediction was an egregious methodological error.

---

## 4. Re-Evaluating the Trailing-Stop Hypothesis

The initial audit claimed:
> *"What does ArxTerminal consistently get wrong? The absence of a trailing stop mechanism is the root cause of losses."*

### The Adversarial Challenge
1. **Unsubstantiated Root-Cause Attribution**: It is true that 9 of the 12 stopped trades had an average favorable excursion of $+7.13\%$ before stopping out. However, concluding that adding a trailing stop at $+5\%$ or $+1.0\times \text{ATR}$ is the "fix" is premature.
2. **Hidden Degradation Risks**:
   - In trend-following and VCP breakout trading, winning compounders regularly retest their breakout level or 20 EMA before making multi-week runs toward $+20\%$ targets.
   - Moving a stop to breakeven at $+5\%$ frequently cuts off winning trades prematurely during normal market noise.
   - Turning three $+18\%$ winning trades into $0\%$ breakeven exits would immediately collapse the $+4.29\%$ expectancy into negative territory.
3. **Correct Designation**: A trailing breakeven stop is an **untested hypothesis for shadow evaluation**, NOT an established cure for strategy drawdowns.

---

## 5. Purging Marketing Jargon: "Institutional-Grade Asymmetry"

The phrase *"confirming institutional-grade asymmetry"* has been formally **expunged**.

A $2.32:1$ average win/loss payoff ratio across 28 historical trades (heavily clustered across 4 dates and buoyed by look-ahead fundamental data) does not prove institutional grade anything. Institutional systems are validated across thousands of trades across multiple economic cycles, factoring in slippage, execution drag, margin costs, and market impact.

---

## 6. Learning vs. Adaptation vs. Iteration

To eliminate ambiguity regarding ArxTerminal’s evolution, we establish the following strict taxonomy:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          SYSTEM EVOLUTION TAXONOMY                                     │
├─────────────────┬─────────────────────────────────────────────────┬────────────────────┤
│ Classification  │ Mechanism                                       │ ArxTerminal Status │
├─────────────────┼─────────────────────────────────────────────────┼────────────────────┤
│ 1. Learning     │ Autonomous online parameter/weight update driven│ ABSENT             │
│                 │ directly by observed trade outcomes.            │ (Class C)          │
├─────────────────┼─────────────────────────────────────────────────┼────────────────────┤
│ 2. Adaptation   │ Algorithmic, rule-encoded parameter shifting    │ MINIMAL / ABSENT   │
│                 │ in response to detected market regimes.         │ (Fixed parameters) │
├─────────────────┼─────────────────────────────────────────────────┼────────────────────┤
│ 3. Iteration    │ Manual human developer analysis of failures,    │ SOLE MECHANISM     │
│                 │ followed by code commits and new deployments.   │ (Phases 21 to 26)  │
└─────────────────┴─────────────────────────────────────────────────┴────────────────────┘
```

ArxTerminal evolves exclusively through **human software iteration**, not machine learning or autonomous adaptation.

---

## 7. Dual-Hash Cryptographic Verification: Scope & Limitations

The initial report praised the dual-hash architecture:
- `decisionSnapshotHash`: SHA-256 of trade parameters ($P_0, Stop, TP_1, TP_2, Conf$).
- `inputsSnapshotHash`: SHA-256 of input metadata.

### The Adversarial Caveat
- Dual hashes prove that **the recorded decision has not been modified since it was written to `paper_trading_ledger.json`**.
- They do **NOT** prove that the underlying market environment has been preserved if raw price candles or fundamentals in SQLite are updated in-place.
- To achieve true cryptographic provenance, future versions of the ledger must store the **complete serialized input payload** (raw price arrays and raw SEC filing dates) rather than just a hash of metadata.

---

## 8. Revised Executive Verdict

> **“Can we trust the evidence telling us that ArxTerminal is good?”**

### The Unvarnished Truth:
1. **We CANNOT trust the historical +4.29% expectancy and 2.51 profit factor as proof of edge**:
   - The historical benchmark was contaminated by **look-ahead fundamental snapshots** (using September 2026 fundamentals on early 2026 cutoffs).
   - The 28 actionable setups were **heavily clustered across just 4 calendar dates** ($N_{eff} \approx 4\text{–}5$), invalidating claims of statistical independence.
   - The confirmation candle gate was tuned in Phase 23 by developer inspection of this exact market period.
2. **We CANNOT evaluate live performance yet**:
   - Zero market sessions have elapsed since launch. All 9 live recommendations are open.
3. **What we CAN verify**:
   - The production codebase is cleanly frozen at commit `4e36862`.
   - The 9 live recommendations are immutably registered in `paper_trading_ledger.json` with point-in-time hashes, preventing future look-ahead tampering.
   - The system is an iterative rules engine, not an autonomous learning system.

---

## 9. Revised Roadmap: The 3 Genuine Priorities for the Next 30–90 Days

1. **Priority 1: Pure Prospective Forward Observation (Uncontaminated Baseline)**
   - Keep commit `4e36862` strictly frozen.
   - Harvest daily candles across the 9 live positions for the full 20 trading sessions.
   - This prospective cohort is the **only true, uncontaminated out-of-sample test** the system has ever had.
2. **Priority 2: Construct a Point-in-Time Historical Database (Fixing the Data Leakage)**
   - Build historical snapshot tables for fundamentals (`as_of_date` SEC filing dates), ensuring backtests cannot see future financial statements.
   - Re-run the Phase 24 benchmark strictly under point-in-time constraints to measure the true, leak-free historical expectancy.
3. **Priority 3: Shadow-Only Trailing Stop Simulation**
   - Model the proposed breakeven trailing stop against both live forward data and clean historical data.
   - Measure whether it increases or decreases net expectancy before writing a single line of production trading code.
