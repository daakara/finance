# ArxTerminal Findings Backlog: Forensic Integrity & Evaluation Audit

**Audit Date**: September 6, 2026
**Auditor**: Quantitative Risk Guardian (`quant-guardian`) & Methods Auditor
**Baseline**: Git Tag `v2.4.0-phase24-freeze` (Commit `4e36862`)
**Scope**: Production System Integrity, Prediction Performance, Architecture, and Observability

---

## Severity Definitions

- **P0 — Critical / Integrity**: Defects that render predictions, data provenance, or post-launch evaluation unreliable or invalid.
- **P1 — Major**: Issues materially degrading prediction quality, risk management, or client-side trade execution.
- **P2 — Improvement**: High-value enhancements that improve system observability, data pipeline resilience, or statistical tracking.
- **P3 — Nice-to-Have**: Minor enhancements, UI polish, or cosmetic improvements.

---

## Prioritized Findings Matrix

| Finding ID | Title | Category | Severity | Confidence | Affected Components |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P0-1** | Zero-Session Live Production Horizon | Operational | **P0** | **HIGH (100%)** | `paper_trading_ledger.json` |
| **P0-2** | Fundamental Look-Ahead Contamination | Data Integrity | **P0** | **HIGH (100%)** | `asset_factor_snapshots` |
| **P0-3** | Cross-Sectional Setup Clustering ($N_{eff} \approx 4\text{–}6$) | Methodology | **P0** | **HIGH (100%)** | `phase24_out_of_sample_validation.py` |
| **P1-1** | Strategy In-Sample Heuristic Tuning | Methodology | **P1** | **HIGH (100%)** | `optimal_execution.py` (Commit `4e36862`) |
| **P1-2** | Confluence Score Over-Confidence (-37% Error)| Calibration | **P1** | **HIGH (100%)** | `confluence_engine.py` |
| **P1-3** | Untested Trailing Stop Hypothesis | Strategy | **P1** | **HIGH (95%)** | `optimal_execution.py` |
| **P1-4** | Cloudflare Edge Cache Invalidation Gap | Architecture | **P1** | **HIGH (90%)** | `api/routes/screener.py` |
| **P2-1** | Automated Forward Harvester Cron | Observability | **P2** | **HIGH (95%)** | `experiment_ledger.py` |
| **P2-2** | Missing Point-in-Time SEC XBRL Store | Infrastructure | **P2** | **HIGH (100%)** | `market_db.py`, `sec_edgar_fetcher.py` |
| **P2-3** | SQLite Concurrent Reader/Writer Lock | Storage | **P2** | **MEDIUM (80%)** | `market_db.py` |
| **P3-1** | Confluence Naming Rebranding (Ranking vs Conf)| UX / Copy | **P3** | **HIGH (100%)** | UI components |

---

## Detailed Findings

### P0-1: Zero-Session Live Production Horizon
- **Finding**: ArxTerminal launched Friday, September 4, 2026 at 17:32 UTC (post-market close). Between launch and the audit cutoff (Sunday, September 6, 2026 at 13:34 UTC), zero market sessions have traded. All 9 live recommendations remain open with 0.00% excursion.
- **Evidence**: `paper_trading_ledger.json` shows `sessionsObserved: 0`, `resolvedOutcome: null`, `currentPrice == entryPrice`.
- **Impact**: Any claim regarding live production profitability, win rate, or live edge is mathematically unprovable today.
- **Recommended Solution**: Maintain the absolute freeze (`v2.4.0-phase24-freeze`) without code alterations across the full 20-session prospective evaluation horizon.
- **Dependencies**: Real-world exchange tape progression.
- **Validation Method**: Automated daily forward observation updates from SQLite OHLCV candles.

---

### P0-2: Fundamental Look-Ahead Contamination (100% of Setups)
- **Finding**: The historical evaluation in Phase 24 sliced price candles historically ($T-160$ to $T-80$) but retrieved factor scores from `asset_factor_snapshots`. That table contains only September 2026 factor data. 28 of 28 historical actionable setups (100.0%) were evaluated using future fundamental scores.
- **Evidence**: `scratch/generate_contamination_table.py` verified that all 28 setups used factor rows carrying timestamps from September 1–6, 2026.
- **Impact**: The reported $+4.29\%$ historical expectancy cannot be trusted. It is contaminated by future survivorship and quality knowledge.
- **Recommended Solution**: Formally declare that a clean historical fundamental backtest is currently impossible. Build a point-in-time SEC XBRL database table carrying `filing_acceptance_date`.
- **Dependencies**: New point-in-time database schema.
- **Validation Method**: Point-in-time temporal assertion test enforcing `filing_date <= cutoff_date`.

---

### P0-3: Cross-Sectional Setup Clustering & Non-Independence ($N_{eff} \approx 4\text{–}6$)
- **Finding**: 22 of the 28 historical setups (78.6%) occurred on just 4 calendar dates (8 on March 13, 6 on Feb 12, 4 on April 13, 4 on Jan 14).
- **Evidence**: `scratch/analyze_clustering.py` demonstrated that when SPY gained $+7.75\%$ (April 13), 100% of setups won. When SPY fell $-2.79\%$ (Feb 12), 83.3% of setups stopped out.
- **Impact**: Effective degrees of freedom is $N_{eff} \approx 4\text{–}6$ macro events. The sample size is statistically powerless to prove durable edge.
- **Recommended Solution**: Require evaluations to span $\ge 20$ distinct calendar dates across multiple market regimes before calculating aggregate statistics.
- **Dependencies**: Prospective forward tracking over 60–90 days.
- **Validation Method**: Intraclass correlation and cluster standard error calculation.

---

### P1-1: Strategy In-Sample Heuristic Tuning (Confirmation Candle Rule)
- **Finding**: Commit `4e36862` (Phase 23) introduced the confirmation candle gate specifically to eliminate the 76.2% stop hit rate observed during Phase 22 testing on this exact SQLite dataset. Evaluating it on earlier historical slices ($T-160$ to $T-80$) was an in-sample calibration, not an untouched out-of-sample test.
- **Evidence**: Git log commit message for `4e36862` and `phase23_decision_quality_audit.md`.
- **Impact**: The historical performance reflects fitting heuristics to known historical patterns.
- **Recommended Solution**: Burn the historical SQLite dataset for tuning. Establish strict 4-tier data partitioning (Development / Validation / Holdout / Prospective Live).
- **Dependencies**: Governance standard `ARX-GOV-2026.1`.
- **Validation Method**: Track only prospective live signals going forward.

---

### P1-2: Confluence Score Over-Confidence (-37% Error)
- **Finding**: Confluence scores of $85.0\%-90.0\%$ produced an empirical win rate of only $50.0\%$ across the historical sample, representing a -37.1 percentage point over-confidence error.
- **Evidence**: `scratch/phase24_out_of_sample_validation.py` confluence calibration breakdown.
- **Impact**: Users and risk systems interpreting confluence scores as win probabilities will severely misprice risk.
- **Recommended Solution**: Rebrand "Confidence" to "Confluence Rank" across UI and API. In the future, calibrate probabilistic models using Platt scaling or isotonic regression on holdout data.
- **Dependencies**: Frontend copy and API schema.
- **Validation Method**: Brier score calculation on prospective outcomes.

---

### P1-3: Untested Trailing Stop Hypothesis
- **Finding**: The finding that 9 of 12 stopped trades achieved an average favorable excursion of $+7.13\%$ before pulling back was prematurely declared a "root cause" requiring trailing stops.
- **Evidence**: No simulation was conducted to determine how many $+15\%$ to $+20\%$ winners would be prematurely aborted at breakeven.
- **Impact**: Premature production deployment could severely damage strategy expectancy.
- **Recommended Solution**: Test trailing breakeven stops strictly in shadow mode. Do not deploy to production unless shadow net expectancy expands.
- **Dependencies**: Shadow testing framework.
- **Validation Method**: Comparative shadow vs production backtest.

---

### P1-4: Cloudflare Edge Cache Invalidation Gap
- **Finding**: Screener GET endpoint sets `Cloudflare-CDN-Cache-Control: max-age=120`, allowing edge caches to serve stale execution levels for 2 minutes.
- **Evidence**: `api/routes/screener.py` lines 83–85, 110–112.
- **Impact**: High-volatility market open trades could execute on stale stop/entry levels.
- **Recommended Solution**: Lower edge cache TTL during trading hours (13:30–20:00 UTC) to `max-age=10, s-maxage=30`.
- **Dependencies**: FastAPI response headers.
- **Validation Method**: Header inspection in simulated trading sessions.
