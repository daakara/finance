# ARX Model Governance — Phase 26 Promotion Criteria & Production Readiness Specification
**Document ID**: `GOV-PHASE26-LIQUIDITYGUARD-PROM`
**Current Architecture Status**: `SHADOW_OBSERVATION_ONLY` (Frozen)
**Baseline Engine Version**: `v2.4.0-phase24-freeze` (`commit 4e36862`)
**Target Engine Version**: `v2.5.0-candidate`
**Author**: Antigravity Quantitative Systems Engineering & Governance
**Enforcement Level**: Fail-Closed (Binding Model Invariant)

---

## 1. Executive Summary & Epistemic Boundary

Under Phase 24 and Phase 25 Model Governance, the ARX Technical Signal and Confluence Decision Engine is **strictly frozen** (`v2.4.0-phase24-freeze`). The directional setup rules (Minervini VCP, Stage Analysis, 14-period ATR volatility corridors, stop-loss floors, asymmetric take-profit targets, and 4-stream Bayesian confluence scoring) are frozen to guarantee uncontaminated prospective evaluation.

The system strictly decouples two fundamentally different questions:
1. **Decision Model (Frozen Alpha Engine)**: *"Is this asset technically an actionable setup?"*
2. **Execution Diagnostic Layer (`LiquidityGuard`)**: *"If you trade it, what estimated historical trading friction might you encounter?"*

### Epistemic Classification: Historical Trading Liquidity Diagnostic
`LiquidityGuard` must **never** be described as an "execution risk model" in the strong predictive sense. Because its primary inputs are daily OHLCV bars, it cannot observe real-time Level-2 queue depth, displayed liquidity at the inside, hidden liquidity, or actual institutional market impact:
$$\text{Daily OHLCV} \longrightarrow \text{Estimated Historical Trading Friction} \quad (\text{Defensible})$$
$$\text{Daily OHLCV} \centernot\longrightarrow \text{Actual Execution Risk} \quad (\text{Overstated})$$

This boundary is made explicit across the codebase, API contracts, UI components, and documentation.

---

## 2. Independent Acceptance Criteria: Gate A (Safety) vs. Gate B (Utility)

| Criterion | Formal Question | Current Empirical Status | Verdict |
| :--- | :--- | :--- | :--- |
| **Gate A — Safety** | Does `LiquidityGuard` modify or contaminate the frozen directional decision engine in any scenario? | 100% Non-interference proven across full decision vector ($N > 1,000$). Decoupled. | **PASSED (Locked)** |
| **Gate B — Utility** | Does `LiquidityGuard`'s diagnostic information change something demonstrably useful about execution outcomes? | Catalog universe imposes an upstream $\$250\text{M}$ MCAP / $100\text{K}$ volume floor; historical tests show 100% `HIGH_TRADING_LIQUIDITY`. Efficacy on illiquid setups is not yet empirically proven. | **UNPROVEN (Shadow Mode Retained)** |

**Scientific Conclusion**: *Safe and potentially useful diagnostic; execution efficacy not yet established.*
This position prevents premature promotion based on unvalidated assumptions.

---

## 3. Mathematical & Operational Diagnostic Invariants

### 3.1. Participation Advisory Threshold (Operational Heuristic)
$$\text{Participation Rate} = \frac{\text{OrderSize}_{\text{USD}}}{ADV_{20D,\text{USD}}}$$
- The $1.0\%$ threshold is designated as a **Participation Advisory Threshold**, **not** a universal physical law of market impact.
- Actual market impact depends upon spread, intraday volume distribution, volatility, order type, execution horizon, market regime, free float, and venue fragmentation.
- The UI presents this threshold purely as an operational heuristic for retail sizing.

### 3.2. 20-Day vs. 5-Day ADV and Liquidity Trend Diagnostic
To resolve distortion from single-session volume outliers (e.g. 19 days at $\$100\text{K}$ and 1 day at $\$5\text{M}$), `LiquidityGuard` computes both 20-day and 5-day baselines:
$$\text{LiquidityTrend} = \frac{ADV_{5D}}{ADV_{20D}}$$
- $\text{LiquidityTrend} > 1.0$: Expanding short-term volume liquidity.
- $\text{LiquidityTrend} < 1.0$: Contracting short-term volume / liquidity deterioration.
- Purely an observational diagnostic; does not alter decision states.

### 3.3. Amihud Dimensional & Unit Semantics
Let $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ be daily fractional price return, and $D_t = P_t \times V_t$ be daily dollar volume traded.
$$ILLIQ_{\text{raw}} = \frac{1}{N} \sum_{t=1}^N \frac{|R_t|}{D_t} \quad \left[\frac{\text{fractional return}}{\$ \text{ traded}}\right]$$
$$ILLIQ_{\text{scaled}} = ILLIQ_{\text{raw}} \times 10^6 \quad \left[\frac{\text{fractional return}}{\$1\text{M dollar volume traded}}\right]$$
- Example: $ILLIQ_{\text{scaled}} = 0.0004$ represents an expected fractional return move of $0.0004$ ($4\text{ bps}$) per $\$1\text{M}$ traded.
- The platform documents this strictly as `return / $1M traded`, not percentage return.

---

## 4. Production Readiness Gate Audit Matrix

Every check was formally implemented and verified via automated test suite `tests/test_liquidity_production_readiness_audit.py` (20/20 PASSED):

| Level | Check ID | Audit Description | Test Function | Status |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | P0-1 | Full Decision-Vector Non-Interference: $\text{Decision}(P, V_{\text{high}}) = \text{Decision}(P, V_{\text{low}})$ across all 18 keys | `test_p0_1_full_decision_vector_non_interference` | ✅ PASSED |
| **P0** | P0-2 | API Serialization/Deserialization roundtrip across all 4 liquidity tiers | `test_p0_2_api_serialization_roundtrip_all_tiers` | ✅ PASSED |
| **P0** | P0-3 | Unknown/missing-data semantics audit (empty DF, None, $<3$ sessions produce `UNKNOWN_LIQUIDITY`) | `test_p0_3_unknown_missing_data_semantics` | ✅ PASSED |
| **P0** | P0-4 | Amihud dimensional/unit audit ($ILLIQ_{\text{scaled}} = ILLIQ_{\text{raw}} \times 10^6$ return/$\$1\text{M}$) | `test_p0_4_amihud_dimensional_unit_audit` | ✅ PASSED |
| **P0** | P0-5 | Zero hidden liquidity dependencies anywhere in frozen decision engine | `test_p0_5_no_hidden_liquidity_dependency_in_frozen_engine` | ✅ PASSED |
| **P0** | P0-6 | Split-adjusted price and volume alignment guard ($|\Delta P| > 80\%$ filter) | `test_p0_6_split_adjusted_price_and_volume_alignment` | ✅ PASSED |
| **P0** | P0-7 | Point-in-time temporal verification & zero look-ahead in 20D rolling window | `test_p0_7_no_lookahead_in_rolling_windows` | ✅ PASSED |
| **P0** | P0-8 | Signal-time liquidity is permanently immutable in forward ledger | `test_p0_8_signal_time_liquidity_immutability` | ✅ PASSED |
| **P0** | P0-9 | Forward liquidity observations have zero effect on trade outcome resolutions | `test_p0_9_forward_liquidity_cannot_affect_outcomes` | ✅ PASSED |
| **P0** | P0-10 | Cryptographic dual-hashes (`decisionSnapshotHash`, `inputsSnapshotHash`) cover all immutable fields | `test_p0_10_dual_hash_coverage` | ✅ PASSED |
| **P1** | P1-11 | Liquidity deterioration trajectory ($HIGH \to MODERATE \to EXECUTION\_RISK$, trend $< 0.5$) | `test_p1_11_liquidity_deterioration_trajectory` | ✅ PASSED |
| **P1** | P1-12 | Liquidity recovery trajectory ($EXECUTION\_RISK \to MODERATE \to HIGH$, trend $> 1.5$) | `test_p1_12_liquidity_recovery_trajectory` | ✅ PASSED |
| **P1** | P1-13 | Sudden volume spike behavior ($> 2.5\times$ 20D mean flags `is_volume_spike`) | `test_p1_13_sudden_volume_spike_behavior` | ✅ PASSED |
| **P1** | P1-14 | Intermittent stale/missing sessions resilience (zero-volume days handled gracefully) | `test_p1_14_stale_missing_sessions_resilience` | ✅ PASSED |
| **P1** | P1-15 | Corporate action overnight gap guard (95% drop sanitized from Amihud numerator) | `test_p1_15_corporate_action_gap_guard` | ✅ PASSED |
| **P1** | P1-16 | Ultra-high priced equities ($>\$500\text{k}$ spot, e.g. BRK.A) evaluate accurately | `test_p1_16_very_high_priced_stocks` | ✅ PASSED |
| **P1** | P1-17 | Penny / micro-cap equities ($<\$1$ spot, $\$10\text{k}$ ADV) evaluate to `EXECUTION_RISK` | `test_p1_17_penny_micro_cap_stocks` | ✅ PASSED |
| **P1** | P1-18 | Order size participation advisory spectrum ($0.1\%$ to $10.0\%$ ADV) scales linearly | `test_p1_18_order_size_participation_spectrum` | ✅ PASSED |
| **P1** | P1-19 | UX wording strictly advisory; zero claims of guaranteed slippage or untradeability | `test_p1_19_ux_wording_advisory_invariants` | ✅ PASSED |
| **P1** | P1-20 | Graceful neutral state on `UNKNOWN_LIQUIDITY` (hazard=False, slate badge) | `test_p1_20_disabled_gracefully_when_unknown` | ✅ PASSED |

---

## 5. The 5 Non-Negotiable Promotion Gates (Phase 26)

Promotion from `SHADOW_OBSERVATION` to `ACTIVE_DECISION_GATE` requires unanimous clearance across all five gates:

1. **Gate 1: Comprehensive Non-Interference Verification**: Zero mutation across $\ge 1,000$ setups.
2. **Gate 2: Prospective Forward Validation Cohort**: Accumulation of $N_{\text{resolved}} \ge 50$ completed paper trades in the cryptographic ledger.
3. **Gate 3: Net Economic Value & Alpha Trade-Off Proof**: Statistical proof ($p < 0.05$) that slippage reduction exceeds the opportunity cost of rejected winning setups:
   $$\mathbb{E}[R_{\text{filtered}} - C_{\text{slippage}}] - \text{OpportunityCost}(\text{rejected winners}) > \mathbb{E}[R_{\text{raw}} - C_{\text{slippage}}]$$
4. **Gate 4: Real-Time Microstructure Integration**: Direct ingestion of Level-2 NBBO spread or depth rather than backward-looking daily Amihud proxies.
5. **Gate 5: Dual-Key Governance Signing**: Model version increment to `v2.5.0` and immutable archival of the Phase 25 frozen ledger.

---

## 6. Phase 25 Freeze Declaration

**The `LiquidityGuard` methodology, parameters, and thresholds are now permanently FROZEN for Phase 25**:
- $ADV_{\text{HIGH\_FLOOR}} = \$2,000,000$
- $ADV_{\text{MIN\_SAFETY\_FLOOR}} = \$500,000$
- $ILLIQ_{\text{TRAP}} = 5.0 \times 10^{-6}$
- $ILLIQ_{\text{THIN}} = 1.0 \times 10^{-6}$
- $\text{Participation Advisory Threshold} = 0.01$ ($1\%$ of ADV)
- Rolling windows: 20-day baseline, 5-day trend

**No further heuristic tuning against the existing catalog dataset is permitted.** Forward observations will accumulate prospective evidence in `paper_trading_ledger.json` without post-hoc modification until the Phase 26 cohort evaluation milestone is reached.

---

## 7. Phase 26 Prospective Validation Infrastructure

To scientifically evaluate `LiquidityGuard` without contaminating the frozen `v2.4.0-phase24-freeze` engine, the platform implements dedicated prospective observation infrastructure (`Phase26ValidationEngine` in `analyst_dashboard/governance/liquidity_validation.py`).

### 7.1. Decoupling of Three Core Concepts
The platform strictly distinguishes:
1. **Realized Execution Friction**: Observed fill prices from broker execution:
   $$\text{Slippage}_{\text{bps}} = \frac{|P_{\text{fill}} - P_{\text{reference}}|}{P_{\text{reference}}} \times 10{,}000$$
2. **Estimated Execution Friction**: Heuristic cost model proxy based on historical Amihud illiquidity and order participation rate:
   $$\text{Cost}_{\text{est, bps}} = \text{Baseline}(\text{Grade}) + \text{Participation} \times ILLIQ_{\text{scaled}} \times 10$$
3. **Liquidity Diagnostic**: Historical 20-day OHLCV classifications (`HIGH`, `MODERATE`, `EXECUTION_RISK`, `UNKNOWN`).

Under no circumstances is estimated friction substituted for realized friction, nor is causality asserted between diagnostic tiers and execution outcomes without empirical proof.

### 7.2. Experiment 26-A — Realized Execution Friction
- **Hypothesis**:
  $$H_0: \mu_{\text{slippage}}(\text{EXECUTION\_RISK}) = \mu_{\text{slippage}}(\text{HIGH\_TRADING\_LIQUIDITY})$$
  $$H_1: \mu_{\text{slippage}}(\text{EXECUTION\_RISK}) > \mu_{\text{slippage}}(\text{HIGH\_TRADING\_LIQUIDITY})$$
- **Captured Schema**:
  - `observationId`: Unique observation identifier (`{signalId}_exec_{idx}`)
  - `signalId`: Foreign key to registered signal
  - `symbol`: Ticker symbol
  - `signalTimestamp`: Signal-time UTC ISO timestamp
  - `referencePrice`: Signal arrival / entry midpoint price ($>0$)
  - `executionTimestamp`: Fill UTC ISO timestamp ($t_{\text{exec}} \ge t_{\text{signal}}$)
  - `fillPrice`: Realized fill price ($>0$ or `null` if unfilled)
  - `side`: `"BUY"` or `"SELL"`
  - `orderSizeUsd`: Order dollar value
  - `adv20dUsd`, `adv5dUsd`, `liquidityTrend`: Point-in-time ADV volume metrics
  - `amihudIlliqRaw`, `amihudIlliqScaled`: Point-in-time Amihud ratios
  - `liquidityGrade`: Classification tier at signal time
  - `participationRate`: Order size / 20D ADV
  - `slippageBps`: Realized slippage in basis points (`null` if unfilled)
  - `executionSource`: `"BROKER_FILL"`, `"SIMULATED_FILL"`, `"MANUAL_RECORD"`, or `"UNFILLED"`
  - `isSimulated`: Boolean flag separating simulated fills from real broker executions
  - `specVersion`: `"LiquidityGuard Shadow Spec v1.0"`
- **Missing Data Treatment**:
  - If an order is unfilled, `fillPrice` is `null` and `slippageBps` is `null`.
  - Missing data is NEVER fabricated or imputed.
- **Statistical Outputs**:
  - Sample size $N$ per cohort
  - Real vs. simulated fill counts
  - Mean slippage $\bar{s}$, median slippage $\tilde{s}$, standard deviation $\sigma$, standard error $\text{SE} = \frac{\sigma}{\sqrt{N}}$
  - $95\%$ Confidence Interval: $[\bar{s} - 1.96 \cdot \text{SE}, \; \bar{s} + 1.96 \cdot \text{SE}]$
  - Pairwise Welch's $t$-test ($p$-value) and Cohen's $d$ effect size
  - Safe evaluation: Cohorts with $N < 2$ return `"INSUFFICIENT_SAMPLE_FOR_INFERENCE"` without division-by-zero or NaNs.

### 7.3. Experiment 26-B — Economic Counterfactual
- **Hypothesis**:
  $$H_0: \text{Sharpe}_{\text{net}}(\text{Hypothetically Gated}) \le \text{Sharpe}_{\text{net}}(\text{Ungated})$$
  $$H_1: \text{Sharpe}_{\text{net}}(\text{Hypothetically Gated}) > \text{Sharpe}_{\text{net}}(\text{Ungated})$$
- **Protocol**:
  - For every resolved forward signal, preserve the original ungated return:
    $$R_{\text{gross}} = \text{realizedReturnPct}$$
    $$R_{\text{net}} = R_{\text{gross}} - \frac{\text{Slippage}_{\text{bps}}}{100.0}$$
  - Calculate hypothetical outcome if `LiquidityGuard` had gated the setup (`EXECUTION_RISK` or `executionHazard = True`), WITHOUT actually modifying the production trade.
- **Comparative Metrics**:
  - **Ungated Portfolio** vs. **Gated Portfolio**:
    - Trade count, win rate, mean gross/net return, total gross/net return
    - Net Profit Factor, Net Sharpe Ratio, Max Drawdown
  - **Trade-Off Analysis**:
    - $\text{Excluded Winners}$: Winning setups rejected by hypothetical gate (Opportunity Cost)
    - $\text{Avoided Losers}$: Losing setups avoided by hypothetical gate (Saved Loss)
    - $\text{Net Economic Benefit} = \sum \text{Losses Avoided} - \sum \text{Opportunity Cost} + \Delta \text{Execution Friction Saved}$
- **Non-Mutating Invariant**:
  - The counterfactual calculation is strictly read-only. It NEVER mutates underlying `signals`, forward tracking, or historical ledger states.

---

## 8. Epistemic Status, Evidence Standards & Promotion Criteria

| Condition | Verdict | Required Action |
| :--- | :--- | :--- |
| $\ge 50$ resolved signals, $p < 0.05$ on slippage, and $\text{NetBenefit} > 0$ with higher Sharpe | **PROMOTION CANDIDATE** | Convene governance committee for Gate 4 & 5 sign-off |
| Opportunity cost of excluded winners exceeds losses avoided ($\text{OppCost} > \text{LossesAvoided}$) | **PROMOTION PERMANENTLY BLOCKED** | Retain shadow observation mode; gating destroys strategy alpha |
| No statistical difference in realized slippage between cohorts ($p \ge 0.05$) | **HYPOTHESIS NOT SUPPORTED** | Diagnostic cannot be used as an execution filter; retain informational label only |
| Sample size $N < 50$ or insufficient risk cohort fills | **ACTIVE OBSERVATION** | Continue prospective forward logging; no promotion decisions allowed |

---

*End of Governance Specification `GOV-PHASE26-LIQUIDITYGUARD-PROM`.*
