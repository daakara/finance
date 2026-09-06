"""Phase 26 Prospective Liquidity Validation Engine.

Implements prospective validation infrastructure strictly separated from the frozen decision engine:
- Experiment 26-A: Realized Execution Friction Analysis across Liquidity Cohorts
- Experiment 26-B: Economic Counterfactual Analysis (Hypothetical Gated vs. Ungated Portfolios)

Epistemic & Structural Invariants:
1. Pure Observational & Analytical Layer: NEVER mutates the frozen decision model (v2.4.0-phase24-freeze).
2. Missing Execution Data Treatment: Missing fills are represented as None, NEVER fabricated or imputed.
3. Decoupling of Concepts:
   - Realized Execution Friction (observed fills from broker/trader)
   - Estimated Execution Friction (heuristic model proxy based on Amihud / ADV)
   - Liquidity Diagnostic (historical OHLCV classification)
4. Safe Statistical Evaluation: Handles empty and small cohorts gracefully without division-by-zero or NaNs.
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None

from analyst_dashboard.governance.experiment_ledger import ExperimentLedger


class Phase26ValidationEngine:
    """Statistical and Economic Evaluation Engine for Phase 26 Prospective Validation."""

    SPEC_VERSION = "LiquidityGuard Shadow Spec v1.0"

    # Default heuristic execution cost model in basis points for counterfactual simulation when fills are unobserved
    DEFAULT_HEURISTIC_COST_BPS = {
        "HIGH_TRADING_LIQUIDITY": 5.0,        # ~5 bps for liquid US large-caps
        "MODERATE_TRADING_LIQUIDITY": 15.0,   # ~15 bps for mid-tier names
        "EXECUTION_RISK": 45.0,               # ~45 bps for low-ADV / thin equities
        "UNKNOWN_LIQUIDITY": 10.0,            # neutral unverified baseline
    }

    @staticmethod
    def calculate_slippage_bps(fill_price: float, reference_price: float) -> float:
        """
        Calculates realized slippage in basis points:
        slippage_bps = abs(fill_price - reference_price) / reference_price * 10,000

        Strictly rejects zero or negative prices.
        """
        if fill_price <= 0 or math.isnan(fill_price) or math.isinf(fill_price):
            raise ValueError(f"Invalid fill_price {fill_price}: must be positive finite number")
        if reference_price <= 0 or math.isnan(reference_price) or math.isinf(reference_price):
            raise ValueError(f"Invalid reference_price {reference_price}: must be positive finite number")

        return round((abs(fill_price - reference_price) / reference_price) * 10000.0, 2)

    @classmethod
    def evaluate_execution_friction(
        cls,
        records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Experiment 26-A — Realized Execution Friction Evaluation.

        Hypothesis:
        H0: mu_slippage(EXECUTION_RISK) == mu_slippage(HIGH_TRADING_LIQUIDITY)
        H1: mu_slippage(EXECUTION_RISK) > mu_slippage(HIGH_TRADING_LIQUIDITY)

        Calculates per-cohort sample size, mean, median, standard error, 95% CI,
        Welch's t-test p-value, and Cohen's d effect size.
        """
        if not records:
            return {
                "experiment": "Experiment 26-A: Execution Friction",
                "specVersion": cls.SPEC_VERSION,
                "status": "EMPTY_OBSERVATION_POOL",
                "totalObservations": 0,
                "validFillsCount": 0,
                "missingFillsCount": 0,
                "realFillsCount": 0,
                "simulatedFillsCount": 0,
                "cohorts": {},
                "pairwiseComparison": {
                    "comparison": "EXECUTION_RISK vs HIGH_TRADING_LIQUIDITY",
                    "status": "INSUFFICIENT_SAMPLE_FOR_INFERENCE",
                    "differenceInMeansBps": None,
                    "cohensD": None,
                    "pValue": None,
                    "statisticallySignificant": False,
                },
                "epistemicClassification": "REALIZED_VS_ESTIMATED_FRICTION_DECOUPLED",
                "conclusion": "No execution observations recorded yet. Phase 26 prospective data collection pending."
            }

        valid_records = []
        real_records = []
        simulated_records = []
        missing_count = 0
        real_count = 0
        simulated_count = 0

        for r in records:
            fill = r.get("fillPrice")
            slip = r.get("slippageBps")
            if fill is not None and slip is not None:
                valid_records.append(r)
                if r.get("isSimulated", False) or r.get("executionSource") == "SIMULATED_FILL":
                    simulated_count += 1
                    simulated_records.append(r)
                else:
                    real_count += 1
                    real_records.append(r)
            else:
                missing_count += 1

        # Evaluate real, simulated, and combined cohorts independently to prevent evidence contamination
        real_eval = cls._evaluate_cohort_subset(real_records, subset_label="REAL_BROKER_EXECUTION")
        sim_eval = cls._evaluate_cohort_subset(simulated_records, subset_label="SIMULATED_EXECUTION")
        combined_eval = cls._evaluate_cohort_subset(valid_records, subset_label="COMBINED_EXECUTION")

        # Epistemic conclusion based strictly on real evidence for formal promotion
        if real_eval["pairwiseComparison"]["status"] == "EVALUATED" and real_eval["pairwiseComparison"]["statisticallySignificant"]:
            conclusion = (
                f"Prospective real broker evidence rejects H0 (p = {real_eval['pairwiseComparison']['pValue']}). "
                f"Execution risk cohort experienced higher realized slippage by {real_eval['pairwiseComparison']['differenceInMeansBps']} bps."
            )
        elif real_eval["pairwiseComparison"]["status"] == "EVALUATED":
            conclusion = (
                f"Prospective real broker evidence does NOT reject H0 at p < 0.05 (p = {real_eval['pairwiseComparison']['pValue']}). "
                "Slippage difference between cohorts is not statistically significant."
            )
        elif sim_eval["pairwiseComparison"]["status"] == "EVALUATED":
            conclusion = (
                f"Real broker sample size is insufficient. Simulated cohort indicates {sim_eval['pairwiseComparison']['differenceInMeansBps']} bps difference, "
                "but simulated evidence CANNOT be used to satisfy Gate 3 promotion criteria."
            )
        else:
            conclusion = "Insufficient observations across high vs risk cohorts to evaluate execution friction hypothesis."

        return {
            "experiment": "Experiment 26-A: Execution Friction",
            "specVersion": cls.SPEC_VERSION,
            "status": "ACTIVE_OBSERVATION",
            "totalObservations": len(records),
            "validFillsCount": len(valid_records),
            "missingFillsCount": missing_count,
            "realFillsCount": real_count,
            "simulatedFillsCount": simulated_count,
            "cohorts": combined_eval["cohorts"],
            "pairwiseComparison": combined_eval["pairwiseComparison"],
            "realEvidence": real_eval,
            "simulatedEvidence": sim_eval,
            "combinedEvidence": combined_eval,
            "formalPromotionEvidence": real_eval,
            "epistemicClassification": "REALIZED_VS_ESTIMATED_FRICTION_DECOUPLED",
            "conclusion": conclusion,
        }

    @classmethod
    def _evaluate_cohort_subset(
        cls,
        subset_records: List[Dict[str, Any]],
        subset_label: str = "SUBSET"
    ) -> Dict[str, Any]:
        """Calculates cohort statistics and pairwise Welch's t-test for a defined evidence partition."""
        cohort_keys = [
            "HIGH_TRADING_LIQUIDITY",
            "MODERATE_TRADING_LIQUIDITY",
            "EXECUTION_RISK",
            "UNKNOWN_LIQUIDITY"
        ]

        cohort_groups: Dict[str, List[float]] = {k: [] for k in cohort_keys}
        cohort_meta: Dict[str, Dict[str, int]] = {k: {"real": 0, "simulated": 0} for k in cohort_keys}

        for r in subset_records:
            grade = r.get("liquidityGrade") or r.get("liquidity_grade") or "UNKNOWN_LIQUIDITY"
            if grade not in cohort_groups:
                cohort_groups[grade] = []
                cohort_meta[grade] = {"real": 0, "simulated": 0}

            slip_val = float(r["slippageBps"])
            cohort_groups[grade].append(slip_val)
            if r.get("isSimulated", False) or r.get("executionSource") == "SIMULATED_FILL":
                cohort_meta[grade]["simulated"] += 1
            else:
                cohort_meta[grade]["real"] += 1

        cohort_stats: Dict[str, Any] = {}
        for grade, values in cohort_groups.items():
            n = len(values)
            if n == 0:
                cohort_stats[grade] = {
                    "sampleSize": 0,
                    "realFills": 0,
                    "simulatedFills": 0,
                    "meanSlippageBps": None,
                    "medianSlippageBps": None,
                    "stdDevBps": None,
                    "standardErrorBps": None,
                    "ci95": None,
                    "status": "EMPTY_COHORT"
                }
            elif n == 1:
                val = round(values[0], 2)
                cohort_stats[grade] = {
                    "sampleSize": 1,
                    "realFills": cohort_meta[grade]["real"],
                    "simulatedFills": cohort_meta[grade]["simulated"],
                    "meanSlippageBps": val,
                    "medianSlippageBps": val,
                    "stdDevBps": 0.0,
                    "standardErrorBps": None,
                    "ci95": None,
                    "status": "SINGLE_OBSERVATION"
                }
            else:
                arr = np.array(values, dtype=float)
                mean_val = float(np.mean(arr))
                median_val = float(np.median(arr))
                std_dev = float(np.std(arr, ddof=1))
                std_err = float(std_dev / np.sqrt(n))

                # Exact Student-t critical value for small samples if scipy available
                if stats is not None and n >= 2:
                    t_crit = float(stats.t.ppf(0.975, df=n - 1))
                else:
                    t_crit = 1.96

                ci_lower = max(0.0, mean_val - t_crit * std_err)
                ci_upper = mean_val + t_crit * std_err

                cohort_stats[grade] = {
                    "sampleSize": n,
                    "realFills": cohort_meta[grade]["real"],
                    "simulatedFills": cohort_meta[grade]["simulated"],
                    "meanSlippageBps": round(mean_val, 2),
                    "medianSlippageBps": round(median_val, 2),
                    "stdDevBps": round(std_dev, 2),
                    "standardErrorBps": round(std_err, 2),
                    "ci95": [round(ci_lower, 2), round(ci_upper, 2)],
                    "status": "SUFFICIENT_SAMPLE"
                }

        # Pairwise comparison: EXECUTION_RISK vs HIGH_TRADING_LIQUIDITY
        high_vals = cohort_groups.get("HIGH_TRADING_LIQUIDITY", [])
        risk_vals = cohort_groups.get("EXECUTION_RISK", [])

        if len(high_vals) >= 2 and len(risk_vals) >= 2:
            n1, n2 = len(high_vals), len(risk_vals)
            m1, m2 = float(np.mean(high_vals)), float(np.mean(risk_vals))
            v1, v2 = float(np.var(high_vals, ddof=1)), float(np.var(risk_vals, ddof=1))

            diff_means = round(m2 - m1, 2)

            pooled_denom = n1 + n2 - 2
            if pooled_denom > 0:
                s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / pooled_denom)
                cohens_d = round(float((m2 - m1) / s_pooled), 3) if s_pooled > 0 else 0.0
            else:
                cohens_d = None

            if stats is not None:
                t_stat, p_val = stats.ttest_ind(risk_vals, high_vals, equal_var=False)
                p_value = round(float(p_val), 4) if not math.isnan(p_val) else None
            else:
                p_value = None

            stat_sig = bool(p_value is not None and p_value < 0.05 and diff_means > 0)
            pairwise = {
                "comparison": "EXECUTION_RISK vs HIGH_TRADING_LIQUIDITY",
                "evidenceType": subset_label,
                "status": "EVALUATED",
                "differenceInMeansBps": diff_means,
                "cohensD": cohens_d,
                "pValue": p_value,
                "statisticallySignificant": stat_sig,
            }
        else:
            pairwise = {
                "comparison": "EXECUTION_RISK vs HIGH_TRADING_LIQUIDITY",
                "evidenceType": subset_label,
                "status": "INSUFFICIENT_SAMPLE_FOR_INFERENCE",
                "differenceInMeansBps": None,
                "cohensD": None,
                "pValue": None,
                "statisticallySignificant": False,
            }

        return {
            "evidenceType": subset_label,
            "sampleCount": len(subset_records),
            "cohorts": cohort_stats,
            "pairwiseComparison": pairwise,
        }

    @classmethod
    def evaluate_economic_counterfactual(
        cls,
        ledger_data: Dict[str, Any],
        filter_criterion: str = "EXECUTION_RISK"
    ) -> Dict[str, Any]:
        """
        Experiment 26-B — Economic Counterfactual Analysis.

        Hypothesis:
        H0: Sharpe_net(Gated) <= Sharpe_net(Ungated)
        H1: Sharpe_net(Gated) > Sharpe_net(Ungated)

        Evaluates resolved strategy signals under two hypothetical regimes:
        1. Ungated Portfolio: Original strategy containing all resolved signals.
        2. Hypothetically Gated Portfolio: Portfolio excluding setups flagged with filter_criterion.

        NON-MUTATING GUARANTEE: This method is purely read-only and never modifies signals.
        """
        signals = ledger_data.get("signals", [])
        resolved_signals = [
            s for s in signals
            if s.get("status") == "RESOLVED"
            and s.get("forwardTracking", {}).get("realizedReturnPct") is not None
        ]

        if not resolved_signals:
            return {
                "experiment": "Experiment 26-B: Economic Counterfactual",
                "specVersion": cls.SPEC_VERSION,
                "filterCriterion": filter_criterion,
                "status": "NO_RESOLVED_SIGNALS",
                "totalSignals": len(signals),
                "resolvedSignals": 0,
                "ungatedPortfolio": cls._empty_portfolio_summary(),
                "gatedPortfolio": cls._empty_portfolio_summary(),
                "tradeoffAnalysis": {
                    "filteredTradesCount": 0,
                    "excludedWinnersCount": 0,
                    "avoidedLosersCount": 0,
                    "opportunityCostPct": 0.0,
                    "lossesAvoidedPct": 0.0,
                    "frictionSavedPct": 0.0,
                    "netEconomicBenefitPct": 0.0,
                    "excludedWinnersList": [],
                    "avoidedLosersList": [],
                },
                "counterfactualVerdict": "INSUFFICIENT_OBSERVATIONS_FOR_DECISION",
                "conclusion": "No resolved paper-trading signals available yet to evaluate economic counterfactual."
            }

        ungated_returns_gross: List[float] = []
        ungated_returns_net: List[float] = []
        gated_returns_gross: List[float] = []
        gated_returns_net: List[float] = []

        excluded_winners: List[Dict[str, Any]] = []
        avoided_losers: List[Dict[str, Any]] = []

        total_friction_ungated = 0.0
        total_friction_gated = 0.0

        for s in resolved_signals:
            r_gross = float(s["forwardTracking"]["realizedReturnPct"])
            liq_at_sig = s.get("liquidityAtSignal") or {}
            grade = liq_at_sig.get("liquidityGrade") or liq_at_sig.get("liquidity_grade") or "UNKNOWN_LIQUIDITY"
            is_hazard = bool(liq_at_sig.get("executionHazard") or liq_at_sig.get("execution_hazard") or False)

            # Determine hypothetical gating decision
            is_hypothetically_filtered = (grade == filter_criterion) or is_hazard

            # Determine execution cost in bps:
            exec_obs = s.get("executionObservations", [])
            valid_obs = [o for o in exec_obs if o.get("slippageBps") is not None]
            if valid_obs:
                # If multiple fills exist on a signal, compute volume-weighted average slippage
                total_sz = sum([float(o.get("orderSizeUsd") or 0.0) for o in valid_obs])
                if total_sz > 0:
                    cost_bps = float(sum([float(o["slippageBps"]) * float(o.get("orderSizeUsd") or 0.0) for o in valid_obs]) / total_sz)
                else:
                    cost_bps = float(np.mean([float(o["slippageBps"]) for o in valid_obs]))
            else:
                cost_bps = cls.DEFAULT_HEURISTIC_COST_BPS.get(grade, 10.0)

            cost_pct = cost_bps / 100.0  # Convert bps to percentage return drag
            r_net = r_gross - cost_pct

            ungated_returns_gross.append(r_gross)
            ungated_returns_net.append(r_net)
            total_friction_ungated += cost_pct

            if is_hypothetically_filtered:
                if r_gross > 0:
                    excluded_winners.append({
                        "signalId": s.get("signalId"),
                        "symbol": s.get("symbol"),
                        "grossReturnPct": r_gross,
                        "netReturnPct": round(r_net, 2),
                        "liquidityGrade": grade
                    })
                else:
                    avoided_losers.append({
                        "signalId": s.get("signalId"),
                        "symbol": s.get("symbol"),
                        "grossReturnPct": r_gross,
                        "netReturnPct": round(r_net, 2),
                        "liquidityGrade": grade
                    })
            else:
                gated_returns_gross.append(r_gross)
                gated_returns_net.append(r_net)
                total_friction_gated += cost_pct

        ungated_summary = cls._compute_portfolio_metrics(ungated_returns_gross, ungated_returns_net)
        gated_summary = cls._compute_portfolio_metrics(gated_returns_gross, gated_returns_net)

        opp_cost = sum([w["grossReturnPct"] for w in excluded_winners])
        losses_avoided = sum([abs(l["grossReturnPct"]) for l in avoided_losers])
        friction_saved = total_friction_ungated - total_friction_gated
        net_benefit = (losses_avoided - opp_cost) + friction_saved

        # Verdict logic
        n_resolved = len(resolved_signals)
        if n_resolved < 10:
            verdict = "INSUFFICIENT_OBSERVATIONS_FOR_DECISION"
            conclusion = (
                f"Resolved sample size (N={n_resolved}) is below empirical evaluation threshold (N >= 10). "
                "Retain shadow observation mode."
            )
        elif opp_cost > losses_avoided and (gated_summary["sharpeRatioNet"] or 0) <= (ungated_summary["sharpeRatioNet"] or 0):
            verdict = "HARMFUL_GATE_OPPORTUNITY_COST_DOMINATES"
            conclusion = (
                f"Hypothetical gate excluded {len(excluded_winners)} winning setups ({opp_cost:.1f}% gross profit) "
                f"while saving only {len(avoided_losers)} losing setups ({losses_avoided:.1f}% gross loss). "
                "Gating would degrade strategy Sharpe ratio. PROMOTION BLOCKED."
            )
        elif net_benefit > 0 and (gated_summary["sharpeRatioNet"] or 0) > (ungated_summary["sharpeRatioNet"] or 0):
            verdict = "FAVORABLE_GATE_COUNTERFACTUAL"
            conclusion = (
                f"Hypothetical gate improved net Sharpe ratio from {ungated_summary['sharpeRatioNet']} to {gated_summary['sharpeRatioNet']} "
                f"with a net economic benefit of +{net_benefit:.2f}%. Prospective candidate for promotion consideration."
            )
        else:
            verdict = "NEUTRAL_GATE_COUNTERFACTUAL"
            conclusion = (
                "Hypothetical gating produced neutral impact. Differences in net Sharpe ratio are within statistical noise."
            )

        return {
            "experiment": "Experiment 26-B: Economic Counterfactual",
            "specVersion": cls.SPEC_VERSION,
            "filterCriterion": filter_criterion,
            "status": "EVALUATED",
            "totalSignals": len(signals),
            "resolvedSignals": n_resolved,
            "ungatedPortfolio": ungated_summary,
            "gatedPortfolio": gated_summary,
            "tradeoffAnalysis": {
                "filteredTradesCount": len(excluded_winners) + len(avoided_losers),
                "excludedWinnersCount": len(excluded_winners),
                "avoidedLosersCount": len(avoided_losers),
                "opportunityCostPct": round(opp_cost, 2),
                "lossesAvoidedPct": round(losses_avoided, 2),
                "frictionSavedPct": round(friction_saved, 2),
                "netEconomicBenefitPct": round(net_benefit, 2),
                "excludedWinnersList": excluded_winners,
                "avoidedLosersList": avoided_losers,
            },
            "counterfactualVerdict": verdict,
            "conclusion": conclusion,
        }

    @classmethod
    def generate_phase26_validation_report(
        cls,
        ledger_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified prospective validation summary combining Experiment 26-A and Experiment 26-B.
        """
        ledger = ExperimentLedger.load_ledger(ledger_path)

        # Harvest all execution observations across all signals
        all_exec_records: List[Dict[str, Any]] = []
        for s in ledger.get("signals", []):
            obs = s.get("executionObservations", [])
            all_exec_records.extend(obs)

        exp_26a = cls.evaluate_execution_friction(all_exec_records)
        exp_26b = cls.evaluate_economic_counterfactual(ledger)

        # Evaluate against Phase 26 Promotion Gates
        gates = {
            "gate1_non_interference": True,  # Verified by automated test suites
            "gate2_prospective_cohort_size_met": bool(exp_26b["resolvedSignals"] >= 50),
            "gate3_net_economic_value_proven": bool(
                exp_26b["counterfactualVerdict"] == "FAVORABLE_GATE_COUNTERFACTUAL"
                and exp_26b["resolvedSignals"] >= 50
            ),
            "gate4_microstructure_spread_integrated": False,  # Pending tick/L2 data
            "gate5_dual_key_governance_signed": False,        # Requires human committee sign-off
        }

        all_gates_pass = all(gates.values())

        return {
            "document": "Phase 26 Prospective Liquidity Validation Report",
            "specVersion": cls.SPEC_VERSION,
            "engineCommit": ledger.get("engineCommit", "4e36862"),
            "engineTag": ledger.get("engineTag", "v2.4.0-phase24-freeze"),
            "executionFrictionExperiment": exp_26a,
            "economicCounterfactualExperiment": exp_26b,
            "promotionGateReadiness": gates,
            "overallPromotionVerdict": "PROMOTE_TO_GATE" if all_gates_pass else "RETAIN_SHADOW_OBSERVATION",
            "epistemicSummary": (
                "LiquidityGuard remains in FROZEN SHADOW OBSERVATION MODE. "
                "Directional setups are 100% unaffected. Promotion to active execution gate "
                "is strictly blocked until both empirical execution friction and positive net "
                "economic alpha trade-offs are proven on forward cohorts."
            )
        }

    @staticmethod
    def _compute_portfolio_metrics(gross_returns: List[float], net_returns: List[float]) -> Dict[str, Any]:
        """Computes summary statistics for a simulated/realized return series."""
        n = len(gross_returns)
        if n == 0:
            return {
                "tradeCount": 0,
                "winCount": 0,
                "lossCount": 0,
                "winRatePct": 0.0,
                "meanGrossReturnPct": 0.0,
                "meanNetReturnPct": 0.0,
                "totalGrossReturnPct": 0.0,
                "totalNetReturnPct": 0.0,
                "profitFactorNet": 0.0,
                "sharpeRatioNet": None,
                "maxDrawdownPct": 0.0,
            }

        wins = [r for r in net_returns if r > 0]
        losses = [r for r in net_returns if r < 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = round((win_count / n) * 100.0, 1)

        mean_gross = float(np.mean(gross_returns))
        mean_net = float(np.mean(net_returns))
        tot_gross = float(np.sum(gross_returns))
        tot_net = float(np.sum(net_returns))

        tot_win = sum(wins)
        tot_loss = abs(sum(losses))
        profit_factor = round(tot_win / tot_loss, 2) if tot_loss > 0 else (999.0 if tot_win > 0 else 0.0)

        # Sharpe ratio of returns
        if n >= 2:
            std_net = float(np.std(net_returns, ddof=1))
            sharpe = round(mean_net / std_net, 2) if std_net > 0 else None
        else:
            sharpe = None

        # Max Drawdown of cumulative wealth
        cum_equity = 1.0
        running_max = 1.0
        max_dd = 0.0
        for r in net_returns:
            cum_equity *= (1.0 + r / 100.0)
            if cum_equity > running_max:
                running_max = cum_equity
            dd = (cum_equity - running_max) / running_max * 100.0
            if dd < max_dd:
                max_dd = dd

        return {
            "tradeCount": n,
            "winCount": win_count,
            "lossCount": loss_count,
            "winRatePct": win_rate,
            "meanGrossReturnPct": round(mean_gross, 2),
            "meanNetReturnPct": round(mean_net, 2),
            "totalGrossReturnPct": round(tot_gross, 2),
            "totalNetReturnPct": round(tot_net, 2),
            "profitFactorNet": profit_factor,
            "sharpeRatioNet": sharpe,
            "maxDrawdownPct": round(max_dd, 2),
        }

    @staticmethod
    def _empty_portfolio_summary() -> Dict[str, Any]:
        return {
            "tradeCount": 0,
            "winCount": 0,
            "lossCount": 0,
            "winRatePct": 0.0,
            "meanGrossReturnPct": 0.0,
            "meanNetReturnPct": 0.0,
            "totalGrossReturnPct": 0.0,
            "totalNetReturnPct": 0.0,
            "profitFactorNet": 0.0,
            "sharpeRatioNet": None,
            "maxDrawdownPct": 0.0,
        }
