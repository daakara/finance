"""Descriptive historical forecast diagnostics (no automatic model calibration)."""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class SelfHealingForecastAuditor:
    """Summarizes a retrospective momentum proxy and observed VaR breach frequency.

    This implementation does not replay archived model forecasts, perform the
    Kupiec likelihood-ratio POF test, estimate confidence intervals, or update
    model parameters. The legacy method name and output keys remain stable.
    """

    def audit_and_calibrate(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        current_risk_metrics: Dict[str, Any],
        expected_return_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute descriptive overlapping-window diagnostic statistics only."""
        if len(price_df) < 35:
            return {
                "auditStatus": "Awaiting Minimum Historical Sample (N < 35)",
                "accuracyScore": None,
                "hitRatePct": None,
                "rmsePct": None,
                "varBreachRatePct": None,
                "varBreachStatus": "Insufficient History (< 35 bars)",
                "autoCalibrationAdjustments": "No model parameters changed (insufficient history)",
                "confidenceInterval": "Insufficient Sample Size",
            }

        closes = price_df["Close"].values
        daily_returns = price_df["Close"].pct_change().dropna().values

        # 1. Observed daily VaR threshold breach frequency (NOT a Kupiec POF test)
        var_95_threshold = abs(float(current_risk_metrics.get("Modified_VaR_95", 2.5))) / 100.0
        breaches = sum(1 for r in daily_returns if r < -var_95_threshold)
        total_days = len(daily_returns)
        actual_breach_rate = round((breaches / max(1, total_days)) * 100, 2)

        # Expected breach rate for 95% VaR is ~5.0%
        if actual_breach_rate <= 5.5:
            var_status = "Observed breach rate at or below 5.5% (no statistical test)"
            var_adjustment = "No model parameters changed; descriptive breach diagnostic only"
        else:
            var_status = "Observed breach rate above 5.5% (no statistical test)"
            var_adjustment = "No model parameters changed; descriptive breach diagnostic only"

        # 2. Retrospective overlapping 30-day momentum proxy fit and RMSE
        # Ensure safely bounded indices
        n_closes = len(closes)
        sample_windows = min(20, n_closes - 31)
        errors = []
        hits = 0

        for i in range(sample_windows):
            past_idx = n_closes - 31 - i
            target_idx = past_idx + 30
            if target_idx < n_closes and past_idx >= 0:
                actual_30d_return = (closes[target_idx] - closes[past_idx]) / max(0.0001, closes[past_idx])
                
                # Retrospective momentum proxy, not a stored contemporaneous forecast
                past_start = max(0, past_idx - 60)
                past_momentum = (closes[past_idx] - closes[past_start]) / max(0.0001, closes[past_start])
                projected_return = past_momentum * 0.35

                error = (projected_return - actual_30d_return)
                errors.append(error ** 2)

                if (projected_return >= 0 and actual_30d_return >= 0) or (projected_return < 0 and actual_30d_return < 0):
                    hits += 1

        if errors and sample_windows > 0:
            rmse = round(float(np.sqrt(np.mean(errors))) * 100, 2)
            hit_rate = round((hits / sample_windows) * 100, 1)
            accuracy_score = round(max(0.0, min(100.0, 100.0 - (rmse * 3.5) + (hit_rate * 0.15))), 1)
            audit_status = "Retrospective heuristic audit (not calibrated)"
        else:
            rmse = None
            hit_rate = None
            accuracy_score = None
            audit_status = "Awaiting Valid Walk-Forward Samples"

        return {
            "auditStatus": audit_status,
            "accuracyScore": accuracy_score,
            "hitRatePct": hit_rate,
            "rmsePct": rmse,
            "varBreachRatePct": actual_breach_rate,
            "varBreachStatus": var_status,
            "autoCalibrationAdjustments": var_adjustment,
            "confidenceInterval": "Not estimated (no statistical confidence interval)",
        }

