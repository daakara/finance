"""ARCH-002: diagnostic claim honesty, without changing engine math.

Run: python -m pytest -q tests/test_self_healing_audit_claims.py
"""
import pandas as pd
import pytest

from analyst_dashboard.analyzers.self_healing_engine import SelfHealingForecastAuditor


@pytest.mark.parametrize(
    "prices,expected_status",
    [
        ([100.0 + i * 0.05 for i in range(80)], "at or below 5.5%"),
        ([100.0 if i % 2 == 0 else 90.0 for i in range(80)], "above 5.5%"),
    ],
)
def test_observed_breach_rate_is_not_called_kupiec_or_auto_calibration(prices, expected_status):
    auditor = SelfHealingForecastAuditor()
    risk = {"Modified_VaR_95": 2.5}
    expected = {"p50Expected": 4.0}
    result = auditor.audit_and_calibrate(
        symbol="TEST",
        price_df=pd.DataFrame({"Close": prices}),
        current_risk_metrics=risk,
        expected_return_data=expected,
    )
    assert expected_status in result["varBreachStatus"]
    assert "no statistical test" in result["varBreachStatus"]
    assert result["auditStatus"] == "Retrospective heuristic audit (not calibrated)"
    assert result["confidenceInterval"] == "Not estimated (no statistical confidence interval)"
    assert result["autoCalibrationAdjustments"].startswith("No model parameters changed")
    assert result["accuracyScore"] is not None
    assert result["rmsePct"] is not None
    assert result["hitRatePct"] is not None
    assert risk == {"Modified_VaR_95": 2.5}
    assert expected == {"p50Expected": 4.0}


def test_insufficient_history_does_not_claim_statistical_confidence():
    result = SelfHealingForecastAuditor().audit_and_calibrate(
        symbol="TEST",
        price_df=pd.DataFrame({"Close": [100.0] * 34}),
        current_risk_metrics={},
        expected_return_data={},
    )
    assert result["accuracyScore"] is None
    assert result["varBreachRatePct"] is None
    assert result["confidenceInterval"] == "Insufficient Sample Size"
    assert "No model parameters changed" in result["autoCalibrationAdjustments"]


def test_diagnostic_does_not_claim_95_percent_interval_or_kupiec_even_with_many_samples():
    result = SelfHealingForecastAuditor().audit_and_calibrate(
        "TEST",
        pd.DataFrame({"Close": [100.0 + i for i in range(200)]}),
        {"Modified_VaR_95": 2.5},
        {"p50Expected": 1.0},
    )
    serialized = " ".join(str(value) for value in result.values())
    assert "Passed Kupiec" not in serialized
    assert "Self-Healed" not in serialized
    assert "95% Statistical Confidence" not in serialized
    assert "calibrated" not in result["autoCalibrationAdjustments"].lower()
