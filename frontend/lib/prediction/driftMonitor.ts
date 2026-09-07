import { DriftMetric, DriftReport } from "../../types/predictive-intelligence";

export class DriftMonitor {
  /**
   * Evaluates feature distribution shifts and prediction accuracy degradation.
   * Thresholds: Warning > 10%, Critical > 20% (Deployment freeze)
   */
  public static evaluateDrift(metricsData: {
    featureShiftPct: number;
    regimeDivergencePct: number;
    outcomeErrorShiftPct: number;
  }): DriftReport {
    const metrics: DriftMetric[] = [
      {
        metricName: "Feature Distribution Shift",
        warningThreshold: 10.0,
        criticalThreshold: 20.0,
        currentShiftPct: metricsData.featureShiftPct,
        status:
          metricsData.featureShiftPct > 20.0
            ? "CRITICAL"
            : metricsData.featureShiftPct > 10.0
            ? "WARNING"
            : "STABLE",
      },
      {
        metricName: "Macro Regime Divergence",
        warningThreshold: 10.0,
        criticalThreshold: 20.0,
        currentShiftPct: metricsData.regimeDivergencePct,
        status:
          metricsData.regimeDivergencePct > 20.0
            ? "CRITICAL"
            : metricsData.regimeDivergencePct > 10.0
            ? "WARNING"
            : "STABLE",
      },
      {
        metricName: "Outcome Error Shift",
        warningThreshold: 10.0,
        criticalThreshold: 20.0,
        currentShiftPct: metricsData.outcomeErrorShiftPct,
        status:
          metricsData.outcomeErrorShiftPct > 20.0
            ? "CRITICAL"
            : metricsData.outcomeErrorShiftPct > 10.0
            ? "WARNING"
            : "STABLE",
      },
    ];

    const hasCritical = metrics.some((m) => m.status === "CRITICAL");
    const hasWarning = metrics.some((m) => m.status === "WARNING");
    const overallStatus = hasCritical ? "CRITICAL" : hasWarning ? "WARNING" : "STABLE";

    return {
      generatedAt: new Date().toISOString(),
      overallStatus,
      metrics,
    };
  }
}
