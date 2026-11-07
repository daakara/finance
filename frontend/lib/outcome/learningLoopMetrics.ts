import { OutcomeRecord, OutcomeSummary } from "../../types/outcome-intelligence";

export class LearningLoopMetrics {
  /**
   * Aggregates resolved outcomes into institutional performance indicators.
   * Computes Prediction Actionability Rate (PAR).
   */
  public static computeSummary(
    outcomes: OutcomeRecord[],
    displayedPredictionsCount = 3000
  ): OutcomeSummary {
    const total = outcomes.length;
    if (total === 0) {
      return {
        totalReviewed: 0,
        successCount: 0,
        partialSuccessCount: 0,
        failureCount: 0,
        expiredCount: 0,
        successRate: 0,
        partialSuccessRate: 0,
        failureRate: 0,
        expiredRate: 0,
        topDrivers: [],
        failureDrivers: [],
        actionability: {
          reviewedPredictions: 0,
          displayedPredictions: displayedPredictionsCount,
          par: 0,
          target: 0.50,
          status: "FAIL",
        },
      };
    }

    const successes = outcomes.filter((o) => o.outcomeClass === "SUCCESS").length;
    const partials = outcomes.filter((o) => o.outcomeClass === "PARTIAL_SUCCESS").length;
    const failures = outcomes.filter((o) => o.outcomeClass === "FAILURE").length;
    const expired = outcomes.filter((o) => o.outcomeClass === "EXPIRED" || o.outcomeClass === "INVALIDATED").length;

    const par = Number((total / Math.max(total, displayedPredictionsCount)).toFixed(3));

    // Driver Win Rates
    const driverStats: Record<string, { total: number; wins: number }> = {
      "Institutional Accumulation": { total: 0, wins: 0 },
      "Regime Alignment": { total: 0, wins: 0 },
      "Validation Promotion": { total: 0, wins: 0 },
    };

    // Failure Drivers
    const failureStats: Record<string, number> = {
      "Regime Deterioration": 0,
      "Flow Reversal": 0,
      "Stop Triggered": 0,
    };

    for (const o of outcomes) {
      if (o.outcomeClass === "SUCCESS") {
        if (o.attributionCategory === "TARGET_REACHED") {
          driverStats["Institutional Accumulation"].total++;
          driverStats["Institutional Accumulation"].wins++;
        } else {
          driverStats["Regime Alignment"].total++;
          driverStats["Regime Alignment"].wins++;
        }
      } else if (o.outcomeClass === "FAILURE") {
        if (o.attributionCategory === "STOP_TRIGGERED") {
          failureStats["Stop Triggered"]++;
        } else {
          failureStats["Flow Reversal"]++;
        }
      } else if (o.outcomeClass === "INVALIDATED") {
        failureStats["Regime Deterioration"]++;
      }
    }

    const topDrivers = Object.entries(driverStats)
      .filter(([_, stats]) => stats.total > 0)
      .map(([driver, stats]) => ({
        driver,
        winRate: Number((stats.wins / stats.total).toFixed(2)),
      }))
      .sort((a, b) => b.winRate - a.winRate);

    const failureDrivers = Object.entries(failureStats)
      .filter(([_, count]) => count > 0)
      .map(([driver, count]) => ({
        driver,
        percentage: Number((count / Math.max(1, failures)).toFixed(2)),
      }))
      .sort((a, b) => b.percentage - a.percentage);

    return {
      totalReviewed: total,
      successCount: successes,
      partialSuccessCount: partials,
      failureCount: failures,
      expiredCount: expired,
      successRate: Number(((successes / total) * 100).toFixed(1)),
      partialSuccessRate: Number(((partials / total) * 100).toFixed(1)),
      failureRate: Number(((failures / total) * 100).toFixed(1)),
      expiredRate: Number(((expired / total) * 100).toFixed(1)),
      topDrivers,
      failureDrivers,
      actionability: {
        reviewedPredictions: total,
        displayedPredictions: displayedPredictionsCount,
        par,
        target: 0.50,
        status: par >= 0.50 ? "PASS" : "FAIL",
      },
    };
  }
}
