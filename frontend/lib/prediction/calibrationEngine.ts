import {
  CalibrationBucket,
  CalibrationReport,
} from "../../types/predictive-intelligence";

export interface PredictionOutcomePair {
  probability: number;
  actualOutcome: boolean; // true = occurred, false = did not occur
}

export class CalibrationEngine {
  /**
   * Computes Brier Score and Expected Calibration Error (ECE) across 10 confidence buckets.
   * Target: ECE <= 0.05 (5%), Brier Score <= 0.15
   */
  public static calculateCalibration(pairs: PredictionOutcomePair[]): CalibrationReport {
    if (pairs.length === 0) {
      return {
        ece: 0,
        brierScore: 0,
        samples: 0,
        buckets: [],
        status: "PASS",
      };
    }

    // 1. Calculate Brier Score: (1/N) * sum( (p_i - y_i)^2 )
    let totalBrier = 0;
    for (const pair of pairs) {
      const y = pair.actualOutcome ? 1 : 0;
      totalBrier += Math.pow(pair.probability - y, 2);
    }
    const brierScore = Number((totalBrier / pairs.length).toFixed(4));

    // 2. Partition into 10 buckets: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    const bucketCounts = new Array(10).fill(0);
    const bucketSuccesses = new Array(10).fill(0);
    const bucketConfidenceSums = new Array(10).fill(0);

    for (const pair of pairs) {
      let bucketIdx = Math.floor(pair.probability * 10);
      if (bucketIdx >= 10) bucketIdx = 9; // 1.0 falls into bucket 9

      bucketCounts[bucketIdx]++;
      bucketConfidenceSums[bucketIdx] += pair.probability;
      if (pair.actualOutcome) {
        bucketSuccesses[bucketIdx]++;
      }
    }

    const buckets: CalibrationBucket[] = [];
    let ece = 0;

    for (let i = 0; i < 10; i++) {
      const rangeMin = Number((i * 0.1).toFixed(1));
      const rangeMax = Number(((i + 1) * 0.1).toFixed(1));
      const count = bucketCounts[i];

      if (count > 0) {
        const actualSuccessRate = Number((bucketSuccesses[i] / count).toFixed(4));
        const avgConfidence = bucketConfidenceSums[i] / count;
        const bucketError = Math.abs(actualSuccessRate - avgConfidence);

        ece += (count / pairs.length) * bucketError;

        buckets.push({
          rangeMin,
          rangeMax,
          predictionCount: count,
          actualSuccessRate,
        });
      } else {
        buckets.push({
          rangeMin,
          rangeMax,
          predictionCount: 0,
          actualSuccessRate: 0,
        });
      }
    }

    ece = Number(ece.toFixed(4));
    const status = ece <= 0.05 && brierScore <= 0.15 ? "PASS" : "FAIL";

    return {
      ece,
      brierScore,
      samples: pairs.length,
      buckets,
      status,
    };
  }
}
