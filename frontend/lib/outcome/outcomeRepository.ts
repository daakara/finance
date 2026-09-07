import {
  AttributionResult,
  OutcomeRecord,
  validateOutcomeRecord,
} from "../../types/outcome-intelligence";

export class OutcomeRepository {
  private static instance: OutcomeRepository;
  private outcomes: Map<string, OutcomeRecord> = new Map();
  private attributions: Map<string, AttributionResult> = new Map();

  private constructor() {
    this.seedDefaultOutcomes();
  }

  public static getInstance(): OutcomeRepository {
    if (!OutcomeRepository.instance) {
      OutcomeRepository.instance = new OutcomeRepository();
    }
    return OutcomeRepository.instance;
  }

  public reset(): void {
    this.outcomes.clear();
    this.attributions.clear();
    this.seedDefaultOutcomes();
  }

  private seedDefaultOutcomes(): void {
    const defaultOutcomes: OutcomeRecord[] = [
      {
        outcomeId: "out-cprx-01",
        predictionId: "pred-cprx-orig",
        ticker: "CPRX",
        predictedAt: "2026-09-01T08:00:00Z",
        resolvedAt: "2026-09-05T16:00:00Z",
        outcomeClass: "SUCCESS",
        attributionCategory: "TARGET_REACHED",
        explanation: "Institutional accumulation velocity (+2.1σ) absorbed supply and achieved Target 1",
        outcomeReturnPct: 7.4,
        outcomeConfidence: 0.89,
        snapshotHash: "sha256:cprxbaselinehash1",
      },
      {
        outcomeId: "out-nvda-01",
        predictionId: "pred-nvda-orig",
        ticker: "NVDA",
        predictedAt: "2026-09-02T08:00:00Z",
        resolvedAt: "2026-09-06T16:00:00Z",
        outcomeClass: "SUCCESS",
        attributionCategory: "TARGET_REACHED",
        explanation: "Execution buy zone entry triggered with strong semiconductor momentum",
        outcomeReturnPct: 8.2,
        outcomeConfidence: 0.86,
        snapshotHash: "sha256:nvdabaselinehash1",
      },
      {
        outcomeId: "out-intc-01",
        predictionId: "pred-intc-orig",
        ticker: "INTC",
        predictedAt: "2026-09-01T08:00:00Z",
        resolvedAt: "2026-09-04T16:00:00Z",
        outcomeClass: "FAILURE",
        attributionCategory: "STOP_TRIGGERED",
        explanation: "Adverse price movement breached stop floor at $19.20",
        outcomeReturnPct: -4.1,
        outcomeConfidence: 0.82,
        snapshotHash: "sha256:intcbaselinehash1",
      },
      {
        outcomeId: "out-spy-01",
        predictionId: "pred-spy-orig",
        ticker: "SPY",
        predictedAt: "2026-09-03T08:00:00Z",
        resolvedAt: "2026-09-06T16:00:00Z",
        outcomeClass: "INVALIDATED",
        attributionCategory: "REGIME_CHANGE",
        explanation: "VIX surge to 24.5 triggered macro transition to DEFENSIVE regime",
        outcomeReturnPct: -0.8,
        outcomeConfidence: 0.78,
        snapshotHash: "sha256:spybaselinehash1",
      },
    ];

    for (const out of defaultOutcomes) {
      this.outcomes.set(out.outcomeId, out);
    }
  }

  public saveOutcome(outcome: OutcomeRecord, attribution?: AttributionResult): void {
    const validated = validateOutcomeRecord(outcome);
    this.outcomes.set(validated.outcomeId, validated);
    if (attribution) {
      this.attributions.set(validated.predictionId, attribution);
    }
  }

  /**
   * Enforces INV-O3 / INV-P4: Historical outcomes are permanently immutable.
   */
  public updateOutcome(outcomeId: string): void {
    throw new Error("OUTCOME_IMMUTABLE");
  }

  public getOutcome(outcomeId: string): OutcomeRecord | null {
    return this.outcomes.get(outcomeId) || null;
  }

  public getOutcomesForTicker(ticker: string): OutcomeRecord[] {
    return Array.from(this.outcomes.values()).filter(
      (o) => o.ticker.toUpperCase() === ticker.toUpperCase()
    );
  }

  public getAllOutcomes(): OutcomeRecord[] {
    return Array.from(this.outcomes.values());
  }

  public getAttribution(predictionId: string): AttributionResult | null {
    return this.attributions.get(predictionId) || null;
  }
}
