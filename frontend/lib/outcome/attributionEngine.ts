import {
  AttributionCategory,
  AttributionResult,
  OutcomeRecord,
} from "../../types/outcome-intelligence";

export interface AttributionInputFactors {
  flowZScore: number;
  regimeAlignment: boolean;
  setupScore: number;
  validationTier: string;
}

export class AttributionEngine {
  /**
   * Deterministic causal attribution generator (INV-O4, AC-OI-09).
   */
  public static generateAttribution(
    outcome: OutcomeRecord,
    factors: AttributionInputFactors
  ): AttributionResult {
    let primaryDriver = "Setup Confluence";
    const secondaryDrivers: string[] = [];

    if (outcome.attributionCategory === "STOP_TRIGGERED") {
      primaryDriver = factors.flowZScore < 0 ? "Flow Reversal" : "Stop Triggered";
      secondaryDrivers.push("Volatility expansion past risk buffer");
      if (!factors.regimeAlignment) {
        secondaryDrivers.push("Macro regime headwind");
      }
    } else if (outcome.attributionCategory === "REGIME_CHANGE") {
      primaryDriver = "Regime Deterioration";
      secondaryDrivers.push("Macro volatility divergence");
      secondaryDrivers.push("Term structure inversion");
    } else if (outcome.attributionCategory === "TARGET_REACHED" || outcome.attributionCategory === "EXECUTION_SUCCESS") {
      if (factors.flowZScore >= 1.2) {
        primaryDriver = "Institutional Accumulation";
        secondaryDrivers.push(`Strong institutional absorption (+${factors.flowZScore.toFixed(1)}σ)`);
      } else if (factors.regimeAlignment) {
        primaryDriver = "Regime Alignment";
        secondaryDrivers.push("Macro tailwind and equity risk-on premium");
      } else {
        primaryDriver = "Validation Promotion";
        secondaryDrivers.push(`Multi-factor technical setup score (${factors.setupScore})`);
      }
    } else if (outcome.attributionCategory === "THESIS_EXPIRED") {
      primaryDriver = "Volume Exhaustion";
      secondaryDrivers.push("Low institutional participation within observation limit");
    }

    return {
      attributionId: `attr-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      predictionId: outcome.predictionId,
      primaryDriver,
      secondaryDrivers,
      confidence: outcome.outcomeConfidence,
      createdAt: new Date().toISOString(),
    };
  }
}
