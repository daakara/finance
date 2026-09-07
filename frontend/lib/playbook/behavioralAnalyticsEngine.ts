import { AdoptionMetrics } from "../../types/personal-intelligence";

export class BehavioralAnalyticsEngine {
  /**
   * Calculates Behavioral Adoption Rate (BAR):
   * BAR = (Recommendations Followed / Total Recommendations Issued) * 100
   */
  public static calculateBAR(followed: number, issued: number): number {
    if (issued <= 0) return 0;
    return Number(((followed / issued) * 100).toFixed(1));
  }

  /**
   * Classifies decision drift into institutional risk bands:
   * 0 - 29: LOW
   * 30 - 59: MEDIUM
   * 60+: HIGH
   */
  public static classifyDecisionDrift(driftScore: number): "LOW" | "MEDIUM" | "HIGH" {
    if (driftScore < 30) return "LOW";
    if (driftScore < 60) return "MEDIUM";
    return "HIGH";
  }

  /**
   * Calculates percentage change in repeat mistakes between quarters:
   * ((current - prior) / prior) * 100
   */
  public static calculateRepeatMistakeReduction(current: number, prior: number): number {
    if (prior <= 0) return 0;
    return Number((((current - prior) / prior) * 100).toFixed(1));
  }

  /**
   * Computes full adoption metrics snapshot.
   */
  public static computeMetrics(
    issued = 112,
    followed = 79,
    currentMistakes = 12,
    priorMistakes = 21,
    driftScore = 21.0
  ): AdoptionMetrics {
    const bar = this.calculateBAR(followed, issued);
    const repeatReduction = this.calculateRepeatMistakeReduction(currentMistakes, priorMistakes);
    const classification = this.classifyDecisionDrift(driftScore);

    return {
      recommendationsIssued: issued,
      recommendationsFollowed: followed,
      behavioralAdoptionRate: bar,
      repeatMistakeRate: repeatReduction,
      decisionDrift: driftScore,
      driftClassification: classification,
      ruleAdherence: {
        overall: 87.0,
        stopDiscipline: 91.0,
        macroRules: 72.0,
        positionSizing: 87.0,
        riskControls: 94.0,
      },
      complianceBreakdown: {
        doMore: 81.0,
        stopDoing: 74.0,
        calibrate: 63.0,
      },
    };
  }
}
