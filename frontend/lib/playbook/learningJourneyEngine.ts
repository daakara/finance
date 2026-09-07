import {
  JourneyMilestone,
  LearningJourneyData,
} from "../../types/personal-intelligence";

export class LearningJourneyEngine {
  /**
   * Evaluates quarterly trajectory and computes points remaining to next milestone tier.
   */
  public static computeJourneyData(
    currentScore = 74,
    priorYearScore = 62,
    targetScore = 80
  ): LearningJourneyData {
    const improvementPct = Number(
      (((currentScore - priorYearScore) / priorYearScore) * 100).toFixed(1)
    );
    const pointsRemaining = Math.max(0, targetScore - currentScore);

    const milestones: JourneyMilestone[] = [
      {
        period: "2025 Q1",
        score: 62,
        delta: 0,
        milestone: "Baseline Established",
        keyImprovement: "Identified high discretionary slippage and absence of macro regime gating.",
        impactPts: 0,
        status: "COMPLETED",
      },
      {
        period: "2025 Q2",
        score: 64,
        delta: 2.0,
        milestone: "Stop Governance Framework",
        keyImprovement: "Introduced hard stop floor at -3.5% capping maximum drawdown per trade.",
        impactPts: 2.0,
        status: "COMPLETED",
      },
      {
        period: "2025 Q3",
        score: 67,
        delta: 3.0,
        milestone: "Macro Regime Gating",
        keyImprovement: "Filtered equity breakout entries by macro regime, eliminating 65% of whipsaws.",
        impactPts: 3.0,
        status: "COMPLETED",
      },
      {
        period: "2025 Q4",
        score: 71,
        delta: 4.0,
        milestone: "Conviction Scaling & Dark Pool Filter",
        keyImprovement: "Calibrated volume absorption >+2.0σ, lifting setup win rate to 68%.",
        impactPts: 4.0,
        status: "COMPLETED",
      },
      {
        period: "Today",
        score: currentScore,
        delta: 3.0,
        milestone: "Institutional Playbook & Attribution Loop",
        keyImprovement: "Operationalized learning into explicit behavioral rules; reached Top 18% Decile.",
        impactPts: 3.0,
        status: "ACTIVE",
      },
    ];

    return {
      currentScore,
      priorYearScore,
      improvementPct,
      largestContributor: {
        name: "Institutional Flow Filter",
        impactPts: 6.2,
      },
      nextOpportunity: {
        action: "Reduce exposure during macro deterioration environments",
        projectedGainPts: 3.4,
        confidence: 0.89,
      },
      nextTargetScore: targetScore,
      pointsRemaining,
      milestones,
    };
  }
}
