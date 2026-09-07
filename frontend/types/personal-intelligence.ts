/**
 * ARX Terminal vNext - Personal Decision Intelligence Contracts (Sprint 8)
 * Source: docs/architecture/PERSONAL_DECISION_INTELLIGENCE.md
 */

export type PlaybookRuleCategory = "DO_MORE" | "STOP_DOING" | "CALIBRATE";

export const PlaybookRuleCategory = {
  DO_MORE: "DO_MORE" as const,
  STOP_DOING: "STOP_DOING" as const,
  CALIBRATE: "CALIBRATE" as const,
};

export type RecommendationCategory =
  | "POSITION_SIZING"
  | "ENTRY"
  | "EXIT"
  | "RISK"
  | "MACRO";

export const RecommendationCategory = {
  POSITION_SIZING: "POSITION_SIZING" as const,
  ENTRY: "ENTRY" as const,
  EXIT: "EXIT" as const,
  RISK: "RISK" as const,
  MACRO: "MACRO" as const,
};

export interface PlaybookRule {
  ruleId: string;
  category: PlaybookRuleCategory;
  title: string;
  explanation: string;
  supportingOutcomes: number;
  winRate: number;
  averageReturn: number;
  confidence: number;
  status: "ACTIVE" | "REVIEW" | "ARCHIVED";
}

export interface BehavioralRecommendation {
  recommendationId: string;
  category: RecommendationCategory;
  recommendation: string;
  projectedImpact: number;
  confidence: number;
  generatedAt: string;
}

export interface AdoptionMetrics {
  recommendationsIssued: number;
  recommendationsFollowed: number;
  behavioralAdoptionRate: number; // (followed / issued) * 100
  repeatMistakeRate: number; // percentage change vs prior quarter (e.g. -43%)
  decisionDrift: number; // 0-100 score
  driftClassification: "LOW" | "MEDIUM" | "HIGH";
  ruleAdherence: {
    overall: number;
    stopDiscipline: number;
    macroRules: number;
    positionSizing: number;
    riskControls: number;
  };
  complianceBreakdown: {
    doMore: number;
    stopDoing: number;
    calibrate: number;
  };
}

export interface JourneyMilestone {
  period: string;
  score: number;
  delta: number;
  milestone: string;
  keyImprovement: string;
  impactPts: number;
  status: "COMPLETED" | "ACTIVE" | "FUTURE";
}

export interface LearningJourneyData {
  currentScore: number;
  priorYearScore: number;
  improvementPct: number;
  largestContributor: {
    name: string;
    impactPts: number;
  };
  nextOpportunity: {
    action: string;
    projectedGainPts: number;
    confidence: number;
  };
  nextTargetScore: number;
  pointsRemaining: number;
  milestones: JourneyMilestone[];
}

export interface PersonalPlaybook {
  playbookId: string;
  userId: string;
  generatedAt: string;
  version: number;
  qualityScore: number;
  strengths: PlaybookRule[];
  weaknesses: PlaybookRule[];
  recommendations: BehavioralRecommendation[];
  adoptionMetrics: AdoptionMetrics;
  journey: LearningJourneyData;
  confidenceScore: number;
}

export function validatePersonalPlaybook(pb: PersonalPlaybook): PersonalPlaybook {
  if (!pb.playbookId || !pb.userId) {
    throw new Error("PersonalPlaybook missing mandatory playbookId or userId");
  }
  if (pb.qualityScore < 0 || pb.qualityScore > 100) {
    throw new Error(`Invalid qualityScore: ${pb.qualityScore}. Must be [0, 100].`);
  }
  if (pb.adoptionMetrics.behavioralAdoptionRate < 0 || pb.adoptionMetrics.behavioralAdoptionRate > 100) {
    throw new Error(`Invalid BAR: ${pb.adoptionMetrics.behavioralAdoptionRate}. Must be [0, 100].`);
  }
  return pb;
}
