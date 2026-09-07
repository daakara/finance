/**
 * Behavioral Story & Narrative Synthesis Engine
 * 
 * Generates:
 * 1. Executive Story (Today's Story, Monthly Outcome Narrative, AI Chief of Staff)
 * 2. Morning Briefing 2.0 Narrative (Market Changed -> Why It Matters -> What Is Affected -> What To Do)
 * 3. Behavioral Evolution Timeline (Chronological behavior updates & impact)
 * 4. Forward AI Behavioral Coach Forecast (74 -> 80 in 4 months)
 * 
 * Phase 26 Quantitative Freeze Compliant: Strictly frontend narrative orchestration.
 */

import {
  BehavioralStory,
  MorningBriefingV2Story,
  BehavioralStrength,
  BehavioralRisk,
  ImprovementForecast,
  BehavioralTimelineEvent,
  BehavioralIntelligenceProfile,
} from '@/types/behavioral-intelligence';
import { CANONICAL_LEARNING_VELOCITY } from './learningVelocityEngine';
import { CANONICAL_CONFIDENCE_METRICS } from './statisticalConfidenceEngine';

export const CANONICAL_BEHAVIORAL_STORY: BehavioralStory = {
  userName: 'David',
  dateString: 'Monday, September 7, 2026',
  weeklyAdoptionRate: 84.0,
  weeklyScoreDelta: 2.0,
  repeatMistakeDelta: -12.0,
  macroExposureTrend: 'Macro exposure increased significantly during the last three trading sessions.',
  recommendedActionToday: 'Tighten regime filters and review 3 exposed momentum positions.',
  monthlyStats: {
    totalDecisions: 42,
    successCount: 30,
    failureCount: 12,
    largestSuccessDriver: 'Institutional Accumulation filter (+72% win-rate)',
    largestFailureDriver: 'Late-day momentum entries after 2:30 PM EST',
    netLearningTakeaway: 'Position timing remains your highest leverage improvement area.',
  },
  chiefOfStaffHighlight: {
    actionTitle: 'Reduce exposure in weakening macro regimes',
    drawdownReductionPct: 4.2,
    confidence: 92,
    evidenceDetail: 'Historical drawdown simulation under >2.0 sigma macro shift indicates -4.2% capital protection.',
  },
};

export const CANONICAL_MORNING_BRIEFING_V2: MorningBriefingV2Story = {
  marketRiskScorePrev: 42,
  marketRiskScoreCurrent: 56,
  narrativeSummary: 'Market risk environment deteriorated overnight. Treasuries strengthened, semiconductors weakened, and growth leadership narrowed.',
  marketShifts: [
    { dimension: 'Treasuries (10Y Yield)', direction: 'STRENGTHENED', detail: 'Flight to quality with 10Y yield dropping 8 bps' },
    { dimension: 'Semiconductors (SOX)', direction: 'WEAKENED', detail: 'Break of 20-day moving average on institutional distribution volume' },
    { dimension: 'Growth Breadth', direction: 'NARROWED', detail: 'Only 32% of Russell 1000 members trading above 50-day SMA' },
  ],
  affectedPositions: [
    {
      symbol: 'NVDA',
      shares: 450,
      currentPrice: 118.50,
      capitalAtRisk: 53325,
      violatedCondition: 'SOX regime break & elevated beta (>1.8)',
      suggestedAction: 'TRIM_50',
    },
    {
      symbol: 'AMD',
      shares: 600,
      currentPrice: 142.20,
      capitalAtRisk: 85320,
      violatedCondition: 'Late-day momentum entry violating macro gate',
      suggestedAction: 'TIGHTEN_STOP',
    },
    {
      symbol: 'CRWD',
      shares: 180,
      currentPrice: 252.00,
      capitalAtRisk: 45360,
      violatedCondition: 'Software sector breadth divergence',
      suggestedAction: 'TIGHTEN_STOP',
    },
  ],
  totalCapitalAtRisk: 184000,
  recommendationConfidence: 91,
  isFeedStale: false,
};

export const CANONICAL_BEHAVIORAL_STRENGTHS: BehavioralStrength[] = [
  {
    id: 'STRENGTH-01',
    title: 'Institutional Flow Discipline',
    qualityPointContribution: 6.2,
    confidence: 91,
    description: 'Adhering to >2.0 sigma institutional volume confirmation before entering Stage 2 breakouts.',
    evidenceHash: 'SHA256:8b4f1c9d2e...',
  },
  {
    id: 'STRENGTH-02',
    title: 'Consistent Pre-Market Journaling',
    qualityPointContribution: 3.4,
    confidence: 88,
    description: 'Recording thesis invalidate levels and max drawdowns prior to market open.',
    evidenceHash: 'SHA256:3a7e9f1c4b...',
  },
];

export const CANONICAL_BEHAVIORAL_RISKS: BehavioralRisk[] = [
  {
    id: 'RISK-01',
    title: 'Macro Deterioration Blindness',
    exposurePercentage: 21.0,
    confidence: 87,
    description: 'Maintaining high-beta long exposure when macro stress index rises above 50.',
    mitigation: 'Enable automated macro-gate prompt requiring committee override for beta >1.5.',
  },
  {
    id: 'RISK-02',
    title: 'Late-Day Momentum Chases',
    exposurePercentage: 14.0,
    confidence: 84,
    description: 'Entering breakouts after 2:30 PM EST without volume support.',
    mitigation: 'Hard curfew on new breakout orders after 2:00 PM EST.',
  },
];

export const CANONICAL_IMPROVEMENT_FORECAST: ImprovementForecast = {
  currentScore: 74,
  projectedScoreFourMonths: 80,
  projectedScoreSixMonths: 82,
  confidence: 87,
  isLowConfidenceWarning: false,
  expectedGain: 6.0,
  keyCatalysts: [
    'Better position sizing across high-volatility names',
    'Stronger macro discipline during regime transitions',
    'Elimination of late-day momentum chases',
  ],
};

export const CANONICAL_BEHAVIORAL_TIMELINE: BehavioralTimelineEvent[] = [
  {
    quarter: 'Q1 2026',
    problemIdentified: 'Poor stop-loss discipline on gap-downs',
    governanceImprovement: 'Enforced Stop Governance & automated sizing bounds',
    qualityDelta: 2.0,
    supportingEvidence: 'Stop slippage decreased from -2.4% to -0.6% on gap days.',
    category: 'STOP_LOSS',
  },
  {
    quarter: 'Q2 2026',
    problemIdentified: 'Macro regime blindness during rate hike cycles',
    governanceImprovement: 'Macro Gating filter integrated into Morning Briefing',
    qualityDelta: 3.0,
    supportingEvidence: 'Avoided 3 major tech pullbacks by reducing beta exposure in June.',
    category: 'MACRO_GATING',
  },
  {
    quarter: 'Q3 2026',
    problemIdentified: 'Chasing low-liquidity breakout traps',
    governanceImprovement: 'Institutional flow accumulation surge requirement (>2.0 sigma)',
    qualityDelta: 4.0,
    supportingEvidence: 'Breakout win rate elevated from 54% to 72% across 38 trades.',
    category: 'FLOW_ACCUMULATION',
  },
];

export const CANONICAL_BEHAVIORAL_PROFILE: BehavioralIntelligenceProfile = {
  profileId: 'BIP-DAVID-2026',
  userId: 'usr_david_001',
  generatedAt: '2026-09-08T00:00:00Z',
  story: CANONICAL_BEHAVIORAL_STORY,
  decisionQuality: {
    currentScore: 74,
    previousScore: 62,
    quarterlyChange: 6,
    annualChange: 12,
    percentileRank: 18,
    targetScore: 80,
    trend: 'IMPROVING',
  },
  learningVelocity: CANONICAL_LEARNING_VELOCITY,
  behaviorAdoption: CANONICAL_CONFIDENCE_METRICS.behavioralAdoption,
  ruleAdherence: CANONICAL_CONFIDENCE_METRICS.ruleAdherence,
  decisionDrift: {
    driftScore: 21.0,
    targetThreshold: 20.0,
    driftCategory: 'LOW',
    majorDeviationDrivers: [
      'Late-day order submissions (3 instances)',
      'Position size override above recommended 5% equity cap (1 instance)',
    ],
  },
  strengths: CANONICAL_BEHAVIORAL_STRENGTHS,
  risks: CANONICAL_BEHAVIORAL_RISKS,
  projectedImprovement: CANONICAL_IMPROVEMENT_FORECAST,
  timeline: CANONICAL_BEHAVIORAL_TIMELINE,
};
