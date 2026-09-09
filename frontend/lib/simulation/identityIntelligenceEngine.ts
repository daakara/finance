/**
 * Horizon 13: Identity Intelligence Engine
 *
 * Shifting the optimization target from "Actions" to "Identity Trajectory":
 * "Help David Become Who David Wants To Be."
 *
 * Computes:
 * - IdentityTwin (Current Identity vs Target Identity)
 * - Identity Alignment Index (IAI) [0 - 100]
 * - Identity Drift Alerts
 * - Emerging and Fading Identities
 * - Identity Leverage Scoring for Next Best Actions
 */

export interface IdentityTrait {
  id: string;
  name: string;
  currentLevel: number; // 0 to 100
  targetLevel: number;  // 0 to 100
  delta: number;        // targetLevel - currentLevel
  domain: 'CAREER' | 'FINANCE' | 'HEALTH' | 'TRADING' | 'HOUSEHOLD' | 'SYSTEMS';
  evidenceChain: string[];
}

export interface IdentityState {
  roleTitle: string;
  archetype: string;
  domainFoci: string[];
  competenceScore: number; // 0 to 100
  traits: IdentityTrait[];
}

export interface SkillGap {
  skill: string;
  domain: string;
  current: number;
  required: number;
  gap: number;
  estimatedWeeks: number;
}

export interface IdentityDelta {
  overallGap: number;
  skillGaps: SkillGap[];
  estimatedEffortWeeks: number;
}

export interface IdentityDriftAlert {
  id: string;
  targetRole: string;
  domain: string;
  observedInactionDays: number;
  inactivityThresholdDays: number;
  driftSeverity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  causalExplanation: string;
  correctiveActionRecommendation: string;
}

export interface EmergingIdentity {
  name: string;
  confidenceScore: number; // 0 to 100
  catalyst: string;
  evidencePoints: string[];
}

export interface FadingIdentity {
  name: string;
  decayRatePct: number;    // 0 to 100
  supersededBy: string;
  observedReduction: string;
}

export interface IdentityTwin {
  userId: string;
  currentIdentity: IdentityState;
  targetIdentity: IdentityState;
  identityGap: IdentityDelta;
  identityMomentum: number;       // 0 to 100
  identityAlignmentIndex: number; // 0 to 100 (IAI)
  emergingIdentities: EmergingIdentity[];
  fadingIdentities: FadingIdentity[];
  driftAlerts: IdentityDriftAlert[];
  lastEvaluated: string;
}

export const CANONICAL_IDENTITY_TWIN: IdentityTwin = {
  userId: 'usr-david-canonical',
  currentIdentity: {
    roleTitle: 'Analytics Manager',
    archetype: 'Quantitative Analyst & Engineering Leader',
    domainFoci: ['Data Architecture', 'Execution Management', 'Portfolio Risk'],
    competenceScore: 72,
    traits: [
      {
        id: 'trait-arch',
        name: 'AI & Systems Architecture',
        currentLevel: 64,
        targetLevel: 88,
        delta: 24,
        domain: 'CAREER',
        evidenceChain: [
          'Completed Distributed Systems & Causal DAG Module 3',
          'Authored ADR-014 on Edge Signal Verification & Invariant Gates',
          'Shipped closed-loop H5-H12 simulation engines',
        ],
      },
      {
        id: 'trait-lead',
        name: 'Strategic Technical Leadership',
        currentLevel: 70,
        targetLevel: 88,
        delta: 18,
        domain: 'CAREER',
        evidenceChain: [
          'Chaired Q3 Platform Invariant Review Board',
          'Mentored 2 senior engineers on formal verification',
        ],
      },
      {
        id: 'trait-alloc',
        name: 'Systematic Capital Allocation',
        currentLevel: 75,
        targetLevel: 90,
        delta: 15,
        domain: 'TRADING',
        evidenceChain: [
          'Maintained 0.91 Poise Ratio across Minervini VCP trades',
          'Strict compliance with INV-OI98-P 6-month liquid cash floor',
          'Zero emotional trade overrides over past 90 days',
        ],
      },
      {
        id: 'trait-infl',
        name: 'Public Industry Influence',
        currentLevel: 41,
        targetLevel: 50,
        delta: 9,
        domain: 'SYSTEMS',
        evidenceChain: [
          'Drafted comprehensive open-source specification on Invariant Engines',
        ],
      },
    ],
  },
  targetIdentity: {
    roleTitle: 'AI Strategy Leader & Systematic Investor',
    archetype: 'World-Class Consumer AI Architect & Quantitative Sovereign',
    domainFoci: ['Autonomous Systems', 'Macro Risk Sovereignty', 'Generative Architecture'],
    competenceScore: 90,
    traits: [
      {
        id: 'trait-arch',
        name: 'AI & Systems Architecture',
        currentLevel: 88,
        targetLevel: 88,
        delta: 0,
        domain: 'CAREER',
        evidenceChain: [],
      },
      {
        id: 'trait-lead',
        name: 'Strategic Technical Leadership',
        currentLevel: 88,
        targetLevel: 88,
        delta: 0,
        domain: 'CAREER',
        evidenceChain: [],
      },
      {
        id: 'trait-alloc',
        name: 'Systematic Capital Allocation',
        currentLevel: 90,
        targetLevel: 90,
        delta: 0,
        domain: 'TRADING',
        evidenceChain: [],
      },
      {
        id: 'trait-infl',
        name: 'Public Industry Influence',
        currentLevel: 50,
        targetLevel: 50,
        delta: 0,
        domain: 'SYSTEMS',
        evidenceChain: [],
      },
    ],
  },
  identityGap: {
    overallGap: 18,
    estimatedEffortWeeks: 36,
    skillGaps: [
      { skill: 'Distributed AI Architecture', domain: 'CAREER', current: 64, required: 88, gap: 24, estimatedWeeks: 14 },
      { skill: 'Executive Strategy & Alignment', domain: 'CAREER', current: 70, required: 88, gap: 18, estimatedWeeks: 10 },
      { skill: 'Multi-Asset Risk Sovereignty', domain: 'TRADING', current: 75, required: 90, gap: 15, estimatedWeeks: 8 },
      { skill: 'Ecosystem Thought Leadership', domain: 'SYSTEMS', current: 41, required: 50, gap: 9, estimatedWeeks: 4 },
    ],
  },
  identityMomentum: 64,
  identityAlignmentIndex: 61,
  emergingIdentities: [
    {
      name: 'Systematic Value & Momentum Investor',
      confidenceScore: 84,
      catalyst: 'Adoption of fail-closed sizing clamps and Minervini VCP discipline',
      evidencePoints: [
        '92% adherence to volatility stop-loss thresholds',
        'Automatic refusal to trade during sleep debt spikes',
      ],
    },
    {
      name: 'Autonomous Systems Architect',
      confidenceScore: 78,
      catalyst: 'Multi-horizon invariant and closed-loop engine implementation',
      evidencePoints: [
        'Authored 4,200+ fail-closed programmatic assertions',
        'Implemented state-machine-driven decision reductions',
      ],
    },
  ],
  fadingIdentities: [
    {
      name: 'Ad-Hoc Discretionary Speculator',
      decayRatePct: 92,
      supersededBy: 'Systematic Value & Momentum Investor',
      observedReduction: 'Zero unhedged or impulsive market entries recorded for 6 consecutive months',
    },
    {
      name: 'Overextended Firefighter',
      decayRatePct: 74,
      supersededBy: 'Autonomous Systems Architect',
      observedReduction: 'Replaced reactive daily firefighting with 30-Second Cockpit decision triage',
    },
  ],
  driftAlerts: [
    {
      id: 'drift-public-influence',
      targetRole: 'AI Strategy Leader',
      domain: 'SYSTEMS',
      observedInactionDays: 68,
      inactivityThresholdDays: 60,
      driftSeverity: 'MEDIUM',
      causalExplanation: 'No external publication or architectural thesis shared in 68 days despite +9 Public Influence target.',
      correctiveActionRecommendation: 'Publish 1 technical briefing summarizing the Closed-Loop Personal Life OS architecture.',
    },
  ],
  lastEvaluated: '2026-09-09T13:40:00Z',
};

/**
 * Calculates the Identity Alignment Index (IAI) [0 - 100]
 * Measures how well current actions reinforce target identity traits
 */
export function calculateIdentityAlignmentIndex(
  recentActions: Array<{ domain: string; completed: boolean; identityLeverage: number }>,
  targetTraits: IdentityTrait[]
): number {
  if (!recentActions || recentActions.length === 0) {
    return 50; // Neutral baseline
  }

  const completedActions = recentActions.filter((a) => a.completed);
  if (completedActions.length === 0) {
    return 30; // Depleted momentum
  }

  // Weighted average of identity leverage
  const totalLeverage = completedActions.reduce((acc, a) => acc + a.identityLeverage, 0);
  const averageLeverage = totalLeverage / completedActions.length;

  // Domain diversity factor (spread across target traits)
  const targetDomains = new Set(targetTraits.map((t) => t.domain));
  const activeDomains = new Set(completedActions.map((a) => a.domain));
  let domainCoverage = 0;
  targetDomains.forEach((d) => {
    if (activeDomains.has(d as any)) domainCoverage++;
  });
  const coverageMultiplier = targetDomains.size > 0 ? domainCoverage / targetDomains.size : 1;

  // Compute composite IAI score clamped between 0 and 100
  const rawScore = averageLeverage * 0.75 + coverageMultiplier * 25;
  return Math.max(0, Math.min(100, Math.round(rawScore)));
}

/**
 * Evaluates whether an inactivity gap triggers an Identity Drift Alert
 */
export function evaluateIdentityDrift(
  domain: string,
  targetRole: string,
  daysInactive: number,
  thresholdDays: number = 60
): IdentityDriftAlert | null {
  if (daysInactive < thresholdDays) {
    return null;
  }

  let severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' = 'LOW';
  if (daysInactive >= 120) severity = 'CRITICAL';
  else if (daysInactive >= 90) severity = 'HIGH';
  else if (daysInactive >= 60) severity = 'MEDIUM';

  return {
    id: `drift-${domain.toLowerCase()}-${Date.now()}`,
    targetRole,
    domain,
    observedInactionDays: daysInactive,
    inactivityThresholdDays: thresholdDays,
    driftSeverity: severity,
    causalExplanation: `Observed ${daysInactive} days of zero measurable progression in "${domain}" supporting target role "${targetRole}".`,
    correctiveActionRecommendation: `Execute 1 micro-commitment (< 25 min) in "${domain}" to arrest identity drift.`,
  };
}

/**
 * Evaluates candidate action identity leverage [0 to 100]
 */
export function scoreActionIdentityLeverage(
  action: { domain: string; estimatedMinutes: number; tags?: string[] },
  twin: IdentityTwin
): { identityLeverage: number; alignedTrait?: IdentityTrait; narrative: string } {
  // Find highest delta trait matching action domain
  const matchingTraits = twin.currentIdentity.traits.filter((t) => t.domain === action.domain);
  if (matchingTraits.length === 0) {
    return {
      identityLeverage: 25,
      narrative: `Action maintains general baseline but does not directly compound target identity.`,
    };
  }

  const primaryTrait = matchingTraits.reduce((max, t) => (t.delta > max.delta ? t : max), matchingTraits[0]);
  
  // Higher leverage if addressing a large trait delta
  let leverage = Math.min(100, 50 + primaryTrait.delta * 2);
  
  // Shorter, high-efficiency actions score higher leverage per minute
  if (action.estimatedMinutes <= 30) {
    leverage = Math.min(100, leverage + 10);
  }

  return {
    identityLeverage: leverage,
    alignedTrait: primaryTrait,
    narrative: `Compounds target trait "${primaryTrait.name}" (+${primaryTrait.delta} delta needed toward "${twin.targetIdentity.roleTitle}").`,
  };
}
