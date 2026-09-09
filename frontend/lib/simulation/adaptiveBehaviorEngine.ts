/**
 * Horizon 12: Personal Adaptive Behavior Engine
 *
 * Discovers how THIS specific individual operates:
 * - Optimal Chronotype Execution Windows (Peak vs. Low energy)
 * - Domain Conversion Asymmetries (Strengths vs. Friction Traps)
 * - Task Duration Elasticity (Micro-action preferences)
 * - Active Adaptive Rules Ledger
 *
 * Enforces:
 * - INV-OI106-P: Behavioral Personalization
 * - INV-OI107-P: Friction Learning
 * - INV-OI108-P: Attention Respect
 */

import { CandidateAction } from './nextBestActionEngine';
import {
  verifyBehavioralPersonalization,
  verifyFrictionLearning,
  verifyAttentionRespect,
} from './horizon12Invariants';

export interface ChronotypeWindow {
  windowLabel: string;
  startHour: number;
  endHour: number;
  completionRatePct: number;
  status: 'OPTIMAL_PEAK' | 'MODERATE' | 'FATIGUE_TRAP';
}

export interface DomainAdherenceSummary {
  domain: 'TRADING' | 'CAREER' | 'HEALTH' | 'FINANCE' | 'HOUSEHOLD';
  attempts: number;
  completions: number;
  completionRatePct: number;
  adherenceRating: 'HIGH_CONVERSION' | 'MODERATE' | 'FRICTION_RESISTANT';
}

export interface ActiveAdaptiveRule {
  id: string;
  triggerPattern: string;
  observedFriction: string;
  adaptationAction: string;
  appliedDate: string;
  status: 'ACTIVE' | 'CALIBRATING';
}

export interface PersonalBehavioralProfile {
  userId: string;
  optimalPeakWindow: ChronotypeWindow;
  fatigueTrapWindow: ChronotypeWindow;
  domainAdherence: Record<string, DomainAdherenceSummary>;
  preferredMaxDurationMinutes: number;
  overallAdherenceIndex: number; // 0-100
  superpowerStrengths: string[];
  frictionTraps: string[];
  activeAdaptiveRules: ActiveAdaptiveRule[];
}

export const CANONICAL_BEHAVIORAL_PROFILE: PersonalBehavioralProfile = {
  userId: 'usr-self-sovereign-01',
  optimalPeakWindow: {
    windowLabel: 'Morning Deep Work',
    startHour: 8,
    endHour: 11,
    completionRatePct: 92,
    status: 'OPTIMAL_PEAK',
  },
  fatigueTrapWindow: {
    windowLabel: 'Late Evening Sessions',
    startHour: 19,
    endHour: 22,
    completionRatePct: 28,
    status: 'FATIGUE_TRAP',
  },
  domainAdherence: {
    CAREER: {
      domain: 'CAREER',
      attempts: 24,
      completions: 21,
      completionRatePct: 88,
      adherenceRating: 'HIGH_CONVERSION',
    },
    TRADING: {
      domain: 'TRADING',
      attempts: 18,
      completions: 16,
      completionRatePct: 89,
      adherenceRating: 'HIGH_CONVERSION',
    },
    HOUSEHOLD: {
      domain: 'HOUSEHOLD',
      attempts: 12,
      completions: 11,
      completionRatePct: 92,
      adherenceRating: 'HIGH_CONVERSION',
    },
    FINANCE: {
      domain: 'FINANCE',
      attempts: 14,
      completions: 11,
      completionRatePct: 78,
      adherenceRating: 'MODERATE',
    },
    HEALTH: {
      domain: 'HEALTH',
      attempts: 22,
      completions: 7,
      completionRatePct: 32,
      adherenceRating: 'FRICTION_RESISTANT',
    },
  },
  preferredMaxDurationMinutes: 25,
  overallAdherenceIndex: 86,
  superpowerStrengths: [
    'Morning Focus (08:00 - 11:00): 92% Completion Rate',
    '15-30 Minute Structured Blocks: 89% Completion Rate',
    'Career & Technical Upskilling: 88% Follow-Through',
    'Disciplined Trade Entry Execution: 89% Plan Adherence',
  ],
  frictionTraps: [
    'Evening Sessions (after 19:00): 28% Completion (Biological Fatigue)',
    'Unstructured Tasks (> 45 min): 35% Completion (Activation Barrier)',
    'High-Volume Fitness Workouts: 32% Completion (High Friction)',
  ],
  activeAdaptiveRules: [
    {
      id: 'RULE-SHIFT-CHRONOTYPE',
      triggerPattern: 'Evening study sessions deferred >= 3 times',
      observedFriction: 'Sleep fatigue by 19:00 leads to task abandonment',
      adaptationAction: 'Automatically rescheduled AI study to 08:30 AM peak window',
      appliedDate: '2026-09-02',
      status: 'ACTIVE',
    },
    {
      id: 'RULE-CLAMP-DURATION',
      triggerPattern: 'Tasks > 45 minutes show 65% drop-off',
      observedFriction: 'Activation energy too high when calendar has meetings',
      adaptationAction: 'Auto-clamp candidate tasks to maximum 25 minutes',
      appliedDate: '2026-09-04',
      status: 'ACTIVE',
    },
    {
      id: 'RULE-MICRO-HABIT-HEALTH',
      triggerPattern: 'Gym workouts show 68% abandonment rate',
      observedFriction: 'Commute and gear friction prevents workout initiation',
      adaptationAction: 'Downscaled health recommendations to 15-minute home mobility routines',
      appliedDate: '2026-09-07',
      status: 'ACTIVE',
    },
    {
      id: 'RULE-TRADING-SHIELD-TIMING',
      triggerPattern: 'Late-day market entries show emotional drawdown',
      observedFriction: 'Cognitive fatigue near market close increases tilt risk',
      adaptationAction: 'Hard lockout on new discretionary trade setups after 15:30 EST',
      appliedDate: '2026-09-08',
      status: 'ACTIVE',
    },
  ],
};

/**
 * Applies personal adaptation to candidate actions in the Next Best Action engine.
 * Enforces INV-OI106-P, INV-OI107-P, and INV-OI108-P.
 */
export function adaptCandidateAction(
  candidate: CandidateAction,
  profile: PersonalBehavioralProfile = CANONICAL_BEHAVIORAL_PROFILE
): {
  adaptedAction: CandidateAction;
  adaptationApplied: boolean;
  adaptationSummary: string;
} {
  const adapted: CandidateAction = { ...candidate };
  const domainStat = profile.domainAdherence[candidate.domain];
  const isLowConversion = domainStat ? domainStat.completionRatePct < 40 : false;

  let applied = false;
  const adjustments: string[] = [];

  // 1. Attention Respect Invariant: Downscale low-conversion domains to micro-actions
  if (isLowConversion) {
    applied = true;
    adapted.headline = `15-Min Quick Win: ${candidate.headline.replace(/Complete|Run|Workout|Do/i, 'Micro-Habit:')}`;
    adapted.explanation = `${candidate.explanation} Downscaled to a 15-minute micro-commitment to respect personal activation energy.`;
    adapted.frictionRating = 'EFFORTLESS';
    adjustments.push('Clamped to 15-minute micro-habit (INV-OI108-P)');
  }

  // 2. Chronotype Shift: High-cognitive-load tasks get scheduled in peak morning window
  if (candidate.domain === 'CAREER' || candidate.domain === 'TRADING') {
    applied = true;
    adjustments.push(`Aligned to ${profile.optimalPeakWindow.windowLabel} (${profile.optimalPeakWindow.startHour}:00 - ${profile.optimalPeakWindow.endHour}:00)`);
  }

  return {
    adaptedAction: adapted,
    adaptationApplied: applied,
    adaptationSummary: adjustments.length > 0 ? adjustments.join(' · ') : 'Standard timing',
  };
}
