/**
 * ARX User Outcome Telemetry & Executive Execution Tracker Types
 * 
 * Philosophy:
 * 1. Was it seen? (SEEN)
 * 2. Was it understood? (UNDERSTOOD)
 * 3. Was it acted upon? (ACTED_UPON)
 * 4. Did behavior improve? (BEHAVIOR_IMPROVED)
 * 
 * Event Taxonomy:
 * ENGAGEMENT | DECISION | LEARNING | PLAYBOOK | MENTOR | GOVERNANCE | OUTCOME
 */

export type OutcomeTelemetryPhase =
  | 'SEEN'
  | 'UNDERSTOOD'
  | 'ACTED_UPON'
  | 'BEHAVIOR_IMPROVED';

export type UserOutcomeEventCategory =
  | 'ENGAGEMENT'
  | 'DECISION'
  | 'LEARNING'
  | 'PLAYBOOK'
  | 'MENTOR'
  | 'GOVERNANCE'
  | 'OUTCOME';

export type UserOutcomeEventName =
  | 'mentor_viewed'
  | 'mentor_recommendation_clicked'
  | 'mentor_evidence_opened'
  | 'learning_journey_viewed'
  | 'milestone_expanded'
  | 'next_milestone_clicked'
  | 'playbook_viewed'
  | 'rule_opened'
  | 'rule_followed'
  | 'adoption_dashboard_viewed'
  | 'drift_warning_seen'
  | 'drift_warning_acknowledged'
  | 'lifecycle_stage_viewed'
  | 'lifecycle_transition_completed'
  // Phase 28 Milestone 1 Behavioral Telemetry
  | 'outcome_review_viewed'
  | 'decision_journal_viewed'
  | 'learning_coach_opened'
  | 'learning_coach_accepted'
  | 'learning_recommendation_completed'
  | 'repeat_error_occurred'
  | 'behavior_improvement_detected';

export interface MentorViewedPayload {
  screen: string;
  mentor_type: 'attention' | 'decision' | 'attribution' | 'learning' | 'playbook' | 'governance';
  user_id: string;
  timestamp: string;
}

export interface MentorRecommendationClickedPayload {
  recommendation_type: 'DO_MORE' | 'STOP_DOING' | 'CALIBRATE';
  confidence: number;
  projected_impact: string;
  ticker?: string;
}

export interface MentorEvidenceOpenedPayload {
  sample_size: number;
  p_value: number;
  ledger_hash: string;
}

export interface LearningJourneyViewedPayload {
  current_score: number;
  baseline_score: number;
  quarters_tracked: number;
}

export interface MilestoneExpandedPayload {
  milestone: string;
  score: number;
  contributor: string;
}

export interface NextMilestoneClickedPayload {
  target_score: number;
  points_remaining: number;
}

export interface PlaybookViewedPayload {
  active_rules_count: number;
  user_id: string;
}

export interface RuleOpenedPayload {
  rule_type: 'DO_MORE' | 'STOP_DOING' | 'CALIBRATE';
  rule_id: string;
  rule_title: string;
}

export interface RuleFollowedPayload {
  rule_id: string;
  rule_type: string;
  action_detected: string;
  adherence_rate: number;
}

export interface AdoptionDashboardViewedPayload {
  bar_percentage: number;
  drift_score: number;
  repeat_mistake_reduction: number;
}

export interface DriftWarningSeenPayload {
  drift_score: number;
  drift_band: 'LOW' | 'MEDIUM' | 'HIGH';
  trigger_reason: string;
}

export interface DriftWarningAcknowledgedPayload {
  drift_score: number;
  acknowledged_at: string;
  action_selected: 'RE_CALIBRATE' | 'DISMISS';
}

export interface LifecycleStageViewedPayload {
  stage: string;
  ticker?: string;
  duration_ms?: number;
}

export interface LifecycleTransitionCompletedPayload {
  from: string;
  to: string;
  ticker?: string;
  actor: string;
}

export interface OutcomeReviewViewedPayload {
  review_id: string;
  decision_id: string;
  quality_delta: number;
  user_id: string;
}

export interface DecisionJournalViewedPayload {
  journal_entries_count: number;
  user_id: string;
}

export interface LearningCoachOpenedPayload {
  context: string;
  source: 'executive_home' | 'decision_workspace' | 'playbook' | 'timeline';
  user_id: string;
}

export interface LearningCoachAcceptedPayload {
  recommendation_id: string;
  coach_focus_area: string;
  projected_impact: number;
}

export interface LearningRecommendationCompletedPayload {
  recommendation_id: string;
  completion_time_ms: number;
  evidence_verified: boolean;
}

export interface RepeatErrorOccurredPayload {
  error_type: string;
  prior_occurrence_count: number;
  drift_penalty: number;
}

export interface BehaviorImprovementDetectedPayload {
  behavior_category: string;
  points_gained: number;
  rule_adherence_rate: number;
}

export interface UserOutcomeTelemetryEvent {
  id: string;
  event: UserOutcomeEventName;
  category: UserOutcomeEventCategory;
  phase: OutcomeTelemetryPhase;
  timestamp: string;
  userId: string;
  payload: Record<string, unknown>;
}

export interface ExecutiveTrackerMetrics {
  // Platform Health
  currentReadinessScore: number; // 95.0%
  targetReadinessScore: number; // 98.0%
  releaseStatus: 'Release Candidate' | 'Institutional Production Certified';
  openCriticalIssues: number;
  openMajorIssues: number;

  // Tier 1: Strategic Outcome Metrics
  decisionQualityScore: number; // 74
  decisionQualityDeltaYearly: number; // +12
  behavioralAdoptionRate: number; // 70.5% (Target > 70%)
  repeatMistakeReduction: number; // -43% (Target > 30%)
  decisionDrift: number; // 21% (Target < 25%)

  // Tier 2: Product Metrics
  mentorVisibilityRate: number; // 94% (Target > 95%)
  recommendationEngagementRate: number; // 67% (Target > 60%)
  trustValidationRate: number; // 44% (Target 30%-70%)
  playbookAdoptionReach: number; // 82% (Target > 75%)
  learningJourneyReach: number; // 76% (Target > 70%)
  ruleAdherenceRate: number; // 87% (Target > 80%)

  // Tier 3: UX Speed Test
  questionResolutionAvgTimeSec: number; // 2.1s (Target < 5s)
}

export interface ReleaseGateItem {
  id: string;
  name: string;
  status: 'CERTIFIED' | 'IN_PROGRESS' | 'PENDING';
  evidence: string;
}
