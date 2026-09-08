/**
 * Phase 31-M10.1: Safety Policy Engine
 *
 * Implements:
 * - Invariant INV-OI52 / INV-OI60: Policy Boundary Enforcement
 * - Invariant INV-OI50 / INV-OI58: Autonomous Recommendation Safety
 * - Typed Fail-Close Policy Boundary Violations:
 *   - GOV-POL-001: Action Outside Approved Policy
 *   - GOV-POL-002: Autonomous Execution Boundary Breach
 *   - GOV-POL-003: Risk Tolerance Breach
 *   - GOV-POL-004: Mandatory Dissent Protection Breach (INV-OI14 / INV-OI21)
 *   - GOV-POL-005: Explainability Boundary Breach (INV-OI23 / INV-OI32)
 */

import {
  FailCloseStateResponse,
  PolicyBoundaryViolationError,
  AutonomousBoundaryViolationError,
  RiskToleranceViolationError,
  DissentProtectionViolationError,
  ExplainabilityBoundaryViolationError,
  PolicyViolationErrorM10,
} from '@/types/fail-close-governance';
import { GovernancePolicy, AutonomousActionRequest, CANONICAL_POLICIES } from '@/types/autonomous-governance';
import { sha256Hex } from '@/lib/governance/sha256';

export interface PolicyEvaluationAssessment {
  passed: boolean;
  actionAllowed: boolean;
  violations: PolicyViolationErrorM10[];
  stateResponse: FailCloseStateResponse;
  evaluatedAtUtc: string;
}

export class SafetyPolicyEngine {
  private activePolicies: GovernancePolicy[];

  constructor(policies: GovernancePolicy[] = CANONICAL_POLICIES) {
    this.activePolicies = JSON.parse(JSON.stringify(policies));
  }

  /**
   * Evaluates action safety against defined governance and risk boundaries.
   */
  public evaluateActionSafety(
    request: AutonomousActionRequest,
    context?: {
      dissentIds?: string[];
      evidenceCount?: number;
      hasRationale?: boolean;
      agentId?: string;
    }
  ): PolicyEvaluationAssessment {
    const violations: PolicyViolationErrorM10[] = [];
    const correlationId = `CORR-${request.requestId}-${Date.now().toString(36)}`;
    const nowUtc = new Date().toISOString();

    // 1. GOV-POL-003: Risk Tolerance Breach
    const targetRisk = request.targetRiskScore ?? 50.0;
    const maxApprovedRisk = 80.0;
    if (targetRisk > maxApprovedRisk) {
      const err: RiskToleranceViolationError = {
        errorCode: 'GOV-POL-003',
        errorType: 'POLICY_BOUNDARY_VIOLATION',
        severity: 'CRITICAL',
        certificationImpact: 'FAILED',
        message: `Projected risk score ${targetRisk.toFixed(1)} exceeds approved tolerance ${maxApprovedRisk.toFixed(1)}`,
        invariantId: 'INV-OI60',
        affectedArtifactId: request.requestId,
        correlationId,
        detectedAtUtc: nowUtc,
        failCloseActivated: true,
        recoveryRequired: true,
        recommendedRecoveryMode: 'SAFE_MODE',
        approvedRiskLimit: maxApprovedRisk,
        projectedRisk: targetRisk,
        excessRisk: targetRisk - maxApprovedRisk,
      };
      violations.push(err);
    }

    // 2. GOV-POL-001: Action Outside Approved Policy (Budget limits)
    const budgetRequested = request.budgetRequestedDollars ?? 0;
    if (budgetRequested > 100000) {
      const err: PolicyBoundaryViolationError = {
        errorCode: 'GOV-POL-001',
        errorType: 'POLICY_BOUNDARY_VIOLATION',
        severity: 'HIGH',
        certificationImpact: 'DEGRADED',
        message: `Action budget $${budgetRequested.toLocaleString()} exceeds authorized policy limit of $100,000 without prior executive approval`,
        invariantId: 'INV-OI60',
        affectedArtifactId: request.requestId,
        correlationId,
        detectedAtUtc: nowUtc,
        failCloseActivated: true,
        recoveryRequired: true,
        recommendedRecoveryMode: 'MANUAL_REVIEW',
        policyId: 'POL-RISK-001',
        attemptedAction: request.proposedAction,
        allowedActions: ['BUDGET_REALLOCATION_TIER_1', 'PORTFOLIO_DAMPENING'],
      };
      violations.push(err);
    }

    // 3. GOV-POL-002: Autonomous Execution Boundary Breach (Restricted operations)
    const actionLower = request.proposedAction.toLowerCase();
    if (actionLower.includes('charter') || actionLower.includes('emergency_charter') || actionLower.includes('restricted')) {
      const err: AutonomousBoundaryViolationError = {
        errorCode: 'GOV-POL-002',
        errorType: 'POLICY_BOUNDARY_VIOLATION',
        severity: 'CRITICAL',
        certificationImpact: 'FAILED',
        message: `Autonomous execution prohibited on charter-modifying or restricted governance domain`,
        invariantId: 'INV-OI58',
        affectedArtifactId: request.requestId,
        correlationId,
        detectedAtUtc: nowUtc,
        failCloseActivated: true,
        recoveryRequired: true,
        recommendedRecoveryMode: 'SAFE_MODE',
        autonomousAgentId: context?.agentId ?? 'AGENT-AUTONOMOUS-CORE',
        actionCategory: 'CHARTER_MODIFICATION',
        policyBoundaryId: 'POL-GOV-001',
      };
      violations.push(err);
    }

    // 4. GOV-POL-004: Mandatory Dissent Protection Breach (INV-OI14 / INV-OI21)
    if (context && context.dissentIds !== undefined && context.dissentIds.length === 0 && targetRisk > 70.0) {
      const err: DissentProtectionViolationError = {
        errorCode: 'GOV-POL-004',
        errorType: 'POLICY_BOUNDARY_VIOLATION',
        severity: 'HIGH',
        certificationImpact: 'DEGRADED',
        message: `Execution suppresses mandatory dissent review on high-conviction decision (DIRatio floor violated)`,
        invariantId: 'INV-OI14',
        affectedArtifactId: request.requestId,
        correlationId,
        detectedAtUtc: nowUtc,
        failCloseActivated: true,
        recoveryRequired: true,
        recommendedRecoveryMode: 'ROLLBACK',
        dissentIds: [],
        suppressionAttemptDetected: true,
      };
      violations.push(err);
    }

    // 5. GOV-POL-005: Explainability Boundary Breach (INV-OI23 / INV-OI32)
    const evidenceCount = context?.evidenceCount ?? 2;
    const hasRationale = context?.hasRationale ?? (request.rationale.length > 10);
    if (evidenceCount < 1 || !hasRationale) {
      const err: ExplainabilityBoundaryViolationError = {
        errorCode: 'GOV-POL-005',
        errorType: 'POLICY_BOUNDARY_VIOLATION',
        severity: 'HIGH',
        certificationImpact: 'FAILED',
        message: `Autonomous action lacks mandatory explainability evidence or rationale trace`,
        invariantId: 'INV-OI62',
        affectedArtifactId: request.requestId,
        correlationId,
        detectedAtUtc: nowUtc,
        failCloseActivated: true,
        recoveryRequired: false,
        recommendedRecoveryMode: 'MANUAL_REVIEW',
        artifactId: request.requestId,
        missingEvidenceCount: evidenceCount === 0 ? 1 : 0,
        missingRationale: !hasRationale,
      };
      violations.push(err);
    }

    const passed = violations.length === 0;
    const actionAllowed = passed;

    const stateResponse: FailCloseStateResponse = {
      state: passed ? 'CERTIFIED' : violations.some(v => v.severity === 'CRITICAL') ? 'FAIL_CLOSED' : 'DEGRADED',
      triggeringErrorCode: passed ? 'NONE' : violations[0].errorCode,
      certificationRestored: passed,
      safeModeEnabled: !passed && violations.some(v => v.severity === 'CRITICAL'),
      blockedCapabilities: passed ? [] : ['AUTONOMOUS_EXECUTION', 'RECOMMENDATION_APPROVAL'],
      timestampUtc: nowUtc,
    };

    return {
      passed,
      actionAllowed,
      violations,
      stateResponse,
      evaluatedAtUtc: nowUtc,
    };
  }
}

export const safetyPolicyEngine = new SafetyPolicyEngine();
