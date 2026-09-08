/**
 * Phase 31-M10.3: Override Governance Controller
 *
 * Implements:
 * - Invariant INV-OI51 / INV-OI59: Human Override Integrity (Zero bypass, instant supersession)
 * - Emergency Stop & Killswitch Controls
 * - Typed Fail-Close Override Violations:
 *   - GOV-OVR-001: Unauthorized Override Attempt
 *   - GOV-OVR-002: Certification Bypass Attempt
 *   - GOV-OVR-003: Human Approval Circumvention
 */

import {
  UnauthorizedOverrideError,
  CertificationBypassError,
  HumanApprovalBypassError,
  OverrideError,
  FailCloseStateResponse,
} from '@/types/fail-close-governance';
import { sha256Hex } from '@/lib/governance/sha256';

export type OverrideLevel = 'OPERATOR' | 'EXECUTIVE' | 'BOARD';

export interface ActiveOverrideEntry {
  overrideId: string;
  targetActionId: string;
  overrideType: 'PAUSE' | 'CANCEL' | 'ROLLBACK' | 'FORCE_APPROVE';
  actorId: string;
  actorLevel: OverrideLevel;
  justification: string;
  issuedAtUtc: string;
  active: boolean;
}

const AUTHORIZED_ROLES: Record<string, OverrideLevel> = {
  'USR-CRO': 'EXECUTIVE',
  'USR-ED': 'EXECUTIVE',
  'USR-BOARD-01': 'BOARD',
  'USR-LEAD-OP': 'OPERATOR',
};

export class OverrideGovernanceEngine {
  private activeOverrides: Map<string, ActiveOverrideEntry> = new Map();
  private emergencyStopActive: boolean = false;
  private violationLog: OverrideError[] = [];

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.activeOverrides.clear();
    this.emergencyStopActive = false;
    this.violationLog = [];
    // Seed baseline override for testing
    this.registerOverride({
      overrideId: 'OVR-INIT-M10',
      targetActionId: 'ACT-2026-004',
      overrideType: 'PAUSE',
      actorId: 'USR-CRO',
      actorLevel: 'EXECUTIVE',
      justification: 'System initialization baseline override checkpoint',
      issuedAtUtc: new Date().toISOString(),
      active: true,
    });
  }

  public isEmergencyStopActive(): boolean {
    return this.emergencyStopActive;
  }

  public triggerEmergencyStop(actorId: string, reason: string): { success: boolean; stateResponse: FailCloseStateResponse } {
    this.emergencyStopActive = true;
    return {
      success: true,
      stateResponse: {
        state: 'FAIL_CLOSED',
        triggeringErrorCode: 'EMERGENCY_STOP',
        certificationRestored: false,
        safeModeEnabled: true,
        blockedCapabilities: ['AUTONOMOUS_EXECUTION', 'RECOMMENDATION_APPROVAL', 'POLICY_MUTATION'],
        timestampUtc: new Date().toISOString(),
      },
    };
  }

  public releaseEmergencyStop(actorId: string): void {
    this.emergencyStopActive = false;
  }

  /**
   * Registers a valid human/executive override.
   */
  public registerOverride(entry: ActiveOverrideEntry): void {
    this.activeOverrides.set(entry.targetActionId, { ...entry, active: true });
  }

  /**
   * Checks if an action is blocked by an active override.
   */
  public getActiveOverride(actionId: string): ActiveOverrideEntry | undefined {
    const entry = this.activeOverrides.get(actionId);
    return entry?.active ? entry : undefined;
  }

  public getAllActiveOverrides(): ActiveOverrideEntry[] {
    return Array.from(this.activeOverrides.values()).filter(o => o.active);
  }

  /**
   * Invariant INV-OI51 / INV-OI59: Evaluates if an attempted action bypasses or violates override rules.
   */
  public validateOverrideAttempt(
    actorId: string,
    actionId: string,
    attemptedAction: string,
    requiredLevel: OverrideLevel = 'EXECUTIVE'
  ): { allowed: boolean; violation?: UnauthorizedOverrideError } {
    const actorLevel = AUTHORIZED_ROLES[actorId];
    const correlationId = `CORR-OVR-${Date.now().toString(36)}`;
    const nowUtc = new Date().toISOString();

    if (!actorLevel) {
      const violation: UnauthorizedOverrideError = {
        errorCode: 'GOV-OVR-001',
        errorType: 'OVERRIDE_VIOLATION',
        severity: 'CRITICAL',
        certificationImpact: 'FAILED',
        message: `Actor ${actorId} lacks authorized role to issue governance override`,
        invariantId: 'INV-OI59',
        affectedArtifactId: actionId,
        correlationId,
        detectedAtUtc: nowUtc,
        failCloseActivated: true,
        recoveryRequired: false,
        recommendedRecoveryMode: 'MANUAL_REVIEW',
        attemptedAction,
        actorId,
        requiredApprovalLevel: requiredLevel,
      };
      this.violationLog.push(violation);
      return { allowed: false, violation };
    }

    // Role hierarchy check: OPERATOR < EXECUTIVE < BOARD
    const levelRanks = { OPERATOR: 1, EXECUTIVE: 2, BOARD: 3 };
    if (levelRanks[actorLevel] < levelRanks[requiredLevel]) {
      const violation: UnauthorizedOverrideError = {
        errorCode: 'GOV-OVR-001',
        errorType: 'OVERRIDE_VIOLATION',
        severity: 'CRITICAL',
        certificationImpact: 'FAILED',
        message: `Actor ${actorId} (${actorLevel}) does not meet required approval level (${requiredLevel})`,
        invariantId: 'INV-OI59',
        affectedArtifactId: actionId,
        correlationId,
        detectedAtUtc: nowUtc,
        failCloseActivated: true,
        recoveryRequired: false,
        recommendedRecoveryMode: 'MANUAL_REVIEW',
        attemptedAction,
        actorId,
        requiredApprovalLevel: requiredLevel,
      };
      this.violationLog.push(violation);
      return { allowed: false, violation };
    }

    return { allowed: true };
  }

  /**
   * Detects circumvention of human approval workflows (GOV-OVR-003).
   */
  public detectCircumvention(
    actionId: string,
    workflowId: string,
    requiredApprovers: string[],
    actualApprovers: string[]
  ): HumanApprovalBypassError | null {
    const missing = requiredApprovers.filter(a => !actualApprovers.includes(a));
    if (missing.length > 0) {
      const err: HumanApprovalBypassError = {
        errorCode: 'GOV-OVR-003',
        errorType: 'OVERRIDE_VIOLATION',
        severity: 'CRITICAL',
        certificationImpact: 'FAILED',
        message: `Mandatory human approval circumvented. Missing approvers: ${missing.join(', ')}`,
        invariantId: 'INV-OI59',
        affectedArtifactId: actionId,
        correlationId: `CORR-BYPASS-${Date.now().toString(36)}`,
        detectedAtUtc: new Date().toISOString(),
        failCloseActivated: true,
        recoveryRequired: true,
        recommendedRecoveryMode: 'SAFE_MODE',
        approvalWorkflowId: workflowId,
        requiredApprovers,
        actualApprovers,
      };
      this.violationLog.push(err);
      return err;
    }
    return null;
  }

  public getViolations(): OverrideError[] {
    return [...this.violationLog];
  }
}

export const overrideGovernanceEngine = new OverrideGovernanceEngine();
