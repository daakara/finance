/**
 * Phase 31-M10.2: Autonomous Action Orchestrator
 *
 * Implements:
 * - Invariant INV-OI55 / INV-OI63: Safe Termination (Unsafe paths terminate before state mutation)
 * - Invariant INV-OI50 / INV-OI58: Autonomous Recommendation Safety
 * - Action Orchestration Lifecycle: Execute, Pause, Reject, Rollback
 * - Deterministic State Inversion & SHA-256 State Locking
 */

import {
  FailCloseStateResponse,
  AnyFailCloseError,
} from '@/types/fail-close-governance';
import {
  AutonomousAction,
  AutonomousActionRequest,
  AutonomousExecutionRecord,
} from '@/types/autonomous-governance';
import { safetyPolicyEngine } from './safetyPolicyEngine';
import { overrideGovernanceEngine } from './overrideGovernanceEngine';
import { actionRegistry } from './autonomousActionRegistry';
import { sha256Hex } from '@/lib/governance/sha256';

export interface ActionExecutionOutcome {
  success: boolean;
  status: 'EXECUTED' | 'PAUSED' | 'REJECTED' | 'SAFE_TERMINATION';
  stateMutated: boolean;
  actionId: string;
  executionRecord?: AutonomousExecutionRecord;
  violations?: AnyFailCloseError[];
  stateResponse: FailCloseStateResponse;
  decisionHash: string;
}

export class AutonomousActionEngine {
  /**
   * Invariant INV-OI55 / INV-OI63: Safe Termination.
   * Unsafe execution paths terminate BEFORE any state mutation occurs.
   */
  public executeAutonomousAction(
    request: AutonomousActionRequest,
    context?: {
      dissentIds?: string[];
      evidenceCount?: number;
      hasRationale?: boolean;
    }
  ): ActionExecutionOutcome {
    const actionId = request.requestId;
    const nowUtc = new Date().toISOString();

    // 1. Check Emergency Stop
    if (overrideGovernanceEngine.isEmergencyStopActive()) {
      return {
        success: false,
        status: 'SAFE_TERMINATION',
        stateMutated: false,
        actionId,
        stateResponse: {
          state: 'FAIL_CLOSED',
          triggeringErrorCode: 'EMERGENCY_STOP',
          certificationRestored: false,
          safeModeEnabled: true,
          blockedCapabilities: ['AUTONOMOUS_EXECUTION'],
          timestampUtc: nowUtc,
        },
        decisionHash: sha256Hex(`SAFE_TERMINATION-${actionId}-EMERGENCY_STOP`),
      };
    }

    // 2. Check Active Human Overrides (INV-OI51 / INV-OI59)
    const activeOverride = overrideGovernanceEngine.getActiveOverride(actionId);
    if (activeOverride) {
      return {
        success: false,
        status: activeOverride.overrideType === 'PAUSE' ? 'PAUSED' : 'REJECTED',
        stateMutated: false,
        actionId,
        stateResponse: {
          state: 'DEGRADED',
          triggeringErrorCode: `OVERRIDE_${activeOverride.overrideType}`,
          certificationRestored: true,
          safeModeEnabled: false,
          blockedCapabilities: ['AUTONOMOUS_EXECUTION'],
          timestampUtc: nowUtc,
        },
        decisionHash: sha256Hex(`OVERRIDE-${actionId}-${activeOverride.overrideType}`),
      };
    }

    // 3. Evaluate Policy Boundaries (INV-OI52 / INV-OI60)
    const policyCheck = safetyPolicyEngine.evaluateActionSafety(request, context);
    if (!policyCheck.passed) {
      return {
        success: false,
        status: 'SAFE_TERMINATION',
        stateMutated: false,
        actionId,
        violations: policyCheck.violations,
        stateResponse: policyCheck.stateResponse,
        decisionHash: sha256Hex(`SAFE_TERMINATION-${actionId}-${policyCheck.violations[0].errorCode}`),
      };
    }

    // 4. Safe to execute - perform state mutation
    const stateMutated = true;
    let executionRecord: AutonomousExecutionRecord;

    try {
      // Find or register action in registry
      let registeredAction = actionRegistry.getAction(actionId);
      if (!registeredAction) {
        registeredAction = {
          actionId,
          category: 'OPTIMIZATION',
          title: request.proposedAction,
          proposedAtUtc: request.initiatedAtUtc,
          approved: true,
          executed: false,
          policyApprovalId: request.policyEvaluationId,
          expectedOutcome: `Validated against policy boundaries. Rationale: ${request.rationale}`,
          rollbackAvailable: true,
          targetCommitteeId: request.committeeId,
          confidenceScore: 95.0,
          status: 'APPROVED',
        };
        actionRegistry.registerAction(registeredAction);
      } else {
        registeredAction.approved = true;
      }

      executionRecord = actionRegistry.executeAction(actionId);
    } catch (err: unknown) {
      return {
        success: false,
        status: 'REJECTED',
        stateMutated: false,
        actionId,
        stateResponse: {
          state: 'FAIL_CLOSED',
          triggeringErrorCode: 'EXECUTION_DISPATCH_FAILURE',
          certificationRestored: false,
          safeModeEnabled: true,
          blockedCapabilities: ['AUTONOMOUS_EXECUTION'],
          timestampUtc: nowUtc,
        },
        decisionHash: sha256Hex(`FAIL-${actionId}`),
      };
    }

    return {
      success: true,
      status: 'EXECUTED',
      stateMutated,
      actionId,
      executionRecord,
      stateResponse: {
        state: 'CERTIFIED',
        triggeringErrorCode: 'NONE',
        certificationRestored: true,
        safeModeEnabled: false,
        blockedCapabilities: [],
        timestampUtc: nowUtc,
      },
      decisionHash: executionRecord.replayHash,
    };
  }

  /**
   * Rolls back an action to pre-execution state.
   */
  public rollbackAction(actionId: string, rationale: string) {
    return actionRegistry.executeRollback(actionId, rationale);
  }
}

export const autonomousActionEngine = new AutonomousActionEngine();
