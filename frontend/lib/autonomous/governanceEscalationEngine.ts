/**
 * Phase 31-M10.4: Governance Escalation Engine
 *
 * Implements:
 * - Invariant INV-OI53 / INV-OI61: Escalation Integrity (Zero silent drops; SLA enforcement)
 * - Automatic Severity Upgrades (HIGH -> CRITICAL upon SLA breach)
 * - Typed Fail-Close Escalation Violations:
 *   - GOV-ESC-001: Escalation Suppression
 *   - GOV-ESC-002: SLA Breach
 *   - GOV-ESC-003: Escalation Routing Failure
 *   - GOV-ESC-004: Severity Downgrade Manipulation
 */

import {
  EscalationSuppressionError,
  EscalationSlaViolationError,
  EscalationRoutingError,
  SeverityDowngradeViolationError,
  EscalationError,
  FailCloseSeverity,
} from '@/types/fail-close-governance';

export interface EscalationIncident {
  incidentId: string;
  title: string;
  sourceAlertId: string;
  severity: FailCloseSeverity;
  status: 'PENDING' | 'ESCALATED' | 'RESOLVED';
  escalatedToRole: string;
  targetSlaHours: number;
  createdAtUtc: string;
  resolvedAtUtc?: string;
  suppressed?: boolean;
}

export class GovernanceEscalationEngine {
  private incidents: Map<string, EscalationIncident> = new Map();
  private escalationViolations: EscalationError[] = [];

  constructor() {
    this.reset();
  }

  public reset(): void {
    this.incidents.clear();
    this.escalationViolations = [];
    // Seed baseline escalation incident
    this.incidents.set('INC-ESC-001', {
      incidentId: 'INC-ESC-001',
      title: 'VaR 99% Drawdown Floor Exceeded',
      sourceAlertId: 'ALT-RSK-001',
      severity: 'CRITICAL',
      status: 'ESCALATED',
      escalatedToRole: 'CHIEF_RISK_OFFICER',
      targetSlaHours: 1.0,
      createdAtUtc: new Date().toISOString(),
    });
  }

  public getIncidents(): EscalationIncident[] {
    return Array.from(this.incidents.values());
  }

  public getViolations(): EscalationError[] {
    return [...this.escalationViolations];
  }

  /**
   * Evaluates an incident for automatic escalation (INV-OI53 / INV-OI61).
   * Guarantees zero silent drops.
   */
  public processEscalation(
    incident: EscalationIncident
  ): { escalated: boolean; incident: EscalationIncident; violation?: EscalationError } {
    const correlationId = `CORR-ESC-${Date.now().toString(36)}`;
    const nowUtc = new Date().toISOString();

    // Check for suppression (GOV-ESC-001)
    if (incident.suppressed) {
      const err: EscalationSuppressionError = {
        errorCode: 'GOV-ESC-001',
        errorType: 'ESCALATION_VIOLATION',
        severity: 'CRITICAL',
        certificationImpact: 'FAILED',
        message: `Critical alert ${incident.sourceAlertId} was marked suppressed without executive authority`,
        invariantId: 'INV-OI61',
        affectedArtifactId: incident.incidentId,
        correlationId,
        detectedAtUtc: nowUtc,
        failCloseActivated: true,
        recoveryRequired: true,
        recommendedRecoveryMode: 'AUTO_REPAIR',
        alertId: incident.sourceAlertId,
        severityLevel: incident.severity,
        escalationTarget: incident.escalatedToRole,
      };
      this.escalationViolations.push(err);
      // Auto-repair: un-suppress and escalate
      incident.suppressed = false;
      incident.status = 'ESCALATED';
      this.incidents.set(incident.incidentId, incident);
      return { escalated: true, incident, violation: err };
    }

    // Check for routing resolution (GOV-ESC-003)
    if (!incident.escalatedToRole || incident.escalatedToRole === 'UNKNOWN') {
      const err: EscalationRoutingError = {
        errorCode: 'GOV-ESC-003',
        errorType: 'ESCALATION_VIOLATION',
        severity: 'HIGH',
        certificationImpact: 'DEGRADED',
        message: `Notification target role cannot be resolved for incident ${incident.incidentId}`,
        invariantId: 'INV-OI61',
        affectedArtifactId: incident.incidentId,
        correlationId,
        detectedAtUtc: nowUtc,
        failCloseActivated: true,
        recoveryRequired: true,
        recommendedRecoveryMode: 'SAFE_MODE',
        escalationRole: incident.escalatedToRole || 'UNRESOLVED',
        alertId: incident.sourceAlertId,
        routingAttempts: 3,
      };
      this.escalationViolations.push(err);
      incident.escalatedToRole = 'BOARD_ADMIN'; // Fallback to fail-closed board admin
      incident.status = 'ESCALATED';
      this.incidents.set(incident.incidentId, incident);
      return { escalated: true, incident, violation: err };
    }

    if (incident.status === 'RESOLVED') {
      return { escalated: false, incident };
    }

    incident.status = 'ESCALATED';
    this.incidents.set(incident.incidentId, incident);
    return { escalated: true, incident };
  }

  /**
   * Checks SLA duration and auto-escalates HIGH to CRITICAL on breach (GOV-ESC-002).
   */
  public evaluateSlaBreach(
    incidentId: string,
    elapsedHours: number
  ): { breached: boolean; violation?: EscalationSlaViolationError } {
    const incident = this.incidents.get(incidentId);
    if (!incident) throw new Error(`Incident ${incidentId} not found`);

    if (incident.status !== 'RESOLVED' && elapsedHours > incident.targetSlaHours) {
      incident.severity = 'CRITICAL';
      const violation: EscalationSlaViolationError = {
        errorCode: 'GOV-ESC-002',
        errorType: 'ESCALATION_VIOLATION',
        severity: 'HIGH',
        certificationImpact: 'DEGRADED',
        message: `Escalation SLA breached for incident ${incidentId}: ${elapsedHours.toFixed(1)}h elapsed (Target: ${incident.targetSlaHours.toFixed(1)}h)`,
        invariantId: 'INV-OI61',
        affectedArtifactId: incidentId,
        correlationId: `CORR-SLA-${Date.now().toString(36)}`,
        detectedAtUtc: new Date().toISOString(),
        failCloseActivated: true,
        recoveryRequired: true,
        recommendedRecoveryMode: 'MANUAL_REVIEW',
        alertId: incident.sourceAlertId,
        targetSlaHours: incident.targetSlaHours,
        actualElapsedHours: elapsedHours,
      };
      this.escalationViolations.push(violation);
      this.incidents.set(incidentId, incident);
      return { breached: true, violation };
    }

    return { breached: false };
  }

  /**
   * Prevents severity downgrade without justification (GOV-ESC-004).
   */
  public attemptSeverityDowngrade(
    incidentId: string,
    newSeverity: FailCloseSeverity,
    justification?: string
  ): { allowed: boolean; violation?: SeverityDowngradeViolationError } {
    const incident = this.incidents.get(incidentId);
    if (!incident) throw new Error(`Incident ${incidentId} not found`);

    const originalSeverity = incident.severity;
    const isDowngrade = originalSeverity === 'CRITICAL' && newSeverity === 'HIGH';

    if (isDowngrade && (!justification || justification.length < 10)) {
      const violation: SeverityDowngradeViolationError = {
        errorCode: 'GOV-ESC-004',
        errorType: 'ESCALATION_VIOLATION',
        severity: 'CRITICAL',
        certificationImpact: 'FAILED',
        message: `Severity downgrade from ${originalSeverity} to ${newSeverity} rejected without verified evidence and rationale`,
        invariantId: 'INV-OI61',
        affectedArtifactId: incidentId,
        correlationId: `CORR-DOWNSCALE-${Date.now().toString(36)}`,
        detectedAtUtc: new Date().toISOString(),
        failCloseActivated: true,
        recoveryRequired: false,
        recommendedRecoveryMode: 'MANUAL_REVIEW',
        originalSeverity,
        modifiedSeverity: newSeverity,
        justificationPresent: false,
      };
      this.escalationViolations.push(violation);
      return { allowed: false, violation };
    }

    incident.severity = newSeverity;
    this.incidents.set(incidentId, incident);
    return { allowed: true };
  }
}

export const governanceEscalationEngine = new GovernanceEscalationEngine();
