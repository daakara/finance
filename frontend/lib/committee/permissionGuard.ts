/**
 * ARX Terminal vNext - Committee Permission Guard
 * Enforces role-based access control for baselines, approvals, and audit trails.
 * Reference: docs/architecture/COMMITTEE_INTELLIGENCE_ARCHITECTURE.md
 */

import { CommitteeRole } from "../../types/committee-intelligence";

export function canCreateBaseline(role: CommitteeRole): boolean {
  return role === CommitteeRole.PORTFOLIO_MANAGER || role === CommitteeRole.CIO;
}

export function canApproveBaseline(role: CommitteeRole): boolean {
  return role === CommitteeRole.CIO;
}

export function canAcknowledge(role: CommitteeRole): boolean {
  return role === CommitteeRole.PORTFOLIO_MANAGER || role === CommitteeRole.CIO;
}

export function canAddResearchNote(role: CommitteeRole): boolean {
  return (
    role === CommitteeRole.ANALYST ||
    role === CommitteeRole.PORTFOLIO_MANAGER ||
    role === CommitteeRole.CIO
  );
}

export function canAlterAudit(role: CommitteeRole): boolean {
  // Invariant: Even ADMIN is strictly forbidden from altering historical audit records
  return false;
}
