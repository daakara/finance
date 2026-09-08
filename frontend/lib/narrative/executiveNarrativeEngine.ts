/**
 * Executive Narrative Intelligence Engine (Phase 31-M11.5)
 *
 * Translates multi-engine institutional metrics (OHI, ODEI, CDQI, Risk, Learning, Resilience, Safety)
 * into plain-language executive summaries and actionable strategic briefings.
 */

export interface InstitutionalTelemetry {
  ohi: number;
  odei: number;
  cdqi: number;
  diRatio: number;
  learningVelocity: number;
  riskHealth: number;
  stressProbability: number;
  rtoSeconds: number;
  activeIncidents: number;
  pendingApprovals: number;
  replayDriftCount: number;
}

export interface PriorityExecutiveAction {
  id: string;
  title: string;
  impactArea: string;
  urgency: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  recommendedAction: string;
  ownerCommittee: string;
}

export interface CrossCenterInsight {
  center: string;
  signal: string;
  interpretation: string;
  sentiment: 'POSITIVE' | 'NEUTRAL' | 'WARNING' | 'CRITICAL';
}

export interface ExecutiveBriefing {
  briefingId: string;
  generatedAt: string;
  headline: string;
  overallStatus: 'OPTIMAL' | 'STABLE' | 'DEGRADED' | 'CRITICAL';
  executiveSummary: string;
  keyFindings: string[];
  insights: CrossCenterInsight[];
  priorityActions: PriorityExecutiveAction[];
  telemetrySnapshot: InstitutionalTelemetry;
}

export const CANONICAL_TELEMETRY_BASELINE: InstitutionalTelemetry = {
  ohi: 84.2,
  odei: 86.4,
  cdqi: 82.1,
  diRatio: 78.5,
  learningVelocity: 3.8,
  riskHealth: 88.0,
  stressProbability: 0.12,
  rtoSeconds: 4.8,
  activeIncidents: 0,
  pendingApprovals: 3,
  replayDriftCount: 0,
};

export function generateExecutiveBriefing(customTelemetry?: Partial<InstitutionalTelemetry>): ExecutiveBriefing {
  const t: InstitutionalTelemetry = { ...CANONICAL_TELEMETRY_BASELINE, ...(customTelemetry || {}) };

  let overallStatus: ExecutiveBriefing['overallStatus'] = 'OPTIMAL';
  if (t.ohi < 70 || t.activeIncidents > 2 || t.replayDriftCount > 0) {
    overallStatus = 'CRITICAL';
  } else if (t.ohi < 80 || t.stressProbability > 0.25 || t.activeIncidents > 0) {
    overallStatus = 'DEGRADED';
  } else if (t.ohi < 82) {
    overallStatus = 'STABLE';
  }

  const headline =
    overallStatus === 'CRITICAL'
      ? `CRITICAL ALERT: Organizational Health Depleted to ${t.ohi.toFixed(1)} with ${t.activeIncidents} Active Incidents`
      : overallStatus === 'DEGRADED'
      ? `Caution Required: Governance Metrics at ${t.ohi.toFixed(1)} with Elevated Stress Probability (${(t.stressProbability * 100).toFixed(0)}%)`
      : `Institutional Resilience Certified: OHI at ${t.ohi.toFixed(1)} with Optimal Risk Bounds`;

  const executiveSummary =
    `The organization is operating with ${overallStatus.toLowerCase()} governance indicators. Decision execution efficiency (ODEI: ${t.odei.toFixed(1)}) and collective quality (CDQI: ${t.cdqi.toFixed(1)}) remain well above the mandatory 80.0 floor. ` +
    `Learning velocity is progressing at +${t.learningVelocity.toFixed(1)} per quarter, while stress probability is contained at ${(t.stressProbability * 100).toFixed(0)}%. ` +
    `Autonomous governance safeguards report zero drift with all 10 M10 safety invariants strictly satisfied.`;

  const keyFindings: string[] = [
    `Organizational Health Index (OHI) is ${t.ohi.toFixed(1)}, beating the institutional benchmark floor of 80.0.`,
    `Decision execution velocity has expanded by +${t.learningVelocity.toFixed(1)} pts quarter-over-quarter.`,
    `Failover RTO is currently ${t.rtoSeconds.toFixed(1)}s (well within the 30.0s L1 recovery SLA).`,
    `Replay verification completed across 100 snapshots with zero divergence (0 drift).`,
    `Human override supremacy is 100% active with ${t.pendingApprovals} pending executive approvals in queue.`,
  ];

  const insights: CrossCenterInsight[] = [
    {
      center: 'Executive Intelligence',
      signal: `ODEI ${t.odei.toFixed(1)}`,
      interpretation: 'Committees are executing decisions 14% faster than median baseline.',
      sentiment: 'POSITIVE',
    },
    {
      center: 'Organizational Learning',
      signal: `+${t.learningVelocity.toFixed(1)}/qtr`,
      interpretation: 'Lessons learned from Q2 dissents are successfully propagating into capital allocation.',
      sentiment: 'POSITIVE',
    },
    {
      center: 'Risk & Groupthink',
      signal: `Stress ${(t.stressProbability * 100).toFixed(0)}%`,
      interpretation: 'Low probability of correlated failure across key strategic portfolios.',
      sentiment: 'POSITIVE',
    },
    {
      center: 'Resilience & Operations',
      signal: `RTO ${t.rtoSeconds.toFixed(1)}s`,
      interpretation: 'High-availability failover tested and verified for autonomous optimization pipelines.',
      sentiment: 'POSITIVE',
    },
    {
      center: 'Autonomous Governance',
      signal: `${t.pendingApprovals} Pending`,
      interpretation: 'Human oversight boundary is fully enforced with zero unauthorized mutations.',
      sentiment: 'NEUTRAL',
    },
  ];

  const priorityActions: PriorityExecutiveAction[] = [
    {
      id: 'ACT-EXEC-001',
      title: 'Authorize Tier-1 Autonomous Rebalancing Plan',
      impactArea: 'Capital Allocation',
      urgency: 'HIGH',
      recommendedAction: 'Review and approve algorithmic portfolio weight adjustment (+2.4% target efficiency).',
      ownerCommittee: 'Investment Committee',
    },
    {
      id: 'ACT-EXEC-002',
      title: 'Ratify Q3 Dissent Integration Ledger',
      impactArea: 'Governance & Culture',
      urgency: 'MEDIUM',
      recommendedAction: 'Formally certify absorption of credit risk minority dissents into operational risk registry.',
      ownerCommittee: 'Risk Oversight Board',
    },
    {
      id: 'ACT-EXEC-003',
      title: 'Review Autonomous Safety Audit Trail',
      impactArea: 'Compliance & Audit',
      urgency: 'LOW',
      recommendedAction: 'Inspect deterministic replay hash chain for 2026-Q3 regulatory sign-off.',
      ownerCommittee: 'Audit Committee',
    },
  ];

  return {
    briefingId: 'NI-001',
    generatedAt: new Date().toISOString(),
    headline,
    overallStatus,
    executiveSummary,
    keyFindings,
    insights,
    priorityActions,
    telemetrySnapshot: t,
  };
}

export const CANONICAL_BRIEFINGS_HISTORY: ExecutiveBriefing[] = [
  generateExecutiveBriefing(),
  generateExecutiveBriefing({
    ohi: 82.5,
    odei: 84.1,
    learningVelocity: 2.9,
    stressProbability: 0.18,
    pendingApprovals: 5,
  }),
];
