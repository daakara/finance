'use client';

import React from 'react';
import { DAY_30_EXECUTIVE_REVIEW_DATA } from '@/lib/telemetry/behavioralCohortEngine';

export default function Day30ProductionReviewDashboard() {
  const data = DAY_30_EXECUTIVE_REVIEW_DATA;

  return (
    <div className="space-y-6" data-testid="day30-production-review-dashboard">
      {/* 1. Executive Summary & Production Status */}
      <div className="p-6 bg-gradient-to-r from-bg-surface-raised via-bg-surface to-bg-surface-raised border-2 border-accent-positive/40 rounded-2xl shadow-xl space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 text-caption-mono font-bold tracking-wider uppercase bg-accent-positive/15 text-accent-positive border border-accent-positive/40 rounded-md">
                Institutional Production Review
              </span>
              <span className="text-text-muted text-caption-mono">
                {data.reviewPeriod}
              </span>
              <span className="text-text-muted text-caption-mono">
                Build: {data.evaluatedBuild}
              </span>
            </div>
            <h3 className="text-display-2 font-bold text-text-primary">
              ARX Terminal vNext: 30-Day Executive Production Review
            </h3>
            <p className="text-body-ui text-text-secondary max-w-3xl">
              ARX successfully completed its first production operating cycle. The platform demonstrated
              strong reliability across Decision, Prediction, Outcome, Learning, and Governance Intelligence.
            </p>
          </div>

          <div className="p-4 bg-bg-surface/80 backdrop-blur border border-accent-positive/30 rounded-xl text-center min-w-[200px]">
            <div className="text-display-1 font-mono font-extrabold text-accent-positive">
              {data.productionExcellenceScore} <span className="text-body-ui font-normal text-text-muted">/ 100</span>
            </div>
            <div className="text-caption-mono font-bold uppercase text-accent-positive mt-1">
              {data.productionStatus}
            </div>
            <div className="text-[11px] font-mono text-text-muted mt-0.5">
              Status: CERTIFIED
            </div>
          </div>
        </div>

        {/* Stakeholder Approval Sign-off Grid */}
        <div className="pt-4 border-t border-border-subtle grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-2.5 bg-bg-surface rounded border border-border-subtle">
            <div className="text-text-muted text-[11px]">CIO &amp; Committee Chair</div>
            <div className="text-text-primary font-semibold truncate">{data.certifiedBy.cio}</div>
            <div className="text-accent-positive text-[10px] mt-0.5">✓ Formal Approval</div>
          </div>
          <div className="p-2.5 bg-bg-surface rounded border border-border-subtle">
            <div className="text-text-muted text-[11px]">Product Steering Committee</div>
            <div className="text-text-primary font-semibold truncate">{data.certifiedBy.productSteeringCommittee}</div>
            <div className="text-accent-positive text-[10px] mt-0.5">✓ Sign-off Granted</div>
          </div>
          <div className="p-2.5 bg-bg-surface rounded border border-border-subtle">
            <div className="text-text-muted text-[11px]">Governance Board</div>
            <div className="text-text-primary font-semibold truncate">{data.certifiedBy.governanceBoard}</div>
            <div className="text-accent-positive text-[10px] mt-0.5">✓ Audit Compliant</div>
          </div>
          <div className="p-2.5 bg-bg-surface rounded border border-border-subtle">
            <div className="text-text-muted text-[11px]">Chief Systems Architect</div>
            <div className="text-text-primary font-semibold truncate">{data.certifiedBy.chiefSystemsArchitect}</div>
            <div className="text-accent-positive text-[10px] mt-0.5">✓ Architecture Certified</div>
          </div>
        </div>
      </div>

      {/* 2. Adoption Metrics & 3. Behavioral Metrics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Adoption Metrics */}
        <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
          <h4 className="text-header-1 text-text-primary flex items-center justify-between">
            <span>2. User Adoption Metrics</span>
            <span className="text-caption-mono text-accent-positive font-bold">ALL PASS</span>
          </h4>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
              <div className="text-caption-mono text-text-muted text-xs uppercase">Total Active Users</div>
              <div className="text-header-1 font-mono font-bold text-text-primary mt-1">{data.totalActiveUsers.toLocaleString()}</div>
              <div className="text-caption text-text-secondary">Across 42 institutional desks</div>
            </div>
            <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
              <div className="text-caption-mono text-text-muted text-xs uppercase">Daily Active Users</div>
              <div className="text-header-1 font-mono font-bold text-text-primary mt-1">{data.dailyActiveUsers.toLocaleString()}</div>
              <div className="text-caption text-accent-positive">48.4% DAU / MAU ratio</div>
            </div>
            <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
              <div className="text-caption-mono text-text-muted text-xs uppercase">Weekly Retention</div>
              <div className="text-header-1 font-mono font-bold text-accent-positive mt-1">{data.weeklyRetentionPct}%</div>
              <div className="text-caption text-text-secondary">High institutional stickiness</div>
            </div>
            <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
              <div className="text-caption-mono text-text-muted text-xs uppercase">AI Coach Adoption</div>
              <div className="text-header-1 font-mono font-bold text-accent-positive mt-1">{data.aiCoachAdoptionPct}%</div>
              <div className="text-caption text-text-secondary">Weekly active engagement</div>
            </div>
          </div>
        </div>

        {/* Behavioral Metrics */}
        <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
          <h4 className="text-header-1 text-text-primary flex items-center justify-between">
            <span>3. Decision Quality &amp; Outcomes</span>
            <span className="text-caption-mono text-accent-positive font-bold">TARGET MET</span>
          </h4>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
              <div className="text-caption-mono text-text-muted text-xs uppercase">Average Decision Score</div>
              <div className="text-header-1 font-mono font-bold text-accent-positive mt-1">74 / 100</div>
              <div className="text-caption text-accent-positive font-medium">+6 pts vs baseline 68</div>
            </div>
            <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
              <div className="text-caption-mono text-text-muted text-xs uppercase">Prediction Actionability (PAR)</div>
              <div className="text-header-1 font-mono font-bold text-accent-positive mt-1">{data.predictionActionabilityRate}%</div>
              <div className="text-caption text-text-muted">Target &ge; 50.0% (PASS)</div>
            </div>
            <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
              <div className="text-caption-mono text-text-muted text-xs uppercase">Outcome Resolution</div>
              <div className="text-header-1 font-mono font-bold text-accent-positive mt-1">{data.outcomeResolutionCoverage}%</div>
              <div className="text-caption text-text-muted">Zero unverified outcomes</div>
            </div>
            <div className="p-3 bg-bg-surface-raised rounded-lg border border-border-subtle">
              <div className="text-caption-mono text-text-muted text-xs uppercase">Learning Velocity</div>
              <div className="text-header-1 font-mono font-bold text-accent-positive mt-1">+{data.learningVelocityPct}%</div>
              <div className="text-caption text-text-muted">Quarter-over-quarter gain</div>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Operational Metrics Scorecard */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
        <h4 className="text-header-1 text-text-primary">
          4. Operational Reliability Scorecard
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-body-ui font-mono text-xs">
            <thead>
              <tr className="border-b border-border-subtle text-text-muted">
                <th className="pb-2">Metric Dimension</th>
                <th className="pb-2">Target Standard</th>
                <th className="pb-2">Actual Production Value</th>
                <th className="pb-2">Audit Verdict</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle">
              <tr>
                <td className="py-2.5 font-sans font-semibold text-text-primary">Platform Availability (SLO)</td>
                <td className="py-2.5 text-text-secondary">&ge; 99.90%</td>
                <td className="py-2.5 text-accent-positive font-bold">99.96%</td>
                <td className="py-2.5"><span className="text-accent-positive font-bold">✓ PASS</span></td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-text-primary">Critical Production Defects</td>
                <td className="py-2.5 text-text-secondary">0 Allowed</td>
                <td className="py-2.5 text-accent-positive font-bold">0</td>
                <td className="py-2.5"><span className="text-accent-positive font-bold">✓ PASS</span></td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-text-primary">Major Defects</td>
                <td className="py-2.5 text-text-secondary">&le; 2 Allowed</td>
                <td className="py-2.5 text-accent-positive font-bold">1 (Resolved)</td>
                <td className="py-2.5"><span className="text-accent-positive font-bold">✓ PASS</span></td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-text-primary">Telemetry Coverage (TQ-1)</td>
                <td className="py-2.5 text-text-secondary">&ge; 99.50%</td>
                <td className="py-2.5 text-accent-positive font-bold">99.70%</td>
                <td className="py-2.5"><span className="text-accent-positive font-bold">✓ PASS</span></td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-text-primary">Attribution Completeness (TQ-2)</td>
                <td className="py-2.5 text-text-secondary">100.0%</td>
                <td className="py-2.5 text-accent-positive font-bold">100.0%</td>
                <td className="py-2.5"><span className="text-accent-positive font-bold">✓ PASS</span></td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-semibold text-text-primary">Audit Trail Integrity (TQ-4)</td>
                <td className="py-2.5 text-text-secondary">100.0%</td>
                <td className="py-2.5 text-accent-positive font-bold">100.0%</td>
                <td className="py-2.5"><span className="text-accent-positive font-bold">✓ PASS</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 5. Governance Review & Strategic Findings */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-3">
          <h4 className="text-header-2 text-text-primary">
            5. Committee Governance Review
          </h4>
          <div className="space-y-2 text-body-ui">
            <div className="flex justify-between py-1 border-b border-border-subtle">
              <span className="text-text-secondary">Committee Decisions Recorded:</span>
              <span className="font-mono font-bold text-text-primary">{data.committeeDecisionsCount}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-border-subtle">
              <span className="text-text-secondary">Cryptographic Audit Chain:</span>
              <span className="font-mono font-bold text-accent-positive">100% SEC Rule 17a-4 Compliant</span>
            </div>
            <div className="flex justify-between py-1 border-b border-border-subtle">
              <span className="text-text-secondary">Decision Traceability:</span>
              <span className="font-mono text-caption text-text-muted">
                Decision &rarr; Approval &rarr; Prediction &rarr; Outcome &rarr; Attribution
              </span>
            </div>
          </div>
        </div>

        <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-3">
          <h4 className="text-header-2 text-text-primary">
            6. Strategic Findings &amp; Next Phase
          </h4>
          <div className="space-y-2 text-caption">
            <div className="flex items-start gap-2">
              <span className="text-accent-positive font-bold">✓</span>
              <span className="text-text-secondary">Outcome Intelligence adoption exceeded initial expectations (68% weekly).</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-accent-positive font-bold">✓</span>
              <span className="text-text-secondary">AI Learning Coach established as the primary daily analytical surface.</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-accent-warning font-bold">&bull;</span>
              <span className="text-text-secondary">Opportunity: Expand personalized learning pathways and deep behavioral coaching.</span>
            </div>
            <div className="pt-2 border-t border-border-subtle text-[11px] font-mono text-accent-info">
              Recommendation: Proceed to Sprint 8 Institutional Reliability &amp; Adaptive Coaching.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
