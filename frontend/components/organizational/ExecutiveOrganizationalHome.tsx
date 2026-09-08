'use client';

/**
 * Phase 29: Executive Organizational Home — CEO Intelligence Dashboard
 *
 * Implements OI-501/502/503:
 * - CEO Intelligence Dashboard (ODEI, narrative, KPIs, top opportunity/risk)
 * - Executive Narrative Briefing (Story-First format)
 * - Organizational Readiness Index
 *
 * INV-OI9: Every recommendation exposes Evidence, Learning, Benchmark,
 * Expected Impact, and Confidence.
 */

import React, { useState } from 'react';
import {
  generateExecutiveBriefing,
  getOrganizationalReadinessIndex,
  getTopStrategicOpportunity,
  getTopOrganizationalRisk,
  CANONICAL_EXECUTIVE_IMPACT,
} from '@/lib/telemetry/executiveOrganizationalEngine';
import { CANONICAL_ORGANIZATIONAL_NARRATIVE } from '@/lib/telemetry/executiveOrganizationalEngine';
import { classifyODEI } from '@/lib/telemetry/odeiEngine';

function CeoNarrativeCard() {
  const briefing = generateExecutiveBriefing();
  const narrative = briefing.narrative;
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 mb-6" data-testid="ceo-narrative-card" role="region" aria-label="Executive Organizational Briefing">
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-gray-400 text-sm uppercase tracking-widest mb-1">CEO Intelligence Briefing</p>
          <p className="text-white text-xl font-bold">{briefing.greeting}</p>
        </div>
        <div className="text-right">
          <p className="text-5xl font-black text-green-400">{briefing.odei}</p>
          <p className="text-green-400 text-xs font-semibold">{briefing.classification.replace('_', ' ')}</p>
          <p className="text-gray-500 text-xs">{narrative.confidence}% confidence</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <p className="text-gray-400 text-xs">Capital Preserved</p>
          <p className="text-green-400 font-black text-2xl">{briefing.capitalPreserved}</p>
        </div>
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <p className="text-gray-400 text-xs">Excess Return</p>
          <p className="text-blue-400 font-black text-2xl">+{briefing.excessReturn}%</p>
        </div>
      </div>

      <div className="space-y-3">
        <div className="bg-gray-800 rounded-lg p-4">
          <p className="text-gray-400 text-xs uppercase mb-1">What Changed</p>
          <p className="text-white text-sm">{narrative.observation}</p>
        </div>
        <div className="bg-gray-800 rounded-lg p-4">
          <p className="text-gray-400 text-xs uppercase mb-1">What We Learned</p>
          <p className="text-white text-sm">{narrative.learning}</p>
        </div>
        <div className="bg-blue-950 border border-blue-700 rounded-lg p-4">
          <p className="text-blue-300 text-xs uppercase mb-1">Recommended Action</p>
          <p className="text-white text-sm">{narrative.recommendedAction}</p>
        </div>
      </div>

      {expanded && (
        <div className="mt-3 space-y-3" data-testid="narrative-detail">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-xs uppercase mb-1">Who Is Affected</p>
            <p className="text-white text-sm">{narrative.whoAffected}</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-xs uppercase mb-1">Expected Outcome</p>
            <p className="text-white text-sm">{narrative.expectedOutcome}</p>
          </div>
          <p className="text-gray-500 text-xs">Evidence ID: {narrative.evidenceId}</p>
        </div>
      )}

      <button
        onClick={() => setExpanded(!expanded)}
        className="mt-4 text-blue-400 text-sm font-semibold hover:text-blue-300 min-h-[44px] px-3 rounded"
        aria-expanded={expanded}
      >
        {expanded ? 'Show less ▲' : 'Show full evidence ▼'}
      </button>
    </div>
  );
}

function ExecutiveImpactMetrics() {
  const impact = CANONICAL_EXECUTIVE_IMPACT;
  const metrics = [
    { label: 'Decision Cycle Time', value: `-${impact.decisionCycleTimeReduction}%`, target: '-25%', pass: impact.decisionCycleTimeReduction >= 25, description: 'Observation → Decision → Execution' },
    { label: 'Repeat Mistake Prevention', value: `${impact.repeatMistakePreventionRate}%`, target: '>50%', pass: impact.repeatMistakePreventionRate >= 50, description: 'Reduction in recurring errors' },
    { label: 'Cross-Team Learning', value: `${impact.crossTeamLearningAdoption}%`, target: '>60%', pass: impact.crossTeamLearningAdoption >= 60, description: 'Learning adoption across teams' },
    { label: 'Institutional Alpha', value: `${impact.institutionalAlphaAttribution}%`, target: '40%+', pass: impact.institutionalAlphaAttribution >= 40, description: 'Excess performance traceable to ARX' },
  ];

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 mb-6" data-testid="executive-impact-metrics" role="region" aria-label="Executive Impact Metrics">
      <h3 className="text-white font-bold text-lg mb-4">Executive Impact Metrics</h3>
      <div className="grid grid-cols-2 gap-4">
        {metrics.map(m => (
          <div key={m.label} className="bg-gray-800 rounded-xl p-4" data-testid={`impact-metric-${m.label.toLowerCase().replace(/ /g, '-')}`}>
            <p className="text-gray-400 text-xs uppercase tracking-wide mb-1">{m.label}</p>
            <p className={`text-3xl font-black ${m.pass ? 'text-green-400' : 'text-rose-400'}`}>{m.value}</p>
            <p className="text-gray-500 text-xs">target {m.target} {m.pass ? '✅' : '❌'}</p>
            <p className="text-gray-400 text-xs mt-1">{m.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function OrganizationalReadiness() {
  const ori = getOrganizationalReadinessIndex();
  const dims = [
    { label: 'Decision Quality', value: ori.decisionQuality },
    { label: 'Learning Velocity', value: ori.learningVelocity },
    { label: 'Governance Compliance', value: ori.governanceCompliance },
    { label: 'Adoption Rate', value: ori.adoptionRate },
    { label: 'Knowledge Reuse', value: ori.knowledgeReuse },
  ];

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 mb-6" data-testid="organizational-readiness-index" role="region" aria-label="Organizational Readiness Index">
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-gray-400 text-xs uppercase tracking-widest mb-1">Organizational Readiness Index</p>
          <p className="text-white text-sm">Predicts future organizational performance trajectory.</p>
        </div>
        <div className="text-right">
          <p className="text-4xl font-black text-green-400" data-testid="ori-score">{ori.score}</p>
          <p className="text-green-400 text-xs">▲ Above 80 target</p>
          <p className="text-gray-500 text-xs">{ori.confidence}% confidence</p>
        </div>
      </div>
      <div className="space-y-2">
        {dims.map(d => (
          <div key={d.label}>
            <div className="flex justify-between text-xs mb-0.5">
              <span className="text-gray-300">{d.label}</span>
              <span className="text-white font-semibold">{d.value}{d.label.includes('Quality') || d.label.includes('Velocity') ? '' : '%'}</span>
            </div>
            <div className="h-1.5 bg-gray-700 rounded-full">
              <div className={`h-full rounded-full ${d.value >= 80 ? 'bg-green-500' : d.value >= 60 ? 'bg-amber-500' : 'bg-rose-500'}`}
                style={{ width: `${Math.min(d.value, 100)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StrategicOpportunityRisk() {
  const opportunity = getTopStrategicOpportunity();
  const risk = getTopOrganizationalRisk();

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
      <div className="bg-green-950 border border-green-700 rounded-xl p-5" data-testid="top-opportunity">
        <p className="text-green-400 text-xs uppercase font-bold mb-2">Top Strategic Opportunity</p>
        <p className="text-white font-bold mb-1">{opportunity.title}</p>
        <p className="text-green-300 text-sm">{opportunity.impact}</p>
        <p className="text-gray-400 text-xs mt-2">{opportunity.confidence}% confidence</p>
      </div>
      <div className="bg-amber-950 border border-amber-700 rounded-xl p-5" data-testid="top-risk">
        <p className="text-amber-400 text-xs uppercase font-bold mb-2">Top Organizational Risk</p>
        <p className="text-white font-bold mb-1">{risk.title}</p>
        <p className="text-amber-300 text-sm">{risk.explanation}</p>
        <span className={`inline-block mt-2 text-xs font-bold px-2 py-0.5 rounded-full ${risk.severity === 'HIGH' ? 'bg-rose-900 text-rose-300' : 'bg-amber-900 text-amber-300'}`}>
          {risk.severity}
        </span>
      </div>
    </div>
  );
}

export default function ExecutiveOrganizationalHome() {
  return (
    <div role="main" aria-label="CEO Intelligence Dashboard">
      <CeoNarrativeCard />
      <StrategicOpportunityRisk />
      <ExecutiveImpactMetrics />
      <OrganizationalReadiness />
    </div>
  );
}

