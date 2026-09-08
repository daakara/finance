'use client';

/**
 * Phase 29: ODEI Dashboard
 *
 * Organizational Decision Effectiveness Index — Executive Command Center
 *
 * Sections:
 * 1. ODEI Hero Banner (score, delta, classification, confidence model)
 * 2. 4-Component Breakdown (DQ, OE, LE, OH)
 * 3. Team Leaderboard (5 teams)
 * 4. Organizational Cohort Distribution
 * 5. Strategic KPIs (OM-01 through OM-05)
 */

import React from 'react';
import {
  CANONICAL_ODEI_RESULT,
  CANONICAL_TEAM_BENCHMARKS,
  CANONICAL_ORGANIZATIONAL_COHORTS,
  CANONICAL_STRATEGIC_KPIS,
  classifyODEI,
} from '@/lib/telemetry/odeiEngine';
import type { ODEIClassification, StrategicOrgKPI, TeamBenchmark } from '@/types/organizational-intelligence';

function classificationLabel(c: ODEIClassification): string {
  const map: Record<ODEIClassification, string> = {
    ELITE: 'Elite Organization',
    HIGH_PERFORMING: 'High Performing',
    EFFECTIVE: 'Effective',
    DEVELOPING: 'Developing',
    AT_RISK: 'At Risk',
    CRITICAL: 'Critical',
  };
  return map[c];
}

function classificationColor(c: ODEIClassification): string {
  const map: Record<ODEIClassification, string> = {
    ELITE: 'text-emerald-400',
    HIGH_PERFORMING: 'text-green-400',
    EFFECTIVE: 'text-blue-400',
    DEVELOPING: 'text-amber-400',
    AT_RISK: 'text-orange-400',
    CRITICAL: 'text-rose-400',
  };
  return map[c];
}

function OdeiBanner() {
  const d = CANONICAL_ODEI_RESULT;
  const cls = classifyODEI(d.score);
  return (
    <div
      className="bg-gray-900 border border-gray-700 rounded-xl p-6 mb-6"
      data-testid="odei-banner"
      role="region"
      aria-label="Organizational Decision Effectiveness Index"
    >
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <p className="text-gray-400 text-sm uppercase tracking-widest mb-1">Organizational Decision Effectiveness Index</p>
          <div className="flex items-baseline gap-4">
            <span className="text-7xl font-black text-white" data-testid="odei-score">{d.score}</span>
            <div>
              <span className={`text-2xl font-bold ${classificationColor(cls)}`} data-testid="odei-classification">
                {classificationLabel(cls)}
              </span>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-green-400 text-sm font-semibold" data-testid="odei-delta">▲ +{d.delta} pts</span>
                <span className="text-gray-500 text-sm">vs prior 90 days</span>
              </div>
            </div>
          </div>
        </div>
        <div className="text-right text-sm text-gray-400 space-y-1">
          <p>Confidence <span className="text-white font-semibold" data-testid="odei-confidence">{d.confidence.confidencePct}%</span></p>
          <p>N = <span className="text-white font-semibold">{d.confidence.sampleSize.toLocaleString()}</span> decisions</p>
          <p>vs <span className="text-white font-semibold">{d.confidence.organizationsCompared}</span> organizations</p>
          <p>Observation window: <span className="text-white font-semibold">{d.confidence.observationWindowDays} days</span></p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
        {[
          { label: 'Decision Quality', value: d.components.decisionQuality, weight: '35%' },
          { label: 'Outcome Effectiveness', value: d.components.outcomeEffectiveness, weight: '30%' },
          { label: 'Learning Effectiveness', value: d.components.learningEffectiveness, weight: '20%' },
          { label: 'Org Health', value: d.components.organizationalHealth, weight: '15%' },
        ].map(comp => (
          <div key={comp.label} className="bg-gray-800 rounded-lg p-4" data-testid={`odei-component-${comp.label.toLowerCase().replace(/ /g, '-')}`}>
            <p className="text-gray-400 text-xs mb-1">{comp.label}</p>
            <p className="text-2xl font-bold text-white">{comp.value}</p>
            <p className="text-gray-500 text-xs mt-1">Weight {comp.weight}</p>
            <div className="mt-2 h-1.5 bg-gray-700 rounded-full">
              <div className="h-full bg-green-500 rounded-full" style={{ width: `${comp.value}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TeamLeaderboard() {
  const teams = [...CANONICAL_TEAM_BENCHMARKS].sort((a, b) => b.odei - a.odei);
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 mb-6" data-testid="team-leaderboard" role="region" aria-label="Team ODEI Leaderboard">
      <h3 className="text-white font-bold text-lg mb-4">Team Leaderboard</h3>
      <div className="space-y-3">
        {teams.map((team, i) => {
          const cls = classifyODEI(team.odei);
          return (
            <div key={team.teamId} className="flex items-center gap-3 bg-gray-800 rounded-lg px-4 py-3" data-testid={`team-${team.teamId}`}>
              <span className="text-gray-500 text-sm w-5">{i + 1}</span>
              <div className="flex-1">
                <p className="text-white font-semibold text-sm">{team.teamName}</p>
                <p className="text-gray-400 text-xs">{classificationLabel(cls)} · P{team.percentile}</p>
              </div>
              <div className="text-right">
                <p className={`text-xl font-bold ${classificationColor(cls)}`}>{team.odei}</p>
                <p className={`text-xs ${team.trend === 'UP' ? 'text-green-400' : team.trend === 'DOWN' ? 'text-rose-400' : 'text-gray-400'}`}>
                  {team.trend === 'UP' ? '▲' : team.trend === 'DOWN' ? '▼' : '—'}
                </p>
              </div>
              <div className="w-16 h-1.5 bg-gray-700 rounded-full">
                <div className="h-full bg-blue-500 rounded-full" style={{ width: `${team.odei}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CohortDistribution() {
  const cohorts = CANONICAL_ORGANIZATIONAL_COHORTS;
  const bars = [
    { label: 'Cohort D: Elite (90+)', pct: cohorts.elitePct, color: 'bg-emerald-500' },
    { label: 'Cohort C: High Performing (80–89)', pct: cohorts.highPerformingPct, color: 'bg-green-500' },
    { label: 'Cohort B: Developing (70–79)', pct: cohorts.developingPct, color: 'bg-amber-500' },
    { label: 'Cohort A: Emerging (<70)', pct: cohorts.emergingPct, color: 'bg-rose-500' },
  ];
  const total = bars.reduce((s, b) => s + b.pct, 0);

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 mb-6" data-testid="cohort-distribution" role="region" aria-label="Organizational Cohort Distribution">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-bold text-lg">Organizational Cohort Distribution</h3>
        <span className="text-gray-400 text-xs">Total = {total}%</span>
      </div>
      <div className="space-y-3">
        {bars.map(bar => (
          <div key={bar.label}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-300">{bar.label}</span>
              <span className="text-white font-semibold">{bar.pct}%</span>
            </div>
            <div className="h-3 bg-gray-700 rounded-full">
              <div className={`h-full ${bar.color} rounded-full`} style={{ width: `${bar.pct}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StrategicKpiGrid() {
  const kpis = CANONICAL_STRATEGIC_KPIS;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6" data-testid="strategic-kpis" role="region" aria-label="Strategic Organizational KPIs">
      <h3 className="text-white font-bold text-lg mb-4">Strategic KPIs</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {kpis.map(kpi => (
          <div key={kpi.id} className="bg-gray-800 rounded-lg p-4" data-testid={`kpi-${kpi.id.toLowerCase()}`}>
            <div className="flex items-start justify-between mb-2">
              <span className="text-gray-400 text-xs uppercase tracking-wide">{kpi.id}</span>
              <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${kpi.status === 'PASS' ? 'bg-green-900 text-green-400' : kpi.status === 'FAIL' ? 'bg-rose-900 text-rose-400' : 'bg-amber-900 text-amber-400'}`}>
                {kpi.status === 'PASS' ? '✅' : kpi.status === 'FAIL' ? '❌' : '⚠️'} {kpi.status}
              </span>
            </div>
            <p className="text-white font-semibold text-sm mb-1">{kpi.name}</p>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-black text-white">{kpi.current}{kpi.unit === '%' || kpi.unit === '% QoQ' || kpi.unit === '% variance' ? '%' : ''}</span>
              <span className="text-gray-500 text-xs">target {kpi.target}{kpi.unit === '%' || kpi.unit === '% QoQ' || kpi.unit === '% variance' ? '%' : ''}</span>
            </div>
            <p className="text-gray-400 text-xs mt-1">{kpi.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ODEIDashboard() {
  return (
    <div className="space-y-0" role="main" aria-label="ODEI Executive Dashboard">
      <OdeiBanner />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <TeamLeaderboard />
        <CohortDistribution />
      </div>
      <StrategicKpiGrid />
    </div>
  );
}

