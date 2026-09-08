'use client';

/**
 * Phase 29: Team Benchmark Dashboard
 *
 * Implements OI-301/302/303:
 * - Team comparison across DQ, LV, Rule Adherence, Drift
 * - Role cohort analysis (Analyst / PM / Leadership)
 * - Groupthink detection (INV-OI5)
 * - Top performer pattern analysis
 */

import React, { useState } from 'react';
import {
  CANONICAL_TOP_PERFORMER_PATTERNS,
  CANONICAL_GROUPTHINK_RESULT,
  CANONICAL_ISOLATION_TEST_RESULT,
  CANONICAL_CONCENTRATION_VIOLATION,
} from '@/lib/telemetry/organizationalBenchmarkEngine';
import { CANONICAL_TEAM_BENCHMARKS, CANONICAL_ROLE_COHORTS, classifyODEI } from '@/lib/telemetry/odeiEngine';

type ActiveTab = 'teams' | 'roles' | 'groupthink' | 'top-performers';

function GroupthinkAlert() {
  const result = CANONICAL_GROUPTHINK_RESULT;
  const isolation = CANONICAL_ISOLATION_TEST_RESULT;
  const fairness = CANONICAL_CONCENTRATION_VIOLATION;

  return (
    <div className="space-y-4" data-testid="groupthink-panel" role="tabpanel">
      <div className={`rounded-xl p-5 border ${result.groupthinkRisk ? 'bg-rose-950 border-rose-700' : 'bg-green-950 border-green-700'}`} data-testid="groupthink-result">
        <div className="flex items-start justify-between mb-3">
          <div>
            <p className="text-white font-bold">INV-OI5: Groupthink Detection</p>
            <p className="text-gray-300 text-sm mt-0.5">Synthetic committee scenario · {result.committeeId}</p>
          </div>
          <span className={`text-sm font-bold px-3 py-1 rounded-full ${result.groupthinkRisk ? 'bg-rose-700 text-rose-100' : 'bg-green-700 text-green-100'}`}>
            {result.groupthinkRisk ? '⚠️ RISK DETECTED' : '✅ HEALTHY'}
          </span>
        </div>
        <div className="grid grid-cols-3 gap-3 mb-3">
          <div className="bg-black/20 rounded-lg p-3 text-center">
            <p className="text-gray-400 text-xs">Approvals</p>
            <p className="text-white font-bold text-xl">{result.approvalCount}</p>
          </div>
          <div className="bg-black/20 rounded-lg p-3 text-center">
            <p className="text-gray-400 text-xs">Dissent</p>
            <p className={`font-bold text-xl ${result.dissentCount === 0 ? 'text-rose-400' : 'text-green-400'}`}>{result.dissentCount}</p>
          </div>
          <div className="bg-black/20 rounded-lg p-3 text-center">
            <p className="text-gray-400 text-xs">Diversity Index</p>
            <p className={`font-bold text-xl ${result.diversityIndex < 0.3 ? 'text-rose-400' : 'text-green-400'}`}>{(result.diversityIndex * 100).toFixed(0)}%</p>
          </div>
        </div>
        <p className="text-gray-300 text-sm">{result.explanation}</p>
      </div>

      <div className="bg-gray-900 border border-gray-700 rounded-xl p-5" data-testid="benchmark-isolation">
        <p className="text-white font-bold mb-2">INV-OI6: Benchmark Isolation</p>
        <div className="flex items-center gap-3">
          <span className={`text-sm font-bold px-3 py-1 rounded-full ${isolation.isIsolated ? 'bg-green-900 text-green-300' : 'bg-rose-900 text-rose-300'}`}>
            {isolation.isIsolated ? '✅ ISOLATED' : '❌ CONTAMINATION'}
          </span>
          <p className="text-gray-400 text-sm">{isolation.violations.length} violations · {isolation.benchmarkPopulation} organizations compared</p>
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-700 rounded-xl p-5" data-testid="fairness-result">
        <p className="text-white font-bold mb-2">INV-OI8: Organizational Fairness</p>
        <div className="flex items-center gap-3 mb-2">
          <span className={`text-sm font-bold px-3 py-1 rounded-full ${fairness.isConcentrated ? 'bg-rose-900 text-rose-300' : 'bg-green-900 text-green-300'}`}>
            {fairness.isConcentrated ? '⚠️ CONCENTRATION RISK' : '✅ FAIR'}
          </span>
          <p className="text-gray-400 text-sm">Max actor: {fairness.maxInfluencePct}% · Threshold: {fairness.threshold}%</p>
        </div>
        <p className="text-gray-300 text-sm">{fairness.explanation}</p>
      </div>
    </div>
  );
}

export default function TeamBenchmarkDashboard() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('teams');

  const tabs: Array<{ id: ActiveTab; label: string }> = [
    { id: 'teams', label: 'Team Benchmarks' },
    { id: 'roles', label: 'Role Cohorts' },
    { id: 'groupthink', label: 'Governance Risks' },
    { id: 'top-performers', label: 'Top Performers' },
  ];

  return (
    <div role="main" aria-label="Organizational Benchmark Dashboard">
      <div className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden">
        <div className="flex flex-wrap border-b border-gray-700" role="tablist">
          {tabs.map(tab => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-3 text-sm font-semibold min-h-[44px] transition-colors ${activeTab === tab.id ? 'bg-gray-800 text-white border-b-2 border-blue-500' : 'text-gray-400 hover:text-gray-200'}`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="p-6">
          {activeTab === 'teams' && (
            <div data-testid="team-benchmarks-table" role="tabpanel">
              <div className="overflow-x-auto">
                <table className="w-full text-sm" role="table" aria-label="Team Benchmark Comparison">
                  <thead>
                    <tr className="text-gray-400 text-xs border-b border-gray-700">
                      <th className="text-left pb-3">Team</th>
                      <th className="text-center pb-3">ODEI</th>
                      <th className="text-center pb-3">DQ</th>
                      <th className="text-center pb-3">Learn. Vel.</th>
                      <th className="text-center pb-3">Rule Adh.</th>
                      <th className="text-center pb-3">Drift</th>
                      <th className="text-center pb-3">Pctile</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {CANONICAL_TEAM_BENCHMARKS.map(team => {
                      const cls = classifyODEI(team.odei);
                      const clsColors: Record<string, string> = {
                        ELITE: 'text-emerald-400',
                        HIGH_PERFORMING: 'text-green-400',
                        EFFECTIVE: 'text-blue-400',
                        DEVELOPING: 'text-amber-400',
                        AT_RISK: 'text-orange-400',
                        CRITICAL: 'text-rose-400',
                      };
                      return (
                        <tr key={team.teamId} data-testid={`team-row-${team.teamId}`}>
                          <td className="py-3 text-white font-semibold">{team.teamName}</td>
                          <td className={`py-3 text-center font-black text-lg ${clsColors[cls]}`}>{team.odei}</td>
                          <td className="py-3 text-center text-gray-200">{team.decisionQuality}</td>
                          <td className="py-3 text-center text-gray-200">{team.learningVelocity}</td>
                          <td className="py-3 text-center text-gray-200">{team.ruleAdherence}%</td>
                          <td className={`py-3 text-center font-semibold ${team.drift < 15 ? 'text-green-400' : team.drift < 20 ? 'text-amber-400' : 'text-rose-400'}`}>{team.drift}%</td>
                          <td className="py-3 text-center text-gray-400">P{team.percentile}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'roles' && (
            <div className="space-y-4" data-testid="role-cohorts" role="tabpanel">
              {CANONICAL_ROLE_COHORTS.map(cohort => (
                <div key={cohort.role} className="bg-gray-800 rounded-xl p-5" data-testid={`role-${cohort.role}`}>
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-white font-bold text-lg">{cohort.role.replace('_', ' ')}</p>
                    <p className="text-gray-400 text-sm">N = {cohort.sampleSize.toLocaleString()}</p>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="text-center"><p className="text-gray-400 text-xs">Avg DQ</p><p className="text-white font-bold text-xl">{cohort.avgDecisionQuality}</p></div>
                    <div className="text-center"><p className="text-gray-400 text-xs">Avg LV</p><p className="text-white font-bold text-xl">{cohort.avgLearningVelocity}</p></div>
                    <div className="text-center"><p className="text-gray-400 text-xs">Rule Adh.</p><p className="text-white font-bold text-xl">{cohort.avgRuleAdherence}%</p></div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'groupthink' && <GroupthinkAlert />}

          {activeTab === 'top-performers' && (
            <div className="space-y-4" data-testid="top-performer-patterns" role="tabpanel">
              {CANONICAL_TOP_PERFORMER_PATTERNS.map(pattern => (
                <div key={pattern.rank} className="bg-gray-800 rounded-xl p-5" data-testid={`top-performer-${pattern.rank}`}>
                  <div className="flex items-start gap-4">
                    <div className="text-4xl font-black text-gray-600">#{pattern.rank}</div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-1">
                        <p className="text-white font-bold">{pattern.teamName}</p>
                        <span className="text-2xl font-black text-green-400">{pattern.odei}</span>
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${pattern.replicability === 'HIGH' ? 'bg-green-900 text-green-300' : 'bg-amber-900 text-amber-300'}`}>
                          {pattern.replicability} replicability
                        </span>
                      </div>
                      <p className="text-gray-300 text-sm mb-1"><span className="text-gray-400">Key behavior:</span> {pattern.differentiatingBehavior}</p>
                      <p className="text-green-400 text-sm font-semibold">{pattern.behaviorImpact}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

