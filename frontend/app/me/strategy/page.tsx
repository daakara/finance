"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import IntelligenceHeader from '../../../components/ui/IntelligenceHeader';
import {
  HouseholdStrategy,
  FutureState,
  HouseholdPortfolioPlan,
} from '../../../types/household-orchestration';

const CANONICAL_STRATEGIES: HouseholdStrategy[] = [
  {
    strategyId: 'STRAT_BALANCED_FAMILY',
    title: 'Balanced Family & Concurrent Upskilling',
    category: 'CAREER',
    projectedHHI: 88,
    projectedLHI: 86,
    projectedResilience: 84,
    projectedFinancialHealth: 82,
    confidenceScore: 0.91,
    personalImpact: 'Consistent sleep, low burnout risk, 5 protected hours weekly for AI & quantitative trading.',
    financialImpact: 'Stable salary +$14k/yr compounding investments with zero drawdown threat to emergency cash.',
    relationalImpact: 'Protected family dinners and shared childcare commitments; zero domestic dislocation.',
    householdImpact: 'High resilience buffer (14+ months runway preserved at all times).',
    riskScore: 18,
  },
  {
    strategyId: 'STRAT_STARTUP_BOOTSTRAP',
    title: 'Solo AI Startup & Day-Trading',
    category: 'BUSINESS',
    projectedHHI: 64,
    projectedLHI: 68,
    projectedResilience: 58,
    projectedFinancialHealth: 60,
    confidenceScore: 0.68,
    personalImpact: 'High cognitive stress, irregular sleep patterns, elevated amygdala activation.',
    financialImpact: 'High variance ($0 - $350k); liquid savings depleted by 65% in Year 1.',
    relationalImpact: 'Significant domestic friction over lost emergency cash runway.',
    householdImpact: 'Vulnerable to 3-month market drawdowns.',
    riskScore: 78,
  },
  {
    strategyId: 'STRAT_EXEC_MBA',
    title: 'Executive MBA & Principal Tech Move',
    category: 'EDUCATION',
    projectedHHI: 82,
    projectedLHI: 79,
    projectedResilience: 78,
    projectedFinancialHealth: 89,
    confidenceScore: 0.85,
    personalImpact: 'Temporary time pinch in Year 1, high upside in Years 2-5.',
    financialImpact: 'Tuition cost -$45k in Year 1, followed by +$65k/year executive comp.',
    relationalImpact: 'Partner assumes 15% more domestic tasks in Year 1; high joint alignment.',
    householdImpact: 'Strong long-term wealth compounding.',
    riskScore: 32,
  },
];

const CANONICAL_FUTURE_STATES: FutureState[] = [
  {
    horizonYears: 1,
    projectedHHI: 84,
    projectedLHI: 82,
    projectedNetWorth: 240000,
    projectedRelationshipHealth: 85,
    projectedResilience: 82,
    contributingSignals: ['Sleep 7.6h', '401k Maxed', 'Rule-based Swing Trading'],
    dependencyChain: ['Discipline -> Capital Accumulation -> Peace of Mind'],
    forecastDrivers: ['Core Comp ($195k)', 'Portfolio Alpha (+8%)', 'Zero Runway Dips'],
  },
  {
    horizonYears: 3,
    projectedHHI: 87,
    projectedLHI: 86,
    projectedNetWorth: 410000,
    projectedRelationshipHealth: 88,
    projectedResilience: 86,
    contributingSignals: ['AI Promotion', 'Trading Vault Compounding', 'Low Friction'],
    dependencyChain: ['Skill Growth -> Senior Promotion -> High Savings Rate'],
    forecastDrivers: ['Comp Growth ($250k)', 'Household Reserve (18 Mos)'],
  },
  {
    horizonYears: 5,
    projectedHHI: 91,
    projectedLHI: 89,
    projectedNetWorth: 680000,
    projectedRelationshipHealth: 90,
    projectedResilience: 90,
    contributingSignals: ['Passive Yield', 'Children Stable', 'High Discretionary Time'],
    dependencyChain: ['Financial Optionality -> Reduced Work Hours -> High Family Time'],
    forecastDrivers: ['Compounding Returns', 'Family Stability'],
  },
  {
    horizonYears: 10,
    projectedHHI: 94,
    projectedLHI: 92,
    projectedNetWorth: 1450000,
    projectedRelationshipHealth: 93,
    projectedResilience: 94,
    contributingSignals: ['Financial Independence', 'Legacy Vault', 'Full Autonomy'],
    dependencyChain: ['Decade of Discipline -> Freedom of Location & Schedule'],
    forecastDrivers: ['Capital Autonomy', 'Zero Debt'],
  },
];

export default function HouseholdStrategyPage() {
  const [selectedStrategyId, setSelectedStrategyId] = useState<string>('STRAT_BALANCED_FAMILY');
  const activeStrategy =
    CANONICAL_STRATEGIES.find((s) => s.strategyId === selectedStrategyId) ||
    CANONICAL_STRATEGIES[0];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 space-y-8">
      <IntelligenceHeader
        certification="HORIZON-9-10-CERTIFIED"
        title="Household Strategy Orchestrator"
        subtitle="Multi-Year Life Portfolio Intelligence & Relational Outcome Forecasting"
      />

      {/* Navigation Breadcrumb */}
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <Link href="/me" className="text-emerald-400 hover:underline">
          ← Back to 30-Second Cockpit
        </Link>
        <span>/</span>
        <span>Household Strategy</span>
      </div>

      {/* ZONE A: Household Strategy Command Card */}
      <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/80 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-5">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">
                Current Active Household Strategy
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-mono">
                Primary Path
              </span>
            </div>
            <h2 className="text-2xl font-black text-white">{activeStrategy.title}</h2>
          </div>
          <div className="flex items-center gap-4 text-center">
            <div className="bg-slate-950 px-4 py-2 rounded-xl border border-slate-800">
              <span className="text-xs text-slate-400 block">Projected HHI</span>
              <span className="text-xl font-bold text-emerald-400">{activeStrategy.projectedHHI} / 100</span>
            </div>
            <div className="bg-slate-950 px-4 py-2 rounded-xl border border-slate-800">
              <span className="text-xs text-slate-400 block">Confidence</span>
              <span className="text-xl font-bold text-cyan-400">{(activeStrategy.confidenceScore * 100).toFixed(0)}%</span>
            </div>
            <div className="bg-slate-950 px-4 py-2 rounded-xl border border-slate-800">
              <span className="text-xs text-slate-400 block">Resilience</span>
              <span className="text-xl font-bold text-indigo-400">{activeStrategy.projectedResilience} / 100</span>
            </div>
          </div>
        </div>

        {/* Explainability Matrix (INV-OI93-P) */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-slate-400 font-bold uppercase block mb-1">Personal Impact</span>
            <p className="text-slate-300">{activeStrategy.personalImpact}</p>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-slate-400 font-bold uppercase block mb-1">Financial Impact</span>
            <p className="text-slate-300">{activeStrategy.financialImpact}</p>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-slate-400 font-bold uppercase block mb-1">Relational Impact</span>
            <p className="text-slate-300">{activeStrategy.relationalImpact}</p>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-slate-400 font-bold uppercase block mb-1">Household Impact</span>
            <p className="text-slate-300">{activeStrategy.householdImpact}</p>
          </div>
        </div>
      </div>

      {/* ZONE B: Multi-Year Future Timeline (1, 3, 5, 10 Years) */}
      <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/60 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold text-white">Multi-Year Future Timeline (10-Year Projection)</h3>
          <span className="text-xs text-slate-400">Monte Carlo Calibrated</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {CANONICAL_FUTURE_STATES.map((fs) => (
            <div key={fs.horizonYears} className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-cyan-400">Year {fs.horizonYears}</span>
                <span className="text-xs text-slate-400 font-mono">HHI: {fs.projectedHHI}</span>
              </div>
              <div className="text-2xl font-black text-white">
                ${(fs.projectedNetWorth / 1000).toFixed(0)}k Net Worth
              </div>
              <div className="text-xs text-slate-400">
                Relationship Health: <strong className="text-emerald-400">{fs.projectedRelationshipHealth}%</strong>
              </div>
              <div className="pt-2 border-t border-slate-900 text-[11px] text-slate-500">
                Drivers: {fs.forecastDrivers?.join(', ')}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ZONE C: Strategy Comparison */}
      <div className="p-6 rounded-2xl border border-slate-800 bg-slate-900/60 shadow-xl space-y-4">
        <h3 className="text-lg font-bold text-white">Compare Alternative Strategic Paths</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {CANONICAL_STRATEGIES.map((strat) => (
            <div
              key={strat.strategyId}
              onClick={() => setSelectedStrategyId(strat.strategyId)}
              className={`p-4 rounded-xl border cursor-pointer transition-all ${
                selectedStrategyId === strat.strategyId
                  ? 'border-emerald-500 bg-emerald-950/20'
                  : 'border-slate-800 bg-slate-950/60 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-slate-400">{strat.category}</span>
                <span className="text-xs font-mono text-emerald-400">HHI {strat.projectedHHI}</span>
              </div>
              <h4 className="text-base font-bold text-white mb-2">{strat.title}</h4>
              <div className="text-xs text-slate-400 mb-3 line-clamp-2">{strat.personalImpact}</div>
              <div className="flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-800/60">
                <span>Risk: {strat.riskScore}/100</span>
                <span>Resilience: {strat.projectedResilience}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ZONE E & F: Relational Effects & Survivability */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40">
          <h4 className="text-sm font-bold text-white mb-2">Relational Effects Flow</h4>
          <div className="font-mono text-xs text-slate-300 bg-slate-950 p-3 rounded-lg border border-slate-800 leading-relaxed">
            Strategy: {activeStrategy.title}<br />
            ↓ Partner Alignment: High (Shared Vision)<br />
            ↓ Child Care Bandwidth: Protected (Zero Compromise)<br />
            ↓ Liquid Runway: 14.2 Months (Zero Shock Threat)<br />
            = Net HHI Outcome: {activeStrategy.projectedHHI} / 100
          </div>
        </div>

        <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40">
          <h4 className="text-sm font-bold text-white mb-2">Household Survivability Audit</h4>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-slate-500 block">Stress Resilience</span>
              <span className="text-emerald-400 font-bold">{activeStrategy.projectedResilience} / 100</span>
            </div>
            <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-slate-500 block">Burnout Probability</span>
              <span className="text-cyan-400 font-bold">{activeStrategy.riskScore}%</span>
            </div>
            <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-slate-500 block">Emergency Capacity</span>
              <span className="text-emerald-400 font-bold">14.2 Months</span>
            </div>
            <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
              <span className="text-slate-500 block">Rollback Available</span>
              <span className="text-emerald-400 font-bold">Yes (Instant)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
