'use client';

/**
 * Horizon 5: /me/allocator — Weekly Time, Energy, and Money Allocator
 *
 * Implements:
 * - INV-OI75-P (Personal Capacity Feasibility Invariant)
 * - 168-Hour Weekly Time Budget Matrix
 * - Live Multidimensional Outcome Simulation (Current, 6m, 12m)
 * - Constraint Warnings & Automated Resource Rebalancing
 */

import React, { useState, useMemo, Suspense } from 'react';
import Link from 'next/link';
import IntelligenceHeader from '../../../components/ui/IntelligenceHeader';
import HorizonCard from '../../../components/ui/HorizonCard';
import HorizonMetricCard from '../../../components/ui/HorizonMetricCard';
import SeverityBadge from '../../../components/ui/SeverityBadge';
import {
  DEFAULT_PERSONAL_CAPACITY,
  DEFAULT_WEEKLY_TIME_BUDGET,
  DEFAULT_ALLOCATION_DOMAINS,
  verifyPersonalCapacity,
  calculateProjectedOutcomes,
} from '../../../lib/simulation/personalCapacityEngine';

function AllocatorContent() {
  const capacity = DEFAULT_PERSONAL_CAPACITY;
  const timeBudget = DEFAULT_WEEKLY_TIME_BUDGET;

  // Domain hours state
  const [allocations, setAllocations] = useState<Record<string, number>>({
    career: 7,
    learning: 4,
    health: 3,
    relationships: 2,
    finance: 1,
  });

  const [monthlySpend, setMonthlySpend] = useState<number>(300);

  // Calculate total allocated hours
  const totalAllocatedHours = useMemo(() => {
    return Object.values(allocations).reduce((sum, h) => sum + (h || 0), 0);
  }, [allocations]);

  // Derived demands
  const demand = useMemo(() => {
    const energyDemand = Math.round(totalAllocatedHours * 3.8);
    const attentionDemand = Math.round(
      (allocations.career || 0) * 5 + (allocations.learning || 0) * 6 + (allocations.finance || 0) * 3
    );

    return {
      weeklyHours: totalAllocatedHours,
      monthlyBudget: monthlySpend,
      energyDemand: Math.min(100, energyDemand),
      attentionDemand: Math.min(100, attentionDemand),
    };
  }, [totalAllocatedHours, allocations, monthlySpend]);

  // Capacity verification against INV-OI75-P
  const capacityCheck = useMemo(() => {
    return verifyPersonalCapacity(capacity, demand, timeBudget.sleepHours);
  }, [capacity, demand, timeBudget.sleepHours]);

  // Dynamic outcomes
  const projectedOutcomes = useMemo(() => {
    return calculateProjectedOutcomes(allocations, 82.4);
  }, [allocations]);

  const handleSliderChange = (domainId: string, hours: number) => {
    setAllocations((prev) => ({
      ...prev,
      [domainId]: hours,
    }));
  };

  const handleRebalance = () => {
    // Intelligent auto-rebalancing: scale down to fit available weekly hours
    const targetHours = capacity.weeklyHours - 3; // safe buffer
    const factor = targetHours / Math.max(1, totalAllocatedHours);
    setAllocations({
      career: Math.max(1, Math.round((allocations.career || 7) * factor)),
      learning: Math.max(1, Math.round((allocations.learning || 4) * factor)),
      health: Math.max(1, Math.round((allocations.health || 3) * factor)),
      relationships: Math.max(1, Math.round((allocations.relationships || 2) * factor)),
      finance: Math.max(1, Math.round((allocations.finance || 1) * factor)),
    });
    setMonthlySpend(Math.min(capacity.monthlyBudget, monthlySpend));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 space-y-6 max-w-7xl mx-auto">
      {/* Navigation breadcrumb */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <Link
            href="/me"
            className="text-xs uppercase tracking-wider font-semibold text-slate-400 hover:text-white transition-colors"
          >
            ← Back to Life Command Center
          </Link>
          <span className="text-slate-600">/</span>
          <span className="text-xs uppercase tracking-wider font-semibold text-emerald-400">
            Resource Allocator
          </span>
        </div>
        <div className="flex items-center gap-2">
          <SeverityBadge
            level={capacityCheck.isFeasible ? 'LOW' : 'HIGH'}
            status={capacityCheck.isFeasible ? 'HEALTHY' : 'CRITICAL'}
          />
        </div>
      </div>

      <IntelligenceHeader
        title="Personal Resource & Capacity Allocator"
        subtitle="Where should I spend my limited resources? Balance weekly hours, budget, energy, and attention with real-time INV-OI75-P feasibility bounds."
        certification="HORIZON-5-CERTIFIED"
        status={capacityCheck.isFeasible ? 'CERTIFIED' : 'FAILED'}
        breadcrumbs={[
          { label: 'Life OS', href: '/me' },
          { label: 'Resource Allocator' },
        ]}
      />

      {/* Top Life Health Summary & Capacity Gauges */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <HorizonMetricCard
          label="TIME CAPACITY"
          value={`${totalAllocatedHours} / ${capacity.weeklyHours}h`}
          delta={`${Math.max(0, capacity.weeklyHours - totalAllocatedHours)}h slack`}
          deltaPositive={totalAllocatedHours <= capacity.weeklyHours}
          severity={totalAllocatedHours <= capacity.weeklyHours ? 'PASS' : 'CRITICAL'}
        />
        <HorizonMetricCard
          label="GROWTH BUDGET"
          value={`€${monthlySpend} / €${capacity.monthlyBudget}`}
          delta={`€${Math.max(0, capacity.monthlyBudget - monthlySpend)} remaining`}
          deltaPositive={monthlySpend <= capacity.monthlyBudget}
          severity={monthlySpend <= capacity.monthlyBudget ? 'PASS' : 'CRITICAL'}
        />
        <HorizonMetricCard
          label="ENERGY BATTERY"
          value={`${demand.energyDemand} / ${capacity.energyCapacity}`}
          delta="Autonomic reserve"
          deltaPositive={demand.energyDemand <= capacity.energyCapacity}
          severity={demand.energyDemand <= capacity.energyCapacity ? 'PASS' : 'WARN'}
        />
        <HorizonMetricCard
          label="ATTENTION UNITS"
          value={`${demand.attentionDemand} / ${capacity.attentionCapacity}`}
          delta="Cognitive load"
          deltaPositive={demand.attentionDemand <= capacity.attentionCapacity}
          severity={demand.attentionDemand <= capacity.attentionCapacity ? 'PASS' : 'CRITICAL'}
        />
      </div>

      {/* Constraint Warning Banner if Infeasible */}
      {!capacityCheck.isFeasible && (
        <div className="p-4 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="font-semibold text-rose-200 flex items-center gap-2">
              <span>⚠</span> INV-OI75-P Constraint Violation Detected
            </div>
            <div className="text-xs text-rose-300/80">
              Violations:{' '}
              <span className="font-mono">{capacityCheck.violations.join(', ')}</span>. The
              proposed plan demands more resources than physically or cognitively available.
            </div>
          </div>
          <button
            onClick={handleRebalance}
            className="px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white rounded text-xs font-semibold uppercase tracking-wider transition-colors shadow-sm whitespace-nowrap"
          >
            Auto-Rebalance to Capacity
          </button>
        </div>
      )}

      {/* Main Grid: Allocation Matrix + Simulation Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Interactive Allocation Matrix */}
        <div className="lg:col-span-2 space-y-6">
          <HorizonCard
            title="Weekly Allocation Matrix (Hours)"
            subtitle="Adjust weekly commitment sliders to observe immediate trajectory and LHI recalculation"
          >
            <div className="space-y-6">
              {DEFAULT_ALLOCATION_DOMAINS.map((domain) => {
                const currentHours = allocations[domain.id] ?? domain.hours;
                return (
                  <div key={domain.id} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: domain.color }}
                        />
                        <span className="font-semibold text-slate-200">{domain.name}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-emerald-400 font-mono">
                          +{Math.round(currentHours * 2.2)} Impact
                        </span>
                        <span className="font-mono text-sm font-bold text-white bg-slate-800 px-2 py-0.5 rounded">
                          {currentHours}h / wk
                        </span>
                      </div>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="15"
                      step="1"
                      value={currentHours}
                      onChange={(e) => handleSliderChange(domain.id, parseInt(e.target.value, 10))}
                      className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                    />
                    <div className="text-xs text-slate-400 flex justify-between">
                      <span>{domain.description}</span>
                      <span>Max: 15h</span>
                    </div>
                  </div>
                );
              })}

              {/* Monthly Budget Slider */}
              <div className="pt-4 border-t border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-semibold text-slate-200">
                    Monthly Discretionary Growth Budget
                  </span>
                  <span className="font-mono text-sm font-bold text-white bg-slate-800 px-2 py-0.5 rounded">
                    €{monthlySpend} / mo
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="800"
                  step="25"
                  value={monthlySpend}
                  onChange={(e) => setMonthlySpend(parseInt(e.target.value, 10))}
                  className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <div className="text-xs text-slate-400 flex justify-between">
                  <span>Coaching, cohorts, tool subscriptions, fitness recovery</span>
                  <span>Cap: €{capacity.monthlyBudget}</span>
                </div>
              </div>
            </div>
          </HorizonCard>

          {/* 168-Hour Weekly Time Architecture */}
          <HorizonCard
            title="168-Hour Hard Physiological Horizon"
            subtitle="Immutable weekly balance confirming sleep and recovery floors (INV-OI75-P)"
          >
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
              <div className="p-3 bg-slate-900/60 rounded border border-slate-800">
                <div className="text-xs text-slate-400">SLEEP FLOOR</div>
                <div className="text-lg font-mono font-bold text-emerald-400">
                  {timeBudget.sleepHours}h
                </div>
                <div className="text-[10px] text-slate-500">8.0h / night (Safe)</div>
              </div>
              <div className="p-3 bg-slate-900/60 rounded border border-slate-800">
                <div className="text-xs text-slate-400">WORK & CORE</div>
                <div className="text-lg font-mono font-bold text-blue-400">
                  {timeBudget.workHours}h
                </div>
                <div className="text-[10px] text-slate-500">Employment baseline</div>
              </div>
              <div className="p-3 bg-slate-900/60 rounded border border-slate-800">
                <div className="text-xs text-slate-400">FAMILY & CARE</div>
                <div className="text-lg font-mono font-bold text-pink-400">
                  {timeBudget.familyHours}h
                </div>
                <div className="text-[10px] text-slate-500">Non-negotiable</div>
              </div>
              <div className="p-3 bg-slate-900/60 rounded border border-slate-800">
                <div className="text-xs text-slate-400">REMAINING SLACK</div>
                <div className="text-lg font-mono font-bold text-amber-400">
                  {Math.max(0, capacity.weeklyHours - totalAllocatedHours)}h
                </div>
                <div className="text-[10px] text-slate-500">Buffer protection</div>
              </div>
            </div>
          </HorizonCard>
        </div>

        {/* Right Col: Simulation Panel */}
        <div className="space-y-6">
          <HorizonCard
            title="Live Trajectory Simulation"
            subtitle="Calculated from real-time slider allocation dynamics"
          >
            <div className="space-y-5">
              <div className="p-4 bg-slate-900/80 rounded-lg border border-slate-800 space-y-3">
                <div className="flex justify-between items-center text-xs uppercase tracking-wider text-slate-400">
                  <span>Current Life Health</span>
                  <span className="font-mono text-emerald-400 font-bold">
                    {projectedOutcomes.projectedLhiCurrent}
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs uppercase tracking-wider text-slate-400">
                  <span>6 Months Projected</span>
                  <span className="font-mono text-blue-400 font-bold">
                    {projectedOutcomes.projectedLhi6m}
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs uppercase tracking-wider text-slate-400">
                  <span>12 Months Projected</span>
                  <span className="font-mono text-purple-400 font-bold">
                    {projectedOutcomes.projectedLhi12m}
                  </span>
                </div>
                <div className="pt-2 border-t border-slate-800 flex justify-between items-center text-xs">
                  <span className="text-slate-400">Monte Carlo Confidence</span>
                  <span className="font-mono font-bold text-emerald-400">
                    {projectedOutcomes.confidencePct}%
                  </span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Resource Utilization Radar
                </div>
                <div className="space-y-2 text-xs font-mono">
                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Time Utilization</span>
                      <span>{capacityCheck.utilization.timePct}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-2 rounded overflow-hidden">
                      <div
                        className={`h-full transition-all ${
                          capacityCheck.utilization.timePct > 100
                            ? 'bg-rose-500'
                            : 'bg-emerald-500'
                        }`}
                        style={{ width: `${Math.min(100, capacityCheck.utilization.timePct)}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Attention Load</span>
                      <span>{capacityCheck.utilization.attentionPct}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-2 rounded overflow-hidden">
                      <div
                        className={`h-full transition-all ${
                          capacityCheck.utilization.attentionPct > 100
                            ? 'bg-rose-500'
                            : 'bg-blue-500'
                        }`}
                        style={{
                          width: `${Math.min(100, capacityCheck.utilization.attentionPct)}%`,
                        }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Energy Demand</span>
                      <span>{capacityCheck.utilization.energyPct}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-2 rounded overflow-hidden">
                      <div
                        className={`h-full transition-all ${
                          capacityCheck.utilization.energyPct > 100
                            ? 'bg-rose-500'
                            : 'bg-amber-500'
                        }`}
                        style={{
                          width: `${Math.min(100, capacityCheck.utilization.energyPct)}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {capacityCheck.rebalanceSuggestions.length > 0 && (
                <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded text-xs space-y-2 text-amber-200">
                  <div className="font-semibold flex items-center gap-1">
                    <span>💡</span> Rebalancing Recommendation
                  </div>
                  {capacityCheck.rebalanceSuggestions.map((sug, idx) => (
                    <div key={idx} className="text-[11px] text-amber-300/80">
                      • {sug.reason}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </HorizonCard>
        </div>
      </div>
    </div>
  );
}

export default function AllocatorPage() {
  return (
    <Suspense
      fallback={
        <div className="p-8 text-slate-400 bg-slate-950 min-h-screen">
          Loading Resource Allocator...
        </div>
      }
    >
      <AllocatorContent />
    </Suspense>
  );
}
