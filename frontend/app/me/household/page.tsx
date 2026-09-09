'use client';

/**
 * Horizon 8: /me/household — Relational Digital Twin & Household Intelligence
 *
 * Implements:
 * - Multi-Twin Relational Model (Me + Partner + Child + Parent + Co-founder)
 * - Household Health Index (HHI) vs Personal Life Health Index (LHI)
 * - Shared Resource Ledger & Time-Window Double-Booking Detection (INV-OI90-P)
 * - 360-Degree Relational Impact Visibility (INV-OI91-P)
 * - 4 Canonical Cross-Twin Scenarios (Relocation, Startup, Master's, Rhythm)
 */

import React, { useState, useMemo, Suspense } from 'react';
import Link from 'next/link';
import IntelligenceHeader from '../../../components/ui/IntelligenceHeader';
import HorizonCard from '../../../components/ui/HorizonCard';
import HorizonMetricCard from '../../../components/ui/HorizonMetricCard';
import SeverityBadge from '../../../components/ui/SeverityBadge';
import {
  CANONICAL_RELATIONSHIP_NODES,
  CANONICAL_SHARED_RESOURCES,
  CANONICAL_RELATIONAL_SCENARIOS,
  calculateHouseholdHealthIndex,
  verifySharedResourceIntegrity,
  simulateRelationalScenario,
} from '../../../lib/simulation/relationshipTwinEngine';
import {
  RelationshipRole,
  RelationshipNode,
  SharedResourceAllocation,
  HouseholdHealthIndex,
  RelationalScenarioResult,
} from '../../../types/personal-digital-twin';

const ROLE_BADGES: Record<RelationshipRole, { text: string; bg: string; border: string }> = {
  PARTNER: { text: 'text-rose-400', bg: 'bg-rose-950/40', border: 'border-rose-700/50' },
  CHILD: { text: 'text-amber-400', bg: 'bg-amber-950/40', border: 'border-amber-700/50' },
  PARENT: { text: 'text-emerald-400', bg: 'bg-emerald-950/40', border: 'border-emerald-700/50' },
  COFOUNDER: { text: 'text-cyan-400', bg: 'bg-cyan-950/40', border: 'border-cyan-700/50' },
  FRIEND: { text: 'text-indigo-400', bg: 'bg-indigo-950/40', border: 'border-indigo-700/50' },
};

function HouseholdContent() {
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('move_city_relocation');
  const [simulatedDoubleBooking, setSimulatedDoubleBooking] = useState<boolean>(false);

  const activeScenarioDef = useMemo(() => {
    return (
      CANONICAL_RELATIONAL_SCENARIOS.find((s) => s.id === selectedScenarioId) ||
      CANONICAL_RELATIONAL_SCENARIOS[0]
    );
  }, [selectedScenarioId]);

  // Construct resources, optionally injecting a test clash for INV-OI90-P demonstration
  const resources: SharedResourceAllocation[] = useMemo(() => {
    if (!simulatedDoubleBooking) {
      return CANONICAL_SHARED_RESOURCES;
    }
    // Inject synthetic collision on Friday 19:00-21:00
    return CANONICAL_SHARED_RESOURCES.map((res) => {
      if (res.resourceId === 'SHARED_EVENING_BLOCKS') {
        return {
          ...res,
          allocatedCommitments: [
            ...res.allocatedCommitments,
            {
              commitmentId: 'COM_WORK_SPRINT_CLASH',
              allocatedTo: 'CRITICAL_CLIENT_LAUNCH',
              amount: 2,
              timeWindow: 'WEEKDAYS_1900_2030', // Double-booking with FAMILY_DINNER_AND_BEDTIME!
              priority: 1,
            },
          ],
        };
      }
      return res;
    });
  }, [simulatedDoubleBooking]);

  // Run simulation
  const simResult: RelationalScenarioResult = useMemo(() => {
    return simulateRelationalScenario(
      activeScenarioDef,
      82.4,
      CANONICAL_RELATIONSHIP_NODES,
      resources
    );
  }, [activeScenarioDef, resources]);

  const hhi = useMemo(() => {
    return calculateHouseholdHealthIndex(82.4, CANONICAL_RELATIONSHIP_NODES, resources);
  }, [resources]);

  const resourceAudit = useMemo(() => {
    return verifySharedResourceIntegrity(resources);
  }, [resources]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 space-y-6 max-w-7xl mx-auto">
      {/* Top Breadcrumb Nav */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <Link
            href="/me"
            className="text-xs uppercase tracking-wider font-semibold text-slate-400 hover:text-white transition-colors"
          >
            ← Back to Life Command Center
          </Link>
          <span className="text-slate-600">/</span>
          <span className="text-xs uppercase tracking-wider font-semibold text-rose-400">
            Household & Relational Twin
          </span>
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/me/twin"
            className="text-xs bg-slate-900 border border-slate-700 hover:border-slate-500 px-3 py-1.5 rounded-lg text-slate-300 hover:text-white transition-colors"
          >
            Personal Twin (DAG) →
          </Link>
          <Link
            href="/me/signals"
            className="text-xs bg-slate-900 border border-slate-700 hover:border-slate-500 px-3 py-1.5 rounded-lg text-slate-300 hover:text-white transition-colors"
          >
            Signal Layer →
          </Link>
          <SeverityBadge
            level={resourceAudit.valid && simResult.isRelationalImpactVisible ? 'LOW' : 'HIGH'}
            status={resourceAudit.valid && simResult.isRelationalImpactVisible ? 'PASS' : 'CLASH'}
          />
        </div>
      </div>

      {/* Primary Header */}
      <IntelligenceHeader
        title="Household & Relational Digital Twin"
        subtitle="Multi-Twin Relational Model · Shared Resource Integrity & 360-Degree Relational Visibility · INV-OI90-P & INV-OI91-P"
        certification="CERTIFIED MULTI-TWIN ENGINE"
        status={resourceAudit.valid ? 'OPTIMAL' : 'WARNING'}
        replayHash="0xRELATIONAL_H8"
        breadcrumbs={[
          { label: 'Home', href: '/' },
          { label: 'Life OS', href: '/me' },
          { label: 'Household', href: '/me/household' },
        ]}
      />

      {/* Metrics Banner */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <HorizonMetricCard
          label="Household Health Index (HHI)"
          value={`${(hhi.compositeHhi + simResult.householdHhiDelta).toFixed(1)} / 100`}
          delta={`${simResult.householdHhiDelta >= 0 ? '+' : ''}${simResult.householdHhiDelta.toFixed(1)} pts`}
          deltaPositive={simResult.householdHhiDelta >= 0}
          subtext={`Personal LHI: ${(82.4 + simResult.personalLhiDelta).toFixed(1)} | Weighted Family: 81.2`}
          severity={simResult.householdHhiDelta >= 0 ? 'PASS' : 'WARN'}
        />
        <HorizonMetricCard
          label="Active Relational Twins"
          value="4 Stakeholders"
          delta="Elena, Leo, David, Marcus"
          deltaPositive={true}
          subtext="Calibrated relevance weights & impact sensitivities"
          severity="INFO"
        />
        <HorizonMetricCard
          label="Shared Resource Integrity (INV-OI90-P)"
          value={resourceAudit.valid ? 'Zero Conflicts' : `${resourceAudit.violations.length} Clash Detected`}
          delta={resourceAudit.valid ? '4 Protected Reserves' : 'Double-Booked Slot'}
          deltaPositive={resourceAudit.valid}
          subtext="Calendar time-windows & shared capital pool"
          severity={resourceAudit.valid ? 'PASS' : 'CRITICAL'}
        />
        <HorizonMetricCard
          label="Relational Visibility (INV-OI91-P)"
          value="100% Disclosed"
          delta={`${simResult.relationalVisibilityCard.unvarnishedTradeOffs.length} Trade-Offs`}
          deltaPositive={true}
          subtext="Unvarnished multi-twin impact transparency"
          severity="PASS"
        />
      </div>

      {/* Strategic Household Scenarios */}
      <HorizonCard
        title="Multi-Twin Strategic Scenarios"
        badge="Household Impact Simulator"
      >
        <div className="space-y-4">
          <p className="text-xs text-slate-400 leading-relaxed">
            Major life choices affect more than one person. Compare the divergent trajectory between your isolated
            Personal LHI and the collective Household Health Index (HHI).
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {CANONICAL_RELATIONAL_SCENARIOS.map((sc) => {
              const isSelected = sc.id === selectedScenarioId;
              const isHhiPositive = sc.id === 'balanced_household_rhythm' || sc.id === 'bootstrap_startup';
              return (
                <button
                  key={sc.id}
                  onClick={() => setSelectedScenarioId(sc.id)}
                  className={`text-left p-4 rounded-xl border transition-all ${
                    isSelected
                      ? 'bg-rose-950/40 border-rose-500 shadow-lg shadow-rose-950/30'
                      : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-800/50">
                      Personal LHI: +{sc.personalLhiDelta.toFixed(1)}
                    </span>
                    <span
                      className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                        isHhiPositive
                          ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800/60'
                          : 'bg-rose-950/80 text-rose-300 border border-rose-800/60'
                      }`}
                    >
                      HHI: {sc.id === 'move_city_relocation' ? '-6.8' : sc.id === 'executive_masters' ? '-2.4' : '+4.5'}
                    </span>
                  </div>
                  <h4 className="text-sm font-semibold text-white mb-1">{sc.title}</h4>
                  <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                    {sc.description}
                  </p>
                </button>
              );
            })}
          </div>
        </div>
      </HorizonCard>

      {/* Main Dual Grid: Member Impacts & Shared Resource Ledger */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Col: Multi-Twin Member Impact Breakdown (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <HorizonCard
            title="Cross-Twin Impact & Member Wellbeing"
            badge="Relational Ripple Effect"
          >
            <div className="space-y-4">
              <p className="text-xs text-slate-400">
                Observing member-by-member emotional and operational ramifications across the household.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {simResult.memberImpacts.map((member) => {
                  const roleCfg = ROLE_BADGES[member.role] || ROLE_BADGES.PARTNER;
                  const isPositive = member.delta >= 0;
                  return (
                    <div
                      key={member.relationshipId}
                      className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-800 space-y-2.5 hover:border-slate-700 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-white text-sm">{member.name}</span>
                          <span
                            className={`text-[10px] uppercase tracking-wider font-semibold px-2 py-0.2 rounded border ${roleCfg.text} ${roleCfg.bg} ${roleCfg.border}`}
                          >
                            {member.role}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5 font-mono text-xs">
                          <span className="text-slate-400">{member.wellbeingPrior}</span>
                          <span className="text-slate-600">→</span>
                          <span className="font-bold text-white">{member.wellbeingNew}</span>
                          <span
                            className={`font-bold px-1.5 py-0.2 rounded text-[10px] ${
                              isPositive
                                ? 'text-emerald-400 bg-emerald-950/40'
                                : 'text-rose-400 bg-rose-950/40'
                            }`}
                          >
                            {isPositive ? '+' : ''}
                            {member.delta}
                          </span>
                        </div>
                      </div>

                      {/* Wellbeing meter */}
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 ${
                            member.wellbeingNew >= 80
                              ? 'bg-emerald-500'
                              : member.wellbeingNew >= 65
                              ? 'bg-amber-500'
                              : 'bg-rose-500'
                          }`}
                          style={{ width: `${Math.min(100, Math.max(0, member.wellbeingNew))}%` }}
                        />
                      </div>

                      {/* Key Concerns */}
                      <div className="space-y-1 pt-1">
                        <div className="text-[10px] uppercase font-semibold text-slate-500 tracking-wider">
                          Primary Ramifications:
                        </div>
                        {member.keyConcerns.map((concern, idx) => (
                          <div
                            key={idx}
                            className="text-[11px] text-slate-300 bg-slate-950/60 p-1.5 rounded border border-slate-800/80 leading-relaxed flex items-start gap-1.5"
                          >
                            <span className="text-slate-500">•</span>
                            <span>{concern}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </HorizonCard>

          {/* Relational Visibility Card (INV-OI91-P) */}
          <HorizonCard
            title="Unvarnished Relational Impact Disclosure"
            badge="INV-OI91-P Mandate"
          >
            <div className="space-y-4 text-xs">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 space-y-1">
                  <span className="text-[10px] uppercase font-bold text-sky-400 tracking-wider">
                    Personal Impact
                  </span>
                  <p className="text-slate-300 leading-relaxed text-[11px]">
                    {simResult.relationalVisibilityCard.personalImpactSummary}
                  </p>
                </div>
                <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 space-y-1">
                  <span className="text-[10px] uppercase font-bold text-amber-400 tracking-wider">
                    Financial Impact
                  </span>
                  <p className="text-slate-300 leading-relaxed text-[11px]">
                    {simResult.relationalVisibilityCard.financialImpactSummary}
                  </p>
                </div>
                <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 space-y-1">
                  <span className="text-[10px] uppercase font-bold text-rose-400 tracking-wider">
                    Relational Impact
                  </span>
                  <p className="text-slate-300 leading-relaxed text-[11px]">
                    {simResult.relationalVisibilityCard.relationalImpactSummary}
                  </p>
                </div>
              </div>

              {/* Unvarnished Trade-Offs List */}
              <div className="pt-2 border-t border-slate-800 space-y-2">
                <h5 className="font-semibold text-slate-300 uppercase tracking-wider text-[11px]">
                  Unvarnished Trade-Offs ({simResult.relationalVisibilityCard.unvarnishedTradeOffs.length})
                </h5>
                <div className="space-y-1.5">
                  {simResult.relationalVisibilityCard.unvarnishedTradeOffs.map((tradeoff, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-2 bg-rose-950/20 border border-rose-900/30 p-2.5 rounded-lg text-slate-300 text-[11px]"
                    >
                      <span className="text-rose-400 font-bold">⚖</span>
                      <span>{tradeoff}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </HorizonCard>
        </div>

        {/* Right Col: Shared Resource Ledger & Collision Radar (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <HorizonCard
            title="Shared Resource Ledger"
            badge="INV-OI90-P Integrity"
          >
            <div className="space-y-4 text-xs">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-slate-400">Time Windows & Capital Pool</span>
                <button
                  onClick={() => setSimulatedDoubleBooking(!simulatedDoubleBooking)}
                  className={`text-[11px] font-semibold px-2.5 py-1 rounded border transition-all ${
                    simulatedDoubleBooking
                      ? 'bg-rose-950 text-rose-300 border-rose-700'
                      : 'bg-slate-900 text-slate-300 border-slate-700 hover:border-slate-500'
                  }`}
                >
                  {simulatedDoubleBooking ? '⚠ Revert Clash Demo' : '⚡ Simulate Double-Booking Clash'}
                </button>
              </div>

              {/* Resource List */}
              <div className="space-y-3">
                {resources.map((res) => {
                  const hasConflict = res.isDoubleBooked;
                  return (
                    <div
                      key={res.resourceId}
                      className={`p-3 rounded-lg border transition-colors ${
                        hasConflict
                          ? 'bg-rose-950/30 border-rose-700 shadow-sm shadow-rose-950/30'
                          : 'bg-slate-900/70 border-slate-800'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="font-semibold text-white truncate max-w-[200px]">
                          {res.name}
                        </span>
                        <div className="flex items-center gap-2 font-mono text-[11px]">
                          <span className={hasConflict ? 'text-rose-400 font-bold' : 'text-slate-400'}>
                            {res.totalAllocated} / {res.capacityUnits} {res.unit}
                          </span>
                          <span
                            className={`text-[9px] font-bold px-1.5 py-0.2 rounded ${
                              hasConflict
                                ? 'bg-rose-950 text-rose-300 border border-rose-800'
                                : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                            }`}
                          >
                            {hasConflict ? 'CLASH' : 'PASS'}
                          </span>
                        </div>
                      </div>

                      {/* Commitments breakdown */}
                      <div className="space-y-1">
                        {res.allocatedCommitments.map((com) => (
                          <div
                            key={com.commitmentId}
                            className="flex items-center justify-between text-[10px] text-slate-400 bg-slate-950/50 px-2 py-1 rounded"
                          >
                            <span className="font-mono text-slate-300 truncate">{com.allocatedTo}</span>
                            <div className="flex items-center gap-2 font-mono">
                              {com.timeWindow && (
                                <span className="text-cyan-400/80 bg-cyan-950/30 px-1 rounded">
                                  {com.timeWindow}
                                </span>
                              )}
                              <span>
                                {com.amount} {res.unit}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Conflict details if any */}
                      {res.conflictDetails.length > 0 && (
                        <div className="mt-2 p-2 rounded bg-rose-950/50 border border-rose-800/80 text-[10px] text-rose-200 space-y-0.5">
                          {res.conflictDetails.map((det, i) => (
                            <div key={i} className="flex items-start gap-1">
                              <span>✕</span>
                              <span>{det}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Invariant Status Footer */}
              <div className="p-3 bg-slate-900/50 rounded-lg border border-slate-800 text-[11px] text-slate-400 leading-relaxed">
                <span className="font-semibold text-white">INV-OI90-P Rule: </span>
                Zero shared resource double-booking. If Friday 19:00-21:00 is scheduled for family connection,
                scheduling a client sprint on that slot triggers an immediate fail-closed conflict penalty on HHI.
              </div>
            </div>
          </HorizonCard>
        </div>
      </div>
    </div>
  );
}

export default function HouseholdPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-500">Loading Household Twin...</div>}>
      <HouseholdContent />
    </Suspense>
  );
}
