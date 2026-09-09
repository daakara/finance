'use client';

/**
 * Horizon 6: /me/signals — Personal Signal Health & Real-World Integration Layer
 *
 * Implements:
 * - INV-OI84-P (Personal Signal Freshness Invariant)
 * - INV-OI86-P (Conflict Resolution Integrity & Audit Ledger)
 * - INV-OI85-P (Decision Outcome Capture & Calibration)
 * - INV-OI87-P (Signal Quality Coverage Score)
 * - Connected Sources Management (Calendar, Health, Finance, Learning)
 */

import React, { useState, useMemo, Suspense } from 'react';
import Link from 'next/link';
import IntelligenceHeader from '../../../components/ui/IntelligenceHeader';
import HorizonCard from '../../../components/ui/HorizonCard';
import HorizonMetricCard from '../../../components/ui/HorizonMetricCard';
import SeverityBadge from '../../../components/ui/SeverityBadge';
import {
  CANONICAL_CONNECTED_SOURCES,
  CANONICAL_SIGNALS,
  CANONICAL_CONFLICTS,
  CANONICAL_DECISION_OUTCOMES,
  calculateSignalQuality,
  verifySignalFreshness,
} from '../../../lib/simulation/personalSignalEngine';

function SignalsContent() {
  const [sources, setSources] = useState(CANONICAL_CONNECTED_SOURCES);
  const [activeSignals] = useState(CANONICAL_SIGNALS);
  const [conflicts] = useState(CANONICAL_CONFLICTS);
  const [outcomes] = useState(CANONICAL_DECISION_OUTCOMES);
  const [syncingSourceId, setSyncingSourceId] = useState<string | null>(null);

  // Compute composite signal quality
  const quality = useMemo(() => {
    return calculateSignalQuality(activeSignals);
  }, [activeSignals]);

  const handleForceSync = (sourceId: string) => {
    setSyncingSourceId(sourceId);
    setTimeout(() => {
      setSources((prev) =>
        prev.map((s) =>
          s.id === sourceId
            ? { ...s, lastSyncUtc: new Date().toISOString(), freshnessScore: 100 }
            : s
        )
      );
      setSyncingSourceId(null);
    }, 600);
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
          <span className="text-xs uppercase tracking-wider font-semibold text-cyan-400">
            Signal Health & Data Quality
          </span>
        </div>
        <div className="flex items-center gap-2">
          <SeverityBadge
            level={quality.overallQualityScore >= 75 ? 'LOW' : 'HIGH'}
            status={quality.overallQualityScore >= 75 ? 'HEALTHY' : 'WARNING'}
          />
        </div>
      </div>

      <IntelligenceHeader
        title="Personal Signal & Data Integration Layer"
        subtitle="Real-world data ingestion, INV-OI84-P freshness decay verification, conflict resolution ledger, and decision outcome calibration."
        certification="HORIZON-6-CERTIFIED"
        status="CERTIFIED"
        breadcrumbs={[
          { label: 'Life OS', href: '/me' },
          { label: 'Signals & Integrations' },
        ]}
      />

      {/* Top Signal Quality Composite (INV-OI87-P) */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <HorizonMetricCard
          label="OVERALL SIGNAL QUALITY"
          value={`${quality.overallQualityScore} / 100`}
          delta={quality.status}
          deltaPositive={quality.overallQualityScore >= 75}
          severity={quality.overallQualityScore >= 75 ? 'PASS' : 'WARN'}
        />
        <HorizonMetricCard
          label="SIGNAL FRESHNESS"
          value={`${quality.freshnessScore}%`}
          delta="INV-OI84-P Verified"
          deltaPositive={quality.freshnessScore >= 70}
          severity={quality.freshnessScore >= 70 ? 'PASS' : 'WARN'}
        />
        <HorizonMetricCard
          label="CATEGORY COVERAGE"
          value={`${quality.coverageScore}%`}
          delta="5 / 5 Domains Active"
          deltaPositive={quality.coverageScore >= 80}
          severity="PASS"
        />
        <HorizonMetricCard
          label="SOURCE CONFIDENCE"
          value={`${quality.confidenceScore}%`}
          delta="Multi-source verified"
          deltaPositive={true}
          severity="PASS"
        />
      </div>

      {/* Connected Sources Health Cards */}
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-xs uppercase tracking-wider font-semibold text-slate-300">
            Connected Real-World Sources (Layer 1 Connectors)
          </h2>
          <span className="text-xs text-slate-500 font-mono">
            {sources.length} Active Data Feeds
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {sources.map((src) => {
            const isSyncing = syncingSourceId === src.id;
            return (
              <div
                key={src.id}
                className="p-4 bg-slate-900/80 rounded-lg border border-slate-800 space-y-3"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <span className="text-[10px] font-mono text-cyan-400 uppercase">
                      {src.category}
                    </span>
                    <div className="text-sm font-semibold text-slate-200">{src.name}</div>
                  </div>
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    {src.status}
                  </span>
                </div>

                <div className="space-y-1 text-xs text-slate-400 font-mono">
                  <div className="flex justify-between">
                    <span>Freshness:</span>
                    <span className="text-slate-200 font-bold">{src.freshnessScore}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Confidence:</span>
                    <span className="text-slate-200">{src.confidencePct}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Signals:</span>
                    <span className="text-slate-200">{src.totalSignalsTracked} feeds</span>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-800/80 flex justify-between items-center text-[10px]">
                  <span className="text-slate-500">Auto-synced</span>
                  <button
                    onClick={() => handleForceSync(src.id)}
                    disabled={isSyncing}
                    className="text-cyan-400 hover:text-cyan-300 font-semibold transition-colors disabled:opacity-50"
                  >
                    {isSyncing ? 'Syncing...' : 'Force Sync ↻'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Freshness Timeline & Degradation Radar (INV-OI84-P) */}
      <HorizonCard
        title="Signal Freshness & Natural Degradation (INV-OI84-P)"
        subtitle="Mathematical decay: Freshness = 100 * (1 - age / maxAge). Stale signals age out and are fail-closed rejected."
      >
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {activeSignals.map((sig) => {
              const fresh = verifySignalFreshness(sig);
              return (
                <div
                  key={sig.signalId}
                  className="p-3 bg-slate-900/60 rounded border border-slate-800/80 space-y-2 text-xs"
                >
                  <div className="flex justify-between items-center font-mono">
                    <span className="font-semibold text-slate-200">{sig.metricId}</span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        fresh.status === 'FRESH'
                          ? 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20'
                          : fresh.status === 'AGING'
                          ? 'text-amber-400 bg-amber-500/10 border border-amber-500/20'
                          : 'text-rose-400 bg-rose-500/10 border border-rose-500/20'
                      }`}
                    >
                      {fresh.status} ({fresh.freshnessScore}%)
                    </span>
                  </div>

                  <div className="flex justify-between text-slate-400 text-[11px] font-mono">
                    <span>
                      Observed: {sig.value} {sig.unit}
                    </span>
                    <span>
                      Age: {sig.metadata.freshnessHours}h / Max: {sig.metadata.maxAllowedAgeHours}h
                    </span>
                  </div>

                  {/* Degradation bar */}
                  <div className="w-full bg-slate-800 h-1.5 rounded overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        fresh.freshnessScore >= 70
                          ? 'bg-emerald-500'
                          : fresh.freshnessScore >= 30
                          ? 'bg-amber-500'
                          : 'bg-rose-500'
                      }`}
                      style={{ width: `${fresh.freshnessScore}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </HorizonCard>

      {/* Conflict Resolution Ledger (INV-OI86-P) */}
      <HorizonCard
        title="Signal Conflict Resolution Ledger (INV-OI86-P)"
        subtitle="Never silently overwrite: All conflicting observations between devices, APIs, and manual entries are reconciled via deterministic priority or weighted merge."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border border-slate-800 rounded-lg overflow-hidden font-mono">
            <thead className="bg-slate-900 text-slate-400 uppercase text-[10px]">
              <tr>
                <th className="p-3">Conflict ID</th>
                <th className="p-3">Metric</th>
                <th className="p-3">Source A</th>
                <th className="p-3">Source B</th>
                <th className="p-3">Method</th>
                <th className="p-3">Resolved Value</th>
                <th className="p-3">Audit Lineage</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-300">
              {conflicts.map((cnf) => (
                <tr key={cnf.conflictId} className="hover:bg-slate-900/40 transition-colors">
                  <td className="p-3 text-cyan-400 font-semibold">{cnf.conflictId}</td>
                  <td className="p-3">{cnf.metricId}</td>
                  <td className="p-3 text-slate-300">
                    {cnf.sourceA}: {cnf.valueA} ({cnf.confidenceA}%)
                  </td>
                  <td className="p-3 text-slate-400">
                    {cnf.sourceB}: {cnf.valueB} ({cnf.confidenceB}%)
                  </td>
                  <td className="p-3">
                    <span className="px-1.5 py-0.5 rounded text-[9px] bg-blue-500/10 text-blue-300 border border-blue-500/20 font-bold">
                      {cnf.resolutionMethod}
                    </span>
                  </td>
                  <td className="p-3 text-emerald-400 font-bold">{cnf.resolvedValue}</td>
                  <td className="p-3 text-[11px] text-slate-400 font-sans">{cnf.auditReason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </HorizonCard>

      {/* Decision Outcome Calibration Matrix (INV-OI85-P) */}
      <HorizonCard
        title="Decision Outcome Capture & Model Calibration (INV-OI85-P)"
        subtitle="Closed-loop learning: Comparing expected recommendation impact vs. observed real-world outcomes to calibrate digital twin accuracy."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border border-slate-800 rounded-lg overflow-hidden font-mono">
            <thead className="bg-slate-900 text-slate-400 uppercase text-[10px]">
              <tr>
                <th className="p-3">Decision ID</th>
                <th className="p-3">Recommendation</th>
                <th className="p-3">Domain</th>
                <th className="p-3">Expected Gain</th>
                <th className="p-3">Actual Gain</th>
                <th className="p-3">Calibration Delta</th>
                <th className="p-3">Brier Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-300">
              {outcomes.map((dec) => (
                <tr key={dec.decisionId} className="hover:bg-slate-900/40 transition-colors">
                  <td className="p-3 text-cyan-400 font-semibold">{dec.decisionId}</td>
                  <td className="p-3 font-sans font-medium text-slate-200">
                    {dec.recommendationTitle}
                  </td>
                  <td className="p-3">
                    <span className="text-[10px] text-slate-400">{dec.category}</span>
                  </td>
                  <td className="p-3 text-slate-300">+{dec.expectedMetricGain}</td>
                  <td className="p-3 text-emerald-400 font-bold">+{dec.actualMetricGain}</td>
                  <td className="p-3">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        dec.calibrationDeltaPct >= 0
                          ? 'text-emerald-400 bg-emerald-500/10'
                          : 'text-amber-400 bg-amber-500/10'
                      }`}
                    >
                      {dec.calibrationDeltaPct > 0 ? '+' : ''}
                      {dec.calibrationDeltaPct}%
                    </span>
                  </td>
                  <td className="p-3 text-slate-400">{dec.brierScoreContribution}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </HorizonCard>
    </div>
  );
}

export default function SignalsPage() {
  return (
    <Suspense
      fallback={
        <div className="p-8 text-slate-400 bg-slate-950 min-h-screen">
          Loading Signal Health Dashboard...
        </div>
      }
    >
      <SignalsContent />
    </Suspense>
  );
}
