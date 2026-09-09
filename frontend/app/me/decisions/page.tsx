"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import IntelligenceHeader from '../../../components/ui/IntelligenceHeader';
import {
  CANONICAL_JOURNAL_ENTRIES,
  calculateCalibrationSummary,
  JournalEntry,
  JournalDomain,
} from '../../../lib/simulation/decisionJournalEngine';

export default function DecisionsPage() {
  const [entries] = useState<JournalEntry[]>(CANONICAL_JOURNAL_ENTRIES);
  const [filterDomain, setFilterDomain] = useState<JournalDomain | 'ALL'>('ALL');

  const calibration = calculateCalibrationSummary(entries);

  const filteredEntries =
    filterDomain === 'ALL'
      ? entries
      : entries.filter((e) => e.domain === filterDomain);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 space-y-8 max-w-5xl mx-auto">
      <IntelligenceHeader
        certification="HORIZON-11-CERTIFIED"
        title="Decision & Outcome Journal"
        subtitle="Tracking Expected vs. Actual Outcomes to Calibrate Human Judgment & Invariant Sizing"
        breadcrumbs={[
          { label: 'Life OS', href: '/me' },
          { label: 'Decision Journal' },
        ]}
      />

      {/* Navigation Breadcrumb */}
      <div className="flex items-center justify-between text-xs text-slate-400">
        <Link href="/me" className="text-emerald-400 hover:underline">
          ← Back to 30-Second Cockpit
        </Link>
        <Link href="/me/execute" className="text-emerald-400 hover:underline">
          Open Execution Cockpit →
        </Link>
      </div>

      {/* Top Scorecard: Brier Calibration & Accuracy */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-md">
          <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-1">
            Calibration Accuracy
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white">
              {calibration.calibrationAccuracyPct}%
            </span>
            <span className="text-xs text-emerald-400 font-semibold">
              Brier: {calibration.brierScore}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-2">
            Status: <strong className="text-emerald-400">{calibration.calibrationStatus}</strong> (Low forecast drift)
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-md">
          <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-1">
            Evaluated Decisions
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-white">
              {calibration.totalEvaluated}
            </span>
            <span className="text-xs text-cyan-400 font-semibold">
              Across 4 Domains
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-2">
            Trading (92%), Career (94%), Household (95%)
          </p>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 shadow-md">
          <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-1">
            Empirical Key Lesson
          </span>
          <p className="text-xs text-slate-300 leading-relaxed mt-1">
            &ldquo;{calibration.keyLessonLearned}&rdquo;
          </p>
        </div>
      </div>

      {/* Domain Filter Switcher */}
      <div className="flex flex-wrap items-center gap-2 bg-slate-900/60 p-2 rounded-xl border border-slate-800">
        {(['ALL', 'TRADING', 'CAREER', 'HOUSEHOLD', 'FINANCE', 'HEALTH'] as const).map((dom) => (
          <button
            key={dom}
            onClick={() => setFilterDomain(dom)}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              filterDomain === dom
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {dom}
          </button>
        ))}
      </div>

      {/* Journal Entry Feed */}
      <div className="space-y-4">
        {filteredEntries.map((entry) => (
          <div
            key={entry.id}
            className="p-6 rounded-2xl border border-slate-800 bg-slate-900/80 hover:border-slate-700 transition-all space-y-4"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
              <div className="flex items-center gap-2.5">
                <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 uppercase tracking-wider">
                  {entry.domain}
                </span>
                <h3 className="text-base font-bold text-white">{entry.title}</h3>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400 font-mono">
                <span>Decided: {entry.decisionDate}</span>
                {entry.status === 'EVALUATED' && (
                  <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold">
                    Evaluated
                  </span>
                )}
              </div>
            </div>

            {/* Thesis */}
            <p className="text-sm text-slate-300 leading-relaxed">
              <strong>Thesis:</strong> {entry.thesis}
            </p>

            {/* Expected vs. Actual Comparison Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-3 bg-slate-950/60 rounded-xl border border-slate-800/60 text-xs">
              <div>
                <span className="text-slate-500 block">Expected Outcome</span>
                <span className="text-slate-200 font-semibold">
                  {entry.expectedFinancialDeltaDollars ? `+$${entry.expectedFinancialDeltaDollars} ` : ''}
                  (+{entry.expectedLhiImpact} LHI)
                </span>
              </div>
              <div>
                <span className="text-slate-500 block">Realized Outcome</span>
                <span className="text-emerald-400 font-semibold">
                  {entry.realizedFinancialDeltaDollars ? `+$${entry.realizedFinancialDeltaDollars} ` : ''}
                  {entry.realizedLhiImpact ? `(+${entry.realizedLhiImpact} LHI)` : 'In Progress'}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block">Biometric Context</span>
                <span className="text-cyan-400 font-semibold">
                  Recovery {entry.biometricContext.recoveryScore}% · {entry.biometricContext.restingStress} Stress
                </span>
              </div>
            </div>

            {/* Retrospective Reflection */}
            {entry.retrospectiveReflection && (
              <div className="p-3 bg-indigo-950/20 rounded-xl border border-indigo-900/40 text-xs text-slate-300">
                <span className="font-bold text-indigo-300 block mb-1">
                  Retrospective Reflection:
                </span>
                <p>{entry.retrospectiveReflection}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
