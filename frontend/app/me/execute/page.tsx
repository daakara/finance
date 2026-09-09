"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import IntelligenceHeader from '../../../components/ui/IntelligenceHeader';
import {
  CANONICAL_ACTIVE_COMMITMENT,
  transitionActionState,
  diagnoseExecutionFriction,
  ActionCommitment,
} from '../../../lib/simulation/executionIntelligenceEngine';

export default function ExecutePage() {
  const [commitment, setCommitment] = useState<ActionCommitment>(CANONICAL_ACTIVE_COMMITMENT);
  const [isCompleted, setIsCompleted] = useState<boolean>(false);
  const [deferNotice, setDeferNotice] = useState<string | null>(null);

  const handleStart = () => {
    const started = transitionActionState(commitment, 'IN_PROGRESS');
    setCommitment(started);
    setDeferNotice(null);
  };

  const handleComplete = () => {
    const completed = transitionActionState(commitment, 'COMPLETED', {
      predictedOutcomeDelta: 3.2,
      observedOutcomeDelta: 3.5,
    });
    setCommitment(completed);
    setIsCompleted(true);
    setDeferNotice(null);
  };

  const handleDefer = () => {
    const deferred = transitionActionState(commitment, 'DEFERRED');
    const diagnosed = diagnoseExecutionFriction(deferred, 58); // recovery simulation
    setCommitment(diagnosed);
    setDeferNotice(
      diagnosed.frictionDiagnosis?.explanation ||
        'Commitment deferred to tomorrow. Zero shame—preserving personal bandwidth.'
    );
  };

  const toggleChecklist = (idx: number) => {
    const updatedChecklist = commitment.checklist.map((item, i) =>
      i === idx ? { ...item, completed: !item.completed } : item
    );
    setCommitment({ ...commitment, checklist: updatedChecklist });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 md:p-10 space-y-8 max-w-4xl mx-auto">
      <IntelligenceHeader
        certification="HORIZON-11-CERTIFIED"
        title="Behavioral Execution Engine"
        subtitle="Today's Primary Move. Zero distraction, seamless follow-through, and outcome learning."
        breadcrumbs={[
          { label: 'Life OS', href: '/me' },
          { label: 'Execution Cockpit' },
        ]}
      />

      {/* Navigation Breadcrumb */}
      <div className="flex items-center justify-between text-xs text-slate-400">
        <Link href="/me" className="text-emerald-400 hover:underline">
          ← Back to 30-Second Cockpit
        </Link>
        <Link href="/me/decisions" className="text-cyan-400 hover:underline">
          View Decision Journal →
        </Link>
      </div>

      {/* Primary Action Card */}
      <div className="p-8 rounded-3xl border-2 border-emerald-500/50 bg-gradient-to-br from-slate-900 via-slate-900/95 to-emerald-950/20 shadow-2xl space-y-6 relative overflow-hidden">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-2">
            <span className="text-xs px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 font-bold tracking-wider uppercase border border-emerald-500/40">
              {commitment.domain}
            </span>
            <span className="text-xs text-slate-400 font-mono">
              Est. {commitment.estimatedMinutes} Minutes
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Status:</span>
            <span
              className={`text-xs font-bold px-2.5 py-1 rounded-full uppercase tracking-wider ${
                commitment.currentState === 'COMPLETED'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                  : commitment.currentState === 'IN_PROGRESS'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 animate-pulse'
                  : commitment.currentState === 'DEFERRED'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                  : 'bg-slate-800 text-slate-300'
              }`}
            >
              {commitment.currentState}
            </span>
          </div>
        </div>

        {/* Headline */}
        <div>
          <h2 className="text-2xl md:text-3xl font-black text-white tracking-tight leading-snug">
            {commitment.headline}
          </h2>
          <p className="text-slate-300 text-sm mt-2 leading-relaxed">
            Completing this key move unlocks promotion eligibility with high focus and preserves tonight&apos;s protected family dinner.
          </p>
        </div>

        {/* Checklist */}
        <div className="space-y-3 p-5 rounded-2xl bg-slate-950/70 border border-slate-800/80">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block mb-2">
            Action Steps:
          </span>
          {commitment.checklist.map((item, idx) => (
            <div
              key={item.stepId}
              onClick={() => toggleChecklist(idx)}
              className="flex items-center gap-3 cursor-pointer group"
            >
              <div
                className={`w-5 h-5 rounded-md border flex items-center justify-center text-xs font-bold transition-all ${
                  item.completed
                    ? 'bg-emerald-600 border-emerald-500 text-white'
                    : 'border-slate-700 bg-slate-900 group-hover:border-slate-500'
                }`}
              >
                {item.completed && '✓'}
              </div>
              <span
                className={`text-sm transition-all ${
                  item.completed ? 'text-slate-500 line-through' : 'text-slate-200'
                }`}
              >
                {item.label}
              </span>
            </div>
          ))}
        </div>

        {/* Tactile Action Buttons */}
        <div className="pt-2 flex flex-wrap items-center gap-4">
          {commitment.currentState !== 'COMPLETED' && commitment.currentState !== 'IN_PROGRESS' && (
            <button
              onClick={handleStart}
              className="px-8 py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-base rounded-xl shadow-lg transition-all"
            >
              [ Start Focus Session ]
            </button>
          )}

          {commitment.currentState === 'IN_PROGRESS' && (
            <button
              onClick={handleComplete}
              className="px-8 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-base rounded-xl shadow-lg transition-all animate-pulse"
            >
              [ Mark Complete ✓ ]
            </button>
          )}

          {commitment.currentState === 'COMPLETED' && (
            <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/40 text-emerald-300 text-sm font-semibold flex items-center gap-2">
              <span>★</span> Action completed! Post-telemetry and +3.5 LHI outcome attribution recorded.
            </div>
          )}

          {commitment.currentState !== 'COMPLETED' && (
            <button
              onClick={handleDefer}
              className="px-6 py-3 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 text-sm font-semibold rounded-xl border border-slate-800 transition-all"
            >
              [ Defer to Tomorrow ]
            </button>
          )}
        </div>

        {/* Deferral / Friction Diagnosis Notification (INV-OI104-P) */}
        {deferNotice && (
          <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-500/40 text-xs text-amber-200 space-y-1">
            <span className="font-bold block">✦ Non-Punitive Friction Diagnosis (INV-OI104-P):</span>
            <p>{deferNotice}</p>
          </div>
        )}

        {/* Outcome Attribution Banner (INV-OI103-P) */}
        {isCompleted && commitment.outcomeAttribution && (
          <div className="p-4 rounded-xl bg-slate-950 border border-emerald-500/40 text-xs space-y-2">
            <div className="flex items-center justify-between text-emerald-400 font-bold">
              <span>INV-OI103-P Outcome Learning Recorded</span>
              <span>Variance: +{commitment.outcomeAttribution.variance} LHI</span>
            </div>
            <p className="text-slate-400">
              Predicted lift: +{commitment.outcomeAttribution.predictedLhiDelta} LHI · Realized lift: +{commitment.outcomeAttribution.observedLhiDelta} LHI · Telemetry updated.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
