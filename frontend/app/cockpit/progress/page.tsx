"use client";

import CockpitShell from "../../../components/cockpit/CockpitShell";

import React from "react";
import Link from "next/link";
import { getUnifiedCockpitState } from "../../../lib/simulation/unifiedCockpitStore";
import SemanticZoom from "../../../components/cockpit/SemanticZoom";

export default function ProgressHubPage() {
  const state = getUnifiedCockpitState();
  const { triad, identityDrift, skillTrajectories, calibrationScore, targetIdentityRole } = state;

  return (
    <CockpitShell activeHub="progress"><div className="space-y-8">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center md:justify-between pb-6 border-b border-gray-800/80 gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <span className="text-xs font-mono font-bold tracking-wider uppercase text-purple-400 bg-purple-950/60 border border-purple-800/60 px-2.5 py-1 rounded">
              Core Hub 3
            </span>
            <span className="text-xs font-mono text-gray-400">Horizons 11, 12, 13 Progress</span>
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white mt-1">
            Progress & Calibration
          </h1>
          <p className="text-sm text-gray-400 mt-0.5">
            Identity twin evolution, behavioral drift monitoring, and decision calibration.
          </p>
        </div>

        {/* Global Triad Banner */}
        <div className="flex items-center space-x-3 bg-gray-900/90 border border-gray-800 rounded-xl p-3 shadow-inner">
          <div className="text-center px-3 border-r border-gray-800">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">LHI</span>
            <span className="text-xl font-mono font-bold text-emerald-400">{triad.lhi}</span>
          </div>
          <div className="text-center px-3 border-r border-gray-800">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">HHI</span>
            <span className="text-xl font-mono font-bold text-blue-400">{triad.hhi}</span>
          </div>
          <div className="text-center px-3">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">IAI</span>
            <span className="text-xl font-mono font-bold text-purple-400">{triad.iai}</span>
          </div>
        </div>
      </header>

      {/* Behavioral Drift Alert Banner */}
      {identityDrift.hasActiveDrift && (
        <section className="p-5 rounded-xl bg-amber-950/30 border border-amber-800/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1 max-w-2xl">
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono font-bold text-amber-300 uppercase px-2 py-0.5 rounded bg-black/40 border border-amber-800">
                Drift Detected: {identityDrift.domain}
              </span>
              <span className="text-xs font-mono text-gray-400">
                {identityDrift.inactiveDays} days inactive (Threshold: {identityDrift.thresholdDays}d)
              </span>
            </div>
            <p className="text-xs text-amber-200/90">{identityDrift.impactExplanation}</p>
          </div>
          <button className="px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-black font-semibold text-xs transition-colors shrink-0 shadow">
            Remedy: {identityDrift.remedyAction}
          </button>
        </section>
      )}

      {/* Semantic Zoom on Progress & Calibration */}
      <SemanticZoom
        hubTitle="Identity Twin & Calibration Matrix"
        workbenchRoute="/workbench/journal"
        workbenchName="Decision Journal Workbench"
        level0Content={
          <div className="space-y-6">
            {/* Identity Target Overview */}
            <div className="p-6 rounded-xl bg-gradient-to-br from-gray-900 to-gray-950 border border-purple-800/50 space-y-4">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                <div>
                  <span className="text-xs font-mono text-purple-400 uppercase font-semibold">Target Identity Vector</span>
                  <h3 className="text-xl font-bold text-white">{targetIdentityRole}</h3>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <span className="text-[10px] font-mono text-gray-400 uppercase block">IAI Score</span>
                    <span className="text-2xl font-mono font-bold text-purple-400">{triad.iai}/100</span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] font-mono text-gray-400 uppercase block">Brier Score</span>
                    <span className="text-2xl font-mono font-bold text-emerald-400">{calibrationScore.brierScore}</span>
                  </div>
                </div>
              </div>

              {/* Skills Trajectory Bars */}
              <div className="space-y-3 pt-2">
                {skillTrajectories.map((st) => (
                  <div key={st.skill} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-300 font-medium">{st.skill}</span>
                      <span className="font-mono text-gray-400">
                        {st.currentScore} → <strong className="text-purple-300">{st.targetScore}</strong> (+{st.gapPoints} gap)
                      </span>
                    </div>
                    <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full"
                        style={{ width: `${(st.currentScore / st.targetScore) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Calibration & Accuracy Glance */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 space-y-1">
                <span className="text-xs font-mono text-gray-400 uppercase">Probabilistic Accuracy</span>
                <p className="text-xl font-mono font-bold text-white">{calibrationScore.accuracyPct}%</p>
                <p className="text-[11px] text-gray-400">Audited across {calibrationScore.sampleDecisionsAudited} decisions.</p>
              </div>
              <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 space-y-1">
                <span className="text-xs font-mono text-gray-400 uppercase">Overconfidence Bias</span>
                <p className="text-xl font-mono font-bold text-emerald-400">{calibrationScore.overconfidenceBias}</p>
                <p className="text-[11px] text-gray-400">Predicted probabilities strictly mirror actual event frequencies.</p>
              </div>
              <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 space-y-1">
                <span className="text-xs font-mono text-gray-400 uppercase">Judgment Quality</span>
                <p className="text-xl font-mono font-bold text-cyan-400">{calibrationScore.trend}</p>
                <p className="text-[11px] text-gray-400">Brier score {calibrationScore.brierScore} (Threshold $\le 0.25$).</p>
              </div>
            </div>
          </div>
        }
        level1Content={
          <div className="space-y-6">
            <div className="p-5 rounded-xl bg-gray-950 border border-gray-800 space-y-4">
              <h4 className="text-sm font-bold text-white">Detailed Competence Gap Diagnostic</h4>
              <div className="space-y-3">
                {skillTrajectories.map((st) => (
                  <div key={st.skill} className="p-3 rounded-lg bg-gray-900/70 border border-gray-800/80 flex items-center justify-between text-xs">
                    <div>
                      <p className="font-semibold text-white">{st.skill}</p>
                      <p className="text-gray-400">Momentum compounding velocity: {st.momentumVelocityPct}%</p>
                    </div>
                    <div className="text-right font-mono">
                      <span className="text-gray-400 block">Gap Points</span>
                      <span className="text-purple-400 font-bold">+{st.gapPoints} pts</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        }
      />
    </div></CockpitShell>
  );
}
