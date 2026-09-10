"use client";

import CockpitShell from "../../components/cockpit/CockpitShell";

import React from "react";
import Link from "next/link";
import { useUnifiedCockpit } from "../../lib/simulation/unifiedCockpitStore";
import SemanticZoom from "../../components/cockpit/SemanticZoom";

export default function ProgressHubPage() {
  const state = useUnifiedCockpit();
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
            Progress &amp; Calibration
          </h1>
          <p className="text-sm text-gray-400 mt-0.5">
            Identity twin evolution, behavioral drift monitoring, and decision calibration.
          </p>
        </div>

        {/* Global Triad Banner */}
        <div className="flex items-center space-x-3 bg-gray-900/90 border border-gray-800 rounded-xl p-3 shadow-inner">
          <div className="text-center px-3 border-r border-gray-800">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">LHI</span>
            <span className="text-xl font-mono font-bold text-emerald-400">{triad?.lhi ?? "--"}</span>
          </div>
          <div className="text-center px-3 border-r border-gray-800">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">HHI</span>
            <span className="text-xl font-mono font-bold text-blue-400">{triad?.hhi ?? "--"}</span>
          </div>
          <div className="text-center px-3">
            <span className="text-[10px] uppercase font-mono text-gray-400 block">IAI</span>
            <span className="text-xl font-mono font-bold text-purple-400">{triad?.iai ?? "--"}</span>
          </div>
        </div>
      </header>

      {/* Behavioral Drift Alert Banner */}
      {identityDrift && identityDrift.hasActiveDrift ? (
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
            <p className="text-sm text-gray-200">{identityDrift.impactExplanation}</p>
            <p className="text-xs font-mono text-amber-400">Remedy: {identityDrift.remedyAction}</p>
          </div>
          <Link
            href="/workbench/journal"
            className="text-xs font-mono text-amber-300 hover:text-white border border-amber-800 bg-amber-950/60 px-4 py-2 rounded-lg transition-colors shrink-0 font-bold"
          >
            Audit Drift in Journal →
          </Link>
        </section>
      ) : (
        <section className="p-4 rounded-xl bg-gray-900/40 border border-gray-800 text-xs font-mono text-gray-400 flex items-center justify-between">
          <span>Behavioral Drift Monitoring: Alignment nominal. Zero unmitigated drift detected.</span>
          <span className="text-emerald-400 font-bold">NOMINAL</span>
        </section>
      )}

      {/* Semantic Zoom Container for Progress & Calibration */}
      <SemanticZoom
        hubTitle="Identity Compounding & Calibration Engine"
        workbenchRoute="/workbench/journal"
        workbenchName="Decision Journal Workbench"
        level0Content={
          <div className="space-y-6">
            <div className="p-6 rounded-2xl bg-gradient-to-r from-purple-950/40 via-gray-900 to-gray-900 border border-purple-800/50 flex flex-col md:flex-row md:items-center justify-between gap-6 shadow-xl">
              <div className="space-y-1">
                <span className="text-xs font-mono uppercase font-bold text-purple-400 bg-purple-950 px-2 py-0.5 rounded border border-purple-800">
                  Target Identity Role
                </span>
                <h2 className="text-2xl font-bold text-white tracking-tight">
                  {targetIdentityRole || "Identity Role Unassigned"}
                </h2>
                <p className="text-xs text-gray-400">
                  IAI measures daily compounding toward declared systemic mastery and strategic leverage.
                </p>
              </div>

              {calibrationScore ? (
                <div className="flex items-center space-x-4 bg-gray-950/70 border border-gray-800 p-4 rounded-xl">
                  <div>
                    <span className="text-[10px] font-mono uppercase text-gray-400 block">Brier Score</span>
                    <span className="text-xl font-mono font-bold text-emerald-400">{calibrationScore.brierScore}</span>
                  </div>
                  <div className="h-8 w-px bg-gray-800" />
                  <div>
                    <span className="text-[10px] font-mono uppercase text-gray-400 block">Accuracy</span>
                    <span className="text-xl font-mono font-bold text-white">{calibrationScore.accuracyPct}%</span>
                  </div>
                </div>
              ) : null}
            </div>

            {/* Trajectories Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {skillTrajectories && skillTrajectories.map((s) => (
                <div key={s.skill} className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 space-y-2">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-white font-bold">{s.skill}</span>
                    <span className="text-purple-400 font-bold">{s.currentScore} / {s.targetScore}</span>
                  </div>
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-emerald-400"
                      style={{ width: `${(s.currentScore / s.targetScore) * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] font-mono text-gray-400">
                    <span>Gap: -{s.gapPoints} pts</span>
                    <span className="text-emerald-400">{s.momentumVelocityPct}% velocity</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        }
        level1Content={
          <div className="space-y-4">
            <h4 className="text-sm font-bold text-white">Decision Journal Calibration Audit</h4>
            <div className="p-5 rounded-xl bg-gray-950 border border-gray-800 text-xs font-mono space-y-2">
              <p className="text-gray-300">Audited Sample: {calibrationScore?.sampleDecisionsAudited ?? 0} high-stakes capital and career decisions.</p>
              <p className="text-gray-300">Overconfidence Bias: <strong className="text-emerald-400">{calibrationScore?.overconfidenceBias ?? "NONE"}</strong></p>
              <p className="text-gray-300">Historical Trend: <strong className="text-cyan-400">{calibrationScore?.trend ?? "CALIBRATED"}</strong></p>
            </div>
          </div>
        }
      />
    </div></CockpitShell>
  );
}
