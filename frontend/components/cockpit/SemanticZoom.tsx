"use client";

import React, { useState } from "react";
import Link from "next/link";

export type ZoomLevel = 0 | 1 | 2;

export interface SemanticZoomLevel {
  level: number;
  label: string;
  detail: string;
}

export interface SemanticZoomProps {
  hubTitle?: string;
  workbenchRoute?: string;
  workbenchName?: string;
  level0Content?: React.ReactNode;
  level1Content?: React.ReactNode;
  level2Content?: React.ReactNode;
  level2PreviewText?: string;
  initialLevel?: ZoomLevel;
  defaultLevel?: number;
  activeLevel?: number;
  onLevelChange?: (level: number) => void;
  levels?: SemanticZoomLevel[];
}

export default function SemanticZoom({
  hubTitle = "Semantic Zoom",
  workbenchRoute = "/workbench/allocator",
  workbenchName = "Allocator Workbench",
  level0Content,
  level1Content,
  level2Content,
  level2PreviewText,
  initialLevel = 0,
  defaultLevel,
  activeLevel: controlledLevel,
  onLevelChange,
  levels,
}: SemanticZoomProps) {
  const [internalLevel, setInternalLevel] = useState<number>(defaultLevel ?? initialLevel);
  const activeLevel = controlledLevel !== undefined ? controlledLevel : internalLevel;

  const handleLevelSelect = (lvl: number) => {
    setInternalLevel(lvl);
    onLevelChange?.(lvl);
  };

  if (levels && levels.length > 0 && !level0Content) {
    const current = levels.find((l) => l.level === activeLevel) || levels[0];
    return (
      <div className="w-full bg-[#0d131f] border border-gray-800 rounded-xl p-4 shadow-xl space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-gray-800/80 pb-3">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-2 py-0.5 rounded">
              Semantic Zoom
            </span>
            <span className="text-xs text-gray-400 font-mono">
              Level {activeLevel}: {current?.label}
            </span>
          </div>
          <div className="flex items-center space-x-1 bg-gray-900 border border-gray-800 rounded-lg p-1">
            {levels.map((lvl) => (
              <button
                key={lvl.level}
                onClick={() => handleLevelSelect(lvl.level)}
                className={`px-3 py-1 rounded text-xs font-mono transition-colors ${
                  activeLevel === lvl.level
                    ? "bg-emerald-600 text-white font-semibold shadow"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                L{lvl.level}
              </button>
            ))}
          </div>
        </div>
        <p className="text-xs text-gray-300 font-mono">{current?.detail}</p>
      </div>
    );
  }

  return (
    <div className="w-full bg-[#0d131f] border border-gray-800 rounded-xl overflow-hidden shadow-2xl transition-all duration-300">
      {/* Semantic Zoom Header & Controls */}
      <div className="flex flex-wrap items-center justify-between px-6 py-4 border-b border-gray-800/80 bg-gray-950/60 gap-3">
        <div className="flex items-center space-x-3">
          <span className="text-xs font-mono font-semibold uppercase tracking-wider text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-2.5 py-0.5 rounded-md">
            Semantic Zoom
          </span>
          <h3 className="text-sm font-medium text-gray-200">
            {hubTitle}
          </h3>
        </div>

        {/* Zoom Level Switcher */}
        <div className="flex items-center space-x-1 bg-gray-900 border border-gray-800 rounded-lg p-1">
          <button
            onClick={() => handleLevelSelect(0)}
            className={`px-3 py-1 rounded text-xs font-mono transition-colors ${
              activeLevel === 0
                ? "bg-emerald-600 text-white font-semibold shadow"
                : "text-gray-400 hover:text-gray-200"
            }`}
            title="Level 0: 30-Second Overview"
          >
            L0 Overview
          </button>
          <button
            onClick={() => handleLevelSelect(1)}
            className={`px-3 py-1 rounded text-xs font-mono transition-colors ${
              activeLevel === 1
                ? "bg-cyan-600 text-white font-semibold shadow"
                : "text-gray-400 hover:text-gray-200"
            }`}
            title="Level 1: Context Drawer"
          >
            L1 Context
          </button>
          <button
            onClick={() => handleLevelSelect(2)}
            className={`px-3 py-1 rounded text-xs font-mono transition-colors ${
              activeLevel === 2
                ? "bg-purple-600 text-white font-semibold shadow"
                : "text-gray-400 hover:text-gray-200"
            }`}
            title="Level 2: Specialist Workbench"
          >
            L2 Workbench
          </button>
        </div>
      </div>

      {/* Semantic Zoom Dynamic Canvas */}
      <div className="p-6">
        {/* Level 0: 30-Second Overview */}
        {activeLevel === 0 && (
          <div className="space-y-4 animate-fadeIn">
            <div className="flex items-center justify-between text-xs text-gray-400 border-b border-gray-800/60 pb-2">
              <span className="font-mono">FOCUS: 30-Second At-A-Glance Execution</span>
              <button
                onClick={() => handleLevelSelect(1)}
                className="text-cyan-400 hover:text-cyan-300 font-medium hover:underline flex items-center space-x-1"
              >
                <span>Expand Context (L1)</span>
                <span>→</span>
              </button>
            </div>
            {level0Content}
          </div>
        )}

        {/* Level 1: Context Drawer */}
        {activeLevel === 1 && (
          <div className="space-y-4 animate-fadeIn">
            <div className="flex items-center justify-between text-xs text-gray-400 border-b border-gray-800/60 pb-2">
              <span className="font-mono text-cyan-400 font-semibold">
                FOCUS: Diagnostic Lineage & Factor Breakdown
              </span>
              <div className="flex items-center space-x-3">
                <button
                  onClick={() => handleLevelSelect(0)}
                  className="text-gray-400 hover:text-gray-200 hover:underline"
                >
                  ← Collapse to L0
                </button>
                <Link
                  href={workbenchRoute}
                  className="text-purple-400 hover:text-purple-300 font-medium hover:underline flex items-center space-x-1"
                >
                  <span>Open {workbenchName} (L2)</span>
                  <span>→</span>
                </Link>
              </div>
            </div>
            {level1Content}
          </div>
        )}

        {/* Level 2: Specialist Workbench Handoff */}
        {activeLevel === 2 && (
          <div className="space-y-6 py-4 animate-fadeIn">
            {level2Content ? (
              <div className="space-y-4">
                {level2Content}
              </div>
            ) : (
              <div className="border border-purple-900/50 bg-purple-950/20 rounded-xl p-6 text-center space-y-4">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-purple-900/40 border border-purple-700/60 text-purple-300 text-xl font-mono">
                  ⚙️
                </div>
                <h4 className="text-lg font-semibold text-white">
                  Launch Specialist Workbench: {workbenchName}
                </h4>
                <p className="text-sm text-gray-300 max-w-xl mx-auto">
                  {level2PreviewText ||
                    `Deep-dive into multi-parameter simulation, raw signal telemetry, node topology, and high-frequency parameter calibration.`}
                </p>
                <div className="pt-2">
                  <Link
                    href={workbenchRoute}
                    className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-white font-medium text-sm transition-colors shadow-lg shadow-purple-900/40"
                  >
                    <span>Enter {workbenchName}</span>
                    <span className="font-mono">→</span>
                  </Link>
                </div>
              </div>
            )}

            <div className="flex justify-center">
              <button
                onClick={() => handleLevelSelect(0)}
                className="text-xs text-gray-400 hover:text-gray-200 hover:underline"
              >
                ← Return to Level 0 Overview
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
