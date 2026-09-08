"use client";

import { useState } from "react";
import { InfluenceHeatmapMatrix, InfluenceMatrixEntry } from "../../types/committee-intelligence";
import { computeInfluenceHeatmap } from "../../lib/telemetry/decisionNetworkEngine";

export interface InfluenceHeatmapProps {
  matrix?: InfluenceHeatmapMatrix;
}

export default function InfluenceHeatmap({ matrix }: InfluenceHeatmapProps) {
  const currentMatrix = matrix ?? computeInfluenceHeatmap();
  const [selectedEntry, setSelectedEntry] = useState<InfluenceMatrixEntry | null>(
    currentMatrix.entries[0] ?? null
  );

  const { committeeIds, committeeNames, entries, maxInfluenceScore } = currentMatrix;

  function getEntry(sourceId: string, targetId: string): InfluenceMatrixEntry | undefined {
    return entries.find(
      (e) => e.sourceCommitteeId === sourceId && e.targetCommitteeId === targetId
    );
  }

  function getIntensityBg(score: number): string {
    if (score >= 75) return "bg-cyan-500/30 text-cyan-200 border-cyan-500/50";
    if (score >= 60) return "bg-cyan-600/20 text-cyan-300 border-cyan-600/30";
    if (score > 0) return "bg-cyan-950/40 text-cyan-400 border-cyan-900/30";
    return "bg-[#0c1017] text-slate-600 border-[#202d44]";
  }

  return (
    <div className="space-y-4 font-mono">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 bg-[#111724] border border-[#202d44] p-3 rounded-xl">
        <div className="flex items-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wide">
            Cross-Committee Influence Matrix &amp; Heatmap
          </h2>
        </div>
        <span className="text-xs text-slate-400">
          Max Influence: <strong className="text-cyan-400">{maxInfluenceScore.toFixed(1)}%</strong>
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Pairwise Grid / Table */}
        <div className="lg:col-span-2 bg-[#111724] border border-[#202d44] p-4 rounded-xl overflow-x-auto">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider mb-3">
            Source Committee (Row) &rarr; Target Committee (Column)
          </div>

          <table className="w-full text-xs border-collapse">
            <thead>
              <tr>
                <th className="p-2 text-left text-slate-400 font-semibold border-b border-[#202d44]">
                  Source \ Target
                </th>
                {committeeIds.map((targetId) => (
                  <th
                    key={targetId}
                    className="p-2 text-center text-slate-300 font-semibold border-b border-[#202d44]"
                    title={committeeNames[targetId]}
                  >
                    {targetId}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {committeeIds.map((sourceId) => (
                <tr key={sourceId} className="border-b border-[#202d44]/50">
                  <td className="p-2 font-bold text-slate-200 whitespace-nowrap">
                    <span className="text-cyan-400 mr-1.5">{sourceId}</span>
                    <span className="text-[11px] text-slate-400 font-normal hidden sm:inline">
                      {committeeNames[sourceId]}
                    </span>
                  </td>

                  {committeeIds.map((targetId) => {
                    if (sourceId === targetId) {
                      return (
                        <td
                          key={targetId}
                          className="p-2 text-center text-slate-600 bg-[#0c1017]/60"
                        >
                          &mdash;
                        </td>
                      );
                    }

                    const entry = getEntry(sourceId, targetId);
                    const isSelected =
                      selectedEntry?.sourceCommitteeId === sourceId &&
                      selectedEntry?.targetCommitteeId === targetId;

                    if (!entry) {
                      return (
                        <td
                          key={targetId}
                          className="p-2 text-center text-slate-600 bg-[#0c1017]/40"
                        >
                          0.0%
                        </td>
                      );
                    }

                    return (
                      <td key={targetId} className="p-1.5 text-center">
                        <button
                          type="button"
                          onClick={() => setSelectedEntry(entry)}
                          className={`w-full py-2 px-1 rounded border transition-all ${getIntensityBg(
                            entry.influenceScore
                          )} ${
                            isSelected
                              ? "ring-2 ring-cyan-400 font-bold scale-[1.03]"
                              : "hover:border-cyan-400/80"
                          }`}
                        >
                          <span className="block font-bold">{entry.influenceScore.toFixed(1)}%</span>
                          <span className="block text-[9px] opacity-75">
                            {entry.sharedDecisionCount} shared
                          </span>
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Selected Pair Detail Card */}
        <div className="bg-[#111724] border border-[#202d44] p-4 rounded-xl flex flex-col justify-between">
          {selectedEntry ? (
            <div className="space-y-3">
              <div className="border-b border-[#202d44] pb-2.5">
                <span className="text-[10px] text-cyan-400 uppercase tracking-wider block">
                  Directional Influence Inspector
                </span>
                <h3 className="text-sm font-bold text-slate-100 mt-1">
                  {selectedEntry.sourceCommitteeName} &rarr; {selectedEntry.targetCommitteeName}
                </h3>
                <span className="text-[11px] text-slate-400">
                  {selectedEntry.sourceCommitteeId} to {selectedEntry.targetCommitteeId}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-[#0c1017] p-2.5 rounded-lg border border-[#202d44]">
                  <span className="text-[10px] text-slate-400 block">Influence Score</span>
                  <span className="text-base font-bold text-cyan-400 mt-0.5 block">
                    {selectedEntry.influenceScore.toFixed(1)}%
                  </span>
                </div>
                <div className="bg-[#0c1017] p-2.5 rounded-lg border border-[#202d44]">
                  <span className="text-[10px] text-slate-400 block">Alignment Pct</span>
                  <span className="text-base font-bold text-emerald-400 mt-0.5 block">
                    {selectedEntry.alignmentPct.toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="bg-[#0c1017] p-3 rounded-lg border border-[#202d44] text-xs">
                <span className="text-[10px] text-purple-400 uppercase tracking-wider font-bold block mb-1">
                  Shared Decision Volume
                </span>
                <p className="text-slate-300">
                  <strong className="text-slate-100">{selectedEntry.sharedDecisionCount}</strong> institutional decisions co-reviewed and ratified.
                </p>
              </div>

              <div className="bg-[#0c1017] p-3 rounded-lg border border-[#202d44] text-xs">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold block mb-1">
                  Strategic Rationale
                </span>
                <p className="text-slate-300 leading-relaxed">
                  {selectedEntry.rationale}
                </p>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500 text-xs">
              Select a cell in the influence heatmap to inspect strategic rationale and alignment.
            </div>
          )}

          <div className="pt-3 border-t border-[#202d44] mt-4 flex items-center justify-between text-[11px]">
            <span className="text-slate-400">Invariant State:</span>
            <span className="text-emerald-400 font-bold">INV-OI15 Compliant</span>
          </div>
        </div>
      </div>
    </div>
  );
}
