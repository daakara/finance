"use client";

import React, { useState } from "react";

export interface TLXScores {
  mental_demand: number;
  physical_demand: number;
  temporal_demand: number;
  performance: number;
  effort: number;
  frustration: number;
}

interface NASAWorkloadModalProps {
  isOpen: boolean;
  blockName: string;
  variantId: string;
  onSubmit: (scores: TLXScores) => void;
}

export function NASAWorkloadModal({ isOpen, blockName, variantId, onSubmit }: NASAWorkloadModalProps) {
  const [scores, setScores] = useState<TLXScores>({
    mental_demand: 5,
    physical_demand: 1,
    temporal_demand: 5,
    performance: 7,
    effort: 5,
    frustration: 2,
  });

  if (!isOpen) return null;

  const handleChange = (dimension: keyof TLXScores, value: number) => {
    setScores((prev) => ({ ...prev, [dimension]: value }));
  };

  const dimensions: { key: keyof TLXScores; label: string; desc: string; low: string; high: string }[] = [
    {
      key: "mental_demand",
      label: "Mental Demand",
      desc: "How mentally demanding was the task (thinking, deciding, calculating)?",
      low: "Very Low",
      high: "Very High",
    },
    {
      key: "physical_demand",
      label: "Physical Demand",
      desc: "How physically demanding was the interaction (clicking, scanning, typing)?",
      low: "Very Low",
      high: "Very High",
    },
    {
      key: "temporal_demand",
      label: "Temporal Demand",
      desc: "How hurried or rushed was the pace of the task?",
      low: "Very Low",
      high: "Very High",
    },
    {
      key: "performance",
      label: "Performance",
      desc: "How successful were you in accomplishing the task directives?",
      low: "Failure",
      high: "Perfect",
    },
    {
      key: "effort",
      label: "Effort",
      desc: "How hard did you have to work to accomplish your level of performance?",
      low: "Very Low",
      high: "Very High",
    },
    {
      key: "frustration",
      label: "Frustration",
      desc: "How insecure, discouraged, irritated, or annoyed were you?",
      low: "Very Low",
      high: "Very High",
    },
  ];

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
      <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-xl w-full p-6 shadow-2xl flex flex-col gap-5">
        <div>
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold bg-sky-950 text-sky-400 border border-sky-800 px-2.5 py-0.5 rounded">
              NASA-TLX WORKLOAD ASSESSMENT
            </span>
            <span className="text-xs font-mono text-slate-400">
              {blockName} • Variant: {variantId}
            </span>
          </div>
          <h2 className="text-lg font-bold text-slate-100 mt-2">
            Subjective Cognitive Workload Rating
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Please evaluate your cognitive workload experience for the trials just completed. Rate each dimension on a scale of 1 (Lowest) to 10 (Highest).
          </p>
        </div>

        <div className="flex flex-col gap-4 max-h-[60vh] overflow-y-auto pr-2">
          {dimensions.map((dim) => (
            <div key={dim.key} className="bg-slate-950 p-3.5 rounded border border-slate-800 flex flex-col gap-2">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-slate-200">{dim.label}</span>
                <span className="text-xs font-mono font-bold text-sky-400 bg-sky-950/60 px-2 py-0.5 rounded">
                  {scores[dim.key]} / 10
                </span>
              </div>
              <p className="text-[11px] text-slate-400">{dim.desc}</p>

              <div className="flex items-center gap-3 pt-1">
                <span className="text-[10px] text-slate-500 font-mono w-14 text-left">{dim.low}</span>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={scores[dim.key]}
                  onChange={(e) => handleChange(dim.key, parseInt(e.target.value, 10))}
                  className="w-full accent-sky-500 cursor-pointer"
                />
                <span className="text-[10px] text-slate-500 font-mono w-14 text-right">{dim.high}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="pt-3 border-t border-slate-800 flex justify-end">
          <button
            type="button"
            onClick={() => onSubmit(scores)}
            className="bg-sky-600 hover:bg-sky-500 text-white font-mono font-bold text-xs py-2.5 px-6 rounded shadow transition-colors"
          >
            Submit Workload Assessment & Continue
          </button>
        </div>
      </div>
    </div>
  );
}
