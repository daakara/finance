"use client";

import React, { useState, useEffect } from "react";

export interface TaskPromptProps {
  taskId: string;
  taskTitle: string;
  taskInstructions: string;
  maxDurationSec: number;
  onTaskSubmit: (result: {
    participantStatedState: string;
    participantProposedAction: string;
    participantExplanation: string;
    confidenceRating: number;
  }) => void;
  onTimeout: () => void;
}

export function TaskPrompter({
  taskId,
  taskTitle,
  taskInstructions,
  maxDurationSec,
  onTaskSubmit,
  onTimeout,
}: TaskPromptProps) {
  const [elapsedSec, setElapsedSec] = useState(0);
  const [statedDecision, setStatedDecision] = useState("");
  const [explanation, setExplanation] = useState("");
  const [confidence, setConfidence] = useState<number>(3);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    setElapsedSec(0);
    const interval = setInterval(() => {
      setElapsedSec((prev) => {
        const next = prev + 1;
        if (next >= maxDurationSec) {
          clearInterval(interval);
          onTimeout();
        }
        return next;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [taskId, maxDurationSec, onTimeout]);

  const remainingSec = Math.max(0, maxDurationSec - elapsedSec);
  const pctRemaining = (remainingSec / maxDurationSec) * 100;
  const isUrgent = remainingSec <= 30;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!statedDecision.trim()) return;
    setIsSubmitting(true);
    onTaskSubmit({
      participantStatedState: statedDecision,
      participantProposedAction: statedDecision,
      participantExplanation: explanation,
      confidenceRating: confidence,
    });
    setStatedDecision("");
    setExplanation("");
    setIsSubmitting(false);
  };

  return (
    <aside
      aria-label="Research Task Evaluation Panel"
      className="bg-slate-900 border border-slate-800 rounded-lg p-4 shadow-xl flex flex-col gap-4 text-slate-100"
    >
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-emerald-400 bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded">
            TASK {taskId}
          </span>
          <h2 className="text-sm font-semibold mt-1 text-slate-200">{taskTitle}</h2>
        </div>
        <div className="text-right font-mono">
          <div className="text-xs text-slate-400">TIME REMAINING</div>
          <div className={`text-base font-bold tabular-nums ${isUrgent ? "text-rose-400 animate-pulse" : "text-slate-200"}`}>
            {Math.floor(remainingSec / 60)}:{(remainingSec % 60).toString().padStart(2, "0")}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden" role="progressbar" aria-valuenow={pctRemaining} aria-valuemin={0} aria-valuemax={100}>
        <div
          className={`h-full transition-all duration-1000 ${isUrgent ? "bg-rose-500" : "bg-emerald-500"}`}
          style={{ width: `${pctRemaining}%` }}
        />
      </div>

      <div className="bg-slate-950/80 border border-slate-800 rounded p-3 text-xs text-slate-300 leading-relaxed">
        <span className="font-semibold text-slate-100 block mb-1">TASK DIRECTIVE:</span>
        {taskInstructions}
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <div>
          <label htmlFor="statedDecision" className="block text-xs font-medium text-slate-300 mb-1">
            Your Decision / Proposed Action:
          </label>
          <input
            id="statedDecision"
            type="text"
            required
            value={statedDecision}
            onChange={(e) => setStatedDecision(e.target.value)}
            placeholder="e.g. Approve 1,492 shares at $6.83 stop / Reject due to stale quote"
            className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 font-mono"
          />
        </div>

        <div>
          <label htmlFor="explanation" className="block text-xs font-medium text-slate-300 mb-1">
            Reasoning / Evidence Observed:
          </label>
          <textarea
            id="explanation"
            rows={2}
            value={explanation}
            onChange={(e) => setExplanation(e.target.value)}
            placeholder="Cite observed price levels, risks, dissent, or data state..."
            className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 text-xs"
          />
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-medium text-slate-300">Decision Confidence:</span>
            <span className="text-xs font-mono text-emerald-400 font-bold">{confidence} / 5</span>
          </div>
          <div className="flex gap-2" role="radiogroup" aria-label="Confidence rating from 1 to 5">
            {[1, 2, 3, 4, 5].map((lvl) => (
              <button
                key={lvl}
                type="button"
                onClick={() => setConfidence(lvl)}
                className={`flex-1 py-1 rounded text-xs font-mono font-bold border transition-colors ${
                  confidence === lvl
                    ? "bg-emerald-600 border-emerald-500 text-white"
                    : "bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800"
                }`}
                aria-checked={confidence === lvl}
                role="radio"
              >
                {lvl}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={!statedDecision.trim() || isSubmitting}
          className="mt-1 w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-semibold py-2 px-4 rounded text-xs transition-colors shadow focus:outline-none focus:ring-2 focus:ring-emerald-400"
        >
          Submit Decision & Continue
        </button>
      </form>
    </aside>
  );
}
