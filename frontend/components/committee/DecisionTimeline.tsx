"use client";

import { DecisionTimelineStep } from "../../types/committee-intelligence";

export interface DecisionTimelineProps {
  steps: DecisionTimelineStep[];
  decisionId: string;
}

export default function DecisionTimeline({ steps, decisionId }: DecisionTimelineProps) {
  return (
    <div className="bg-[#111723] border border-[#1f2c42] rounded-xl p-5 shadow-lg shadow-black/40">
      <div className="flex items-center justify-between pb-4 mb-4 border-b border-[#1f2c42]">
        <div>
          <h3 className="text-sm font-semibold font-mono text-slate-100 flex items-center space-x-2">
            <span>AUDIT TRAIL TIMELINE</span>
            <span className="px-2 py-0.5 rounded bg-[#1c273a] text-cyan-400 text-xs">
              {decisionId}
            </span>
          </h3>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            INV-OI13 Full-Chain Traceability (100% Verified)
          </p>
        </div>
        <div className="text-right font-mono text-xs">
          <span className="text-slate-400">Steps Recovered: </span>
          <span className="text-emerald-400 font-bold">{steps.length} / 7</span>
        </div>
      </div>

      <div className="relative border-l-2 border-[#24344d] ml-3 md:ml-4 space-y-6 pl-4 md:pl-6 my-2">
        {steps.map((step, idx) => {
          const isPassed = step.status === "COMPLETED";
          const isWarning = step.status === "WARNING";

          return (
            <div key={step.stepId || idx} className="relative group">
              {/* Bullet Node */}
              <div
                className={`absolute -left-[23px] md:-left-[31px] top-1 w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all ${
                  isPassed
                    ? "bg-emerald-950 border-emerald-400 text-emerald-400"
                    : isWarning
                    ? "bg-amber-950 border-amber-400 text-amber-400"
                    : "bg-rose-950 border-rose-400 text-rose-400"
                }`}
              >
                <div
                  className={`w-1.5 h-1.5 rounded-full ${
                    isPassed ? "bg-emerald-400" : isWarning ? "bg-amber-400" : "bg-rose-400"
                  }`}
                />
              </div>

              {/* Step Content Card */}
              <div className="bg-[#0c111a] border border-[#1d293d] rounded-lg p-3 hover:border-cyan-500/30 transition-colors">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-1 font-mono text-xs">
                  <div className="flex items-center space-x-2">
                    <span className="px-1.5 py-0.5 rounded bg-[#182334] text-[10px] text-cyan-400 font-bold uppercase">
                      {step.stepName}
                    </span>
                    <span className="font-semibold text-slate-200 text-xs">
                      {step.title}
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-400">
                    {new Date(step.timestampUtc).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} UTC
                  </span>
                </div>

                <p className="text-xs text-slate-400 font-mono mt-1 leading-relaxed">
                  {step.details}
                </p>

                {/* Meta details / IDs */}
                {(step.actorId || step.artifactId) && (
                  <div className="mt-2 pt-2 border-t border-[#172030] flex flex-wrap items-center gap-2 font-mono text-[10px] text-slate-400">
                    {step.actorId && (
                      <span className="bg-[#121927] px-2 py-0.5 rounded border border-[#202d44]">
                        Actor: <span className="text-slate-200">{step.actorId}</span>
                        {step.actorRole && ` (${step.actorRole})`}
                      </span>
                    )}
                    {step.artifactId && (
                      <span className="bg-[#121927] px-2 py-0.5 rounded border border-[#202d44]">
                        Artifact: <span className="text-cyan-400">{step.artifactId}</span>
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
