'use client';

import React, { useState } from 'react';
import { ValidationPhase } from '@/types/phase27-observability';
import { VALIDATION_ROADMAP_30_DAY } from '@/lib/telemetry/productionObservabilityEngine';

interface ValidationRoadmap30DayProps {
  phases?: ValidationPhase[];
}

export default function ValidationRoadmap30Day({
  phases = VALIDATION_ROADMAP_30_DAY,
}: ValidationRoadmap30DayProps) {
  const [selectedPhaseNumber, setSelectedPhaseNumber] = useState<number>(2); // Default to active Phase 2

  const selectedPhase = phases.find((p) => p.phaseNumber === selectedPhaseNumber);

  return (
    <div className="space-y-6" data-testid="validation-roadmap-30-day">
      <div>
        <h3 className="text-header-1 text-text-primary">
          30-Day Production Validation Roadmap
        </h3>
        <p className="text-body-ui text-text-secondary mt-1">
          Four structured validation sprints transitioning ARX Terminal from 97% readiness
          to verified 99%+ Production Excellence.
        </p>
      </div>

      {/* 4 Phases Timeline Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {phases.map((phase) => {
          const isSelected = selectedPhaseNumber === phase.phaseNumber;
          const isCompleted = phase.status === 'COMPLETED';
          const isActive = phase.status === 'ACTIVE';

          return (
            <div
              key={phase.phaseNumber}
              onClick={() => setSelectedPhaseNumber(phase.phaseNumber)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSelectedPhaseNumber(phase.phaseNumber);
                }
              }}
              tabIndex={0}
              role="button"
              aria-expanded={isSelected}
              className={`p-5 rounded-xl border transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent-info ${
                isSelected
                  ? 'bg-bg-surface-elevated border-accent-info shadow-md'
                  : 'bg-bg-surface hover:bg-bg-surface-raised border-border-subtle'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-caption-mono font-bold text-accent-info">
                  {phase.daysRange}
                </span>
                <span
                  className={`px-2 py-0.5 text-[10px] font-mono font-bold uppercase rounded ${
                    isCompleted
                      ? 'bg-accent-positive/20 text-accent-positive border border-accent-positive/40'
                      : isActive
                      ? 'bg-accent-info/20 text-accent-info border border-accent-info/40 animate-pulse'
                      : 'bg-bg-surface-raised text-text-muted border border-border-subtle'
                  }`}
                >
                  {phase.status}
                </span>
              </div>

              <h4 className="text-header-2 font-bold text-text-primary mt-2">
                Phase {phase.phaseNumber}
              </h4>
              <p className="text-caption text-text-secondary mt-0.5 font-medium line-clamp-1">
                {phase.name}
              </p>

              {/* Progress bar */}
              <div className="mt-4 space-y-1">
                <div className="flex justify-between text-caption-mono text-text-muted text-[11px]">
                  <span>Progress</span>
                  <span>{phase.completionPct}%</span>
                </div>
                <div className="w-full bg-bg-surface-raised h-2 rounded-full overflow-hidden border border-border-subtle">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isCompleted
                        ? 'bg-accent-positive'
                        : isActive
                        ? 'bg-accent-info'
                        : 'bg-text-muted'
                    }`}
                    style={{ width: `${phase.completionPct}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Detailed Phase Inspection Card */}
      {selectedPhase && (
        <div className="p-6 bg-bg-surface-elevated border border-border-subtle rounded-xl space-y-5 animate-in fade-in duration-150">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border-subtle pb-4">
            <div>
              <div className="flex items-center gap-3">
                <span className="px-2.5 py-1 text-caption-mono font-bold bg-accent-info/10 text-accent-info border border-accent-info/30 rounded">
                  {selectedPhase.daysRange}
                </span>
                <span className="text-caption-mono text-text-muted uppercase">
                  Phase {selectedPhase.phaseNumber} Detail
                </span>
              </div>
              <h4 className="text-header-1 text-text-primary mt-1">
                {selectedPhase.name}
              </h4>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-caption-mono text-text-muted">Status:</span>
              <span className="px-3 py-1 text-caption-mono font-bold uppercase rounded bg-accent-positive/15 text-accent-positive border border-accent-positive/40">
                {selectedPhase.status} ({selectedPhase.completionPct}%)
              </span>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <h5 className="text-body-ui font-semibold text-text-primary">Phase Objective</h5>
              <p className="text-body-ui text-text-secondary mt-1">
                {selectedPhase.goal}
              </p>
            </div>

            <div>
              <h5 className="text-body-ui font-semibold text-text-primary">Key Verification Activities</h5>
              <ul className="mt-2 space-y-2">
                {selectedPhase.activities.map((act, idx) => (
                  <li key={idx} className="flex items-start gap-3 text-caption">
                    <span className="text-accent-positive font-bold mt-0.5">✓</span>
                    <span className="text-text-secondary">{act}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="pt-3 border-t border-border-subtle">
              <h5 className="text-body-ui font-semibold text-text-primary">Gate Exit Criteria</h5>
              <p className="text-caption font-mono text-accent-info mt-1">
                {selectedPhase.successCriteria}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
