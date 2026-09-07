'use client';

import React from 'react';
import { DecisionLifecycleState, LifecycleStep } from '@/types/ux-foundations';

interface DecisionLifecycleTimelineProps {
  steps: LifecycleStep[];
  currentStepIndex: number;
  onStepClick?: (step: LifecycleStep, index: number) => void;
  interactive?: boolean;
}

const STEP_DEFINITIONS: Array<{ state: DecisionLifecycleState; label: string; shortCode: string }> = [
  { state: 'OBSERVED', label: '1. Observed', shortCode: 'OBS' },
  { state: 'PREDICTED', label: '2. Predicted', shortCode: 'PRD' },
  { state: 'APPROVED', label: '3. Approved', shortCode: 'APP' },
  { state: 'EXECUTING', label: '4. Executing', shortCode: 'EXE' },
  { state: 'RESOLVED', label: '5. Resolved', shortCode: 'RES' },
  { state: 'LEARNED', label: '6. Learned', shortCode: 'LRN' },
  { state: 'PLAYBOOK_UPDATED', label: '7. Playbook Updated', shortCode: 'PBK' },
];

export const DecisionLifecycleTimeline: React.FC<DecisionLifecycleTimelineProps> = ({
  steps,
  currentStepIndex,
  onStepClick,
  interactive = true,
}) => {
  return (
    <nav
      aria-label="Decision Lifecycle Progress"
      className="w-full bg-slate-900/90 border border-slate-800 rounded-lg p-3 backdrop-blur-sm"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Decision Lifecycle State Model (7-Stage)
          </span>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          Stage {currentStepIndex + 1} of {STEP_DEFINITIONS.length}:{' '}
          <strong className="text-white">
            {STEP_DEFINITIONS[currentStepIndex]?.label || 'Active'}
          </strong>
        </span>
      </div>

      {/* Stepper Container */}
      <ol className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-1.5 list-none p-0 m-0">
        {STEP_DEFINITIONS.map((def, idx) => {
          const stepData = steps[idx];
          const isCompleted = idx < currentStepIndex;
          const isActive = idx === currentStepIndex;
          const isPending = idx > currentStepIndex;

          return (
            <li key={def.state} className="relative">
              <button
                type="button"
                disabled={!interactive}
                onClick={() => onStepClick && stepData && onStepClick(stepData, idx)}
                aria-current={isActive ? 'step' : undefined}
                className={`w-full text-left p-2 rounded-md border transition-all duration-150 group focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-1 focus:ring-offset-slate-900 ${
                  isActive
                    ? 'bg-cyan-950/40 border-cyan-500 text-cyan-200 shadow-sm shadow-cyan-500/20'
                    : isCompleted
                    ? 'bg-slate-800/60 border-emerald-900/60 text-slate-300 hover:bg-slate-800'
                    : 'bg-slate-900/40 border-slate-800/80 text-slate-500 opacity-75'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span
                    className={`inline-flex items-center justify-center text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                      isActive
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : isCompleted
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                        : 'bg-slate-800 text-slate-400 border border-slate-700'
                    }`}
                  >
                    {isCompleted ? '✓' : def.shortCode}
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">
                    {stepData?.timestamp ? stepData.timestamp.split(' ')[1] || stepData.timestamp : `0${idx + 1}`}
                  </span>
                </div>

                <div className="text-xs font-medium truncate text-slate-200 group-hover:text-white">
                  {def.label.split('. ')[1]}
                </div>

                <div className="text-[10px] text-slate-400 truncate mt-0.5">
                  {isActive ? 'In Progress' : isCompleted ? 'Verified' : 'Pending'}
                </div>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
};
