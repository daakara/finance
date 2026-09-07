'use client';

import React, { useState } from 'react';
import { MentorContext, LifecycleStep } from '@/types/ux-foundations';
import {
  MOCK_DECISION_PROFILE,
  MOCK_LIFECYCLE_STEPS,
  MOCK_MENTOR_INSIGHTS,
  MOCK_INSIGHT_CARDS,
  MOCK_RECOMMENDATION_CARDS,
  MOCK_LEARNING_CARDS,
  MOCK_EVIDENCE_CARDS,
} from '@/lib/ux-foundations/mockData';
import { UnifiedWorkspaceShell } from '@/components/layout/UnifiedWorkspaceShell';
import { StandardInsightCard } from '@/components/intelligence/StandardInsightCard';
import { StandardRecommendationCard } from '@/components/intelligence/StandardRecommendationCard';
import { StandardLearningCard } from '@/components/intelligence/StandardLearningCard';
import { StandardEvidenceCard } from '@/components/intelligence/StandardEvidenceCard';
import { ProductionReadinessScorecard } from '@/components/audit/ProductionReadinessScorecard';
import { ExecutiveUATDashboard } from '@/components/audit/ExecutiveUATDashboard';
import { ExecutiveExecutionTracker } from '@/components/audit/ExecutiveExecutionTracker';

export const Sprint85Showcase: React.FC = () => {
  const [activeContext, setActiveContext] = useState<MentorContext>('DECISION');
  const [activeTicker, setActiveTicker] = useState<string>('CPRX');
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(3); // Stage 4 Executing
  const [simulatedDevice, setSimulatedDevice] = useState<'desktop' | 'tablet' | 'mobile'>('desktop');
  const [showScorecardModal, setShowScorecardModal] = useState<boolean>(false);

  const currentMentor = MOCK_MENTOR_INSIGHTS[activeContext];

  return (
    <div className="space-y-8">
      {/* Executive Hero Banner */}
      <div className="p-6 bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/40 border border-slate-800 rounded-2xl shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-bold bg-cyan-950 text-cyan-400 border border-cyan-800/80">
                Sprint 8.5 Release
              </span>
              <span className="text-xs font-mono text-emerald-400 font-semibold">
                ✓ Operating Model for Production Readiness
              </span>
            </div>
            <h1 className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              UX Foundations & Operating Architecture
            </h1>
            <p className="text-sm text-slate-300 max-w-3xl leading-relaxed">
              Standardizes the ARX Terminal from a fragmented intelligence engine into an institutional-grade decision platform. Unifies navigation shells, cognitive mentor structures, the 7-stage decision lifecycle state model, and the formal 6-dimension Production UX Audit Scorecard.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="p-3 bg-slate-950 border border-slate-800 rounded-xl text-center min-w-[120px]">
              <div className="text-[10px] uppercase font-mono text-slate-400">Scorecard</div>
              <div className="text-2xl font-bold font-mono text-emerald-400">93.6%</div>
              <div className="text-[9px] font-mono text-emerald-500">PRODUCTION READY</div>
            </div>

            <div className="p-3 bg-slate-950 border border-slate-800 rounded-xl text-center min-w-[120px]">
              <div className="text-[10px] uppercase font-mono text-slate-400">Cognitive Steps</div>
              <div className="text-2xl font-bold font-mono text-cyan-400">5 / 5</div>
              <div className="text-[9px] font-mono text-cyan-500">OBS → LRN FORMAT</div>
            </div>

            <div className="p-3 bg-slate-950 border border-slate-800 rounded-xl text-center min-w-[120px]">
              <div className="text-[10px] uppercase font-mono text-slate-400">Lifecycle States</div>
              <div className="text-2xl font-bold font-mono text-indigo-400">7-Stage</div>
              <div className="text-[9px] font-mono text-indigo-500">IMMUTABLE PROVENANCE</div>
            </div>
          </div>
        </div>

        {/* Interactive Controls Bar */}
        <div className="mt-6 pt-4 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-mono font-medium text-slate-400 mr-1">Workspace Context:</span>
            {(['ATTENTION', 'DECISION', 'ATTRIBUTION', 'LEARNING', 'PLAYBOOK', 'GOVERNANCE'] as MentorContext[]).map((ctx) => (
              <button
                key={ctx}
                type="button"
                onClick={() => setActiveContext(ctx)}
                className={`px-2.5 py-1 text-xs font-mono rounded border transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
                  activeContext === ctx
                    ? 'bg-cyan-600 border-cyan-500 text-white font-bold shadow-sm'
                    : 'bg-slate-800/80 hover:bg-slate-800 text-slate-300 border-slate-700'
                }`}
              >
                {ctx}
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono font-medium text-slate-400">Viewport Simulation:</span>
            {(['desktop', 'tablet', 'mobile'] as const).map((dev) => (
              <button
                key={dev}
                type="button"
                onClick={() => setSimulatedDevice(dev)}
                className={`px-2 py-0.5 text-xs font-mono rounded capitalize transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
                  simulatedDevice === dev
                    ? 'bg-indigo-600 text-white font-bold'
                    : 'bg-slate-800 text-slate-400 hover:text-white'
                }`}
              >
                {dev}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 1. Live Unified Workspace Shell Demonstration */}
      <section aria-labelledby="section-unified-shell" className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="h-2 w-2 rounded-full bg-cyan-400" />
            <h2 id="section-unified-shell" className="text-lg font-bold text-white">
              1. Unified Workspace Architecture & Live Shell (Epic UXF-100)
            </h2>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono text-slate-400">Advance Lifecycle Step:</span>
            <button
              type="button"
              onClick={() => setCurrentStepIndex(Math.max(0, currentStepIndex - 1))}
              disabled={currentStepIndex === 0}
              className="px-2 py-0.5 text-xs font-mono rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-white"
            >
              ◀ Prev
            </button>
            <span className="text-xs font-mono font-bold text-cyan-400 px-1.5">
              Step {currentStepIndex + 1}/7
            </span>
            <button
              type="button"
              onClick={() => setCurrentStepIndex(Math.min(MOCK_LIFECYCLE_STEPS.length - 1, currentStepIndex + 1))}
              disabled={currentStepIndex === MOCK_LIFECYCLE_STEPS.length - 1}
              className="px-2 py-0.5 text-xs font-mono rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-white"
            >
              Next ▶
            </button>
          </div>
        </div>

        {/* Viewport Frame */}
        <div
          className={`mx-auto border border-slate-800 rounded-2xl overflow-hidden shadow-2xl transition-all duration-300 ${
            simulatedDevice === 'desktop'
              ? 'w-full'
              : simulatedDevice === 'tablet'
              ? 'max-w-[834px]'
              : 'max-w-[390px]'
          }`}
        >
          <UnifiedWorkspaceShell
            activeContext={activeContext}
            onContextChange={setActiveContext}
            activeTicker={activeTicker}
            onTickerChange={setActiveTicker}
            profile={MOCK_DECISION_PROFILE}
            steps={MOCK_LIFECYCLE_STEPS}
            currentStepIndex={currentStepIndex}
            mentorInsight={currentMentor}
          >
            {/* Embedded Active Canvas View */}
            <div className="space-y-4">
              <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg flex items-center justify-between">
                <div>
                  <span className="text-[10px] uppercase font-mono text-slate-400 block">
                    Active Canvas Screen
                  </span>
                  <div className="text-sm font-bold text-white">
                    {activeContext} WORKSPACE :: {activeTicker}
                  </div>
                </div>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
                  Ready for Execution
                </span>
              </div>

              {/* Sample Injected Intelligence Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <StandardInsightCard insight={MOCK_INSIGHT_CARDS[0]} />
                <StandardRecommendationCard recommendation={MOCK_RECOMMENDATION_CARDS[0]} />
              </div>
            </div>
          </UnifiedWorkspaceShell>
        </div>
      </section>

      {/* 2. Intelligence Design System Component Library */}
      <section aria-labelledby="section-card-library" className="space-y-4">
        <div className="flex items-center space-x-2">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          <h2 id="section-card-library" className="text-lg font-bold text-white">
            2. Standard Intelligence Design System Component Library (Epic UXF-500)
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Card 1: Insight */}
          <div className="space-y-2">
            <span className="text-xs font-mono text-slate-400 font-semibold">
              StandardInsightCard (&ldquo;What happened?&rdquo;)
            </span>
            <StandardInsightCard insight={MOCK_INSIGHT_CARDS[0]} />
          </div>

          {/* Card 2: Recommendation */}
          <div className="space-y-2">
            <span className="text-xs font-mono text-slate-400 font-semibold">
              StandardRecommendationCard (&ldquo;What should I do?&rdquo;)
            </span>
            <StandardRecommendationCard recommendation={MOCK_RECOMMENDATION_CARDS[1]} />
          </div>

          {/* Card 3: Learning */}
          <div className="space-y-2">
            <span className="text-xs font-mono text-slate-400 font-semibold">
              StandardLearningCard (&ldquo;What did we learn?&rdquo;)
            </span>
            <StandardLearningCard learning={MOCK_LEARNING_CARDS[0]} />
          </div>

          {/* Card 4: Evidence */}
          <div className="space-y-2">
            <span className="text-xs font-mono text-slate-400 font-semibold">
              StandardEvidenceCard (&ldquo;Why should I trust this?&rdquo;)
            </span>
            <StandardEvidenceCard evidence={MOCK_EVIDENCE_CARDS[0]} />
          </div>
        </div>
      </section>

      {/* 3. Formal Production UX Audit & Readiness Scorecard */}
      <section aria-labelledby="section-scorecard" className="space-y-4">
        <div className="flex items-center space-x-2">
          <span className="h-2 w-2 rounded-full bg-indigo-400" />
          <h2 id="section-scorecard" className="text-lg font-bold text-white">
            3. Production UX Audit & Readiness Scorecard (Epic UXF-600)
          </h2>
        </div>

        <ProductionReadinessScorecard />
      </section>

      {/* 4. Executive UAT Test Pack & Institutional Certification */}
      <section aria-labelledby="section-executive-uat" className="space-y-4">
        <div className="flex items-center space-x-2">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          <h2 id="section-executive-uat" className="text-lg font-bold text-white">
            4. Executive UAT Test Pack & Institutional Release Certification (97.0% Score)
          </h2>
        </div>

        <ExecutiveUATDashboard />
      </section>

      {/* 5. Executive Execution Tracker & User Outcome Telemetry */}
      <section aria-labelledby="section-execution-tracker" className="space-y-4">
        <div className="flex items-center space-x-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400" />
          <h2 id="section-execution-tracker" className="text-lg font-bold text-white">
            5. Executive Execution Tracker & User Outcome Telemetry Framework (95% Release Candidate)
          </h2>
        </div>

        <ExecutiveExecutionTracker />
      </section>
    </div>
  );
};
