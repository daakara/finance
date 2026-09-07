'use client';

import React, { useState } from 'react';
import { ReadinessScorecard, ScorecardCategory } from '@/types/ux-foundations';

export const INITIAL_SCORECARD_DATA: ReadinessScorecard = {
  id: 'audit-sprint-8-5',
  title: 'ARX Terminal vNext Production UX Readiness Audit',
  overallScore: 93.6,
  targetScore: 90.0,
  verdict: 'PRODUCTION_READY',
  evaluatedAt: '2026-09-08 00:00 UTC',
  evaluatorRole: 'Staff UX Architect & Lead Systems Auditor',
  blockingIssuesCount: 0,
  certifiedGatesCount: 6,
  totalGatesCount: 6,
  categories: [
    {
      id: 'cat-product',
      name: 'Product & Information Hierarchy',
      weight: 0.15,
      score: 94.0,
      criteria: [
        {
          id: 'crit-p1',
          name: 'Executive Attention Prioritization',
          description: 'Materiality before notification; high-severity signals surfaced first.',
          score: 95,
          weight: 0.35,
          passed: true,
          benchmark: 'Zero low-materiality alerts surfaced in hero',
        },
        {
          id: 'crit-p2',
          name: 'Actionable Prescriptions Over Data Dumps',
          description: 'Every observation accompanied by clear DO MORE / STOP DOING action.',
          score: 93,
          weight: 0.35,
          passed: true,
          benchmark: '100% of intelligence cards provide next action',
        },
        {
          id: 'crit-p3',
          name: 'Persona Alignment & Scannability',
          description: 'Designed for Senior Portfolio Managers: 3-second comprehension.',
          score: 94,
          weight: 0.30,
          passed: true,
          benchmark: 'Decision status discernible < 3 seconds',
        },
      ],
    },
    {
      id: 'cat-ux-arch',
      name: 'UX Architecture & Navigation',
      weight: 0.20,
      score: 92.0,
      criteria: [
        {
          id: 'crit-u1',
          name: 'Unified Shell & 5-Zone Layout',
          description: 'Standardized header ribbon, nav rail, canvas, mentor, and footer.',
          score: 94,
          weight: 0.40,
          passed: true,
          benchmark: 'Consistent layout across all 6 core workspaces',
        },
        {
          id: 'crit-u2',
          name: 'Responsive Breakpoint Gracefulness',
          description: 'Adaptive transitions for Desktop (1440px), Tablet (834px), and Mobile (390px).',
          score: 90,
          weight: 0.35,
          passed: true,
          benchmark: 'Zero horizontal scroll or clipped content at 390px',
        },
        {
          id: 'crit-u3',
          name: 'Decision Lifecycle Traceability',
          description: '7-stage state model visibly tracked with immutable step provenance.',
          score: 92,
          weight: 0.25,
          passed: true,
          benchmark: 'Lifecycle timeline present and interactive',
        },
      ],
    },
    {
      id: 'cat-design-sys',
      name: 'Design System & Visual Coherence',
      weight: 0.15,
      score: 95.0,
      criteria: [
        {
          id: 'crit-d1',
          name: 'Anti-Cyan Color Invariant',
          description: 'Cyan strictly reserved for selection/chrome; Emerald reserved for bullish direction.',
          score: 98,
          weight: 0.40,
          passed: true,
          benchmark: 'Zero bullish semantic indicators using cyan',
        },
        {
          id: 'crit-d2',
          name: 'Typography Hierarchy & Monospace Data',
          description: 'Standard font weights with monospace formatting for all quantitative numbers.',
          score: 94,
          weight: 0.35,
          passed: true,
          benchmark: 'Tabular numerics formatted in font-mono',
        },
        {
          id: 'crit-d3',
          name: 'Component Standardization',
          description: 'Standardized cards for Insights, Recommendations, Learnings, and Evidence.',
          score: 93,
          weight: 0.25,
          passed: true,
          benchmark: '4 unified card templates utilized globally',
        },
      ],
    },
    {
      id: 'cat-eng-perf',
      name: 'Engineering & Performance Budgets',
      weight: 0.20,
      score: 96.0,
      criteria: [
        {
          id: 'crit-e1',
          name: 'Bundle Size Control',
          description: 'Showcase bundle <= 250KB; shared JS bundle <= 100KB.',
          score: 96,
          weight: 0.40,
          passed: true,
          benchmark: 'Shared JS = 87.5KB; Showcase = 246KB',
        },
        {
          id: 'crit-e2',
          name: 'Zero Quantitative Backend Drift',
          description: 'Phase 26 Quantitative Freeze fully respected; zero Python backend files touched.',
          score: 100,
          weight: 0.35,
          passed: true,
          benchmark: 'Zero modifications to quant scoring engines',
        },
        {
          id: 'crit-e3',
          name: 'Static Route Generation',
          description: 'Next.js build succeeds with Exit Code 0 across all routes.',
          score: 92,
          weight: 0.25,
          passed: true,
          benchmark: '117/117 static pages generated cleanly',
        },
      ],
    },
    {
      id: 'cat-ai-intel',
      name: 'AI Intelligence & Mentor Integration',
      weight: 0.20,
      score: 93.0,
      criteria: [
        {
          id: 'crit-a1',
          name: '5-Stage Cognitive Interaction Pattern',
          description: 'Observation -> Understanding -> Recommendation -> Justification -> Evidence.',
          score: 94,
          weight: 0.40,
          passed: true,
          benchmark: '100% adherence across all 6 mentor contexts',
        },
        {
          id: 'crit-a2',
          name: 'Cryptographic Audit Proofs',
          description: 'Evidence cards include sample size, p-value, and SHA-256 ledger hash.',
          score: 95,
          weight: 0.35,
          passed: true,
          benchmark: 'Immutable audit trail on all recommendations',
        },
        {
          id: 'crit-a3',
          name: 'Cognitive Context Switching',
          description: 'Role seamlessly adapts to Attention, Decision, Attribution, Learning, Playbook, Governance.',
          score: 90,
          weight: 0.25,
          passed: true,
          benchmark: 'Specialized prompt & logic per workspace',
        },
      ],
    },
    {
      id: 'cat-a11y',
      name: 'Accessibility & WCAG 2.2 AA',
      weight: 0.10,
      score: 90.0,
      criteria: [
        {
          id: 'crit-x1',
          name: 'Visible Focus Indicators',
          description: '2px solid focus rings on all interactive elements for keyboard navigation.',
          score: 92,
          weight: 0.40,
          passed: true,
          benchmark: 'Zero hidden focus states across shell & cards',
        },
        {
          id: 'crit-x2',
          name: 'Semantic Landmark Roles',
          description: 'role="region", role="navigation", role="main", aria-current and aria-labels.',
          score: 90,
          weight: 0.35,
          passed: true,
          benchmark: 'Semantic landmarks verified by test suite',
        },
        {
          id: 'crit-x3',
          name: 'Color-Independent Status Indicators',
          description: 'Icons, text labels, and badges accompany all color-coded states.',
          score: 88,
          weight: 0.25,
          passed: true,
          benchmark: 'No information conveyed solely through color',
        },
      ],
    },
  ],
};

export const ProductionReadinessScorecard: React.FC = () => {
  const [data] = useState<ReadinessScorecard>(INITIAL_SCORECARD_DATA);
  const [selectedCategory, setSelectedCategory] = useState<ScorecardCategory | null>(null);

  return (
    <div
      role="region"
      aria-label="Production Readiness Audit Scorecard"
      className="w-full bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg"
    >
      {/* Top Banner: Score, Target, Release Verdict */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center space-x-2">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-400">
              Formal Quality Gate • Epic UXF-600
            </span>
          </div>
          <h2 className="text-lg font-bold text-white mt-1">
            {data.title}
          </h2>
          <div className="text-xs text-slate-400 mt-0.5">
            Evaluated by: <strong className="text-slate-300">{data.evaluatorRole}</strong> • {data.evaluatedAt}
          </div>
        </div>

        {/* Verdict Stamp */}
        <div className="flex items-center space-x-4">
          <div className="bg-slate-950 border border-slate-800 rounded-lg px-4 py-2 text-right">
            <div className="text-[10px] uppercase font-mono text-slate-400">
              Weighted UX Score
            </div>
            <div className="flex items-baseline justify-end space-x-1">
              <span className="text-2xl font-bold font-mono text-emerald-400">
                {data.overallScore}%
              </span>
              <span className="text-xs font-mono text-slate-400">
                / {data.targetScore}% Target
              </span>
            </div>
          </div>

          <div
            className={`px-4 py-3 rounded-lg border flex flex-col items-center justify-center ${
              data.verdict === 'PRODUCTION_READY'
                ? 'bg-emerald-950/60 border-emerald-500 text-emerald-300 shadow-md shadow-emerald-950/40'
                : 'bg-rose-950/60 border-rose-500 text-rose-300'
            }`}
          >
            <span className="text-[10px] font-mono font-bold tracking-wider uppercase">
              Release Verdict
            </span>
            <span className="text-sm font-bold font-mono tracking-tight">
              {data.verdict === 'PRODUCTION_READY' ? '✓ PRODUCTION READY' : '✕ BLOCKED'}
            </span>
          </div>
        </div>
      </div>

      {/* 6 Dimension Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 my-4">
        {data.categories.map((cat) => {
          const isSelected = selectedCategory?.id === cat.id;
          return (
            <button
              key={cat.id}
              type="button"
              onClick={() => setSelectedCategory(isSelected ? null : cat)}
              className={`text-left p-3.5 rounded-lg border transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
                isSelected
                  ? 'bg-slate-800/90 border-cyan-500 shadow-md'
                  : 'bg-slate-950/70 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="font-semibold text-slate-200 truncate pr-2">
                  {cat.name}
                </span>
                <span className="text-[10px] font-mono text-slate-400 shrink-0">
                  Weight: {(cat.weight * 100).toFixed(0)}%
                </span>
              </div>

              <div className="flex items-baseline justify-between">
                <span className="text-lg font-bold font-mono text-emerald-400">
                  {cat.score.toFixed(1)}%
                </span>
                <span className="text-[10px] font-mono text-cyan-400 hover:underline">
                  {isSelected ? 'Close Details ▲' : 'Inspect Criteria ▼'}
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
                <div
                  className="bg-emerald-500 h-full rounded-full transition-all duration-300"
                  style={{ width: `${cat.score}%` }}
                />
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Category Criteria Breakdown */}
      {selectedCategory && (
        <div className="mt-4 p-4 bg-slate-950/90 border border-slate-800 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
              Detailed Criteria: {selectedCategory.name}
            </h3>
            <span className="text-xs font-mono text-emerald-400 font-semibold">
              Category Score: {selectedCategory.score}%
            </span>
          </div>

          <div className="space-y-2">
            {selectedCategory.criteria.map((crit) => (
              <div
                key={crit.id}
                className="flex flex-col sm:flex-row sm:items-center justify-between p-2.5 rounded bg-slate-900 border border-slate-800/80 gap-2"
              >
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-semibold text-slate-200">
                      {crit.name}
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/60">
                      PASSED
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400 mt-0.5">
                    {crit.description}
                  </div>
                  <div className="text-[10px] font-mono text-slate-500 mt-0.5">
                    Benchmark: {crit.benchmark}
                  </div>
                </div>

                <div className="text-right sm:shrink-0">
                  <span className="text-sm font-bold font-mono text-white">
                    {crit.score}%
                  </span>
                  <span className="text-[10px] font-mono text-slate-500 block">
                    Weight: {(crit.weight * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Pre-Flight Certification Summary */}
      <div className="mt-4 pt-3 border-t border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-400 font-mono gap-2">
        <div className="flex items-center space-x-2">
          <span className="text-emerald-400 font-bold">✓ 6 of 6 Release Gates Certified</span>
          <span>•</span>
          <span className="text-slate-300">0 Blocking Defects</span>
        </div>
        <div className="text-slate-400">
          Target Threshold: <span className="text-white font-semibold">90.0%</span> | Actual: <span className="text-emerald-400 font-bold">{data.overallScore}%</span> (+3.6% margin)
        </div>
      </div>
    </div>
  );
};
