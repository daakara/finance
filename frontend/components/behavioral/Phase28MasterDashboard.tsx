'use client';

import React, { useState } from 'react';
import ExecutiveStoryHome from './ExecutiveStoryHome';
import MorningBriefingV2 from './MorningBriefingV2';
import BehavioralIntelligenceCenter from './BehavioralIntelligenceCenter';
import AIBehavioralCoachCard from './AIBehavioralCoachCard';
import ConfidenceBandMetric from './ConfidenceBandMetric';
import EnhancedConfidenceBand from './EnhancedConfidenceBand';
import BehavioralMaturityCohortMatrix from './BehavioralMaturityCohortMatrix';
import DIRHeroCard from './DIRHeroCard';
import DIREvolutionTimeline from './DIREvolutionTimeline';
import CohortMigrationDashboard from './CohortMigrationDashboard';
import { CANONICAL_CONFIDENCE_METRICS } from '@/lib/telemetry/statisticalConfidenceEngine';
import { CANONICAL_DIR_RESULT } from '@/lib/telemetry/dirEngine';

export default function Phase28MasterDashboard() {
  const [activeTab, setActiveTab] = useState<
    'dir' | 'home' | 'briefing' | 'center' | 'coach' | 'confidence' | 'migration'
  >('dir');

  return (
    <div className="space-y-8" data-testid="phase28-master-dashboard">
      {/* Top Banner */}
      <div className="p-6 bg-bg-surface border border-border-subtle rounded-2xl shadow-sm space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-caption-mono font-bold uppercase bg-accent-positive/10 text-accent-positive border border-accent-positive/30 rounded">
                Phase 28 Active
              </span>
              <span className="text-caption-mono text-text-muted">
                Decision Improvement Rating (DIR) &bull; Story-First Decision Operating System
              </span>
            </div>
            <h2 className="text-display-1 font-bold text-text-primary mt-1">
              Decision Improvement Operating System
            </h2>
            <p className="text-body-ui text-text-secondary mt-0.5">
              Transforming ARX from evaluating decisions to continuously improving decision-makers through DIR,
              cohort migration velocity, learning velocity, and adaptive behavioral coaching.
            </p>
          </div>

          <div className="text-right">
            <div className="text-caption-mono text-text-muted uppercase text-xs">Decision Improvement Rating</div>
            <div className="text-display-1 font-mono font-extrabold text-accent-positive">
              {CANONICAL_DIR_RESULT.finalDIR} <span className="text-body-ui font-normal text-text-muted">/ 100</span>
            </div>
            <div className="text-caption-mono text-accent-positive font-bold text-xs">
              ▲ +8.0 Annual Growth ({CANONICAL_DIR_RESULT.classification})
            </div>
          </div>
        </div>

        {/* Sub-Navigation Strip */}
        <div className="flex items-center gap-2 border-t border-border-subtle pt-4 overflow-x-auto">
          {[
            { id: 'dir', label: '1. Decision Improvement Rating (DIR 63)' },
            { id: 'migration', label: '2. Cohort Migration (CAR 31%)' },
            { id: 'home', label: '3. Story-First Executive Home' },
            { id: 'briefing', label: '4. Morning Briefing 2.0 (Story Flow)' },
            { id: 'center', label: '5. Behavioral Intelligence Center' },
            { id: 'coach', label: '6. AI Behavioral Coach (Forecast 80)' },
            { id: 'confidence', label: '7. Statistical Confidence & Bands' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`px-4 py-2 text-body-ui font-medium rounded-lg whitespace-nowrap transition-colors ${
                activeTab === tab.id
                  ? 'bg-accent-info/15 text-accent-info border border-accent-info/30 font-semibold'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-surface-raised'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Views */}
      {activeTab === 'dir' && (
        <div className="space-y-6">
          <DIRHeroCard result={CANONICAL_DIR_RESULT} />
          <DIREvolutionTimeline currentScore={CANONICAL_DIR_RESULT.finalDIR} />
        </div>
      )}

      {activeTab === 'migration' && (
        <CohortMigrationDashboard />
      )}

      {activeTab === 'home' && <ExecutiveStoryHome />}

      {activeTab === 'briefing' && <MorningBriefingV2 />}

      {activeTab === 'center' && <BehavioralIntelligenceCenter />}

      {activeTab === 'coach' && <AIBehavioralCoachCard />}

      {activeTab === 'confidence' && (
        <div className="space-y-6" data-testid="statistical-confidence-view">
          <div className="p-6 bg-bg-surface border border-border-subtle rounded-xl space-y-4">
            <h3 className="text-header-1 text-text-primary">
              Statistical Confidence Bands &amp; Trend Velocity
            </h3>
            <p className="text-body-ui text-text-secondary">
              Every executive KPI receives uncertainty bounds calculated via Wilson score confidence intervals (95% CI) with 3-tier color coding and sample size guards.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <EnhancedConfidenceBand
                metric={CANONICAL_CONFIDENCE_METRICS.mentorEngagement}
                sampleSize={42}
              />
              <EnhancedConfidenceBand
                metric={CANONICAL_CONFIDENCE_METRICS.behavioralAdoption}
                sampleSize={112}
              />
              <EnhancedConfidenceBand
                metric={CANONICAL_CONFIDENCE_METRICS.ruleAdherence}
                sampleSize={28}
              />
            </div>
          </div>

          <BehavioralMaturityCohortMatrix />
        </div>
      )}
    </div>
  );
}
