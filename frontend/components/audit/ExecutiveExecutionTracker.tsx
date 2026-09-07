'use client';

import React, { useState } from 'react';
import { ExecutiveTrackerMetrics, ReleaseGateItem, UserOutcomeTelemetryEvent } from '@/types/user-outcome-telemetry';
import { userOutcomeTelemetry } from '@/telemetry/userOutcomeTelemetry';

export const ExecutiveExecutionTracker: React.FC = () => {
  const [metrics] = useState<ExecutiveTrackerMetrics>(userOutcomeTelemetry.getExecutiveTrackerMetrics());
  const [gates] = useState<ReleaseGateItem[]>(userOutcomeTelemetry.getReleaseGates());
  const [recentEvents, setRecentEvents] = useState<UserOutcomeTelemetryEvent[]>(userOutcomeTelemetry.getBuffer());
  const [lastDispatched, setLastDispatched] = useState<string>('Ready for simulation');

  // Interactive Live Telemetry Simulator
  const triggerEvent = (type: 'view' | 'click' | 'evidence' | 'rule' | 'drift') => {
    let evt: UserOutcomeTelemetryEvent;
    if (type === 'view') {
      evt = userOutcomeTelemetry.trackMentorViewed({
        screen: 'learning_center',
        mentor_type: 'learning',
        user_id: 'pm-inst-042',
        timestamp: new Date().toISOString(),
      });
      setLastDispatched('Dispatched mentor_viewed (SEEN)');
    } else if (type === 'click') {
      evt = userOutcomeTelemetry.trackMentorRecommendationClicked({
        recommendation_type: 'DO_MORE',
        confidence: 91,
        projected_impact: '+3.4 pts',
        ticker: 'CPRX',
      });
      setLastDispatched('Dispatched mentor_recommendation_clicked (UNDERSTOOD)');
    } else if (type === 'evidence') {
      evt = userOutcomeTelemetry.trackMentorEvidenceOpened({
        sample_size: 124,
        p_value: 0.0004,
        ledger_hash: '8f9e0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f',
      });
      setLastDispatched('Dispatched mentor_evidence_opened (UNDERSTOOD - Trust Validation)');
    } else if (type === 'rule') {
      evt = userOutcomeTelemetry.trackRuleFollowed({
        rule_id: 'rule_12',
        rule_type: 'DO_MORE',
        action_detected: 'Stage 2 breakout entered with >2.0σ flow',
        adherence_rate: 87,
      });
      setLastDispatched('Dispatched rule_followed (BEHAVIOR_IMPROVED)');
    } else {
      evt = userOutcomeTelemetry.trackDriftWarningAcknowledged({
        drift_score: 21,
        acknowledged_at: new Date().toISOString(),
        action_selected: 'RE_CALIBRATE',
      });
      setLastDispatched('Dispatched drift_warning_acknowledged (ACTED_UPON)');
    }
    setRecentEvents(userOutcomeTelemetry.getBuffer());
  };

  return (
    <div
      role="region"
      aria-label="Executive Execution Tracker"
      className="w-full bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-2xl space-y-6"
    >
      {/* Executive Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center space-x-2">
            <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-cyan-400">
              Executive Overview • Leadership Operating Dashboard
            </span>
          </div>
          <h2 className="text-xl font-extrabold text-white tracking-tight mt-1">
            ARX Decision Intelligence :: Executive Execution Tracker
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time bridge from System Quality to User Outcomes & Behavioral Telemetry.
          </p>
        </div>

        {/* Status Stamp */}
        <div className="flex items-center space-x-3">
          <div className="bg-slate-950 border border-slate-800 rounded-xl px-4 py-2 text-right">
            <div className="text-[10px] uppercase font-mono text-slate-400">Current Readiness</div>
            <div className="flex items-baseline justify-end space-x-1">
              <span className="text-2xl font-bold font-mono text-cyan-300">
                {metrics.currentReadinessScore}%
              </span>
              <span className="text-xs font-mono text-emerald-400">▲ +4.1%</span>
            </div>
          </div>

          <div className="bg-cyan-950/60 border border-cyan-500 rounded-xl px-4 py-2 text-center">
            <div className="text-[10px] uppercase font-mono text-cyan-300 font-bold">Status</div>
            <div className="text-sm font-bold font-mono text-white">
              {metrics.releaseStatus}
            </div>
            <div className="text-[9px] font-mono text-slate-400">Target: {metrics.targetReadinessScore}%</div>
          </div>
        </div>
      </div>

      {/* 4 Core Sections Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* SECTION 1: PLATFORM HEALTH */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-bold text-slate-300 uppercase tracking-wider">
              1. Platform Health
            </span>
            <span className="text-emerald-400">HEALTHY</span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-slate-800/80">
              <span className="text-slate-400">Readiness:</span>
              <span className="font-bold text-cyan-300">{metrics.currentReadinessScore}% (95/100)</span>
            </div>
            <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-slate-800/80">
              <span className="text-slate-400">Target Benchmark:</span>
              <span className="font-bold text-white">98.0% Excellence</span>
            </div>
            <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-slate-800/80">
              <span className="text-slate-400">Critical Defects:</span>
              <span className="font-bold text-emerald-400">0 Open</span>
            </div>
            <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-slate-800/80">
              <span className="text-slate-400">Major Defects:</span>
              <span className="font-bold text-emerald-400">0 Open</span>
            </div>
          </div>
        </div>

        {/* SECTION 2: USER OUTCOME METRICS (TIER 1) */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-bold text-slate-300 uppercase tracking-wider">
              2. User Outcomes (Tier 1)
            </span>
            <span className="text-emerald-400">ON TRACK</span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            {/* Quality Score */}
            <div className="p-2 rounded bg-slate-900 border border-slate-800/80">
              <div className="flex justify-between mb-1">
                <span className="text-slate-400">Decision Quality:</span>
                <span className="font-bold text-white">{metrics.decisionQualityScore} <span className="text-slate-500">/ 80 Target</span></span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${(metrics.decisionQualityScore / 80) * 100}%` }} />
              </div>
            </div>

            {/* BAR */}
            <div className="p-2 rounded bg-slate-900 border border-slate-800/80">
              <div className="flex justify-between mb-1">
                <span className="text-slate-400">Behavior Adoption (BAR):</span>
                <span className="font-bold text-emerald-400">{metrics.behavioralAdoptionRate}% <span className="text-slate-500">(&gt;70%)</span></span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${metrics.behavioralAdoptionRate}%` }} />
              </div>
            </div>

            {/* Repeat Mistakes */}
            <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-slate-800/80">
              <span className="text-slate-400">Repeat Mistakes:</span>
              <span className="font-bold text-emerald-400">{metrics.repeatMistakeReduction}% (&gt;30% Target)</span>
            </div>

            {/* Decision Drift */}
            <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-slate-800/80">
              <span className="text-slate-400">Decision Drift:</span>
              <span className="font-bold text-cyan-300">{metrics.decisionDrift}% (&lt;25% Target)</span>
            </div>
          </div>
        </div>

        {/* SECTION 3: MENTOR METRICS (TIER 2) */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-bold text-slate-300 uppercase tracking-wider">
              3. Mentor Metrics (Tier 2)
            </span>
            <span className="text-cyan-400">ENGAGED</span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="p-2 rounded bg-slate-900 border border-slate-800/80">
              <div className="flex justify-between mb-1">
                <span className="text-slate-400">Mentor Views:</span>
                <span className="font-bold text-white">{metrics.mentorVisibilityRate}% <span className="text-slate-500">(&gt;95%)</span></span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${metrics.mentorVisibilityRate}%` }} />
              </div>
            </div>

            <div className="p-2 rounded bg-slate-900 border border-slate-800/80">
              <div className="flex justify-between mb-1">
                <span className="text-slate-400">Recommendation Clicks:</span>
                <span className="font-bold text-emerald-400">{metrics.recommendationEngagementRate}% <span className="text-slate-500">(&gt;60%)</span></span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${metrics.recommendationEngagementRate}%` }} />
              </div>
            </div>

            <div className="p-2 rounded bg-slate-900 border border-slate-800/80">
              <div className="flex justify-between mb-1">
                <span className="text-slate-400">Evidence Opens (Trust):</span>
                <span className="font-bold text-indigo-400">{metrics.trustValidationRate}% <span className="text-slate-500">(30-70%)</span></span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div className="bg-indigo-500 h-full rounded-full" style={{ width: `${metrics.trustValidationRate}%` }} />
              </div>
            </div>

            <div className="flex justify-between items-center p-2 rounded bg-slate-900 border border-slate-800/80">
              <span className="text-slate-400">CEO Speed Target:</span>
              <span className="font-bold text-emerald-400">{metrics.questionResolutionAvgTimeSec}s (&lt;5s)</span>
            </div>
          </div>
        </div>

        {/* SECTION 4: GO / NO-GO RELEASE GATES */}
        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-bold text-slate-300 uppercase tracking-wider">
              4. Release Gate Tracker
            </span>
            <span className="text-amber-400">6/8 CERTIFIED</span>
          </div>

          <div className="space-y-1.5 text-[11px] font-mono">
            {gates.map((g) => (
              <div key={g.id} className="flex items-center justify-between p-1.5 rounded bg-slate-900 border border-slate-800/70">
                <span className="text-slate-300 truncate mr-2" title={g.evidence}>
                  {g.name.split(' (')[0]}
                </span>
                <span className={`px-1.5 py-0.2 rounded text-[10px] font-bold shrink-0 ${
                  g.status === 'CERTIFIED'
                    ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                    : 'bg-amber-950 text-amber-300 border border-amber-800'
                }`}>
                  {g.status === 'CERTIFIED' ? '✅' : '🟡'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* SECTION 5: INTERACTIVE LIVE TELEMETRY SIMULATOR */}
      <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-2">
          <div className="flex items-center space-x-2">
            <span className="text-sm">📡</span>
            <span className="text-xs font-bold uppercase tracking-wider text-white font-mono">
              Live Outcome Telemetry Event Stream & Verification Harness
            </span>
          </div>
          <span className="text-[11px] font-mono text-cyan-400">
            {lastDispatched}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-mono text-slate-400 mr-2">Simulate Event:</span>
          <button
            type="button"
            onClick={() => triggerEvent('view')}
            className="px-2.5 py-1 text-xs font-mono rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            + mentor_viewed (SEEN)
          </button>
          <button
            type="button"
            onClick={() => triggerEvent('click')}
            className="px-2.5 py-1 text-xs font-mono rounded bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            + recommendation_clicked (UNDERSTOOD)
          </button>
          <button
            type="button"
            onClick={() => triggerEvent('evidence')}
            className="px-2.5 py-1 text-xs font-mono rounded bg-indigo-950 hover:bg-indigo-900 text-indigo-300 border border-indigo-800 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            + evidence_opened (TRUST)
          </button>
          <button
            type="button"
            onClick={() => triggerEvent('rule')}
            className="px-2.5 py-1 text-xs font-mono rounded bg-emerald-950 hover:bg-emerald-900 text-emerald-300 border border-emerald-800 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            + rule_followed (BEHAVIOR)
          </button>
          <button
            type="button"
            onClick={() => triggerEvent('drift')}
            className="px-2.5 py-1 text-xs font-mono rounded bg-rose-950 hover:bg-rose-900 text-rose-300 border border-rose-800 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            + drift_acknowledged (ACTED)
          </button>
        </div>

        {/* Event Log Output */}
        {recentEvents.length > 0 && (
          <div className="mt-3 p-2.5 bg-slate-900/90 rounded-lg border border-slate-800 max-h-36 overflow-y-auto space-y-1 font-mono text-[11px]">
            {recentEvents.slice(-5).reverse().map((ev) => (
              <div key={ev.id} className="flex items-center justify-between text-slate-300 py-0.5 border-b border-slate-800/60 last:border-0">
                <div className="flex items-center space-x-2 truncate">
                  <span className="text-[10px] px-1.5 rounded bg-slate-800 text-cyan-400 font-bold">
                    {ev.phase}
                  </span>
                  <span className="text-white font-semibold truncate">{ev.event}</span>
                  <span className="text-slate-500 text-[10px]">[{ev.category}]</span>
                </div>
                <span className="text-[10px] text-slate-500 shrink-0 ml-2">
                  {ev.timestamp.split('T')[1].slice(0, 8)} UTC
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
