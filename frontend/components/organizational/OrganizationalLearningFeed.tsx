'use client';

/**
 * Phase 29: Organizational Learning Feed
 *
 * Implements OI-201/202/203:
 * - Cross-Team Learning Feed with relevance scores
 * - Best Practice Propagation status
 * - Learning Impact Tracking (adopted / improved / ignored)
 */

import React, { useState } from 'react';
import {
  CANONICAL_LEARNING_FEED,
  CANONICAL_BEST_PRACTICE_PROPAGATIONS,
  CANONICAL_LEARNING_IMPACT_TRACKING,
  verifyLearningConservation,
  getKnowledgeReuseRate,
  getLearningVelocityQoQ,
} from '@/lib/telemetry/organizationalLearningEngine';

type ActiveTab = 'feed' | 'propagation' | 'impact';

export default function OrganizationalLearningFeed() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('feed');
  const conservation = verifyLearningConservation();
  const reuseRate = getKnowledgeReuseRate();
  const velocity = getLearningVelocityQoQ();

  const tabs: Array<{ id: ActiveTab; label: string }> = [
    { id: 'feed', label: 'Cross-Team Feed' },
    { id: 'propagation', label: 'Best Practice Propagation' },
    { id: 'impact', label: 'Learning Impact Tracking' },
  ];

  return (
    <div role="main" aria-label="Organizational Learning Intelligence">
      {/* KPI Strip */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {[
          { label: 'Knowledge Reuse Rate', value: `${reuseRate}%`, target: '>70%', pass: reuseRate >= 70 },
          { label: 'Learning Velocity QoQ', value: `+${velocity}%`, target: '>10%', pass: velocity >= 10 },
          { label: 'Learning Conservation', value: conservation.isConservationSatisfied ? 'SATISFIED' : 'BREACH', target: '±1%', pass: conservation.isConservationSatisfied },
        ].map(kpi => (
          <div key={kpi.label} className="bg-gray-900 border border-gray-700 rounded-xl p-4 text-center" data-testid={`learning-kpi-${kpi.label.toLowerCase().replace(/ /g, '-')}`}>
            <p className="text-gray-400 text-xs uppercase tracking-widest mb-1">{kpi.label}</p>
            <p className={`text-2xl font-black ${kpi.pass ? 'text-green-400' : 'text-rose-400'}`}>{kpi.value}</p>
            <p className="text-gray-500 text-xs">target {kpi.target} {kpi.pass ? '✅' : '❌'}</p>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden">
        <div className="flex border-b border-gray-700" role="tablist">
          {tabs.map(tab => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 px-4 py-3 text-sm font-semibold min-h-[44px] transition-colors ${activeTab === tab.id ? 'bg-gray-800 text-white border-b-2 border-blue-500' : 'text-gray-400 hover:text-gray-200'}`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="p-6">
          {activeTab === 'feed' && (
            <div className="space-y-4" data-testid="cross-team-feed" role="tabpanel">
              {CANONICAL_LEARNING_FEED.map(item => (
                <div key={item.itemId} className="bg-gray-800 rounded-lg p-4" data-testid={`feed-item-${item.itemId}`}>
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <div>
                      <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${item.recommendedAction === 'ADOPT' ? 'bg-green-900 text-green-300' : 'bg-amber-900 text-amber-300'}`}>
                        {item.recommendedAction}
                      </span>
                      <span className="ml-2 text-gray-400 text-xs">{item.sourceTeam}</span>
                    </div>
                    <div className="text-right">
                      <p className="text-gray-400 text-xs">Relevance</p>
                      <p className="text-white font-bold text-sm">{item.relevanceScore}%</p>
                    </div>
                  </div>
                  <p className="text-white font-semibold text-sm mb-1">{item.learningTitle}</p>
                  <p className="text-green-400 text-xs">{item.potentialImpact}</p>
                  <p className="text-gray-500 text-xs mt-1">Confidence: {item.confidence}%</p>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'propagation' && (
            <div className="space-y-3" data-testid="best-practice-propagation" role="tabpanel">
              {CANONICAL_BEST_PRACTICE_PROPAGATIONS.map(bp => (
                <div key={bp.propagationId} className="flex items-center gap-4 bg-gray-800 rounded-lg px-4 py-3" data-testid={`propagation-${bp.propagationId}`}>
                  <div className="flex-1 min-w-0">
                    <p className="text-white text-sm font-semibold">{bp.ruleName}</p>
                    <p className="text-gray-400 text-xs">{bp.sourceTeam} → {bp.receivingTeam}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-400 text-xs">Adoption</p>
                    <p className="text-white font-bold text-sm">{bp.adoptionRate}%</p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-400 text-xs">Impact</p>
                    <p className="text-white font-bold text-sm">{bp.impactScore}</p>
                  </div>
                  <span className={`text-xs font-bold px-2 py-1 rounded-full ${bp.status === 'ADOPTED' ? 'bg-green-900 text-green-300' : bp.status === 'PENDING' ? 'bg-amber-900 text-amber-300' : 'bg-rose-900 text-rose-300'}`}>
                    {bp.status}
                  </span>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'impact' && (
            <div data-testid="learning-impact-tracking" role="tabpanel">
              <div className="overflow-x-auto">
                <table className="w-full text-sm" role="table" aria-label="Learning Impact by Team">
                  <thead>
                    <tr className="text-gray-400 text-xs border-b border-gray-700">
                      <th className="text-left pb-3">Team</th>
                      <th className="text-center pb-3">Adopted</th>
                      <th className="text-center pb-3">Improved</th>
                      <th className="text-center pb-3">Ignored</th>
                      <th className="text-center pb-3">Adoption Rate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-800">
                    {CANONICAL_LEARNING_IMPACT_TRACKING.map(rec => (
                      <tr key={rec.teamId} data-testid={`impact-${rec.teamId}`}>
                        <td className="py-3 text-white font-semibold">{rec.teamName}</td>
                        <td className="py-3 text-center text-green-400 font-bold">{rec.adopted}</td>
                        <td className="py-3 text-center text-blue-400 font-bold">{rec.improved}</td>
                        <td className="py-3 text-center text-rose-400 font-bold">{rec.ignored}</td>
                        <td className="py-3 text-center">
                          <span className={`font-bold ${rec.adoptionRate >= 70 ? 'text-green-400' : rec.adoptionRate >= 50 ? 'text-amber-400' : 'text-rose-400'}`}>
                            {rec.adoptionRate}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

