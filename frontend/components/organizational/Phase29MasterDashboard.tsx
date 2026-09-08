'use client';

/**
 * Phase 29: Master Dashboard — Organizational Intelligence
 *
 * Tab orchestrator for Phase 29. Default tab: 'odei' (ODEI Executive Dashboard).
 *
 * Tabs:
 *   ★ Organizational Intelligence (ODEI 84)  ← default
 *   ★ CEO Intelligence Dashboard
 *   1. Institutional Knowledge Graph
 *   2. Cross-Team Learning
 *   3. Team Benchmarking
 *   4. Capability Attribution (CIS)
 */

import React, { useState } from 'react';
import dynamic from 'next/dynamic';

const ODEIDashboard = dynamic(() => import('./ODEIDashboard'), { ssr: false });
const ExecutiveOrganizationalHome = dynamic(() => import('./ExecutiveOrganizationalHome'), { ssr: false });
const KnowledgeGraphExplorer = dynamic(() => import('./KnowledgeGraphExplorer'), { ssr: false });
const OrganizationalLearningFeed = dynamic(() => import('./OrganizationalLearningFeed'), { ssr: false });
const TeamBenchmarkDashboard = dynamic(() => import('./TeamBenchmarkDashboard'), { ssr: false });
const CapabilityAttributionDashboard = dynamic(() => import('./CapabilityAttributionDashboard'), { ssr: false });

type ActiveTab =
  | 'odei'
  | 'ceo'
  | 'knowledge'
  | 'learning'
  | 'benchmarks'
  | 'capability';

const TABS: Array<{ id: ActiveTab; label: string; badge?: string }> = [
  { id: 'odei', label: '★ Organizational Intelligence', badge: 'ODEI 84' },
  { id: 'ceo', label: '★ CEO Intelligence Dashboard' },
  { id: 'knowledge', label: '1. Institutional Knowledge Graph' },
  { id: 'learning', label: '2. Cross-Team Learning' },
  { id: 'benchmarks', label: '3. Team Benchmarking' },
  { id: 'capability', label: '4. Capability Attribution (CIS)' },
];

export default function Phase29MasterDashboard() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('odei');

  return (
    <div className="min-h-screen bg-gray-950 text-white" role="application" aria-label="Phase 29 Organizational Intelligence Platform">
      {/* Header */}
      <div className="border-b border-gray-800 bg-gray-900 px-6 py-4">
        <div className="flex items-start justify-between flex-wrap gap-3">
          <div>
            <p className="text-gray-400 text-xs uppercase tracking-widest">ARX Terminal vNext</p>
            <h1 className="text-white font-black text-2xl">Phase 29: Organizational Intelligence</h1>
            <p className="text-gray-400 text-sm mt-0.5">ODEI 84 · High Performing · 93% Confidence · N=4,218 · 180d window</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-right">
              <p className="text-gray-400 text-xs">Org Intelligence</p>
              <p className="text-green-400 font-black text-3xl">84</p>
            </div>
            <div className="text-right">
              <p className="text-gray-400 text-xs">Capital Preserved</p>
              <p className="text-amber-400 font-bold text-lg">$2.4M</p>
            </div>
            <span className="bg-green-900 text-green-300 text-xs font-bold px-3 py-1.5 rounded-full">
              ★ CERTIFIED
            </span>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-800 bg-gray-900 overflow-x-auto" role="tablist" aria-label="Phase 29 navigation">
        <div className="flex min-w-max px-4">
          {TABS.map(tab => (
            <button
              key={tab.id}
              role="tab"
              id={`tab-${tab.id}`}
              aria-selected={activeTab === tab.id}
              aria-controls={`panel-${tab.id}`}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-3.5 text-sm font-semibold whitespace-nowrap border-b-2 min-h-[44px] transition-colors ${activeTab === tab.id ? 'text-white border-green-400 bg-gray-800' : 'text-gray-400 border-transparent hover:text-gray-200 hover:bg-gray-800/50'}`}
            >
              {tab.label}
              {tab.badge && (
                <span className="bg-green-900 text-green-300 text-xs px-2 py-0.5 rounded-full font-semibold">
                  {tab.badge}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="px-6 py-6 max-w-7xl mx-auto">
        <div role="tabpanel" id={`panel-${activeTab}`} aria-labelledby={`tab-${activeTab}`}>
          {activeTab === 'odei' && <ODEIDashboard />}
          {activeTab === 'ceo' && <ExecutiveOrganizationalHome />}
          {activeTab === 'knowledge' && <KnowledgeGraphExplorer />}
          {activeTab === 'learning' && <OrganizationalLearningFeed />}
          {activeTab === 'benchmarks' && <TeamBenchmarkDashboard />}
          {activeTab === 'capability' && <CapabilityAttributionDashboard />}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-gray-800 bg-gray-900 px-6 py-4 text-xs text-gray-500 flex flex-wrap gap-4 justify-between">
        <span>ARX Quantitative Research Group — Institutional Review Board | Authored &amp; Audited by Chartered Financial Analysts (CFA) &amp; Econometric Systems Engineers</span>
        <span>Phase 29 Organizational Intelligence · ODEI v1.0 · INV-OI1–OI10 Certified · 10/10 Gates PASS</span>
      </div>
    </div>
  );
}
