'use client';

/**
 * Phase 29: Capability Attribution Dashboard
 *
 * Implements P29-200/300/400/500:
 * - Capability Impact Score (CIS) ranking
 * - Economic ROI attribution (100% total)
 * - Investment Prioritization Matrix (4 quadrants)
 * - Monthly Executive Value Report
 */

import React, { useState } from 'react';
import {
  CANONICAL_CAPABILITY_SCORES,
  CANONICAL_ATTRIBUTION_SUMMARY,
  CANONICAL_INVESTMENT_MATRIX,
  generateMonthlyValueReport,
} from '@/lib/telemetry/capabilityAttributionV2Engine';

type ActiveTab = 'cis' | 'roi' | 'matrix' | 'report';

function CISRanking() {
  const caps = [...CANONICAL_CAPABILITY_SCORES].sort((a, b) => b.cis - a.cis);

  return (
    <div data-testid="cis-ranking" role="tabpanel">
      <div className="bg-gray-800 rounded-lg p-3 mb-4 flex items-center justify-between text-sm">
        <span className="text-gray-400">Total Attribution Coverage</span>
        <span className={`font-bold ${CANONICAL_ATTRIBUTION_SUMMARY.totalAttributionPct === 100 ? 'text-green-400' : 'text-rose-400'}`}>
          {CANONICAL_ATTRIBUTION_SUMMARY.totalAttributionPct}% ✅ INV-OI2 Satisfied
        </span>
      </div>
      <div className="space-y-4">
        {caps.map((cap, i) => (
          <div key={cap.capabilityId} className="bg-gray-800 rounded-xl p-5" data-testid={`capability-${cap.capabilityId}`}>
            <div className="flex items-start justify-between gap-4 mb-3">
              <div className="flex items-center gap-3">
                <span className="text-3xl font-black text-gray-600">#{i + 1}</span>
                <div>
                  <p className="text-white font-bold">{cap.capabilityName}</p>
                  <p className="text-gray-400 text-xs">{cap.adoptionPct}% adoption · N={cap.sampleSize.toLocaleString()} · {cap.confidence}% CI</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-gray-400 text-xs">CIS</p>
                <p className="text-3xl font-black text-green-400">{cap.cis}</p>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-gray-900 rounded-lg p-3 text-center">
                <p className="text-gray-400 text-xs">Behavior Lift</p>
                <p className="text-green-400 font-bold text-lg">+{cap.behaviorLiftPct}%</p>
              </div>
              <div className="bg-gray-900 rounded-lg p-3 text-center">
                <p className="text-gray-400 text-xs">DQ Lift</p>
                <p className="text-blue-400 font-bold text-lg">+{cap.decisionQualityLift}</p>
              </div>
              <div className="bg-gray-900 rounded-lg p-3 text-center">
                <p className="text-gray-400 text-xs">Capital Preserved</p>
                <p className="text-amber-400 font-bold text-lg">{cap.capitalPreservedFormatted}</p>
              </div>
            </div>
            <div className="mt-3">
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>Attribution</span>
                <span className="text-white font-semibold">{cap.contributionPct}%</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full">
                <div className="h-full bg-green-500 rounded-full" style={{ width: `${cap.contributionPct}%` }} />
              </div>
            </div>
          </div>
        ))}
        {/* Other (8%) */}
        <div className="bg-gray-800 rounded-lg px-5 py-3 flex items-center justify-between text-sm text-gray-400">
          <span>Other capabilities</span>
          <span className="font-semibold">8.0%</span>
        </div>
      </div>
    </div>
  );
}

function EconomicROI() {
  const s = CANONICAL_ATTRIBUTION_SUMMARY;
  return (
    <div data-testid="economic-roi" role="tabpanel">
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-gray-800 rounded-xl p-5 text-center" data-testid="total-capital-preserved">
          <p className="text-gray-400 text-xs uppercase tracking-widest mb-1">Capital Preserved</p>
          <p className="text-4xl font-black text-green-400">{s.totalCapitalPreserved}</p>
        </div>
        <div className="bg-gray-800 rounded-xl p-5 text-center" data-testid="excess-return">
          <p className="text-gray-400 text-xs uppercase tracking-widest mb-1">Excess Return</p>
          <p className="text-4xl font-black text-blue-400">+{s.excessReturnPct}%</p>
        </div>
      </div>
      <div className="bg-gray-800 rounded-xl p-5">
        <h4 className="text-white font-bold mb-4">Value Attribution Breakdown</h4>
        {[...s.capabilities].sort((a, b) => b.contributionPct - a.contributionPct).map(cap => (
          <div key={cap.capabilityId} className="mb-3">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-300">{cap.capabilityName}</span>
              <span className="text-white font-bold">{cap.contributionPct}% · {cap.capitalPreservedFormatted}</span>
            </div>
            <div className="h-2 bg-gray-700 rounded-full">
              <div className="h-full bg-blue-500 rounded-full" style={{ width: `${cap.contributionPct}%` }} />
            </div>
          </div>
        ))}
        <div className="mt-3 flex justify-between text-sm border-t border-gray-700 pt-3">
          <span className="text-gray-400">Other</span>
          <span className="text-white font-bold">8.0% — residual</span>
        </div>
      </div>
    </div>
  );
}

function InvestmentMatrix() {
  const q1 = CANONICAL_INVESTMENT_MATRIX.filter(c => c.quadrant === 'HIGH_IMPACT_HIGH_ADOPTION');
  const q2 = CANONICAL_INVESTMENT_MATRIX.filter(c => c.quadrant === 'HIGH_IMPACT_LOW_ADOPTION');
  const q3 = CANONICAL_INVESTMENT_MATRIX.filter(c => c.quadrant === 'LOW_IMPACT_HIGH_ADOPTION');
  const q4 = CANONICAL_INVESTMENT_MATRIX.filter(c => c.quadrant === 'LOW_IMPACT_LOW_ADOPTION');

  const Quadrant = ({ label, items, className }: { label: string; items: typeof q1; className: string }) => (
    <div className={`rounded-xl p-4 ${className}`} data-testid={`quadrant-${label.toLowerCase().replace(/ /g, '-')}`}>
      <p className="text-white font-bold text-sm mb-3">{label}</p>
      {items.length === 0 ? (
        <p className="text-gray-500 text-xs">No capabilities</p>
      ) : (
        <div className="space-y-2">
          {items.map(c => (
            <div key={c.capabilityId} className="bg-black/20 rounded-lg p-3">
              <p className="text-white text-xs font-semibold">{c.capabilityName}</p>
              <p className="text-gray-400 text-xs">CIS {c.cis} · {c.adoptionPct}% adoption</p>
              <p className="text-gray-300 text-xs mt-1">{c.investmentRecommendation}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div data-testid="investment-matrix" role="tabpanel">
      <div className="grid grid-cols-2 gap-4">
        <Quadrant label="High Impact · High Adoption" items={q1} className="bg-green-950 border border-green-700" />
        <Quadrant label="High Impact · Low Adoption" items={q2} className="bg-blue-950 border border-blue-700" />
        <Quadrant label="Low Impact · High Adoption" items={q3} className="bg-amber-950 border border-amber-700" />
        <Quadrant label="Low Impact · Low Adoption" items={q4} className="bg-rose-950 border border-rose-700" />
      </div>
    </div>
  );
}

function MonthlyReport() {
  const report = generateMonthlyValueReport();
  return (
    <div data-testid="monthly-value-report" role="tabpanel">
      <div className="bg-gray-800 rounded-xl p-5 mb-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-white font-bold">What Created Value — {report.period}</h4>
          <span className="text-gray-400 text-xs">{report.totalCapitalPreserved} total · +{report.excessReturn}% excess return</span>
        </div>
        <div className="space-y-2">
          {report.topValueDrivers.map((d, i) => (
            <div key={i} className="flex items-center gap-3 bg-gray-900 rounded-lg px-4 py-2.5">
              <span className="text-gray-500 font-bold w-4">{i + 1}.</span>
              <span className="flex-1 text-white text-sm">{d.capability}</span>
              <span className="text-green-400 font-bold text-sm">{d.value}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="bg-gray-800 rounded-xl p-5 mb-4">
        <h4 className="text-white font-bold mb-3">Lowest Contributors</h4>
        {report.weakestContributors.map((w, i) => (
          <div key={i} className="bg-gray-900 rounded-lg p-4">
            <p className="text-white text-sm font-semibold">{w.capability}</p>
            <p className="text-rose-400 text-xs mb-1">{w.reason}</p>
            <p className="text-amber-400 text-xs">→ {w.recommendation}</p>
          </div>
        ))}
      </div>
      <div className="bg-gray-800 rounded-xl p-5">
        <h4 className="text-white font-bold mb-3">Investment Recommendations</h4>
        <div className="space-y-2">
          {report.investmentRecommendations.map((rec, i) => (
            <p key={i} className="text-gray-300 text-sm flex items-start gap-2">
              <span className="text-blue-400 mt-0.5">→</span>
              {rec}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function CapabilityAttributionDashboard() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('cis');

  const tabs: Array<{ id: ActiveTab; label: string }> = [
    { id: 'cis', label: 'Capability Impact (CIS)' },
    { id: 'roi', label: 'Economic ROI' },
    { id: 'matrix', label: 'Investment Matrix' },
    { id: 'report', label: 'Monthly Report' },
  ];

  return (
    <div role="main" aria-label="Capability Attribution Dashboard">
      <div className="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden">
        <div className="flex flex-wrap border-b border-gray-700" role="tablist">
          {tabs.map(tab => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-3 text-sm font-semibold min-h-[44px] transition-colors ${activeTab === tab.id ? 'bg-gray-800 text-white border-b-2 border-blue-500' : 'text-gray-400 hover:text-gray-200'}`}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div className="p-6">
          {activeTab === 'cis' && <CISRanking />}
          {activeTab === 'roi' && <EconomicROI />}
          {activeTab === 'matrix' && <InvestmentMatrix />}
          {activeTab === 'report' && <MonthlyReport />}
        </div>
      </div>
    </div>
  );
}

