'use client';

import React, { useState, useMemo } from 'react';
import {
  IssuerEvidenceSummary,
  EvidenceState,
  CohortAccessionItem,
  CohortReconciliationReceipt,
  ReceiptValidationReport,
} from '../../types/cohortPilotEvidence';
import {
  SYNTHETIC_ISSUER_SUMMARIES,
  SYNTHETIC_COHORT_RECEIPT,
  buildSecEdgarUrl,
  parseReceiptToSummaries,
  validateReceiptAgainstExpected,
} from '../../fixtures/cohortPilotEvidenceFixtures';

interface Props {
  initialSummaries?: IssuerEvidenceSummary[];
  className?: string;
  isPrivateMode?: boolean;
}

export const Form4PilotEvidencePreview: React.FC<Props> = ({
  initialSummaries = SYNTHETIC_ISSUER_SUMMARIES,
  className = '',
  isPrivateMode = true,
}) => {
  const [summaries, setSummaries] = useState<IssuerEvidenceSummary[]>(initialSummaries);
  const [selectedStateFilter, setSelectedStateFilter] = useState<string>('ALL');
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const [customReceiptJson, setCustomReceiptJson] = useState<string>('');
  const [validationReport, setValidationReport] = useState<ReceiptValidationReport | null>(null);
  const [customReceiptError, setCustomReceiptError] = useState<string | null>(null);
  const [dataSource, setDataSource] = useState<'SYNTHETIC' | 'CUSTOM'>('SYNTHETIC');

  // Filtered summaries
  const filteredSummaries = useMemo(() => {
    if (selectedStateFilter === 'ALL') return summaries;
    return summaries.filter((s) => s.state === selectedStateFilter);
  }, [summaries, selectedStateFilter]);

  // KPI counts
  const kpis = useMemo(() => {
    return {
      total: summaries.length,
      filingsFound: summaries.filter((s) => s.state === 'FILINGS_FOUND').length,
      polledQuiet: summaries.filter((s) => s.state === 'POLLED_QUIET').length,
      incomplete: summaries.filter((s) => s.state === 'INCOMPLETE').length,
      notPolled: summaries.filter((s) => s.state === 'NOT_POLLED').length,
      readError: summaries.filter((s) => s.state === 'READ_ERROR').length,
    };
  }, [summaries]);

  // Handle switching to synthetic fixtures
  const handleLoadSynthetic = () => {
    setDataSource('SYNTHETIC');
    setSummaries(SYNTHETIC_ISSUER_SUMMARIES);
    setValidationReport(null);
    setCustomReceiptError(null);
  };

  // Handle parsing user-pasted / uploaded custom receipt
  const handleParseCustomReceipt = () => {
    setCustomReceiptError(null);
    try {
      if (!customReceiptJson.trim()) {
        throw new Error('Please paste a valid JSON reconciliation receipt.');
      }
      const parsed: CohortReconciliationReceipt = JSON.parse(customReceiptJson);
      if (!parsed.failure_reasons?.cohort_evidence?.issuers) {
        throw new Error('Invalid receipt structure: missing failure_reasons.cohort_evidence.issuers');
      }
      const parsedSummaries = parseReceiptToSummaries(parsed);
      setSummaries(parsedSummaries);
      setDataSource('CUSTOM');

      // Run automatic validation against expected synthetic profile
      const expectedMap: Record<string, Partial<IssuerEvidenceSummary>> = {};
      SYNTHETIC_ISSUER_SUMMARIES.forEach((s) => {
        expectedMap[s.ticker] = {
          state: s.state,
          new_filings_count: s.new_filings_count,
          accounted_through: s.accounted_through,
          gaps: s.gaps,
        };
      });
      const report = validateReceiptAgainstExpected(parsed, expectedMap);
      setValidationReport(report);
    } catch (err: any) {
      setCustomReceiptError(err.message || 'Failed to parse receipt JSON');
    }
  };

  // State badge styling
  const renderStateBadge = (state: EvidenceState) => {
    switch (state) {
      case 'FILINGS_FOUND':
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-950/80 text-blue-400 border border-blue-700/60">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mr-1.5 animate-pulse" />
            FILINGS FOUND
          </span>
        );
      case 'POLLED_QUIET':
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-700/60">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5" />
            QUIET — ACCOUNTED
          </span>
        );
      case 'INCOMPLETE':
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-950/80 text-amber-400 border border-amber-700/60">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mr-1.5" />
            INCOMPLETE / GAPS
          </span>
        );
      case 'NOT_POLLED':
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-800 text-slate-400 border border-slate-700">
            <span className="w-1.5 h-1.5 rounded-full bg-slate-500 mr-1.5" />
            NOT POLLED
          </span>
        );
      case 'READ_ERROR':
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-950/80 text-rose-400 border border-rose-700/60">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mr-1.5" />
            READ ERROR
          </span>
        );
    }
  };

  // Format timestamp helper
  const formatTime = (ts: string | null) => {
    if (!ts) return '—';
    try {
      const d = new Date(ts);
      return `${d.toISOString().slice(11, 19)} UTC (${d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })})`;
    } catch {
      return ts;
    }
  };

  return (
    <div
      className={`w-full bg-slate-950 text-slate-100 rounded-xl border border-slate-800 p-4 sm:p-6 font-sans ${className}`}
      data-testid="form4-pilot-evidence-preview"
    >
      {/* 1. Header & Governance Guardrail Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
              Form 4 Pilot Evidence Preview
            </h2>
            {isPrivateMode && (
              <span className="px-2 py-0.5 text-xs font-bold uppercase tracking-wider rounded bg-purple-950 text-purple-300 border border-purple-700">
                Private Preview
              </span>
            )}
          </div>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            ARX Issue #165 &bull; Trial <code className="text-slate-300">ARX-TRIAL-10-ISSUER-001</code> &bull; Candidate <code className="text-slate-300">1063a10f</code>
          </p>
        </div>

        {/* Source Switcher */}
        <div className="flex items-center gap-2 self-start md:self-auto">
          <button
            type="button"
            onClick={handleLoadSynthetic}
            className={`min-h-[44px] px-3.5 py-2 text-xs font-medium rounded-lg transition-colors ${
              dataSource === 'SYNTHETIC'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
            }`}
          >
            Synthetic Fixtures
          </button>
          <button
            type="button"
            onClick={() => setDataSource('CUSTOM')}
            className={`min-h-[44px] px-3.5 py-2 text-xs font-medium rounded-lg transition-colors ${
              dataSource === 'CUSTOM'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
            }`}
          >
            Load Real Receipt
          </button>
        </div>
      </div>

      {/* Operator Safety Banner */}
      <div className="mt-3 p-3 bg-slate-900/90 border border-slate-800 rounded-lg text-xs text-slate-400 flex items-start gap-2">
        <svg
          className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <span>
          <strong className="text-slate-200">Operator Isolation Fencing:</strong> This preview connects only to offline fixtures or read-only reconciliation receipts. Production API remains on canonical database with frozen Radar and Smart Money snapshots preserved.
        </span>
      </div>

      {/* 2. Custom Receipt Input & Validation Section (Collapsible) */}
      {dataSource === 'CUSTOM' && (
        <div className="mt-4 p-4 bg-slate-900 border border-slate-800 rounded-lg">
          <h3 className="text-sm font-semibold text-white mb-2">
            Paste Reconciled Receipt JSON
          </h3>
          <textarea
            value={customReceiptJson}
            onChange={(e) => setCustomReceiptJson(e.target.value)}
            placeholder="Paste raw sm_reconciliation_receipts JSON payload here..."
            className="w-full h-28 bg-slate-950 border border-slate-800 rounded p-2 text-xs font-mono text-slate-300 focus:outline-none focus:border-blue-500"
          />
          <div className="mt-2 flex items-center gap-2">
            <button
              type="button"
              onClick={handleParseCustomReceipt}
              className="min-h-[44px] px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded"
            >
              Parse &amp; Validate Receipt
            </button>
            <button
              type="button"
              onClick={() => {
                setCustomReceiptJson(JSON.stringify(SYNTHETIC_COHORT_RECEIPT, null, 2));
              }}
              className="min-h-[44px] px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded"
            >
              Fill Sample JSON
            </button>
          </div>

          {customReceiptError && (
            <div className="mt-2 text-xs text-rose-400 font-medium">
              Error: {customReceiptError}
            </div>
          )}

          {/* Validation Discrepancy Report */}
          {validationReport && (
            <div className="mt-3 p-3 bg-slate-950 border border-slate-800 rounded text-xs">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-slate-300">
                  Receipt Validation Report (ID: {validationReport.receipt_id})
                </span>
                <span
                  className={`px-2 py-0.5 rounded font-bold text-[10px] ${
                    validationReport.is_valid
                      ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                      : 'bg-rose-950 text-rose-300 border border-rose-800'
                  }`}
                >
                  {validationReport.is_valid ? '10/10 MATCH' : 'DISCREPANCIES DETECTED'}
                </span>
              </div>
              {validationReport.discrepancies.length === 0 ? (
                <p className="text-emerald-400">All 10 issuer rows perfectly reconcile against expectations.</p>
              ) : (
                <ul className="space-y-1 text-slate-400">
                  {validationReport.discrepancies.map((d, i) => (
                    <li key={i} className="text-rose-300">
                      [{d.severity}] {d.ticker}: {d.field} expected &lsquo;{String(d.expected)}&rsquo; but got &lsquo;{String(d.actual)}&rsquo;
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      )}

      {/* 3. KPI / Filter Strip */}
      <div className="mt-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 sm:gap-3">
        <button
          type="button"
          onClick={() => setSelectedStateFilter('ALL')}
          className={`min-h-[44px] p-2.5 rounded-lg border text-left transition-all ${
            selectedStateFilter === 'ALL'
              ? 'bg-slate-850 border-slate-600 ring-1 ring-slate-500'
              : 'bg-slate-900 border-slate-800 hover:border-slate-700'
          }`}
        >
          <div className="text-[11px] font-medium text-slate-400">Total Cohort</div>
          <div className="text-lg font-bold text-white">{kpis.total}</div>
        </button>

        <button
          type="button"
          onClick={() => setSelectedStateFilter('FILINGS_FOUND')}
          className={`min-h-[44px] p-2.5 rounded-lg border text-left transition-all ${
            selectedStateFilter === 'FILINGS_FOUND'
              ? 'bg-blue-950 border-blue-600 ring-1 ring-blue-500'
              : 'bg-slate-900 border-slate-800 hover:border-slate-700'
          }`}
        >
          <div className="text-[11px] font-medium text-blue-400">Filings Found</div>
          <div className="text-lg font-bold text-blue-300">{kpis.filingsFound}</div>
        </button>

        <button
          type="button"
          onClick={() => setSelectedStateFilter('POLLED_QUIET')}
          className={`min-h-[44px] p-2.5 rounded-lg border text-left transition-all ${
            selectedStateFilter === 'POLLED_QUIET'
              ? 'bg-emerald-950 border-emerald-600 ring-1 ring-emerald-500'
              : 'bg-slate-900 border-slate-800 hover:border-slate-700'
          }`}
        >
          <div className="text-[11px] font-medium text-emerald-400">Polled Quiet</div>
          <div className="text-lg font-bold text-emerald-300">{kpis.polledQuiet}</div>
        </button>

        <button
          type="button"
          onClick={() => setSelectedStateFilter('INCOMPLETE')}
          className={`min-h-[44px] p-2.5 rounded-lg border text-left transition-all ${
            selectedStateFilter === 'INCOMPLETE'
              ? 'bg-amber-950 border-amber-600 ring-1 ring-amber-500'
              : 'bg-slate-900 border-slate-800 hover:border-slate-700'
          }`}
        >
          <div className="text-[11px] font-medium text-amber-400">Incomplete / Gaps</div>
          <div className="text-lg font-bold text-amber-300">{kpis.incomplete}</div>
        </button>

        <button
          type="button"
          onClick={() => setSelectedStateFilter('NOT_POLLED')}
          className={`min-h-[44px] p-2.5 rounded-lg border text-left transition-all ${
            selectedStateFilter === 'NOT_POLLED'
              ? 'bg-slate-800 border-slate-600 ring-1 ring-slate-500'
              : 'bg-slate-900 border-slate-800 hover:border-slate-700'
          }`}
        >
          <div className="text-[11px] font-medium text-slate-400">Not Polled</div>
          <div className="text-lg font-bold text-slate-300">{kpis.notPolled}</div>
        </button>

        <button
          type="button"
          onClick={() => setSelectedStateFilter('READ_ERROR')}
          className={`min-h-[44px] p-2.5 rounded-lg border text-left transition-all ${
            selectedStateFilter === 'READ_ERROR'
              ? 'bg-rose-950 border-rose-600 ring-1 ring-rose-500'
              : 'bg-slate-900 border-slate-800 hover:border-slate-700'
          }`}
        >
          <div className="text-[11px] font-medium text-rose-400">Read Error</div>
          <div className="text-lg font-bold text-rose-300">{kpis.readError}</div>
        </button>
      </div>

      {/* 4. Desktop View: Comprehensive Evidence Table */}
      <div className="mt-5 hidden md:block overflow-x-auto rounded-lg border border-slate-800 bg-slate-900/50">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400">
              <th className="py-3 px-4 font-semibold">Issuer</th>
              <th className="py-3 px-4 font-semibold">Evidence State</th>
              <th className="py-3 px-4 font-semibold">Last Completed Poll</th>
              <th className="py-3 px-4 font-semibold">Accounted Interval</th>
              <th className="py-3 px-4 font-semibold text-center">New Filings</th>
              <th className="py-3 px-4 font-semibold text-center">Deferred / Unres.</th>
              <th className="py-3 px-4 font-semibold">Recorded Gaps</th>
              <th className="py-3 px-4 font-semibold text-right">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {filteredSummaries.map((summary) => {
              const isExpanded = expandedTicker === summary.ticker;
              return (
                <React.Fragment key={summary.ticker}>
                  <tr
                    className={`hover:bg-slate-800/40 transition-colors ${
                      isExpanded ? 'bg-slate-850/60' : ''
                    }`}
                  >
                    {/* Ticker & Name */}
                    <td className="py-3.5 px-4 font-sans">
                      <div className="font-bold text-white tracking-wide">
                        {summary.ticker}
                      </div>
                      <div className="text-[11px] text-slate-400 font-mono">
                        CIK {summary.cik}
                      </div>
                    </td>

                    {/* State Badge */}
                    <td className="py-3.5 px-4 font-sans">
                      {renderStateBadge(summary.state)}
                    </td>

                    {/* Last Completed Poll */}
                    <td className="py-3.5 px-4 text-slate-300">
                      {summary.last_polled_at ? (
                        <div>
                          <div>{formatTime(summary.last_polled_at)}</div>
                          {summary.is_older_receipt && (
                            <span className="inline-block mt-0.5 px-1.5 py-0.2 rounded text-[10px] font-sans font-medium bg-amber-950 text-amber-300 border border-amber-800">
                              Historical Poll
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-slate-500">—</span>
                      )}
                    </td>

                    {/* Accounted Interval */}
                    <td className="py-3.5 px-4 text-slate-300">
                      {summary.interval_start && summary.accounted_through ? (
                        <div className="text-[11px]">
                          <div>{summary.interval_start.slice(11, 19)} &rarr; {summary.accounted_through.slice(11, 19)} UTC</div>
                          {summary.cutoff && summary.accounted_through < summary.cutoff && (
                            <div className="text-[10px] text-amber-400 font-sans">
                              (Lagging cutoff {summary.cutoff.slice(11, 19)} UTC)
                            </div>
                          )}
                        </div>
                      ) : (
                        <span className="text-slate-500">—</span>
                      )}
                    </td>

                    {/* New Filings (Omitted on READ_ERROR or NOT_POLLED) */}
                    <td className="py-3.5 px-4 text-center font-sans">
                      {summary.new_filings_count !== null ? (
                        <div className="inline-flex items-center gap-1 font-semibold text-slate-200">
                          <span>{summary.new_filings_count}</span>
                          {summary.new_filings_count > 0 && (
                            <span className="text-[10px] text-blue-400">
                              ({summary.form4_count} F4, {summary.form4a_count} 4/A)
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-slate-500">—</span>
                      )}
                    </td>

                    {/* Deferred & Unresolved */}
                    <td className="py-3.5 px-4 text-center font-sans">
                      {summary.deferred_count !== null || summary.unresolved_amendments_count !== null ? (
                        <div className="text-xs">
                          {summary.deferred_count ? (
                            <span className="text-amber-400 font-semibold mr-1">
                              {summary.deferred_count} def
                            </span>
                          ) : null}
                          {summary.unresolved_amendments_count ? (
                            <span className="text-rose-400 font-semibold">
                              {summary.unresolved_amendments_count} unres
                            </span>
                          ) : null}
                          {!summary.deferred_count && !summary.unresolved_amendments_count ? (
                            <span className="text-slate-500">0</span>
                          ) : null}
                        </div>
                      ) : (
                        <span className="text-slate-500">—</span>
                      )}
                    </td>

                    {/* Recorded Gaps */}
                    <td className="py-3.5 px-4 font-sans text-xs">
                      {summary.gaps.length > 0 ? (
                        <div className="space-y-0.5">
                          {summary.gaps.map((g, idx) => (
                            <div
                              key={idx}
                              className="text-amber-400 text-[11px] font-mono truncate max-w-[180px]"
                              title={g}
                            >
                              &bull; {g}
                            </div>
                          ))}
                        </div>
                      ) : summary.state === 'READ_ERROR' ? (
                        <span className="text-rose-400 text-xs italic">
                          {summary.read_error_message || 'Receipt payload unreadable'}
                        </span>
                      ) : (
                        <span className="text-slate-500 font-mono">None</span>
                      )}
                    </td>

                    {/* Action Trigger */}
                    <td className="py-3.5 px-4 text-right">
                      <button
                        type="button"
                        onClick={() =>
                          setExpandedTicker(isExpanded ? null : summary.ticker)
                        }
                        className="min-h-[44px] px-3 py-1.5 text-xs font-sans font-medium text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                        aria-expanded={isExpanded}
                      >
                        {isExpanded ? 'Hide' : 'Inspect'}
                      </button>
                    </td>
                  </tr>

                  {/* Expanded Detail Row */}
                  {isExpanded && (
                    <tr className="bg-slate-900/90 border-b border-slate-800">
                      <td colSpan={8} className="p-4 font-sans">
                        <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-3">
                          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
                            <div>
                              <span className="text-sm font-bold text-white">
                                {summary.company_name} ({summary.ticker})
                              </span>
                              <span className="text-xs text-slate-400 ml-2">
                                CIK: {summary.cik}
                              </span>
                            </div>
                            <div className="text-xs text-slate-400 flex items-center gap-2">
                              <span>Artifact: <code className="text-slate-300">{summary.artifact_id || 'N/A'}</code></span>
                              {summary.receipt_sha256 && (
                                <span title={summary.receipt_sha256}>
                                  SHA: <code className="text-slate-300">{summary.receipt_sha256.slice(0, 12)}...</code>
                                </span>
                              )}
                            </div>
                          </div>

                          {/* Accessions List */}
                          <div>
                            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
                              Inventory Accessions ({summary.inventory.length})
                            </h4>
                            {summary.inventory.length === 0 ? (
                              <p className="text-xs text-slate-500 italic">
                                {summary.state === 'NOT_POLLED'
                                  ? 'No completed poll available.'
                                  : summary.state === 'READ_ERROR'
                                  ? 'Cannot display accessions due to read error.'
                                  : 'No eligible Form 4 or 4/A accessions recorded in interval.'}
                              </p>
                            ) : (
                              <div className="space-y-1.5">
                                {summary.inventory.map((item) => (
                                  <div
                                    key={item.accession}
                                    className="flex flex-wrap items-center justify-between p-2 bg-slate-900 border border-slate-800/80 rounded text-xs gap-2"
                                  >
                                    <div className="flex items-center gap-2">
                                      <span className="px-1.5 py-0.5 font-bold rounded bg-slate-800 text-slate-200 text-[11px]">
                                        Form {item.form}
                                      </span>
                                      <span className="font-mono text-slate-300">
                                        {item.accession}
                                      </span>
                                      <span className="text-slate-500">
                                        ({item.accepted_at.slice(11, 19)} UTC)
                                      </span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                      <span
                                        className={`px-2 py-0.5 text-[10px] font-bold rounded ${
                                          item.disposition === 'INGESTED'
                                            ? 'bg-blue-950 text-blue-300 border border-blue-800'
                                            : item.disposition === 'ALREADY_ACCOUNTED'
                                            ? 'bg-slate-800 text-slate-400'
                                            : item.disposition === 'DEFERRED'
                                            ? 'bg-amber-950 text-amber-300 border border-amber-800'
                                            : 'bg-rose-950 text-rose-300 border border-rose-800'
                                        }`}
                                      >
                                        {item.disposition}
                                      </span>
                                      <a
                                        href={buildSecEdgarUrl(summary.cik, item.accession, item.primary_document)}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="min-h-[44px] flex items-center text-blue-400 hover:text-blue-300 underline text-xs"
                                      >
                                        SEC EDGAR &nearr;
                                      </a>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* 5. Mobile View: Touch-Friendly Responsive Cards (<768px) */}
      <div className="mt-4 md:hidden space-y-3">
        {filteredSummaries.map((summary) => {
          const isExpanded = expandedTicker === summary.ticker;
          return (
            <div
              key={summary.ticker}
              className="bg-slate-900 border border-slate-800 rounded-lg p-3.5 space-y-3"
            >
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-base font-bold text-white tracking-wide">
                    {summary.ticker}
                  </span>
                  <span className="text-xs text-slate-400 ml-2 font-mono">
                    CIK {summary.cik}
                  </span>
                </div>
                {renderStateBadge(summary.state)}
              </div>

              {/* Meta Grid */}
              <div className="grid grid-cols-2 gap-2 text-xs pt-1 border-t border-slate-800/80">
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-semibold">
                    Last Polled
                  </span>
                  <span className="text-slate-300 font-mono text-[11px]">
                    {summary.last_polled_at ? formatTime(summary.last_polled_at) : '—'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-semibold">
                    Accounted To
                  </span>
                  <span className="text-slate-300 font-mono text-[11px]">
                    {summary.accounted_through ? summary.accounted_through.slice(11, 19) + ' UTC' : '—'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-semibold">
                    New Filings
                  </span>
                  <span className="text-slate-200 font-bold">
                    {summary.new_filings_count !== null ? summary.new_filings_count : '—'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-semibold">
                    Deferred / Gaps
                  </span>
                  <span className="text-slate-200">
                    {summary.deferred_count !== null ? `${summary.deferred_count} / ${summary.gaps.length}` : '—'}
                  </span>
                </div>
              </div>

              {summary.read_error_message && (
                <div className="text-xs text-rose-400 bg-rose-950/40 p-2 rounded border border-rose-900/60">
                  {summary.read_error_message}
                </div>
              )}

              {/* Mobile Touch Trigger (Min 44px) */}
              <button
                type="button"
                onClick={() => setExpandedTicker(isExpanded ? null : summary.ticker)}
                className="w-full min-h-[44px] flex items-center justify-center text-xs font-semibold text-slate-300 bg-slate-800 hover:bg-slate-750 active:bg-slate-700 rounded transition-colors"
                aria-expanded={isExpanded}
              >
                {isExpanded ? 'Close Details' : `Inspect Accessions (${summary.inventory.length})`}
              </button>

              {/* Mobile Expanded Drawer */}
              {isExpanded && (
                <div className="pt-2 border-t border-slate-800 space-y-2">
                  <div className="text-[11px] text-slate-400">
                    {summary.company_name}
                  </div>
                  {summary.inventory.length === 0 ? (
                    <div className="text-xs text-slate-500 italic">
                      No accessions recorded in interval.
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {summary.inventory.map((item) => (
                        <div
                          key={item.accession}
                          className="p-2.5 bg-slate-950 border border-slate-800 rounded text-xs space-y-1.5"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-white">
                              Form {item.form}
                            </span>
                            <span
                              className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${
                                item.disposition === 'INGESTED'
                                  ? 'bg-blue-950 text-blue-300 border border-blue-800'
                                  : 'bg-slate-800 text-slate-400'
                              }`}
                            >
                              {item.disposition}
                            </span>
                          </div>
                          <div className="font-mono text-[10px] text-slate-400 truncate">
                            {item.accession}
                          </div>
                          <a
                            href={buildSecEdgarUrl(summary.cik, item.accession, item.primary_document)}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="min-h-[44px] flex items-center text-blue-400 hover:text-blue-300 underline text-xs pt-1"
                          >
                            Open on SEC EDGAR &nearr;
                          </a>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer Info */}
      <div className="mt-6 pt-3 border-t border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between text-xs text-slate-500 gap-2 font-mono">
        <div>
          Universe: <span className="text-slate-400">UB-US-EQUITIES-20261025-001</span> (10 Mega-Caps)
        </div>
        <div>
          Rehearsal Horizon: <span className="text-slate-400">Monday, 30 November 2026</span>
        </div>
      </div>
    </div>
  );
};
export default Form4PilotEvidencePreview;
