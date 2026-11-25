'use client';

import React, { useState, useMemo, useEffect } from 'react';
import {
  IssuerEvidenceSummary,
  EvidenceState,
  CohortAccessionItem,
  WorkerReceiptValidation,
  IndependentlyReconciledExport,
} from '../../types/cohortPilotEvidence';
import {
  SYNTHETIC_ISSUER_SUMMARIES,
  SYNTHETIC_COHORT_RECEIPT,
  QUALIFIED_RELEASE_SHA,
  QUALIFIED_TRIAL_ID,
  buildSecEdgarUrl,
  parseReceiptToSummaries,
  validateWorkerReceiptFormat,
  validateReceiptAgainstExpected,
  extractCohortEvidence,
} from '../../fixtures/cohortPilotEvidenceFixtures';

interface Props {
  initialSummaries?: IssuerEvidenceSummary[];
  className?: string;
  isPrivateMode?: boolean;
  isOperatorAuthenticated?: boolean;
  defaultActiveSlot?: string;
}

export const Form4PilotEvidencePreview: React.FC<Props> = ({
  initialSummaries = SYNTHETIC_ISSUER_SUMMARIES,
  className = '',
  isPrivateMode = true,
  isOperatorAuthenticated,
  defaultActiveSlot = 'SLOT-1',
}) => {
  const [summaries, setSummaries] = useState<IssuerEvidenceSummary[]>(initialSummaries);
  const [selectedStateFilter, setSelectedStateFilter] = useState<string>('ALL');
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const [activeSlotId, setActiveSlotId] = useState<string>(defaultActiveSlot);

  // Data source toggle: SYNTHETIC (fixture mode) vs IMPORTED (real receipt mode)
  const [dataSource, setDataSource] = useState<'SYNTHETIC' | 'IMPORTED'>('SYNTHETIC');

  // Custom receipt inputs
  const [customReceiptJson, setCustomReceiptJson] = useState<string>('');
  const [independentExportJson, setIndependentExportJson] = useState<string>('');

  // Validation results
  const [workerValidation, setWorkerValidation] = useState<WorkerReceiptValidation | null>(null);
  const [customReceiptError, setCustomReceiptError] = useState<string | null>(null);

  // Retained state for failed import handling
  const [lastSuccessfulReceiptMeta, setLastSuccessfulReceiptMeta] = useState<{
    receiptId: string;
    slotId: string;
    source: 'SYNTHETIC' | 'IMPORTED';
  } | null>({
    receiptId: 'rcpt-pilot-165-20261130-slot1',
    slotId: 'SLOT-1',
    source: 'SYNTHETIC',
  });
  const [failedImportError, setFailedImportError] = useState<string | null>(null);

  // Check environment for local operator preview enforcement:
  // Allowed locally (localhost/127.0.0.1/dev/test) unless hosted environment enforces operator auth
  const isEnvironmentAuthorized = useMemo(() => {
    if (isOperatorAuthenticated === false) {
      return false;
    }
    if (isOperatorAuthenticated === true) {
      return true;
    }
    if (typeof window !== 'undefined') {
      const hostname = window.location?.hostname || '';
      if (!hostname || hostname === 'localhost' || hostname === '127.0.0.1' || hostname.includes('test')) {
        return true;
      }
      if (process.env.NODE_ENV === 'test' && hostname === 'arxterminal.com') {
        return true;
      }
      return false;
    }
    return true;
  }, [isOperatorAuthenticated]);

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

  // Handle switching back to synthetic fixtures mode
  const handleLoadSynthetic = () => {
    setDataSource('SYNTHETIC');
    setSummaries(SYNTHETIC_ISSUER_SUMMARIES);
    setWorkerValidation(null);
    setCustomReceiptError(null);
    setFailedImportError(null);
    setLastSuccessfulReceiptMeta({
      receiptId: 'rcpt-pilot-165-20261130-slot1',
      slotId: 'SLOT-1',
      source: 'SYNTHETIC',
    });
  };

  // Handle parsing user-pasted / imported worker receipt
  const handleParseCustomReceipt = () => {
    setCustomReceiptError(null);
    setFailedImportError(null);

    try {
      if (!customReceiptJson.trim()) {
        throw new Error('Please paste a valid JSON reconciliation receipt.');
      }

      let parsed: any;
      try {
        parsed = JSON.parse(customReceiptJson);
      } catch (jsonErr: any) {
        throw new Error(`JSON Syntax Error: ${jsonErr.message}`);
      }

      // Optional independently reconciled export
      let independentExport: IndependentlyReconciledExport | null = null;
      if (independentExportJson.trim()) {
        try {
          independentExport = JSON.parse(independentExportJson);
        } catch (indepErr: any) {
          throw new Error(`Independent Export JSON Syntax Error: ${indepErr.message}`);
        }
      }

      // Validate actual worker receipt format, trial, slot, release SHA, issuer/CIK pairs, and receipt fields
      const validation = validateWorkerReceiptFormat(parsed, activeSlotId, independentExport);
      setWorkerValidation(validation);

      if (!validation.is_valid_format && validation.validation_errors.length > 0) {
        // If malformed structure completely prevents parsing issuers
        if (validation.issuer_count === 0) {
          throw new Error(validation.validation_errors.join('; '));
        }
      }

      // Parse into summaries
      const parsedSummaries = parseReceiptToSummaries(parsed, undefined, activeSlotId);
      setSummaries(parsedSummaries);
      setDataSource('IMPORTED');

      const { receiptId, slotId } = extractCohortEvidence(parsed);
      setLastSuccessfulReceiptMeta({
        receiptId: receiptId || 'imported-receipt',
        slotId: slotId || activeSlotId,
        source: 'IMPORTED',
      });
    } catch (err: any) {
      const errMsg = err.message || 'Failed to parse receipt JSON';
      setCustomReceiptError(errMsg);
      setFailedImportError(errMsg);
      // NOTE: Still-visible results are NOT wiped out, they remain belonging to last successful receipt
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
      case 'READ_ERROR':
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-950/80 text-rose-400 border border-rose-700/60">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mr-1.5" />
            READ ERROR
          </span>
        );
      case 'NOT_POLLED':
      default:
        return (
          <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-zinc-800 text-zinc-400 border border-zinc-700">
            <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 mr-1.5" />
            NOT POLLED
          </span>
        );
    }
  };

  // Operator Auth Guard for Hosted Non-Local Sessions
  if (!isEnvironmentAuthorized) {
    return (
      <div className="p-8 bg-slate-900 border border-rose-800 rounded-lg text-slate-100 max-w-2xl mx-auto my-12 text-center space-y-4">
        <div className="w-12 h-12 rounded-full bg-rose-950 border border-rose-600 flex items-center justify-center mx-auto text-rose-400 font-bold text-xl">
          &times;
        </div>
        <h2 className="text-xl font-bold text-rose-300">Operator Authentication Required</h2>
        <p className="text-xs text-slate-300 leading-relaxed">
          The Form 4 Pilot Evidence Preview contains internal SEC audit evidence for ARX Issue #165. Access is restricted to local developer sessions (<code className="text-slate-400 font-mono">localhost</code>) or authenticated operator environments.
        </p>
        <div className="text-[11px] font-mono text-slate-500 pt-2 border-t border-slate-800">
          ARX Gate 3 Boundary Enforcement &bull; Candidate 1063a10f
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header & Guardrail Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-5 shadow-lg">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-white tracking-tight">
                Form 4 Pilot Evidence Preview
              </h1>
              {isPrivateMode && (
                <span className="px-2 py-0.5 text-[10px] font-bold uppercase rounded bg-purple-950 text-purple-300 border border-purple-800">
                  Private Preview
                </span>
              )}
              {dataSource === 'IMPORTED' && (
                <span
                  className={`px-2 py-0.5 text-[10px] font-bold uppercase rounded border ${
                    workerValidation?.reconciliation_status === 'RECONCILED'
                      ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                      : workerValidation?.reconciliation_status === 'RECONCILIATION_DISCREPANCY'
                      ? 'bg-rose-950 text-rose-300 border-rose-800'
                      : 'bg-amber-950 text-amber-300 border-amber-800'
                  }`}
                >
                  {workerValidation?.reconciliation_label || 'Imported — reconciliation pending'}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Read-only evidence inspection for ARX Issue #165 recurring Form 4 cohort pilot.
            </p>
          </div>

          {/* Operational Invariant Tags */}
          <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
            <div className="px-2.5 py-1 rounded bg-slate-950 border border-slate-800 text-slate-300">
              <span className="text-slate-500 mr-1.5">Trial:</span>
              <span className="text-purple-400 font-semibold">{QUALIFIED_TRIAL_ID}</span>
            </div>
            <div className="px-2.5 py-1 rounded bg-slate-950 border border-slate-800 text-slate-300">
              <span className="text-slate-500 mr-1.5">Candidate:</span>
              <span className="text-emerald-400 font-semibold">1063a10f</span>
            </div>
            <div className="px-2.5 py-1 rounded bg-rose-950/60 border border-rose-800 text-rose-300 font-bold">
              Worker: DISABLED (0 SEC Egress)
            </div>
          </div>
        </div>

        {/* Slot Selector & Mode Bar */}
        <div className="mt-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2 font-mono">
            <span className="text-slate-400 font-semibold">Active Expected Slot:</span>
            {(['SLOT-1', 'SLOT-2', 'SLOT-3'] as const).map((slot) => (
              <button
                key={slot}
                type="button"
                onClick={() => setActiveSlotId(slot)}
                className={`px-2.5 py-1 rounded text-xs font-semibold transition-colors min-h-[36px] ${
                  activeSlotId === slot
                    ? 'bg-purple-900/80 text-purple-200 border border-purple-600'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-750'
                }`}
              >
                {slot}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-slate-400 text-xs font-semibold">Evidence Mode:</span>
            <div className="inline-flex rounded-md shadow-sm border border-slate-800 p-0.5 bg-slate-950">
              <button
                type="button"
                onClick={handleLoadSynthetic}
                className={`px-3 py-1 text-xs font-semibold rounded min-h-[36px] transition-colors ${
                  dataSource === 'SYNTHETIC'
                    ? 'bg-purple-600 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Synthetic Fixtures
              </button>
              <button
                type="button"
                onClick={() => setDataSource('IMPORTED')}
                className={`px-3 py-1 text-xs font-semibold rounded min-h-[36px] transition-colors ${
                  dataSource === 'IMPORTED'
                    ? 'bg-purple-600 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Import Worker Receipt
              </button>
            </div>
          </div>
        </div>

        {/* Failed Import Warning with Retained Previous State Alert */}
        {failedImportError && (
          <div className="mt-4 p-3 bg-amber-950/80 border border-amber-600/90 rounded-md text-amber-200 text-xs flex items-start gap-2.5">
            <span className="text-amber-400 font-bold text-sm leading-none mt-0.5">&#9888;</span>
            <div>
              <div className="font-bold text-amber-100">Previous Receipt Retained</div>
              <div className="text-amber-300 mt-0.5">
                The latest import attempt failed: <span className="font-mono text-amber-200">{failedImportError}</span>.
              </div>
              <div className="text-amber-400/90 text-[11px] mt-1 font-mono">
                Currently displaying results from previously loaded receipt:{' '}
                <span className="font-semibold">{lastSuccessfulReceiptMeta?.receiptId || 'initial-receipt'}</span>{' '}
                (Slot: {lastSuccessfulReceiptMeta?.slotId || 'SLOT-1'}, Source: {lastSuccessfulReceiptMeta?.source}).
              </div>
            </div>
          </div>
        )}

        {/* Mode-specific Panels */}
        {dataSource === 'IMPORTED' && (
          <div className="mt-4 p-4 bg-slate-950 border border-slate-800 rounded-md space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                Import Issue #165 Worker Receipt & Reconciled Export
              </h2>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => {
                    setCustomReceiptJson(JSON.stringify(SYNTHETIC_COHORT_RECEIPT, null, 2));
                  }}
                  className="px-2.5 py-1 text-[11px] font-mono text-slate-400 bg-slate-800 hover:bg-slate-750 rounded min-h-[36px]"
                >
                  Fill Sample Worker JSON
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                  1. Worker Reconciliation Receipt JSON <span className="text-rose-400">*</span>
                </label>
                <textarea
                  value={customReceiptJson}
                  onChange={(e) => setCustomReceiptJson(e.target.value)}
                  placeholder="Paste raw sm_reconciliation_receipts JSON row or worker cohort_evidence output..."
                  rows={5}
                  className="w-full p-2.5 text-xs font-mono bg-slate-900 border border-slate-700 rounded text-slate-200 placeholder-slate-600 focus:outline-none focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                  2. Optional Independently Reconciled Export JSON (for counts/gaps comparison)
                </label>
                <textarea
                  value={independentExportJson}
                  onChange={(e) => setIndependentExportJson(e.target.value)}
                  placeholder="Paste independently reconciled export to verify counts and gaps, or leave empty to keep 'Imported — reconciliation pending'..."
                  rows={5}
                  className="w-full p-2.5 text-xs font-mono bg-slate-900 border border-slate-700 rounded text-slate-200 placeholder-slate-600 focus:outline-none focus:border-purple-500"
                />
              </div>
            </div>

            <div className="flex items-center gap-3 pt-1">
              <button
                type="button"
                onClick={handleParseCustomReceipt}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-500 active:bg-purple-700 text-white font-semibold text-xs rounded min-h-[44px] transition-colors"
              >
                Validate & Import Receipt
              </button>
            </div>

            {/* Worker Validation Summary Strip */}
            {workerValidation && (
              <div className="mt-3 p-3 bg-slate-900 border border-slate-800 rounded text-xs space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
                  <span className="font-bold text-slate-200">Worker Receipt Audit Verdict:</span>
                  <span
                    className={`px-2 py-0.5 rounded font-bold uppercase text-[10px] ${
                      workerValidation.is_valid_format
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-rose-950 text-rose-400 border border-rose-800'
                    }`}
                  >
                    {workerValidation.is_valid_format ? 'PASSED QUALIFICATION FORMAT' : 'FORMAT DISCREPANCY'}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
                  <div className="p-1.5 bg-slate-950 rounded border border-slate-800">
                    <span className="text-slate-500 block text-[9px] uppercase">Trial ID</span>
                    <span className={workerValidation.trial_id_matches ? 'text-emerald-400' : 'text-rose-400 font-bold'}>
                      {workerValidation.trial_id || 'MISSING'}
                    </span>
                  </div>
                  <div className="p-1.5 bg-slate-950 rounded border border-slate-800">
                    <span className="text-slate-500 block text-[9px] uppercase">Release SHA</span>
                    <span className={workerValidation.release_sha_matches ? 'text-emerald-400' : 'text-rose-400 font-bold'}>
                      {workerValidation.release_sha ? workerValidation.release_sha.slice(0, 10) : 'MISSING'}
                    </span>
                  </div>
                  <div className="p-1.5 bg-slate-950 rounded border border-slate-800">
                    <span className="text-slate-500 block text-[9px] uppercase">Slot ID</span>
                    <span className={workerValidation.is_prior_slot ? 'text-amber-400 font-bold' : 'text-slate-300'}>
                      {workerValidation.slot_id || 'N/A'}{' '}
                      {workerValidation.is_prior_slot && `(Prior Slot)`}
                    </span>
                  </div>
                  <div className="p-1.5 bg-slate-950 rounded border border-slate-800">
                    <span className="text-slate-500 block text-[9px] uppercase">Issuer/CIK Cohort</span>
                    <span className={workerValidation.issuers_matched ? 'text-emerald-400' : 'text-rose-400 font-bold'}>
                      {workerValidation.issuer_count} / 10 Match
                    </span>
                  </div>
                </div>

                {workerValidation.validation_errors.length > 0 && (
                  <div className="p-2 bg-rose-950/60 border border-rose-800/80 rounded text-rose-300 text-[11px] space-y-1">
                    <div className="font-bold text-rose-200">Validation Errors:</div>
                    <ul className="list-disc list-inside">
                      {workerValidation.validation_errors.map((err, i) => (
                        <li key={i}>{err}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Discrepancies vs Independent Export */}
                {workerValidation.discrepancies.length > 0 && (
                  <div className="p-2 bg-amber-950/60 border border-amber-800/80 rounded text-amber-300 text-[11px] space-y-1">
                    <div className="font-bold text-amber-200">Reconciliation Discrepancies vs Independent Export:</div>
                    <ul className="list-disc list-inside">
                      {workerValidation.discrepancies.map((d, i) => (
                        <li key={i}>
                          <span className="font-semibold text-white">{d.ticker}</span> [{d.field}]: expected{' '}
                          <code className="text-amber-100">{JSON.stringify(d.expected)}</code>, got{' '}
                          <code className="text-rose-300">{JSON.stringify(d.actual)}</code>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* KPI & Filter Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        <button
          type="button"
          onClick={() => setSelectedStateFilter('ALL')}
          className={`p-3 rounded-lg border text-left transition-colors min-h-[44px] ${
            selectedStateFilter === 'ALL'
              ? 'bg-purple-950/60 border-purple-500 text-purple-200'
              : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-850'
          }`}
        >
          <div className="text-[10px] uppercase font-bold tracking-wider">Total Cohort</div>
          <div className="text-xl font-bold text-white mt-1">{kpis.total}</div>
        </button>

        <button
          type="button"
          onClick={() => setSelectedStateFilter('FILINGS_FOUND')}
          className={`p-3 rounded-lg border text-left transition-colors min-h-[44px] ${
            selectedStateFilter === 'FILINGS_FOUND'
              ? 'bg-blue-950/80 border-blue-500 text-blue-200'
              : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-850'
          }`}
        >
          <div className="text-[10px] uppercase font-bold tracking-wider">Filings Found</div>
          <div className="text-xl font-bold text-blue-400 mt-1">{kpis.filingsFound}</div>
        </button>

        <button
          type="button"
          onClick={() => setSelectedStateFilter('POLLED_QUIET')}
          className={`p-3 rounded-lg border text-left transition-colors min-h-[44px] ${
            selectedStateFilter === 'POLLED_QUIET'
              ? 'bg-emerald-950/80 border-emerald-500 text-emerald-200'
              : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-850'
          }`}
        >
          <div className="text-[10px] uppercase font-bold tracking-wider">Quiet Polls</div>
          <div className="text-xl font-bold text-emerald-400 mt-1">{kpis.polledQuiet}</div>
        </button>

        <button
          type="button"
          onClick={() => setSelectedStateFilter('INCOMPLETE')}
          className={`p-3 rounded-lg border text-left transition-colors min-h-[44px] ${
            selectedStateFilter === 'INCOMPLETE'
              ? 'bg-amber-950/80 border-amber-500 text-amber-200'
              : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-850'
          }`}
        >
          <div className="text-[10px] uppercase font-bold tracking-wider">Incomplete / Gaps</div>
          <div className="text-xl font-bold text-amber-400 mt-1">{kpis.incomplete}</div>
        </button>

        <button
          type="button"
          onClick={() => setSelectedStateFilter('NOT_POLLED')}
          className={`p-3 rounded-lg border text-left transition-colors min-h-[44px] ${
            selectedStateFilter === 'NOT_POLLED'
              ? 'bg-zinc-800/80 border-zinc-500 text-zinc-200'
              : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-850'
          }`}
        >
          <div className="text-[10px] uppercase font-bold tracking-wider">Not Polled</div>
          <div className="text-xl font-bold text-zinc-300 mt-1">{kpis.notPolled}</div>
        </button>

        <button
          type="button"
          onClick={() => setSelectedStateFilter('READ_ERROR')}
          className={`p-3 rounded-lg border text-left transition-colors min-h-[44px] ${
            selectedStateFilter === 'READ_ERROR'
              ? 'bg-rose-950/80 border-rose-500 text-rose-200'
              : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-850'
          }`}
        >
          <div className="text-[10px] uppercase font-bold tracking-wider">Read Error</div>
          <div className="text-xl font-bold text-rose-400 mt-1">{kpis.readError}</div>
        </button>
      </div>

      {/* Desktop Evidence Table */}
      <div className="hidden md:block overflow-x-auto bg-slate-900 border border-slate-800 rounded-lg shadow">
        <table className="w-full text-left text-xs text-slate-300 border-collapse">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-950/60 font-mono text-[11px] text-slate-400 uppercase">
              <th className="py-3 px-4">Issuer</th>
              <th className="py-3 px-4">Evidence State</th>
              <th className="py-3 px-4">Poll Time (observed_at)</th>
              <th className="py-3 px-4">Coverage Cutoff</th>
              <th className="py-3 px-4">Accounted To</th>
              <th className="py-3 px-4 text-center">New Filings (4 / 4-A)</th>
              <th className="py-3 px-4 text-center">Deferred / Gaps</th>
              <th className="py-3 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {filteredSummaries.map((summary) => {
              const isExpanded = expandedTicker === summary.ticker;
              return (
                <React.Fragment key={summary.ticker}>
                  <tr
                    className={`hover:bg-slate-850/60 transition-colors ${
                      summary.state === 'READ_ERROR'
                        ? 'bg-rose-950/10'
                        : summary.state === 'INCOMPLETE'
                        ? 'bg-amber-950/10'
                        : ''
                    }`}
                  >
                    <td className="py-3 px-4 font-mono">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-white text-sm">{summary.ticker}</span>
                        {summary.is_prior_slot && (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase bg-amber-950 text-amber-300 border border-amber-800">
                            Prior Slot ({summary.slot_id || 'SLOT-1'})
                          </span>
                        )}
                        {summary.is_older_receipt && !summary.is_prior_slot && (
                          <span className="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase bg-amber-950 text-amber-300 border border-amber-800">
                            Historical Poll
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-slate-500 truncate max-w-[160px]">
                        CIK {summary.cik} &bull; {summary.company_name}
                      </div>
                    </td>

                    <td className="py-3 px-4">{renderStateBadge(summary.state)}</td>

                    {/* Poll Time (observed_at) */}
                    <td className="py-3 px-4 font-mono text-[11px]">
                      {summary.observed_at ? (
                        <div>
                          <span className="text-slate-200">{summary.observed_at.slice(11, 19)} UTC</span>
                          <span className="text-slate-500 block text-[9px]">{summary.observed_at.slice(0, 10)}</span>
                        </div>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>

                    {/* Coverage Cutoff */}
                    <td className="py-3 px-4 font-mono text-[11px]">
                      {summary.cutoff ? (
                        <div>
                          <span className="text-slate-300">{summary.cutoff.slice(11, 19)} UTC</span>
                          <span className="text-slate-500 block text-[9px]">{summary.cutoff.slice(0, 10)}</span>
                        </div>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>

                    {/* Accounted To */}
                    <td className="py-3 px-4 font-mono text-[11px]">
                      {summary.accounted_through ? (
                        <span
                          className={
                            summary.cutoff && summary.accounted_through < summary.cutoff
                              ? 'text-amber-400 font-semibold'
                              : 'text-slate-300'
                          }
                        >
                          {summary.accounted_through.slice(11, 19)} UTC
                        </span>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>

                    {/* New Filings */}
                    <td className="py-3 px-4 text-center font-mono">
                      {summary.new_filings_count !== null ? (
                        <div className="inline-flex items-center gap-1 font-bold">
                          <span className={summary.new_filings_count > 0 ? 'text-blue-400' : 'text-slate-400'}>
                            {summary.new_filings_count}
                          </span>
                          <span className="text-slate-500 text-[10px] font-normal">
                            ({summary.form4_count || 0} / {summary.form4a_count || 0})
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-600 font-bold">—</span>
                      )}
                    </td>

                    {/* Deferred / Gaps */}
                    <td className="py-3 px-4 text-center font-mono">
                      {summary.state === 'NOT_POLLED' || summary.state === 'READ_ERROR' ? (
                        <span className="text-slate-600">—</span>
                      ) : (
                        <div className="inline-flex items-center gap-1.5">
                          <span className={summary.deferred_count ? 'text-amber-400 font-bold' : 'text-slate-500'}>
                            {summary.deferred_count || 0}
                          </span>
                          <span className="text-slate-600">/</span>
                          <span className={summary.gaps.length > 0 ? 'text-amber-400 font-bold' : 'text-slate-500'}>
                            {summary.gaps.length}
                          </span>
                        </div>
                      )}
                    </td>

                    {/* Action */}
                    <td className="py-3 px-4 text-right">
                      <button
                        type="button"
                        onClick={() => setExpandedTicker(isExpanded ? null : summary.ticker)}
                        className="px-3 py-1.5 text-xs font-semibold rounded bg-slate-800 hover:bg-slate-750 text-slate-300 border border-slate-700 transition-colors min-h-[44px]"
                        aria-expanded={isExpanded}
                      >
                        {isExpanded ? 'Collapse' : `Inspect (${summary.inventory.length})`}
                      </button>
                    </td>
                  </tr>

                  {/* Desktop Expanded Inspection Row */}
                  {isExpanded && (
                    <tr className="bg-slate-950/90 border-b border-slate-800">
                      <td colSpan={8} className="p-4 space-y-3 font-mono text-xs">
                        <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-800">
                          <div>
                            <span className="font-bold text-white text-sm">{summary.ticker} Accession Audit Trail</span>
                            <span className="text-slate-500 ml-2">({summary.company_name})</span>
                          </div>
                          <div className="text-[11px] text-slate-400">
                            Artifact ID: <span className="text-slate-200">{summary.artifact_id || 'N/A'}</span>
                          </div>
                        </div>

                        {summary.read_error_message && (
                          <div className="p-2.5 bg-rose-950/80 border border-rose-800 rounded text-rose-300">
                            <strong>Audit Error:</strong> {summary.read_error_message}
                          </div>
                        )}

                        {summary.gaps.length > 0 && (
                          <div className="p-2.5 bg-amber-950/60 border border-amber-800 rounded text-amber-300">
                            <strong>Recorded Coverage Gaps:</strong> {summary.gaps.join(', ')}
                          </div>
                        )}

                        {summary.inventory.length === 0 ? (
                          <div className="text-slate-500 italic py-2">
                            {summary.state === 'NOT_POLLED'
                              ? 'No completed receipt exists for this issuer in current cohort execution.'
                              : 'Retained catalog accounts for interval with 0 new eligible Form 4 or 4/A accessions.'}
                          </div>
                        ) : (
                          <div className="overflow-x-auto">
                            <table className="w-full text-left text-[11px] text-slate-300">
                              <thead>
                                <tr className="border-b border-slate-800 text-slate-500 uppercase">
                                  <th className="py-2 px-3">Form</th>
                                  <th className="py-2 px-3">Accession Number</th>
                                  <th className="py-2 px-3">Filing Date</th>
                                  <th className="py-2 px-3">Accepted At</th>
                                  <th className="py-2 px-3">Disposition</th>
                                  <th className="py-2 px-3">SEC Source Link</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-800/40">
                                {summary.inventory.map((item) => (
                                  <tr key={item.accession} className="hover:bg-slate-900">
                                    <td className="py-2 px-3 font-bold text-white">Form {item.form}</td>
                                    <td className="py-2 px-3 font-mono text-slate-400">{item.accession}</td>
                                    <td className="py-2 px-3">{item.filing_date}</td>
                                    <td className="py-2 px-3 text-slate-400">{item.accepted_at}</td>
                                    <td className="py-2 px-3">
                                      <span
                                        className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                          item.disposition === 'INGESTED'
                                            ? 'bg-blue-950 text-blue-300 border border-blue-800'
                                            : item.disposition === 'DEFERRED'
                                            ? 'bg-amber-950 text-amber-300 border border-amber-800'
                                            : 'bg-slate-800 text-slate-400'
                                        }`}
                                      >
                                        {item.disposition}
                                      </span>
                                    </td>
                                    <td className="py-2 px-3">
                                      <a
                                        href={buildSecEdgarUrl(summary.cik, item.accession, item.primary_document)}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-blue-400 hover:text-blue-300 underline font-semibold flex items-center gap-1 min-h-[36px]"
                                      >
                                        <span>SEC EDGAR Document</span>
                                        <span>&nearr;</span>
                                      </a>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Mobile Card Layout with Minimum 44px Touch Targets */}
      <div className="md:hidden space-y-3">
        {filteredSummaries.map((summary) => {
          const isExpanded = expandedTicker === summary.ticker;
          return (
            <div
              key={summary.ticker}
              className={`p-4 bg-slate-900 border rounded-lg space-y-3 shadow ${
                summary.state === 'READ_ERROR'
                  ? 'border-rose-900/80'
                  : summary.state === 'INCOMPLETE'
                  ? 'border-amber-900/80'
                  : 'border-slate-800'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-base font-bold text-white font-mono">{summary.ticker}</span>
                    {summary.is_prior_slot && (
                      <span className="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase bg-amber-950 text-amber-300 border border-amber-800">
                        Prior Slot ({summary.slot_id || 'SLOT-1'})
                      </span>
                    )}
                    {summary.is_older_receipt && !summary.is_prior_slot && (
                      <span className="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase bg-amber-950 text-amber-300 border border-amber-800">
                        Historical
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-slate-400 font-mono">
                    CIK {summary.cik} &bull; {summary.company_name}
                  </div>
                </div>
                <div>{renderStateBadge(summary.state)}</div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-slate-950 p-2.5 rounded border border-slate-800/80">
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-semibold">
                    Poll Time (observed_at)
                  </span>
                  <span className="text-slate-200">
                    {summary.observed_at ? summary.observed_at.slice(11, 19) + ' UTC' : '—'}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-semibold">
                    Coverage Cutoff
                  </span>
                  <span className="text-slate-300 font-mono text-[11px]">
                    {summary.cutoff ? summary.cutoff.slice(11, 19) + ' UTC' : '—'}
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
                <div className="text-xs text-rose-400 bg-rose-950/40 p-2 rounded border border-rose-900/60 font-mono">
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
                            className="min-h-[44px] flex items-center text-blue-400 hover:text-blue-300 underline text-xs pt-1 font-mono"
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
