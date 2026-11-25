/**
 * ARX Terminal — Cohort Pilot Evidence Synthetic Fixtures & Reconciliation Validator
 * Issue #165 / Trial ARX-TRIAL-10-ISSUER-001
 * Aligned with candidate 1063a10fed1ed312feadeb671980a642e757e601
 */

import {
  CohortReconciliationReceipt,
  CohortEvidencePayload,
  IssuerEvidenceSummary,
  RawIssuerReceipt,
  ReceiptValidationReport,
  CohortValidationDiscrepancy,
  WorkerReceiptValidation,
  IndependentlyReconciledExport,
} from '../types/cohortPilotEvidence';

export interface GovernedIssuerDefinition {
  ticker: string;
  cik: string;
  company_name: string;
}

export const QUALIFIED_RELEASE_SHA = '1063a10fed1ed312feadeb671980a642e757e601';
export const QUALIFIED_TRIAL_ID = 'ARX-TRIAL-10-ISSUER-001';
export const QUALIFIED_UNIVERSE = 'UB-US-EQUITIES-20261025-001';

export const COHORT_CIKS: Record<string, string> = {
  NVDA: '0001045810',
  AAPL: '0000320193',
  MSFT: '0000789019',
  AMZN: '0001018724',
  GOOGL: '0001652044',
  META: '0001326801',
  TSLA: '0001318605',
  'BRK-B': '0001067983',
  JPM: '0000019617',
  V: '0001403161',
};

export const GOVERNED_COHORT_ISSUERS: GovernedIssuerDefinition[] = [
  { ticker: 'NVDA', cik: '0001045810', company_name: 'NVIDIA Corporation' },
  { ticker: 'AAPL', cik: '0000320193', company_name: 'Apple Inc.' },
  { ticker: 'MSFT', cik: '0000789019', company_name: 'Microsoft Corporation' },
  { ticker: 'AMZN', cik: '0001018724', company_name: 'Amazon.com, Inc.' },
  { ticker: 'GOOGL', cik: '0001652044', company_name: 'Alphabet Inc.' },
  { ticker: 'META', cik: '0001326801', company_name: 'Meta Platforms, Inc.' },
  { ticker: 'TSLA', cik: '0001318605', company_name: 'Tesla, Inc.' },
  { ticker: 'BRK-B', cik: '0001067983', company_name: 'Berkshire Hathaway Inc.' },
  { ticker: 'JPM', cik: '0000019617', company_name: 'JPMorgan Chase & Co.' },
  { ticker: 'V', cik: '0001403161', company_name: 'Visa Inc.' },
];

/**
 * Builds authoritative SEC EDGAR document URL.
 */
export function buildSecEdgarUrl(cik: string, accession: string, primaryDoc?: string): string {
  const cleanCik = String(parseInt(cik, 10));
  const compactAccession = accession.replace(/-/g, '');
  if (primaryDoc) {
    return `https://www.sec.gov/Archives/edgar/data/${cleanCik}/${compactAccession}/${primaryDoc}`;
  }
  return `https://www.sec.gov/Archives/edgar/data/${cleanCik}/${compactAccession}/${accession}-index.htm`;
}

/**
 * Extracts cohort evidence payload from raw worker receipt object or db row.
 */
export function extractCohortEvidence(raw: any): {
  evidence: CohortEvidencePayload | null;
  receiptId: string | null;
  releaseSha: string | null;
  trialId: string | null;
  slotId: string | null;
} {
  if (!raw || typeof raw !== 'object') {
    return { evidence: null, receiptId: null, releaseSha: null, trialId: null, slotId: null };
  }

  // Check 1: In failure_reasons.cohort_evidence (PostgreSQL sm_reconciliation_receipts row)
  if (raw.failure_reasons?.cohort_evidence?.issuers) {
    const ce = raw.failure_reasons.cohort_evidence;
    return {
      evidence: ce,
      receiptId: raw.receipt_id || null,
      releaseSha: ce.release_sha || raw.release_sha || null,
      trialId: ce.trial_id || raw.trial_id || null,
      slotId: ce.slot_id || raw.slot_id || null,
    };
  }

  // Check 2: At root cohort_evidence (in-memory candidate output)
  if (raw.cohort_evidence?.issuers) {
    const ce = raw.cohort_evidence;
    return {
      evidence: ce,
      receiptId: raw.receipt_id || null,
      releaseSha: ce.release_sha || raw.release_sha || null,
      trialId: ce.trial_id || raw.trial_id || null,
      slotId: ce.slot_id || raw.slot_id || null,
    };
  }

  // Check 3: Raw evidence object directly
  if (raw.issuers && typeof raw.issuers === 'object') {
    return {
      evidence: {
        version: raw.version || 1,
        trial_id: raw.trial_id || '',
        slot_id: raw.slot_id || '',
        release_sha: raw.release_sha || '',
        issuers: raw.issuers,
      },
      receiptId: raw.receipt_id || null,
      releaseSha: raw.release_sha || null,
      trialId: raw.trial_id || null,
      slotId: raw.slot_id || null,
    };
  }

  return { evidence: null, receiptId: null, releaseSha: null, trialId: null, slotId: null };
}

/**
 * Full synthetic reconciliation receipt modeled on candidate 1063a10f output.
 */
export const SYNTHETIC_COHORT_RECEIPT: CohortReconciliationReceipt = {
  receipt_id: 'rcpt-pilot-165-20261130-slot1',
  trial_id: 'ARX-TRIAL-10-ISSUER-001',
  slot_id: 'SLOT-1',
  created_at: '2026-11-30T14:26:00Z',
  interval_start: '2026-11-30T14:00:00Z',
  interval_end: '2026-11-30T14:25:00Z',
  coverage_status: 'PARTIAL',
  reconciliation_hash: '9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b',
  release_sha: '1063a10fed1ed312feadeb671980a642e757e601',
  failure_reasons: {
    trial_id: 'ARX-TRIAL-10-ISSUER-001',
    cohort_evidence: {
      trial_id: 'ARX-TRIAL-10-ISSUER-001',
      slot_id: 'SLOT-1',
      universe_id: 'UB-US-EQUITIES-20261025-001',
      release_sha: '1063a10fed1ed312feadeb671980a642e757e601',
      issuers: {
        NVDA: {
          ticker: 'NVDA',
          cik: '0001045810',
          observed_at: '2026-11-30T14:20:15Z',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:00Z',
          artifact_id: 'evt-audit-1795620716-nvda',
          catalog_sha256: 'a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e',
          catalog_gaps: [],
          inventory: [
            {
              accession: '0001045810-26-000045',
              form: '4',
              filing_date: '2026-11-30',
              accepted_at: '2026-11-30T14:04:12Z',
              primary_document: 'form4.xml',
              disposition: 'INGESTED',
              source_sha256: 'fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210',
            },
            {
              accession: '0001045810-26-000046',
              form: '4/A',
              filing_date: '2026-11-30',
              accepted_at: '2026-11-30T14:10:05Z',
              primary_document: 'amendment.xml',
              disposition: 'INGESTED',
              source_sha256: '1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
            },
          ],
          gaps: [],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:20:00Z',
          status: 'CATALOG_ACCOUNTED',
          sha256: 'b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5',
        },
        AAPL: {
          ticker: 'AAPL',
          cik: '0000320193',
          observed_at: '2026-11-30T14:20:25Z',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:00Z',
          artifact_id: 'evt-audit-1795620716-aapl',
          catalog_sha256: 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4',
          catalog_gaps: [],
          inventory: [],
          gaps: [],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:20:00Z',
          status: 'CATALOG_ACCOUNTED',
          sha256: 'd5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6',
        },
        MSFT: {
          ticker: 'MSFT',
          cik: '0000789019',
          observed_at: '2026-11-30T14:21:05Z',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:00Z',
          artifact_id: 'evt-audit-1795620716-msft',
          catalog_sha256: 'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6',
          catalog_gaps: [],
          inventory: [
            {
              accession: '0000789019-26-000088',
              form: '4',
              filing_date: '2026-11-30',
              accepted_at: '2026-11-30T14:15:30Z',
              primary_document: 'doc4.xml',
              disposition: 'DEFERRED',
              source_sha256: null,
            },
          ],
          gaps: ['DEFERRED:0000789019-26-000088'],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:00:00Z', // held at previous watermark due to gap
          status: 'PARTIAL',
          sha256: 'f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7',
        },
        AMZN: {
          ticker: 'AMZN',
          cik: '0001018724',
          observed_at: '2026-11-30T14:21:40Z',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:00Z',
          artifact_id: 'evt-audit-1795620716-amzn',
          catalog_sha256: '0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b',
          catalog_gaps: [],
          inventory: [
            {
              accession: '0001018724-26-000042',
              form: '4/A',
              filing_date: '2026-11-30',
              accepted_at: '2026-11-30T14:18:10Z',
              primary_document: 'amend.xml',
              disposition: 'AMENDMENT_UNRESOLVED',
              source_sha256: 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2',
            },
          ],
          gaps: ['AMENDMENT_UNRESOLVED:0001018724-26-000042'],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:00:00Z',
          status: 'PARTIAL',
          sha256: '1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b',
        },
        GOOGL: {
          ticker: 'GOOGL',
          cik: '0001652044',
          observed_at: '2026-11-30T09:22:10Z', // historical earlier timestamp
          interval_start: '2026-11-30T09:00:00Z',
          cutoff: '2026-11-30T09:20:00Z',
          artifact_id: 'evt-audit-1795600000-googl',
          catalog_sha256: '2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c',
          catalog_gaps: [],
          inventory: [],
          gaps: [],
          previous_watermark: '2026-11-30T09:00:00Z',
          accounted_through: '2026-11-30T09:20:00Z',
          status: 'CATALOG_ACCOUNTED',
          sha256: '3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d',
        },
        META: {
          ticker: 'META',
          cik: '0001326801',
          observed_at: '2026-11-30T14:22:50Z',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:00Z',
          artifact_id: 'evt-audit-1795620716-meta',
          catalog_sha256: '4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e',
          catalog_gaps: ['HISTORICAL_PAGE_MAY_OVERLAP'],
          inventory: [],
          gaps: ['HISTORICAL_PAGE_MAY_OVERLAP'],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:00:00Z',
          status: 'PARTIAL',
          sha256: '5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f',
        },
        // TSLA is deliberately omitted from receipt to simulate NOT_POLLED
        'BRK-B': {
          ticker: 'BRK-B',
          cik: '0001067983',
          observed_at: '2026-11-30T14:23:30Z',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:00Z',
          artifact_id: 'evt-audit-1795620716-brkb',
          catalog_sha256: 'corrupted-sha256-hash',
          catalog_gaps: [],
          inventory: [],
          gaps: [],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: null,
          status: 'READ_ERROR',
          sha256: '6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a',
        },
        JPM: {
          ticker: 'JPM',
          cik: '0000019617',
          observed_at: '2026-11-30T14:24:00Z',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:00Z',
          artifact_id: 'evt-audit-1795620716-jpm',
          catalog_sha256: '7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b',
          catalog_gaps: [],
          inventory: [],
          gaps: [],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:20:00Z',
          status: 'CATALOG_ACCOUNTED',
          sha256: '8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c',
        },
        V: {
          ticker: 'V',
          cik: '0001403161',
          observed_at: '2026-11-30T14:24:20Z',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:00Z',
          artifact_id: 'evt-audit-1795620716-v',
          catalog_sha256: '9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d',
          catalog_gaps: [],
          inventory: [],
          gaps: [],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:20:00Z',
          status: 'CATALOG_ACCOUNTED',
          sha256: '0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e',
        },
      },
    },
  },
};

/**
 * Authoritative Synthetic Summaries representing each required evidence state.
 */
export const SYNTHETIC_ISSUER_SUMMARIES: IssuerEvidenceSummary[] = [
  {
    ticker: 'NVDA',
    cik: '0001045810',
    company_name: 'NVIDIA Corporation',
    state: 'FILINGS_FOUND',
    observed_at: '2026-11-30T14:20:15Z',
    cutoff: '2026-11-30T14:20:00Z',
    last_polled_at: '2026-11-30T14:20:15Z',
    interval_start: '2026-11-30T14:00:00Z',
    accounted_through: '2026-11-30T14:20:00Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: 'a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e',
    catalog_gaps: [],
    new_filings_count: 2,
    form4_count: 1,
    form4a_count: 1,
    deferred_count: 0,
    unresolved_amendments_count: 0,
    gaps: [],
    inventory: [
      {
        accession: '0001045810-26-000045',
        form: '4',
        filing_date: '2026-11-30',
        accepted_at: '2026-11-30T14:04:12Z',
        primary_document: 'form4.xml',
        disposition: 'INGESTED',
        source_sha256: 'fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210',
      },
      {
        accession: '0001045810-26-000046',
        form: '4/A',
        filing_date: '2026-11-30',
        accepted_at: '2026-11-30T14:10:05Z',
        primary_document: 'amendment.xml',
        disposition: 'INGESTED',
        source_sha256: '1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
      },
    ],
    artifact_id: 'evt-audit-1795620716-nvda',
    receipt_sha256: 'b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    slot_id: 'SLOT-1',
    is_older_receipt: false,
  },
  {
    ticker: 'AAPL',
    cik: '0000320193',
    company_name: 'Apple Inc.',
    state: 'POLLED_QUIET',
    observed_at: '2026-11-30T14:20:25Z',
    cutoff: '2026-11-30T14:20:00Z',
    last_polled_at: '2026-11-30T14:20:25Z',
    interval_start: '2026-11-30T14:00:00Z',
    accounted_through: '2026-11-30T14:20:00Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4',
    catalog_gaps: [],
    new_filings_count: 0,
    form4_count: 0,
    form4a_count: 0,
    deferred_count: 0,
    unresolved_amendments_count: 0,
    gaps: [],
    inventory: [],
    artifact_id: 'evt-audit-1795620716-aapl',
    receipt_sha256: 'd5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    slot_id: 'SLOT-1',
    is_older_receipt: false,
  },
  {
    ticker: 'MSFT',
    cik: '0000789019',
    company_name: 'Microsoft Corporation',
    state: 'INCOMPLETE',
    observed_at: '2026-11-30T14:21:05Z',
    cutoff: '2026-11-30T14:20:00Z',
    last_polled_at: '2026-11-30T14:21:05Z',
    interval_start: '2026-11-30T14:00:00Z',
    accounted_through: '2026-11-30T14:00:00Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: 'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6',
    catalog_gaps: [],
    new_filings_count: 0,
    form4_count: 0,
    form4a_count: 0,
    deferred_count: 1,
    unresolved_amendments_count: 0,
    gaps: ['DEFERRED:0000789019-26-000088'],
    inventory: [
      {
        accession: '0000789019-26-000088',
        form: '4',
        filing_date: '2026-11-30',
        accepted_at: '2026-11-30T14:15:30Z',
        primary_document: 'doc4.xml',
        disposition: 'DEFERRED',
        source_sha256: null,
      },
    ],
    artifact_id: 'evt-audit-1795620716-msft',
    receipt_sha256: 'f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    slot_id: 'SLOT-1',
    is_older_receipt: false,
  },
  {
    ticker: 'AMZN',
    cik: '0001018724',
    company_name: 'Amazon.com, Inc.',
    state: 'INCOMPLETE',
    observed_at: '2026-11-30T14:21:40Z',
    cutoff: '2026-11-30T14:20:00Z',
    last_polled_at: '2026-11-30T14:21:40Z',
    interval_start: '2026-11-30T14:00:00Z',
    accounted_through: '2026-11-30T14:00:00Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: '0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b',
    catalog_gaps: [],
    new_filings_count: 0,
    form4_count: 0,
    form4a_count: 0,
    deferred_count: 0,
    unresolved_amendments_count: 1,
    gaps: ['AMENDMENT_UNRESOLVED:0001018724-26-000042'],
    inventory: [
      {
        accession: '0001018724-26-000042',
        form: '4/A',
        filing_date: '2026-11-30',
        accepted_at: '2026-11-30T14:18:10Z',
        primary_document: 'amend.xml',
        disposition: 'AMENDMENT_UNRESOLVED',
        source_sha256: 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2',
      },
    ],
    artifact_id: 'evt-audit-1795620716-amzn',
    receipt_sha256: '1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    slot_id: 'SLOT-1',
    is_older_receipt: false,
  },
  {
    ticker: 'GOOGL',
    cik: '0001652044',
    company_name: 'Alphabet Inc.',
    state: 'POLLED_QUIET',
    observed_at: '2026-11-30T09:22:10Z',
    cutoff: '2026-11-30T09:20:00Z',
    last_polled_at: '2026-11-30T09:22:10Z',
    interval_start: '2026-11-30T09:00:00Z',
    accounted_through: '2026-11-30T09:20:00Z',
    previous_watermark: '2026-11-30T09:00:00Z',
    catalog_sha256: '2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c',
    catalog_gaps: [],
    new_filings_count: 0,
    form4_count: 0,
    form4a_count: 0,
    deferred_count: 0,
    unresolved_amendments_count: 0,
    gaps: [],
    inventory: [],
    artifact_id: 'evt-audit-1795600000-googl',
    receipt_sha256: '3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    slot_id: 'SLOT-1',
    is_older_receipt: true, // Marked historical for demonstration
  },
  {
    ticker: 'META',
    cik: '0001326801',
    company_name: 'Meta Platforms, Inc.',
    state: 'INCOMPLETE',
    observed_at: '2026-11-30T14:22:50Z',
    cutoff: '2026-11-30T14:20:00Z',
    last_polled_at: '2026-11-30T14:22:50Z',
    interval_start: '2026-11-30T14:00:00Z',
    accounted_through: '2026-11-30T14:00:00Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: '4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e',
    catalog_gaps: ['HISTORICAL_PAGE_MAY_OVERLAP'],
    new_filings_count: 0,
    form4_count: 0,
    form4a_count: 0,
    deferred_count: 0,
    unresolved_amendments_count: 0,
    gaps: ['HISTORICAL_PAGE_MAY_OVERLAP'],
    inventory: [],
    artifact_id: 'evt-audit-1795620716-meta',
    receipt_sha256: '5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    slot_id: 'SLOT-1',
    is_older_receipt: false,
  },
  {
    ticker: 'TSLA',
    cik: '0001318605',
    company_name: 'Tesla, Inc.',
    state: 'NOT_POLLED',
    observed_at: null,
    cutoff: null,
    last_polled_at: null,
    interval_start: null,
    accounted_through: null,
    previous_watermark: null,
    catalog_sha256: null,
    catalog_gaps: [],
    new_filings_count: null, // Strictly null
    form4_count: null,
    form4a_count: null,
    deferred_count: null,
    unresolved_amendments_count: null,
    gaps: [],
    inventory: [],
    artifact_id: null,
    receipt_sha256: null,
    receipt_id: null,
    slot_id: null,
    is_older_receipt: false,
  },
  {
    ticker: 'BRK-B',
    cik: '0001067983',
    company_name: 'Berkshire Hathaway Inc.',
    state: 'READ_ERROR',
    observed_at: '2026-11-30T14:23:30Z',
    cutoff: '2026-11-30T14:20:00Z',
    last_polled_at: '2026-11-30T14:23:30Z',
    interval_start: '2026-11-30T14:00:00Z',
    accounted_through: null,
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: 'corrupted-sha256-hash',
    catalog_gaps: [],
    new_filings_count: null, // Strictly null on READ_ERROR
    form4_count: null,
    form4a_count: null,
    deferred_count: null,
    unresolved_amendments_count: null,
    gaps: [],
    inventory: [],
    artifact_id: 'evt-audit-1795620716-brkb',
    receipt_sha256: '6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    slot_id: 'SLOT-1',
    is_older_receipt: false,
    read_error_message: 'SEC Catalog Retained Hash Mismatch: expected sha256 digest does not match audit payload.',
  },
  {
    ticker: 'JPM',
    cik: '0000019617',
    company_name: 'JPMorgan Chase & Co.',
    state: 'POLLED_QUIET',
    observed_at: '2026-11-30T14:24:00Z',
    cutoff: '2026-11-30T14:20:00Z',
    last_polled_at: '2026-11-30T14:24:00Z',
    interval_start: '2026-11-30T14:00:00Z',
    accounted_through: '2026-11-30T14:20:00Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: '7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b',
    catalog_gaps: [],
    new_filings_count: 0,
    form4_count: 0,
    form4a_count: 0,
    deferred_count: 0,
    unresolved_amendments_count: 0,
    gaps: [],
    inventory: [],
    artifact_id: 'evt-audit-1795620716-jpm',
    receipt_sha256: '8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    slot_id: 'SLOT-1',
    is_older_receipt: false,
  },
  {
    ticker: 'V',
    cik: '0001403161',
    company_name: 'Visa Inc.',
    state: 'POLLED_QUIET',
    observed_at: '2026-11-30T14:24:20Z',
    cutoff: '2026-11-30T14:20:00Z',
    last_polled_at: '2026-11-30T14:24:20Z',
    interval_start: '2026-11-30T14:00:00Z',
    accounted_through: '2026-11-30T14:20:00Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: '9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d',
    catalog_gaps: [],
    new_filings_count: 0,
    form4_count: 0,
    form4a_count: 0,
    deferred_count: 0,
    unresolved_amendments_count: 0,
    gaps: [],
    inventory: [],
    artifact_id: 'evt-audit-1795620716-v',
    receipt_sha256: '0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    slot_id: 'SLOT-1',
    is_older_receipt: false,
  },
];

/**
 * Validates the actual worker receipt format, trial, slot, release SHA,
 * issuer/CIK pairs, and required receipt fields.
 * If independent export is available, compares counts and gaps.
 * If not, labels the receipt "Imported — reconciliation pending".
 */
export function validateWorkerReceiptFormat(
  rawInput: any,
  activeSlotId?: string,
  independentExport?: IndependentlyReconciledExport | null,
): WorkerReceiptValidation {
  const errors: string[] = [];
  const discrepancies: CohortValidationDiscrepancy[] = [];

  const { evidence, receiptId, releaseSha, trialId, slotId } = extractCohortEvidence(rawInput);

  if (!evidence) {
    return {
      is_valid_format: false,
      trial_id: null,
      trial_id_matches: false,
      slot_id: null,
      is_prior_slot: false,
      prior_slot_id: null,
      release_sha: null,
      release_sha_matches: false,
      issuers_matched: false,
      issuer_count: 0,
      missing_issuers: Object.keys(COHORT_CIKS),
      mismatched_issuers: [],
      receipt_fields_valid: false,
      validation_errors: ['Malformed receipt: missing cohort evidence or issuers object.'],
      reconciliation_status: 'RECONCILIATION_DISCREPANCY',
      reconciliation_label: 'Imported — malformed format',
      discrepancies: [],
    };
  }

  // 1. Trial ID validation
  const effectiveTrialId = trialId || evidence.trial_id || rawInput.trial_id;
  const trialIdMatches = effectiveTrialId === QUALIFIED_TRIAL_ID;
  if (!trialIdMatches) {
    errors.push(`Trial ID mismatch: expected '${QUALIFIED_TRIAL_ID}', got '${effectiveTrialId || 'MISSING'}'`);
  }

  // 2. Release SHA validation
  const effectiveReleaseSha = releaseSha || evidence.release_sha || rawInput.release_sha;
  const releaseShaMatches = effectiveReleaseSha === QUALIFIED_RELEASE_SHA;
  if (!releaseShaMatches) {
    errors.push(
      `Release SHA mismatch: expected '${QUALIFIED_RELEASE_SHA}', got '${effectiveReleaseSha || 'MISSING'}'`,
    );
  }

  // 3. Slot ID & Prior Slot identification
  const effectiveSlotId = slotId || evidence.slot_id || rawInput.slot_id || null;
  let isPriorSlot = false;
  let priorSlotId: string | null = null;
  if (effectiveSlotId && activeSlotId && effectiveSlotId !== activeSlotId) {
    isPriorSlot = true;
    priorSlotId = effectiveSlotId;
  }

  // 4. Issuer / CIK pairs validation
  const issuers = evidence.issuers || {};
  const issuerKeys = Object.keys(issuers);
  const missingIssuers: string[] = [];
  const mismatchedIssuers: string[] = [];

  for (const [expectedTicker, expectedCik] of Object.entries(COHORT_CIKS)) {
    const issuer = issuers[expectedTicker];
    if (!issuer) {
      missingIssuers.push(expectedTicker);
    } else {
      const rawCik = String(issuer.cik || '').padStart(10, '0');
      if (rawCik !== expectedCik) {
        mismatchedIssuers.push(`${expectedTicker} (expected CIK ${expectedCik}, got ${issuer.cik})`);
        errors.push(`Mismatched CIK for ${expectedTicker}: expected ${expectedCik}, got ${issuer.cik}`);
      }
    }
  }

  // Check for unexpected extra issuers outside cohort
  for (const ticker of issuerKeys) {
    if (!COHORT_CIKS[ticker]) {
      mismatchedIssuers.push(`${ticker} (unexpected issuer outside authorized cohort)`);
      errors.push(`Unexpected issuer ${ticker} present in receipt outside authorized universe`);
    }
  }

  const issuersMatched = missingIssuers.length === 0 && mismatchedIssuers.length === 0;

  // 5. Receipt fields validity
  let receiptFieldsValid = true;
  for (const [ticker, issuer] of Object.entries(issuers)) {
    if (!issuer || typeof issuer !== 'object') {
      receiptFieldsValid = false;
      errors.push(`Invalid issuer record for ${ticker}`);
      continue;
    }
    if (!issuer.interval_start || !issuer.cutoff) {
      receiptFieldsValid = false;
      errors.push(`Missing accounting interval (interval_start/cutoff) for ${ticker}`);
    }
  }

  const isValidFormat = trialIdMatches && releaseShaMatches && issuersMatched && receiptFieldsValid;

  // 6. Reconciliation comparison
  let reconciliationStatus: 'RECONCILIATION_PENDING' | 'RECONCILED' | 'RECONCILIATION_DISCREPANCY' =
    'RECONCILIATION_PENDING';
  let reconciliationLabel = 'Imported — reconciliation pending';

  if (independentExport && independentExport.issuers) {
    for (const [ticker, issuer] of Object.entries(issuers)) {
      const indep = independentExport.issuers[ticker];
      if (!indep) {
        discrepancies.push({
          ticker,
          field: 'issuer_in_independent_export',
          expected: 'PRESENT',
          actual: 'MISSING',
          severity: 'ERROR',
        });
        continue;
      }

      // Calculate actual counts from inventory
      const inventory = issuer.inventory || [];
      const actualIngested = inventory.filter((i) => i.disposition === 'INGESTED').length;
      const actualDeferred = inventory.filter((i) => i.disposition === 'DEFERRED').length;
      const actualGaps = issuer.gaps || [];

      if (indep.expected_count !== undefined && indep.expected_count !== inventory.length) {
        discrepancies.push({
          ticker,
          field: 'expected_count',
          expected: indep.expected_count,
          actual: inventory.length,
          severity: 'ERROR',
        });
      }

      if (indep.accounted_count !== undefined && indep.accounted_count !== actualIngested) {
        discrepancies.push({
          ticker,
          field: 'accounted_count',
          expected: indep.accounted_count,
          actual: actualIngested,
          severity: 'ERROR',
        });
      }

      if (indep.missing_count !== undefined && indep.missing_count !== actualDeferred) {
        discrepancies.push({
          ticker,
          field: 'missing_count',
          expected: indep.missing_count,
          actual: actualDeferred,
          severity: 'ERROR',
        });
      }

      if (indep.gaps && JSON.stringify(indep.gaps.sort()) !== JSON.stringify(actualGaps.sort())) {
        discrepancies.push({
          ticker,
          field: 'gaps',
          expected: indep.gaps,
          actual: actualGaps,
          severity: 'ERROR',
        });
      }
    }

    if (discrepancies.length === 0) {
      reconciliationStatus = 'RECONCILED';
      reconciliationLabel = 'Imported — reconciled against independent export';
    } else {
      reconciliationStatus = 'RECONCILIATION_DISCREPANCY';
      reconciliationLabel = 'Imported — reconciliation discrepancies detected';
    }
  }

  return {
    is_valid_format: isValidFormat,
    trial_id: effectiveTrialId || null,
    trial_id_matches: trialIdMatches,
    slot_id: effectiveSlotId,
    is_prior_slot: isPriorSlot,
    prior_slot_id: priorSlotId,
    release_sha: effectiveReleaseSha || null,
    release_sha_matches: releaseShaMatches,
    issuers_matched: issuersMatched,
    issuer_count: issuerKeys.length,
    missing_issuers: missingIssuers,
    mismatched_issuers: mismatchedIssuers,
    receipt_fields_valid: receiptFieldsValid,
    validation_errors: errors,
    reconciliation_status: reconciliationStatus,
    reconciliation_label: reconciliationLabel,
    discrepancies,
  };
}

/**
 * Parses raw reconciliation receipt JSON into 10-issuer summary objects.
 * Populates observed_at as the poll time, and displays cutoff separately.
 * Retains distinct evidence states: verified zero, not polled, incomplete evidence, read error.
 */
export function parseReceiptToSummaries(
  receipt: any,
  referenceTimeIso?: string,
  activeSlotId?: string,
): IssuerEvidenceSummary[] {
  const currentTime = referenceTimeIso ? new Date(referenceTimeIso) : new Date();

  const { evidence, receiptId, slotId } = extractCohortEvidence(receipt);
  const rawIssuers = evidence?.issuers || {};

  return GOVERNED_COHORT_ISSUERS.map((gov) => {
    const raw: RawIssuerReceipt | undefined = rawIssuers[gov.ticker];

    if (!raw) {
      // NOT POLLED state: no completed receipt exists
      return {
        ticker: gov.ticker,
        cik: gov.cik,
        company_name: gov.company_name,
        state: 'NOT_POLLED' as const,
        observed_at: null,
        cutoff: null,
        last_polled_at: null,
        interval_start: null,
        accounted_through: null,
        previous_watermark: null,
        catalog_sha256: null,
        catalog_gaps: [],
        new_filings_count: null, // Strictly null
        form4_count: null,
        form4a_count: null,
        deferred_count: null,
        unresolved_amendments_count: null,
        gaps: [],
        inventory: [],
        artifact_id: null,
        receipt_sha256: null,
        receipt_id: receiptId || null,
        slot_id: null,
        is_prior_slot: false,
        prior_slot_id: null,
        is_older_receipt: false,
      };
    }

    if (raw.status === 'READ_ERROR' || raw.catalog_sha256 === 'corrupted-sha256-hash') {
      // READ ERROR state: receipt could not be loaded, strictly omit unverified counts
      return {
        ticker: gov.ticker,
        cik: gov.cik,
        company_name: gov.company_name,
        state: 'READ_ERROR' as const,
        observed_at: raw.observed_at || raw.cutoff || null,
        cutoff: raw.cutoff || null,
        last_polled_at: raw.observed_at || raw.cutoff || null,
        interval_start: raw.interval_start || null,
        accounted_through: null,
        previous_watermark: raw.previous_watermark || null,
        catalog_sha256: raw.catalog_sha256 || null,
        catalog_gaps: raw.catalog_gaps || [],
        new_filings_count: null, // Strictly null
        form4_count: null,
        form4a_count: null,
        deferred_count: null,
        unresolved_amendments_count: null,
        gaps: raw.gaps || [],
        inventory: [],
        artifact_id: raw.artifact_id || null,
        receipt_sha256: raw.sha256 || null,
        receipt_id: receiptId || null,
        slot_id: raw.slot_id || slotId || null,
        is_prior_slot: Boolean(activeSlotId && (raw.slot_id || slotId) && (raw.slot_id || slotId) !== activeSlotId),
        prior_slot_id: raw.slot_id || slotId || null,
        is_older_receipt: false,
        read_error_message: 'Receipt catalog read error: could not load verified catalog payload.',
      };
    }

    const inventory = raw.inventory || [];
    const gaps = raw.gaps || [];
    const catalogGaps = raw.catalog_gaps || [];

    const ingested = inventory.filter((item) => item.disposition === 'INGESTED');
    const deferred = inventory.filter((item) => item.disposition === 'DEFERRED');
    const unresolved = inventory.filter((item) => item.disposition === 'AMENDMENT_UNRESOLVED');

    const form4Count = ingested.filter((i) => i.form === '4').length;
    const form4aCount = ingested.filter((i) => i.form === '4/A').length;
    const newFilingsCount = ingested.length;

    // Distinct evidence states
    let state: IssuerEvidenceSummary['state'];
    if (gaps.length > 0 || catalogGaps.length > 0 || raw.status === 'PARTIAL') {
      state = 'INCOMPLETE';
    } else if (newFilingsCount > 0) {
      state = 'FILINGS_FOUND';
    } else {
      state = 'POLLED_QUIET';
    }

    // Historical poll check: if cutoff/observed_at is >2 hours before reference time
    let isOlder = false;
    const compareTimeStr = raw.observed_at || raw.cutoff;
    if (compareTimeStr) {
      const dateVal = new Date(compareTimeStr);
      isOlder = currentTime.getTime() - dateVal.getTime() > 2 * 60 * 60 * 1000;
    }

    const effectiveSlot = raw.slot_id || slotId || null;
    const isPrior = Boolean(activeSlotId && effectiveSlot && effectiveSlot !== activeSlotId);

    return {
      ticker: gov.ticker,
      cik: gov.cik,
      company_name: gov.company_name,
      state,
      observed_at: raw.observed_at || raw.cutoff || null,
      cutoff: raw.cutoff || null,
      last_polled_at: raw.observed_at || raw.cutoff || null,
      interval_start: raw.interval_start || null,
      accounted_through: raw.accounted_through || null,
      previous_watermark: raw.previous_watermark || null,
      catalog_sha256: raw.catalog_sha256 || null,
      catalog_gaps: catalogGaps,
      new_filings_count: newFilingsCount,
      form4_count: form4Count,
      form4a_count: form4aCount,
      deferred_count: deferred.length,
      unresolved_amendments_count: unresolved.length,
      gaps,
      inventory: inventory.map((inv) => ({
        accession: inv.accession,
        form: inv.form,
        filing_date: inv.filing_date,
        accepted_at: inv.accepted_at,
        primary_document: inv.primary_document,
        disposition: inv.disposition,
        source_sha256: inv.source_sha256,
      })),
      artifact_id: raw.artifact_id || null,
      receipt_sha256: raw.sha256 || null,
      receipt_id: receiptId || null,
      slot_id: effectiveSlot,
      is_prior_slot: isPrior,
      prior_slot_id: isPrior ? effectiveSlot : null,
      is_older_receipt: isOlder,
    };
  });
}

/**
 * Validates an exported receipt against an expected summary map (used in synthetic fixture mode).
 */
export function validateReceiptAgainstExpected(
  receipt: any,
  expectedMap: Record<string, Partial<IssuerEvidenceSummary>>,
): ReceiptValidationReport {
  const summaries = parseReceiptToSummaries(receipt);
  const discrepancies: CohortValidationDiscrepancy[] = [];

  for (const actual of summaries) {
    const exp = expectedMap[actual.ticker];
    if (!exp) {
      discrepancies.push({
        ticker: actual.ticker,
        field: 'existence',
        expected: 'PRESENT_IN_EXPECTATION',
        actual: 'MISSING_IN_EXPECTATION',
        severity: 'WARNING',
      });
      continue;
    }

    if (exp.state && exp.state !== actual.state) {
      discrepancies.push({
        ticker: actual.ticker,
        field: 'state',
        expected: exp.state,
        actual: actual.state,
        severity: 'ERROR',
      });
    }

    if (exp.new_filings_count !== undefined && exp.new_filings_count !== actual.new_filings_count) {
      discrepancies.push({
        ticker: actual.ticker,
        field: 'new_filings_count',
        expected: exp.new_filings_count,
        actual: actual.new_filings_count,
        severity: 'ERROR',
      });
    }

    if (exp.accounted_through !== undefined && exp.accounted_through !== actual.accounted_through) {
      discrepancies.push({
        ticker: actual.ticker,
        field: 'accounted_through',
        expected: exp.accounted_through,
        actual: actual.accounted_through,
        severity: 'ERROR',
      });
    }

    if (exp.gaps && JSON.stringify(exp.gaps.sort()) !== JSON.stringify(actual.gaps.sort())) {
      discrepancies.push({
        ticker: actual.ticker,
        field: 'gaps',
        expected: exp.gaps,
        actual: actual.gaps,
        severity: 'ERROR',
      });
    }
  }

  const { receiptId } = extractCohortEvidence(receipt);

  return {
    receipt_id: receiptId || 'unidentified-receipt',
    validated_at: new Date().toISOString(),
    is_valid: discrepancies.filter((d) => d.severity === 'ERROR').length === 0,
    issuer_count: summaries.length,
    discrepancies,
  };
}
