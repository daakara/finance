/**
 * ARX Terminal — Cohort Pilot Evidence Synthetic Fixtures & Reconciliation Validator
 * Issue #165 / Trial ARX-TRIAL-10-ISSUER-001
 * Aligned with candidate 1063a10fed1ed312feadeb671980a642e757e601
 */

import {
  CohortReconciliationReceipt,
  IssuerEvidenceSummary,
  RawIssuerReceipt,
  ReceiptValidationReport,
  CohortValidationDiscrepancy,
} from '../types/cohortPilotEvidence';

export interface GovernedIssuerDefinition {
  ticker: string;
  cik: string;
  company_name: string;
}

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
 * Full synthetic reconciliation receipt modeled on candidate 1063a10f output.
 */
export const SYNTHETIC_COHORT_RECEIPT: CohortReconciliationReceipt = {
  receipt_id: 'rcpt-pilot-165-20261130-slot1',
  trial_id: 'ARX-TRIAL-10-ISSUER-001',
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
      universe_id: 'UB-US-EQUITIES-20261025-001',
      release_sha: '1063a10fed1ed312feadeb671980a642e757e601',
      issuers: {
        NVDA: {
          ticker: 'NVDA',
          cik: '0001045810',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:15Z',
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
          accounted_through: '2026-11-30T14:20:15Z',
          status: 'CATALOG_ACCOUNTED',
          sha256: 'b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5',
        },
        AAPL: {
          ticker: 'AAPL',
          cik: '0000320193',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:25Z',
          artifact_id: 'evt-audit-1795620716-aapl',
          catalog_sha256: 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4',
          catalog_gaps: [],
          inventory: [],
          gaps: [],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:20:25Z',
          status: 'CATALOG_ACCOUNTED',
          sha256: 'd5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6',
        },
        MSFT: {
          ticker: 'MSFT',
          cik: '0000789019',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:40Z',
          artifact_id: 'evt-audit-1795620716-msft',
          catalog_sha256: 'e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7',
          catalog_gaps: [],
          inventory: [
            {
              accession: '0000789019-26-000085',
              form: '4',
              filing_date: '2026-11-30',
              accepted_at: '2026-11-30T14:02:10Z',
              primary_document: 'doc4_1.xml',
              disposition: 'INGESTED',
              source_sha256: 'aaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccdddd',
            },
            {
              accession: '0000789019-26-000086',
              form: '4',
              filing_date: '2026-11-30',
              accepted_at: '2026-11-30T14:05:30Z',
              primary_document: 'doc4_2.xml',
              disposition: 'INGESTED',
              source_sha256: 'bbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaa',
            },
            {
              accession: '0000789019-26-000087',
              form: '4',
              filing_date: '2026-11-30',
              accepted_at: '2026-11-30T14:08:45Z',
              primary_document: 'doc4_3.xml',
              disposition: 'INGESTED',
              source_sha256: 'ccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbb',
            },
            {
              accession: '0000789019-26-000088',
              form: '4',
              filing_date: '2026-11-30',
              accepted_at: '2026-11-30T14:15:00Z',
              primary_document: 'doc4_4.xml',
              disposition: 'DEFERRED',
              source_sha256: null,
            },
          ],
          gaps: ['DEFERRED:0000789019-26-000088'],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:00:00Z', // Lags cutoff due to budget deferral
          status: 'PARTIAL',
          sha256: 'f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8',
        },
        AMZN: {
          ticker: 'AMZN',
          cik: '0001018724',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:20:55Z',
          artifact_id: 'evt-audit-1795620716-amzn',
          catalog_sha256: '0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b',
          catalog_gaps: [],
          inventory: [
            {
              accession: '0001018724-26-000042',
              form: '4/A',
              filing_date: '2026-11-30',
              accepted_at: '2026-11-30T14:07:20Z',
              primary_document: 'amendment4.xml',
              disposition: 'AMENDMENT_UNRESOLVED',
              source_sha256: '999888777666555444333222111000aaabbbcccdddeeefff9998887776665554',
            },
          ],
          gaps: ['AMENDMENT_UNRESOLVED:0001018724-26-000042'],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:00:00Z', // Lags cutoff due to missing original Form 4
          status: 'PARTIAL',
          sha256: '1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c',
        },
        GOOGL: {
          ticker: 'GOOGL',
          cik: '0001652044',
          interval_start: '2026-11-30T09:00:00Z',
          cutoff: '2026-11-30T09:22:10Z', // Older morning poll
          artifact_id: 'evt-audit-1795600000-googl',
          catalog_sha256: '2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d',
          catalog_gaps: [],
          inventory: [],
          gaps: [],
          previous_watermark: '2026-11-30T09:00:00Z',
          accounted_through: '2026-11-30T09:22:10Z',
          status: 'CATALOG_ACCOUNTED',
          sha256: '3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e',
        },
        META: {
          ticker: 'META',
          cik: '0001326801',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:21:10Z',
          artifact_id: 'evt-audit-1795620716-meta',
          catalog_sha256: '4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f',
          catalog_gaps: [],
          inventory: [
            {
              accession: '0001326801-26-000055',
              form: '4',
              filing_date: '2026-11-30',
              accepted_at: '2026-11-30T14:12:00Z',
              primary_document: 'form4.xml',
              disposition: 'INGESTED',
              source_sha256: '555544443333222211110000ffffeeeeddddccccbbbbaaaa5555444433332222',
            },
          ],
          gaps: [],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:21:10Z',
          status: 'CATALOG_ACCOUNTED',
          sha256: '5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a',
        },
        JPM: {
          ticker: 'JPM',
          cik: '0000019617',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:21:30Z',
          artifact_id: 'evt-audit-1795620716-jpm',
          catalog_sha256: '6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b',
          catalog_gaps: [],
          inventory: [],
          gaps: [],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:21:30Z',
          status: 'CATALOG_ACCOUNTED',
          sha256: '7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c',
        },
        V: {
          ticker: 'V',
          cik: '0001403161',
          interval_start: '2026-11-30T14:00:00Z',
          cutoff: '2026-11-30T14:21:45Z',
          artifact_id: 'evt-audit-1795620716-v',
          catalog_sha256: '8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d',
          catalog_gaps: ['HISTORICAL_PAGE_MAY_OVERLAP'],
          inventory: [],
          gaps: ['HISTORICAL_PAGE_MAY_OVERLAP'],
          previous_watermark: '2026-11-30T14:00:00Z',
          accounted_through: '2026-11-30T14:00:00Z', // Lags cutoff due to catalog gap
          status: 'PARTIAL',
          sha256: '9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e',
        },
      },
    },
  },
};

/**
 * Pre-computed synthetic summaries for all 10 issuers demonstrating the 5 evidence states:
 * - NVDA: FILINGS_FOUND (1 Form 4, 1 Form 4/A)
 * - AAPL: POLLED_QUIET (quiet, accounted)
 * - MSFT: INCOMPLETE (budget deferral: 4-req cap reached)
 * - AMZN: INCOMPLETE (unresolved amendment)
 * - GOOGL: POLLED_QUIET (older historical receipt)
 * - META: FILINGS_FOUND (1 Form 4)
 * - TSLA: NOT_POLLED (no receipt exists)
 * - BRK-B: READ_ERROR (checksum error, counts omitted)
 * - JPM: POLLED_QUIET (quiet, accounted)
 * - V: INCOMPLETE (catalog overlap gap)
 */
export const SYNTHETIC_ISSUER_SUMMARIES: IssuerEvidenceSummary[] = [
  {
    ticker: 'NVDA',
    cik: '0001045810',
    company_name: 'NVIDIA Corporation',
    state: 'FILINGS_FOUND',
    last_polled_at: '2026-11-30T14:20:15Z',
    interval_start: '2026-11-30T14:00:00Z',
    cutoff: '2026-11-30T14:20:15Z',
    accounted_through: '2026-11-30T14:20:15Z',
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
    is_older_receipt: false,
  },
  {
    ticker: 'AAPL',
    cik: '0000320193',
    company_name: 'Apple Inc.',
    state: 'POLLED_QUIET',
    last_polled_at: '2026-11-30T14:20:25Z',
    interval_start: '2026-11-30T14:00:00Z',
    cutoff: '2026-11-30T14:20:25Z',
    accounted_through: '2026-11-30T14:20:25Z',
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
    is_older_receipt: false,
  },
  {
    ticker: 'MSFT',
    cik: '0000789019',
    company_name: 'Microsoft Corporation',
    state: 'INCOMPLETE',
    last_polled_at: '2026-11-30T14:20:40Z',
    interval_start: '2026-11-30T14:00:00Z',
    cutoff: '2026-11-30T14:20:40Z',
    accounted_through: '2026-11-30T14:00:00Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: 'e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7',
    catalog_gaps: [],
    new_filings_count: 3,
    form4_count: 3,
    form4a_count: 0,
    deferred_count: 1,
    unresolved_amendments_count: 0,
    gaps: ['DEFERRED:0000789019-26-000088'],
    inventory: [
      {
        accession: '0000789019-26-000085',
        form: '4',
        filing_date: '2026-11-30',
        accepted_at: '2026-11-30T14:02:10Z',
        primary_document: 'doc4_1.xml',
        disposition: 'INGESTED',
        source_sha256: 'aaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccdddd',
      },
      {
        accession: '0000789019-26-000086',
        form: '4',
        filing_date: '2026-11-30',
        accepted_at: '2026-11-30T14:05:30Z',
        primary_document: 'doc4_2.xml',
        disposition: 'INGESTED',
        source_sha256: 'bbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaa',
      },
      {
        accession: '0000789019-26-000087',
        form: '4',
        filing_date: '2026-11-30',
        accepted_at: '2026-11-30T14:08:45Z',
        primary_document: 'doc4_3.xml',
        disposition: 'INGESTED',
        source_sha256: 'ccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbbccccddddaaaabbbb',
      },
      {
        accession: '0000789019-26-000088',
        form: '4',
        filing_date: '2026-11-30',
        accepted_at: '2026-11-30T14:15:00Z',
        primary_document: 'doc4_4.xml',
        disposition: 'DEFERRED',
        source_sha256: null,
      },
    ],
    artifact_id: 'evt-audit-1795620716-msft',
    receipt_sha256: 'f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    is_older_receipt: false,
  },
  {
    ticker: 'AMZN',
    cik: '0001018724',
    company_name: 'Amazon.com, Inc.',
    state: 'INCOMPLETE',
    last_polled_at: '2026-11-30T14:20:55Z',
    interval_start: '2026-11-30T14:00:00Z',
    cutoff: '2026-11-30T14:20:55Z',
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
        accepted_at: '2026-11-30T14:07:20Z',
        primary_document: 'amendment4.xml',
        disposition: 'AMENDMENT_UNRESOLVED',
        source_sha256: '999888777666555444333222111000aaabbbcccdddeeefff9998887776665554',
      },
    ],
    artifact_id: 'evt-audit-1795620716-amzn',
    receipt_sha256: '1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    is_older_receipt: false,
  },
  {
    ticker: 'GOOGL',
    cik: '0001652044',
    company_name: 'Alphabet Inc.',
    state: 'POLLED_QUIET',
    last_polled_at: '2026-11-30T09:22:10Z', // Actual older poll timestamp
    interval_start: '2026-11-30T09:00:00Z',
    cutoff: '2026-11-30T09:22:10Z',
    accounted_through: '2026-11-30T09:22:10Z',
    previous_watermark: '2026-11-30T09:00:00Z',
    catalog_sha256: '2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d',
    catalog_gaps: [],
    new_filings_count: 0,
    form4_count: 0,
    form4a_count: 0,
    deferred_count: 0,
    unresolved_amendments_count: 0,
    gaps: [],
    inventory: [],
    artifact_id: 'evt-audit-1795600000-googl',
    receipt_sha256: '3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e',
    receipt_id: 'rcpt-pilot-165-20261130-slot0',
    is_older_receipt: true, // Specifically presents older receipt with actual timestamp
  },
  {
    ticker: 'META',
    cik: '0001326801',
    company_name: 'Meta Platforms, Inc.',
    state: 'FILINGS_FOUND',
    last_polled_at: '2026-11-30T14:21:10Z',
    interval_start: '2026-11-30T14:00:00Z',
    cutoff: '2026-11-30T14:21:10Z',
    accounted_through: '2026-11-30T14:21:10Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: '4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f',
    catalog_gaps: [],
    new_filings_count: 1,
    form4_count: 1,
    form4a_count: 0,
    deferred_count: 0,
    unresolved_amendments_count: 0,
    gaps: [],
    inventory: [
      {
        accession: '0001326801-26-000055',
        form: '4',
        filing_date: '2026-11-30',
        accepted_at: '2026-11-30T14:12:00Z',
        primary_document: 'form4.xml',
        disposition: 'INGESTED',
        source_sha256: '555544443333222211110000ffffeeeeddddccccbbbbaaaa5555444433332222',
      },
    ],
    artifact_id: 'evt-audit-1795620716-meta',
    receipt_sha256: '5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    is_older_receipt: false,
  },
  {
    ticker: 'TSLA',
    cik: '0001318605',
    company_name: 'Tesla, Inc.',
    state: 'NOT_POLLED',
    last_polled_at: null,
    interval_start: null,
    cutoff: null,
    accounted_through: null,
    previous_watermark: null,
    catalog_sha256: null,
    catalog_gaps: [],
    new_filings_count: null, // Omitted
    form4_count: null,
    form4a_count: null,
    deferred_count: null,
    unresolved_amendments_count: null,
    gaps: [],
    inventory: [],
    artifact_id: null,
    receipt_sha256: null,
    receipt_id: null,
    is_older_receipt: false,
  },
  {
    ticker: 'BRK-B',
    cik: '0001067983',
    company_name: 'Berkshire Hathaway Inc.',
    state: 'READ_ERROR',
    last_polled_at: null,
    interval_start: null,
    cutoff: null,
    accounted_through: null,
    previous_watermark: null,
    catalog_sha256: null,
    catalog_gaps: [],
    new_filings_count: null, // Strictly omitted on read error
    form4_count: null,
    form4a_count: null,
    deferred_count: null,
    unresolved_amendments_count: null,
    gaps: ['EVIDENCE_CORRUPT:SHA256_MISMATCH'],
    inventory: [],
    artifact_id: 'evt-audit-1795620716-brkb',
    receipt_sha256: null,
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    is_older_receipt: false,
    read_error_message: 'Receipt corrupted: catalog payload SHA-256 mismatch detected in storage ledger. Unverified counts omitted.',
  },
  {
    ticker: 'JPM',
    cik: '0000019617',
    company_name: 'JPMorgan Chase & Co.',
    state: 'POLLED_QUIET',
    last_polled_at: '2026-11-30T14:21:30Z',
    interval_start: '2026-11-30T14:00:00Z',
    cutoff: '2026-11-30T14:21:30Z',
    accounted_through: '2026-11-30T14:21:30Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: '6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b',
    catalog_gaps: [],
    new_filings_count: 0,
    form4_count: 0,
    form4a_count: 0,
    deferred_count: 0,
    unresolved_amendments_count: 0,
    gaps: [],
    inventory: [],
    artifact_id: 'evt-audit-1795620716-jpm',
    receipt_sha256: '7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    is_older_receipt: false,
  },
  {
    ticker: 'V',
    cik: '0001403161',
    company_name: 'Visa Inc.',
    state: 'INCOMPLETE',
    last_polled_at: '2026-11-30T14:21:45Z',
    interval_start: '2026-11-30T14:00:00Z',
    cutoff: '2026-11-30T14:21:45Z',
    accounted_through: '2026-11-30T14:00:00Z',
    previous_watermark: '2026-11-30T14:00:00Z',
    catalog_sha256: '8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d',
    catalog_gaps: ['HISTORICAL_PAGE_MAY_OVERLAP'],
    new_filings_count: 0,
    form4_count: 0,
    form4a_count: 0,
    deferred_count: 0,
    unresolved_amendments_count: 0,
    gaps: ['HISTORICAL_PAGE_MAY_OVERLAP'],
    inventory: [],
    artifact_id: 'evt-audit-1795620716-v',
    receipt_sha256: '9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e',
    receipt_id: 'rcpt-pilot-165-20261130-slot1',
    is_older_receipt: false,
  },
];

/**
 * Transforms a raw reconciliation receipt into standardized IssuerEvidenceSummary models for all 10 cohort issuers.
 */
export function parseReceiptToSummaries(
  receipt: CohortReconciliationReceipt | null,
  currentSimulatedTime?: string,
): IssuerEvidenceSummary[] {
  const issuerEvidence = receipt?.failure_reasons?.cohort_evidence?.issuers || {};
  const currentTime = currentSimulatedTime ? new Date(currentSimulatedTime) : new Date();

  return GOVERNED_COHORT_ISSUERS.map((gov) => {
    const raw: RawIssuerReceipt | undefined = issuerEvidence[gov.ticker];

    if (!raw) {
      return {
        ticker: gov.ticker,
        cik: gov.cik,
        company_name: gov.company_name,
        state: 'NOT_POLLED',
        last_polled_at: null,
        interval_start: null,
        cutoff: null,
        accounted_through: null,
        previous_watermark: null,
        catalog_sha256: null,
        catalog_gaps: [],
        new_filings_count: null,
        form4_count: null,
        form4a_count: null,
        deferred_count: null,
        unresolved_amendments_count: null,
        gaps: [],
        inventory: [],
        artifact_id: null,
        receipt_sha256: null,
        receipt_id: receipt?.receipt_id || null,
        is_older_receipt: false,
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

    // Determine state
    let state: IssuerEvidenceSummary['state'];
    if (gaps.length > 0 || catalogGaps.length > 0 || raw.status === 'PARTIAL') {
      state = 'INCOMPLETE';
    } else if (newFilingsCount > 0) {
      state = 'FILINGS_FOUND';
    } else {
      state = 'POLLED_QUIET';
    }

    // Check if receipt cutoff is >2 hours older than current observation time
    let isOlder = false;
    if (raw.cutoff) {
      const cutoffDate = new Date(raw.cutoff);
      isOlder = currentTime.getTime() - cutoffDate.getTime() > 2 * 60 * 60 * 1000;
    }

    return {
      ticker: gov.ticker,
      cik: gov.cik,
      company_name: gov.company_name,
      state,
      last_polled_at: raw.cutoff,
      interval_start: raw.interval_start,
      cutoff: raw.cutoff,
      accounted_through: raw.accounted_through,
      previous_watermark: raw.previous_watermark,
      catalog_sha256: raw.catalog_sha256,
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
      artifact_id: raw.artifact_id,
      receipt_sha256: raw.sha256,
      receipt_id: receipt?.receipt_id || null,
      is_older_receipt: isOlder,
    };
  });
}

/**
 * Validates an exported receipt against an expected summary map.
 * Compares all 10 issuer rows for counts, timestamps, coverage gaps, and links.
 */
export function validateReceiptAgainstExpected(
  receipt: CohortReconciliationReceipt,
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

    // Compare state
    if (exp.state && exp.state !== actual.state) {
      discrepancies.push({
        ticker: actual.ticker,
        field: 'state',
        expected: exp.state,
        actual: actual.state,
        severity: 'ERROR',
      });
    }

    // Compare new_filings_count
    if (exp.new_filings_count !== undefined && exp.new_filings_count !== actual.new_filings_count) {
      discrepancies.push({
        ticker: actual.ticker,
        field: 'new_filings_count',
        expected: exp.new_filings_count,
        actual: actual.new_filings_count,
        severity: 'ERROR',
      });
    }

    // Compare accounted_through timestamp
    if (exp.accounted_through !== undefined && exp.accounted_through !== actual.accounted_through) {
      discrepancies.push({
        ticker: actual.ticker,
        field: 'accounted_through',
        expected: exp.accounted_through,
        actual: actual.accounted_through,
        severity: 'ERROR',
      });
    }

    // Compare gaps count
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

  return {
    receipt_id: receipt.receipt_id,
    validated_at: new Date().toISOString(),
    is_valid: discrepancies.filter((d) => d.severity === 'ERROR').length === 0,
    issuer_count: summaries.length,
    discrepancies,
  };
}
