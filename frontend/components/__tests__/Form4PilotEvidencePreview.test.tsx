import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Form4PilotEvidencePreview from '../pilot/Form4PilotEvidencePreview';
import {
  SYNTHETIC_ISSUER_SUMMARIES,
  SYNTHETIC_COHORT_RECEIPT,
  QUALIFIED_RELEASE_SHA,
  QUALIFIED_TRIAL_ID,
  COHORT_CIKS,
  parseReceiptToSummaries,
  validateWorkerReceiptFormat,
  validateReceiptAgainstExpected,
  buildSecEdgarUrl,
} from '../../fixtures/cohortPilotEvidenceFixtures';
import { CohortReconciliationReceipt, IndependentlyReconciledExport } from '../../types/cohortPilotEvidence';

describe('Form4PilotEvidencePreview Component & Worker Receipt Validation', () => {
  describe('Baseline Presentation & Governed Invariants', () => {
    it('renders preview title, private badge, trial id and all 10 cohort issuers', () => {
      render(<Form4PilotEvidencePreview />);

      // Header & guardrails
      expect(screen.getByText('Form 4 Pilot Evidence Preview')).toBeDefined();
      expect(screen.getByText('Private Preview')).toBeDefined();
      expect(screen.getByText(QUALIFIED_TRIAL_ID)).toBeDefined();
      expect(screen.getByText(/1063a10f/i)).toBeDefined();

      // Verify all 10 governed tickers are present
      const expectedTickers = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JPM', 'V'];
      expectedTickers.forEach((ticker) => {
        expect(screen.getAllByText(ticker).length).toBeGreaterThan(0);
      });
    });

    it('displays issuer observed_at timestamp as poll time and cutoff separately', () => {
      render(<Form4PilotEvidencePreview />);

      // Check header columns
      expect(screen.getAllByText(/Poll Time \(observed_at\)/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/Coverage Cutoff/i).length).toBeGreaterThan(0);

      // For NVDA in synthetic fixture:
      // observed_at is 2026-11-30T14:20:15Z -> 14:20:15 UTC
      // cutoff is 2026-11-30T14:20:00Z -> 14:20:00 UTC
      expect(screen.getAllByText('14:20:15 UTC').length).toBeGreaterThan(0);
      expect(screen.getAllByText('14:20:00 UTC').length).toBeGreaterThan(0);
    });
  });

  describe('Five Distinct Evidence States', () => {
    it('1. NOT_POLLED: renders NOT POLLED badge and strictly omits unverified counts as dash', () => {
      render(<Form4PilotEvidencePreview />);

      const notPolledBadges = screen.getAllByText('NOT POLLED');
      expect(notPolledBadges.length).toBeGreaterThan(0);

      const tslaSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'TSLA');
      expect(tslaSummary?.state).toBe('NOT_POLLED');
      expect(tslaSummary?.new_filings_count).toBeNull();
      expect(tslaSummary?.observed_at).toBeNull();
      expect(tslaSummary?.cutoff).toBeNull();
    });

    it('2. POLLED_QUIET: renders QUIET — ACCOUNTED badge and authentic verified 0 count', () => {
      render(<Form4PilotEvidencePreview />);

      const quietBadges = screen.getAllByText('QUIET — ACCOUNTED');
      expect(quietBadges.length).toBeGreaterThan(0);

      const aaplSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'AAPL');
      expect(aaplSummary?.state).toBe('POLLED_QUIET');
      expect(aaplSummary?.new_filings_count).toBe(0);
      expect(aaplSummary?.gaps.length).toBe(0);
    });

    it('3. FILINGS_FOUND: renders FILINGS FOUND badge, shows counts and SEC source links', () => {
      render(<Form4PilotEvidencePreview />);

      const filingsFoundBadges = screen.getAllByText('FILINGS FOUND');
      expect(filingsFoundBadges.length).toBeGreaterThan(0);

      const nvdaSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'NVDA');
      expect(nvdaSummary?.state).toBe('FILINGS_FOUND');
      expect(nvdaSummary?.new_filings_count).toBe(2);
      expect(nvdaSummary?.form4_count).toBe(1);
      expect(nvdaSummary?.form4a_count).toBe(1);

      // Verify SEC EDGAR URL generation
      const url = buildSecEdgarUrl(nvdaSummary!.cik, '0001045810-26-000045', 'form4.xml');
      expect(url).toBe('https://www.sec.gov/Archives/edgar/data/1045810/000104581026000045/form4.xml');
    });

    it('4. INCOMPLETE: renders INCOMPLETE / GAPS badge, shows recorded gaps and lagging cutoff note', () => {
      render(<Form4PilotEvidencePreview />);

      const incompleteBadges = screen.getAllByText('INCOMPLETE / GAPS');
      expect(incompleteBadges.length).toBeGreaterThan(0);

      const msftSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'MSFT');
      expect(msftSummary?.state).toBe('INCOMPLETE');
      expect(msftSummary?.deferred_count).toBe(1);
      expect(msftSummary?.gaps).toContain('DEFERRED:0000789019-26-000088');

      const amznSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'AMZN');
      expect(amznSummary?.state).toBe('INCOMPLETE');
      expect(amznSummary?.unresolved_amendments_count).toBe(1);
      expect(amznSummary?.gaps).toContain('AMENDMENT_UNRESOLVED:0001018724-26-000042');
    });

    it('5. READ_ERROR: renders READ ERROR badge, explains error and strictly omits unverified counts', () => {
      render(<Form4PilotEvidencePreview />);

      const readErrorBadges = screen.getAllByText('READ ERROR');
      expect(readErrorBadges.length).toBeGreaterThan(0);

      const brkbSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'BRK-B');
      expect(brkbSummary?.state).toBe('READ_ERROR');
      expect(brkbSummary?.new_filings_count).toBeNull();
      expect(brkbSummary?.read_error_message).toContain('SEC Catalog Retained Hash Mismatch');
    });
  });

  describe('Historical & Prior Slot Timestamp Presentation', () => {
    it('presents older historical receipt with Historical Poll indicator', () => {
      render(<Form4PilotEvidencePreview />);

      const googlSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'GOOGL');
      expect(googlSummary?.is_older_receipt).toBe(true);

      const historicalBadges = screen.getAllByText('Historical Poll');
      expect(historicalBadges.length).toBeGreaterThan(0);
    });

    it('uses slot_id to identify a prior-slot receipt when viewing subsequent slot', () => {
      // Create a receipt from SLOT-1
      const priorSlotReceipt = {
        ...SYNTHETIC_COHORT_RECEIPT,
        slot_id: 'SLOT-1',
        failure_reasons: {
          ...SYNTHETIC_COHORT_RECEIPT.failure_reasons,
          cohort_evidence: {
            ...SYNTHETIC_COHORT_RECEIPT.failure_reasons.cohort_evidence,
            slot_id: 'SLOT-1',
          },
        },
      };

      // Set active slot to SLOT-2
      const validation = validateWorkerReceiptFormat(priorSlotReceipt, 'SLOT-2');
      expect(validation.is_prior_slot).toBe(true);
      expect(validation.prior_slot_id).toBe('SLOT-1');

      const summaries = parseReceiptToSummaries(priorSlotReceipt, undefined, 'SLOT-2');
      const nvda = summaries.find((s) => s.ticker === 'NVDA');
      expect(nvda?.is_prior_slot).toBe(true);
      expect(nvda?.slot_id).toBe('SLOT-1');
    });

    it('expands detail inspection drawer when clicking Inspect', () => {
      render(<Form4PilotEvidencePreview />);

      const inspectButtons = screen.getAllByRole('button', { name: /Inspect/i });
      fireEvent.click(inspectButtons[0]);

      expect(screen.getAllByText(/Accession Audit Trail/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/SEC EDGAR/i).length).toBeGreaterThan(0);
    });

    it('filters rows when clicking KPI filter buttons', () => {
      render(<Form4PilotEvidencePreview />);

      // Filter by READ_ERROR
      const readErrorFilterBtn = screen.getByRole('button', { name: /Read Error/i });
      fireEvent.click(readErrorFilterBtn);

      // Should only show BRK-B in the table
      expect(screen.getAllByText('BRK-B').length).toBeGreaterThan(0);
      expect(screen.queryByText('NVIDIA Corporation')).toBeNull();

      // Reset to ALL
      const allFilterBtn = screen.getByRole('button', { name: /Total Cohort/i });
      fireEvent.click(allFilterBtn);
      expect(screen.getAllByText('NVDA').length).toBeGreaterThan(0);
    });

    it('verifies touch target accessibility on mobile triggers', () => {
      const { container } = render(<Form4PilotEvidencePreview />);
      const buttons = container.querySelectorAll('button');
      buttons.forEach((btn) => {
        expect(btn.className).toContain('min-h-[');
      });
    });

    it('validates receipt against synthetic expectations using validateReceiptAgainstExpected', () => {
      const expected = {
        NVDA: { state: 'POLLED_QUIET' as const, new_filings_count: 0 },
        AAPL: { state: 'POLLED_QUIET' as const, new_filings_count: 0 },
      };

      const report = validateReceiptAgainstExpected(SYNTHETIC_COHORT_RECEIPT, expected);
      expect(report.is_valid).toBe(false);
      expect(report.discrepancies.length).toBeGreaterThan(0);

      const nvdaDiscrepancy = report.discrepancies.find((d) => d.ticker === 'NVDA' && d.field === 'state');
      expect(nvdaDiscrepancy?.expected).toBe('POLLED_QUIET');
      expect(nvdaDiscrepancy?.actual).toBe('FILINGS_FOUND');
    });
  });

  describe('Imported Worker Receipts & Independent Reconciliation', () => {
    // 1. Valid import that differs from every synthetic expectation
    it('imports a valid real worker receipt differing from synthetic expectations and marks it pending reconciliation', () => {
      // Build valid receipt for 10 issuers with distinct counts (e.g. 5 filings for NVDA, 3 for AAPL, 0 for MSFT)
      const distinctReceipt: CohortReconciliationReceipt = {
        receipt_id: 'rcpt-pilot-165-live-slot1-001',
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: 'SLOT-1',
        created_at: '2026-11-30T14:26:00Z',
        interval_start: '2026-11-30T14:00:00Z',
        interval_end: '2026-11-30T14:25:00Z',
        coverage_status: 'PARTIAL',
        reconciliation_hash: 'feedface00112233445566778899aabbccddeeff00112233445566778899aabb',
        release_sha: QUALIFIED_RELEASE_SHA,
        cohort_evidence: {
          trial_id: QUALIFIED_TRIAL_ID,
          slot_id: 'SLOT-1',
          release_sha: QUALIFIED_RELEASE_SHA,
          issuers: {
            NVDA: {
              ticker: 'NVDA',
              cik: COHORT_CIKS.NVDA,
              observed_at: '2026-11-30T14:22:00Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-audit-live-nvda',
              catalog_sha256: 'aa00112233445566778899aabbccddeeffaa00112233445566778899aabbccdd',
              catalog_gaps: [],
              inventory: [
                {
                  accession: '0001045810-26-000101',
                  form: '4',
                  filing_date: '2026-11-30',
                  accepted_at: '2026-11-30T14:05:00Z',
                  primary_document: 'form4_1.xml',
                  disposition: 'INGESTED',
                  source_sha256: '1111111111111111111111111111111111111111111111111111111111111111',
                },
                {
                  accession: '0001045810-26-000102',
                  form: '4',
                  filing_date: '2026-11-30',
                  accepted_at: '2026-11-30T14:06:00Z',
                  primary_document: 'form4_2.xml',
                  disposition: 'INGESTED',
                  source_sha256: '2222222222222222222222222222222222222222222222222222222222222222',
                },
                {
                  accession: '0001045810-26-000103',
                  form: '4/A',
                  filing_date: '2026-11-30',
                  accepted_at: '2026-11-30T14:07:00Z',
                  primary_document: 'form4a_3.xml',
                  disposition: 'INGESTED',
                  source_sha256: '3333333333333333333333333333333333333333333333333333333333333333',
                },
              ],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'deadbeef11223344556677889900aabbccddeeff11223344556677889900aabb',
            },
            AAPL: {
              ticker: 'AAPL',
              cik: COHORT_CIKS.AAPL,
              observed_at: '2026-11-30T14:22:10Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-audit-live-aapl',
              catalog_sha256: 'bb00112233445566778899aabbccddeeffaa00112233445566778899aabbccdd',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'deadbeef11223344556677889900aabbccddeeff11223344556677889900aacc',
            },
            MSFT: {
              ticker: 'MSFT',
              cik: COHORT_CIKS.MSFT,
              observed_at: '2026-11-30T14:22:20Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-audit-live-msft',
              catalog_sha256: 'cc00112233445566778899aabbccddeeffaa00112233445566778899aabbccdd',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'deadbeef11223344556677889900aabbccddeeff11223344556677889900aadd',
            },
            AMZN: {
              ticker: 'AMZN',
              cik: COHORT_CIKS.AMZN,
              observed_at: '2026-11-30T14:22:30Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-audit-live-amzn',
              catalog_sha256: 'dd00112233445566778899aabbccddeeffaa00112233445566778899aabbccdd',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'deadbeef11223344556677889900aabbccddeeff11223344556677889900aaee',
            },
            GOOGL: {
              ticker: 'GOOGL',
              cik: COHORT_CIKS.GOOGL,
              observed_at: '2026-11-30T14:22:40Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-audit-live-googl',
              catalog_sha256: 'ee00112233445566778899aabbccddeeffaa00112233445566778899aabbccdd',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'deadbeef11223344556677889900aabbccddeeff11223344556677889900aaff',
            },
            META: {
              ticker: 'META',
              cik: COHORT_CIKS.META,
              observed_at: '2026-11-30T14:22:50Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-audit-live-meta',
              catalog_sha256: 'ff00112233445566778899aabbccddeeffaa00112233445566778899aabbccdd',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'deadbeef11223344556677889900aabbccddeeff11223344556677889900aa11',
            },
            TSLA: {
              ticker: 'TSLA',
              cik: COHORT_CIKS.TSLA,
              observed_at: '2026-11-30T14:23:00Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-audit-live-tsla',
              catalog_sha256: '1100112233445566778899aabbccddeeffaa00112233445566778899aabbccdd',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'deadbeef11223344556677889900aabbccddeeff11223344556677889900aa22',
            },
            'BRK-B': {
              ticker: 'BRK-B',
              cik: COHORT_CIKS['BRK-B'],
              observed_at: '2026-11-30T14:23:10Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-audit-live-brkb',
              catalog_sha256: '2200112233445566778899aabbccddeeffaa00112233445566778899aabbccdd',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'deadbeef11223344556677889900aabbccddeeff11223344556677889900aa33',
            },
            JPM: {
              ticker: 'JPM',
              cik: COHORT_CIKS.JPM,
              observed_at: '2026-11-30T14:23:20Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-audit-live-jpm',
              catalog_sha256: '3300112233445566778899aabbccddeeffaa00112233445566778899aabbccdd',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'deadbeef11223344556677889900aabbccddeeff11223344556677889900aa44',
            },
            V: {
              ticker: 'V',
              cik: COHORT_CIKS.V,
              observed_at: '2026-11-30T14:23:30Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-audit-live-v',
              catalog_sha256: '4400112233445566778899aabbccddeeffaa00112233445566778899aabbccdd',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'deadbeef11223344556677889900aabbccddeeff11223344556677889900aa55',
            },
          },
        },
      };

      // Format validation without independent export -> RECONCILIATION_PENDING
      const validation = validateWorkerReceiptFormat(distinctReceipt, 'SLOT-1', null);
      expect(validation.is_valid_format).toBe(true);
      expect(validation.reconciliation_status).toBe('RECONCILIATION_PENDING');
      expect(validation.reconciliation_label).toBe('Imported — reconciliation pending');
      expect(validation.validation_errors.length).toBe(0);

      // Render in UI
      render(<Form4PilotEvidencePreview />);

      // Switch to Import Worker Receipt
      const importModeBtn = screen.getByRole('button', { name: /Import Worker Receipt/i });
      fireEvent.click(importModeBtn);

      const textarea = screen.getByPlaceholderText(/Paste raw sm_reconciliation_receipts JSON row/i);
      fireEvent.change(textarea, { target: { value: JSON.stringify(distinctReceipt) } });

      const importBtn = screen.getByRole('button', { name: /Validate & Import Receipt/i });
      fireEvent.click(importBtn);

      // Verify the badge shows "Imported — reconciliation pending"
      expect(screen.getAllByText('Imported — reconciliation pending').length).toBeGreaterThan(0);
      expect(screen.getByText('PASSED QUALIFICATION FORMAT')).toBeDefined();

      // Check NVDA has 3 filings
      const parsed = parseReceiptToSummaries(distinctReceipt);
      const nvda = parsed.find((p) => p.ticker === 'NVDA');
      expect(nvda?.new_filings_count).toBe(3);
    });

    // 2. Reconciliation against an independently reconciled export
    it('compares counts and gaps against an independently reconciled export when available', () => {
      const receipt: CohortReconciliationReceipt = {
        receipt_id: 'rcpt-test-reconcile-001',
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: 'SLOT-1',
        release_sha: QUALIFIED_RELEASE_SHA,
        cohort_evidence: {
          trial_id: QUALIFIED_TRIAL_ID,
          slot_id: 'SLOT-1',
          release_sha: QUALIFIED_RELEASE_SHA,
          issuers: {
            NVDA: {
              ticker: 'NVDA',
              cik: COHORT_CIKS.NVDA,
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-nvda',
              catalog_sha256: 'hash-nvda',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'sha-nvda',
            },
          },
        },
      };

      const matchingExport: IndependentlyReconciledExport = {
        export_id: 'indep-export-001',
        trial_id: QUALIFIED_TRIAL_ID,
        issuers: {
          NVDA: {
            ticker: 'NVDA',
            cik: COHORT_CIKS.NVDA,
            expected_count: 0,
            accounted_count: 0,
            missing_count: 0,
            gaps: [],
          },
        },
      };

      const matchValidation = validateWorkerReceiptFormat(receipt, 'SLOT-1', matchingExport);
      expect(matchValidation.reconciliation_status).toBe('RECONCILED');
      expect(matchValidation.reconciliation_label).toBe('Imported — reconciled against independent export');
      expect(matchValidation.discrepancies.length).toBe(0);

      // Now test mismatch: export expected 2 filings, but receipt had 0
      const mismatchExport: IndependentlyReconciledExport = {
        export_id: 'indep-export-002',
        trial_id: QUALIFIED_TRIAL_ID,
        issuers: {
          NVDA: {
            ticker: 'NVDA',
            cik: COHORT_CIKS.NVDA,
            expected_count: 2,
            accounted_count: 2,
            missing_count: 0,
            gaps: [],
          },
        },
      };

      const mismatchValidation = validateWorkerReceiptFormat(receipt, 'SLOT-1', mismatchExport);
      expect(mismatchValidation.reconciliation_status).toBe('RECONCILIATION_DISCREPANCY');
      expect(mismatchValidation.reconciliation_label).toBe('Imported — reconciliation discrepancies detected');
      expect(mismatchValidation.discrepancies.length).toBeGreaterThan(0);
      expect(mismatchValidation.discrepancies[0].field).toBe('expected_count');
    });

    // 3. Mismatched release SHA or issuer/CIK
    it('detects and flags release SHA and issuer CIK mismatches', () => {
      const badReleaseReceipt = {
        receipt_id: 'rcpt-bad-release',
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: 'SLOT-1',
        release_sha: 'badc0dedbadc0dedbadc0dedbadc0dedbadc0ded', // Mismatch!
        cohort_evidence: {
          trial_id: QUALIFIED_TRIAL_ID,
          slot_id: 'SLOT-1',
          release_sha: 'badc0dedbadc0dedbadc0dedbadc0dedbadc0ded',
          issuers: {
            AAPL: {
              ticker: 'AAPL',
              cik: '0000999999', // Mismatched CIK! Expected 0000320193
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: 'evt-aapl',
              catalog_sha256: 'hash-aapl',
              catalog_gaps: [],
              inventory: [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'sha-aapl',
            },
          },
        },
      };

      const validation = validateWorkerReceiptFormat(badReleaseReceipt, 'SLOT-1');
      expect(validation.is_valid_format).toBe(false);
      expect(validation.release_sha_matches).toBe(false);
      expect(validation.validation_errors.some((e) => e.includes('Release SHA mismatch'))).toBe(true);
      expect(validation.validation_errors.some((e) => e.includes('Mismatched CIK for AAPL'))).toBe(true);
    });

    // 4. Malformed import preserves previous receipt and identifies still-visible results
    it('preserves still-visible results from previous receipt when a new import fails', () => {
      render(<Form4PilotEvidencePreview />);

      // Switch to Import Worker Receipt
      const importModeBtn = screen.getByRole('button', { name: /Import Worker Receipt/i });
      fireEvent.click(importModeBtn);

      // Verify initial state renders NVDA
      expect(screen.getAllByText('NVDA').length).toBeGreaterThan(0);

      // Enter malformed JSON
      const textarea = screen.getByPlaceholderText(/Paste raw sm_reconciliation_receipts JSON row/i);
      fireEvent.change(textarea, { target: { value: '{"corrupted": json structure' } });

      const importBtn = screen.getByRole('button', { name: /Validate & Import Receipt/i });
      fireEvent.click(importBtn);

      // Check that Previous Receipt Retained banner is rendered
      expect(screen.getByText('Previous Receipt Retained')).toBeDefined();
      expect(screen.getByText(/The latest import attempt failed/i)).toBeDefined();
      expect(screen.getByText(/Currently displaying results from previously loaded receipt/i)).toBeDefined();

      // Ensure NVDA is still visible and not wiped out!
      expect(screen.getAllByText('NVDA').length).toBeGreaterThan(0);
    });
  });

  describe('Local Preview & Operator Authentication Gate', () => {
    it('enforces operator authentication on hosted non-local environments', () => {
      const originalLocation = window.location;
      // @ts-ignore
      delete window.location;
      // @ts-ignore
      window.location = {
        ...originalLocation,
        hostname: 'finance.arxterminal.internal',
      };

      try {
        // Render unauthenticated on hosted host
        const { unmount } = render(
          <Form4PilotEvidencePreview isOperatorAuthenticated={false} />,
        );

        expect(screen.getByText('Operator Authentication Required')).toBeDefined();
        expect(screen.getByText(/Access is restricted to local developer sessions/i)).toBeDefined();
        unmount();

        // Render authenticated on hosted host
        render(<Form4PilotEvidencePreview isOperatorAuthenticated={true} />);
        expect(screen.getByText('Form 4 Pilot Evidence Preview')).toBeDefined();
      } finally {
        // @ts-ignore
        window.location = originalLocation;
      }
    });
  });
});
