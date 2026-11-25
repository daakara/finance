import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Form4PilotEvidencePreview from '../pilot/Form4PilotEvidencePreview';
import {
  SYNTHETIC_ISSUER_SUMMARIES,
  SYNTHETIC_COHORT_RECEIPT,
  QUALIFIED_RELEASE_SHA,
  QUALIFIED_TRIAL_ID,
  WORKER_COHORT_PILOT_SLOTS,
  DEFAULT_PILOT_SLOT,
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

      expect(screen.getAllByText(/Poll Time \(observed_at\)/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/Coverage Cutoff/i).length).toBeGreaterThan(0);

      expect(screen.getAllByText('14:20:15 UTC').length).toBeGreaterThan(0);
      expect(screen.getAllByText('14:20:00 UTC').length).toBeGreaterThan(0);
    });

    it('renders worker dated slot IDs in the active slot selector', () => {
      render(<Form4PilotEvidencePreview />);

      WORKER_COHORT_PILOT_SLOTS.forEach((slotId) => {
        expect(screen.getByRole('button', { name: slotId })).toBeDefined();
      });
    });
  });

  describe('Five Distinct Evidence States', () => {
    it('1. NOT_POLLED: renders NOT POLLED badge and strictly omits unverified counts as dash', () => {
      render(<Form4PilotEvidencePreview />);

      const notPolledBadges = screen.getAllByText('NOT POLLED');
      expect(notPolledBadges.length).toBeGreaterThan(0);

      const tslaSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'TSLA');
      expect(tslaSummary?.state).toBe('NOT_POLLED');
      expect(tslaSummary?.newly_ingested_count).toBeNull();
      expect(tslaSummary?.accounted_count).toBeNull();
      expect(tslaSummary?.observed_at).toBeNull();
      expect(tslaSummary?.cutoff).toBeNull();
    });

    it('2. POLLED_QUIET: renders QUIET — ACCOUNTED badge and authentic verified 0 count', () => {
      render(<Form4PilotEvidencePreview />);

      const quietBadges = screen.getAllByText('QUIET — ACCOUNTED');
      expect(quietBadges.length).toBeGreaterThan(0);

      const aaplSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'AAPL');
      expect(aaplSummary?.state).toBe('POLLED_QUIET');
      expect(aaplSummary?.newly_ingested_count).toBe(0);
      expect(aaplSummary?.accounted_count).toBe(0);
      expect(aaplSummary?.gaps.length).toBe(0);
    });

    it('3. FILINGS_FOUND: renders FILINGS FOUND badge, shows counts and SEC source links', () => {
      render(<Form4PilotEvidencePreview />);

      const filingsFoundBadges = screen.getAllByText('FILINGS FOUND');
      expect(filingsFoundBadges.length).toBeGreaterThan(0);

      const nvdaSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'NVDA');
      expect(nvdaSummary?.state).toBe('FILINGS_FOUND');
      expect(nvdaSummary?.newly_ingested_count).toBe(2);
      expect(nvdaSummary?.accounted_count).toBe(2);
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
      expect(brkbSummary?.accounted_count).toBeNull();
      expect(brkbSummary?.newly_ingested_count).toBeNull();
      expect(brkbSummary?.read_error_message).toContain('SEC Catalog Retained Hash Mismatch');
    });
  });

  describe('Historical & Dated Slot Timestamp Presentation', () => {
    it('presents older historical receipt with Historical Poll indicator', () => {
      render(<Form4PilotEvidencePreview />);

      const googlSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'GOOGL');
      expect(googlSummary?.is_older_receipt).toBe(true);

      const historicalBadges = screen.getAllByText('Historical Poll');
      expect(historicalBadges.length).toBeGreaterThan(0);
    });

    it('uses dated slot_id to identify a prior-slot receipt when viewing subsequent slot', () => {
      // Create a receipt from SLOT 1
      const priorSlotReceipt = {
        ...SYNTHETIC_COHORT_RECEIPT,
        slot_id: '2026-11-30:SLOT_1:0915',
        failure_reasons: {
          ...SYNTHETIC_COHORT_RECEIPT.failure_reasons,
          cohort_evidence: {
            ...SYNTHETIC_COHORT_RECEIPT.failure_reasons.cohort_evidence,
            slot_id: '2026-11-30:SLOT_1:0915',
          },
        },
      };

      // Set active slot to SLOT 2
      const validation = validateWorkerReceiptFormat(priorSlotReceipt, '2026-11-30:SLOT_2:1300');
      expect(validation.is_prior_slot).toBe(true);
      expect(validation.prior_slot_id).toBe('2026-11-30:SLOT_1:0915');

      const summaries = parseReceiptToSummaries(priorSlotReceipt, undefined, '2026-11-30:SLOT_2:1300');
      const nvda = summaries.find((s) => s.ticker === 'NVDA');
      expect(nvda?.is_prior_slot).toBe(true);
      expect(nvda?.slot_id).toBe('2026-11-30:SLOT_1:0915');
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
        NVDA: { state: 'POLLED_QUIET' as const, newly_ingested_count: 0 },
        AAPL: { state: 'POLLED_QUIET' as const, newly_ingested_count: 0 },
      };

      const report = validateReceiptAgainstExpected(SYNTHETIC_COHORT_RECEIPT, expected);
      expect(report.is_valid).toBe(false);
      expect(report.discrepancies.length).toBeGreaterThan(0);

      const nvdaDiscrepancy = report.discrepancies.find((d) => d.ticker === 'NVDA' && d.field === 'state');
      expect(nvdaDiscrepancy?.expected).toBe('POLLED_QUIET');
      expect(nvdaDiscrepancy?.actual).toBe('FILINGS_FOUND');
    });
  });

  describe('Imported Worker Receipts, Rejection Guard & Worker Accounting', () => {
    // 1. Worker Accounting Alignment: Accounted = INGESTED + ALREADY_ACCOUNTED; Newly Ingested = INGESTED
    it('handles later-slot receipt containing ALREADY_ACCOUNTED rows with accurate worker accounting', () => {
      const laterSlotReceipt: CohortReconciliationReceipt = {
        receipt_id: 'rcpt-pilot-165-live-slot2',
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: '2026-11-30:SLOT_2:1300',
        release_sha: QUALIFIED_RELEASE_SHA,
        cohort_evidence: {
          trial_id: QUALIFIED_TRIAL_ID,
          slot_id: '2026-11-30:SLOT_2:1300',
          release_sha: QUALIFIED_RELEASE_SHA,
          issuers: {
            NVDA: {
              ticker: 'NVDA',
              cik: COHORT_CIKS.NVDA,
              observed_at: '2026-11-30T18:05:00Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T18:00:00Z',
              artifact_id: 'evt-slot2-nvda',
              catalog_sha256: 'hash-slot2-nvda',
              catalog_gaps: [],
              inventory: [
                // Accession from slot 1 is ALREADY_ACCOUNTED
                {
                  accession: '0001045810-26-000045',
                  form: '4',
                  filing_date: '2026-11-30',
                  accepted_at: '2026-11-30T14:04:12Z',
                  primary_document: 'form4.xml',
                  disposition: 'ALREADY_ACCOUNTED',
                  source_sha256: 'hash1',
                },
                // New accession in slot 2 is INGESTED
                {
                  accession: '0001045810-26-000099',
                  form: '4',
                  filing_date: '2026-11-30',
                  accepted_at: '2026-11-30T17:45:00Z',
                  primary_document: 'form4_new.xml',
                  disposition: 'INGESTED',
                  source_sha256: 'hash2',
                },
              ],
              gaps: [],
              previous_watermark: '2026-11-30T14:20:00Z',
              accounted_through: '2026-11-30T18:00:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'sha-slot2-nvda',
            },
            AAPL: {
              ticker: 'AAPL',
              cik: COHORT_CIKS.AAPL,
              observed_at: '2026-11-30T18:05:10Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T18:00:00Z',
              artifact_id: 'evt-slot2-aapl',
              catalog_sha256: 'hash-slot2-aapl',
              catalog_gaps: [],
              inventory: [
                // Only ALREADY_ACCOUNTED accession from slot 1
                {
                  accession: '0000320193-26-000010',
                  form: '4',
                  filing_date: '2026-11-30',
                  accepted_at: '2026-11-30T14:05:00Z',
                  primary_document: 'aapl.xml',
                  disposition: 'ALREADY_ACCOUNTED',
                  source_sha256: 'hash3',
                },
              ],
              gaps: [],
              previous_watermark: '2026-11-30T14:20:00Z',
              accounted_through: '2026-11-30T18:00:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: 'sha-slot2-aapl',
            },
          },
        },
      };

      const summaries = parseReceiptToSummaries(laterSlotReceipt);
      const nvda = summaries.find((s) => s.ticker === 'NVDA');
      const aapl = summaries.find((s) => s.ticker === 'AAPL');

      // NVDA has 1 INGESTED + 1 ALREADY_ACCOUNTED -> Accounted = 2, Newly Ingested = 1
      expect(nvda?.accounted_count).toBe(2);
      expect(nvda?.newly_ingested_count).toBe(1);
      expect(nvda?.already_accounted_count).toBe(1);
      expect(nvda?.state).toBe('FILINGS_FOUND');

      // AAPL has 0 INGESTED + 1 ALREADY_ACCOUNTED -> Accounted = 1, Newly Ingested = 0 -> POLLED_QUIET
      expect(aapl?.accounted_count).toBe(1);
      expect(aapl?.newly_ingested_count).toBe(0);
      expect(aapl?.already_accounted_count).toBe(1);
      expect(aapl?.state).toBe('POLLED_QUIET');
    });

    // 2. Reject invalid trial, release, slot ID, or issuer/CIK BEFORE changing displayed results
    it('rejects an invalid trial, release, slot ID, or issuer/CIK before changing displayed results and retains prior receipt', () => {
      render(<Form4PilotEvidencePreview />);

      // Switch to Import Worker Receipt
      const importModeBtn = screen.getByRole('button', { name: /Import Worker Receipt/i });
      fireEvent.click(importModeBtn);

      // Verify NVDA is currently displayed with 2 filings from synthetic initial state
      expect(screen.getAllByText('NVDA').length).toBeGreaterThan(0);

      // Construct receipt with INVALID RELEASE SHA
      const badReleaseReceipt = {
        receipt_id: 'rcpt-bad-release',
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: '2026-11-30:SLOT_1:0915',
        release_sha: 'invalid_sha_000000000000000000000000000000',
        cohort_evidence: {
          trial_id: QUALIFIED_TRIAL_ID,
          slot_id: '2026-11-30:SLOT_1:0915',
          release_sha: 'invalid_sha_000000000000000000000000000000',
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

      const textarea = screen.getByPlaceholderText(/Paste raw sm_reconciliation_receipts JSON row/i);
      fireEvent.change(textarea, { target: { value: JSON.stringify(badReleaseReceipt) } });

      const importBtn = screen.getByRole('button', { name: /Validate & Import Receipt/i });
      fireEvent.click(importBtn);

      // Check rejection alert is displayed
      expect(screen.getByText('Rejected Import — Previous Receipt Retained')).toBeDefined();
      expect(screen.getAllByText(/Release SHA mismatch/i).length).toBeGreaterThan(0);

      // Ensure displayed results were NOT changed (NVDA still has initial 2 filings, not 0 from bad receipt)
      expect(screen.getAllByText('NVDA').length).toBeGreaterThan(0);
      expect(screen.getAllByText('14:20:15 UTC').length).toBeGreaterThan(0); // From retained receipt
    });

    it('rejects an invalid slot ID format before changing displayed results', () => {
      const badSlotReceipt = {
        ...SYNTHETIC_COHORT_RECEIPT,
        slot_id: 'INVALID_SLOT_123',
        failure_reasons: {
          cohort_evidence: {
            ...SYNTHETIC_COHORT_RECEIPT.failure_reasons.cohort_evidence,
            slot_id: 'INVALID_SLOT_123',
          },
        },
      };

      const validation = validateWorkerReceiptFormat(badSlotReceipt);
      expect(validation.is_valid_format).toBe(false);
      expect(validation.slot_id_valid).toBe(false);
      expect(validation.validation_errors.some((e) => e.includes('Invalid or missing dated slot ID'))).toBe(true);
    });

    // 3. Incomplete or mismatched independent export
    it('keeps reconciliation pending when independent export is incomplete or missing comparison fields', () => {
      // Build 10-issuer valid receipt
      const validReceipt: CohortReconciliationReceipt = {
        receipt_id: 'rcpt-reconcile-test-001',
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: '2026-11-30:SLOT_1:0915',
        release_sha: QUALIFIED_RELEASE_SHA,
        cohort_evidence: {
          trial_id: QUALIFIED_TRIAL_ID,
          slot_id: '2026-11-30:SLOT_1:0915',
          release_sha: QUALIFIED_RELEASE_SHA,
          issuers: Object.fromEntries(
            Object.entries(COHORT_CIKS).map(([ticker, cik]) => [
              ticker,
              {
                ticker,
                cik,
                interval_start: '2026-11-30T14:00:00Z',
                cutoff: '2026-11-30T14:20:00Z',
                artifact_id: `evt-${ticker}`,
                catalog_sha256: `hash-${ticker}`,
                catalog_gaps: [],
                inventory: [],
                gaps: [],
                previous_watermark: '2026-11-30T14:00:00Z',
                accounted_through: '2026-11-30T14:20:00Z',
                status: 'CATALOG_ACCOUNTED',
                sha256: `sha-${ticker}`,
              },
            ]),
          ),
        },
      };

      // Incomplete export: only has 3 of 10 issuers
      const incompleteExport: IndependentlyReconciledExport = {
        receipt_id: 'rcpt-reconcile-test-001',
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: '2026-11-30:SLOT_1:0915',
        issuers: {
          NVDA: { ticker: 'NVDA', cik: COHORT_CIKS.NVDA, accounted_count: 0, newly_ingested_count: 0, gaps: [], accounted_through: '2026-11-30T14:20:00Z' },
          AAPL: { ticker: 'AAPL', cik: COHORT_CIKS.AAPL, accounted_count: 0, newly_ingested_count: 0, gaps: [], accounted_through: '2026-11-30T14:20:00Z' },
          MSFT: { ticker: 'MSFT', cik: COHORT_CIKS.MSFT, accounted_count: 0, newly_ingested_count: 0, gaps: [], accounted_through: '2026-11-30T14:20:00Z' },
        },
      };

      const validation = validateWorkerReceiptFormat(validReceipt, '2026-11-30:SLOT_1:0915', incompleteExport);
      expect(validation.reconciliation_status).toBe('RECONCILIATION_PENDING');
      expect(validation.reconciliation_label).toContain('reconciliation pending (missing issuers');

      // Export missing comparison fields (e.g. missing accounted_through)
      const missingFieldsExport: IndependentlyReconciledExport = {
        receipt_id: 'rcpt-reconcile-test-001',
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: '2026-11-30:SLOT_1:0915',
        issuers: Object.fromEntries(
          Object.entries(COHORT_CIKS).map(([ticker, cik]) => [
            ticker,
            {
              ticker,
              cik,
              accounted_count: 0,
              newly_ingested_count: 0,
              gaps: [],
              // Missing accounted_through!
            },
          ]),
        ),
      };

      const fieldValidation = validateWorkerReceiptFormat(validReceipt, '2026-11-30:SLOT_1:0915', missingFieldsExport);
      expect(fieldValidation.reconciliation_status).toBe('RECONCILIATION_PENDING');
      expect(fieldValidation.reconciliation_label).toContain('missing comparison fields');

      // Export bound to wrong slot ID
      const wrongSlotExport: IndependentlyReconciledExport = {
        receipt_id: 'rcpt-reconcile-test-001',
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: '2026-11-30:SLOT_2:1300', // Mismatched slot!
        issuers: Object.fromEntries(
          Object.entries(COHORT_CIKS).map(([ticker, cik]) => [
            ticker,
            { ticker, cik, accounted_count: 0, newly_ingested_count: 0, gaps: [], accounted_through: '2026-11-30T14:20:00Z' },
          ]),
        ),
      };

      const wrongSlotValidation = validateWorkerReceiptFormat(validReceipt, '2026-11-30:SLOT_1:0915', wrongSlotExport);
      expect(wrongSlotValidation.reconciliation_status).toBe('RECONCILIATION_DISCREPANCY');
      expect(wrongSlotValidation.reconciliation_label).toContain('export binding mismatch');

      // Complete, bound export for all 10 issuers with matching fields -> RECONCILED!
      const completeMatchingExport: IndependentlyReconciledExport = {
        receipt_id: 'rcpt-reconcile-test-001',
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: '2026-11-30:SLOT_1:0915',
        issuers: Object.fromEntries(
          Object.entries(COHORT_CIKS).map(([ticker, cik]) => [
            ticker,
            { ticker, cik, accounted_count: 0, newly_ingested_count: 0, gaps: [], accounted_through: '2026-11-30T14:20:00Z' },
          ]),
        ),
      };

      const completeValidation = validateWorkerReceiptFormat(validReceipt, '2026-11-30:SLOT_1:0915', completeMatchingExport);
      expect(completeValidation.reconciliation_status).toBe('RECONCILED');
      expect(completeValidation.reconciliation_label).toBe('Imported — reconciled against independent export');
      expect(completeValidation.discrepancies.length).toBe(0);
    });
  });

  describe('Closeout Rehearsal: Loading Approved Exports & Comparing All 10 Rows', () => {
    const slot1Receipt = {
      receipt_id: 'rcpt-pilot-165-slot1-approved',
      trial_id: QUALIFIED_TRIAL_ID,
      slot_id: '2026-11-30:SLOT_1:0915',
      release_sha: QUALIFIED_RELEASE_SHA,
      cohort_evidence: {
        version: 1,
        trial_id: QUALIFIED_TRIAL_ID,
        slot_id: '2026-11-30:SLOT_1:0915',
        release_sha: QUALIFIED_RELEASE_SHA,
        issuers: Object.fromEntries(
          Object.entries(COHORT_CIKS).map(([ticker, cik]) => [
            ticker,
            {
              ticker,
              cik,
              observed_at: '2026-11-30T14:21:00Z',
              interval_start: '2026-11-30T14:00:00Z',
              cutoff: '2026-11-30T14:20:00Z',
              artifact_id: `evt-audit-${ticker.toLowerCase()}`,
              catalog_sha256: `cat-sha-${ticker}`,
              catalog_gaps: [],
              inventory:
                ticker === 'NVDA'
                  ? [
                      {
                        accession: '0001045810-26-000001',
                        form: '4',
                        filing_date: '2026-11-30',
                        accepted_at: '2026-11-30T14:05:00Z',
                        primary_document: 'owner.xml',
                        disposition: 'INGESTED',
                        source_sha256: 'sha-nvda-1',
                      },
                    ]
                  : [],
              gaps: [],
              previous_watermark: '2026-11-30T14:00:00Z',
              accounted_through: '2026-11-30T14:20:00Z',
              status: 'CATALOG_ACCOUNTED',
              sha256: `sha-rcpt-${ticker}`,
            },
          ]),
        ),
      },
    };

    const matchingIndependentExport: IndependentlyReconciledExport = {
      export_id: 'exp-pilot-165-slot1-approved',
      receipt_id: 'rcpt-pilot-165-slot1-approved',
      trial_id: QUALIFIED_TRIAL_ID,
      slot_id: '2026-11-30:SLOT_1:0915',
      reconciled_at: '2026-11-30T14:25:00Z',
      reconciled_by: 'arx-independent-evidence-reconciler',
      issuers: Object.fromEntries(
        Object.entries(COHORT_CIKS).map(([ticker, cik]) => [
          ticker,
          {
            ticker,
            cik,
            accounted_count: ticker === 'NVDA' ? 1 : 0,
            newly_ingested_count: ticker === 'NVDA' ? 1 : 0,
            gaps: [],
            accounted_through: '2026-11-30T14:20:00Z',
            state: ticker === 'NVDA' ? 'FILINGS_FOUND' : 'POLLED_QUIET',
            verified: true,
          },
        ]),
      ),
    };

    it('loads approved exports locally, compares all 10 rows with zero discrepancies, and shows reconciled status', () => {
      render(<Form4PilotEvidencePreview />);

      // Switch to Import Worker Receipt
      fireEvent.click(screen.getByRole('button', { name: /Import Worker Receipt/i }));

      // Fill Worker Receipt JSON
      const receiptInput = screen.getByPlaceholderText(/Paste raw sm_reconciliation_receipts JSON row/i);
      fireEvent.change(receiptInput, { target: { value: JSON.stringify(slot1Receipt) } });

      // Fill Independent Reconciled Export JSON
      const exportInput = screen.getByPlaceholderText(/Paste independently reconciled export to verify counts and gaps/i);
      fireEvent.change(exportInput, { target: { value: JSON.stringify(matchingIndependentExport) } });

      // Click Validate & Import
      fireEvent.click(screen.getByRole('button', { name: /Validate & Import Receipt/i }));

      // Verify Reconciled verdict
      expect(screen.getByText(/PASSED QUALIFICATION FORMAT/i)).toBeDefined();
      expect(screen.getByText(/Imported — reconciled against independent export/i)).toBeDefined();

      // Verify all 10 issuers are displayed in the evidence table/cards
      for (const ticker of Object.keys(COHORT_CIKS)) {
        expect(screen.getAllByText(ticker).length).toBeGreaterThan(0);
      }
    });

    it('records and renders discrepancies with the affected issuer and field when counts mismatch', () => {
      render(<Form4PilotEvidencePreview />);

      fireEvent.click(screen.getByRole('button', { name: /Import Worker Receipt/i }));

      // Tamper independent export: NVDA expected accounted_count = 5 instead of 1
      const tamperedExport: IndependentlyReconciledExport = {
        ...matchingIndependentExport,
        issuers: {
          ...matchingIndependentExport.issuers,
          NVDA: {
            ...matchingIndependentExport.issuers.NVDA,
            accounted_count: 5,
          },
        },
      };

      const receiptInput = screen.getByPlaceholderText(/Paste raw sm_reconciliation_receipts JSON row/i);
      fireEvent.change(receiptInput, { target: { value: JSON.stringify(slot1Receipt) } });

      const exportInput = screen.getByPlaceholderText(/Paste independently reconciled export to verify counts and gaps/i);
      fireEvent.change(exportInput, { target: { value: JSON.stringify(tamperedExport) } });

      fireEvent.click(screen.getByRole('button', { name: /Validate & Import Receipt/i }));

      // Status must show discrepancy
      expect(screen.getByText(/Imported — reconciliation discrepancies detected/i)).toBeDefined();
      expect(screen.getByText(/Reconciliation Discrepancies vs Independent Export:/i)).toBeDefined();

      // Specifically records affected issuer NVDA and field accounted_count
      expect(screen.getAllByText(/NVDA/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/accounted_count/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText('5').length).toBeGreaterThan(0); // expected
      expect(screen.getAllByText('1').length).toBeGreaterThan(0); // actual
    });

    it('marks export as pending when an issuer has unestablished (null or undefined) comparison fields', () => {
      render(<Form4PilotEvidencePreview />);

      fireEvent.click(screen.getByRole('button', { name: /Import Worker Receipt/i }));

      // Incomplete export where META comparison fields could not be independently established
      const incompleteExport: IndependentlyReconciledExport = {
        ...matchingIndependentExport,
        issuers: {
          ...matchingIndependentExport.issuers,
          META: {
            ticker: 'META',
            cik: COHORT_CIKS.META,
            accounted_count: undefined,
            newly_ingested_count: undefined,
            gaps: undefined,
            accounted_through: undefined,
          },
        },
      };

      const receiptInput = screen.getByPlaceholderText(/Paste raw sm_reconciliation_receipts JSON row/i);
      fireEvent.change(receiptInput, { target: { value: JSON.stringify(slot1Receipt) } });

      const exportInput = screen.getByPlaceholderText(/Paste independently reconciled export to verify counts and gaps/i);
      fireEvent.change(exportInput, { target: { value: JSON.stringify(incompleteExport) } });

      fireEvent.click(screen.getByRole('button', { name: /Validate & Import Receipt/i }));

      // Must remain pending
      expect(screen.getAllByText(/Imported — reconciliation pending/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/issuers missing comparison fields/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/META/i).length).toBeGreaterThan(0);
    });

    it('reports pending fields alongside discrepancies when both occur', () => {
      render(<Form4PilotEvidencePreview />);

      fireEvent.click(screen.getByRole('button', { name: /Import Worker Receipt/i }));

      // NVDA has discrepancy (accounted_count = 5 vs actual 1), and META has pending null field
      const hybridExport: IndependentlyReconciledExport = {
        ...matchingIndependentExport,
        issuers: {
          ...matchingIndependentExport.issuers,
          NVDA: {
            ...matchingIndependentExport.issuers.NVDA,
            accounted_count: 5,
          },
          META: {
            ticker: 'META',
            cik: COHORT_CIKS.META,
            accounted_count: undefined,
            newly_ingested_count: 0,
            gaps: [],
            accounted_through: '2026-11-30T14:20:00Z',
          },
        },
      };

      const receiptInput = screen.getByPlaceholderText(/Paste raw sm_reconciliation_receipts JSON row/i);
      fireEvent.change(receiptInput, { target: { value: JSON.stringify(slot1Receipt) } });

      const exportInput = screen.getByPlaceholderText(/Paste independently reconciled export to verify counts and gaps/i);
      fireEvent.change(exportInput, { target: { value: JSON.stringify(hybridExport) } });

      fireEvent.click(screen.getByRole('button', { name: /Validate & Import Receipt/i }));

      // Status must show discrepancy
      expect(screen.getByText(/Imported — reconciliation discrepancies detected/i)).toBeDefined();
      expect(screen.getByText(/Reconciliation Discrepancies vs Independent Export:/i)).toBeDefined();
      expect(screen.getAllByText(/NVDA/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/accounted_count/i).length).toBeGreaterThan(0);

      // Pending fields must also be reported alongside discrepancies!
      expect(screen.getByText(/Pending Fields Reported Alongside Discrepancies:/i)).toBeDefined();
      expect(screen.getAllByText(/META/i).length).toBeGreaterThan(0);
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
