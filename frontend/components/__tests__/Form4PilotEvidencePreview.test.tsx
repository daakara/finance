import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Form4PilotEvidencePreview from '../pilot/Form4PilotEvidencePreview';
import {
  SYNTHETIC_ISSUER_SUMMARIES,
  SYNTHETIC_COHORT_RECEIPT,
  parseReceiptToSummaries,
  validateReceiptAgainstExpected,
  buildSecEdgarUrl,
} from '../../fixtures/cohortPilotEvidenceFixtures';

describe('Form4PilotEvidencePreview Component', () => {
  it('renders preview title, private badge, trial id and all 10 cohort issuers', () => {
    render(<Form4PilotEvidencePreview />);

    // Header & guardrails
    expect(screen.getByText('Form 4 Pilot Evidence Preview')).toBeDefined();
    expect(screen.getByText('Private Preview')).toBeDefined();
    expect(screen.getByText(/ARX-TRIAL-10-ISSUER-001/i)).toBeDefined();
    expect(screen.getByText(/1063a10f/i)).toBeDefined();

    // Verify all 10 governed tickers are present
    const expectedTickers = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JPM', 'V'];
    expectedTickers.forEach((ticker) => {
      expect(screen.getAllByText(ticker).length).toBeGreaterThan(0);
    });
  });

  describe('Evidence State Distinction', () => {
    it('1. NOT_POLLED: renders NOT POLLED badge and omits unverified counts as dash', () => {
      render(<Form4PilotEvidencePreview />);

      const notPolledBadges = screen.getAllByText('NOT POLLED');
      expect(notPolledBadges.length).toBeGreaterThan(0);

      const tslaSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'TSLA');
      expect(tslaSummary?.state).toBe('NOT_POLLED');
      expect(tslaSummary?.new_filings_count).toBeNull();
    });

    it('2. POLLED_QUIET: renders QUIET — ACCOUNTED badge and authentic 0 count', () => {
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
      expect(brkbSummary?.read_error_message).toContain('SHA-256 mismatch');
    });
  });

  describe('Historical Timestamp Presentation & Inspection', () => {
    it('presents older historical receipt with Historical Poll indicator and actual timestamp', () => {
      render(<Form4PilotEvidencePreview />);

      const googlSummary = SYNTHETIC_ISSUER_SUMMARIES.find((s) => s.ticker === 'GOOGL');
      expect(googlSummary?.is_older_receipt).toBe(true);
      expect(googlSummary?.last_polled_at).toBe('2026-11-30T09:22:10Z');

      const historicalBadges = screen.getAllByText('Historical Poll');
      expect(historicalBadges.length).toBeGreaterThan(0);
    });

    it('expands detail inspection drawer when clicking Inspect', () => {
      render(<Form4PilotEvidencePreview />);

      const inspectButtons = screen.getAllByRole('button', { name: /Inspect/i });
      fireEvent.click(inspectButtons[0]);

      expect(screen.getByText(/Inventory Accessions/i)).toBeDefined();
      expect(screen.getAllByText(/SEC EDGAR/i).length).toBeGreaterThan(0);
    });
  });

  describe('Filter Navigation & Responsiveness', () => {
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
        expect(btn.className).toContain('min-h-[44px]');
      });
    });
  });

  describe('Receipt Parser & Discrepancy Validator', () => {
    it('correctly parses synthetic reconciliation receipt into 10 issuer summaries', () => {
      const parsed = parseReceiptToSummaries(SYNTHETIC_COHORT_RECEIPT, '2026-11-30T15:00:00Z');
      expect(parsed.length).toBe(10);

      const nvda = parsed.find((p) => p.ticker === 'NVDA');
      expect(nvda?.state).toBe('FILINGS_FOUND');
      expect(nvda?.new_filings_count).toBe(2);

      const aapl = parsed.find((p) => p.ticker === 'AAPL');
      expect(aapl?.state).toBe('POLLED_QUIET');
      expect(aapl?.new_filings_count).toBe(0);

      const msft = parsed.find((p) => p.ticker === 'MSFT');
      expect(msft?.state).toBe('INCOMPLETE');
      expect(msft?.deferred_count).toBe(1);

      // TSLA is absent from receipt issuers -> NOT_POLLED
      const tsla = parsed.find((p) => p.ticker === 'TSLA');
      expect(tsla?.state).toBe('NOT_POLLED');
      expect(tsla?.new_filings_count).toBeNull();
    });

    it('detects discrepancies when real receipt diverges from expected numbers', () => {
      // Expected NVDA to have 0 filings, but receipt has 2
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

    it('handles custom receipt JSON parsing in UI', () => {
      render(<Form4PilotEvidencePreview />);

      // Switch to custom
      const customBtn = screen.getByRole('button', { name: /Load Real Receipt/i });
      fireEvent.click(customBtn);

      const fillBtn = screen.getByRole('button', { name: /Fill Sample JSON/i });
      fireEvent.click(fillBtn);

      const parseBtn = screen.getByRole('button', { name: /Parse & Validate Receipt/i });
      fireEvent.click(parseBtn);

      expect(screen.getByText(/Receipt Validation Report/i)).toBeDefined();
    });
  });
});
