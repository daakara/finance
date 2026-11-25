/**
 * ARX Terminal — Smart Money Form 4 Cohort Pilot Evidence Types
 * Aligned with candidate 1063a10fed1ed312feadeb671980a642e757e601
 * Issue #165 / Trial ARX-TRIAL-10-ISSUER-001
 */

export type AccessionDisposition =
  | 'INGESTED'
  | 'ALREADY_ACCOUNTED'
  | 'DEFERRED'
  | 'AMENDMENT_UNRESOLVED';

export type EvidenceState =
  | 'NOT_POLLED'
  | 'POLLED_QUIET'
  | 'FILINGS_FOUND'
  | 'INCOMPLETE'
  | 'READ_ERROR';

export interface CohortAccessionItem {
  accession: string;
  form: '4' | '4/A' | string;
  filing_date: string;
  accepted_at: string;
  primary_document: string;
  disposition: AccessionDisposition;
  source_sha256: string | null;
}

export interface IssuerEvidenceSummary {
  ticker: string;
  cik: string;
  company_name: string;
  state: EvidenceState;
  last_polled_at: string | null;
  interval_start: string | null;
  cutoff: string | null;
  accounted_through: string | null;
  previous_watermark: string | null;
  catalog_sha256: string | null;
  catalog_gaps: string[];
  new_filings_count: number | null; // null on READ_ERROR or NOT_POLLED
  form4_count: number | null;
  form4a_count: number | null;
  deferred_count: number | null;
  unresolved_amendments_count: number | null;
  gaps: string[];
  inventory: CohortAccessionItem[];
  artifact_id: string | null;
  receipt_sha256: string | null;
  receipt_id: string | null;
  is_older_receipt?: boolean;
  read_error_message?: string;
}

export interface RawIssuerReceipt {
  ticker: string;
  cik: string;
  interval_start: string;
  cutoff: string;
  artifact_id: string;
  catalog_sha256: string;
  catalog_gaps: string[];
  inventory: Array<{
    accession: string;
    form: string;
    filing_date: string;
    accepted_at: string;
    primary_document: string;
    disposition: AccessionDisposition;
    source_sha256: string | null;
  }>;
  gaps: string[];
  previous_watermark: string | null;
  accounted_through: string | null;
  status: 'CATALOG_ACCOUNTED' | 'PARTIAL' | string;
  sha256: string;
}

export interface CohortReconciliationReceipt {
  receipt_id: string;
  trial_id: string;
  created_at: string;
  interval_start: string;
  interval_end: string;
  coverage_status: 'PARTIAL' | 'COMPLETE' | string;
  reconciliation_hash: string;
  release_sha: string;
  failure_reasons: {
    trial_id: string;
    cohort_evidence: {
      trial_id: string;
      universe_id: string;
      release_sha: string;
      issuers: Record<string, RawIssuerReceipt>;
    };
  };
}

export interface CohortValidationDiscrepancy {
  ticker: string;
  field: string;
  expected: any;
  actual: any;
  severity: 'ERROR' | 'WARNING';
}

export interface ReceiptValidationReport {
  receipt_id: string;
  validated_at: string;
  is_valid: boolean;
  issuer_count: number;
  discrepancies: CohortValidationDiscrepancy[];
}
