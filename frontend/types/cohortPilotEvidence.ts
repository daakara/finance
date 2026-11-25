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
  /**
   * The exact observed_at timestamp from the SEC submissions catalog poll.
   * This represents the actual poll time.
   */
  observed_at: string | null;
  /**
   * The bounded coverage cutoff timestamp accounted through by the worker.
   */
  cutoff: string | null;
  /**
   * Backwards-compatible alias for observed_at/cutoff.
   */
  last_polled_at: string | null;
  interval_start: string | null;
  accounted_through: string | null;
  previous_watermark: string | null;
  catalog_sha256: string | null;
  catalog_gaps: string[];
  /**
   * Accounting alignment with worker:
   * Accounted = INGESTED + ALREADY_ACCOUNTED
   * Newly Ingested = INGESTED
   */
  accounted_count: number | null;
  newly_ingested_count: number | null;
  already_accounted_count: number | null;
  new_filings_count: number | null; // Alias to newly_ingested_count
  form4_count: number | null;
  form4a_count: number | null;
  deferred_count: number | null;
  unresolved_amendments_count: number | null;
  gaps: string[];
  inventory: CohortAccessionItem[];
  artifact_id: string | null;
  receipt_sha256: string | null;
  receipt_id: string | null;
  slot_id?: string | null;
  is_prior_slot?: boolean;
  prior_slot_id?: string | null;
  is_older_receipt?: boolean;
  read_error_message?: string;
}

export interface RawIssuerReceipt {
  cycle_id?: string;
  slot_id?: string;
  grant_sha256?: string;
  ticker: string;
  cik: string;
  request_url?: string;
  response_url?: string;
  observed_at?: string;
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

export interface CohortEvidencePayload {
  version?: number;
  trial_id: string;
  slot_id?: string;
  universe_id?: string;
  release_sha: string;
  issuers: Record<string, RawIssuerReceipt>;
}

export interface CohortReconciliationReceipt {
  receipt_id?: string;
  trial_id?: string;
  slot_id?: string;
  created_at?: string;
  interval_start?: string;
  interval_end?: string;
  coverage_status?: 'PARTIAL' | 'COMPLETE' | string;
  reconciliation_hash?: string;
  release_sha?: string;
  cohort_evidence?: CohortEvidencePayload;
  failure_reasons?: {
    trial_id?: string;
    cohort_evidence?: CohortEvidencePayload;
    [key: string]: any;
  };
  issuers?: Record<string, RawIssuerReceipt>;
  [key: string]: any;
}

export interface IndependentlyReconciledIssuerExport {
  ticker: string;
  cik: string;
  accounted_count?: number; // INGESTED + ALREADY_ACCOUNTED
  newly_ingested_count?: number; // INGESTED
  gaps?: string[];
  cutoff?: string;
  accounted_through?: string;
  state?: EvidenceState;
  verified?: boolean;
}

export interface IndependentlyReconciledExport {
  export_id?: string;
  receipt_id?: string;
  trial_id?: string;
  slot_id?: string;
  reconciled_at?: string;
  reconciled_by?: string;
  issuers: Record<string, IndependentlyReconciledIssuerExport>;
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

export interface WorkerReceiptValidation {
  is_valid_format: boolean;
  trial_id: string | null;
  trial_id_matches: boolean;
  slot_id: string | null;
  slot_id_valid: boolean;
  is_prior_slot: boolean;
  prior_slot_id: string | null;
  release_sha: string | null;
  release_sha_matches: boolean;
  issuers_matched: boolean;
  issuer_count: number;
  missing_issuers: string[];
  mismatched_issuers: string[];
  receipt_fields_valid: boolean;
  validation_errors: string[];
  reconciliation_status: 'RECONCILIATION_PENDING' | 'RECONCILED' | 'RECONCILIATION_DISCREPANCY';
  reconciliation_label: string;
  reconciliation_reasons: string[];
  discrepancies: CohortValidationDiscrepancy[];
}
