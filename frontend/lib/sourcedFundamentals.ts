/**
 * frontend/lib/sourcedFundamentals.ts
 *
 * Client and SSG data accessor for SEC EDGAR-sourced company fundamentals evidence.
 * Reads the immutable snapshot generated from analyst_dashboard research ledger.
 * Fails closed (returns null) for uncataloged assets.
 */

import snapshotData from "./sourcedFundamentalsSnapshot.json";

export interface SourcedReportedFact {
  metric_key: string;
  value: number | null;
  units: string | null;
  concept: string | null;
  accession_number: string | null;
  form: string | null;
  period_start: string | null;
  period_end: string | null;
  filing_date: string | null;
  retrieval_time?: string | null;
  status: string | null;
  source_url: string | null;
  formula?: string | null;
  input_references?: string[];
  notes?: string | null;
  duration_type: "INSTANT" | "DURATION";
}

export interface SourcedDerivedMetric {
  value: number | null;
  units?: string;
  status?: string;
  label?: string;
  formula?: string;
  notes?: string;
  period_end?: string;
}

export interface SourcedCompanyIdentity {
  symbol: string;
  company_name: string;
  cik: string | null;
  primary_exchange: string | null;
  sector_industry: string | null;
  business_model?: string | null;
  fiscal_year_end?: string | null;
  latest_reported_period: string | null;
  latest_filing_date: string | null;
  sec_edgar_company_url: string | null;
}

export interface SourcedResearchDisposition {
  disposition: string | null;
  disposition_rationale: string | null;
  growth_durability: string | null;
  reinvestment_economics: string | null;
  balance_sheet_resilience: string | null;
  evidence_sufficiency: string | null;
  valuation_uncertainty: string | null;
  remaining_evidence_gaps: string[];
  thesis_invalidation_conditions: string[];
  specific_next_evidence_to_monitor: string[];
  valuation_assessment_status: string;
  valuation_suppression_reason: string | null;
}

export interface SourcedDeepValuation {
  valuation_status: string;
  suppression_reason?: string | null;
  current_price?: number | null;
  current_price_date?: string | null;
  thesis_invalidation_thresholds?: string[];
  next_quarter_evidence_milestones?: string[];
  [key: string]: any;
}

export interface SourcedCompanySnapshot {
  snapshot_metadata: {
    dataset_version?: string;
    as_of_date?: string;
    research_cutoff_date?: string;
    api_contract_version?: string;
    read_only?: boolean;
    [key: string]: any;
  };
  company_identity: SourcedCompanyIdentity;
  reported_facts: SourcedReportedFact[];
  derived_calculations: Record<string, SourcedDerivedMetric>;
  research_disposition_and_evidence: SourcedResearchDisposition;
  deep_valuation: SourcedDeepValuation;
}

interface SourcedSnapshotFile {
  metadata: Record<string, any>;
  companies: Record<string, SourcedCompanySnapshot>;
}

const typedSnapshot = snapshotData as unknown as SourcedSnapshotFile;

/**
 * Returns the company fundamentals snapshot for a symbol, or null if uncataloged.
 */
export function getSourcedFundamentals(symbol: string): SourcedCompanySnapshot | null {
  if (!symbol) return null;
  const clean = symbol.trim().toUpperCase().replace("-USD", "");
  return typedSnapshot.companies[clean] || null;
}

/**
 * Returns all cataloged company symbols with sourced fundamentals evidence.
 */
export function listSourcedCompanySymbols(): string[] {
  return Object.keys(typedSnapshot.companies).sort();
}

/**
 * Returns all snapshots in the current ledger.
 */
export function getAllSourcedFundamentals(): Record<string, SourcedCompanySnapshot> {
  return typedSnapshot.companies;
}
