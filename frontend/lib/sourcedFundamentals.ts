/**
 * frontend/lib/sourcedFundamentals.ts
 *
 * Client and SSG data accessor for SEC EDGAR-sourced company fundamentals evidence.
 * Reads the immutable snapshot generated from analyst_dashboard research ledger.
 * Fails closed (returns null) for uncataloged assets.
 * Strictly facts-only: Excludes DCF scenarios and investment rankings.
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
  fact_type: "PRIMARY_REPORTED" | "DERIVED_CALCULATION" | "RESEARCH_ESTIMATE" | "UNAVAILABLE" | "UNCLASSIFIED";
}

export interface SourcedDerivedMetric {
  value: number | null;
  units?: string;
  status?: string;
  label?: string;
  concept?: string | null;
  formula?: string;
  notes?: string;
  period_start?: string;
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

export interface SourcedResearchEvidenceNotes {
  evidence_sufficiency?: string | null;
  remaining_evidence_gaps?: string[];
  thesis_invalidation_conditions?: string[];
  specific_next_evidence_to_monitor?: string[];
}

export interface SourcedCompanySnapshot {
  snapshot_metadata: {
    dataset_version?: string;
    as_of_date?: string;
    research_cutoff_date?: string;
    api_contract_version?: string;
    read_only?: boolean;
    status?: string;
    is_available?: boolean;
    [key: string]: any;
  };
  company_identity: SourcedCompanyIdentity;
  reported_facts: SourcedReportedFact[];
  derived_calculations: Record<string, SourcedDerivedMetric>;
  research_evidence_notes: SourcedResearchEvidenceNotes;
}

export interface SourcedSnapshotFile {
  metadata: Record<string, any>;
  companies: Record<string, SourcedCompanySnapshot>;
}

export const ATTRIBUTABLE_SEC_IDENTITY_MAP: Record<string, string> = {
  FIX: "0001035983",
  DUOL: "0001562088",
  ULTA: "0001403568",
  MEDP: "0001668397",
  DECK: "0000910521",
  KLAC: "0000319201",
  LULU: "0001397187",
  CELH: "0001341766",
};

const typedSnapshot = snapshotData as unknown as SourcedSnapshotFile;

/**
 * Returns the company fundamentals snapshot for a symbol, or null if uncataloged or invalid.
 */
export function getSourcedFundamentals(symbol: string): SourcedCompanySnapshot | null {
  if (!symbol || typeof symbol !== "string") return null;
  const clean = symbol.trim().toUpperCase();
  const company = typedSnapshot.companies?.[clean];
  if (!company) return null;
  // Strict identity verification: company record must match requested symbol and canonical CIK
  if (company.company_identity?.symbol !== clean) return null;
  const expectedCik = ATTRIBUTABLE_SEC_IDENTITY_MAP[clean];
  if (expectedCik && company.company_identity?.cik !== expectedCik) return null;
  return company;
}

/**
 * Returns all cataloged company symbols with sourced fundamentals evidence.
 */
export function listSourcedCompanySymbols(): string[] {
  if (!typedSnapshot.companies) return [];
  return Object.keys(typedSnapshot.companies).sort();
}

/**
 * Returns all snapshots in the current ledger.
 */
export function getAllSourcedFundamentals(): Record<string, SourcedCompanySnapshot> {
  return typedSnapshot.companies || {};
}
