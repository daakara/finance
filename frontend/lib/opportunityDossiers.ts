/**
 * frontend/lib/opportunityDossiers.ts
 *
 * Client and SSG data accessor for the ARX Read-Only Opportunity Dossiers.
 * Reads the immutable, statically bundled snapshot generated from accepted ODP-05 research ledger.
 *
 * Invariants:
 * - Pure read-only bundled data path (no live API fetch or dynamic solver).
 * - Fails closed: returns null for uncataloged symbols.
 * - Retains exact research dispositions (WATCH_CONDITIONS, NO_SUPPORTED_OPPORTUNITY).
 * - Preserves fail-closed valuation suppression (IREN, CORZ) and qualification (MU).
 * - Distinguishes documented zero debt (DUOL) from unassessed/missing debt.
 */

import snapshotData from "./opportunityDossierSnapshot.json";

export interface PrimaryFilingReference {
  form: string;
  period_end: string;
  filing_date?: string;
  accession_number: string;
  url: string | null;
  source_role?: string;
}

export interface ScenarioProjection {
  scenario_key: "conservative" | "base" | "optimistic" | string;
  name: string;
  per_share_value: number;
  potential_return_pct: number;
  conditional_wording: string;
  revenue_cagr_pct?: number;
  operating_margin_pct?: number;
  wacc_pct?: number;
  terminal_growth_pct?: number;
}

export interface OpportunityDossierIdentity {
  symbol: string;
  company_name: string;
  cik: string;
  exchange: string;
  opportunity_route: string;
  sec_edgar_browse_url: string | null;
}

export interface OpportunityCentralAssessment {
  disposition: "WATCH_CONDITIONS" | "NO_SUPPORTED_OPPORTUNITY" | "RESEARCH_FURTHER" | string;
  disposition_label: string;
  presentation_summary: string;
  summary_provenance: "DERIVED_SYNTHESIS";
  strongest_supporting_evidence: string;
  strongest_opposing_evidence: string;
  main_uncertainty: string;
  price_watch_condition: string;
  confidence_assessment: string;
  financing_dilution_exposure?: string;
}

export interface OpportunityValuation {
  valuation_status: "COMPUTED_VERIFIED" | "COMPUTED_QUALIFIED" | "VALUATION_SUPPRESSED" | string;
  valuation_status_label: string;
  model_type: string | null;
  reference_price: number;
  reference_price_label: string;
  reference_date: string;
  solved_5y_cagr_pct: number | null;
  roundtrip_value: number | null;
  roundtrip_delta_cents: number | null;
  convention_summary: string;
  qualification_notice: string | null;
  suppression_reason: string | null;
  resume_conditions: string[];
  scenarios: Record<string, ScenarioProjection> | null;
}

export interface SelectedShareBasis {
  basis_type: string;
  source?: string;
  measurement_period?: string;
  value: number;
  rationale?: string;
}

export interface OpportunityFinancialProfile {
  accounting_basis: string;
  currency: string;
  revenue: number;
  revenue_display: string;
  revenue_period: string;
  operating_income: number | null;
  operating_margin_pct: number | null;
  cash_and_equivalents: number | null;
  total_cash_and_investments: number | null;
  total_financial_debt: number | null;
  debt_display: string;
  is_zero_debt: boolean;
  zero_debt_support: string | null;
  net_debt_bridge: number | null;
  capital_structure_position: string | null;
  selected_share_basis?: SelectedShareBasis;
  primary_filings: PrimaryFilingReference[];
}

export interface OpportunityMonitoring {
  observable_invalidation_trigger: string;
  next_monitoring_trigger: string;
  price_watch_condition: string;
  baseline_factual_evidence: string;
}

export interface OpportunityNavigation {
  radar_url: string;
  stock_url: string;
  radar_note: string;
}

export interface OpportunityDossier {
  identity: OpportunityDossierIdentity;
  central_assessment: OpportunityCentralAssessment;
  valuation: OpportunityValuation;
  financial_profile: OpportunityFinancialProfile;
  monitoring: OpportunityMonitoring;
  navigation: OpportunityNavigation;
}

export interface OpportunityCohortItem {
  symbol: string;
  company_name: string;
  exchange: string;
  opportunity_route: string;
  disposition: string;
  disposition_label: string;
  reference_price: number;
  valuation_status: string;
}

export interface OpportunitySnapshotMetadata {
  snapshot_identifier: string;
  source_ledger_path: string;
  source_ledger_sha256: string;
  research_candidate_commit: string;
  research_candidate_tree: string;
  canonical_ledger_blob: string;
  governing_contract: string;
  market_session_date: string;
  filing_availability_cutoff_utc: string;
  authoring_timestamp_utc: string;
  review_timestamp_utc: string | null;
  cohort_symbols: string[];
  is_read_only: boolean;
}

export interface OpportunityDossierSnapshotFile {
  schema_version: string;
  snapshot_metadata: OpportunitySnapshotMetadata;
  cohort: OpportunityCohortItem[];
  dossiers: Record<string, OpportunityDossier>;
}

export const ODP05_COHORT_SYMBOLS = ["FIX", "DUOL", "IREN", "CORZ", "MU"] as const;

export const ATTRIBUTABLE_SEC_CIK_MAP: Record<string, string> = {
  FIX: "0001035983",
  DUOL: "0001562088",
  IREN: "0001878848",
  CORZ: "0001839341",
  MU: "0000723125",
};

const typedSnapshot = snapshotData as unknown as OpportunityDossierSnapshotFile;

/**
 * Normalizes symbol to uppercase trimmed string.
 */
export function normalizeSymbol(symbol: string): string {
  if (!symbol || typeof symbol !== "string") return "";
  return symbol.trim().toUpperCase();
}

/**
 * Returns true if symbol belongs to the frozen 5-company ODP-05 research cohort.
 */
export function isCohortSymbol(symbol: string): boolean {
  const clean = normalizeSymbol(symbol);
  return (ODP05_COHORT_SYMBOLS as readonly string[]).includes(clean);
}

/**
 * Returns the opportunity dossier for a symbol, or null if outside cohort or uncataloged.
 */
export function getOpportunityDossier(symbol: string): OpportunityDossier | null {
  const clean = normalizeSymbol(symbol);
  if (!clean) return null;
  const dossier = typedSnapshot.dossiers?.[clean];
  if (!dossier) return null;

  // Identity verification: ensure internal symbol matches requested
  if (dossier.identity?.symbol !== clean) return null;

  return dossier;
}

/**
 * Returns the list of available cohort companies.
 */
export function listOpportunityCohort(): OpportunityCohortItem[] {
  return typedSnapshot.cohort || [];
}

/**
 * Returns top-level snapshot metadata.
 */
export function getOpportunitySnapshotMetadata(): OpportunitySnapshotMetadata {
  return typedSnapshot.snapshot_metadata;
}
