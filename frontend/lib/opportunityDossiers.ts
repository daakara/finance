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
import publishedWatchManifest from "./publishedWatchManifest.json";

// Explicit deployment / build-time separation
// Pilot drafts are ONLY enabled when NEXT_PUBLIC_ENABLE_RESEARCH_PILOT_DRAFTS is explicitly "true".
// In standard production builds, draftSnapshot is null and draft JSON is omitted from client bundles.
export const IS_PILOT_DRAFTS_BUILD_ENABLED =
  process.env.NEXT_PUBLIC_ENABLE_RESEARCH_PILOT_DRAFTS === "true";

let typedDraftSnapshot: OpportunityDossierSnapshotFile | null = null;
if (IS_PILOT_DRAFTS_BUILD_ENABLED) {
  try {
    typedDraftSnapshot = require("./opportunityDossierPilotDrafts.json") as OpportunityDossierSnapshotFile;
  } catch {
    typedDraftSnapshot = null;
  }
}

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
  tax_rate_pct?: number;
  reinvestment_rate_pct?: number;
  enterprise_value?: number;
  equity_value?: number;
}

export interface ReverseDcfParameters {
  target_price: number;
  solved_revenue_cagr_pct: number;
  operating_margin_pct: number;
  wacc_pct: number;
  terminal_growth_pct: number;
  tax_rate_pct: number;
  reinvestment_rate_pct: number;
  base_revenue: number;
  current_shares: number;
  net_debt_bridge: number;
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
  reverse_dcf_parameters?: ReverseDcfParameters | null;
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
  short_term_investments?: number | null;
  restricted_cash?: number | null;
  total_cash_and_investments: number | null;
  total_cash_and_restricted?: number | null;
  total_financial_debt: number | null;
  debt_display: string;
  is_zero_debt: boolean;
  zero_debt_support: string | null;
  operating_lease_liabilities_asc842?: number | null;
  lease_convention_treatment?: string | null;
  restriction_treatment?: string | null;
  digital_assets?: number | null;
  total_liquidity_with_digital_assets?: number | null;
  net_debt_bridge: number | null;
  capital_structure_position: string | null;
  bridge_note?: string | null;
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

export interface PublishedWatchManifestFile {
  schema_version: string;
  task: string;
  governing_standard: string;
  standard_doc_path?: string;
  created_at_utc: string;
  market_session_date: string;
  filing_availability_cutoff_utc: string;
  published_symbols: string[];
  published_count: number;
  baseline_watch_symbols: string[];
  excluded_historical_symbols: string[];
  excluded_draft_symbols: string[];
  accepted_additions: Record<string, OpportunityDossier>;
  additions_provenance: Record<string, any>;
}

export const ODP05_COHORT_SYMBOLS = ["FIX", "DUOL", "IREN", "CORZ", "MU"] as const;

export const PUBLISHED_WATCH_SYMBOLS = ["FIX", "DUOL", "IREN", "CORZ", "ULTA"] as const;
export const BASELINE_WATCH_SYMBOLS = ["FIX", "DUOL", "IREN", "CORZ"] as const;
export const EXCLUDED_REJECTED_SYMBOLS = ["MU", "PLTR"] as const;

export const PILOT_EXPANSION_SYMBOLS = [
  "ULTA", "MEDP", "DECK", "KLAC", "LULU",
  "EME", "VRT", "ANET", "ISRG",
  "CDNS", "PWR", "SNPS", "NOW", "ELF"
] as const;

export const TOTAL_PILOT_SYMBOLS = [
  ...BASELINE_WATCH_SYMBOLS,
  ...PILOT_EXPANSION_SYMBOLS
] as const;

export const ATTRIBUTABLE_SEC_CIK_MAP: Record<string, string> = {
  FIX: "0001035983",
  DUOL: "0001562088",
  IREN: "0001878848",
  CORZ: "0001839341",
  MU: "0000723125",
  ULTA: "0001403568",
  MEDP: "0001668397",
  DECK: "0000910521",
  KLAC: "0000319201",
  LULU: "0001397187",
  EME: "0000105634",
  VRT: "0001674101",
  ANET: "0001596532",
  PLTR: "0001321655",
  ISRG: "0001035267",
  CDNS: "0000813672",
  PWR: "0001050915",
  SNPS: "0000883241",
  NOW: "0001373715",
  ELF: "0001600033",
};

const typedSnapshot = snapshotData as unknown as OpportunityDossierSnapshotFile;
const typedPublishedManifest = publishedWatchManifest as unknown as PublishedWatchManifestFile;

/**
 * Normalizes symbol to uppercase trimmed string.
 */
export function normalizeSymbol(symbol: string): string {
  if (!symbol || typeof symbol !== "string") return "";
  return symbol.trim().toUpperCase();
}

/**
 * Returns true if symbol belongs to the published Quality Watch research collection.
 * Only qualifying WATCH assets (FIX, DUOL, IREN, CORZ, ULTA) return true.
 * Excluded rejected records (MU, PLTR) return false.
 */
export function isCohortSymbol(symbol: string): boolean {
  const clean = normalizeSymbol(symbol);
  return (PUBLISHED_WATCH_SYMBOLS as readonly string[]).includes(clean);
}

export function isPublishedWatchSymbol(symbol: string): boolean {
  return isCohortSymbol(symbol);
}

export function isExcludedRejectedSymbol(symbol: string): boolean {
  const clean = normalizeSymbol(symbol);
  return (EXCLUDED_REJECTED_SYMBOLS as readonly string[]).includes(clean);
}

/**
 * Returns true if symbol belongs to the pilot staging cohort (preview only).
 * Excludes rejected records (MU, PLTR). In production builds, falls closed to published WATCH cohort.
 */
export function isPilotDraftSymbol(symbol: string): boolean {
  const clean = normalizeSymbol(symbol);
  if (!IS_PILOT_DRAFTS_BUILD_ENABLED || !typedDraftSnapshot) {
    return isCohortSymbol(symbol);
  }
  return (TOTAL_PILOT_SYMBOLS as readonly string[]).includes(clean);
}

/**
 * Returns the opportunity dossier for a symbol from published Quality Watch collection,
 * or optionally from pilot draft staging when previewDrafts is enabled and supported by the build.
 * Fails closed (returns null) for excluded rejected symbols (MU, PLTR) and uncataloged symbols.
 */
export function getOpportunityDossier(
  symbol: string,
  options?: { previewDrafts?: boolean }
): OpportunityDossier | null {
  const clean = normalizeSymbol(symbol);
  if (!clean) return null;

  // Rejected records strictly excluded from active research
  if ((EXCLUDED_REJECTED_SYMBOLS as readonly string[]).includes(clean)) {
    return null;
  }

  // Preview branch for draft staging
  if (options?.previewDrafts && IS_PILOT_DRAFTS_BUILD_ENABLED && typedDraftSnapshot) {
    const draftDossier = typedDraftSnapshot.dossiers?.[clean];
    if (!draftDossier || draftDossier.identity?.symbol !== clean) return null;
    return draftDossier;
  }

  // 1. Published Additions from manifest (e.g., ULTA)
  const acceptedAddition = typedPublishedManifest.accepted_additions?.[clean];
  if (acceptedAddition && acceptedAddition.identity?.symbol === clean) {
    return acceptedAddition;
  }

  // 2. Baseline Published Watch assets (FIX, DUOL, IREN, CORZ)
  if ((BASELINE_WATCH_SYMBOLS as readonly string[]).includes(clean)) {
    const dossier = typedSnapshot.dossiers?.[clean];
    if (dossier && dossier.identity?.symbol === clean) {
      return dossier;
    }
  }

  return null;
}

/**
 * Returns the opportunity dossier from the draft staging file.
 */
export function getPilotDraftDossier(symbol: string): OpportunityDossier | null {
  return getOpportunityDossier(symbol, { previewDrafts: true });
}

/**
 * Returns the list of 5 published Quality Watch cohort companies.
 */
export function listOpportunityCohort(): OpportunityCohortItem[] {
  const baselineItems = (typedSnapshot.cohort || []).filter(item =>
    (BASELINE_WATCH_SYMBOLS as readonly string[]).includes(item.symbol)
  );

  const ultaDossier = typedPublishedManifest.accepted_additions?.["ULTA"];
  const ultaItem: OpportunityCohortItem = ultaDossier ? {
    symbol: "ULTA",
    company_name: ultaDossier.identity.company_name,
    exchange: ultaDossier.identity.exchange,
    opportunity_route: ultaDossier.identity.opportunity_route,
    disposition: ultaDossier.central_assessment.disposition,
    disposition_label: ultaDossier.central_assessment.disposition_label,
    reference_price: ultaDossier.valuation.reference_price,
    valuation_status: ultaDossier.valuation.valuation_status,
  } : {
    symbol: "ULTA",
    company_name: "Ulta Beauty, Inc.",
    exchange: "NASDAQ",
    opportunity_route: "Consumer Cyclical / Specialty Retail",
    disposition: "WATCH_CONDITIONS",
    disposition_label: "Watch — Conditions Required for Entry",
    reference_price: 592.94,
    valuation_status: "COMPUTED_VERIFIED",
  };

  return [...baselineItems, ultaItem];
}

/**
 * Returns the list of pilot cohort companies from draft staging when enabled (excluding rejected MU and PLTR),
 * or falls back strictly to the 5 published Quality Watch companies in production builds.
 */
export function listPilotDraftCohort(): OpportunityCohortItem[] {
  if (!IS_PILOT_DRAFTS_BUILD_ENABLED || !typedDraftSnapshot) {
    return listOpportunityCohort();
  }
  return (typedDraftSnapshot.cohort || []).filter(item =>
    !(EXCLUDED_REJECTED_SYMBOLS as readonly string[]).includes(item.symbol)
  );
}

/**
 * Returns top-level snapshot metadata.
 */
export function getOpportunitySnapshotMetadata(): OpportunitySnapshotMetadata {
  return typedSnapshot.snapshot_metadata;
}

/**
 * Returns top-level draft snapshot metadata when enabled,
 * or falls back to production snapshot metadata in production builds.
 */
export function getPilotDraftSnapshotMetadata(): OpportunitySnapshotMetadata {
  if (!IS_PILOT_DRAFTS_BUILD_ENABLED || !typedDraftSnapshot) {
    return getOpportunitySnapshotMetadata();
  }
  return typedDraftSnapshot.snapshot_metadata;
}

export function getPublishedWatchManifest(): PublishedWatchManifestFile {
  return typedPublishedManifest;
}
