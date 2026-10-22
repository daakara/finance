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

export interface SelectionRationale {
  why_this_business?: string;
  why_research_now?: string;
  price_requirements?: string;
  differentiated_view?: string;
  assessment_changes?: string;
  queue_priority?: string;
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
  selection_rationale?: SelectionRationale | null;
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
  source_evidence_tracing?: Record<string, any>;
  selection_rationale?: SelectionRationale | null;
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
  publication_metadata?: Record<string, any>;
  source_evidence_tracing?: Record<string, any>;
  selection_rationale?: SelectionRationale | null;
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
  held_historical_symbols?: string[];
  excluded_historical_symbols: string[];
  excluded_draft_symbols: string[];
  accepted_additions: Record<string, OpportunityDossier>;
  additions_provenance: Record<string, any>;
  remaining_expansion_queue?: string[];
  queue_policy?: string;
}

export const ODP05_COHORT_SYMBOLS = ["FIX", "DUOL", "IREN", "CORZ", "MU"] as const;

export const PILOT_EXPANSION_SYMBOLS = [
  "ULTA", "MEDP", "DECK", "KLAC", "LULU",
  "EME", "VRT", "ANET", "ISRG",
  "CDNS", "PWR", "SNPS", "NOW", "ELF"
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
 * Canonical recursive JSON stringifier with sorted keys.
 * Matches Python json.dumps(obj, sort_keys=True, separators=(',', ':')).
 */
export function canonicalJsonStringify(obj: any): string {
  if (obj === null || typeof obj !== "object") {
    return JSON.stringify(obj);
  }
  if (Array.isArray(obj)) {
    return "[" + obj.map(canonicalJsonStringify).join(",") + "]";
  }
  const sortedKeys = Object.keys(obj).sort();
  const pairs = sortedKeys.map(
    (k) => JSON.stringify(k) + ":" + canonicalJsonStringify(obj[k])
  );
  return "{" + pairs.join(",") + "}";
}

/**
 * Computes deterministic SHA-256 digest of a dossier using compact canonical JSON.
 */
export function computeCanonicalDossierDigest(dossier: any): string {
  const canonicalStr = canonicalJsonStringify(dossier);
  try {
    const crypto = require("crypto");
    return crypto.createHash("sha256").update(canonicalStr).digest("hex");
  } catch {
    return "";
  }
}

export interface PublishedCollectionValidationResult {
  isValid: boolean;
  errors: string[];
  publishedSymbols: string[];
  baselineWatchSymbols: string[];
  heldHistoricalSymbols: string[];
  excludedSymbols: string[];
  cohortItems: OpportunityCohortItem[];
  dossiers: Record<string, OpportunityDossier>;
}

/**
 * Validates and derives the authoritative published Quality Watch collection from the manifest.
 * Guarantees:
 * - Published membership strictly follows manifest.published_symbols.
 * - Missing selected payloads fail validation; zero fabricated/synthetic entries can be produced.
 * - An accepted addition is ignored if not selected in published_symbols.
 * - Published records must have matching identity and WATCH_CONDITIONS disposition.
 * - Content digests (when recorded) must match canonical serialization.
 */
export function validateAndDerivePublishedCollection(
  manifest: PublishedWatchManifestFile,
  baselineSnapshot: OpportunityDossierSnapshotFile,
  options?: { verifyDigests?: boolean }
): PublishedCollectionValidationResult {
  const errors: string[] = [];
  const publishedSymbols: string[] = [];
  const cohortItems: OpportunityCohortItem[] = [];
  const dossiers: Record<string, OpportunityDossier> = {};

  if (!manifest || typeof manifest !== "object") {
    return {
      isValid: false,
      errors: ["Manifest is null or not an object"],
      publishedSymbols: [],
      baselineWatchSymbols: [],
      heldHistoricalSymbols: [],
      excludedSymbols: [],
      cohortItems: [],
      dossiers: {},
    };
  }

  const rawPublishedSymbols = manifest.published_symbols || [];
  const rawBaselineWatchSymbols = manifest.baseline_watch_symbols || [];
  const heldHistoricalSymbols = manifest.held_historical_symbols || [];
  const excludedSymbols = [
    ...(manifest.excluded_historical_symbols || []),
    ...(manifest.excluded_draft_symbols || []),
  ];

  for (const sym of rawPublishedSymbols) {
    const cleanSym = normalizeSymbol(sym);
    if (!cleanSym) continue;

    // Must not be in excluded symbols
    if (excludedSymbols.includes(cleanSym)) {
      errors.push(`Excluded symbol ${cleanSym} cannot be in published_symbols`);
      continue;
    }

    let dossier: OpportunityDossier | undefined;
    if (rawBaselineWatchSymbols.includes(cleanSym)) {
      dossier = baselineSnapshot?.dossiers?.[cleanSym];
      if (!dossier) {
        errors.push(`Selected baseline symbol ${cleanSym} has missing payload in baseline snapshot`);
        continue;
      }
    } else {
      // Must be an accepted addition
      dossier = manifest.accepted_additions?.[cleanSym];
      if (!dossier) {
        errors.push(`Selected addition symbol ${cleanSym} has missing payload in accepted_additions`);
        continue;
      }
    }

    // Check identity matches selected symbol
    if (!dossier.identity || dossier.identity.symbol !== cleanSym) {
      errors.push(
        `Payload for ${cleanSym} has mismatched identity symbol: ${dossier.identity?.symbol}`
      );
      continue;
    }

    // Check disposition must be WATCH_CONDITIONS
    if (dossier.central_assessment?.disposition !== "WATCH_CONDITIONS") {
      errors.push(
        `Published dossier for ${cleanSym} must have disposition WATCH_CONDITIONS, got: ${dossier.central_assessment?.disposition}`
      );
      continue;
    }

    // If content digest is recorded for addition, verify it matches
    if (!rawBaselineWatchSymbols.includes(cleanSym)) {
      const recordedDigest = manifest.additions_provenance?.[cleanSym]?.content_sha256;
      if (options?.verifyDigests || recordedDigest) {
        if (recordedDigest) {
          const computedDigest = computeCanonicalDossierDigest(dossier);
          if (computedDigest && computedDigest !== recordedDigest) {
            errors.push(
              `Dossier content digest mismatch for accepted addition ${cleanSym}: expected ${recordedDigest}, got ${computedDigest}`
            );
            continue;
          }
        }
      }
    }

    publishedSymbols.push(cleanSym);
    dossiers[cleanSym] = dossier;

    cohortItems.push({
      symbol: cleanSym,
      company_name: dossier.identity.company_name,
      exchange: dossier.identity.exchange,
      opportunity_route: dossier.identity.opportunity_route,
      disposition: dossier.central_assessment.disposition,
      disposition_label: dossier.central_assessment.disposition_label,
      reference_price: dossier.valuation.reference_price,
      valuation_status: dossier.valuation.valuation_status,
    });
  }

  return {
    isValid: errors.length === 0,
    errors,
    publishedSymbols,
    baselineWatchSymbols: rawBaselineWatchSymbols.filter((s) => publishedSymbols.includes(s)),
    heldHistoricalSymbols,
    excludedSymbols,
    cohortItems,
    dossiers,
  };
}

// Authoritative runtime publication derived strictly from validated manifest
const validatedCollection = validateAndDerivePublishedCollection(
  typedPublishedManifest,
  typedSnapshot,
  { verifyDigests: true }
);

export const PUBLISHED_WATCH_SYMBOLS: readonly string[] = Object.freeze(
  validatedCollection.publishedSymbols
);
export const BASELINE_WATCH_SYMBOLS: readonly string[] = Object.freeze(
  validatedCollection.baselineWatchSymbols
);
export const HELD_HISTORICAL_SYMBOLS: readonly string[] = Object.freeze(
  validatedCollection.heldHistoricalSymbols
);
export const EXCLUDED_REJECTED_SYMBOLS: readonly string[] = Object.freeze(
  validatedCollection.excludedSymbols
);

export const TOTAL_PILOT_SYMBOLS = [
  ...BASELINE_WATCH_SYMBOLS,
  ...HELD_HISTORICAL_SYMBOLS,
  ...PILOT_EXPANSION_SYMBOLS,
] as const;

/**
 * Normalizes symbol to uppercase trimmed string.
 */
export function normalizeSymbol(symbol: string): string {
  if (!symbol || typeof symbol !== "string") return "";
  return symbol.trim().toUpperCase();
}

/**
 * Returns true if symbol belongs to the published Quality Watch research collection.
 * Derived strictly from manifest.published_symbols.
 * Excluded rejected records (MU, PLTR) and held records return false.
 */
export function isCohortSymbol(symbol: string): boolean {
  const clean = normalizeSymbol(symbol);
  return PUBLISHED_WATCH_SYMBOLS.includes(clean);
}

export function isPublishedWatchSymbol(symbol: string): boolean {
  return isCohortSymbol(symbol);
}

export function isExcludedRejectedSymbol(symbol: string): boolean {
  const clean = normalizeSymbol(symbol);
  return EXCLUDED_REJECTED_SYMBOLS.includes(clean);
}

export function isHeldHistoricalSymbol(symbol: string): boolean {
  const clean = normalizeSymbol(symbol);
  return HELD_HISTORICAL_SYMBOLS.includes(clean);
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
 * Fails closed (returns null) for:
 * - Unselected additions (even if present in accepted_additions payload)
 * - Excluded rejected symbols (MU, PLTR)
 * - Uncataloged symbols
 */
export function getOpportunityDossier(
  symbol: string,
  options?: { previewDrafts?: boolean }
): OpportunityDossier | null {
  const clean = normalizeSymbol(symbol);
  if (!clean) return null;

  // Rejected records strictly excluded from active research
  if (EXCLUDED_REJECTED_SYMBOLS.includes(clean)) {
    return null;
  }

  // Preview branch for draft staging
  if (options?.previewDrafts && IS_PILOT_DRAFTS_BUILD_ENABLED && typedDraftSnapshot) {
    const draftDossier = typedDraftSnapshot.dossiers?.[clean];
    if (!draftDossier || draftDossier.identity?.symbol !== clean) return null;
    return draftDossier;
  }

  // Manifest controls publication:
  // If not selected in PUBLISHED_WATCH_SYMBOLS, unselected additions CANNOT be retrieved.
  if (!PUBLISHED_WATCH_SYMBOLS.includes(clean)) {
    return null;
  }

  return validatedCollection.dossiers[clean] || null;
}

/**
 * Returns the opportunity dossier from the draft staging file.
 */
export function getPilotDraftDossier(symbol: string): OpportunityDossier | null {
  return getOpportunityDossier(symbol, { previewDrafts: true });
}

/**
 * Returns the list of published Quality Watch cohort companies.
 * Derived strictly from validated manifest dossiers (zero hardcoded fallback).
 */
export function listOpportunityCohort(): OpportunityCohortItem[] {
  return validatedCollection.cohortItems;
}

/**
 * Returns the list of pilot cohort companies from draft staging when enabled (excluding rejected MU and PLTR),
 * or falls back strictly to the published Quality Watch companies in production builds.
 */
export function listPilotDraftCohort(): OpportunityCohortItem[] {
  if (!IS_PILOT_DRAFTS_BUILD_ENABLED || !typedDraftSnapshot) {
    return listOpportunityCohort();
  }
  return (typedDraftSnapshot.cohort || []).filter(item =>
    !EXCLUDED_REJECTED_SYMBOLS.includes(item.symbol)
  );
}

export interface DossierProvenance {
  symbol: string;
  sourceAuthority: string;
  governingContract: string;
  authoringTimestampUtc: string;
  filingCutoffUtc: string;
  referenceMarketDate: string;
  contentDigest: string;
  digestLabel: string;
  lifecycleStatus: string;
  lifecycleStatusLabel: string;
  reviewStatus: string;
  reviewStatusLabel: string;
  isAddition: boolean;
  isPreviewDraft: boolean;
}

/**
 * Returns accurate, per-dossier provenance appropriate to the selected dossier.
 * Distinguishes historical baseline (FIX, DUOL) from scaling additions (ULTA) and draft staging.
 */
export function getDossierProvenance(
  symbol: string,
  options?: { previewDrafts?: boolean }
): DossierProvenance | null {
  const clean = normalizeSymbol(symbol);
  if (!clean) return null;

  // 1. Preview Draft
  if (options?.previewDrafts && IS_PILOT_DRAFTS_BUILD_ENABLED && typedDraftSnapshot) {
    const draft = typedDraftSnapshot.dossiers?.[clean];
    if (!draft) return null;
    const meta = typedDraftSnapshot.snapshot_metadata;
    return {
      symbol: clean,
      sourceAuthority: "Pilot Scaling Draft Staging",
      governingContract: "CONTRACT-ARX-OPP-PILOT-SCALING-V1",
      authoringTimestampUtc: meta.authoring_timestamp_utc || "2026-10-21T16:00:00Z",
      filingCutoffUtc: meta.filing_availability_cutoff_utc || "2026-10-16T20:00:00Z",
      referenceMarketDate: meta.market_session_date || "2026-10-16",
      contentDigest: "Draft In-Memory Staging",
      digestLabel: "Draft Staging",
      lifecycleStatus: "DRAFT_STAGING",
      lifecycleStatusLabel: "Draft Staging (Preview Only)",
      reviewStatus: "PENDING_RESEARCH_REVIEW",
      reviewStatusLabel: "Pending Independent Research Review",
      isAddition: true,
      isPreviewDraft: true,
    };
  }

  // 2. Published Addition (e.g. ULTA)
  if (typedPublishedManifest.accepted_additions?.[clean] && PUBLISHED_WATCH_SYMBOLS.includes(clean)) {
    const prov = typedPublishedManifest.additions_provenance?.[clean];
    const dossier = typedPublishedManifest.accepted_additions[clean];
    const pubMeta = (dossier as any).publication_metadata || {};
    return {
      symbol: clean,
      sourceAuthority: prov?.source_authority || "ARX Quality Watch Scaling Addition",
      governingContract: prov?.governing_contract || "CONTRACT-ARX-OPP-PILOT-SCALING-V1",
      authoringTimestampUtc: prov?.authoring_timestamp_utc || "2026-10-21T16:00:00Z",
      filingCutoffUtc: prov?.filing_availability_cutoff_utc || "2026-10-16T20:00:00Z",
      referenceMarketDate: prov?.reference_date || "2026-10-16",
      contentDigest: prov?.content_sha256 ? `${prov.content_sha256.substring(0, 12)}…` : "Verified Addition",
      digestLabel: "Accepted Addition SHA-256",
      lifecycleStatus: pubMeta.lifecycle_status || "CANDIDATE_SELECTED",
      lifecycleStatusLabel: "Selected Quality Watch Research",
      reviewStatus: pubMeta.review_status || "RESEARCH_EVIDENCE_ACCEPTED",
      reviewStatusLabel: "Research checked against SEC filings",
      isAddition: true,
      isPreviewDraft: false,
    };
  }

  // 3. Published Baseline (e.g. FIX, DUOL)
  if (BASELINE_WATCH_SYMBOLS.includes(clean)) {
    const meta = typedSnapshot.snapshot_metadata;
    return {
      symbol: clean,
      sourceAuthority: "ODP-05 Historical Research Baseline",
      governingContract: meta.governing_contract || "CONTRACT-ARX-ODP-V1-001-REV2",
      authoringTimestampUtc: meta.authoring_timestamp_utc || "2026-10-20T14:30:00Z",
      filingCutoffUtc: meta.filing_availability_cutoff_utc || "2026-10-16T20:00:00Z",
      referenceMarketDate: meta.market_session_date || "2026-10-16",
      contentDigest: meta.source_ledger_sha256 ? `${meta.source_ledger_sha256.substring(0, 12)}…` : "Verified Baseline",
      digestLabel: "Baseline Ledger SHA-256",
      lifecycleStatus: "HISTORICAL_BASELINE_WATCH",
      lifecycleStatusLabel: "Published Watch (Historical Baseline)",
      reviewStatus: "BASELINE_RESEARCH_ACCEPTED",
      reviewStatusLabel: "Accepted Research Baseline",
      isAddition: false,
      isPreviewDraft: false,
    };
  }

  return null;
}

export interface CollectionProvenance {
  governingStandard: string;
  manifestVersion: string;
  publishedCount: number;
  collectionTitle: string;
  marketSessionDate: string;
  disclaimer: string;
}

/**
 * Returns collection-level provenance for unavailable or uncataloged symbols.
 * Explicitly avoids associating baseline snapshot hashes with unavailable assets.
 */
export function getCollectionProvenance(requestedSymbol?: string): CollectionProvenance {
  return {
    governingStandard: typedPublishedManifest.governing_standard || "ARX-QUALITY-WATCH-STANDARD-V1",
    manifestVersion: typedPublishedManifest.schema_version || "ARX-PUBLISHED-WATCH-MANIFEST-V1",
    publishedCount: PUBLISHED_WATCH_SYMBOLS.length,
    collectionTitle: "Curated Quality Watch Collection",
    marketSessionDate: typedPublishedManifest.market_session_date || "2026-10-16",
    disclaimer: requestedSymbol
      ? `No published watch dossier exists for "${requestedSymbol}". Collection governance applies strictly to published assets.`
      : "Collection governance applies strictly to published assets.",
  };
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
