/**
 * ARX Terminal Prototype Sandbox — Scenario Loader
 *
 * Loads deterministic, versioned scenario fixtures with integrity verification.
 */

export interface ScenarioFixture {
  fixture_id: string;
  version: string;
  filename: string;
  type: "NORMATIVE" | "ADVERSARIAL_TRAP";
  market_data_provenance: string;
  provenance_reference: string;
  ticker: string;
  underlying_reference?: string;
  company_name: string;
  asset_class: string;
  primary_exchange: string;
  market_data: {
    spot_price: number;
    quote_timestamp: string;
    session_current_timestamp?: string;
    is_stale: boolean;
    latency_seconds?: number;
    display_label: string;
  };
  technical_context: {
    entry_corridor?: [number, number];
    planned_entry_pivot?: number;
    breakout_pivot_trigger?: number;
    structural_invalidation_floor?: number;
    distance_to_invalidation_dollar?: number;
    distance_to_invalidation_pct?: number;
    take_profit_1?: number;
    take_profit_2?: number;
    execution_readiness?: string;
    radar_screening_badge?: string;
    setup_state?: string;
    spot_relative_to_invalidation?: number;
    pattern?: string;
    contractions?: number;
    is_actionable: boolean;
    reason_suppressed?: string;
  };
  statistical_context?: {
    engine: string;
    raw_mvar_95_pct: number;
    engine_multiplier: number;
    effective_stop_distance_pct: number;
    effective_stop_dollar: number;
    effective_stop_price: number;
    statistical_shares_per_1000_risk: number;
    statistical_capital_committed: number;
  };
  sizing_context?: {
    authorized_engine?: string;
    portfolio_equity?: number;
    risk_budget_dollar?: number;
    risk_budget_pct?: number;
    recommended_shares?: number;
    capital_committed?: number;
    portfolio_exposure_pct?: number;
    maximum_modeled_loss?: number;
  };
  fundamental_context?: {
    revenue_growth_yoy_pct: number;
    net_cash_balance_sheet: boolean;
    stock_based_compensation_pct_rev: number;
    forward_pe: number;
  };
  research_dossier?: {
    core_thesis: string;
    bearish_dissent: string;
  };
  heuristic_context?: {
    confluence_score: number;
    confluence_label: string;
    corporate_event_alert?: string;
  };
  action_contract?: {
    button_text: string;
    actual_event_action: string;
    network_requests_triggered: number;
    broker_routing_present: boolean;
    feedback_banner: string;
  };
  pipeline_health?: {
    status: string;
    failure_reason?: string;
    last_successful_batch_run?: string;
    is_safe_for_trading: boolean;
  };
  expected_evaluations: {
    authorized_decision: string;
    authorized_stop_price?: number;
    authorized_shares?: number;
    rejected_stop_price?: number;
    primary_rationale: string;
  };
  canonical_cm_codes: string[];
  supplemental_sm_codes: string[];
}

export const SCENARIO_FILE_MAP: Record<string, string> = {
  "SCN-NORM-PLSE-SWING": "scn_norm_plse_v3.json",
  "SCN-NORM-FIX-VCP": "scn_norm_fix_v3.json",
  "SCN-NORM-ALAB-MOM": "scn_norm_alab_v3.json",
  "SCN-NORM-DUOL-FUND": "scn_norm_duol_v3.json",
  "SCN-ADVR-STALE-TAPE": "scn_advr_stale_v3.json",
  "SCN-ADVR-CONTRADICT": "scn_advr_contradict_v3.json",
  "SCN-ADVR-PHANTOM": "scn_advr_phantom_v3.json",
  "SCN-ADVR-SCREEN-ONLY": "scn_advr_screen_v3.json",
  "SCN-ADVR-PIPE-FAIL": "scn_advr_pipe_v3.json",
};

export async function loadScenarioFixture(scenarioId: string): Promise<ScenarioFixture> {
  const filename = SCENARIO_FILE_MAP[scenarioId];
  if (!filename) {
    throw new Error(`[SCENARIO_LOADER] Unknown scenario ID: ${scenarioId}`);
  }

  const res = await fetch(`/fixtures/scenarios/${filename}`);
  if (!res.ok) {
    throw new Error(`[SCENARIO_LOADER] Failed to fetch fixture /fixtures/scenarios/${filename} (HTTP ${res.status})`);
  }

  const data: ScenarioFixture = await res.json();
  if (data.fixture_id !== scenarioId) {
    throw new Error(`[SCENARIO_LOADER] Fixture ID mismatch: expected ${scenarioId}, got ${data.fixture_id}`);
  }

  return data;
}
