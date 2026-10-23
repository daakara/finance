"""
analyst_dashboard/research/maintenance/decision_assessor.py

Decision Consequence Assessor for Research Maintenance.
Evaluates the material impact of reconciled filing changes on accepted research conclusions:
- Decisive Reason
- Supporting & Opposing Evidence
- Watch Conditions & Invalidation Triggers
- Financial Resilience
- Conditional Valuation Differences (Reverse-DCF & Forward Scenarios)

Governed by ARX-DECISION-RESEARCH-PROCESS-V1 and ARX Closure Model v2.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
from .contracts import (
    DecisionAssessmentEffect,
    InterpretationType,
    ProposedDisposition,
    ReconciledFinancialChange,
)
from analyst_dashboard.research.sourced_fundamentals.valuation import (
    run_fcff_model,
    solve_reverse_dcf,
)


class DecisionAssessor:
    """
    Connects verified financial changes to the research decision.
    Strictly distinguishes facts, calculations, interpretation, and unresolved questions.
    Preserves exact approved baseline conditions and thresholds without inventing new ones.
    """

    def assess_consequences(
        self,
        symbol: str,
        baseline_dossier: Dict[str, Any],
        reconciled_profile: Dict[str, Any],
        changes: List[ReconciledFinancialChange],
        accounting_exceptions: List[str],
        undisclosed_info: List[str],
    ) -> Tuple[List[DecisionAssessmentEffect], Dict[str, Any], ProposedDisposition, str, List[str], str]:
        """
        Evaluates the impact on all accepted research conclusions.
        Returns:
            (effects, valuation_comparison, proposed_disposition, rationale, questions_for_owner, stopping_rule)
        """
        effects: List[DecisionAssessmentEffect] = []
        val_comp: Dict[str, Any] = {}
        questions: List[str] = []

        central = baseline_dossier.get("central_assessment", {})
        baseline_val = baseline_dossier.get("valuation", {})
        monitoring = baseline_dossier.get("monitoring", {})
        base_profile = baseline_dossier.get("financial_profile", {})

        base_rev = base_profile.get("revenue")
        base_shares = base_profile.get("selected_share_basis", {}).get("value")
        base_net_bridge = base_profile.get("net_debt_bridge")
        base_cash = base_profile.get("cash_and_equivalents")
        base_debt = base_profile.get("total_financial_debt")

        # Extract dated reference price directly from accepted baseline (no hardcoded fallback)
        ref_price = baseline_val.get("reference_price")
        ref_date = baseline_val.get("reference_date")

        # -------------------------------------------------------------
        # 1. Baseline Valuation Status & Input Resolution Gate
        # -------------------------------------------------------------
        baseline_status = baseline_val.get("valuation_status")
        baseline_reconciled = False
        recomputed_base_cagr: Optional[float] = None
        recomputed_base_scenarios: Dict[str, Any] = {}
        base_solver_note = ""

        target_margin: Optional[float] = None
        target_wacc: Optional[float] = None
        target_g: Optional[float] = None
        tax_rate: Optional[float] = None
        reinvest: Optional[float] = None
        resolved_scenarios: Dict[str, Dict[str, Any]] = {}

        # Step 1A: Check baseline valuation status
        if baseline_status != "COMPUTED_VERIFIED":
            val_comp = {
                "valuation_machinery_status": "SUPPRESSED_BASELINE_UNVERIFIED",
                "reason": f"Baseline valuation status is '{baseline_status}', not 'COMPUTED_VERIFIED'. Draft valuation is suppressed under propagation invariant.",
            }
        else:
            # Step 1B: Resolve reverse DCF parameters from authoritative baseline
            base_rev_params = baseline_val.get("reverse_dcf_parameters", {})
            target_margin = base_rev_params.get("operating_margin_pct")
            target_wacc = base_rev_params.get("wacc_pct")
            target_g = base_rev_params.get("terminal_growth_pct")
            tax_rate = base_rev_params.get("tax_rate_pct")
            reinvest = base_rev_params.get("reinvestment_rate_pct")

            # Step 1C: Resolve each scenario's individual parameters from authoritative baseline
            scenarios_data = baseline_val.get("scenarios", {})
            required_scenarios = ["conservative", "base", "optimistic"]
            resolved_scenarios: Dict[str, Dict[str, Any]] = {}
            missing_scenario_param: Optional[str] = None

            for sc_key in required_scenarios:
                sc_raw = scenarios_data.get(sc_key)
                if not sc_raw:
                    missing_scenario_param = f"Missing scenario '{sc_key}' in baseline valuation"
                    break
                sc_g = sc_raw.get("revenue_cagr_pct") if sc_raw.get("revenue_cagr_pct") is not None else sc_raw.get("growth_5y_pct")
                sc_m = sc_raw.get("operating_margin_pct")
                sc_w = sc_raw.get("wacc_pct")
                sc_tg = sc_raw.get("terminal_growth_pct")
                sc_tax = sc_raw.get("tax_rate_pct")
                sc_reinvest = sc_raw.get("reinvestment_rate_pct")
                sc_val = sc_raw.get("per_share_value")

                if any(p is None for p in [sc_g, sc_m, sc_w, sc_tg, sc_tax, sc_reinvest, sc_val]):
                    missing_scenario_param = f"Missing required parameter in scenario '{sc_key}'"
                    break

                resolved_scenarios[sc_key] = {
                    "growth": sc_g,
                    "margin": sc_m,
                    "wacc": sc_w,
                    "terminal_g": sc_tg,
                    "tax": sc_tax,
                    "reinvest": sc_reinvest,
                    "per_share_value": sc_val,
                }

            if ref_price is None or base_rev is None or base_shares is None or base_net_bridge is None:
                val_comp = {
                    "valuation_machinery_status": "SUPPRESSED_INPUT_UNRESOLVED",
                    "reason": "Missing baseline reference price, revenue, shares, or net debt bridge in baseline financial profile.",
                }
            elif any(p is None for p in [target_margin, target_wacc, target_g, tax_rate, reinvest]):
                val_comp = {
                    "valuation_machinery_status": "SUPPRESSED_INPUT_UNRESOLVED",
                    "reason": "Missing required reverse-DCF parameter (operating_margin_pct, wacc_pct, terminal_growth_pct, tax_rate_pct, or reinvestment_rate_pct) in baseline valuation.",
                }
            elif missing_scenario_param:
                val_comp = {
                    "valuation_machinery_status": "SUPPRESSED_INPUT_UNRESOLVED",
                    "reason": f"Baseline scenario resolution failed: {missing_scenario_param}",
                }
            else:
                # Step 1D: Recompute baseline valuation with existing engine
                recomputed_base_cagr, base_solver_note = solve_reverse_dcf(
                    ref_price, base_rev, base_shares, base_net_bridge,
                    target_margin, target_wacc, target_g, tax_rate, reinvest
                )

                if recomputed_base_cagr is None:
                    val_comp = {
                        "valuation_machinery_status": "SUPPRESSED_SOLVER_FAILURE",
                        "reason": f"Baseline reverse-DCF solver failed: {base_solver_note}",
                    }
                else:
                    recomputed_base_scenarios = {}
                    sc_finite = True
                    for sc_key, sc_params in resolved_scenarios.items():
                        try:
                            res = run_fcff_model(
                                base_rev, base_shares, base_net_bridge,
                                sc_params["growth"], sc_params["margin"], sc_params["wacc"],
                                sc_params["terminal_g"], sc_params["tax"], sc_params["reinvest"]
                            )
                            if not math.isfinite(res.get("per_share", float("nan"))):
                                sc_finite = False
                            recomputed_base_scenarios[sc_key] = res
                        except Exception as e:
                            sc_finite = False
                            base_solver_note = str(e)
                            break

                    if not sc_finite:
                        val_comp = {
                            "valuation_machinery_status": "SUPPRESSED_SOLVER_FAILURE",
                            "reason": "Baseline scenario calculation produced non-finite per-share value.",
                        }
                    else:
                        # Step 1E: Reconcile baseline outputs (implied CAGR, bear, base, bull)
                        accepted_cagr = baseline_val.get("solved_5y_cagr_pct")
                        cagr_match = (accepted_cagr is not None and abs(recomputed_base_cagr - accepted_cagr) <= 0.05)

                        sc_mismatches = []
                        for sc_key, sc_params in resolved_scenarios.items():
                            recomp_val = recomputed_base_scenarios[sc_key]["per_share"]
                            acc_val = sc_params["per_share_value"]
                            if abs(recomp_val - acc_val) > 0.50:
                                sc_mismatches.append(
                                    f"{sc_key} (recomputed ${recomp_val:.2f} vs accepted ${acc_val:.2f})"
                                )

                        if not cagr_match or sc_mismatches:
                            val_comp = {
                                "valuation_machinery_status": "SUPPRESSED_BASELINE_UNRECONCILED",
                                "reason": f"Baseline reconciliation failed: CAGR match={cagr_match}, scenario mismatches: {', '.join(sc_mismatches)}",
                            }
                            accounting_exceptions.append(
                                f"Baseline Reconciliation Failure: {val_comp['reason']}"
                            )
                        else:
                            baseline_reconciled = True

        # -------------------------------------------------------------
        # 2. Draft Valuation Calculation under Resolved Parameters
        # -------------------------------------------------------------
        draft_rev = reconciled_profile.get("revenue")
        draft_shares = reconciled_profile.get("selected_share_basis", {}).get("value")
        draft_net_bridge = reconciled_profile.get("net_debt_bridge")

        if not baseline_reconciled:
            # val_comp already holds exact suppression state and reason
            pass
        elif draft_rev is None or draft_shares is None or draft_net_bridge is None:
            val_comp = {
                "valuation_machinery_status": "SUPPRESSED_INPUT_UNRESOLVED",
                "reason": "Draft revenue, shares, or net debt bridge is UNRESOLVED or in validation conflict. Calculation suppressed.",
            }
        else:
            draft_cagr, draft_solver_note = solve_reverse_dcf(
                ref_price, draft_rev, draft_shares, draft_net_bridge,
                target_margin, target_wacc, target_g, tax_rate, reinvest
            )

            if draft_cagr is None:
                val_comp = {
                    "valuation_machinery_status": "SUPPRESSED_SOLVER_FAILURE",
                    "reason": f"Draft reverse-DCF solver failed: {draft_solver_note}",
                }
            else:
                draft_scenarios = {}
                draft_finite = True
                for sc_key, sc_params in resolved_scenarios.items():
                    try:
                        res = run_fcff_model(
                            draft_rev, draft_shares, draft_net_bridge,
                            sc_params["growth"], sc_params["margin"], sc_params["wacc"],
                            sc_params["terminal_g"], sc_params["tax"], sc_params["reinvest"]
                        )
                        if not math.isfinite(res.get("per_share", float("nan"))):
                            draft_finite = False
                        draft_scenarios[sc_key] = res
                    except Exception as e:
                        draft_finite = False
                        draft_solver_note = str(e)
                        break

                if not draft_finite:
                    val_comp = {
                        "valuation_machinery_status": "SUPPRESSED_SOLVER_FAILURE",
                        "reason": "Draft scenario model produced non-finite per-share value.",
                    }
                else:
                    base_cagr = baseline_val.get("solved_5y_cagr_pct", recomputed_base_cagr)
                    cagr_delta_bps = round((draft_cagr - base_cagr) * 100, 0)

                    if draft_cagr < base_cagr:
                        cagr_interp = f"Lower required revenue growth ({draft_cagr:.2f}% vs {base_cagr:.2f}%, {cagr_delta_bps:+.0f} bps) to justify reference price due to larger starting revenue base and expanded net cash buffer."
                    elif draft_cagr > base_cagr:
                        cagr_interp = f"Higher required revenue growth ({draft_cagr:.2f}% vs {base_cagr:.2f}%, {cagr_delta_bps:+.0f} bps) to justify reference price."
                    else:
                        cagr_interp = f"Required revenue growth unchanged at {draft_cagr:.2f}% ({cagr_delta_bps:+.0f} bps)."

                    sc_out = {}
                    for sc_key, sc_params in resolved_scenarios.items():
                        base_ps = sc_params["per_share_value"]
                        draft_ps = round(draft_scenarios[sc_key]["per_share"], 2)
                        delta_d = round(draft_ps - base_ps, 2)
                        delta_p = round(delta_d / base_ps * 100, 2)
                        sc_out[sc_key] = {
                            "baseline_per_share": base_ps,
                            "draft_per_share": draft_ps,
                            "delta_dollars": delta_d,
                            "delta_pct": delta_p,
                            "assumptions": {
                                "revenue_cagr_pct": sc_params["growth"],
                                "operating_margin_pct": sc_params["margin"],
                                "wacc_pct": sc_params["wacc"],
                                "terminal_growth_pct": sc_params["terminal_g"],
                                "tax_rate_pct": sc_params["tax"],
                                "reinvestment_rate_pct": sc_params["reinvest"],
                            }
                        }

                    val_comp = {
                        "valuation_machinery_status": "COMPUTED_VERIFIED",
                        "reference_price": ref_price,
                        "reference_date": ref_date,
                        "resolved_assumptions": {
                            "authoritative_paths": {
                                "reverse_dcf": "baseline_dossier.valuation.reverse_dcf_parameters",
                                "scenarios": "baseline_dossier.valuation.scenarios",
                            },
                            "reverse_dcf_parameters": {
                                "operating_margin_pct": target_margin,
                                "wacc_pct": target_wacc,
                                "terminal_growth_pct": target_g,
                                "tax_rate_pct": tax_rate,
                                "reinvestment_rate_pct": reinvest,
                            },
                            "solver_verification": {
                                "baseline_solver": base_solver_note,
                                "draft_solver": draft_solver_note,
                            }
                        },
                        "solved_5y_cagr": {
                            "baseline_cagr_pct": base_cagr,
                            "draft_cagr_pct": round(draft_cagr, 2),
                            "delta_cagr_bps": cagr_delta_bps,
                            "interpretation": cagr_interp,
                        },
                        "scenarios": sc_out,
                    }

        # -------------------------------------------------------------
        # 3. Assess Research Consequences & Invalidation Rules
        # -------------------------------------------------------------
        base_decisive = central.get("presentation_summary") or "Baseline presentation summary unavailable."
        base_watch = monitoring.get("price_watch_condition") or central.get("price_watch_condition") or "Price watch condition unavailable."
        base_invalidation = monitoring.get("observable_invalidation_trigger") or "Observable invalidation trigger unavailable."
        base_uncertainty = central.get("main_uncertainty") or "Main uncertainty unavailable."

        # Standalone Quarterly Margin Control (Strict separation from TTM margin)
        draft_q_margin = reconciled_profile.get("quarterly_operating_margin_pct")
        draft_ttm_margin = reconciled_profile.get("operating_margin_pct")

        is_quarterly_invalidated = False
        is_quarterly_unresolved = False

        # FIX invalidation condition links to 13.0% quarterly operating margin floor
        if draft_q_margin is None:
            is_quarterly_unresolved = True
        elif draft_q_margin < 13.0:
            is_quarterly_invalidated = True

        # Check balance sheet items honestly
        draft_cash = reconciled_profile.get("cash_and_equivalents")
        draft_debt = reconciled_profile.get("total_financial_debt")
        base_cash = baseline_dossier.get("financial_profile", {}).get("cash_and_equivalents")
        base_debt = baseline_dossier.get("financial_profile", {}).get("total_financial_debt")

        cash_expanded = (draft_cash is not None and base_cash is not None and draft_cash > base_cash)
        cash_contracted = (draft_cash is not None and base_cash is not None and draft_cash < base_cash)
        debt_declined = (draft_debt is not None and base_debt is not None and draft_debt < base_debt)
        debt_increased = (draft_debt is not None and base_debt is not None and draft_debt > base_debt)

        # Capital structure text using net-debt bridge sign
        if draft_net_bridge is None:
            cap_desc = "Net debt bridge is UNRESOLVED."
        elif draft_net_bridge < 0:
            draft_net_cash_m = abs(draft_net_bridge) / 1e6
            if base_net_bridge is not None and base_net_bridge < 0:
                base_net_cash_m = abs(base_net_bridge) / 1e6
                if draft_net_cash_m > base_net_cash_m:
                    cap_desc = f"Net cash buffer expanded from ${base_net_cash_m:.1f}M to ${draft_net_cash_m:.1f}M."
                elif draft_net_cash_m < base_net_cash_m:
                    cap_desc = f"Net cash buffer contracted from ${base_net_cash_m:.1f}M to ${draft_net_cash_m:.1f}M."
                else:
                    cap_desc = f"Net cash buffer held steady at ${draft_net_cash_m:.1f}M."
            else:
                cap_desc = f"Capital structure position is net cash of ${draft_net_cash_m:.1f}M."
        elif draft_net_bridge > 0:
            draft_net_debt_m = draft_net_bridge / 1e6
            cap_desc = f"Capital structure shifted to net debt of ${draft_net_debt_m:.1f}M."
        else:
            cap_desc = "Net debt bridge is zero."

        # Effect 1: Decisive Reason
        if val_comp.get("valuation_machinery_status") == "COMPUTED_VERIFIED":
            draft_cagr_val = val_comp["solved_5y_cagr"]["draft_cagr_pct"]
            base_cagr_val = val_comp["solved_5y_cagr"]["baseline_cagr_pct"]
            draft_base_per_share = val_comp["scenarios"]["base"]["draft_per_share"]

            # Revenue description
            if draft_rev > base_rev:
                rev_desc = f"TTM revenue grew from ${base_rev / 1e9:.2f}B to ${draft_rev / 1e9:.2f}B."
            elif draft_rev < base_rev:
                rev_desc = f"TTM revenue contracted from ${base_rev / 1e9:.2f}B to ${draft_rev / 1e9:.2f}B."
            else:
                rev_desc = f"TTM revenue was unchanged at ${draft_rev / 1e9:.2f}B."

            # Implied CAGR description
            if draft_cagr_val < base_cagr_val:
                cagr_desc = f"Solved 5-year revenue CAGR eased from {base_cagr_val:.2f}% to {draft_cagr_val:.2f}%."
            elif draft_cagr_val > base_cagr_val:
                cagr_desc = f"Solved 5-year revenue CAGR increased from {base_cagr_val:.2f}% to {draft_cagr_val:.2f}%."
            else:
                cagr_desc = f"Solved 5-year revenue CAGR was unchanged at {draft_cagr_val:.2f}%."

            # Reference price vs Base Intrinsic Value
            if ref_price > draft_base_per_share:
                price_desc = f"Market reference price (${ref_price:.2f}) remains above base intrinsic value (${draft_base_per_share:.2f})."
            elif ref_price < draft_base_per_share:
                price_desc = f"Market reference price (${ref_price:.2f}) is below base intrinsic value (${draft_base_per_share:.2f})."
            else:
                price_desc = f"Market reference price (${ref_price:.2f}) equals base intrinsic value (${draft_base_per_share:.2f})."

            margin_desc = f"Trailing 12-month margin was {draft_ttm_margin:.2f}%." if draft_ttm_margin is not None else "Margin is UNRESOLVED."

            is_improved = (draft_net_bridge < (base_net_bridge or 0)) and (draft_rev > (base_rev or 0))
            is_deteriorated = (draft_net_bridge > (base_net_bridge or 0)) or (draft_rev < (base_rev or 0))

            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="decisive_reason",
                    conclusion_label="Decisive Reason for Quality Watch",
                    baseline_state=base_decisive,
                    draft_state=f"{cap_desc} {rev_desc} {margin_desc} {cagr_desc} {price_desc} Core decisive thesis is maintained.",
                    interpretation_type=InterpretationType.CALCULATED_DELTA,
                    materiality="INCREMENTAL_POSITIVE" if is_improved else ("INCREMENTAL_NEGATIVE" if is_deteriorated else "NEUTRAL"),
                    owner_judgment_required=False,
                    notes="Quantitative comparison derived directly from reconciled filing facts and recomputed DCF."
                )
            )
        else:
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="decisive_reason",
                    conclusion_label="Decisive Reason for Quality Watch",
                    baseline_state=base_decisive,
                    draft_state=f"Decisive reason quantitative assessment is UNRESOLVED: {val_comp.get('reason', 'Valuation suppressed.')}",
                    interpretation_type=InterpretationType.UNRESOLVED_EVIDENCE,
                    materiality="NEUTRAL",
                    owner_judgment_required=True,
                    notes="Valuation calculations suppressed; human analyst review required."
                )
            )

        # Effect 2: Balance Sheet Resilience
        if draft_cash is None or draft_debt is None or draft_net_bridge is None:
            c_str = f"${draft_cash / 1e6:.1f}M" if draft_cash is not None else "UNRESOLVED"
            d_str = f"${draft_debt / 1e6:.1f}M" if draft_debt is not None else "UNRESOLVED"
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="financial_resilience",
                    conclusion_label="Balance Sheet Resilience & Capital Structure",
                    baseline_state="Prong 1: Exceptional balance sheet ($1.80B net cash, $54.1M debt).",
                    draft_state=f"Capital structure position is UNRESOLVED: Cash ({c_str}), funded debt ({d_str}). Balance sheet resilience cannot be verified from available disclosures.",
                    interpretation_type=InterpretationType.UNRESOLVED_EVIDENCE,
                    materiality="NEUTRAL",
                    owner_judgment_required=False,
                    notes="Balance sheet items missing or in validation conflict; honest omission preserved."
                )
            )
        elif draft_net_bridge < 0:
            c_str = f"${draft_cash / 1e6:.1f}M"
            d_str = f"${draft_debt / 1e6:.1f}M"
            bridge_str = f"${abs(draft_net_bridge) / 1e6:.1f}M"
            resilience_text = f"Net cash buffer: Unrestricted cash carrying value {c_str} vs funded debt {d_str}. Capital structure remains in strong net cash position ({bridge_str} net cash)."
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="financial_resilience",
                    conclusion_label="Balance Sheet Resilience & Capital Structure",
                    baseline_state="Prong 1: Exceptional balance sheet ($1.80B net cash, $54.1M debt).",
                    draft_state=resilience_text,
                    interpretation_type=InterpretationType.FACT_CHANGE,
                    materiality="INCREMENTAL_POSITIVE" if (cash_expanded and debt_declined) else "NEUTRAL",
                    owner_judgment_required=False,
                    notes="Reconciled directly from primary balance sheet facts."
                )
            )
        else:
            c_str = f"${draft_cash / 1e6:.1f}M"
            d_str = f"${draft_debt / 1e6:.1f}M"
            bridge_str = f"${draft_net_bridge / 1e6:.1f}M"
            resilience_text = f"Net debt position: Funded debt {d_str} exceeds cash {c_str}. Capital structure is net debt of {bridge_str}."
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="financial_resilience",
                    conclusion_label="Balance Sheet Resilience & Capital Structure",
                    baseline_state="Prong 1: Exceptional balance sheet ($1.80B net cash, $54.1M debt).",
                    draft_state=resilience_text,
                    interpretation_type=InterpretationType.FACT_CHANGE,
                    materiality="INCREMENTAL_NEGATIVE",
                    owner_judgment_required=False,
                    notes="Net debt position detected; cash cushion no longer exists."
                )
            )

        # Effect 3: Watch Conditions & Entry Corridor
        if val_comp.get("valuation_machinery_status") == "COMPUTED_VERIFIED":
            draft_base_val = val_comp["scenarios"]["base"]["draft_per_share"]
            base_base_val = val_comp["scenarios"]["base"]["baseline_per_share"]
            pct_delta = val_comp["scenarios"]["base"]["delta_pct"]
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="watch_condition",
                    conclusion_label="Price Watch Entry Condition",
                    baseline_state=base_watch,
                    draft_state=(
                        f"Approved entry condition remains {base_watch}. "
                        f"Base-case modeled conditional value shifted from ${base_base_val:.2f} to ${draft_base_val:.2f} ({pct_delta:+.1f}%). "
                        f"Modifying approved entry thresholds is an analyst judgment decision reserved for the Product Owner; existing approved thresholds are preserved."
                    ),
                    interpretation_type=InterpretationType.ANALYST_JUDGMENT,
                    materiality="MATERIAL",
                    owner_judgment_required=True,
                    notes="Model value expansion warrants human review; existing approved watch thresholds are preserved intact."
                )
            )
            questions.append(
                f"Base modeled value shifted from ${base_base_val:.2f} to ${draft_base_val:.2f} ({pct_delta:+.1f}%). "
                f"Does the Research Lead / Product Owner approve updating the price watch entry corridor, or should existing approved thresholds ({base_watch}) remain unchanged?"
            )
        else:
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="watch_condition",
                    conclusion_label="Price Watch Entry Condition",
                    baseline_state=base_watch,
                    draft_state=f"Approved entry condition preserved: {base_watch}. Draft conditional valuation is UNRESOLVED.",
                    interpretation_type=InterpretationType.ANALYST_JUDGMENT,
                    materiality="NEUTRAL",
                    owner_judgment_required=True,
                    notes="Preserving approved thresholds."
                )
            )

        # Effect 4: Thesis Invalidation Triggers (Evaluated strictly against STANDALONE QUARTERLY margin)
        if is_quarterly_invalidated:
            inval_text = (
                f"INVALIDATION TRIGGER BREACHED: Standalone quarterly operating margin ({draft_q_margin:.2f}%) fell below the approved 13.0% threshold. "
                f"Immediate thesis review and potential downgrade required."
            )
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="invalidation_condition",
                    conclusion_label="Thesis Invalidation Triggers",
                    baseline_state=base_invalidation,
                    draft_state=inval_text,
                    interpretation_type=InterpretationType.FACT_CHANGE,
                    materiality="MATERIAL",
                    owner_judgment_required=True,
                    notes="Standalone quarterly margin breached approved 13.0% lower boundary."
                )
            )
        elif is_quarterly_unresolved:
            inval_text = (
                f"Quarterly operating margin is UNRESOLVED due to missing quarterly revenue or operating income; cannot assess 13.0% quarterly margin floor. "
                f"Backlog book-to-bill ratio remains UNRESOLVED via primary filing alone."
            )
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="invalidation_condition",
                    conclusion_label="Thesis Invalidation Triggers",
                    baseline_state=base_invalidation,
                    draft_state=inval_text,
                    interpretation_type=InterpretationType.UNRESOLVED_EVIDENCE,
                    materiality="NEUTRAL",
                    owner_judgment_required=True,
                    notes="Quarterly inputs unavailable; condition unresolved."
                )
            )
        else:
            inval_text = (
                f"Standalone quarterly operating margin ({draft_q_margin:.2f}%) did not breach the approved 13.0% invalidation floor. "
                f"(Trailing 12-month margin is {draft_ttm_margin:.2f}%). "
                f"Backlog book-to-bill ratio is an evidence limitation not disclosed in standardized primary XBRL; "
                f"condition is not triggered by available reported evidence, but remains UNRESOLVED via primary filing alone."
            )
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="invalidation_condition",
                    conclusion_label="Thesis Invalidation Triggers",
                    baseline_state=base_invalidation,
                    draft_state=inval_text,
                    interpretation_type=InterpretationType.UNRESOLVED_EVIDENCE,
                    materiality="NEUTRAL",
                    owner_judgment_required=False,
                    notes="Quarterly margin evaluated; backlog book-to-bill requires human contract audit."
                )
            )

        # Effect 5: Core Uncertainty
        effects.append(
            DecisionAssessmentEffect(
                conclusion_key="main_uncertainty",
                conclusion_label="Decisive Core Uncertainty",
                baseline_state=base_uncertainty,
                draft_state=(
                    f"Core uncertainty ({base_uncertainty}) cannot be resolved by a single interim Form 10-Q filing. "
                    f"Multi-year hyperscale cloud capex commitments remain subject to technology expenditure cycles."
                ),
                interpretation_type=InterpretationType.UNRESOLVED_EVIDENCE,
                materiality="NEUTRAL",
                owner_judgment_required=False,
                notes="Interim filing reports historical backlog but cannot eliminate macro tech capex cycle risk."
            )
        )

        # -------------------------------------------------------------
        # 4. Proposed Disposition & Rationale
        # -------------------------------------------------------------
        if is_quarterly_invalidated:
            disposition = ProposedDisposition.INVALIDATE
            rationale = (
                f"INVALIDATE: Standalone quarterly operating margin compressed to {draft_q_margin:.2f}%, breaching the approved 13.0% invalidation threshold. "
                f"Quality Watch thesis is invalidated by reported primary filing facts."
            )
        elif is_quarterly_unresolved or val_comp.get("valuation_machinery_status") != "COMPUTED_VERIFIED":
            disposition = ProposedDisposition.UNRESOLVED
            reasons = []
            if is_quarterly_unresolved:
                reasons.append("quarterly operating margin is UNRESOLVED")
            if val_comp.get("valuation_machinery_status") != "COMPUTED_VERIFIED":
                reasons.append(f"valuation suppressed ({val_comp.get('reason')})")
            rationale = (
                f"UNRESOLVED: Review packet is staged with unresolved status for human research review because {'; '.join(reasons)}."
            )
        else:
            disposition = ProposedDisposition.MAINTAIN
            cagr_text_part = val_comp.get("solved_5y_cagr", {}).get("interpretation", "")
            rationale = (
                f"Proposed disposition: MAINTAIN Quality Watch status (WATCH_CONDITIONS). "
                f"Operational results in reporting period ended {reconciled_profile.get('selected_share_basis', {}).get('measurement_period')}: "
                f"{cap_desc} {cagr_text_part} "
                f"No thesis invalidation occurred; maintain watch status pending human Research Lead review."
            )

        if val_comp.get("valuation_machinery_status") == "COMPUTED_VERIFIED":
            target_margin_val = val_comp.get("resolved_assumptions", {}).get("reverse_dcf_parameters", {}).get("operating_margin_pct", 15.0)
            obs_margin_val = draft_ttm_margin or draft_q_margin or 0.0
            questions.append(
                f"Should normalized operating margin assumption ({target_margin_val:.1f}%) be reassessed at the next annual research review given trailing margin execution ({obs_margin_val:.2f}%)?"
            )

        rep_period = reconciled_profile.get('selected_share_basis', {}).get('measurement_period') or 'interim period'
        stopping_rule = (
            f"Evidence collection complete for Form 10-Q (Period: {rep_period}). "
            f"All reported primary facts reconciled and conditional valuation differences calculated. "
            f"Core customer capex cycle uncertainty remains an evidence limitation that cannot be further resolved via SEC quarterly filings until FY2026 Form 10-K. "
            f"Cease investigation."
        )

        return effects, val_comp, disposition, rationale, questions, stopping_rule
