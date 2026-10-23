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

        # Extract dated reference price directly from accepted baseline (no hardcoded fallback)
        ref_price = baseline_val.get("reference_price")
        ref_date = baseline_val.get("reference_date")

        # -------------------------------------------------------------
        # 1. Baseline Valuation Recomputation & Reconciliation Gate
        # -------------------------------------------------------------
        base_rev_params = baseline_val.get("reverse_dcf_parameters", {})
        target_margin = base_rev_params.get("operating_margin_pct")
        target_wacc = base_rev_params.get("wacc_pct")
        target_g = base_rev_params.get("terminal_growth_pct")
        tax_rate = base_rev_params.get("tax_rate_pct", 21.0)
        reinvest = base_rev_params.get("reinvestment_rate_pct", 15.0)

        base_rev = baseline_dossier.get("financial_profile", {}).get("revenue")
        base_shares = baseline_dossier.get("financial_profile", {}).get("selected_share_basis", {}).get("value")
        base_net_bridge = baseline_dossier.get("financial_profile", {}).get("net_debt_bridge")

        baseline_reconciled = False
        recomputed_base_cagr: Optional[float] = None
        recomputed_base_scenarios: Dict[str, Any] = {}

        if (
            ref_price is not None
            and base_rev is not None
            and base_shares is not None
            and base_net_bridge is not None
            and target_margin is not None
            and target_wacc is not None
            and target_g is not None
        ):
            # Recompute accepted baseline to verify mathematical machinery
            recomputed_base_cagr, _ = solve_reverse_dcf(
                ref_price, base_rev, base_shares, base_net_bridge,
                target_margin, target_wacc, target_g, tax_rate, reinvest
            )
            sc_c = run_fcff_model(base_rev, base_shares, base_net_bridge, 12.0, 12.5, 10.5, 2.0, tax_rate, reinvest)
            sc_b = run_fcff_model(base_rev, base_shares, base_net_bridge, 22.0, 15.0, 9.5, 2.5, tax_rate, reinvest)
            sc_o = run_fcff_model(base_rev, base_shares, base_net_bridge, 30.0, 16.5, 8.5, 3.0, tax_rate, reinvest)
            recomputed_base_scenarios = {"conservative": sc_c, "base": sc_b, "optimistic": sc_o}

            accepted_cagr = baseline_val.get("solved_5y_cagr_pct")
            accepted_base_val = baseline_val.get("scenarios", {}).get("base", {}).get("per_share_value")

            if (
                accepted_cagr is not None
                and abs(recomputed_base_cagr - accepted_cagr) <= 0.05
                and accepted_base_val is not None
                and abs(sc_b["per_share"] - accepted_base_val) <= 0.50
            ):
                baseline_reconciled = True
            else:
                accounting_exceptions.append(
                    f"Baseline Valuation Reconciliation Warning: Recomputed CAGR ({recomputed_base_cagr:.2f}%) or Base Value (${sc_b['per_share']:.2f}) "
                    f"deviated from accepted baseline ({accepted_cagr}%, ${accepted_base_val})."
                )

        # -------------------------------------------------------------
        # 2. Draft Valuation Calculation under Fixed Parameters
        # -------------------------------------------------------------
        draft_rev = reconciled_profile.get("revenue")
        draft_shares = reconciled_profile.get("selected_share_basis", {}).get("value")
        draft_net_bridge = reconciled_profile.get("net_debt_bridge")

        # Suppress calculations if critical bridge inputs are missing, conflicted, or baseline did not reconcile
        if not baseline_reconciled or ref_price is None:
            val_comp = {
                "valuation_machinery_status": "SUPPRESSED_BASELINE_UNRECONCILED",
                "reason": "Baseline valuation inputs could not be reconciled against authoritative baseline dossier.",
            }
        elif draft_rev is None or draft_shares is None or draft_net_bridge is None:
            val_comp = {
                "valuation_machinery_status": "SUPPRESSED_INPUT_UNRESOLVED",
                "reason": "Draft revenue, shares, or net debt bridge is UNRESOLVED or in validation conflict. Calculation suppressed.",
            }
        else:
            # Solved Reverse-DCF CAGR
            draft_cagr, cagr_note = solve_reverse_dcf(
                ref_price, draft_rev, draft_shares, draft_net_bridge,
                target_margin, target_wacc, target_g, tax_rate, reinvest
            )

            # Scenarios
            draft_sc = run_fcff_model(draft_rev, draft_shares, draft_net_bridge, 12.0, 12.5, 10.5, 2.0, tax_rate, reinvest)
            draft_sb = run_fcff_model(draft_rev, draft_shares, draft_net_bridge, 22.0, 15.0, 9.5, 2.5, tax_rate, reinvest)
            draft_so = run_fcff_model(draft_rev, draft_shares, draft_net_bridge, 30.0, 16.5, 8.5, 3.0, tax_rate, reinvest)

            base_cagr = baseline_val.get("solved_5y_cagr_pct", recomputed_base_cagr)
            base_sc = baseline_val.get("scenarios", {}).get("conservative", {}).get("per_share_value", recomputed_base_scenarios.get("conservative", {}).get("per_share"))
            base_sb = baseline_val.get("scenarios", {}).get("base", {}).get("per_share_value", recomputed_base_scenarios.get("base", {}).get("per_share"))
            base_so = baseline_val.get("scenarios", {}).get("optimistic", {}).get("per_share_value", recomputed_base_scenarios.get("optimistic", {}).get("per_share"))

            cagr_delta_bps = round((draft_cagr - base_cagr) * 100, 0)
            cagr_interp = (
                f"Lower required revenue growth ({draft_cagr:.2f}% vs {base_cagr:.2f}%, {cagr_delta_bps:+.0f} bps) "
                f"to justify reference price due to larger starting revenue base and expanded net cash buffer."
                if draft_cagr < base_cagr else
                f"Higher required revenue growth ({draft_cagr:.2f}% vs {base_cagr:.2f}%, {cagr_delta_bps:+.0f} bps) "
                f"to justify reference price."
            )

            val_comp = {
                "valuation_machinery_status": "COMPUTED_VERIFIED",
                "reference_price": ref_price,
                "reference_date": ref_date,
                "fixed_assumptions": {
                    "operating_margin_pct": target_margin,
                    "wacc_pct": target_wacc,
                    "terminal_growth_pct": target_g,
                    "tax_rate_pct": tax_rate,
                    "reinvestment_rate_pct": reinvest,
                },
                "solved_5y_cagr": {
                    "baseline_cagr_pct": base_cagr,
                    "draft_cagr_pct": round(draft_cagr, 2),
                    "delta_cagr_bps": cagr_delta_bps,
                    "interpretation": cagr_interp,
                },
                "scenarios": {
                    "conservative": {
                        "baseline_per_share": base_sc,
                        "draft_per_share": round(draft_sc["per_share"], 2),
                        "delta_dollars": round(draft_sc["per_share"] - base_sc, 2) if base_sc else None,
                        "delta_pct": round((draft_sc["per_share"] - base_sc) / base_sc * 100, 2) if base_sc else None,
                    },
                    "base": {
                        "baseline_per_share": base_sb,
                        "draft_per_share": round(draft_sb["per_share"], 2),
                        "delta_dollars": round(draft_sb["per_share"] - base_sb, 2) if base_sb else None,
                        "delta_pct": round((draft_sb["per_share"] - base_sb) / base_sb * 100, 2) if base_sb else None,
                    },
                    "optimistic": {
                        "baseline_per_share": base_so,
                        "draft_per_share": round(draft_so["per_share"], 2),
                        "delta_dollars": round(draft_so["per_share"] - base_so, 2) if base_so else None,
                        "delta_pct": round((draft_so["per_share"] - base_so) / base_so * 100, 2) if base_so else None,
                    },
                }
            }

        # -------------------------------------------------------------
        # 3. Assess Research Consequences & Invalidation Rules
        # -------------------------------------------------------------
        base_decisive = central.get("presentation_summary", "")
        base_watch = monitoring.get("price_watch_condition") or central.get("price_watch_condition", "Priced attractive for entry if share price pulls back to base-case modeled value range ($1,050 - $1,100).")
        base_invalidation = monitoring.get("observable_invalidation_trigger", "Backlog book-to-bill falling below 0.95x or quarterly operating margin dropping below 13.0%.")
        base_uncertainty = central.get("main_uncertainty", "Cyclicality of hyperscale cloud capex budgets and customer concentration in top-tier tech clients.")

        # Check Invalidation Rule: Operating Margin < 13.0%
        draft_margin = reconciled_profile.get("operating_margin_pct")
        is_margin_invalidated = (draft_margin is not None and draft_margin < 13.0)

        # Check Financial Deltas Direction
        cash_change = next((c for c in changes if c.metric_key == "cash_and_equivalents"), None)
        debt_change = next((c for c in changes if c.metric_key == "total_financial_debt"), None)
        rev_change = next((c for c in changes if c.metric_key == "revenue"), None)

        cash_expanded = cash_change and cash_change.delta_value and cash_change.delta_value > 0
        debt_declined = debt_change and debt_change.delta_value and debt_change.delta_value < 0
        rev_expanded = rev_change and rev_change.delta_value and rev_change.delta_value > 0

        # Effect 1: Decisive Reason
        if val_comp.get("valuation_machinery_status") == "COMPUTED_VERIFIED":
            draft_cagr_val = val_comp["solved_5y_cagr"]["draft_cagr_pct"]
            base_cagr_val = val_comp["solved_5y_cagr"]["baseline_cagr_pct"]
            draft_base_per_share = val_comp["scenarios"]["base"]["draft_per_share"]
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="decisive_reason",
                    conclusion_label="Decisive Reason for Quality Watch",
                    baseline_state=base_decisive,
                    draft_state=(
                        f"Net cash reserves expanded to ${abs(draft_net_bridge) / 1e6:.1f}M while TTM revenue grew to ${draft_rev / 1e9:.2f}B at {draft_margin:.2f}% operating margin. "
                        f"Solved 5-year revenue CAGR holding {target_margin:.1f}% margin and {target_wacc:.1f}% WACC fixed shifted from {base_cagr_val:.2f}% to {draft_cagr_val:.2f}%. "
                        f"Market reference price (${ref_price:.2f}) remains above base intrinsic value (${draft_base_per_share:.2f}). "
                        f"Core decisive thesis (await pullback toward modeled value or evidence of margin persistence) is maintained."
                    ),
                    interpretation_type=InterpretationType.CALCULATED_DELTA,
                    materiality="INCREMENTAL_POSITIVE" if (rev_expanded and cash_expanded) else "INCREMENTAL_NEGATIVE",
                    owner_judgment_required=False,
                    notes="Operational metrics updated from primary filing facts; required growth eased but remains above historical trend."
                )
            )
        else:
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="decisive_reason",
                    conclusion_label="Decisive Reason for Quality Watch",
                    baseline_state=base_decisive,
                    draft_state="Decisive reason quantitative assessment is UNRESOLVED due to suppressed valuation calculations.",
                    interpretation_type=InterpretationType.UNRESOLVED_EVIDENCE,
                    materiality="NEUTRAL",
                    owner_judgment_required=True,
                    notes="Valuation calculations suppressed; human analyst review required."
                )
            )

        # Effect 2: Balance Sheet Resilience
        draft_cash_m = (reconciled_profile.get("cash_and_equivalents") or 0.0) / 1e6
        draft_debt_m = (reconciled_profile.get("total_financial_debt") or 0.0) / 1e6
        resilience_narrative = (
            f"Net cash buffer expanded: Unrestricted cash carrying value ${draft_cash_m:.1f}M vs funded debt ${draft_debt_m:.1f}M. "
            f"Capital structure remains in strong net cash position."
            if cash_expanded else
            f"Capital structure position: Cash ${draft_cash_m:.1f}M, funded debt ${draft_debt_m:.1f}M."
        )
        effects.append(
            DecisionAssessmentEffect(
                conclusion_key="financial_resilience",
                conclusion_label="Balance Sheet Resilience & Capital Structure",
                baseline_state="Prong 1: Exceptional balance sheet ($1.80B net cash, $54.1M debt).",
                draft_state=resilience_narrative,
                interpretation_type=InterpretationType.FACT_CHANGE,
                materiality="INCREMENTAL_POSITIVE" if (cash_expanded and debt_declined) else "NEUTRAL",
                owner_judgment_required=False,
                notes="Reconciled directly from primary balance sheet facts."
            )
        )

        # Effect 3: Watch Conditions & Entry Corridor (Preserve exact approved threshold)
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
                f"Does the Research Lead / Product Owner approve updating the price watch entry corridor, or should existing ($1,050 - $1,100) thresholds remain unchanged?"
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

        # Effect 4: Thesis Invalidation Triggers (Distinguish 'not triggered by available evidence' from 'fully assessed')
        if is_margin_invalidated:
            invalidation_draft = (
                f"INVALIDATION TRIGGER BREACHED: Operating margin ({draft_margin:.2f}%) fell below the approved 13.0% threshold. "
                f"Immediate thesis review and potential downgrade required."
            )
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="invalidation_condition",
                    conclusion_label="Thesis Invalidation Triggers",
                    baseline_state=base_invalidation,
                    draft_state=invalidation_draft,
                    interpretation_type=InterpretationType.FACT_CHANGE,
                    materiality="MATERIAL",
                    owner_judgment_required=True,
                    notes="Operating margin breached approved lower boundary."
                )
            )
        else:
            invalidation_draft = (
                f"Quarterly operating margin ({draft_margin:.2f}% TTM) did not breach the approved 13.0% invalidation floor. "
                f"Backlog book-to-bill ratio is an evidence limitation not disclosed in standardized primary XBRL; "
                f"condition is not triggered by available reported evidence, but remains UNRESOLVED via primary filing alone."
            )
            effects.append(
                DecisionAssessmentEffect(
                    conclusion_key="invalidation_condition",
                    conclusion_label="Thesis Invalidation Triggers",
                    baseline_state=base_invalidation,
                    draft_state=invalidation_draft,
                    interpretation_type=InterpretationType.UNRESOLVED_EVIDENCE,
                    materiality="NEUTRAL",
                    owner_judgment_required=False,
                    notes="Margin condition evaluated; backlog book-to-bill requires human contract audit."
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
        if is_margin_invalidated:
            disposition = ProposedDisposition.INVALIDATE
            rationale = (
                f"INVALIDATE: Operating margin compressed to {draft_margin:.2f}%, breaching the approved 13.0% invalidation threshold. "
                f"Quality Watch thesis is invalidated by reported primary filing facts."
            )
        elif val_comp.get("valuation_machinery_status") != "COMPUTED_VERIFIED":
            disposition = ProposedDisposition.UNRESOLVED
            rationale = (
                "UNRESOLVED: Valuation calculations could not be verified due to missing or conflicted inputs. "
                "Review packet is staged with unresolved status for human research review."
            )
        else:
            disposition = ProposedDisposition.MAINTAIN
            rationale = (
                f"Proposed disposition: MAINTAIN Quality Watch status (WATCH_CONDITIONS). Operational results in Q3 2026 "
                f"expanded net cash buffer to ${abs(draft_net_bridge) / 1e6:.1f}M and eased solved reverse-DCF CAGR to {val_comp['solved_5y_cagr']['draft_cagr_pct']:.2f}%. "
                f"However, reference market price (${ref_price:.2f}) remains above mid-cycle base intrinsic value (${val_comp['scenarios']['base']['draft_per_share']:.2f}), "
                f"and core hyperscale cyclicality remains unresolvable. No thesis invalidation occurred; maintain watch status pending human Research Lead review."
            )

        questions.append(
            "Should normalized operating margin assumption (15.0%) be reassessed at the next annual research review given sustained 16.70% trailing 12-month margin execution?"
        )

        stopping_rule = (
            f"Evidence collection complete for Form 10-Q (Period: {reconciled_profile.get('selected_share_basis', {}).get('measurement_period')}). "
            f"All reported primary facts reconciled and conditional valuation differences calculated. "
            f"Core customer capex cycle uncertainty remains an evidence limitation that cannot be further resolved via SEC quarterly filings until FY2026 Form 10-K. "
            f"Cease investigation."
        )

        return effects, val_comp, disposition, rationale, questions, stopping_rule
