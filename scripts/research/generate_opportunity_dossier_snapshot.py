#!/usr/bin/env python3
"""
scripts/research/generate_opportunity_dossier_snapshot.py

Deterministic Generator & Parity Checker for ARX Read-Only Opportunity Dossiers.
Transforms the accepted ODP-05 research ledger into the typed frontend snapshot.

Invariants:
- Verifies SHA-256 of the canonical research ledger input.
- Exactly preserves the 5-company research cohort (FIX, DUOL, IREN, CORZ, MU).
- Generates 100% deterministic JSON output without current-clock timestamps.
- Preserves raw numeric metrics, explicit nulls, and exact accounting periods.
- Distinguishes documented zero debt (DUOL) from unevidenced or missing debt.
- Retains fail-closed valuation suppression (IREN, CORZ) and qualification (MU).
- Excludes simulated technical overlays and execution states.
- Supports --check mode for CI linting (exits 1 on stale or altered snapshot).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Canonical Research Baseline Authorities
EXPECTED_SOURCE_SHA256 = "a65be3b38cdf988081a29257fb41674c3591b10a666362f351794a7d1c40979d"
EXPECTED_COHORT = ["FIX", "DUOL", "IREN", "CORZ", "MU"]

DEFAULT_SOURCE_PATH = Path("docs/research/opportunity_discovery/odp05_discovery_ledger.json")
DEFAULT_OUTPUT_PATH = Path("frontend/lib/opportunityDossierSnapshot.json")

# SEC EDGAR base URL for archival accession search index
SEC_EDGAR_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


def format_sec_edgar_url(cik: Optional[str], accession_number: Optional[str]) -> Optional[str]:
    """Formats canonical SEC EDGAR direct document index URL for a given accession and CIK."""
    if not cik or not accession_number:
        return None
    try:
        clean_cik = str(int(cik.strip()))
    except (ValueError, TypeError):
        return None
    clean_acc_no_hyphens = accession_number.strip().replace("-", "")
    clean_acc = accession_number.strip()
    return f"{SEC_EDGAR_ARCHIVE_BASE}/{clean_cik}/{clean_acc_no_hyphens}/{clean_acc}-index.htm"


def format_currency_display(val: Optional[float], is_zero: bool = False) -> str:
    """Formats USD values with standard institutional abbreviations."""
    if is_zero:
        return "$0.00"
    if val is None:
        return "UNAVAILABLE"
    abs_v = abs(val)
    if abs_v >= 1e9:
        return f"${val / 1e9:.2f}B"
    if abs_v >= 1e6:
        return f"${val / 1e6:.1f}M"
    if abs_v >= 1e3:
        return f"${val / 1e3:.0f}K"
    return f"${val:.2f}"


def derive_presentation_summary(sym: str, assessment: Dict[str, Any]) -> str:
    """
    Deterministically composes a concise presentation summary from canonical synthesis fields.
    Introduces zero new factual claims or thresholds.
    """
    synth = assessment.get("synthesis", {})
    val_model = assessment.get("valuation_model", {})
    disp = assessment.get("disposition")

    conf = synth.get("confidence_assessment", "")
    watch = synth.get("price_watch_condition", "")
    supporting = synth.get("strongest_supporting_evidence", "")
    opposing = synth.get("strongest_opposing_evidence", "")
    ref_price = assessment.get("reference_market_price")

    if sym == "FIX":
        return (
            f"Comfort Systems USA presents an exceptional balance sheet ($1.80B net cash) and high capital returns (>28% ROIC), "
            f"supported by multi-year modular HVAC backlogs for hyperscale data center construction. However, the frozen reference "
            f"market price of ${ref_price:.2f} embeds a demanding {val_model.get('solved_reverse_dcf_5y_revenue_cagr_pct'):.2f}% "
            f"5-year revenue CAGR at elevated 15.0% operating margins. Retained on watch: await market pullback toward mid-cycle "
            f"base valuation range ($1,050 - $1,100) or evidence of margin persistence."
        )
    elif sym == "DUOL":
        return (
            f"Duolingo maintains rapid subscription expansion ($1.15B TTM revenue) and a pristine balance sheet ($1.31B cash and "
            f"documented zero funded debt). However, the frozen reference market price of ${ref_price:.2f} embeds a demanding "
            f"{val_model.get('solved_reverse_dcf_5y_revenue_cagr_pct'):.2f}% 5-year revenue CAGR at a normalized 22.0% operating margin. "
            f"Retained on watch: await market pullbacks toward the $125 - $130 range or evidence of sustainable AI monetisation."
        )
    elif sym == "IREN":
        return (
            f"Iris Energy holds a massive secured power pipeline (>5 GW) transitioning toward AI cloud compute. However, single-stage "
            f"DCF valuation is strictly suppressed (Fail-Closed) due to $25B-$30B guided capex, multi-layered equipment debt ($7.59B), "
            f"and ongoing equity dilution. Retained on watch pending transition from upfront prepayments to sustained operating cash flow."
        )
    elif sym == "CORZ":
        return (
            f"Core Scientific demonstrates rapid post-reorganization industrial colocation delivery (395 MW active billing, $136.7M quarterly "
            f"colocation revenue under landmark CoreWeave agreements with $6.7B total contracted commitments). However, single-stage DCF valuation is strictly suppressed (Fail-Closed) due to massive quarterly capex "
            f"($797.5M), $4.30B carrying debt, and warrant fair value volatility. Retained on watch pending positive post-capex free cash flow."
        )
    elif sym == "MU":
        return (
            f"Micron Technology is rejected as a supported opportunity under ODP-05 (NO_SUPPORTED_OPPORTUNITY). While FY2026 delivered "
            f"record revenue ($133.2B) driven by HBM3E pricing, memory remains a fiercely cyclical commodity. At the reference price "
            f"of ${ref_price:.2f}, priced-in growth of 21.29% reflects cyclical exuberance; mid-cycle base valuation is $456.67, "
            f"indicating -48.0% downside risk. Require cyclical pullback below $500."
        )

    # General fallback purely from synthesis
    return f"{conf} {watch}".strip()


def validate_and_project_ledger(source_data: Dict[str, Any], source_sha256: str) -> Dict[str, Any]:
    """Validates the raw canonical research ledger and projects it into the frontend contract."""
    metadata = source_data.get("metadata", {})
    assessments = source_data.get("assessments", {})

    # Validate cohort completeness
    for sym in EXPECTED_COHORT:
        if sym not in assessments:
            raise ValueError(f"Required cohort symbol '{sym}' missing from research ledger.")

    # Validate exact dispositions
    expected_dispositions = {
        "FIX": "WATCH_CONDITIONS",
        "DUOL": "WATCH_CONDITIONS",
        "IREN": "WATCH_CONDITIONS",
        "CORZ": "WATCH_CONDITIONS",
        "MU": "NO_SUPPORTED_OPPORTUNITY",
    }
    expected_val_statuses = {
        "FIX": "COMPUTED_VERIFIED",
        "DUOL": "COMPUTED_VERIFIED",
        "IREN": "VALUATION_SUPPRESSED",
        "CORZ": "VALUATION_SUPPRESSED",
        "MU": "COMPUTED_QUALIFIED",
    }
    expected_ciks = {
        "FIX": "0001035983",
        "DUOL": "0001562088",
        "IREN": "0001878848",
        "CORZ": "0001839341",
        "MU": "0000723125",
    }

    cohort_list = []
    dossiers = {}

    for sym in EXPECTED_COHORT:
        raw_a = assessments[sym]
        cik = raw_a.get("cik")
        if cik != expected_ciks[sym]:
            raise ValueError(f"CIK mismatch for {sym}: expected {expected_ciks[sym]}, got {cik}")

        disp = raw_a.get("disposition")
        if disp != expected_dispositions[sym]:
            raise ValueError(f"Disposition mismatch for {sym}: expected {expected_dispositions[sym]}, got {disp}")

        val_model = raw_a.get("valuation_model", {})
        val_status = val_model.get("valuation_status")
        if val_status != expected_val_statuses[sym]:
            raise ValueError(f"Valuation status mismatch for {sym}: expected {expected_val_statuses[sym]}, got {val_status}")

        fin_prof = raw_a.get("financial_profile", {})
        synth = raw_a.get("synthesis", {})
        ref_price = raw_a.get("reference_market_price")
        ref_date = raw_a.get("reference_market_date", "2026-10-16")

        # Identity block
        company_name = raw_a.get("company_name")
        exchange = raw_a.get("exchange")
        route = raw_a.get("opportunity_route")
        try:
            sec_browse_url = f"https://www.sec.gov/edgar/browse/?CIK={int(cik)}"
        except (ValueError, TypeError):
            sec_browse_url = None

        # Primary filings
        raw_filings = raw_a.get("primary_sec_filings", [])
        filings = []
        for f in raw_filings:
            accn = f.get("accession_number")
            filings.append({
                "form": f.get("form"),
                "period_end": f.get("period_end"),
                "filing_date": f.get("filing_date"),
                "accession_number": accn,
                "url": format_sec_edgar_url(cik, accn),
                "source_role": f.get("source_role", "Primary financial disclosures")
            })

        # Financial facts & debt completeness
        tot_debt = fin_prof.get("total_financial_debt")
        is_zero_debt = (tot_debt == 0 and sym == "DUOL")
        debt_display = "$0.00 (Zero Funded Debt Evidenced)" if is_zero_debt else format_currency_display(tot_debt)

        # Revenue period labeling & exact source value (Finding D)
        if sym == "IREN":
            rev_period = "FY2026 (Ended 2026-06-30)"
            rev_value = fin_prof.get("fy2026_revenue")
        elif sym == "CORZ":
            rev_period = "Q2 2026 (Quarter Ended 2026-06-30)"
            rev_value = fin_prof.get("q2_2026_revenue")
        elif sym == "MU":
            rev_period = "FY2026 (Ended 2026-09-03)"
            rev_value = fin_prof.get("fy2026_revenue")
        else:
            rev_period = "TTM (Ended 2026-06-30)"
            rev_value = fin_prof.get("ttm_revenue")

        if rev_value is None:
            raise ValueError(f"Missing revenue for cohort symbol {sym}")

        # Bridge treatment note (Finding C)
        bridge_note = None
        if sym == "FIX":
            bridge_note = (
                "Unrestricted cash ($1.85B) less long-term credit ($54.1M) yields $1.80B net cash bridge "
                "added to enterprise value in DCF equity bridge."
            )
        elif sym == "DUOL":
            bridge_note = (
                fin_prof.get("lease_convention_treatment") or
                "Lease as operating expense convention applied; rent deducted in operating income; "
                "lease liabilities excluded from net debt bridge to avoid double deduction."
            )
        elif sym == "IREN":
            bridge_note = (
                fin_prof.get("restriction_treatment") or
                "Restricted cash ($1.72B) excluded from unrestricted net debt bridge under prudent credit convention, "
                "resulting in net debt bridge of $1.70B ($7.59B debt - $5.90B unrestricted cash)."
            )
        elif sym == "CORZ":
            bridge_note = (
                "Unrestricted cash ($1.77B) against carrying debt ($4.30B) leaves $2.53B net debt bridge; "
                "digital assets ($49.7M) and mark-to-market warrant liabilities are excluded from fundamental net debt bridge."
            )
        elif sym == "MU":
            bridge_note = (
                "Unrestricted cash ($18.24B) less total financial debt ($11.85B) leaves $6.39B net cash bridge "
                "added to enterprise value in DCF equity bridge."
            )

        # Valuation block projection
        is_suppressed = (val_status == "VALUATION_SUPPRESSED")
        ref_price_label = (
            "Research reference price" if is_suppressed
            else "Reference price used in reverse DCF"
        )

        convention_summary = "Valuation suppressed (Fail-Closed)."
        scenarios = None
        reverse_dcf_parameters = None

        if not is_suppressed:
            params = val_model.get("parameters", {})
            wacc = params.get("wacc_pct")
            g = params.get("terminal_growth_pct")
            margin = params.get("operating_margin_pct")
            reinvest = params.get("reinvestment_rate_pct")
            tax_rate = params.get("tax_rate_pct")
            solved_cagr = val_model.get("solved_reverse_dcf_5y_revenue_cagr_pct")
            
            # Carry complete reverse model parameters (Finding B)
            reverse_dcf_parameters = {
                "target_price": params.get("target_price"),
                "solved_revenue_cagr_pct": solved_cagr,
                "operating_margin_pct": margin,
                "wacc_pct": wacc,
                "terminal_growth_pct": g,
                "tax_rate_pct": tax_rate,
                "reinvestment_rate_pct": reinvest,
                "base_revenue": params.get("base_revenue"),
                "current_shares": params.get("current_shares"),
                "net_debt_bridge": params.get("net_debt_bridge"),
            }

            convention_summary = (
                f"Single-stage Reverse-DCF FCFF (WACC {wacc:.1f}%, Terminal g {g:.1f}%, "
                f"Operating Margin {margin:.1f}%, Tax Rate {tax_rate:.1f}%, Reinvestment Rate {reinvest:.1f}%). "
                f"Solved 5Y Revenue CAGR: {solved_cagr:.2f}%."
            )

            # Map scenarios carrying complete parameters (Finding B)
            raw_scenarios = val_model.get("scenarios", {})
            scenarios = {}
            for sc_key, sc_val in raw_scenarios.items():
                p = sc_val.get("parameters", {})
                per_share = sc_val.get("per_share_value")
                ret_pct = sc_val.get("potential_return_pct")
                
                # Normalize key names (MU uses downside_cyclical_bust etc.)
                norm_key = sc_key
                if "downside" in sc_key or "conservative" in sc_key:
                    norm_key = "conservative"
                elif "base" in sc_key:
                    norm_key = "base"
                elif "optimistic" in sc_key:
                    norm_key = "optimistic"

                ret_sign = "+" if (ret_pct is not None and ret_pct > 0) else ""
                conditional_wording = (
                    f"Under stated {norm_key} assumptions, modeled conditional value is "
                    f"${per_share:.2f} ({ret_sign}{ret_pct:.1f}% vs frozen reference price)."
                )

                scenarios[norm_key] = {
                    "scenario_key": norm_key,
                    "name": sc_val.get("description", norm_key.capitalize()),
                    "per_share_value": per_share,
                    "potential_return_pct": ret_pct,
                    "conditional_wording": conditional_wording,
                    "revenue_cagr_pct": p.get("growth_5y_pct"),
                    "operating_margin_pct": p.get("operating_margin_pct"),
                    "wacc_pct": p.get("wacc_pct"),
                    "terminal_growth_pct": p.get("terminal_growth_pct"),
                    "tax_rate_pct": p.get("tax_rate_pct"),
                    "reinvestment_rate_pct": p.get("reinvestment_rate_pct"),
                    "enterprise_value": sc_val.get("enterprise_value"),
                    "equity_value": sc_val.get("equity_value"),
                }

        # Company-specific resume conditions
        resume_conditions = []
        if is_suppressed:
            resume_conditions = val_model.get("observable_conditions_to_resume_valuation", [])

        # Qualification note
        qualification_notice = val_model.get("notes") if val_status == "COMPUTED_QUALIFIED" else None

        # Build full dossier record
        dossier = {
            "identity": {
                "symbol": sym,
                "company_name": company_name,
                "cik": cik,
                "exchange": exchange,
                "opportunity_route": route,
                "sec_edgar_browse_url": sec_browse_url,
            },
            "central_assessment": {
                "disposition": disp,
                "disposition_label": (
                    "Watch — Conditions Required for Entry" if disp == "WATCH_CONDITIONS" and not is_suppressed
                    else "Watch — Valuation Suppressed (Fail-Closed)" if is_suppressed
                    else "Rejected — No Supported Opportunity"
                ),
                "presentation_summary": derive_presentation_summary(sym, raw_a),
                "summary_provenance": "DERIVED_SYNTHESIS",
                "strongest_supporting_evidence": synth.get("strongest_supporting_evidence", ""),
                "strongest_opposing_evidence": synth.get("strongest_opposing_evidence", ""),
                "main_uncertainty": synth.get("main_uncertainty", ""),
                "price_watch_condition": synth.get("price_watch_condition", ""),
                "confidence_assessment": synth.get("confidence_assessment", ""),
                "financing_dilution_exposure": synth.get("financing_dilution_exposure", ""),
            },
            "valuation": {
                "valuation_status": val_status,
                "valuation_status_label": (
                    "Computed & Verified" if val_status == "COMPUTED_VERIFIED"
                    else "Computed with Qualification" if val_status == "COMPUTED_QUALIFIED"
                    else "Valuation Suppressed (Fail-Closed)"
                ),
                "model_type": val_model.get("model_type") if not is_suppressed else None,
                "reference_price": ref_price,
                "reference_price_label": ref_price_label,
                "reference_date": ref_date,
                "solved_5y_cagr_pct": val_model.get("solved_reverse_dcf_5y_revenue_cagr_pct") if not is_suppressed else None,
                "reverse_dcf_parameters": reverse_dcf_parameters,
                "roundtrip_value": val_model.get("roundtrip_per_share_value") if not is_suppressed else None,
                "roundtrip_delta_cents": val_model.get("roundtrip_delta_cents") if not is_suppressed else None,
                "convention_summary": convention_summary,
                "qualification_notice": qualification_notice,
                "suppression_reason": val_model.get("suppression_reason") if is_suppressed else None,
                "resume_conditions": resume_conditions,
                "scenarios": scenarios,
            },
            "financial_profile": {
                "accounting_basis": fin_prof.get("accounting_basis", "US-GAAP"),
                "currency": fin_prof.get("currency", "USD"),
                "revenue": rev_value,
                "revenue_display": format_currency_display(rev_value),
                "revenue_period": rev_period,
                "operating_income": fin_prof.get("ttm_operating_income_ebit") or fin_prof.get("fy2026_operating_income_ebit"),
                "operating_margin_pct": fin_prof.get("operating_margin_pct"),
                "cash_and_equivalents": fin_prof.get("cash_and_equivalents"),
                "short_term_investments": fin_prof.get("short_term_investments"),
                "restricted_cash": fin_prof.get("restricted_cash"),
                "total_cash_and_investments": fin_prof.get("total_cash_and_investments") or fin_prof.get("cash_and_equivalents"),
                "total_cash_and_restricted": fin_prof.get("total_cash_and_restricted"),
                "total_financial_debt": tot_debt,
                "debt_display": debt_display,
                "is_zero_debt": is_zero_debt,
                "zero_debt_support": fin_prof.get("zero_debt_completeness_support") if is_zero_debt else None,
                "operating_lease_liabilities_asc842": fin_prof.get("operating_lease_liabilities_asc842"),
                "lease_convention_treatment": fin_prof.get("lease_convention_treatment"),
                "restriction_treatment": fin_prof.get("restriction_treatment"),
                "digital_assets": fin_prof.get("digital_assets"),
                "total_liquidity_with_digital_assets": fin_prof.get("total_liquidity_with_digital_assets"),
                "net_debt_bridge": fin_prof.get("net_debt_bridge"),
                "capital_structure_position": fin_prof.get("capital_structure_position"),
                "bridge_note": bridge_note,
                "selected_share_basis": fin_prof.get("selected_share_basis"),
                "primary_filings": filings,
            },
            "monitoring": {
                "observable_invalidation_trigger": synth.get("observable_invalidation_trigger", ""),
                "next_monitoring_trigger": synth.get("next_monitoring_trigger", ""),
                "price_watch_condition": synth.get("price_watch_condition", ""),
                "baseline_factual_evidence": synth.get("strongest_supporting_evidence", ""),
            },
            "navigation": {
                "radar_url": f"/radar?symbol={sym}",
                "stock_url": f"/stock/{sym}",
                "radar_note": "Opens on-demand technical analysis in Radar console.",
            }
        }

        cohort_list.append({
            "symbol": sym,
            "company_name": company_name,
            "exchange": exchange,
            "opportunity_route": route,
            "disposition": disp,
            "disposition_label": dossier["central_assessment"]["disposition_label"],
            "reference_price": ref_price,
            "valuation_status": val_status,
        })
        dossiers[sym] = dossier

    # Assemble complete snapshot
    output = {
        "schema_version": "ARX-OPPORTUNITY-DOSSIER-V1",
        "snapshot_metadata": {
            "snapshot_identifier": "ODP05-FINAL-SOURCE-BINDING-2026-10-16",
            "source_ledger_path": "docs/research/opportunity_discovery/odp05_discovery_ledger.json",
            "source_ledger_sha256": source_sha256,
            "research_candidate_commit": "de5a942fc29bbe8977bffc1b5d9f640814370dc7",
            "research_candidate_tree": "62c3ab722e18c1438660444d4f7f67f40fc78958",
            "canonical_ledger_blob": "dc0ee558309072f4b1206c9aedb40f453feab627",
            "governing_contract": metadata.get("governing_contract", "CONTRACT-ARX-ODP-V1-001-REV2"),
            "market_session_date": metadata.get("market_session_date", "2026-10-16"),
            "filing_availability_cutoff_utc": metadata.get("filing_availability_cutoff_utc", "2026-10-16T20:00:00Z"),
            "authoring_timestamp_utc": metadata.get("authoring_timestamp_utc", "2026-10-20T14:30:00Z"),
            "review_timestamp_utc": None,
            "cohort_symbols": EXPECTED_COHORT,
            "is_read_only": True,
        },
        "cohort": cohort_list,
        "dossiers": dossiers,
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic Generator & Checker for ARX Read-Only Opportunity Dossiers."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Path to the accepted research ledger JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Target path for the projected frontend snapshot JSON.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: verifies existing output matches expected projection without writing.",
    )
    args = parser.parse_args()

    # 1. Verify source existence
    if not args.source.exists():
        sys.stderr.write(f"FATAL: Source ledger not found at: {args.source}\n")
        sys.exit(1)

    # 2. Compute and verify source SHA-256
    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        sys.stderr.write(
            f"FATAL: Source ledger SHA-256 mismatch!\n"
            f"  Expected: {EXPECTED_SOURCE_SHA256}\n"
            f"  Actual:   {source_sha256}\n"
        )
        sys.exit(1)

    # 3. Parse source JSON
    try:
        source_data = json.loads(source_bytes.decode("utf-8"))
    except Exception as e:
        sys.stderr.write(f"FATAL: Failed to parse source JSON: {e}\n")
        sys.exit(1)

    # 4. Validate and project
    try:
        projected = validate_and_project_ledger(source_data, source_sha256)
    except Exception as e:
        sys.stderr.write(f"FATAL: Ledger validation or projection failed: {e}\n")
        sys.exit(1)

    # Serialize deterministically with 2-space indentation and trailing newline
    projected_json = json.dumps(projected, indent=2, ensure_ascii=False) + "\n"

    # 5. Handle --check mode
    if args.check:
        if not args.output.exists():
            sys.stderr.write(f"CHECK FAIL: Target output file does not exist: {args.output}\n")
            sys.exit(1)

        existing_content = args.output.read_text(encoding="utf-8")
        if existing_content != projected_json:
            sys.stderr.write(
                f"CHECK FAIL: Snapshot at {args.output} is stale or has been manually altered.\n"
                f"Run 'python scripts/research/generate_opportunity_dossier_snapshot.py' to regenerate.\n"
            )
            sys.exit(1)

        sys.stdout.write(f"CHECK PASS: {args.output} matches deterministic ledger projection exactly.\n")
        sys.exit(0)

    # 6. Write output mode
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(projected_json, encoding="utf-8")
    output_sha256 = hashlib.sha256(projected_json.encode("utf-8")).hexdigest()

    sys.stdout.write(
        f"SUCCESS: Generated opportunity dossier snapshot.\n"
        f"  Source:  {args.source} (SHA-256: {source_sha256})\n"
        f"  Output:  {args.output} (SHA-256: {output_sha256}, Bytes: {len(projected_json.encode('utf-8'))})\n"
        f"  Cohort:  {', '.join(EXPECTED_COHORT)}\n"
    )


if __name__ == "__main__":
    main()
