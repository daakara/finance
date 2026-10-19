"""
analyst_dashboard/research/sourced_fundamentals/normalization.py

Reproducible Cash Flow Normalization Engine for Comfort Systems USA (FIX).
Unpacks reported GAAP cash conversion ($2,159.81M CFO less Capex) against operating income ($1,849.30M EBIT),
separating reported financials, observed contract liability financing inflows, and explicit analyst assumptions.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationAssumptions:
    """Explicit versioned analyst assumptions for sustainable run-rate cash flow."""
    model_version: str = "2.0.0"
    assumed_da_millions: float = 112.50
    normalized_wc_reinvestment_millions: float = 280.00
    effective_tax_rate_override: Optional[float] = None


def calculate_fix_normalized_cash_flow(
    ebit_raw: Optional[float],
    effective_tax_rate_pct: Optional[float],
    capex_raw: Optional[float],
    reported_cfo_raw: Optional[float],
    assumptions: NormalizationAssumptions = NormalizationAssumptions(),
) -> Dict[str, Any]:
    """
    Computes illustrative sustainable run-rate cash flow from reported ledger inputs
    and versioned analyst assumptions.

    Inputs (in raw USD):
    - ebit_raw: e.g. 1,849,302,000.0 ($1,849.30M)
    - effective_tax_rate_pct: e.g. 21.93106 (%)
    - capex_raw: e.g. 390,259,000.0 ($390.26M)
    - reported_cfo_raw: e.g. 2,550,065,000.0 ($2,550.07M)
    """
    # Guard against missing inputs (R1)
    if any(v is None for v in (ebit_raw, effective_tax_rate_pct, capex_raw, reported_cfo_raw)):
        missing = [
            k for k, v in [
                ("ebit_raw", ebit_raw),
                ("effective_tax_rate_pct", effective_tax_rate_pct),
                ("capex_raw", capex_raw),
                ("reported_cfo_raw", reported_cfo_raw),
            ] if v is None
        ]
        return {
            "status": "INCOMPLETE",
            "error": f"Missing required financial inputs for normalization walk: {', '.join(missing)}.",
            "sustainable_cash_flow_millions": None,
            "normalization_walk": None,
        }

    # 1. Convert to millions for standardized reporting precision ($M rounded to 2 decimal places)
    ebit_m = round(ebit_raw / 1e6, 2)
    tax_rate = (
        assumptions.effective_tax_rate_override
        if assumptions.effective_tax_rate_override is not None
        else effective_tax_rate_pct
    ) / 100.0
    taxes_m = round(ebit_m * tax_rate, 2)
    nopat_m = round(ebit_m - taxes_m, 2)

    capex_m = round(capex_raw / 1e6, 2)
    cfo_m = round(reported_cfo_raw / 1e6, 2)
    reported_cfo_less_capex_m = round(cfo_m - capex_m, 2)

    da_m = assumptions.assumed_da_millions
    wc_reinvest_m = assumptions.normalized_wc_reinvestment_millions

    # 2. Arithmetic sustainable cash flow walk
    sustainable_cf_m = round(nopat_m + da_m - capex_m - wc_reinvest_m, 2)

    # 3. Dynamic explanatory formula string derived from actual calculation values (R1)
    formula_str = (
        f"NOPAT (${nopat_m:,.2f}M) + D&A (${da_m:,.2f}M) - Capex (${capex_m:,.2f}M) - "
        f"Normalized WC Reinvestment (${wc_reinvest_m:,.2f}M) = ${sustainable_cf_m:,.2f}M"
    )

    return {
        "status": "COMPLETE",
        "model_version": assumptions.model_version,
        "calculation_precision": "Standardized in millions ($M) rounded to two decimal places; illustrative modeling precision, not audited penny accounting.",
        "reported_inputs": {
            "ebit_millions": ebit_m,
            "effective_tax_rate_pct": round(tax_rate * 100.0, 2),
            "taxes_millions": taxes_m,
            "nopat_millions": nopat_m,
            "capex_millions": capex_m,
            "cfo_millions": cfo_m,
            "cfo_less_capex_millions": reported_cfo_less_capex_m,
        },
        "analyst_assumptions": {
            "assumed_da_millions": da_m,
            "normalized_wc_reinvestment_millions": wc_reinvest_m,
            "is_illustrative_assumption": True,
            "label_da": "Analyst-assumed annual run-rate depreciation & amortization (illustrative)",
            "label_wc": "Analyst-assumed steady-state working capital reinvestment drag (illustrative)",
        },
        "normalization_walk": {
            "nopat": nopat_m,
            "plus_da": da_m,
            "minus_capex": capex_m,
            "minus_normalized_wc_reinvestment": wc_reinvest_m,
            "sustainable_cash_flow_millions": sustainable_cf_m,
            "formula": formula_str,
        },
        "caveats": [
            "D&A and normalized working capital reinvestment are analyst-selected assumptions, not GAAP audited cash flows.",
            "Reported TTM cash flow ($2,159.81M) was significantly expanded by interim customer contract liabilities (billings in excess of costs).",
            "This model represents an illustrative sensitivity scenario and does not establish certified intrinsic value.",
        ],
    }


def calculate_fix_sustainable_cash_flow_from_ledger(
    ledger_or_service: Any = None,
    assumptions: NormalizationAssumptions = NormalizationAssumptions(),
) -> Dict[str, Any]:
    """
    Adapter connecting the research calculation directly to retained ledger observations (R1).
    Extracts FIX operating income, tax rate, capex, and cash flow from ledger.
    """
    if ledger_or_service is None:
        from analyst_dashboard.research.sourced_fundamentals.service import get_sourced_fundamentals_service
        service = get_sourced_fundamentals_service()
        ledger_data = service._data
    elif hasattr(ledger_or_service, "_data"):
        ledger_data = ledger_or_service._data
    elif isinstance(ledger_or_service, dict):
        ledger_data = ledger_or_service
    elif isinstance(ledger_or_service, (str, Path)):
        p = Path(ledger_or_service)
        if not p.exists():
            return {
                "status": "INCOMPLETE",
                "error": f"Ledger file not found at {p}",
                "sustainable_cash_flow_millions": None,
                "normalization_walk": None,
            }
        with open(p, "r", encoding="utf-8") as f:
            ledger_data = json.load(f)
    else:
        ledger_data = None

    if not isinstance(ledger_data, dict) or "companies" not in ledger_data or "FIX" not in ledger_data["companies"]:
        return {
            "status": "INCOMPLETE",
            "error": "Ledger data missing or company 'FIX' not cataloged in dataset.",
            "sustainable_cash_flow_millions": None,
            "normalization_walk": None,
        }

    cdata = ledger_data["companies"]["FIX"]
    f_facts = cdata.get("fundamentals", {}).get("facts", {})
    dm = cdata.get("derived_metrics", {})

    ebit_raw = dm.get("operating_income_ttm") or f_facts.get("operating_income_ttm", {}).get("value")
    capex_raw = dm.get("capex_ttm") or f_facts.get("capex_ttm", {}).get("value")
    cfo_raw = dm.get("cfo_ttm") or f_facts.get("cfo_ttm", {}).get("value")

    tax_exp_raw = f_facts.get("tax_expense_ttm", {}).get("value")
    pretax_raw = f_facts.get("pretax_income_ttm", {}).get("value")

    if tax_exp_raw is not None and pretax_raw is not None and pretax_raw > 0:
        tax_rate_pct = (tax_exp_raw / pretax_raw) * 100.0
    elif dm.get("effective_tax_rate_pct") is not None:
        tax_rate_pct = dm.get("effective_tax_rate_pct")
    else:
        tax_rate_pct = None

    return calculate_fix_normalized_cash_flow(
        ebit_raw=ebit_raw,
        effective_tax_rate_pct=tax_rate_pct,
        capex_raw=capex_raw,
        reported_cfo_raw=cfo_raw,
        assumptions=assumptions,
    )
