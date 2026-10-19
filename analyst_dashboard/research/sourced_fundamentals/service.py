"""
analyst_dashboard/research/sourced_fundamentals/service.py

Read-Only Sourced Fundamentals & Financial Evidence Service.
Exposes bounded, versioned company fundamentals snapshots through pure read-only access.
Zero side-effects: GET requests do not trigger SEC harvesting, valuation solving,
filesystem writes, or database mutations.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger("analyst_dashboard.research.sourced_fundamentals.service")

LEDGER_FILENAME = "sourced_fundamentals_ledger.json"
LEDGER_DIR = Path(__file__).resolve().parent
LEDGER_PATH = LEDGER_DIR / LEDGER_FILENAME

# SEC EDGAR base URL for archival search index
SEC_EDGAR_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


def format_sec_edgar_url(cik: Optional[str], accession_number: Optional[str]) -> Optional[str]:
    """Formats canonical SEC EDGAR direct document index URL for a given accession and CIK."""
    if not cik or not accession_number:
        return None
    clean_cik = str(int(cik.strip()))  # strip leading zeroes for directory path
    clean_acc_no_hyphens = accession_number.strip().replace("-", "")
    clean_acc = accession_number.strip()
    return f"{SEC_EDGAR_ARCHIVE_BASE}/{clean_cik}/{clean_acc_no_hyphens}/{clean_acc}-index.htm"


class SourcedFundamentalsService:
    """Read-only service for querying company fundamentals snapshots."""

    def __init__(self, ledger_path: Optional[Path] = None):
        self._ledger_path = ledger_path or LEDGER_PATH
        self._data: Optional[Dict[str, Any]] = None
        self._load_ledger()

    def _load_ledger(self) -> None:
        """Loads the serialized immutable research ledger snapshot into memory."""
        if not self._ledger_path.exists():
            logger.warning(f"Sourced fundamentals ledger not found at {self._ledger_path}")
            self._data = {"metadata": {}, "companies": {}}
            return

        try:
            with open(self._ledger_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            logger.info(
                f"Successfully loaded sourced fundamentals ledger from {self._ledger_path} "
                f"({len(self._data.get('companies', {}))} companies)"
            )
        except Exception as e:
            logger.error(f"Error loading sourced fundamentals ledger: {e}", exc_info=True)
            self._data = {"metadata": {}, "companies": {}}

    def get_snapshot_metadata(self) -> Dict[str, Any]:
        """Returns top-level snapshot metadata, research cutoff date, and compliance attestation."""
        if not self._data:
            return {}
        meta = dict(self._data.get("metadata", {}))
        meta["api_contract_version"] = "1.0.0"
        meta["read_only"] = True
        return meta

    def list_available_companies(self) -> List[Dict[str, Any]]:
        """Returns catalog summary of all companies present in the current snapshot."""
        if not self._data or "companies" not in self._data:
            return []

        results = []
        for sym, cdata in sorted(self._data["companies"].items()):
            f = cdata.get("fundamentals") or {}
            dm = cdata.get("derived_metrics") or {}
            asmt = cdata.get("assessment") or {}
            dv = cdata.get("deep_valuation") or {}
            results.append({
                "symbol": sym,
                "company_name": f.get("company_name", sym),
                "cik": f.get("cik"),
                "primary_exchange": f.get("primary_exchange"),
                "sector_industry": f.get("sector_industry"),
                "latest_reported_period": f.get("latest_reported_period"),
                "latest_filing_date": f.get("latest_filing_date"),
                "disposition": asmt.get("disposition"),
                "valuation_status": dv.get("valuation_status", "UNAVAILABLE"),
                "total_financial_debt_evidenced": dm.get("total_financial_debt") is not None,
            })
        return results

    def get_company_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves full, structured financial evidence snapshot for a company.
        Fails closed (returns None) if symbol is uncataloged or missing in snapshot.
        """
        if not symbol or not self._data or "companies" not in self._data:
            return None

        clean_sym = symbol.strip().upper()
        if clean_sym not in self._data["companies"]:
            return None

        cdata = self._data["companies"][clean_sym]
        f = cdata.get("fundamentals") or {}
        dm = cdata.get("derived_metrics") or {}
        asmt = cdata.get("assessment") or {}
        dv = cdata.get("deep_valuation") or {}
        cik = f.get("cik")

        # 1. Transform reported facts with attributable SEC links and duration types
        reported_facts = []
        raw_facts = f.get("facts", {})
        for metric_key, fact in raw_facts.items():
            acc = fact.get("accession_number")
            sec_url = format_sec_edgar_url(cik, acc)
            
            # Determine duration / instant type
            p_start = fact.get("period_start")
            p_end = fact.get("period_end")
            duration_type = "INSTANT"
            if p_start and p_end and p_start != p_end:
                duration_type = "DURATION"

            reported_facts.append({
                "metric_key": metric_key,
                "value": fact.get("value"),
                "units": fact.get("units"),
                "concept": fact.get("concept"),
                "accession_number": acc,
                "form": fact.get("form"),
                "period_start": p_start,
                "period_end": p_end,
                "filing_date": fact.get("filing_date"),
                "retrieval_time": fact.get("retrieval_time"),
                "status": fact.get("status"),
                "source_url": sec_url,
                "formula": fact.get("formula"),
                "input_references": fact.get("input_references", []),
                "notes": fact.get("notes"),
                "duration_type": duration_type,
            })

        # Sort facts deterministically by metric_key
        reported_facts.sort(key=lambda x: x["metric_key"])

        # 2. Extract derived calculations with explicit basis & formulas
        derived_calculations = {
            "period_label": dm.get("period_label"),
            "revenue_ttm": {
                "value": dm.get("revenue_ttm"),
                "units": "USD",
                "status": "DERIVED" if dm.get("revenue_ttm") is not None else "UNAVAILABLE",
                "formula": "Sum of 4 consecutive non-overlapping fiscal quarters",
            },
            "revenue_prior_ttm": {
                "value": dm.get("revenue_prior_ttm"),
                "units": "USD",
                "status": "DERIVED" if dm.get("revenue_prior_ttm") is not None else "UNAVAILABLE",
            },
            "revenue_growth_yoy_pct": {
                "value": dm.get("revenue_growth_yoy_pct"),
                "units": "percent",
                "status": "DERIVED" if dm.get("revenue_growth_yoy_pct") is not None else "UNAVAILABLE",
                "formula": "(Revenue TTM - Prior TTM Revenue) / Prior TTM Revenue * 100",
            },
            "operating_income_ttm": {
                "value": dm.get("operating_income_ttm"),
                "units": "USD",
                "status": "DERIVED" if dm.get("operating_income_ttm") is not None else "UNAVAILABLE",
            },
            "operating_margin_pct": {
                "value": dm.get("operating_margin_pct"),
                "units": "percent",
                "status": "DERIVED" if dm.get("operating_margin_pct") is not None else "UNAVAILABLE",
                "formula": "Operating Income TTM / Revenue TTM * 100",
            },
            "net_income_ttm": {
                "value": dm.get("net_income_ttm"),
                "units": "USD",
                "status": "DERIVED" if dm.get("net_income_ttm") is not None else "UNAVAILABLE",
            },
            "cfo_ttm": {
                "value": dm.get("cfo_ttm"),
                "units": "USD",
                "status": "DERIVED" if dm.get("cfo_ttm") is not None else "UNAVAILABLE",
                "formula": "Sum of 4 quarterly operating cash flows using matching prior-period YTD subtractions",
            },
            "capex_ttm": {
                "value": dm.get("capex_ttm"),
                "units": "USD",
                "status": "DERIVED" if dm.get("capex_ttm") is not None else "UNAVAILABLE",
                "formula": "Sum of 4 quarterly property, plant & equipment investments",
            },
            "cfo_minus_capex_ttm": {
                "value": dm.get("cfo_minus_capex_ttm"),
                "units": "USD",
                "status": "DERIVED" if dm.get("cfo_minus_capex_ttm") is not None else "UNAVAILABLE",
                "label": "Operating Cash Flow less Capex (Non-GAAP Free Cash Flow Indicator)",
                "formula": "CFO TTM - Capex TTM",
            },
            "sbc_ttm": {
                "value": dm.get("sbc_ttm"),
                "units": "USD",
                "status": "DERIVED" if dm.get("sbc_ttm") is not None else "UNAVAILABLE",
            },
            "sbc_to_revenue_pct": {
                "value": dm.get("sbc_to_revenue_pct"),
                "units": "percent",
                "status": "DERIVED" if dm.get("sbc_to_revenue_pct") is not None else "UNAVAILABLE",
                "formula": "Stock-Based Compensation TTM / Revenue TTM * 100",
            },
            "cash_and_equivalents": {
                "value": dm.get("cash_and_equivalents"),
                "units": "USD",
                "status": "REPORTED" if dm.get("cash_and_equivalents") is not None else "UNAVAILABLE",
                "period_end": f.get("latest_reported_period"),
            },
            "short_term_debt": {
                "value": dm.get("short_term_debt"),
                "units": "USD",
                "status": "REPORTED" if dm.get("short_term_debt") is not None else "UNAVAILABLE",
                "period_end": f.get("latest_reported_period"),
            },
            "long_term_debt": {
                "value": dm.get("long_term_debt"),
                "units": "USD",
                "status": "REPORTED" if dm.get("long_term_debt") is not None else "UNAVAILABLE",
                "period_end": f.get("latest_reported_period"),
            },
            "total_financial_debt": {
                "value": dm.get("total_financial_debt"),
                "units": "USD",
                "status": "REPORTED" if dm.get("total_financial_debt") is not None else "UNAVAILABLE",
                "period_end": f.get("latest_reported_period"),
                "notes": (
                    "Fully evidenced total financial debt"
                    if dm.get("total_financial_debt") is not None
                    else "UNAVAILABLE under aggregate completeness rules: component or tag missing in primary filing XBRL."
                ),
            },
            "net_debt": {
                "value": dm.get("net_debt"),
                "units": "USD",
                "status": "DERIVED" if dm.get("net_debt") is not None else "UNAVAILABLE",
                "formula": "Total Financial Debt - Total Cash & Short-Term Investments",
            },
            "operating_lease_liabilities": {
                "value": dm.get("operating_lease_liabilities"),
                "units": "USD",
                "status": "REPORTED" if dm.get("operating_lease_liabilities") is not None else "UNAVAILABLE",
                "notes": "Contractual operating lease obligation under ASC 842 (treated as operating expense, not financial debt)",
            },
            "diluted_shares_weighted_avg": {
                "value": dm.get("diluted_shares_weighted_avg"),
                "units": "shares",
                "status": "REPORTED" if dm.get("diluted_shares_weighted_avg") is not None else "UNAVAILABLE",
            },
            "share_dilution_1y_pct": {
                "value": dm.get("share_dilution_1y_pct"),
                "units": "percent",
                "status": "DERIVED" if dm.get("share_dilution_1y_pct") is not None else "UNAVAILABLE",
                "formula": "1-year split-adjusted change in diluted share count",
            },
            "share_dilution_3y_pct": {
                "value": dm.get("share_dilution_3y_pct"),
                "units": "percent",
                "status": "DERIVED" if dm.get("share_dilution_3y_pct") is not None else "UNAVAILABLE",
                "formula": "3-year split-adjusted change in diluted share count",
            },
            "ttm_net_income_per_latest_quarter_diluted_share": {
                "value": dm.get("ttm_net_income_per_latest_quarter_diluted_share"),
                "units": "USD/share",
                "status": "RESEARCH_ESTIMATE",
                "label": "Interim TTM Net Income per Latest-Quarter Diluted Share (Research Estimate)",
                "formula": "TTM Net Income / Latest-Quarter Diluted Weighted-Average Shares",
                "notes": "Research estimate only; not reported TTM diluted EPS under US-GAAP.",
            },
            "market_price": dm.get("market_price"),
            "market_price_date": dm.get("market_price_date"),
            "market_cap": dm.get("market_cap"),
            "enterprise_value": dm.get("enterprise_value"),
        }

        # 3. Assemble full response contract
        return {
            "snapshot_metadata": self.get_snapshot_metadata(),
            "company_identity": {
                "symbol": clean_sym,
                "company_name": f.get("company_name", clean_sym),
                "cik": cik,
                "primary_exchange": f.get("primary_exchange"),
                "sector_industry": f.get("sector_industry"),
                "business_model": f.get("business_model"),
                "fiscal_year_end": f.get("fiscal_year_end"),
                "latest_reported_period": f.get("latest_reported_period"),
                "latest_filing_date": f.get("latest_filing_date"),
                "sec_edgar_company_url": f"https://www.sec.gov/edgar/browse/?CIK={int(cik)}" if cik else None,
            },
            "reported_facts": reported_facts,
            "derived_calculations": derived_calculations,
            "research_disposition_and_evidence": {
                "disposition": asmt.get("disposition"),
                "disposition_rationale": asmt.get("disposition_rationale"),
                "growth_durability": asmt.get("growth_durability"),
                "reinvestment_economics": asmt.get("reinvestment_economics"),
                "balance_sheet_resilience": asmt.get("balance_sheet_resilience"),
                "evidence_sufficiency": asmt.get("evidence_sufficiency"),
                "valuation_uncertainty": asmt.get("valuation_uncertainty"),
                "remaining_evidence_gaps": asmt.get("remaining_evidence_gaps", []),
                "thesis_invalidation_conditions": asmt.get("thesis_invalidation_conditions", []),
                "specific_next_evidence_to_monitor": asmt.get("specific_next_evidence_to_monitor", []),
                "valuation_assessment_status": dv.get("valuation_status", "UNAVAILABLE"),
                "valuation_suppression_reason": dv.get("suppression_reason"),
            },
            "deep_valuation": dv if dv.get("valuation_status") == "AVAILABLE" else {
                "valuation_status": dv.get("valuation_status", "SUPPRESSED_UNAVAILABLE"),
                "suppression_reason": dv.get("suppression_reason"),
                "current_price": dv.get("current_price"),
                "current_price_date": dv.get("current_price_date"),
                "thesis_invalidation_thresholds": dv.get("thesis_invalidation_thresholds", []),
                "next_quarter_evidence_milestones": dv.get("next_quarter_evidence_milestones", []),
            },
        }


# Global singleton instance
_SERVICE_INSTANCE: Optional[SourcedFundamentalsService] = None


def get_sourced_fundamentals_service() -> SourcedFundamentalsService:
    """Returns singleton SourcedFundamentalsService instance."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = SourcedFundamentalsService()
    return _SERVICE_INSTANCE
