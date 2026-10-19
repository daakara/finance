"""
analyst_dashboard/research/sourced_fundamentals/service.py

Read-Only Sourced Fundamentals & Financial Evidence Service.
Exposes bounded, versioned company fundamentals snapshots through pure read-only access.
Zero side-effects: GET requests do not trigger SEC harvesting, valuation solving,
filesystem writes, or database mutations.

Enforces strict facts-only scope:
- Excludes DCF scenarios, fair values, and investment rankings from the product-facing contract.
- Truthful state handling for missing, malformed, or valid snapshots.
- Strict symbol identity validation.
- Preservation of null values without synthetic zero substitution.
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, Any, List, Optional
from pathlib import Path

from analyst_dashboard.research.sourced_fundamentals.identity import (
    verify_company_identity,
    validate_cik_format,
    ATTRIBUTABLE_SEC_IDENTITY_AUTHORITY,
)

logger = logging.getLogger("analyst_dashboard.research.sourced_fundamentals.service")

LEDGER_FILENAME = "sourced_fundamentals_ledger.json"
LEDGER_DIR = Path(__file__).resolve().parent
LEDGER_PATH = LEDGER_DIR / LEDGER_FILENAME

# SEC EDGAR base URL for archival search index
SEC_EDGAR_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"


class IdentityIntegrityError(ValueError):
    """Raised when an internal company record violates attributable SEC identity ground truth."""
    pass


class SnapshotState(str, Enum):
    READY = "READY"
    MISSING_LEDGER = "MISSING_LEDGER"
    MALFORMED_LEDGER = "MALFORMED_LEDGER"
    EMPTY_LEDGER = "EMPTY_LEDGER"


def format_sec_edgar_url(cik: Optional[str], accession_number: Optional[str]) -> Optional[str]:
    """Formats canonical SEC EDGAR direct document index URL for a given accession and CIK."""
    if not cik or not accession_number:
        return None
    try:
        clean_cik = str(int(cik.strip()))  # strip leading zeroes for directory path
    except (ValueError, TypeError):
        return None
    clean_acc_no_hyphens = accession_number.strip().replace("-", "")
    clean_acc = accession_number.strip()
    return f"{SEC_EDGAR_ARCHIVE_BASE}/{clean_cik}/{clean_acc_no_hyphens}/{clean_acc}-index.htm"


class SourcedFundamentalsService:
    """Read-only service for querying company fundamentals snapshots."""

    def __init__(self, ledger_path: Optional[Path] = None):
        self._ledger_path = ledger_path or LEDGER_PATH
        self._data: Optional[Dict[str, Any]] = None
        self._state: SnapshotState = SnapshotState.MISSING_LEDGER
        self._error_detail: Optional[str] = None
        self._load_ledger()

    def _load_ledger(self) -> None:
        """Loads the serialized immutable research ledger snapshot into memory with strict validation."""
        if not self._ledger_path.exists():
            self._state = SnapshotState.MISSING_LEDGER
            self._error_detail = "Sourced fundamentals ledger snapshot file is not present on disk."
            self._data = None
            logger.warning(f"Sourced fundamentals ledger not found at {self._ledger_path}")
            return

        try:
            with open(self._ledger_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if not isinstance(loaded, dict):
                self._state = SnapshotState.MALFORMED_LEDGER
                self._error_detail = "Sourced fundamentals ledger root must be a JSON object."
                self._data = None
                return

            if "metadata" not in loaded or not isinstance(loaded.get("metadata"), dict):
                self._state = SnapshotState.MALFORMED_LEDGER
                self._error_detail = "Sourced fundamentals ledger metadata is missing or not a dictionary."
                self._data = None
                return

            if "companies" not in loaded or not isinstance(loaded.get("companies"), dict):
                self._state = SnapshotState.MALFORMED_LEDGER
                self._error_detail = "Sourced fundamentals ledger companies mapping is missing or not a dictionary."
                self._data = None
                return

            companies = loaded.get("companies", {})
            if len(companies) == 0:
                self._state = SnapshotState.EMPTY_LEDGER
                self._error_detail = "Sourced fundamentals ledger contains no company records."
                self._data = None
                return

            # Comprehensive structural & identity validation per company record
            for sym, cdata in companies.items():
                if not isinstance(cdata, dict):
                    self._state = SnapshotState.MALFORMED_LEDGER
                    self._error_detail = f"Company record for '{sym}' must be a JSON object."
                    self._data = None
                    return

                fundamentals = cdata.get("fundamentals")
                if not isinstance(fundamentals, dict):
                    self._state = SnapshotState.MALFORMED_LEDGER
                    self._error_detail = f"Company record for '{sym}' is missing valid fundamentals object."
                    self._data = None
                    return

                derived_metrics = cdata.get("derived_metrics")
                if not isinstance(derived_metrics, dict):
                    self._state = SnapshotState.MALFORMED_LEDGER
                    self._error_detail = f"Company record for '{sym}' is missing valid derived_metrics object."
                    self._data = None
                    return

                # Validate identity agreement against attributable SEC ground truth
                ident_res = verify_company_identity(
                    requested_symbol=sym,
                    record_symbol=fundamentals.get("symbol"),
                    record_cik=fundamentals.get("cik"),
                )
                if not ident_res.is_valid:
                    self._state = SnapshotState.MALFORMED_LEDGER
                    self._error_detail = f"Identity integrity validation failed for '{sym}': {ident_res.error}"
                    self._data = None
                    return

            self._data = loaded
            self._state = SnapshotState.READY
            self._error_detail = None
            logger.info(
                f"Successfully loaded sourced fundamentals ledger from {self._ledger_path} "
                f"({len(companies)} companies validated)"
            )
        except json.JSONDecodeError as jde:
            self._state = SnapshotState.MALFORMED_LEDGER
            self._error_detail = f"Sourced fundamentals ledger contains invalid JSON: {jde.msg}."
            self._data = None
            logger.error(f"Malformed JSON in sourced fundamentals ledger: {jde}")
        except Exception as e:
            self._state = SnapshotState.MALFORMED_LEDGER
            self._error_detail = "Failed to load sourced fundamentals ledger due to an unexpected read error."
            self._data = None
            logger.error(f"Error loading sourced fundamentals ledger: {e}", exc_info=True)

    @property
    def state(self) -> SnapshotState:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == SnapshotState.READY

    @property
    def error_detail(self) -> Optional[str]:
        return self._error_detail

    def get_snapshot_metadata(self) -> Dict[str, Any]:
        """Returns top-level snapshot metadata, research cutoff date, and compliance attestation."""
        if not self.is_ready or not self._data:
            return {
                "status": self._state.value,
                "read_only": True,
                "is_available": False,
                "error": self._error_detail,
            }
        meta = dict(self._data.get("metadata", {}))
        meta["api_contract_version"] = "2.0.0"
        meta["read_only"] = True
        meta["status"] = self._state.value
        meta["is_available"] = True
        return meta

    def list_available_companies(self) -> List[Dict[str, Any]]:
        """Returns catalog summary of all companies present in the current snapshot."""
        if not self.is_ready or not self._data or "companies" not in self._data:
            return []

        results = []
        for sym, cdata in sorted(self._data["companies"].items()):
            f = cdata.get("fundamentals") or {}
            dm = cdata.get("derived_metrics") or {}
            results.append({
                "symbol": sym,
                "company_name": f.get("company_name", sym),
                "cik": f.get("cik"),
                "primary_exchange": f.get("primary_exchange"),
                "sector_industry": f.get("sector_industry"),
                "latest_reported_period": f.get("latest_reported_period"),
                "latest_filing_date": f.get("latest_filing_date"),
                "total_financial_debt_evidenced": (
                    dm.get("total_financial_debt") is not None
                ),
            })
        return results

    def validate_symbol_identity(self, requested_symbol: str, company_data: Dict[str, Any]) -> bool:
        """
        Validates that requested symbol matches company record's canonical symbol and CIK
        against authoritative SEC EDGAR ground truth.
        Fails closed on any mismatch, whitespace/format anomaly, or ungrounded binding.
        """
        if not requested_symbol or not isinstance(requested_symbol, str):
            return False
        clean_sym = requested_symbol.strip().upper()
        if not clean_sym:
            return False
        if not isinstance(company_data, dict):
            return False
        f = company_data.get("fundamentals") or {}
        if not isinstance(f, dict):
            return False
        canonical_sym = f.get("symbol")
        cik = f.get("cik")
        ident_res = verify_company_identity(clean_sym, canonical_sym, cik)
        return ident_res.is_valid

    def get_company_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves full, structured financial evidence snapshot for a company.
        Fails closed: returns None if symbol is uncataloged.
        Raises IdentityIntegrityError if an existing record has an identity violation or corrupt data.
        Facts-only: Excludes DCF scenarios, fair values, and investment rankings.
        """
        if not symbol or not isinstance(symbol, str) or not self.is_ready or not self._data:
            return None

        clean_sym = symbol.strip().upper()
        if not clean_sym or clean_sym not in self._data["companies"]:
            return None

        cdata = self._data["companies"][clean_sym]
        if not isinstance(cdata, dict):
            raise IdentityIntegrityError(f"Corrupt company record for '{clean_sym}' in ledger.")

        f = cdata.get("fundamentals")
        if not isinstance(f, dict):
            raise IdentityIntegrityError(f"Missing fundamentals object for '{clean_sym}'.")

        dm = cdata.get("derived_metrics")
        if not isinstance(dm, dict):
            raise IdentityIntegrityError(f"Missing derived_metrics object for '{clean_sym}'.")

        asmt = cdata.get("assessment") or {}
        cik = f.get("cik")

        # Identity validation (F01 / P02): Verify that internal symbol and CIK match ground truth
        ident_res = verify_company_identity(clean_sym, f.get("symbol"), cik)
        if not ident_res.is_valid:
            logger.error(f"Identity integrity error in snapshot for '{clean_sym}': {ident_res.error}")
            raise IdentityIntegrityError(ident_res.error)

        # 1. Transform reported facts with attributable SEC links and fact types (F02)
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

            raw_status = fact.get("status")
            if raw_status == "REPORTED":
                fact_type = "PRIMARY_REPORTED"
                fact_status = "REPORTED"
            elif raw_status == "DERIVED":
                fact_type = "DERIVED_CALCULATION"
                fact_status = "DERIVED"
            elif raw_status == "RESEARCH_ESTIMATE":
                fact_type = "RESEARCH_ESTIMATE"
                fact_status = "RESEARCH_ESTIMATE"
            elif raw_status == "UNAVAILABLE":
                fact_type = "UNAVAILABLE"
                fact_status = "UNAVAILABLE"
            else:
                if acc and not fact.get("formula"):
                    fact_type = "PRIMARY_REPORTED"
                    fact_status = "REPORTED"
                else:
                    fact_type = "DERIVED_CALCULATION"
                    fact_status = "DERIVED"

            reported_facts.append({
                "metric_key": metric_key,
                "value": fact.get("value"),
                "units": fact.get("units", "USD"),
                "concept": fact.get("concept"),
                "accession_number": acc,
                "form": fact.get("form"),
                "period_start": p_start,
                "period_end": p_end,
                "filing_date": fact.get("filing_date"),
                "retrieval_time": fact.get("retrieval_time"),
                "status": fact_status,
                "source_url": sec_url,
                "formula": fact.get("formula"),
                "input_references": fact.get("input_references", []),
                "notes": fact.get("notes"),
                "duration_type": duration_type,
                "fact_type": fact_type,
            })

        # Sort facts deterministically by metric_key
        reported_facts.sort(key=lambda x: x["metric_key"])

        # Total debt status and formula (F03)
        total_debt_fact = raw_facts.get("total_financial_debt", {})
        if dm.get("total_financial_debt") is None:
            total_debt_status = "UNAVAILABLE"
            total_debt_formula = None
        elif total_debt_fact.get("status") == "DERIVED":
            total_debt_status = "DERIVED"
            total_debt_formula = total_debt_fact.get("formula") or "Short-Term Debt + Long-Term Debt"
        elif total_debt_fact.get("status") == "REPORTED":
            total_debt_status = "REPORTED"
            total_debt_formula = None
        else:
            total_debt_status = "DERIVED" if (dm.get("short_term_debt") is not None or dm.get("long_term_debt") is not None) else "REPORTED"
            total_debt_formula = "Short-Term Debt + Long-Term Debt" if total_debt_status == "DERIVED" else None

        # Diluted shares period extraction (F03)
        diluted_fact = raw_facts.get("diluted_shares_weighted_avg", {})
        share_period_start = diluted_fact.get("period_start")
        share_period_end = diluted_fact.get("period_end") or f.get("latest_reported_period")
        ttm_denom_desc = dm.get("ttm_eps_denominator_period")
        if not share_period_start and ttm_denom_desc and " to " in ttm_denom_desc:
            share_period_start = ttm_denom_desc.split(" to ")[0].strip()
        share_period_type = "ANNUAL_WEIGHTED_AVERAGE" if (clean_sym == "KLAC") else "QUARTERLY_WEIGHTED_AVERAGE"

        # 2. Extract derived calculations with explicit basis & formulas (Facts only)
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
                "status": total_debt_status,
                "formula": total_debt_formula,
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
                "label": "Operating Lease Liabilities (Noncurrent)",
                "concept": raw_facts.get("operating_lease_liabilities", {}).get("concept"),
                "period_end": f.get("latest_reported_period"),
                "notes": "Contractual noncurrent operating lease obligation under ASC 842 (treated as operating expense, not financial debt)",
            },
            "diluted_shares_weighted_avg": {
                "value": dm.get("diluted_shares_weighted_avg"),
                "units": "shares",
                "status": "REPORTED" if dm.get("diluted_shares_weighted_avg") is not None else "UNAVAILABLE",
                "period_start": share_period_start,
                "period_end": share_period_end,
                "period_type": share_period_type,
                "period_description": ttm_denom_desc,
                "notes": "Reported weighted-average diluted shares from latest periodic filing (not an annual average for quarterly filers).",
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

        # 3. Assemble full facts-only response contract (P09: No DCF scenarios or investment rankings)
        try:
            sec_browse_url = f"https://www.sec.gov/edgar/browse/?CIK={int(cik)}"
        except (ValueError, TypeError):
            sec_browse_url = None

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
                "sec_edgar_company_url": sec_browse_url,
            },
            "reported_facts": reported_facts,
            "derived_calculations": derived_calculations,
            "research_evidence_notes": {
                "evidence_sufficiency": asmt.get("evidence_sufficiency"),
                "remaining_evidence_gaps": asmt.get("remaining_evidence_gaps", []),
                "thesis_invalidation_conditions": asmt.get("thesis_invalidation_conditions", []),
                "specific_next_evidence_to_monitor": asmt.get("specific_next_evidence_to_monitor", []),
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


def reset_sourced_fundamentals_service() -> None:
    """Resets singleton instance for test fixtures."""
    global _SERVICE_INSTANCE
    _SERVICE_INSTANCE = None
