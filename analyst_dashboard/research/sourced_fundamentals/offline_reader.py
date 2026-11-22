"""
analyst_dashboard/research/sourced_fundamentals/offline_reader.py

Offline Fundamentals Reader Integration with Strict Custody Verification.
Enforces:
- Attributable symbol-to-CIK identity validation against ATTRIBUTABLE_SEC_IDENTITY_AUTHORITY.
- Strict SHA-256 cryptographic hash and size verification against custody manifest.json.
- Internal issuer CIK and entity matching inside raw SEC JSON payloads.
- Resolution of acceptanceDateTime directly from retained submissions JSON (unmatched remain None).
- Strict temporal separation of research cutoff date (2026-10-16) from retrieval currentness (2026-11-22).
- Preservation of missing financial metrics as None (MetricStatus.UNAVAILABLE) without synthetic zero substitution.
- Anti-masking protection: prevents runtime estimates (e.g. source: "live_runtime") from masking filing-backed evidence.
- Fails closed with production scoring and publication held under HOLD_SEC_001.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from analyst_dashboard.research.sourced_fundamentals.identity import (
    get_attributable_binding,
    verify_company_identity,
    validate_cik_format,
    IdentityBinding,
)
from analyst_dashboard.research.sourced_fundamentals.contracts import (
    CompanyFundamentals,
    FinancialFact,
    MetricStatus,
)
from analyst_dashboard.research.sourced_fundamentals.normalizer import SecFactNormalizer

logger = logging.getLogger("analyst_dashboard.research.sourced_fundamentals.offline_reader")

# Canonical project paths
DEFAULT_RAW_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "research"
    / "sourced_fundamentals"
    / "raw"
)

RESEARCH_CUTOFF_DATE_DEFAULT = "2026-10-16"
HOLDING_CODE_DEFAULT = "HOLD_SEC_001"


class ManifestIntegrityError(ValueError):
    """Raised when the custody manifest is missing, corrupt, or fails cryptographic hash checks."""
    pass


class IdentityIntegrityError(ValueError):
    """Raised when an issuer symbol or CIK violates attributable SEC identity authority."""
    pass


class EvidenceProtectionError(ValueError):
    """Raised when runtime factor estimates attempt to mask filing-backed evidence."""
    pass


def compute_file_sha256(file_path: Path) -> str:
    """Computes hex SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class OfflineFundamentalsResult:
    """Immutable result container for offline-loaded company fundamentals."""

    def __init__(
        self,
        symbol: str,
        cik: str,
        company_name: str,
        fundamentals: CompanyFundamentals,
        manifest: Dict[str, Any],
        verified_hashes: Dict[str, str],
        retrieval_date: str,
        research_cutoff_date: str,
        scoring_held: bool = True,
        holding_code: str = HOLDING_CODE_DEFAULT,
        provenance: Optional[Dict[str, Any]] = None,
        unresolved_lineage: Optional[Dict[str, Any]] = None,
    ):
        self.symbol = symbol
        self.cik = cik
        self.company_name = company_name
        self.fundamentals = fundamentals
        self.manifest = manifest
        self.verified_hashes = verified_hashes
        self.retrieval_date = retrieval_date
        self.research_cutoff_date = research_cutoff_date
        self.scoring_held = scoring_held
        self.holding_code = holding_code
        self.status = "unavailable" if scoring_held else "ready"
        self.provenance = provenance or {}
        self.unresolved_lineage = unresolved_lineage or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "cik": self.cik,
            "company_name": self.company_name,
            "retrieval_date": self.retrieval_date,
            "research_cutoff_date": self.research_cutoff_date,
            "scoring_held": self.scoring_held,
            "holding_code": self.holding_code,
            "status": self.status,
            "verified_hashes": self.verified_hashes,
            "fundamentals": self.fundamentals.to_dict(),
            "provenance": self.provenance,
            "unresolved_lineage": self.unresolved_lineage,
        }


class OfflineFundamentalsReader:
    """
    Offline reader for retained SEC company facts and submissions.
    Enforces cryptographic hash checks, identity verification, null preservation,
    and anti-masking protection while keeping production scoring held.
    """

    def __init__(self, raw_base_dir: Optional[Path] = None):
        self.raw_base_dir = Path(raw_base_dir) if raw_base_dir else DEFAULT_RAW_DIR
        self.normalizer = SecFactNormalizer()

    def load_symbol(
        self,
        symbol: str,
        research_cutoff_date: str = RESEARCH_CUTOFF_DATE_DEFAULT,
        as_of_date: Optional[str] = None,
        enforce_hash_check: bool = True,
    ) -> OfflineFundamentalsResult:
        """
        Loads and normalizes offline fundamentals for a cataloged symbol.
        Fails closed on identity mismatch, hash mismatch, or payload tampering.
        """
        clean_sym = symbol.strip().upper()
        binding = get_attributable_binding(clean_sym)
        if binding is None:
            raise IdentityIntegrityError(
                f"Symbol '{clean_sym}' is not cataloged in authoritative SEC identity authority."
            )

        cik = binding.cik
        expected_dir = self.raw_base_dir / f"CIK{cik}"
        if not expected_dir.exists():
            # Fall back to root raw directory if individual CIK dir not present
            if (self.raw_base_dir / f"CIK{cik}_submissions.json").exists():
                expected_dir = self.raw_base_dir
            else:
                raise FileNotFoundError(
                    f"Raw SEC custody directory not found for {clean_sym} (CIK {cik}) at {expected_dir}"
                )

        manifest_path = expected_dir / "manifest.json"
        if not manifest_path.exists():
            raise ManifestIntegrityError(
                f"Required custody manifest.json is missing at {manifest_path}"
            )

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            raise ManifestIntegrityError(f"Failed to parse manifest.json at {manifest_path}: {e}")

        # Verify manifest identity binding
        manifest_cik = str(manifest_data.get("cik", "")).zfill(10)
        manifest_symbol = str(manifest_data.get("symbol", "")).strip().upper()
        if manifest_cik != cik:
            raise IdentityIntegrityError(
                f"Manifest CIK '{manifest_cik}' does not match authoritative CIK '{cik}' for {clean_sym}."
            )
        if manifest_symbol and manifest_symbol != clean_sym:
            raise IdentityIntegrityError(
                f"Manifest symbol '{manifest_symbol}' does not match requested symbol '{clean_sym}'."
            )

        verified_hashes: Dict[str, str] = {}
        manifest_files = manifest_data.get("files", {})

        # Cryptographic hash verification of retained files
        expected_submissions_fn = f"CIK{cik}_submissions.json"
        expected_companyfacts_fn = f"CIK{cik}_companyfacts.json"

        for fn in [expected_submissions_fn, expected_companyfacts_fn]:
            file_path = expected_dir / fn
            if not file_path.exists():
                raise FileNotFoundError(f"Required raw evidence file {fn} missing at {file_path}")

            if enforce_hash_check:
                if fn not in manifest_files:
                    raise ManifestIntegrityError(
                        f"Custody file '{fn}' is missing from manifest.json file inventory."
                    )
                expected_meta = manifest_files[fn]
                expected_hash = expected_meta.get("sha256")
                expected_size = expected_meta.get("size_bytes")

                actual_size = file_path.stat().st_size
                if expected_size is not None and actual_size != expected_size:
                    raise ManifestIntegrityError(
                        f"Byte size mismatch for {fn}: manifest specifies {expected_size}, observed {actual_size}"
                    )

                actual_hash = compute_file_sha256(file_path)
                if expected_hash and actual_hash.lower() != expected_hash.lower():
                    raise ManifestIntegrityError(
                        f"Cryptographic hash mismatch for {fn}: manifest specifies {expected_hash}, observed {actual_hash}"
                    )
                verified_hashes[fn] = actual_hash
            else:
                verified_hashes[fn] = compute_file_sha256(file_path)

        # Load raw JSON payloads
        with open(expected_dir / expected_submissions_fn, "r", encoding="utf-8") as f:
            submissions_dict = json.load(f)
        with open(expected_dir / expected_companyfacts_fn, "r", encoding="utf-8") as f:
            companyfacts_dict = json.load(f)

        # Strict inner payload CIK & Entity validation
        inner_sub_cik = str(submissions_dict.get("cik", "")).zfill(10)
        inner_cf_cik = str(companyfacts_dict.get("cik", "")).zfill(10)
        if inner_sub_cik != cik:
            raise IdentityIntegrityError(
                f"Submissions payload CIK '{inner_sub_cik}' does not match authoritative CIK '{cik}'."
            )
        if inner_cf_cik != cik:
            raise IdentityIntegrityError(
                f"Company facts payload CIK '{inner_cf_cik}' does not match authoritative CIK '{cik}'."
            )

        # Extract retrieval currentness date from manifest or submissions
        retrieval_date = "2026-11-22"
        if "generated_at_utc" in manifest_data:
            retrieval_date = str(manifest_data["generated_at_utc"])[:10]
        elif expected_submissions_fn in manifest_files:
            meta = manifest_files[expected_submissions_fn]
            if "retrieval_timestamp_utc" in meta:
                retrieval_date = str(meta["retrieval_timestamp_utc"])[:10]

        # Normalize facts using SecFactNormalizer
        fundamentals = self.normalizer.normalize(
            symbol=clean_sym,
            cik=cik,
            facts_dict=companyfacts_dict,
            submissions_dict=submissions_dict,
            as_of_date=as_of_date,
        )

        # Validate null preservation (no synthetic zero substitution)
        for key in ["short_term_investments", "total_cash_and_investments", "common_shares_outstanding"]:
            fact = fundamentals.facts.get(key)
            if fact is not None and fact.status == MetricStatus.UNAVAILABLE:
                if fact.value is not None:
                    raise ValueError(
                        f"Null preservation violation: metric '{key}' has status UNAVAILABLE but non-null value {fact.value}."
                    )

        # Explicit unresolved lineage calculation across Company Facts vs Submissions
        all_cf_accns = set()
        for tax in companyfacts_dict.get("facts", {}).values():
            if isinstance(tax, dict):
                for c_data in tax.values():
                    if isinstance(c_data, dict):
                        for unit_items in c_data.get("units", {}).values():
                            if isinstance(unit_items, list):
                                for item in unit_items:
                                    if isinstance(item, dict):
                                        a = item.get("accn")
                                        if a:
                                            all_cf_accns.add(str(a).strip())

        recent_filings = submissions_dict.get("filings", {}).get("recent", {})
        sub_accns = set(str(a).strip() for a in recent_filings.get("accessionNumber", []) if a)

        matched_accns = all_cf_accns.intersection(sub_accns)
        unresolved_accns = all_cf_accns - sub_accns

        unresolved_lineage = {
            "totalCompanyFactsAccessions": len(all_cf_accns),
            "matchedSubmissionsAccessions": len(matched_accns),
            "unresolvedAccessionsCount": len(unresolved_accns),
            "unresolvedAccessions": sorted(list(unresolved_accns)),
            "lineageStatus": "PARTIAL_HISTORICAL_COVERAGE",
            "explanation": (
                f"{len(unresolved_accns)} historical accessions precede the Submissions recent filing envelope "
                f"(covering 2020-2026) and reside in older submission partitions; explicitly retained as unresolved."
            ),
        }

        # Build provenance envelope holding production scoring
        revenue_fact = fundamentals.facts.get("revenue_ttm")
        latest_accn = revenue_fact.accession_number if revenue_fact else None
        latest_acc_dt = revenue_fact.acceptance_time if revenue_fact else None

        provenance = {
            "symbol": clean_sym,
            "cik": cik,
            "entity_name": binding.company_name,
            "source": "sec_edgar_offline",
            "sourceId": "sec_edgar",
            "evidenceStatus": "UNAVAILABLE",  # Production scoring held
            "isVerified": False,              # Production scoring held
            "isSynthetic": False,
            "isStale": False,
            "retrievalDate": retrieval_date,
            "researchCutoffDate": research_cutoff_date,
            "filingDate": fundamentals.latest_filing_date,
            "latestReportedPeriod": fundamentals.latest_reported_period,
            "reportingForm": fundamentals.operating_indicators.get("reporting_form"),
            "accessionNumber": latest_accn,
            "acceptanceDateTime": latest_acc_dt,
            "scoringHeld": True,
            "holdingCode": HOLDING_CODE_DEFAULT,
            "unresolvedLineage": unresolved_lineage,
            "disqualificationReason": (
                f"Filing-backed evidence for {clean_sym} verified offline from SEC EDGAR "
                f"(accession: {latest_accn}); production scoring and publication held under "
                f"{HOLDING_CODE_DEFAULT} pending formal release authorization."
            ),
            "profileLabel": "Offline Verified Filing Evidence (Scoring Held)",
        }

        return OfflineFundamentalsResult(
            symbol=clean_sym,
            cik=cik,
            company_name=binding.company_name,
            fundamentals=fundamentals,
            manifest=manifest_data,
            verified_hashes=verified_hashes,
            retrieval_date=retrieval_date,
            research_cutoff_date=research_cutoff_date,
            scoring_held=True,
            holding_code=HOLDING_CODE_DEFAULT,
            provenance=provenance,
            unresolved_lineage=unresolved_lineage,
        )

    def protect_against_runtime_masking(
        self,
        filing_result: Optional[OfflineFundamentalsResult],
        runtime_factor_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Anti-masking protection guard.
        Ensures that when filing-backed evidence exists in offline research custody,
        ungrounded runtime estimates (e.g. source: 'LIVE_RUNTIME' or synthetic scores)
        cannot mask, override, or masquerade as verified filing-backed evidence.
        Enforces that production scoring remains held under HOLD_SEC_001.
        """
        runtime_dict = runtime_factor_data if isinstance(runtime_factor_data, dict) else {}
        runtime_source = str(runtime_dict.get("source", "")).strip().lower()

        if filing_result is not None:
            # Filing-backed evidence is present
            revenue_fact = filing_result.fundamentals.facts.get("revenue_ttm")
            accn = revenue_fact.accession_number if revenue_fact else None
            acc_time = revenue_fact.acceptance_time if revenue_fact else None

            # If runtime factor data attempts to present ungrounded estimates
            is_runtime_estimate = runtime_source in ("live_runtime", "runtime_estimate", "synthetic")

            return {
                "symbol": filing_result.symbol,
                "hasFilingEvidence": True,
                "runtimeMaskingPrevented": is_runtime_estimate,
                "filingSource": "sec_edgar_offline",
                "runtimeSource": runtime_dict.get("source"),
                "latestFilingDate": filing_result.fundamentals.latest_filing_date,
                "latestAccession": accn,
                "acceptanceDateTime": acc_time,
                "scoringHeld": True,
                "holdingCode": HOLDING_CODE_DEFAULT,
                "status": "unavailable",
                "disqualificationReason": (
                    f"Filing-backed evidence for {filing_result.symbol} is present in offline research custody; "
                    f"ungrounded runtime estimates cannot mask filing facts. Production scoring remains held "
                    f"under {HOLDING_CODE_DEFAULT}."
                ),
                "retainedFactsCount": len(filing_result.fundamentals.facts),
            }

        # No filing evidence available
        return {
            "symbol": runtime_dict.get("symbol", "UNKNOWN"),
            "hasFilingEvidence": False,
            "runtimeMaskingPrevented": False,
            "scoringHeld": True,
            "holdingCode": HOLDING_CODE_DEFAULT,
            "status": "unavailable",
            "disqualificationReason": "No filing-backed evidence available; runtime estimates remain unverified.",
        }


def load_offline_fundamentals(
    symbol: str,
    raw_base_dir: Optional[Path] = None,
    research_cutoff_date: str = RESEARCH_CUTOFF_DATE_DEFAULT,
) -> OfflineFundamentalsResult:
    """Convenience function to load and verify offline fundamentals."""
    reader = OfflineFundamentalsReader(raw_base_dir=raw_base_dir)
    return reader.load_symbol(symbol=symbol, research_cutoff_date=research_cutoff_date)


def protect_against_runtime_masking(
    filing_result: Optional[OfflineFundamentalsResult],
    runtime_factor_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convenience function to enforce anti-masking protection."""
    reader = OfflineFundamentalsReader()
    return reader.protect_against_runtime_masking(filing_result, runtime_factor_data)
