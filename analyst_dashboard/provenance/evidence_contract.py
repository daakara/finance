"""Canonical Evidence Provenance Contract & Governance Authority (ARX-CONFLUENCE-INTEGRITY-002).

Establishes formal classification and verification contracts for multi-factor evidence:
    VERIFIED
    UNVERIFIED
    STALE
    SYNTHETIC
    UNAVAILABLE
    NOT_APPLICABLE
    FAILED

Enforces non-negotiable invariants:
- Numerical presence does not imply verification (POPULATED != VERIFIED).
- Synthetic fixtures fail closed and cannot establish an actionable trade setup.
- Legitimate identical scores from independent authentic sources are never rejected solely for matching.
- Instrument-specific exemptions (ETFs corporate filings = NOT_APPLICABLE) are strictly preserved.
- Historical point-in-time evidence is never masqueraded as current live evidence.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple


PROVENANCE_SPEC_VERSION = "2.0.0-INTEGRITY-002"


class EvidenceStatus(str, Enum):
    """Authoritative evidence classifications for ARX investment decision pillars."""
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    STALE = "STALE"
    SYNTHETIC = "SYNTHETIC"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    FAILED = "FAILED"


class EvidenceProvenance:
    """Canonical evidence provenance record container."""
    def __init__(
        self,
        status: EvidenceStatus,
        source_id: str,
        as_of_date: Optional[str] = None,
        filing_date: Optional[str] = None,
        observed_at: Optional[str] = None,
        transformation_version: str = PROVENANCE_SPEC_VERSION,
        content_hash: Optional[str] = None,
        is_verified: bool = False,
        is_synthetic: bool = False,
        is_stale: bool = False,
        disqualification_reason: Optional[str] = None,
        profile_label: Optional[str] = None,
        symbol: Optional[str] = None,
        asset_class: Optional[str] = None,
    ):
        self.status = status
        self.source_id = source_id
        self.as_of_date = as_of_date
        self.filing_date = filing_date
        self.observed_at = observed_at
        self.transformation_version = transformation_version
        self.content_hash = content_hash
        self.is_verified = is_verified
        self.is_synthetic = is_synthetic
        self.is_stale = is_stale
        self.disqualification_reason = disqualification_reason
        self.profile_label = profile_label
        self.symbol = symbol
        self.asset_class = asset_class

    def to_dict(self) -> Dict[str, Any]:
        status_val = self.status.value if isinstance(self.status, EvidenceStatus) else str(self.status)
        return {
            "status": status_val,
            "evidenceStatus": status_val,
            "sourceId": self.source_id,
            "asOfDate": self.as_of_date,
            "filingDate": self.filing_date,
            "observedAt": self.observed_at,
            "transformationVersion": self.transformation_version,
            "provenanceSpecVersion": self.transformation_version,
            "symbol": self.symbol,
            "assetClass": self.asset_class,
            "contentHash": self.content_hash,
            "isVerified": self.is_verified,
            "isSynthetic": self.is_synthetic,
            "isStale": self.is_stale,
            "disqualificationReason": self.disqualification_reason,
            "profileLabel": self.profile_label,
        }


PROVENANCE_SPEC_VERSION = "2.0.0-INTEGRITY-002"

# Known synthetic fixture timestamps and signatures from legacy database seeding
KNOWN_SYNTHETIC_TIMESTAMPS = {
    "2026-11-09 12:42:30",
    "2026-11-09T12:42:30",
}

AUTHENTIC_SOURCES = {
    "sec_edgar",
    "sec_10k",
    "sec_10q",
    "sec_form4",
    "sec_13f",
    "financial_modeling_prep",
    "fmp",
    "alphavantage",
    "bloomberg",
    "factset",
}


def compute_evidence_hash(data: Dict[str, Any]) -> str:
    """Computes deterministic SHA256 hash of evidence payload."""
    try:
        clean_data = {
            k: v for k, v in data.items()
            if not k.startswith("_") and k not in ("hash", "content_hash")
        }
        serialized = json.dumps(clean_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()


def evaluate_factor_provenance(
    factor_snap: Optional[Dict[str, Any]],
    symbol: str,
    asset_class: str = "EQUITY",
    max_staleness_days: int = 450,
) -> Dict[str, Any]:
    """Evaluates the provenance, authenticity, and verification status of a fundamental factor snapshot.

    Args:
        factor_snap: Raw dictionary returned from market_db.get_factor_snapshot(symbol).
        symbol: Asset ticker symbol.
        asset_class: 'EQUITY', 'ETF', 'ADR', etc.
        max_staleness_days: Maximum calendar days allowed between filing/as_of and current date.

    Returns:
        Structured provenance dictionary with canonical EvidenceStatus and audit metadata.
    """
    clean_sym = symbol.upper().strip()
    clean_asset = asset_class.upper().strip() if asset_class else "EQUITY"

    # ETF Exemption: Corporate financial filings are legally NOT_APPLICABLE for fund units
    if clean_asset == "ETF" or clean_sym in {"SPY", "QQQ", "IWM", "XLK", "DIA", "SOXX", "SMH", "XLF", "XLE", "VTI", "VOO"}:
        return {
            "status": EvidenceStatus.NOT_APPLICABLE.value,
            "evidenceStatus": EvidenceStatus.NOT_APPLICABLE.value,
            "sourceId": "SECURITY_MASTER_EXEMPTION",
            "asOfDate": None,
            "filingDate": None,
            "observedAt": datetime.now(timezone.utc).isoformat(),
            "transformationVersion": PROVENANCE_SPEC_VERSION,
            "provenanceSpecVersion": PROVENANCE_SPEC_VERSION,
            "symbol": clean_sym,
            "assetClass": clean_asset,
            "contentHash": None,
            "isVerified": True,
            "isSynthetic": False,
            "isStale": False,
            "disqualificationReason": None,
            "profileLabel": "ETF Portfolio Holdings (10-K/10-Q Corporate Filings Not Applicable)",
        }

    # Missing Record
    if not factor_snap or not isinstance(factor_snap, dict):
        return {
            "status": EvidenceStatus.UNAVAILABLE.value,
            "evidenceStatus": EvidenceStatus.UNAVAILABLE.value,
            "sourceId": "NONE",
            "asOfDate": None,
            "filingDate": None,
            "observedAt": datetime.now(timezone.utc).isoformat(),
            "transformationVersion": PROVENANCE_SPEC_VERSION,
            "provenanceSpecVersion": PROVENANCE_SPEC_VERSION,
            "symbol": clean_sym,
            "assetClass": clean_asset,
            "contentHash": None,
            "isVerified": False,
            "isSynthetic": False,
            "isStale": False,
            "disqualificationReason": f"Fundamental financial records unavailable for {clean_sym}.",
            "profileLabel": "Fundamental Disclosures Unavailable",
        }

    source_raw = str(factor_snap.get("source") or "").lower().strip()
    created_at = str(factor_snap.get("created_at") or "").strip()
    as_of = factor_snap.get("as_of_date")
    filing = factor_snap.get("filing_date")
    available_from = factor_snap.get("available_from")
    pit_status = factor_snap.get("point_in_time_status")

    # 1. Detect Synthetic Template Fixtures
    # A fixture is identified by missing authentic source attribution AND absence of any filing dates/accession numbers
    is_unknown_source = (source_raw in ("unknown", "synthetic", "template", "fixture", "mock", ""))
    dates_missing = (as_of is None and filing is None and available_from is None)
    is_known_template_timestamp = any(created_at.startswith(ts) for ts in KNOWN_SYNTHETIC_TIMESTAMPS)

    if (is_unknown_source and dates_missing) or is_known_template_timestamp:
        return {
            "status": EvidenceStatus.SYNTHETIC.value,
            "evidenceStatus": EvidenceStatus.SYNTHETIC.value,
            "sourceId": f"SYNTHETIC_FIXTURE_{source_raw or 'UNKNOWN'}",
            "asOfDate": as_of,
            "filingDate": filing,
            "observedAt": created_at or datetime.now(timezone.utc).isoformat(),
            "transformationVersion": PROVENANCE_SPEC_VERSION,
            "provenanceSpecVersion": PROVENANCE_SPEC_VERSION,
            "symbol": clean_sym,
            "assetClass": clean_asset,
            "contentHash": compute_evidence_hash(factor_snap),
            "isVerified": False,
            "isSynthetic": True,
            "isStale": False,
            "disqualificationReason": "Synthetic template factors: Lacks audited SEC filing dates, accession lineage, and authentic source attribution.",
            "profileLabel": "Synthetic / Unverified Template Data",
        }

    # 2. Evidence freshness and source authority must be demonstrated.
    # Source labels, timestamps alone, or invalid/future dates do not prove
    # point-in-time authenticity; never silently ignore malformed evidence.
    eval_date_str = filing or as_of or available_from
    is_stale = False
    valid_evidence_date = False
    invalid_evidence_date = False
    if eval_date_str:
        try:
            clean_date = str(eval_date_str)[:10]
            dt = datetime.strptime(clean_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta_days = (now - dt).days
            valid_evidence_date = (0 <= delta_days <= max_staleness_days)
            is_stale = delta_days > max_staleness_days
            invalid_evidence_date = delta_days < 0
        except (ValueError, TypeError, OverflowError):
            invalid_evidence_date = True

    if is_stale:
        return {
            "status": EvidenceStatus.STALE.value,
            "evidenceStatus": EvidenceStatus.STALE.value,
            "sourceId": source_raw or "UNVERIFIED_SOURCE",
            "asOfDate": as_of,
            "filingDate": filing,
            "observedAt": created_at or datetime.now(timezone.utc).isoformat(),
            "transformationVersion": PROVENANCE_SPEC_VERSION,
            "provenanceSpecVersion": PROVENANCE_SPEC_VERSION,
            "symbol": clean_sym,
            "assetClass": clean_asset,
            "contentHash": compute_evidence_hash(factor_snap),
            "isVerified": False,
            "isSynthetic": False,
            "isStale": True,
            "disqualificationReason": f"Fundamental filing records are stale (> {max_staleness_days} days without updated financial report).",
            "profileLabel": "Stale Fundamental Disclosures",
        }

    # 3. Check for Authenticity. Require BOTH a recognized provider and
    # a valid non-future filing/as-of date. A self-asserted source label is
    # not independent proof of authenticated upstream provenance.
    if source_raw in AUTHENTIC_SOURCES and valid_evidence_date and not invalid_evidence_date:
        return {
            "status": EvidenceStatus.VERIFIED.value,
            "evidenceStatus": EvidenceStatus.VERIFIED.value,
            "sourceId": source_raw or "AUDITED_SEC_FILING",
            "asOfDate": as_of,
            "filingDate": filing,
            "observedAt": created_at or datetime.now(timezone.utc).isoformat(),
            "transformationVersion": PROVENANCE_SPEC_VERSION,
            "provenanceSpecVersion": PROVENANCE_SPEC_VERSION,
            "symbol": clean_sym,
            "assetClass": clean_asset,
            "contentHash": compute_evidence_hash(factor_snap),
            "isVerified": True,
            "isSynthetic": False,
            "isStale": False,
            "disqualificationReason": None,
            "profileLabel": "Source-Attributed Fundamental Evidence",
        }

    # Fallback to UNVERIFIED if numerical fields exist but source lineage cannot be verified
    return {
        "status": EvidenceStatus.UNVERIFIED.value,
        "evidenceStatus": EvidenceStatus.UNVERIFIED.value,
        "sourceId": source_raw or "UNVERIFIED_SOURCE",
        "asOfDate": as_of,
        "filingDate": filing,
        "observedAt": created_at or datetime.now(timezone.utc).isoformat(),
        "transformationVersion": PROVENANCE_SPEC_VERSION,
        "provenanceSpecVersion": PROVENANCE_SPEC_VERSION,
        "symbol": clean_sym,
        "assetClass": clean_asset,
        "contentHash": compute_evidence_hash(factor_snap),
        "isVerified": False,
        "isSynthetic": False,
        "isStale": False,
        "disqualificationReason": f"Fundamental factor data for {clean_sym} is unverified by an authorized regulator or exchange source.",
        "profileLabel": "Unverified Fundamental Disclosures",
    }
