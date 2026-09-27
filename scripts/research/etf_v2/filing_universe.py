"""
scripts/research/etf_v2/filing_universe.py

Filing Universe Authority for Pipeline V2.
Enforces the temporal boundary rule:
SEC_ACCEPTANCE_TIMESTAMP <= 2026-09-24T23:59:59Z.
Strictly distinguishes filing date, acceptance timestamp, report period, and effective date.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from .models import EntityIdentity, FilingMetadata

SNAPSHOT_BOUNDARY_ISO = "2026-09-24T23:59:59Z"


class FilingUniverse:
    """Manages pre-boundary regulatory filing universe for target entities."""

    def __init__(
        self,
        submissions_dir: Path,
        prospectus_dir: Path,
        boundary_iso: str = SNAPSHOT_BOUNDARY_ISO,
    ):
        self.submissions_dir = submissions_dir
        self.prospectus_dir = prospectus_dir
        self.boundary_iso = boundary_iso
        self._prosp_by_acc: Dict[str, Path] = {}
        if self.prospectus_dir.exists():
            for p in self.prospectus_dir.glob("*_*"):
                acc = p.name.split("_", 1)[0]
                self._prosp_by_acc[acc] = p

    def get_candidate_prospectuses(self, identity: EntityIdentity) -> List[FilingMetadata]:
        """Discovers pre-boundary statutory filing candidates for a given target entity."""
        candidates = []
        cik_clean = identity.cik.lstrip("0")
        sub_file = self.submissions_dir / f"CIK{cik_clean.zfill(10)}.json"

        if not sub_file.exists():
            return candidates

        with open(sub_file, "r", encoding="utf-8") as f:
            sub_data = json.load(f)

        recent = sub_data.get("filings", {}).get("recent", {})
        accessions = recent.get("accessionNumber", [])
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        acceptance_datetimes = recent.get("acceptanceDateTime", [])
        primary_doc_names = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        n_filings = len(accessions)
        for i in range(n_filings):
            acc = accessions[i]
            form = forms[i]
            accept_time = acceptance_datetimes[i]
            filing_date = filing_dates[i]
            doc_name = primary_doc_names[i]
            report_period = report_dates[i] if i < len(report_dates) else None

            # 1. Enforce Temporal Boundary Invariant
            if accept_time > self.boundary_iso:
                continue

            # 2. Filter for statutory prospectus form types
            if form not in {"497K", "485BPOS", "485APOS", "497"}:
                continue

            # Locate local prospectus cache file in O(1)
            file_path = self._prosp_by_acc.get(acc)
            if not file_path:
                continue

            candidates.append(
                FilingMetadata(
                    accession=acc,
                    form=form,
                    filing_date=filing_date,
                    acceptance_timestamp=accept_time,
                    document_filename=doc_name,
                    file_path=file_path,
                    raw_sha256="",
                    report_period=report_period,
                )
            )

        # Sort by acceptance timestamp descending (most recent pre-boundary first)
        candidates.sort(key=lambda x: x.acceptance_timestamp, reverse=True)
        return candidates
