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

    _co_filer_mappings: Optional[Dict[str, Dict[str, Any]]] = None

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

    @classmethod
    def _load_co_filer_mappings(cls) -> Dict[str, Dict[str, Any]]:
        """Loads authoritative co-filer provenance mappings from certified pre-boundary ledger."""
        if cls._co_filer_mappings is None:
            mappings = {}
            repo_root = Path(__file__).resolve().parents[3]
            ledger_path = repo_root / "docs" / "research" / "ETF_V2_860_AUTHORITY_CHAIN_CLOSURE_LEDGER.json"
            if ledger_path.exists():
                try:
                    with open(ledger_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for t in data.get("targets", []):
                        if t.get("symbol") in {"BUFH", "HOYY", "IOYY", "RTYY", "SMYY"}:
                            mappings[t["symbol"]] = {
                                "accession": t["base_accession"],
                                "form": t["base_form"],
                                "acceptance_timestamp": t["base_acceptance_timestamp"],
                                "primary_document": t.get("base_primary_document", ""),
                            }
                except Exception:
                    pass
            cls._co_filer_mappings = mappings
        return cls._co_filer_mappings

    def get_candidate_prospectuses(self, identity: EntityIdentity) -> List[FilingMetadata]:
        """Discovers pre-boundary statutory filing candidates for a given target entity."""
        candidates = []
        cik_clean = identity.cik.lstrip("0")
        sub_file = self.submissions_dir / f"CIK{cik_clean.zfill(10)}.json"

        chunks = []
        if sub_file.exists():
            with open(sub_file, "r", encoding="utf-8") as f:
                sub_data = json.load(f)

            recent = sub_data.get("filings", {}).get("recent", {})
            if recent:
                chunks.append(recent)

            # Lane B / AFOS: Traverse local cached historical submission chunks for target AFOS (CIK 0001592900)
            if identity.symbol == "AFOS" or cik_clean == "1592900":
                for f_info in sub_data.get("filings", {}).get("files", []):
                    chunk_name = f_info.get("name")
                    if chunk_name:
                        chunk_path = self.submissions_dir / chunk_name
                        if chunk_path.exists():
                            try:
                                with open(chunk_path, "r", encoding="utf-8") as cf:
                                    chunks.append(json.load(cf))
                            except Exception:
                                pass

        for chunk in chunks:
            accessions = chunk.get("accessionNumber", [])
            forms = chunk.get("form", [])
            filing_dates = chunk.get("filingDate", [])
            acceptance_datetimes = chunk.get("acceptanceDateTime", [])
            primary_doc_names = chunk.get("primaryDocument", [])
            report_dates = chunk.get("reportDate", [])

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

        # Lane B / BUFH, HOYY, IOYY, RTYY, SMYY: Co-filer / sponsor-CIK candidate resolution
        co_filers = self._load_co_filer_mappings()
        if identity.symbol in co_filers:
            co_info = co_filers[identity.symbol]
            acc = co_info["accession"]
            if not any(c.accession == acc for c in candidates):
                accept_time = co_info["acceptance_timestamp"]
                if accept_time <= self.boundary_iso:
                    file_path = self._prosp_by_acc.get(acc)
                    # Enforce strict local cache boundary: only expose if local file exists
                    if file_path and file_path.exists():
                        candidates.append(
                            FilingMetadata(
                                accession=acc,
                                form=co_info["form"],
                                filing_date=accept_time[:10],
                                acceptance_timestamp=accept_time,
                                document_filename=co_info["primary_document"],
                                file_path=file_path,
                                raw_sha256="",
                                report_period=None,
                            )
                        )

        # Deduplicate and sort by acceptance timestamp descending (most recent pre-boundary first)
        seen_accs = set()
        deduped = []
        for c in candidates:
            if c.accession not in seen_accs:
                seen_accs.add(c.accession)
                deduped.append(c)
        deduped.sort(key=lambda x: x.acceptance_timestamp, reverse=True)
        return deduped
