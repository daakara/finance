"""Evidence for the recurring cohort's bounded SEC submissions polls."""

import hashlib
import json
import re
from datetime import date, datetime, timezone


def evidence_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inventory_from_catalog(body: bytes, cik: str, start: datetime, cutoff: datetime):
    """Return eligible recent rows and any gap that prevents interval accounting."""
    try:
        data = json.loads(body)
        if str(data["cik"]).zfill(10) != cik:
            raise RuntimeError("SEC_SUBMISSIONS_ISSUER_MISMATCH")
        filings = data["filings"]
        recent = filings["recent"]
        columns = [recent[field] for field in (
            "form", "accessionNumber", "filingDate", "acceptanceDateTime", "primaryDocument"
        )]
        if not all(isinstance(column, list) for column in columns):
            raise ValueError("invalid recent columns")
        pages = filings["files"]
        if not isinstance(pages, list):
            raise ValueError("invalid historical page inventory")
        # EDGAR catalog page bounds are dates, so a page ending on the start
        # date is conservatively treated as potentially overlapping.
        dates = [date.fromisoformat(page["filingTo"]) for page in pages]
        gaps = []
        if any(end >= start.date() for end in dates):
            gaps.append("HISTORICAL_PAGE_MAY_OVERLAP")

        earliest = None
        inventory = {}
        for form, accession, filing_date, accepted, doc in zip(*columns, strict=True):
            timestamp = datetime.fromisoformat(accepted)
            if timestamp.tzinfo is None:
                raise ValueError("acceptance timestamp missing timezone")
            timestamp = timestamp.astimezone(timezone.utc)
            earliest = timestamp if earliest is None else min(earliest, timestamp)
            if form not in ("4", "4/A", "4A") or not start <= timestamp <= cutoff:
                continue
            if not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession):
                raise ValueError("invalid accession")
            row = {
                "accession": accession,
                "form": "4/A" if form == "4A" else form,
                "filing_date": date.fromisoformat(filing_date).isoformat(),
                "accepted_at": timestamp.isoformat(),
                "primary_document": doc,
            }
            if inventory.setdefault(accession, row) != row:
                raise ValueError("conflicting duplicate accession")
        if earliest is None or earliest > start:
            gaps.append("RECENT_CATALOG_START_UNVERIFIED")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("SEC_SUBMISSIONS_SCHEMA_INVALID") from exc
    return sorted(inventory.values(), key=lambda row: (row["accepted_at"], row["accession"])), gaps


def build_issuer_receipt(observation, stored_hashes, fetched_hashes, unresolved, previous_watermark):
    """Bind the retained catalog, exact accession dispositions and poll checkpoint."""
    gaps = list(observation["catalog_gaps"])
    entries = []
    for row in observation["inventory"]:
        accession = row["accession"]
        digest = fetched_hashes.get(accession) or stored_hashes.get(accession)
        if accession in unresolved:
            disposition = "AMENDMENT_UNRESOLVED"
        elif not digest:
            disposition = "DEFERRED"
        else:
            disposition = "INGESTED" if accession in fetched_hashes else "ALREADY_ACCOUNTED"
        if disposition in ("DEFERRED", "AMENDMENT_UNRESOLVED"):
            gaps.append(f"{disposition}:{accession}")
        entries.append({**row, "disposition": disposition, "source_sha256": digest})
    receipt = {
        **observation,
        "inventory": entries,
        "gaps": gaps,
        # Reconcile the full one-day grant horizon on every poll so a filing
        # that appeared late cannot be skipped by the last checkpoint.
        "previous_watermark": previous_watermark,
        "accounted_through": previous_watermark if gaps else observation["cutoff"],
        "status": "PARTIAL" if gaps else "CATALOG_ACCOUNTED",
    }
    return {**receipt, "sha256": evidence_sha256(receipt)}
