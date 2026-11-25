#!/usr/bin/env python3
"""
ARX Terminal — Smart Money Form 4 Cohort Pilot Receipt & Reconciliation Exporter
Issue #165 / Trial ARX-TRIAL-10-ISSUER-001

Exports the stored worker reconciliation receipt and derives an independent
reconciliation directly from underlying PostgreSQL evidence in
finance_smart_money_cohort_001.

Crucial Governance Invariant:
DO NOT copy expected totals or counts from the stored worker receipt.
All issuer comparison fields (accounted_count, newly_ingested_count, gaps,
accounted_through) must be independently computed from sm_audit_events,
form4_raw_filings, sm_source_availability_ledger, and sm_amendments.
If any field cannot be independently established, report it as pending.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Cohort Universe: UB-US-EQUITIES-20261025-001 (10 Issuers)
COHORT_CIKS: Mapping[str, str] = {
    "NVDA": "0001045810",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "TSLA": "0001318605",
    "BRK-B": "0001067983",
    "JPM": "0000019617",
    "V": "0001403161",
}

DEFAULT_TRIAL_ID = "ARX-TRIAL-10-ISSUER-001"
DEFAULT_RELEASE_SHA = "1063a10fed1ed312feadeb671980a642e757e601"
WORKER_COHORT_PILOT_SLOTS = (
    "2026-11-30:SLOT_1:0915",
    "2026-11-30:SLOT_2:1300",
    "2026-11-30:SLOT_3:1630",
)


def evidence_sha256(payload: Any) -> str:
    """Canonical compact JSON SHA256 digest."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inventory_from_catalog(
    body: bytes,
    cik: str,
    start: datetime,
    cutoff: datetime,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Pure SEC submissions catalog parser matching candidate 1063a10f.
    Extracts eligible Form 4/4A rows and flags catalog gaps (e.g. historical overlap).
    """
    try:
        data = json.loads(body)
        if str(data.get("cik", "")).zfill(10) != cik:
            raise RuntimeError(f"SEC_SUBMISSIONS_ISSUER_MISMATCH: expected CIK {cik}, got {data.get('cik')}")
        filings = data["filings"]
        recent = filings["recent"]
        columns = [recent[field] for field in (
            "form", "accessionNumber", "filingDate", "acceptanceDateTime", "primaryDocument"
        )]
        if not all(isinstance(column, list) for column in columns):
            raise ValueError("invalid recent columns")
        pages = filings.get("files", [])
        if not isinstance(pages, list):
            raise ValueError("invalid historical page inventory")

        dates = [date.fromisoformat(page["filingTo"]) for page in pages if "filingTo" in page]
        gaps: List[str] = []
        if any(end >= start.date() for end in dates):
            gaps.append("HISTORICAL_PAGE_MAY_OVERLAP")

        earliest = None
        inventory: Dict[str, Dict[str, Any]] = {}
        for form, accession, filing_date, accepted, doc in zip(*columns, strict=True):
            timestamp = datetime.fromisoformat(accepted.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("acceptance timestamp missing timezone")
            timestamp = timestamp.astimezone(timezone.utc)
            earliest = timestamp if earliest is None else min(earliest, timestamp)
            if form not in ("4", "4/A", "4A") or not (start <= timestamp <= cutoff):
                continue
            if not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession):
                raise ValueError(f"invalid accession format: {accession}")
            row = {
                "accession": accession,
                "form": "4/A" if form == "4A" else form,
                "filing_date": date.fromisoformat(filing_date).isoformat(),
                "accepted_at": timestamp.isoformat(),
                "primary_document": doc,
            }
            if inventory.setdefault(accession, row) != row:
                raise ValueError(f"conflicting duplicate accession: {accession}")

        if earliest is None or earliest > start:
            gaps.append("RECENT_CATALOG_START_UNVERIFIED")

    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"SEC_SUBMISSIONS_SCHEMA_INVALID: {exc}") from exc

    sorted_inv = sorted(inventory.values(), key=lambda r: (r["accepted_at"], r["accession"]))
    return sorted_inv, gaps


def extract_stored_worker_receipt(
    cur: Any,
    slot_id: str,
    trial_id: str = DEFAULT_TRIAL_ID,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves the immutable worker reconciliation receipt stored in sm_reconciliation_receipts.
    Returns the raw database record serialized as a dictionary.
    """
    query = """
        SELECT
            receipt_id,
            interval_start,
            interval_end,
            certified_watermark,
            receipt_version,
            supersedes_receipt_id,
            expected_count,
            accounted_count,
            missing_count,
            material_gap_count,
            coverage_status,
            failure_reasons,
            reconciliation_hash,
            release_sha,
            created_at
        FROM sm_reconciliation_receipts
        WHERE (
            failure_reasons->'cohort_evidence'->>'slot_id' = %s
            OR failure_reasons->>'slot_id' = %s
        )
        AND (
            failure_reasons->'cohort_evidence'->>'trial_id' = %s
            OR failure_reasons->>'trial_id' = %s
        )
        ORDER BY created_at DESC
        LIMIT 1;
    """
    cur.execute(query, (slot_id, slot_id, trial_id, trial_id))
    row = cur.fetchone()
    if not row:
        return None

    if isinstance(row, dict):
        record = dict(row)
    else:
        # Tuple from standard cursor
        columns = [
            "receipt_id", "interval_start", "interval_end", "certified_watermark",
            "receipt_version", "supersedes_receipt_id", "expected_count", "accounted_count",
            "missing_count", "material_gap_count", "coverage_status", "failure_reasons",
            "reconciliation_hash", "release_sha", "created_at",
        ]
        record = dict(zip(columns, row))

    # Format UUID and datetimes for JSON serialization
    for k, v in record.items():
        if isinstance(v, datetime):
            record[k] = v.isoformat().replace("+00:00", "Z")
        elif isinstance(v, date):
            record[k] = v.isoformat()
        elif hasattr(v, "hex"):  # UUID
            record[k] = str(v)

    # Top-level slot_id and trial_id convenience
    record["slot_id"] = slot_id
    record["trial_id"] = trial_id
    return record


def derive_independent_reconciliation(
    cur: Any,
    slot_id: str,
    trial_id: str = DEFAULT_TRIAL_ID,
    receipt_id: Optional[str] = None,
    slot_created_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Derives the independent reconciliation export directly from underlying evidence tables.
    CRUCIAL: Does NOT copy expected totals or counts from the stored worker receipt.

    For each of the 10 cohort issuers:
    1. Looks up the SEC_COHORT_CATALOG_OBSERVED audit event in sm_audit_events.
    2. Validates event_hash and catalog_sha256 digest.
    3. Recomputes raw accession inventory and catalog gaps via inventory_from_catalog().
    4. Looks up previous watermark from sm_reconciliation_receipts prior to this receipt.
    5. Checks form4_raw_filings and sm_amendments to determine accession dispositions:
       - INGESTED: present in form4_raw_filings with created_at during this slot's window
       - ALREADY_ACCOUNTED: present in form4_raw_filings with created_at prior to this slot
       - AMENDMENT_UNRESOLVED: unresolved amendment in sm_amendments
       - DEFERRED: not present in form4_raw_filings
    6. If any field or audit event cannot be established, reports comparison fields as None/pending.
    """
    reconciled_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    issuer_exports: Dict[str, Dict[str, Any]] = {}

    for ticker, cik in COHORT_CIKS.items():
        # Step 1: Query audit event for this issuer in this slot
        audit_query = """
            SELECT event_id, payload, event_hash, created_at
            FROM sm_audit_events
            WHERE event_type = 'SEC_COHORT_CATALOG_OBSERVED'
              AND subject_id = %s
              AND payload->>'slot_id' = %s
              AND payload->>'ticker' = %s
            ORDER BY created_at DESC
            LIMIT 1;
        """
        cur.execute(audit_query, (trial_id, slot_id, ticker))
        event_row = cur.fetchone()

        if not event_row:
            # Audit event missing: report comparison fields as None (pending)
            issuer_exports[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": None,
                "newly_ingested_count": None,
                "gaps": None,
                "accounted_through": None,
                "state": "NOT_POLLED",
                "verified": False,
            }
            continue

        if isinstance(event_row, dict):
            event_id = event_row["event_id"]
            payload = event_row["payload"]
            event_hash = event_row["event_hash"]
            event_created_at = event_row["created_at"]
        else:
            event_id, payload, event_hash, event_created_at = event_row

        if isinstance(payload, str):
            payload = json.loads(payload)

        # Step 2: Validate event hash and catalog hash
        computed_event_hash = evidence_sha256(payload)
        if computed_event_hash != event_hash:
            issuer_exports[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": None,
                "newly_ingested_count": None,
                "gaps": None,
                "accounted_through": None,
                "state": "READ_ERROR",
                "verified": False,
            }
            continue

        raw_body_b64 = payload.get("body_base64")
        if not raw_body_b64:
            issuer_exports[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": None,
                "newly_ingested_count": None,
                "gaps": None,
                "accounted_through": None,
                "state": "READ_ERROR",
                "verified": False,
            }
            continue

        body = base64.b64decode(raw_body_b64)
        if hashlib.sha256(body).hexdigest() != payload.get("catalog_sha256"):
            issuer_exports[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": None,
                "newly_ingested_count": None,
                "gaps": None,
                "accounted_through": None,
                "state": "READ_ERROR",
                "verified": False,
            }
            continue

        # Step 3: Parse catalog independently
        start = datetime.fromisoformat(payload["interval_start"].replace("Z", "+00:00"))
        cutoff = datetime.fromisoformat(payload["cutoff"].replace("Z", "+00:00"))
        try:
            inventory, catalog_gaps = inventory_from_catalog(body, cik, start, cutoff)
        except Exception:
            issuer_exports[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": None,
                "newly_ingested_count": None,
                "gaps": None,
                "accounted_through": None,
                "state": "READ_ERROR",
                "verified": False,
            }
            continue

        # Step 4: Lookup prior watermark
        watermark_query = """
            SELECT failure_reasons -> 'cohort_evidence' -> 'issuers' -> %s ->> 'accounted_through' AS watermark
            FROM sm_reconciliation_receipts
            WHERE (failure_reasons ->> 'trial_id' = %s OR failure_reasons -> 'cohort_evidence' ->> 'trial_id' = %s)
              AND failure_reasons -> 'cohort_evidence' -> 'issuers' -> %s ->> 'cik' = %s
              AND (supersedes_receipt_id IS NULL OR supersedes_receipt_id != %s)
              AND (%s IS NULL OR receipt_id != %s::uuid)
            ORDER BY interval_end DESC, created_at DESC
            LIMIT 1;
        """
        cur.execute(watermark_query, (ticker, trial_id, trial_id, ticker, cik, receipt_id or "", receipt_id, receipt_id or ""))
        w_row = cur.fetchone()
        previous_watermark = (w_row["watermark"] if isinstance(w_row, dict) else w_row[0]) if w_row else None

        # Step 5: Check raw filings and amendments for each accession
        gaps = list(catalog_gaps)
        accounted_count = 0
        newly_ingested_count = 0

        for item in inventory:
            acc = item["accession"]

            # Check amendments first
            cur.execute(
                "SELECT resolution_status FROM sm_amendments WHERE amendment_accession = %s LIMIT 1;",
                (acc,),
            )
            amend_row = cur.fetchone()
            res_status = (amend_row["resolution_status"] if isinstance(amend_row, dict) else amend_row[0]) if amend_row else None
            is_unresolved = res_status and res_status not in (
                "RESOLVED", "FULL_REPLACEMENT", "PARTIAL_CORRECTION", "ADDITIONAL_TRANSACTION"
            )

            if is_unresolved:
                gaps.append(f"AMENDMENT_UNRESOLVED:{acc}")
                continue

            # Check form4_raw_filings
            cur.execute(
                """SELECT f.content_hash, f.created_at
                   FROM form4_raw_filings f
                   JOIN sm_source_availability_ledger av USING (accession_number)
                   WHERE f.accession_number = %s;""",
                (acc,),
            )
            raw_row = cur.fetchone()
            if not raw_row:
                gaps.append(f"DEFERRED:{acc}")
                continue

            # Accession exists in raw filings
            raw_created_at = raw_row["created_at"] if isinstance(raw_row, dict) else raw_row[1]
            accounted_count += 1

            # Determine whether newly ingested in this slot or already accounted
            # Filings created during or after this slot's audit event are newly ingested
            if event_created_at and raw_created_at >= event_created_at:
                newly_ingested_count += 1

        cutoff_iso = cutoff.isoformat().replace("+00:00", "Z")
        if previous_watermark and "+00:00" in previous_watermark:
            previous_watermark = previous_watermark.replace("+00:00", "Z")

        # Checkpoint: previous_watermark if gaps else cutoff
        accounted_through = previous_watermark if gaps else cutoff_iso

        # Determine evidence state
        if gaps:
            state = "INCOMPLETE"
        elif newly_ingested_count > 0 or accounted_count > 0:
            state = "FILINGS_FOUND"
        else:
            state = "POLLED_QUIET"

        issuer_exports[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "accounted_count": accounted_count,
            "newly_ingested_count": newly_ingested_count,
            "gaps": sorted(gaps),
            "accounted_through": accounted_through,
            "cutoff": cutoff_iso,
            "state": state,
            "verified": True,
        }

    export_payload: Dict[str, Any] = {
        "export_id": f"exp-indep-{trial_id}-{slot_id.replace(':', '_')}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "receipt_id": receipt_id,
        "trial_id": trial_id,
        "slot_id": slot_id,
        "reconciled_at": reconciled_at,
        "reconciled_by": "arx-independent-evidence-reconciler",
        "issuers": issuer_exports,
    }
    return export_payload


def export_slot_evidence(
    dsn_or_cur: Any,
    slot_id: str,
    output_dir: str = "exports",
    trial_id: str = DEFAULT_TRIAL_ID,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Exports both stored worker receipt and independently reconciled export for a slot.
    Writes:
      - <output_dir>/receipt_<slot_slug>.json
      - <output_dir>/independent_reconciliation_<slot_slug>.json
    """
    is_cursor = hasattr(dsn_or_cur, "execute")
    conn = None

    if is_cursor:
        cur = dsn_or_cur
    else:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(dsn_or_cur)
        cur = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # 1. Fetch worker receipt
        stored_receipt = extract_stored_worker_receipt(cur, slot_id=slot_id, trial_id=trial_id)
        if not stored_receipt:
            raise RuntimeError(f"RECEIPT_NOT_FOUND: No stored receipt found for slot '{slot_id}' and trial '{trial_id}'")

        receipt_id = stored_receipt.get("receipt_id")
        slot_created_at = stored_receipt.get("created_at")

        # 2. Derive independent reconciliation without copying counts
        independent_export = derive_independent_reconciliation(
            cur,
            slot_id=slot_id,
            trial_id=trial_id,
            receipt_id=receipt_id,
            slot_created_at=datetime.fromisoformat(slot_created_at.replace("Z", "+00:00")) if slot_created_at else None,
        )

        os.makedirs(output_dir, exist_ok=True)
        slot_slug = slot_id.replace(":", "_").replace("-", "")

        receipt_path = os.path.join(output_dir, f"receipt_{slot_slug}.json")
        indep_path = os.path.join(output_dir, f"independent_reconciliation_{slot_slug}.json")

        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(stored_receipt, f, indent=2, sort_keys=True)

        with open(indep_path, "w", encoding="utf-8") as f:
            json.dump(independent_export, f, indent=2, sort_keys=True)

        return stored_receipt, independent_export

    finally:
        if conn:
            conn.close()


class ComparisonReport:
    """Structured report of reconciliation comparison between worker receipt and independent export."""

    def __init__(
        self,
        status: str,  # "RECONCILED", "PENDING", "DISCREPANCY"
        discrepancies: List[Dict[str, Any]],
        pending_fields: List[Dict[str, Any]],
        binding_errors: List[Dict[str, Any]],
    ):
        self.status = status
        self.discrepancies = discrepancies
        self.pending_fields = pending_fields
        self.binding_errors = binding_errors

    @property
    def is_reconciled(self) -> bool:
        return self.status == "RECONCILED"

    @property
    def is_pending(self) -> bool:
        return self.status == "PENDING"

    @property
    def is_discrepancy(self) -> bool:
        return self.status == "DISCREPANCY"

    def format_cli_output(self) -> str:
        lines = []
        if self.status == "RECONCILED":
            lines.append("RECONCILED: All 10 cohort issuer rows match on trial, receipt, and dated slot.")
            lines.append("Zero discrepancies detected across all comparison fields.")
        elif self.status == "PENDING":
            lines.append("PENDING: Reconciliation cannot be established because evidence is pending or incomplete.")
            lines.append(f"Pending fields ({len(self.pending_fields)}):")
            for p in self.pending_fields:
                lines.append(f"  - [{p['ticker']}] field '{p['field']}' is pending / null")
        elif self.status == "DISCREPANCY":
            lines.append("DISCREPANCY: Export identities differ or known values disagree.")
            if self.binding_errors:
                lines.append(f"Identity / Binding Errors ({len(self.binding_errors)}):")
                for err in self.binding_errors:
                    lines.append(
                        f"  - [{err['ticker']}] field '{err['field']}': expected {err['expected']!r}, actual {err['actual']!r}"
                    )
            if self.discrepancies:
                lines.append(f"Value Discrepancies ({len(self.discrepancies)}):")
                for d in self.discrepancies:
                    lines.append(
                        f"  - [{d['ticker']}] field '{d['field']}': expected {d['expected']!r}, actual {d['actual']!r}"
                    )
            if self.pending_fields:
                lines.append(f"Pending Fields Reported Alongside Discrepancies ({len(self.pending_fields)}):")
                for p in self.pending_fields:
                    lines.append(f"  - [{p['ticker']}] field '{p['field']}' is pending / null")

        return "\n".join(lines)

    # For sequence-like backwards compatibility:
    def __iter__(self):
        all_items = self.binding_errors + self.discrepancies
        return iter(all_items)

    def __len__(self):
        return len(self.binding_errors) + len(self.discrepancies)

    def __getitem__(self, item):
        all_items = self.binding_errors + self.discrepancies
        return all_items[item]

    def __eq__(self, other):
        all_items = self.binding_errors + self.discrepancies
        if isinstance(other, list):
            return all_items == other
        if isinstance(other, ComparisonReport):
            return (
                self.status == other.status
                and self.discrepancies == other.discrepancies
                and self.pending_fields == other.pending_fields
                and self.binding_errors == other.binding_errors
            )
        return False

    def __repr__(self):
        return (
            f"<ComparisonReport status={self.status} "
            f"discrepancies={len(self.discrepancies)} binding_errors={len(self.binding_errors)} "
            f"pending={len(self.pending_fields)}>"
        )


def reconcile_and_compare_local(
    stored_receipt: Dict[str, Any],
    independent_export: Dict[str, Any],
) -> ComparisonReport:
    """
    Compares all ten cohort issuer rows between the stored receipt and independent reconciliation.
    Adheres strictly to the ARX Issue #165 qualification contract:
    - RECONCILED: only when identities match and all 10 issuers have complete, matching comparison fields.
    - PENDING: when any required issuer or comparison field is missing or null.
    - DISCREPANCY: when export identities differ or a known value disagrees.
      Reports pending fields alongside discrepancies when both occur.
    """
    cohort_evidence = (
        stored_receipt.get("cohort_evidence")
        or stored_receipt.get("failure_reasons", {}).get("cohort_evidence", {})
    )
    worker_issuers = cohort_evidence.get("issuers", {})
    export_issuers = independent_export.get("issuers", {})

    binding_errors: List[Dict[str, Any]] = []
    discrepancies: List[Dict[str, Any]] = []
    pending_fields: List[Dict[str, Any]] = []

    # 1. Identity / Binding checks (trial_id, receipt_id, slot_id)
    worker_trial_id = (
        stored_receipt.get("trial_id")
        or stored_receipt.get("failure_reasons", {}).get("cohort_evidence", {}).get("trial_id")
        or stored_receipt.get("failure_reasons", {}).get("trial_id")
    )
    worker_receipt_id = stored_receipt.get("receipt_id")
    worker_slot_id = (
        stored_receipt.get("slot_id")
        or stored_receipt.get("failure_reasons", {}).get("cohort_evidence", {}).get("slot_id")
        or stored_receipt.get("failure_reasons", {}).get("slot_id")
    )

    export_trial_id = independent_export.get("trial_id")
    export_receipt_id = independent_export.get("receipt_id")
    export_slot_id = independent_export.get("slot_id")

    if export_trial_id is not None and worker_trial_id is not None and export_trial_id != worker_trial_id:
        binding_errors.append({
            "ticker": "GLOBAL",
            "field": "trial_id",
            "expected": export_trial_id,
            "actual": worker_trial_id,
            "severity": "ERROR",
        })
    elif export_trial_id is None and worker_trial_id is not None:
        binding_errors.append({
            "ticker": "GLOBAL",
            "field": "trial_id",
            "expected": None,
            "actual": worker_trial_id,
            "severity": "ERROR",
        })

    if export_receipt_id is not None and worker_receipt_id is not None and export_receipt_id != worker_receipt_id:
        binding_errors.append({
            "ticker": "GLOBAL",
            "field": "receipt_id",
            "expected": export_receipt_id,
            "actual": worker_receipt_id,
            "severity": "ERROR",
        })
    elif export_receipt_id is None and worker_receipt_id is not None:
        binding_errors.append({
            "ticker": "GLOBAL",
            "field": "receipt_id",
            "expected": None,
            "actual": worker_receipt_id,
            "severity": "ERROR",
        })

    if export_slot_id is not None and worker_slot_id is not None and export_slot_id != worker_slot_id:
        binding_errors.append({
            "ticker": "GLOBAL",
            "field": "slot_id",
            "expected": export_slot_id,
            "actual": worker_slot_id,
            "severity": "ERROR",
        })
    elif export_slot_id is None and worker_slot_id is not None:
        binding_errors.append({
            "ticker": "GLOBAL",
            "field": "slot_id",
            "expected": None,
            "actual": worker_slot_id,
            "severity": "ERROR",
        })

    # 2. Check all ten cohort issuers and comparison fields
    required_fields = ("accounted_count", "newly_ingested_count", "gaps", "accounted_through")

    for ticker in COHORT_CIKS:
        w_iss = worker_issuers.get(ticker)
        indep = export_issuers.get(ticker)

        if not w_iss:
            discrepancies.append({
                "ticker": ticker,
                "field": "worker_evidence_presence",
                "expected": "present",
                "actual": "missing in worker receipt",
                "severity": "ERROR",
            })
            continue

        if not indep:
            for f in required_fields:
                pending_fields.append({
                    "ticker": ticker,
                    "field": f,
                    "reason": "missing issuer in independent export",
                })
            continue

        inventory = w_iss.get("inventory", [])
        actual_newly_ingested = sum(1 for item in inventory if item.get("disposition") == "INGESTED")
        actual_accounted = sum(1 for item in inventory if item.get("disposition") in ("INGESTED", "ALREADY_ACCOUNTED"))
        actual_gaps = sorted(w_iss.get("gaps", []))
        actual_accounted_through = w_iss.get("accounted_through")

        # Field 1: accounted_count
        exp_accounted = indep.get("accounted_count")
        if exp_accounted is None:
            pending_fields.append({
                "ticker": ticker,
                "field": "accounted_count",
                "reason": "null or missing",
            })
        elif exp_accounted != actual_accounted:
            discrepancies.append({
                "ticker": ticker,
                "field": "accounted_count",
                "expected": exp_accounted,
                "actual": actual_accounted,
                "severity": "ERROR",
            })

        # Field 2: newly_ingested_count
        exp_newly_ingested = indep.get("newly_ingested_count")
        if exp_newly_ingested is None:
            pending_fields.append({
                "ticker": ticker,
                "field": "newly_ingested_count",
                "reason": "null or missing",
            })
        elif exp_newly_ingested != actual_newly_ingested:
            discrepancies.append({
                "ticker": ticker,
                "field": "newly_ingested_count",
                "expected": exp_newly_ingested,
                "actual": actual_newly_ingested,
                "severity": "ERROR",
            })

        # Field 3: gaps
        exp_gaps = indep.get("gaps")
        if exp_gaps is None:
            pending_fields.append({
                "ticker": ticker,
                "field": "gaps",
                "reason": "null or missing",
            })
        else:
            sorted_exp_gaps = sorted(exp_gaps)
            if sorted_exp_gaps != actual_gaps:
                discrepancies.append({
                    "ticker": ticker,
                    "field": "gaps",
                    "expected": sorted_exp_gaps,
                    "actual": actual_gaps,
                    "severity": "ERROR",
                })

        # Field 4: accounted_through
        exp_accounted_through = indep.get("accounted_through")
        if exp_accounted_through is None or exp_accounted_through == "":
            pending_fields.append({
                "ticker": ticker,
                "field": "accounted_through",
                "reason": "null or missing",
            })
        elif (
            actual_accounted_through is not None
            and exp_accounted_through != actual_accounted_through
        ):
            discrepancies.append({
                "ticker": ticker,
                "field": "accounted_through",
                "expected": exp_accounted_through,
                "actual": actual_accounted_through,
                "severity": "ERROR",
            })

    # Status verdict determination
    if binding_errors or discrepancies:
        status = "DISCREPANCY"
    elif pending_fields:
        status = "PENDING"
    else:
        status = "RECONCILED"

    return ComparisonReport(
        status=status,
        discrepancies=discrepancies,
        pending_fields=pending_fields,
        binding_errors=binding_errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Issue #165 worker receipt and independent reconciliation exporter."
    )
    parser.add_argument("--slot-id", help="Dated slot ID (e.g. 2026-11-30:SLOT_1:0915)")
    parser.add_argument("--trial-id", default=DEFAULT_TRIAL_ID, help="Trial ID (default ARX-TRIAL-10-ISSUER-001)")
    parser.add_argument("--dsn", default=os.getenv("SMART_MONEY_COHORT_PILOT_DATABASE_URL"), help="PostgreSQL DSN")
    parser.add_argument("--output-dir", default="exports", help="Directory for JSON export files")
    parser.add_argument("--compare", action="store_true", help="Run local discrepancy comparison across all 10 issuers")
    parser.add_argument("--receipt-file", help="Path to stored worker receipt JSON file for offline comparison")
    parser.add_argument("--export-file", help="Path to independent reconciliation JSON file for offline comparison")
    args = parser.parse_args()

    # Offline file comparison mode
    if args.receipt_file and args.export_file:
        try:
            with open(args.receipt_file, "r", encoding="utf-8") as f:
                stored_receipt = json.load(f)
            with open(args.export_file, "r", encoding="utf-8") as f:
                independent_export = json.load(f)

            report = reconcile_and_compare_local(stored_receipt, independent_export)
            print(report.format_cli_output())

            if report.is_discrepancy:
                return 2
            elif report.is_pending:
                return 3
            else:
                return 0
        except Exception as exc:
            print(f"Comparison failed: {exc}", file=sys.stderr)
            return 1

    # Live or database export mode
    if not args.slot_id:
        print("Error: --slot-id is required for database export mode (or specify --receipt-file and --export-file).", file=sys.stderr)
        return 1

    if not args.dsn:
        print("Error: PostgreSQL DSN not provided and SMART_MONEY_COHORT_PILOT_DATABASE_URL is unset.", file=sys.stderr)
        return 1

    try:
        stored_receipt, independent_export = export_slot_evidence(
            dsn_or_cur=args.dsn,
            slot_id=args.slot_id,
            output_dir=args.output_dir,
            trial_id=args.trial_id,
        )
        print(f"Successfully exported receipt and independent reconciliation for slot {args.slot_id} to '{args.output_dir}'.")
        print(f"Receipt ID: {stored_receipt.get('receipt_id')}")
        print(f"Export ID: {independent_export.get('export_id')}")

        if args.compare:
            report = reconcile_and_compare_local(stored_receipt, independent_export)
            print(f"\n{report.format_cli_output()}")

            if report.is_discrepancy:
                return 2
            elif report.is_pending:
                return 3
            else:
                return 0

        return 0

    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
