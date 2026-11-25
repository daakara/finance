"""
Tests and offline rehearsal for Issue #165 pilot receipt and independent reconciliation exporter.
Rehearses read-only export and independent derivation against worker-shaped test data.
"""

import base64
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from scripts.export_pilot_receipt_and_reconciliation import (
    COHORT_CIKS,
    DEFAULT_RELEASE_SHA,
    DEFAULT_TRIAL_ID,
    derive_independent_reconciliation,
    evidence_sha256,
    export_slot_evidence,
    extract_stored_worker_receipt,
    inventory_from_catalog,
    reconcile_and_compare_local,
)


def make_submissions_json(cik: str, rows: list) -> bytes:
    """Helper to generate standard SEC submissions catalog JSON."""
    prior = ["10-Q", "0001045810-26-000021", "2026-11-19", "2026-11-19T18:00:00Z", "prior.htm"]
    columns = ("form", "accessionNumber", "filingDate", "acceptanceDateTime", "primaryDocument")
    all_rows = [prior, *rows]
    data = {
        "cik": int(cik),
        "filings": {
            "recent": dict(zip(columns, map(list, zip(*all_rows)))),
            "files": [],
        },
    }
    return json.dumps(data).encode("utf-8")


class MockEvidenceDb:
    """In-memory mock representing finance_smart_money_cohort_001 tables."""

    def __init__(self):
        self.reconciliation_receipts = []
        self.audit_events = []
        self.raw_filings = {}
        self.source_availability = {}
        self.amendments = {}

    def get_cursor(self):
        db = self

        class MockCursor:
            def __init__(self):
                self._results = []
                self._last_query = ""

            def execute(self, query, params=None):
                self._last_query = query
                params = params or ()
                q = " ".join(query.split())

                if "FROM sm_reconciliation_receipts" in q:
                    if "AS watermark" in q:
                        ticker = params[0]
                        cik = params[3]
                        # Find prior receipt watermark
                        watermark = None
                        for r in db.reconciliation_receipts:
                            ev = r.get("failure_reasons", {}).get("cohort_evidence", {}).get("issuers", {})
                            if ticker in ev and ev[ticker].get("cik") == cik:
                                watermark = ev[ticker].get("accounted_through")
                        self._results = [{"watermark": watermark}] if watermark else []
                    else:
                        slot_id = params[0]
                        matches = [
                            r for r in db.reconciliation_receipts
                            if r.get("slot_id") == slot_id or r.get("failure_reasons", {}).get("cohort_evidence", {}).get("slot_id") == slot_id
                        ]
                        self._results = matches[:1]

                elif "FROM sm_audit_events" in q:
                    trial_id, slot_id, ticker = params[0], params[1], params[2]
                    matches = [
                        ev for ev in db.audit_events
                        if ev["event_type"] == "SEC_COHORT_CATALOG_OBSERVED"
                        and ev["subject_id"] == trial_id
                        and ev["payload"].get("slot_id") == slot_id
                        and ev["payload"].get("ticker") == ticker
                    ]
                    self._results = matches[:1]

                elif "FROM sm_amendments" in q:
                    acc = params[0]
                    amend = db.amendments.get(acc)
                    self._results = [{"resolution_status": amend["resolution_status"]}] if amend else []

                elif "FROM form4_raw_filings" in q:
                    acc = params[0]
                    raw = db.raw_filings.get(acc)
                    self._results = [{"content_hash": raw["content_hash"], "created_at": raw["created_at"]}] if raw else []

                else:
                    self._results = []

            def fetchone(self):
                return self._results[0] if self._results else None

            def fetchall(self):
                return list(self._results)

        return MockCursor()


def setup_worker_shaped_slot_1(db: MockEvidenceDb):
    """Populates db with realistic worker-shaped slot 1 evidence for all 10 issuers."""
    slot_id = "2026-11-30:SLOT_1:0915"
    receipt_id = "rcpt-pilot-165-20261130-slot1"
    start_iso = "2026-11-30T14:00:00Z"
    cutoff_iso = "2026-11-30T14:20:00Z"
    slot_time = datetime(2026, 11, 30, 14, 21, tzinfo=timezone.utc)

    # NVDA has 1 new Form 4
    nvda_acc = "0001045810-26-000001"
    nvda_hash = "a" * 64
    db.raw_filings[nvda_acc] = {
        "accession_number": nvda_acc,
        "ticker": "NVDA",
        "cik": COHORT_CIKS["NVDA"],
        "content_hash": nvda_hash,
        "created_at": slot_time,
    }
    db.source_availability[nvda_acc] = {"public_availability_at": slot_time}

    worker_issuers = {}

    for ticker, cik in COHORT_CIKS.items():
        if ticker == "NVDA":
            inv_rows = [["4", nvda_acc, "2026-11-30", "2026-11-30T14:05:00Z", "owner.xml"]]
            inv = [{
                "accession": nvda_acc,
                "form": "4",
                "filing_date": "2026-11-30",
                "accepted_at": "2026-11-30T14:05:00Z",
                "primary_document": "owner.xml",
                "disposition": "INGESTED",
                "source_sha256": nvda_hash,
            }]
            gaps = []
            status = "CATALOG_ACCOUNTED"
            acc_through = cutoff_iso
        else:
            inv_rows = []
            inv = []
            gaps = []
            status = "CATALOG_ACCOUNTED"
            acc_through = cutoff_iso

        body = make_submissions_json(cik, inv_rows)
        cat_digest = hashlib.sha256(body).hexdigest()

        obs_payload = {
            "cycle_id": "cycle-001",
            "slot_id": slot_id,
            "grant_sha256": "g" * 64,
            "ticker": ticker,
            "cik": cik,
            "request_url": f"https://data.sec.gov/submissions/CIK{cik}.json",
            "response_url": f"https://data.sec.gov/submissions/CIK{cik}.json",
            "observed_at": slot_time.isoformat(),
            "interval_start": start_iso,
            "cutoff": cutoff_iso,
            "catalog_sha256": cat_digest,
            "body_base64": base64.b64encode(body).decode("ascii"),
        }
        event_id = f"evt-audit-slot1-{ticker.lower()}"
        db.audit_events.append({
            "event_id": event_id,
            "event_type": "SEC_COHORT_CATALOG_OBSERVED",
            "subject_id": DEFAULT_TRIAL_ID,
            "payload": obs_payload,
            "event_hash": evidence_sha256(obs_payload),
            "created_at": slot_time,
        })

        worker_issuers[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "observed_at": slot_time.isoformat(),
            "interval_start": start_iso,
            "cutoff": cutoff_iso,
            "artifact_id": event_id,
            "catalog_sha256": cat_digest,
            "catalog_gaps": [],
            "inventory": inv,
            "gaps": gaps,
            "previous_watermark": start_iso,
            "accounted_through": acc_through,
            "status": status,
            "sha256": "s" * 64,
        }

    receipt = {
        "receipt_id": receipt_id,
        "interval_start": start_iso,
        "interval_end": cutoff_iso,
        "certified_watermark": start_iso,
        "receipt_version": 1,
        "supersedes_receipt_id": None,
        "expected_count": 1,
        "accounted_count": 1,
        "missing_count": 0,
        "material_gap_count": 0,
        "coverage_status": "PARTIAL",
        "reconciliation_hash": "r" * 64,
        "release_sha": DEFAULT_RELEASE_SHA,
        "created_at": slot_time.isoformat(),
        "slot_id": slot_id,
        "trial_id": DEFAULT_TRIAL_ID,
        "failure_reasons": {
            "trial_id": DEFAULT_TRIAL_ID,
            "slot_id": slot_id,
            "cohort_evidence": {
                "version": 1,
                "trial_id": DEFAULT_TRIAL_ID,
                "slot_id": slot_id,
                "release_sha": DEFAULT_RELEASE_SHA,
                "issuers": worker_issuers,
            },
        },
    }
    db.reconciliation_receipts.append(receipt)


def test_offline_rehearsal_slot_1_all_10_issuers_reconcile():
    """Verify that export extracts receipt and independently derives reconciliation with zero discrepancies."""
    db = MockEvidenceDb()
    setup_worker_shaped_slot_1(db)
    cur = db.get_cursor()

    slot_id = "2026-11-30:SLOT_1:0915"

    with tempfile.TemporaryDirectory() as tmpdir:
        stored_receipt, independent_export = export_slot_evidence(
            dsn_or_cur=cur,
            slot_id=slot_id,
            output_dir=tmpdir,
            trial_id=DEFAULT_TRIAL_ID,
        )

        # 1. Verify export binding
        assert stored_receipt["receipt_id"] == "rcpt-pilot-165-20261130-slot1"
        assert independent_export["receipt_id"] == "rcpt-pilot-165-20261130-slot1"
        assert stored_receipt["slot_id"] == slot_id
        assert independent_export["slot_id"] == slot_id
        assert stored_receipt["trial_id"] == DEFAULT_TRIAL_ID
        assert independent_export["trial_id"] == DEFAULT_TRIAL_ID

        # 2. Verify all 10 issuers derived independently
        issuers = independent_export["issuers"]
        assert set(issuers.keys()) == set(COHORT_CIKS.keys())

        # NVDA has 1 newly ingested filing
        nvda = issuers["NVDA"]
        assert nvda["accounted_count"] == 1
        assert nvda["newly_ingested_count"] == 1
        assert nvda["gaps"] == []
        assert nvda["accounted_through"] == "2026-11-30T14:20:00Z"
        assert nvda["state"] == "FILINGS_FOUND"

        # The other 9 issuers are quiet
        for ticker in COHORT_CIKS:
            if ticker != "NVDA":
                iss = issuers[ticker]
                assert iss["accounted_count"] == 0
                assert iss["newly_ingested_count"] == 0
                assert iss["gaps"] == []
                assert iss["accounted_through"] == "2026-11-30T14:20:00Z"
                assert iss["state"] == "POLLED_QUIET"

        # 3. Verify local closeout comparison across all 10 rows
        discrepancies = reconcile_and_compare_local(stored_receipt, independent_export)
        assert discrepancies == [], f"Expected 0 discrepancies, got: {discrepancies}"

        # 4. Verify JSON files exist on disk
        slot_slug = slot_id.replace(":", "_").replace("-", "")
        receipt_file = os.path.join(tmpdir, f"receipt_{slot_slug}.json")
        indep_file = os.path.join(tmpdir, f"independent_reconciliation_{slot_slug}.json")
        assert os.path.isfile(receipt_file)
        assert os.path.isfile(indep_file)


def test_offline_rehearsal_slot_2_already_accounted_filing():
    """Verify that earlier ingested filing in slot 2 is accounted without double-counting newly_ingested."""
    db = MockEvidenceDb()
    setup_worker_shaped_slot_1(db)

    # In Slot 2, NVDA catalog still lists the same accession, but it was already ingested in slot 1
    slot_2_id = "2026-11-30:SLOT_2:1300"
    receipt_2_id = "rcpt-pilot-165-20261130-slot2"
    slot_2_time = datetime(2026, 11, 30, 18, 5, tzinfo=timezone.utc)
    cutoff_2_iso = "2026-11-30T18:00:00Z"

    nvda_acc = "0001045810-26-000001"
    # Raw filing created_at is from slot 1 (14:21 UTC), which is earlier than slot 2 time (18:05 UTC)
    body = make_submissions_json(COHORT_CIKS["NVDA"], [["4", nvda_acc, "2026-11-30", "2026-11-30T14:05:00Z", "owner.xml"]])
    cat_digest = hashlib.sha256(body).hexdigest()

    obs_payload = {
        "cycle_id": "cycle-002",
        "slot_id": slot_2_id,
        "grant_sha256": "g" * 64,
        "ticker": "NVDA",
        "cik": COHORT_CIKS["NVDA"],
        "observed_at": slot_2_time.isoformat(),
        "interval_start": "2026-11-30T14:00:00Z",
        "cutoff": cutoff_2_iso,
        "catalog_sha256": cat_digest,
        "body_base64": base64.b64encode(body).decode("ascii"),
    }
    db.audit_events.append({
        "event_id": "evt-audit-slot2-nvda",
        "event_type": "SEC_COHORT_CATALOG_OBSERVED",
        "subject_id": DEFAULT_TRIAL_ID,
        "payload": obs_payload,
        "event_hash": evidence_sha256(obs_payload),
        "created_at": slot_2_time,
    })

    # Add other 9 quiet issuers for slot 2
    for ticker, cik in COHORT_CIKS.items():
        if ticker != "NVDA":
            b = make_submissions_json(cik, [])
            d = hashlib.sha256(b).hexdigest()
            p = {
                "cycle_id": "cycle-002", "slot_id": slot_2_id, "grant_sha256": "g" * 64,
                "ticker": ticker, "cik": cik, "observed_at": slot_2_time.isoformat(),
                "interval_start": "2026-11-30T14:00:00Z", "cutoff": cutoff_2_iso,
                "catalog_sha256": d, "body_base64": base64.b64encode(b).decode("ascii"),
            }
            db.audit_events.append({
                "event_id": f"evt-audit-slot2-{ticker.lower()}",
                "event_type": "SEC_COHORT_CATALOG_OBSERVED",
                "subject_id": DEFAULT_TRIAL_ID,
                "payload": p,
                "event_hash": evidence_sha256(p),
                "created_at": slot_2_time,
            })

    cur = db.get_cursor()
    indep = derive_independent_reconciliation(cur, slot_id=slot_2_id, trial_id=DEFAULT_TRIAL_ID, receipt_id=receipt_2_id)

    nvda = indep["issuers"]["NVDA"]
    # Accounted count is 1 (ALREADY_ACCOUNTED), newly_ingested is 0!
    assert nvda["accounted_count"] == 1
    assert nvda["newly_ingested_count"] == 0
    assert nvda["gaps"] == []
    assert nvda["accounted_through"] == cutoff_2_iso


def test_offline_rehearsal_missing_audit_event_reports_pending():
    """Verify that when an issuer lacks an audit event, its comparison fields are None/pending."""
    db = MockEvidenceDb()
    setup_worker_shaped_slot_1(db)

    # Remove META's audit event
    db.audit_events = [ev for ev in db.audit_events if ev["payload"].get("ticker") != "META"]

    cur = db.get_cursor()
    indep = derive_independent_reconciliation(cur, slot_id="2026-11-30:SLOT_1:0915", trial_id=DEFAULT_TRIAL_ID)

    meta = indep["issuers"]["META"]
    assert meta["accounted_count"] is None
    assert meta["newly_ingested_count"] is None
    assert meta["gaps"] is None
    assert meta["accounted_through"] is None
    assert meta["state"] == "NOT_POLLED"
    assert meta["verified"] is False


def test_offline_rehearsal_unresolved_amendment_stops_checkpoint():
    """Verify that an unresolved amendment records gap and keeps watermark bounded."""
    db = MockEvidenceDb()
    setup_worker_shaped_slot_1(db)

    # Add an amendment for AAPL in sm_amendments
    aapl_acc = "0000320193-26-000099"
    db.amendments[aapl_acc] = {"resolution_status": "AMENDMENT_UNRESOLVED"}

    # Update AAPL audit event to include this amendment
    for ev in db.audit_events:
        if ev["payload"].get("ticker") == "AAPL":
            body = make_submissions_json(COHORT_CIKS["AAPL"], [["4/A", aapl_acc, "2026-11-30", "2026-11-30T14:10:00Z", "amend.xml"]])
            ev["payload"]["body_base64"] = base64.b64encode(body).decode("ascii")
            ev["payload"]["catalog_sha256"] = hashlib.sha256(body).hexdigest()
            ev["event_hash"] = evidence_sha256(ev["payload"])

    cur = db.get_cursor()
    indep = derive_independent_reconciliation(cur, slot_id="2026-11-30:SLOT_1:0915", trial_id=DEFAULT_TRIAL_ID)

    aapl = indep["issuers"]["AAPL"]
    assert f"AMENDMENT_UNRESOLVED:{aapl_acc}" in aapl["gaps"]
    assert aapl["state"] == "INCOMPLETE"
    assert aapl["accounted_through"] is None  # Stops at previous watermark (which was None for slot 1)


def test_local_closeout_records_discrepancy_details():
    """Verify that discrepancies record exact ticker, field, expected, and actual values."""
    # Build full 10-issuer stored receipt
    worker_issuers = {}
    export_issuers = {}

    for ticker, cik in COHORT_CIKS.items():
        if ticker == "NVDA":
            worker_issuers[ticker] = {
                "ticker": ticker,
                "inventory": [{"disposition": "INGESTED", "accession": "acc-1"}],
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
            }
            # Intentional mismatch on NVDA: export reports accounted_count = 2 (worker has 1)
            export_issuers[ticker] = {
                "ticker": ticker,
                "accounted_count": 2,
                "newly_ingested_count": 1,
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
            }
        else:
            worker_issuers[ticker] = {
                "ticker": ticker,
                "inventory": [],
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
            }
            export_issuers[ticker] = {
                "ticker": ticker,
                "accounted_count": 0,
                "newly_ingested_count": 0,
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
            }

    stored_receipt = {
        "receipt_id": "rcpt-001",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": "2026-11-30:SLOT_1:0915",
        "cohort_evidence": {
            "trial_id": DEFAULT_TRIAL_ID,
            "slot_id": "2026-11-30:SLOT_1:0915",
            "issuers": worker_issuers,
        },
    }

    independent_export = {
        "receipt_id": "rcpt-001",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": "2026-11-30:SLOT_1:0915",
        "issuers": export_issuers,
    }

    report = reconcile_and_compare_local(stored_receipt, independent_export)
    assert report.status == "DISCREPANCY"
    assert report.is_discrepancy is True
    assert len(report.discrepancies) == 1
    d = report.discrepancies[0]
    assert d["ticker"] == "NVDA"
    assert d["field"] == "accounted_count"
    assert d["expected"] == 2
    assert d["actual"] == 1
    assert d["severity"] == "ERROR"


def test_compare_scenario_1_complete_match_reconciled():
    """SCENARIO 1: RECONCILED only when both exports match on trial, receipt, and dated slot, and all 10 issuers have complete, matching comparison fields."""
    worker_issuers = {}
    export_issuers = {}

    for ticker, cik in COHORT_CIKS.items():
        inv = [{"disposition": "INGESTED", "accession": "acc-1"}] if ticker == "NVDA" else []
        worker_issuers[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "inventory": inv,
            "gaps": [],
            "accounted_through": "2026-11-30T14:20:00Z",
        }
        export_issuers[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "accounted_count": 1 if ticker == "NVDA" else 0,
            "newly_ingested_count": 1 if ticker == "NVDA" else 0,
            "gaps": [],
            "accounted_through": "2026-11-30T14:20:00Z",
        }

    stored_receipt = {
        "receipt_id": "rcpt-complete-match",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": "2026-11-30:SLOT_1:0915",
        "cohort_evidence": {
            "trial_id": DEFAULT_TRIAL_ID,
            "slot_id": "2026-11-30:SLOT_1:0915",
            "issuers": worker_issuers,
        },
    }

    independent_export = {
        "receipt_id": "rcpt-complete-match",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": "2026-11-30:SLOT_1:0915",
        "issuers": export_issuers,
    }

    report = reconcile_and_compare_local(stored_receipt, independent_export)
    assert report.status == "RECONCILED"
    assert report.is_reconciled is True
    assert len(report.discrepancies) == 0
    assert len(report.pending_fields) == 0
    cli_output = report.format_cli_output()
    assert "RECONCILED: All 10 cohort issuer rows match" in cli_output
    assert "Zero discrepancies detected" in cli_output


def test_compare_scenario_2_missing_or_null_evidence_pending():
    """SCENARIO 2: PENDING when a required issuer or comparison field is missing or null. Prints affected issuer and field."""
    worker_issuers = {}
    export_issuers = {}

    for ticker, cik in COHORT_CIKS.items():
        worker_issuers[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "inventory": [],
            "gaps": [],
            "accounted_through": "2026-11-30T14:20:00Z",
        }
        if ticker == "META":
            # META has null / missing accounted_count and gaps
            export_issuers[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": None,
                "newly_ingested_count": 0,
                "gaps": None,
                "accounted_through": "2026-11-30T14:20:00Z",
            }
        elif ticker == "TSLA":
            # TSLA missing entirely from export_issuers
            continue
        else:
            export_issuers[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": 0,
                "newly_ingested_count": 0,
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
            }

    stored_receipt = {
        "receipt_id": "rcpt-pending-test",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": "2026-11-30:SLOT_1:0915",
        "cohort_evidence": {
            "trial_id": DEFAULT_TRIAL_ID,
            "slot_id": "2026-11-30:SLOT_1:0915",
            "issuers": worker_issuers,
        },
    }

    independent_export = {
        "receipt_id": "rcpt-pending-test",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": "2026-11-30:SLOT_1:0915",
        "issuers": export_issuers,
    }

    report = reconcile_and_compare_local(stored_receipt, independent_export)
    assert report.status == "PENDING"
    assert report.is_pending is True
    assert len(report.discrepancies) == 0
    assert len(report.pending_fields) > 0

    cli_output = report.format_cli_output()
    assert "PENDING: Reconciliation cannot be established" in cli_output
    assert "[META] field 'accounted_count' is pending / null" in cli_output
    assert "[META] field 'gaps' is pending / null" in cli_output
    assert "[TSLA] field 'accounted_count' is pending / null" in cli_output


def test_compare_scenario_3_mismatched_binding_discrepancy():
    """SCENARIO 3: DISCREPANCY when export identities differ (slot_id, receipt_id, or trial_id). Prints field, expected, and actual."""
    worker_issuers = {ticker: {"ticker": ticker, "inventory": [], "gaps": [], "accounted_through": "2026-11-30T14:20:00Z"} for ticker in COHORT_CIKS}
    export_issuers = {ticker: {"ticker": ticker, "accounted_count": 0, "newly_ingested_count": 0, "gaps": [], "accounted_through": "2026-11-30T14:20:00Z"} for ticker in COHORT_CIKS}

    stored_receipt = {
        "receipt_id": "rcpt-slot-1",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": "2026-11-30:SLOT_1:0915",
        "cohort_evidence": {
            "trial_id": DEFAULT_TRIAL_ID,
            "slot_id": "2026-11-30:SLOT_1:0915",
            "issuers": worker_issuers,
        },
    }

    # Export declared with wrong slot_id
    independent_export = {
        "receipt_id": "rcpt-slot-1",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": "2026-11-30:SLOT_2:1300",
        "issuers": export_issuers,
    }

    report = reconcile_and_compare_local(stored_receipt, independent_export)
    assert report.status == "DISCREPANCY"
    assert report.is_discrepancy is True
    assert len(report.binding_errors) == 1
    err = report.binding_errors[0]
    assert err["field"] == "slot_id"
    assert err["expected"] == "2026-11-30:SLOT_2:1300"
    assert err["actual"] == "2026-11-30:SLOT_1:0915"

    cli_output = report.format_cli_output()
    assert "DISCREPANCY: Export identities differ" in cli_output
    assert "field 'slot_id': expected '2026-11-30:SLOT_2:1300', actual '2026-11-30:SLOT_1:0915'" in cli_output


def test_compare_scenario_4_known_discrepancy_alongside_pending_evidence():
    """SCENARIO 4: DISCREPANCY when a known value disagrees, and reports pending fields alongside discrepancies when both occur."""
    worker_issuers = {}
    export_issuers = {}

    for ticker, cik in COHORT_CIKS.items():
        inv = [{"disposition": "INGESTED", "accession": "acc-1"}] if ticker == "NVDA" else []
        worker_issuers[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "inventory": inv,
            "gaps": [],
            "accounted_through": "2026-11-30T14:20:00Z",
        }

        if ticker == "NVDA":
            # Discrepancy on NVDA: expected accounted_count = 5 vs actual 1
            export_issuers[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": 5,
                "newly_ingested_count": 1,
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
            }
        elif ticker == "META":
            # Pending on META: accounted_count is None
            export_issuers[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": None,
                "newly_ingested_count": 0,
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
            }
        else:
            export_issuers[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": 0,
                "newly_ingested_count": 0,
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
            }

    stored_receipt = {
        "receipt_id": "rcpt-hybrid-test",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": "2026-11-30:SLOT_1:0915",
        "cohort_evidence": {
            "trial_id": DEFAULT_TRIAL_ID,
            "slot_id": "2026-11-30:SLOT_1:0915",
            "issuers": worker_issuers,
        },
    }

    independent_export = {
        "receipt_id": "rcpt-hybrid-test",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": "2026-11-30:SLOT_1:0915",
        "issuers": export_issuers,
    }

    report = reconcile_and_compare_local(stored_receipt, independent_export)
    assert report.status == "DISCREPANCY"
    assert report.is_discrepancy is True
    assert len(report.discrepancies) == 1
    assert len(report.pending_fields) == 1

    cli_output = report.format_cli_output()
    assert "DISCREPANCY: Export identities differ or known values disagree" in cli_output
    assert "[NVDA] field 'accounted_count': expected 5, actual 1" in cli_output
    assert "Pending Fields Reported Alongside Discrepancies" in cli_output
    assert "[META] field 'accounted_count' is pending / null" in cli_output


def test_rehearsal_export_cli_and_preview_parity():
    """Verify that export -> CLI comparison -> local preview produces identical RECONCILED results."""
    from scripts.export_pilot_receipt_and_reconciliation import main as cli_main
    import sys

    db = MockEvidenceDb()
    setup_worker_shaped_slot_1(db)
    cur = db.get_cursor()

    slot_id = "2026-11-30:SLOT_1:0915"

    with tempfile.TemporaryDirectory() as tmpdir:
        stored_receipt, independent_export = export_slot_evidence(
            dsn_or_cur=cur,
            slot_id=slot_id,
            output_dir=tmpdir,
            trial_id=DEFAULT_TRIAL_ID,
        )

        slot_slug = slot_id.replace(":", "_").replace("-", "")
        receipt_file = os.path.join(tmpdir, f"receipt_{slot_slug}.json")
        indep_file = os.path.join(tmpdir, f"independent_reconciliation_{slot_slug}.json")

        # 1. Test CLI comparison via Python API
        report = reconcile_and_compare_local(stored_receipt, independent_export)
        assert report.status == "RECONCILED"
        assert report.is_reconciled is True
        assert len(report.discrepancies) == 0
        assert len(report.pending_fields) == 0

        # 2. Test CLI execution via main()
        old_argv = sys.argv
        sys.argv = ["export_pilot_receipt_and_reconciliation.py", "--compare", "--receipt-file", receipt_file, "--export-file", indep_file]
        try:
            exit_code = cli_main()
            assert exit_code == 0
        finally:
            sys.argv = old_argv
