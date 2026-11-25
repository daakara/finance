#!/usr/bin/env python3
"""
Generate worker-shaped offline test evidence files for Issue #165 operator rehearsal:
1. Scenario A: Complete export (RECONCILED, exit 0)
2. Scenario B: Missing evidence (PENDING, exit 3)
3. Scenario C: Known mismatch with pending fields (DISCREPANCY, exit 2)
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.export_pilot_receipt_and_reconciliation import (
    COHORT_CIKS,
    DEFAULT_RELEASE_SHA,
    DEFAULT_TRIAL_ID,
)

FIXTURE_DIR = "rehearsal_fixtures"
SLOT_ID = "2026-11-30:SLOT_1:0915"


def build_base_worker_issuers():
    issuers = {}
    for ticker, cik in COHORT_CIKS.items():
        if ticker == "NVDA":
            inv = [{
                "accession": "0001045810-26-000001",
                "form": "4",
                "filing_date": "2026-11-30",
                "accepted_at": "2026-11-30T14:05:00Z",
                "primary_document": "owner.xml",
                "disposition": "INGESTED",
                "source_sha256": "a" * 64,
            }]
        else:
            inv = []
        issuers[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "observed_at": "2026-11-30T14:21:00Z",
            "interval_start": "2026-11-30T14:00:00Z",
            "cutoff": "2026-11-30T14:20:00Z",
            "artifact_id": f"evt-audit-slot1-{ticker.lower()}",
            "catalog_sha256": "c" * 64,
            "catalog_gaps": [],
            "inventory": inv,
            "gaps": [],
            "previous_watermark": "2026-11-30T14:00:00Z",
            "accounted_through": "2026-11-30T14:20:00Z",
            "status": "CATALOG_ACCOUNTED",
            "sha256": "s" * 64,
        }
    return issuers


def build_base_receipt(receipt_id: str, issuers: dict):
    return {
        "receipt_id": receipt_id,
        "interval_start": "2026-11-30T14:00:00Z",
        "interval_end": "2026-11-30T14:20:00Z",
        "certified_watermark": "2026-11-30T14:00:00Z",
        "receipt_version": 1,
        "supersedes_receipt_id": None,
        "expected_count": 1,
        "accounted_count": 1,
        "missing_count": 0,
        "material_gap_count": 0,
        "coverage_status": "PARTIAL",
        "reconciliation_hash": "r" * 64,
        "release_sha": DEFAULT_RELEASE_SHA,
        "created_at": "2026-11-30T14:21:00Z",
        "slot_id": SLOT_ID,
        "trial_id": DEFAULT_TRIAL_ID,
        "failure_reasons": {
            "trial_id": DEFAULT_TRIAL_ID,
            "slot_id": SLOT_ID,
            "cohort_evidence": {
                "version": 1,
                "trial_id": DEFAULT_TRIAL_ID,
                "slot_id": SLOT_ID,
                "release_sha": DEFAULT_RELEASE_SHA,
                "issuers": issuers,
            },
        },
    }


def generate_all_fixtures():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    issuers = build_base_worker_issuers()

    # Scenario A: Complete export (RECONCILED)
    receipt_a = build_base_receipt("rcpt-rehearsal-scenario-a", issuers)
    export_a_issuers = {}
    for ticker, cik in COHORT_CIKS.items():
        export_a_issuers[ticker] = {
            "ticker": ticker,
            "cik": cik,
            "accounted_count": 1 if ticker == "NVDA" else 0,
            "newly_ingested_count": 1 if ticker == "NVDA" else 0,
            "gaps": [],
            "accounted_through": "2026-11-30T14:20:00Z",
            "state": "FILINGS_FOUND" if ticker == "NVDA" else "POLLED_QUIET",
            "verified": True,
        }
    export_a = {
        "export_id": "exp-rehearsal-scenario-a",
        "receipt_id": "rcpt-rehearsal-scenario-a",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": SLOT_ID,
        "reconciled_at": "2026-11-30T14:25:00Z",
        "reconciled_by": "operator-rehearsal",
        "issuers": export_a_issuers,
    }

    # Scenario B: Missing evidence (PENDING)
    receipt_b = build_base_receipt("rcpt-rehearsal-scenario-b", issuers)
    export_b_issuers = {}
    for ticker, cik in COHORT_CIKS.items():
        if ticker == "META":
            # META has unestablished comparison fields (null)
            export_b_issuers[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": None,
                "newly_ingested_count": None,
                "gaps": None,
                "accounted_through": None,
                "state": "NOT_POLLED",
                "verified": False,
            }
        else:
            export_b_issuers[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": 1 if ticker == "NVDA" else 0,
                "newly_ingested_count": 1 if ticker == "NVDA" else 0,
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
                "state": "FILINGS_FOUND" if ticker == "NVDA" else "POLLED_QUIET",
                "verified": True,
            }
    export_b = {
        "export_id": "exp-rehearsal-scenario-b",
        "receipt_id": "rcpt-rehearsal-scenario-b",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": SLOT_ID,
        "reconciled_at": "2026-11-30T14:25:00Z",
        "reconciled_by": "operator-rehearsal",
        "issuers": export_b_issuers,
    }

    # Scenario C: Known mismatch with pending fields (DISCREPANCY)
    receipt_c = build_base_receipt("rcpt-rehearsal-scenario-c", issuers)
    export_c_issuers = {}
    for ticker, cik in COHORT_CIKS.items():
        if ticker == "NVDA":
            # Count discrepancy: expected 5 vs actual 1
            export_c_issuers[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": 5,
                "newly_ingested_count": 1,
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
                "state": "FILINGS_FOUND",
                "verified": True,
            }
        elif ticker == "META":
            # Pending null field alongside discrepancy
            export_c_issuers[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": None,
                "newly_ingested_count": 0,
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
                "state": "INCOMPLETE",
                "verified": False,
            }
        else:
            export_c_issuers[ticker] = {
                "ticker": ticker,
                "cik": cik,
                "accounted_count": 0,
                "newly_ingested_count": 0,
                "gaps": [],
                "accounted_through": "2026-11-30T14:20:00Z",
                "state": "POLLED_QUIET",
                "verified": True,
            }
    export_c = {
        "export_id": "exp-rehearsal-scenario-c",
        "receipt_id": "rcpt-rehearsal-scenario-c",
        "trial_id": DEFAULT_TRIAL_ID,
        "slot_id": SLOT_ID,
        "reconciled_at": "2026-11-30T14:25:00Z",
        "reconciled_by": "operator-rehearsal",
        "issuers": export_c_issuers,
    }

    pairs = [
        ("scenario_a", receipt_a, export_a),
        ("scenario_b", receipt_b, export_b),
        ("scenario_c", receipt_c, export_c),
    ]

    for name, r, e in pairs:
        r_path = os.path.join(FIXTURE_DIR, f"{name}_receipt.json")
        e_path = os.path.join(FIXTURE_DIR, f"{name}_export.json")
        with open(r_path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, sort_keys=True)
        with open(e_path, "w", encoding="utf-8") as f:
            json.dump(e, f, indent=2, sort_keys=True)
        print(f"Generated {r_path} and {e_path}")


if __name__ == "__main__":
    generate_all_fixtures()
