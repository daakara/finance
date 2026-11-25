"""Offline rehearsal of the recurring cohort through the real worker and PostgreSQL."""

import base64
import hashlib
import json
from datetime import datetime, timezone
from functools import partial
from unittest.mock import Mock
from urllib.parse import urlsplit

import psycopg2
from psycopg2.sql import Identifier, SQL
import pytest
import requests
from requests.adapters import HTTPAdapter

from analyst_dashboard.smart_money import continuous_authorization, continuous_pipeline, smart_money_scheduler, worker
from analyst_dashboard.smart_money.cohort_catalog import evidence_sha256, inventory_from_catalog
from analyst_dashboard.smart_money.continuous_authorization import COHORT_CIKS, COHORT_PILOT_ID, SCOPE_TRIAL_10
from analyst_dashboard.smart_money.storage.backup import freshness_authority
from analyst_dashboard.smart_money.storage.config import COHORT_PILOT_BACKUP_PREFIX, COHORT_PILOT_DATABASE
from analyst_dashboard.smart_money.storage.postgres_connection import transaction
from analyst_dashboard.smart_money.storage.postgres_schema import initialize_postgres_schema
from test_smart_money_two_issuer_trial import (
    disposable_db, generate_form4_xml, make_mock_backup_engine, pg_server_instance, trial_grant,
)


def submissions(cik, rows):
    prior = ("10-Q", "0001045810-26-000021", "2026-11-19", "2026-11-19T18:00:00Z", "prior.htm")
    columns = ("form", "accessionNumber", "filingDate", "acceptanceDateTime", "primaryDocument")
    return json.dumps({
        "cik": int(cik),
        "filings": {"recent": dict(zip(columns, map(list, zip(*[prior, *rows])))), "files": []},
    })


def filing(number, form="4"):
    return (form, f"0001045810-26-{number:06d}", "2026-11-20", "2026-11-20T14:04:00Z", "owner.xml")


@pytest.fixture(scope="module")
def cohort_dsn(pg_server_instance):
    canonical = psycopg2.connect(pg_server_instance.get_uri())
    canonical.autocommit = True
    with canonical.cursor() as cur:
        cur.execute(SQL("CREATE DATABASE {}").format(Identifier(COHORT_PILOT_DATABASE)))
    canonical.close()
    return urlsplit(pg_server_instance.get_uri())._replace(path=f"/{COHORT_PILOT_DATABASE}").geturl()


@pytest.fixture
def rehearsal(disposable_db, cohort_dsn, monkeypatch):
    conn = psycopg2.connect(cohort_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
    initialize_postgres_schema(conn)
    conn.close()
    monkeypatch.setenv("SMART_MONEY_COHORT_PILOT_DATABASE_URL", cohort_dsn)
    grant = trial_grant(**{
        "ARX_CONTINUOUS_AUTH_SCOPE": SCOPE_TRIAL_10,
        "ARX_CONTINUOUS_AUTH_ISSUED_AT": "2026-11-20T14:00:00Z",
        "ARX_CONTINUOUS_AUTH_EXPIRES_AT": "2026-11-20T21:41:00Z",
        "SEC_USER_AGENT": "ARX Terminal Research (ops@arxterminal.com)",
        "ARX_EMERGENCY_STOP_ACTIVE": "false",
    })
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("ARX_SMART_MONEY_INGESTION_ONLY", raising=False)
    for key, value in grant.items():
        monkeypatch.setenv(key, value)

    clock = [datetime(2026, 11, 20, 14, 16, tzinfo=timezone.utc)]

    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz) if tz else clock[0].replace(tzinfo=None)

    for module in (continuous_authorization, continuous_pipeline, smart_money_scheduler):
        monkeypatch.setattr(module, "datetime", FixedClock)
    storage_adapter = Mock(provider_name="cloudflare_r2", prefix=COHORT_PILOT_BACKUP_PREFIX)
    monkeypatch.setattr(freshness_authority, "get_backup_storage_adapter", lambda: storage_adapter)
    monkeypatch.setattr(continuous_pipeline, "get_backup_storage_adapter", lambda: storage_adapter)
    monkeypatch.setattr(continuous_pipeline, "validate_storage_configuration", lambda: None)
    monkeypatch.setattr(worker, "validate_storage_configuration", lambda: None)
    monkeypatch.setattr(continuous_pipeline.RateLimiter, "acquire", lambda self: None)
    monkeypatch.setattr(worker, "SmartMoneyContinuousPipeline", partial(
        continuous_pipeline.SmartMoneyContinuousPipeline, backup_engine=make_mock_backup_engine(),
    ))

    rows = []
    documents = {}
    requests_seen = []

    def send(_adapter, request, **_kwargs):
        url = request.url
        requests_seen.append(url)
        response = requests.Response()
        response.url = url
        response.request = request
        if url.startswith("https://data.sec.gov/submissions/CIK"):
            cik = url.rsplit("CIK", 1)[1].split(".")[0]
            assert cik in COHORT_CIKS.values()
            status, body = documents.get(cik, (200, submissions(cik, rows if cik == COHORT_CIKS["NVDA"] else [])))
        else:
            assert url.startswith("https://www.sec.gov/Archives/edgar/data/1045810/")
            compact_accession = url.split("/")[-2]
            status, body = documents[compact_accession]
        response.status_code = status
        response._content = body.encode()
        return response

    monkeypatch.setattr(HTTPAdapter, "send", send)
    gate = Mock()
    ingestion_worker = worker.SmartMoneyIngestionWorker(dsn=disposable_db, containment_gate=gate)
    assert ingestion_worker.validate_preflight()
    assert ingestion_worker.acquire_egress_authority(max_attempts=1)
    yield ingestion_worker, clock, rows, documents, requests_seen
    ingestion_worker.release_egress_authority()


def latest_receipt(ingestion_worker):
    with transaction(dsn=ingestion_worker.dsn) as cur:
        cur.execute("SELECT * FROM sm_reconciliation_receipts ORDER BY interval_end DESC LIMIT 1")
        return cur.fetchone()


def verify_retained_evidence(ingestion_worker, receipt):
    evidence = receipt["failure_reasons"]["cohort_evidence"]
    assert receipt["coverage_status"] == "PARTIAL"
    assert evidence_sha256(evidence) == receipt["reconciliation_hash"]
    assert evidence["issuers"].keys() == COHORT_CIKS.keys()
    with transaction(dsn=ingestion_worker.dsn) as cur:
        for issuer in evidence["issuers"].values():
            cur.execute("SELECT payload, event_hash FROM sm_audit_events WHERE event_id = %s", (issuer["artifact_id"],))
            event = cur.fetchone()
            payload = event["payload"]
            assert event["event_hash"] == evidence_sha256(payload)
            body = base64.b64decode(payload["body_base64"])
            assert hashlib.sha256(body).hexdigest() == issuer["catalog_sha256"]
            inventory, gaps = inventory_from_catalog(
                body, issuer["cik"], datetime.fromisoformat(issuer["interval_start"]),
                datetime.fromisoformat(issuer["cutoff"]),
            )
            assert gaps == issuer["catalog_gaps"]
            assert inventory == [
                {key: value for key, value in row.items() if key not in ("disposition", "source_sha256")}
                for row in issuer["inventory"]
            ]
            assert evidence_sha256({k: v for k, v in issuer.items() if k != "sha256"}) == issuer["sha256"]
    return evidence["issuers"]


@pytest.mark.parametrize(("remarks", "amendment_disposition"), [
    ("This amendment replaces the original Form 4 in its entirety.", "INGESTED"),
    ("Corrected filing.", "AMENDMENT_UNRESOLVED"),
])
def test_quiet_late_filing_and_amendment_across_three_slots(rehearsal, remarks, amendment_disposition):
    pilot, clock, rows, documents, requests_seen = rehearsal
    operator_id = pilot.repository.create_run_request("OPERATOR_INGESTION", "unrelated")
    pilot.run_cycle()
    first = latest_receipt(pilot)
    first_receipts = verify_retained_evidence(pilot, first)
    assert first["expected_count"] == 0
    assert len(requests_seen) == 10
    assert all(entry["status"] == "CATALOG_ACCOUNTED" for entry in first_receipts.values())
    pilot.run_cycle()
    assert len(requests_seen) == 10

    # The filing predates slot 1's watermark but only appears in slot 2's catalog.
    new_filing = filing(1)
    rows.extend([new_filing, new_filing])
    documents[new_filing[1].replace("-", "")] = (200, generate_form4_xml(date="2026-11-20"))
    clock[0] = datetime(2026, 11, 20, 18, 1, tzinfo=timezone.utc)
    pilot.run_cycle()
    second = latest_receipt(pilot)
    nvda = verify_retained_evidence(pilot, second)["NVDA"]
    assert nvda["previous_watermark"] == first_receipts["NVDA"]["accounted_through"]
    assert nvda["inventory"][0]["disposition"] == "INGESTED"
    assert second["accounted_count"] == 1
    assert len(requests_seen) == 21

    amendment = filing(2, "4/A")
    rows.append(amendment)
    documents[amendment[1].replace("-", "")] = (200, generate_form4_xml(
        date="2026-11-20", is_amendment=True, original_date="2026-11-20", remarks=remarks,
    ))
    clock[0] = datetime(2026, 11, 20, 21, 31, tzinfo=timezone.utc)
    pilot.run_cycle()
    third = latest_receipt(pilot)
    nvda = verify_retained_evidence(pilot, third)["NVDA"]
    assert [entry["disposition"] for entry in nvda["inventory"]] == ["ALREADY_ACCOUNTED", amendment_disposition]
    checkpoint = nvda["cutoff"] if amendment_disposition == "INGESTED" else nvda["previous_watermark"]
    assert nvda["accounted_through"] == checkpoint
    assert len(requests_seen) == 32
    progress = pilot.repository.get_trial_progress(COHORT_PILOT_ID, "2026-11-20")
    assert progress["cycles_trial"] == 3
    assert progress["requests_trial"] == 120
    assert pilot.repository.get_run_request(operator_id)["status"] == "PENDING"
    with transaction(dsn=pilot.dsn) as cur:
        cur.execute("SELECT status, result_generation_id FROM sm_run_requests WHERE request_type = 'TRIAL_SLOT_FORM4'")
        assert {(row["status"], row["result_generation_id"]) for row in cur.fetchall()} == {
            ("COMPLETED", str(receipt["receipt_id"])) for receipt in (first, second, third)
        }
        cur.execute("SELECT count(*) AS n FROM scanner_active_snapshots")
        assert cur.fetchone()["n"] == 0
        cur.execute("SELECT count(*) AS n FROM form4_parsed_transactions")
        assert cur.fetchone()["n"] == 2
        cur.execute("SELECT base_accession FROM sm_amendments WHERE amendment_accession = %s", (amendment[1],))
        assert cur.fetchone()["base_accession"] == new_filing[1]
    with psycopg2.connect(pilot.owner_dsn) as canonical:
        with canonical.cursor() as cur:
            for table in ("form4_raw_filings", "sm_amendments", "sm_reconciliation_receipts", "sm_run_requests"):
                cur.execute(SQL("SELECT count(*) FROM {}").format(Identifier(table)))
                assert cur.fetchone()[0] == 0
    clock[0] = datetime(2026, 11, 20, 21, 41, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_EXPIRED"):
        pilot.run_cycle()
    assert len(requests_seen) == 32


def test_per_issuer_exhaustion_defers_unfetched_documents_to_next_slot(rehearsal):
    pilot, clock, rows, documents, requests_seen = rehearsal
    rows.extend(filing(i) for i in range(1, 5))
    for row in rows:
        documents[row[1].replace("-", "")] = (200, generate_form4_xml(date="2026-11-20"))
    pilot.run_cycle()
    first = latest_receipt(pilot)
    nvda = verify_retained_evidence(pilot, first)["NVDA"]
    assert nvda["accounted_through"] is None
    assert first["expected_count"] == 4
    assert first["accounted_count"] == 3
    assert first["missing_count"] == 1
    assert nvda["inventory"][-1]["disposition"] == "DEFERRED"
    assert len(requests_seen) == 13

    clock[0] = datetime(2026, 11, 20, 18, 1, tzinfo=timezone.utc)
    pilot.run_cycle()
    second = latest_receipt(pilot)
    nvda = verify_retained_evidence(pilot, second)["NVDA"]
    assert nvda["accounted_through"] == nvda["cutoff"]
    assert second["accounted_count"] == 4
    assert second["missing_count"] == 0
    assert len(requests_seen) == 24


@pytest.mark.parametrize(("failure", "catalog_count", "transmissions"), [
    ("429", 0, 1),
    ("malformed_catalog", 1, 1),
    ("invalid_document", 10, 11),
    ("cycle_budget", 9, 9),
])
def test_failures_keep_reserved_charge_and_catalogs_without_advancing_watermark(
    rehearsal, monkeypatch, failure, catalog_count, transmissions,
):
    pilot, _, rows, documents, requests_seen = rehearsal
    if failure == "429":
        documents[COHORT_CIKS["NVDA"]] = (429, "")
    elif failure == "malformed_catalog":
        documents[COHORT_CIKS["NVDA"]] = (200, "{broken JSON")
    elif failure == "invalid_document":
        rows.append(filing(1))
        documents[rows[0][1].replace("-", "")] = (200, "<ownershipDocument><invalid></ownershipDocument>")
    else:
        monkeypatch.setenv("ARX_SEC_MAX_TOTAL_REQUESTS_PER_CYCLE", "9")

    pilot.run_cycle()
    pilot.run_cycle()
    assert len(requests_seen) == transmissions
    assert latest_receipt(pilot) is None
    assert pilot.repository.get_trial_progress(COHORT_PILOT_ID, "2026-11-20")["requests_trial"] == 40
    with transaction(dsn=pilot.dsn) as cur:
        cur.execute("SELECT count(*) AS n FROM sm_audit_events WHERE event_type = 'SEC_COHORT_CATALOG_OBSERVED'")
        assert cur.fetchone()["n"] == catalog_count
        cur.execute("SELECT status FROM sm_run_requests WHERE request_type = 'TRIAL_SLOT_FORM4'")
        assert cur.fetchone()["status"] == "FAILED"
        cur.execute("SELECT count(*) AS n FROM form4_raw_filings")
        assert cur.fetchone()["n"] == 0


def test_no_send_after_grant_change_or_owner_lock_loss(rehearsal, monkeypatch):
    pilot, clock, _, _, requests_seen = rehearsal
    pilot.run_cycle()
    clock[0] = datetime(2026, 11, 20, 18, 1, tzinfo=timezone.utc)
    monkeypatch.setenv("ARX_CONTINUOUS_AUTH_ID", "c" * 32)
    pilot.run_cycle()
    assert len(requests_seen) == 10
    assert pilot.repository.get_trial_progress(COHORT_PILOT_ID, "2026-11-20")["requests_trial"] == 40

    monkeypatch.setenv("ARX_CONTINUOUS_AUTH_ID", "b" * 32)
    clock[0] = datetime(2026, 11, 20, 21, 31, tzinfo=timezone.utc)
    def lose_lock(_limiter):
        with pilot._owner_conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock_all()")
    monkeypatch.setattr(continuous_pipeline.RateLimiter, "acquire", lose_lock)
    pilot.run_cycle()
    assert len(requests_seen) == 10


def test_cohort_grant_requires_exact_ciks_and_one_ny_day():
    grant = trial_grant(**{
        "ARX_CONTINUOUS_AUTH_SCOPE": SCOPE_TRIAL_10,
        "ARX_CONTINUOUS_AUTH_EXPIRES_AT": "2026-11-21T03:00:00Z",
    })
    now = datetime(2026, 11, 20, 16, tzinfo=timezone.utc)
    issuers = continuous_pipeline.DEFAULT_GOVERNED_ISSUERS
    continuous_authorization.require_continuous_authorization(env=grant, issuers=issuers, now=now)
    bad = [{**issuers[0], "cik": "99999"}, *issuers[1:]]
    with pytest.raises(RuntimeError, match="CONTINUOUS_AUTH_ISSUER_MISMATCH"):
        continuous_authorization.require_continuous_authorization(env=grant, issuers=bad, now=now)
    grant["ARX_CONTINUOUS_AUTH_EXPIRES_AT"] = "2026-11-21T05:00:00Z"
    with pytest.raises(RuntimeError, match="COHORT_PILOT_GRANT_MUST_FIT_ONE_NY_DAY"):
        continuous_authorization.require_continuous_authorization(env=grant, issuers=issuers, now=now)


def test_rejects_canonical_evidence_store_and_pilot_owner_db(rehearsal, monkeypatch):
    pilot, _, _, _, requests_seen = rehearsal
    with monkeypatch.context() as config:
        config.setenv("SMART_MONEY_COHORT_PILOT_DATABASE_URL", pilot.owner_dsn)
        wrong_evidence_db = worker.SmartMoneyIngestionWorker(dsn=pilot.owner_dsn)
        with pytest.raises(RuntimeError, match="COHORT_PILOT_DATABASE_ISOLATION_REQUIRED"):
            wrong_evidence_db.validate_preflight()

    wrong_owner_db = worker.SmartMoneyIngestionWorker(dsn=pilot.dsn)
    with pytest.raises(RuntimeError, match="COHORT_PILOT_DATABASE_ISOLATION_REQUIRED"):
        wrong_owner_db.acquire_egress_authority(max_attempts=1)
    assert requests_seen == []


def test_catalog_without_verified_start_or_historical_traversal():
    cik = COHORT_CIKS["NVDA"]
    body = json.loads(submissions(cik, []))
    body["filings"]["recent"] = {key: [] for key in body["filings"]["recent"]}
    body["filings"]["files"] = [{"name": "older.json", "filingTo": "2026-11-20"}]
    start = datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc)
    cutoff = datetime(2026, 11, 20, 14, 16, tzinfo=timezone.utc)
    inventory, gaps = inventory_from_catalog(json.dumps(body).encode(), cik, start, cutoff)
    assert inventory == []
    assert gaps == ["HISTORICAL_PAGE_MAY_OVERLAP", "RECENT_CATALOG_START_UNVERIFIED"]
