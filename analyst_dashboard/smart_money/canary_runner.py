"""
ARX Terminal — Dedicated NVDA-Only Bounded SEC Ingestion Canary Runner.

Implements the surgical, bounded production canary path (SM-ACT-014 / SM-ACT-014R):
- Target: Strictly NVDA (CIK 0001045810)
- Request Ceiling: <= 4 total HTTPS requests strictly enforced at HTTP boundary
- Duration Ceiling: <= 30.0 seconds wall-clock timeout
- Source Rate Limits: <= 8.0 req/s enforced via SecRateLimiter
- Queue Isolation: Zero interaction with sm_run_requests operator queue
- Snapshot Protection: 1/10 partial coverage does NOT overwrite active 10-issuer snapshot
- Fail-Closed Teardown: Invariant finally: restores STANDBY role, false authorization, and releases advisory lock
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import requests

from analyst_dashboard.smart_money.storage.postgres_connection import get_connection
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    acquire_sec_owner_lock,
    release_sec_owner_lock,
    LOCK_SEC_EGRESS_OWNER,
)
from analyst_dashboard.smart_money.storage.postgres_repository import PostgresSmartMoneyRepository
from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    BackupFreshnessAuthority,
    BackupContainmentPolicy,
)
from analyst_dashboard.smart_money.storage.backup.containment_gate import BackupAgeContainmentGate
from analyst_dashboard.data.sec_edgar_fetcher import (
    SecRateLimiter,
    validate_sec_user_agent,
    get_sec_user_agent,
    get_sec_egress_role,
    is_sec_egress_authorized,
    SecEdgarFetcher,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.form4_models import OwnershipNature, SmartMoneyFiling

logger = logging.getLogger("smart_money.canary_runner")

CANARY_TICKER = "NVDA"
CANARY_CIK = "0001045810"
CANARY_MAX_REQUESTS = 4
CANARY_MAX_DURATION_SEC = 30.0
CANARY_MAX_RPS = 8.0


class CanarySecurityError(RuntimeError):
    """Raised when an environment safety invariant is breached."""
    pass


class CanaryBudgetExceededError(RuntimeError):
    """Raised when the 4 HTTPS request budget ceiling is exceeded."""
    pass


class CanaryTimeoutError(RuntimeError):
    """Raised when the 30.0 second wall-clock execution ceiling is exceeded."""
    pass


class CanaryLockContentionError(RuntimeError):
    """Raised when the SEC owner advisory lock cannot be acquired."""
    pass


class CanaryContainmentError(RuntimeError):
    """Raised when backup containment gate or freshness authority denies execution."""
    pass


class NvdaCanaryRunner:
    """
    Dedicated single-shot production Form 4 canary runner for NVDA.
    Guarantees isolated execution without touching operator queues or expanding universe.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        max_requests: int = CANARY_MAX_REQUESTS,
        max_duration_sec: float = CANARY_MAX_DURATION_SEC,
        max_rps: float = CANARY_MAX_RPS,
        session: Optional[requests.Session] = None,
    ):
        self.dsn = dsn or os.environ.get("SMART_MONEY_DATABASE_URL") or os.environ.get("DATABASE_URL")
        self.max_requests = max_requests
        self.max_duration_sec = max_duration_sec
        self.max_rps = max_rps
        self.rate_limiter = SecRateLimiter(max_per_second=self.max_rps)
        self.session = session or requests.Session()
        self.request_count = 0
        self.start_time = 0.0
        self.requests_log: List[Dict[str, Any]] = []

    def validate_preflight(self, ticker: str = CANARY_TICKER, cik: str = CANARY_CIK) -> bool:
        """
        Validates all fail-closed preflight gates prior to claiming lock or executing network requests:
        1. Issuer Scope: NVDA / 0001045810 only.
        2. Emergency Stop: ARX_EMERGENCY_STOP_ACTIVE must not be TRUE.
        3. Containment Ratification: ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED must be TRUE.
        4. User Agent: SEC_USER_AGENT must be valid non-placeholder.
        5. Backup Freshness: BackupAgeContainmentGate must verify fresh backup (< 12 hours).
        """
        # 1. Target scope constraint
        if ticker.upper().strip() != CANARY_TICKER or str(cik).zfill(10) != CANARY_CIK:
            raise CanarySecurityError(
                f"UNAUTHORIZED_CANARY_SCOPE: Canary runner is strictly locked to {CANARY_TICKER} "
                f"(CIK {CANARY_CIK}). Attempted: ticker='{ticker}', cik='{cik}'."
            )

        # 2. Emergency stop check
        if os.environ.get("ARX_EMERGENCY_STOP_ACTIVE", "").strip().upper() == "TRUE":
            raise CanarySecurityError("EMERGENCY_STOP_ACTIVE: Canary execution forbidden.")

        # 3. Containment ratification check
        is_auth = os.environ.get("ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED", "").strip().lower() in ("true", "1", "yes")
        if not is_auth:
            raise CanaryContainmentError(
                "UNAUTHORIZED_PRODUCTION_BACKUP_CONTAINMENT: Canary execution requires explicit "
                "ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED=True ratification."
            )

        # 4. User agent check
        ua = get_sec_user_agent()
        is_ua_valid, ua_err = validate_sec_user_agent(ua)
        if not is_ua_valid:
            raise CanarySecurityError(f"INVALID_SEC_USER_AGENT: {ua_err}")

        # 5. Backup freshness containment check
        policy = BackupContainmentPolicy.from_environment()
        gate = BackupAgeContainmentGate(policy=policy)
        if not gate.check_ingestion_eligibility():
            raise CanaryContainmentError(
                f"INGESTION_CONTAINMENT_BREACH: Ingestion suspended due to backup staleness "
                f"or unverifiability: {gate.last_error_code}"
            )

        return True

    def check_deadline(self) -> None:
        """Enforces wall-clock execution ceiling."""
        if self.start_time > 0 and (time.time() - self.start_time) > self.max_duration_sec:
            raise CanaryTimeoutError(
                f"MECHANICAL_STOP: Canary execution deadline ({self.max_duration_sec}s) exceeded."
            )

    def tracked_get(self, url: str, timeout: float = 10.0) -> requests.Response:
        """
        Executes outbound HTTPS request with rate-limiting and strict budget enforcement.
        Enforced at the physical HTTP boundary.
        """
        self.check_deadline()
        self.request_count += 1
        if self.request_count > self.max_requests:
            raise CanaryBudgetExceededError(
                f"MECHANICAL_STOP: Canary request count ({self.request_count}) exceeded "
                f"strict cap ({self.max_requests})."
            )

        self.rate_limiter.acquire()
        t0 = time.time()
        resp = self.session.get(url, timeout=timeout)
        duration = round(time.time() - t0, 3)

        self.requests_log.append({
            "request_num": self.request_count,
            "url": url,
            "status_code": resp.status_code,
            "bytes_received": len(resp.content),
            "duration_sec": duration,
        })
        logger.info(
            f"[CANARY REQ {self.request_count}/{self.max_requests}] "
            f"GET {url} -> {resp.status_code} ({len(resp.content)} bytes, {duration}s)"
        )
        return resp

    def execute(self) -> Dict[str, Any]:
        """
        Executes the bounded, single-shot NVDA canary with guaranteed teardown.
        """
        self.start_time = time.time()
        self.request_count = 0
        self.requests_log = []

        telemetry: Dict[str, Any] = {
            "directive": "SM-ACT-014R",
            "timestamp_start": datetime.now(timezone.utc).isoformat(),
            "issuer": CANARY_TICKER,
            "cik": CANARY_CIK,
            "request_budget": self.max_requests,
            "requests_made": 0,
            "sec_requests": [],
            "filings_discovered": [],
            "transactions_persisted": 0,
            "errors": [],
            "lock_acquired": False,
            "lock_released": False,
            "role_restored": False,
            "containment_auth_revoked": False,
            "active_snapshot_preserved": True,
            "verdict": "PENDING",
        }

        # Step 1: Preflight Verification
        logger.info("Executing preflight validations...")
        self.validate_preflight()

        # Step 2: Acquire Dedicated Advisory Lock
        logger.info(f"Acquiring PostgreSQL advisory lock {LOCK_SEC_EGRESS_OWNER}...")
        pg_ctx = get_connection(self.dsn) if self.dsn else None
        conn = None
        if pg_ctx:
            conn = pg_ctx.__enter__()
            has_lock = acquire_sec_owner_lock(conn)
            if not has_lock:
                pg_ctx.__exit__(None, None, None)
                raise CanaryLockContentionError(
                    f"EXCLUSIVE_OWNERSHIP_UNAVAILABLE: Could not acquire lock {LOCK_SEC_EGRESS_OWNER}."
                )
            telemetry["lock_acquired"] = True
            logger.info("Exclusive advisory lock claimed successfully.")

        try:
            # Step 3: Temporary In-Process Egress Authorization
            logger.info("Applying temporary ARX_SEC_EGRESS_ROLE=OWNER in-process...")
            os.environ["ARX_SEC_EGRESS_ROLE"] = "OWNER"
            if not is_sec_egress_authorized():
                raise CanarySecurityError("Failed to transition in-process role to OWNER.")

            ua = get_sec_user_agent()
            self.session.headers.update({
                "User-Agent": ua,
                "Accept-Encoding": "gzip, deflate",
            })

            # Step 4: Fetch NVDA Submissions Metadata (CIK 0001045810)
            sub_url = f"https://data.sec.gov/submissions/CIK{CANARY_CIK}.json"
            logger.info(f"Fetching submissions catalog: {sub_url}")
            sub_resp = self.tracked_get(sub_url)
            if sub_resp.status_code != 200:
                raise RuntimeError(f"SEC EDGAR submissions returned HTTP {sub_resp.status_code}")

            sub_data = sub_resp.json()
            recent = sub_data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])
            filing_dates = recent.get("filingDate", [])
            acceptance_dts = recent.get("acceptanceDateTime", [])
            primary_docs = recent.get("primaryDocument", [])

            # Step 5: Locate Target Form 4 Filing
            existing_accs = set()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT accession_number FROM form4_raw_filings WHERE ticker = 'NVDA';")
                    existing_accs = {r[0] for r in cur.fetchall()}

            target_idx = None
            # Prefer newest Form 4 not yet in database
            for idx, f_type in enumerate(forms):
                if f_type in ("4", "4/A"):
                    if accessions[idx] not in existing_accs:
                        target_idx = idx
                        break

            # Fallback to newest Form 4 for idempotency test if all exist
            if target_idx is None:
                for idx, f_type in enumerate(forms):
                    if f_type in ("4", "4/A"):
                        target_idx = idx
                        break

            if target_idx is None:
                raise RuntimeError(f"No Form 4 or 4/A filing found in recent filings for {CANARY_TICKER}!")

            form_type = forms[target_idx]
            acc_num = accessions[target_idx]
            clean_acc = acc_num.replace("-", "")
            f_date = filing_dates[target_idx]
            acc_dt = acceptance_dts[target_idx]
            p_doc = primary_docs[target_idx]

            logger.info(
                f"Selected canary filing: {form_type} {acc_num} "
                f"(filing_date={f_date}, accepted={acc_dt}, doc={p_doc})"
            )

            # Step 6: Fetch Primary XML Document
            clean_p_doc = p_doc.split("/")[-1] if p_doc else ""
            cik_int = int(CANARY_CIK)
            xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{clean_p_doc}"
            doc_resp = self.tracked_get(xml_url)

            raw_xml = ""
            source_url = xml_url
            if doc_resp.status_code == 200 and ("<XML" in doc_resp.text.upper() or "<OWNERSHIPDOCUMENT" in doc_resp.text.upper()):
                raw_xml = doc_resp.text
            else:
                txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{acc_num}.txt"
                logger.info(f"Primary doc not standalone XML; fetching full submission: {txt_url}")
                txt_resp = self.tracked_get(txt_url)
                if txt_resp.status_code == 200:
                    raw_xml = Form4XmlParser.extract_xml_from_submission_text(txt_resp.text)
                    source_url = txt_url
                else:
                    raise RuntimeError(f"Failed to fetch submission txt: HTTP {txt_resp.status_code}")

            if not raw_xml:
                raise RuntimeError(f"Could not extract valid XML from SEC EDGAR payload for {acc_num}")

            content_hash = hashlib.sha256(raw_xml.encode("utf-8")).hexdigest()
            logger.info(f"Extracted Form 4 XML: {len(raw_xml)} chars, sha256: {content_hash}")

            # Step 7: Parse Filing and Semantics
            filing_obj = Form4XmlParser.parse_filing(
                raw_content=raw_xml,
                accession_number=acc_num,
                filing_date_hint=f_date,
                acceptance_datetime=acc_dt,
                source_url=source_url,
                source_fetch_time=datetime.now(timezone.utc).isoformat(),
            )
            filing_obj.source_content_hash = content_hash
            filing_obj.document_type = form_type

            logger.info(f"Parsed {len(filing_obj.transactions)} transaction(s) from {acc_num}")

            # Step 8: Persist to PostgreSQL Idempotently
            persisted_tx_count = 0
            if conn:
                with conn.cursor() as cur:
                    # 1. Raw filing persistence (append-only)
                    cur.execute("""
                        INSERT INTO form4_raw_filings (
                            accession_number, cik, ticker, company_name, filing_date,
                            accepted_datetime, xml_content, content_hash, source_url
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (accession_number) DO NOTHING;
                    """, (acc_num, CANARY_CIK, CANARY_TICKER, "NVIDIA CORP", f_date, acc_dt, raw_xml, content_hash, source_url))

                    # 2. Availability ledger
                    cur.execute("""
                        INSERT INTO sm_source_availability_ledger (
                            accession_number, public_availability_at, source, source_reference
                        ) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (accession_number) DO NOTHING;
                    """, (acc_num, acc_dt, "SEC_EDGAR_LIVE_PROD", source_url))

                    # 3. Parsed transactions with Schema V2 classification
                    for tx in filing_obj.transactions:
                        ownership_str = "D" if tx.ownership_nature == OwnershipNature.DIRECT else "I"
                        sig_str = tx.signal_type.value if hasattr(tx.signal_type, "value") else str(tx.signal_type)

                        cur.execute("""
                            INSERT INTO form4_parsed_transactions (
                                transaction_id, accession_number, transaction_date, transaction_code,
                                shares, price_per_share, total_value_usd, reporting_owner_name,
                                reporting_owner_role, is_director, is_officer, is_ten_percent,
                                officer_title, ownership_type, is_derivative, signal_type,
                                acquired_disposed, is_positive_buying_conviction, is_open_market_purchase_evidence
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            )
                            ON CONFLICT (transaction_id) DO UPDATE SET
                                shares = EXCLUDED.shares,
                                price_per_share = EXCLUDED.price_per_share,
                                total_value_usd = EXCLUDED.total_value_usd,
                                is_derivative = EXCLUDED.is_derivative,
                                signal_type = EXCLUDED.signal_type,
                                acquired_disposed = EXCLUDED.acquired_disposed,
                                is_positive_buying_conviction = EXCLUDED.is_positive_buying_conviction,
                                is_open_market_purchase_evidence = EXCLUDED.is_open_market_purchase_evidence;
                        """, (
                            tx.record_key, acc_num, tx.transaction_date, tx.transaction_code,
                            tx.shares, tx.price_per_share, tx.total_value_usd, tx.reporting_owner_name,
                            tx.officer_title or ("Director" if tx.is_director else "Insider"),
                            tx.is_director, tx.is_officer, tx.is_ten_percent_owner, tx.officer_title,
                            ownership_str, bool(tx.is_derivative), sig_str, tx.acquired_disposed,
                            bool(tx.is_positive_buying_conviction), bool(tx.is_open_market_purchase_evidence)
                        ))
                        persisted_tx_count += 1

                    # 4. Amendment ledger if 4/A
                    if filing_obj.is_amendment or form_type in ("4/A", "4A"):
                        base_acc = filing_obj.amendment_of_accession
                        rel_status = "PARTIAL_CORRECTION" if base_acc else "AMENDMENT_TARGET_AMBIGUOUS"
                        cur.execute("""
                            INSERT INTO sm_amendments (
                                amendment_accession, base_accession, amendment_number, resolution_status, resolved_at
                            ) VALUES (%s, %s, %s, %s, NOW())
                            ON CONFLICT (amendment_accession) DO UPDATE SET
                                base_accession = EXCLUDED.base_accession,
                                resolution_status = EXCLUDED.resolution_status;
                        """, (acc_num, base_acc, 1, rel_status))

                    # 5. Audit event
                    ev_payload = json.dumps({
                        "ticker": CANARY_TICKER,
                        "accession_number": acc_num,
                        "form_type": form_type,
                        "transactions_count": persisted_tx_count,
                        "content_hash": content_hash,
                    })
                    ev_hash = hashlib.sha256(ev_payload.encode("utf-8")).hexdigest()
                    cur.execute("""
                        INSERT INTO sm_audit_events (
                            event_id, event_type, subject_id, payload, event_hash
                        ) VALUES (%s, %s, %s, %s, %s);
                    """, (
                        str(uuid.uuid4()),
                        "CANARY_INGESTION_COMPLETED",
                        acc_num,
                        ev_payload,
                        ev_hash,
                    ))

                conn.commit()
                logger.info(f"Committed {acc_num} and {persisted_tx_count} transaction(s) to PostgreSQL.")

            # Record telemetry data
            telemetry["requests_made"] = self.request_count
            telemetry["sec_requests"] = self.requests_log
            telemetry["filings_discovered"].append({
                "accession_number": acc_num,
                "form_type": form_type,
                "filing_date": f_date,
                "acceptance_datetime": acc_dt,
                "content_hash": content_hash,
                "source_url": source_url,
                "transactions_count": len(filing_obj.transactions),
            })
            telemetry["transactions_persisted"] = persisted_tx_count
            telemetry["verdict"] = "SUCCESS"

        except Exception as e:
            logger.error(f"Canary execution encountered error: {e}", exc_info=True)
            telemetry["errors"].append(str(e))
            telemetry["verdict"] = "FAILED"
            raise

        finally:
            # Step 9: INVARIANT TEARDOWN (Guaranteed fail-closed restoration)
            logger.info("Executing invariant fail-closed teardown...")

            # 1. Revert in-process role to STANDBY
            os.environ["ARX_SEC_EGRESS_ROLE"] = "STANDBY"
            telemetry["role_restored"] = True

            # 2. Revoke containment authorization
            os.environ["ARX_BACKUP_CONTAINMENT_PRODUCTION_AUTHORIZED"] = "false"
            telemetry["containment_auth_revoked"] = True

            # 3. Release exclusive advisory lock
            if conn:
                try:
                    release_sec_owner_lock(conn)
                    telemetry["lock_released"] = True
                except Exception as lock_err:
                    logger.warning(f"Advisory lock release warning: {lock_err}")
                finally:
                    if pg_ctx:
                        pg_ctx.__exit__(None, None, None)

            # 4. Verify post-teardown egress blockage
            try:
                probe_fetcher = SecEdgarFetcher()
                probe_fetcher._get("https://data.sec.gov/probe")
                raise CanarySecurityError("SECURITY_BREACH: Outbound SEC request succeeded after teardown!")
            except PermissionError as pe:
                if "SEC_EGRESS_FORBIDDEN" in str(pe):
                    logger.info("Post-teardown egress blockage verified: SEC_EGRESS_FORBIDDEN.")
                else:
                    raise

            telemetry["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            telemetry["duration_seconds"] = round(time.time() - self.start_time, 3)

        return telemetry


def run_nvda_canary(dsn: Optional[str] = None) -> Dict[str, Any]:
    """Convenience CLI entrypoint for running the bounded NVDA canary."""
    runner = NvdaCanaryRunner(dsn=dsn)
    return runner.execute()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print("=== EXECUTING ARX NVDA SEC CANARY ===")
    try:
        receipt = run_nvda_canary()
        print("\nCanary Execution Receipt:")
        print(json.dumps(receipt, indent=2, default=str))
        if receipt.get("verdict") == "SUCCESS":
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as exc:
        print(f"\nFATAL: Canary failed with exception: {exc}")
        sys.exit(1)
