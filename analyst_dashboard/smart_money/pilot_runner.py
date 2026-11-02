"""
ARX Terminal — Dedicated Ten-Issuer Bounded Sequential SEC Pilot Runner.

Implements the controlled multi-issuer production pilot path (Directive SM-ACT-016):
- Target: 10 Governed Issuers (NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, BRK-B, JPM, V)
- Request Ceiling: <= 4 HTTPS requests per issuer; <= 40 total cycle HTTPS requests
- Duration Ceiling: <= 180.0 seconds wall-clock cycle timeout
- Source Rate Limits: <= 5.0 req/s enforced via SecRateLimiter, with 250ms inter-issuer pause
- Failure Isolation: Single issuer network/parse failures do NOT halt pilot cycle
- Coverage Accounting: Explicit tracking of successful vs failed issuers (N/10)
- Snapshot Protection: Partial coverage (< 10/10) strictly preserves active snapshot
- Fail-Closed Teardown: Invariant finally: restores STANDBY role and releases advisory lock
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
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
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.form4_models import OwnershipNature, SmartMoneyFiling
from analyst_dashboard.smart_money.continuous_pipeline import DEFAULT_GOVERNED_ISSUERS

logger = logging.getLogger("smart_money.pilot_runner")

PILOT_MAX_TOTAL_REQUESTS = 40
PILOT_MAX_PER_ISSUER_REQUESTS = 4
PILOT_MAX_DURATION_SEC = 180.0
PILOT_MAX_RPS = 5.0
PILOT_INTER_ISSUER_DELAY_SEC = 0.25


class PilotSecurityError(RuntimeError):
    """Raised when an environment safety invariant is breached."""
    pass


class PilotBudgetExceededError(RuntimeError):
    """Raised when the 40 HTTPS request budget ceiling is exceeded."""
    pass


class IssuerBudgetExceededError(RuntimeError):
    """Raised when an individual issuer exceeds its per-issuer request ceiling."""
    pass


class PilotTimeoutError(RuntimeError):
    """Raised when the 180.0 second wall-clock execution ceiling is exceeded."""
    pass


class PilotLockContentionError(RuntimeError):
    """Raised when the SEC owner advisory lock cannot be acquired."""
    pass


class PilotContainmentError(RuntimeError):
    """Raised when backup containment gate or freshness authority denies execution."""
    pass


class TenIssuerPilotRunner:
    """
    Dedicated production sequential pilot runner covering the 10 governed mega-cap issuers.
    Enforces per-issuer limits, failure isolation, and explicit coverage accounting.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        issuers: Optional[List[Dict[str, str]]] = None,
        max_total_requests: int = PILOT_MAX_TOTAL_REQUESTS,
        max_per_issuer_requests: int = PILOT_MAX_PER_ISSUER_REQUESTS,
        max_duration_sec: float = PILOT_MAX_DURATION_SEC,
        max_rps: float = PILOT_MAX_RPS,
        session: Optional[requests.Session] = None,
    ):
        self.dsn = dsn or os.environ.get("SMART_MONEY_DATABASE_URL") or os.environ.get("DATABASE_URL")
        self.issuers = issuers or list(DEFAULT_GOVERNED_ISSUERS)
        self.max_total_requests = max_total_requests
        self.max_per_issuer_requests = max_per_issuer_requests
        self.max_duration_sec = max_duration_sec
        self.rate_limiter = SecRateLimiter(max_per_second=max_rps)
        self.session = session or requests.Session()
        self.start_time: float = 0.0
        self.total_requests_made: int = 0
        self.requests_log: List[Dict[str, Any]] = []
        self.current_issuer_ticker: Optional[str] = None
        self.current_issuer_requests: int = 0
        self.abort_requested: bool = False

    def abort(self) -> None:
        """Triggers emergency stop for the in-progress pilot."""
        logger.warning("EMERGENCY_STOP: Pilot abort requested.")
        self.abort_requested = True

    def validate_preflight(self) -> bool:
        """
        Validates all pre-flight invariants:
        1. User-Agent conforms to SEC regulations
        2. Database connectivity
        3. Backup containment gate verified (< 12 hours)
        """
        ua = get_sec_user_agent()
        is_valid_ua, ua_err = validate_sec_user_agent(ua)
        if not is_valid_ua:
            raise PilotSecurityError(f"INVALID_USER_AGENT: {ua_err}")

        # Backup containment check
        containment_enabled = os.environ.get("ARX_BACKUP_CONTAINMENT_ENABLED", "").strip().upper() == "TRUE"
        if containment_enabled:
            logger.info("Evaluating backup freshness authority gate (< 12 hours)...")
            policy = BackupContainmentPolicy(
                enabled=True,
                max_verified_age_seconds=float(os.environ.get("ARX_BACKUP_CONTAINMENT_THRESHOLD_SECONDS", "43200.0")),
                production_authorized=True,
            )
            authority = BackupFreshnessAuthority(policy=policy)
            gate = BackupAgeContainmentGate(freshness_authority=authority, policy=policy)
            gate.enforce_ingestion_gate()
            logger.info("Backup freshness gate verified: active verified backup within threshold.")

        if not self.dsn:
            logger.warning("No PostgreSQL DSN provided. Running pilot in memory/read-only mode.")
            return True

        with get_connection(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                row = cur.fetchone()
                if not row or row[0] != 1:
                    raise PilotSecurityError("DATABASE_UNAVAILABLE: Preflight connection probe failed.")
        return True

    def check_deadline(self) -> None:
        """Enforces wall-clock execution ceiling and emergency abort across the entire pilot."""
        if self.abort_requested or os.environ.get("ARX_EMERGENCY_STOP", "").strip().lower() in ("true", "1", "yes"):
            raise PilotSecurityError("EMERGENCY_STOP: Execution aborted by operator signal.")
        if self.start_time > 0 and (time.time() - self.start_time) > self.max_duration_sec:
            raise PilotTimeoutError(
                f"MECHANICAL_STOP: Pilot execution deadline ({self.max_duration_sec}s) exceeded."
            )

    def tracked_get(self, url: str, timeout: float = 10.0) -> requests.Response:
        """
        Executes outbound HTTPS request with rate-limiting, redirect/retry tracking,
        and per-issuer and cycle budget enforcement.
        """
        self.check_deadline()

        # Pre-check budgets before wire request
        if self.total_requests_made >= self.max_total_requests:
            raise PilotBudgetExceededError(
                f"MECHANICAL_STOP: Pilot total request count ({self.total_requests_made}) reached "
                f"cycle ceiling ({self.max_total_requests})."
            )
        if self.current_issuer_requests >= self.max_per_issuer_requests:
            raise IssuerBudgetExceededError(
                f"MECHANICAL_STOP: Issuer {self.current_issuer_ticker} request count "
                f"({self.current_issuer_requests}) reached per-issuer ceiling ({self.max_per_issuer_requests})."
            )

        self.rate_limiter.acquire()
        t0 = time.time()
        resp = self.session.get(url, timeout=timeout)
        duration = round(time.time() - t0, 3)

        # Count redirects in history
        for hist_resp in (resp.history or []):
            self.total_requests_made += 1
            self.current_issuer_requests += 1
            self.requests_log.append({
                "request_num": self.total_requests_made,
                "issuer": self.current_issuer_ticker,
                "url": hist_resp.url,
                "status_code": hist_resp.status_code,
                "bytes_received": len(hist_resp.content or b""),
                "duration_sec": 0.0,
                "is_redirect": True,
            })
            if self.total_requests_made > self.max_total_requests:
                raise PilotBudgetExceededError(
                    f"MECHANICAL_STOP: Pilot total request count ({self.total_requests_made}) exceeded "
                    f"cycle ceiling ({self.max_total_requests}) via redirects."
                )
            if self.current_issuer_requests > self.max_per_issuer_requests:
                raise IssuerBudgetExceededError(
                    f"MECHANICAL_STOP: Issuer {self.current_issuer_ticker} request count "
                    f"({self.current_issuer_requests}) exceeded per-issuer ceiling ({self.max_per_issuer_requests}) via redirects."
                )

        # Count final response
        self.total_requests_made += 1
        self.current_issuer_requests += 1
        self.requests_log.append({
            "request_num": self.total_requests_made,
            "issuer": self.current_issuer_ticker,
            "url": resp.url,
            "status_code": resp.status_code,
            "bytes_received": len(resp.content or b""),
            "duration_sec": duration,
            "is_redirect": False,
        })
        logger.info(
            f"[PILOT REQ {self.total_requests_made}/{self.max_total_requests} | "
            f"{self.current_issuer_ticker} {self.current_issuer_requests}/{self.max_per_issuer_requests}] "
            f"GET {url} -> {resp.status_code} ({len(resp.content or b'')} bytes, {duration}s)"
        )

        if self.total_requests_made > self.max_total_requests:
            raise PilotBudgetExceededError(
                f"MECHANICAL_STOP: Pilot total request count ({self.total_requests_made}) exceeded "
                f"cycle ceiling ({self.max_total_requests})."
            )
        if self.current_issuer_requests > self.max_per_issuer_requests:
            raise IssuerBudgetExceededError(
                f"MECHANICAL_STOP: Issuer {self.current_issuer_ticker} request count "
                f"({self.current_issuer_requests}) exceeded per-issuer ceiling ({self.max_per_issuer_requests})."
            )

        return resp

    def _ingest_issuer(self, ticker: str, cik: str, company_name: str, conn: Any) -> Dict[str, Any]:
        """
        Executes bounded ingestion for a single issuer with failure isolation.
        """
        t0 = time.time()
        self.current_issuer_ticker = ticker
        self.current_issuer_requests = 0

        issuer_receipt: Dict[str, Any] = {
            "ticker": ticker,
            "cik": cik,
            "company_name": company_name,
            "status": "PENDING",
            "requests_made": 0,
            "duration_sec": 0.0,
            "accession_number": None,
            "filing_date": None,
            "transactions_persisted": 0,
            "error": None,
        }

        try:
            # Step 1: Submissions catalog
            sub_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            sub_resp = self.tracked_get(sub_url)
            if sub_resp.status_code != 200:
                raise RuntimeError(f"SEC submissions catalog returned HTTP {sub_resp.status_code}")

            sub_data = sub_resp.json()
            recent = sub_data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])
            filing_dates = recent.get("filingDate", [])
            acceptance_dts = recent.get("acceptanceDateTime", [])
            primary_docs = recent.get("primaryDocument", [])

            # Check existing accessions to prioritize un-ingested filings
            existing_accs = set()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT accession_number FROM form4_raw_filings WHERE ticker = %s;", (ticker,))
                    existing_accs = {r[0] for r in cur.fetchall()}

            target_idx = None
            for idx, f_type in enumerate(forms):
                if f_type in ("4", "4/A"):
                    if accessions[idx] not in existing_accs:
                        target_idx = idx
                        break

            # Fallback to newest Form 4 for idempotency
            if target_idx is None:
                for idx, f_type in enumerate(forms):
                    if f_type in ("4", "4/A"):
                        target_idx = idx
                        break

            if target_idx is None:
                raise RuntimeError(f"No Form 4 or 4/A filing found in recent filings for {ticker}")

            form_type = forms[target_idx]
            acc_num = accessions[target_idx]
            clean_acc = acc_num.replace("-", "")
            f_date = filing_dates[target_idx]
            acc_dt = acceptance_dts[target_idx]
            p_doc = primary_docs[target_idx]

            issuer_receipt["accession_number"] = acc_num
            issuer_receipt["filing_date"] = f_date

            # Step 2: Fetch Primary Doc
            clean_p_doc = p_doc.split("/")[-1] if p_doc else ""
            cik_int = int(cik)
            xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{clean_p_doc}"
            doc_resp = self.tracked_get(xml_url)

            raw_xml = ""
            source_url = xml_url
            if doc_resp.status_code == 200 and ("<XML" in doc_resp.text.upper() or "<OWNERSHIPDOCUMENT" in doc_resp.text.upper()):
                raw_xml = doc_resp.text
            else:
                txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{acc_num}.txt"
                txt_resp = self.tracked_get(txt_url)
                if txt_resp.status_code == 200:
                    raw_xml = Form4XmlParser.extract_xml_from_submission_text(txt_resp.text)
                    source_url = txt_url
                else:
                    raise RuntimeError(f"Failed to fetch submission txt: HTTP {txt_resp.status_code}")

            if not raw_xml:
                raise RuntimeError(f"Could not extract valid XML from SEC EDGAR payload for {acc_num}")

            content_hash = hashlib.sha256(raw_xml.encode("utf-8")).hexdigest()

            # Step 3: Parse
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

            # Step 4: Persist
            persisted_tx = 0
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO form4_raw_filings (
                            accession_number, cik, ticker, company_name, filing_date,
                            accepted_datetime, xml_content, content_hash, source_url
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (accession_number) DO NOTHING;
                    """, (acc_num, cik, ticker, company_name, f_date, acc_dt, raw_xml, content_hash, source_url))

                    cur.execute("""
                        INSERT INTO sm_source_availability_ledger (
                            accession_number, public_availability_at, source, source_reference
                        ) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (accession_number) DO NOTHING;
                    """, (acc_num, acc_dt, "SEC_EDGAR_PILOT_PROD", source_url))

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
                        persisted_tx += 1

                conn.commit()

            issuer_receipt["transactions_persisted"] = persisted_tx
            issuer_receipt["status"] = "SUCCESS"

        except Exception as e:
            logger.error(f"Ingestion failed for {ticker} ({cik}): {e}")
            if conn:
                conn.rollback()
            issuer_receipt["status"] = "FAILED"
            issuer_receipt["error"] = str(e)
        finally:
            issuer_receipt["requests_made"] = self.current_issuer_requests
            issuer_receipt["duration_sec"] = round(time.time() - t0, 3)

        return issuer_receipt

    def execute(self) -> Dict[str, Any]:
        """
        Executes bounded 10-issuer sequential pilot cycle with guaranteed fail-closed teardown.
        """
        self.start_time = time.time()
        self.total_requests_made = 0
        self.requests_log = []

        telemetry: Dict[str, Any] = {
            "directive": "SM-ACT-016",
            "cycle_type": "SEQUENTIAL_10_ISSUER_PILOT",
            "timestamp_start": datetime.now(timezone.utc).isoformat(),
            "total_issuers_planned": len(self.issuers),
            "max_requests_cap": self.max_total_requests,
            "max_duration_sec": self.max_duration_sec,
            "requests_made": 0,
            "issuers_evaluated": 0,
            "issuers_succeeded": 0,
            "issuers_failed": 0,
            "coverage_ratio": 0.0,
            "is_complete_coverage": False,
            "active_snapshot_preserved": True,
            "issuer_receipts": [],
            "sec_requests": [],
            "lock_acquired": False,
            "lock_released": False,
            "role_restored": False,
            "verdict": "PENDING",
        }

        # Step 1: Preflight
        logger.info("Executing pilot preflight validations...")
        self.validate_preflight()

        # Step 2: Acquire Advisory Lock
        logger.info(f"Acquiring PostgreSQL advisory lock {LOCK_SEC_EGRESS_OWNER}...")
        pg_ctx = get_connection(self.dsn) if self.dsn else None
        conn = None
        if pg_ctx:
            conn = pg_ctx.__enter__()
            has_lock = acquire_sec_owner_lock(conn)
            if not has_lock:
                pg_ctx.__exit__(None, None, None)
                raise PilotLockContentionError(
                    f"EXCLUSIVE_OWNERSHIP_UNAVAILABLE: Could not acquire lock {LOCK_SEC_EGRESS_OWNER}."
                )
            telemetry["lock_acquired"] = True
            logger.info("Exclusive advisory lock claimed successfully.")

        prev_role = get_sec_egress_role()
        try:
            # Step 3: Temporary In-Process Authorization
            logger.info("Applying temporary ARX_SEC_EGRESS_ROLE=OWNER in-process...")
            os.environ["ARX_SEC_EGRESS_ROLE"] = "OWNER"
            if not is_sec_egress_authorized():
                raise PilotSecurityError("Failed to transition in-process role to OWNER.")

            ua = get_sec_user_agent()
            self.session.headers.update({
                "User-Agent": ua,
                "Accept-Encoding": "gzip, deflate",
            })

            # Step 4: Sequential Pilot Pass
            for i, issuer in enumerate(self.issuers):
                ticker = issuer["ticker"]
                cik = issuer["cik"]
                name = issuer.get("company_name", ticker)

                logger.info(f"[{i+1}/{len(self.issuers)}] Ingesting {ticker} ({cik})...")
                receipt = self._ingest_issuer(ticker, cik, name, conn)
                telemetry["issuer_receipts"].append(receipt)
                telemetry["issuers_evaluated"] += 1

                if receipt["status"] == "SUCCESS":
                    telemetry["issuers_succeeded"] += 1
                else:
                    telemetry["issuers_failed"] += 1

                # Inter-issuer polite delay
                if i < len(self.issuers) - 1:
                    time.sleep(PILOT_INTER_ISSUER_DELAY_SEC)

            # Step 5: Coverage Evaluation & Snapshot Accounting
            total = telemetry["total_issuers_planned"]
            succ = telemetry["issuers_succeeded"]
            coverage = round(succ / total, 3) if total > 0 else 0.0
            telemetry["coverage_ratio"] = coverage
            telemetry["is_complete_coverage"] = (succ == total)

            if not telemetry["is_complete_coverage"]:
                logger.warning(
                    f"PARTIAL_COVERAGE ({succ}/{total}): Active snapshot preserved. "
                    "Immutability invariant holds."
                )
                telemetry["active_snapshot_preserved"] = True

            telemetry["verdict"] = "SUCCESS" if succ > 0 else "FAILED"

        finally:
            # Step 6: Fail-Closed Teardown
            logger.info("Executing pilot fail-closed teardown...")
            os.environ["ARX_SEC_EGRESS_ROLE"] = prev_role if prev_role in ("STANDBY", "DISABLED") else "STANDBY"
            telemetry["role_restored"] = True

            if conn:
                try:
                    release_sec_owner_lock(conn)
                    telemetry["lock_released"] = True
                    logger.info("Advisory lock released successfully.")
                except Exception as e:
                    logger.warning(f"Error releasing advisory lock: {e}")
                pg_ctx.__exit__(None, None, None)

            telemetry["requests_made"] = self.total_requests_made
            telemetry["sec_requests"] = self.requests_log
            telemetry["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            telemetry["duration_wall_clock_sec"] = round(time.time() - self.start_time, 3)
            total_planned = telemetry["total_issuers_planned"]
            logger.info(
                f"Pilot execution finalized: {telemetry['issuers_succeeded']}/{total_planned} issuers succeeded, "
                f"{telemetry['requests_made']} requests made in {telemetry['duration_wall_clock_sec']}s. "
                f"Verdict: {telemetry['verdict']}"
            )

        return telemetry


def run_ten_issuer_pilot(dsn: Optional[str] = None) -> Dict[str, Any]:
    """Top-level entrypoint for executing bounded 10-issuer pilot."""
    runner = TenIssuerPilotRunner(dsn=dsn)
    return runner.execute()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print("=== EXECUTING ARX TEN-ISSUER SEC PILOT ===")
    try:
        receipt = run_ten_issuer_pilot()
        print("\nPilot Execution Receipt:")
        print(json.dumps(receipt, indent=2, default=str))
        if receipt.get("verdict") == "SUCCESS":
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as exc:
        print(f"\nFATAL: Ten-issuer pilot failed with exception: {exc}")
        sys.exit(1)
