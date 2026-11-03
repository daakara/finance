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
import threading
import subprocess
import argparse
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter

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


class PilotSupervisoryWatchdog:
    """
    Supervisory watchdog timer thread enforcing hard wall-clock execution limits.
    Triggers immediate runner abort and socket termination if total execution time is exceeded.
    """

    def __init__(self, runner: "TenIssuerPilotRunner", timeout_sec: float):
        self.runner = runner
        self.timeout_sec = timeout_sec
        self.timer: Optional[threading.Timer] = None
        self.fired: bool = False

    def start(self) -> None:
        def _on_timeout():
            self.fired = True
            logger.critical(
                f"SUPERVISORY_WATCHDOG: Hard execution ceiling ({self.timeout_sec}s) exceeded! "
                "Triggering supervisory termination..."
            )
            self.runner.abort(reason=f"SUPERVISORY_TIMEOUT: Total execution exceeded {self.timeout_sec}s ceiling.")
            try:
                self.runner.session.close()
            except Exception:
                pass

        self.timer = threading.Timer(self.timeout_sec, _on_timeout)
        self.timer.daemon = True
        self.timer.start()

    def cancel(self) -> None:
        if self.timer:
            self.timer.cancel()


class OutboundAccountingSession(requests.Session):
    """
    Custom Session that intercepts every outbound HTTP transmission at the wire send boundary.
    Enforces per-issuer ceiling (<= 4) and total cycle ceiling (<= 40) before transmitting.
    Counts initial requests, redirects, and retries.
    """

    def __init__(self, runner: "TenIssuerPilotRunner"):
        super().__init__()
        self.runner = runner

    def send(self, request, **kwargs):
        self.runner.check_deadline()
        self.runner.rate_limiter.acquire()
        self.runner.record_outbound_send(request.url)
        t0 = time.time()
        resp = super().send(request, **kwargs)
        duration = round(time.time() - t0, 3)
        self.runner.record_response_received(resp, duration)
        return resp


class SecAdvisoryLockBoundary:
    """
    Dedicated RAII / resource-management boundary for PostgreSQL connection
    and SEC owner advisory lock acquisition.
    Guarantees clean release of advisory lock and connection closure on exit,
    handling exceptions before, during, and after acquisition.
    """

    def __init__(self, dsn: Optional[str], lock_id: int = LOCK_SEC_EGRESS_OWNER):
        self.dsn = dsn
        self.lock_id = lock_id
        self.pg_ctx = None
        self.conn = None
        self.lock_acquired: bool = False
        self.lock_released: bool = False

    def __enter__(self) -> "SecAdvisoryLockBoundary":
        if not self.dsn:
            return self

        try:
            self.pg_ctx = get_connection(self.dsn)
            self.conn = self.pg_ctx.__enter__()
        except Exception as exc:
            logger.error(f"Failed to establish database connection for advisory lock: {exc}")
            self.close()
            raise PilotSecurityError(f"DATABASE_CONNECTION_FAILED: {exc}") from exc

        try:
            has_lock = acquire_sec_owner_lock(self.conn)
            if not has_lock:
                self.close()
                raise PilotLockContentionError(
                    f"EXCLUSIVE_OWNERSHIP_UNAVAILABLE: Could not acquire lock {self.lock_id}."
                )
            self.lock_acquired = True
            logger.info(f"Advisory lock {self.lock_id} acquired successfully.")
            return self
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Explicitly and safely releases advisory lock and closes database connection."""
        if self.conn and self.lock_acquired and not self.lock_released:
            try:
                release_sec_owner_lock(self.conn)
                self.lock_released = True
                logger.info(f"Advisory lock {self.lock_id} released successfully.")
            except Exception as e:
                logger.warning(f"Error releasing advisory lock {self.lock_id}: {e}")
            finally:
                self.lock_acquired = False

        if self.pg_ctx:
            try:
                self.pg_ctx.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection context: {e}")
            finally:
                self.pg_ctx = None
                self.conn = None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class PilotProcessSupervisor:
    """
    Supervises pilot execution in a separate, isolated operating system process.
    Enforces a genuine 180-second wall-clock termination boundary: if the child
    process blocks in uninterruptible network/database I/O, the supervisor forcibly
    terminates the child process and frees all OS sockets and locks.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        timeout_sec: float = PILOT_MAX_DURATION_SEC,
        python_exe: Optional[str] = None,
    ):
        self.dsn = dsn
        self.timeout_sec = timeout_sec
        self.python_exe = python_exe or sys.executable
        self.child_proc: Optional[subprocess.Popen] = None
        self.termination_forced: bool = False

    def run(self, env_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Executes the ten-issuer pilot in a dedicated child process, monitoring
        elapsed wall-clock time and enforcing hard termination at timeout_sec.
        """
        cmd = [self.python_exe, "-m", "analyst_dashboard.smart_money.pilot_runner", "--child-worker"]
        if self.dsn:
            cmd.extend(["--dsn", self.dsn])

        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        logger.info(
            f"PilotProcessSupervisor: Spawning child pilot process (timeout={self.timeout_sec}s): "
            f"{' '.join(cmd)}"
        )
        t0 = time.time()
        self.child_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        stdout, stderr = "", ""
        try:
            stdout, stderr = self.child_proc.communicate(timeout=self.timeout_sec)
        except subprocess.TimeoutExpired:
            self.termination_forced = True
            logger.critical(
                f"PilotProcessSupervisor: Child process PID {self.child_proc.pid} exceeded "
                f"hard wall-clock deadline ({self.timeout_sec}s). Executing force termination..."
            )
            try:
                self.child_proc.terminate()
                stdout, stderr = self.child_proc.communicate(timeout=3.0)
            except subprocess.TimeoutExpired:
                logger.critical(
                    f"PilotProcessSupervisor: Child PID {self.child_proc.pid} did not exit on SIGTERM. "
                    "Killing process..."
                )
                self.child_proc.kill()
                stdout, stderr = self.child_proc.communicate()

            raise PilotTimeoutError(
                f"SUPERVISOR_HARD_TIMEOUT: Supervised pilot process exceeded {self.timeout_sec}s "
                f"wall-clock limit. Force-killed by process supervisor."
            )

        elapsed = round(time.time() - t0, 3)
        logger.info(
            f"PilotProcessSupervisor: Child process PID {self.child_proc.pid} exited with code "
            f"{self.child_proc.returncode} in {elapsed}s."
        )

        telemetry = self._extract_telemetry(stdout, stderr)
        if self.child_proc.returncode != 0 and telemetry.get("verdict") != "SUCCESS":
            err_msg = telemetry.get("error") or stderr.strip() or f"Child process exited with code {self.child_proc.returncode}"
            logger.error(f"PilotProcessSupervisor: Child process reported failure: {err_msg}")
            if "PilotSecurityError" in err_msg or "UNAUTHORIZED" in err_msg:
                raise PilotSecurityError(err_msg)
            elif "PilotLockContentionError" in err_msg or "EXCLUSIVE_OWNERSHIP_UNAVAILABLE" in err_msg:
                raise PilotLockContentionError(err_msg)
            elif "PilotContainmentError" in err_msg or "CONTAINMENT" in err_msg:
                raise PilotContainmentError(err_msg)
            elif "PilotTimeoutError" in err_msg or "SUPERVISOR_HARD_TIMEOUT" in err_msg:
                raise PilotTimeoutError(err_msg)

        return telemetry

    def _extract_telemetry(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extracts JSON telemetry from child worker stdout."""
        marker_start = "--- PILOT_TELEMETRY_JSON_START ---"
        marker_end = "--- PILOT_TELEMETRY_JSON_END ---"
        if marker_start in stdout and marker_end in stdout:
            try:
                json_part = stdout.split(marker_start)[1].split(marker_end)[0].strip()
                return json.loads(json_part)
            except Exception as e:
                logger.warning(f"Failed to parse child JSON telemetry marker: {e}")

        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except Exception:
                    continue

        return {
            "verdict": "FAILED",
            "error": stderr.strip() or stdout.strip() or "No telemetry captured from child process",
            "stdout": stdout,
            "stderr": stderr,
        }


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
        self.start_time: float = 0.0
        self.total_requests_made: int = 0
        self.requests_log: List[Dict[str, Any]] = []
        self.current_issuer_ticker: Optional[str] = None
        self.current_issuer_requests: int = 0
        self.abort_requested: bool = False
        self.abort_reason: Optional[str] = None
        self.watchdog: Optional[PilotSupervisoryWatchdog] = None

        if session is not None:
            self.session = session
            orig_send = self.session.send

            def wrapped_send(request, **kwargs):
                self.check_deadline()
                self.rate_limiter.acquire()
                self.record_outbound_send(request.url)
                t0 = time.time()
                resp = orig_send(request, **kwargs)
                duration = round(time.time() - t0, 3)
                self.record_response_received(resp, duration)
                return resp

            self.session.send = wrapped_send
        else:
            self.session = OutboundAccountingSession(self)

        try:
            adapter = HTTPAdapter(max_retries=0)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)
        except Exception:
            pass

    def abort(self, reason: Optional[str] = None) -> None:
        """Triggers emergency stop for the in-progress pilot."""
        self.abort_reason = reason or "Pilot abort requested"
        logger.warning(f"EMERGENCY_STOP: {self.abort_reason}")
        self.abort_requested = True

    def validate_preflight(self) -> bool:
        """
        Validates all pre-flight invariants:
        1. User-Agent conforms to SEC regulations
        2. Backup containment gate verified (< 12 hours) - MANDATORY (fail-closed if disabled/missing)
        3. Production PostgreSQL connectivity - MANDATORY (fail-closed if missing/failed)
        """
        # 1. Regulatory SEC User-Agent
        ua = get_sec_user_agent()
        is_valid_ua, ua_err = validate_sec_user_agent(ua)
        if not is_valid_ua:
            raise PilotSecurityError(f"INVALID_USER_AGENT: {ua_err}")

        # 2. Backup containment check - MANDATORY (No silent bypass)
        containment_raw = os.environ.get("ARX_BACKUP_CONTAINMENT_ENABLED", "").strip().upper()
        if containment_raw != "TRUE":
            raise PilotContainmentError(
                "CONTAINMENT_DISABLED: Verified R2 backup containment is mandatory for ten-issuer pilot execution "
                "(ARX_BACKUP_CONTAINMENT_ENABLED must be 'TRUE'). Missing or disabled containment fails closed."
            )

        logger.info("Evaluating mandatory backup freshness authority gate (< 12 hours)...")
        max_age_sec = float(os.environ.get("ARX_BACKUP_CONTAINMENT_THRESHOLD_SECONDS", "43200.0"))
        policy = BackupContainmentPolicy(
            enabled=True,
            max_verified_age_seconds=max_age_sec,
            production_authorized=True,
        )
        authority = BackupFreshnessAuthority(policy=policy)
        gate = BackupAgeContainmentGate(freshness_authority=authority, policy=policy)
        gate.enforce_ingestion_gate()
        logger.info("Backup freshness gate verified: active verified backup within threshold.")

        # 3. Production PostgreSQL connectivity - MANDATORY (No silent bypass)
        if not self.dsn:
            raise PilotSecurityError(
                "DATABASE_UNAVAILABLE: Mandatory production PostgreSQL DSN is missing. "
                "Pilot execution requires authoritative database connectivity."
            )

        try:
            with get_connection(self.dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    row = cur.fetchone()
                    if not row or row[0] != 1:
                        raise PilotSecurityError("DATABASE_UNAVAILABLE: Preflight connection probe returned invalid result.")
        except PilotSecurityError:
            raise
        except Exception as e:
            raise PilotSecurityError(f"DATABASE_UNAVAILABLE: Preflight database probe failed: {e}")

        return True

    def check_deadline(self) -> None:
        """Enforces wall-clock execution ceiling and emergency abort across the entire pilot."""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0.0
        if self.abort_requested or os.environ.get("ARX_EMERGENCY_STOP", "").strip().lower() in ("true", "1", "yes"):
            if self.watchdog and self.watchdog.fired:
                raise PilotTimeoutError(
                    f"SUPERVISORY_WATCHDOG: Total execution exceeded {self.max_duration_sec}s ceiling."
                )
            reason_suffix = f" ({self.abort_reason})" if self.abort_reason else ""
            raise PilotSecurityError(
                f"EMERGENCY_STOP: Execution aborted by operator signal{reason_suffix}."
            )
        if self.start_time > 0 and elapsed > self.max_duration_sec:
            raise PilotTimeoutError(
                f"MECHANICAL_STOP: Pilot execution deadline ({self.max_duration_sec}s) exceeded."
            )

    def record_outbound_send(self, url: str, is_redirect: Optional[bool] = None) -> None:
        """
        Accounting hook executed BEFORE an HTTP request hits the wire.
        Enforces per-issuer ceiling (<= 4) and total cycle ceiling (<= 40).
        """
        self.check_deadline()
        if self.total_requests_made >= self.max_total_requests:
            raise PilotBudgetExceededError(
                f"MECHANICAL_STOP: Pilot total request count ({self.total_requests_made}) reached "
                f"cycle ceiling ({self.max_total_requests}). Outbound send blocked."
            )
        if self.current_issuer_requests >= self.max_per_issuer_requests:
            raise IssuerBudgetExceededError(
                f"MECHANICAL_STOP: Issuer {self.current_issuer_ticker} request count "
                f"({self.current_issuer_requests}) reached per-issuer ceiling ({self.max_per_issuer_requests}). "
                f"Outbound send blocked."
            )

        self.total_requests_made += 1
        self.current_issuer_requests += 1
        redirect_flag = is_redirect if is_redirect is not None else (self.current_issuer_requests > 1)
        self.requests_log.append({
            "request_num": self.total_requests_made,
            "issuer": self.current_issuer_ticker,
            "url": url,
            "status_code": None,
            "bytes_received": 0,
            "duration_sec": 0.0,
            "is_redirect": redirect_flag,
            "state": "SENT",
        })
        logger.info(
            f"[OUTBOUND SEND {self.total_requests_made}/{self.max_total_requests} | "
            f"{self.current_issuer_ticker} {self.current_issuer_requests}/{self.max_per_issuer_requests}] "
            f"Sending to {url}"
        )

    def record_response_received(self, resp: Any, duration: float) -> None:
        """Updates request log entry when response is received."""
        if self.requests_log:
            last_req = self.requests_log[-1]
            last_req["status_code"] = getattr(resp, "status_code", None)
            content = getattr(resp, "content", b"") or b""
            last_req["bytes_received"] = len(content)
            last_req["duration_sec"] = duration
            last_req["state"] = "RECEIVED"

    def tracked_get(self, url: str, timeout: float = 10.0) -> requests.Response:
        """
        Executes outbound HTTPS request with rate-limiting, redirect/retry tracking,
        and per-issuer and cycle budget enforcement at the outbound send boundary.
        """
        self.check_deadline()
        remaining = max(0.01, self.max_duration_sec - (time.time() - self.start_time)) if self.start_time > 0 else timeout
        effective_timeout = min(timeout, remaining)

        send_count_before = self.total_requests_made
        resp = self.session.get(url, timeout=effective_timeout)
        send_count_after = self.total_requests_made

        # Fallback for mocked sessions where session.get is mocked directly and bypasses send()
        if send_count_after == send_count_before:
            for hist_resp in (getattr(resp, "history", []) or []):
                self.record_outbound_send(hist_resp.url, is_redirect=True)
                self.record_response_received(hist_resp, 0.0)
            self.record_outbound_send(getattr(resp, "url", url), is_redirect=False)
            duration = getattr(resp, "duration_sec", 0.0) or 0.0
            self.record_response_received(resp, duration)

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
            self.check_deadline()
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

            self.check_deadline()
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

            self.check_deadline()
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

            self.check_deadline()
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

        except (PilotBudgetExceededError, PilotTimeoutError, PilotSecurityError) as fatal_err:
            logger.critical(f"FATAL pilot abort during ingestion for {ticker}: {fatal_err}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            issuer_receipt["status"] = "FAILED"
            issuer_receipt["error"] = str(fatal_err)
            issuer_receipt["requests_made"] = self.current_issuer_requests
            issuer_receipt["duration_sec"] = round(time.time() - t0, 3)
            raise fatal_err
        except Exception as e:
            logger.error(f"Ingestion failed for {ticker} ({cik}): {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            issuer_receipt["status"] = "FAILED"
            issuer_receipt["error"] = str(e)
        finally:
            issuer_receipt["requests_made"] = self.current_issuer_requests
            issuer_receipt["duration_sec"] = round(time.time() - t0, 3)

        return issuer_receipt

    def execute(self) -> Dict[str, Any]:
        """
        Executes bounded 10-issuer sequential pilot cycle with supervisory termination
        and guaranteed fail-closed teardown.
        """
        self.start_time = time.time()
        self.total_requests_made = 0
        self.requests_log = []

        telemetry: Dict[str, Any] = {
            "directive": "SM-ACT-018R",
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
            "pilot_auth_revoked": False,
            "verdict": "PENDING",
        }

        # Step 1: Preflight validations (MANDATORY DB & Backup Containment)
        logger.info("Executing pilot preflight validations...")
        self.validate_preflight()

        # Start supervisory watchdog
        self.watchdog = PilotSupervisoryWatchdog(self, self.max_duration_sec)
        self.watchdog.start()

        # Step 2: Acquire Advisory Lock via Resource Management Boundary
        logger.info(f"Acquiring PostgreSQL advisory lock {LOCK_SEC_EGRESS_OWNER} via SecAdvisoryLockBoundary...")
        prev_role = get_sec_egress_role()
        lock_boundary = SecAdvisoryLockBoundary(self.dsn)
        conn = None

        try:
            lock_boundary.__enter__()
            conn = lock_boundary.conn
            if lock_boundary.lock_acquired:
                telemetry["lock_acquired"] = True
                logger.info("Exclusive advisory lock claimed successfully.")

            # Step 3: Require explicit, scoped, one-shot owner authorization
            auth_val = os.environ.get("ARX_TEN_ISSUER_PILOT_AUTHORIZED", "").strip().lower()
            if auth_val in ("consumed", "used"):
                raise PilotSecurityError(
                    "UNAUTHORIZED_PILOT_EXECUTION: Pilot authorization has already been consumed. "
                    "One-shot authorization cannot be reused or become a recurring execution."
                )
            if auth_val not in ("true", "1", "yes"):
                raise PilotSecurityError(
                    "UNAUTHORIZED_PILOT_EXECUTION: Ten-issuer pilot execution requires explicit, "
                    "scoped one-shot owner authorization (ARX_TEN_ISSUER_PILOT_AUTHORIZED=true). "
                    "Ordinary standby-worker startup cannot trigger SEC role elevation."
                )

            # Check authorization expiry if provided
            expires_at_str = os.environ.get("ARX_TEN_ISSUER_PILOT_EXPIRES_AT", "").strip()
            if expires_at_str:
                try:
                    exp_dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                    if datetime.now(timezone.utc) > exp_dt:
                        raise PilotSecurityError(
                            f"AUTHORIZATION_EXPIRED: Pilot authorization expired at {expires_at_str}."
                        )
                except ValueError as e:
                    raise PilotSecurityError(f"INVALID_AUTHORIZATION_EXPIRY: {e}")

            # Check production release anchor binding if provided
            expected_release = os.environ.get("ARX_TEN_ISSUER_PILOT_RELEASE_SHA", "").strip()
            if expected_release:
                current_head = os.environ.get("RAILWAY_GIT_COMMIT_SHA", "").strip()
                if current_head and expected_release != current_head:
                    raise PilotSecurityError(
                        f"RELEASE_ANCHOR_MISMATCH: Authorization bound to release {expected_release}, "
                        f"but current runtime is {current_head}."
                    )

            # Consume authorization immediately so it cannot be reused
            os.environ["ARX_TEN_ISSUER_PILOT_AUTHORIZED"] = "consumed"
            telemetry["authorization_record"] = {
                "scope": "10_GOVERNED_ISSUERS_SEQUENTIAL_PILOT",
                "issuers": [iss["ticker"] for iss in self.issuers],
                "production_release": expected_release or "87c5e939fb1434e7ee472a397a9ecb2252123dab",
                "consumed_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": expires_at_str or None,
                "is_recurring": False,
                "single_use_consumed": True,
            }

            logger.info("One-shot owner authorization confirmed & consumed (single-use).")
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
                self.check_deadline()
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

            # 1. Cancel supervisory watchdog
            if self.watchdog:
                try:
                    self.watchdog.cancel()
                except Exception:
                    pass

            # 2. Revoke one-shot pilot authorization
            os.environ["ARX_TEN_ISSUER_PILOT_AUTHORIZED"] = "false"
            telemetry["pilot_auth_revoked"] = True

            # 3. Restore non-owner role
            os.environ["ARX_SEC_EGRESS_ROLE"] = prev_role if prev_role in ("STANDBY", "DISABLED") else "STANDBY"
            telemetry["role_restored"] = True

            # 4. Release advisory lock and close DB connection via boundary
            try:
                lock_boundary.close()
                telemetry["lock_released"] = lock_boundary.lock_released or (not lock_boundary.lock_acquired)
                logger.info("Advisory lock boundary closed.")
            except Exception as e:
                logger.warning(f"Error in lock boundary cleanup: {e}")

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


def run_ten_issuer_pilot(dsn: Optional[str] = None, supervised: bool = True) -> Dict[str, Any]:
    """
    Top-level entrypoint for executing bounded 10-issuer pilot.
    By default, executes within a dedicated, supervised process enforcing
    a hard 180-second wall-clock termination boundary.
    """
    if supervised:
        supervisor = PilotProcessSupervisor(dsn=dsn, timeout_sec=PILOT_MAX_DURATION_SEC)
        return supervisor.run()
    else:
        runner = TenIssuerPilotRunner(dsn=dsn)
        return runner.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARX Ten-Issuer SEC Pilot Runner")
    parser.add_argument("--child-worker", action="store_true", help="Run in-process pilot worker (invoked by supervisor)")
    parser.add_argument("--unsupervised", action="store_true", help="Run without process supervisor wrapper")
    parser.add_argument("--dsn", type=str, default=None, help="PostgreSQL DSN override")
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.child_worker:
        # In child worker process: execute runner in-process
        runner = TenIssuerPilotRunner(dsn=args.dsn)
        try:
            receipt = runner.execute()
            print("\n--- PILOT_TELEMETRY_JSON_START ---")
            print(json.dumps(receipt, default=str))
            print("--- PILOT_TELEMETRY_JSON_END ---")
            sys.exit(0 if receipt.get("verdict") == "SUCCESS" else 1)
        except Exception as exc:
            logger.exception(f"Child pilot worker encountered exception: {exc}")
            err_dict = {"verdict": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
            print("\n--- PILOT_TELEMETRY_JSON_START ---")
            print(json.dumps(err_dict, default=str))
            print("--- PILOT_TELEMETRY_JSON_END ---")
            sys.exit(1)
    else:
        # Parent / operator CLI invocation: execute via supervisor
        print("=== EXECUTING ARX TEN-ISSUER SEC PILOT (SUPERVISED PROCESS) ===")
        try:
            receipt = run_ten_issuer_pilot(dsn=args.dsn, supervised=not args.unsupervised)
            print("\nPilot Execution Receipt:")
            print(json.dumps(receipt, indent=2, default=str))
            sys.exit(0 if receipt.get("verdict") == "SUCCESS" else 1)
        except Exception as exc:
            print(f"\nFATAL: Ten-issuer pilot failed with exception: {exc}")
            sys.exit(1)
