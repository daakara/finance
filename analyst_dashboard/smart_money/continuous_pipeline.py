"""
ARX Terminal — Smart Money Continuous Ingestion Pipeline (Phase 1).

Encapsulates the canonical 10-step daily continuous ingestion cycle:
1. Preflight check: Validates environment, credentials, DB connectivity, SEC User-Agent.
2. Pre-ingestion backup: Logical pg_dump -Fc of Smart Money tables via smart_money_backup role.
3. R2 backup upload: Compute and verify SHA-256 digest on Cloudflare R2 / storage adapter.
4. Backup freshness gate: Confirm backup age is <= 12 hours (freshly captured pre-backup satisfies this).
5. Advisory lock claim: Acquire session-level LOCK_SEC_EGRESS_OWNER (401007311509093371).
6. Genuine SEC Form 4 ingestion: Rate-limited (<= 8.0 req/s) ingestion for 10 governed US equities.
7. Transaction normalization & reconciliation: Parse XML, persist transactions, generate & save cryptographic receipt.
8. Atomic snapshot publication: Generate immutable snapshot and atomically advance scanner_active_snapshots pointer.
9. Post-ingestion backup: Logical pg_dump -Fc uploaded to R2 (failure degrades to DEGRADED_POST_BACKUP_FAILED).
10. Retention pruning & lock release: Prune older backups protecting the latest recovery point, release advisory lock.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import json
import hashlib
import logging
import math
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set, Callable

from analyst_dashboard.smart_money.storage.config import (
    get_database_url,
    get_database_role,
    ROLE_INGESTION,
    validate_storage_configuration,
)
from analyst_dashboard.smart_money.storage.postgres_connection import get_connection
from analyst_dashboard.smart_money.continuous_authorization import (
    require_continuous_authorization,
    resolve_release_identity,
    SCOPE_DAILY_10,
    SCOPE_TRIAL_2,
)
from analyst_dashboard.smart_money.storage.postgres_coordination import (
    acquire_sec_owner_lock,
    release_sec_owner_lock,
    LOCK_SEC_EGRESS_OWNER,
)
from analyst_dashboard.smart_money.storage.postgres_repository import (
    PostgresSmartMoneyRepository,
    StaleLeaseOwnershipLostError,
)
from analyst_dashboard.smart_money.smart_money_scheduler import (
    SmartMoneyTimezoneScheduler,
    TwoIssuerTrialScheduler,
)
from analyst_dashboard.smart_money.freshness_authority import NY_TZ
from analyst_dashboard.data.sec_edgar_fetcher import (
    get_sec_egress_role,
    get_sec_user_agent,
    RateLimiter,
)
from analyst_dashboard.smart_money.storage.backup.backup_engine import SmartMoneyBackupEngine
from analyst_dashboard.smart_money.storage.backup.retention import BackupRetentionManager
from analyst_dashboard.smart_money.storage.backup.storage_adapter import (
    get_backup_storage_adapter,
    BackupStorageAdapter,
)
from analyst_dashboard.smart_money.storage.backup.freshness_authority import (
    BackupFreshnessAuthority,
    BackupContainmentPolicy,
)
from analyst_dashboard.smart_money.storage.backup.containment_gate import (
    BackupAgeContainmentGate,
    BackupAgeContainmentViolationException,
)
from analyst_dashboard.smart_money.form4_models import (
    SmartMoneyTransaction,
    SmartMoneyFiling,
    OwnershipNature,
    SmartMoneySignalType,
    SmartMoneyIssuerState,
)
from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
from analyst_dashboard.smart_money.amendment_classifier import (
    classify_amendment_effect,
    SCOPE_FULL_REPLACEMENT,
    SCOPE_PARTIAL_CORRECTION,
    SCOPE_ADDITIONAL_TRANSACTION,
    SCOPE_AMBIGUOUS,
)
from analyst_dashboard.smart_money.form4_scanner import SmartMoneyForm4Scanner
from analyst_dashboard.analyzers.scanner_contract import (
    ScannerCandidateResult,
    ScannerStatus,
    PublicationDecision,
    FreshnessStatus,
)

logger = logging.getLogger("smart_money.continuous_pipeline")

GOVERNED_UNIVERSE_ID = "UB-US-EQUITIES-20261025-001"

DEFAULT_GOVERNED_ISSUERS = [
    {"ticker": "NVDA", "cik": "0001045810", "company_name": "NVIDIA CORP"},
    {"ticker": "AAPL", "cik": "0000320193", "company_name": "APPLE INC"},
    {"ticker": "MSFT", "cik": "0000789019", "company_name": "MICROSOFT CORP"},
    {"ticker": "AMZN", "cik": "0001018724", "company_name": "AMAZON COM INC"},
    {"ticker": "GOOGL", "cik": "0001652044", "company_name": "ALPHABET INC"},
    {"ticker": "META", "cik": "0001326801", "company_name": "META PLATFORMS INC"},
    {"ticker": "TSLA", "cik": "0001318605", "company_name": "TESLA INC"},
    {"ticker": "BRK-B", "cik": "0001067983", "company_name": "BERKSHIRE HATHAWAY INC"},
    {"ticker": "JPM", "cik": "0000019617", "company_name": "JPMORGAN CHASE & CO"},
    {"ticker": "V", "cik": "0001403161", "company_name": "VISA INC"},
]

TRIAL_2_GOVERNED_ISSUERS = [
    {"ticker": "NVDA", "cik": "0001045810", "company_name": "NVIDIA CORP"},
    {"ticker": "AAPL", "cik": "0000320193", "company_name": "APPLE INC"},
]

PROSPECTIVE_TRIAL_START_BOUNDARY = "2026-11-17T00:00:00Z"


def _normalize_to_transaction_obj(tx: Any) -> SmartMoneyTransaction:
    """Normalizes a transaction dict or SmartMoneyTransaction instance into SmartMoneyTransaction."""
    if isinstance(tx, SmartMoneyTransaction):
        return tx
    if isinstance(tx, dict):
        is_deriv = bool(tx.get("is_derivative", False))
        is_buy = (tx.get("is_open_market_purchase_evidence", False) or (tx.get("transaction_code") == "P" and not is_deriv and tx.get("acquired_disposed", "A") == "A")) and not is_deriv
        t_code = (tx.get("transaction_code") or "").upper().strip()
        raw_sig = tx.get("signal_type")
        raw_sig_str = raw_sig.value if isinstance(raw_sig, SmartMoneySignalType) else str(raw_sig or "")

        if is_deriv:
            sig_type = SmartMoneySignalType.OTHER
        elif is_buy:
            sig_type = SmartMoneySignalType.OPEN_MARKET_PURCHASE
        elif t_code == "S" or raw_sig_str == "OPEN_MARKET_SALE":
            sig_type = SmartMoneySignalType.OPEN_MARKET_SALE
        elif t_code == "F" or raw_sig_str == "TAX_WITHHOLDING":
            sig_type = SmartMoneySignalType.TAX_WITHHOLDING
        elif t_code == "G" or raw_sig_str == "GIFT":
            sig_type = SmartMoneySignalType.GIFT
        elif t_code == "D" or raw_sig_str == "ISSUER_DISPOSITION":
            sig_type = SmartMoneySignalType.ISSUER_DISPOSITION
        elif t_code == "M" or raw_sig_str == "OPTION_EXERCISE":
            sig_type = SmartMoneySignalType.OPTION_EXERCISE
        elif t_code == "A" or raw_sig_str == "EQUITY_GRANT":
            sig_type = SmartMoneySignalType.EQUITY_GRANT
        else:
            sig_type = SmartMoneySignalType.OTHER
        return SmartMoneyTransaction(
            record_key=tx.get("transaction_id", tx.get("record_key", str(uuid.uuid4()))),
            accession_number=tx.get("accession_number", "0000000000-00-000000"),
            issuer_cik=tx.get("issuer_cik", ""),
            issuer_symbol=tx.get("issuer_symbol", tx.get("ticker", tx.get("symbol", ""))),
            issuer_name=tx.get("issuer_name", tx.get("company_name", "")),
            reporting_owner_cik=tx.get("reporting_owner_cik", tx.get("reporting_owner_name", "insider-1")),
            reporting_owner_name=tx.get("reporting_owner_name", "Insider"),
            is_officer=tx.get("is_officer", False),
            officer_title=tx.get("officer_title"),
            is_director=tx.get("is_director", False),
            is_ten_percent_owner=tx.get("is_ten_percent", tx.get("is_ten_percent_owner", False)),
            filing_date=tx.get("filing_date", tx.get("transaction_date", "2026-10-25")),
            transaction_date=tx.get("transaction_date", "2026-10-25"),
            transaction_code=tx.get("transaction_code", "P" if is_buy else "S"),
            signal_type=sig_type,
            shares=float(tx.get("shares", 0.0)),
            price_per_share=float(tx.get("price_per_share", 0.0)),
            total_value_usd=float(tx.get("total_value_usd", 0.0)),
            acquired_disposed=tx.get("acquired_disposed", "A" if is_buy else "D"),
            ownership_nature=OwnershipNature.DIRECT if tx.get("ownership_type", "D") in ("D", "DIRECT", OwnershipNature.DIRECT) else OwnershipNature.INDIRECT,
            is_derivative=is_deriv,
            is_open_market_purchase_evidence=is_buy and not is_deriv,
            is_positive_buying_conviction=is_buy and not is_deriv,
        )
    raise TypeError(f"UNSUPPORTED_TRANSACTION_TYPE: Cannot convert {type(tx)} to SmartMoneyTransaction")



import requests
import re
from urllib.parse import urlsplit
from requests.adapters import HTTPAdapter


def resolve_sec_primary_document_url(archive_root: str, doc: str) -> str:
    """Safely resolves legitimate SEC primary-document path within accession directory.

    Accepts relative paths like 'form4.xml', 'xslF345X06/form4.xml', 'primary_doc.xml'.
    Rejects directory traversal ('..'), absolute paths, host substitutions, and untrusted URLs.
    """
    if not isinstance(doc, str) or not doc.strip():
        raise RuntimeError("SEC_PRIMARY_DOCUMENT_INVALID")
    doc_clean = doc.strip()

    # Reject URLs, schemes, host substitutions, backslashes, credentials
    if (
        "://" in doc_clean
        or doc_clean.startswith("//")
        or doc_clean.startswith("/")
        or "\\" in doc_clean
        or ":" in doc_clean
        or "@" in doc_clean
        or "?" in doc_clean
        or "#" in doc_clean
    ):
        raise RuntimeError("SEC_PRIMARY_DOCUMENT_INVALID")

    # Reject invalid characters (only alphanumeric, _, ., -, / allowed)
    if not re.fullmatch(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$", doc_clean):
        raise RuntimeError("SEC_PRIMARY_DOCUMENT_INVALID")

    # Reject traversal segments
    segments = doc_clean.split("/")
    if any(s in ("", ".", "..") for s in segments):
        raise RuntimeError("SEC_PRIMARY_DOCUMENT_INVALID")

    # Ensure archive_root is valid SEC archive root
    if not archive_root.startswith("https://www.sec.gov/Archives/edgar/data/"):
        raise RuntimeError("SEC_EGRESS_DESTINATION_REJECTED")

    resolved = f"{archive_root}{doc_clean}"
    if not resolved.startswith(archive_root):
        raise RuntimeError("SEC_PRIMARY_DOCUMENT_INVALID")
    return resolved


def validate_cycle_duration(val: Any) -> float:
    """
    Validates cycle duration setting:
    - Rejects boolean types.
    - Rejects non-numeric types.
    - Rejects non-finite values (NaN, Inf).
    - Rejects non-positive values (val <= 0).
    - Rejects excessive values (> 60.0s): config may lower, never raise above 60.0s.
    Returns validated float.
    """
    if isinstance(val, bool):
        raise ValueError("CYCLE_DURATION_INVALID: Boolean value is not a valid duration")
    try:
        fval = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CYCLE_DURATION_INVALID: Non-numeric duration value: {val}") from exc
    if math.isnan(fval) or math.isinf(fval):
        raise ValueError(f"CYCLE_DURATION_INVALID: Non-finite duration value: {fval}")
    if fval <= 0.0:
        raise ValueError(f"CYCLE_DURATION_INVALID: Non-positive duration value: {fval}")
    if fval > 60.0:
        raise ValueError(f"CYCLE_DURATION_INVALID: Duration {fval} exceeds maximum allowed ceiling of 60.0s")
    return fval


def parse_sec_datetime_utc(dt_str: str) -> datetime:
    """Parses SEC acceptanceDateTime or filingDate string into timezone-aware UTC datetime."""
    if not dt_str or not isinstance(dt_str, str):
        raise ValueError("Invalid datetime string")
    s = dt_str.strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


class ContinuousBudgetExceededError(RuntimeError):
    """Raised when total cycle (40) or per-issuer (4) HTTP transmission ceiling is reached."""
    pass


class ContinuousOutboundSession(requests.Session):
    """
    Hardened outbound session for Smart Money continuous ingestion.
    Enforces wire-level invariants on EVERY transmission (initial, redirect, retry):
    1. ARX_EMERGENCY_STOP_ACTIVE: fail closed if TRUE.
    2. Cooperative cancellation: fail closed if cancellation requested.
    3. 24h continuous authorization: fail closed if expired, future, or invalid via require_continuous_authorization.
    4. Rate limiting: strictly throttles to <= 5.0 requests/sec.
    5. Per-cycle request budget: maximum 40 HTTP transmissions total per cycle (8 for 2-issuer trial).
    6. Per-issuer request budget: maximum 4 HTTP transmissions per issuer.
    """

    def __init__(
        self,
        rate_limiter: Optional[Any] = None,
        max_total_requests: int = 40,
        max_per_issuer_requests: int = 4,
        issuers: Optional[Sequence[Dict[str, str]]] = None,
        allow_test_mode: bool = False,
        cancellation_token: Optional[Callable[[], bool]] = None,
        preacquired_lock_conn: Optional[Any] = None,
        cycle_deadline: Optional[float] = None,
    ):
        super().__init__()
        # Explicitly prohibit urllib3 adapter-managed retries, which would bypass
        # Session.send accounting. Any future retry must call this guarded send.
        self.mount("https://", HTTPAdapter(max_retries=0))
        self.mount("http://", HTTPAdapter(max_retries=0))
        if rate_limiter is not None:
            if hasattr(rate_limiter, "max_per_second") and float(rate_limiter.max_per_second) > 5.0:
                rate_limiter.max_per_second = 5.0
            self.rate_limiter = rate_limiter
        else:
            self.rate_limiter = RateLimiter(max_per_second=5.0)

        self.issuers = issuers or DEFAULT_GOVERNED_ISSUERS
        is_trial = (
            os.environ.get("ARX_CONTINUOUS_AUTH_SCOPE") == "RECURRING_2_ISSUER_FORM4_TRIAL"
            or (self.issuers is not None and len(self.issuers) <= 2)
        )
        self.hard_max_total_ceiling = 8 if is_trial else 40
        self.hard_max_issuer_ceiling = 4
        self.max_total_requests = min(max_total_requests, self.hard_max_total_ceiling)
        self.max_per_issuer_requests = min(max_per_issuer_requests, self.hard_max_issuer_ceiling)
        self.allow_test_mode = allow_test_mode
        self.cancellation_token = cancellation_token
        self.cycle_deadline = cycle_deadline
        self.deadline_monotonic: Optional[float] = None
        self.daily_requests_baseline: int = 0
        self.trial_requests_baseline: int = 0
        self.max_daily_requests: int = 32
        self.max_trial_requests: int = 96
        self.is_trial: bool = is_trial
        self.total_transmissions: int = 0
        self.issuer_transmissions: Dict[str, int] = {}
        self.current_issuer: Optional[str] = None
        self.transmission_log: List[Dict[str, Any]] = []

    def set_current_issuer(self, ticker: str) -> None:
        self.current_issuer = ticker.upper().strip()
        if self.current_issuer not in self.issuer_transmissions:
            self.issuer_transmissions[self.current_issuer] = 0

    def send(self, request, **kwargs):
        # 0. Monotonic & cycle deadline check
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            raise RuntimeError("CYCLE_TIMEOUT_EXCEEDED: Outbound send blocked by monotonic cycle deadline.")
        if self.cycle_deadline and time.time() >= self.cycle_deadline:
            raise RuntimeError("CYCLE_TIMEOUT_EXCEEDED: Outbound send blocked by cycle deadline.")

        # Every redirect must remain on the two explicitly trusted SEC domains.
        # Reject non-HTTPS schemes, host confusion and alternate ports.
        target = urlsplit(getattr(request, "url", "") or "")
        if (
            target.scheme != "https"
            or target.hostname not in ("data.sec.gov", "www.sec.gov")
            or target.port not in (None, 443)
            or target.username is not None
            or target.password is not None
        ):
            raise RuntimeError("SEC_EGRESS_DESTINATION_REJECTED")

        # Disallow unaccounted HTTPAdapter/urllib3 retries. Redirects are still
        # followed by requests.Session and are intercepted on each send().
        adapter = self.get_adapter(request.url)
        retry_policy = getattr(adapter, "max_retries", None)
        if getattr(retry_policy, "total", 0) != 0:
            raise RuntimeError("SEC_EGRESS_UNACCOUNTED_RETRY_POLICY")

        # 1. Emergency stop check
        if os.environ.get("ARX_EMERGENCY_STOP_ACTIVE", "").strip().upper() in ("TRUE", "1", "YES"):
            raise RuntimeError("EMERGENCY_STOP_ACTIVE: Outbound send blocked by emergency stop killswitch.")

        # 2. In-flight cooperative cancellation
        if self.cancellation_token and self.cancellation_token():
            raise RuntimeError("IN_FLIGHT_CANCELLATION: Outbound send blocked by cooperative cancellation.")

        # 3. 24h continuous authorization check (fails closed if expired mid-cycle or before retry)
        if not self.allow_test_mode:
            require_continuous_authorization(issuers=self.issuers)

        # 4. Total cycle request budget check (config may lower, never raise)
        configured_max_total = int(os.environ.get("ARX_SEC_MAX_TOTAL_REQUESTS_PER_CYCLE", str(self.max_total_requests)))
        effective_max_total = min(self.max_total_requests, configured_max_total)
        if self.total_transmissions >= effective_max_total:
            raise ContinuousBudgetExceededError(
                f"PILOT_BUDGET_EXCEEDED: CYCLE_BUDGET_EXCEEDED: Total cycle HTTP transmissions ({self.total_transmissions}) "
                f"reached ceiling ({effective_max_total}). Outbound send blocked."
            )

        # 4b. Daily and trial request budgets under two-issuer trial profile
        if self.is_trial:
            total_today = self.daily_requests_baseline + self.total_transmissions
            if total_today >= self.max_daily_requests:
                raise ContinuousBudgetExceededError(
                    f"PILOT_BUDGET_EXCEEDED: DAILY_BUDGET_EXCEEDED: Daily HTTP transmissions ({total_today}) "
                    f"reached ceiling ({self.max_daily_requests}). Outbound send blocked."
                )
            total_trial = self.trial_requests_baseline + self.total_transmissions
            if total_trial >= self.max_trial_requests:
                raise ContinuousBudgetExceededError(
                    f"PILOT_BUDGET_EXCEEDED: TRIAL_BUDGET_EXCEEDED: Trial HTTP transmissions ({total_trial}) "
                    f"reached ceiling ({self.max_trial_requests}). Outbound send blocked."
                )

        # 5. Per-issuer request budget check (config may lower, never raise)
        configured_max_issuer = int(os.environ.get("ARX_SEC_MAX_REQUESTS_PER_ISSUER", str(self.max_per_issuer_requests)))
        effective_max_issuer = min(self.max_per_issuer_requests, configured_max_issuer)
        issuer_key = self.current_issuer or "UNKNOWN"
        count_for_issuer = self.issuer_transmissions.get(issuer_key, 0)
        if count_for_issuer >= effective_max_issuer:
            raise ContinuousBudgetExceededError(
                f"PILOT_BUDGET_EXCEEDED: ISSUER_BUDGET_EXCEEDED: Issuer {issuer_key} HTTP transmissions ({count_for_issuer}) "
                f"reached ceiling ({effective_max_issuer}). Outbound send blocked."
            )

        # 6. Rate limiting (<= 5.0 req/s)
        self.rate_limiter.acquire()
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            raise RuntimeError("CYCLE_TIMEOUT_EXCEEDED: Execution exceeded monotonic cycle duration ceiling after rate-limiting wait.")
        if self.cycle_deadline and time.time() >= self.cycle_deadline:
            raise RuntimeError("CYCLE_TIMEOUT_EXCEEDED: Execution exceeded cycle duration ceiling after rate-limiting wait.")

        # A grant or kill switch may expire while rate limiting blocks.
        if os.environ.get("ARX_EMERGENCY_STOP_ACTIVE", "").strip().upper() in ("TRUE", "1", "YES"):
            raise RuntimeError("EMERGENCY_STOP_ACTIVE")
        if self.cancellation_token and self.cancellation_token():
            raise RuntimeError("IN_FLIGHT_CANCELLATION")
        if not self.allow_test_mode:
            require_continuous_authorization(issuers=self.issuers)

        # 7. Record transmission count before wire transmission
        self.total_transmissions += 1
        self.issuer_transmissions[issuer_key] = count_for_issuer + 1
        t0 = time.time()

        # 8. Wire transmission with chunked transfer and monotonic deadline enforcement
        remaining_sec = None
        if self.deadline_monotonic is not None:
            remaining_sec = max(0.001, self.deadline_monotonic - time.monotonic())
            orig_timeout = kwargs.get("timeout")
            if orig_timeout is None:
                kwargs["timeout"] = (min(5.0, remaining_sec), remaining_sec)
            elif isinstance(orig_timeout, (int, float)):
                kwargs["timeout"] = min(float(orig_timeout), remaining_sec)
            elif isinstance(orig_timeout, tuple) and len(orig_timeout) == 2:
                conn_t, read_t = orig_timeout
                kwargs["timeout"] = (min(float(conn_t), remaining_sec), min(float(read_t), remaining_sec))

        kwargs["stream"] = True
        resp = super().send(request, **kwargs)

        # Ingest chunk-by-chunk with monotonic deadline checks across body transfer
        if resp.raw is not None and not getattr(resp, "_content_consumed", False):
            chunks = []
            try:
                for chunk in resp.iter_content(chunk_size=4096):
                    if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
                        if hasattr(resp, "close"):
                            try:
                                resp.close()
                            except Exception:
                                pass
                        raise RuntimeError("CYCLE_TIMEOUT_EXCEEDED: Response transfer exceeded monotonic cycle deadline.")
                    if chunk:
                        chunks.append(chunk)
                resp._content = b"".join(chunks)
            except Exception:
                if hasattr(resp, "close") and resp.raw is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                raise

        duration = round(time.time() - t0, 3)
        self.transmission_log.append({
            "seq": self.total_transmissions,
            "issuer": issuer_key,
            "url": getattr(request, "url", ""),
            "status_code": getattr(resp, "status_code", None),
            "duration_sec": duration,
        })
        return resp


class SmartMoneyContinuousPipeline:
    """
    10-step continuous daily ingestion and snapshot publication pipeline.
    Ensures safe, verifiable, and observable Form 4 market-wide updates.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        backup_engine: Optional[SmartMoneyBackupEngine] = None,
        storage_adapter: Optional[BackupStorageAdapter] = None,
        retention_manager: Optional[BackupRetentionManager] = None,
        repository: Optional[PostgresSmartMoneyRepository] = None,
        containment_gate: Optional[BackupAgeContainmentGate] = None,
        issuers: Optional[List[Dict[str, str]]] = None,
        rate_limiter: Optional[RateLimiter] = None,
        sec_fetcher: Optional[Any] = None,
        scanner: Optional[SmartMoneyForm4Scanner] = None,
        scheduler: Optional[SmartMoneyTimezoneScheduler] = None,
        allow_test_mode: bool = False,
        cancellation_token: Optional[Callable[[], bool]] = None,
        preacquired_lock_conn: Optional[Any] = None,
        ingestion_only: Optional[bool] = None,
        max_duration_sec: float = 60.0,
    ):
        arx_env = os.environ.get("ARX_ENVIRONMENT", "").strip().lower()
        if arx_env == "production" and allow_test_mode:
            raise RuntimeError(
                "TEST_MODE_PROHIBITED_IN_PRODUCTION: allow_test_mode is strictly prohibited in production environment."
            )

        self.allow_test_mode = allow_test_mode
        self.cancellation_token = cancellation_token
        self._cancelled: bool = False
        self.dsn = dsn or get_database_url()
        self.storage_adapter = storage_adapter or get_backup_storage_adapter()
        self.backup_engine = backup_engine or (
            SmartMoneyBackupEngine(dsn=self.dsn, storage_adapter=self.storage_adapter)
            if self.dsn
            else None
        )
        if not self.allow_test_mode and self.backup_engine is None:
            raise RuntimeError(
                "BACKUP_ENGINE_REQUIRED: Live continuous pipeline requires verified backup engine"
            )

        self.retention_manager = retention_manager or BackupRetentionManager(
            storage_adapter=self.storage_adapter,
            retention_count=10,
            retention_days=7,
        )
        self.containment_policy = BackupContainmentPolicy.from_environment()
        self.containment_gate = containment_gate or (
            BackupAgeContainmentGate(
                freshness_authority=BackupFreshnessAuthority(
                    dsn=self.dsn, policy=self.containment_policy
                ),
                policy=self.containment_policy,
            )
            if self.dsn
            else None
        )
        self.repository = repository or (
            PostgresSmartMoneyRepository(
                dsn=self.dsn, containment_gate=self.containment_gate
            )
            if self.dsn
            else None
        )
        self.issuers = issuers or DEFAULT_GOVERNED_ISSUERS
        scope = os.environ.get("ARX_CONTINUOUS_AUTH_SCOPE", "")
        self.is_trial_2_issuer = bool(
            scope == "RECURRING_2_ISSUER_FORM4_TRIAL"
            or (self.issuers is not None and len(self.issuers) == 2 and {i.get("ticker") for i in self.issuers} == {"NVDA", "AAPL"})
        )
        env_ingestion_only = os.environ.get("ARX_SMART_MONEY_INGESTION_ONLY", "").strip().lower()

        if self.is_trial_2_issuer:
            if env_ingestion_only in ("false", "0", "no"):
                raise RuntimeError(
                    "TRIAL_INGESTION_ONLY_NON_OVERRIDABLE: ARX_SMART_MONEY_INGESTION_ONLY cannot be disabled under two-issuer trial profile."
                )
            if ingestion_only is False:
                raise RuntimeError(
                    "TRIAL_INGESTION_ONLY_NON_OVERRIDABLE: ingestion_only=False is prohibited under two-issuer trial profile."
                )
            self.ingestion_only = True
            self.trial_scheduler = scheduler if isinstance(scheduler, TwoIssuerTrialScheduler) else TwoIssuerTrialScheduler()
            self.trial_scheduler.validate_issuers(self.issuers)
        elif ingestion_only is not None:
            self.ingestion_only = bool(ingestion_only)
        else:
            self.ingestion_only = bool(env_ingestion_only in ("true", "1", "yes"))

        env_dur = os.environ.get("ARX_CYCLE_MAX_DURATION_SEC")
        raw_dur = env_dur if (env_dur is not None and env_dur != "") else max_duration_sec
        self.max_duration_sec = validate_cycle_duration(raw_dur)
        self.cycle_deadline: Optional[float] = None
        self.cycle_deadline_monotonic: Optional[float] = None
        # Strict token bucket <= 5.0 req/s (SEC ceiling is 10)
        limiter = rate_limiter or RateLimiter(max_per_second=5.0)
        if hasattr(limiter, "max_per_second") and limiter.max_per_second > 5.0:
            limiter = RateLimiter(max_per_second=5.0)
        self.rate_limiter = limiter
        max_total_reqs = 8 if (self.ingestion_only or len(self.issuers) <= 2) else 40
        self.outbound_session = ContinuousOutboundSession(
            rate_limiter=self.rate_limiter,
            max_total_requests=max_total_reqs,
            max_per_issuer_requests=4,
            issuers=self.issuers,
            allow_test_mode=self.allow_test_mode,
            cancellation_token=self.cancellation_token,
        )
        self.outbound_session.is_trial = self.is_trial_2_issuer
        self.sec_fetcher = sec_fetcher
        self.scanner = scanner or SmartMoneyForm4Scanner()
        self.scheduler = scheduler or SmartMoneyTimezoneScheduler()
        self._preacquired_lock_conn = preacquired_lock_conn
        self._lock_conn_is_preacquired = preacquired_lock_conn is not None
        self._lock_conn = preacquired_lock_conn

    def cancel(self) -> None:
        """Sets internal cooperative cancellation flag for in-flight checks."""
        self._cancelled = True

    def execute_cycle(
        self,
        run_request_id: Optional[str] = None,
        lease_token: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False,
        as_of: Optional[str] = None,
        trigger_type: str = "SCHEDULED",
        recovery_token: Optional[str] = None,
        recovery_reason: Optional[str] = None,
        recovery_scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the authoritative 10-step ingestion cycle.
        Returns execution result summary with cryptographic receipt.

        Production override governance:
        - In production (not allow_test_mode or ARX_ENVIRONMENT=production):
          * trigger_type='TEST' is strictly rejected fail-closed.
          * raw force=True or as_of overrides require explicit, verified recovery authorization
            (matching ARX_PRODUCTION_FORCE_RECOVERY_TOKEN, recovery_reason >= 10 chars, and approved scope).
          * force overrides NEVER bypass advisory locking, pre/post backup integrity, or snapshot atomicity.
        """
        start_time = datetime.now(timezone.utc)
        self.cycle_deadline = time.time() + self.max_duration_sec
        self.cycle_deadline_monotonic = time.monotonic() + self.max_duration_sec
        self.outbound_session.cycle_deadline = self.cycle_deadline
        self.outbound_session.deadline_monotonic = self.cycle_deadline_monotonic
        cycle_id = f"sm-cycle-{start_time.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        steps_completed: List[str] = []
        errors: List[str] = []
        pre_backup_manifest: Optional[Dict[str, Any]] = None
        post_backup_manifest: Optional[Dict[str, Any]] = None
        snapshot_id: Optional[str] = None
        receipt_id: Optional[str] = None
        reconciliation_hash: Optional[str] = None
        matched_count = 0
        status = "IN_PROGRESS"
        trial_budget_reserved = False
        reserved_slot_id = None
        reserved_ny_date = None

        logger.info(
            f"Starting Smart Money 10-step continuous cycle {cycle_id} "
            f"(trigger={trigger_type}, dry_run={dry_run}, req_id={run_request_id})"
        )

        is_production = not self.allow_test_mode or os.environ.get("ARX_ENVIRONMENT", "").strip().lower() == "production"

        # Production boundary checks: fail-closed rejection of test triggers outside test mode
        if is_production and str(trigger_type).upper() == "TEST":
            raise RuntimeError(
                "TEST_TRIGGER_PROHIBITED_IN_PRODUCTION: trigger_type='TEST' is strictly prohibited outside authorized test mode."
            )

        # Production recovery authorization evaluation
        recovery_authorized = False
        if is_production and (force or as_of is not None):
            expected_token = os.environ.get("ARX_PRODUCTION_FORCE_RECOVERY_TOKEN")
            effective_token = recovery_token or os.environ.get("ARX_RECOVERY_TOKEN")

            if not effective_token or not expected_token or effective_token != expected_token:
                if force:
                    raise RuntimeError(
                        "PRODUCTION_FORCE_OVERRIDE_UNAUTHORIZED: force=True is prohibited in production without verified recovery authorization token."
                    )
                else:
                    raise RuntimeError(
                        "PRODUCTION_AS_OF_OVERRIDE_UNAUTHORIZED: as_of date override in production requires verified recovery authorization token."
                    )

            effective_reason = recovery_reason or os.environ.get("ARX_RECOVERY_REASON")
            if not effective_reason or len(effective_reason.strip()) < 10:
                if force:
                    raise RuntimeError(
                        "PRODUCTION_FORCE_OVERRIDE_UNAUTHORIZED: Valid recovery_reason (minimum 10 characters) is required for production force override."
                    )
                else:
                    raise RuntimeError(
                        "PRODUCTION_AS_OF_OVERRIDE_UNAUTHORIZED: Valid recovery_reason (minimum 10 characters) is required for production as_of override."
                    )

            effective_scope = recovery_scope or os.environ.get("ARX_RECOVERY_SCOPE", "SINGLE_CYCLE_RECOVERY")
            if effective_scope not in ("SINGLE_CYCLE_RECOVERY", "SAME_DAY_RESUMPTION", "MANUAL_RECOVERY"):
                valid_scopes = "('SINGLE_CYCLE_RECOVERY', 'SAME_DAY_RESUMPTION', 'MANUAL_RECOVERY')"
                if force:
                    raise RuntimeError(
                        f"PRODUCTION_FORCE_OVERRIDE_UNAUTHORIZED: Invalid recovery_scope '{effective_scope}'. Must be one of {valid_scopes}."
                    )
                else:
                    raise RuntimeError(
                        f"PRODUCTION_AS_OF_OVERRIDE_UNAUTHORIZED: Invalid recovery_scope '{effective_scope}'. Must be one of {valid_scopes}."
                    )

            recovery_authorized = True
            logger.warning(
                f"PRODUCTION_RECOVERY_OVERRIDE_AUTHORIZED: Cycle {cycle_id} operating under approved recovery scope '{effective_scope}' "
                f"(reason: '{effective_reason}')."
            )

        try:
            # ── Step 1: Preflight check ───────────────────────────────────────
            self._step1_preflight(dry_run=dry_run)
            steps_completed.append("STEP_1_PREFLIGHT_VERIFIED")

            if dry_run:
                logger.info(f"Dry-run requested: Pipeline preflight verified cleanly for cycle {cycle_id}.")
                return {
                    "cycle_id": cycle_id,
                    "status": "DRY_RUN_SUCCESS",
                    "steps_completed": steps_completed,
                    "universe_id": GOVERNED_UNIVERSE_ID,
                    "issuers_count": len(self.issuers),
                    "timestamp": start_time.isoformat(),
                }

            # ── Step 2: Session advisory lock claim (FIRST MUTATION GATE) ──────
            self._step2_claim_advisory_lock()
            steps_completed.append("STEP_2_ADVISORY_LOCK_CLAIMED")

            # ── Step 3: Atomic durable eligibility check under lock ───────────
            effective_force = bool(force or (self.allow_test_mode and trigger_type == "TEST"))
            should_run, sched_reason = self._step3_check_durable_date_eligibility_under_lock(
                force=effective_force,
                as_of=as_of,
                recovery_authorized=recovery_authorized,
                recovery_reason=recovery_reason or os.environ.get("ARX_RECOVERY_REASON"),
                recovery_scope=recovery_scope or os.environ.get("ARX_RECOVERY_SCOPE", "SINGLE_CYCLE_RECOVERY"),
            )
            if not should_run:
                logger.info(
                    f"DURABLE_ELIGIBILITY_DECLINED_UNDER_LOCK for cycle {cycle_id}: {sched_reason}"
                )
                skip_status = (
                    "SKIPPED_ALREADY_EXECUTED_TODAY"
                    if "ALREADY_EXECUTED_TODAY" in sched_reason
                    else "SKIPPED_NOT_ELIGIBLE"
                )
                steps_completed.append(f"STEP_3_{skip_status}")
                if self.repository and run_request_id and lease_token:
                    try:
                        self.repository.complete_run_request(
                            run_request_id,
                            lease_token=lease_token,
                            status=skip_status,
                            failure_code=None,
                        )
                        logger.info(
                            f"Marked run request {run_request_id} as terminal {skip_status} under lock."
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not record skip terminal status for {run_request_id}: {e}"
                        )
                return {
                    "cycle_id": cycle_id,
                    "status": skip_status,
                    "reason": sched_reason,
                    "steps_completed": steps_completed,
                    "universe_id": GOVERNED_UNIVERSE_ID,
                    "issuers_count": len(self.issuers),
                    "timestamp": start_time.isoformat(),
                }
            steps_completed.append("STEP_3_DURABLE_DATE_ELIGIBILITY_VERIFIED")

            if self.is_trial_2_issuer:
                if self.repository is None or not self.current_slot_id:
                    raise RuntimeError("TRIAL_BUDGET_RESERVATION_UNAVAILABLE")
                reserved_slot_id = self.current_slot_id
                reserved_ny_date = reserved_slot_id.split(":", 1)[0]
                # A committed reservation survives SIGKILL and database rollback.
                # Keep the full eight-request charge even after clean completion;
                # actual transmissions remain separate telemetry, never a refund.
                self.repository.reserve_trial_cycle_budget(
                    trial_id=self.trial_scheduler.trial_id,
                    slot_id=reserved_slot_id,
                    ny_date=reserved_ny_date,
                )
                trial_budget_reserved = True
                steps_completed.append("STEP_3_TRIAL_BUDGET_RESERVED")

            # ── Step 4: Pre-ingestion backup & R2 upload ──────────────────────
            pre_backup_manifest = self._step4_pre_backup(cycle_id)
            steps_completed.append("STEP_4_PRE_BACKUP_COMPLETED")
            steps_completed.append("STEP_4_PRE_BACKUP_UPLOADED_TO_R2")

            # ── Step 5: Backup freshness gate check ───────────────────────────
            self._step5_freshness_gate_check()
            steps_completed.append("STEP_5_FRESHNESS_GATE_VERIFIED")

            # ── Step 6: Genuine SEC Form 4 ingestion ──────────────────────────
            self._check_in_flight_authorization(step_name="STEP_6_SEC_INGESTION_START")
            raw_filings = self._step6_ingest_form4_filings(as_of=as_of)
            steps_completed.append("STEP_6_SEC_FORM4_INGESTED")

            # ── Step 7: Transaction normalization & reconciliation receipt ────
            parsed_txs, receipt_meta = self._step7_parse_and_reconcile(
                raw_filings, cycle_id, as_of=as_of
            )
            receipt_id = receipt_meta.get("receipt_id")
            reconciliation_hash = receipt_meta.get("reconciliation_hash")
            steps_completed.append("STEP_7_TRANSACTIONS_NORMALIZED_AND_RECONCILED")

            # ── Step 8: Atomic snapshot publication or Ingestion-Only completion ──
            if self.ingestion_only:
                self._check_in_flight_authorization(step_name="STEP_8_INGESTION_ONLY_COMPLETION")
                if self.repository and run_request_id and lease_token:
                    self.repository.complete_run_request(
                        run_request_id,
                        result_generation_id=receipt_id,
                        lease_token=lease_token,
                        status="COMPLETED",
                    )
                steps_completed.append("STEP_8_INGESTION_ONLY_COMPLETED")
                snapshot_id = None
                matched_count = 0
            else:
                self._check_in_flight_authorization(step_name="STEP_8_ATOMIC_SNAPSHOT_PUBLICATION")
                snapshot_meta = self._step8_publish_snapshot(
                    parsed_txs,
                    receipt_meta,
                    cycle_id,
                    run_request_id=run_request_id,
                    lease_token=lease_token,
                    as_of=as_of,
                )
                snapshot_id = snapshot_meta.get("snapshot_id")
                matched_count = snapshot_meta.get("matched_count", 0)
                steps_completed.append("STEP_8_ATOMIC_SNAPSHOT_PUBLISHED")

            # ── Step 9: Post-ingestion backup ─────────────────────────────────
            try:
                post_backup_manifest = self._step9_post_backup(cycle_id)
                steps_completed.append("STEP_9_POST_BACKUP_COMPLETED")
            except Exception as backup_err:
                logger.error(
                    f"Post-ingestion backup failed for cycle {cycle_id}: {backup_err}. "
                    "Snapshot was published; marking cycle as DEGRADED_POST_BACKUP_FAILED."
                )
                errors.append(f"POST_BACKUP_FAILED: {backup_err}")
                status = "DEGRADED_POST_BACKUP_FAILED"

            # ── Step 10: Retention pruning ────────────────────────────────────
            pruned = self._step10_prune_retention()
            steps_completed.append(f"STEP_10_RETENTION_PRUNED_{len(pruned)}")

            if status != "DEGRADED_POST_BACKUP_FAILED":
                status = "SUCCESS"

        except Exception as e:
            logger.error(f"Cycle {cycle_id} failed at step {steps_completed[-1] if steps_completed else 'START'}: {e}")
            errors.append(str(e))
            if status != "DEGRADED_POST_BACKUP_FAILED":
                status = "FAILED"
            raise
        finally:
            if trial_budget_reserved:
                try:
                    self.repository.record_trial_cycle_audit(
                        trial_id=self.trial_scheduler.trial_id,
                        slot_id=reserved_slot_id,
                        ny_date=reserved_ny_date,
                        slot_index=getattr(self, "current_slot_index", 0),
                        status=status,
                        requests_transmitted=self.outbound_session.total_transmissions,
                        error="; ".join(errors) if errors else None,
                    )
                except Exception as audit_err:
                    logger.error(f"Failed to record trial cycle audit: {audit_err}")

            # Advisory lock release in finally block
            t_cleanup_start = time.monotonic()
            self._cleanup_lock()
            t_cleanup_end = time.monotonic()
            cleanup_dur = round(t_cleanup_end - t_cleanup_start, 3)
            steps_completed.append("LOCK_RELEASED_CLEANLY")
            cleanup_after_expiry = False
            if self.cycle_deadline_monotonic is not None and t_cleanup_end > self.cycle_deadline_monotonic:
                cleanup_after_expiry = True
                overshoot = round(t_cleanup_end - self.cycle_deadline_monotonic, 3)
                logger.warning(
                    f"CLEANUP_AFTER_EXPIRY: Bounded cleanup finished after cycle execution expiry "
                    f"(cleanup_duration={cleanup_dur}s, overshoot={overshoot}s)"
                )
                steps_completed.append(f"CLEANUP_AFTER_EXPIRY:{cleanup_dur}s")

        return {
            "cycle_id": cycle_id,
            "status": status,
            "steps_completed": steps_completed,
            "snapshot_id": snapshot_id,
            "matched_count": matched_count,
            "universe_size": len(self.issuers),
            "universe_id": GOVERNED_UNIVERSE_ID,
            "receipt_id": receipt_id,
            "reconciliation_hash": reconciliation_hash,
            "pre_backup_id": pre_backup_manifest.get("backup_id") if pre_backup_manifest else None,
            "post_backup_id": post_backup_manifest.get("backup_id") if post_backup_manifest else None,
            "errors": errors,
            "duration_sec": (datetime.now(timezone.utc) - start_time).total_seconds(),
            "cleanup_duration_sec": cleanup_dur,
            "cleanup_after_expiry": cleanup_after_expiry,
        }

    # ── Pipeline Step Implementations ─────────────────────────────────────────

    def _step1_preflight(self, dry_run: bool = False) -> None:
        if os.environ.get("ARX_EMERGENCY_STOP_ACTIVE", "").strip().upper() in ("TRUE", "1", "YES"):
            raise RuntimeError("EMERGENCY_STOP_ACTIVE: Pipeline preflight halted by emergency stop killswitch.")
        """Validates configuration, role authorization, DB connection, and credentials."""
        # 1. Check worker role
        role = get_sec_egress_role()
        # In non-dry-run environments, worker must have OWNER role
        if not dry_run and not self.allow_test_mode:
            require_continuous_authorization(issuers=self.issuers)
        if not dry_run and role != "OWNER":
            raise RuntimeError(
                f"UNAUTHORIZED_WORKER_ROLE: Worker requires ARX_SEC_EGRESS_ROLE=OWNER. Active: '{role}'."
            )

        # 2. Check SEC user agent
        ua = get_sec_user_agent()
        if not ua or "sample" in ua.lower() or "@" not in ua:
            raise RuntimeError(
                f"INVALID_SEC_USER_AGENT: Valid genuine User-Agent with contact email required. Active: '{ua}'."
            )

        # 3. Check storage & db connectivity
        if self.dsn and not dry_run:
            validate_storage_configuration()
            try:
                with get_connection(dsn=self.dsn) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
            except Exception as e:
                raise RuntimeError(f"DATABASE_CONNECTIVITY_FAILED: {e}")

    def _check_in_flight_authorization(self, step_name: str = "EXECUTION") -> None:
        """
        Static stage-boundary authorization and cooperative cancellation check within the worker process.
        """
        # 0a. Check cycle deadline
        if self.cycle_deadline and time.time() >= self.cycle_deadline:
            raise RuntimeError(f"CYCLE_TIMEOUT_EXCEEDED: Execution halted at {step_name} by cycle duration ceiling.")

        # 0. Emergency stop check
        if os.environ.get("ARX_EMERGENCY_STOP_ACTIVE", "").strip().upper() in ("TRUE", "1", "YES"):
            raise RuntimeError(f"EMERGENCY_STOP_ACTIVE: Execution halted at {step_name} by emergency stop killswitch.")

        # 0b. A production OWNER grant must remain valid at every stage boundary.
        if not self.allow_test_mode:
            require_continuous_authorization(issuers=self.issuers)

        # 1. Cooperative in-process cancellation
        if self._cancelled or (self.cancellation_token and self.cancellation_token()):
            raise RuntimeError(
                f"IN_FLIGHT_CANCELLATION: Execution cancelled cooperatively at {step_name}."
            )

        raw_role = os.environ.get("ARX_SEC_EGRESS_ROLE")
        role = raw_role.strip().upper() if raw_role and isinstance(raw_role, str) else None

        # If explicitly revoked or disabled, abort unconditionally (production or test)
        if role in ("DISABLED", "REVOKED"):
            raise RuntimeError(
                f"IN_FLIGHT_REVOCATION_ABORT: Execution aborted at {step_name}. "
                f"Worker requires ARX_SEC_EGRESS_ROLE=OWNER. Active: '{role}'."
            )

        # In live / production mode (outside allow_test_mode), role MUST be OWNER
        if not self.allow_test_mode and role != "OWNER":
            raise RuntimeError(
                f"IN_FLIGHT_REVOCATION_ABORT: Execution aborted at {step_name}. "
                f"Live worker requires ARX_SEC_EGRESS_ROLE=OWNER. Active: '{role or 'UNSET'}'."
            )

    def _require_verified_owner_lock(self) -> None:
        """Require the exclusive bigint lock on this exact PostgreSQL session."""
        conn = self._lock_conn
        if conn is None or getattr(conn, "closed", 0):
            raise RuntimeError("LOCK_LOST_BEFORE_COMMIT: Owner connection unavailable")
        with conn.cursor() as cur:
            cur.execute(
                """SELECT EXISTS (
                    SELECT 1 FROM pg_locks
                    WHERE locktype = 'advisory' AND pid = pg_backend_pid()
                      AND granted AND mode = 'ExclusiveLock' AND objsubid = 1
                      AND classid::bigint = %s AND objid::bigint = %s
                ) AS owner_lock_held""",
                ((LOCK_SEC_EGRESS_OWNER >> 32) & 0xFFFFFFFF,
                 LOCK_SEC_EGRESS_OWNER & 0xFFFFFFFF),
            )
            row = cur.fetchone()
            held = row.get("owner_lock_held") if isinstance(row, dict) else row[0] if row else False
            if not held:
                raise RuntimeError("LOCK_LOST_BEFORE_COMMIT: Session lacks exclusive SEC owner lock")

    def _step2_claim_advisory_lock(self) -> None:
        """Acquires exclusive session advisory lock (LOCK_SEC_EGRESS_OWNER)."""
        if not self.dsn:
            return

        if getattr(self, "_lock_conn_is_preacquired", False) and self._preacquired_lock_conn is not None:
            is_closed = getattr(self._preacquired_lock_conn, "closed", False)
            if is_closed is True or (isinstance(is_closed, int) and not isinstance(is_closed, bool) and is_closed > 0):
                raise RuntimeError("PREACQUIRED_LOCK_CONN_CLOSED: Preacquired advisory lock connection is closed.")
            self._lock_conn = self._preacquired_lock_conn
            logger.info("Reusing preacquired worker advisory lock connection.")
            return

        import psycopg2
        self._lock_conn = psycopg2.connect(self.dsn)
        acquired = acquire_sec_owner_lock(self._lock_conn)
        if not acquired:
            try:
                self._lock_conn.close()
            except Exception:
                pass
            self._lock_conn = None
            raise RuntimeError(
                "CONCURRENT_INGESTION_BLOCKED: Another worker currently holds LOCK_SEC_EGRESS_OWNER."
            )
        logger.info("Acquired session advisory lock LOCK_SEC_EGRESS_OWNER.")

    def _step3_check_durable_date_eligibility_under_lock(
        self,
        force: bool = False,
        as_of: Optional[str] = None,
        recovery_authorized: bool = False,
        recovery_reason: Optional[str] = None,
        recovery_scope: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Durable date and schedule eligibility evaluation under exclusive advisory lock.
        Enforces strict fail-closed acceptance invariant: UNKNOWN_EXECUTION_STATE != ELIGIBLE_TO_RUN.

        Checks:
        1. Production lock connection & repository presence (unconditionally under lock)
        2. Authoritative latest active snapshot query & data_as_of parsing
        3. Force / Recovery authorization boundary
        4. America/New_York dissemination calendar & 22:30 execution window
        """
        # 1. Lock and Repository Validation (Fail-closed)
        if not self.allow_test_mode:
            if not self.repository:
                raise RuntimeError(
                    "DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Active repository required to verify execution state."
                )
            if not self._lock_conn:
                raise RuntimeError(
                    "DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Active advisory lock connection required to verify execution state."
                )
            if getattr(self._lock_conn, "closed", 0) > 0:
                raise RuntimeError(
                    "DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Advisory lock connection closed prior to eligibility check."
                )

        # 2. Query durable state under lock
        if self.is_trial_2_issuer:
            ny_now = self.trial_scheduler.get_ny_now()
            today_ny_str = ny_now.date().isoformat()
            as_of_date = None
            if as_of:
                try:
                    as_of_date = datetime.strptime(as_of, "%Y-%m-%d").date()
                except Exception as e:
                    raise RuntimeError(
                        f"INVALID_AS_OF_DATE: Malformed as_of format '{as_of}', expected YYYY-MM-DD: {e}"
                    ) from e

            # Trial scheduling is slot-based. A receipt from an earlier slot
            # must not suppress all remaining slots in the same NY day.
            progress = None
            if self.repository:
                try:
                    progress = self.repository.get_trial_progress(
                        trial_id=self.trial_scheduler.trial_id,
                        ny_date=today_ny_str,
                        conn=self._lock_conn,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Database query for trial progress failed under lock: {e}"
                    ) from e
            elif not self.allow_test_mode:
                raise RuntimeError("DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Missing repository outside authorized test mode.")

            if progress:
                self.outbound_session.daily_requests_baseline = progress.get("requests_today", 0)
                self.outbound_session.trial_requests_baseline = progress.get("requests_trial", 0)

            eval_dt = None
            if as_of_date:
                eval_dt = datetime(
                    as_of_date.year,
                    as_of_date.month,
                    as_of_date.day,
                    ny_now.hour,
                    ny_now.minute,
                    tzinfo=NY_TZ,
                )

            should_run, slot_id, reason = self.trial_scheduler.evaluate_execution_eligibility(
                now=eval_dt,
                progress=progress,
                issuers=self.issuers,
                force=force,
            )
            self.current_slot_id = slot_id
            return should_run, reason

        last_executed_ny_date = None
        if self.repository:
            # Confirm lock connection is valid if present
            if self._lock_conn and getattr(self._lock_conn, "closed", 0) > 0:
                raise RuntimeError(
                    "DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Advisory lock connection closed prior to eligibility check."
                )

            if self.ingestion_only:
                try:
                    receipt = self.repository.get_latest_reconciliation_receipt(
                        covered_tickers=[i["ticker"] for i in self.issuers],
                        conn=self._lock_conn,
                    )
                    if receipt:
                        wm = receipt.get("certified_watermark") or receipt.get("interval_end")
                        if wm:
                            if isinstance(wm, str):
                                s = wm.strip().replace(" ", "T")
                                if s.endswith("Z"):
                                    s = s[:-1] + "+00:00"
                                dt_utc = datetime.fromisoformat(s)
                            elif isinstance(wm, datetime):
                                dt_utc = wm
                            else:
                                dt_utc = None
                            if dt_utc is not None:
                                if dt_utc.tzinfo is None:
                                    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                                last_executed_ny_date = dt_utc.astimezone(NY_TZ).date()
                except Exception as e:
                    raise RuntimeError(
                        f"DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Database query for latest reconciliation receipt failed under lock: {e}"
                    ) from e
            else:
                try:
                    snap = self.repository.get_latest_active_snapshot(conn=self._lock_conn)
                except Exception as e:
                    raise RuntimeError(
                        f"DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Database query for latest active snapshot failed: {e}"
                    ) from e

                if snap is not None:
                    as_of_val = snap.get("data_as_of")
                    if not as_of_val:
                        raise RuntimeError(
                            "DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Active snapshot record missing mandatory 'data_as_of' field."
                        )
                    try:
                        if isinstance(as_of_val, str):
                            last_executed_ny_date = datetime.strptime(as_of_val[:10], "%Y-%m-%d").date()
                        elif hasattr(as_of_val, "date"):
                            last_executed_ny_date = as_of_val.date()
                        else:
                            raise ValueError(f"Unsupported data_as_of type: {type(as_of_val)}")
                    except Exception as e:
                        raise RuntimeError(
                            f"DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Active snapshot has unparseable or malformed data_as_of '{as_of_val}': {e}"
                        ) from e
        elif not self.allow_test_mode:
            raise RuntimeError(
                "DURABLE_EXECUTION_ELIGIBILITY_UNVERIFIED: Missing repository outside authorized test mode."
            )

        # 3. Force override evaluation under lock
        if force:
            if not self.allow_test_mode:
                if not recovery_authorized:
                    raise RuntimeError(
                        "PRODUCTION_FORCE_OVERRIDE_UNAUTHORIZED: force=True is prohibited in production without verified recovery authorization."
                    )
                scope = recovery_scope or "SINGLE_CYCLE_RECOVERY"
                logger.warning(
                    f"Executing authorized production force recovery under lock: scope={scope}, reason={recovery_reason}"
                )
                return True, f"PRODUCTION_RECOVERY_OVERRIDE:{scope}"
            return True, "FORCE_OVERRIDE"

        eval_dt = None
        if as_of:
            try:
                as_of_date = datetime.strptime(as_of, "%Y-%m-%d").date()
            except Exception as e:
                raise RuntimeError(
                    f"INVALID_AS_OF_DATE: Malformed as_of format '{as_of}', expected YYYY-MM-DD: {e}"
                ) from e
            eval_dt = datetime(
                as_of_date.year,
                as_of_date.month,
                as_of_date.day,
                self.scheduler.target_hour,
                self.scheduler.target_minute,
                tzinfo=NY_TZ,
            )

        return self.scheduler.should_execute_now(
            now=eval_dt,
            last_executed_ny_date=last_executed_ny_date,
            force=force,
        )

    def _validate_backup_manifest(
        self, manifest: Optional[Dict[str, Any]], expected_backup_id: Optional[str] = None
    ) -> None:
        """
        Unconditionally validates pre-ingestion and post-ingestion backup manifests.
        Enforces acceptance invariant: BACKUP_SUCCESS != VERIFIED_RECOVERABLE_BACKUP.

        Validates every mandatory property independently:
        1. Nonempty dictionary manifest.
        2. Expected backup identity (backup_id).
        3. Recognized successful status (status in ("SUCCESS", "VERIFIED") and != "FAILED").
        4. Explicit upload_verified is True (Boolean True, not truthy string or missing).
        5. Valid external storage destination.
        6. Cryptographic integrity evidence (artifact_sha256 or database_sha256).
        7. Artifact size in bytes (> 0).
        """
        if not manifest or not isinstance(manifest, dict) or len(manifest) == 0:
            raise RuntimeError("PRE_BACKUP_FAILED: Empty or invalid backup manifest returned")

        # 1. Status check: must not be FAILED, and must be in ("SUCCESS", "VERIFIED")
        status = manifest.get("status")
        if status == "FAILED":
            raise RuntimeError("PRE_BACKUP_FAILED: Manifest status is FAILED")
        if status not in ("SUCCESS", "VERIFIED"):
            raise RuntimeError(f"PRE_BACKUP_FAILED: Manifest status is not successful: {status}")

        # 2. Upload verification: MUST be explicitly Boolean True
        upload_verified = manifest.get("upload_verified")
        if upload_verified is not True:
            raise RuntimeError(
                f"PRE_BACKUP_FAILED: Backup upload was not verified (upload_verified={upload_verified})"
            )

        # 3. Expected backup identity
        bid = manifest.get("backup_id")
        if not bid or not isinstance(bid, str):
            raise RuntimeError("PRE_BACKUP_FAILED: Manifest missing mandatory 'backup_id'")
        if expected_backup_id and bid != expected_backup_id:
            raise RuntimeError(f"PRE_BACKUP_FAILED: Manifest backup_id mismatch: {bid} != {expected_backup_id}")

        # 4. Storage destination
        dest = manifest.get("storage_destination")
        if not dest or not isinstance(dest, str):
            raise RuntimeError("PRE_BACKUP_FAILED: Manifest missing mandatory 'storage_destination'")

        # 5. Cryptographic integrity evidence (strictly valid 64-character hexadecimal SHA-256)
        sha = manifest.get("artifact_sha256") or manifest.get("database_sha256")
        if (
            not sha
            or not isinstance(sha, str)
            or len(sha) != 64
            or not all(c in "0123456789abcdefABCDEF" for c in sha)
        ):
            raise RuntimeError(f"PRE_BACKUP_FAILED: Manifest missing or malformed integrity sha256: {sha}")

        # Consistency invariant: if both artifact_sha256 and database_sha256 exist, they must match
        art_sha = manifest.get("artifact_sha256")
        db_sha = manifest.get("database_sha256")
        if art_sha and db_sha and art_sha != db_sha:
            raise RuntimeError(
                f"PRE_BACKUP_FAILED: Manifest integrity sha256 mismatch: artifact_sha256 '{art_sha}' != database_sha256 '{db_sha}'"
            )

        # 6. Artifact size
        size = manifest.get("artifact_size_bytes")
        if size is None or not isinstance(size, (int, float)) or size <= 0:
            raise RuntimeError(f"PRE_BACKUP_FAILED: Manifest missing or invalid artifact_size_bytes: {size}")

    def _step4_pre_backup(self, cycle_id: str) -> Dict[str, Any]:
        """Captures pre-ingestion logical dump and uploads to off-host storage."""
        if not self.backup_engine:
            if not self.allow_test_mode:
                raise RuntimeError(
                    "BACKUP_ENGINE_REQUIRED: Live continuous pipeline requires verified backup engine"
                )
            logger.warning("No backup engine configured, skipping pre-backup (test mode).")
            return {
                "backup_id": f"mock-pre-backup-{cycle_id}",
                "status": "VERIFIED",
                "upload_verified": True,
                "storage_destination": "mock://storage",
                "artifact_sha256": "0" * 64,
                "database_sha256": "0" * 64,
                "artifact_size_bytes": 1024,
            }

        backup_id = f"sm-pre-ingest-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
        manifest = self.backup_engine.create_backup(custom_backup_id=backup_id, include_globals=False)
        self._validate_backup_manifest(manifest, expected_backup_id=backup_id)
        return manifest

    def _step5_freshness_gate_check(self) -> None:
        """Confirms that the latest verified backup age is strictly <= 12 hours."""
        if not self.containment_gate:
            return
        if not self.containment_gate.check_ingestion_eligibility():
            state = getattr(self.containment_gate, "current_state", "UNKNOWN")
            err = getattr(self.containment_gate, "last_error_code", "CONTAINMENT_GATE_CLOSED")
            raise BackupAgeContainmentViolationException(
                f"PRE_INGESTION_CONTAINMENT_VIOLATION: Ingestion blocked by containment gate ({state}: {err})."
            )

    # Backward compatibility aliases
    _step5_claim_advisory_lock = _step2_claim_advisory_lock
    _step2_and_3_pre_backup = _step4_pre_backup
    _step4_freshness_gate_check = _step5_freshness_gate_check

    def _step6_ingest_form4_filings(self, as_of: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches SEC Form 4 filings for the governed issuers.
        Persists raw filings into PostgreSQL form4_raw_filings idempotently.
        """
        raw_filings: List[Dict[str, Any]] = []

        # In-flight authorization check before starting ingestion loop
        self._check_in_flight_authorization(step_name="STEP_6_SEC_INGESTION_START")

        max_total_ceiling = 8 if (self.ingestion_only or len(self.issuers) <= 2) else int(os.environ.get("ARX_SEC_MAX_TOTAL_REQUESTS_PER_CYCLE", "40"))
        cycle_request_count = 0

        for issuer in self.issuers:
            if cycle_request_count >= max_total_ceiling:
                raise RuntimeError(
                    f"PILOT_BUDGET_EXCEEDED: Cycle request budget ({max_total_ceiling}) exceeded before fetching {issuer['ticker']}."
                )
            ticker = issuer["ticker"]
            cik = issuer["cik"]
            company_name = issuer.get("company_name", ticker)

            # Bind current issuer to outbound guarded session
            self.outbound_session.set_current_issuer(ticker)

            # In-flight authorization check before each network egress request
            self._check_in_flight_authorization(step_name=f"STEP_6_SEC_INGESTION_{ticker}")

            # Retrieve filings via fetcher or mock
            if self.sec_fetcher and self.allow_test_mode:
                if hasattr(self.sec_fetcher, "session"):
                    self.sec_fetcher.session = self.outbound_session
                if hasattr(self.sec_fetcher, "rate_limiter"):
                    self.sec_fetcher.rate_limiter = self.rate_limiter
                filing_dicts = self.sec_fetcher.fetch_form4_filings(ticker=ticker, cik=cik)
                cycle_request_count += 1
            else:
                # Query SEC EDGAR Atom feed or use database-cached filings via guarded session
                filing_dicts = self._fetch_filings_from_sec(
                    ticker=ticker,
                    cik=cik,
                    company_name=company_name,
                    as_of=as_of,
                )
                cycle_request_count += 1

            for filing in filing_dicts:
                # Calculate deterministic content hash if not already computed
                if not filing.get("content_hash"):
                    filing["content_hash"] = hashlib.sha256(
                        filing.get("xml_content", "").encode("utf-8")
                    ).hexdigest()

                # Persist only after parsing, with availability and reconciliation
                # in the shared Step 7 transaction. A parse failure must be retryable.
                raw_filings.append(filing)

        # Never publish a new empty snapshot after a successful but data-free scan.
        # Individual issuers may legitimately have no recent filings, but a total
        # absence of new filings must preserve the existing active snapshot.
        if not raw_filings and not self.allow_test_mode and not self.ingestion_only:
            raise RuntimeError("NO_NEW_FORM4_FILINGS: Active Smart Money snapshot preserved.")
        logger.info(f"Ingested {len(raw_filings)} Form 4 filing(s) across {len(self.issuers)} governed issuers.")
        return raw_filings

    def _fetch_filings_from_sec(
        self,
        ticker: str,
        cik: str,
        company_name: str,
        as_of: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read all new Form 4/4-A filings within issuer budget via guarded HTTP.

        Default production path: worker -> pipeline -> ContinuousOutboundSession.
        Pure offline tests inject a mocked adapter. No unguarded requests are used.
        A rate limit, invalid SEC response, or malformed URL aborts the cycle.
        """
        if not re.fullmatch(r"[0-9]{1,10}", str(cik)):
            raise RuntimeError("SEC_CIK_INVALID")
        ua = get_sec_user_agent()
        if not ua:
            raise RuntimeError("SEC_USER_AGENT_INVALID")
        self.outbound_session.headers.update({
            "User-Agent": ua,
            "Accept": "application/json, application/xml, text/plain",
        })

        def receive(url: str, allow_404: bool = False) -> Optional[requests.Response]:
            # No hidden HTTP adapter retries. All sends/redirects pass through
            # the guarded session's authorization, budget, rate and host gates.
            response = self.outbound_session.get(
                url, timeout=(5, 15), allow_redirects=True
            )
            if response.status_code == 429:
                raise RuntimeError("SEC_RATE_LIMITED_429")
            if response.status_code == 404:
                if allow_404:
                    return None
                raise RuntimeError("SEC_UPSTREAM_HTTP_ERROR: 404 Not Found")
            if response.status_code != 200:
                raise RuntimeError(f"SEC_UPSTREAM_HTTP_ERROR: {response.status_code}")
            return response

        # Determine prospective start boundary and public-availability cutoff
        prospective_start_utc: Optional[datetime] = None
        if self.is_trial_2_issuer and hasattr(self, "trial_scheduler"):
            st = self.trial_scheduler.start_date
            prospective_start_utc = datetime(st.year, st.month, st.day, 0, 0, tzinfo=NY_TZ).astimezone(timezone.utc)
        elif os.environ.get("ARX_PROSPECTIVE_START_UTC"):
            prospective_start_utc = parse_sec_datetime_utc(os.environ["ARX_PROSPECTIVE_START_UTC"])

        public_cutoff_utc: Optional[datetime] = None
        if as_of:
            public_cutoff_utc = self._snapshot_cutoff(as_of)
        else:
            public_cutoff_utc = datetime.now(timezone.utc)

        catalog = receive(f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json")
        if catalog is None:
            raise RuntimeError("SEC_UPSTREAM_HTTP_ERROR: Submissions catalog not found")
        try:
            recent = catalog.json()["filings"]["recent"]
            forms = recent["form"]
            accessions = recent["accessionNumber"]
            filings_dates = recent["filingDate"]
            accepted_dates = recent["acceptanceDateTime"]
            primary_documents = recent["primaryDocument"]
            if not all(isinstance(v, list) for v in (
                forms, accessions, filings_dates, accepted_dates, primary_documents
            )):
                raise ValueError("SEC submissions arrays not lists")
            if len({len(v) for v in (
                forms, accessions, filings_dates, accepted_dates, primary_documents
            )}) != 1:
                raise ValueError("SEC submissions arrays inconsistent")
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise RuntimeError("SEC_SUBMISSIONS_SCHEMA_INVALID") from e

        if self.repository is None:
            if not self.allow_test_mode:
                raise RuntimeError("SEC_FILING_WATERMARK_DB_UNAVAILABLE")
            known_accessions = set()
        else:
            try:
                known_accessions = set(
                    self.repository.get_existing_accessions_for_issuer(ticker)
                )
            except Exception as exc:
                raise RuntimeError("SEC_FILING_WATERMARK_UNVERIFIED") from exc

        # A governed issuer permits only four actual wire transmissions per cycle.
        # One is reserved for the submissions index; at most three new primary
        # documents may be fetched, absent redirects/fallbacks. If the catalog
        # advertises more new filings, fail closed instead of silently truncating.
        new_indices = []
        seen_accessions = set()
        for index, form in enumerate(forms):
            if form not in ("4", "4/A"):
                continue
            accession = accessions[index]
            if not isinstance(accession, str) or not re.fullmatch(
                r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession
            ):
                raise RuntimeError("SEC_ACCESSION_INVALID")

            # Prospective boundary and public-cutoff check
            try:
                acc_dt = parse_sec_datetime_utc(accepted_dates[index])
            except Exception:
                try:
                    acc_dt = parse_sec_datetime_utc(f"{filings_dates[index]}T00:00:00Z")
                except Exception:
                    acc_dt = None

            if acc_dt is not None:
                if public_cutoff_utc and acc_dt > public_cutoff_utc:
                    continue
                if prospective_start_utc and acc_dt < prospective_start_utc:
                    continue

            if accession not in known_accessions and accession not in seen_accessions:
                new_indices.append(index)
            seen_accessions.add(accession)
        if len(new_indices) > 3:
            raise RuntimeError("SEC_NEW_FILINGS_EXCEED_ISSUER_BUDGET")

        retrieved = []
        for index in new_indices:
            form = forms[index]
            accession = accessions[index]
            doc = primary_documents[index]
            if not isinstance(accession, str) or not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession):
                raise RuntimeError("SEC_ACCESSION_INVALID")

            accession_compact = accession.replace("-", "")
            archive_root = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_compact}/"
            )
            # Safely resolve primary document URL (validates doc path, rejects traversal & host substitution)
            resolved_doc_url = resolve_sec_primary_document_url(archive_root, doc)
            response = receive(resolved_doc_url, allow_404=True)
            xml = response.text if response is not None else ""
            source_url = resolved_doc_url

            # If the primary document was 404 or does not contain ownershipDocument/xml,
            # attempt explicitly classified fallback representations:
            # 1. Cleaned filename if nested path was provided
            # 2. Complete submission text (.txt)
            # Failures like 429, budget exhaustion, cancellation, auth expiry, deadline
            # are raised immediately by receive() or OutboundSession.send() and NEVER swallowed.
            if response is None or ("<ownershipdocument" not in xml.lower() and "<xml" not in xml.lower()):
                clean_doc = doc.split("/")[-1] if "/" in doc else None
                if clean_doc and clean_doc != doc:
                    clean_url = resolve_sec_primary_document_url(archive_root, clean_doc)
                    clean_resp = receive(clean_url, allow_404=True)
                    if clean_resp is not None and clean_resp.status_code == 200:
                        c_text = clean_resp.text
                        if "<ownershipdocument" in c_text.lower() or "<xml" in c_text.lower():
                            xml = c_text
                            source_url = clean_url
                if not xml or ("<ownershipdocument" not in xml.lower() and "<xml" not in xml.lower()):
                    txt_url = resolve_sec_primary_document_url(archive_root, f"{accession}.txt")
                    txt_resp = receive(txt_url)
                    if txt_resp is not None:
                        xml = Form4XmlParser.extract_xml_from_submission_text(txt_resp.text)
                        source_url = txt_url
            if not xml or "<ownershipdocument" not in xml.lower():
                raise RuntimeError("SEC_FORM4_XML_INVALID")

            retrieved.append({
                "ticker": ticker,
                "cik": str(cik).zfill(10),
                "company_name": company_name,
                "accession_number": accession,
                "filing_date": filings_dates[index],
                "accepted_datetime": accepted_dates[index],
                "document_type": form,
                "xml_content": xml,
                "source_url": source_url,
                "content_hash": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
            })

        return retrieved

    def _step7_parse_and_reconcile(
        self,
        raw_filings: List[Dict[str, Any]],
        cycle_id: str,
        as_of: Optional[str] = None,
    ) -> Tuple[List[SmartMoneyTransaction], Dict[str, Any]]:
        """
        Parses Form 4 XML into normalized transactions and persists them.
        Generates and saves immutable cryptographic reconciliation receipt.
        """
        context = self.repository.ingestion_transaction() if self.repository else nullcontext(None)
        with context as conn:
            return self._parse_and_reconcile_transaction(raw_filings, cycle_id, as_of, conn)

    def _parse_and_reconcile_transaction(self, raw_filings, cycle_id, as_of, conn):
        parsed_tx_objs: List[SmartMoneyTransaction] = []
        raw_tx_dicts: List[Dict[str, Any]] = []
        unresolved_amendments = []
        now_iso = datetime.now(timezone.utc).isoformat()
        watermark = as_of or now_iso

        for f in raw_filings:
            if self.repository:
                if not f.get("accepted_datetime"):
                    raise RuntimeError("SEC_PUBLIC_AVAILABILITY_MISSING")
                self.repository.save_raw_filing(f, conn=conn)
                self.repository.save_availability(
                    f["accession_number"], f["accepted_datetime"], source="SEC_INDEX",
                    source_reference=f.get("source_url"), conn=conn,
                )

        parsed_filings_info = []
        for f in raw_filings:
            xml = f.get("xml_content")
            if not xml:
                raise RuntimeError("SEC_FORM4_XML_MISSING")
            accession = f.get("accession_number", "UNKNOWN")
            parsed_filing = Form4XmlParser.parse_filing(
                xml, accession_number=accession,
                acceptance_datetime=f.get("accepted_datetime"),
                source_url=f.get("source_url", ""),
            )
            if not parsed_filing or parsed_filing.disposition not in (
                "FETCHED_AND_PARSED", "FETCHED_NO_RELEVANT_TRANSACTIONS"
            ):
                raise RuntimeError("SEC_FORM4_PARSE_FAILED")
            if parsed_filing.issuer_cik != str(f.get("cik", "")).zfill(10):
                raise RuntimeError("SEC_FORM4_ISSUER_MISMATCH")
            parsed_filings_info.append((f, xml, accession, parsed_filing))
            if parsed_filing.transactions:
                for tx in parsed_filing.transactions:
                    if not tx.issuer_symbol:
                        tx.issuer_symbol = f.get("ticker") or f.get("symbol")
                    if not tx.issuer_cik:
                        tx.issuer_cik = f.get("cik") or parsed_filing.issuer_cik
                    if not tx.issuer_name:
                        tx.issuer_name = f.get("company_name")

                    parsed_tx_objs.append(tx)
                    tx_dict = {
                        "transaction_id": tx.record_key,
                        "accession_number": tx.accession_number,
                        "transaction_date": tx.transaction_date,
                        "transaction_code": tx.transaction_code,
                        "shares": tx.shares,
                        "price_per_share": tx.price_per_share,
                        "total_value_usd": tx.total_value_usd,
                        "reporting_owner_name": tx.reporting_owner_name,
                        "reporting_owner_role": tx.officer_title or ("Director" if tx.is_director else "Insider"),
                        "is_director": tx.is_director,
                        "is_officer": tx.is_officer,
                        "is_ten_percent": tx.is_ten_percent_owner,
                        "officer_title": tx.officer_title,
                        "ownership_type": tx.ownership_nature.value if hasattr(tx.ownership_nature, "value") else str(tx.ownership_nature),
                        "is_derivative": bool(tx.is_derivative),
                        "signal_type": tx.signal_type.value if hasattr(tx.signal_type, "value") else (str(tx.signal_type) if tx.signal_type else None),
                        "acquired_disposed": tx.acquired_disposed,
                        "is_positive_buying_conviction": bool(tx.is_positive_buying_conviction),
                        "is_open_market_purchase_evidence": bool(tx.is_open_market_purchase_evidence),
                    }
                    raw_tx_dicts.append(tx_dict)

        # Persist parsed transactions first so amendment mapping can reference both base and amendment transactions
        if self.repository and raw_tx_dicts:
            self.repository.save_parsed_transactions(raw_tx_dicts, conn=conn)

        # Process amendment relationships and evidence-backed transaction mappings
        for f, xml, accession, parsed_filing in parsed_filings_info:
            is_amend = getattr(parsed_filing, "is_amendment", False) or f.get("document_type") in ("4/A", "4A")
            if is_amend and self.repository:
                base_acc = self.repository.resolve_amendment_base(
                    f, parsed_filing.original_submission_date,
                    parsed_filing.reporting_owner_ciks, conn,
                )
                if not base_acc:
                    unresolved_amendments.append(accession)
                    rel_status = "UNRESOLVED"
                else:
                    base_raw = self.repository.get_raw_filing(base_acc, conn=conn)
                    base_txs = self.repository.get_parsed_transactions_for_accession(base_acc, conn=conn)
                    clf_res = classify_amendment_effect(
                        amendment_filing=parsed_filing,
                        base_filing=base_raw,
                        base_transactions=base_txs,
                        raw_xml=xml,
                    )
                    scope, _ = clf_res
                    mappings = getattr(clf_res, "mappings", [])
                    if scope == SCOPE_PARTIAL_CORRECTION:
                        if mappings:
                            rel_status = SCOPE_PARTIAL_CORRECTION
                            self.repository.record_amendment_transaction_mappings(
                                base_accession=base_acc,
                                amendment_accession=accession,
                                mappings=mappings,
                                base_content_hash=base_raw.get("content_hash"),
                                amendment_content_hash=f.get("content_hash"),
                                conn=conn,
                            )
                        else:
                            rel_status = SCOPE_AMBIGUOUS
                    elif scope in (SCOPE_FULL_REPLACEMENT, SCOPE_ADDITIONAL_TRANSACTION):
                        rel_status = scope
                    else:
                        rel_status = SCOPE_AMBIGUOUS

                self.repository.save_amendment(
                    amendment_accession=accession, base_accession=base_acc,
                    amendment_number=1,
                    resolution_status=rel_status,
                    resolved_at=now_iso if base_acc else None, conn=conn,
                )

        # Cryptographic reconciliation receipt
        reconcile_str = f"{cycle_id}:{watermark}:{len(raw_filings)}:{len(parsed_tx_objs)}"
        reconciliation_hash = hashlib.sha256(reconcile_str.encode("utf-8")).hexdigest()

        cycle_ciks = [i["cik"] for i in self.issuers]
        cycle_tickers = [i["ticker"] for i in self.issuers]
        acc_inventory = [f.get("accession_number") for f in raw_filings if f.get("accession_number")]
        source_hashes = {f.get("accession_number"): f.get("content_hash") for f in raw_filings if f.get("accession_number")}

        # Historical coverage for trial profile is always PARTIAL (never marks historical gaps as COMPLETE)
        if self.ingestion_only or os.environ.get("ARX_CONTINUOUS_AUTH_SCOPE") == "RECURRING_2_ISSUER_FORM4_TRIAL":
            coverage_status = "PARTIAL"
            failure_reasons = [f"UNRESOLVED_AMENDMENT:{a}" for a in unresolved_amendments]
            failure_reasons.append("TRIAL_INTERVAL_PARTIAL_COVERAGE")
        else:
            coverage_status = "PARTIAL" if unresolved_amendments else "COMPLETE"
            failure_reasons = [f"UNRESOLVED_AMENDMENT:{a}" for a in unresolved_amendments]

        if self.is_trial_2_issuer and hasattr(self, "trial_scheduler"):
            cur_slot = getattr(self, "current_slot_id", None)
            failure_reasons.append({
                "trial_id": self.trial_scheduler.trial_id,
                "slot_id": cur_slot,
            })

        receipt_meta = {
            "receipt_id": str(uuid.uuid4()),
            "interval_start": PROSPECTIVE_TRIAL_START_BOUNDARY if self.ingestion_only else (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "interval_end": watermark,
            "certified_watermark": watermark,
            "receipt_version": 1,
            "expected_count": len(raw_filings),
            "accounted_count": len(raw_filings) - len(unresolved_amendments),
            "missing_count": 0,
            "material_gap_count": len(unresolved_amendments),
            "coverage_status": coverage_status,
            "failure_reasons": failure_reasons,
            "covered_ciks": cycle_ciks,
            "covered_tickers": cycle_tickers,
            "accession_inventory": acc_inventory,
            "source_hashes": source_hashes,
            "reconciliation_hash": reconciliation_hash,
            "release_sha": resolve_release_identity(),
            "trial_id": getattr(self.trial_scheduler, "trial_id", None) if hasattr(self, "trial_scheduler") else None,
            "slot_id": getattr(self, "current_slot_id", None),
        }

        # Pre-commit invariants verification:
        # 1. Monotonic cycle deadline check
        if self.cycle_deadline_monotonic is not None and time.monotonic() >= self.cycle_deadline_monotonic:
            raise RuntimeError("CYCLE_TIMEOUT_EXCEEDED: Absolute cycle deadline expired prior to database commit.")
        # 2. Continuous authorization check
        if not self.allow_test_mode:
            require_continuous_authorization(issuers=self.issuers)
            self._require_verified_owner_lock()

        if self.repository:
            self.repository.save_reconciliation_receipt(receipt_meta, conn=conn)

        return parsed_tx_objs, receipt_meta

    @staticmethod
    def _snapshot_cutoff(as_of):
        """Date-only replays include that UTC day; timestamps require an explicit zone."""
        if as_of is None:
            value = datetime.now(timezone.utc)
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(as_of)):
            value = datetime.fromisoformat(str(as_of)).replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc,
            )
        else:
            value = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
            if value.tzinfo is None:
                raise ValueError("SNAPSHOT_CUTOFF_TIMEZONE_REQUIRED")
        return value.astimezone(timezone.utc)

    def _historical_snapshot_inputs(self, cutoff):
        if self.repository is None:
            raise RuntimeError("SNAPSHOT_HISTORICAL_REPOSITORY_REQUIRED")
        watermark = cutoff.isoformat().replace("+00:00", "Z")
        min_date = (cutoff.date() - timedelta(days=30)).isoformat()
        evidence = self.repository.get_snapshot_evidence_as_of(watermark, self.issuers, min_date)
        expected = {i["ticker"]: i["cik"] for i in self.issuers}
        transactions = []
        for symbol, rows in evidence["transactions"].items():
            if symbol not in expected:
                raise RuntimeError("SNAPSHOT_UNEXPECTED_ISSUER")
            for row in rows:
                if str(row.get("issuer_cik", "")).zfill(10) != expected[symbol]:
                    raise RuntimeError("SNAPSHOT_ISSUER_IDENTITY_MISMATCH")
                public_at = datetime.fromisoformat(str(row["public_availability_at"]).replace("Z", "+00:00"))
                if public_at.tzinfo is None or public_at > cutoff:
                    raise RuntimeError("SNAPSHOT_PUBLIC_AVAILABILITY_INVALID")
                item = dict(row)
                item["transaction_date"] = str(row["transaction_date"])[:10]
                if not min_date <= item["transaction_date"] <= cutoff.date().isoformat():
                    raise RuntimeError("SNAPSHOT_TRANSACTION_DATE_INVALID")
                # Canonical parser record keys retain the XML reporting-owner CIK.
                parts = str(item["transaction_id"]).split(":")
                if len(parts) != 4 or parts[0] != item["accession_number"] or not re.fullmatch(r"[0-9]{10}", parts[1]):
                    raise RuntimeError("SNAPSHOT_REPORTING_OWNER_UNVERIFIED")
                item["reporting_owner_cik"] = parts[1]
                tx = _normalize_to_transaction_obj(item)
                # Stored evidence flags remain authoritative; do not infer a buy from code P.
                tx.is_open_market_purchase_evidence = row.get("is_open_market_purchase_evidence") is True
                tx.is_positive_buying_conviction = row.get("is_positive_buying_conviction") is True
                tx.public_availability_at = public_at.isoformat()
                transactions.append(tx)
        return transactions, {
            "source": "POSTGRES_PUBLIC_AVAILABLE",
            "watermark": watermark,
            "interval_start": min_date,
            "coverage_receipt_ids": evidence["coverage_receipt_ids"],
        }

    def _step8_publish_snapshot(
        self,
        parsed_txs: List[Any],
        receipt_meta: Dict[str, Any],
        cycle_id: str,
        run_request_id: Optional[str] = None,
        lease_token: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds immutable snapshot and atomically publishes to scanner_snapshots and
        scanner_active_snapshots via repository.publish_scanner_snapshot_atomic.
        Fails closed: Rejects publication if coverage is incomplete, any issuer fails
        evaluation, or reconciliation hash is missing.
        """
        now_ts = int(time.time())
        snapshot_id = f"SNAP_SM_{now_ts}_{uuid.uuid4().hex[:8]}"
        run_id = f"RUN_SM_{now_ts}_{uuid.uuid4().hex[:6]}"
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cutoff = self._snapshot_cutoff(as_of)
        as_of_watermark = cutoff.isoformat().replace("+00:00", "Z")
        release_sha = resolve_release_identity()

        # 1. Reconciliation gate check
        recon_hash = receipt_meta.get("reconciliation_hash") if receipt_meta else None
        if not recon_hash:
            raise RuntimeError("SNAPSHOT_PUBLICATION_REJECTED: Missing reconciliation receipt hash")

        if receipt_meta.get("material_gap_count", 0) or receipt_meta.get("coverage_status", "COMPLETE") != "COMPLETE":
            raise RuntimeError("SNAPSHOT_PUBLICATION_REJECTED: Reconciliation evidence incomplete")

        # Live snapshots use all eligible persisted evidence, never the latest batch.
        input_provenance = {"source": "EXPLICIT_TEST_FIXTURE"}
        if self.allow_test_mode:
            tx_objects = [_normalize_to_transaction_obj(tx) for tx in parsed_txs]
        else:
            tx_objects, input_provenance = self._historical_snapshot_inputs(cutoff)

        # 3. Evaluate scanner candidate results for each governed issuer
        scanned_successfully_count = 0
        data_complete_count = 0
        unavailable_symbol_count = 0
        unavailable_reasons: Dict[str, str] = {}
        qualifying_states: List[Tuple[SmartMoneyIssuerState, List[str]]] = []

        total_universe = len(self.issuers)
        if total_universe == 0:
            raise RuntimeError("SNAPSHOT_PUBLICATION_REJECTED: Governed universe is empty")

        for issuer in self.issuers:
            ticker = issuer["ticker"]
            cik = issuer["cik"]
            company_name = issuer.get("company_name", ticker)

            # Filter transactions for this issuer
            issuer_txs = [
                tx for tx in tx_objects
                if (tx.issuer_symbol and tx.issuer_symbol.upper() == ticker.upper())
                or (tx.issuer_cik and tx.issuer_cik.lstrip("0") == cik.lstrip("0"))
            ]

            try:
                state = self.scanner.evaluate_issuer_transactions(
                    symbol=ticker,
                    transactions=issuer_txs,
                    lookback_days=30,
                    now_date=cutoff.date().isoformat(),
                )
                scanned_successfully_count += 1
                data_complete_count += 1

                if state.qualifies_for_radar:
                    accessions = list({tx.accession_number for tx in issuer_txs if tx.accession_number})
                    qualifying_states.append((state, accessions))
            except Exception as e:
                unavailable_symbol_count += 1
                unavailable_reasons[ticker] = f"SCANNER_EVALUATION_FAILED: {str(e)}"
                logger.error(f"Scanner evaluation failed for {ticker}: {e}")

        scan_coverage_pct = round((scanned_successfully_count / total_universe) * 100.0, 1)
        data_completeness_pct = round((data_complete_count / total_universe) * 100.0, 1)
        is_complete = (unavailable_symbol_count == 0 and scanned_successfully_count == total_universe)
        coverage_status = "COMPLETE" if is_complete else ("FAILED" if scanned_successfully_count == 0 else "PARTIAL")

        # 4. Fail closed if coverage is incomplete
        if not is_complete or coverage_status != "COMPLETE":
            logger.error(
                f"Snapshot publication rejected: Coverage status is '{coverage_status}' "
                f"(unavailable={unavailable_symbol_count}/{total_universe}). Preserving existing snapshot."
            )
            raise RuntimeError(
                f"SNAPSHOT_PUBLICATION_REJECTED: Coverage incomplete (status={coverage_status}, "
                f"unavailable_count={unavailable_symbol_count})"
            )

        # 5. Build candidate results sorted by conviction score descending
        qualifying_states.sort(key=lambda item: item[0].conviction_score, reverse=True)
        matched_results: List[Dict[str, Any]] = []

        for rank, (state, accessions) in enumerate(qualifying_states, start=1):
            evidence = {
                "symbol": state.symbol,
                "companyName": state.issuer_name,
                "subsignal": "INSIDER_FORM4",
                "convictionScore": round(float(state.conviction_score), 1),
                "reasons": state.conviction_reasons,
                "openMarketPurchaseCount": state.open_market_purchase_count,
                "openMarketSaleCount": state.open_market_sale_count,
                "distinctBuyerCount": state.distinct_buyer_count,
                "grossPurchaseValueUsd": round(float(state.gross_purchase_value_usd), 2),
                "netPurchaseValueUsd": round(float(state.net_purchase_value_usd), 2),
                "clusterBuy": state.cluster_buy,
                "cSuiteBuying": state.c_suite_buying,
                "topInsiderTitles": state.top_insider_titles,
                "signalStatus": "QUALIFIED_INSIDER_ACCUMULATION",
                "provenance": {
                    "issuer_cik": state.issuer_cik,
                    "statutory_basis": "SEC Form 4 Section 16(a)",
                    "filing_accession_numbers": sorted(accessions),
                    "conviction_score": round(float(state.conviction_score), 1),
                },
            }
            candidate = ScannerCandidateResult(
                symbol=state.symbol,
                rank=rank,
                score=round(float(state.conviction_score), 1),
                current_price=0.0,
                scanner_evidence=evidence,
            )
            matched_results.append(candidate.to_dict())

        membership_hash = hashlib.sha256(
            "".join(sorted(i["ticker"] for i in self.issuers)).encode("utf-8")
        ).hexdigest()

        fingerprint_payload = {
            "scanner_id": "SMART_MONEY",
            "snapshot_id": snapshot_id,
            "data_as_of": as_of_watermark,
            "universe_id": GOVERNED_UNIVERSE_ID,
            "matched_count": len(matched_results),
            "results": matched_results,
            "reconciliation_hash": recon_hash,
        }
        semantic_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

        snapshot = {
            "snapshot_id": snapshot_id,
            "scanner_id": "SMART_MONEY",
            "run_id": run_id,
            "api_contract_version": "1.0.0",
            "ruleset_version": "1.0.0",
            "evidence_schema_version": "1.0.0",
            "score_model_version": "1.0.0",
            "data_provenance_version": "1.0.0",
            "universe_version": "sm-eligibility-v2.0",
            "freshness_policy_version": "1.0.0",
            "implementation_release_sha": release_sha,
            "semantic_fingerprint": semantic_fingerprint,
            "generated_at": now_iso,
            "data_as_of": as_of_watermark,
            "status_at_publication": "AVAILABLE",
            "universe_id": GOVERNED_UNIVERSE_ID,
            "universe_size": total_universe,
            "matched_count": len(matched_results),
            "results": matched_results,
            "results_json": matched_results,
            "publication_decision": "PUBLISH",
            "coverage_status": "COMPLETE",
            "reconciliation_hash": recon_hash,
            "provenance": {
                "subsignal": "INSIDER_FORM4",
                "statutory_basis": "SEC Form 4 Section 16(a)",
                "role": "INSIDER_FORM4_POINT_IN_TIME_FLOW",
                "historical_input": input_provenance,
                "reconciliation_receipt_id": receipt_meta.get("receipt_id"),
                "reconciliation_hash": recon_hash,
                "release_sha": release_sha,
            },
            "freshness": {
                "status": "LIVE",
                "data_as_of": as_of_watermark,
                "generated_at": now_iso,
            },
            "universe_metadata": {
                "scope_class": "US_EQUITIES",
                "display_name": "Smart Money Governed Equities Cohort",
                "source_population_count": total_universe,
                "eligible_universe_count": total_universe,
                "universe_id": GOVERNED_UNIVERSE_ID,
                "universe_version": "sm-eligibility-v2.0",
                "membership_hash": membership_hash,
                "construction_status": "COMPLETE",
            },
            "coverage_metadata": {
                "coverage_status": "COMPLETE",
                "scanned_successfully_count": scanned_successfully_count,
                "data_complete_count": data_complete_count,
                "matched_count": len(matched_results),
                "unavailable_symbol_count": unavailable_symbol_count,
                "unresolved_symbol_count": 0,
                "data_completeness_pct": data_completeness_pct,
                "scan_coverage_pct": scan_coverage_pct,
                "unavailable_reasons": unavailable_reasons,
            },
        }

        # In-flight authorization check before advancing active snapshot pointer
        self._check_in_flight_authorization(step_name="STEP_8_ATOMIC_SNAPSHOT_PUBLICATION")

        if self.repository:
            self.repository.publish_scanner_snapshot_atomic(
                snapshot=snapshot,
                run_request_id=run_request_id,
                lease_token=lease_token,
            )

        logger.info(
            f"Published immutable snapshot {snapshot_id} (matches={len(matched_results)}/{total_universe})."
        )
        return snapshot

    def _step9_post_backup(self, cycle_id: str) -> Dict[str, Any]:
        """Captures post-ingestion logical dump and uploads to off-host storage."""
        if not self.backup_engine:
            if not self.allow_test_mode:
                raise RuntimeError(
                    "BACKUP_ENGINE_REQUIRED: Live continuous pipeline requires verified backup engine"
                )
            logger.warning("No backup engine configured, skipping post-backup (test mode).")
            return {
                "backup_id": f"mock-post-backup-{cycle_id}",
                "status": "VERIFIED",
                "upload_verified": True,
                "storage_destination": "mock://storage",
                "artifact_sha256": "0" * 64,
                "database_sha256": "0" * 64,
                "artifact_size_bytes": 1024,
            }

        backup_id = f"sm-post-ingest-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
        manifest = self.backup_engine.create_backup(custom_backup_id=backup_id, include_globals=False)
        self._validate_backup_manifest(manifest, expected_backup_id=backup_id)
        return manifest

    def _step10_prune_retention(self) -> List[str]:
        """Safely prunes older backups beyond retention policy."""
        try:
            return self.retention_manager.prune_backups()
        except Exception as e:
            logger.warning(f"Error during retention pruning: {e}")
            return []

    def _cleanup_lock(self) -> None:
        """Releases session advisory lock on completion or error."""
        if self._lock_conn:
            if getattr(self, "_lock_conn_is_preacquired", False):
                logger.info("Skipping release and close of preacquired worker advisory lock connection.")
                return
            is_closed = getattr(self._lock_conn, "closed", 0)
            if not (is_closed is True or (isinstance(is_closed, int) and not isinstance(is_closed, bool) and is_closed > 0)):
                try:
                    release_sec_owner_lock(self._lock_conn)
                except Exception as e:
                    logger.warning(f"Error releasing session advisory lock: {e}")
                finally:
                    try:
                        self._lock_conn.close()
                    except Exception:
                        pass
                    self._lock_conn = None
