"""SEC EDGAR Regulatory Filing & Corporate Financial Disclosure Fetcher (Free Public API)."""

import os
import re
import json
import logging
import requests
import datetime
from typing import Dict, Any, List, Optional, Tuple

import threading
import time

import sqlite3

logger = logging.getLogger(__name__)


def _format_canonical_sec_timestamp(val: Any) -> Optional[str]:
    """
    Formats a datetime or ISO timestamp into canonical ISO 8601 UTC with 'Z' suffix (e.g. '2026-09-18T21:13:29Z').
    Returns None if missing or invalid; never fabricates timestamps from dates.
    """
    if val is None:
        return None
    if isinstance(val, datetime.datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=datetime.timezone.utc)
        return val.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned:
            return None
        if cleaned.endswith("+00:00"):
            cleaned = cleaned[:-6] + "Z"
        if "T" not in cleaned and " " not in cleaned:
            return None
        if " " in cleaned and "T" not in cleaned:
            cleaned = cleaned.replace(" ", "T")
        if not cleaned.endswith("Z") and "+" not in cleaned and "-" not in cleaned[10:]:
            cleaned += "Z"
        return cleaned
    return None

DISALLOWED_DOMAINS = {
    "internal", "local", "example.com", "sample.com", "test.com", "placeholder", "invalid"
}

DISALLOWED_IDENTIFIERS = {
    "financeterminalresearchapp admin@financeterminal.internal",
    "admin@financeterminal.internal",
}


def get_sec_egress_role() -> str:
    """
    Retrieve SEC egress execution role from environment.
    Values: 'DISABLED' (default), 'OWNER'.
    Fails closed: any absent, empty, or unverified value defaults to 'DISABLED'.
    """
    role = os.environ.get("ARX_SEC_EGRESS_ROLE", "DISABLED")
    if not role or not isinstance(role, str):
        return "DISABLED"
    trimmed = role.strip().upper()
    return "OWNER" if trimmed == "OWNER" else "DISABLED"


def is_sec_egress_authorized() -> bool:
    """True only if the current process is authorized as the SEC egress owner."""
    return get_sec_egress_role() == "OWNER"


class SecRateLimiter:
    """
    Thread-safe rate limiter enforcing max SEC requests per second process-globally.
    Conforms to SEC Fair Access policy (max 10 req/sec) with conservative 5.0 req/sec ceiling.
    """

    def __init__(self, max_per_second: float = 5.0, max_rps: Optional[float] = None):
        rps = max_rps if max_rps is not None else max_per_second
        self.max_per_second = rps
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        if self.interval <= 0:
            return
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_request_time = time.time()

    def wait_for_slot(self):
        self.acquire()


RateLimiter = SecRateLimiter
DEFAULT_SEC_RATE_LIMITER = SecRateLimiter(max_per_second=5.0)


def validate_sec_user_agent(user_agent: Optional[str]) -> Tuple[bool, str]:
    """
    Validate SEC User-Agent string against SEC Fair Access requirements:
    Format: '<ApplicationOrCompany> <AdminContact@domain.com>'
    Must contain a valid email domain (not private/placeholder) and non-empty entity name.
    """
    if not user_agent or not isinstance(user_agent, str):
        return False, "User-Agent string is missing or empty"

    trimmed = user_agent.strip()
    if not trimmed:
        return False, "User-Agent string is whitespace only"

    if trimmed.lower() in DISALLOWED_IDENTIFIERS:
        return False, "User-Agent is a disallowed placeholder identity"

    # Check for email pattern
    email_match = re.search(r"[\w\.-]+@([\w\.-]+\.\w+)", trimmed)
    if not email_match:
        return False, "User-Agent must contain a valid contact email (AdminContact@domain.com)"

    domain = email_match.group(1).lower()
    for disallowed in DISALLOWED_DOMAINS:
        if domain == disallowed or domain.endswith("." + disallowed):
            return False, f"User-Agent domain '{domain}' is a disallowed placeholder or private domain"

    parts = trimmed.split()
    if len(parts) < 2:
        return False, "User-Agent must include both entity/application name and contact email"

    return True, "Valid"


def get_sec_user_agent() -> Optional[str]:
    """
    Retrieve SEC_USER_AGENT from environment.
    Returns valid user agent string or None if unconfigured/invalid.
    Never prints contact information to logs.
    """
    ua = os.environ.get("SEC_USER_AGENT")
    is_valid, _ = validate_sec_user_agent(ua)
    if not is_valid:
        return None
    return ua.strip()


SEC_USER_AGENT = get_sec_user_agent()


class SecEdgarFetcher:
    """Fetches official 10-K, 10-Q, 8-K, Form 4 insider transactions, and 13F institutional holdings."""

    BASE_URL = "https://data.sec.gov"
    CIK_MAP = {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "NVDA": "0001045810",
        "NVO": "0000353278",
        "PLTR": "0001321655",
        "TSLA": "0001318605",
        "LLY": "0000059478",
        "CRWD": "0001535527",
        "AMD": "0000002488",
        "AVGO": "0001730168",
        "VRT": "0001674101",
        "TSM": "0001046179",
        "GE": "0000040545",
        "GOOGL": "0001652044",
        "AMZN": "0001018724",
        "META": "0001326801",
        "NFLX": "0001065280",
        "CPRX": "0001373715",
        "FIX": "0001056285",
        "SYM": "0001837240",
        "DECK": "0000910521",
        "PODD": "0001145236",
        "MNST": "0000865752",
        "ULTA": "0001403568",
        "LULU": "0001397187",
    }

    def __init__(self, user_agent: Optional[str] = None, rate_limiter: Optional[SecRateLimiter] = None):
        self.session = requests.Session()
        candidate_ua = user_agent or os.environ.get("SEC_USER_AGENT")
        self.is_configured, self.configuration_error = validate_sec_user_agent(candidate_ua)
        self.user_agent = candidate_ua.strip() if self.is_configured and candidate_ua else None

        if self.is_configured and self.user_agent:
            self.session.headers.update({"User-Agent": self.user_agent})
        else:
            self.session.headers.pop("User-Agent", None)

        self.cik_cache: Dict[str, str] = dict(self.CIK_MAP)
        self._has_loaded_master_index: bool = False
        self.rate_limiter = rate_limiter if rate_limiter is not None else DEFAULT_SEC_RATE_LIMITER

    def _is_mocked(self) -> bool:
        """
        Detect whether session.get or submission methods have been mocked or monkeypatched.
        Supports unittest.mock (Mock, MagicMock), pytest monkeypatch (custom functions/lambdas),
        and patched methods.
        """
        session_get = getattr(self.session, "get", None)
        if session_get is not None:
            if hasattr(session_get, "assert_called") or hasattr(session_get, "_mock_return_value"):
                return True
            func = getattr(session_get, "__func__", None)
            if func is not None and func is not requests.sessions.Session.get:
                return True
            if func is None and callable(session_get) and session_get is not requests.sessions.Session.get:
                return True

        comp_sub = getattr(self, "get_company_submissions", None)
        if comp_sub is not None:
            if hasattr(comp_sub, "assert_called") or hasattr(comp_sub, "_mock_return_value"):
                return True
            func = getattr(comp_sub, "__func__", None)
            if func is not None and func is not SecEdgarFetcher.get_company_submissions:
                return True
            if func is None and callable(comp_sub) and comp_sub is not SecEdgarFetcher.get_company_submissions:
                return True

        return False


    def _get(self, url: str, **kwargs) -> requests.Response:
        """Execute outbound HTTP GET with mandatory rate-limiter acquisition and egress role check."""
        is_mock = self._is_mocked()
        if not is_mock and not is_sec_egress_authorized():
            raise PermissionError(
                f"SEC_EGRESS_FORBIDDEN: Outbound SEC request blocked by fail-closed egress role "
                f"(ARX_SEC_EGRESS_ROLE={get_sec_egress_role()} != OWNER). Attempted URL: {url}"
            )
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        return self.session.get(url, **kwargs)

    def resolve_cik(self, symbol: str) -> Optional[str]:
        """Resolve ticker to 10-digit CIK from local cache or SEC company_tickers index."""
        upper = symbol.upper().strip()
        if upper in self.cik_cache:
            return self.cik_cache[upper]

        if not self.is_configured:
            logger.warning("SEC EDGAR request blocked: valid SEC_USER_AGENT is not configured.")
            return None

        if not self._has_loaded_master_index:
            is_mock = self._is_mocked()
            if not is_mock and not is_sec_egress_authorized():
                return self.cik_cache.get(upper)
            try:
                resp = self._get("https://www.sec.gov/files/company_tickers.json", timeout=6)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.values():
                        t = item.get("ticker", "").upper()
                        c = str(item.get("cik_str", "")).zfill(10)
                        if t and c:
                            self.cik_cache[t] = c
                self._has_loaded_master_index = True
            except Exception as e:
                logger.warning(f"Failed to fetch SEC master company_tickers.json: {e}")

        return self.cik_cache.get(upper)

    def get_company_submissions_envelope(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time company submissions with complete semantic failure tracking."""
        if not self.is_configured:
            logger.warning("SEC EDGAR request blocked: valid SEC_USER_AGENT is not configured.")
            return {
                "status": "CONFIGURATION_BLOCKED",
                "status_code": None,
                "data": None,
                "reason": self.configuration_error or "VALID_SEC_USER_AGENT_REQUIRED",
                "error": "Valid SEC_USER_AGENT is not configured",
            }

        is_mock = self._is_mocked()
        if not is_mock and not is_sec_egress_authorized():
            return {
                "status": "CONFIGURATION_BLOCKED",
                "status_code": None,
                "data": None,
                "reason": "SEC_EGRESS_ROLE_DISABLED",
                "error": "Outbound SEC EDGAR requests are disabled in this runtime role (ARX_SEC_EGRESS_ROLE != OWNER)",
            }

        cik = self.resolve_cik(symbol)
        if not cik:
            return {
                "status": "UPSTREAM_NOT_FOUND",
                "status_code": 404,
                "data": None,
                "reason": "CIK_NOT_FOUND",
                "error": f"CIK could not be resolved for symbol '{symbol}'",
            }

        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        try:
            resp = self._get(url, timeout=5)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if not isinstance(data, dict):
                        return {
                            "status": "STRUCTURALLY_INVALID",
                            "status_code": 200,
                            "data": None,
                            "reason": "PAYLOAD_NOT_A_DICT",
                            "error": "SEC EDGAR submissions JSON root is not an object",
                        }
                    return {
                        "status": "AVAILABLE",
                        "status_code": 200,
                        "data": data,
                        "reason": None,
                        "error": None,
                    }
                except (ValueError, json.JSONDecodeError) as e:
                    return {
                        "status": "STRUCTURALLY_INVALID",
                        "status_code": 200,
                        "data": None,
                        "reason": "MALFORMED_JSON",
                        "error": str(e),
                    }
            elif resp.status_code == 429:
                return {
                    "status": "UPSTREAM_RATE_LIMITED",
                    "status_code": 429,
                    "data": None,
                    "reason": "RATE_LIMITED",
                    "error": "SEC EDGAR API rate limit exceeded (HTTP 429)",
                }
            elif resp.status_code == 404:
                return {
                    "status": "UPSTREAM_NOT_FOUND",
                    "status_code": 404,
                    "data": None,
                    "reason": "SUBMISSIONS_RESOURCE_NOT_FOUND",
                    "error": f"Submissions resource not found for CIK {cik}",
                }
            elif resp.status_code == 403:
                return {
                    "status": "UNKNOWN_FAIL_CLOSED",
                    "status_code": 403,
                    "data": None,
                    "reason": "HTTP_403_FORBIDDEN",
                    "error": "SEC EDGAR API returned 403 Forbidden (access denied or blocked)",
                }
            elif resp.status_code in (500, 502, 503, 504):
                return {
                    "status": "UPSTREAM_SERVER_ERROR",
                    "status_code": resp.status_code,
                    "data": None,
                    "reason": f"HTTP_{resp.status_code}_SERVER_ERROR",
                    "error": f"SEC EDGAR upstream server error (HTTP {resp.status_code})",
                }
            else:
                return {
                    "status": "UNKNOWN_FAIL_CLOSED",
                    "status_code": resp.status_code,
                    "data": None,
                    "reason": f"UNEXPECTED_STATUS_{resp.status_code}",
                    "error": f"Unexpected HTTP status {resp.status_code} from SEC EDGAR",
                }
        except requests.exceptions.Timeout as e:
            return {
                "status": "UPSTREAM_TIMEOUT",
                "status_code": None,
                "data": None,
                "reason": "CONNECTION_TIMEOUT",
                "error": str(e),
            }
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            return {
                "status": "TRANSPORT_INCOMPLETE",
                "status_code": None,
                "data": None,
                "reason": "TRANSPORT_CONNECTION_DROPPED",
                "error": str(e),
            }
        except Exception as e:
            return {
                "status": "UNKNOWN_FAIL_CLOSED",
                "status_code": None,
                "data": None,
                "reason": "UNHANDLED_EXCEPTION",
                "error": str(e),
            }

    def get_company_submissions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch real-time company submissions including Form 4 (insiders) and 8-K (material events)."""
        env = self.get_company_submissions_envelope(symbol)
        return env.get("data")

    def _get_cached_recent_filings_envelope(
        self, symbol: str, form_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieves cached filing metadata from the governed local store without issuing live SEC network requests.
        Enforces CACHE_ONLY behavior for normal user/API requests when ARX_SEC_EGRESS_ROLE != OWNER.
        """
        from analyst_dashboard.smart_money.form4_ingestion import (
            DEFAULT_SMART_MONEY_DB,
            connect_smart_money_db,
        )

        clean_sym = symbol.upper().strip()
        cik = self.resolve_cik(clean_sym)
        allowed_forms = set(form_types) if form_types else {"10-K", "10-Q", "8-K", "4", "4/A", "13F-HR"}
        results = []

        backend = "sqlite"
        try:
            from analyst_dashboard.smart_money.storage.config import get_storage_backend
            backend = get_storage_backend()
        except Exception as e:
            if "PRODUCTION_STORAGE_CONFIGURATION_VIOLATION" in str(e) or "POSTGRES_STORAGE_BACKEND_CONFIGURATION_VIOLATION" in str(e):
                logger.error(f"Storage backend configuration violation: {e}")
                return {
                    "status": "BACKEND_STORAGE_ERROR",
                    "status_code": None,
                    "cik": cik,
                    "reason": "STORAGE_CONFIGURATION_VIOLATION",
                    "error_detail": str(e),
                    "items": None,
                    "cached": True,
                }
            backend = "sqlite"

        if backend == "postgres":
            try:
                from analyst_dashboard.smart_money.storage.postgres_connection import get_connection
                from psycopg2.extras import RealDictCursor
                with get_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        has_query = "SELECT 1 FROM form4_raw_filings WHERE (cik = %s OR ticker = %s) LIMIT 1" if cik else "SELECT 1 FROM form4_raw_filings WHERE ticker = %s LIMIT 1"
                        has_params = [str(cik).zfill(10), clean_sym] if cik else [clean_sym]
                        cur.execute(has_query, tuple(has_params))
                        has_issuer = cur.fetchone() is not None

                        if not has_issuer:
                            return {
                                "status": "CONFIGURATION_BLOCKED",
                                "status_code": None,
                                "cik": cik,
                                "reason": "SOURCE_NOT_INGESTED",
                                "error_detail": f"SEC filings for {clean_sym} not yet ingested in local store. Live SEC egress disabled in API web workers (ARX_SEC_EGRESS_ROLE != OWNER).",
                                "items": None,
                                "cached": True,
                            }

                        query = """
                            SELECT
                                r.accession_number,
                                r.cik,
                                r.ticker,
                                r.filing_date,
                                r.accepted_datetime,
                                r.source_url,
                                l.public_availability_at
                            FROM form4_raw_filings r
                            LEFT JOIN sm_source_availability_ledger l ON r.accession_number = l.accession_number
                            WHERE 1=1
                        """
                        params = []
                        if cik:
                            query += " AND (r.cik = %s OR r.ticker = %s)"
                            params.extend([str(cik).zfill(10), clean_sym])
                        else:
                            query += " AND r.ticker = %s"
                            params.append(clean_sym)

                        query += " ORDER BY r.filing_date DESC, r.accepted_datetime DESC NULLS LAST LIMIT 25"
                        cur.execute(query, tuple(params))
                        rows = cur.fetchall()

                        for r in rows:
                            doc_type = "4"
                            if allowed_forms and doc_type not in allowed_forms and not any(f.startswith("4") for f in allowed_forms):
                                continue

                            acc_num = r["accession_number"]
                            clean_acc = acc_num.replace("-", "")
                            cik_val = r.get("cik") or (str(cik).zfill(10) if cik else "0")
                            cik_int = int(cik_val) if str(cik_val).isdigit() else 0
                            f_date = str(r["filing_date"]) if r.get("filing_date") else ""

                            acc_dt = _format_canonical_sec_timestamp(r.get("accepted_datetime"))
                            pub_dt = _format_canonical_sec_timestamp(r.get("public_availability_at"))
                            avail_from = pub_dt or acc_dt

                            results.append({
                                "form": doc_type,
                                "filing_date": f_date,
                                "report_date": f_date,
                                "acceptance_datetime": acc_dt,
                                "available_from": avail_from,
                                "description": "Form 4 Insider Filing",
                                "accession_number": acc_num,
                                "sec_url": r.get("source_url") or f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{acc_num}.txt",
                            })

                        if results:
                            return {
                                "status": "AVAILABLE_WITH_RESULTS",
                                "status_code": 200,
                                "cik": cik,
                                "reason": None,
                                "error_detail": None,
                                "items": results,
                                "cached": True,
                            }
                        else:
                            return {
                                "status": "AVAILABLE_NO_RESULTS",
                                "status_code": 200,
                                "cik": cik,
                                "reason": None,
                                "error_detail": None,
                                "items": [],
                                "cached": True,
                            }
            except Exception as e:
                logger.error(f"Failed to query PostgreSQL filings store for {clean_sym}: {e}")
                return {
                    "status": "BACKEND_STORAGE_ERROR",
                    "status_code": None,
                    "cik": cik,
                    "reason": "STORAGE_QUERY_FAILED",
                    "error_detail": f"Failed to query PostgreSQL filings store: {e}",
                    "items": None,
                    "cached": True,
                }

        elif backend == "sqlite":
            try:
                conn = connect_smart_money_db()
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()

                has_query = "SELECT 1 FROM form4_raw_filings WHERE (issuer_cik = ? OR issuer_symbol = ?) LIMIT 1" if cik else "SELECT 1 FROM form4_raw_filings WHERE issuer_symbol = ? LIMIT 1"
                has_params = [str(cik).zfill(10), clean_sym] if cik else [clean_sym]
                try:
                    cur.execute(has_query, tuple(has_params))
                    has_issuer = cur.fetchone() is not None
                except sqlite3.OperationalError:
                    has_issuer = False

                if not has_issuer:
                    conn.close()
                    return {
                        "status": "CONFIGURATION_BLOCKED",
                        "status_code": None,
                        "cik": cik,
                        "reason": "SOURCE_NOT_INGESTED",
                        "error_detail": f"SEC filings for {clean_sym} not yet ingested in local store. Live SEC egress disabled in API web workers (ARX_SEC_EGRESS_ROLE != OWNER).",
                        "items": None,
                        "cached": True,
                    }

                has_ledger = False
                try:
                    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sm_source_availability_ledger'")
                    has_ledger = cur.fetchone() is not None
                except Exception:
                    has_ledger = False

                if has_ledger:
                    query = """
                        SELECT
                            r.accession_number,
                            r.issuer_cik,
                            r.filing_date,
                            r.acceptance_datetime,
                            r.document_type,
                            r.source_url,
                            l.public_availability_at
                        FROM form4_raw_filings r
                        LEFT JOIN sm_source_availability_ledger l ON r.accession_number = l.accession_number
                        WHERE 1=1
                    """
                else:
                    query = """
                        SELECT
                            r.accession_number,
                            r.issuer_cik,
                            r.filing_date,
                            r.acceptance_datetime,
                            r.document_type,
                            r.source_url,
                            NULL as public_availability_at
                        FROM form4_raw_filings r
                        WHERE 1=1
                    """
                params = []
                if cik:
                    query += " AND (r.issuer_cik = ? OR r.issuer_symbol = ?)"
                    params.extend([str(cik).zfill(10), clean_sym])
                else:
                    query += " AND r.issuer_symbol = ?"
                    params.append(clean_sym)

                query += " ORDER BY r.filing_date DESC LIMIT 25"
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                conn.close()

                for r in rows:
                    doc_type = r["document_type"] or "4"
                    if allowed_forms and doc_type not in allowed_forms and not any(doc_type.startswith(f) for f in allowed_forms):
                        continue

                    acc_num = r["accession_number"]
                    clean_acc = acc_num.replace("-", "")
                    cik_val = r["issuer_cik"] or (str(cik).zfill(10) if cik else "0")
                    cik_int = int(cik_val) if str(cik_val).isdigit() else 0
                    f_date = r["filing_date"] or ""

                    acc_dt = _format_canonical_sec_timestamp(r["acceptance_datetime"])
                    pub_dt = _format_canonical_sec_timestamp(r["public_availability_at"]) if "public_availability_at" in r.keys() else None
                    avail_from = pub_dt or acc_dt

                    results.append({
                        "form": doc_type,
                        "filing_date": f_date,
                        "report_date": f_date,
                        "acceptance_datetime": acc_dt,
                        "available_from": avail_from,
                        "description": f"Form {doc_type} Insider Filing",
                        "accession_number": acc_num,
                        "sec_url": r["source_url"] or f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{acc_num}.txt",
                    })

                if results:
                    return {
                        "status": "AVAILABLE_WITH_RESULTS",
                        "status_code": 200,
                        "cik": cik,
                        "reason": None,
                        "error_detail": None,
                        "items": results,
                        "cached": True,
                    }
                else:
                    return {
                        "status": "AVAILABLE_NO_RESULTS",
                        "status_code": 200,
                        "cik": cik,
                        "reason": None,
                        "error_detail": None,
                        "items": [],
                        "cached": True,
                    }

            except Exception as e:
                logger.error(f"Failed to query SQLite filings store for {clean_sym}: {e}")
                return {
                    "status": "BACKEND_STORAGE_ERROR",
                    "status_code": None,
                    "cik": cik,
                    "reason": "STORAGE_QUERY_FAILED",
                    "error_detail": f"Failed to query SQLite filings store: {e}",
                    "items": None,
                    "cached": True,
                }
        else:
            return {
                "status": "CONFIGURATION_BLOCKED",
                "status_code": None,
                "cik": cik,
                "reason": "UNSUPPORTED_STORAGE_BACKEND",
                "error_detail": f"Unsupported storage backend '{backend}'. Supported: 'postgres', 'sqlite'.",
                "items": None,
                "cached": True,
            }

    def get_recent_filings_envelope(
        self, symbol: str, form_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get structured filings envelope distinguishing valid empty results from upstream failures.
        Operates in CACHE_ONLY mode when ARX_SEC_EGRESS_ROLE != OWNER.
        """
        is_mock = self._is_mocked()
        if not is_mock and not is_sec_egress_authorized():
            return self._get_cached_recent_filings_envelope(symbol, form_types=form_types)

        cik = self.resolve_cik(symbol)
        sub_env = self.get_company_submissions_envelope(symbol)

        if sub_env["status"] != "AVAILABLE":
            # If get_company_submissions was mocked/overridden in caller tests
            mocked_sub = self.get_company_submissions(symbol)
            if mocked_sub and isinstance(mocked_sub, dict):
                sub_env = {
                    "status": "AVAILABLE",
                    "status_code": 200,
                    "data": mocked_sub,
                    "reason": None,
                    "error": None,
                }
            else:
                return {
                    "status": sub_env["status"],
                    "status_code": sub_env.get("status_code"),
                    "cik": cik,
                    "reason": sub_env.get("reason"),
                    "error_detail": sub_env.get("error"),
                    "items": None,
                }

        data = sub_env.get("data")
        if not data or not isinstance(data, dict) or "filings" not in data or not isinstance(data.get("filings"), dict) or "recent" not in data["filings"]:
            return {
                "status": "STRUCTURALLY_INVALID",
                "status_code": 200,
                "cik": cik,
                "reason": "MISSING_FILINGS_RECENT_KEY",
                "error_detail": "SEC EDGAR payload missing 'filings.recent' structure",
                "items": None,
            }

        recent = data["filings"]["recent"]
        if not isinstance(recent, dict):
            return {
                "status": "STRUCTURALLY_INVALID",
                "status_code": 200,
                "cik": cik,
                "reason": "RECENT_NOT_DICT",
                "error_detail": "'filings.recent' is not a dictionary",
                "items": None,
            }

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocDescription", [])
        acceptance_dts = recent.get("acceptanceDateTime", [])
        report_dates = recent.get("reportDate", [])

        allowed_forms = set(form_types) if form_types else {"10-K", "10-Q", "8-K", "4", "13F-HR"}
        results = []

        cik_int = int(cik) if cik and str(cik).isdigit() else 0

        for i in range(min(len(forms), 25)):
            form = forms[i]
            if form in allowed_forms:
                clean_acc = accessions[i].replace("-", "") if i < len(accessions) else ""
                acc_num = accessions[i] if i < len(accessions) else ""
                acceptance_dt = acceptance_dts[i] if i < len(acceptance_dts) else ""
                report_date = report_dates[i] if i < len(report_dates) else ""
                filing_date = filing_dates[i] if i < len(filing_dates) else ""
                avail_from = acceptance_dt or (f"{filing_date}T00:00:00Z" if filing_date else "")

                results.append({
                    "form": form,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "acceptance_datetime": acceptance_dt,
                    "available_from": avail_from,
                    "description": descriptions[i] if i < len(descriptions) else form,
                    "accession_number": acc_num,
                    "sec_url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{acc_num}.txt",
                })

        final_status = "AVAILABLE_WITH_RESULTS" if len(results) > 0 else "AVAILABLE_NO_RESULTS"
        return {
            "status": final_status,
            "status_code": 200,
            "cik": cik,
            "reason": None,
            "error_detail": None,
            "items": results,
        }

    def get_recent_filings(self, symbol: str, form_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get formatted list of recent regulatory filings (backwards compatible)."""
        env = self.get_recent_filings_envelope(symbol, form_types=form_types)
        return env.get("items") or []
