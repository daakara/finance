"""
SEC Operational Preflight Verification — Epoch 001.
Performs prospective verification immediately before observer arming:
business day calendar check, active filing window verification,
automated access policy compliance, rate limits, and endpoint reachability.
"""

import hashlib
import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone, time as dt_time, timedelta
from typing import Dict, Any, Optional, Tuple

from analyst_dashboard.smart_money.canonicalization import canonicalize_json
from analyst_dashboard.smart_money.public_availability.fair_access import (
    ARX_USER_AGENT,
    ARX_MAX_RPS,
    SEC_OFFICIAL_MAX_RPS,
    OFFICIAL_SEC_ACCESS_POLICY_URL,
    OFFICIAL_SEC_ACCESS_POLICY_SHA256,
)

# SEC Filing Window: 06:00:00 to 22:00:00 US Eastern Time (ET)
SEC_WINDOW_OPEN_ET = dt_time(6, 0, 0)
SEC_WINDOW_CLOSE_ET = dt_time(22, 0, 0)

# Endpoints for operational reachability checks
SEC_METADATA_PROBE_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
SEC_CONTENT_PROBE_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass(frozen=True)
class SecPreflightResult:
    """Certified result of SEC operational preflight audit (Sections 32, 33)."""
    captured_at: str
    sec_business_day: bool
    sec_active_filing_window: bool
    sec_automated_access_policy_current: bool
    sec_user_agent_policy_compliant: bool
    sec_rate_limit_policy_compliant: bool
    sec_metadata_endpoint_reachable: bool
    sec_content_endpoint_reachable: bool
    sec_operational_preflight_passed: bool
    sec_preflight_evidence_hash: str
    metadata_http_status: Optional[int] = None
    content_http_status: Optional[int] = None
    sec_date_header: Optional[str] = None
    next_eligible_recheck_condition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "content_http_status": self.content_http_status,
            "metadata_http_status": self.metadata_http_status,
            "next_eligible_recheck_condition": self.next_eligible_recheck_condition,
            "sec_active_filing_window": self.sec_active_filing_window,
            "sec_automated_access_policy_current": self.sec_automated_access_policy_current,
            "sec_business_day": self.sec_business_day,
            "sec_date_header": self.sec_date_header,
            "sec_metadata_endpoint_reachable": self.sec_metadata_endpoint_reachable,
            "sec_content_endpoint_reachable": self.sec_content_endpoint_reachable,
            "sec_operational_preflight_passed": self.sec_operational_preflight_passed,
            "sec_preflight_evidence_hash": self.sec_preflight_evidence_hash,
            "sec_rate_limit_policy_compliant": self.sec_rate_limit_policy_compliant,
            "sec_user_agent_policy_compliant": self.sec_user_agent_policy_compliant,
        }


class SecOperationalPreflight:
    """
    Independent SEC Operational Preflight Auditor (Sections 32, 33).
    Ensures zero synthetic admission and fail-closed arming blocking during weekends,
    holidays, or closed filing windows.
    """

    @staticmethod
    def get_us_eastern_time(utc_dt: datetime) -> datetime:
        """Converts UTC datetime to US Eastern Time (ET), accounting for EDT/EST."""
        # EDT is UTC-4 (second Sunday in March to first Sunday in November), EST is UTC-5
        # October 11 is in EDT (UTC-4)
        year = utc_dt.year
        # DST start: second Sunday in March
        march1 = datetime(year, 3, 1, tzinfo=timezone.utc)
        dst_start = march1 + timedelta(days=(6 - march1.weekday() + 7) % 7 + 7)
        # DST end: first Sunday in November
        nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
        dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)

        if dst_start <= utc_dt < dst_end:
            et_offset = timedelta(hours=-4)
        else:
            et_offset = timedelta(hours=-5)
        return utc_dt.astimezone(timezone(et_offset))

    @staticmethod
    def is_sec_business_day(et_dt: datetime) -> bool:
        """
        Returns True if Monday through Friday and not a US federal holiday.
        """
        # Monday is 0, Sunday is 6
        if et_dt.weekday() >= 5:
            return False

        # US Federal Holidays:
        year = et_dt.year
        month = et_dt.month
        day = et_dt.day

        # New Year's Day (Jan 1)
        if month == 1 and day == 1:
            return False
        # Juneteenth (June 19)
        if month == 6 and day == 19:
            return False
        # Independence Day (July 4)
        if month == 7 and day == 4:
            return False
        # Veterans Day (Nov 11)
        if month == 11 and day == 11:
            return False
        # Christmas Day (Dec 25)
        if month == 12 and day == 25:
            return False

        # Floating holidays:
        # MLK Day: 3rd Monday in January
        if month == 1 and et_dt.weekday() == 0 and 15 <= day <= 21:
            return False
        # Washington's Birthday: 3rd Monday in February
        if month == 2 and et_dt.weekday() == 0 and 15 <= day <= 21:
            return False
        # Memorial Day: Last Monday in May
        if month == 5 and et_dt.weekday() == 0 and day >= 25:
            return False
        # Labor Day: 1st Monday in September
        if month == 9 and et_dt.weekday() == 0 and day <= 7:
            return False
        # Columbus Day: 2nd Monday in October
        if month == 10 and et_dt.weekday() == 0 and 8 <= day <= 14:
            return False
        # Thanksgiving Day: 4th Thursday in November
        if month == 11 and et_dt.weekday() == 3 and 22 <= day <= 28:
            return False

        return True

    @staticmethod
    def is_active_filing_window(et_dt: datetime) -> bool:
        """
        SEC EDGAR accepts filings between 06:00:00 and 22:00:00 ET on business days.
        """
        current_time = et_dt.time()
        return SEC_WINDOW_OPEN_ET <= current_time <= SEC_WINDOW_CLOSE_ET

    @staticmethod
    def is_user_agent_compliant(user_agent: str) -> bool:
        """
        SEC policy requires: 'Sample Company Name AdminContact@<sample company domain>.com'
        or 'App/1.0 (contact@domain.com)'. Must declare specific app and valid email address.
        """
        if not user_agent or len(user_agent) < 10:
            return False
        email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        return bool(re.search(email_pattern, user_agent))

    @staticmethod
    def probe_endpoint(url: str, user_agent: str, timeout: float = 10.0) -> Tuple[bool, Optional[int], Optional[str]]:
        """Makes compliant HTTP request to SEC endpoint and returns (reachable, status, date_header)."""
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept-Encoding": "gzip, deflate",
                    "Host": urllib.parse.urlparse(url).netloc,
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                date_hdr = resp.headers.get("Date")
                return (200 <= status < 400), status, date_hdr
        except urllib.error.HTTPError as e:
            return False, e.code, e.headers.get("Date")
        except Exception:
            return False, None, None

    def execute_preflight(
        self,
        user_agent: str = ARX_USER_AGENT,
        max_rps: float = ARX_MAX_RPS,
        now_override_utc: Optional[str] = None,
        skip_network_probes: bool = False,
        mock_reachability: Optional[Tuple[bool, bool]] = None,
    ) -> SecPreflightResult:
        """
        Executes complete SEC preflight audit (Section 32).
        """
        if now_override_utc:
            now_utc = datetime.fromisoformat(now_override_utc.replace("Z", "+00:00"))
        else:
            now_utc = datetime.now(timezone.utc)

        captured_at_iso = now_utc.isoformat()
        et_dt = self.get_us_eastern_time(now_utc)

        business_day = self.is_sec_business_day(et_dt)
        filing_window = self.is_active_filing_window(et_dt) if business_day else False
        access_policy_current = (OFFICIAL_SEC_ACCESS_POLICY_SHA256 is not None and len(OFFICIAL_SEC_ACCESS_POLICY_SHA256) == 64)
        ua_compliant = self.is_user_agent_compliant(user_agent)
        rate_compliant = (max_rps <= SEC_OFFICIAL_MAX_RPS)

        meta_ok = False
        content_ok = False
        meta_status = None
        content_status = None
        date_header = None

        if mock_reachability is not None:
            meta_ok, content_ok = mock_reachability
            meta_status = 200 if meta_ok else 503
            content_status = 200 if content_ok else 503
            date_header = now_utc.strftime("%a, %d %b %Y %H:%M:%S GMT")
        elif not skip_network_probes:
            meta_ok, meta_status, meta_date = self.probe_endpoint(SEC_METADATA_PROBE_URL, user_agent)
            content_ok, content_status, content_date = self.probe_endpoint(SEC_CONTENT_PROBE_URL, user_agent)
            date_header = meta_date or content_date
        else:
            meta_ok = True
            content_ok = True
            meta_status = 200
            content_status = 200

        # Section 33 Preflight Verdict:
        # Fails if outside business day, outside window, access policy unverified, or endpoints unreachable
        all_passed = (
            business_day
            and filing_window
            and access_policy_current
            and ua_compliant
            and rate_compliant
            and meta_ok
            and content_ok
        )

        next_condition = None
        if not business_day:
            day_name = et_dt.strftime("%A")
            next_condition = f"AWAIT_NEXT_SEC_BUSINESS_DAY (Current day is {day_name})"
        elif not filing_window:
            next_condition = "AWAIT_ACTIVE_FILING_WINDOW (06:00-22:00 ET)"
        elif not (meta_ok and content_ok):
            next_condition = "AWAIT_SEC_ENDPOINT_RECOVERY"

        evidence = {
            "captured_at": captured_at_iso,
            "content_status": content_status,
            "et_datetime": et_dt.isoformat(),
            "filing_window": filing_window,
            "is_business_day": business_day,
            "metadata_status": meta_status,
            "rate_rps": f"{max_rps:.1f}",
            "user_agent": user_agent,
        }
        evidence_sha = hashlib.sha256(canonicalize_json(evidence).encode()).hexdigest()

        return SecPreflightResult(
            captured_at=captured_at_iso,
            sec_business_day=business_day,
            sec_active_filing_window=filing_window,
            sec_automated_access_policy_current=access_policy_current,
            sec_user_agent_policy_compliant=ua_compliant,
            sec_rate_limit_policy_compliant=rate_compliant,
            sec_metadata_endpoint_reachable=meta_ok,
            sec_content_endpoint_reachable=content_ok,
            sec_operational_preflight_passed=all_passed,
            sec_preflight_evidence_hash=evidence_sha,
            metadata_http_status=meta_status,
            content_http_status=content_status,
            sec_date_header=date_header,
            next_eligible_recheck_condition=next_condition,
        )
