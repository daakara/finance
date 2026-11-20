"""
ARX Terminal — Smart Money Market-Wide Scheduler Design (Gate B6 & Phase M).

Implements distributed-fenced, lease-governed scheduler execution for Smart Money.
Adheres to strict fail-closed production safety:
- Stable job identity: SM_MARKET_WIDE_DAILY_SCAN_V1
- Distributed fencing lease via DurableRunCoordinator (resource: smart_money_pipeline)
- Anti-rollback, lease TTL expiration, and crash recovery
- Rejects overlapping runs immediately with 409 Conflict / OVERLAPPING_SCANS_PROHIBITED
- Separates local manual test invocation from un-armed prospective production accession
"""

import os
import time
import uuid
import logging
from datetime import datetime, date, timedelta, timezone
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

from analyst_dashboard.smart_money.freshness_authority import (
    is_sec_dissemination_date,
    NY_TZ,
)

from analyst_dashboard.coordination import (
    DurableRunCoordinator,
    CoordinationStore,
    TriggerType,
    AcquisitionStatus,
    PRODUCTION_LEASE_POLICY,
    LeasePolicy,
)
from analyst_dashboard.analyzers.scanner_runner import (
    SmartMoneyScannerRunner,
    RESOURCE_KEY_SMART_MONEY_PIPELINE,
)

logger = logging.getLogger(__name__)

JOB_ID_SMART_MONEY_MARKET_WIDE = "SM_MARKET_WIDE_DAILY_SCAN_V1"


@dataclass
class SchedulerExecutionResult:
    job_id: str
    run_id: str
    triggered_at: str
    completed_at: str
    trigger_type: str
    status: str  # SUCCESS, FAILED, OVERLAP_REJECTED, PREFLIGHT_BLOCKED
    snapshot_id: Optional[str] = None
    candidate_count: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SmartMoneyMarketWideScheduler:
    """
    Durable scheduler runner for market-wide Smart Money pipeline.
    Ensures safe distributed execution without production armed status.
    """

    def __init__(
        self,
        scanner_runner: Optional[SmartMoneyScannerRunner] = None,
        coordinator: Optional[DurableRunCoordinator] = None,
        lease_policy: Optional[LeasePolicy] = None,
    ):
        self.scanner_runner = scanner_runner or SmartMoneyScannerRunner()
        self.lease_policy = lease_policy or PRODUCTION_LEASE_POLICY
        self.coordinator = coordinator or DurableRunCoordinator(
            resource_key=RESOURCE_KEY_SMART_MONEY_PIPELINE,
            policy=self.lease_policy,
        )

    def execute_scheduled_cycle(
        self,
        trigger_type: TriggerType = TriggerType.SCHEDULED,
        as_of_watermark: Optional[str] = None,
        universe_build_id: Optional[str] = None,
        operator_request_id: Optional[str] = None,
    ) -> SchedulerExecutionResult:
        """
        Executes a single fenced scheduler cycle.
        Rejects overlapping concurrent execution fail-closed.
        """
        start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        run_id = f"SCHED_RUN_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        try:
            envelope = self.scanner_runner.execute_market_wide_scan(
                as_of_watermark=as_of_watermark,
                universe_build_id=universe_build_id,
                trigger_type=trigger_type,
                operator_request_id=operator_request_id or run_id,
            )
            end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            snap_meta = envelope.get("snapshot", {})
            return SchedulerExecutionResult(
                job_id=JOB_ID_SMART_MONEY_MARKET_WIDE,
                run_id=run_id,
                triggered_at=start_time,
                completed_at=end_time,
                trigger_type=trigger_type.value,
                status="SUCCESS",
                snapshot_id=snap_meta.get("snapshot_id"),
                candidate_count=snap_meta.get("matched_count"),
            )
        except RuntimeError as e:
            if "OVERLAPPING_SMART_MONEY_SCANS_PROHIBITED" in str(e):
                logger.warning(f"Smart Money scheduled run {run_id} rejected due to overlap: {e}")
                end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                return SchedulerExecutionResult(
                    job_id=JOB_ID_SMART_MONEY_MARKET_WIDE,
                    run_id=run_id,
                    triggered_at=start_time,
                    completed_at=end_time,
                    trigger_type=trigger_type.value,
                    status="OVERLAP_REJECTED",
                    error=str(e),
                )
            raise
        except Exception as e:
            logger.error(f"Smart Money scheduled cycle {run_id} failed: {e}")
            end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return SchedulerExecutionResult(
                job_id=JOB_ID_SMART_MONEY_MARKET_WIDE,
                run_id=run_id,
                triggered_at=start_time,
                completed_at=end_time,
                trigger_type=trigger_type.value,
                status="FAILED",
                error=str(e),
            )


class SmartMoneyTimezoneScheduler:
    """
    Timezone-safe scheduler for Smart Money continuous daily ingestion.
    Enforces:
    - 22:30 America/New_York target execution window (Mon-Fri)
    - DST invariance (automatically handles EDT vs EST shifts relative to UTC)
    - SEC EDGAR dissemination calendar awareness (skips weekends and federal dissemination holidays)
    - Duplicate execution avoidance for the same calendar date
    - Execution window tolerance and catchup policy
    """
    def __init__(
        self,
        target_hour: int = 22,
        target_minute: int = 30,
        window_minutes: int = 60,
    ):
        self.target_hour = target_hour
        self.target_minute = target_minute
        self.window_minutes = window_minutes

    def get_ny_now(self, now: Optional[datetime] = None) -> datetime:
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(NY_TZ)

    def is_dissemination_day(self, dt: Optional[datetime] = None) -> bool:
        ny_dt = self.get_ny_now(dt)
        return is_sec_dissemination_date(ny_dt.date())

    def should_execute_now(
        self,
        now: Optional[datetime] = None,
        last_executed_ny_date: Optional[date] = None,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        Evaluates whether an ingestion cycle should execute at this moment.
        Returns: (should_run: bool, reason: str)
        """
        if force:
            return True, "FORCE_OVERRIDE"

        ny_now = self.get_ny_now(now)
        today = ny_now.date()

        # Check if today is a dissemination day
        if not is_sec_dissemination_date(today):
            return False, f"NON_DISSEMINATION_DAY: {today.strftime('%A')} ({today.isoformat()}) is not an SEC filing day"

        # Check if already executed today
        if last_executed_ny_date and last_executed_ny_date == today:
            return False, f"ALREADY_EXECUTED_TODAY: cycle already recorded for {today.isoformat()}"

        # Target time today in NY
        target_time = datetime(today.year, today.month, today.day, self.target_hour, self.target_minute, tzinfo=NY_TZ)
        window_end = target_time + timedelta(minutes=self.window_minutes)

        if ny_now < target_time:
            diff = (target_time - ny_now).total_seconds() / 60
            return False, f"BEFORE_SCHEDULED_TIME: current time {ny_now.strftime('%H:%M %Z')} is {diff:.1f}m before target {target_time.strftime('%H:%M %Z')}"

        if ny_now > window_end:
            return False, f"MISSED_WINDOW: current time {ny_now.strftime('%H:%M %Z')} is past execution window closing {window_end.strftime('%H:%M %Z')}"

        return True, f"WITHIN_EXECUTION_WINDOW: {ny_now.strftime('%H:%M %Z')} matches daily target {target_time.strftime('%H:%M %Z')}"

    def next_scheduled_run(self, from_dt: Optional[datetime] = None) -> datetime:
        """Calculates the next scheduled run datetime in America/New_York."""
        ny_from = self.get_ny_now(from_dt)
        candidate_date = ny_from.date()
        target_today = datetime(candidate_date.year, candidate_date.month, candidate_date.day, self.target_hour, self.target_minute, tzinfo=NY_TZ)

        if ny_from < target_today and is_sec_dissemination_date(candidate_date):
            return target_today

        # Find next valid dissemination day
        curr = candidate_date + timedelta(days=1)
        while not is_sec_dissemination_date(curr):
            curr += timedelta(days=1)

        return datetime(curr.year, curr.month, curr.day, self.target_hour, self.target_minute, tzinfo=NY_TZ)


TRIAL_ID_DEFAULT = "ARX-TRIAL-2026-NVDA-AAPL-001"
TRIAL_START_DATE_DEFAULT = date(2026, 11, 17)
TRIAL_END_DATE_DEFAULT = date(2026, 11, 30)

TRIAL_DAILY_SLOTS: Tuple[Tuple[int, int], ...] = (
    (9, 15),
    (13, 0),
    (16, 30),
    (20, 30),
)

MAX_CYCLES_PER_DAY = 4
MAX_CYCLES_PER_TRIAL = 12
MAX_REQUESTS_PER_DAY = 32
MAX_REQUESTS_PER_TRIAL = 96


class TwoIssuerTrialScheduler:
    """
    Timezone-safe and quota-governed scheduler for the bounded AAPL/NVDA recurring trial.
    Enforces:
    - Stable trial identity and explicit start/end dates.
    - Exact AAPL/NVDA ticker/CIK pairs; missing CIK is rejected.
    - 4 distinct daily slots in America/New_York: 09:15, 13:00, 16:30, 20:30.
    - Unique durable slot identity: YYYY-MM-DD:SLOT_{idx}:{HH}{MM}.
    - Hard quotas: <= 4 cycles/day, <= 12 cycles/trial, <= 32 requests/day, <= 96 requests/trial.
    - SEC EDGAR dissemination calendar awareness (skips weekends and federal holidays).
    - Quotas and duplicate prevention survive restarts and short-lived authorization renewal.
    """
    def __init__(
        self,
        trial_id: str = TRIAL_ID_DEFAULT,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        slots: Tuple[Tuple[int, int], ...] = TRIAL_DAILY_SLOTS,
        window_minutes: int = 60,
        max_cycles_per_day: int = MAX_CYCLES_PER_DAY,
        max_cycles_per_trial: int = MAX_CYCLES_PER_TRIAL,
        max_requests_per_day: int = MAX_REQUESTS_PER_DAY,
        max_requests_per_trial: int = MAX_REQUESTS_PER_TRIAL,
    ):
        self.trial_id = trial_id or TRIAL_ID_DEFAULT
        env_sd = os.environ.get("ARX_TRIAL_START_DATE")
        self.start_date = start_date or (date.fromisoformat(env_sd) if env_sd else TRIAL_START_DATE_DEFAULT)
        env_ed = os.environ.get("ARX_TRIAL_END_DATE")
        self.end_date = end_date or (date.fromisoformat(env_ed) if env_ed else TRIAL_END_DATE_DEFAULT)
        self.slots = slots
        self.window_minutes = window_minutes
        self.max_cycles_per_day = max_cycles_per_day
        self.max_cycles_per_trial = max_cycles_per_trial
        self.max_requests_per_day = max_requests_per_day
        self.max_requests_per_trial = max_requests_per_trial

    def get_ny_now(self, now: Optional[datetime] = None) -> datetime:
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(NY_TZ)

    def validate_issuers(self, issuers: Optional[List[Dict[str, Any]]]) -> None:
        """Enforces exact NVDA and AAPL ticker and CIK pairs; missing or empty CIK fails closed."""
        if not issuers or len(issuers) != 2:
            raise RuntimeError("TRIAL_ISSUERS_INVALID: Two-issuer trial strictly requires exactly 2 issuers (NVDA, AAPL)")
        expected_ciks = {"NVDA": "0001045810", "AAPL": "0000320193"}
        found_ciks = {}
        for iss in issuers:
            ticker = iss.get("ticker")
            cik = iss.get("cik")
            if not ticker or ticker not in expected_ciks:
                raise RuntimeError(f"TRIAL_ISSUERS_INVALID: Unauthorized ticker '{ticker}' for trial")
            if not cik or not str(cik).strip():
                raise RuntimeError(f"TRIAL_ISSUERS_INVALID: Missing mandatory CIK for ticker '{ticker}'")
            norm_cik = str(cik).strip().zfill(10)
            if norm_cik != expected_ciks[ticker]:
                raise RuntimeError(f"TRIAL_ISSUERS_INVALID: CIK mismatch for ticker '{ticker}': expected {expected_ciks[ticker]}, got {norm_cik}")
            found_ciks[ticker] = norm_cik
        if set(found_ciks.keys()) != set(expected_ciks.keys()):
            raise RuntimeError("TRIAL_ISSUERS_INVALID: Both NVDA and AAPL must be present")

    def get_slot_for_time(self, ny_now: datetime) -> Tuple[Optional[int], Optional[str], str]:
        """
        Determines the current slot index (1..4) and slot_id for a given America/New_York datetime.
        Returns: (slot_index: Optional[int], slot_id: Optional[str], status_reason: str)
        """
        today = ny_now.date()
        for idx, (sh, sm) in enumerate(self.slots, start=1):
            slot_start = datetime(today.year, today.month, today.day, sh, sm, tzinfo=NY_TZ)
            slot_end = slot_start + timedelta(minutes=self.window_minutes)
            if slot_start <= ny_now <= slot_end:
                slot_id = f"{today.isoformat()}:SLOT_{idx}:{sh:02d}{sm:02d}"
                return idx, slot_id, f"WITHIN_EXECUTION_WINDOW:{slot_id}"

        # If not inside any slot window, categorize why
        first_start = datetime(today.year, today.month, today.day, self.slots[0][0], self.slots[0][1], tzinfo=NY_TZ)
        last_end = datetime(today.year, today.month, today.day, self.slots[-1][0], self.slots[-1][1], tzinfo=NY_TZ) + timedelta(minutes=self.window_minutes)
        if ny_now < first_start:
            diff = (first_start - ny_now).total_seconds() / 60
            return None, None, f"BEFORE_SCHEDULED_TIME: current time {ny_now.strftime('%H:%M %Z')} is {diff:.1f}m before first daily slot {first_start.strftime('%H:%M %Z')}"
        if ny_now > last_end:
            return None, None, f"PAST_DAILY_WINDOW: current time {ny_now.strftime('%H:%M %Z')} is past final daily slot window closing {last_end.strftime('%H:%M %Z')}"
        return None, None, f"BETWEEN_SLOTS: current time {ny_now.strftime('%H:%M %Z')} is between scheduled execution slots"

    def evaluate_execution_eligibility(
        self,
        now: Optional[datetime] = None,
        progress: Optional[Dict[str, Any]] = None,
        issuers: Optional[List[Dict[str, Any]]] = None,
        force: bool = False,
    ) -> Tuple[bool, Optional[str], str]:
        """
        Comprehensive eligibility evaluation against dissemination calendar, trial window, slot identity, and quotas.
        Returns: (should_run: bool, slot_id: Optional[str], reason: str)
        """
        if issuers is not None:
            self.validate_issuers(issuers)

        ny_now = self.get_ny_now(now)
        today = ny_now.date()

        if force:
            slot_idx, slot_id, _ = self.get_slot_for_time(ny_now)
            effective_slot_id = slot_id or f"{today.isoformat()}:FORCE_{int(time.time())}"
            return True, effective_slot_id, "FORCE_OVERRIDE"

        # 1. Trial Window Check
        if today < self.start_date:
            return False, None, f"BEFORE_TRIAL_START: current date {today.isoformat()} is before trial start {self.start_date.isoformat()}"
        if today > self.end_date:
            return False, None, f"TRIAL_WINDOW_EXPIRED: current date {today.isoformat()} is after trial end {self.end_date.isoformat()}"

        # 2. SEC Dissemination Day Check (Mon-Fri, non-holiday)
        if not is_sec_dissemination_date(today):
            if today.weekday() in (5, 6):
                return False, None, f"WEEKEND_EXCLUDED: {today.strftime('%A')} ({today.isoformat()}) is not an SEC dissemination day"
            return False, None, f"DISSEMINATION_HOLIDAY_EXCLUDED: {today.isoformat()} is an SEC holiday"

        # 3. Quotas Checks (survives restarts/auth renewals via persistent progress)
        if progress:
            cycles_today = progress.get("cycles_today", 0)
            if cycles_today >= self.max_cycles_per_day:
                return False, None, f"DAILY_CYCLE_QUOTA_REACHED: TRIAL_DAILY_CYCLE_LIMIT_REACHED: Daily cycle limit of {self.max_cycles_per_day} reached ({cycles_today}/{self.max_cycles_per_day})"

            cycles_trial = progress.get("cycles_trial", 0)
            if cycles_trial >= self.max_cycles_per_trial:
                return False, None, f"TRIAL_CYCLE_QUOTA_REACHED: TOTAL_CYCLE_LIMIT_REACHED: Total trial cycle limit of {self.max_cycles_per_trial} reached ({cycles_trial}/{self.max_cycles_per_trial})"

            requests_today = progress.get("requests_today", 0)
            if requests_today >= self.max_requests_per_day:
                return False, None, f"DAILY_REQUEST_QUOTA_REACHED: TRIAL_DAILY_REQUEST_BUDGET_EXCEEDED: Daily request budget of {self.max_requests_per_day} reached ({requests_today}/{self.max_requests_per_day})"

            requests_trial = progress.get("requests_trial", 0)
            if requests_trial >= self.max_requests_per_trial:
                return False, None, f"TRIAL_REQUEST_QUOTA_REACHED: TOTAL_REQUEST_BUDGET_EXCEEDED: Total trial request budget of {self.max_requests_per_trial} reached ({requests_trial}/{self.max_requests_per_trial})"

        # 4. Time slot window check
        slot_idx, slot_id, window_reason = self.get_slot_for_time(ny_now)
        if not slot_idx or not slot_id:
            return False, None, window_reason

        # 5. Duplicate Slot Execution Checks
        if progress:
            executed_slots = progress.get("executed_slots") or set()
            if slot_id in executed_slots:
                return False, slot_id, f"SLOT_ALREADY_EXECUTED: ALREADY_EXECUTED_TODAY: Slot {slot_id} has already been executed"

        return True, slot_id, window_reason


