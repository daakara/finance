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

