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
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

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
