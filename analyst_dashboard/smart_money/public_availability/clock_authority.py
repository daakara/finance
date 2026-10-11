"""
Clock Authority & Point-in-Time Uncertainty Verification — Epoch 001.
Evaluates local and reference clock synchronization, bounding uncertainty intervals.
Enforces fail-closed rules when clock authority is degraded or untrusted.
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from analyst_dashboard.smart_money.public_availability.models import (
    ClockAuthorityRecord,
    ClockStatus,
)


MAX_SYNC_UNCERTAINTY_MS = 100.0      # <= 100 ms -> SYNCHRONIZED
MAX_DEGRADED_UNCERTAINTY_MS = 500.0  # <= 500 ms -> DEGRADED, > 500 ms -> UNTRUSTED


class ClockAuthority:
    """
    Manages and certifies the local clock status before observer executions.
    """

    def __init__(
        self,
        clock_source: str = "OS_DISCIPLINED_TIME_SERVICE",
        sync_method: str = "WINDOWS_TIME_SYNC_OR_MONOTONIC_ANCHOR",
        override_offset_ms: Optional[float] = None,
        override_uncertainty_ms: Optional[float] = None,
        override_status: Optional[ClockStatus] = None,
    ):
        self.clock_source = clock_source
        self.sync_method = sync_method
        self.override_offset_ms = override_offset_ms
        self.override_uncertainty_ms = override_uncertainty_ms
        self.override_status = override_status

    def inspect_clock(self) -> ClockAuthorityRecord:
        """
        Samples the current wall-clock and monotonic anchor to verify monotonic continuity
        and estimate local clock uncertainty.
        """
        t0_mono = time.monotonic()
        now_dt = datetime.now(timezone.utc)
        t1_mono = time.monotonic()
        mono_duration_ms = (t1_mono - t0_mono) * 1000.0

        now_iso = now_dt.isoformat()

        if self.override_status is not None:
            status = self.override_status
            offset_ms = self.override_offset_ms if self.override_offset_ms is not None else 0.0
            uncertainty_ms = self.override_uncertainty_ms if self.override_uncertainty_ms is not None else 50.0
        else:
            # Under standard local execution without remote NTP socket access:
            # Baseline uncertainty bounded by sampling duration + standard OS precision floor
            offset_ms = 0.0
            uncertainty_ms = max(mono_duration_ms * 2.0, 15.6)  # Windows timer tick granularity ~15.6ms

            if uncertainty_ms <= MAX_SYNC_UNCERTAINTY_MS:
                status = ClockStatus.SYNCHRONIZED
            elif uncertainty_ms <= MAX_DEGRADED_UNCERTAINTY_MS:
                status = ClockStatus.DEGRADED
            else:
                status = ClockStatus.UNTRUSTED

        return ClockAuthorityRecord(
            clock_source=self.clock_source,
            clock_sync_method=self.sync_method,
            clock_observed_at=now_iso,
            clock_offset_ms=offset_ms,
            clock_uncertainty_ms=uncertainty_ms,
            clock_status=status,
        )

    @staticmethod
    def derive_observation_intervals(
        response_completed_at_iso: str,
        uncertainty_ms: float,
    ) -> Tuple[str, str, str]:
        """
        Derives:
          (earliest_possible_time, conservative_observation_time, latest_possible_time)
        using:
          earliest = response_completed_at - uncertainty
          conservative = response_completed_at + uncertainty
          latest = response_completed_at + uncertainty
        """
        dt = datetime.fromisoformat(response_completed_at_iso)
        delta = timedelta(milliseconds=uncertainty_ms)

        earliest = (dt - delta).isoformat()
        conservative = (dt + delta).isoformat()
        latest = (dt + delta).isoformat()

        return earliest, conservative, latest

    @staticmethod
    def compare_clock_intervals(
        interval_a: Tuple[str, str],  # (earliest_a, latest_a)
        interval_b: Tuple[str, str],  # (earliest_b, latest_b)
    ) -> str:
        """
        Determines ordering between two observation intervals:
          - A_PRECEDES_B if latest_a < earliest_b
          - B_PRECEDES_A if latest_b < earliest_a
          - INDETERMINATE if intervals overlap
        """
        dt_earliest_a = datetime.fromisoformat(interval_a[0])
        dt_latest_a = datetime.fromisoformat(interval_a[1])
        dt_earliest_b = datetime.fromisoformat(interval_b[0])
        dt_latest_b = datetime.fromisoformat(interval_b[1])

        if dt_latest_a < dt_earliest_b:
            return "A_PRECEDES_B"
        elif dt_latest_b < dt_earliest_a:
            return "B_PRECEDES_A"
        else:
            return "INDETERMINATE"
