"""
HVA-01 Wave 4C-3 Clock Drift Daemon.
Provides authoritative continuous monitoring of wall-clock progression against monotonic reference time,
clock rollback detection, drift threshold enforcement, time-source health assertions,
and fail-closed admissibility enforcement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, List, Optional


class ClockHealthStatus(str, Enum):
    """Health classification for system and reference clocks."""
    HEALTHY = "HEALTHY"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    ROLLBACK_DETECTED = "ROLLBACK_DETECTED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED_CLOSED = "FAILED_CLOSED"


class ClockDaemonError(Exception):
    """Base exception for all ClockDriftDaemon failures."""
    pass


class ClockRollbackError(ClockDaemonError):
    """Raised when negative time travel or clock rollback is detected."""
    pass


class ClockDriftExceededError(ClockDaemonError):
    """Raised when divergence between wall clock and monotonic clock exceeds threshold."""
    pass


class TimeSourceUnavailableError(ClockDaemonError):
    """Raised when the underlying time source fails or returns invalid readings."""
    pass


class ClockAnomalyError(ClockDaemonError):
    """Raised when conflicting or non-physical clock readings occur."""
    pass


@dataclass(frozen=True)
class ClockSample:
    """Immutable audit record of a single synchronized clock observation."""
    sample_index: int
    timestamp_utc: str
    wall_time: float
    monotonic_time: float
    wall_delta: float
    monotonic_delta: float
    drift_delta: float
    cumulative_drift: float
    status: ClockHealthStatus
    details: Optional[str] = None


class ClockDriftDaemon:
    """
    Monitors wall-clock progression against a monotonic clock baseline.
    Enforces strict drift thresholds, monotonic forward progression,
    instantaneous rollback detection, and fail-closed admissibility checks.
    """

    def __init__(
        self,
        max_drift_seconds: float = 5.0,
        max_rollback_tolerance_seconds: float = 0.0,
        time_provider: Optional[Callable[[], float]] = None,
        monotonic_provider: Optional[Callable[[], float]] = None,
    ):
        """
        Initialize the Clock Drift Daemon.

        Args:
            max_drift_seconds: Maximum tolerable divergence between wall and monotonic time (seconds).
            max_rollback_tolerance_seconds: Maximum tolerable wall-clock backward jump (seconds, default 0.0).
            time_provider: Custom callable returning wall time epoch seconds (for synthetic testing).
            monotonic_provider: Custom callable returning monotonic seconds (for synthetic testing).
        """
        if max_drift_seconds <= 0.0:
            raise ValueError(f"max_drift_seconds must be positive, got {max_drift_seconds}")

        self.max_drift_seconds = float(max_drift_seconds)
        self.max_rollback_tolerance_seconds = float(max_rollback_tolerance_seconds)

        self._time_provider = time_provider or time.time
        self._monotonic_provider = monotonic_provider or time.monotonic

        self._status: ClockHealthStatus = ClockHealthStatus.HEALTHY
        self._history: List[ClockSample] = []
        self._sample_counter: int = 0

        # Anchor points
        self._anchor_wall: Optional[float] = None
        self._anchor_mono: Optional[float] = None
        self._last_wall: Optional[float] = None
        self._last_mono: Optional[float] = None

        # Auto-initialize baseline anchor
        self._initialize_baseline()

    def _initialize_baseline(self) -> None:
        """Capture the initial reference anchor for wall and monotonic clocks."""
        try:
            wall = float(self._time_provider())
            mono = float(self._monotonic_provider())
        except Exception as e:
            self._status = ClockHealthStatus.UNAVAILABLE
            raise TimeSourceUnavailableError(f"Initial time source sampling failed: {e}") from e

        self._anchor_wall = wall
        self._anchor_mono = mono
        self._last_wall = wall
        self._last_mono = mono
        self._sample_counter = 0

        sample = ClockSample(
            sample_index=0,
            timestamp_utc=datetime.fromtimestamp(wall, tz=timezone.utc).isoformat(),
            wall_time=wall,
            monotonic_time=mono,
            wall_delta=0.0,
            monotonic_delta=0.0,
            drift_delta=0.0,
            cumulative_drift=0.0,
            status=ClockHealthStatus.HEALTHY,
            details="Initial reference baseline anchored",
        )
        self._history.append(sample)

    @property
    def status(self) -> ClockHealthStatus:
        """Current health status of the clock subsystem."""
        return self._status

    @property
    def is_healthy(self) -> bool:
        """True if the clock is healthy and within all drift/monotonic invariants."""
        return self._status == ClockHealthStatus.HEALTHY

    def sample(self) -> ClockSample:
        """
        Sample the current wall and monotonic clocks, evaluate invariants, and update audit log.

        Returns:
            ClockSample with measurement metrics and status.

        Raises:
            TimeSourceUnavailableError: If time provider fails or raises an error.
            ClockRollbackError: If negative time travel is detected.
            ClockDriftExceededError: If drift relative to monotonic progression exceeds limit.
            ClockAnomalyError: If monotonic clock travels backwards.
        """
        # If already failed closed, refuse further sampling without explicit recalibration
        if self._status in (ClockHealthStatus.FAILED_CLOSED, ClockHealthStatus.ROLLBACK_DETECTED):
            raise ClockAnomalyError(f"Clock daemon is in fail-closed state: {self._status}")

        try:
            current_wall = float(self._time_provider())
            current_mono = float(self._monotonic_provider())
        except Exception as e:
            self._status = ClockHealthStatus.UNAVAILABLE
            raise TimeSourceUnavailableError(f"Time source reading failed: {e}") from e

        self._sample_counter += 1
        iso_str = datetime.fromtimestamp(current_wall, tz=timezone.utc).isoformat()

        assert self._last_wall is not None
        assert self._last_mono is not None
        assert self._anchor_wall is not None
        assert self._anchor_mono is not None

        wall_delta = current_wall - self._last_wall
        mono_delta = current_mono - self._last_mono

        # 1. Monotonic integrity assertion
        if mono_delta < 0.0:
            self._status = ClockHealthStatus.ROLLBACK_DETECTED
            sample = ClockSample(
                sample_index=self._sample_counter,
                timestamp_utc=iso_str,
                wall_time=current_wall,
                monotonic_time=current_mono,
                wall_delta=wall_delta,
                monotonic_delta=mono_delta,
                drift_delta=wall_delta - mono_delta,
                cumulative_drift=(current_wall - self._anchor_wall) - (current_mono - self._anchor_mono),
                status=ClockHealthStatus.ROLLBACK_DETECTED,
                details=f"Monotonic clock reversal detected: mono_delta={mono_delta:.6f}s < 0",
            )
            self._history.append(sample)
            raise ClockRollbackError(sample.details)

        # 2. Wall-clock rollback assertion
        if wall_delta < -self.max_rollback_tolerance_seconds:
            self._status = ClockHealthStatus.ROLLBACK_DETECTED
            sample = ClockSample(
                sample_index=self._sample_counter,
                timestamp_utc=iso_str,
                wall_time=current_wall,
                monotonic_time=current_mono,
                wall_delta=wall_delta,
                monotonic_delta=mono_delta,
                drift_delta=wall_delta - mono_delta,
                cumulative_drift=(current_wall - self._anchor_wall) - (current_mono - self._anchor_mono),
                status=ClockHealthStatus.ROLLBACK_DETECTED,
                details=f"Wall-clock rollback detected: wall_delta={wall_delta:.6f}s < -{self.max_rollback_tolerance_seconds}s",
            )
            self._history.append(sample)
            raise ClockRollbackError(sample.details)

        # 3. Drift computation
        step_drift = wall_delta - mono_delta
        cumulative_drift = (current_wall - self._anchor_wall) - (current_mono - self._anchor_mono)

        # Update last positions
        self._last_wall = current_wall
        self._last_mono = current_mono

        # 4. Drift threshold check
        if abs(cumulative_drift) > self.max_drift_seconds:
            self._status = ClockHealthStatus.DRIFT_DETECTED
            sample = ClockSample(
                sample_index=self._sample_counter,
                timestamp_utc=iso_str,
                wall_time=current_wall,
                monotonic_time=current_mono,
                wall_delta=wall_delta,
                monotonic_delta=mono_delta,
                drift_delta=step_drift,
                cumulative_drift=cumulative_drift,
                status=ClockHealthStatus.DRIFT_DETECTED,
                details=f"Cumulative clock drift ({cumulative_drift:+.4f}s) exceeded limit (+/-{self.max_drift_seconds:.4f}s)",
            )
            self._history.append(sample)
            raise ClockDriftExceededError(sample.details)

        # Healthy sample
        sample = ClockSample(
            sample_index=self._sample_counter,
            timestamp_utc=iso_str,
            wall_time=current_wall,
            monotonic_time=current_mono,
            wall_delta=wall_delta,
            monotonic_delta=mono_delta,
            drift_delta=step_drift,
            cumulative_drift=cumulative_drift,
            status=ClockHealthStatus.HEALTHY,
            details="Clock progress within bounds",
        )
        self._history.append(sample)
        return sample

    def assert_admissible(self, candidate_timestamp_utc: Optional[str] = None) -> bool:
        """
        Fail-closed assertion of clock admissibility.
        
        Args:
            candidate_timestamp_utc: Optional ISO 8601 string to check against current wall time.

        Returns:
            True if clock is admissible.

        Raises:
            ClockDaemonError: If clock is in any non-HEALTHY state or candidate timestamp exhibits excessive drift.
        """
        if self._status == ClockHealthStatus.ROLLBACK_DETECTED:
            raise ClockRollbackError("Evidence rejected: clock rollback previously detected in daemon")
        if self._status == ClockHealthStatus.DRIFT_DETECTED:
            raise ClockDriftExceededError("Evidence rejected: cumulative clock drift exceeded permissible limit")
        if self._status == ClockHealthStatus.UNAVAILABLE:
            raise TimeSourceUnavailableError("Evidence rejected: time source is currently unavailable")
        if self._status != ClockHealthStatus.HEALTHY:
            raise ClockAnomalyError(f"Evidence rejected: clock daemon in non-healthy state ({self._status})")

        if candidate_timestamp_utc:
            try:
                cand_dt = datetime.fromisoformat(candidate_timestamp_utc.replace("Z", "+00:00"))
                now_dt = datetime.fromtimestamp(float(self._time_provider()), tz=timezone.utc)
                diff = abs((cand_dt - now_dt).total_seconds())
                if diff > self.max_drift_seconds:
                    raise ClockDriftExceededError(
                        f"Candidate timestamp drift {diff:.2f}s exceeds daemon threshold {self.max_drift_seconds:.2f}s"
                    )
            except ClockDriftExceededError:
                raise
            except Exception as e:
                raise ClockAnomalyError(f"Malformed candidate timestamp '{candidate_timestamp_utc}': {e}") from e

        return True

    def get_audit_history(self) -> List[ClockSample]:
        """Return an immutable copy of all clock observation samples."""
        return list(self._history)

    def reset(
        self,
        time_provider: Optional[Callable[[], float]] = None,
        monotonic_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        """Recalibrate and re-anchor daemon baseline."""
        if time_provider:
            self._time_provider = time_provider
        if monotonic_provider:
            self._monotonic_provider = monotonic_provider
        self._status = ClockHealthStatus.HEALTHY
        self._history = []
        self._initialize_baseline()
