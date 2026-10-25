"""
ARX Terminal — Smart Money Snapshot & Ingestion Freshness Authority.

Authoritative evaluator for Smart Money continuous ingestion data freshness.
Differentiates:
1. Calendar Age: Physical elapsed hours since snapshot generation.
2. Expected Dissemination Cycles: SEC Form 4 dissemination occurs Mon-Fri up to 22:00 ET.
   Over weekends (Friday 22:30 ET to Monday 22:30 ET), 0 dissemination cycles occur.
   A snapshot published Friday remains FRESH through Monday 22:30 ET.
3. Freshness States:
   - FRESH: 0 expected cycles missed.
   - AGING: 1 expected cycle missed (emits informational UI badge).
   - STALE: >= 2 expected cycles missed (> 72h of market days; emits warning).
4. Worker Availability vs Data Freshness:
   Worker downtime does not invalidate an existing verified snapshot until staleness threshold.
5. Backup Freshness: Independent 12-hour containment gate enforcement.
"""

from __future__ import annotations

import os
import logging
from enum import Enum
from datetime import datetime, timezone, timedelta, date
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

try:
    import zoneinfo
    NY_TZ = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    # Fallback to fixed offset if zoneinfo not available
    NY_TZ = timezone(timedelta(hours=-5))

logger = logging.getLogger("smart_money.freshness_authority")

# Standard SEC EDGAR federal dissemination holidays (fixed dates or calculated)
SEC_HOLIDAYS_FIXED = {
    (1, 1): "New Year's Day",
    (6, 19): "Juneteenth",
    (7, 4): "Independence Day",
    (12, 25): "Christmas Day",
}


def is_sec_dissemination_date(d: date) -> bool:
    """Returns True if date d is an active SEC EDGAR filing dissemination business day."""
    # Weekend check
    if d.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Fixed holidays check (with weekend observation rule)
    month, day = d.month, d.day
    if (month, day) in SEC_HOLIDAYS_FIXED:
        return False
    # If holiday was on Sunday, Monday is observed
    if d.weekday() == 0:
        sun = d - timedelta(days=1)
        if (sun.month, sun.day) in SEC_HOLIDAYS_FIXED:
            return False
    # If holiday was on Saturday, Friday is observed
    if d.weekday() == 4:
        sat = d + timedelta(days=1)
        if (sat.month, sat.day) in SEC_HOLIDAYS_FIXED:
            return False

    # Variable holidays:
    # MLK Day: 3rd Monday of January
    if month == 1 and d.weekday() == 0 and 15 <= day <= 21:
        return False
    # Presidents' Day / Washington's Birthday: 3rd Monday of February
    if month == 2 and d.weekday() == 0 and 15 <= day <= 21:
        return False
    # Memorial Day: Last Monday of May
    if month == 5 and d.weekday() == 0 and day >= 25:
        return False
    # Labor Day: 1st Monday of September
    if month == 9 and d.weekday() == 0 and day <= 7:
        return False
    # Thanksgiving: 4th Thursday of November
    if month == 11 and d.weekday() == 3 and 22 <= day <= 28:
        return False

    return True


class SnapshotFreshnessState(str, Enum):
    """Authoritative lifecycle freshness classification."""
    FRESH = "FRESH"
    AGING = "AGING"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SmartMoneyFreshnessEvaluation:
    """Comprehensive evaluation receipt for Smart Money freshness."""
    snapshot_id: Optional[str]
    published_at_utc: Optional[str]
    evaluated_at_utc: str
    calendar_age_hours: float
    expected_cycles_missed: int
    freshness_state: SnapshotFreshnessState
    backup_freshness_state: str
    worker_health_state: str
    emergency_read_enabled: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["freshness_state"] = self.freshness_state.value
        return d


class SmartMoneyFreshnessAuthority:
    """
    Evaluates Smart Money data freshness adhering to the authoritative calendar-aware contract.
    """

    def __init__(
        self,
        stale_cycle_threshold: int = 2,
        aging_cycle_threshold: int = 1,
    ):
        self.stale_cycle_threshold = stale_cycle_threshold
        self.aging_cycle_threshold = aging_cycle_threshold

    def calculate_expected_cycles(
        self,
        from_utc: datetime,
        to_utc: datetime,
    ) -> int:
        """
        Calculates how many SEC Form 4 daily dissemination cycles (at 22:30 America/New_York)
        were expected to occur between from_utc and to_utc.
        """
        if from_utc >= to_utc:
            return 0

        # Convert to Eastern Time
        from_et = from_utc.astimezone(NY_TZ)
        to_et = to_utc.astimezone(NY_TZ)

        cycles = 0
        current_date = from_et.date()
        end_date = to_et.date()

        while current_date <= end_date:
            if is_sec_dissemination_date(current_date):
                # Cutoff is 22:30 ET
                cycle_dt = datetime(
                    current_date.year, current_date.month, current_date.day,
                    22, 30, 0, tzinfo=NY_TZ
                )
                if from_et < cycle_dt <= to_et:
                    cycles += 1
            current_date += timedelta(days=1)

        return cycles

    def evaluate_snapshot_freshness(
        self,
        snapshot: Optional[Dict[str, Any]],
        now_utc: Optional[datetime] = None,
        backup_freshness_state: str = "FRESH",
        worker_health_state: str = "HEALTHY",
    ) -> SmartMoneyFreshnessEvaluation:
        """
        Evaluates snapshot freshness differentiating calendar age from expected cycles missed.
        """
        eval_now = now_utc or datetime.now(timezone.utc)
        if eval_now.tzinfo is None:
            eval_now = eval_now.replace(tzinfo=timezone.utc)

        read_enabled = os.environ.get("ARX_SMART_MONEY_POSTGRES_READ_ENABLED", "true").strip().lower() not in ("false", "0", "no")

        if not snapshot:
            return SmartMoneyFreshnessEvaluation(
                snapshot_id=None,
                published_at_utc=None,
                evaluated_at_utc=eval_now.isoformat(),
                calendar_age_hours=0.0,
                expected_cycles_missed=0,
                freshness_state=SnapshotFreshnessState.UNKNOWN,
                backup_freshness_state=backup_freshness_state,
                worker_health_state=worker_health_state,
                emergency_read_enabled=read_enabled,
                rationale="No active snapshot published; pipeline pending.",
            )

        snap_id = snapshot.get("snapshot_id")
        pub_at_str = snapshot.get("generated_at") or snapshot.get("published_at") or snapshot.get("data_as_of")

        if not pub_at_str:
            return SmartMoneyFreshnessEvaluation(
                snapshot_id=snap_id,
                published_at_utc=None,
                evaluated_at_utc=eval_now.isoformat(),
                calendar_age_hours=0.0,
                expected_cycles_missed=0,
                freshness_state=SnapshotFreshnessState.UNKNOWN,
                backup_freshness_state=backup_freshness_state,
                worker_health_state=worker_health_state,
                emergency_read_enabled=read_enabled,
                rationale="Snapshot missing publication timestamp.",
            )

        try:
            clean_ts = pub_at_str.replace("Z", "+00:00")
            pub_dt = datetime.fromisoformat(clean_ts)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        except Exception as e:
            logger.warning(f"Failed to parse snapshot publication timestamp '{pub_at_str}': {e}")
            return SmartMoneyFreshnessEvaluation(
                snapshot_id=snap_id,
                published_at_utc=pub_at_str,
                evaluated_at_utc=eval_now.isoformat(),
                calendar_age_hours=0.0,
                expected_cycles_missed=0,
                freshness_state=SnapshotFreshnessState.UNKNOWN,
                backup_freshness_state=backup_freshness_state,
                worker_health_state=worker_health_state,
                emergency_read_enabled=read_enabled,
                rationale=f"Invalid timestamp format: {e}",
            )

        calendar_age_sec = max(0.0, (eval_now - pub_dt).total_seconds())
        calendar_age_hours = round(calendar_age_sec / 3600.0, 2)

        # Expected cycles missed
        cycles_missed = self.calculate_expected_cycles(pub_dt, eval_now)

        if cycles_missed == 0:
            state = SnapshotFreshnessState.FRESH
            rationale = (
                f"Data is FRESH. 0 expected dissemination cycles missed. "
                f"Calendar age is {calendar_age_hours}h (weekend/holiday adjusted)."
            )
        elif cycles_missed < self.stale_cycle_threshold:
            state = SnapshotFreshnessState.AGING
            rationale = (
                f"Data is AGING. {cycles_missed} expected dissemination cycle missed. "
                f"Calendar age is {calendar_age_hours}h."
            )
        else:
            state = SnapshotFreshnessState.STALE
            rationale = (
                f"Data is STALE. {cycles_missed} expected dissemination cycles missed "
                f"(>= threshold {self.stale_cycle_threshold}). Calendar age is {calendar_age_hours}h."
            )

        return SmartMoneyFreshnessEvaluation(
            snapshot_id=snap_id,
            published_at_utc=pub_dt.isoformat(),
            evaluated_at_utc=eval_now.isoformat(),
            calendar_age_hours=calendar_age_hours,
            expected_cycles_missed=cycles_missed,
            freshness_state=state,
            backup_freshness_state=backup_freshness_state,
            worker_health_state=worker_health_state,
            emergency_read_enabled=read_enabled,
            rationale=rationale,
        )
