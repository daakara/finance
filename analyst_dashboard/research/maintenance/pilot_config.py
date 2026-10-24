"""
analyst_dashboard/research/maintenance/pilot_config.py

Configuration and operational boundaries for the ARX Scheduled Research-Maintenance Pilot.
Governed by ARX Closure Model v2, ARX-DECISION-RESEARCH-PROCESS-V1, and PILOT_MAINTENANCE_POLICY.md.

Invariants:
- Strictly limited to Comfort Systems USA, Inc. (FIX), CIK 0001035983.
- Once-daily weekday schedule (UTC).
- Explicit activation flag (dormant by default).
- Hard-coded pilot limits: 30-day window, max 30 runs.
- Strictly bounded request budget (max 5 requests/run, min 0.25s interval).
- Financial reconciliation validated ONLY for Q3 2026 Form 10-Q (2026-09-30).
- Unsupported periods generate bounded event-review artifacts without valuation.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Set, Tuple


# Pilot Target Identity
TARGET_SYMBOL = "FIX"
TARGET_CIK = "0001035983"
TARGET_COMPANY_NAME = "Comfort Systems USA, Inc."

# Baseline Research Cutoff
ACCEPTED_BASELINE_ID = "OPP-FIX-20261016"
ACCEPTED_CUTOFF_DATE = "2026-10-16"
ACCEPTED_CUTOFF_TIMESTAMP_UTC = "2026-10-16T20:00:00Z"

# Already Recorded Known Accessions (pre-existing baseline events)
RECORDED_KNOWN_ACCESSIONS: Set[str] = {
    "0001104659-26-118159",  # Q3 2026 Form 10-Q (accepted worked example)
    "0001104659-26-118156",  # 8-K earnings release (correctly rejected non-periodic)
}

# Forms Classification
QUALIFYING_PERIODIC_FORMS: Set[str] = {"10-Q", "10-K"}
PERIODIC_AMENDMENT_FORMS: Set[str] = {"10-Q/A", "10-K/A"}
NON_PERIODIC_DISCLOSURE_FORMS: Set[str] = {
    "8-K", "4", "144", "SCHEDULE 13G", "SCHEDULE 13G/A",
    "SC 13G", "SC 13D", "DEF 14A"
}

# Supported Reconciliation Scope: ONLY Q3 2026 Form 10-Q is currently validated
SUPPORTED_RECONCILIATION_SCOPES: Set[Tuple[str, str, str]] = {
    ("FIX", "10-Q", "2026-09-30"),
}

# Schedule & Timing
SCHEDULE_CRON = "17 21 * * 1-5"  # Monday through Friday at 21:17 UTC
SCHEDULE_TIMEZONE = "UTC"

# Pilot Lifespan & Enforcement Limits
PILOT_START_DATE = "2026-10-24"
PILOT_END_DATE = "2026-11-24"
MAX_PILOT_RUNS = 30

# Operational Budgets per Run
MAX_REQUESTS_PER_RUN = 5
MAX_RUNTIME_SECONDS = 120
MIN_REQUEST_INTERVAL_SECONDS = 0.25

# Activation Flag Environment Variables
ACTIVATION_ENV_VAR = "ARX_MAINTENANCE_PILOT_ACTIVE"
REPO_VAR_NAME = "ENABLE_RESEARCH_MAINTENANCE_PILOT"


@dataclass(frozen=True)
class PilotStatus:
    """Read-only evaluation of pilot activation and boundary compliance."""
    is_active: bool
    is_expired: bool
    run_count: int
    max_runs: int
    current_date: str
    end_date: str
    message: str


def evaluate_pilot_status(
    current_run_count: int = 0,
    force_run: bool = False,
    override_date: str = None,
    override_active: bool = None,
) -> PilotStatus:
    """
    Evaluates whether the scheduled pilot is authorized to execute.
    Enforces activation flag and lifespan/run limits.
    """
    now_utc = datetime.now(timezone.utc)
    current_date_str = override_date or now_utc.strftime("%Y-%m-%d")

    # 1. Check expiration limits
    date_expired = current_date_str > PILOT_END_DATE
    runs_exhausted = current_run_count >= MAX_PILOT_RUNS

    if date_expired or runs_exhausted:
        reason = f"Date {current_date_str} > end date {PILOT_END_DATE}" if date_expired else f"Run count {current_run_count} >= limit {MAX_PILOT_RUNS}"
        return PilotStatus(
            is_active=False,
            is_expired=True,
            run_count=current_run_count,
            max_runs=MAX_PILOT_RUNS,
            current_date=current_date_str,
            end_date=PILOT_END_DATE,
            message=f"PILOT_EXPIRED: {reason}. Recurring execution is halted by code guard.",
        )

    # 2. Check activation flag
    if override_active is not None:
        active_flag = override_active
    else:
        env_val = os.environ.get(ACTIVATION_ENV_VAR, "").strip().lower()
        active_flag = env_val in ("1", "true", "yes", "enabled")

    if not active_flag and not force_run:
        return PilotStatus(
            is_active=False,
            is_expired=False,
            run_count=current_run_count,
            max_runs=MAX_PILOT_RUNS,
            current_date=current_date_str,
            end_date=PILOT_END_DATE,
            message="PILOT_DORMANT: Scheduled research maintenance pilot is disabled (ARX_MAINTENANCE_PILOT_ACTIVE=false).",
        )

    return PilotStatus(
        is_active=True,
        is_expired=False,
        run_count=current_run_count,
        max_runs=MAX_PILOT_RUNS,
        current_date=current_date_str,
        end_date=PILOT_END_DATE,
        message="PILOT_ACTIVE: Scheduled research maintenance pilot is authorized to run.",
    )


def is_reconciliation_supported(symbol: str, form: str, reporting_period: str) -> bool:
    """
    Checks whether the automated financial reconciler supports this specific filing.
    Fails closed: only returns True for explicitly tested, validated periods.
    """
    clean_sym = symbol.strip().upper()
    clean_form = form.strip().upper()
    clean_period = reporting_period.strip()
    return (clean_sym, clean_form, clean_period) in SUPPORTED_RECONCILIATION_SCOPES
