"""
analyst_dashboard/research/maintenance/pilot_config.py

Configuration and operational boundaries for the ARX Scheduled Research-Maintenance Pilot.
Governed by ARX Closure Model v2, ARX-DECISION-RESEARCH-PROCESS-V1, and PILOT_MAINTENANCE_POLICY.md.

Invariants:
- Strictly limited to Comfort Systems USA, Inc. (FIX), CIK 0001035983.
- Once-daily weekday schedule (UTC): Monday through Friday at 21:17 UTC.
- Explicit activation flag (dormant by default: ENABLE_RESEARCH_MAINTENANCE_PILOT=false).
- Hard-coded pilot limits: 30-day window (2026-10-24 to 2026-11-24), max 30 runs.
- Strictly bounded request budget (max 5 requests/run, min 0.25s interval, 120s runtime limit).
- Shared run-level budget enforcement across all SEC network requests.
- Financial reconciliation validated ONLY for Q3 2026 Form 10-Q (2026-09-30).
- Unsupported periods generate bounded event-review artifacts without valuation.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Set, Tuple, Optional, Callable, Union


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

# Schedule & Timing: Mon-Fri at 21:17 UTC
SCHEDULE_CRON = "17 21 * * 1-5"
SCHEDULE_TIMEZONE = "UTC"

# Pilot Lifespan & Enforcement Limits
PILOT_START_DATE = "2026-10-24"
PILOT_END_DATE = "2026-11-24"
MAX_PILOT_RUNS = 30

# Operational Budgets per Run
MAX_REQUESTS_PER_RUN = 5
MAX_RUNTIME_SECONDS = 120.0
MIN_REQUEST_INTERVAL_SECONDS = 0.25

# Isolated Namespace Constants: Operational vs. Rehearsal
OPERATIONAL_STATE_DIR = "durable_state/research_maintenance"
OPERATIONAL_OUTPUT_DIR = "docs/research/maintenance/packets"
OPERATIONAL_CACHE_DIR = "data/research/sourced_fundamentals/raw"
OPERATIONAL_ARTIFACT_NAME = "research-maintenance-fix-state"
OPERATIONAL_CACHE_KEY_PREFIX = "research-maintenance-durable-state-"

REHEARSAL_STATE_DIR = "durable_state/rehearsal_maintenance"
REHEARSAL_OUTPUT_DIR = "docs/research/maintenance/rehearsal_packets"
REHEARSAL_CACHE_DIR = "data/research/sourced_fundamentals/rehearsal_raw"
REHEARSAL_ARTIFACT_NAME = "research-maintenance-fix-rehearsal-state"
REHEARSAL_CACHE_KEY_PREFIX = "research-maintenance-rehearsal-state-"

# Activation Flag Environment Variables
ACTIVATION_ENV_VAR = "ARX_MAINTENANCE_PILOT_ACTIVE"
REPO_VAR_NAME = "ENABLE_RESEARCH_MAINTENANCE_PILOT"
OPERATIONAL_BOOTSTRAP_ENV_VAR = "ARX_OPERATIONAL_BOOTSTRAP"

# GitHub Actions Workflow & Repository Boundaries
TARGET_REPOSITORY = "daakara/finance"
DESIGNATED_WORKFLOW_FILE = "scheduled_research_maintenance.yml"
DESIGNATED_WORKFLOW_PATH = ".github/workflows/scheduled_research_maintenance.yml"


def validate_rehearsal_paths(
    state_dir: Union[str, Path],
    output_dir: Union[str, Path],
    cache_dir: Union[str, Path],
    repo_root: Optional[Path] = None,
) -> None:
    """
    Validates that rehearsal paths do not collide or overlap with protected operational roots.
    Rejects exact equality and parent/child overlap in either direction.
    Resolves existing symlinks and junction aliases using filesystem behavior.
    Uses path-aware comparisons (is_relative_to), not string prefixes.
    Preserves safe custom temporary directories.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
    else:
        repo_root = Path(repo_root).resolve()

    op_roots = [
        ("operational state dir", (repo_root / OPERATIONAL_STATE_DIR).resolve()),
        ("operational output dir", (repo_root / OPERATIONAL_OUTPUT_DIR).resolve()),
        ("operational cache dir", (repo_root / OPERATIONAL_CACHE_DIR).resolve()),
    ]

    rehearsal_paths = [
        ("state_dir", Path(state_dir).resolve()),
        ("output_dir", Path(output_dir).resolve()),
        ("cache_dir", Path(cache_dir).resolve()),
    ]

    for r_name, r_resolved in rehearsal_paths:
        for op_name, op_root in op_roots:
            # 1. Exact equality
            if r_resolved == op_root:
                raise ValueError(
                    f"BOUNDARY_VIOLATION: Rehearsal {r_name} '{r_resolved}' matches protected {op_name} '{op_root}'."
                )
            # 2. Child overlap (r_resolved is inside op_root)
            if r_resolved.is_relative_to(op_root):
                raise ValueError(
                    f"BOUNDARY_VIOLATION: Rehearsal {r_name} '{r_resolved}' is inside protected {op_name} '{op_root}'."
                )
            # 3. Parent overlap (op_root is inside r_resolved)
            if op_root.is_relative_to(r_resolved):
                raise ValueError(
                    f"BOUNDARY_VIOLATION: Rehearsal {r_name} '{r_resolved}' contains protected {op_name} '{op_root}'."
                )


class BudgetExhaustedError(RuntimeError):
    """Raised when request budget or monotonic runtime deadline is exhausted."""
    pass


class RunBudgetTracker:
    """
    Shared run-level budget tracker enforcing:
    - Overall monotonic runtime deadline (default 120.0s).
    - Per-run HTTP request count ceiling for primary-source SEC retrieval (max 5 requests).
    - Explicit separation of SEC requests from recovery HTTP requests.
    - Rate limit interval between network requests (default 0.25s).
    - Request-level timeouts dynamically bounded by remaining runtime (never inflated).
    """

    def __init__(
        self,
        max_requests: int = MAX_REQUESTS_PER_RUN,
        max_runtime_sec: Optional[float] = None,
        min_interval_sec: Optional[float] = None,
        clock_fn: Optional[Callable[[], float]] = None,
        max_runtime_seconds: Optional[float] = None,
        min_request_interval_seconds: Optional[float] = None,
    ):
        self.max_sec_requests = max_requests
        self.max_runtime_sec = (
            max_runtime_seconds
            if max_runtime_seconds is not None
            else (max_runtime_sec if max_runtime_sec is not None else MAX_RUNTIME_SECONDS)
        )
        self.min_interval_sec = (
            min_request_interval_seconds
            if min_request_interval_seconds is not None
            else (min_interval_sec if min_interval_sec is not None else MIN_REQUEST_INTERVAL_SECONDS)
        )
        self._clock = clock_fn or time.monotonic
        self.start_time = self._clock()
        self.deadline = self.start_time + self.max_runtime_sec
        self.last_sec_request_time = 0.0
        self.sec_request_count = 0
        self.recovery_request_count = 0

    @property
    def max_requests(self) -> int:
        return self.max_sec_requests

    @property
    def request_count(self) -> int:
        """Backward-compatible access to SEC request count."""
        return self.sec_request_count

    @property
    def total_requests(self) -> int:
        """Combined SEC and recovery HTTP requests."""
        return self.sec_request_count + self.recovery_request_count

    def elapsed_time(self) -> float:
        """Returns elapsed time in seconds since tracker creation."""
        return self._clock() - self.start_time

    def remaining_runtime(self) -> float:
        """Returns seconds remaining before monotonic deadline."""
        return max(0.0, self.deadline - self._clock())

    def check_deadline(self, boundary_name: str = "") -> None:
        """
        Enforces monotonic deadline at consequential processing boundaries.
        Raises BudgetExhaustedError if remaining runtime is exhausted or insufficient.
        """
        rem = self.remaining_runtime()
        if rem <= 0.05:
            prefix = f" [{boundary_name}]" if boundary_name else ""
            raise BudgetExhaustedError(
                f"RUNTIME_BUDGET_EXHAUSTED{prefix}: Monotonic deadline ({self.max_runtime_sec}s) expired "
                f"(remaining: {rem:.2f}s)."
            )

    def check_can_request(self) -> Tuple[bool, Optional[str]]:
        """Checks if another SEC HTTP request is permitted under both budgets."""
        if self.sec_request_count >= self.max_sec_requests:
            return False, f"REQUEST_BUDGET_EXHAUSTED: Run reached maximum {self.max_sec_requests} SEC HTTP requests."
        rem = self.remaining_runtime()
        if rem < 0.1:
            return False, f"RUNTIME_BUDGET_EXHAUSTED: Insufficient runtime ({rem:.2f}s) remaining before monotonic deadline."
        return True, None

    def acquire_request_slot(self, default_timeout_sec: float = 15.0) -> float:
        """Backward-compatible alias for acquire_sec_request_slot."""
        return self.acquire_sec_request_slot(default_timeout_sec=default_timeout_sec)

    def acquire_sec_request_slot(self, default_timeout_sec: float = 15.0) -> float:
        """
        Enforces SEC rate limiting, consumes 1 SEC request budget slot, and returns bounded timeout.
        Dynamically bounds timeout by remaining monotonic runtime without inflation.
        Raises BudgetExhaustedError if budget or runtime has expired.
        """
        can_req, reason = self.check_can_request()
        if not can_req:
            raise BudgetExhaustedError(reason)

        elapsed = self._clock() - self.last_sec_request_time
        if elapsed < self.min_interval_sec:
            sleep_needed = self.min_interval_sec - elapsed
            rem = self.remaining_runtime()
            if sleep_needed >= rem:
                raise BudgetExhaustedError(
                    f"RUNTIME_BUDGET_EXHAUSTED: Monotonic deadline reached while rate limiting."
                )
            time.sleep(sleep_needed)

        rem_before = self.remaining_runtime()
        if rem_before < 0.1:
            raise BudgetExhaustedError(
                f"RUNTIME_BUDGET_EXHAUSTED: Insufficient runtime ({rem_before:.2f}s) to initiate SEC request."
            )

        self.last_sec_request_time = self._clock()
        self.sec_request_count += 1

        rem_after = self.remaining_runtime()
        if rem_after <= 0.0:
            raise BudgetExhaustedError(
                f"RUNTIME_BUDGET_EXHAUSTED: Monotonic deadline reached during rate limit sleep."
            )

        # Dynamic timeout: strictly bounded by remaining runtime (never inflated above rem_after)
        return min(default_timeout_sec, rem_after)

    def acquire_recovery_request_slot(self, default_timeout_sec: float = 10.0) -> float:
        """
        Consumes 1 recovery HTTP request slot (tracked separately from SEC retrieval budget)
        and bounds timeout by remaining monotonic deadline.
        """
        rem = self.remaining_runtime()
        if rem < 0.1:
            raise BudgetExhaustedError(
                f"RUNTIME_BUDGET_EXHAUSTED: Insufficient runtime ({rem:.2f}s) to initiate state recovery request."
            )
        self.recovery_request_count += 1
        return min(default_timeout_sec, rem)


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
    operational_bootstrap: bool = False,
) -> PilotStatus:
    """
    Evaluates whether the scheduled pilot is authorized to execute.
    Enforces activation flag and lifespan/run limits.
    """
    now_utc = datetime.now(timezone.utc)
    current_date_str = override_date or now_utc.strftime("%Y-%m-%d")

    # 1. Check expiration limits (enforced unconditionally even for bootstrap)
    date_expired = current_date_str > PILOT_END_DATE
    runs_exhausted = current_run_count >= MAX_PILOT_RUNS

    if date_expired or runs_exhausted:
        reason = (
            f"Date {current_date_str} > end date {PILOT_END_DATE}"
            if date_expired
            else f"Run count {current_run_count} >= limit {MAX_PILOT_RUNS}"
        )
        return PilotStatus(
            is_active=False,
            is_expired=True,
            run_count=current_run_count,
            max_runs=MAX_PILOT_RUNS,
            current_date=current_date_str,
            end_date=PILOT_END_DATE,
            message=f"PILOT_EXPIRED: {reason}. Recurring execution is halted by code guard.",
        )

    # 2. Check activation flag and operational bootstrap
    is_bootstrap = (
        operational_bootstrap
        or os.environ.get(OPERATIONAL_BOOTSTRAP_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "enabled")
    )

    if override_active is not None:
        active_flag = override_active
    else:
        env_val = os.environ.get(ACTIVATION_ENV_VAR, "").strip().lower()
        active_flag = env_val in ("1", "true", "yes", "enabled")

    if not active_flag and not force_run and not is_bootstrap:
        return PilotStatus(
            is_active=False,
            is_expired=False,
            run_count=current_run_count,
            max_runs=MAX_PILOT_RUNS,
            current_date=current_date_str,
            end_date=PILOT_END_DATE,
            message="PILOT_DORMANT: Scheduled research maintenance pilot is disabled (ARX_MAINTENANCE_PILOT_ACTIVE=false).",
        )

    if is_bootstrap:
        return PilotStatus(
            is_active=True,
            is_expired=False,
            run_count=current_run_count,
            max_runs=MAX_PILOT_RUNS,
            current_date=current_date_str,
            end_date=PILOT_END_DATE,
            message="PILOT_ACTIVE: Authorized manual operational bootstrap invocation.",
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
