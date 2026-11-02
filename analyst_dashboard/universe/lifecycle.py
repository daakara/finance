"""
analyst_dashboard/universe/lifecycle.py

Authoritative Security Lifecycle Classification and Corporate Action Engine for ARX Terminal.
Governs point-in-time listing states: ACTIVE, SUSPENDED, DELISTED, ACQUIRED, UNRESOLVED, REINSTATED.
Enforces fail-closed delisting exclusion across Universe Builder, VCP Scanner, Value/GARP,
Smart Money discovery, Radar rankings, and trade-plan actionability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SecurityLifecycleStatus(str, Enum):
    """Canonical lifecycle states for securities."""
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    DELISTED = "DELISTED"
    ACQUIRED = "ACQUIRED"
    UNRESOLVED = "UNRESOLVED"
    REINSTATED = "REINSTATED"


class CorporateActionType(str, Enum):
    """Authoritative corporate action classifications."""
    CASH_MERGER_ACQUISITION = "CASH_MERGER_ACQUISITION"
    STOCK_MERGER_ACQUISITION = "STOCK_MERGER_ACQUISITION"
    VOLUNTARY_DELISTING = "VOLUNTARY_DELISTING"
    REGULATORY_DELISTING = "REGULATORY_DELISTING"
    BANKRUPTCY_LIQUIDATION = "BANKRUPTCY_LIQUIDATION"
    TRADING_HALT_SUSPENSION = "TRADING_HALT_SUSPENSION"
    REINSTATEMENT = "REINSTATEMENT"
    TICKER_RENAME = "TICKER_RENAME"


@dataclass(frozen=True)
class SecurityLifecycleRecord:
    """Immutable authoritative corporate action and listing lifecycle record."""
    canonical_id: str
    symbol: str
    lifecycle_status: SecurityLifecycleStatus
    action_type: CorporateActionType
    effective_date: str  # YYYY-MM-DD
    delisting_date: Optional[str] = None  # YYYY-MM-DD
    trading_suspension_date: Optional[str] = None  # YYYY-MM-DD
    reinstatement_date: Optional[str] = None  # YYYY-MM-DD
    acquirer: Optional[str] = None
    consideration: Optional[str] = None
    filings: List[Dict[str, str]] = field(default_factory=list)
    notes: str = ""
    execution_status: str = "DELISTED_SECURITY"
    status_label: str = "🛑 Delisted / Inactive"
    status_color: str = "rose"
    decision_state: str = "INELIGIBLE_INACTIVE"
    is_screening_eligible: bool = False
    is_investable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["lifecycle_status"] = self.lifecycle_status.value
        res["action_type"] = self.action_type.value if self.action_type else None
        return res

    def to_corporate_action_dict(self) -> Dict[str, Any]:
        """Convert to canonical corporate action dictionary format for API responses."""
        return {
            "status": "DELISTED" if self.lifecycle_status in (SecurityLifecycleStatus.DELISTED, SecurityLifecycleStatus.ACQUIRED) else self.lifecycle_status.value,
            "action_type": self.action_type.value if self.action_type else None,
            "effective_date": self.effective_date,
            "delisting_date": self.delisting_date or self.effective_date,
            "trading_suspension_date": self.trading_suspension_date,
            "reinstatement_date": self.reinstatement_date,
            "acquirer": self.acquirer,
            "consideration": self.consideration,
            "filings": list(self.filings),
            "execution_status": self.execution_status,
            "status_label": self.status_label,
            "status_color": self.status_color,
            "decision_state": self.decision_state,
            "screening_eligible": self.is_screening_eligible,
            "is_investable": self.is_investable,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class LifecycleEvaluationResult:
    """Point-in-time security lifecycle evaluation outcome."""
    canonical_id: Optional[str]
    symbol: str
    lifecycle_status: SecurityLifecycleStatus
    is_active_for_trading: bool
    is_screening_eligible: bool
    is_delisted: bool
    is_suspended: bool
    is_reinstated: bool
    as_of_date: Optional[str]
    effective_date: Optional[str]
    exclusion_reason: Optional[str]
    record: Optional[SecurityLifecycleRecord]
    corporate_action: Optional[Dict[str, Any]]


# ── Authoritative Historical Corporate Actions Registry ──────────────────────
# CIK / Exchange primary citations. CPRX is primary benchmark regression fixture.
_CONFIRMED_RECORDS: List[SecurityLifecycleRecord] = [
    SecurityLifecycleRecord(
        canonical_id="SEC_CPRX_001",
        symbol="CPRX",
        lifecycle_status=SecurityLifecycleStatus.ACQUIRED,
        action_type=CorporateActionType.CASH_MERGER_ACQUISITION,
        effective_date="2026-07-21",
        delisting_date="2026-07-21",
        trading_suspension_date="2026-07-15",
        acquirer="Angelini Pharma S.p.A.",
        consideration="$31.50 per share in cash",
        filings=[
            {"form": "25-NSE", "filing_date": "2026-07-10", "description": "Notification of Removal from Listing on Nasdaq"},
            {"form": "15-12B", "filing_date": "2026-07-20", "description": "Certification and Notice of Termination of Registration"},
        ],
        notes="Acquired by Angelini Pharma for $31.50/share in cash in July 2026. Common stock delisted and deregistered from Nasdaq.",
        execution_status="DELISTED_SECURITY",
        status_label="🛑 Delisted / Inactive",
        status_color="rose",
        decision_state="INELIGIBLE_INACTIVE",
        is_screening_eligible=False,
        is_investable=False,
    )
]


class SecurityLifecycleRegistry:
    """
    Central server-authoritative registry and evaluation engine for security lifecycle status.
    Ensures that confirmed corporate actions (delistings, mergers, suspensions, reinstatements)
    are strictly enforced at canonical identity and symbol levels.
    """

    def __init__(self, records: Optional[List[SecurityLifecycleRecord]] = None):
        self._by_id: Dict[str, SecurityLifecycleRecord] = {}
        self._by_symbol: Dict[str, List[SecurityLifecycleRecord]] = {}
        initial = records if records is not None else _CONFIRMED_RECORDS
        for r in initial:
            self.register(r)

    def register(self, record: SecurityLifecycleRecord) -> None:
        """Register an authoritative security lifecycle record."""
        clean_id = record.canonical_id.strip()
        clean_sym = record.symbol.strip().upper()
        self._by_id[clean_id] = record
        if clean_sym not in self._by_symbol:
            self._by_symbol[clean_sym] = []
        # Replace existing record for same canonical_id or append
        existing_idx = next(
            (i for i, r in enumerate(self._by_symbol[clean_sym]) if r.canonical_id == clean_id),
            None,
        )
        if existing_idx is not None:
            self._by_symbol[clean_sym][existing_idx] = record
        else:
            self._by_symbol[clean_sym].append(record)

    def get_record_by_id(self, canonical_id: str) -> Optional[SecurityLifecycleRecord]:
        """Look up by canonical security ID."""
        return self._by_id.get(canonical_id.strip())

    def get_latest_record_by_symbol(self, symbol: str) -> Optional[SecurityLifecycleRecord]:
        """Look up latest record for symbol."""
        records = self._by_symbol.get(symbol.strip().upper(), [])
        if not records:
            return None
        # Return record with newest effective_date
        return max(records, key=lambda r: r.effective_date)

    def get_all_records(self) -> List[SecurityLifecycleRecord]:
        """Retrieve all registered lifecycle records."""
        return list(self._by_id.values())

    def get_confirmed_corporate_actions_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Export dictionary compatible with legacy CONFIRMED_CORPORATE_ACTIONS mapping.
        Keys are uppercase ticker symbols.
        """
        res: Dict[str, Dict[str, Any]] = {}
        for sym, records in self._by_symbol.items():
            latest = max(records, key=lambda r: r.effective_date)
            res[sym] = latest.to_corporate_action_dict()
        return res

    def evaluate_lifecycle(
        self,
        symbol: str,
        canonical_id: Optional[str] = None,
        as_of_date: Optional[str] = None,
    ) -> LifecycleEvaluationResult:
        """
        Evaluate security lifecycle state point-in-time against an observation session date.

        Parameters
        ----------
        symbol : str
            Security ticker symbol (e.g. 'CPRX')
        canonical_id : Optional[str]
            Canonical security identifier (e.g. 'SEC_CPRX_001'). If provided, ensures
            exact entity binding and supports ticker reuse / reinstatement.
        as_of_date : Optional[str]
            Evaluation date YYYY-MM-DD or ISO timestamp. If None, evaluates against current state.

        Returns
        -------
        LifecycleEvaluationResult
            Authoritative lifecycle evaluation outcome.
        """
        clean_sym = symbol.strip().upper()
        clean_id = canonical_id.strip() if canonical_id else None
        obs_date = as_of_date[:10] if as_of_date else None

        # 1. Resolve relevant record: prioritize canonical_id, fallback to symbol
        record: Optional[SecurityLifecycleRecord] = None
        if clean_id:
            record = self._by_id.get(clean_id)
        if record is None and clean_sym in self._by_symbol:
            # If multiple records for symbol, pick the one matching date context or latest
            records = self._by_symbol[clean_sym]
            if obs_date:
                # Find the most recent record whose effective_date or suspension_date is <= obs_date,
                # or the earliest record if all are in the future
                past_or_current = [r for r in records if (r.trading_suspension_date or r.effective_date) <= obs_date]
                if past_or_current:
                    record = max(past_or_current, key=lambda r: r.effective_date)
                else:
                    record = min(records, key=lambda r: r.effective_date)
            else:
                record = max(records, key=lambda r: r.effective_date)

        # 2. If no record registered, security has no known corporate action disqualification
        if record is None:
            return LifecycleEvaluationResult(
                canonical_id=clean_id,
                symbol=clean_sym,
                lifecycle_status=SecurityLifecycleStatus.ACTIVE,
                is_active_for_trading=True,
                is_screening_eligible=True,
                is_delisted=False,
                is_suspended=False,
                is_reinstated=False,
                as_of_date=obs_date,
                effective_date=None,
                exclusion_reason=None,
                record=None,
                corporate_action=None,
            )

        # 3. Point-in-time evaluation against obs_date
        susp_date = record.trading_suspension_date or record.effective_date
        delist_date = record.delisting_date or record.effective_date
        reinstate_date = record.reinstatement_date

        # Case A: Future date with explicit reinstatement
        if obs_date and reinstate_date and obs_date >= reinstate_date:
            return LifecycleEvaluationResult(
                canonical_id=record.canonical_id,
                symbol=clean_sym,
                lifecycle_status=SecurityLifecycleStatus.REINSTATED,
                is_active_for_trading=True,
                is_screening_eligible=True,
                is_delisted=False,
                is_suspended=False,
                is_reinstated=True,
                as_of_date=obs_date,
                effective_date=reinstate_date,
                exclusion_reason=None,
                record=record,
                corporate_action=record.to_corporate_action_dict(),
            )

        # Case B: On or after confirmed delisting / acquisition date
        is_past_delisting = (obs_date >= delist_date) if obs_date else (
            record.lifecycle_status in (SecurityLifecycleStatus.DELISTED, SecurityLifecycleStatus.ACQUIRED)
        )
        if is_past_delisting:
            status = record.lifecycle_status
            if status not in (SecurityLifecycleStatus.DELISTED, SecurityLifecycleStatus.ACQUIRED):
                status = SecurityLifecycleStatus.DELISTED
            reason = f"Security {clean_sym} delisted/acquired effective {delist_date}: {record.notes}"
            return LifecycleEvaluationResult(
                canonical_id=record.canonical_id,
                symbol=clean_sym,
                lifecycle_status=status,
                is_active_for_trading=False,
                is_screening_eligible=False,
                is_delisted=True,
                is_suspended=False,
                is_reinstated=False,
                as_of_date=obs_date,
                effective_date=delist_date,
                exclusion_reason=reason,
                record=record,
                corporate_action=record.to_corporate_action_dict(),
            )

        # Case C: Trading suspension window (halted/suspended, but not yet legally delisted)
        is_past_suspension = (obs_date >= susp_date) if obs_date else (
            record.lifecycle_status == SecurityLifecycleStatus.SUSPENDED
        )
        if is_past_suspension:
            reason = f"Security {clean_sym} trading suspended effective {susp_date} pending corporate action."
            return LifecycleEvaluationResult(
                canonical_id=record.canonical_id,
                symbol=clean_sym,
                lifecycle_status=SecurityLifecycleStatus.SUSPENDED,
                is_active_for_trading=False,
                is_screening_eligible=False,
                is_delisted=False,  # Temporary halt != permanent delisting
                is_suspended=True,
                is_reinstated=False,
                as_of_date=obs_date,
                effective_date=susp_date,
                exclusion_reason=reason,
                record=record,
                corporate_action=record.to_corporate_action_dict(),
            )

        # Case D: Prior to suspension and delisting (historical backtest point-in-time)
        return LifecycleEvaluationResult(
            canonical_id=record.canonical_id,
            symbol=clean_sym,
            lifecycle_status=SecurityLifecycleStatus.ACTIVE,
            is_active_for_trading=True,
            is_screening_eligible=True,
            is_delisted=False,
            is_suspended=False,
            is_reinstated=False,
            as_of_date=obs_date,
            effective_date=None,
            exclusion_reason=None,
            record=record,
            corporate_action=None,
        )


# Singleton instance for process-wide authoritative lifecycle evaluation
GLOBAL_LIFECYCLE_REGISTRY = SecurityLifecycleRegistry()


def get_security_lifecycle_registry() -> SecurityLifecycleRegistry:
    """Retrieve global authoritative security lifecycle registry."""
    return GLOBAL_LIFECYCLE_REGISTRY
