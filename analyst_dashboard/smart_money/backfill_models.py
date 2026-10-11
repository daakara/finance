"""
Canonical Data Models, Enumerations, Specifications, and Manifests
for SEC Form 4 Historical Ingestion and Point-in-Time Backfill.
Adheres strictly to ARX Terminal Smart Money Wave 2B Governance.
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional


class BackfillMode(str, Enum):
    """Governed historical backfill modes."""
    CURRENT_UNIVERSE_HISTORICAL_BACKFILL = "CURRENT_UNIVERSE_HISTORICAL_BACKFILL"
    PIT_HISTORICAL_UNIVERSE_BACKFILL = "PIT_HISTORICAL_UNIVERSE_BACKFILL"


class BackfillStatus(str, Enum):
    """Lifecycle state of a backfill job."""
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    SEALED_COMPLETE = "SEALED_COMPLETE"
    SEALED_PARTIAL = "SEALED_PARTIAL"
    SEALED_FAILED = "SEALED_FAILED"


class ExecutionScopeType(str, Enum):
    """Classification of target execution scope."""
    EXPLICIT_TARGETS = "EXPLICIT_TARGETS"
    TOP_N_UNIVERSE = "TOP_N_UNIVERSE"
    FULL_UNIVERSE = "FULL_UNIVERSE"


class FilingDisposition(str, Enum):
    """Exhaustive closed set of source filing outcomes (no silent skips)."""
    FETCHED_AND_PARSED = "FETCHED_AND_PARSED"
    FETCHED_NO_RELEVANT_TRANSACTIONS = "FETCHED_NO_RELEVANT_TRANSACTIONS"
    FETCH_FAILED_RETRYABLE = "FETCH_FAILED_RETRYABLE"
    FETCH_FAILED_TERMINAL = "FETCH_FAILED_TERMINAL"
    PARSE_FAILED = "PARSE_FAILED"
    UNSUPPORTED_DOCUMENT = "UNSUPPORTED_DOCUMENT"
    AMENDMENT_UNRESOLVED = "AMENDMENT_UNRESOLVED"


class AmendmentRelationship(str, Enum):
    """Form 4 vs Form 4/A lineage relationship."""
    ORIGINAL_FILING = "ORIGINAL_FILING"
    AMENDMENT = "AMENDMENT"
    SUPERSEDED_OR_CORRECTED = "SUPERSEDED_OR_CORRECTED"
    UNRESOLVED_AMENDMENT = "UNRESOLVED_AMENDMENT"


@dataclass
class SourceAvailabilityRecord:
    """
    Append-only ledger entry tracking exact public SEC availability time.
    INVARIANT: SOURCE_AVAILABILITY_CAN_BE_REWRITTEN_EARLIER = NO.
    """
    accession_number: str
    issuer_cik: str
    public_availability_at: str  # ISO8601 UTC
    first_fetched_at: str        # ISO8601 UTC
    source_hash: str             # SHA-256 of raw submission
    form_type: str               # "4" or "4/A"
    normalization_version: str   # e.g. "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BackfillPlan:
    """
    Semantic specification of an intended historical backfill plan.
    Plan hash excludes wall-clock runtime and execution-specific target scopes.
    """
    universe_build_id: str
    universe_content_hash: str
    backfill_mode: str  # BackfillMode value
    requested_date_start: str  # YYYY-MM-DD
    requested_date_end: str    # YYYY-MM-DD
    source_contract_version: str = "1.0.0"
    normalization_version: str = "1.0.0"
    parser_version: str = "1.0.0"
    eligibility_policy_version: str = "sm-eligibility-v2.0"
    cik_mapping_policy_version: str = "cik-mapping-v1.0"
    request_policy_version: str = "1.0.0"

    def compute_plan_hash(self) -> str:
        """Computes deterministic SHA-256 over canonical JSON representation."""
        payload = {
            "universe_build_id": self.universe_build_id,
            "universe_content_hash": self.universe_content_hash,
            "backfill_mode": self.backfill_mode,
            "requested_date_start": self.requested_date_start,
            "requested_date_end": self.requested_date_end,
            "source_contract_version": self.source_contract_version,
            "normalization_version": self.normalization_version,
            "parser_version": self.parser_version,
            "eligibility_policy_version": self.eligibility_policy_version,
            "cik_mapping_policy_version": self.cik_mapping_policy_version,
            "request_policy_version": self.request_policy_version,
        }
        dumped = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def get_plan_id(self) -> str:
        return f"PLAN-{self.compute_plan_hash()[:16]}"


@dataclass
class BackfillExecution:
    """
    Concrete execution scope binding plan identity to target population, batch boundaries, and stage.
    Execution ID is strictly bound to scope hash, preventing identity collisions across stages.
    """
    plan_id: str
    plan_hash: str
    scope_type: str  # ExecutionScopeType value
    target_symbols: List[str]
    target_ciks: List[str]
    stage_label: str
    max_issuers: Optional[int] = None
    max_filings_per_issuer: Optional[int] = None
    execution_policy_version: str = "exec-v2.0"

    def compute_scope_hash(self) -> str:
        payload = {
            "plan_hash": self.plan_hash,
            "scope_type": self.scope_type,
            "target_symbols": sorted(self.target_symbols),
            "target_ciks": sorted(self.target_ciks),
            "max_issuers": self.max_issuers,
            "max_filings_per_issuer": self.max_filings_per_issuer,
            "stage_label": self.stage_label,
            "execution_policy_version": self.execution_policy_version,
        }
        dumped = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def get_execution_id(self) -> str:
        return f"EXEC-{self.compute_scope_hash()[:16]}"


@dataclass
class SealedExecutionManifest:
    """
    Immutable audit receipt of an executed backfill run with distinct execution identity.
    Once sealed, mutations are strictly rejected (EXECUTION_MANIFEST_OVERWRITE_COUNT = 0).
    """
    execution_id: str
    plan_id: str
    plan_hash: str
    scope_hash: str
    stage_label: str
    scope_type: str
    universe_build_id: str
    universe_content_hash: str
    requested_start: str
    requested_end: str
    started_at: str
    completed_at: Optional[str] = None
    target_symbols: List[str] = field(default_factory=list)
    target_ciks: List[str] = field(default_factory=list)
    issuer_count_requested: int = 0
    issuer_count_attempted: int = 0
    issuer_count_completed: int = 0
    issuer_count_failed: int = 0
    filings_discovered: int = 0
    filings_fetched: int = 0
    filings_cached: int = 0
    filings_failed: int = 0
    transactions_parsed: int = 0
    transactions_rejected: int = 0
    open_market_purchase_count: int = 0
    duplicates_detected: int = 0
    amendments_detected: int = 0
    source_manifest_hash: str = ""
    transaction_manifest_hash: str = ""
    status: str = BackfillStatus.PLANNED.value
    disposition_counts: Dict[str, int] = field(default_factory=dict)
    survivorship_bias_warning: str = (
        "CURRENT_UNIVERSE_HISTORICAL_BACKFILL: Uses present-day universe members; "
        "contains survivorship bias; MUST NOT be used for historical whole-market performance claims."
    )
    historical_market_coverage_claim: str = "NOT_AUTHORIZED"
    historical_performance_claim: str = "NOT_AUTHORIZED"
    is_sealed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_manifest_sha256(self) -> str:
        dumped = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def compute_sealed_manifest_hash(self) -> str:
        return self.compute_manifest_sha256()


@dataclass
class SmartMoneyBackfillSpec:
    """
    Immutable specification of an intended historical backfill execution.
    Deterministic specification hash excludes wall-clock runtime.
    Maintained for backward compatibility; bridges to BackfillPlan.
    """
    universe_build_id: str
    universe_content_hash: str
    backfill_mode: str  # BackfillMode value
    requested_date_start: str  # YYYY-MM-DD
    requested_date_end: str    # YYYY-MM-DD
    source_contract_version: str = "1.0.0"
    normalization_version: str = "1.0.0"
    parser_version: str = "1.0.0"
    eligibility_policy_version: str = "sm-eligibility-v1.0"
    cik_mapping_policy_version: str = "cik-mapping-v1.0"
    request_policy_version: str = "1.0.0"

    def compute_spec_hash(self) -> str:
        """Computes deterministic SHA-256 over canonical JSON representation."""
        payload = {
            "universe_build_id": self.universe_build_id,
            "universe_content_hash": self.universe_content_hash,
            "backfill_mode": self.backfill_mode,
            "requested_date_start": self.requested_date_start,
            "requested_date_end": self.requested_date_end,
            "source_contract_version": self.source_contract_version,
            "normalization_version": self.normalization_version,
            "parser_version": self.parser_version,
            "eligibility_policy_version": self.eligibility_policy_version,
            "cik_mapping_policy_version": self.cik_mapping_policy_version,
            "request_policy_version": self.request_policy_version,
        }
        dumped = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def get_backfill_id(self) -> str:
        return f"BF-{self.compute_spec_hash()[:16]}"

    def to_plan(self) -> BackfillPlan:
        return BackfillPlan(
            universe_build_id=self.universe_build_id,
            universe_content_hash=self.universe_content_hash,
            backfill_mode=self.backfill_mode,
            requested_date_start=self.requested_date_start,
            requested_date_end=self.requested_date_end,
            source_contract_version=self.source_contract_version,
            normalization_version=self.normalization_version,
            parser_version=self.parser_version,
            eligibility_policy_version=self.eligibility_policy_version,
            cik_mapping_policy_version=self.cik_mapping_policy_version,
            request_policy_version=self.request_policy_version,
        )


@dataclass
class SmartMoneyBackfillManifest:
    """
    Immutable audit receipt of an executed backfill run.
    Binds universe, mode, counts, raw/normalized content hashes, and survivorship warnings.
    Supports execution-aware identity fields.
    """
    backfill_id: str
    backfill_spec_hash: str
    backfill_mode: str
    universe_build_id: str
    universe_content_hash: str
    requested_start: str
    requested_end: str
    started_at: str
    completed_at: Optional[str] = None
    issuer_count_requested: int = 0
    issuer_count_attempted: int = 0
    issuer_count_completed: int = 0
    issuer_count_failed: int = 0
    filings_discovered: int = 0
    filings_fetched: int = 0
    filings_cached: int = 0
    filings_failed: int = 0
    transactions_parsed: int = 0
    transactions_rejected: int = 0
    open_market_purchase_count: int = 0
    duplicates_detected: int = 0
    amendments_detected: int = 0
    source_manifest_hash: str = ""
    transaction_manifest_hash: str = ""
    status: str = BackfillStatus.PLANNED.value
    survivorship_bias_warning: str = (
        "CURRENT_UNIVERSE_HISTORICAL_BACKFILL: Uses present-day universe members; "
        "contains survivorship bias; MUST NOT be used for historical whole-market performance claims."
    )
    historical_market_coverage_claim: str = "NOT_AUTHORIZED"
    historical_performance_claim: str = "NOT_AUTHORIZED"
    disposition_counts: Dict[str, int] = field(default_factory=dict)
    execution_id: Optional[str] = None
    plan_id: Optional[str] = None
    scope_hash: Optional[str] = None
    stage_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_manifest_sha256(self) -> str:
        dumped = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()
