"""
analyst_dashboard/research/maintenance/contracts.py

Data contracts, schemas, and event registers for the ARX Research-Maintenance Pilot.
Governed by ARX Closure Model v2, ARX-DECISION-RESEARCH-PROCESS-V1, and PILOT_MAINTENANCE_POLICY.md.

Invariants:
- Durable event register keyed by issuer and accession number.
- Strict separation of event publication time, reporting period, retrieval time, and processing state.
- Tracks source evidence with exact locators, units, scale, and accounting context.
- Distinguishes missing data (UNRESOLVED) from explicit numeric zero.
- Preserves honest omission without inventing fallbacks.
- Reusable across all Quality Watch assets.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Any, Union


class EventProcessingState(str, Enum):
    """Lifecycle states for primary-source maintenance events."""
    DISCOVERED = "DISCOVERED"
    REGISTERED = "REGISTERED"
    RECONCILED = "RECONCILED"
    ASSESSED = "ASSESSED"
    PACKET_GENERATED = "PACKET_GENERATED"
    REVIEW_PENDING = "REVIEW_PENDING"
    REVIEW_ACCEPTED = "REVIEW_ACCEPTED"
    REVIEW_REJECTED = "REVIEW_REJECTED"
    REJECTED_MISIDENTIFIED = "REJECTED_MISIDENTIFIED"
    REJECTED_DUPLICATE = "REJECTED_DUPLICATE"
    REJECTED_VALIDATION_FAILED = "REJECTED_VALIDATION_FAILED"
    FAILED = "FAILED"


class ProposedDisposition(str, Enum):
    """Research review disposition outcome under Step A of decision process."""
    MAINTAIN = "MAINTAIN"
    REVISE = "REVISE"
    INVALIDATE = "INVALIDATE"
    UNRESOLVED = "UNRESOLVED"
    CONSIDER_WITHDRAWAL = "CONSIDER_WITHDRAWAL"


class InterpretationType(str, Enum):
    """Epistemological classification separating objective facts from opinion."""
    FACT_CHANGE = "FACT_CHANGE"
    CALCULATED_DELTA = "CALCULATED_DELTA"
    ANALYST_JUDGMENT = "ANALYST_JUDGMENT"
    UNRESOLVED_EVIDENCE = "UNRESOLVED_EVIDENCE"


class FactStatus(str, Enum):
    """Status of an individual reconciled financial fact."""
    REPORTED = "REPORTED"
    DERIVED = "DERIVED"
    UNRESOLVED = "UNRESOLVED"
    EXPLICIT_ZERO = "EXPLICIT_ZERO"
    VALIDATION_CONFLICT = "VALIDATION_CONFLICT"


@dataclass
class SourceReceipt:
    """
    Cryptographic content receipt for a primary-source artifact.
    Identifies exact object type, URL, byte count, SHA-256 digest, retrieval mode,
    retrieval timestamps (remote vs cache-read), byte representation, and limitations.
    """
    object_type: str  # "INDEX_HTML", "PRIMARY_DOC_HTML", "SUBMISSIONS_JSON", "COMPANY_FACTS_JSON", "SYNTHETIC_FIXTURE"
    url: str
    retrieval_time_utc: Optional[str] = None
    byte_count: int = 0
    sha256: Optional[str] = None
    content_type: str = "text/html"
    retained_path: Optional[str] = None
    retrieval_mode: str = "LIVE"  # "LIVE", "CACHED", "UNAVAILABLE", "SYNTHETIC_FIXTURE"
    remote_retrieval_time_utc: Optional[str] = None
    cache_read_time_utc: Optional[str] = None
    byte_representation: str = "decoded_http_response_bytes"
    limitation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaintenanceEvent:
    """
    Durable primary-source filing event record.
    Preserves exact provenance, accession, and timing metadata.
    """
    event_id: str
    symbol: str
    cik: str
    company_name: str
    form: str
    accession_number: str
    filing_date: str  # Event publication date (YYYY-MM-DD)
    reporting_period: str  # Reporting period end date (YYYY-MM-DD)
    retrieval_time_utc: str  # Ingestion timestamp
    source_url: str
    primary_document: str
    acceptance_datetime_utc: Optional[str] = None
    index_url: Optional[str] = None
    index_sha256: Optional[str] = None
    submissions_url: Optional[str] = None
    company_facts_url: Optional[str] = None
    processing_state: EventProcessingState = EventProcessingState.REGISTERED
    is_replay: bool = False
    sha256: Optional[str] = None
    source_receipts: Optional[Dict[str, Dict[str, Any]]] = None
    successor_draft_id: Optional[str] = None
    review_packet_id: Optional[str] = None
    error_message: Optional[str] = None
    failed_stage: Optional[str] = None
    reconciliation_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["processing_state"] = self.processing_state.value if isinstance(self.processing_state, EventProcessingState) else str(self.processing_state)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaintenanceEvent":
        d = dict(data)
        if "processing_state" in d and isinstance(d["processing_state"], str):
            try:
                d["processing_state"] = EventProcessingState(d["processing_state"])
            except ValueError:
                d["processing_state"] = EventProcessingState.REGISTERED
        return cls(**d)


@dataclass
class ReconciledFinancialChange:
    """
    Material financial change between accepted baseline and draft successor.
    Preserves exact evidence locators, units, scale, and accounting context.
    """
    metric_key: str
    label: str
    baseline_value: Optional[float]
    baseline_display: str
    draft_value: Optional[float]
    draft_display: str
    delta_value: Optional[float]
    delta_pct: Optional[float]
    units: str  # "USD", "shares", "percent"
    source_accession: str
    concept: Optional[str]
    period_start: Optional[str]
    period_end: str
    accounting_context: str
    status: FactStatus = FactStatus.REPORTED
    is_missing_unresolved: bool = False
    source_accessions: List[str] = field(default_factory=list)
    derivation_formula: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, FactStatus) else str(self.status)
        return d


@dataclass
class DecisionAssessmentEffect:
    """
    Assessment of how a reconciled material change affects an accepted research conclusion.
    Separates verified facts, calculated effects, analytical interpretation, and unresolved evidence.
    """
    conclusion_key: str  # decisive_reason, supporting_evidence, opposing_evidence, watch_condition, invalidation_condition, financial_resilience, valuation_model
    conclusion_label: str
    baseline_state: str
    draft_state: str
    interpretation_type: InterpretationType
    materiality: str  # MATERIAL, INCREMENTAL_POSITIVE, INCREMENTAL_NEGATIVE, NEUTRAL
    owner_judgment_required: bool
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["interpretation_type"] = self.interpretation_type.value if isinstance(self.interpretation_type, InterpretationType) else str(self.interpretation_type)
        return d


@dataclass
class ReviewPacket:
    """
    Compact, decision-focused review packet linking accepted baseline to draft successor.
    Follows ARX-DECISION-RESEARCH-PROCESS-V1 template.
    """
    packet_id: str
    symbol: str
    company_name: str
    cik: str
    accepted_baseline_id: str
    accepted_baseline_cutoff: str
    draft_successor_id: str
    filing_metadata: Dict[str, Any]
    material_financial_changes: List[ReconciledFinancialChange]
    assessment_effects: List[DecisionAssessmentEffect]
    accounting_exceptions: List[str]
    undisclosed_information: List[str]
    valuation_comparison: Dict[str, Any]
    proposed_disposition: ProposedDisposition
    disposition_rationale: str
    questions_requiring_owner_judgment: List[str]
    stopping_rule: str
    created_at_utc: str
    is_replay: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "cik": self.cik,
            "accepted_baseline_id": self.accepted_baseline_id,
            "accepted_baseline_cutoff": self.accepted_baseline_cutoff,
            "draft_successor_id": self.draft_successor_id,
            "filing_metadata": self.filing_metadata,
            "material_financial_changes": [c.to_dict() for c in self.material_financial_changes],
            "assessment_effects": [e.to_dict() for e in self.assessment_effects],
            "accounting_exceptions": self.accounting_exceptions,
            "undisclosed_information": self.undisclosed_information,
            "valuation_comparison": self.valuation_comparison,
            "proposed_disposition": self.proposed_disposition.value,
            "disposition_rationale": self.disposition_rationale,
            "questions_requiring_owner_judgment": self.questions_requiring_owner_judgment,
            "stopping_rule": self.stopping_rule,
            "created_at_utc": self.created_at_utc,
            "is_replay": self.is_replay,
        }


class EventRegister:
    """
    Durable event register keyed by issuer symbol and accession number.
    Ensures repeated processing of the same accession does not create duplicate events.
    Supports atomic persistence and safe retry of incomplete / failed processing.
    """

    def __init__(self, register_path: Optional[Union[str, Path]] = None):
        self.register_path = Path(register_path) if register_path else None
        self.events: Dict[str, MaintenanceEvent] = {}
        if self.register_path and self.register_path.exists():
            self.load()

    def _make_key(self, symbol: str, accession_number: str, is_replay: bool = False) -> str:
        clean_sym = symbol.strip().upper()
        clean_acc = accession_number.strip()
        prefix = "REPLAY:" if is_replay else ""
        return f"{prefix}{clean_sym}:{clean_acc}"

    def has_event(self, symbol: str, accession_number: str, is_replay: bool = False) -> bool:
        key = self._make_key(symbol, accession_number, is_replay=is_replay)
        return key in self.events

    def get_event(self, symbol: str, accession_number: str, is_replay: bool = False) -> Optional[MaintenanceEvent]:
        key = self._make_key(symbol, accession_number, is_replay=is_replay)
        return self.events.get(key)

    def is_completed(self, symbol: str, accession_number: str, is_replay: bool = False) -> bool:
        evt = self.get_event(symbol, accession_number, is_replay=is_replay)
        if not evt:
            return False
        return evt.processing_state in (
            EventProcessingState.REVIEW_PENDING,
            EventProcessingState.REVIEW_ACCEPTED,
            EventProcessingState.REJECTED_MISIDENTIFIED,
        )

    def can_retry(self, symbol: str, accession_number: str, is_replay: bool = False) -> bool:
        evt = self.get_event(symbol, accession_number, is_replay=is_replay)
        if not evt:
            return True
        return evt.processing_state in (
            EventProcessingState.FAILED,
            EventProcessingState.REGISTERED,
            EventProcessingState.RECONCILED,
            EventProcessingState.ASSESSED,
        )

    def register_event(
        self,
        event: MaintenanceEvent,
        fail_if_duplicate: bool = True,
        allow_overwrite: bool = False,
    ) -> MaintenanceEvent:
        key = self._make_key(event.symbol, event.accession_number, is_replay=event.is_replay)
        if key in self.events:
            existing = self.events[key]
            # If the event previously failed or was interrupted, or allow_overwrite is set, allow safe update
            if allow_overwrite or self.can_retry(event.symbol, event.accession_number, is_replay=event.is_replay):
                self.events[key] = event
                if self.register_path:
                    self.save()
                return event

            if fail_if_duplicate:
                raise ValueError(
                    f"Duplicate accession error: Filing {event.accession_number} for {event.symbol} "
                    f"is already registered (event_id: {existing.event_id}, state: {existing.processing_state.value})."
                )
            return existing

        self.events[key] = event
        if self.register_path:
            self.save()
        return event

    def update_event_state(
        self,
        symbol: str,
        accession_number: str,
        state: EventProcessingState,
        is_replay: bool = False,
        successor_draft_id: Optional[str] = None,
        review_packet_id: Optional[str] = None,
        error_message: Optional[str] = None,
        failed_stage: Optional[str] = None,
        reconciliation_summary: Optional[Dict[str, Any]] = None,
        index_sha256: Optional[str] = None,
        sha256: Optional[str] = None,
        source_receipts: Optional[Dict[str, Any]] = None,
    ) -> MaintenanceEvent:
        key = self._make_key(symbol, accession_number, is_replay=is_replay)
        if key not in self.events:
            raise KeyError(f"Event not found in register: {key}")
        evt = self.events[key]
        evt.processing_state = state
        if successor_draft_id:
            evt.successor_draft_id = successor_draft_id
        if review_packet_id:
            evt.review_packet_id = review_packet_id
        if error_message:
            evt.error_message = error_message
        if failed_stage:
            evt.failed_stage = failed_stage
        if reconciliation_summary:
            evt.reconciliation_summary = reconciliation_summary
        if index_sha256:
            evt.index_sha256 = index_sha256
        if sha256:
            evt.sha256 = sha256
        if source_receipts:
            evt.source_receipts = source_receipts
        if self.register_path:
            self.save()
        return evt

    def list_events(self, symbol: Optional[str] = None, include_replays: bool = True) -> List[MaintenanceEvent]:
        evts = list(self.events.values())
        if not include_replays:
            evts = [e for e in evts if not e.is_replay]
        if symbol:
            clean_sym = symbol.strip().upper()
            return [e for e in evts if e.symbol == clean_sym]
        return evts

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        target = Path(path) if path else self.register_path
        if not target or not target.exists():
            return
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.events = {}
        for key, evt_data in data.get("events", {}).items():
            self.events[key] = MaintenanceEvent.from_dict(evt_data)

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Atomically saves register payload using a temporary file to prevent corruption."""
        target = Path(path) if path else self.register_path
        if not target:
            raise ValueError("No register path specified for saving.")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0.0",
            "title": "ARX Research Maintenance Event Register",
            "updated_at_utc": datetime.utcnow().isoformat() + "Z",
            "total_events": len(self.events),
            "events": {k: v.to_dict() for k, v in self.events.items()},
        }
        # Atomic file write
        temp_dir = target.parent
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            json.dump(payload, tf, indent=2)
            temp_path = Path(tf.name)
        os.replace(temp_path, target)
