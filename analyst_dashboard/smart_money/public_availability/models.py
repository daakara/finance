"""
Public Availability Observer Domain Models — Epoch 001.
Implements prospective multi-channel observation classes, clock authority,
transport integrity, decision input completeness, and knowledge-time models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple


class EndpointChannel(str, Enum):
    """Governed SEC public channels."""
    RSS = "RSS"
    LATEST_FILINGS = "LATEST_FILINGS"
    FILING_INDEX = "FILING_INDEX"
    COMPLETE_SUBMISSION_TEXT = "COMPLETE_SUBMISSION_TEXT"
    PRIMARY_FORM_DOCUMENT = "PRIMARY_FORM_DOCUMENT"
    OWNERSHIP_XML = "OWNERSHIP_XML"
    SUBMISSIONS_API = "SUBMISSIONS_API"
    PDS = "PDS"  # Out of scope for first-public timing authority


class ChannelObservationRole(str, Enum):
    """Semantic role of each channel."""
    METADATA_PUBLIC_OBSERVATION = "METADATA_PUBLIC_OBSERVATION"
    CONTENT_OR_STRUCTURAL_PUBLIC_OBSERVATION = "CONTENT_OR_STRUCTURAL_PUBLIC_OBSERVATION"
    CONTENT_PUBLIC_OBSERVATION = "CONTENT_PUBLIC_OBSERVATION"
    STRUCTURED_METADATA_OBSERVATION = "STRUCTURED_METADATA_OBSERVATION"
    OUT_OF_SCOPE_FOR_FIRST_PUBLIC_OBSERVER = "OUT_OF_SCOPE_FOR_FIRST_PUBLIC_OBSERVER"


CHANNEL_ROLE_MAP: Dict[EndpointChannel, ChannelObservationRole] = {
    EndpointChannel.RSS: ChannelObservationRole.METADATA_PUBLIC_OBSERVATION,
    EndpointChannel.LATEST_FILINGS: ChannelObservationRole.METADATA_PUBLIC_OBSERVATION,
    EndpointChannel.FILING_INDEX: ChannelObservationRole.CONTENT_OR_STRUCTURAL_PUBLIC_OBSERVATION,
    EndpointChannel.COMPLETE_SUBMISSION_TEXT: ChannelObservationRole.CONTENT_PUBLIC_OBSERVATION,
    EndpointChannel.PRIMARY_FORM_DOCUMENT: ChannelObservationRole.CONTENT_PUBLIC_OBSERVATION,
    EndpointChannel.OWNERSHIP_XML: ChannelObservationRole.CONTENT_PUBLIC_OBSERVATION,
    EndpointChannel.SUBMISSIONS_API: ChannelObservationRole.STRUCTURED_METADATA_OBSERVATION,
    EndpointChannel.PDS: ChannelObservationRole.OUT_OF_SCOPE_FOR_FIRST_PUBLIC_OBSERVER,
}


class PublicObservationClass(str, Enum):
    """
    Governed observation classifications.
    Never collapsed to simple AVAILABLE / UNAVAILABLE.
    """
    METADATA_PUBLIC = "METADATA_PUBLIC"
    CONTENT_PUBLIC = "CONTENT_PUBLIC"
    DECISION_INPUT_COMPLETE = "DECISION_INPUT_COMPLETE"
    ENDPOINT_NOT_PRESENT = "ENDPOINT_NOT_PRESENT"
    TRANSPORT_INDETERMINATE = "TRANSPORT_INDETERMINATE"
    IDENTITY_INVALID = "IDENTITY_INVALID"
    TRANSPORT_INCOMPLETE = "TRANSPORT_INCOMPLETE"
    STRUCTURALLY_INVALID = "STRUCTURALLY_INVALID"
    DECISION_INPUT_INCOMPLETE = "DECISION_INPUT_INCOMPLETE"
    CONTENT_VERSION_CONFLICT = "CONTENT_VERSION_CONFLICT"


class TransportIntegrity(str, Enum):
    COMPLETE = "COMPLETE"
    TRUNCATED = "TRUNCATED"
    UNKNOWN = "UNKNOWN"


class ClockStatus(str, Enum):
    SYNCHRONIZED = "SYNCHRONIZED"
    DEGRADED = "DEGRADED"
    UNTRUSTED = "UNTRUSTED"


class DecisionInputCompleteness(str, Enum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    UNKNOWN = "UNKNOWN"


class DecisionInputReasonCode(str, Enum):
    REQUIRED_TRANSACTION_CODE_MISSING = "REQUIRED_TRANSACTION_CODE_MISSING"
    REQUIRED_ACQUIRED_DISPOSED_MISSING = "REQUIRED_ACQUIRED_DISPOSED_MISSING"
    REQUIRED_SHARES_MISSING = "REQUIRED_SHARES_MISSING"
    REQUIRED_PRICE_MISSING = "REQUIRED_PRICE_MISSING"
    REQUIRED_OWNER_RELATIONSHIP_MISSING = "REQUIRED_OWNER_RELATIONSHIP_MISSING"
    REQUIRED_FOOTNOTE_UNRESOLVED = "REQUIRED_FOOTNOTE_UNRESOLVED"
    REQUIRED_DOCUMENT_MISSING = "REQUIRED_DOCUMENT_MISSING"
    SOURCE_IDENTITY_CONFLICT = "SOURCE_IDENTITY_CONFLICT"
    TRANSPORT_INCOMPLETE = "TRANSPORT_INCOMPLETE"
    SATISFIED = "SATISFIED"


class ContentChangeClass(str, Enum):
    INITIAL = "INITIAL"
    UNRESOLVED = "UNRESOLVED"
    REPRESENTATION_CHANGE = "REPRESENTATION_CHANGE"
    SOURCE_CORRECTION = "SOURCE_CORRECTION"
    PAC_RELATED_CHANGE = "PAC_RELATED_CHANGE"
    AMENDMENT = "AMENDMENT"
    PUBLICATION_COMPLETION = "PUBLICATION_COMPLETION"
    CONTENT_VERSION_CONFLICT = "CONTENT_VERSION_CONFLICT"
    SEMANTIC_CONTENT_CHANGE = "SEMANTIC_CONTENT_CHANGE"
    CONTENT_AVAILABILITY_REGRESSION = "CONTENT_AVAILABILITY_REGRESSION"


class ChannelOrder(str, Enum):
    METADATA_PRECEDED_CONTENT = "METADATA_PRECEDED_CONTENT"
    CONTENT_PRECEDED_METADATA = "CONTENT_PRECEDED_METADATA"
    INDETERMINATE = "INDETERMINATE"
    METADATA_ONLY = "METADATA_ONLY"
    CONTENT_ONLY = "CONTENT_ONLY"
    NONE = "NONE"


# Canonical Frozen Decision-Input Contract (Section 21 Path B)
CANONICAL_DECISION_INPUT_CONTRACT_ID = "SMART_MONEY_REPORTED_PURCHASE_TRANSACTION_V1"
LEGACY_DECISION_INPUT_CONTRACT_ID = "SMART_MONEY_OPEN_MARKET_PURCHASE_V1"
CANONICAL_DECISION_INPUT_CONTRACT_SHA256 = "c32aacab8f5e7d309655720f99fb8a3952029960d2a5a263ae65c889194698f0"


@dataclass(frozen=True)
class ClockAuthorityRecord:
    clock_source: str
    clock_sync_method: str
    clock_observed_at: str
    clock_offset_ms: float
    clock_uncertainty_ms: float
    clock_status: ClockStatus

    def is_trusted(self) -> bool:
        return self.clock_status != ClockStatus.UNTRUSTED


@dataclass(frozen=True)
class AccessPolicyRecord:
    policy_id: str
    policy_version: str
    official_sec_source_url: str
    official_sec_policy_sha256: str
    official_sec_fetched_at: str
    max_requests_per_second: float
    user_agent_format: str
    policy_content_hash: str


@dataclass(frozen=True)
class PublicObservationRecord:
    observation_id: str
    run_id: str
    accession_number: str
    issuer_cik: str
    form_type: str
    endpoint_class: EndpointChannel
    endpoint_url: str
    request_sequence: int
    request_started_at: str
    response_completed_at: str
    clock_uncertainty_ms: float
    conservative_observation_time: str
    earliest_possible_time: str
    latest_possible_time: str
    http_status: int
    transport_integrity: TransportIntegrity
    bytes_received: int
    raw_response_sha256: str
    raw_response_retained: bool
    identity_valid: bool
    structural_valid: bool
    decision_input_completeness: DecisionInputCompleteness
    decision_input_reason_code: Optional[str]
    observation_class: PublicObservationClass
    created_at: str
    content_type: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None


@dataclass(frozen=True)
class PublicContentVersionRecord:
    version_id: str
    observation_id: str
    accession_number: str
    endpoint_class: EndpointChannel
    raw_content_sha256: str
    normalized_semantic_hash: str
    observed_at: str
    content_change_class: ContentChangeClass
    created_at: str


@dataclass(frozen=True)
class PublicKnowledgeDerivationRecord:
    derivation_id: str
    accession_number: str
    issuer_cik: str
    form_type: str
    decision_input_contract_id: str
    decision_input_contract_sha256: str
    first_proven_metadata_public_at: Optional[str]
    metadata_observation_id: Optional[str]
    first_proven_content_public_at: Optional[str]
    content_observation_id: Optional[str]
    first_proven_decision_input_complete_at: Optional[str]
    decision_input_observation_id: Optional[str]
    conservative_knowledge_time: Optional[str]
    exact_first_public_available_at: str  # ALWAYS "NOT_ESTABLISHED"
    channel_order: ChannelOrder
    derived_from_observation_ids: List[str]
    derivation_hash: str
    created_at: str
    content_version_count: int = 1
    has_semantic_content_change: bool = False
    has_content_regression: bool = False
