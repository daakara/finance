"""
scripts/research/etf_v2/openfigi_models.py

Domain models and Pydantic schemas for the OpenFIGI Corroboration Engine.
Strictly separates:
1. Governed canonical input records
2. Provider wire mapping jobs
3. Provider raw response envelopes
4. Normalized corroboration results
5. Authoritative operational observation records
6. Derived operational active projection records
7. Execution summaries

Invariants Enforced:
- OFIGI-INV-001: Operational evidence only; zero canonical authority.
- OFIGI-INV-002: ISIN is sole legal identity key.
- OFIGI-INV-013: API keys and secrets never persisted.
- OFIGI-INV-014: Operational persistence physically separate from canonical store.
"""

from __future__ import annotations

from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class CorroborationOutcomeClass(str, Enum):
    """Normalized outcome classifications for OpenFIGI corroboration."""
    NO_OPERATIONAL_CORROBORATION = "NO_OPERATIONAL_CORROBORATION"
    EXACT_OPERATIONAL_CORROBORATION = "EXACT_OPERATIONAL_CORROBORATION"
    AMBIGUOUS_OPERATIONAL_CORROBORATION = "AMBIGUOUS_OPERATIONAL_CORROBORATION"
    PROVIDER_RESPONSE_INVALID = "PROVIDER_RESPONSE_INVALID"
    PROVIDER_AUTHENTICATION_FAILURE = "PROVIDER_AUTHENTICATION_FAILURE"
    PROVIDER_RATE_LIMIT_FAILURE = "PROVIDER_RATE_LIMIT_FAILURE"
    PROVIDER_TRANSPORT_FAILURE = "PROVIDER_TRANSPORT_FAILURE"
    PROVIDER_SERVER_FAILURE = "PROVIDER_SERVER_FAILURE"
    INVALID_REQUEST = "INVALID_REQUEST"
    CONFIGURATION_FAILURE = "CONFIGURATION_FAILURE"


class AuthorizedCanonicalInputRecord(BaseModel):
    """
    Caller-supplied governed canonical input record.
    OpenFIGI service must NOT self-select canonical population.
    """
    model_config = ConfigDict(frozen=True)

    canonical_internal_id: str = Field(..., description="Deterministic canonical internal ID (e.g. etfs:v1:ISIN:...)")
    isin: str = Field(..., description="ISO 6166 12-character ISIN")
    source_population_version: str = Field(..., description="Canonical population release version (e.g. 2.0.0)")
    source_snapshot_sha256: str = Field(..., description="Cryptographic SHA-256 digest of upstream canonical snapshot")

    # Optional contextual filters for candidate disambiguation
    mic_code: Optional[str] = Field(None, description="ISO 10383 market identifier code")
    exch_code: Optional[str] = Field(None, description="OpenFIGI exchange code")
    currency: Optional[str] = Field(None, description="ISO 4217 3-letter currency code")
    legal_name: Optional[str] = Field(None, description="Contextual legal name for disambiguation")

    def compute_idempotency_key(self, contract_version: str = "1.0.0") -> str:
        """
        Deterministic idempotency key mandated by OFIGI-INV-012:
        SHA256(canonical_internal_id + ":" + isin + ":" + contract_version + ":" + source_snapshot_sha256)
        """
        raw = f"{self.canonical_internal_id}:{self.isin}:{contract_version}:{self.source_snapshot_sha256}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class OpenFIGIMappingJob(BaseModel):
    """Wire mapping job payload conforming to OpenFIGI V3 API."""
    model_config = ConfigDict(frozen=True)

    idType: str = Field("ID_ISIN", description="Primary lookup key is strictly ID_ISIN")
    idValue: str = Field(..., description="Normalized 12-character ISIN string")
    micCode: Optional[str] = Field(None, description="Optional MIC code filter")
    exchCode: Optional[str] = Field(None, description="Optional exchange code filter")
    currency: Optional[str] = Field(None, description="Optional currency filter")


class OpenFIGICandidate(BaseModel):
    """Single candidate result returned by OpenFIGI mapping endpoint."""
    model_config = ConfigDict(extra="ignore")

    figi: str = Field(..., description="Bloomberg Financial Instrument Global Identifier")
    name: Optional[str] = None
    ticker: Optional[str] = None
    exchCode: Optional[str] = None
    compositeFIGI: Optional[str] = None
    securityType: Optional[str] = None
    marketSector: Optional[str] = None
    shareClassFIGI: Optional[str] = None
    securityType2: Optional[str] = None
    securityDescription: Optional[str] = None


class OpenFIGIResultEnvelope(BaseModel):
    """Wire envelope returned in array position by OpenFIGI V3 endpoint."""
    model_config = ConfigDict(extra="ignore")

    data: Optional[List[OpenFIGICandidate]] = None
    warning: Optional[str] = None
    error: Optional[str] = None


class NormalizedCorroborationResult(BaseModel):
    """Normalized domain result derived from provider response and contextual filtering."""
    canonical_internal_id: str
    isin: str
    outcome_class: CorroborationOutcomeClass
    figi: Optional[str] = None
    composite_figi: Optional[str] = None
    share_class_figi: Optional[str] = None
    ticker: Optional[str] = None
    exch_code: Optional[str] = None
    security_type: Optional[str] = None
    market_sector: Optional[str] = None
    name: Optional[str] = None
    candidates_count: int = 0
    warning_message: Optional[str] = None
    error_message: Optional[str] = None
    is_terminal: bool = True
    is_retryable: bool = False


class OpenFIGIObservation(BaseModel):
    """
    Authoritative operational audit observation record.
    Appended to openfigi_observations table. Never updated or deleted.
    """
    observation_id: str
    execution_id: str
    correlation_id: str
    canonical_internal_id: str
    isin: str
    idempotency_key: str
    source_population_version: str
    source_snapshot_sha256: str
    contract_version: str = "1.0.0"
    request_position: int
    request_filters: Dict[str, Any]
    attempt_count: int
    retry_count: int
    http_status: Optional[int] = None
    outcome_class: str
    normalized_result: Dict[str, Any]
    provider_response_evidence: Dict[str, Any]
    provider_response_digest: str
    observed_at: str
    created_at: str


class OpenFIGIActiveMapping(BaseModel):
    """
    Derived operational projection record.
    Stored in openfigi_active_mappings table, keyed by isin.
    """
    isin: str
    canonical_internal_id: str
    figi: Optional[str] = None
    composite_figi: Optional[str] = None
    share_class_figi: Optional[str] = None
    ticker: Optional[str] = None
    exch_code: Optional[str] = None
    security_type: Optional[str] = None
    market_sector: Optional[str] = None
    name: Optional[str] = None
    outcome_class: str
    last_observation_id: str
    source_population_version: str
    source_snapshot_sha256: str
    contract_version: str = "1.0.0"
    updated_at: str


class CorroborationExecutionSummary(BaseModel):
    """Summary metrics of an OpenFIGI corroboration execution run."""
    execution_id: str
    total_input_records: int
    total_observations_recorded: int
    exact_matches: int
    ambiguous_matches: int
    no_matches: int
    failures: int
    live_mapping_requests_executed: int = 0
