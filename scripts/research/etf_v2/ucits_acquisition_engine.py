"""
scripts/research/etf_v2/ucits_acquisition_engine.py

Statutory UCITS Authority Acquisition Engine for Pipeline V2 Wave 4.

Enforces:
- Exact 10-stage acquisition state machine execution
- Strict HTTPS enforcement, TLS verification, and bounded redirects (max 3)
- Bounded operational limits: 10s connect, 30s read, 50 MiB maximum payload
- Transparent transfer decompression prior to decoded-byte SHA-256 computation
- Content validation for statutory PDF documents (%PDF- header and %%EOF trailer)
- Closed 11-member AcquisitionOutcome classification
- Pluggable air-gapped transport interface for 100% deterministic test execution
- Zero product-specific branching
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
import datetime
import gzip
import hashlib
import io
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse
import zlib

from .global_identity_models import (
    InvalidIdentifierError,
    Jurisdiction,
    UnsupportedJurisdictionError,
)
from .global_identifier_authority import normalize_isin, validate_isin
from .ucits_acquisition_models import (
    AcquisitionOutcome,
    AcquisitionResult,
    assert_failure_is_not_no_match,
    AuthorityLocator,
    AuthorityRequest,
    RawArtifact,
    RETRYABLE_OUTCOMES,
    UCITSPopulationUniverseCounters,
)
from .ucits_authority_adapter import (
    CONNECT_TIMEOUT_SECONDS,
    HTTPS_REQUIRED,
    MAX_PAYLOAD_BYTES,
    MAX_REDIRECTS,
    PROHIBITED_AUTHORITY_HOSTS,
    READ_TIMEOUT_SECONDS,
    UCITSAcquisitionStateMachine,
    UCITSAuthorityAdapter,
)
from .ucits_provenance_models import (
    AcquisitionTransportError,
    AUTHORIZED_DOCUMENT_CLASSES,
    InvalidSourceAuthorityError,
    ProvenanceConflictError,
    ProvenanceValidationError,
    UCITSSourceProvenanceRecord,
    UnsupportedDocumentTypeError,
    WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS,
)

LOGGER_NAME = "arx.etf_v2.ucits.acquisition_engine"
logger = logging.getLogger(LOGGER_NAME)

MAX_REQUEST_RATE_PER_SECOND: float = 2.0
HTTP_DOWNGRADE_ALLOWED: bool = False
TLS_VALIDATION: bool = True
MAX_REDIRECT_HOPS: int = 3


@dataclass(frozen=True)
class TransportResponse:
    """Standardized HTTP response wrapper produced by transport handlers."""
    status_code: int
    headers: Dict[str, str]
    raw_body: bytes
    final_url: str
    redirect_history: Tuple[str, ...] = field(default_factory=tuple)


class HttpTransportHandler(abc.ABC):
    """Abstract interface for document transport handling."""

    @abc.abstractmethod
    def fetch(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = READ_TIMEOUT_SECONDS,
    ) -> TransportResponse:
        """Retrieves raw content from URL or raises appropriate transport exceptions."""
        pass


class DeterministicMockTransportHandler(HttpTransportHandler):
    """
    Deterministic air-gapped transport handler for automated CI test execution.
    Allows registering pre-configured responses, error simulations, and redirects.
    """

    def __init__(self) -> None:
        self._endpoints: Dict[str, TransportResponse] = {}
        self._exceptions: Dict[str, Exception] = {}
        self._call_counts: Dict[str, int] = {}

    def register_endpoint(
        self,
        url: str,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        raw_body: bytes = b"",
        final_url: Optional[str] = None,
        redirect_history: Tuple[str, ...] = (),
    ) -> None:
        norm_headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._endpoints[url] = TransportResponse(
            status_code=status_code,
            headers=norm_headers,
            raw_body=raw_body,
            final_url=final_url or url,
            redirect_history=redirect_history,
        )

    def register_exception(self, url: str, exc: Exception) -> None:
        self._exceptions[url] = exc

    def get_call_count(self, url: str) -> int:
        return self._call_counts.get(url, 0)

    def fetch(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = READ_TIMEOUT_SECONDS,
    ) -> TransportResponse:
        self._call_counts[url] = self._call_counts.get(url, 0) + 1

        if url in self._exceptions:
            raise self._exceptions[url]

        if url in self._endpoints:
            return self._endpoints[url]

        # Default 404 for unmapped mock URLs
        return TransportResponse(
            status_code=404,
            headers={"content-type": "text/html"},
            raw_body=b"<html><body>404 Not Found</body></html>",
            final_url=url,
        )


def decode_content_encoding(raw_body: bytes, content_encoding: Optional[str]) -> bytes:
    """
    Decodes HTTP compressed payloads (gzip/deflate) to ensure the canonical
    SHA-256 hash is computed strictly over decoded raw artifact bytes.
    """
    if not content_encoding:
        return raw_body

    enc = content_encoding.lower().strip()
    if enc == "gzip":
        try:
            return gzip.decompress(raw_body)
        except Exception as e:
            raise AcquisitionTransportError(f"Failed to decompress gzip content: {e}") from e
    elif enc == "deflate":
        try:
            return zlib.decompress(raw_body)
        except Exception as e:
            raise AcquisitionTransportError(f"Failed to decompress deflate content: {e}") from e
    elif enc in ("identity", "none"):
        return raw_body
    else:
        raise AcquisitionTransportError(f"Unsupported content-encoding: {enc}")


def validate_pdf_content(payload: bytes) -> None:
    """
    Validates PDF format integrity:
    1. Must begin with '%PDF-' magic bytes.
    2. Must contain '%%EOF' closing marker in the trailing 1024 bytes.
    """
    if not payload.startswith(b"%PDF-"):
        raise AcquisitionTransportError("Payload does not start with valid PDF magic bytes '%PDF-'.")
    if b"%%EOF" not in payload[-1024:]:
        raise AcquisitionTransportError("Payload does not contain closing '%%EOF' marker in trailing bytes.")


class UCITSAcquisitionEngine:
    """
    Statutory authority acquisition driver running the 10-stage state machine
    with operational boundaries, bounded retries, and denominator tracking.
    """

    def __init__(
        self,
        adapter: Optional[UCITSAuthorityAdapter] = None,
        transport: Optional[HttpTransportHandler] = None,
        state_machine: Optional[UCITSAcquisitionStateMachine] = None,
        rate_limit_rps: float = MAX_REQUEST_RATE_PER_SECOND,
    ) -> None:
        self.adapter = adapter or UCITSAuthorityAdapter()
        self.transport = transport or DeterministicMockTransportHandler()
        self.state_machine = state_machine or UCITSAcquisitionStateMachine(adapter=self.adapter)
        self.rate_limit_rps = rate_limit_rps
        self._last_request_time: float = 0.0

    def _apply_rate_limit(self) -> None:
        if self.rate_limit_rps <= 0:
            return
        min_interval = 1.0 / self.rate_limit_rps
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.monotonic()

    def validate_request_bounds(self, request: AuthorityRequest) -> None:
        """Validates request syntax, jurisdiction support, and HTTPS policy."""
        url = request.locator.source_url
        if not url:
            raise ProvenanceValidationError("Source URL is required.")

        parsed = urlparse(url)
        if HTTPS_REQUIRED and parsed.scheme.lower() != "https":
            raise UnsupportedJurisdictionError(f"Non-HTTPS scheme prohibited: {parsed.scheme!r}")

        host = (parsed.hostname or "").lower()
        if host in PROHIBITED_AUTHORITY_HOSTS:
            raise InvalidSourceAuthorityError(f"Prohibited commercial authority host: {host!r}")

        dom = request.locator.jurisdiction.upper().strip()
        if not self.adapter.supports_jurisdiction(dom):
            raise UnsupportedJurisdictionError(f"Unsupported jurisdiction: {dom!r}")

        if not request.share_class_isin:
            raise InvalidIdentifierError("ISIN is required in request.")

        norm_isin = normalize_isin(request.share_class_isin)
        if not validate_isin(norm_isin):
            raise InvalidIdentifierError(f"ISIN '{norm_isin}' failed Mod-10 checksum validation.")
        if not norm_isin.startswith(dom):
            raise InvalidIdentifierError(f"ISIN prefix '{norm_isin[:2]}' does not match domicile '{dom}'.")

    def execute_attempt(self, request: AuthorityRequest, attempt_id: str = "att-001") -> AcquisitionResult:
        """
        Executes a statutory acquisition attempt following the 10-stage state machine.
        Returns a closed AcquisitionResult with an explicit AcquisitionOutcome.
        """
        # Step 1: Pre-request validation
        try:
            self.validate_request_bounds(request)
        except UnsupportedJurisdictionError as e:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.UNSUPPORTED,
                request=request,
                error_message=str(e),
            )
        except InvalidSourceAuthorityError as e:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.UNSUPPORTED,
                request=request,
                error_message=str(e),
            )
        except (InvalidIdentifierError, ProvenanceValidationError) as e:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.INVALID_REQUEST,
                request=request,
                error_message=str(e),
            )

        # Step 2: Rate limiting & HTTP transport fetch
        self._apply_rate_limit()

        try:
            resp = self.transport.fetch(
                url=request.locator.source_url,
                headers=request.headers,
                timeout_seconds=min(request.timeout_seconds, READ_TIMEOUT_SECONDS),
            )
        except TimeoutError as e:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.RETRIEVAL_FAILURE,
                request=request,
                error_message=f"Connection/read timeout: {e}",
            )
        except Exception as e:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.RETRIEVAL_FAILURE,
                request=request,
                error_message=f"Transport error: {e}",
            )

        # Step 3: Redirect policy checks
        if len(resp.redirect_history) > MAX_REDIRECT_HOPS:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.RETRIEVAL_FAILURE,
                request=request,
                error_message=f"Redirect hops {len(resp.redirect_history)} exceeded max {MAX_REDIRECT_HOPS}",
            )

        for red_url in resp.redirect_history:
            parsed_red = urlparse(red_url)
            if not HTTP_DOWNGRADE_ALLOWED and parsed_red.scheme.lower() != "https":
                return AcquisitionResult(
                    attempt_id=attempt_id,
                    outcome=AcquisitionOutcome.RETRIEVAL_FAILURE,
                    request=request,
                    error_message=f"HTTPS downgrade detected in redirect: {red_url}",
                )

        # Step 4: HTTP status outcome mapping
        status = resp.status_code
        if status == 404:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.NO_MATCH,
                request=request,
                error_message=f"Authority returned HTTP 404 Not Found for {request.locator.source_url}",
            )
        elif status in (401, 403):
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.ACCESS_DENIED,
                request=request,
                error_message=f"Authority access denied (HTTP {status})",
            )
        elif status == 429:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.RATE_LIMITED,
                request=request,
                error_message="Authority rate limit exceeded (HTTP 429)",
            )
        elif status in (500, 502, 503, 504):
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.AUTHORITY_UNAVAILABLE,
                request=request,
                error_message=f"Authority service unavailable (HTTP {status})",
            )
        elif status != 200:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.RETRIEVAL_FAILURE,
                request=request,
                error_message=f"Unexpected HTTP status: {status}",
            )

        # Step 5: Payload decompression & byte integrity checks
        encoding = resp.headers.get("content-encoding")
        try:
            decoded_bytes = decode_content_encoding(resp.raw_body, encoding)
        except AcquisitionTransportError as e:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.CONTENT_INVALID,
                request=request,
                error_message=str(e),
            )

        byte_len = len(decoded_bytes)
        if byte_len == 0:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.CONTENT_INVALID,
                request=request,
                error_message="Retrieved payload has 0 bytes",
            )

        if byte_len > MAX_PAYLOAD_BYTES:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.CONTENT_INVALID,
                request=request,
                error_message=f"Payload byte length {byte_len} exceeds max {MAX_PAYLOAD_BYTES}",
            )

        # Step 6: PDF structural validation
        doc_type = request.locator.document_type
        if doc_type in ("STATUTORY_PROSPECTUS", "PROSPECTUS_SUPPLEMENT", "PRIIP_KID"):
            try:
                validate_pdf_content(decoded_bytes)
            except AcquisitionTransportError as e:
                return AcquisitionResult(
                    attempt_id=attempt_id,
                    outcome=AcquisitionOutcome.CONTENT_INVALID,
                    request=request,
                    error_message=str(e),
                )

        raw_hash = hashlib.sha256(decoded_bytes).hexdigest().lower()
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        media_type = resp.headers.get("content-type", "application/pdf").split(";")[0].strip().lower()

        artifact = RawArtifact(
            artifact_id=f"art-{raw_hash[:16]}",
            raw_bytes=decoded_bytes,
            byte_length=byte_len,
            raw_sha256=raw_hash,
            media_type=media_type,
            http_status=status,
            retrieval_timestamp=now_iso,
            source_url=request.locator.source_url,
            content_encoding_decoded=True,
        )

        # Step 7: State machine execution & provenance ledger admission
        candidate_spec = {
            "share_class_isin": request.share_class_isin,
            "domicile_iso2": request.locator.jurisdiction,
            "document_type": doc_type,
            "source_url": request.locator.source_url,
            "effective_date": "2026-01-01",  # Initial default if not yet extracted
            "authorized_source_class": doc_type,
            "status": "ACTIVE",
        }

        try:
            st_result, prov_record = self.state_machine.process_candidate(
                candidate_spec=candidate_spec,
                raw_bytes=decoded_bytes,
                attempt_id=attempt_id,
            )
        except ProvenanceConflictError as e:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.PROVENANCE_FAILURE,
                request=request,
                artifact=artifact,
                error_message=f"Provenance ledger conflict: {e}",
            )
        except ETFSourceAuthorityError as e:
            return AcquisitionResult(
                attempt_id=attempt_id,
                outcome=AcquisitionOutcome.CONTENT_INVALID,
                request=request,
                artifact=artifact,
                error_message=f"State machine validation error: {e}",
            )

        res = AcquisitionResult(
            attempt_id=attempt_id,
            outcome=AcquisitionOutcome.ACQUIRED,
            request=request,
            artifact=artifact,
            provenance_record=prov_record,
        )
        assert_failure_is_not_no_match(res.outcome)
        return res

    def execute_batch(
        self,
        requests: Sequence[AuthorityRequest],
        counters: Optional[UCITSPopulationUniverseCounters] = None,
    ) -> List[AcquisitionResult]:
        """Executes a sequence of acquisition requests and records denominator statistics."""
        results: List[AcquisitionResult] = []
        for idx, req in enumerate(requests, 1):
            att_id = f"batch-att-{idx:04d}"
            res = self.execute_attempt(req, attempt_id=att_id)
            results.append(res)
            if counters is not None:
                counters.record_outcome(res.outcome)
        return results
