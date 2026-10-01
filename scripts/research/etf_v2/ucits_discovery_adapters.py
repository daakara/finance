"""
scripts/research/etf_v2/ucits_discovery_adapters.py

Standardized registry adapters and mock transport interfaces for European
National Competent Authorities (CBI, CSSF, BaFin, AMF) and statutory issuers.
Strictly encapsulates source-specific parsing and request semantics.
Supports offline deterministic execution via injectable transports.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from .global_identifier_authority import normalize_isin, validate_isin
from .ucits_discovery_models import (
    DiscoveryJurisdiction,
    QuarantineReason,
    RawDiscoveryObservation,
    RawRegisterPayload,
    SchemaDriftError,
    SourceAdapterError,
    SourceAuthorityId,
    SourceAuthorityTier,
    SourceEnumerationState,
)


# =============================================================================
# Transport Abstractions
# =============================================================================

class BaseDiscoveryTransport(ABC):
    """Abstract transport layer allowing zero-network testing and mock injection."""

    @abstractmethod
    def fetch(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
    ) -> Tuple[int, bytes, Dict[str, str]]:
        """
        Executes HTTP/transport request.
        Returns (status_code, response_bytes, response_headers).
        """
        pass


class MockDiscoveryTransport(BaseDiscoveryTransport):
    """
    Offline deterministic mock transport with programmable responses,
    rate-limit simulation, error injection, and call tracking.
    """

    def __init__(self) -> None:
        self._endpoints: Dict[str, Tuple[int, bytes, Dict[str, str]]] = {}
        self._endpoint_call_counts: Dict[str, int] = {}
        self._interception_callbacks: Dict[str, Callable[[], Tuple[int, bytes, Dict[str, str]]]] = {}
        self.call_history: List[str] = []

    def register_response(
        self,
        url: str,
        status_code: int = 200,
        content: bytes = b"",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        norm_headers = headers or {"content-type": "application/json"}
        self._endpoints[url] = (status_code, content, norm_headers)

    def register_callback(
        self,
        url: str,
        callback: Callable[[], Tuple[int, bytes, Dict[str, str]]],
    ) -> None:
        self._interception_callbacks[url] = callback

    def get_call_count(self, url: str) -> int:
        return self._endpoint_call_counts.get(url, 0)

    def fetch(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
    ) -> Tuple[int, bytes, Dict[str, str]]:
        self.call_history.append(url)
        self._endpoint_call_counts[url] = self._endpoint_call_counts.get(url, 0) + 1

        if url in self._interception_callbacks:
            return self._interception_callbacks[url]()

        if url in self._endpoints:
            return self._endpoints[url]

        # Fail closed for unregistered endpoints to enforce isolation
        return (404, b"Not Found (Mock Transport Isolation)", {"content-type": "text/plain"})


# =============================================================================
# Base Discovery Adapter
# =============================================================================

class BaseDiscoveryAdapter(ABC):
    """Common contract for all European registry harvesting adapters."""

    @property
    @abstractmethod
    def source_authority(self) -> SourceAuthorityId:
        """Specific regulatory or statutory authority ID."""
        pass

    @property
    @abstractmethod
    def source_tier(self) -> SourceAuthorityTier:
        """Authority tier (Tier 1 NCA, Tier 2 Issuer, Tier 3 Exchange)."""
        pass

    @property
    @abstractmethod
    def jurisdiction(self) -> DiscoveryJurisdiction:
        """Target fund domicile."""
        pass

    @abstractmethod
    def fetch_raw_register(
        self,
        run_context: Any,
        transport: BaseDiscoveryTransport,
    ) -> List[RawRegisterPayload]:
        """Fetches raw register bytes and returns raw payload structures."""
        pass

    @abstractmethod
    def parse_observations(
        self,
        payloads: List[RawRegisterPayload],
    ) -> List[RawDiscoveryObservation]:
        """Parses raw register payloads into normalized observations without loss of fidelity."""
        pass

    @abstractmethod
    def verify_completeness(
        self,
        payloads: List[RawRegisterPayload],
        observations: List[RawDiscoveryObservation],
    ) -> Tuple[SourceEnumerationState, Optional[str]]:
        """Verifies full registry enumeration against source headers or checksums."""
        pass


# =============================================================================
# Central Bank of Ireland (CBI) Adapter
# =============================================================================

class CentralBankOfIrelandAdapter(BaseDiscoveryAdapter):
    """
    Tier 1 NCA Adapter for Ireland (CBI).
    Harvests register of Collective Investment Schemes authorized under UCITS Regulations.
    """

    @property
    def source_authority(self) -> SourceAuthorityId:
        return SourceAuthorityId.CENTRAL_BANK_OF_IRELAND

    @property
    def source_tier(self) -> SourceAuthorityTier:
        return SourceAuthorityTier.TIER_1_NCA

    @property
    def jurisdiction(self) -> DiscoveryJurisdiction:
        return DiscoveryJurisdiction.IE

    def fetch_raw_register(
        self,
        run_context: Any,
        transport: BaseDiscoveryTransport,
    ) -> List[RawRegisterPayload]:
        url = getattr(run_context, "cbi_register_url", "https://registers.centralbank.ie/cis/ucits_etfs.json")
        status, content, hdrs = transport.fetch(url)
        if status != 200:
            raise SourceAdapterError(f"CBI register fetch failed with HTTP {status}")

        sha = hashlib.sha256(content).hexdigest()
        retrieved_at = getattr(run_context, "current_timestamp", "2026-10-01T08:00:00Z")

        # Extract declared header count if available
        declared_count = None
        try:
            doc = json.loads(content.decode("utf-8"))
            if isinstance(doc, dict):
                declared_count = doc.get("total_records")
        except Exception:
            pass

        return [
            RawRegisterPayload(
                source_authority=self.source_authority.value,
                jurisdiction=self.jurisdiction.value,
                request_uri=url,
                response_status=status,
                content_type=hdrs.get("content-type", "application/json"),
                raw_bytes=content,
                raw_sha256=sha,
                retrieved_at=retrieved_at,
                header_declared_count=declared_count,
            )
        ]

    def parse_observations(
        self,
        payloads: List[RawRegisterPayload],
    ) -> List[RawDiscoveryObservation]:
        observations: List[RawDiscoveryObservation] = []
        for payload in payloads:
            try:
                data = json.loads(payload.raw_bytes.decode("utf-8"))
            except Exception as e:
                raise SchemaDriftError(f"CBI JSON decode failure: {e}")

            records = data.get("records", []) if isinstance(data, dict) else data
            if not isinstance(records, list):
                raise SchemaDriftError("CBI records root must be a list")

            for idx, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                # Expected fields: isin, fund_name, sub_fund_name, share_class, cis_type, is_etf, status
                raw_id = str(rec.get("isin", "")).strip()
                fund_name = str(rec.get("fund_name", "")).strip()
                share_class = str(rec.get("share_class_name", rec.get("share_class", ""))).strip()
                cis_type = str(rec.get("cis_type", "")).strip().upper()
                is_etf = bool(rec.get("is_etf", False) or "ETF" in fund_name.upper() or "ETF" in share_class.upper())
                is_ucits = (cis_type == "UCITS" or "UCITS" in str(rec.get("legal_framework", "")).upper())
                status = str(rec.get("status", "ACTIVE")).strip().upper()

                obs_id = f"cbi_obs_{payload.raw_sha256[:8]}_{idx:05d}"
                observations.append(
                    RawDiscoveryObservation(
                        observation_id=obs_id,
                        source_authority=self.source_authority.value,
                        source_authority_tier=self.source_tier.value,
                        retrieved_at=payload.retrieved_at,
                        source_as_of=data.get("as_of", payload.retrieved_at) if isinstance(data, dict) else payload.retrieved_at,
                        raw_identifier=raw_id,
                        normalized_isin=raw_id.upper(),
                        fund_name_raw=fund_name,
                        share_class_name_raw=share_class,
                        domicile_raw="IE",
                        is_ucits_raw=is_ucits,
                        is_etf_raw=is_etf,
                        listing_status_raw=status,
                        source_record_uri=f"{payload.request_uri}#record_{idx}",
                        source_payload_sha256=payload.raw_sha256,
                        raw_attributes=rec,
                    )
                )
        return observations

    def verify_completeness(
        self,
        payloads: List[RawRegisterPayload],
        observations: List[RawDiscoveryObservation],
    ) -> Tuple[SourceEnumerationState, Optional[str]]:
        if not payloads:
            return SourceEnumerationState.FAILED, "No payloads retrieved for CBI"
        payload = payloads[0]
        if payload.header_declared_count is not None:
            if len(observations) != payload.header_declared_count:
                return (
                    SourceEnumerationState.PARTIAL,
                    f"CBI observation count ({len(observations)}) != declared header count ({payload.header_declared_count})",
                )
        return SourceEnumerationState.COMPLETE, None


# =============================================================================
# CSSF Luxembourg Adapter
# =============================================================================

class CSSFLuxembourgAdapter(BaseDiscoveryAdapter):
    """
    Tier 1 NCA Adapter for Luxembourg (CSSF).
    Harvests register of UCITS Part I authorized sub-funds and ETF share classes.
    """

    @property
    def source_authority(self) -> SourceAuthorityId:
        return SourceAuthorityId.CSSF_LUXEMBOURG

    @property
    def source_tier(self) -> SourceAuthorityTier:
        return SourceAuthorityTier.TIER_1_NCA

    @property
    def jurisdiction(self) -> DiscoveryJurisdiction:
        return DiscoveryJurisdiction.LU

    def fetch_raw_register(
        self,
        run_context: Any,
        transport: BaseDiscoveryTransport,
    ) -> List[RawRegisterPayload]:
        url = getattr(run_context, "cssf_register_url", "https://registers.cssf.lu/api/v1/ucits_etfs.json")
        status, content, hdrs = transport.fetch(url)
        if status != 200:
            raise SourceAdapterError(f"CSSF register fetch failed with HTTP {status}")

        sha = hashlib.sha256(content).hexdigest()
        retrieved_at = getattr(run_context, "current_timestamp", "2026-10-01T08:00:00Z")

        declared_count = None
        try:
            doc = json.loads(content.decode("utf-8"))
            if isinstance(doc, dict):
                declared_count = doc.get("total_records")
        except Exception:
            pass

        return [
            RawRegisterPayload(
                source_authority=self.source_authority.value,
                jurisdiction=self.jurisdiction.value,
                request_uri=url,
                response_status=status,
                content_type=hdrs.get("content-type", "application/json"),
                raw_bytes=content,
                raw_sha256=sha,
                retrieved_at=retrieved_at,
                header_declared_count=declared_count,
            )
        ]

    def parse_observations(
        self,
        payloads: List[RawRegisterPayload],
    ) -> List[RawDiscoveryObservation]:
        observations: List[RawDiscoveryObservation] = []
        for payload in payloads:
            try:
                data = json.loads(payload.raw_bytes.decode("utf-8"))
            except Exception as e:
                raise SchemaDriftError(f"CSSF JSON decode failure: {e}")

            records = data.get("records", []) if isinstance(data, dict) else data
            if not isinstance(records, list):
                raise SchemaDriftError("CSSF records root must be a list")

            for idx, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                raw_id = str(rec.get("isin", "")).strip()
                fund_name = str(rec.get("fund_name", "")).strip()
                share_class = str(rec.get("share_class_name", "")).strip()
                law_part = str(rec.get("law_part", "PART_I")).strip().upper()
                is_ucits = (law_part == "PART_I" or "UCITS" in str(rec.get("regime", "")).upper())
                is_etf = bool(rec.get("is_etf", False) or "ETF" in fund_name.upper() or "ETF" in share_class.upper())
                status = str(rec.get("status", "ACTIVE")).strip().upper()

                obs_id = f"cssf_obs_{payload.raw_sha256[:8]}_{idx:05d}"
                observations.append(
                    RawDiscoveryObservation(
                        observation_id=obs_id,
                        source_authority=self.source_authority.value,
                        source_authority_tier=self.source_tier.value,
                        retrieved_at=payload.retrieved_at,
                        source_as_of=data.get("as_of", payload.retrieved_at) if isinstance(data, dict) else payload.retrieved_at,
                        raw_identifier=raw_id,
                        normalized_isin=raw_id.upper(),
                        fund_name_raw=fund_name,
                        share_class_name_raw=share_class,
                        domicile_raw="LU",
                        is_ucits_raw=is_ucits,
                        is_etf_raw=is_etf,
                        listing_status_raw=status,
                        source_record_uri=f"{payload.request_uri}#record_{idx}",
                        source_payload_sha256=payload.raw_sha256,
                        raw_attributes=rec,
                    )
                )
        return observations

    def verify_completeness(
        self,
        payloads: List[RawRegisterPayload],
        observations: List[RawDiscoveryObservation],
    ) -> Tuple[SourceEnumerationState, Optional[str]]:
        if not payloads:
            return SourceEnumerationState.FAILED, "No payloads retrieved for CSSF"
        payload = payloads[0]
        if payload.header_declared_count is not None:
            if len(observations) != payload.header_declared_count:
                return (
                    SourceEnumerationState.PARTIAL,
                    f"CSSF count ({len(observations)}) != header count ({payload.header_declared_count})",
                )
        return SourceEnumerationState.COMPLETE, None


# =============================================================================
# BaFin Germany Adapter
# =============================================================================

class BaFinGermanyAdapter(BaseDiscoveryAdapter):
    """
    Tier 1 NCA Adapter for Germany (BaFin).
    Harvests register of KAGB-authorized domestic and passported UCITS ETFs.
    """

    @property
    def source_authority(self) -> SourceAuthorityId:
        return SourceAuthorityId.BAFIN_GERMANY

    @property
    def source_tier(self) -> SourceAuthorityTier:
        return SourceAuthorityTier.TIER_1_NCA

    @property
    def jurisdiction(self) -> DiscoveryJurisdiction:
        return DiscoveryJurisdiction.DE

    def fetch_raw_register(
        self,
        run_context: Any,
        transport: BaseDiscoveryTransport,
    ) -> List[RawRegisterPayload]:
        url = getattr(run_context, "bafin_register_url", "https://portal.mvp.bafin.de/database/fonds/ucits_etfs.csv")
        status, content, hdrs = transport.fetch(url)
        if status != 200:
            raise SourceAdapterError(f"BaFin register fetch failed with HTTP {status}")

        sha = hashlib.sha256(content).hexdigest()
        retrieved_at = getattr(run_context, "current_timestamp", "2026-10-01T08:00:00Z")

        return [
            RawRegisterPayload(
                source_authority=self.source_authority.value,
                jurisdiction=self.jurisdiction.value,
                request_uri=url,
                response_status=status,
                content_type=hdrs.get("content-type", "text/csv"),
                raw_bytes=content,
                raw_sha256=sha,
                retrieved_at=retrieved_at,
            )
        ]

    def parse_observations(
        self,
        payloads: List[RawRegisterPayload],
    ) -> List[RawDiscoveryObservation]:
        observations: List[RawDiscoveryObservation] = []
        for payload in payloads:
            try:
                text = payload.raw_bytes.decode("utf-8", errors="replace")
                reader = csv.DictReader(io.StringIO(text), delimiter=";")
                rows = list(reader)
            except Exception as e:
                raise SchemaDriftError(f"BaFin CSV parse failure: {e}")

            for idx, row in enumerate(rows):
                raw_id = str(row.get("ISIN", row.get("isin", ""))).strip()
                fund_name = str(row.get("Fondsname", row.get("fund_name", ""))).strip()
                share_class = str(row.get("Anteilklasse", row.get("share_class", ""))).strip()
                legal_form = str(row.get("Rechtsform", row.get("legal_form", ""))).strip().upper()
                is_ucits = ("OGAW" in legal_form or "UCITS" in legal_form)
                is_etf = bool("ETF" in fund_name.upper() or "ETF" in share_class.upper())
                status = "ACTIVE" if "LIQ" not in row.get("Status", "").upper() else "TERMINATED"

                obs_id = f"bafin_obs_{payload.raw_sha256[:8]}_{idx:05d}"
                observations.append(
                    RawDiscoveryObservation(
                        observation_id=obs_id,
                        source_authority=self.source_authority.value,
                        source_authority_tier=self.source_tier.value,
                        retrieved_at=payload.retrieved_at,
                        source_as_of=payload.retrieved_at,
                        raw_identifier=raw_id,
                        normalized_isin=raw_id.upper(),
                        fund_name_raw=fund_name,
                        share_class_name_raw=share_class,
                        domicile_raw="DE",
                        is_ucits_raw=is_ucits,
                        is_etf_raw=is_etf,
                        listing_status_raw=status,
                        source_record_uri=f"{payload.request_uri}#row_{idx}",
                        source_payload_sha256=payload.raw_sha256,
                        raw_attributes=dict(row),
                    )
                )
        return observations

    def verify_completeness(
        self,
        payloads: List[RawRegisterPayload],
        observations: List[RawDiscoveryObservation],
    ) -> Tuple[SourceEnumerationState, Optional[str]]:
        if not payloads:
            return SourceEnumerationState.FAILED, "No payloads retrieved for BaFin"
        return SourceEnumerationState.COMPLETE, None


# =============================================================================
# AMF France Adapter
# =============================================================================

class AMFFranceAdapter(BaseDiscoveryAdapter):
    """
    Tier 1 NCA Adapter for France (AMF).
    Harvests register of French UCITS ETFs (FCP/SICAV).
    """

    @property
    def source_authority(self) -> SourceAuthorityId:
        return SourceAuthorityId.AMF_FRANCE

    @property
    def source_tier(self) -> SourceAuthorityTier:
        return SourceAuthorityTier.TIER_1_NCA

    @property
    def jurisdiction(self) -> DiscoveryJurisdiction:
        return DiscoveryJurisdiction.FR

    def fetch_raw_register(
        self,
        run_context: Any,
        transport: BaseDiscoveryTransport,
    ) -> List[RawRegisterPayload]:
        url = getattr(run_context, "amf_register_url", "https://geco.amf-france.org/api/funds/ucits_etfs.json")
        status, content, hdrs = transport.fetch(url)
        if status != 200:
            raise SourceAdapterError(f"AMF register fetch failed with HTTP {status}")

        sha = hashlib.sha256(content).hexdigest()
        retrieved_at = getattr(run_context, "current_timestamp", "2026-10-01T08:00:00Z")

        declared_count = None
        try:
            doc = json.loads(content.decode("utf-8"))
            if isinstance(doc, dict):
                declared_count = doc.get("total_records")
        except Exception:
            pass

        return [
            RawRegisterPayload(
                source_authority=self.source_authority.value,
                jurisdiction=self.jurisdiction.value,
                request_uri=url,
                response_status=status,
                content_type=hdrs.get("content-type", "application/json"),
                raw_bytes=content,
                raw_sha256=sha,
                retrieved_at=retrieved_at,
                header_declared_count=declared_count,
            )
        ]

    def parse_observations(
        self,
        payloads: List[RawRegisterPayload],
    ) -> List[RawDiscoveryObservation]:
        observations: List[RawDiscoveryObservation] = []
        for payload in payloads:
            try:
                data = json.loads(payload.raw_bytes.decode("utf-8"))
            except Exception as e:
                raise SchemaDriftError(f"AMF JSON decode failure: {e}")

            records = data.get("records", []) if isinstance(data, dict) else data
            if not isinstance(records, list):
                raise SchemaDriftError("AMF records root must be a list")

            for idx, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                raw_id = str(rec.get("isin", "")).strip()
                fund_name = str(rec.get("fund_name", "")).strip()
                share_class = str(rec.get("share_class_name", "")).strip()
                form = str(rec.get("legal_form", "")).strip().upper()
                is_ucits = (form in ("SICAV", "FCP", "UCITS") or bool(rec.get("is_ucits", True)))
                is_etf = bool(rec.get("is_etf", False) or "ETF" in fund_name.upper() or "ETF" in share_class.upper())
                status = str(rec.get("status", "ACTIVE")).strip().upper()

                obs_id = f"amf_obs_{payload.raw_sha256[:8]}_{idx:05d}"
                observations.append(
                    RawDiscoveryObservation(
                        observation_id=obs_id,
                        source_authority=self.source_authority.value,
                        source_authority_tier=self.source_tier.value,
                        retrieved_at=payload.retrieved_at,
                        source_as_of=data.get("as_of", payload.retrieved_at) if isinstance(data, dict) else payload.retrieved_at,
                        raw_identifier=raw_id,
                        normalized_isin=raw_id.upper(),
                        fund_name_raw=fund_name,
                        share_class_name_raw=share_class,
                        domicile_raw="FR",
                        is_ucits_raw=is_ucits,
                        is_etf_raw=is_etf,
                        listing_status_raw=status,
                        source_record_uri=f"{payload.request_uri}#record_{idx}",
                        source_payload_sha256=payload.raw_sha256,
                        raw_attributes=rec,
                    )
                )
        return observations

    def verify_completeness(
        self,
        payloads: List[RawRegisterPayload],
        observations: List[RawDiscoveryObservation],
    ) -> Tuple[SourceEnumerationState, Optional[str]]:
        if not payloads:
            return SourceEnumerationState.FAILED, "No payloads retrieved for AMF"
        payload = payloads[0]
        if payload.header_declared_count is not None:
            if len(observations) != payload.header_declared_count:
                return (
                    SourceEnumerationState.PARTIAL,
                    f"AMF count ({len(observations)}) != header count ({payload.header_declared_count})",
                )
        return SourceEnumerationState.COMPLETE, None


# =============================================================================
# Statutory Issuer Adapter (Tier 2 Corroboration)
# =============================================================================

class StatutoryIssuerAdapter(BaseDiscoveryAdapter):
    """
    Tier 2 Statutory Issuer Adapter.
    Harvests official issuer listings (e.g. iShares, Vanguard, Xtrackers).
    Secondary corroboration only; cannot unilaterally expand canonical denominator.
    """

    def __init__(self, issuer_name: str = "GENERIC_STATUTORY_ISSUER", jurisdiction: DiscoveryJurisdiction = DiscoveryJurisdiction.IE) -> None:
        self._issuer_name = issuer_name
        self._jurisdiction = jurisdiction

    @property
    def source_authority(self) -> SourceAuthorityId:
        return SourceAuthorityId.STATUTORY_ISSUER

    @property
    def source_tier(self) -> SourceAuthorityTier:
        return SourceAuthorityTier.TIER_2_STATUTORY_ISSUER

    @property
    def jurisdiction(self) -> DiscoveryJurisdiction:
        return self._jurisdiction

    def fetch_raw_register(
        self,
        run_context: Any,
        transport: BaseDiscoveryTransport,
    ) -> List[RawRegisterPayload]:
        url = getattr(run_context, "issuer_register_url", f"https://www.{self._issuer_name.lower()}.com/products.json")
        status, content, hdrs = transport.fetch(url)
        if status != 200:
            raise SourceAdapterError(f"Issuer register fetch failed with HTTP {status}")

        sha = hashlib.sha256(content).hexdigest()
        retrieved_at = getattr(run_context, "current_timestamp", "2026-10-01T08:00:00Z")

        return [
            RawRegisterPayload(
                source_authority=f"{self.source_authority.value}:{self._issuer_name}",
                jurisdiction=self.jurisdiction.value,
                request_uri=url,
                response_status=status,
                content_type=hdrs.get("content-type", "application/json"),
                raw_bytes=content,
                raw_sha256=sha,
                retrieved_at=retrieved_at,
            )
        ]

    def parse_observations(
        self,
        payloads: List[RawRegisterPayload],
    ) -> List[RawDiscoveryObservation]:
        observations: List[RawDiscoveryObservation] = []
        for payload in payloads:
            try:
                data = json.loads(payload.raw_bytes.decode("utf-8"))
            except Exception as e:
                raise SchemaDriftError(f"Issuer JSON decode failure: {e}")

            records = data.get("products", data.get("records", [])) if isinstance(data, dict) else data
            for idx, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                raw_id = str(rec.get("isin", "")).strip()
                fund_name = str(rec.get("fund_name", "")).strip()
                share_class = str(rec.get("share_class_name", "")).strip()
                is_ucits = bool(rec.get("is_ucits", True))
                is_etf = bool(rec.get("is_etf", True))
                status = str(rec.get("status", "ACTIVE")).strip().upper()

                obs_id = f"issuer_obs_{payload.raw_sha256[:8]}_{idx:05d}"
                observations.append(
                    RawDiscoveryObservation(
                        observation_id=obs_id,
                        source_authority=payload.source_authority,
                        source_authority_tier=self.source_tier.value,
                        retrieved_at=payload.retrieved_at,
                        source_as_of=payload.retrieved_at,
                        raw_identifier=raw_id,
                        normalized_isin=raw_id.upper(),
                        fund_name_raw=fund_name,
                        share_class_name_raw=share_class,
                        domicile_raw=self.jurisdiction.value,
                        is_ucits_raw=is_ucits,
                        is_etf_raw=is_etf,
                        listing_status_raw=status,
                        source_record_uri=f"{payload.request_uri}#product_{idx}",
                        source_payload_sha256=payload.raw_sha256,
                        raw_attributes=rec,
                    )
                )
        return observations

    def verify_completeness(
        self,
        payloads: List[RawRegisterPayload],
        observations: List[RawDiscoveryObservation],
    ) -> Tuple[SourceEnumerationState, Optional[str]]:
        if not payloads:
            return SourceEnumerationState.FAILED, "No payloads retrieved for Issuer"
        return SourceEnumerationState.COMPLETE, None
