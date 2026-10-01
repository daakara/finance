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
import zipfile
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from .global_identifier_authority import normalize_isin, validate_isin
from .ucits_discovery_models import (
    AuthorityFunction,
    DiscoveryJurisdiction,
    ParentJoinStatus,
    QuarantineReason,
    RawDiscoveryObservation,
    RawRegisterPayload,
    SchemaDriftError,
    ShareClassExpansionCompleteness,
    SourceAdapterError,
    SourceAuthorityId,
    SourceAuthorityTier,
    SourceEnumerationState,
    Tier1ParentAccounting,
    Tier1ParentObservation,
    Tier2ShareClassExpansion,
    normalize_fund_name,
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
            raw_stripped = payload.raw_bytes.strip()
            if raw_stripped.startswith(b"<!DOCTYPE") or b"<html" in raw_stripped[:500].lower():
                raise SchemaDriftError("CBI source returned HTML landing page instead of structured register JSON (automated machine-readable API unavailable)")
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

    def parse_parent_observations(
        self,
        payloads: List[RawRegisterPayload],
    ) -> List[Tier1ParentObservation]:
        """
        Parses CBI register payloads into authoritative Tier-1 Umbrella / Sub-Fund parent observations.
        Strictly zero share-class ISIN fabrication: does not emit or guess ISINs.
        """
        parents: List[Tier1ParentObservation] = []
        for payload in payloads:
            raw_stripped = payload.raw_bytes.strip()
            if raw_stripped.startswith(b"<!DOCTYPE") or b"<html" in raw_stripped[:500].lower():
                raise SchemaDriftError("CBI source returned HTML landing page instead of structured register")
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
                # Only parse as parent observation if structured as umbrella/subfund parent
                if not ("sub_fund_name" in rec or "umbrella_name" in rec or bool(rec.get("is_parent")) or rec.get("entity_type") == "SUB_FUND"):
                    continue
                umbrella_name = str(rec.get("umbrella_name", rec.get("fund_name", ""))).strip()
                subfund_name = str(rec.get("sub_fund_name", rec.get("fund_name", ""))).strip()
                cis_type = str(rec.get("cis_type", "")).strip().upper()
                is_etf = bool(rec.get("is_etf", False) or "ETF" in umbrella_name.upper() or "ETF" in subfund_name.upper())
                is_ucits = (cis_type == "UCITS" or "UCITS" in str(rec.get("legal_framework", "")).upper() or bool(rec.get("is_ucits", True)))
                status = str(rec.get("status", "ACTIVE")).strip().upper()

                parent_id = f"cbi_parent_{payload.raw_sha256[:8]}_{idx:05d}"
                parents.append(
                    Tier1ParentObservation(
                        parent_id=parent_id,
                        source_authority=self.source_authority.value,
                        source_authority_tier=self.source_tier.value,
                        jurisdiction="IE",
                        umbrella_name=umbrella_name,
                        subfund_name=subfund_name,
                        is_ucits=is_ucits,
                        is_etf=is_etf,
                        status=status,
                        retrieved_at=payload.retrieved_at,
                        source_as_of=data.get("as_of", payload.retrieved_at) if isinstance(data, dict) else payload.retrieved_at,
                        source_record_uri=f"{payload.request_uri}#subfund_{idx}",
                        source_payload_sha256=payload.raw_sha256,
                        authorization_date=rec.get("authorization_date"),
                        termination_date=rec.get("termination_date"),
                        raw_attributes=rec,
                    )
                )
        return parents

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
        url = getattr(run_context, "cssf_register_url", "https://www.cssf.lu/wp-content/uploads/OPC_COMP_TP_TOUS_OUVERTS.zip")
        status, content, hdrs = transport.fetch(url)
        if status != 200:
            raise SourceAdapterError(f"CSSF register fetch failed with HTTP {status}")

        sha = hashlib.sha256(content).hexdigest()
        retrieved_at = getattr(run_context, "current_timestamp", "2026-10-01T08:00:00Z")

        declared_count = None
        content_type = hdrs.get("content-type", "application/zip" if content.startswith(b"PK\x03\x04") else "application/json")
        try:
            if not content.startswith(b"PK\x03\x04"):
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
                content_type=content_type,
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
            raw_bytes = payload.raw_bytes
            raw_stripped = raw_bytes.strip()

            if raw_stripped.startswith(b"<!DOCTYPE") or b"<html" in raw_stripped[:500].lower():
                raise SchemaDriftError("CSSF source returned HTML page instead of structured register archive or JSON")

            # 1. Official Bulk ZIP format
            if raw_bytes.startswith(b"PK\x03\x04") or "zip" in payload.content_type.lower():
                try:
                    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
                        if not csv_names:
                            raise SchemaDriftError("CSSF ZIP does not contain any CSV files")
                        csv_content = zf.read(csv_names[0])
                        try:
                            csv_text = csv_content.decode("utf-16")
                        except UnicodeDecodeError:
                            csv_text = csv_content.decode("utf-8", errors="replace")
                except Exception as e:
                    if isinstance(e, SchemaDriftError):
                        raise
                    raise SchemaDriftError(f"CSSF ZIP extraction failure: {e}")

                lines = csv_text.splitlines()
                if not lines:
                    return []

                reader = csv.reader(lines, delimiter="\t")
                rows = list(reader)
                if not rows:
                    return []

                header_row = [c.strip().upper() for c in rows[0]]
                isin_col = 2
                nomopc_col = 3
                nomcomp_col = 5
                agrcomp_col = 6
                nomtype_col = 9
                for c_idx, h in enumerate(header_row):
                    if "ISIN" in h:
                        isin_col = c_idx
                    elif h == "NOMOPC":
                        nomopc_col = c_idx
                    elif h == "NOMCOMPARTIMENT":
                        nomcomp_col = c_idx
                    elif "AGREEMENT" in h:
                        agrcomp_col = c_idx
                    elif "NOMTYPEPART" in h:
                        nomtype_col = c_idx

                record_idx = 0
                for r in rows[1:]:
                    if not r or not any(r):
                        continue
                    if r[0].strip().startswith("-"):
                        continue
                    if len(r) <= max(isin_col, nomopc_col):
                        continue
                    raw_id = r[isin_col].strip() if len(r) > isin_col else ""
                    if not raw_id:
                        continue
                    fund_name = r[nomopc_col].strip() if len(r) > nomopc_col else ""
                    comp_name = r[nomcomp_col].strip() if len(r) > nomcomp_col else ""
                    share_class = r[nomtype_col].strip() if len(r) > nomtype_col else ""
                    as_of = r[agrcomp_col].strip() if len(r) > agrcomp_col else payload.retrieved_at

                    # In official CSSF register "Identifiants des OPCVM", all listed open funds are UCITS (Part I)
                    is_ucits = True
                    is_etf = bool("ETF" in comp_name.upper() or "ETF" in share_class.upper() or "ETF" in fund_name.upper())
                    status = "ACTIVE"

                    obs_id = f"cssf_obs_{payload.raw_sha256[:8]}_{record_idx:05d}"
                    observations.append(
                        RawDiscoveryObservation(
                            observation_id=obs_id,
                            source_authority=self.source_authority.value,
                            source_authority_tier=self.source_tier.value,
                            retrieved_at=payload.retrieved_at,
                            source_as_of=as_of,
                            raw_identifier=raw_id,
                            normalized_isin=raw_id.upper(),
                            fund_name_raw=comp_name or fund_name,
                            share_class_name_raw=share_class,
                            domicile_raw="LU",
                            is_ucits_raw=is_ucits,
                            is_etf_raw=is_etf,
                            listing_status_raw=status,
                            source_record_uri=f"{payload.request_uri}#record_{record_idx}",
                            source_payload_sha256=payload.raw_sha256,
                            raw_attributes={
                                "isin": raw_id,
                                "fund_name": fund_name,
                                "compartment_name": comp_name,
                                "share_class_name": share_class,
                                "authorization_date": as_of,
                            },
                        )
                    )
                    record_idx += 1

            # 2. Backward-compatible JSON format (for synthetic mock tests)
            else:
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
        default_url = "https://portal.mvp.bafin.de/database/FondsInfo/sucheFonds.do?nameFondsISIN=&nameFonds=&d-16544-e=1&nameFondsButton=Suche&nameFondsId=&6578706f7274=1&filterParagraph=%27OGAW%27%2C%27OOAGA%27%2C%27OGAWA%27"
        url = getattr(run_context, "bafin_register_url", default_url)
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
            raw_bytes = payload.raw_bytes
            raw_stripped = raw_bytes.strip()
            if raw_stripped.startswith(b"<!DOCTYPE") or b"<html" in raw_stripped[:500].lower():
                raise SchemaDriftError("BaFin source returned HTML page instead of structured CSV register")

            try:
                try:
                    text = raw_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = raw_bytes.decode("iso-8859-1", errors="replace")

                first_line = text.splitlines()[0] if text.splitlines() else ""
                delimiter = ";" if ";" in first_line else ","
                reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
                rows = list(reader)
            except Exception as e:
                raise SchemaDriftError(f"BaFin CSV parse failure: {e}")

            for idx, row in enumerate(rows):
                raw_id = str(row.get("ISIN", row.get("isin", ""))).strip()
                fund_name = str(row.get("Fonds", row.get("Fondsname", row.get("fund_name", "")))).strip()
                share_class = str(row.get("Anteilklasse", row.get("share_class", ""))).strip()
                legal_form = str(row.get("Fondsart", row.get("Rechtsform", row.get("legal_form", "")))).strip().upper()
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
        default_url = "https://geco.amf-france.org/back-office/funds/compartments"
        url = getattr(run_context, "amf_register_url", default_url)
        status, content, hdrs = transport.fetch(url)
        if status != 200:
            raise SourceAdapterError(f"AMF register fetch failed with HTTP {status}")

        sha = hashlib.sha256(content).hexdigest()
        retrieved_at = getattr(run_context, "current_timestamp", "2026-10-01T08:00:00Z")

        declared_count = None
        try:
            doc = json.loads(content.decode("utf-8"))
            if isinstance(doc, dict):
                declared_count = doc.get("total", doc.get("total_records"))
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
            raw_bytes = payload.raw_bytes
            raw_stripped = raw_bytes.strip()
            if raw_stripped.startswith(b"<!DOCTYPE") or b"<html" in raw_stripped[:500].lower():
                raise SchemaDriftError("AMF source returned HTML portal page instead of structured register JSON")

            try:
                data = json.loads(raw_bytes.decode("utf-8"))
            except Exception as e:
                raise SchemaDriftError(f"AMF JSON decode failure: {e}")

            # 1. Official GECO REST payload (compartmentDtos)
            if isinstance(data, dict) and "compartmentDtos" in data:
                compartments = data.get("compartmentDtos", [])
                for idx, comp in enumerate(compartments):
                    if not isinstance(comp, dict):
                        continue
                    cmp_nom = str(comp.get("cmpNom", "")).strip()
                    fund_dto = comp.get("fundDTO", {}) if isinstance(comp.get("fundDTO"), dict) else {}
                    fund_name = str(fund_dto.get("prdNom", cmp_nom)).strip()
                    prd_faml = str(comp.get("prdFaml", fund_dto.get("prdFaml", ""))).strip().upper()
                    # In AMF, UCITS are OPCVM (or SICAV/FCP registered under UCITS)
                    is_ucits = (prd_faml == "OPCVM" or "UCITS" in str(comp).upper())
                    is_etf = bool("ETF" in cmp_nom.upper() or "ETF" in fund_name.upper())
                    status = "ACTIVE" if comp.get("cmpStatutCode") == "VIV" else "INACTIVE"

                    shares_isins = comp.get("sharesIsins", [])
                    if isinstance(shares_isins, list) and shares_isins:
                        for s_idx, isin_item in enumerate(shares_isins):
                            raw_id = str(isin_item).strip() if isinstance(isin_item, str) else str(isin_item.get("isin", "")).strip()
                            if not raw_id:
                                continue
                            obs_id = f"amf_obs_{payload.raw_sha256[:8]}_{idx:05d}_{s_idx:02d}"
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
                                    share_class_name_raw=cmp_nom,
                                    domicile_raw="FR",
                                    is_ucits_raw=is_ucits,
                                    is_etf_raw=is_etf,
                                    listing_status_raw=status,
                                    source_record_uri=f"{payload.request_uri}#comp_{idx}_share_{s_idx}",
                                    source_payload_sha256=payload.raw_sha256,
                                    raw_attributes=comp,
                                )
                            )
                    else:
                        cmp_code = str(comp.get("cmpCodeParPrincp", "")).strip()
                        if cmp_code and cmp_code != "FR0000000000":
                            obs_id = f"amf_obs_{payload.raw_sha256[:8]}_{idx:05d}"
                            observations.append(
                                RawDiscoveryObservation(
                                    observation_id=obs_id,
                                    source_authority=self.source_authority.value,
                                    source_authority_tier=self.source_tier.value,
                                    retrieved_at=payload.retrieved_at,
                                    source_as_of=payload.retrieved_at,
                                    raw_identifier=cmp_code,
                                    normalized_isin=cmp_code.upper(),
                                    fund_name_raw=fund_name,
                                    share_class_name_raw=cmp_nom,
                                    domicile_raw="FR",
                                    is_ucits_raw=is_ucits,
                                    is_etf_raw=is_etf,
                                    listing_status_raw=status,
                                    source_record_uri=f"{payload.request_uri}#comp_{idx}",
                                    source_payload_sha256=payload.raw_sha256,
                                    raw_attributes=comp,
                                )
                            )

            # 2. Existing mock JSON payload (records root)
            else:
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
                if "sub_fund_name" in rec or "umbrella_name" in rec or bool(rec.get("is_expansion")) or rec.get("entity_type") == "EXPANSION":
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

    def parse_expansions(
        self,
        payloads: List[RawRegisterPayload],
    ) -> List[Tier2ShareClassExpansion]:
        """
        Parses statutory issuer product schedules / prospectus extracts into Tier-2 share-class expansions.
        Supplies share-class name, ISO 6166 ISIN, and completeness for an authorized sub-fund.
        Can NEVER self-authorize as an independent population parent.
        """
        expansions: List[Tier2ShareClassExpansion] = []
        for payload in payloads:
            try:
                data = json.loads(payload.raw_bytes.decode("utf-8"))
            except Exception as e:
                raise SchemaDriftError(f"Issuer JSON decode failure: {e}")

            records = data.get("products", data.get("records", [])) if isinstance(data, dict) else data
            if not isinstance(records, list):
                continue

            for idx, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                if not ("sub_fund_name" in rec or "umbrella_name" in rec or bool(rec.get("is_expansion")) or rec.get("entity_type") == "EXPANSION"):
                    continue
                raw_id = str(rec.get("isin", "")).strip()
                umbrella_name = str(rec.get("umbrella_name", rec.get("fund_name", ""))).strip()
                subfund_name = str(rec.get("sub_fund_name", rec.get("fund_name", ""))).strip()
                share_class = str(rec.get("share_class_name", rec.get("share_class", ""))).strip()
                is_ucits = bool(rec.get("is_ucits", True))
                is_etf = bool(rec.get("is_etf", True))
                status = str(rec.get("status", "ACTIVE")).strip().upper()
                completeness = str(rec.get("completeness", ShareClassExpansionCompleteness.COMPLETE.value))

                exp_id = f"issuer_exp_{payload.raw_sha256[:8]}_{idx:05d}"
                expansions.append(
                    Tier2ShareClassExpansion(
                        expansion_id=exp_id,
                        source_authority=payload.source_authority,
                        source_authority_tier=self.source_tier.value,
                        jurisdiction=self.jurisdiction.value,
                        umbrella_name=umbrella_name,
                        subfund_name=subfund_name,
                        share_class_name=share_class,
                        share_class_isin=raw_id,
                        is_ucits=is_ucits,
                        is_etf=is_etf,
                        status=status,
                        completeness=completeness,
                        retrieved_at=payload.retrieved_at,
                        source_as_of=payload.retrieved_at,
                        source_record_uri=f"{payload.request_uri}#product_{idx}",
                        source_payload_sha256=payload.raw_sha256,
                        raw_attributes=rec,
                    )
                )
        return expansions

    def verify_completeness(
        self,
        payloads: List[RawRegisterPayload],
        observations: List[RawDiscoveryObservation],
    ) -> Tuple[SourceEnumerationState, Optional[str]]:
        if not payloads:
            return SourceEnumerationState.FAILED, "No payloads retrieved for Issuer"
        return SourceEnumerationState.COMPLETE, None
