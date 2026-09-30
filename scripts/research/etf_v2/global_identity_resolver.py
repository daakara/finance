"""
scripts/research/etf_v2/global_identity_resolver.py

Deterministic GlobalETFIdentityResolver domain facade, declarative adapter registry,
authority adapter protocol, and domain authority wrappers for Pipeline V2 (Wave 3).

Enforces:
- Read-only, air-gapped, idempotent resolution (RESOLVER_MUTATION_MODEL = READ_ONLY)
- Explicit declarative adapter registration with unique adapter_id
- Request-semantics and adapter-capability routing (zero product-specific branches)
- Multi-adapter execution and exact share-class merge reconciliation
- First-class authority execution and partial-authority failure handling (AUTHORITY_FAILURE)
- Fail-closed ambiguity and contradiction handling across identifiers and hints
- Deterministic 6-tuple candidate ordering and output collection ordering
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple, runtime_checkable

from .global_identity_models import (
    ETFInstrument,
    ETFListing,
    ETFShareClass,
    Jurisdiction,
)
from .global_identity_resolver_models import (
    AdapterCapabilities,
    AdapterExecutionOutcome,
    AdapterResolutionResult,
    BrokerAliasRecord,
    ETFIdentityCandidate,
    ETFIdentityQuery,
    ETFIdentityResolution,
    NormalizationOutcome,
    NormalizedETFIdentityQuery,
    ProvenanceReference,
    ResolutionReason,
    ResolutionStatus,
    ResolverIdentifierType,
    ResolverQueryClass,
    is_authoritative_identifier_type,
    normalize_broker_alias_text,
    normalize_identity_query,
    normalize_name_for_share_class,
     normalize_text_nfkc,
    precedence_rank_for_field,
    sort_candidates_deterministically,
    sort_listings_deterministically,
    sort_provenance_deterministically,
)
from .models import EntityIdentity, PopulationRecord
from .ucits_authority_adapter import UCITSAuthorityAdapter
from .ucits_provenance_models import UCITSSourceProvenanceRecord
from .us_sec_authority_adapter import USSECAuthorityAdapter

# Declarative Registry & Reconciliation Constants
ADAPTER_REGISTRATION_MODEL: str = "EXPLICIT_DECLARATIVE_REGISTRY_WITH_UNIQUE_ADAPTER_IDS"
ADAPTER_SELECTION_MODEL: str = "REQUEST_SEMANTICS_AND_ADAPTER_CAPABILITIES_ONLY"
MULTI_ADAPTER_QUERY_ALLOWED: bool = True
CONFLICT_RECONCILIATION_MODEL: str = (
    "COLLECT_ALL_APPLICABLE_RESULTS_THEN_MERGE_EXACT_SHARE_CLASS_MATCHES_"
    "AND_FAIL_CLOSED_ON_INCOMPATIBLE_AUTHORITATIVE_IDENTITIES"
)
REGISTRATION_ORDER_AFFECTS_RESULT: bool = False
RESPONSE_ORDER_AFFECTS_RESULT: bool = False
CANDIDATE_INPUT_ORDER_AFFECTS_RESULT: bool = False
CANONICAL_OUTPUT_ORDER_DETERMINISTIC: bool = True


@runtime_checkable
class ETFAuthorityAdapterProtocol(Protocol):
    """Minimum protocol implemented by every Wave 3 resolver authority adapter (Section 13)."""

    @property
    def adapter_id(self) -> str:
        ...

    def capabilities(self) -> AdapterCapabilities:
        ...

    def supports(self, normalized_query: NormalizedETFIdentityQuery) -> bool:
        ...

    def resolve(self, normalized_query: NormalizedETFIdentityQuery) -> AdapterResolutionResult:
        ...


@dataclass(frozen=True)
class AuthorityShareClassEntry:
    """
    Immutable in-memory domain entry pairing an ETFInstrument, ETFShareClass,
    its ETFListing tuple, optional WKN codes, and backing ProvenanceReference records.
    """
    instrument: ETFInstrument
    share_class: ETFShareClass
    listings: Tuple[ETFListing, ...]
    provenance_references: Tuple[ProvenanceReference, ...]
    wkn_codes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.share_class.instrument_id != self.instrument.canonical_instrument_id:
            raise ValueError(
                f"ShareClass instrument_id '{self.share_class.instrument_id}' does not match "
                f"Instrument '{self.instrument.canonical_instrument_id}'."
            )
        for listing in self.listings:
            if listing.share_class_id != self.share_class.share_class_id:
                raise ValueError(
                    f"Listing share_class_id '{listing.share_class_id}' does not match "
                    f"ShareClass '{self.share_class.share_class_id}'."
                )
        if not self.provenance_references:
            raise ValueError("AuthorityShareClassEntry requires at least one ProvenanceReference.")


class AuthorityAdapterRegistry:
    """
    Explicit declarative registry of ETFAuthorityAdapterProtocol instances.
    Enforces unique adapter_id registration and order-invariant selection.
    """

    def __init__(self, adapters: Optional[Iterable[ETFAuthorityAdapterProtocol]] = None) -> None:
        self._adapters: Dict[str, ETFAuthorityAdapterProtocol] = {}
        if adapters is not None:
            for adapter in adapters:
                self.register(adapter)

    def register(self, adapter: ETFAuthorityAdapterProtocol) -> None:
        if not isinstance(adapter, ETFAuthorityAdapterProtocol):
            raise TypeError(f"Adapter must implement ETFAuthorityAdapterProtocol, got: {type(adapter)}")
        aid = adapter.adapter_id.strip()
        if not aid:
            raise ValueError("Adapter adapter_id must be a non-empty string.")
        if aid in self._adapters:
            raise ValueError(f"Duplicate authority adapter registration for adapter_id: '{aid}'")
        caps = adapter.capabilities()
        if caps.adapter_id != aid:
            raise ValueError(
                f"Adapter adapter_id '{aid}' does not match capabilities().adapter_id '{caps.adapter_id}'"
            )
        self._adapters[aid] = adapter

    def get(self, adapter_id: str) -> Optional[ETFAuthorityAdapterProtocol]:
        return self._adapters.get(adapter_id)

    def all_adapters(self) -> Tuple[ETFAuthorityAdapterProtocol, ...]:
        """Returns registered adapters sorted deterministically by adapter_id ascending."""
        return tuple(self._adapters[aid] for aid in sorted(self._adapters.keys()))

    def select_adapters_for_query(
        self,
        normalized_query: NormalizedETFIdentityQuery,
    ) -> Tuple[ETFAuthorityAdapterProtocol, ...]:
        """
        Selects applicable adapters based strictly on normalized request semantics
        and declarative adapter capabilities (Section 9.1).
        Registration order has zero influence on selection or execution order.
        """
        if normalized_query.outcome != NormalizationOutcome.NORMALIZED or normalized_query.query_class is None:
            return ()

        q_class = normalized_query.query_class
        id_type = normalized_query.inferred_identifier_type
        q_val = normalized_query.normalized_query
        jur_hint = normalized_query.jurisdiction_hint
        broker_hint = normalized_query.broker_source_hint

        # Infer implied jurisdiction from canonical internal ID or ISIN prefix ONLY when no explicit
        # contradictory jurisdiction_hint is supplied (so contradictions can be caught as AUTHORITY_CONFLICT).
        implied_jur: Optional[str] = None
        if id_type == ResolverIdentifierType.CANONICAL_INTERNAL_ID and q_val.startswith("etfi:v1:"):
            parts = q_val.split(":")
            if len(parts) == 5:
                implied_jur = parts[2].upper()
        elif id_type == ResolverIdentifierType.CANONICAL_INTERNAL_ID and q_val.startswith("etfs:v1:SEC_CLASS_ID:"):
            implied_jur = Jurisdiction.US_SEC.value
        elif id_type == ResolverIdentifierType.ISIN and len(q_val) == 12:
            prefix = q_val[:2].upper()
            if prefix == "US":
                implied_jur = Jurisdiction.US_SEC.value
            elif prefix in ("IE", "LU"):
                implied_jur = prefix

        selected: List[ETFAuthorityAdapterProtocol] = []
        for adapter in self.all_adapters():
            caps = adapter.capabilities()
            if q_class not in caps.supported_query_classes:
                continue

            if q_class == ResolverQueryClass.BROKER_ALIAS:
                if not broker_hint or broker_hint not in caps.supported_broker_namespaces:
                    continue
                selected.append(adapter)
                continue

            # If explicit jurisdiction_hint is supplied and no implied jurisdiction conflicts with it,
            # filter to adapters supporting jurisdiction_hint. If implied_jur != jur_hint, select adapters
            # for implied_jur so the resolver can establish the authoritative identity and report CONFLICTING_LISTING_CONTEXT.
            if jur_hint is not None and implied_jur is None:
                if jur_hint not in caps.supported_jurisdictions:
                    continue

            if implied_jur is not None:
                if implied_jur not in caps.supported_jurisdictions:
                    continue

            selected.append(adapter)

        return tuple(selected)


def _venue_matches(listing: ETFListing, venue_hint: str) -> bool:
    """Case-insensitive comparison of venue_hint against listing venue_name or venue_mic."""
    vh = normalize_text_nfkc(venue_hint)
    v_name = normalize_text_nfkc(listing.venue_name) if listing.venue_name else ""
    v_mic = normalize_text_nfkc(listing.venue_mic) if listing.venue_mic else ""
    return vh in (v_name, v_mic)


def _evaluate_entries_against_query(
    adapter_id: str,
    authority_source_label: str,
    default_jurisdiction: str,
    entries: Sequence[AuthorityShareClassEntry],
    normalized_query: NormalizedETFIdentityQuery,
) -> AdapterResolutionResult:
    """
    Shared deterministic evaluation engine for statutory authority entries
    (US SEC and EU UCITS).
    """
    q_class = normalized_query.query_class
    id_type = normalized_query.inferred_identifier_type
    q_val = normalized_query.normalized_query

    if q_class is None or id_type is None:
        return AdapterResolutionResult(
            adapter_id=adapter_id,
            outcome=AdapterExecutionOutcome.UNSUPPORTED_QUERY,
            failure_reason=ResolutionReason.AUTHORITY_CAPABILITY_MISMATCH,
        )

    matched_candidates: List[ETFIdentityCandidate] = []

    for entry in entries:
        inst = entry.instrument
        sc = entry.share_class
        sorted_listings = sort_listings_deterministically(entry.listings)
        jur_val = inst.regulatory_jurisdiction.value

        matched_listings: Optional[Tuple[ETFListing, ...]] = None
        matched_id_str: Optional[str] = None

        if id_type == ResolverIdentifierType.CANONICAL_INTERNAL_ID:
            if q_val == sc.share_class_id:
                matched_listings = sorted_listings
                matched_id_str = sc.share_class_id
            elif q_val == inst.canonical_instrument_id:
                matched_listings = sorted_listings
                matched_id_str = inst.canonical_instrument_id
            else:
                matching_l = tuple(l for l in sorted_listings if l.listing_id == q_val)
                if matching_l:
                    matched_listings = matching_l
                    matched_id_str = q_val

        elif id_type == ResolverIdentifierType.ISIN:
            if sc.isin and sc.isin.upper() == q_val.upper():
                matched_listings = sorted_listings
                matched_id_str = sc.isin.upper()

        elif id_type == ResolverIdentifierType.WKN:
            wkn_set = {w.upper() for w in entry.wkn_codes if w}
            for l in sorted_listings:
                if l.local_code:
                    wkn_set.add(l.local_code.upper())
            if q_val.upper() in wkn_set:
                wkn_listings = tuple(
                    l for l in sorted_listings if l.local_code and l.local_code.upper() == q_val.upper()
                )
                matched_listings = wkn_listings if wkn_listings else sorted_listings
                matched_id_str = q_val.upper()

        elif id_type in (ResolverIdentifierType.TICKER, ResolverIdentifierType.TICKER_WITH_LISTING_CONTEXT):
            t_listings = tuple(l for l in sorted_listings if l.ticker.upper() == q_val.upper())
            if t_listings:
                matched_listings = t_listings
                matched_id_str = q_val.upper()

        elif id_type == ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME:
            full_legal = normalize_name_for_share_class(inst.legal_fund_name, normalized_mode=False)
            sc_name = normalize_name_for_share_class(sc.share_class_name, normalized_mode=False)
            combined = normalize_name_for_share_class(
                f"{inst.legal_fund_name} {sc.share_class_name}".strip(),
                normalized_mode=False,
            )
            if q_val in (full_legal, sc_name, combined):
                matched_listings = sorted_listings
                matched_id_str = (
                    f"{inst.legal_fund_name} {sc.share_class_name}".strip()
                    if q_val == combined and sc.share_class_name
                    else (inst.legal_fund_name if q_val == full_legal else sc.share_class_name)
                )

        elif id_type == ResolverIdentifierType.NORMALIZED_SHARE_CLASS_NAME:
            full_norm = normalize_name_for_share_class(inst.legal_fund_name, normalized_mode=True)
            sc_norm = normalize_name_for_share_class(sc.share_class_name, normalized_mode=True)
            combined_norm = normalize_name_for_share_class(
                f"{inst.legal_fund_name} {sc.share_class_name}".strip(),
                normalized_mode=True,
            )
            if q_val in (full_norm, sc_norm, combined_norm):
                matched_listings = sorted_listings
                matched_id_str = (
                    f"{inst.legal_fund_name} {sc.share_class_name}".strip()
                    if q_val == combined_norm and sc.share_class_name
                    else (inst.legal_fund_name if q_val == full_norm else sc.share_class_name)
                )

        if matched_listings is not None and matched_id_str is not None:
            prov_tuple = tuple(
                ProvenanceReference(
                    adapter_id=adapter_id,
                    authority_jurisdiction=p.authority_jurisdiction or jur_val,
                    authority_source=p.authority_source,
                    matched_identifier=matched_id_str,
                    matched_identifier_type=id_type.value,
                    source_record_id=p.source_record_id,
                    source_document_hash=p.source_document_hash,
                    outcome="COMPLETED_MATCH",
                )
                for p in entry.provenance_references
            )
            matched_candidates.append(
                ETFIdentityCandidate(
                    instrument_identity=inst,
                    share_class_identity=sc,
                    listing_identities=matched_listings,
                    matched_identifier=matched_id_str,
                    matched_identifier_type=id_type,
                    authority_adapter_id=adapter_id,
                    authority_jurisdiction=jur_val,
                    provenance_references=sort_provenance_deterministically(prov_tuple),
                )
            )

    if not matched_candidates:
        no_match_prov = ProvenanceReference(
            adapter_id=adapter_id,
            authority_jurisdiction=default_jurisdiction,
            authority_source=authority_source_label,
            matched_identifier=q_val,
            matched_identifier_type=id_type.value,
            source_record_id=f"{adapter_id}:NO_MATCH",
            source_document_hash=None,
            outcome="COMPLETED_NO_MATCH",
        )
        return AdapterResolutionResult(
            adapter_id=adapter_id,
            outcome=AdapterExecutionOutcome.COMPLETED_NO_MATCH,
            candidates=(),
            provenance_references=(no_match_prov,),
        )

    all_prov: List[ProvenanceReference] = []
    for cand in matched_candidates:
        all_prov.extend(cand.provenance_references)

    return AdapterResolutionResult(
        adapter_id=adapter_id,
        outcome=AdapterExecutionOutcome.COMPLETED_MATCH,
        candidates=sort_candidates_deterministically(matched_candidates),
        provenance_references=sort_provenance_deterministically(all_prov),
    )


class USSECResolverAuthorityAdapter:
    """
    Wave 3 resolver authority adapter wrapping the frozen Wave 1 USSECAuthorityAdapter.
    Read-only; never mutates SEC corpus or Wave 1 models.
    """

    def __init__(
        self,
        adapter_id: str = "us_sec_statutory_adapter",
        entries: Optional[Sequence[AuthorityShareClassEntry]] = None,
        sec_entities: Optional[Sequence[Tuple[EntityIdentity, str, str]]] = None,
        population_records: Optional[Sequence[PopulationRecord]] = None,
    ) -> None:
        self._adapter_id = adapter_id
        built_entries: List[AuthorityShareClassEntry] = list(entries or [])

        if sec_entities:
            for entity, mic, ccy in sec_entities:
                inst, sc, listing = USSECAuthorityAdapter.adapt_entity_identity(
                    entity,
                    venue_mic=mic,
                    trading_currency=ccy,
                )
                prov = ProvenanceReference(
                    adapter_id=self._adapter_id,
                    authority_jurisdiction=Jurisdiction.US_SEC.value,
                    authority_source="US_SEC_EDGAR_STATUTORY",
                    matched_identifier=sc.share_class_id,
                    matched_identifier_type=ResolverIdentifierType.CANONICAL_INTERNAL_ID.value,
                    source_record_id=f"SEC:{entity.cik}:{entity.series_id}:{entity.class_id}",
                    source_document_hash=None,
                    outcome="COMPLETED_MATCH",
                )
                built_entries.append(
                    AuthorityShareClassEntry(
                        instrument=inst,
                        share_class=sc,
                        listings=(listing,),
                        provenance_references=(prov,),
                    )
                )

        if population_records:
            for rec in population_records:
                inst, sc, listing = USSECAuthorityAdapter.adapt_population_record(rec)
                prov = ProvenanceReference(
                    adapter_id=self._adapter_id,
                    authority_jurisdiction=Jurisdiction.US_SEC.value,
                    authority_source="US_SEC_GOLDEN_CORPUS",
                    matched_identifier=sc.share_class_id,
                    matched_identifier_type=ResolverIdentifierType.CANONICAL_INTERNAL_ID.value,
                    source_record_id=f"SEC_ACCESSION:{rec.prospectus_accession}:{rec.class_id}",
                    source_document_hash=rec.prospectus_raw_sha256,
                    outcome="COMPLETED_MATCH",
                )
                built_entries.append(
                    AuthorityShareClassEntry(
                        instrument=inst,
                        share_class=sc,
                        listings=(listing,),
                        provenance_references=(prov,),
                    )
                )

        self._entries: Tuple[AuthorityShareClassEntry, ...] = tuple(built_entries)

        mics: Set[str] = {"ARCX", "XNAS", "BATS", "XNYS"}
        venues: Set[str] = {"nyse arca", "nasdaq", "cboe bzx", "nyse"}
        ccys: Set[str] = {"USD"}
        for entry in self._entries:
            for l in entry.listings:
                if l.venue_mic:
                    mics.add(l.venue_mic.upper())
                if l.venue_name:
                    venues.add(normalize_text_nfkc(l.venue_name))
                if l.trading_currency:
                    ccys.add(l.trading_currency.upper())

        self._capabilities = AdapterCapabilities(
            adapter_id=self._adapter_id,
            supported_jurisdictions=frozenset({Jurisdiction.US_SEC.value, "US"}),
            supported_query_classes=frozenset({
                ResolverQueryClass.CANONICAL_INTERNAL_ID,
                ResolverQueryClass.ISIN,
                ResolverQueryClass.TICKER_WITH_LISTING_CONTEXT,
                ResolverQueryClass.LEGAL_SHARE_CLASS_NAME,
                ResolverQueryClass.NORMALIZED_SHARE_CLASS_NAME,
            }),
            supported_identifier_namespaces=frozenset({
                "CANONICAL_INTERNAL_ID",
                "ISIN",
                "SEC_CLASS_ID",
                "SEC_SERIES_ID",
                "CIK",
                "TICKER",
            }),
            supported_mics=frozenset(mics),
            supported_venues=frozenset(venues),
            supported_currencies=frozenset(ccys),
            supported_broker_namespaces=frozenset(),
            authority_precedence=1,
            supports_provenance=True,
        )

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    def supports(self, normalized_query: NormalizedETFIdentityQuery) -> bool:
        if normalized_query.outcome != NormalizationOutcome.NORMALIZED or normalized_query.query_class is None:
            return False
        return normalized_query.query_class in self._capabilities.supported_query_classes

    def resolve(self, normalized_query: NormalizedETFIdentityQuery) -> AdapterResolutionResult:
        if not self.supports(normalized_query):
            return AdapterResolutionResult(
                adapter_id=self._adapter_id,
                outcome=AdapterExecutionOutcome.UNSUPPORTED_QUERY,
                failure_reason=ResolutionReason.AUTHORITY_CAPABILITY_MISMATCH,
            )
        return _evaluate_entries_against_query(
            adapter_id=self._adapter_id,
            authority_source_label="US_SEC_EDGAR_STATUTORY",
            default_jurisdiction=Jurisdiction.US_SEC.value,
            entries=self._entries,
            normalized_query=normalized_query,
        )


class UCITSResolverAuthorityAdapter:
    """
    Wave 3 resolver authority adapter wrapping the frozen Wave 2 UCITSAuthorityAdapter
    and UCITSSourceProvenanceRecord contracts.
    Read-only; performs zero network or ledger mutations.
    """

    def __init__(
        self,
        adapter_id: str = "eu_ucits_statutory_adapter",
        entries: Optional[Sequence[AuthorityShareClassEntry]] = None,
        ucits_adapter: Optional[UCITSAuthorityAdapter] = None,
    ) -> None:
        self._adapter_id = adapter_id
        self._ucits_adapter = ucits_adapter or UCITSAuthorityAdapter()
        self._entries: Tuple[AuthorityShareClassEntry, ...] = tuple(entries or [])

        # Validate that every injected UCITS entry complies with Wave 2 domicile support
        for entry in self._entries:
            dom = entry.instrument.domicile_iso2.upper().strip()
            if not self._ucits_adapter.supports_jurisdiction(dom):
                raise ValueError(f"UCITSResolverAuthorityAdapter entry has unsupported domicile: {dom}")

        mics: Set[str] = {"XETR", "TGAT", "XLON", "XSWX", "XMIL", "XPAR", "XAMS"}
        venues: Set[str] = {"xetra", "tradegate", "london stock exchange", "six swiss exchange", "borsa italiana"}
        ccys: Set[str] = {"EUR", "USD", "GBP", "CHF"}
        for entry in self._entries:
            for l in entry.listings:
                if l.venue_mic:
                    mics.add(l.venue_mic.upper())
                if l.venue_name:
                    venues.add(normalize_text_nfkc(l.venue_name))
                if l.trading_currency:
                    ccys.add(l.trading_currency.upper())

        self._capabilities = AdapterCapabilities(
            adapter_id=self._adapter_id,
            supported_jurisdictions=frozenset({
                Jurisdiction.EU_UCITS.value,
                Jurisdiction.UK_UCITS.value,
                "IE",
                "LU",
            }),
            supported_query_classes=frozenset({
                ResolverQueryClass.CANONICAL_INTERNAL_ID,
                ResolverQueryClass.ISIN,
                ResolverQueryClass.WKN,
                ResolverQueryClass.TICKER_WITH_LISTING_CONTEXT,
                ResolverQueryClass.LEGAL_SHARE_CLASS_NAME,
                ResolverQueryClass.NORMALIZED_SHARE_CLASS_NAME,
            }),
            supported_identifier_namespaces=frozenset({
                "CANONICAL_INTERNAL_ID",
                "ISIN",
                "WKN",
                "TICKER",
            }),
            supported_mics=frozenset(mics),
            supported_venues=frozenset(venues),
            supported_currencies=frozenset(ccys),
            supported_broker_namespaces=frozenset(),
            authority_precedence=1,
            supports_provenance=True,
        )

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    def supports(self, normalized_query: NormalizedETFIdentityQuery) -> bool:
        if normalized_query.outcome != NormalizationOutcome.NORMALIZED or normalized_query.query_class is None:
            return False
        return normalized_query.query_class in self._capabilities.supported_query_classes

    def resolve(self, normalized_query: NormalizedETFIdentityQuery) -> AdapterResolutionResult:
        if not self.supports(normalized_query):
            return AdapterResolutionResult(
                adapter_id=self._adapter_id,
                outcome=AdapterExecutionOutcome.UNSUPPORTED_QUERY,
                failure_reason=ResolutionReason.AUTHORITY_CAPABILITY_MISMATCH,
            )
        return _evaluate_entries_against_query(
            adapter_id=self._adapter_id,
            authority_source_label="EU_UCITS_STATUTORY_REGISTRY",
            default_jurisdiction=Jurisdiction.EU_UCITS.value,
            entries=self._entries,
            normalized_query=normalized_query,
        )


class GenericBrokerAliasAuthorityAdapter:
    """
    Generic namespace-scoped broker-alias authority adapter (Section 15).
    Resolves broker labels strictly through registered BrokerAliasRecord entries
    backed by statutory AuthorityShareClassEntry records.
    """

    def __init__(
        self,
        adapter_id: str = "generic_broker_alias_adapter",
        alias_records: Optional[Sequence[BrokerAliasRecord]] = None,
        statutory_entries: Optional[Sequence[AuthorityShareClassEntry]] = None,
    ) -> None:
        self._adapter_id = adapter_id
        self._alias_records: Tuple[BrokerAliasRecord, ...] = tuple(alias_records or [])
        self._statutory_by_scid: Dict[str, AuthorityShareClassEntry] = {
            entry.share_class.share_class_id: entry for entry in (statutory_entries or [])
        }

        namespaces = {rec.broker_source_namespace.strip().upper() for rec in self._alias_records if rec.active}
        jurisdictions = {
            entry.instrument.regulatory_jurisdiction.value for entry in self._statutory_by_scid.values()
        } or {Jurisdiction.EU_UCITS.value, Jurisdiction.US_SEC.value}

        self._capabilities = AdapterCapabilities(
            adapter_id=self._adapter_id,
            supported_jurisdictions=frozenset(jurisdictions),
            supported_query_classes=frozenset({ResolverQueryClass.BROKER_ALIAS}),
            supported_identifier_namespaces=frozenset({"BROKER_ALIAS"}),
            supported_mics=frozenset(),
            supported_venues=frozenset(),
            supported_currencies=frozenset(),
            supported_broker_namespaces=frozenset(namespaces),
            authority_precedence=3,
            supports_provenance=True,
        )

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    def supports(self, normalized_query: NormalizedETFIdentityQuery) -> bool:
        if normalized_query.outcome != NormalizationOutcome.NORMALIZED:
            return False
        if normalized_query.query_class != ResolverQueryClass.BROKER_ALIAS:
            return False
        if not normalized_query.broker_source_hint:
            return False
        return normalized_query.broker_source_hint.upper() in self._capabilities.supported_broker_namespaces

    def resolve(self, normalized_query: NormalizedETFIdentityQuery) -> AdapterResolutionResult:
        if not self.supports(normalized_query):
            return AdapterResolutionResult(
                adapter_id=self._adapter_id,
                outcome=AdapterExecutionOutcome.UNSUPPORTED_QUERY,
                failure_reason=ResolutionReason.AUTHORITY_CAPABILITY_MISMATCH,
            )

        ns = (normalized_query.broker_source_hint or "").upper()
        q_alias = normalize_broker_alias_text(normalized_query.normalized_query)

        matching_records = [
            rec
            for rec in self._alias_records
            if rec.active
            and rec.broker_source_namespace.strip().upper() == ns
            and normalize_broker_alias_text(rec.normalized_broker_alias) == q_alias
        ]

        if not matching_records:
            no_match_prov = ProvenanceReference(
                adapter_id=self._adapter_id,
                authority_jurisdiction="BROKER_NAMESPACE",
                authority_source=f"BROKER_ALIAS_REGISTRY:{ns}",
                matched_identifier=q_alias,
                matched_identifier_type=ResolverIdentifierType.BROKER_ALIAS.value,
                source_record_id=f"{self._adapter_id}:{ns}:NO_MATCH",
                source_document_hash=None,
                outcome="COMPLETED_NO_MATCH",
            )
            return AdapterResolutionResult(
                adapter_id=self._adapter_id,
                outcome=AdapterExecutionOutcome.COMPLETED_NO_MATCH,
                candidates=(),
                provenance_references=(no_match_prov,),
            )

        candidates: List[ETFIdentityCandidate] = []
        for rec in matching_records:
            statutory = self._statutory_by_scid.get(rec.target_share_class_id)
            if statutory is None:
                # A broker alias cannot invent a canonical share class without statutory authority backing
                return AdapterResolutionResult(
                    adapter_id=self._adapter_id,
                    outcome=AdapterExecutionOutcome.INVALID_RESPONSE,
                    failure_reason=ResolutionReason.AUTHORITY_RESPONSE_INVALID,
                    diagnostic_detail=(
                        f"BrokerAliasRecord target_share_class_id '{rec.target_share_class_id}' "
                        "not found in statutory authority entries."
                    ),
                )

            listings = sort_listings_deterministically(statutory.listings)
            if rec.target_listing_id:
                filtered = tuple(l for l in listings if l.listing_id == rec.target_listing_id)
                if not filtered:
                    return AdapterResolutionResult(
                        adapter_id=self._adapter_id,
                        outcome=AdapterExecutionOutcome.INVALID_RESPONSE,
                        failure_reason=ResolutionReason.AUTHORITY_RESPONSE_INVALID,
                    )
                listings = filtered

            combined_prov = sort_provenance_deterministically(
                list(statutory.provenance_references) + [rec.provenance_reference]
            )
            candidates.append(
                ETFIdentityCandidate(
                    instrument_identity=statutory.instrument,
                    share_class_identity=statutory.share_class,
                    listing_identities=listings,
                    matched_identifier=rec.raw_broker_alias,
                    matched_identifier_type=ResolverIdentifierType.BROKER_ALIAS,
                    authority_adapter_id=self._adapter_id,
                    authority_jurisdiction=statutory.instrument.regulatory_jurisdiction.value,
                    provenance_references=combined_prov,
                )
            )

        all_prov: List[ProvenanceReference] = []
        for c in candidates:
            all_prov.extend(c.provenance_references)

        return AdapterResolutionResult(
            adapter_id=self._adapter_id,
            outcome=AdapterExecutionOutcome.COMPLETED_MATCH,
            candidates=sort_candidates_deterministically(candidates),
            provenance_references=sort_provenance_deterministically(all_prov),
        )


def _share_classes_compatible(a: ETFShareClass, b: ETFShareClass) -> bool:
    """
    Checks whether two ETFShareClass representations of the same share_class_id
    are structurally and economically compatible across adapters.
    """
    if a.share_class_id != b.share_class_id:
        return False
    if a.instrument_id != b.instrument_id:
        return False
    if a.isin and b.isin and a.isin.upper() != b.isin.upper():
        return False
    if a.sec_class_id and b.sec_class_id and a.sec_class_id.upper() != b.sec_class_id.upper():
        return False
    if a.distribution_policy and b.distribution_policy and a.distribution_policy.upper() != b.distribution_policy.upper():
        return False
    if a.base_currency and b.base_currency and a.base_currency.upper() != b.base_currency.upper():
        return False
    if a.hedging_policy and b.hedging_policy and a.hedging_policy.upper() != b.hedging_policy.upper():
        return False
    return True


def _instruments_compatible(a: ETFInstrument, b: ETFInstrument) -> bool:
    if a.canonical_instrument_id != b.canonical_instrument_id:
        return False
    if a.domicile_iso2.upper() != b.domicile_iso2.upper():
        return False
    if a.regulatory_jurisdiction != b.regulatory_jurisdiction:
        return False
    return True


class GlobalETFIdentityResolver:
    """
    Deterministic GlobalETFIdentityResolver domain facade (Wave 3).
    Unifies US SEC, EU UCITS, and registered generic broker-alias authority adapters
    under a single fail-closed, provenance-preserving, read-only contract.
    """

    def __init__(
        self,
        registry: Optional[AuthorityAdapterRegistry] = None,
        adapters: Optional[Iterable[ETFAuthorityAdapterProtocol]] = None,
    ) -> None:
        if registry is not None and adapters is not None:
            raise ValueError("Provide either 'registry' or 'adapters', not both.")
        if registry is not None:
            self._registry = registry
        else:
            self._registry = AuthorityAdapterRegistry(adapters=adapters)

    @property
    def registry(self) -> AuthorityAdapterRegistry:
        return self._registry

    def resolve(self, query: ETFIdentityQuery) -> ETFIdentityResolution:
        """
        Executes deterministic normalization, capability routing, multi-adapter querying,
        precedence evaluation, hint constraint checking, and candidate reconciliation.
        Never mutates authority state or performs network/disk I/O.
        """
        norm = normalize_identity_query(query)

        if norm.outcome == NormalizationOutcome.INVALID_IDENTIFIER:
            return ETFIdentityResolution(
                resolution_status=ResolutionStatus.INVALID_IDENTIFIER,
                canonical_internal_id=None,
                instrument_identity=None,
                share_class_identity=None,
                listing_identities=(),
                matched_identifier=None,
                matched_identifier_type=norm.inferred_identifier_type,
                authority_adapter_ids=(),
                authority_jurisdictions=(),
                ambiguity_candidates=(),
                provenance_references=(),
                reason_code=norm.failure_reason or ResolutionReason.INVALID_QUERY,
            )

        if norm.outcome == NormalizationOutcome.UNSUPPORTED_QUERY_TYPE or not norm.parsed_fields:
            return ETFIdentityResolution(
                resolution_status=ResolutionStatus.UNSUPPORTED_QUERY_TYPE,
                canonical_internal_id=None,
                instrument_identity=None,
                share_class_identity=None,
                listing_identities=(),
                matched_identifier=None,
                matched_identifier_type=norm.inferred_identifier_type,
                authority_adapter_ids=(),
                authority_jurisdictions=(),
                ambiguity_candidates=(),
                provenance_references=(),
                reason_code=norm.failure_reason or ResolutionReason.NO_APPLICABLE_ADAPTER,
            )

        # Evaluate each supplied identifier field in precedence order (FIRST_MATCH_WINS = False)
        field_evaluations: List[Tuple[ResolverIdentifierType, str, Tuple[ETFIdentityCandidate, ...], Tuple[ProvenanceReference, ...], Tuple[str, ...]]] = []

        for field_type, field_val in norm.parsed_fields:
            field_q_class = (
                ResolverQueryClass.TICKER_WITH_LISTING_CONTEXT
                if field_type in (ResolverIdentifierType.TICKER, ResolverIdentifierType.TICKER_WITH_LISTING_CONTEXT)
                else ResolverQueryClass(field_type.value)
            )
            sub_norm = NormalizedETFIdentityQuery(
                outcome=NormalizationOutcome.NORMALIZED,
                raw_query=norm.raw_query,
                normalized_query=field_val,
                inferred_identifier_type=field_type,
                query_class=field_q_class,
                identifier_type_hint=norm.identifier_type_hint,
                jurisdiction_hint=norm.jurisdiction_hint,
                mic_hint=norm.mic_hint,
                venue_hint=norm.venue_hint,
                currency_hint=norm.currency_hint,
                broker_source_hint=norm.broker_source_hint,
                parsed_fields=((field_type, field_val),),
            )

            selected_adapters = self._registry.select_adapters_for_query(sub_norm)
            if not selected_adapters:
                return ETFIdentityResolution(
                    resolution_status=ResolutionStatus.UNSUPPORTED_QUERY_TYPE,
                    canonical_internal_id=None,
                    instrument_identity=None,
                    share_class_identity=None,
                    listing_identities=(),
                    matched_identifier=None,
                    matched_identifier_type=field_type,
                    authority_adapter_ids=(),
                    authority_jurisdictions=(),
                    ambiguity_candidates=(),
                    provenance_references=(),
                    reason_code=ResolutionReason.NO_APPLICABLE_ADAPTER,
                )

            adapter_results: List[AdapterResolutionResult] = []
            for adapter in selected_adapters:
                try:
                    res = adapter.resolve(sub_norm)
                except Exception as exc:
                    res = AdapterResolutionResult(
                        adapter_id=adapter.adapter_id,
                        outcome=AdapterExecutionOutcome.EXECUTION_FAILURE,
                        failure_reason=ResolutionReason.AUTHORITY_EXECUTION_FAILURE,
                        diagnostic_detail=str(exc),
                    )
                adapter_results.append(res)

            # Check for authority failures / invalid responses / partial failures (Section 14)
            failed_results: List[Tuple[AdapterResolutionResult, ResolutionReason]] = []
            successful_match_count = 0
            collected_prov: List[ProvenanceReference] = []
            raw_candidates: List[ETFIdentityCandidate] = []
            queried_adapter_ids = tuple(sorted({a.adapter_id for a in selected_adapters}))

            for res in adapter_results:
                collected_prov.extend(res.provenance_references)
                if res.outcome == AdapterExecutionOutcome.COMPLETED_MATCH:
                    if not res.candidates:
                        failed_results.append((res, ResolutionReason.AUTHORITY_RESPONSE_INVALID))
                    else:
                        # Validate candidate integrity
                        valid_cands = True
                        for c in res.candidates:
                            if not isinstance(c, ETFIdentityCandidate) or not c.provenance_references:
                                valid_cands = False
                                break
                        if not valid_cands:
                            failed_results.append((res, ResolutionReason.AUTHORITY_RESPONSE_INVALID))
                        else:
                            successful_match_count += 1
                            raw_candidates.extend(res.candidates)
                elif res.outcome == AdapterExecutionOutcome.COMPLETED_NO_MATCH:
                    continue
                elif res.outcome == AdapterExecutionOutcome.UNAVAILABLE:
                    failed_results.append((res, res.failure_reason or ResolutionReason.AUTHORITY_UNAVAILABLE))
                elif res.outcome == AdapterExecutionOutcome.EXECUTION_FAILURE:
                    failed_results.append((res, res.failure_reason or ResolutionReason.AUTHORITY_EXECUTION_FAILURE))
                elif res.outcome == AdapterExecutionOutcome.INVALID_RESPONSE:
                    failed_results.append((res, res.failure_reason or ResolutionReason.AUTHORITY_RESPONSE_INVALID))
                elif res.outcome == AdapterExecutionOutcome.UNSUPPORTED_QUERY:
                    failed_results.append((res, res.failure_reason or ResolutionReason.AUTHORITY_CAPABILITY_MISMATCH))
                elif res.outcome == AdapterExecutionOutcome.CONFLICTING_EVIDENCE:
                    return ETFIdentityResolution(
                        resolution_status=ResolutionStatus.AUTHORITY_CONFLICT,
                        canonical_internal_id=None,
                        instrument_identity=None,
                        share_class_identity=None,
                        listing_identities=(),
                        matched_identifier=field_val,
                        matched_identifier_type=field_type,
                        authority_adapter_ids=queried_adapter_ids,
                        authority_jurisdictions=tuple(
                            sorted({c.authority_jurisdiction for c in res.candidates})
                        ),
                        ambiguity_candidates=sort_candidates_deterministically(res.candidates),
                        provenance_references=sort_provenance_deterministically(collected_prov),
                        reason_code=res.failure_reason or ResolutionReason.CONFLICTING_ADAPTER_IDENTITIES,
                    )

            if failed_results:
                # Partial-authority failure: at least one adapter succeeded with a match while another required adapter failed
                if successful_match_count > 0 and len(selected_adapters) > 1:
                    failure_reason = ResolutionReason.INCOMPLETE_AUTHORITY_SET
                else:
                    failure_reason = failed_results[0][1]
                return ETFIdentityResolution(
                    resolution_status=ResolutionStatus.AUTHORITY_FAILURE,
                    canonical_internal_id=None,
                    instrument_identity=None,
                    share_class_identity=None,
                    listing_identities=(),
                    matched_identifier=None,
                    matched_identifier_type=field_type,
                    authority_adapter_ids=queried_adapter_ids,
                    authority_jurisdictions=(),
                    ambiguity_candidates=(),
                    provenance_references=sort_provenance_deterministically(collected_prov),
                    reason_code=failure_reason,
                )

            # Merge exact share_class_id matches across adapters and detect adapter-level conflicts (Section 9.3)
            merged_by_scid, adapter_conflict_cands = _merge_candidates_by_share_class(raw_candidates)
            if adapter_conflict_cands is not None:
                return ETFIdentityResolution(
                    resolution_status=ResolutionStatus.AUTHORITY_CONFLICT,
                    canonical_internal_id=None,
                    instrument_identity=None,
                    share_class_identity=None,
                    listing_identities=(),
                    matched_identifier=field_val,
                    matched_identifier_type=field_type,
                    authority_adapter_ids=queried_adapter_ids,
                    authority_jurisdictions=tuple(
                        sorted({c.authority_jurisdiction for c in adapter_conflict_cands})
                    ),
                    ambiguity_candidates=sort_candidates_deterministically(adapter_conflict_cands),
                    provenance_references=sort_provenance_deterministically(collected_prov),
                    reason_code=ResolutionReason.CONFLICTING_ADAPTER_IDENTITIES,
                )

            # If the query is a share-class/listing authoritative identifier (ISIN, WKN, or etfs/etfl CANONICAL_INTERNAL_ID)
            # and multiple adapters returned DIFFERENT share_class_ids for that same authoritative ID,
            # fail closed with AUTHORITY_CONFLICT (CONFLICTING_ADAPTER_IDENTITIES).
            is_single_sc_authoritative = (
                field_type in (ResolverIdentifierType.ISIN, ResolverIdentifierType.WKN)
                or (
                    field_type == ResolverIdentifierType.CANONICAL_INTERNAL_ID
                    and (field_val.startswith("etfs:v1:") or field_val.startswith("etfl:v1:"))
                )
            )
            if is_single_sc_authoritative and len(merged_by_scid) > 1:
                distinct_adapters = {c.authority_adapter_id for c in raw_candidates}
                conflict_reason = (
                    ResolutionReason.CONFLICTING_ADAPTER_IDENTITIES
                    if len(distinct_adapters) > 1
                    else ResolutionReason.CONFLICTING_AUTHORITATIVE_IDENTIFIERS
                )
                return ETFIdentityResolution(
                    resolution_status=ResolutionStatus.AUTHORITY_CONFLICT,
                    canonical_internal_id=None,
                    instrument_identity=None,
                    share_class_identity=None,
                    listing_identities=(),
                    matched_identifier=field_val,
                    matched_identifier_type=field_type,
                    authority_adapter_ids=queried_adapter_ids,
                    authority_jurisdictions=tuple(
                        sorted({c.authority_jurisdiction for c in merged_by_scid})
                    ),
                    ambiguity_candidates=sort_candidates_deterministically(merged_by_scid),
                    provenance_references=sort_provenance_deterministically(collected_prov),
                    reason_code=conflict_reason,
                )

            field_evaluations.append(
                (
                    field_type,
                    field_val,
                    merged_by_scid,
                    sort_provenance_deterministically(collected_prov),
                    queried_adapter_ids,
                )
            )

        # Cross-field reconciliation (Section 7 & Section 18)
        primary_type, primary_val, current_candidates, current_prov, current_adapters = field_evaluations[0]
        all_queried_adapters: Set[str] = set(current_adapters)
        all_collected_prov: List[ProvenanceReference] = list(current_prov)

        if len(field_evaluations) > 1:
            for sec_type, sec_val, sec_candidates, sec_prov, sec_adapters in field_evaluations[1:]:
                all_queried_adapters.update(sec_adapters)
                all_collected_prov.extend(sec_prov)

                primary_scids = {c.share_class_identity.share_class_id for c in current_candidates}
                sec_scids = {c.share_class_identity.share_class_id for c in sec_candidates}
                overlap = primary_scids & sec_scids

                if not overlap:
                    # Incompatible supplied identifiers (e.g., valid ISIN + incompatible WKN, or ISIN + incompatible BROKER_ALIAS)
                    combined_conflict_cands = list(current_candidates) + [
                        c for c in sec_candidates if c.share_class_identity.share_class_id not in primary_scids
                    ]
                    if not combined_conflict_cands:
                        return ETFIdentityResolution(
                            resolution_status=ResolutionStatus.NOT_FOUND,
                            canonical_internal_id=None,
                            instrument_identity=None,
                            share_class_identity=None,
                            listing_identities=(),
                            matched_identifier=None,
                            matched_identifier_type=primary_type,
                            authority_adapter_ids=tuple(sorted(all_queried_adapters)),
                            authority_jurisdictions=(),
                            ambiguity_candidates=(),
                            provenance_references=sort_provenance_deterministically(all_collected_prov),
                            reason_code=ResolutionReason.NO_AUTHORITY_MATCH,
                        )
                    return ETFIdentityResolution(
                        resolution_status=ResolutionStatus.AUTHORITY_CONFLICT,
                        canonical_internal_id=None,
                        instrument_identity=None,
                        share_class_identity=None,
                        listing_identities=(),
                        matched_identifier=primary_val,
                        matched_identifier_type=primary_type,
                        authority_adapter_ids=tuple(sorted(all_queried_adapters)),
                        authority_jurisdictions=tuple(
                            sorted({c.authority_jurisdiction for c in combined_conflict_cands})
                        ),
                        ambiguity_candidates=sort_candidates_deterministically(combined_conflict_cands),
                        provenance_references=sort_provenance_deterministically(all_collected_prov),
                        reason_code=ResolutionReason.CONFLICTING_AUTHORITATIVE_IDENTIFIERS,
                    )

                # Corroborating overlap narrows candidates and combines provenance
                sec_by_scid = {c.share_class_identity.share_class_id: c for c in sec_candidates}
                narrowed: List[ETFIdentityCandidate] = []
                for cand in current_candidates:
                    scid = cand.share_class_identity.share_class_id
                    if scid in overlap:
                        sec_c = sec_by_scid[scid]
                        merged_prov = sort_provenance_deterministically(
                            list(cand.provenance_references) + list(sec_c.provenance_references)
                        )
                        narrowed.append(
                            ETFIdentityCandidate(
                                instrument_identity=cand.instrument_identity,
                                share_class_identity=cand.share_class_identity,
                                listing_identities=cand.listing_identities,
                                matched_identifier=cand.matched_identifier,
                                matched_identifier_type=cand.matched_identifier_type,
                                authority_adapter_id=cand.authority_adapter_id,
                                authority_jurisdiction=cand.authority_jurisdiction,
                                provenance_references=merged_prov,
                            )
                        )
                current_candidates = sort_candidates_deterministically(narrowed)

        # If zero candidates before hint filtering -> bounded NOT_FOUND
        if not current_candidates:
            return ETFIdentityResolution(
                resolution_status=ResolutionStatus.NOT_FOUND,
                canonical_internal_id=None,
                instrument_identity=None,
                share_class_identity=None,
                listing_identities=(),
                matched_identifier=None,
                matched_identifier_type=primary_type,
                authority_adapter_ids=tuple(sorted(all_queried_adapters)),
                authority_jurisdictions=(),
                ambiguity_candidates=(),
                provenance_references=sort_provenance_deterministically(all_collected_prov),
                reason_code=ResolutionReason.NO_AUTHORITY_MATCH,
            )

        # Apply supplied jurisdiction_hint, mic_hint, venue_hint, and currency_hint constraints (Section 18)
        filtered_candidates, hint_conflict_cands = _apply_listing_and_jurisdiction_hints(
            candidates=current_candidates,
            jurisdiction_hint=norm.jurisdiction_hint,
            mic_hint=norm.mic_hint,
            venue_hint=norm.venue_hint,
            currency_hint=norm.currency_hint,
        )

        if not filtered_candidates and hint_conflict_cands:
            return ETFIdentityResolution(
                resolution_status=ResolutionStatus.AUTHORITY_CONFLICT,
                canonical_internal_id=None,
                instrument_identity=None,
                share_class_identity=None,
                listing_identities=(),
                matched_identifier=primary_val,
                matched_identifier_type=primary_type,
                authority_adapter_ids=tuple(sorted(all_queried_adapters)),
                authority_jurisdictions=tuple(
                    sorted({c.authority_jurisdiction for c in hint_conflict_cands})
                ),
                ambiguity_candidates=sort_candidates_deterministically(hint_conflict_cands),
                provenance_references=sort_provenance_deterministically(all_collected_prov),
                reason_code=ResolutionReason.CONFLICTING_LISTING_CONTEXT,
            )

        if len(filtered_candidates) == 1:
            winner = filtered_candidates[0]
            resolved_reason = _resolved_reason_for_type(winner.matched_identifier_type)
            adapter_ids_for_winner = tuple(
                sorted({p.adapter_id for p in winner.provenance_references if p.outcome == "COMPLETED_MATCH"})
            ) or tuple(sorted(all_queried_adapters))
            jurisdictions_for_winner = tuple(
                sorted(
                    {
                        winner.instrument_identity.regulatory_jurisdiction.value,
                        *(p.authority_jurisdiction for p in winner.provenance_references if p.outcome == "COMPLETED_MATCH"),
                    }
                )
            )
            return ETFIdentityResolution(
                resolution_status=ResolutionStatus.RESOLVED,
                canonical_internal_id=winner.share_class_identity.share_class_id,
                instrument_identity=winner.instrument_identity,
                share_class_identity=winner.share_class_identity,
                listing_identities=sort_listings_deterministically(winner.listing_identities),
                matched_identifier=winner.matched_identifier,
                matched_identifier_type=winner.matched_identifier_type,
                authority_adapter_ids=adapter_ids_for_winner,
                authority_jurisdictions=jurisdictions_for_winner,
                ambiguity_candidates=(),
                provenance_references=sort_provenance_deterministically(winner.provenance_references),
                reason_code=resolved_reason,
            )

        # >= 2 compatible candidates remain -> fail closed with AMBIGUOUS (Section 8)
        sorted_ambig = sort_candidates_deterministically(filtered_candidates)
        distinct_inst_ids = {c.instrument_identity.canonical_instrument_id for c in sorted_ambig}
        shared_instrument = sorted_ambig[0].instrument_identity if len(distinct_inst_ids) == 1 else None
        ambig_reason = _ambiguous_reason_for_candidates(primary_type, sorted_ambig)

        ambig_prov: List[ProvenanceReference] = []
        for c in sorted_ambig:
            ambig_prov.extend(c.provenance_references)

        return ETFIdentityResolution(
            resolution_status=ResolutionStatus.AMBIGUOUS,
            canonical_internal_id=None,
            instrument_identity=shared_instrument,
            share_class_identity=None,
            listing_identities=(),
            matched_identifier=primary_val,
            matched_identifier_type=primary_type,
            authority_adapter_ids=tuple(sorted(all_queried_adapters)),
            authority_jurisdictions=tuple(sorted({c.authority_jurisdiction for c in sorted_ambig})),
            ambiguity_candidates=sorted_ambig,
            provenance_references=sort_provenance_deterministically(ambig_prov),
            reason_code=ambig_reason,
        )


def _merge_candidates_by_share_class(
    candidates: Sequence[ETFIdentityCandidate],
) -> Tuple[Tuple[ETFIdentityCandidate, ...], Optional[Tuple[ETFIdentityCandidate, ...]]]:
    """
    Merges candidates referencing the same canonical ETFShareClass.share_class_id.
    Returns (merged_candidates, None) on success, or ((), conflicting_candidates) if two
    candidates share a share_class_id or instrument_id with conflicting attributes.
    """
    grouped: Dict[str, List[ETFIdentityCandidate]] = {}
    for cand in candidates:
        scid = cand.share_class_identity.share_class_id
        grouped.setdefault(scid, []).append(cand)

    merged_list: List[ETFIdentityCandidate] = []
    for scid in sorted(grouped.keys()):
        group = grouped[scid]
        first = group[0]
        combined_listings: List[ETFListing] = list(first.listing_identities)
        combined_prov: List[ProvenanceReference] = list(first.provenance_references)

        for other in group[1:]:
            if not _share_classes_compatible(first.share_class_identity, other.share_class_identity):
                return ((), sort_candidates_deterministically(group))
            if not _instruments_compatible(first.instrument_identity, other.instrument_identity):
                return ((), sort_candidates_deterministically(group))
            combined_listings.extend(other.listing_identities)
            combined_prov.extend(other.provenance_references)

        try:
            dedup_listings = sort_listings_deterministically(combined_listings)
        except ValueError:
            return ((), sort_candidates_deterministically(group))

        canonical_adapter_id = sorted(c.authority_adapter_id for c in group)[0]
        merged_sc = ETFShareClass(
            share_class_id=first.share_class_identity.share_class_id,
            instrument_id=first.share_class_identity.instrument_id,
            isin=first.share_class_identity.isin,
            share_class_name=first.share_class_identity.share_class_name,
            distribution_policy=first.share_class_identity.distribution_policy,
            base_currency=first.share_class_identity.base_currency,
            hedging_policy=first.share_class_identity.hedging_policy,
            sec_class_id=first.share_class_identity.sec_class_id,
            identity_status=first.share_class_identity.identity_status,
            listings=dedup_listings,
            metadata=dict(first.share_class_identity.metadata),
        )
        merged_list.append(
            ETFIdentityCandidate(
                instrument_identity=first.instrument_identity,
                share_class_identity=merged_sc,
                listing_identities=dedup_listings,
                matched_identifier=first.matched_identifier,
                matched_identifier_type=first.matched_identifier_type,
                authority_adapter_id=canonical_adapter_id,
                authority_jurisdiction=first.authority_jurisdiction,
                provenance_references=sort_provenance_deterministically(combined_prov),
            )
        )

    return (sort_candidates_deterministically(merged_list), None)


def _apply_listing_and_jurisdiction_hints(
    candidates: Tuple[ETFIdentityCandidate, ...],
    jurisdiction_hint: Optional[str],
    mic_hint: Optional[str],
    venue_hint: Optional[str],
    currency_hint: Optional[str],
) -> Tuple[Tuple[ETFIdentityCandidate, ...], Tuple[ETFIdentityCandidate, ...]]:
    """
    Applies supplied jurisdiction, MIC, venue, and currency hints to candidates.
    If hints eliminate all matched candidates, returns ((), eliminated_candidates) so the
    caller fails closed with AUTHORITY_CONFLICT (CONFLICTING_LISTING_CONTEXT).
    """
    if not (jurisdiction_hint or mic_hint or venue_hint or currency_hint):
        return (candidates, ())

    compatible: List[ETFIdentityCandidate] = []
    eliminated: List[ETFIdentityCandidate] = []

    for cand in candidates:
        inst = cand.instrument_identity
        if jurisdiction_hint is not None:
            jh = jurisdiction_hint.upper()
            accepted_jur_tokens = {
                inst.regulatory_jurisdiction.value.upper(),
                inst.domicile_iso2.upper(),
            }
            if jh not in accepted_jur_tokens:
                eliminated.append(cand)
                continue

        if mic_hint or venue_hint or currency_hint:
            matching_listings: List[ETFListing] = []
            for l in cand.listing_identities:
                if mic_hint and l.venue_mic.upper() != mic_hint.upper():
                    continue
                if venue_hint and not _venue_matches(l, venue_hint):
                    continue
                if currency_hint and l.trading_currency.upper() != currency_hint.upper():
                    continue
                matching_listings.append(l)

            if not matching_listings:
                eliminated.append(cand)
                continue

            compatible.append(
                ETFIdentityCandidate(
                    instrument_identity=cand.instrument_identity,
                    share_class_identity=cand.share_class_identity,
                    listing_identities=sort_listings_deterministically(matching_listings),
                    matched_identifier=cand.matched_identifier,
                    matched_identifier_type=cand.matched_identifier_type,
                    authority_adapter_id=cand.authority_adapter_id,
                    authority_jurisdiction=cand.authority_jurisdiction,
                    provenance_references=cand.provenance_references,
                )
            )
        else:
            compatible.append(cand)

    return (sort_candidates_deterministically(compatible), sort_candidates_deterministically(eliminated))


def _resolved_reason_for_type(id_type: ResolverIdentifierType) -> ResolutionReason:
    if id_type == ResolverIdentifierType.CANONICAL_INTERNAL_ID:
        return ResolutionReason.EXACT_CANONICAL_ID_MATCH
    if id_type == ResolverIdentifierType.ISIN:
        return ResolutionReason.EXACT_ISIN_MATCH
    if id_type == ResolverIdentifierType.WKN:
        return ResolutionReason.EXACT_WKN_MATCH
    if id_type in (ResolverIdentifierType.TICKER, ResolverIdentifierType.TICKER_WITH_LISTING_CONTEXT):
        return ResolutionReason.UNIQUE_TICKER_LISTING_MATCH
    if id_type in (
        ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME,
        ResolverIdentifierType.NORMALIZED_SHARE_CLASS_NAME,
    ):
        return ResolutionReason.UNIQUE_NAME_MATCH
    if id_type == ResolverIdentifierType.BROKER_ALIAS:
        return ResolutionReason.UNIQUE_BROKER_ALIAS_MATCH
    return ResolutionReason.EXACT_CANONICAL_ID_MATCH


def _ambiguous_reason_for_candidates(
    primary_type: ResolverIdentifierType,
    candidates: Sequence[ETFIdentityCandidate],
) -> ResolutionReason:
    if primary_type == ResolverIdentifierType.BROKER_ALIAS:
        return ResolutionReason.MULTIPLE_ALIAS_MATCHES
    if primary_type in (
        ResolverIdentifierType.LEGAL_SHARE_CLASS_NAME,
        ResolverIdentifierType.NORMALIZED_SHARE_CLASS_NAME,
    ):
        return ResolutionReason.MULTIPLE_NAME_MATCHES

    distinct_inst_ids = {c.instrument_identity.canonical_instrument_id for c in candidates}
    if len(distinct_inst_ids) == 1:
        return ResolutionReason.MULTIPLE_SHARE_CLASSES

    distinct_jurisdictions = {c.instrument_identity.regulatory_jurisdiction.value for c in candidates}
    if len(distinct_jurisdictions) > 1:
        return ResolutionReason.MULTIPLE_JURISDICTIONS

    return ResolutionReason.MULTIPLE_LISTINGS_UNRESOLVED
