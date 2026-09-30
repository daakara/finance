"""
scripts/research/etf_v2/ucits_reconciliation_pipeline.py

Deterministic UCITS Evidence Reconciliation Pipeline (Pipeline V2 Wave 4).

Maps and reconciles ExtractedUCITSEvidence and UCITSSourceProvenanceRecord into
canonical three-tier identity models (ETFInstrument -> ETFShareClass -> ETFListing)
and produces Wave 3 AuthorityShareClassEntry instances.

Enforces:
- Strict field ownership across three levels (Instrument -> ShareClass -> Listing)
- TICKER_IS_GLOBAL_CANONICAL_ID = False
- WKN_GLOBAL_CANONICAL_ID = False
- BROKER_ALIAS_IS_CANONICAL_AUTHORITY = False
- ZERO product-specific branching
- Fail-closed ambiguity and contradiction handling across primary authority sources
- Multi-venue listing reconciliation across distinct regulated markets
- Multi-authority corroboration preserving all backing provenance references
- Deterministic sorting of listings, provenance references, and entries
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import datetime
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .global_identity_models import (
    ETFInstrument,
    ETFListing,
    ETFShareClass,
    IdentifierType,
    IdentityStatus,
    InvalidIdentifierError,
    Jurisdiction,
    UnsupportedJurisdictionError,
)
from .global_identifier_authority import (
    generate_instrument_id,
    generate_listing_id,
    generate_share_class_id,
    normalize_isin,
    validate_isin,
    validate_mic,
    validate_wkn,
)
from .global_identity_resolver import (
    AuthorityShareClassEntry,
    UCITSResolverAuthorityAdapter,
)
from .global_identity_resolver_models import (
    ProvenanceReference,
    ResolverIdentifierType,
    sort_listings_deterministically,
    sort_provenance_deterministically,
)
from .ucits_acquisition_models import (
    ExtractedListingEvidence,
    ExtractedUCITSEvidence,
    TemporalMetadata,
    TemporalScope,
)
from .ucits_provenance_models import (
    ETFSourceAuthorityError,
    ProvenanceConflictError,
    ProvenanceValidationError,
    UCITSSourceProvenanceRecord,
    WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS,
)


class AuthorityConflictError(ETFSourceAuthorityError):
    """Raised when contradictory primary authority records exist for the same share class ISIN."""
    pass


class UCITSReconciliationPipeline:
    """
    Deterministic reconciliation pipeline aggregating extracted UCITS evidence
    and official provenance records into canonical three-tier identity models
    and Wave 3 AuthorityShareClassEntry objects.
    """

    def __init__(self, adapter_id: str = "eu_ucits_statutory_adapter") -> None:
        self.adapter_id = adapter_id

    def reconcile_extracted_evidence(
        self,
        evidence_batch: Sequence[ExtractedUCITSEvidence],
        provenance_records: Optional[Sequence[UCITSSourceProvenanceRecord]] = None,
    ) -> Tuple[AuthorityShareClassEntry, ...]:
        """
        Reconciles a batch of extracted evidence records into canonical AuthorityShareClassEntry objects.
        Fails closed on contradictory evidence for the same ISIN.
        """
        if not evidence_batch:
            return ()

        # Group evidence records by normalized ISIN
        by_isin: Dict[str, List[ExtractedUCITSEvidence]] = defaultdict(list)
        for ev in evidence_batch:
            norm_isin = normalize_isin(ev.share_class_isin)
            if not validate_isin(norm_isin):
                raise InvalidIdentifierError(f"Evidence contains invalid ISIN: {ev.share_class_isin!r}")
            by_isin[norm_isin].append(ev)

        # Index provenance records by ISIN and document hash
        prov_by_isin: Dict[str, List[UCITSSourceProvenanceRecord]] = defaultdict(list)
        if provenance_records:
            for prec in provenance_records:
                p_isin = normalize_isin(prec.share_class_isin)
                prov_by_isin[p_isin].append(prec)

        entries: List[AuthorityShareClassEntry] = []

        for isin in sorted(by_isin.keys()):
            records = by_isin[isin]
            primary_ev = records[0]
            domicile = isin[:2]

            if domicile not in WAVE_2_SUPPORTED_SOURCE_JURISDICTIONS:
                raise UnsupportedJurisdictionError(
                    f"Unsupported UCITS domicile '{domicile}' for share class {isin}."
                )

            # Check for authoritative contradictions across corroborating records
            for other in records[1:]:
                if other.legal_domicile != primary_ev.legal_domicile:
                    raise AuthorityConflictError(
                        f"Contradictory legal domicile for ISIN {isin}: "
                        f"{primary_ev.legal_domicile} vs {other.legal_domicile}"
                    )
                if other.distribution_policy != primary_ev.distribution_policy:
                    raise AuthorityConflictError(
                        f"Contradictory distribution policy for ISIN {isin}: "
                        f"{primary_ev.distribution_policy} vs {other.distribution_policy}"
                    )
                if other.share_class_currency != primary_ev.share_class_currency:
                    raise AuthorityConflictError(
                        f"Contradictory share class currency for ISIN {isin}: "
                        f"{primary_ev.share_class_currency} vs {other.share_class_currency}"
                    )
                # Verify fund name compatibility
                if (
                    other.sub_fund_legal_name != primary_ev.sub_fund_legal_name
                    and other.sub_fund_legal_name != "UCITS_SUB_FUND"
                    and primary_ev.sub_fund_legal_name != "UCITS_SUB_FUND"
                ):
                    raise AuthorityConflictError(
                        f"Contradictory sub-fund legal names for ISIN {isin}: "
                        f"'{primary_ev.sub_fund_legal_name}' vs '{other.sub_fund_legal_name}'"
                    )

            # Derive canonical instrument ID
            instrument_id = generate_instrument_id(
                jurisdiction=Jurisdiction.EU_UCITS,
                domicile_iso2=domicile,
                root_id=primary_ev.legal_umbrella_name,
            )

            # Derive canonical share class ID
            share_class_id = generate_share_class_id(
                scheme=IdentifierType.ISIN,
                identifier=isin,
            )

            # Reconcile listings across all corroborating evidence
            collected_listings: List[ETFListing] = []
            seen_listing_keys: Set[Tuple[str, str, str]] = set()

            for rec in records:
                for list_ev in rec.listings:
                    validate_mic(list_ev.mic)
                    dedup_key = (list_ev.ticker, list_ev.mic, list_ev.trading_currency)
                    if dedup_key in seen_listing_keys:
                        continue
                    seen_listing_keys.add(dedup_key)

                    listing_id = generate_listing_id(
                        venue_mic=list_ev.mic,
                        ticker=list_ev.ticker,
                        trading_currency=list_ev.trading_currency,
                    )
                    collected_listings.append(
                        ETFListing(
                            listing_id=listing_id,
                            share_class_id=share_class_id,
                            venue_mic=list_ev.mic,
                            ticker=list_ev.ticker,
                            trading_currency=list_ev.trading_currency,
                            identity_status=IdentityStatus.RESOLVED,
                        )
                    )

            sorted_listings = sort_listings_deterministically(collected_listings)

            # Reconcile WKNs
            wkn_set: Set[str] = set()
            for rec in records:
                if rec.wkn:
                    wkn_clean = rec.wkn.strip().upper()
                    validate_wkn(wkn_clean)
                    wkn_set.add(wkn_clean)
            sorted_wkns = tuple(sorted(wkn_set))

            # Reconcile Provenance References
            prov_refs: List[ProvenanceReference] = []
            backing_precords = prov_by_isin.get(isin, [])

            if backing_precords:
                for prec in backing_precords:
                    prov_refs.append(
                        ProvenanceReference(
                            adapter_id=self.adapter_id,
                            authority_jurisdiction=Jurisdiction.EU_UCITS.value,
                            authority_source=prec.primary_regulator or "EU_UCITS_STATUTORY",
                            matched_identifier=isin,
                            matched_identifier_type=ResolverIdentifierType.ISIN.value,
                            source_record_id=f"UCITS_PROV:{prec.legal_domicile}:{prec.share_class_isin}:{prec.document_type}:{prec.raw_sha256[:16]}",
                            source_document_hash=prec.raw_sha256,
                            outcome="COMPLETED_MATCH",
                        )
                    )
            else:
                for rec in records:
                    h_val = rec.source_provenance_sha256
                    prov_refs.append(
                        ProvenanceReference(
                            adapter_id=self.adapter_id,
                            authority_jurisdiction=Jurisdiction.EU_UCITS.value,
                            authority_source=f"UCITS_{domicile}_STATUTORY",
                            matched_identifier=isin,
                            matched_identifier_type=ResolverIdentifierType.ISIN.value,
                            source_record_id=f"UCITS_EVIDENCE:{domicile}:{isin}:{h_val[:16] if h_val else 'EXTRACTED'}",
                            source_document_hash=h_val or None,
                            outcome="COMPLETED_MATCH",
                        )
                    )

            sorted_prov = sort_provenance_deterministically(prov_refs)

            instrument = ETFInstrument(
                canonical_instrument_id=instrument_id,
                legal_fund_name=primary_ev.legal_umbrella_name,
                domicile_iso2=domicile,
                regulatory_jurisdiction=Jurisdiction.EU_UCITS,
                fund_family=primary_ev.management_company,
                issuer=primary_ev.management_company,
                fund_structure="UCITS",
                identity_status=IdentityStatus.RESOLVED,
            )

            share_class = ETFShareClass(
                share_class_id=share_class_id,
                instrument_id=instrument_id,
                isin=isin,
                share_class_name=primary_ev.share_class_legal_name,
                distribution_policy=primary_ev.distribution_policy,
                base_currency=primary_ev.share_class_currency,
                hedging_policy="UNHEDGED",
                identity_status=IdentityStatus.RESOLVED,
                listings=sorted_listings,
            )

            entry = AuthorityShareClassEntry(
                instrument=instrument,
                share_class=share_class,
                listings=sorted_listings,
                provenance_references=sorted_prov,
                wkn_codes=sorted_wkns,
            )
            entries.append(entry)

        return tuple(entries)

    def build_resolver_adapter(
        self,
        entries: Sequence[AuthorityShareClassEntry],
    ) -> UCITSResolverAuthorityAdapter:
        """
        Instantiates a Wave 3 UCITSResolverAuthorityAdapter populated with reconciled entries.
        Verifies direct compatibility without any Wave 3 adapter contract changes.
        """
        return UCITSResolverAuthorityAdapter(
            adapter_id=self.adapter_id,
            entries=entries,
        )
