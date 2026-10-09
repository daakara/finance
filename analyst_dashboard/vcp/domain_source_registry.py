"""ARX VCP Domain Source Registry & Authority Governance.

Sprint 2B Domain-Authority Resolution.
Enforces explicit domain authority provenance, claim scoping, interpretive distance,
and content-usage separation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AuthorityClass(str, Enum):
    PRIMARY_DOMAIN_AUTHORITY = "PRIMARY_DOMAIN_AUTHORITY"
    DIRECT_AUTHOR_STATEMENT = "DIRECT_AUTHOR_STATEMENT"
    SECONDARY_INTERPRETATION = "SECONDARY_INTERPRETATION"
    EDUCATIONAL_SUMMARY = "EDUCATIONAL_SUMMARY"
    ARX_INTERPRETATION = "ARX_INTERPRETATION"
    ARX_EXTENSION = "ARX_EXTENSION"


class ContentUsageRight(str, Enum):
    REFERENCE_PERMITTED = "REFERENCE_PERMITTED"
    CITATION_ONLY = "CITATION_ONLY"
    INTERNAL_METHODOLOGY_USE = "INTERNAL_METHODOLOGY_USE"
    WHOLESALE_REPRODUCTION_FORBIDDEN = "WHOLESALE_REPRODUCTION_FORBIDDEN"
    CHART_IMAGE_EMBEDDING_FORBIDDEN = "CHART_IMAGE_EMBEDDING_FORBIDDEN"


@dataclass(frozen=True)
class DomainClaimRecord:
    claim_id: str
    supported_term: str
    supported_claim: str
    source_location: str  # e.g., "Chapter 5, pp. 69-92"
    exact_interpretive_scope: str
    interpretive_distance: int  # 0 = verbatim, 1 = direct operationalization, 2 = algorithmic derivation, 3 = arx extension
    authority_class: AuthorityClass


@dataclass(frozen=True)
class DomainSource:
    source_id: str
    title: str
    author: str
    publication_type: str  # BOOK, TREATISE, MONOGRAPH, ARX_SPEC
    publication_version_date: str
    accessible_reference: str
    claim_scope: str
    primary_authority_class: AuthorityClass
    content_usage_rights: List[ContentUsageRight]
    claims: List[DomainClaimRecord] = field(default_factory=list)


# Authoritative Registry Definitions
VCP_DOMAIN_SOURCES: List[DomainSource] = [
    DomainSource(
        source_id="SRC-MINERVINI-2013",
        title="Trade Like a Stock Market Wizard: How to Achieve Superperformance in Stocks in Any Market",
        author="Mark Minervini",
        publication_type="BOOK",
        publication_version_date="2013-04-18, McGraw-Hill Education, ISBN 978-0071807227",
        accessible_reference="Library of Congress / Major Academic & Business Libraries",
        claim_scope="Foundational specification of Volatility Contraction Pattern (VCP), Trend Template, Stage 2 criteria, and pivot mechanics.",
        primary_authority_class=AuthorityClass.PRIMARY_DOMAIN_AUTHORITY,
        content_usage_rights=[
            ContentUsageRight.REFERENCE_PERMITTED,
            ContentUsageRight.CITATION_ONLY,
            ContentUsageRight.INTERNAL_METHODOLOGY_USE,
            ContentUsageRight.WHOLESALE_REPRODUCTION_FORBIDDEN,
            ContentUsageRight.CHART_IMAGE_EMBEDDING_FORBIDDEN,
        ],
        claims=[
            DomainClaimRecord(
                claim_id="CLM-MIN-001",
                supported_term="VCP",
                supported_claim="A volatility contraction pattern is a constructive price consolidation where price volatility narrows through a succession of contractions (waves) accompanied by drying volume, leading to supply exhaustion.",
                source_location="Chapter 5, pp. 71-77",
                exact_interpretive_scope="Pattern concept, consolidation wave narrowing, volume contraction.",
                interpretive_distance=0,
                authority_class=AuthorityClass.PRIMARY_DOMAIN_AUTHORITY,
            ),
            DomainClaimRecord(
                claim_id="CLM-MIN-002",
                supported_term="trend template",
                supported_claim="Minervini 8-point trend template requires: 1) Price > 150 & 200 SMA; 2) 150 SMA > 200 SMA; 3) 200 SMA trending up >= 1 mo; 4) 50 SMA > 150 & 200 SMA; 5) Price > 50 SMA; 6) Price >= 30% above 52w low; 7) Price within 25% of 52w high; 8) Relative strength rank >= 70.",
                source_location="Chapter 5, pp. 78-83",
                exact_interpretive_scope="Macro uptrend verification prior to consolidation.",
                interpretive_distance=0,
                authority_class=AuthorityClass.PRIMARY_DOMAIN_AUTHORITY,
            ),
            DomainClaimRecord(
                claim_id="CLM-MIN-003",
                supported_term="progressive tightening",
                supported_claim="Each successive contraction wave exhibits a smaller percentage depth than the previous wave (e.g., 25% then 15% then 8% then 3%).",
                source_location="Chapter 5, pp. 73-75",
                exact_interpretive_scope="Mathematical depth contraction sequence: Depth_k < Depth_{k-1}.",
                interpretive_distance=1,
                authority_class=AuthorityClass.DIRECT_AUTHOR_STATEMENT,
            ),
            DomainClaimRecord(
                claim_id="CLM-MIN-004",
                supported_term="volume dry-up",
                supported_claim="Volume contracts significantly during pullbacks within the base and dries up substantially on the final contraction, indicating seller exhaustion.",
                source_location="Chapter 5, pp. 76-78; Chapter 10, pp. 185-190",
                exact_interpretive_scope="Declining daily volume compared to historical moving average.",
                interpretive_distance=1,
                authority_class=AuthorityClass.DIRECT_AUTHOR_STATEMENT,
            ),
            DomainClaimRecord(
                claim_id="CLM-MIN-005",
                supported_term="pivot",
                supported_claim="The pivot is the optimal entry point, defined by the resistance high of the final narrowest contraction wave ('cheat' or handle high), where risk can be mathematically bounded.",
                source_location="Chapter 5, pp. 84-89",
                exact_interpretive_scope="Exact price level triggering actionable breakout with tight stop.",
                interpretive_distance=1,
                authority_class=AuthorityClass.DIRECT_AUTHOR_STATEMENT,
            ),
            DomainClaimRecord(
                claim_id="CLM-MIN-006",
                supported_term="prior trend",
                supported_claim="A constructive base must be preceded by a prior primary advance of at least +30% to +100% or greater to qualify as continuation rather than random noise.",
                source_location="Chapter 5, pp. 70-71",
                exact_interpretive_scope="Prior trend minimum velocity and magnitude.",
                interpretive_distance=1,
                authority_class=AuthorityClass.DIRECT_AUTHOR_STATEMENT,
            ),
            DomainClaimRecord(
                claim_id="CLM-MIN-007",
                supported_term="contraction sequence",
                supported_claim="Typically consists of 2 to 4 contractions (denoted 2T, 3T, 4T); rarely up to 6T. Base duration spans 2 to 45 weeks.",
                source_location="Chapter 5, pp. 73-76",
                exact_interpretive_scope="Discrete wave counting and consolidation duration limits.",
                interpretive_distance=1,
                authority_class=AuthorityClass.DIRECT_AUTHOR_STATEMENT,
            ),
        ],
    ),
    DomainSource(
        source_id="SRC-MINERVINI-2017",
        title="Think & Trade Like a Champion: The Secrets, Rules & Blunt Truths of a Stock Market Wizard",
        author="Mark Minervini",
        publication_type="BOOK",
        publication_version_date="2017-01-01, Access Alpha Publishing, ISBN 978-0996307932",
        accessible_reference="Library of Congress / Public & Academic Collections",
        claim_scope="Detailed execution mechanics: cheat pivots, cheat entries, risk-to-reward ratios, and stop placement.",
        primary_authority_class=AuthorityClass.PRIMARY_DOMAIN_AUTHORITY,
        content_usage_rights=[
            ContentUsageRight.REFERENCE_PERMITTED,
            ContentUsageRight.CITATION_ONLY,
            ContentUsageRight.INTERNAL_METHODOLOGY_USE,
            ContentUsageRight.WHOLESALE_REPRODUCTION_FORBIDDEN,
            ContentUsageRight.CHART_IMAGE_EMBEDDING_FORBIDDEN,
        ],
        claims=[
            DomainClaimRecord(
                claim_id="CLM-MIN-101",
                supported_term="pivot",
                supported_claim="The 'Cheat' area is a high-handle or pause in the upper third of the base that allows an early entry before the ultimate multi-month base breakout.",
                source_location="Chapter 4, pp. 65-80",
                exact_interpretive_scope="Early pivot identification and tactical entry execution.",
                interpretive_distance=1,
                authority_class=AuthorityClass.DIRECT_AUTHOR_STATEMENT,
            ),
            DomainClaimRecord(
                claim_id="CLM-MIN-102",
                supported_term="failed setup",
                supported_claim="A setup is invalidated if the breakout reverses below the pivot level or breaches the stop loss floor (typically 5% to 8% max loss).",
                source_location="Chapter 6, pp. 110-125",
                exact_interpretive_scope="Invalidation criteria post-breakout.",
                interpretive_distance=1,
                authority_class=AuthorityClass.DIRECT_AUTHOR_STATEMENT,
            ),
        ],
    ),
    DomainSource(
        source_id="SRC-WEINSTEIN-1988",
        title="Secrets for Profiting in Bull and Bear Markets",
        author="Stan Weinstein",
        publication_type="BOOK",
        publication_version_date="1988-03-01, Dow Jones-Irwin, ISBN 978-1556230790",
        accessible_reference="Major Academic and Financial Collections",
        claim_scope="Authoritative 4-Stage market lifecycle: Stage 1 (Basing), Stage 2 (Advancing), Stage 3 (Topping), Stage 4 (Declining).",
        primary_authority_class=AuthorityClass.PRIMARY_DOMAIN_AUTHORITY,
        content_usage_rights=[
            ContentUsageRight.REFERENCE_PERMITTED,
            ContentUsageRight.CITATION_ONLY,
            ContentUsageRight.INTERNAL_METHODOLOGY_USE,
            ContentUsageRight.WHOLESALE_REPRODUCTION_FORBIDDEN,
            ContentUsageRight.CHART_IMAGE_EMBEDDING_FORBIDDEN,
        ],
        claims=[
            DomainClaimRecord(
                claim_id="CLM-WEI-201",
                supported_term="Stage 1",
                supported_claim="Stage 1 Basing Area: The asset moves sideways in a consolidation range following a major decline; 30-week (150/200 day) MA flattens.",
                source_location="Chapter 2, pp. 11-17",
                exact_interpretive_scope="Dormant accumulation base, flat moving average.",
                interpretive_distance=0,
                authority_class=AuthorityClass.PRIMARY_DOMAIN_AUTHORITY,
            ),
            DomainClaimRecord(
                claim_id="CLM-WEI-202",
                supported_term="Stage 2",
                supported_claim="Stage 2 Advancing Phase: Breakout above resistance of Stage 1 base on heavy volume; price trades above rising 30-week / 200-day moving average.",
                source_location="Chapter 2, pp. 18-24",
                exact_interpretive_scope="Secular uptrend, expanding volume on rallies, higher highs and higher lows.",
                interpretive_distance=0,
                authority_class=AuthorityClass.PRIMARY_DOMAIN_AUTHORITY,
            ),
            DomainClaimRecord(
                claim_id="CLM-WEI-203",
                supported_term="Stage 3",
                supported_claim="Stage 3 The Top Area: Upward momentum stalls; volume is often heavy with churning; 30-week / 200-day moving average flattens and price dips below it.",
                source_location="Chapter 2, pp. 25-29",
                exact_interpretive_scope="Distribution, flattening moving average, high volatility.",
                interpretive_distance=0,
                authority_class=AuthorityClass.PRIMARY_DOMAIN_AUTHORITY,
            ),
            DomainClaimRecord(
                claim_id="CLM-WEI-204",
                supported_term="Stage 4",
                supported_claim="Stage 4 The Declining Phase: Breakdown below Stage 3 support; price trades below declining 30-week / 200-day moving average; avoid or short.",
                source_location="Chapter 2, pp. 30-36",
                exact_interpretive_scope="Markdown, secular downtrend, lower highs and lower lows.",
                interpretive_distance=0,
                authority_class=AuthorityClass.PRIMARY_DOMAIN_AUTHORITY,
            ),
        ],
    ),
    DomainSource(
        source_id="SRC-ONEIL-2009",
        title="How to Make Money in Stocks: A Winning System in Good Times and Bad, 4th Edition",
        author="William J. O'Neil",
        publication_type="BOOK",
        publication_version_date="2009-06-08, McGraw-Hill, ISBN 978-0071625722",
        accessible_reference="Major Academic and Public Libraries",
        claim_scope="Cup-with-Handle, base duration, volume dry-up, institutional accumulation signatures.",
        primary_authority_class=AuthorityClass.SECONDARY_INTERPRETATION,
        content_usage_rights=[
            ContentUsageRight.REFERENCE_PERMITTED,
            ContentUsageRight.CITATION_ONLY,
            ContentUsageRight.INTERNAL_METHODOLOGY_USE,
            ContentUsageRight.WHOLESALE_REPRODUCTION_FORBIDDEN,
            ContentUsageRight.CHART_IMAGE_EMBEDDING_FORBIDDEN,
        ],
        claims=[
            DomainClaimRecord(
                claim_id="CLM-ONL-301",
                supported_term="base",
                supported_claim="Constructive basing patterns require minimum 7 weeks of consolidation (or 3-4 weeks for flat bases) with quiet, dried-up volume on the lows.",
                source_location="Chapter 1, pp. 11-45",
                exact_interpretive_scope="Base duration and handle volume behavior.",
                interpretive_distance=1,
                authority_class=AuthorityClass.SECONDARY_INTERPRETATION,
            ),
            DomainClaimRecord(
                claim_id="CLM-ONL-302",
                supported_term="breakout",
                supported_claim="A breakout occurs when price crosses above pivot resistance on volume expanding by at least 40% to 50% above the 50-day average volume.",
                source_location="Chapter 1, pp. 46-55",
                exact_interpretive_scope="Breakout confirmation volume threshold.",
                interpretive_distance=1,
                authority_class=AuthorityClass.SECONDARY_INTERPRETATION,
            ),
        ],
    ),
    DomainSource(
        source_id="SRC-ARX-SPEC-2026",
        title="ARX Terminal Quantitative VCP & Stage Operationalization Specification",
        author="ARX Research & Architecture Team",
        publication_type="ARX_SPEC",
        publication_version_date="2026-10-09, Specification ARX-EXT-2026-VCP-01",
        accessible_reference="Repository internal: docs/domain/vcp/",
        claim_scope="Algorithmic operationalization of qualitative discretionary VCP criteria into deterministic computational predicates.",
        primary_authority_class=AuthorityClass.ARX_EXTENSION,
        content_usage_rights=[
            ContentUsageRight.REFERENCE_PERMITTED,
            ContentUsageRight.CITATION_ONLY,
            ContentUsageRight.INTERNAL_METHODOLOGY_USE,
        ],
        claims=[
            DomainClaimRecord(
                claim_id="CLM-ARX-401",
                supported_term="candidate contraction",
                supported_claim="A candidate contraction is an algorithmic wave bounded by a local peak and subsequent trough detected using a rolling peak-detection window over daily OHLCV.",
                source_location="Section 8, Predicate PRED_CONTRACTION_EXISTS",
                exact_interpretive_scope="Deterministic wave segment identification.",
                interpretive_distance=2,
                authority_class=AuthorityClass.ARX_EXTENSION,
            ),
            DomainClaimRecord(
                claim_id="CLM-ARX-402",
                supported_term="valid contraction",
                supported_claim="A candidate contraction is valid if its depth is >= 2% and <= 45%, and duration is >= 4 trading sessions.",
                source_location="Section 8, Predicate PRED_CONTRACTION_SEQUENCE_VALID",
                exact_interpretive_scope="Threshold bounds for noise elimination.",
                interpretive_distance=2,
                authority_class=AuthorityClass.ARX_EXTENSION,
            ),
            DomainClaimRecord(
                claim_id="CLM-ARX-403",
                supported_term="insufficient evidence",
                supported_claim="State assigned when historical price/volume data is fewer than required lookback sessions (200 sessions for Trend Template, 50 for volume).",
                source_location="Section 9, Predicate Result Model",
                exact_interpretive_scope="Fail-closed evaluation state.",
                interpretive_distance=1,
                authority_class=AuthorityClass.ARX_INTERPRETATION,
            ),
            DomainClaimRecord(
                claim_id="CLM-ARX-404",
                supported_term="domain unresolved",
                supported_claim="State assigned when data is complete but ambiguous market structure prevents affirmative classification under frozen predicates.",
                source_location="Section 9, Predicate Result Model",
                exact_interpretive_scope="Fail-closed evaluation state.",
                interpretive_distance=1,
                authority_class=AuthorityClass.ARX_INTERPRETATION,
            ),
            DomainClaimRecord(
                claim_id="CLM-ARX-405",
                supported_term="not applicable",
                supported_claim="State assigned to predicates that are conditional on predecessor criteria that failed.",
                source_location="Section 9, Predicate Result Model",
                exact_interpretive_scope="Conditional evaluation bypass.",
                interpretive_distance=1,
                authority_class=AuthorityClass.ARX_INTERPRETATION,
            ),
        ],
    ),
]


class VCPDomainSourceRegistry:
    """Singleton authority registry for VCP domain sources."""

    REGISTRY_ID = "ARX_VCP_DOMAIN_SOURCE_REGISTRY"
    VERSION = "1.0.0"

    def __init__(self, sources: Optional[List[DomainSource]] = None):
        self.sources: Dict[str, DomainSource] = {
            s.source_id: s for s in (sources or VCP_DOMAIN_SOURCES)
        }

    def get_source(self, source_id: str) -> Optional[DomainSource]:
        return self.sources.get(source_id)

    def get_claims_for_term(self, term: str) -> List[DomainClaimRecord]:
        clean_term = term.strip().lower()
        matched: List[DomainClaimRecord] = []
        for src in self.sources.values():
            for clm in src.claims:
                if clm.supported_term.strip().lower() == clean_term:
                    matched.append(clm)
        return matched

    def verify_term_authority(self, term: str) -> Dict[str, Any]:
        claims = self.get_claims_for_term(term)
        if not claims:
            return {
                "term": term,
                "has_authority": False,
                "authority_class": "NONE",
                "claim_count": 0,
            }
        # Best authority class
        has_primary = any(
            c.authority_class in (
                AuthorityClass.PRIMARY_DOMAIN_AUTHORITY,
                AuthorityClass.DIRECT_AUTHOR_STATEMENT,
            )
            for c in claims
        )
        return {
            "term": term,
            "has_authority": True,
            "has_primary": has_primary,
            "authority_class": claims[0].authority_class.value,
            "claim_count": len(claims),
        }

    def compute_registry_hash(self) -> str:
        serialized = []
        for s_id in sorted(self.sources.keys()):
            src = self.sources[s_id]
            s_dict = {
                "source_id": src.source_id,
                "title": src.title,
                "author": src.author,
                "pub_type": src.publication_type,
                "pub_date": src.publication_version_date,
                "claim_scope": src.claim_scope,
                "authority_class": src.primary_authority_class.value,
                "rights": sorted([r.value for r in src.content_usage_rights]),
                "claims": sorted([
                    {
                        "claim_id": c.claim_id,
                        "term": c.supported_term,
                        "location": c.source_location,
                        "scope": c.exact_interpretive_scope,
                        "dist": c.interpretive_distance,
                        "auth": c.authority_class.value,
                    }
                    for c in src.claims
                ], key=lambda x: x["claim_id"]),
            }
            serialized.append(s_dict)
        data_bytes = json.dumps(serialized, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()

    def export_dict(self) -> Dict[str, Any]:
        return {
            "registry_id": self.REGISTRY_ID,
            "version": self.VERSION,
            "hash": self.compute_registry_hash(),
            "sources": [asdict(s) for s in self.sources.values()],
        }
