"""
analyst_dashboard/security_master/source_governance_policy.py

Versioned Authority-Scope Contracts, Normalization Policies,
Field Authority Resolution, and S0–S4 Classification for ARX Security Master.

Invariants Enforced:
- SOURCE_AVAILABLE != SOURCE_AUTHORITATIVE_FOR_ALL_FIELDS
- IMPLICIT_FALLBACKS = PROHIBITED (implicit fallback count = 0)
- Admissibility evaluated BEFORE precedence.
- S2_DEGRADED iff frozen applicable policy yields exactly one authoritative result.
- S3_BLOCKING iff no unique deterministic result exists for a material conflict.
- SYMBOL_IS_IDENTITY = NO
- ALPACA_UUID_IS_UNIVERSAL_CROSS_PROVIDER_IDENTITY = NO
"""

from __future__ import annotations

import re
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field, ConfigDict

from .source_governance_models import (
    canonical_hash,
    canonical_json_dumps,
    ConflictSeverity,
    ConflictResolution,
    ReasonCode,
    SourceRole,
)


class AuthorityGraphValidationError(RuntimeError):
    """Raised when an authority graph contains cycles, duplicate ranks, or undefined sources."""
    pass


# =====================================================================
# MIC & Exchange Normalization Taxonomy (ISO 10383)
# =====================================================================

CANONICAL_MIC_TABLE: Dict[str, str] = {
    "NASDAQ": "XNAS",
    "XNAS": "XNAS",
    "NYSE": "XNYS",
    "XNYS": "XNYS",
    "ARCA": "ARCX",
    "ARCX": "ARCX",
    "BATS": "BATS",
    "BATS EXCHANGE": "BATS",
    "AMEX": "XASE",
    "XASE": "XASE",
    "NYSE AMERICAN": "XASE",
    "OTC": "OTCM",
    "OTCM": "OTCM",
    "PINK": "OTCM",
    "IEX": "IEXG",
    "IEXG": "IEXG",
}


def normalize_exchange_mic(raw_exchange: Optional[str]) -> str:
    """Normalizes raw exchange string into canonical ISO 10383 Operating MIC."""
    if not raw_exchange:
        return "UNKNOWN"
    clean = raw_exchange.strip().upper()
    return CANONICAL_MIC_TABLE.get(clean, "UNKNOWN")


# =====================================================================
# Symbol Normalization
# =====================================================================

UNICODE_LOOKALIKES = {
    "\u2010": "-",  # Hyphen
    "\u2011": "-",  # Non-breaking hyphen
    "\u2012": "-",  # Figure dash
    "\u2013": "-",  # En dash
    "\u2014": "-",  # Em dash
    "\u2212": "-",  # Minus sign
    "\uFF0D": "-",  # Fullwidth hyphen-minus
    "\u00A0": " ",  # Non-breaking space
    "\u200B": "",   # Zero-width space
    "\uFEFF": "",   # Zero-width no-break space
}

CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def normalize_symbol_string(raw_symbol: Optional[str]) -> str:
    """
    Hostile-input safe symbol normalization.
    Strips whitespace, control characters, maps unicode lookalikes, and uppercases.
    Preserves valid share-class notation (e.g. BRK.B).
    """
    if not raw_symbol:
        return ""
    s = raw_symbol
    # Replace unicode lookalikes
    for u_char, repl in UNICODE_LOOKALIKES.items():
        s = s.replace(u_char, repl)
    # Strip control characters
    s = CONTROL_CHAR_RE.sub("", s)
    s = s.strip().upper()
    # Normalize share class delimiters (slash or dash to standard period)
    # e.g. BRK/B -> BRK.B, BRK-B -> BRK.B
    s = re.sub(r"([A-Z0-9]+)[/-]([A-Z0-9]+)", r"\1.\2", s)
    return s


# =====================================================================
# Security Type Taxonomy (Sprint 2A Frozen)
# =====================================================================

CANONICAL_SECURITY_TYPES: Set[str] = {
    "COMMON_STOCK",
    "ETF",
    "ADR",
    "REIT",
    "PREFERRED",
    "WARRANT",
    "UNIT",
    "RIGHT",
    "CRYPTO",
    "OTHER",
    "UNKNOWN",
}


# =====================================================================
# Field Authority Policy Definitions
# =====================================================================

class SingleFieldPolicy(BaseModel):
    """Governed authority configuration for a single canonical field."""
    field_policy_id: str
    canonical_field: str
    authority_chain: List[str]  # Ordered source IDs by strict precedence
    resolution_mode: str = "STRICT_PRECEDENCE"  # STRICT_PRECEDENCE, COMPOSITE, CORROBORATED
    missing_authority_behavior: str = "FAIL_CLOSED_UNRESOLVED"
    allowed_values: Optional[List[str]] = None
    implicit_fallback: bool = False  # MUST be False

    model_config = ConfigDict(frozen=True)


class FieldAuthorityPolicyRegistry:
    """
    Frozen Field Authority Policy Registry for ARX Security Master Sprint 2A.
    Version: 1.0.0
    """
    POLICY_ID = "ARX_SOURCE_GOV_POLICY"
    POLICY_VERSION = "1.0.0"

    POLICIES: Dict[str, SingleFieldPolicy] = {
        "symbol": SingleFieldPolicy(
            field_policy_id="POL_SYMBOL_V1",
            canonical_field="symbol",
            authority_chain=["ALPACA_ASSET_DIRECTORY", "OPENFIGI_V3_MAPPING"],
            resolution_mode="STRICT_PRECEDENCE",
            missing_authority_behavior="FAIL_CLOSED_UNRESOLVED",
            implicit_fallback=False,
        ),
        "primary_exchange": SingleFieldPolicy(
            field_policy_id="POL_EXCHANGE_V1",
            canonical_field="primary_exchange",
            authority_chain=["ALPACA_ASSET_DIRECTORY", "OPENFIGI_V3_MAPPING"],
            resolution_mode="STRICT_PRECEDENCE",
            missing_authority_behavior="FAIL_CLOSED_UNRESOLVED",
            implicit_fallback=False,
        ),
        "listing_status": SingleFieldPolicy(
            field_policy_id="POL_LISTING_STATUS_V1",
            canonical_field="listing_status",
            authority_chain=["ALPACA_ASSET_DIRECTORY"],
            resolution_mode="STRICT_PRECEDENCE",
            missing_authority_behavior="FAIL_CLOSED_UNRESOLVED",
            allowed_values=["ACTIVE", "INACTIVE", "DELISTED", "SUSPENDED", "UNKNOWN"],
            implicit_fallback=False,
        ),
        "security_type": SingleFieldPolicy(
            field_policy_id="POL_SECURITY_TYPE_V1",
            canonical_field="security_type",
            authority_chain=["OPENFIGI_V3_MAPPING"],  # Alpaca us_equity cannot establish subtype
            resolution_mode="STRICT_PRECEDENCE",
            missing_authority_behavior="RESOLVE_AS_UNKNOWN",
            allowed_values=sorted(list(CANONICAL_SECURITY_TYPES)),
            implicit_fallback=False,
        ),
        "share_class": SingleFieldPolicy(
            field_policy_id="POL_SHARE_CLASS_V1",
            canonical_field="share_class",
            authority_chain=["OPENFIGI_V3_MAPPING"],
            resolution_mode="STRICT_PRECEDENCE",
            missing_authority_behavior="NONE",
            implicit_fallback=False,
        ),
        "issuer_identity": SingleFieldPolicy(
            field_policy_id="POL_ISSUER_ID_V1",
            canonical_field="issuer_identity",
            authority_chain=["SEC_EDGAR", "OPENFIGI_V3_MAPPING"],
            resolution_mode="STRICT_PRECEDENCE",
            missing_authority_behavior="FAIL_CLOSED_UNRESOLVED",
            implicit_fallback=False,
        ),
        "corporate_action_state": SingleFieldPolicy(
            field_policy_id="POL_CORP_ACTION_V1",
            canonical_field="corporate_action_state",
            authority_chain=[],  # CORPORATE_ACTION_AUTHORITY = NOT_ESTABLISHED
            resolution_mode="STRICT_PRECEDENCE",
            missing_authority_behavior="UNRESOLVED",
            implicit_fallback=False,
        ),
    }

    @classmethod
    def validate_authority_graph(cls) -> bool:
        """
        Validates graph properties:
        - No duplicate precedence ranks.
        - No cycles.
        - All sources are known.
        - Implicit fallback is strictly prohibited (0 implicit fallbacks).
        """
        known_sources = {
            "ALPACA_ASSET_DIRECTORY",
            "OPENFIGI_V3_MAPPING",
            "SEC_EDGAR",
        }

        for field_name, policy in cls.POLICIES.items():
            if policy.implicit_fallback:
                raise AuthorityGraphValidationError(
                    f"Policy {policy.field_policy_id} for {field_name} specifies implicit_fallback=True (prohibited)."
                )
            # Check unique precedence ranks
            if len(policy.authority_chain) != len(set(policy.authority_chain)):
                raise AuthorityGraphValidationError(
                    f"Policy {policy.field_policy_id} contains duplicate precedence ranks in authority_chain."
                )
            # Check unknown sources
            for src in policy.authority_chain:
                if src not in known_sources:
                    raise AuthorityGraphValidationError(
                        f"Policy {policy.field_policy_id} references unknown source ID: {src}."
                    )

        return True

    @classmethod
    def compute_policy_hash(cls) -> str:
        """Computes deterministic hash across all registered field policies."""
        cls.validate_authority_graph()
        payload = {
            "policy_id": cls.POLICY_ID,
            "policy_version": cls.POLICY_VERSION,
            "policies": {k: p.model_dump() for k, p in sorted(cls.POLICIES.items())},
            "mic_table": sorted(list(CANONICAL_MIC_TABLE.items())),
            "security_types": sorted(list(CANONICAL_SECURITY_TYPES)),
        }
        return canonical_hash(payload)


# =====================================================================
# S0–S4 Severity Classifier (Pure & Deterministic)
# =====================================================================

class SourceConflictClassifier:
    """
    Pure, deterministic classifier mapping field-level multi-source states
    into S0–S4 severity categories according to Sprint 2A Section 19.

    Invariants:
    - S0_INFO: Cosmetic / non-semantic differences (e.g. whitespace, case, alias).
    - S1_WARNING: Genuine disagreement with no material downstream impact (e.g. non-material descriptive text).
    - S2_DEGRADED: Material disagreement resolved uniquely and deterministically by frozen policy.
    - S3_BLOCKING: Material disagreement with no unique deterministic governed resolution.
    - S4_INTEGRITY_FAILURE: Evidence/population construction itself cannot be trusted.
    """

    MATERIAL_FIELDS = {
        "symbol",
        "primary_exchange",
        "listing_status",
        "security_type",
        "canonical_security_id",
        "canonical_listing_id",
    }

    @classmethod
    def classify_conflict(
        cls,
        field_name: str,
        value_a: Any,
        value_b: Any,
        policy: Optional[SingleFieldPolicy] = None,
        source_a: str = "",
        source_b: str = "",
        evidence_corrupt: bool = False,
    ) -> Tuple[ConflictSeverity, ConflictResolution, Any]:
        """
        Evaluates disagreement between source_a and source_b and returns:
        (severity, resolution, resolved_canonical_value)
        """
        if evidence_corrupt:
            return ConflictSeverity.S4_INTEGRITY_FAILURE, ConflictResolution.UNRESOLVED, None

        # S0 check: Identical after basic normalization
        norm_a = str(value_a).strip().upper() if value_a is not None else ""
        norm_b = str(value_b).strip().upper() if value_b is not None else ""
        if norm_a == norm_b:
            return ConflictSeverity.S0_INFO, ConflictResolution.AGREED, value_a

        # If values disagree:
        is_material = field_name in cls.MATERIAL_FIELDS

        if not is_material:
            # Genuine disagreement on non-material field
            # Use source with precedence if available
            res_val = value_a if value_a is not None else value_b
            return ConflictSeverity.S1_WARNING, ConflictResolution.RESOLVED_BY_PRECEDENCE, res_val

        # Material disagreement:
        if policy is None:
            # No policy exists to resolve material disagreement -> S3 Blocking
            return ConflictSeverity.S3_BLOCKING, ConflictResolution.UNRESOLVED, None

        chain = policy.authority_chain
        # Check if one source has clear strictly higher precedence in frozen chain
        rank_a = chain.index(source_a) if source_a in chain else 9999
        rank_b = chain.index(source_b) if source_b in chain else 9999

        if rank_a < rank_b and value_a is not None:
            # Unique authoritative winner
            return ConflictSeverity.S2_DEGRADED, ConflictResolution.RESOLVED_BY_PRECEDENCE, value_a
        elif rank_b < rank_a and value_b is not None:
            # Unique authoritative winner
            return ConflictSeverity.S2_DEGRADED, ConflictResolution.RESOLVED_BY_PRECEDENCE, value_b
        else:
            # Equal rank or neither source in authority chain -> S3 Blocking
            return ConflictSeverity.S3_BLOCKING, ConflictResolution.UNRESOLVED, None
