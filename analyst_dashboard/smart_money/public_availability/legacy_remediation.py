"""
Legacy PIT Consumer Audit & Remediation Authority — Epoch 001.
Categorizes historical code paths using public_availability_at, prevents retroactive data mutation,
and provides explicit semantic accessors (sec_acceptance_at, conservative_knowledge_time).
Enforces HISTORICAL_PIT_CONTAMINATION_RISK = YES where acceptance time was treated as public knowledge time.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple


class LegacyConsumerClass(str, Enum):
    ACCEPTANCE_SEMANTICS_CORRECT = "ACCEPTANCE_SEMANTICS_CORRECT"
    PUBLIC_KNOWLEDGE_SEMANTICS_INCORRECT = "PUBLIC_KNOWLEDGE_SEMANTICS_INCORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    DEAD_CODE = "DEAD_CODE"


@dataclass(frozen=True)
class LegacyConsumerAuditEntry:
    code_path: str
    symbol_or_function: str
    legacy_field_used: str
    actual_source_semantic: str
    classification: LegacyConsumerClass
    remediation_status: str
    notes: str


# Comprehensive Audit Registry of all legacy public_availability_at consumers
LEGACY_CONSUMER_REGISTRY: Tuple[LegacyConsumerAuditEntry, ...] = (
    LegacyConsumerAuditEntry(
        code_path="analyst_dashboard/smart_money/form4_models.py",
        symbol_or_function="SmartMoneyTransaction.__post_init__",
        legacy_field_used="public_availability_at",
        actual_source_semantic="SEC_ACCEPTANCE_DATETIME",
        classification=LegacyConsumerClass.PUBLIC_KNOWLEDGE_SEMANTICS_INCORRECT,
        remediation_status="REMEDIATED_VIA_ACCESSOR_FACADE",
        notes="Defaulted public_availability_at to acceptance_datetime. Remediated via explicit sec_acceptance_at accessor; public knowledge time returns UNKNOWN if no prospective observation.",
    ),
    LegacyConsumerAuditEntry(
        code_path="analyst_dashboard/smart_money/form4_parser.py",
        symbol_or_function="Form4XmlParser.parse_filing",
        legacy_field_used="public_availability_at",
        actual_source_semantic="SEC_ACCEPTANCE_DATETIME",
        classification=LegacyConsumerClass.ACCEPTANCE_SEMANTICS_CORRECT,
        remediation_status="REMEDIATED_VIA_SEMANTIC_LABELING",
        notes="Passed public_avail=acceptance_dt. Verified as raw acceptance timestamp custody.",
    ),
    LegacyConsumerAuditEntry(
        code_path="analyst_dashboard/smart_money/form4_ingestion.py",
        symbol_or_function="query_smart_money_transactions_as_of",
        legacy_field_used="sm_source_availability_ledger.public_availability_at",
        actual_source_semantic="SEC_ACCEPTANCE_DATETIME",
        classification=LegacyConsumerClass.PUBLIC_KNOWLEDGE_SEMANTICS_INCORRECT,
        remediation_status="TAGGED_PIT_CONTAMINATION_RISK",
        notes="Filters transactions as of timestamp using acceptance time. Must not claim strict public knowledge PIT without conservative delta.",
    ),
    LegacyConsumerAuditEntry(
        code_path="analyst_dashboard/smart_money/incremental_engine.py",
        symbol_or_function="compute_source_max_timestamp / check_late_arrival",
        legacy_field_used="sm_source_availability_ledger.public_availability_at",
        actual_source_semantic="SEC_ACCEPTANCE_DATETIME",
        classification=LegacyConsumerClass.ACCEPTANCE_SEMANTICS_CORRECT,
        remediation_status="BOUNDED_SOURCE_WATERMARK",
        notes="Used as internal source arrival watermark. Valid for partition progression, but does not certify true public publication time.",
    ),
    LegacyConsumerAuditEntry(
        code_path="analyst_dashboard/smart_money/policy_bundle.py",
        symbol_or_function="PolicyBundleAuthority.POLICIES",
        legacy_field_used="KNOWLEDGE_TIME text definition",
        actual_source_semantic="SEC_ACCEPTANCE_DATETIME",
        classification=LegacyConsumerClass.PUBLIC_KNOWLEDGE_SEMANTICS_INCORRECT,
        remediation_status="DOCUMENTED_IN_EPOCH_001",
        notes="Prose declared KNOWLEDGE_TIME derived from acceptanceDateTime. Reclassified as acceptance lower bound only.",
    ),
    LegacyConsumerAuditEntry(
        code_path="analyst_dashboard/smart_money/certification_engine.py",
        symbol_or_function="SmartMoneyCertificationEngine._check_source_gap_integrity",
        legacy_field_used="sm_source_availability_ledger.public_availability_at",
        actual_source_semantic="SEC_ACCEPTANCE_DATETIME",
        classification=LegacyConsumerClass.ACCEPTANCE_SEMANTICS_CORRECT,
        remediation_status="GOVERNED_BY_WAVE_2C",
        notes="Verifies partition gap integrity and accession continuity. Valid within governed universe scope.",
    ),
    LegacyConsumerAuditEntry(
        code_path="analyst_dashboard/smart_money/form4_scanner.py",
        symbol_or_function="evaluate_issuer_transactions",
        legacy_field_used="transaction_date (not public_availability_at)",
        actual_source_semantic="FORM4_TRANSACTION_DATE",
        classification=LegacyConsumerClass.ACCEPTANCE_SEMANTICS_CORRECT,
        remediation_status="UNAFFECTED",
        notes="Uses tx.transaction_date for 30-day window lookback; does not claim intraday public knowledge time.",
    ),
)


class LegacyPitSemanticResolver:
    """
    Safely resolves timing semantics for legacy rows without database mutations.
    """

    @staticmethod
    def resolve_timing(
        raw_row_dict: Dict[str, Any],
        prospective_observation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Takes a legacy row dictionary and optionally a prospective observation,
        and outputs explicit decoupled timing semantics.
        """
        # Historical raw value from acceptanceDateTime
        sec_acceptance_at = raw_row_dict.get("acceptance_datetime") or raw_row_dict.get("public_availability_at")

        if prospective_observation:
            first_meta = prospective_observation.get("first_proven_metadata_public_at")
            first_content = prospective_observation.get("first_proven_content_public_at")
            first_decision = prospective_observation.get("first_proven_decision_input_complete_at")
            conservative_kt = prospective_observation.get("conservative_knowledge_time")
        else:
            # Historical records lacking prospective observation: UNKNOWN
            first_meta = "UNKNOWN"
            first_content = "UNKNOWN"
            first_decision = "UNKNOWN"
            conservative_kt = "UNKNOWN"

        return {
            "accession_number": raw_row_dict.get("accession_number"),
            "sec_acceptance_at": sec_acceptance_at,
            "first_proven_metadata_public_at": first_meta,
            "first_proven_content_public_at": first_content,
            "first_proven_decision_input_complete_at": first_decision,
            "conservative_knowledge_time": conservative_kt,
            "exact_first_public_available_at": "UNKNOWN_UNLESS_SEPARATELY_OBSERVED",
            "historical_pit_contamination_risk": True if not prospective_observation else False,
        }
