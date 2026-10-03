"""
scripts/research/etf_v2/openfigi_normalizer.py

Pre-request normalization, regex check, and invariant validator for OpenFIGI mapping jobs.

Invariants Enforced:
- OFIGI-INV-002: ISIN is sole legal identity authority; FIGI cannot replace ISIN.
- OFIGI-INV-004: Operates only on valid canonical inputs; invalid inputs fail closed.
- OFIGI-INV-005: Trims whitespace, enforces uppercase; never synthesizes or repairs invalid ISINs.
- OFIGI-INV-006: Primary request key is strictly ID_ISIN.
- OFIGI-INV-017: A mapping job must not contain both micCode and exchCode.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from .openfigi_models import AuthorizedCanonicalInputRecord, OpenFIGIMappingJob
from .global_identifier_authority import normalize_isin, validate_isin


class InvalidMappingRequestError(ValueError):
    """Raised when an input record violates OpenFIGI mapping specifications or invariants."""
    pass


class OpenFIGINormalizer:
    """Pre-request validator and normalizer for OpenFIGI mapping jobs."""

    @staticmethod
    def validate_and_normalize(
        record: AuthorizedCanonicalInputRecord
    ) -> Tuple[bool, Optional[OpenFIGIMappingJob], Optional[str]]:
        """
        Validates an authorized canonical input record and transforms it into a wire OpenFIGIMappingJob.

        Returns:
            (is_valid, mapping_job, error_message)
        """
        # 1. Enforce OFIGI-INV-017: Mutual exclusivity of micCode and exchCode
        clean_mic = record.mic_code.strip().upper() if record.mic_code else None
        clean_exch = record.exch_code.strip().upper() if record.exch_code else None

        if clean_mic and clean_exch:
            return (
                False,
                None,
                f"OFIGI-INV-017 Violation: Mapping job for ISIN {record.isin} contains both "
                f"micCode ({clean_mic}) and exchCode ({clean_exch}). Fields are mutually exclusive."
            )

        # 2. Enforce ISIN validation and normalization
        raw_isin = record.isin
        if not raw_isin or not isinstance(raw_isin, str):
            return (False, None, f"ISIN must be a non-empty string, got: {raw_isin!r}")

        clean_isin = raw_isin.strip().upper()
        if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", clean_isin):
            return (False, None, f"Malformed ISIN format: {raw_isin!r}")

        # Check ISO 6166 check digit using global identifier authority
        try:
            if not validate_isin(clean_isin, strict=False):
                return (False, None, f"Invalid ISO 6166 check digit for ISIN: {clean_isin}")
        except Exception as e:
            return (False, None, f"ISIN validation error: {str(e)}")

        # 3. Optional currency normalization (ISO 4217 3-letter)
        clean_currency = record.currency.strip().upper() if record.currency else None
        if clean_currency and not re.match(r"^[A-Z]{3}$", clean_currency):
            return (False, None, f"Invalid ISO 4217 currency code: {clean_currency}")

        # 4. Construct wire job
        job = OpenFIGIMappingJob(
            idType="ID_ISIN",
            idValue=clean_isin,
            micCode=clean_mic,
            exchCode=clean_exch,
            currency=clean_currency
        )

        return (True, job, None)
