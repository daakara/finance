"""
analyst_dashboard/research/sourced_fundamentals/identity.py

Attributable Ground Truth Identity Authority for the Sourced Fundamentals Pilot.
Grounded directly in the retained SEC EDGAR submissions files:
- data/research/sourced_fundamentals/raw/*_submissions.json
- data/research/sourced_fundamentals/raw/manifest.json

Provides fail-closed identity verification:
- Strict 10-digit numeric CIK format validation.
- Attributable symbol-to-CIK binding verification against retained primary SEC submissions.
- Strict rejection of mismatches, malformed CIKs, or ungrounded bindings.
"""

import re
from typing import Dict, Any, Optional, NamedTuple

CIK_REGEX = re.compile(r"^\d{10}$")

class IdentityBinding(NamedTuple):
    symbol: str
    cik: str
    company_name: str
    exchange: str


# Authoritative SEC Ground Truth Bindings directly derived from primary Form 10-K/Q submissions index
ATTRIBUTABLE_SEC_IDENTITY_AUTHORITY: Dict[str, IdentityBinding] = {
    "FIX": IdentityBinding(
        symbol="FIX",
        cik="0001035983",
        company_name="COMFORT SYSTEMS USA INC",
        exchange="NYSE",
    ),
    "DUOL": IdentityBinding(
        symbol="DUOL",
        cik="0001562088",
        company_name="Duolingo, Inc.",
        exchange="NASDAQ",
    ),
    "ULTA": IdentityBinding(
        symbol="ULTA",
        cik="0001403568",
        company_name="Ulta Beauty, Inc.",
        exchange="NASDAQ",
    ),
    "MEDP": IdentityBinding(
        symbol="MEDP",
        cik="0001668397",
        company_name="Medpace Holdings, Inc.",
        exchange="NASDAQ",
    ),
    "DECK": IdentityBinding(
        symbol="DECK",
        cik="0000910521",
        company_name="DECKERS OUTDOOR CORP",
        exchange="NYSE",
    ),
    "KLAC": IdentityBinding(
        symbol="KLAC",
        cik="0000319201",
        company_name="KLA CORP",
        exchange="NASDAQ",
    ),
    "LULU": IdentityBinding(
        symbol="LULU",
        cik="0001397187",
        company_name="lululemon athletica inc.",
        exchange="NASDAQ",
    ),
    "CELH": IdentityBinding(
        symbol="CELH",
        cik="0001341766",
        company_name="Celsius Holdings, Inc.",
        exchange="NASDAQ",
    ),
}


class IdentityValidationResult:
    def __init__(self, is_valid: bool, error: Optional[str] = None, binding: Optional[IdentityBinding] = None):
        self.is_valid = is_valid
        self.error = error
        self.binding = binding

    def __bool__(self) -> bool:
        return self.is_valid


def validate_cik_format(cik: Any) -> bool:
    """Verifies that CIK is a valid 10-digit numeric string with leading zeros."""
    if not isinstance(cik, str):
        return False
    return bool(CIK_REGEX.match(cik.strip()))


def get_attributable_binding(symbol: str) -> Optional[IdentityBinding]:
    """Returns attributable SEC ground truth binding for a symbol, if cataloged."""
    if not symbol or not isinstance(symbol, str):
        return None
    return ATTRIBUTABLE_SEC_IDENTITY_AUTHORITY.get(symbol.strip().upper())


def verify_company_identity(
    requested_symbol: str,
    record_symbol: Optional[str],
    record_cik: Optional[str],
) -> IdentityValidationResult:
    """
    Validates company identity against attributable SEC ground truth.
    Fails closed on:
    - Missing or non-string inputs.
    - Malformed CIK format (e.g. 'NOT_A_CIK', non-numeric, not 10 digits).
    - Uncataloged requested symbol.
    - Requested symbol vs record symbol mismatch.
    - Record CIK vs attributable SEC CIK mismatch (e.g. FIX paired with Apple's 0000320193).
    """
    if not requested_symbol or not isinstance(requested_symbol, str):
        return IdentityValidationResult(False, "Requested symbol must be a non-empty string.")

    clean_req_sym = requested_symbol.strip().upper()
    binding = get_attributable_binding(clean_req_sym)

    if binding is None:
        return IdentityValidationResult(False, f"Symbol '{clean_req_sym}' is not cataloged in authoritative SEC pilot authority.")

    if not record_symbol or not isinstance(record_symbol, str):
        return IdentityValidationResult(False, f"Record symbol for '{clean_req_sym}' is missing or invalid.")

    clean_rec_sym = record_symbol.strip().upper()
    if clean_rec_sym != clean_req_sym:
        return IdentityValidationResult(
            False,
            f"Internal symbol mismatch: requested '{clean_req_sym}', but record specifies '{clean_rec_sym}'."
        )

    if not record_cik or not isinstance(record_cik, str):
        return IdentityValidationResult(False, f"Record CIK for '{clean_req_sym}' is missing.")

    clean_rec_cik = record_cik.strip()
    if not validate_cik_format(clean_rec_cik):
        return IdentityValidationResult(
            False,
            f"Malformed CIK '{clean_rec_cik}' for symbol '{clean_req_sym}'. CIK must be a 10-digit numeric string."
        )

    if clean_rec_cik != binding.cik:
        return IdentityValidationResult(
            False,
            f"Attributable identity binding mismatch for '{clean_req_sym}': record specifies CIK '{clean_rec_cik}', "
            f"but authoritative SEC ground truth is '{binding.cik}' ({binding.company_name})."
        )

    return IdentityValidationResult(True, None, binding)
