"""
Deterministic Canonicalization Contract SM_CERT_CANON_V1
for ARX Terminal Smart Money Wave 2C Certification Authority.

Rules:
1. Unicode Normalization: NFC
2. Object Keys: Lexicographically sorted
3. Set-like Arrays: Sorted by canonical identity
4. Ordered Arrays: Order preserved
5. Timestamps: UTC RFC3339 fixed precision (e.g. YYYY-MM-DDTHH:MM:SS.000Z or YYYY-MM-DDTHH:MM:SSZ)
6. Nulls: Explicit nulls (None -> null)
7. Floats: Strictly prohibited in certification payloads (raise ValueError)
8. Unknown Schema Fields: Reject during deserialization
9. Duplicate Object Keys: Reject
"""

import json
import unicodedata
import hashlib
from typing import Any, Dict


CERTIFICATION_CANONICALIZATION_VERSION = "SM_CERT_CANON_V1"


def _validate_no_floats(obj: Any, path: str = "root"):
    """Enforces the invariant: FLOATS_IN_CERTIFICATION_PAYLOAD = PROHIBITED."""
    if isinstance(obj, float):
        raise ValueError(f"FLOAT_PROHIBITED_IN_CERTIFICATION: Float value '{obj}' detected at path '{path}'")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _validate_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, elem in enumerate(obj):
            _validate_no_floats(elem, f"{path}[{i}]")


def _normalize_unicode(obj: Any) -> Any:
    """Recursively normalizes all strings to NFC Unicode."""
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    elif isinstance(obj, dict):
        return {unicodedata.normalize("NFC", k): _normalize_unicode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_unicode(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return [_normalize_unicode(elem) for elem in obj]
    return obj


def canonicalize_json(data: Any) -> str:
    """
    Serializes data to deterministic canonical JSON under SM_CERT_CANON_V1.
    Rejects floats, enforces NFC Unicode normalization, sorts keys, uses fixed separators.
    """
    _validate_no_floats(data)
    normalized = _normalize_unicode(data)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_canonical_hash(data: Any) -> str:
    """Computes SHA-256 over canonical JSON representation."""
    canonical_str = canonicalize_json(data)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
