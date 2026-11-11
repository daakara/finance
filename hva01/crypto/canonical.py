"""
HVA-01 RFC 8785 JSON Canonicalization Scheme (JCS) Engine.
Provides deterministic UTF-8 serialization, IEEE 754 formatting, and SHA-256 digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Dict, List, Union


CANONICAL_KEY_PATTERN = re.compile(r"^[a-z0-9_-]{1,64}$")
MAX_SAFE_INTEGER = 9007199254740991
MIN_SAFE_INTEGER = -9007199254740991


class CanonicalSerializationError(ValueError):
    """Raised when data contains invalid floating-point values or non-canonical structures."""
    pass


class CanonicalJCS:
    """
    Implements RFC 8785 JSON Canonicalization Scheme under HVA-01 Restricted Contract:
    1. UTF-8 encoding with Unicode Normalization Form C (NFC).
    2. Strict lexicographic key sorting by UTF-16 code units.
    3. Restricted property-name validation (^[a-z0-9_-]{1,64}$).
    4. Restricted integer number serialization (safe range [-(2^53-1), 2^53-1]; floats unauthorized).
    5. Deterministic whitespace elimination (separators=(',', ':')).
    6. Preserved array sequence ordering.
    """

    @classmethod
    def serialize_value(cls, val: Any) -> Any:
        """Recursively normalize and validate values according to RFC 8785 restricted profile."""
        if val is None:
            return None
        elif isinstance(val, bool):
            return val
        elif isinstance(val, int):
            if not (MIN_SAFE_INTEGER <= val <= MAX_SAFE_INTEGER):
                raise CanonicalSerializationError(
                    f"Integer value {val} exceeds IEEE 754 safe integer range [{MIN_SAFE_INTEGER}, {MAX_SAFE_INTEGER}]"
                )
            return val
        elif isinstance(val, float):
            raise CanonicalSerializationError(
                f"Floating-point values are unauthorized in HVA-01 canonical payloads, got: {val}"
            )
        elif isinstance(val, str):
            # NFC normalization
            return unicodedata.normalize("NFC", val)
        elif isinstance(val, (list, tuple)):
            return [cls.serialize_value(item) for item in val]
        elif isinstance(val, dict):
            for k in val.keys():
                if not isinstance(k, str):
                    raise CanonicalSerializationError(f"Dictionary keys must be strings, got: {type(k)}")
            # Sort keys by UTF-16 code units
            sorted_dict = {}
            for k in sorted(val.keys(), key=lambda s: s.encode("utf-16be")):
                if not CANONICAL_KEY_PATTERN.match(k):
                    raise CanonicalSerializationError(
                        f"Canonical property name '{k}' violates restricted schema pattern '^[a-z0-9_-]{{1,64}}$'"
                    )
                normalized_k = unicodedata.normalize("NFC", k)
                sorted_dict[normalized_k] = cls.serialize_value(val[k])
            return sorted_dict
        elif hasattr(val, "to_dict") and callable(getattr(val, "to_dict")):
            return cls.serialize_value(val.to_dict())
        elif hasattr(val, "__dict__"):
            return cls.serialize_value(val.__dict__)
        else:
            raise CanonicalSerializationError(f"Unsupported type for canonical serialization: {type(val)}")

    @classmethod
    def canonicalize(cls, data: Any) -> bytes:
        """
        Serialize data to deterministic RFC 8785 canonical UTF-8 bytes.
        Preserves the UTF-16 code unit insertion order established in serialize_value.
        """
        normalized_data = cls.serialize_value(data)
        canonical_str = json.dumps(
            normalized_data,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return canonical_str.encode("utf-8")


def canonicalize_json(data: Any) -> bytes:
    """Convenience wrapper for RFC 8785 canonical serialization."""
    return CanonicalJCS.canonicalize(data)


def canonical_sha256(data: Any) -> str:
    """Compute the hexadecimal lowercase SHA-256 digest of RFC 8785 canonical data."""
    raw_bytes = canonicalize_json(data)
    return f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"
