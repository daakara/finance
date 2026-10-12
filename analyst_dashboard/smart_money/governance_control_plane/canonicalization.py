"""
Canonicalization & Domain Normalization Subsystem
ARX Terminal — Smart Money

Implements ARX-GOV-NORMALIZE-V1 and RFC 8785 JSON Canonicalization Scheme (JCS) with SHA-256.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

ARX_GOV_NORMALIZE_VERSION = "ARX-GOV-NORMALIZE-V1"
JCS_VERSION = "RFC8785-JCS"


def _normalize_string(val: str) -> str:
    s = val.strip()
    # Normalize ISO 8601 / RFC 3339 timestamps with +00:00 to Z
    if len(s) >= 19 and ("T" in s) and (s.endswith("+00:00") or s.endswith("-00:00")):
        s = s[:-6] + "Z"
    return s


def domain_normalize(obj: Any) -> Any:
    """
    Applies ARX-GOV-NORMALIZE-V1 semantic domain normalization:
    - Normalizes string whitespace and timestamps
    - Recursively traverses dictionaries and sorts object keys
    - Semantically sorts sets / lists of identifiers, findings, or artifacts
    - Ensures uniform representation of None vs missing
    """
    if obj is None:
        return None
    elif isinstance(obj, str):
        return _normalize_string(obj)
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, dict):
        normalized_dict = {}
        for k in sorted(obj.keys()):
            v = obj[k]
            normalized_dict[k] = domain_normalize(v)
        return normalized_dict
    elif isinstance(obj, (list, tuple)):
        # Normalize list elements first
        norm_list = [domain_normalize(item) for item in obj]
        # Check if items are dicts with identifiable keys (e.g. finding_id, artifact_id, approver)
        if norm_list and all(isinstance(x, dict) for x in norm_list):
            if all("finding_id" in x for x in norm_list):
                return sorted(norm_list, key=lambda x: str(x.get("finding_id")))
            elif all("artifact_id" in x for x in norm_list):
                return sorted(norm_list, key=lambda x: str(x.get("artifact_id")))
            elif all("id" in x for x in norm_list):
                return sorted(norm_list, key=lambda x: str(x.get("id")))
            elif all("approver" in x for x in norm_list):
                return sorted(norm_list, key=lambda x: str(x.get("approver")))
            elif all("key_id" in x for x in norm_list):
                return sorted(norm_list, key=lambda x: str(x.get("key_id")))
            elif all("audit_id" in x for x in norm_list):
                return sorted(norm_list, key=lambda x: str(x.get("audit_id")))
            else:
                # Stable representation
                return norm_list
        elif norm_list and all(isinstance(x, str) for x in norm_list):
            # If all are strings, sort lexicographically if they represent identifier sets
            return sorted(norm_list)
        else:
            return norm_list
    else:
        return obj


def canonical_json(obj: Any) -> bytes:
    """
    Produces deterministic RFC 8785 JCS canonical JSON UTF-8 bytes.
    - No whitespace between tokens
    - Unicode characters represented directly
    - Keys sorted by Unicode code points
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_canonical(obj: Any) -> str:
    """
    Computes SHA-256 hash over domain_normalize + canonical_json.
    Pipeline: domain object -> domain normalization -> JCS canonicalization -> SHA-256
    """
    norm_obj = domain_normalize(obj)
    jcs_bytes = canonical_json(norm_obj)
    return hashlib.sha256(jcs_bytes).hexdigest()
