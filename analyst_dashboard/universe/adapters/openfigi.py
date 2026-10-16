"""
analyst_dashboard/universe/adapters/openfigi.py

Pure offline parser for saved OpenFIGI request/response mapping pairs.

Strict Non-Inference Invariants:
- Preserves request-to-result association across batch jobs.
- Preserves Listing FIGI, Composite FIGI, and Share Class FIGI without treating them as interchangeable.
- Does NOT automatically select the first result on multi-match outcomes.
- Does NOT infer requested filter criteria were independently returned as verified response evidence.
- Does NOT map REIT/BDC or taxonomy classifications to strategy eligibility.
- Pure in-memory execution: zero file reads, zero network requests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import OpenFigiMatchStatus, ParsedOpenFigiResult


def _compute_sha256(content: str | bytes) -> str:
    """Computes SHA-256 digest of input bytes or UTF-8 encoded string."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def parse_openfigi_response(
    response_content: str | bytes | list,
    request_payload: list[dict[str, Any]] | None = None,
    source_identifier: str = "openfigi_mapping_response.json",
    retrieved_at_utc: str | None = None,
) -> list[ParsedOpenFigiResult]:
    """
    Parses OpenFIGI mapping API response array:
    [
      {"data": [{"figi": "BBG...", "name": "...", "ticker": "...", "exchCode": "US",
                 "compositeFIGI": "...", "securityType": "...", "shareClassFIGI": "...",
                 "securityType2": "...", ...}]},
      {"error": "..."},
      {"warning": "..."},
      {"data": [...]} # Multiple matches
    ]

    Correlates each response item with its corresponding request payload item if provided.
    """
    if isinstance(response_content, list):
        res_list = response_content
    elif isinstance(response_content, (str, bytes)):
        text = response_content.decode("utf-8") if isinstance(response_content, bytes) else response_content
        try:
            res_list = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            return [
                ParsedOpenFigiResult(
                    request_index=0,
                    request_payload={},
                    match_status=OpenFigiMatchStatus.ERROR,
                    error_message=f"Invalid JSON syntax in OpenFIGI response: {e}",
                    matches=[],
                )
            ]
    else:
        raise TypeError(f"Unsupported response content type: {type(response_content)}")

    if not isinstance(res_list, list):
        return [
            ParsedOpenFigiResult(
                request_index=0,
                request_payload={},
                match_status=OpenFigiMatchStatus.ERROR,
                error_message=f"OpenFIGI response must be a list, got {type(res_list)}",
                matches=[],
            )
        ]

    parsed_results: list[ParsedOpenFigiResult] = []

    req_count = len(request_payload) if request_payload is not None else len(res_list)
    max_idx = max(len(res_list), req_count)

    for req_idx in range(max_idx):
        req_item = request_payload[req_idx] if request_payload and req_idx < len(request_payload) else {}

        # Case 1: Missing response item (request was submitted, but response list was truncated)
        if req_idx >= len(res_list):
            parsed_results.append(
                ParsedOpenFigiResult(
                    request_index=req_idx,
                    request_payload=req_item,
                    match_status=OpenFigiMatchStatus.ERROR,
                    error_message=f"Missing response item for request index {req_idx} (received {len(res_list)} responses for {req_count} requests)",
                    matches=[],
                    selected_figi=None,
                )
            )
            continue

        item = res_list[req_idx]

        # Case 2: Extra response item beyond request count
        extra_warning = None
        if request_payload is not None and req_idx >= len(request_payload):
            extra_warning = f"Response item at index {req_idx} has no corresponding request payload item (request payload count: {len(request_payload)})"

        if not isinstance(item, dict):
            parsed_results.append(
                ParsedOpenFigiResult(
                    request_index=req_idx,
                    request_payload=req_item,
                    match_status=OpenFigiMatchStatus.ERROR,
                    error_message=f"Response item at index {req_idx} is not a dictionary: {type(item)}",
                    warning_message=extra_warning,
                    matches=[],
                    selected_figi=None,
                )
            )
            continue

        error_msg = item.get("error")
        warning_msg = item.get("warning")
        if extra_warning:
            warning_msg = f"{warning_msg}; {extra_warning}" if warning_msg else extra_warning
        data_matches = item.get("data")

        if error_msg:
            parsed_results.append(
                ParsedOpenFigiResult(
                    request_index=req_idx,
                    request_payload=req_item,
                    match_status=OpenFigiMatchStatus.ERROR,
                    error_message=str(error_msg),
                    warning_message=str(warning_msg) if warning_msg else None,
                    matches=[],
                    selected_figi=None,
                )
            )
            continue

        if not data_matches or not isinstance(data_matches, list) or len(data_matches) == 0:
            parsed_results.append(
                ParsedOpenFigiResult(
                    request_index=req_idx,
                    request_payload=req_item,
                    match_status=OpenFigiMatchStatus.NO_MATCH,
                    warning_message=str(warning_msg) if warning_msg else None,
                    matches=[],
                    selected_figi=None,
                )
            )
            continue

        # Extract structured matches
        clean_matches: list[dict[str, Any]] = []
        for match in data_matches:
            if not isinstance(match, dict):
                continue
            clean_matches.append({
                "listing_figi": match.get("figi"),
                "composite_figi": match.get("compositeFIGI"),
                "share_class_figi": match.get("shareClassFIGI"),
                "ticker": match.get("ticker"),
                "name": match.get("name"),
                "exch_code": match.get("exchCode"),
                "security_type": match.get("securityType"),
                "security_type2": match.get("securityType2"),
                "market_sector": match.get("marketSector"),
                "security_description": match.get("securityDescription"),
            })

        if len(clean_matches) == 1:
            match_status = OpenFigiMatchStatus.SINGLE_MATCH
            # In a single unambiguous match, the listing FIGI is unambiguous
            selected_figi = clean_matches[0]["listing_figi"]
        elif len(clean_matches) > 1:
            match_status = OpenFigiMatchStatus.MULTI_MATCH
            # INVARIANT: Do NOT select first match automatically
            selected_figi = None
        else:
            match_status = OpenFigiMatchStatus.NO_MATCH
            selected_figi = None

        parsed_results.append(
            ParsedOpenFigiResult(
                request_index=req_idx,
                request_payload=req_item,
                match_status=match_status,
                warning_message=str(warning_msg) if warning_msg else None,
                matches=clean_matches,
                selected_figi=selected_figi,
            )
        )

    return parsed_results
