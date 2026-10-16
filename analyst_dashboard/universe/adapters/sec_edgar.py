"""
analyst_dashboard/universe/adapters/sec_edgar.py

Pure offline parsers for SEC EDGAR company tickers and submissions JSON payloads.

Strict Non-Inference Invariants:
- Preserves CIK strictly as an issuer identifier (formatted as standard 10-digit zero-padded string).
- Does NOT manufacture a security_id from CIK or synthesize a CIK+symbol surrogate key.
- Does NOT infer COMMON_STOCK, EQUITY, USD, ACTIVE, USA, or primary_listing=True.
- Does NOT infer absence of suspension or delisting.
- Pure in-memory execution: zero file reads, zero network requests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import ParsedSourceFile, ParsedSourceRow, SourceFeedType


def _compute_sha256(content: str | bytes) -> str:
    """Computes SHA-256 digest of input bytes or UTF-8 encoded string."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def format_cik(cik_val: Any) -> str | None:
    """
    Normalizes a CIK value to a canonical 10-digit zero-padded string.
    Returns None if CIK is None, empty, or non-numeric.
    """
    if cik_val is None:
        return None
    try:
        cik_int = int(str(cik_val).strip())
        if cik_int < 0:
            return None
        return f"{cik_int:010d}"
    except (ValueError, TypeError):
        return None


def parse_sec_company_tickers_exchange(
    content: str | bytes | dict,
    source_identifier: str = "company_tickers_exchange.json",
    retrieved_at_utc: str | None = None,
) -> ParsedSourceFile:
    """
    Parses documented SEC EDGAR company_tickers_exchange.json format:
    {"fields": ["cik", "name", "ticker", "exchange"], "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"], ...]}

    Strictly validates structure, field alignment, and data row integrity without
    manufacturing missing domain attributes.
    """
    if isinstance(content, dict):
        raw_dict = content
        raw_bytes = json.dumps(content).encode("utf-8")
        content_hash = _compute_sha256(raw_bytes)
        physical_lines = len(raw_dict.get("data", [])) + 2
    elif isinstance(content, (str, bytes)):
        content_hash = _compute_sha256(content)
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        physical_lines = len(text.splitlines())
        try:
            raw_dict = json.loads(text)
        except (json.JSONDecodeError, ValueError) as e:
            # File is completely malformed JSON
            return ParsedSourceFile(
                feed_type=SourceFeedType.SEC_COMPANY_TICKERS_EXCHANGE,
                source_identifier=source_identifier,
                content_sha256=content_hash,
                retrieved_at_utc=retrieved_at_utc,
                physical_line_count=physical_lines,
                metadata_row_count=0,
                data_row_count=0,
                malformed_row_count=1,
                unique_record_count=0,
                rows=[
                    ParsedSourceRow(
                        row_index=0,
                        raw_values={},
                        is_malformed=True,
                        malformed_reason=f"Invalid JSON syntax: {e}",
                    )
                ],
            )
    else:
        raise TypeError(f"Unsupported content type: {type(content)}")

    # Validate structural fields
    fields = raw_dict.get("fields")
    data = raw_dict.get("data")

    if not isinstance(fields, list) or not isinstance(data, list):
        return ParsedSourceFile(
            feed_type=SourceFeedType.SEC_COMPANY_TICKERS_EXCHANGE,
            source_identifier=source_identifier,
            content_sha256=content_hash,
            retrieved_at_utc=retrieved_at_utc,
            physical_line_count=physical_lines,
            metadata_row_count=1,
            data_row_count=0,
            malformed_row_count=1,
            unique_record_count=0,
            rows=[
                ParsedSourceRow(
                    row_index=0,
                    raw_values=raw_dict if isinstance(raw_dict, dict) else {},
                    is_malformed=True,
                    malformed_reason="Top-level payload must contain 'fields' list and 'data' list",
                )
            ],
        )

    required_fields = {"cik", "name", "ticker", "exchange"}
    field_lower = [str(f).strip().lower() for f in fields]
    field_indices = {name: idx for idx, name in enumerate(field_lower)}

    missing_cols = required_fields - set(field_lower)
    if missing_cols:
        return ParsedSourceFile(
            feed_type=SourceFeedType.SEC_COMPANY_TICKERS_EXCHANGE,
            source_identifier=source_identifier,
            content_sha256=content_hash,
            retrieved_at_utc=retrieved_at_utc,
            physical_line_count=physical_lines,
            metadata_row_count=1,
            data_row_count=0,
            malformed_row_count=len(data),
            unique_record_count=0,
            rows=[
                ParsedSourceRow(
                    row_index=0,
                    raw_values={"fields": fields},
                    is_malformed=True,
                    malformed_reason=f"Missing required header fields: {sorted(missing_cols)}",
                )
            ],
        )

    parsed_rows: list[ParsedSourceRow] = []
    seen_symbols: set[str] = set()
    malformed_count = 0
    data_count = 0

    expected_len = len(fields)
    cik_idx = field_indices["cik"]
    name_idx = field_indices["name"]
    ticker_idx = field_indices["ticker"]
    exchange_idx = field_indices["exchange"]

    for row_idx, row in enumerate(data):
        if not isinstance(row, list) or len(row) != expected_len:
            malformed_count += 1
            parsed_rows.append(
                ParsedSourceRow(
                    row_index=row_idx,
                    raw_values={"row_content": row},
                    is_malformed=True,
                    malformed_reason=f"Row width mismatch: expected {expected_len} fields, got {len(row) if isinstance(row, list) else type(row)}",
                )
            )
            continue

        raw_cik = row[cik_idx]
        raw_name = row[name_idx]
        raw_ticker = row[ticker_idx]
        raw_exchange = row[exchange_idx]

        norm_cik = format_cik(raw_cik)
        norm_ticker = str(raw_ticker).strip().upper() if raw_ticker is not None else None
        norm_name = str(raw_name).strip() if raw_name is not None else None
        norm_exchange = str(raw_exchange).strip() if raw_exchange is not None else None

        # Validate mandatory values
        if not norm_cik or not norm_ticker or not norm_exchange:
            malformed_count += 1
            parsed_rows.append(
                ParsedSourceRow(
                    row_index=row_idx,
                    raw_values={
                        "raw_cik": raw_cik,
                        "raw_name": raw_name,
                        "raw_ticker": raw_ticker,
                        "raw_exchange": raw_exchange,
                    },
                    is_malformed=True,
                    malformed_reason="Row contains missing or invalid CIK, ticker, or exchange",
                )
            )
            continue

        data_count += 1
        seen_symbols.add(norm_ticker)

        parsed_rows.append(
            ParsedSourceRow(
                row_index=row_idx,
                raw_values={
                    "issuer_cik": norm_cik,
                    "company_name": norm_name,
                    "symbol": norm_ticker,
                    "sec_exchange": norm_exchange,
                    "raw_cik": raw_cik,
                },
                is_malformed=False,
            )
        )

    return ParsedSourceFile(
        feed_type=SourceFeedType.SEC_COMPANY_TICKERS_EXCHANGE,
        source_identifier=source_identifier,
        content_sha256=content_hash,
        retrieved_at_utc=retrieved_at_utc,
        physical_line_count=physical_lines,
        metadata_row_count=1,  # The header schema definition
        data_row_count=data_count,
        malformed_row_count=malformed_count,
        unique_record_count=len(seen_symbols),
        metadata_fields={
            "fields": fields,
            "total_count": raw_dict.get("total_count"),
            "sample_count": raw_dict.get("sample_count"),
        },
        rows=parsed_rows,
    )


def parse_sec_submissions(
    content: str | bytes | dict,
    source_identifier: str = "submissions.json",
    retrieved_at_utc: str | None = None,
) -> ParsedSourceFile:
    """
    Parses SEC EDGAR company submissions JSON (e.g. data.sec.gov/submissions/CIK{cik}.json).
    Demonstrates multi-share class CIK relationship where one issuer CIK has multiple ticker symbols.
    """
    if isinstance(content, dict):
        raw_dict = content
        raw_bytes = json.dumps(content).encode("utf-8")
        content_hash = _compute_sha256(raw_bytes)
        physical_lines = 1
    elif isinstance(content, (str, bytes)):
        content_hash = _compute_sha256(content)
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        physical_lines = len(text.splitlines())
        raw_dict = json.loads(text)
    else:
        raise TypeError(f"Unsupported content type: {type(content)}")

    raw_cik = raw_dict.get("cik")
    norm_cik = format_cik(raw_cik)
    entity_name = raw_dict.get("name")
    tickers = raw_dict.get("tickers", [])
    exchanges = raw_dict.get("exchanges", [])

    if not norm_cik:
        return ParsedSourceFile(
            feed_type=SourceFeedType.SEC_SUBMISSIONS,
            source_identifier=source_identifier,
            content_sha256=content_hash,
            retrieved_at_utc=retrieved_at_utc,
            physical_line_count=physical_lines,
            metadata_row_count=0,
            data_row_count=0,
            malformed_row_count=1,
            unique_record_count=0,
            rows=[
                ParsedSourceRow(
                    row_index=0,
                    raw_values=raw_dict,
                    is_malformed=True,
                    malformed_reason="Submissions payload missing valid CIK",
                )
            ],
        )

    parsed_rows: list[ParsedSourceRow] = []
    seen_symbols: set[str] = set()

    for idx, ticker in enumerate(tickers):
        norm_ticker = str(ticker).strip().upper() if ticker else None
        exch = exchanges[idx] if idx < len(exchanges) else None
        if not norm_ticker:
            continue
        seen_symbols.add(norm_ticker)
        parsed_rows.append(
            ParsedSourceRow(
                row_index=idx,
                raw_values={
                    "issuer_cik": norm_cik,
                    "company_name": str(entity_name).strip() if entity_name else None,
                    "symbol": norm_ticker,
                    "sec_exchange": str(exch).strip() if exch else None,
                    "sic": raw_dict.get("sic"),
                    "sic_description": raw_dict.get("sicDescription"),
                },
                is_malformed=False,
            )
        )

    return ParsedSourceFile(
        feed_type=SourceFeedType.SEC_SUBMISSIONS,
        source_identifier=source_identifier,
        content_sha256=content_hash,
        retrieved_at_utc=retrieved_at_utc,
        physical_line_count=physical_lines,
        metadata_row_count=1,
        data_row_count=len(parsed_rows),
        malformed_row_count=0,
        unique_record_count=len(seen_symbols),
        metadata_fields={
            "issuer_cik": norm_cik,
            "entity_name": entity_name,
            "ticker_count": len(tickers),
        },
        rows=parsed_rows,
    )
