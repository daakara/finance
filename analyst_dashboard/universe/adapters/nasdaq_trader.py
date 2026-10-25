"""
analyst_dashboard/universe/adapters/nasdaq_trader.py

Pure offline parsers for Nasdaq Trader directory text files (nasdaqlisted.txt and otherlisted.txt).

Strict Non-Inference Invariants:
- Preserves raw symbols and normalized join candidates separately without silently resolving punctuation.
- ETF=N and Test Issue=N must NOT become proof of COMMON_STOCK.
- Directory presence must NOT become proof of unsuspended status, global primary listing,
  currency, or historical listing windows.
- Financial-status flags retain their documented meanings and are NOT reinterpreted as a halt feed.
- Pure in-memory execution: zero file reads, zero network requests.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .models import ParsedSourceFile, ParsedSourceRow, SourceFeedType


def _compute_sha256(content: str | bytes) -> str:
    """Computes SHA-256 digest of input bytes or UTF-8 encoded string."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


# Documented Nasdaq Trader directory header schemas
NASDAQ_LISTED_HEADER = [
    "Symbol",
    "Security Name",
    "Market Category",
    "Test Issue",
    "Financial Status",
    "Round Lot Size",
    "ETF",
    "NextShares",
]

OTHER_LISTED_HEADER = [
    "ACT Symbol",
    "Security Name",
    "Exchange",
    "CQS Symbol",
    "ETF",
    "Round Lot Size",
    "Test Issue",
    "NASDAQ Symbol",
]

TRADING_SYSTEM_ADDS_DELETES_HEADER = [
    "Symbol",
    "Company Name",
    "NASDAQ Action",
    "BX Action",
    "PSX Action",
    "Effective Date",
    "Primary Listing Market",
]


def parse_nasdaq_trader_directory(
    content: str | bytes,
    filename_hint: str | None = None,
    source_identifier: str = "nasdaq_trader_directory.txt",
    retrieved_at_utc: str | None = None,
) -> ParsedSourceFile:
    """
    Parses Nasdaq Trader pipe-delimited directory files:
    - nasdaqlisted.txt (8 columns)
    - otherlisted.txt (8 columns)

    Distinguishes physical lines, metadata rows (header and footer), data rows,
    and malformed rows. Extracts file creation time from footer metadata.
    """
    content_hash = _compute_sha256(content)
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    raw_lines = text.splitlines()
    physical_lines = len(raw_lines)

    if not raw_lines:
        return ParsedSourceFile(
            feed_type=SourceFeedType.NASDAQ_LISTED_DIRECTORY,
            source_identifier=source_identifier,
            content_sha256=content_hash,
            retrieved_at_utc=retrieved_at_utc,
            physical_line_count=0,
            metadata_row_count=0,
            data_row_count=0,
            malformed_row_count=1,
            unique_record_count=0,
            rows=[
                ParsedSourceRow(
                    row_index=0,
                    raw_values={},
                    is_malformed=True,
                    malformed_reason="Empty directory payload",
                )
            ],
        )

    # 1. Inspect Header
    header_line = raw_lines[0].strip()
    header_cols = [c.strip() for c in header_line.split("|")]

    is_nasdaq_listed = header_cols == NASDAQ_LISTED_HEADER or (
        filename_hint and "nasdaqlisted" in filename_hint.lower()
    )
    is_other_listed = header_cols == OTHER_LISTED_HEADER or (
        filename_hint and "otherlisted" in filename_hint.lower()
    )
    is_adds_deletes = header_cols == TRADING_SYSTEM_ADDS_DELETES_HEADER or (
        filename_hint and "tradingsystemaddsdeletes" in filename_hint.lower()
    )

    if is_nasdaq_listed:
        feed_type = SourceFeedType.NASDAQ_LISTED_DIRECTORY
        expected_cols = NASDAQ_LISTED_HEADER
    elif is_other_listed:
        feed_type = SourceFeedType.OTHER_LISTED_DIRECTORY
        expected_cols = OTHER_LISTED_HEADER
    elif is_adds_deletes:
        feed_type = SourceFeedType.NASDAQ_ADDS_DELETES
        expected_cols = TRADING_SYSTEM_ADDS_DELETES_HEADER
    else:
        # Unknown directory schema
        return ParsedSourceFile(
            feed_type=SourceFeedType.NASDAQ_LISTED_DIRECTORY,
            source_identifier=source_identifier,
            content_sha256=content_hash,
            retrieved_at_utc=retrieved_at_utc,
            physical_line_count=physical_lines,
            metadata_row_count=1,
            data_row_count=0,
            malformed_row_count=len(raw_lines) - 1,
            unique_record_count=0,
            rows=[
                ParsedSourceRow(
                    row_index=0,
                    raw_values={"header": header_line},
                    is_malformed=True,
                    malformed_reason=f"Unrecognized directory header: '{header_line}'",
                )
            ],
        )

    expected_width = len(expected_cols)
    parsed_rows: list[ParsedSourceRow] = []
    seen_symbols: set[str] = set()
    metadata_count = 1  # Header line
    data_count = 0
    malformed_count = 0
    footer_metadata: dict[str, Any] = {}

    for line_idx, line in enumerate(raw_lines[1:], start=1):
        clean_line = line.strip()
        if not clean_line:
            continue

        # Check for File Creation Time footer line
        if clean_line.startswith("File Creation Time:"):
            metadata_count += 1
            footer_parts = clean_line.split("|")
            creation_str = footer_parts[0].replace("File Creation Time:", "").strip()
            footer_metadata["file_creation_time_raw"] = creation_str
            footer_metadata["footer_line"] = clean_line
            continue

        parts = clean_line.split("|")
        if len(parts) != expected_width:
            malformed_count += 1
            parsed_rows.append(
                ParsedSourceRow(
                    row_index=line_idx,
                    raw_values={"raw_line": clean_line},
                    is_malformed=True,
                    malformed_reason=f"Column width mismatch: expected {expected_width}, got {len(parts)}",
                )
            )
            continue

        row_dict = {expected_cols[i]: parts[i].strip() for i in range(expected_width)}

        if is_nasdaq_listed:
            raw_sym = row_dict.get("Symbol", "")
            sec_name = row_dict.get("Security Name", "")
            market_cat = row_dict.get("Market Category", "")
            test_issue = row_dict.get("Test Issue", "")
            fin_status = row_dict.get("Financial Status", "")
            lot_size = row_dict.get("Round Lot Size", "")
            etf_flag = row_dict.get("ETF", "")
            nextshares = row_dict.get("NextShares", "")

            # Basic syntax sanity
            if not raw_sym or test_issue not in {"Y", "N"} or etf_flag not in {"Y", "N"}:
                malformed_count += 1
                parsed_rows.append(
                    ParsedSourceRow(
                        row_index=line_idx,
                        raw_values=row_dict,
                        is_malformed=True,
                        malformed_reason="Invalid or missing Symbol, Test Issue, or ETF flag",
                    )
                )
                continue

            data_count += 1
            seen_symbols.add(raw_sym)

            parsed_rows.append(
                ParsedSourceRow(
                    row_index=line_idx,
                    raw_values={
                        "raw_symbol": raw_sym,
                        "normalized_symbol": raw_sym.upper(),
                        "security_name": sec_name,
                        "market_category": market_cat,
                        "test_issue": test_issue == "Y",
                        "financial_status": fin_status,
                        "round_lot_size": lot_size,
                        "is_etf": etf_flag == "Y",
                        "nextshares": nextshares == "Y",
                        "venue_exchange": "NASDAQ",
                        "source_directory": "nasdaqlisted",
                    },
                    is_malformed=False,
                )
            )

        elif is_adds_deletes:
            raw_sym = row_dict.get("Symbol", "")
            comp_name = row_dict.get("Company Name", "")
            nasdaq_action = row_dict.get("NASDAQ Action", "")
            bx_action = row_dict.get("BX Action", "")
            psx_action = row_dict.get("PSX Action", "")
            eff_date = row_dict.get("Effective Date", "")
            primary_market = row_dict.get("Primary Listing Market", "")

            if not raw_sym:
                malformed_count += 1
                parsed_rows.append(
                    ParsedSourceRow(
                        row_index=line_idx,
                        raw_values=row_dict,
                        is_malformed=True,
                        malformed_reason="Invalid or missing Symbol in adds/deletes row",
                    )
                )
                continue

            data_count += 1
            seen_symbols.add(raw_sym)

            parsed_rows.append(
                ParsedSourceRow(
                    row_index=line_idx,
                    raw_values={
                        "raw_symbol": raw_sym,
                        "normalized_symbol": raw_sym.upper(),
                        "company_name": comp_name,
                        "nasdaq_action": nasdaq_action,
                        "bx_action": bx_action,
                        "psx_action": psx_action,
                        "effective_date": eff_date,
                        "primary_listing_market": primary_market,
                        "venue_exchange": "NASDAQ" if primary_market in {"Q", "G", "S"} else primary_market,
                        "source_directory": "TradingSystemAddsDeletes",
                    },
                    is_malformed=False,
                )
            )

        else:  # otherlisted
            act_sym = row_dict.get("ACT Symbol", "")
            sec_name = row_dict.get("Security Name", "")
            exch_code = row_dict.get("Exchange", "")
            cqs_sym = row_dict.get("CQS Symbol", "")
            etf_flag = row_dict.get("ETF", "")
            lot_size = row_dict.get("Round Lot Size", "")
            test_issue = row_dict.get("Test Issue", "")
            nasdaq_sym = row_dict.get("NASDAQ Symbol", "")

            if not act_sym or test_issue not in {"Y", "N"} or etf_flag not in {"Y", "N"}:
                malformed_count += 1
                parsed_rows.append(
                    ParsedSourceRow(
                        row_index=line_idx,
                        raw_values=row_dict,
                        is_malformed=True,
                        malformed_reason="Invalid or missing ACT Symbol, Test Issue, or ETF flag",
                    )
                )
                continue

            data_count += 1
            seen_symbols.add(act_sym)

            parsed_rows.append(
                ParsedSourceRow(
                    row_index=line_idx,
                    raw_values={
                        "raw_symbol": act_sym,
                        "normalized_symbol": act_sym.upper(),
                        "cqs_symbol": cqs_sym,
                        "nasdaq_symbol": nasdaq_sym,
                        "security_name": sec_name,
                        "exchange_code": exch_code,
                        "test_issue": test_issue == "Y",
                        "round_lot_size": lot_size,
                        "is_etf": etf_flag == "Y",
                        "venue_exchange": exch_code,
                        "source_directory": "otherlisted",
                    },
                    is_malformed=False,
                )
            )

    return ParsedSourceFile(
        feed_type=feed_type,
        source_identifier=source_identifier,
        content_sha256=content_hash,
        retrieved_at_utc=retrieved_at_utc,
        physical_line_count=physical_lines,
        metadata_row_count=metadata_count,
        data_row_count=data_count,
        malformed_row_count=malformed_count,
        unique_record_count=len(seen_symbols),
        metadata_fields={
            "header": header_cols,
            "footer": footer_metadata,
        },
        rows=parsed_rows,
    )
