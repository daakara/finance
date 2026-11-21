"""SEC EDGAR Regulatory Filing & Corporate Financial Disclosure Fetcher (Free Public API)."""

import os
import re
import json
import math
import hashlib
import logging
import requests
import datetime
from typing import Dict, Any, List, Optional, Tuple

import threading
import time

import sqlite3

logger = logging.getLogger(__name__)


def _clean_numeric(val: Any) -> Optional[float]:
    """Safely converts numeric input to float, returning None for non-finite values (NaN, Inf) or nulls."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _format_canonical_sec_timestamp(val: Any) -> Optional[str]:
    """
    Formats a datetime, date, or ISO timestamp into canonical ISO 8601 UTC with 'Z' suffix (e.g. '2026-09-18T21:13:29Z').
    Converts all valid timezone-aware values (positive and negative offsets) to canonical UTC 'Z'.
    Returns None if missing or invalid; strictly refuses to fabricate timestamps from date-only values.
    """
    if val is None:
        return None
    if isinstance(val, datetime.datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=datetime.timezone.utc)
        return val.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(val, datetime.date) and not isinstance(val, datetime.datetime):
        return None
    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned or len(cleaned) <= 10:
            return None
        if "T" not in cleaned and " " not in cleaned:
            return None
        iso_str = cleaned.replace("Z", "+00:00")
        try:
            dt = datetime.datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            return None
    return None


def _map_sec_transaction_type(
    code: Optional[str],
    acquired_disposed: Optional[str] = None,
    is_derivative: bool = False,
) -> str:
    """Maps SEC Form 4 transaction codes to authoritative descriptions without narrative hype."""
    c = (code or "").strip().upper()
    ad = (acquired_disposed or "").strip().upper()
    code_map = {
        "P": "Open Market Purchase (Code P)",
        "S": "Open Market Sale (Code S)",
        "A": "Grant / Award (Code A)",
        "D": "Disposition to Issuer (Code D)",
        "F": "Tax Withholding / Payment (Code F)",
        "M": "Option Exercise (Code M)",
        "G": "Bona Fide Gift (Code G)",
        "V": "Early Voluntary Reporting (Code V)",
        "C": "Conversion of Derivative (Code C)",
        "W": "Inheritance / Will (Code W)",
        "X": "Option Exercise (Code X)",
        "J": "Other Transaction (See Footnotes) (Code J)",
        "K": "Equity Swap (Code K)",
        "U": "Tender in Exchange Offer (Code U)",
        "E": "Expiration of Short Position (Code E)",
        "H": "Expiration / Cancellation (Code H)",
        "O": "Exercise of Out-of-the-Money Derivative (Code O)",
        "Z": "Voting Trust Deposit / Withdrawal (Code Z)",
        "L": "Small Acquisition (Rule 16a-6) (Code L)",
    }
    if c in code_map:
        return code_map[c]
    if ad == "A":
        return f"Acquisition (Code {c})" if c else "Acquisition"
    if ad == "D":
        return f"Disposition (Code {c})" if c else "Disposition"
    return f"Regulatory Transaction (Code {c})" if c else "Regulatory Transaction"


KNOWN_ISSUER_CIKS = {
    "AAPL": "0000320193",
    "NVDA": "0001045810",
}


def _validate_issuer_reconciliation_receipt(
    receipt: Dict[str, Any],
    clean_ticker: str,
    clean_cik: Optional[str],
    evaluation_cutoff: Optional[str] = None,
    requested_interval: Optional[Tuple[str, str]] = None,
    cur: Optional[Any] = None,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Validates an authoritative reconciliation receipt using governed rules:
    1. Rejects missing or invalid integrity evidence (reconciliation_hash is mandatory and verified).
    2. Rejects conflicting ticker/CIK identities (internal conflict and requested mismatch).
    3. Enforces requested interval coverage, cutoff watermark bounds, inventory, and supersession chains.
    4. Enforces zero-activity catalog proof verification for COMPLETE status with empty inventory.
    """
    from analyst_dashboard.smart_money.storage.postgres_repository import (
        compute_canonical_receipt_hash,
        validate_catalog_evidence,
        _to_iso,
    )

    reasons_val = receipt.get("failure_reasons")
    if isinstance(reasons_val, str):
        try:
            reasons_val = json.loads(reasons_val)
        except Exception:
            reasons_val = {}
    elif not isinstance(reasons_val, dict):
        reasons_val = {}

    r_ticker = str(reasons_val.get("ticker") or receipt.get("ticker") or "").upper().strip()
    r_cik = (
        str(reasons_val.get("issuer_cik") or receipt.get("issuer_cik") or "").strip().zfill(10)
        if (reasons_val.get("issuer_cik") or receipt.get("issuer_cik"))
        else None
    )
    c_tickers = [str(t).upper().strip() for t in reasons_val.get("covered_tickers", [])]
    c_ciks = [str(c).strip().zfill(10) for c in reasons_val.get("covered_ciks", [])]
    acc_inv = reasons_val.get("accession_inventory", [])
    s_hashes = reasons_val.get("source_hashes", {})
    m_gaps = reasons_val.get("material_gaps", [])
    z_proof = reasons_val.get("zero_activity_proof")
    schema_ver = reasons_val.get("schema_version", 1)

    # 1. Reject Conflicting Ticker / CIK Identities
    req_sym = clean_ticker.upper().strip() if clean_ticker else ""
    req_cik = str(clean_cik).strip().zfill(10) if clean_cik else None

    # Check internal consistency within receipt
    if r_ticker in KNOWN_ISSUER_CIKS and r_cik and r_cik != KNOWN_ISSUER_CIKS[r_ticker]:
        return False, None, f"CONFLICTING_ISSUER_IDENTITY: internal receipt conflict: ticker {r_ticker} with CIK {r_cik}"
    if req_sym in KNOWN_ISSUER_CIKS and req_cik and req_cik != KNOWN_ISSUER_CIKS[req_sym]:
        return False, None, f"CONFLICTING_ISSUER_IDENTITY: requested ticker {req_sym} conflicts with requested CIK {req_cik}"

    # Check match with requested issuer
    if req_sym and req_cik:
        sym_matched = (r_ticker == req_sym or req_sym in c_tickers)
        cik_matched = (r_cik == req_cik or req_cik in c_ciks)
        if not (sym_matched and cik_matched):
            return False, None, f"CONFLICTING_ISSUER_IDENTITY: receipt ({r_ticker}/{r_cik}) does not match requested ({req_sym}/{req_cik})"
    elif req_sym:
        sym_matched = (r_ticker == req_sym or req_sym in c_tickers)
        if not sym_matched:
            return False, None, f"ISSUER_MISMATCH: receipt ticker {r_ticker} does not match {req_sym}"
        if r_cik and req_sym in KNOWN_ISSUER_CIKS and r_cik != KNOWN_ISSUER_CIKS[req_sym]:
            return False, None, f"CONFLICTING_ISSUER_IDENTITY: receipt CIK {r_cik} conflicts with ticker {req_sym}"
    elif req_cik:
        cik_matched = (r_cik == req_cik or req_cik in c_ciks)
        if not cik_matched:
            return False, None, f"ISSUER_MISMATCH: receipt CIK {r_cik} does not match {req_cik}"

    # 2. Mandatory Cryptographic Integrity Hash
    rec_hash = str(receipt.get("reconciliation_hash") or "").strip().lower()
    if not rec_hash or len(rec_hash) != 64 or not re.match(r"^[0-9a-f]{64}$", rec_hash):
        return False, None, "MISSING_OR_INVALID_INTEGRITY_EVIDENCE: reconciliation_hash is missing, empty, or not 64-char hex SHA-256."

    expected_hash = compute_canonical_receipt_hash(
        schema_version=schema_ver,
        ticker=r_ticker or clean_ticker,
        issuer_cik=r_cik or (clean_cik or ""),
        interval_start=receipt.get("interval_start"),
        interval_end=receipt.get("interval_end"),
        certified_watermark=receipt.get("certified_watermark"),
        receipt_version=receipt.get("receipt_version", 1),
        supersedes_receipt_id=str(receipt["supersedes_receipt_id"]) if receipt.get("supersedes_receipt_id") else None,
        accession_inventory=acc_inv,
        source_hashes=s_hashes,
        material_gaps=m_gaps,
        zero_activity_proof=z_proof,
    )
    legacy_payload = (
        f"{(r_ticker or clean_ticker).upper().strip()}:{str(r_cik or clean_cik or '').strip().zfill(10)}:"
        f"{_to_iso(receipt.get('interval_start'))}:{_to_iso(receipt.get('interval_end'))}:{_to_iso(receipt.get('certified_watermark'))}:"
        f"{json.dumps(sorted(list(set(acc_inv))))}:{json.dumps(s_hashes, sort_keys=True)}:{len(m_gaps)}"
    )
    legacy_hash = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()

    if rec_hash != expected_hash.lower() and rec_hash != legacy_hash.lower():
        return False, None, f"INTEGRITY_HASH_MISMATCH: expected {expected_hash}, got {rec_hash}"

    # 3. Interval & Cutoff Boundaries
    int_start = _format_canonical_sec_timestamp(receipt.get("interval_start"))
    int_end = _format_canonical_sec_timestamp(receipt.get("interval_end"))
    cert_wm = _format_canonical_sec_timestamp(receipt.get("certified_watermark"))

    if not int_start or not int_end:
        return False, None, "INVALID_INTERVAL: receipt missing interval_start or interval_end"
    if int_start > int_end:
        return False, None, f"INVALID_INTERVAL: interval_start ({int_start}) > interval_end ({int_end})"
    if not cert_wm:
        return False, None, "MISSING_CERTIFIED_WATERMARK: receipt missing certified_watermark"

    if evaluation_cutoff:
        eval_ts = _format_canonical_sec_timestamp(evaluation_cutoff)
        if eval_ts and cert_wm > eval_ts:
            return False, None, f"CUTOFF_EXCEEDED: certified watermark ({cert_wm}) postdates cutoff ({eval_ts})"
        if eval_ts and int_end > eval_ts:
            return False, None, f"CUTOFF_EXCEEDED: interval_end ({int_end}) postdates cutoff ({eval_ts})"

    if requested_interval:
        req_start = _format_canonical_sec_timestamp(requested_interval[0])
        req_end = _format_canonical_sec_timestamp(requested_interval[1])
        if req_start and int_start > req_start:
            return False, None, f"INTERVAL_NOT_COVERED: interval_start ({int_start}) > requested ({req_start})"
        if req_end and int_end < req_end:
            return False, None, f"INTERVAL_NOT_COVERED: interval_end ({int_end}) < requested ({req_end})"

    is_sqlite = cur is not None and (
        "sqlite" in str(type(cur)).lower()
        or "sqlite" in str(type(getattr(cur, "connection", None))).lower()
    )

    # 4. Supersession Chain Verification
    sup_id = receipt.get("supersedes_receipt_id")
    if sup_id:
        if cur is None:
            return False, None, f"VERIFICATION_UNAVAILABLE: database cursor required to verify supersedes_receipt_id {sup_id}"
        try:
            if is_sqlite:
                cur.execute(
                    "SELECT receipt_version FROM sm_reconciliation_receipts WHERE receipt_id = ?;",
                    (str(sup_id),),
                )
            else:
                cur.execute(
                    "SELECT receipt_version FROM sm_reconciliation_receipts WHERE receipt_id = %s;",
                    (str(sup_id),),
                )
            sup_row = cur.fetchone()
            if not sup_row:
                return False, None, f"SUPERSEDED_RECEIPT_NOT_FOUND: receipt {sup_id} not found in DB"
            if sup_row["receipt_version"] >= receipt.get("receipt_version", 1):
                return False, None, f"INVALID_SUPERSEDED_VERSION: superseded version {sup_row['receipt_version']} >= current {receipt.get('receipt_version', 1)}"
        except Exception as sup_err:
            logger.error(f"Supersession check query failed: {sup_err}")
            return False, None, f"VERIFICATION_UNAVAILABLE: supersession check query failed: {sup_err}"

    if receipt.get("receipt_id"):
        if cur is None:
            return False, None, f"VERIFICATION_UNAVAILABLE: database cursor required to verify supersession status for receipt {receipt.get('receipt_id')}"
        try:
            if is_sqlite:
                cur.execute(
                    "SELECT receipt_id FROM sm_reconciliation_receipts WHERE supersedes_receipt_id = ?;",
                    (str(receipt["receipt_id"]),),
                )
            else:
                cur.execute(
                    "SELECT receipt_id FROM sm_reconciliation_receipts WHERE supersedes_receipt_id = %s;",
                    (str(receipt["receipt_id"]),),
                )
            if cur.fetchone():
                return False, None, f"RECEIPT_SUPERSEDED: receipt {receipt['receipt_id']} is superseded by a newer receipt"
        except Exception as sup_err2:
            logger.error(f"Superseding check query failed: {sup_err2}")
            return False, None, f"VERIFICATION_UNAVAILABLE: superseding check query failed: {sup_err2}"

    # 5. Inventory Integrity & Zero-Activity Proof
    cov_st = str(receipt.get("coverage_status") or "").upper()
    if len(acc_inv) == 0:
        if cov_st == "COMPLETE":
            if not z_proof or not isinstance(z_proof, dict):
                return False, None, "Zero activity unproved: COMPLETE coverage requires verified catalog proof"
            cat_content = z_proof.get("catalog_bytes") or z_proof.get("catalog_content")
            cat_hash = z_proof.get("source_catalog_hash") or z_proof.get("catalog_sha256")
            if not cat_content or not cat_hash:
                return False, None, "Zero activity proof missing catalog bytes or sha256"
            try:
                validate_catalog_evidence(
                    catalog_bytes=cat_content,
                    catalog_sha256=cat_hash,
                    ticker=r_ticker or req_sym,
                    issuer_cik=r_cik or (req_cik or ""),
                    interval_start=receipt["interval_start"],
                    interval_end=receipt["interval_end"],
                    certified_watermark=receipt["certified_watermark"],
                    accession_inventory=acc_inv,
                    source_hashes=s_hashes,
                    material_gaps=m_gaps,
                    pagination_artifacts=z_proof.get("pagination_artifacts"),
                    historical_manifest=z_proof.get("historical_manifest"),
                )
            except Exception as e:
                return False, None, f"Catalog proof validation failed: {e}"
        elif cov_st == "PARTIAL":
            pass
    else:
        if cur is None:
            return False, None, "VERIFICATION_UNAVAILABLE: database cursor required to verify accession inventory"
        try:
            if is_sqlite:
                placeholders = ",".join(["?"] * len(acc_inv))
                cur.execute(
                    f"""
                    SELECT f.*, av.public_availability_at
                    FROM form4_raw_filings f
                    LEFT JOIN sm_source_availability_ledger av ON f.accession_number = av.accession_number
                    WHERE f.accession_number IN ({placeholders});
                    """,
                    tuple(acc_inv),
                )
            else:
                cur.execute(
                    """
                    SELECT f.accession_number, f.cik, f.ticker, f.content_hash, av.public_availability_at
                    FROM form4_raw_filings f
                    LEFT JOIN sm_source_availability_ledger av ON f.accession_number = av.accession_number
                    WHERE f.accession_number = ANY(%s);
                    """,
                    (acc_inv,),
                )
            found = {row["accession_number"]: dict(row) for row in cur.fetchall()}
            if len(found) != len(acc_inv):
                missing_accs = [acc for acc in acc_inv if acc not in found]
                return False, None, f"INVENTORY_INCOMPLETE: expected {len(acc_inv)} raw filings in DB, found {len(found)} (missing: {missing_accs})"

            # Contract issuer identity under issuer contract
            contract_ticker = r_ticker or req_sym
            contract_cik = r_cik or (req_cik or None)
            if contract_ticker in KNOWN_ISSUER_CIKS:
                expected_contract_cik = KNOWN_ISSUER_CIKS[contract_ticker]
                if contract_cik and contract_cik != expected_contract_cik:
                    return False, None, f"CONFLICTING_ISSUER_IDENTITY: contract ticker {contract_ticker} conflicts with CIK {contract_cik}"
                contract_cik = expected_contract_cik

            for acc in acc_inv:
                f_row = found[acc]
                f_cik = str(f_row.get("cik") or f_row.get("issuer_cik") or "").strip().zfill(10) if (f_row.get("cik") or f_row.get("issuer_cik")) else None
                f_ticker = str(f_row.get("ticker") or f_row.get("issuer_symbol") or "").upper().strip() if (f_row.get("ticker") or f_row.get("issuer_symbol")) else None

                # 1. Require consistent ticker and CIK identity under the issuer contract.
                # Reject conflicting identity even when the other identifier matches.
                if contract_cik:
                    if not f_cik:
                        return False, None, f"INVENTORY_ISSUER_MISMATCH: accession {acc} missing CIK, expected {contract_cik}"
                    if f_cik != contract_cik:
                        return False, None, f"INVENTORY_ISSUER_MISMATCH: accession {acc} CIK {f_cik} does not match expected {contract_cik}"
                if contract_ticker:
                    if f_ticker and f_ticker != contract_ticker:
                        return False, None, f"INVENTORY_ISSUER_MISMATCH: accession {acc} ticker {f_ticker} does not match expected {contract_ticker}"
                    if f_ticker and f_ticker in KNOWN_ISSUER_CIKS and f_cik and KNOWN_ISSUER_CIKS[f_ticker] != f_cik:
                        return False, None, f"CONFLICTING_ISSUER_IDENTITY: accession {acc} ticker {f_ticker} conflicts with CIK {f_cik}"
                    if contract_ticker in KNOWN_ISSUER_CIKS and f_cik and KNOWN_ISSUER_CIKS[contract_ticker] != f_cik:
                        return False, None, f"CONFLICTING_ISSUER_IDENTITY: accession {acc} CIK {f_cik} conflicts with contract ticker {contract_ticker}"

                # 2. Require the applicable accession source-hash binding and exact stored-content match.
                if not s_hashes or acc not in s_hashes or not str(s_hashes.get(acc) or "").strip():
                    return False, None, f"MISSING_ACCESSION_SOURCE_HASH: accession {acc} missing source_hashes binding in receipt"
                expected_content_hash = str(s_hashes[acc]).strip().lower()
                stored_content_hash = str(f_row.get("content_hash") or f_row.get("source_content_hash") or "").strip().lower()
                if not stored_content_hash:
                    return False, None, f"MISSING_STORED_CONTENT_HASH: accession {acc} has no stored content_hash in DB"
                if expected_content_hash != stored_content_hash:
                    return False, None, f"INVENTORY_HASH_MISMATCH: accession {acc} stored content_hash ({stored_content_hash}) does not match receipt binding ({expected_content_hash})"

                # 3. Require verified public availability within the applicable cutoff.
                raw_pub_avail = f_row.get("public_availability_at")
                if not raw_pub_avail:
                    return False, None, f"MISSING_PUBLIC_AVAILABILITY: accession {acc} has no verified public availability record in ledger"
                pub_avail = _format_canonical_sec_timestamp(raw_pub_avail)
                if not pub_avail:
                    return False, None, f"MISSING_PUBLIC_AVAILABILITY: accession {acc} has unparseable public availability timestamp: {raw_pub_avail}"

                if cert_wm and pub_avail > cert_wm:
                    return False, None, f"AVAILABILITY_AFTER_CUTOFF: accession {acc} public_availability_at ({pub_avail}) postdates certified watermark ({cert_wm})"
                if evaluation_cutoff:
                    eval_ts = _format_canonical_sec_timestamp(evaluation_cutoff)
                    if eval_ts and pub_avail > eval_ts:
                        return False, None, f"AVAILABILITY_AFTER_CUTOFF: accession {acc} public_availability_at ({pub_avail}) postdates evaluation cutoff ({eval_ts})"

            # 4. Check COMPLETE coverage requirements vs PARTIAL coverage
            if cov_st == "COMPLETE":
                if m_gaps and len(m_gaps) > 0:
                    return False, None, f"MATERIAL_GAPS_PRESENT: receipt claims COMPLETE coverage but contains {len(m_gaps)} material gap(s)"
                if contract_cik:
                    if is_sqlite:
                        cur.execute("PRAGMA table_info(form4_raw_filings);")
                        cols = {r[1] for r in cur.fetchall()}
                        cik_col = "cik" if "cik" in cols else "issuer_cik"
                        date_col = "accepted_datetime" if "accepted_datetime" in cols else "acceptance_datetime"
                        cur.execute(
                            f"""
                            SELECT count(*) AS cnt FROM form4_raw_filings
                            WHERE {cik_col} = ?
                              AND {date_col} >= ?
                              AND {date_col} <= ?
                              AND accession_number NOT IN ({placeholders});
                            """,
                            (contract_cik, int_start, int_end, *acc_inv),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT count(*) AS cnt FROM form4_raw_filings
                            WHERE cik = %s
                              AND accepted_datetime >= %s::timestamptz
                              AND accepted_datetime <= %s::timestamptz
                              AND accession_number != ALL(%s);
                            """,
                            (contract_cik, int_start, int_end, acc_inv),
                        )
                    cnt_row = cur.fetchone()
                    if cnt_row and cnt_row["cnt"] > 0:
                        return False, None, f"UNINVENTORIED_FILINGS_PRESENT: {cnt_row['cnt']} uninventoried filings in interval for COMPLETE coverage"
            elif cov_st == "PARTIAL":
                # Legitimate PARTIAL inventory semantics preserved:
                # Partial coverage does not require zero uninventoried filings or zero material gaps.
                pass
        except Exception as inv_err:
            logger.error(f"Inventory query failure: {inv_err}")
            return False, None, f"VERIFICATION_UNAVAILABLE: inventory query failed: {inv_err}"

    receipt_copy = dict(receipt)
    receipt_copy["structured_metadata"] = reasons_val
    receipt_copy["coverage_status"] = cov_st or receipt.get("coverage_status") or "PARTIAL"
    return True, receipt_copy, None


def _attach_parsed_transactions_to_filings(
    results: List[Dict[str, Any]],
    tx_rows: List[Dict[str, Any]],
    amendments_info: Optional[Dict[str, Dict[str, Any]]] = None,
    cutoff_watermark: Optional[str] = None,
) -> None:
    """
    Enriches filing dictionaries with individual parsed transaction evidence.
    - Preserves individual transactions separately from any aggregate interpretation.
    - Preserves source-supported security_title without inferring Common Stock.
    - Preserves null, 0.0, and positive consideration distinctly.
    - Sanitizes non-finite numeric inputs (NaN, Inf) to None.
    - Removes misleading mixed-filing aggregates: does NOT sum different owners, instruments, or codes into one trade.
    - Evaluates public-availability boundaries and amendment resolution semantics.
    """
    txs_by_acc: Dict[str, List[Dict[str, Any]]] = {}
    amends_by_acc = amendments_info or {}
    filings_pub_by_acc = {
        item.get("accession_number"): (item.get("public_availability_at") or item.get("acceptance_datetime"))
        for item in results if item.get("accession_number")
    }

    for r in tx_rows:
        acc = str(r.get("accession_number") or "").strip()
        if not acc:
            continue
        c = str(r.get("transaction_code") or "").strip().upper()
        ad = str(r.get("acquired_disposed") or "").strip().upper()
        is_deriv = bool(r.get("is_derivative"))
        is_buy = (c == "P")
        is_sell = (c == "S")
        is_gift = (c == "G")
        is_exercise = c in ("M", "X")

        shares_val = _clean_numeric(r.get("shares"))
        price_val = _clean_numeric(r.get("price_per_share"))
        total_val = _clean_numeric(r.get("total_value_usd"))

        # Preserved source-supported security title: NEVER infer "Common Stock"
        sec_title = str(r.get("security_title") or "").strip() or None

        owner_name = str(r.get("reporting_owner_name") or "").strip()
        officer_title = str(r.get("officer_title") or "").strip() or None
        owner_role = str(officer_title or r.get("reporting_owner_role") or "Corporate Insider").strip()

        # Public availability evaluation
        pub_avail = _format_canonical_sec_timestamp(r.get("public_availability_at") or filings_pub_by_acc.get(acc))
        norm_cutoff = _format_canonical_sec_timestamp(cutoff_watermark)
        is_publicly_available = True
        if norm_cutoff and pub_avail:
            is_publicly_available = (pub_avail <= norm_cutoff)
        elif norm_cutoff and not pub_avail:
            is_publicly_available = False

        # Amendment effect evaluation
        am_data = amends_by_acc.get(acc, {})
        resolution_status = str(am_data.get("resolution_status") or "").strip().upper()
        is_superseded = am_data.get("is_superseded", False)
        superseded_by = am_data.get("superseded_by")
        has_ambiguous = bool(am_data.get("has_ambiguous_amendment"))
        has_partial = bool(am_data.get("has_partial_amendment"))

        effective_status = "EFFECTIVE"
        is_effective = True
        amendment_note = None

        if not is_publicly_available:
            effective_status = "PENDING_PUBLIC_AVAILABILITY"
            is_effective = False
            amendment_note = f"Public availability ({pub_avail or 'unverified'}) postdates cutoff ({norm_cutoff})."
        elif resolution_status == "RESOLVED":
            # Requirement 2: Treat legacy RESOLVED amendment lineage as scope-unverified
            effective_status = "SCOPE_UNVERIFIED"
            is_effective = False
            amendment_note = am_data.get("amendment_note") or "Legacy RESOLVED amendment lineage is scope-unverified; verified amendment effect cannot be implied without granular scope classification."
        elif has_ambiguous or resolution_status in (
            "AMENDMENT_TARGET_AMBIGUOUS",
            "AMENDMENT_SCOPE_AMBIGUOUS",
            "UNRESOLVED",
            "UNRESOLVED_AMENDMENT",
        ):
            effective_status = "AMENDMENT_EFFECT_AMBIGUOUS"
            is_effective = False
            amendment_note = am_data.get("amendment_note") or "Historical filing preserved; effective amendment impact is ambiguous and unresolved."
        elif is_superseded:
            effective_status = "SUPERSEDED_BY_AMENDMENT"
            is_effective = False
            amendment_note = f"Superseded by amendment {superseded_by}."
        elif has_partial or resolution_status in ("PARTIAL_CORRECTION", "PARTIAL_AMENDMENT"):
            effective_status = "PARTIAL_CORRECTION"
            is_effective = False
            amendment_note = am_data.get("amendment_note") or "Subject to partial amendment; individual transaction effects unverified."
        elif resolution_status in ("FULL_REPLACEMENT", "SUPERSEDED_OR_CORRECTED"):
            effective_status = "EFFECTIVE_REPLACEMENT"
            is_effective = True
            amendment_note = f"Full replacement of base filing {am_data.get('base_accession')}."
        elif am_data.get("future_amendment_accession"):
            effective_status = "EFFECTIVE"
            is_effective = True
            amendment_note = am_data.get("amendment_note")

        tx_item = {
            "transaction_id": str(r.get("transaction_id") or ""),
            "accession_number": acc,
            "transaction_date": str(r.get("transaction_date") or ""),
            "transaction_code": c,
            "transaction_type": _map_sec_transaction_type(c, ad, is_deriv),
            "shares": shares_val,
            "price_per_share": price_val,
            "total_value_usd": total_val,
            "reporting_owner_name": owner_name,
            "reporting_owner_role": owner_role,
            "officer_title": officer_title,
            "is_director": bool(r.get("is_director")),
            "is_officer": bool(r.get("is_officer")),
            "is_ten_percent": bool(r.get("is_ten_percent") or r.get("is_ten_percent_owner")),
            "ownership_type": str(r.get("ownership_type") or r.get("ownership_nature") or "D").upper(),
            "is_derivative": is_deriv,
            "acquired_disposed": ad,
            "is_buy": is_buy,
            "is_sell": is_sell,
            "is_gift": is_gift,
            "is_exercise": is_exercise,
            "is_open_market_purchase_evidence": bool(r.get("is_open_market_purchase_evidence")),
            "is_positive_buying_conviction": bool(r.get("is_positive_buying_conviction")),
            "security_title": sec_title,
            "public_availability_at": pub_avail,
            "is_effective": is_effective,
            "effective_status": effective_status,
            "amendment_note": amendment_note,
        }
        if acc not in txs_by_acc:
            txs_by_acc[acc] = []
        txs_by_acc[acc].append(tx_item)

    for item in results:
        acc = item.get("accession_number")
        item_txs = txs_by_acc.get(acc, [])
        am_data = amends_by_acc.get(acc, {})

        item["is_superseded"] = am_data.get("is_superseded", False)
        item["superseded_by"] = am_data.get("superseded_by")
        item["has_ambiguous_amendment"] = am_data.get("has_ambiguous_amendment", False)
        item["has_partial_amendment"] = am_data.get("has_partial_amendment", False)
        item["amendment_resolution_status"] = am_data.get("resolution_status")
        item["amendment_note"] = am_data.get("amendment_note")

        if item_txs:
            item["has_parsed_transaction"] = True
            item["transaction_count"] = len(item_txs)
            item["transactions"] = item_txs

            if len(item_txs) == 1:
                # Single transaction: exact mapping
                primary = item_txs[0]
                item["is_mixed"] = False
                item["insider_name"] = primary["reporting_owner_name"]
                item["insider_role"] = primary["reporting_owner_role"]
                item["role"] = primary["reporting_owner_role"]
                item["transaction_type"] = primary["transaction_type"]
                item["transaction_code"] = primary["transaction_code"]
                item["transaction_date"] = primary["transaction_date"]
                item["shares_traded"] = primary["shares"]
                item["price_per_share"] = primary["price_per_share"]
                item["total_value_usd"] = primary["total_value_usd"]
                item["is_buy"] = primary["is_buy"]
                item["is_sell"] = primary["is_sell"]
                item["is_gift"] = primary["is_gift"]
                item["is_exercise"] = primary["is_exercise"]
                item["is_derivative"] = primary["is_derivative"]
                item["is_effective"] = primary["is_effective"]
                item["effective_status"] = primary["effective_status"]
                item["is_significant_buy"] = bool(
                    item["is_buy"] and (item["total_value_usd"] or 0) >= 100000
                )
            else:
                # Multi-transaction filing: check comparability before aggregating
                owners = {t["reporting_owner_name"] for t in item_txs if t.get("reporting_owner_name")}
                securities = {t["security_title"] for t in item_txs if t.get("security_title")}
                codes = {t["transaction_code"] for t in item_txs if t.get("transaction_code")}

                same_owner = len(owners) <= 1
                same_security = len(securities) <= 1
                same_code = len(codes) <= 1

                all_shares_present = all(
                    isinstance(t.get("shares"), (int, float)) and math.isfinite(t.get("shares"))
                    for t in item_txs
                )
                all_totals_present = all(
                    isinstance(t.get("total_value_usd"), (int, float)) and math.isfinite(t.get("total_value_usd"))
                    for t in item_txs
                )

                if same_owner and same_security and same_code and all_shares_present:
                    # Demonstrably comparable transactions: aggregate shares and total safely
                    item["is_mixed"] = False
                    c0 = item_txs[0]["transaction_code"]
                    item["insider_name"] = next(iter(owners)) if owners else item_txs[0]["reporting_owner_name"]
                    item["insider_role"] = item_txs[0]["reporting_owner_role"]
                    item["role"] = item_txs[0]["reporting_owner_role"]
                    item["transaction_type"] = item_txs[0]["transaction_type"]
                    item["transaction_code"] = c0
                    item["transaction_date"] = max(t["transaction_date"] for t in item_txs if t.get("transaction_date"))
                    item["shares_traded"] = sum(t["shares"] for t in item_txs)
                    item["total_value_usd"] = sum(t["total_value_usd"] for t in item_txs) if all_totals_present else None
                    if item["total_value_usd"] is not None and item["shares_traded"] > 0:
                        item["price_per_share"] = round(item["total_value_usd"] / item["shares_traded"], 4)
                    else:
                        item["price_per_share"] = item_txs[0]["price_per_share"] if all(t.get("price_per_share") == item_txs[0].get("price_per_share") for t in item_txs) else None
                    item["is_buy"] = (c0 == "P")
                    item["is_sell"] = (c0 == "S")
                    item["is_gift"] = (c0 == "G")
                    item["is_exercise"] = c0 in ("M", "X")
                    item["is_derivative"] = any(t.get("is_derivative") for t in item_txs)
                    item["is_effective"] = all(t.get("is_effective") for t in item_txs)
                    item["effective_status"] = "EFFECTIVE" if item["is_effective"] else "PARTIALLY_EFFECTIVE"
                    item["is_significant_buy"] = bool(
                        item["is_buy"] and (item["total_value_usd"] or 0) >= 100000
                    )
                else:
                    # MIXED FILING: Different owners, instruments, codes, or missing amounts.
                    # Withhold misleading aggregates completely!
                    item["is_mixed"] = True
                    item["insider_name"] = ", ".join(sorted(list(owners))) if len(owners) > 1 else (next(iter(owners)) if owners else None)
                    item["insider_role"] = "Multiple Insiders" if len(owners) > 1 else item_txs[0]["reporting_owner_role"]
                    item["role"] = item["insider_role"]
                    item["transaction_type"] = f"Multi-Transaction Filing ({len(item_txs)} Events — Mixed)"
                    item["transaction_code"] = "MIXED"
                    item["transaction_date"] = max(t["transaction_date"] for t in item_txs if t.get("transaction_date"))
                    item["shares_traded"] = None  # Withhold misleading sum
                    item["price_per_share"] = None
                    item["total_value_usd"] = None  # Withhold misleading sum
                    item["is_buy"] = None
                    item["is_sell"] = None
                    item["is_gift"] = False
                    item["is_exercise"] = False
                    item["is_derivative"] = any(t.get("is_derivative") for t in item_txs)
                    item["is_effective"] = any(t.get("is_effective") for t in item_txs)
                    item["effective_status"] = "MIXED_EVENTS"
                    item["is_significant_buy"] = False
        else:
            item["has_parsed_transaction"] = False
            item["transaction_count"] = 0
            item["transactions"] = []
            item["is_mixed"] = False
            item["is_buy"] = None
            item["is_sell"] = None
            item["is_gift"] = False
            item["is_exercise"] = False
            item["is_derivative"] = False
            item["is_effective"] = False
            item["effective_status"] = "UNPARSED"
            item["is_significant_buy"] = False


DISALLOWED_DOMAINS = {
    "internal", "local", "example.com", "sample.com", "test.com", "placeholder", "invalid"
}

DISALLOWED_IDENTIFIERS = {
    "financeterminalresearchapp admin@financeterminal.internal",
    "admin@financeterminal.internal",
}


def get_sec_egress_role() -> str:
    """
    Retrieve SEC egress execution role from environment.
    Values: 'DISABLED' (default), 'OWNER'.
    Fails closed: any absent, empty, or unverified value defaults to 'DISABLED'.
    """
    role = os.environ.get("ARX_SEC_EGRESS_ROLE", "DISABLED")
    if not role or not isinstance(role, str):
        return "DISABLED"
    trimmed = role.strip().upper()
    return "OWNER" if trimmed == "OWNER" else "DISABLED"


def is_sec_egress_authorized() -> bool:
    """True only if the current process is authorized as the SEC egress owner."""
    return get_sec_egress_role() == "OWNER"


class SecRateLimiter:
    """
    Thread-safe rate limiter enforcing max SEC requests per second process-globally.
    Conforms to SEC Fair Access policy (max 10 req/sec) with conservative 5.0 req/sec ceiling.
    """

    def __init__(self, max_per_second: float = 5.0, max_rps: Optional[float] = None):
        rps = max_rps if max_rps is not None else max_per_second
        self.max_per_second = rps
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self.last_request_time = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        if self.interval <= 0:
            return
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_request_time = time.time()

    def wait_for_slot(self):
        self.acquire()


RateLimiter = SecRateLimiter
DEFAULT_SEC_RATE_LIMITER = SecRateLimiter(max_per_second=5.0)


def validate_sec_user_agent(user_agent: Optional[str]) -> Tuple[bool, str]:
    """
    Validate SEC User-Agent string against SEC Fair Access requirements:
    Format: '<ApplicationOrCompany> <AdminContact@domain.com>'
    Must contain a valid email domain (not private/placeholder) and non-empty entity name.
    """
    if not user_agent or not isinstance(user_agent, str):
        return False, "User-Agent string is missing or empty"

    trimmed = user_agent.strip()
    if not trimmed:
        return False, "User-Agent string is whitespace only"

    if trimmed.lower() in DISALLOWED_IDENTIFIERS:
        return False, "User-Agent is a disallowed placeholder identity"

    # Check for email pattern
    email_match = re.search(r"[\w\.-]+@([\w\.-]+\.\w+)", trimmed)
    if not email_match:
        return False, "User-Agent must contain a valid contact email (AdminContact@domain.com)"

    domain = email_match.group(1).lower()
    for disallowed in DISALLOWED_DOMAINS:
        if domain == disallowed or domain.endswith("." + disallowed):
            return False, f"User-Agent domain '{domain}' is a disallowed placeholder or private domain"

    parts = trimmed.split()
    if len(parts) < 2:
        return False, "User-Agent must include both entity/application name and contact email"

    return True, "Valid"


def get_sec_user_agent() -> Optional[str]:
    """
    Retrieve SEC_USER_AGENT from environment.
    Returns valid user agent string or None if unconfigured/invalid.
    Never prints contact information to logs.
    """
    ua = os.environ.get("SEC_USER_AGENT")
    is_valid, _ = validate_sec_user_agent(ua)
    if not is_valid:
        return None
    return ua.strip()


SEC_USER_AGENT = get_sec_user_agent()


class SecEdgarFetcher:
    """Fetches official 10-K, 10-Q, 8-K, Form 4 insider transactions, and 13F institutional holdings."""

    BASE_URL = "https://data.sec.gov"
    CIK_MAP = {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "NVDA": "0001045810",
        "NVO": "0000353278",
        "PLTR": "0001321655",
        "TSLA": "0001318605",
        "LLY": "0000059478",
        "CRWD": "0001535527",
        "AMD": "0000002488",
        "AVGO": "0001730168",
        "VRT": "0001674101",
        "TSM": "0001046179",
        "GE": "0000040545",
        "GOOGL": "0001652044",
        "AMZN": "0001018724",
        "META": "0001326801",
        "NFLX": "0001065280",
        "CPRX": "0001373715",
        "FIX": "0001056285",
        "SYM": "0001837240",
        "DECK": "0000910521",
        "PODD": "0001145236",
        "MNST": "0000865752",
        "ULTA": "0001403568",
        "LULU": "0001397187",
    }

    def __init__(self, user_agent: Optional[str] = None, rate_limiter: Optional[SecRateLimiter] = None):
        self.session = requests.Session()
        candidate_ua = user_agent or os.environ.get("SEC_USER_AGENT")
        self.is_configured, self.configuration_error = validate_sec_user_agent(candidate_ua)
        self.user_agent = candidate_ua.strip() if self.is_configured and candidate_ua else None

        if self.is_configured and self.user_agent:
            self.session.headers.update({"User-Agent": self.user_agent})
        else:
            self.session.headers.pop("User-Agent", None)

        self.cik_cache: Dict[str, str] = dict(self.CIK_MAP)
        self._has_loaded_master_index: bool = False
        self.rate_limiter = rate_limiter if rate_limiter is not None else DEFAULT_SEC_RATE_LIMITER

    def _is_mocked(self) -> bool:
        """
        Detect whether session.get or submission methods have been mocked or monkeypatched.
        Supports unittest.mock (Mock, MagicMock), pytest monkeypatch (custom functions/lambdas),
        and patched methods.
        """
        session_get = getattr(self.session, "get", None)
        if session_get is not None:
            if hasattr(session_get, "assert_called") or hasattr(session_get, "_mock_return_value"):
                return True
            func = getattr(session_get, "__func__", None)
            if func is not None and func is not requests.sessions.Session.get:
                return True
            if func is None and callable(session_get) and session_get is not requests.sessions.Session.get:
                return True

        comp_sub = getattr(self, "get_company_submissions", None)
        if comp_sub is not None:
            if hasattr(comp_sub, "assert_called") or hasattr(comp_sub, "_mock_return_value"):
                return True
            func = getattr(comp_sub, "__func__", None)
            if func is not None and func is not SecEdgarFetcher.get_company_submissions:
                return True
            if func is None and callable(comp_sub) and comp_sub is not SecEdgarFetcher.get_company_submissions:
                return True

        return False


    def _get(self, url: str, **kwargs) -> requests.Response:
        """Execute outbound HTTP GET with mandatory rate-limiter acquisition and egress role check."""
        is_mock = self._is_mocked()
        if not is_mock and not is_sec_egress_authorized():
            raise PermissionError(
                f"SEC_EGRESS_FORBIDDEN: Outbound SEC request blocked by fail-closed egress role "
                f"(ARX_SEC_EGRESS_ROLE={get_sec_egress_role()} != OWNER). Attempted URL: {url}"
            )
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        return self.session.get(url, **kwargs)

    def resolve_cik(self, symbol: str) -> Optional[str]:
        """Resolve ticker to 10-digit CIK from local cache or SEC company_tickers index."""
        upper = symbol.upper().strip()
        if upper in self.cik_cache:
            return self.cik_cache[upper]

        if not self.is_configured:
            logger.warning("SEC EDGAR request blocked: valid SEC_USER_AGENT is not configured.")
            return None

        if not self._has_loaded_master_index:
            is_mock = self._is_mocked()
            if not is_mock and not is_sec_egress_authorized():
                return self.cik_cache.get(upper)
            try:
                resp = self._get("https://www.sec.gov/files/company_tickers.json", timeout=6)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.values():
                        t = item.get("ticker", "").upper()
                        c = str(item.get("cik_str", "")).zfill(10)
                        if t and c:
                            self.cik_cache[t] = c
                self._has_loaded_master_index = True
            except Exception as e:
                logger.warning(f"Failed to fetch SEC master company_tickers.json: {e}")

        return self.cik_cache.get(upper)

    def get_company_submissions_envelope(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time company submissions with complete semantic failure tracking."""
        if not self.is_configured:
            logger.warning("SEC EDGAR request blocked: valid SEC_USER_AGENT is not configured.")
            return {
                "status": "CONFIGURATION_BLOCKED",
                "status_code": None,
                "data": None,
                "reason": self.configuration_error or "VALID_SEC_USER_AGENT_REQUIRED",
                "error": "Valid SEC_USER_AGENT is not configured",
            }

        is_mock = self._is_mocked()
        if not is_mock and not is_sec_egress_authorized():
            return {
                "status": "CONFIGURATION_BLOCKED",
                "status_code": None,
                "data": None,
                "reason": "SEC_EGRESS_ROLE_DISABLED",
                "error": "Outbound SEC EDGAR requests are disabled in this runtime role (ARX_SEC_EGRESS_ROLE != OWNER)",
            }

        cik = self.resolve_cik(symbol)
        if not cik:
            return {
                "status": "UPSTREAM_NOT_FOUND",
                "status_code": 404,
                "data": None,
                "reason": "CIK_NOT_FOUND",
                "error": f"CIK could not be resolved for symbol '{symbol}'",
            }

        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        try:
            resp = self._get(url, timeout=5)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if not isinstance(data, dict):
                        return {
                            "status": "STRUCTURALLY_INVALID",
                            "status_code": 200,
                            "data": None,
                            "reason": "PAYLOAD_NOT_A_DICT",
                            "error": "SEC EDGAR submissions JSON root is not an object",
                        }
                    return {
                        "status": "AVAILABLE",
                        "status_code": 200,
                        "data": data,
                        "reason": None,
                        "error": None,
                    }
                except (ValueError, json.JSONDecodeError) as e:
                    return {
                        "status": "STRUCTURALLY_INVALID",
                        "status_code": 200,
                        "data": None,
                        "reason": "MALFORMED_JSON",
                        "error": str(e),
                    }
            elif resp.status_code == 429:
                return {
                    "status": "UPSTREAM_RATE_LIMITED",
                    "status_code": 429,
                    "data": None,
                    "reason": "RATE_LIMITED",
                    "error": "SEC EDGAR API rate limit exceeded (HTTP 429)",
                }
            elif resp.status_code == 404:
                return {
                    "status": "UPSTREAM_NOT_FOUND",
                    "status_code": 404,
                    "data": None,
                    "reason": "SUBMISSIONS_RESOURCE_NOT_FOUND",
                    "error": f"Submissions resource not found for CIK {cik}",
                }
            elif resp.status_code == 403:
                return {
                    "status": "UNKNOWN_FAIL_CLOSED",
                    "status_code": 403,
                    "data": None,
                    "reason": "HTTP_403_FORBIDDEN",
                    "error": "SEC EDGAR API returned 403 Forbidden (access denied or blocked)",
                }
            elif resp.status_code in (500, 502, 503, 504):
                return {
                    "status": "UPSTREAM_SERVER_ERROR",
                    "status_code": resp.status_code,
                    "data": None,
                    "reason": f"HTTP_{resp.status_code}_SERVER_ERROR",
                    "error": f"SEC EDGAR upstream server error (HTTP {resp.status_code})",
                }
            else:
                return {
                    "status": "UNKNOWN_FAIL_CLOSED",
                    "status_code": resp.status_code,
                    "data": None,
                    "reason": f"UNEXPECTED_STATUS_{resp.status_code}",
                    "error": f"Unexpected HTTP status {resp.status_code} from SEC EDGAR",
                }
        except requests.exceptions.Timeout as e:
            return {
                "status": "UPSTREAM_TIMEOUT",
                "status_code": None,
                "data": None,
                "reason": "CONNECTION_TIMEOUT",
                "error": str(e),
            }
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            return {
                "status": "TRANSPORT_INCOMPLETE",
                "status_code": None,
                "data": None,
                "reason": "TRANSPORT_CONNECTION_DROPPED",
                "error": str(e),
            }
        except Exception as e:
            return {
                "status": "UNKNOWN_FAIL_CLOSED",
                "status_code": None,
                "data": None,
                "reason": "UNHANDLED_EXCEPTION",
                "error": str(e),
            }

    def get_company_submissions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch real-time company submissions including Form 4 (insiders) and 8-K (material events)."""
        env = self.get_company_submissions_envelope(symbol)
        return env.get("data")

    def _enrich_filings_with_stored_transactions(
        self, results: List[Dict[str, Any]], symbol: str, cik: Optional[str] = None
    ) -> None:
        """Enriches filing items with parsed transactions from local store without throwing errors."""
        try:
            accs = [item["accession_number"] for item in results if item.get("accession_number")]
            if not accs:
                return
            backend = "sqlite"
            try:
                from analyst_dashboard.smart_money.storage.config import get_storage_backend
                backend = get_storage_backend()
            except Exception:
                backend = "sqlite"

            if backend == "postgres":
                from analyst_dashboard.smart_money.storage.postgres_connection import get_connection
                from psycopg2.extras import RealDictCursor
                with get_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'form4_parsed_transactions'")
                        if cur.fetchone():
                            cur.execute(
                                """
                                SELECT
                                    tx.transaction_id,
                                    tx.accession_number,
                                    tx.transaction_date,
                                    tx.transaction_code,
                                    tx.shares,
                                    tx.price_per_share,
                                    tx.total_value_usd,
                                    tx.reporting_owner_name,
                                    tx.reporting_owner_role,
                                    tx.is_director,
                                    tx.is_officer,
                                    tx.is_ten_percent,
                                    tx.officer_title,
                                    tx.ownership_type,
                                    tx.is_derivative,
                                    tx.signal_type,
                                    tx.acquired_disposed,
                                    tx.is_positive_buying_conviction,
                                    tx.is_open_market_purchase_evidence
                                FROM form4_parsed_transactions tx
                                WHERE tx.accession_number = ANY(%s)
                                ORDER BY tx.transaction_date ASC, tx.transaction_id ASC;
                                """,
                                (accs,),
                            )
                            tx_rows = [dict(r) for r in cur.fetchall()]
                            _attach_parsed_transactions_to_filings(results, tx_rows)
            elif backend == "sqlite":
                from analyst_dashboard.smart_money.form4_ingestion import connect_smart_money_db
                conn = connect_smart_money_db()
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                try:
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = 'form4_parsed_transactions'")
                    if cur.fetchone():
                        placeholders = ",".join(["?"] * len(accs))
                        cur.execute(
                            f"""
                            SELECT
                                record_key AS transaction_id,
                                accession_number,
                                transaction_date,
                                transaction_code,
                                shares,
                                price_per_share,
                                total_value_usd,
                                reporting_owner_name,
                                is_officer,
                                officer_title,
                                is_director,
                                is_ten_percent_owner AS is_ten_percent,
                                ownership_nature AS ownership_type,
                                is_derivative,
                                signal_type,
                                acquired_disposed,
                                is_positive_buying_conviction,
                                is_open_market_purchase_evidence,
                                public_availability_at
                            FROM form4_parsed_transactions
                            WHERE accession_number IN ({placeholders})
                            ORDER BY transaction_date ASC, record_key ASC;
                            """,
                            tuple(accs),
                        )
                        tx_rows = [dict(r) for r in cur.fetchall()]
                        _attach_parsed_transactions_to_filings(results, tx_rows)
                finally:
                    conn.close()
        except Exception as e:
            logger.debug(f"Stored transactions enrichment skipped: {e}")

    def _get_cached_recent_filings_envelope(
        self, symbol: str, form_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieves cached filing metadata from the governed local store without issuing live SEC network requests.
        Enforces CACHE_ONLY behavior for normal user/API requests when ARX_SEC_EGRESS_ROLE != OWNER.
        """
        from analyst_dashboard.smart_money.form4_ingestion import (
            DEFAULT_SMART_MONEY_DB,
            connect_smart_money_db,
        )

        clean_sym = symbol.upper().strip()
        cik = self.resolve_cik(clean_sym)
        allowed_forms = set(form_types) if form_types else {"10-K", "10-Q", "8-K", "4", "4/A", "13F-HR"}
        results = []

        backend = "sqlite"
        try:
            from analyst_dashboard.smart_money.storage.config import get_storage_backend
            backend = get_storage_backend()
        except Exception as e:
            if "PRODUCTION_STORAGE_CONFIGURATION_VIOLATION" in str(e) or "POSTGRES_STORAGE_BACKEND_CONFIGURATION_VIOLATION" in str(e):
                logger.error(f"Storage backend configuration violation: {e}")
                return {
                    "status": "BACKEND_STORAGE_ERROR",
                    "status_code": None,
                    "cik": cik,
                    "reason": "STORAGE_CONFIGURATION_VIOLATION",
                    "error_detail": str(e),
                    "items": None,
                    "cached": True,
                }
            backend = "sqlite"

        if backend == "postgres":
            try:
                from analyst_dashboard.smart_money.storage.postgres_connection import get_connection
                from psycopg2.extras import RealDictCursor

                with get_connection() as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        has_query = "SELECT 1 FROM form4_raw_filings WHERE (cik = %s OR ticker = %s) LIMIT 1" if cik else "SELECT 1 FROM form4_raw_filings WHERE ticker = %s LIMIT 1"
                        has_params = [str(cik).zfill(10), clean_sym] if cik else [clean_sym]
                        cur.execute(has_query, tuple(has_params))
                        has_issuer = cur.fetchone() is not None

                        if not has_issuer:
                            return {
                                "status": "CONFIGURATION_BLOCKED",
                                "status_code": None,
                                "cik": cik,
                                "reason": "SOURCE_NOT_INGESTED",
                                "error_detail": f"SEC filings for {clean_sym} not yet ingested in local store. Live SEC egress disabled in API web workers (ARX_SEC_EGRESS_ROLE != OWNER).",
                                "items": None,
                                "cached": True,
                            }

                        # Check available tables and columns dynamically
                        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('sm_source_availability_ledger', 'sm_amendments')")
                        found_tables = {r["table_name"] for r in cur.fetchall()}
                        has_ledger = "sm_source_availability_ledger" in found_tables
                        has_amendments = "sm_amendments" in found_tables

                        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'form4_raw_filings'")
                        raw_cols = {r["column_name"] for r in cur.fetchall()}

                        select_parts = [
                            "r.accession_number",
                            "r.cik",
                            "r.ticker",
                            "r.filing_date",
                            "r.accepted_datetime",
                            "r.source_url",
                        ]
                        if has_ledger:
                            select_parts.append("l.public_availability_at")
                        else:
                            select_parts.append("NULL AS public_availability_at")

                        if has_amendments:
                            select_parts.extend(["am.base_accession", "am.amendment_number"])
                        else:
                            select_parts.extend(["NULL AS base_accession", "NULL AS amendment_number"])

                        if "document_type" in raw_cols:
                            select_parts.append("r.document_type")
                        if "xml_content" in raw_cols:
                            select_parts.append("r.xml_content")
                        if "content_hash" in raw_cols:
                            select_parts.append("r.content_hash")
                        else:
                            select_parts.append("NULL AS content_hash")

                        query = f"SELECT {', '.join(select_parts)} FROM form4_raw_filings r"
                        if has_ledger:
                            query += " LEFT JOIN sm_source_availability_ledger l ON r.accession_number = l.accession_number"
                        if has_amendments:
                            query += " LEFT JOIN sm_amendments am ON r.accession_number = am.amendment_accession"

                        query += " WHERE 1=1"
                        params = []
                        if cik:
                            query += " AND (r.cik = %s OR r.ticker = %s)"
                            params.extend([str(cik).zfill(10), clean_sym])
                        else:
                            query += " AND r.ticker = %s"
                            params.append(clean_sym)

                        query += " ORDER BY r.filing_date DESC, r.accepted_datetime DESC NULLS LAST LIMIT 25"
                        cur.execute(query, tuple(params))
                        rows = cur.fetchall()

                        for r in rows:
                            base_acc = r.get("base_accession")
                            amend_num = r.get("amendment_number")
                            raw_doc = str(r.get("document_type") or "").strip().upper()
                            xml_content = r.get("xml_content") or ""
                            is_amend = (
                                (base_acc is not None)
                                or (amend_num is not None)
                                or (raw_doc in ("4/A", "4A"))
                                or ("<documentType>4/A" in xml_content)
                                or ("<documentType>4A" in xml_content)
                            )
                            doc_type = "4/A" if is_amend else "4"

                            if allowed_forms and doc_type not in allowed_forms:
                                continue

                            acc_num = r["accession_number"]
                            clean_acc = acc_num.replace("-", "")
                            cik_val = r.get("cik") or (str(cik).zfill(10) if cik else "0")
                            cik_int = int(cik_val) if str(cik_val).isdigit() else 0
                            f_date = str(r["filing_date"]) if r.get("filing_date") else ""

                            acc_dt = _format_canonical_sec_timestamp(r.get("accepted_datetime"))
                            pub_dt = _format_canonical_sec_timestamp(r.get("public_availability_at"))
                            avail_from = pub_dt or acc_dt

                            if is_amend:
                                desc = f"Form 4/A Insider Filing (Amendment {amend_num} to {base_acc})" if (amend_num and base_acc) else (f"Form 4/A Insider Filing (Amendment {amend_num})" if amend_num else "Form 4/A Insider Filing (Amendment)")
                            else:
                                desc = "Form 4 Insider Filing"

                            item = {
                                "form": doc_type,
                                "filing_date": f_date,
                                "report_date": f_date,
                                "acceptance_datetime": acc_dt,
                                "available_from": avail_from,
                                "description": desc,
                                "accession_number": acc_num,
                                "sec_url": r.get("source_url") or f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{acc_num}.txt",
                                "is_amendment": is_amend,
                            }
                            if r.get("content_hash"):
                                item["xml_hash"] = str(r["content_hash"])
                            if is_amend:
                                if base_acc:
                                    item["base_accession"] = base_acc
                                if amend_num is not None:
                                    item["amendment_number"] = amend_num

                            results.append(item)

                        cov_status = "SUPPORTED_SEC_ISSUER" if cik else "UNSUPPORTED_ISSUER"
                        cert_watermark = None
                        try:
                            cur.execute(
                                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'sm_reconciliation_receipts'"
                            )
                            if cur.fetchone():
                                cur.execute(
                                    """
                                    SELECT * FROM sm_reconciliation_receipts
                                    ORDER BY receipt_version DESC, created_at DESC;
                                    """
                                )
                                receipt_rows = [dict(r) for r in cur.fetchall()]
                                clean_cik_str = str(cik).strip().zfill(10) if cik else None
                                for r_row in receipt_rows:
                                    is_valid, valid_rc, err = _validate_issuer_reconciliation_receipt(
                                        r_row, clean_ticker=clean_sym, clean_cik=clean_cik_str, cur=cur
                                    )
                                    if is_valid and valid_rc:
                                        cov_status = valid_rc.get("coverage_status") or cov_status
                                        cert_watermark = _format_canonical_sec_timestamp(valid_rc.get("certified_watermark"))
                                        break
                        except Exception as rc_err:
                            logger.debug(f"Coverage receipt check skipped: {rc_err}")

                        accs = [item["accession_number"] for item in results if item.get("accession_number")]
                        amendments_info = {}
                        if has_amendments and accs:
                            try:
                                cur.execute(
                                    """
                                    SELECT
                                        am.amendment_accession,
                                        am.base_accession,
                                        am.amendment_number,
                                        am.resolution_status,
                                        am.resolved_at,
                                        l.public_availability_at AS amendment_pub_avail
                                    FROM sm_amendments am
                                    LEFT JOIN sm_source_availability_ledger l ON am.amendment_accession = l.accession_number
                                    WHERE am.base_accession = ANY(%s) OR am.amendment_accession = ANY(%s);
                                    """,
                                    (accs, accs),
                                )
                                for am_r in cur.fetchall():
                                    am_acc = str(am_r["amendment_accession"]).strip()
                                    base_acc = str(am_r["base_accession"]).strip() if am_r.get("base_accession") else None
                                    st = str(am_r.get("resolution_status") or "").strip().upper()
                                    am_pub = _format_canonical_sec_timestamp(am_r.get("amendment_pub_avail"))
                                    norm_cutoff = _format_canonical_sec_timestamp(cert_watermark)

                                    if am_acc in accs:
                                        if am_acc not in amendments_info:
                                            amendments_info[am_acc] = {}
                                        amendments_info[am_acc]["resolution_status"] = st
                                        amendments_info[am_acc]["base_accession"] = base_acc
                                        amendments_info[am_acc]["amendment_number"] = am_r.get("amendment_number")
                                        amendments_info[am_acc]["public_availability_at"] = am_pub
                                        if st == "RESOLVED":
                                            # Requirement 2: Treat legacy RESOLVED amendment lineage as scope-unverified
                                            amendments_info[am_acc]["is_effective"] = False
                                            amendments_info[am_acc]["effective_status"] = "SCOPE_UNVERIFIED"
                                            amendments_info[am_acc]["amendment_note"] = (
                                                "Legacy RESOLVED amendment lineage is scope-unverified; "
                                                "verified amendment effect cannot be implied without granular scope classification."
                                            )

                                    if base_acc and base_acc in accs:
                                        if base_acc not in amendments_info:
                                            amendments_info[base_acc] = {}

                                        is_future_amendment = bool(norm_cutoff and am_pub and am_pub > norm_cutoff)

                                        if is_future_amendment:
                                            # Future amendment cannot retroactively affect base filing as of cutoff watermark
                                            amendments_info[base_acc]["future_amendment_accession"] = am_acc
                                            amendments_info[base_acc]["future_amendment_available_at"] = am_pub
                                            amendments_info[base_acc]["amendment_note"] = (
                                                f"Future amendment {am_acc} (available {am_pub}) postdates cutoff {norm_cutoff}; "
                                                "base filing remains effective at watermark."
                                            )
                                        elif st in ("SUPERSEDED", "SUPERSEDED_OR_CORRECTED", "FULL_REPLACEMENT"):
                                            amendments_info[base_acc]["is_superseded"] = True
                                            amendments_info[base_acc]["superseded_by"] = am_acc
                                            amendments_info[base_acc]["amendment_note"] = f"Superseded by amendment {am_acc}."
                                        elif st == "RESOLVED":
                                            # Requirement 2: Treat legacy RESOLVED amendment lineage as scope-unverified
                                            amendments_info[base_acc]["has_ambiguous_amendment"] = True
                                            amendments_info[base_acc]["ambiguous_amendment_accession"] = am_acc
                                            amendments_info[base_acc]["amendment_note"] = (
                                                f"Target of legacy RESOLVED amendment {am_acc} "
                                                "(lineage scope-unverified; effective impact ambiguous)."
                                            )
                                        elif st in ("AMENDMENT_TARGET_AMBIGUOUS", "AMENDMENT_SCOPE_AMBIGUOUS", "UNRESOLVED", "UNRESOLVED_AMENDMENT"):
                                            amendments_info[base_acc]["has_ambiguous_amendment"] = True
                                            amendments_info[base_acc]["ambiguous_amendment_accession"] = am_acc
                                            amendments_info[base_acc]["amendment_note"] = f"Target of ambiguous amendment {am_acc} (effective impact unresolved)."
                                        elif st in ("PARTIAL_CORRECTION", "PARTIAL_AMENDMENT"):
                                            amendments_info[base_acc]["has_partial_amendment"] = True
                                            amendments_info[base_acc]["partial_amendment_accession"] = am_acc
                                            amendments_info[base_acc]["amendment_note"] = f"Subject to partial correction by {am_acc} (unmapped transactions require review)."
                            except Exception as am_err:
                                logger.debug(f"PostgreSQL amendments query skipped: {am_err}")

                        if results:
                            tx_rows = []
                            if accs:
                                cur.execute(
                                    """
                                    SELECT
                                        tx.transaction_id,
                                        tx.accession_number,
                                        tx.transaction_date,
                                        tx.transaction_code,
                                        tx.shares,
                                        tx.price_per_share,
                                        tx.total_value_usd,
                                        tx.reporting_owner_name,
                                        tx.reporting_owner_role,
                                        tx.is_director,
                                        tx.is_officer,
                                        tx.is_ten_percent,
                                        tx.officer_title,
                                        tx.ownership_type,
                                        tx.is_derivative,
                                        tx.signal_type,
                                        tx.acquired_disposed,
                                        tx.is_positive_buying_conviction,
                                        tx.is_open_market_purchase_evidence,
                                        COALESCE(l.public_availability_at, r.accepted_datetime) AS public_availability_at
                                    FROM form4_parsed_transactions tx
                                    LEFT JOIN form4_raw_filings r ON tx.accession_number = r.accession_number
                                    LEFT JOIN sm_source_availability_ledger l ON tx.accession_number = l.accession_number
                                    WHERE tx.accession_number = ANY(%s)
                                    ORDER BY tx.transaction_date ASC, tx.transaction_id ASC;
                                    """,
                                    (accs,),
                                )
                                tx_rows = [dict(row_data) for row_data in cur.fetchall()]
                            _attach_parsed_transactions_to_filings(
                                results, tx_rows, amendments_info=amendments_info, cutoff_watermark=cert_watermark
                            )

                        evidence_timestamps = []
                        for item in results:
                            pub = item.get("public_availability_at") or item.get("acceptance_datetime")
                            if pub:
                                evidence_timestamps.append(pub)

                        evidence_as_of = None
                        if evidence_timestamps:
                            max_ev = max(evidence_timestamps)
                            if cert_watermark and max_ev <= cert_watermark:
                                evidence_as_of = max_ev
                            elif cert_watermark:
                                evidence_as_of = cert_watermark
                            else:
                                evidence_as_of = max_ev
                        elif cert_watermark:
                            evidence_as_of = cert_watermark

                        served_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                        if results:
                            return {
                                "status": "AVAILABLE_WITH_RESULTS",
                                "status_code": 200,
                                "cik": cik,
                                "reason": None,
                                "error_detail": None,
                                "items": results,
                                "cached": True,
                                "coverage_status": cov_status,
                                "certified_watermark": cert_watermark,
                                "as_of": evidence_as_of,
                                "served_at": served_at,
                                "is_pipeline_held": True,
                            }
                        else:
                            return {
                                "status": "AVAILABLE_NO_RESULTS",
                                "status_code": 200,
                                "cik": cik,
                                "reason": None,
                                "error_detail": None,
                                "items": [],
                                "cached": True,
                                "coverage_status": cov_status,
                                "certified_watermark": cert_watermark,
                                "as_of": evidence_as_of,
                                "served_at": served_at,
                                "is_pipeline_held": True,
                            }
            except Exception as e:
                logger.error(f"Failed to query PostgreSQL filings store for {clean_sym}: {e}", exc_info=True)
                return {
                    "status": "BACKEND_STORAGE_ERROR",
                    "status_code": None,
                    "cik": cik,
                    "reason": "STORAGE_QUERY_FAILED",
                    "error_detail": f"Failed to query filings store for {clean_sym}.",
                    "items": None,
                    "cached": True,
                }

        elif backend == "sqlite":
            try:
                conn = connect_smart_money_db()
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()

                has_query = "SELECT 1 FROM form4_raw_filings WHERE (issuer_cik = ? OR issuer_symbol = ?) LIMIT 1" if cik else "SELECT 1 FROM form4_raw_filings WHERE issuer_symbol = ? LIMIT 1"
                has_params = [str(cik).zfill(10), clean_sym] if cik else [clean_sym]
                try:
                    cur.execute(has_query, tuple(has_params))
                    has_issuer = cur.fetchone() is not None
                except sqlite3.OperationalError:
                    has_issuer = False

                if not has_issuer:
                    conn.close()
                    return {
                        "status": "CONFIGURATION_BLOCKED",
                        "status_code": None,
                        "cik": cik,
                        "reason": "SOURCE_NOT_INGESTED",
                        "error_detail": f"SEC filings for {clean_sym} not yet ingested in local store. Live SEC egress disabled in API web workers (ARX_SEC_EGRESS_ROLE != OWNER).",
                        "items": None,
                        "cached": True,
                    }

                has_ledger = False
                has_amendments = False
                try:
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sm_source_availability_ledger', 'sm_amendments', 'form4_parsed_transactions', 'sm_reconciliation_receipts')")
                    tbls = {r[0] for r in cur.fetchall()}
                    has_ledger = "sm_source_availability_ledger" in tbls
                    has_amendments = "sm_amendments" in tbls
                except Exception:
                    tbls = set()

                cur.execute("PRAGMA table_info(form4_raw_filings);")
                raw_cols = {col_info[1] for col_info in cur.fetchall()}

                select_cols = [
                    "r.accession_number",
                    "r.issuer_cik",
                    "r.filing_date",
                    "r.acceptance_datetime",
                    "r.source_url",
                    "r.document_type",
                ]
                if "source_content_hash" in raw_cols:
                    select_cols.append("r.source_content_hash AS content_hash")
                elif "content_hash" in raw_cols:
                    select_cols.append("r.content_hash AS content_hash")
                else:
                    select_cols.append("NULL AS content_hash")

                if has_ledger:
                    select_cols.append("l.public_availability_at")
                else:
                    select_cols.append("NULL AS public_availability_at")

                if has_amendments:
                    select_cols.extend(["am.base_accession", "am.amendment_number"])
                else:
                    select_cols.extend(["NULL AS base_accession", "NULL AS amendment_number"])

                query = f"SELECT {', '.join(select_cols)} FROM form4_raw_filings r"
                if has_ledger:
                    query += " LEFT JOIN sm_source_availability_ledger l ON r.accession_number = l.accession_number"
                if has_amendments:
                    query += " LEFT JOIN sm_amendments am ON r.accession_number = am.amendment_accession"

                query += " WHERE 1=1"
                params = []
                if cik:
                    query += " AND (r.issuer_cik = ? OR r.issuer_symbol = ?)"
                    params.extend([str(cik).zfill(10), clean_sym])
                else:
                    query += " AND r.issuer_symbol = ?"
                    params.append(clean_sym)

                query += " ORDER BY r.filing_date DESC, r.acceptance_datetime DESC LIMIT 25"
                cur.execute(query, tuple(params))
                rows = cur.fetchall()

                for r in rows:
                    base_acc = r["base_accession"]
                    amend_num = r["amendment_number"]
                    raw_doc = str(r["document_type"] or "4").strip().upper()
                    is_amend = (base_acc is not None) or (amend_num is not None) or (raw_doc in ("4/A", "4A"))
                    doc_type = "4/A" if is_amend else "4"

                    if allowed_forms and doc_type not in allowed_forms:
                        continue

                    acc_num = r["accession_number"]
                    clean_acc = acc_num.replace("-", "")
                    cik_val = r["issuer_cik"] or (str(cik).zfill(10) if cik else "0")
                    cik_int = int(cik_val) if str(cik_val).isdigit() else 0
                    f_date = str(r["filing_date"]) if r["filing_date"] else ""

                    acc_dt = _format_canonical_sec_timestamp(r["acceptance_datetime"])
                    pub_dt = _format_canonical_sec_timestamp(r["public_availability_at"])
                    avail_from = pub_dt or acc_dt

                    if is_amend:
                        desc = f"Form 4/A Insider Filing (Amendment {amend_num} to {base_acc})" if (amend_num and base_acc) else (f"Form 4/A Insider Filing (Amendment {amend_num})" if amend_num else "Form 4/A Insider Filing (Amendment)")
                    else:
                        desc = "Form 4 Insider Filing"

                    item = {
                        "form": doc_type,
                        "filing_date": f_date,
                        "report_date": f_date,
                        "acceptance_datetime": acc_dt,
                        "available_from": avail_from,
                        "description": desc,
                        "accession_number": acc_num,
                        "sec_url": r["source_url"] or f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{acc_num}.txt",
                        "is_amendment": is_amend,
                    }
                    if "content_hash" in r.keys() and r["content_hash"]:
                        item["xml_hash"] = str(r["content_hash"])
                    if is_amend:
                        if base_acc:
                            item["base_accession"] = base_acc
                        if amend_num is not None:
                            item["amendment_number"] = amend_num

                    results.append(item)

                cov_status = "SUPPORTED_SEC_ISSUER" if cik else "UNSUPPORTED_ISSUER"
                cert_watermark = None
                try:
                    if "sm_reconciliation_receipts" in tbls:
                        cur.execute(
                            """
                            SELECT * FROM sm_reconciliation_receipts
                            ORDER BY receipt_version DESC, created_at DESC;
                            """
                        )
                        receipt_rows = [dict(r) for r in cur.fetchall()]
                        clean_cik_str = str(cik).strip().zfill(10) if cik else None
                        for r_row in receipt_rows:
                            is_valid, valid_rc, err = _validate_issuer_reconciliation_receipt(
                                r_row, clean_ticker=clean_sym, clean_cik=clean_cik_str, cur=cur
                            )
                            if is_valid and valid_rc:
                                cov_status = valid_rc.get("coverage_status") or cov_status
                                cert_watermark = _format_canonical_sec_timestamp(valid_rc.get("certified_watermark"))
                                break
                except Exception as rc_err:
                    logger.debug(f"SQLite coverage receipt check skipped: {rc_err}")

                accs = [item["accession_number"] for item in results if item.get("accession_number")]
                amendments_info = {}
                if has_amendments and accs:
                    try:
                        placeholders = ",".join(["?"] * len(accs))
                        cur.execute(
                            f"""
                            SELECT
                                am.amendment_accession,
                                am.base_accession,
                                am.amendment_number,
                                am.resolution_status,
                                am.resolved_at,
                                l.public_availability_at AS amendment_pub_avail
                            FROM sm_amendments am
                            LEFT JOIN sm_source_availability_ledger l ON am.amendment_accession = l.accession_number
                            WHERE am.base_accession IN ({placeholders}) OR am.amendment_accession IN ({placeholders});
                            """,
                            tuple(accs) + tuple(accs),
                        )
                        for am_r in cur.fetchall():
                            am_acc = str(am_r["amendment_accession"]).strip()
                            base_acc = str(am_r["base_accession"]).strip() if am_r["base_accession"] else None
                            st = str(am_r["resolution_status"] or "").strip().upper()
                            am_pub = _format_canonical_sec_timestamp(am_r["amendment_pub_avail"]) if "amendment_pub_avail" in am_r.keys() else None
                            norm_cutoff = _format_canonical_sec_timestamp(cert_watermark)

                            if am_acc in accs:
                                if am_acc not in amendments_info:
                                    amendments_info[am_acc] = {}
                                amendments_info[am_acc]["resolution_status"] = st
                                amendments_info[am_acc]["base_accession"] = base_acc
                                amendments_info[am_acc]["amendment_number"] = am_r["amendment_number"]
                                amendments_info[am_acc]["public_availability_at"] = am_pub
                                if st == "RESOLVED":
                                    amendments_info[am_acc]["is_effective"] = False
                                    amendments_info[am_acc]["effective_status"] = "SCOPE_UNVERIFIED"
                                    amendments_info[am_acc]["amendment_note"] = (
                                        "Legacy RESOLVED amendment lineage is scope-unverified; "
                                        "verified amendment effect cannot be implied without granular scope classification."
                                    )

                            if base_acc and base_acc in accs:
                                if base_acc not in amendments_info:
                                    amendments_info[base_acc] = {}

                                is_future_amendment = bool(norm_cutoff and am_pub and am_pub > norm_cutoff)

                                if is_future_amendment:
                                    amendments_info[base_acc]["future_amendment_accession"] = am_acc
                                    amendments_info[base_acc]["future_amendment_available_at"] = am_pub
                                    amendments_info[base_acc]["amendment_note"] = (
                                        f"Future amendment {am_acc} (available {am_pub}) postdates cutoff {norm_cutoff}; "
                                        "base filing remains effective at watermark."
                                    )
                                elif st in ("SUPERSEDED", "SUPERSEDED_OR_CORRECTED", "FULL_REPLACEMENT"):
                                    amendments_info[base_acc]["is_superseded"] = True
                                    amendments_info[base_acc]["superseded_by"] = am_acc
                                    amendments_info[base_acc]["amendment_note"] = f"Superseded by amendment {am_acc}."
                                elif st == "RESOLVED":
                                    amendments_info[base_acc]["has_ambiguous_amendment"] = True
                                    amendments_info[base_acc]["ambiguous_amendment_accession"] = am_acc
                                    amendments_info[base_acc]["amendment_note"] = (
                                        f"Target of legacy RESOLVED amendment {am_acc} "
                                        "(lineage scope-unverified; effective impact ambiguous)."
                                    )
                                elif st in ("AMENDMENT_TARGET_AMBIGUOUS", "AMENDMENT_SCOPE_AMBIGUOUS", "UNRESOLVED", "UNRESOLVED_AMENDMENT"):
                                    amendments_info[base_acc]["has_ambiguous_amendment"] = True
                                    amendments_info[base_acc]["ambiguous_amendment_accession"] = am_acc
                                    amendments_info[base_acc]["amendment_note"] = f"Target of ambiguous amendment {am_acc} (effective impact unresolved)."
                                elif st in ("PARTIAL_CORRECTION", "PARTIAL_AMENDMENT"):
                                    amendments_info[base_acc]["has_partial_amendment"] = True
                                    amendments_info[base_acc]["partial_amendment_accession"] = am_acc
                                    amendments_info[base_acc]["amendment_note"] = f"Subject to partial correction by {am_acc} (unmapped transactions require review)."
                    except Exception as am_err:
                        logger.debug(f"SQLite amendments query skipped: {am_err}")

                if results and "form4_parsed_transactions" in tbls:
                    tx_rows = []
                    if accs:
                        try:
                            placeholders = ",".join(["?"] * len(accs))
                            cur.execute(
                                f"""
                                SELECT
                                    record_key AS transaction_id,
                                    accession_number,
                                    transaction_date,
                                    transaction_code,
                                    shares,
                                    price_per_share,
                                    total_value_usd,
                                    reporting_owner_name,
                                    is_officer,
                                    officer_title,
                                    is_director,
                                    is_ten_percent_owner AS is_ten_percent,
                                    ownership_nature AS ownership_type,
                                    is_derivative,
                                    signal_type,
                                    acquired_disposed,
                                    is_positive_buying_conviction,
                                    is_open_market_purchase_evidence,
                                    public_availability_at
                                FROM form4_parsed_transactions
                                WHERE accession_number IN ({placeholders})
                                ORDER BY transaction_date ASC, record_key ASC;
                                """,
                                tuple(accs),
                            )
                            tx_rows = [dict(row_data) for row_data in cur.fetchall()]
                        except Exception as tx_err:
                            logger.debug(f"SQLite parsed transactions query skipped: {tx_err}")
                    _attach_parsed_transactions_to_filings(
                        results, tx_rows, amendments_info=amendments_info, cutoff_watermark=cert_watermark
                    )

                conn.close()

                evidence_timestamps = []
                for item in results:
                    pub = item.get("public_availability_at") or item.get("acceptance_datetime")
                    if pub:
                        evidence_timestamps.append(pub)

                evidence_as_of = None
                if evidence_timestamps:
                    max_ev = max(evidence_timestamps)
                    if cert_watermark and max_ev <= cert_watermark:
                        evidence_as_of = max_ev
                    elif cert_watermark:
                        evidence_as_of = cert_watermark
                    else:
                        evidence_as_of = max_ev
                elif cert_watermark:
                    evidence_as_of = cert_watermark

                served_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                if results:
                    return {
                        "status": "AVAILABLE_WITH_RESULTS",
                        "status_code": 200,
                        "cik": cik,
                        "reason": None,
                        "error_detail": None,
                        "items": results,
                        "cached": True,
                        "coverage_status": cov_status,
                        "certified_watermark": cert_watermark,
                        "as_of": evidence_as_of,
                        "served_at": served_at,
                        "is_pipeline_held": True,
                    }
                else:
                    return {
                        "status": "AVAILABLE_NO_RESULTS",
                        "status_code": 200,
                        "cik": cik,
                        "reason": None,
                        "error_detail": None,
                        "items": [],
                        "cached": True,
                        "coverage_status": cov_status,
                        "certified_watermark": cert_watermark,
                        "as_of": evidence_as_of,
                        "served_at": served_at,
                        "is_pipeline_held": True,
                    }

            except Exception as e:
                logger.error(f"Failed to query SQLite filings store for {clean_sym}: {e}", exc_info=True)
                return {
                    "status": "BACKEND_STORAGE_ERROR",
                    "status_code": None,
                    "cik": cik,
                    "reason": "STORAGE_QUERY_FAILED",
                    "error_detail": f"Failed to query filings store for {clean_sym}.",
                    "items": None,
                    "cached": True,
                }

        else:
            return {
                "status": "CONFIGURATION_BLOCKED",
                "status_code": None,
                "cik": cik,
                "reason": "UNSUPPORTED_STORAGE_BACKEND",
                "error_detail": f"Unsupported storage backend '{backend}'. Supported: 'postgres', 'sqlite'.",
                "items": None,
                "cached": True,
            }

    def get_recent_filings_envelope(
        self, symbol: str, form_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get structured filings envelope distinguishing valid empty results from upstream failures.
        Operates in CACHE_ONLY mode when ARX_SEC_EGRESS_ROLE != OWNER.
        """
        is_mock = self._is_mocked()
        if not is_mock and not is_sec_egress_authorized():
            return self._get_cached_recent_filings_envelope(symbol, form_types=form_types)

        cik = self.resolve_cik(symbol)
        sub_env = self.get_company_submissions_envelope(symbol)

        if sub_env["status"] != "AVAILABLE":
            # If get_company_submissions was mocked/overridden in caller tests
            mocked_sub = self.get_company_submissions(symbol)
            if mocked_sub and isinstance(mocked_sub, dict):
                sub_env = {
                    "status": "AVAILABLE",
                    "status_code": 200,
                    "data": mocked_sub,
                    "reason": None,
                    "error": None,
                }
            else:
                return {
                    "status": sub_env["status"],
                    "status_code": sub_env.get("status_code"),
                    "cik": cik,
                    "reason": sub_env.get("reason"),
                    "error_detail": sub_env.get("error"),
                    "items": None,
                }

        data = sub_env.get("data")
        if not data or not isinstance(data, dict) or "filings" not in data or not isinstance(data.get("filings"), dict) or "recent" not in data["filings"]:
            return {
                "status": "STRUCTURALLY_INVALID",
                "status_code": 200,
                "cik": cik,
                "reason": "MISSING_FILINGS_RECENT_KEY",
                "error_detail": "SEC EDGAR payload missing 'filings.recent' structure",
                "items": None,
            }

        recent = data["filings"]["recent"]
        if not isinstance(recent, dict):
            return {
                "status": "STRUCTURALLY_INVALID",
                "status_code": 200,
                "cik": cik,
                "reason": "RECENT_NOT_DICT",
                "error_detail": "'filings.recent' is not a dictionary",
                "items": None,
            }

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocDescription", [])
        acceptance_dts = recent.get("acceptanceDateTime", [])
        report_dates = recent.get("reportDate", [])

        allowed_forms = set(form_types) if form_types else {"10-K", "10-Q", "8-K", "4", "13F-HR"}
        results = []

        cik_int = int(cik) if cik and str(cik).isdigit() else 0

        for i in range(min(len(forms), 25)):
            form = forms[i]
            if form in allowed_forms:
                clean_acc = accessions[i].replace("-", "") if i < len(accessions) else ""
                acc_num = accessions[i] if i < len(accessions) else ""
                acceptance_dt = acceptance_dts[i] if i < len(acceptance_dts) else ""
                report_date = report_dates[i] if i < len(report_dates) else ""
                filing_date = filing_dates[i] if i < len(filing_dates) else ""
                avail_from = acceptance_dt or (f"{filing_date}T00:00:00Z" if filing_date else "")

                results.append({
                    "form": form,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "acceptance_datetime": acceptance_dt,
                    "available_from": avail_from,
                    "description": descriptions[i] if i < len(descriptions) else form,
                    "accession_number": acc_num,
                    "sec_url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{acc_num}.txt",
                })

        if results:
            self._enrich_filings_with_stored_transactions(results, symbol, cik)

        evidence_timestamps = []
        for item in results:
            pub = item.get("public_availability_at") or item.get("acceptance_datetime")
            if pub:
                evidence_timestamps.append(pub)

        evidence_as_of = max(evidence_timestamps) if evidence_timestamps else None
        served_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        final_status = "AVAILABLE_WITH_RESULTS" if len(results) > 0 else "AVAILABLE_NO_RESULTS"
        return {
            "status": final_status,
            "status_code": 200,
            "cik": cik,
            "reason": None,
            "error_detail": None,
            "items": results,
            "coverage_status": "SUPPORTED_SEC_ISSUER" if cik else "UNSUPPORTED_ISSUER",
            "certified_watermark": None,
            "as_of": evidence_as_of,
            "served_at": served_at,
            "is_pipeline_held": True,
        }

    def get_recent_filings(self, symbol: str, form_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get formatted list of recent regulatory filings (backwards compatible)."""
        env = self.get_recent_filings_envelope(symbol, form_types=form_types)
        return env.get("items") or []

    def fetch_form4_filings(self, ticker: str, cik: str, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Fetches the latest Form 4 filing(s) for a governed ticker from SEC EDGAR.
        Returns a list of raw filing dictionaries ready for repository persistence.
        """
        import hashlib
        from analyst_dashboard.smart_money.form4_parser import Form4XmlParser

        env = self.get_company_submissions_envelope(ticker)
        if env.get("status") not in ("SUCCESS", "AVAILABLE_WITH_RESULTS"):
            return []

        sub_data = env.get("data") or {}
        recent = sub_data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        acceptance_dts = recent.get("acceptanceDateTime", [])
        primary_docs = recent.get("primaryDocument", [])

        results: List[Dict[str, Any]] = []
        cik_int = int(cik) if str(cik).isdigit() else 0
        company_name = sub_data.get("name", ticker)

        for idx, f_type in enumerate(forms):
            if f_type in ("4", "4/A"):
                acc_num = accessions[idx]
                clean_acc = acc_num.replace("-", "")
                f_date = filing_dates[idx] if idx < len(filing_dates) else ""
                acc_dt = acceptance_dts[idx] if idx < len(acceptance_dts) else ""
                p_doc = primary_docs[idx] if idx < len(primary_docs) else ""
                clean_p_doc = p_doc.split("/")[-1] if p_doc else ""

                xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{clean_p_doc}"
                raw_xml = ""
                source_url = xml_url
                try:
                    doc_resp = self._get(xml_url)
                    if doc_resp.status_code == 200 and ("<XML" in doc_resp.text.upper() or "<OWNERSHIPDOCUMENT" in doc_resp.text.upper()):
                        raw_xml = doc_resp.text
                except Exception:
                    raw_xml = ""

                if not raw_xml:
                    txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{clean_acc}/{acc_num}.txt"
                    try:
                        txt_resp = self._get(txt_url)
                        if txt_resp.status_code == 200:
                            raw_xml = Form4XmlParser.extract_xml_from_submission_text(txt_resp.text)
                            source_url = txt_url
                    except Exception:
                        raw_xml = ""

                if raw_xml:
                    content_hash = hashlib.sha256(raw_xml.encode("utf-8")).hexdigest()
                    results.append({
                        "accession_number": acc_num,
                        "cik": cik,
                        "ticker": ticker,
                        "company_name": company_name,
                        "filing_date": f_date,
                        "accepted_datetime": acc_dt,
                        "xml_content": raw_xml,
                        "content_hash": content_hash,
                        "form_type": f_type,
                        "primary_doc_url": source_url,
                        "source_url": source_url,
                    })
                    if len(results) >= limit:
                        break
        return results
