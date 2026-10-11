"""
Form 4 Decision-Input Contract Validator — Epoch 001.
Evaluates Form 4 XML content against SMART_MONEY_OPEN_MARKET_PURCHASE_V1 contract.
Deterministic reason codes, strict P/A semantics, and precise footnote scoping.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Any, Tuple, Optional, List, Set

from analyst_dashboard.smart_money.public_availability.models import (
    CANONICAL_DECISION_INPUT_CONTRACT_ID,
    CANONICAL_DECISION_INPUT_CONTRACT_SHA256,
    DecisionInputCompleteness,
    DecisionInputReasonCode,
)


# Required fields for decision input
REQUIRED_METADATA_FIELDS = ("schemaVersion", "documentType", "periodOfReport", "issuerCik")
REQUIRED_OWNER_FIELDS = ("rptOwnerCik", "reportingOwnerRelationship")
REQUIRED_TX_FIELDS = ("transactionCode", "transactionShares", "transactionPricePerShare", "transactionAcquiredDisposedCode")


def parse_and_validate_form4_decision_inputs(
    xml_content: str,
) -> Tuple[DecisionInputCompleteness, Optional[DecisionInputReasonCode], Dict[str, Any]]:
    """
    Parses raw Form 4 XML string and evaluates compliance with the frozen decision input contract.
    Returns: (completeness, reason_code, details_dict)
    """
    if not xml_content or not xml_content.strip():
        return (
            DecisionInputCompleteness.INCOMPLETE,
            DecisionInputReasonCode.REQUIRED_DOCUMENT_MISSING,
            {"error": "Empty XML content"},
        )

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        return (
            DecisionInputCompleteness.INCOMPLETE,
            DecisionInputReasonCode.REQUIRED_DOCUMENT_MISSING,
            {"error": f"Malformed XML: {str(e)}"},
        )

    # 1. Document / Metadata checks
    doc_type_el = root.find(".//documentType")
    doc_type = doc_type_el.text.strip() if doc_type_el is not None and doc_type_el.text else ""
    if doc_type not in ("4", "4/A"):
        return (
            DecisionInputCompleteness.INCOMPLETE,
            DecisionInputReasonCode.REQUIRED_DOCUMENT_MISSING,
            {"error": f"Invalid documentType: {doc_type}"},
        )

    issuer_cik_el = root.find(".//issuer/issuerCik")
    issuer_cik = issuer_cik_el.text.strip() if issuer_cik_el is not None and issuer_cik_el.text else ""
    if not issuer_cik:
        return (
            DecisionInputCompleteness.INCOMPLETE,
            DecisionInputReasonCode.REQUIRED_DOCUMENT_MISSING,
            {"error": "Missing issuerCik"},
        )

    # 2. Reporting owner checks
    owner_cik_el = root.find(".//reportingOwner/reportingOwnerId/rptOwnerCik")
    owner_rel_el = root.find(".//reportingOwner/reportingOwnerRelationship")

    if owner_cik_el is None or not owner_cik_el.text:
        return (
            DecisionInputCompleteness.INCOMPLETE,
            DecisionInputReasonCode.REQUIRED_OWNER_RELATIONSHIP_MISSING,
            {"error": "Missing rptOwnerCik"},
        )

    if owner_rel_el is None:
        return (
            DecisionInputCompleteness.INCOMPLETE,
            DecisionInputReasonCode.REQUIRED_OWNER_RELATIONSHIP_MISSING,
            {"error": "Missing reportingOwnerRelationship"},
        )

    # Has at least one relationship marker
    is_officer = owner_rel_el.find("isOfficer") is not None and owner_rel_el.find("isOfficer").text in ("1", "true", "True")
    is_director = owner_rel_el.find("isDirector") is not None and owner_rel_el.find("isDirector").text in ("1", "true", "True")
    is_ten_pct = owner_rel_el.find("isTenPercentOwner") is not None and owner_rel_el.find("isTenPercentOwner").text in ("1", "true", "True")
    is_other = owner_rel_el.find("isOther") is not None and owner_rel_el.find("isOther").text in ("1", "true", "True")

    if not (is_officer or is_director or is_ten_pct or is_other):
        return (
            DecisionInputCompleteness.INCOMPLETE,
            DecisionInputReasonCode.REQUIRED_OWNER_RELATIONSHIP_MISSING,
            {"error": "No valid relationship flags found"},
        )

    # Collect available footnotes
    available_footnotes: Dict[str, str] = {}
    for fn in root.findall(".//footnotes/footnote"):
        fn_id = fn.attrib.get("id")
        if fn_id:
            available_footnotes[fn_id] = fn.text.strip() if fn.text else ""

    # 3. Transaction checks
    non_deriv_txs = root.findall(".//nonDerivativeTransaction")
    if not non_deriv_txs:
        # No non-derivative transactions
        return (
            DecisionInputCompleteness.INCOMPLETE,
            DecisionInputReasonCode.REQUIRED_TRANSACTION_CODE_MISSING,
            {"error": "No nonDerivativeTransaction elements found"},
        )

    qualifying_p_buys = 0
    tx_eval_details: List[Dict[str, Any]] = []

    for tx_idx, tx in enumerate(non_deriv_txs):
        code_el = tx.find(".//transactionCoding/transactionCode")
        code = code_el.text.strip() if code_el is not None and code_el.text else ""

        ad_el = tx.find(".//transactionAmounts/transactionAcquiredDisposedCode/value")
        ad_code = ad_el.text.strip() if ad_el is not None and ad_el.text else ""

        shares_el = tx.find(".//transactionAmounts/transactionShares/value")
        price_el = tx.find(".//transactionAmounts/transactionPricePerShare/value")

        # Check footnotes referenced by required decision-input fields
        required_field_fn_ids: Set[str] = set()
        for field_path in [
            ".//transactionCoding/transactionCode",
            ".//transactionAmounts/transactionShares",
            ".//transactionAmounts/transactionPricePerShare",
            ".//transactionAmounts/transactionAcquiredDisposedCode",
        ]:
            target_el = tx.find(field_path)
            if target_el is not None:
                for fn_ref in target_el.findall(".//footnoteId"):
                    fid = fn_ref.attrib.get("id")
                    if fid:
                        required_field_fn_ids.add(fid)

        # Are referenced footnotes resolved?
        unresolved_required_fns = [fid for fid in required_field_fn_ids if fid not in available_footnotes]

        detail = {
            "tx_idx": tx_idx,
            "transactionCode": code,
            "transactionAcquiredDisposedCode": ad_code,
            "shares": shares_el.text if shares_el is not None else None,
            "price": price_el.text if price_el is not None else None,
            "unresolved_footnotes": unresolved_required_fns,
        }
        tx_eval_details.append(detail)

        # Semantics check: Code P required for open market purchase
        if code == "P" and ad_code == "A":
            # Check shares
            if shares_el is None or not shares_el.text or not shares_el.text.strip():
                return (
                    DecisionInputCompleteness.INCOMPLETE,
                    DecisionInputReasonCode.REQUIRED_SHARES_MISSING,
                    {"tx_idx": tx_idx, "details": detail},
                )
            try:
                shares_val = float(shares_el.text.strip())
                if shares_val <= 0:
                    return (
                        DecisionInputCompleteness.INCOMPLETE,
                        DecisionInputReasonCode.REQUIRED_SHARES_MISSING,
                        {"tx_idx": tx_idx, "shares_val": shares_val},
                    )
            except ValueError:
                return (
                    DecisionInputCompleteness.INCOMPLETE,
                    DecisionInputReasonCode.REQUIRED_SHARES_MISSING,
                    {"tx_idx": tx_idx, "error": "Invalid shares format"},
                )

            # Check price
            if price_el is None or not price_el.text or not price_el.text.strip():
                return (
                    DecisionInputCompleteness.INCOMPLETE,
                    DecisionInputReasonCode.REQUIRED_PRICE_MISSING,
                    {"tx_idx": tx_idx, "details": detail},
                )
            try:
                price_val = float(price_el.text.strip())
                if price_val < 0:
                    return (
                        DecisionInputCompleteness.INCOMPLETE,
                        DecisionInputReasonCode.REQUIRED_PRICE_MISSING,
                        {"tx_idx": tx_idx, "price_val": price_val},
                    )
            except ValueError:
                return (
                    DecisionInputCompleteness.INCOMPLETE,
                    DecisionInputReasonCode.REQUIRED_PRICE_MISSING,
                    {"tx_idx": tx_idx, "error": "Invalid price format"},
                )

            # Check footnotes on required fields
            if unresolved_required_fns:
                return (
                    DecisionInputCompleteness.INCOMPLETE,
                    DecisionInputReasonCode.REQUIRED_FOOTNOTE_UNRESOLVED,
                    {"tx_idx": tx_idx, "unresolved_footnotes": unresolved_required_fns},
                )

            qualifying_p_buys += 1

    if qualifying_p_buys == 0:
        # Distinguish between code A (grants/awards) or other non-P codes vs missing
        has_grant_a = any(d["transactionCode"] == "A" for d in tx_eval_details)
        return (
            DecisionInputCompleteness.INCOMPLETE,
            DecisionInputReasonCode.REQUIRED_TRANSACTION_CODE_MISSING,
            {
                "error": "No open-market purchase transaction (Code P + Acquired A) found",
                "contains_code_A_grant": has_grant_a,
                "evaluated_txs": tx_eval_details,
            },
        )

    return (
        DecisionInputCompleteness.COMPLETE,
        DecisionInputReasonCode.SATISFIED,
        {
            "qualifying_purchase_count": qualifying_p_buys,
            "issuer_cik": issuer_cik,
            "contract_id": CANONICAL_DECISION_INPUT_CONTRACT_ID,
            "contract_sha256": CANONICAL_DECISION_INPUT_CONTRACT_SHA256,
        },
    )
