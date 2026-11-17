"""
ARX Terminal — Smart Money Amendment-Effect Classifier.

Inspects source XML, remarks, footnotes, and base filing transactions to classify
Form 4/A amendment effect into explicit governance categories:
- FULL_REPLACEMENT: The amendment supersedes all transactions of the base filing in their entirety.
- PARTIAL_CORRECTION: The amendment corrects specific line items; non-amended base items remain active.
- ADDITIONAL_TRANSACTION: The amendment reports omitted transactions; both base and amendment remain active.
- AMENDMENT_SCOPE_AMBIGUOUS: Evidence is inconclusive; blocks snapshot publication fail-closed.
- UNRESOLVED: Base filing lineage is missing or unresolvable.

RESOLVED lineage alone must never establish replacement scope.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple, Set

# Governed amendment effect statuses
SCOPE_FULL_REPLACEMENT = "FULL_REPLACEMENT"
SCOPE_PARTIAL_CORRECTION = "PARTIAL_CORRECTION"
SCOPE_ADDITIONAL_TRANSACTION = "ADDITIONAL_TRANSACTION"
SCOPE_AMBIGUOUS = "AMENDMENT_SCOPE_AMBIGUOUS"
SCOPE_UNRESOLVED = "UNRESOLVED"

# Explicit phrase dictionaries for evidence-based classification
FULL_REPLACEMENT_PHRASES = (
    "in its entirety",
    "replaces the original",
    "replaces the prior",
    "replaces the form 4",
    "replaces the initial",
    "amends and restates",
    "amended and restated",
    "supersedes in its entirety",
    "amended in its entirety",
    "restated in its entirety",
    "entire filing",
    "superseded in its entirety",
    "replace in its entirety",
    "amends in its entirety",
    "supersedes the original",
    "supersedes the prior",
    "replaces and supersedes",
)

PARTIAL_CORRECTION_PHRASES = (
    "solely to correct",
    "filed to correct",
    "amended to correct",
    "correcting the",
    "correction of",
    "correction to",
    "correct the",
    "to correct an error",
    "to correct a clerical error",
    "to correct an administrative error",
    "correct the transaction date",
    "correct the price",
    "correct the number of shares",
    "correct the shares",
    "correct the transaction code",
    "correct footnote",
    "amended solely",
    "amends table i solely",
    "amends table ii solely",
    "correction to table i",
    "correction to table ii",
)

ADDITIONAL_TRANSACTION_PHRASES = (
    "inadvertently omitted",
    "inadvertently excluded",
    "report additional",
    "additional transaction",
    "additional reportable",
    "omitted from the original",
    "omitted from original",
    "adding transaction",
    "include transaction",
    "report transactions inadvertently",
)


def extract_amendment_narrative(
    raw_xml: Optional[str] = None,
    filing_obj: Optional[Any] = None,
) -> Tuple[Optional[str], List[str]]:
    """
    Extracts remarks and footnotes from XML or filing object.
    Returns (remarks_text, list_of_footnotes).
    """
    remarks: Optional[str] = None
    footnotes: List[str] = []

    if filing_obj is not None:
        if getattr(filing_obj, "remarks", None):
            remarks = str(filing_obj.remarks).strip()
        if getattr(filing_obj, "all_footnotes", None):
            footnotes.extend([str(f).strip() for f in filing_obj.all_footnotes if str(f).strip()])

    if raw_xml and ("<ownershipDocument" in raw_xml or "<XML>" in raw_xml):
        try:
            # Strip wrapper if needed
            xml_str = raw_xml
            if "<ownershipDocument" in xml_str and not xml_str.strip().startswith("<ownershipDocument"):
                m = re.search(r"(<ownershipDocument.*?</ownershipDocument>)", xml_str, re.DOTALL)
                if m:
                    xml_str = m.group(1)

            root = ET.fromstring(xml_str)

            # Remarks
            if not remarks:
                rem_elem = root.find(".//remarks")
                if rem_elem is not None and rem_elem.text:
                    remarks = rem_elem.text.strip()

            # Footnotes
            if not footnotes:
                for fn in root.findall(".//footnotes/footnote"):
                    if fn.text and fn.text.strip():
                        footnotes.append(fn.text.strip())
        except Exception:
            pass

    return remarks, footnotes


def classify_amendment_effect(
    amendment_filing: Any,
    base_filing: Optional[Dict[str, Any]] = None,
    base_transactions: Optional[List[Dict[str, Any]]] = None,
    raw_xml: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Classifies the effect scope of a Form 4/A filing.

    Returns:
        (status, reason):
            status in (FULL_REPLACEMENT, PARTIAL_CORRECTION, ADDITIONAL_TRANSACTION, AMENDMENT_SCOPE_AMBIGUOUS, UNRESOLVED)
    """
    doc_type = (
        getattr(amendment_filing, "document_type", None)
        or (amendment_filing.get("document_type") if isinstance(amendment_filing, dict) else "")
        or ""
    ).upper().strip()

    is_amend = (
        getattr(amendment_filing, "is_amendment", False)
        or doc_type in ("4/A", "4A")
        or (isinstance(amendment_filing, dict) and amendment_filing.get("is_amendment"))
    )

    if not is_amend:
        return "NOT_AN_AMENDMENT", "Document type is not Form 4/A."

    if not base_filing:
        return SCOPE_UNRESOLVED, "Base filing lineage is unresolved or missing."

    # 1. Extract remarks and footnotes
    xml_content = raw_xml or (
        getattr(amendment_filing, "raw_xml", None)
        or (amendment_filing.get("xml_content") if isinstance(amendment_filing, dict) else None)
    )
    remarks, footnotes = extract_amendment_narrative(raw_xml=xml_content, filing_obj=amendment_filing)

    narrative_corpus = " ".join([remarks or ""] + footnotes).lower()

    # 2. Check phrase matches in narrative
    has_full = any(p in narrative_corpus for p in FULL_REPLACEMENT_PHRASES)
    has_part = any(p in narrative_corpus for p in PARTIAL_CORRECTION_PHRASES)
    has_add = any(p in narrative_corpus for p in ADDITIONAL_TRANSACTION_PHRASES)

    # Contradictory phrases fail closed as ambiguous
    if sum([has_full, has_part, has_add]) > 1:
        return (
            SCOPE_AMBIGUOUS,
            "Contradictory scope phrases detected in amendment remarks/footnotes.",
        )

    if has_full:
        matched = [p for p in FULL_REPLACEMENT_PHRASES if p in narrative_corpus][0]
        return (
            SCOPE_FULL_REPLACEMENT,
            f"Explicit full replacement phrase verified in source narrative: '{matched}'.",
        )

    if has_part:
        matched = [p for p in PARTIAL_CORRECTION_PHRASES if p in narrative_corpus][0]
        return (
            SCOPE_PARTIAL_CORRECTION,
            f"Explicit partial correction phrase verified in source narrative: '{matched}'.",
        )

    if has_add:
        matched = [p for p in ADDITIONAL_TRANSACTION_PHRASES if p in narrative_corpus][0]
        return (
            SCOPE_ADDITIONAL_TRANSACTION,
            f"Explicit additional transaction phrase verified in source narrative: '{matched}'.",
        )

    # 3. Transaction-level comparative inspection (if base transactions available)
    amend_txs = getattr(amendment_filing, "transactions", [])
    if isinstance(amendment_filing, dict) and "transactions" in amendment_filing:
        amend_txs = amendment_filing["transactions"]

    if base_transactions is not None and len(base_transactions) > 0 and len(amend_txs) > 0:
        base_sigs: Set[Tuple[str, str, str]] = set()
        for btx in base_transactions:
            owner = (btx.get("reporting_owner_name") or "").strip().upper()
            t_date = str(btx.get("transaction_date") or "")[:10]
            t_code = (btx.get("transaction_code") or "").strip().upper()
            base_sigs.add((owner, t_date, t_code))

        amend_sigs: Set[Tuple[str, str, str]] = set()
        for atx in amend_txs:
            if hasattr(atx, "reporting_owner_name"):
                owner = (atx.reporting_owner_name or "").strip().upper()
                t_date = str(atx.transaction_date or "")[:10]
                t_code = (atx.transaction_code or "").strip().upper()
            else:
                owner = (atx.get("reporting_owner_name") or "").strip().upper()
                t_date = str(atx.get("transaction_date") or "")[:10]
                t_code = (atx.get("transaction_code") or "").strip().upper()
            amend_sigs.add((owner, t_date, t_code))

        # Check line item overlap
        if amend_sigs and amend_sigs.issubset(base_sigs) and len(amend_sigs) < len(base_sigs):
            return (
                SCOPE_PARTIAL_CORRECTION,
                f"Amendment line items ({len(amend_sigs)}) form a strict subset of base items ({len(base_sigs)}).",
            )

    # 4. Inconclusive: Lineage exists, but replacement scope cannot be established from evidence
    return (
        SCOPE_AMBIGUOUS,
        "Ambiguous amendment scope: Remarks and footnotes lack explicit scope evidence, "
        "and transaction comparison does not establish replacement scope.",
    )
