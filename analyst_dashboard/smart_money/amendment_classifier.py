"""
ARX Terminal — Smart Money Amendment-Effect Classifier.

Inspects source XML, remarks, footnotes, and base filing transactions to classify
Form 4/A amendment effect into explicit governance categories:
- FULL_REPLACEMENT: The amendment supersedes all transactions of the base filing in their entirety.
- PARTIAL_CORRECTION: The amendment corrects specific line items; non-amended base items remain active.
- ADDITIONAL_TRANSACTION: The amendment reports omitted transactions; both base and amendment remain active.
- AMENDMENT_SCOPE_AMBIGUOUS: Evidence is inconclusive; blocks snapshot publication fail-closed.
- UNRESOLVED: Base filing lineage is missing or unresolvable.

RESOLVED lineage and transaction-subset overlap alone must never establish replacement scope.
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

# Exclusion patterns: agreements, plans, or external instruments that must NOT be misclassified as filing replacement
EXCLUSION_INSTRUMENT_PATTERN = re.compile(
    r"\b(?:voting|credit|loan|employment|separation|operating|shareholder|stockholder|security|indemnity|settlement|merger|purchase|contribution)\s+agreement\b"
    r"|\b(?:equity\s+incentive|stock\s+option|benefit|retirement|compensation|omnibus)\s+plan\b"
    r"|\bindenture\b"
    r"|\btrust\s+agreement\b",
    re.IGNORECASE,
)

# Negation and exception guards (e.g. "remains unchanged except for a footnote")
NEGATION_EXCEPTION_PATTERN = re.compile(
    r"\bremains?\s+unchanged\s+except\b"
    r"|\bunchanged\s+other\s+than\b"
    r"|\bno\s+other\s+changes?\b"
    r"|\bexcept\s+(?:for\s+)?(?:a|the)?\s*footnote\b",
    re.IGNORECASE,
)

# Rigorous, contextual full replacement patterns requiring explicit filing / Form 4 replacement in entirety
FULL_REPLACEMENT_PATTERNS = [
    re.compile(
        r"\b(?:this\s+amendment|this\s+form\s+4/?a?|this\s+filing)\s+(?:amends\s+and\s+restates|supersedes|replaces)\s+(?:the\s+original|the\s+prior|the\s+initial)?\s*(?:form\s+4|filing|report)?\s*in\s+its\s+entirety\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:amends\s+and\s+restates|supersedes|replaces|amended|restated|superseded|replaced)\s+(?:the\s+original|the\s+prior|the\s+initial)?\s*(?:form\s+4|filing|report)?\s*in\s+its\s+entirety\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:this\s+form\s+4/?a?|this\s+filing|this\s+amendment)\s+(?:replaces|supersedes)\s+the\s+(?:original|prior|initial)\s+(?:form\s+4|filing|report)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:supersedes|replaces)\s+and\s+(?:supersedes|replaces)\s+the\s+(?:original|prior|initial)\s+(?:form\s+4|filing|report)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:entire\s+filing|entire\s+form\s+4|entire\s+report)\s+(?:is\s+)?(?:amended\s+and\s+restated|superseded|replaced)\s+in\s+its\s+entirety\b",
        re.IGNORECASE,
    ),
]

# Rigorous partial correction patterns
PARTIAL_CORRECTION_PATTERNS = [
    re.compile(r"\b(?:solely|filed|amended)\s+to\s+correct\b", re.IGNORECASE),
    re.compile(r"\bcorrection\s+(?:of|to)\b", re.IGNORECASE),
    re.compile(r"\bcorrecting\s+the\b", re.IGNORECASE),
    re.compile(r"\bto\s+correct\s+an?\s+(?:error|clerical\s+error|administrative\s+error)\b", re.IGNORECASE),
    re.compile(
        r"\bcorrect\s+(?:the\s+)?(?:transaction\s+date|price|number\s+of\s+shares|shares|transaction\s+code|footnote|reporting\s+owner)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bamends?\s+table\s+[i|ii]+\s+solely\b", re.IGNORECASE),
    re.compile(r"\bcorrection\s+to\s+table\s+[i|ii]+\b", re.IGNORECASE),
    re.compile(r"\bremains?\s+unchanged\s+except\s+(?:for\s+)?(?:a|the)?\s*footnote\b", re.IGNORECASE),
    re.compile(r"\bsolely\s+to\s+update\s+footnote\b", re.IGNORECASE),
]

# Additional transaction patterns
ADDITIONAL_TRANSACTION_PATTERNS = [
    re.compile(r"\binadvertently\s+(?:omitted|excluded)\b", re.IGNORECASE),
    re.compile(r"\b(?:report|reporting|add|adding|include)\s+additional\s+(?:reportable\s+)?transactions?\b", re.IGNORECASE),
    re.compile(r"\bomitted\s+from\s+the\s+original\s+filing\b", re.IGNORECASE),
    re.compile(r"\btransactions?\s+(?:was|were)\s+inadvertently\s+omitted\b", re.IGNORECASE),
]


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
    Classifies the effect scope of a Form 4/A filing with conservative, evidence-backed rules.

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

    narrative_chunks = [r for r in [remarks] + footnotes if r and r.strip()]
    full_narrative = " ".join(narrative_chunks).strip()

    if not full_narrative:
        return (
            SCOPE_AMBIGUOUS,
            "Ambiguous amendment scope: Filing contains no remarks or footnotes to evidence scope.",
        )

    # 2. Match signals across narrative
    has_negation = bool(NEGATION_EXCEPTION_PATTERN.search(full_narrative))
    has_instrument_exclusion = bool(EXCLUSION_INSTRUMENT_PATTERN.search(full_narrative))

    matched_full_pats = [p.pattern for p in FULL_REPLACEMENT_PATTERNS if p.search(full_narrative)]
    matched_part_pats = [p.pattern for p in PARTIAL_CORRECTION_PATTERNS if p.search(full_narrative)]
    matched_add_pats = [p.pattern for p in ADDITIONAL_TRANSACTION_PATTERNS if p.search(full_narrative)]

    # If full replacement pattern matched, but narrative also contains negation or refers to an external agreement/plan,
    # reject full replacement.
    if matched_full_pats and (has_negation or has_instrument_exclusion):
        matched_full_pats = []

    has_full = len(matched_full_pats) > 0
    has_part = len(matched_part_pats) > 0
    has_add = len(matched_add_pats) > 0

    # Contradictory phrases or ambiguous negation fail closed
    if sum([has_full, has_part, has_add]) > 1:
        return (
            SCOPE_AMBIGUOUS,
            "Contradictory scope phrases detected in amendment remarks/footnotes.",
        )

    if has_full:
        return (
            SCOPE_FULL_REPLACEMENT,
            "Explicit full replacement phrase verified in source narrative: 'in its entirety'.",
        )

    if has_part:
        # Partial corrections require evidence-backed transaction mapping.
        # If base_transactions are available and base has multiple transactions:
        amend_txs = getattr(amendment_filing, "transactions", [])
        if isinstance(amendment_filing, dict) and "transactions" in amendment_filing:
            amend_txs = amendment_filing["transactions"]

        if base_transactions is not None and len(base_transactions) > 1 and len(amend_txs) > 0:
            # Check if mapping can be established (e.g. sequence numbers match or explicit mapping)
            base_owners = {str(b.get("reporting_owner_name") or "").strip().upper() for b in base_transactions}
            if len(amend_txs) > 1 and len(amend_txs) != len(base_transactions):
                # Multiple transactions with unequal counts and no line-item cross-reference
                return (
                    SCOPE_AMBIGUOUS,
                    "Partial correction lacks unambiguous transaction mapping across multiple base transactions.",
                )

        return (
            SCOPE_PARTIAL_CORRECTION,
            "Explicit partial correction phrase verified in source narrative: 'solely to correct'.",
        )

    if has_add:
        return (
            SCOPE_ADDITIONAL_TRANSACTION,
            "Explicit additional transaction phrase verified in source narrative: 'inadvertently omitted'.",
        )

    # 3. Fail closed: RESOLVED lineage or subset overlap alone does NOT establish scope
    return (
        SCOPE_AMBIGUOUS,
        "Ambiguous amendment scope: Remarks and footnotes lack explicit scope evidence, "
        "and transaction comparison does not establish replacement scope.",
    )
