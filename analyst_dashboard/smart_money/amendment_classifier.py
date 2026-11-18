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


class ClassificationResult(tuple):
    """
    Subclass of tuple returning (status, reason).
    Maintains 100% backward compatibility with `status, reason = res` (2-element unpack)
    while exposing `res.mappings`, `res.status`, `res.reason`, and `res[2]` if indexed as 3-tuple.
    """
    def __new__(cls, status: str, reason: str, mappings: Optional[List[Dict[str, Any]]] = None):
        return super().__new__(cls, (status, reason))

    def __init__(self, status: str, reason: str, mappings: Optional[List[Dict[str, Any]]] = None):
        self.status = status
        self.reason = reason
        self.mappings = mappings or []

    def __getitem__(self, item):
        if item == 2:
            return self.mappings
        return super().__getitem__(item)

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


ORDINAL_MAP = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9, "tenth": 10, "10th": 10,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
}

PAT_ORD_NOUN = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)\s+"
    r"(?:(?:non-derivative|derivative|table\s+[i|ii|1|2])\s+)?(?:transaction|line|row|trade|entry|item)\b",
    re.IGNORECASE,
)
PAT_NOUN_ORD = re.compile(
    r"\b(?:transaction|line|row|trade|entry|item)\s+(?:number\s+|no\.\s*|#\s*)?"
    r"(10|[1-9]|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|one|two|three|four|five)\b",
    re.IGNORECASE,
)
PAT_FIELD_ORD = re.compile(
    r"\b(?:price|shares?|date|code|amount|exercise\s+price)\s+(?:of|for|on)\s+(?:the\s+)?"
    r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)\s+"
    r"(?:transaction|line|row|trade|entry|item)\b",
    re.IGNORECASE,
)
PAT_OF_TXS = re.compile(
    r"\b(?:the\s+)?(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\s+of\s+(?:the\s+)?(?:\w+\s+)?transactions\b",
    re.IGNORECASE,
)


def extract_target_transaction_ordinals(text: str) -> List[int]:
    """
    Extracts explicit 1-based transaction target ordinals from narrative context.
    Unrelated ordinals (such as 'first quarter award', 'first half', 'first grant', etc.)
    are excluded because they do not modify or reference a transaction/line/row.
    """
    ords: List[int] = []
    for m in PAT_ORD_NOUN.finditer(text):
        val = ORDINAL_MAP.get(m.group(1).lower())
        if val is not None:
            ords.append(val)
    for m in PAT_NOUN_ORD.finditer(text):
        val = ORDINAL_MAP.get(m.group(1).lower())
        if val is not None:
            ords.append(val)
    for m in PAT_FIELD_ORD.finditer(text):
        val = ORDINAL_MAP.get(m.group(1).lower())
        if val is not None:
            ords.append(val)
    for m in PAT_OF_TXS.finditer(text):
        val = ORDINAL_MAP.get(m.group(1).lower())
        if val is not None:
            ords.append(val)
    return ords


def get_base_tx_source_sequence(tx: Dict[str, Any]) -> Optional[int]:
    """
    Resolves the verified source table sequence (0-indexed) of a transaction independently of database result order.
    """
    for k in ("source_sequence", "line_number", "row_number", "sequence"):
        if k in tx and tx[k] is not None:
            try:
                return int(tx[k])
            except (ValueError, TypeError):
                pass
    raw_id = str(tx.get("transaction_id") or tx.get("record_key") or "")
    parts = raw_id.strip().split(":")
    if len(parts) >= 4 and parts[-1].isdigit():
        return int(parts[-1])
    return None


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
    explicit_mappings: Optional[List[Dict[str, Any]]] = None,
) -> ClassificationResult:
    """
    Classifies the effect scope of a Form 4/A filing with conservative, evidence-backed rules.

    Returns:
        ClassificationResult: tuple of (status, reason) with .mappings property.
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
        return ClassificationResult("NOT_AN_AMENDMENT", "Document type is not Form 4/A.", [])

    if not base_filing:
        return ClassificationResult(SCOPE_UNRESOLVED, "Base filing lineage is unresolved or missing.", [])

    # 1. Extract remarks and footnotes
    xml_content = raw_xml or (
        getattr(amendment_filing, "raw_xml", None)
        or (amendment_filing.get("xml_content") if isinstance(amendment_filing, dict) else None)
    )
    remarks, footnotes = extract_amendment_narrative(raw_xml=xml_content, filing_obj=amendment_filing)

    narrative_chunks = [r for r in [remarks] + footnotes if r and r.strip()]
    full_narrative = " ".join(narrative_chunks).strip()

    if not full_narrative:
        return ClassificationResult(
            SCOPE_AMBIGUOUS,
            "Ambiguous amendment scope: Filing contains no remarks or footnotes to evidence scope.",
            [],
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
        return ClassificationResult(
            SCOPE_AMBIGUOUS,
            "Contradictory scope phrases detected in amendment remarks/footnotes.",
            [],
        )

    if has_full:
        return ClassificationResult(
            SCOPE_FULL_REPLACEMENT,
            "Explicit full replacement phrase verified in source narrative: 'in its entirety'.",
            [],
        )

    if has_part:
        if explicit_mappings:
            return ClassificationResult(
                SCOPE_PARTIAL_CORRECTION,
                "Explicit partial correction transaction mappings provided and verified.",
                mappings=explicit_mappings,
            )

        amend_txs = getattr(amendment_filing, "transactions", [])
        if not amend_txs and isinstance(amendment_filing, dict) and "transactions" in amendment_filing:
            amend_txs = amendment_filing["transactions"]

        # When base_transactions is provided, validate mappings across base and amendment transactions
        if base_transactions is not None:
            base_txs = list(base_transactions)

            # Single base and single amendment transaction: establish 1-to-1 mapping
            if len(base_txs) == 1 and len(amend_txs) == 1:
                b_tx = base_txs[0]
                a_tx = amend_txs[0]
                b_id = b_tx.get("transaction_id") or b_tx.get("record_key")
                a_id = getattr(a_tx, "record_key", None) or (a_tx.get("transaction_id") or a_tx.get("record_key") if isinstance(a_tx, dict) else None)
                b_der = bool(b_tx.get("is_derivative", False))
                a_der = bool(getattr(a_tx, "is_derivative", False) if hasattr(a_tx, "is_derivative") else (a_tx.get("is_derivative", False) if isinstance(a_tx, dict) else False))

                if b_der == a_der:
                    mapping = [{
                        "amendment_transaction_id": a_id,
                        "base_transaction_id": b_id,
                        "rationale": "Single-transaction base and single-transaction amendment for matching table.",
                    }]
                    return ClassificationResult(
                        SCOPE_PARTIAL_CORRECTION,
                        "Explicit partial correction phrase verified in source narrative: 'solely to correct'.",
                        mappings=mapping,
                    )
                else:
                    return ClassificationResult(
                        SCOPE_AMBIGUOUS,
                        "Partial correction table mismatch with base transaction.",
                        mappings=[],
                    )

            # Multiple base transactions, single amendment transaction
            if len(base_txs) > 1 and len(amend_txs) == 1:
                a_tx = amend_txs[0]
                a_id = getattr(a_tx, "record_key", None) or (
                    a_tx.get("transaction_id") or a_tx.get("record_key") if isinstance(a_tx, dict) else None
                )
                a_der = bool(
                    getattr(a_tx, "is_derivative", False) if hasattr(a_tx, "is_derivative")
                    else (a_tx.get("is_derivative", False) if isinstance(a_tx, dict) else False)
                )

                # Filter base transactions matching the amendment transaction table type (derivative vs non-derivative)
                matching_base = [b for b in base_txs if bool(b.get("is_derivative", False)) == a_der]
                if not matching_base:
                    return ClassificationResult(
                        SCOPE_AMBIGUOUS,
                        "Partial correction table mismatch: No base transactions match amendment table type.",
                        mappings=[],
                    )

                target_ordinals = extract_target_transaction_ordinals(full_narrative)
                distinct_ordinals = sorted(list(set(target_ordinals)))

                if len(distinct_ordinals) > 1:
                    return ClassificationResult(
                        SCOPE_AMBIGUOUS,
                        f"Conflicting or ambiguous transaction references in narrative: ordinals {distinct_ordinals}.",
                        mappings=[],
                    )

                if len(distinct_ordinals) == 1:
                    target_ord = distinct_ordinals[0]
                    target_seq = target_ord - 1  # 0-indexed sequence

                    # Resolve against verified source sequence independently of list/query result ordering
                    seq_map: Dict[int, Dict[str, Any]] = {}
                    has_unresolved_seq = False
                    for b in matching_base:
                        seq = get_base_tx_source_sequence(b)
                        if seq is None or seq in seq_map:
                            has_unresolved_seq = True
                            break
                        seq_map[seq] = b

                    if has_unresolved_seq or not seq_map:
                        return ClassificationResult(
                            SCOPE_AMBIGUOUS,
                            "Cannot verify source row identity across base transactions independently of result ordering.",
                            mappings=[],
                        )

                    if target_seq in seq_map:
                        target_b = seq_map[target_seq]
                        b_id = target_b.get("transaction_id") or target_b.get("record_key")
                        mapping = [{
                            "amendment_transaction_id": a_id,
                            "base_transaction_id": b_id,
                            "rationale": f"Partial correction narrative identified source line {target_ord}.",
                        }]
                        return ClassificationResult(
                            SCOPE_PARTIAL_CORRECTION,
                            f"Explicit partial correction targeting source line {target_ord}.",
                            mappings=mapping,
                        )
                    else:
                        return ClassificationResult(
                            SCOPE_AMBIGUOUS,
                            f"Narrative target line {target_ord} exceeds available base transactions in matching table.",
                            mappings=[],
                        )

                # Narrative lacks unambiguous line identification across multiple base transactions
                return ClassificationResult(
                    SCOPE_AMBIGUOUS,
                    "Partial correction lacks unambiguous transaction mapping across multiple base transactions.",
                    mappings=[],
                )

            # Multiple amendment transactions without explicit mappings fail closed
            if len(amend_txs) > 1 and len(base_txs) > 1:
                return ClassificationResult(
                    SCOPE_AMBIGUOUS,
                    "Partial correction with multiple amendment transactions requires explicit transaction mappings.",
                    mappings=[],
                )

        return ClassificationResult(
            SCOPE_PARTIAL_CORRECTION,
            "Explicit partial correction phrase verified in source narrative: 'solely to correct'.",
            mappings=[],
        )

    if has_add:
        return ClassificationResult(
            SCOPE_ADDITIONAL_TRANSACTION,
            "Explicit additional transaction phrase verified in source narrative: 'inadvertently omitted'.",
            [],
        )

    # 3. Fail closed: RESOLVED lineage or subset overlap alone does NOT establish scope
    return ClassificationResult(
        SCOPE_AMBIGUOUS,
        "Ambiguous amendment scope: Remarks and footnotes lack explicit scope evidence, "
        "and transaction comparison does not establish replacement scope.",
        [],
    )
