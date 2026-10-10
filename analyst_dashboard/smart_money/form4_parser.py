"""
Deterministic XML parser for SEC Form 4 (Statement of Changes in Beneficial Ownership).
Handles <ownershipDocument> XML schema X0508 and extracts normalized Section 16 transactions.
"""

import re
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import xml.etree.ElementTree as ET

from analyst_dashboard.smart_money.form4_models import (
    SmartMoneyTransaction,
    SmartMoneyFiling,
    SmartMoneySignalType,
    OwnershipNature,
)

logger = logging.getLogger(__name__)


class Form4XmlParser:
    """
    Parses official SEC Form 4 XML and extracts normalized, rule-governed transaction objects.
    Enforces strict false-positive controls and fail-closed error handling.
    """

    @classmethod
    def extract_xml_from_submission_text(cls, raw_text: str) -> Optional[str]:
        """
        Extracts <ownershipDocument> XML block from full SEC EDGAR SGML submission text if needed.
        """
        if not raw_text:
            return None

        # Check if text is already bare XML
        if "<ownershipDocument" in raw_text:
            start = raw_text.find("<ownershipDocument")
            end = raw_text.find("</ownershipDocument>")
            if end != -1:
                return raw_text[start : end + len("</ownershipDocument>")]
            return raw_text[start:]

        # Check for <XML>...</XML> container in SGML
        xml_match = re.search(r"<XML>(.*?)</XML>", raw_text, re.DOTALL | re.IGNORECASE)
        if xml_match:
            inner = xml_match.group(1).strip()
            if "<ownershipDocument" in inner:
                start = inner.find("<ownershipDocument")
                end = inner.find("</ownershipDocument>")
                if end != -1:
                    return inner[start : end + len("</ownershipDocument>")]
                return inner[start:]

        return None

    @classmethod
    def _get_text(cls, elem: Optional[ET.Element], path: str, default: str = "") -> str:
        """Helper to get nested element text, resolving optional <value> wrapper."""
        if elem is None:
            return default
        target = elem.find(path)
        if target is None:
            return default
        # If target has a child <value>, return child's text
        val_child = target.find("value")
        if val_child is not None and val_child.text:
            return val_child.text.strip()
        return (target.text or "").strip()

    @classmethod
    def _get_bool(cls, elem: Optional[ET.Element], path: str) -> bool:
        """Helper to parse SEC boolean representations ('1', 'true', etc.)."""
        txt = cls._get_text(elem, path, "0").lower()
        return txt in ["1", "true", "yes"]

    @classmethod
    def _classify_transaction_code(cls, code: str) -> SmartMoneySignalType:
        """Classify SEC Form 4 transaction code into authoritative signal class."""
        code_upper = code.upper().strip()
        if code_upper == "P":
            return SmartMoneySignalType.OPEN_MARKET_PURCHASE
        elif code_upper == "S":
            return SmartMoneySignalType.OPEN_MARKET_SALE
        elif code_upper == "M":
            return SmartMoneySignalType.OPTION_EXERCISE
        elif code_upper == "A":
            return SmartMoneySignalType.EQUITY_GRANT
        elif code_upper == "G":
            return SmartMoneySignalType.GIFT
        elif code_upper == "F":
            return SmartMoneySignalType.TAX_WITHHOLDING
        elif code_upper == "D":
            return SmartMoneySignalType.ISSUER_DISPOSITION
        else:
            return SmartMoneySignalType.OTHER

    @classmethod
    def parse_filing(
        cls,
        raw_content: str,
        accession_number: str,
        filing_date_hint: Optional[str] = None,
        acceptance_datetime: Optional[str] = None,
        source_url: str = "",
        source_fetch_time: Optional[str] = None,
    ) -> SmartMoneyFiling:
        """
        Parses Form 4 XML and returns a SmartMoneyFiling object containing normalized transactions.
        Fail-closed: Returns empty transactions on parse errors.
        """
        source_content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest() if raw_content else ""
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        fetched_at = source_fetch_time or now_iso

        xml_str = cls.extract_xml_from_submission_text(raw_content) or raw_content
        if not xml_str or "<ownershipDocument" not in xml_str:
            return SmartMoneyFiling(
                accession_number=accession_number,
                issuer_cik="",
                issuer_symbol=None,
                filing_date=filing_date_hint or "",
                acceptance_datetime=acceptance_datetime,
                document_type="4",
                is_amendment=False,
                transactions=[],
                raw_xml=xml_str,
                source_content_hash=source_content_hash,
                source_url=source_url,
                fetched_at=fetched_at,
                disposition="UNSUPPORTED_DOCUMENT",
            )

        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            logger.warning(f"XML parse failure for accession {accession_number}: {e}")
            return SmartMoneyFiling(
                accession_number=accession_number,
                issuer_cik="",
                issuer_symbol=None,
                filing_date=filing_date_hint or "",
                acceptance_datetime=acceptance_datetime,
                document_type="4",
                is_amendment=False,
                transactions=[],
                raw_xml=xml_str,
                source_content_hash=source_content_hash,
                source_url=source_url,
                fetched_at=fetched_at,
                disposition="PARSE_FAILED",
            )

        # 1. Document Metadata
        doc_type = cls._get_text(root, "documentType", "4").strip()
        is_amendment = "A" in doc_type or doc_type == "4/A"
        period_of_report = cls._get_text(root, "periodOfReport", filing_date_hint or "")
        filing_date = period_of_report or filing_date_hint or ""

        # 2. Issuer Info
        issuer_elem = root.find("issuer")
        issuer_cik = cls._get_text(issuer_elem, "issuerCik", "").zfill(10)
        issuer_name = cls._get_text(issuer_elem, "issuerName", "")
        issuer_symbol = cls._get_text(issuer_elem, "issuerTradingSymbol", "").upper() or None

        # 3. Reporting Owner(s)
        # Form 4 can have joint filers; primary filer is first
        owners: List[Dict[str, Any]] = []
        for rpt_elem in root.findall("reportingOwner"):
            rpt_id = rpt_elem.find("reportingOwnerId")
            rpt_rel = rpt_elem.find("reportingOwnerRelationship")

            owner_cik = cls._get_text(rpt_id, "rptOwnerCik", "").zfill(10)
            owner_name = cls._get_text(rpt_id, "rptOwnerName", "")

            is_director = cls._get_bool(rpt_rel, "isDirector")
            is_officer = cls._get_bool(rpt_rel, "isOfficer")
            is_ten_pct = cls._get_bool(rpt_rel, "isTenPercentOwner")
            officer_title = cls._get_text(rpt_rel, "officerTitle") or None

            owners.append({
                "cik": owner_cik,
                "name": owner_name,
                "is_director": is_director,
                "is_officer": is_officer,
                "is_ten_pct": is_ten_pct,
                "officer_title": officer_title,
            })

        primary_owner = owners[0] if owners else {
            "cik": "0000000000",
            "name": "UNKNOWN_INSIDER",
            "is_director": False,
            "is_officer": False,
            "is_ten_pct": False,
            "officer_title": None,
        }

        # 4. Footnotes Index
        footnotes: Dict[str, str] = {}
        for fn in root.findall(".//footnotes/footnote"):
            fn_id = fn.attrib.get("id", "")
            if fn_id and fn.text:
                footnotes[fn_id] = fn.text.strip()

        # 5. Non-Derivative Transactions
        transactions: List[SmartMoneyTransaction] = []
        non_deriv_table = root.find("nonDerivativeTable")
        if non_deriv_table is not None:
            seq = 0
            for trans in non_deriv_table.findall("nonDerivativeTransaction"):
                t_date = cls._get_text(trans, "transactionDate", filing_date)
                t_code = cls._get_text(trans, "transactionCoding/transactionCode", "")
                amounts = trans.find("transactionAmounts")

                shares_raw = cls._get_text(amounts, "transactionShares", "0")
                price_raw = cls._get_text(amounts, "transactionPricePerShare", "0")
                acq_disp = cls._get_text(amounts, "transactionAcquiredDisposedCode", "").upper()

                try:
                    shares = float(shares_raw.replace(",", ""))
                except ValueError:
                    shares = 0.0

                try:
                    price = float(price_raw.replace("$", "").replace(",", ""))
                except ValueError:
                    price = 0.0

                own_elem = trans.find("ownershipNature")
                own_code = cls._get_text(own_elem, "directOrIndirectOwnership", "D").upper()
                nature = OwnershipNature.INDIRECT if own_code == "I" else OwnershipNature.DIRECT

                # Linked footnotes
                tx_fns: List[str] = []
                for fn_ref in trans.findall(".//footnoteId"):
                    fid = fn_ref.attrib.get("id")
                    if fid in footnotes:
                        tx_fns.append(footnotes[fid])

                total_val = round(shares * price, 2)
                signal_type = cls._classify_transaction_code(t_code)

                # Strict false-positive control for positive buying conviction:
                # Must be open-market purchase ('P'), acquired ('A'), positive price (>0.0), positive shares (>0)
                is_positive = (
                    signal_type == SmartMoneySignalType.OPEN_MARKET_PURCHASE
                    and acq_disp == "A"
                    and price > 0.0
                    and shares > 0.0
                )

                rec_key = f"{accession_number}:{primary_owner['cik']}:NON_DERIVATIVE:{seq}"
                hash_basis = f"{rec_key}:{shares}:{price}:{t_date}:{t_code}:{acq_disp}"
                content_hash = hashlib.sha256(hash_basis.encode("utf-8")).hexdigest()

                public_avail = acceptance_datetime if acceptance_datetime else None
                tx = SmartMoneyTransaction(
                    record_key=rec_key,
                    accession_number=accession_number,
                    issuer_cik=issuer_cik,
                    issuer_symbol=issuer_symbol,
                    issuer_name=issuer_name,
                    reporting_owner_cik=primary_owner["cik"],
                    reporting_owner_name=primary_owner["name"],
                    is_officer=primary_owner["is_officer"],
                    officer_title=primary_owner["officer_title"],
                    is_director=primary_owner["is_director"],
                    is_ten_percent_owner=primary_owner["is_ten_pct"],
                    filing_date=filing_date,
                    transaction_date=t_date,
                    transaction_code=t_code,
                    signal_type=signal_type,
                    shares=shares,
                    price_per_share=price,
                    total_value_usd=total_val,
                    acquired_disposed=acq_disp,
                    ownership_nature=nature,
                    is_derivative=False,
                    is_open_market_purchase_evidence=is_positive,
                    is_positive_buying_conviction=is_positive,
                    acceptance_datetime=acceptance_datetime,
                    public_availability_at=public_avail,
                    source_fetch_time=source_fetch_time,
                    normalized_at=now_iso,
                    footnotes=tx_fns,
                    source_url=source_url,
                    content_hash=content_hash,
                )
                transactions.append(tx)
                seq += 1

        # 6. Derivative Transactions (for identifying option exercises and grants)
        deriv_table = root.find("derivativeTable")
        if deriv_table is not None:
            d_seq = 0
            for trans in deriv_table.findall("derivativeTransaction"):
                t_date = cls._get_text(trans, "transactionDate", filing_date)
                t_code = cls._get_text(trans, "transactionCoding/transactionCode", "")
                amounts = trans.find("transactionAmounts")

                shares_raw = cls._get_text(amounts, "transactionShares", "0")
                price_raw = cls._get_text(amounts, "transactionPricePerShare", "0")
                acq_disp = cls._get_text(amounts, "transactionAcquiredDisposedCode", "").upper()

                try:
                    shares = float(shares_raw.replace(",", ""))
                except ValueError:
                    shares = 0.0

                try:
                    price = float(price_raw.replace("$", "").replace(",", ""))
                except ValueError:
                    price = 0.0

                own_elem = trans.find("ownershipNature")
                own_code = cls._get_text(own_elem, "directOrIndirectOwnership", "D").upper()
                nature = OwnershipNature.INDIRECT if own_code == "I" else OwnershipNature.DIRECT

                total_val = round(shares * price, 2)
                signal_type = cls._classify_transaction_code(t_code)

                # Derivative transactions NEVER count as direct open-market buying conviction in Wave 1
                rec_key = f"{accession_number}:{primary_owner['cik']}:DERIVATIVE:{d_seq}"
                hash_basis = f"{rec_key}:{shares}:{price}:{t_date}:{t_code}:{acq_disp}"
                content_hash = hashlib.sha256(hash_basis.encode("utf-8")).hexdigest()

                public_avail = acceptance_datetime if acceptance_datetime else None
                tx = SmartMoneyTransaction(
                    record_key=rec_key,
                    accession_number=accession_number,
                    issuer_cik=issuer_cik,
                    issuer_symbol=issuer_symbol,
                    issuer_name=issuer_name,
                    reporting_owner_cik=primary_owner["cik"],
                    reporting_owner_name=primary_owner["name"],
                    is_officer=primary_owner["is_officer"],
                    officer_title=primary_owner["officer_title"],
                    is_director=primary_owner["is_director"],
                    is_ten_percent_owner=primary_owner["is_ten_pct"],
                    filing_date=filing_date,
                    transaction_date=t_date,
                    transaction_code=t_code,
                    signal_type=signal_type,
                    shares=shares,
                    price_per_share=price,
                    total_value_usd=total_val,
                    acquired_disposed=acq_disp,
                    ownership_nature=nature,
                    is_derivative=True,
                    is_open_market_purchase_evidence=False,
                    is_positive_buying_conviction=False,  # Derivatives excluded from open market conviction
                    acceptance_datetime=acceptance_datetime,
                    public_availability_at=public_avail,
                    source_fetch_time=source_fetch_time,
                    normalized_at=now_iso,
                    footnotes=[],
                    source_url=source_url,
                    content_hash=content_hash,
                )
                transactions.append(tx)
                d_seq += 1

        disp = "FETCHED_AND_PARSED" if transactions else "FETCHED_NO_RELEVANT_TRANSACTIONS"
        return SmartMoneyFiling(
            accession_number=accession_number,
            issuer_cik=issuer_cik,
            issuer_symbol=issuer_symbol,
            filing_date=filing_date,
            acceptance_datetime=acceptance_datetime,
            document_type=doc_type,
            is_amendment=is_amendment,
            transactions=transactions,
            raw_xml=xml_str,
            source_content_hash=source_content_hash,
            source_url=source_url,
            fetched_at=fetched_at,
            disposition=disp,
        )
