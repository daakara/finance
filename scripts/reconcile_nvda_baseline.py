#!/usr/bin/env python3
"""
Reconciliation script for ARX Issue #165: NVDA Form 4 Evidence.
Performs offline, deterministic audit of the 16-filing baseline and compares against retained submissions index.
"""

import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

EXPORT_PATH = Path("/Users/iDavid/Projects/finance-sec-expansion/data/research/nvda_form4_frozen_baseline_export.json")
SUBMISSIONS_PATH = Path("/Users/iDavid/Projects/finance-clean/data/research/sourced_fundamentals/raw/CIK0001045810/CIK0001045810_submissions.json")

def main():
    print("=" * 80)
    print("ARX ISSUE #165: NVDA FORM 4 BASELINE RECONCILIATION")
    print("=" * 80)

    with open(EXPORT_PATH, "r", encoding="utf-8") as f:
        export_data = json.load(f)

    # 1. HASH RECOMPUTATION & METADATA VERIFICATION
    print("\n--- SECTION 1: HASH RECOMPUTATION & ISSUER IDENTITY ---")
    filings = export_data["raw_filings"]
    print(f"Total raw filings in baseline: {len(filings)}")

    all_match = True
    content_hashes = []
    accession_inventory = []

    for f in filings:
        acc = f["accession_number"]
        accession_inventory.append(acc)
        xml = f["xml_content"]
        rec_hash = f["content_hash"]
        calc_hash = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        match = (calc_hash == rec_hash)
        if not match:
            all_match = False
        content_hashes.append(rec_hash)
        cik = f["cik"]
        ticker = f["ticker"]
        fdate = f["filing_date"]
        accepted = f["accepted_datetime"]
        print(f"Accession: {acc} | Date: {fdate} | CIK: {cik} | Ticker: {ticker} | Hash Match: {match}")

    print(f"All 16/16 content hashes match exactly: {all_match}")
    ordered_concat_digest = hashlib.sha256("".join(content_hashes).encode("utf-8")).hexdigest()
    print(f"Ordered Concatenated Aggregate SHA-256: {ordered_concat_digest}")

    # Check for duplicate accessions
    acc_counts = Counter(accession_inventory)
    duplicates = [acc for acc, count in acc_counts.items() if count > 1]
    print(f"Duplicate accessions in baseline: {duplicates if duplicates else 'None (0 duplicates)'}")

    # 2. TRANSACTION DERIVATION & BREAKDOWN
    print("\n--- SECTION 2: TRANSACTION DERIVATION & ZERO-TRANSACTION AUDIT ---")
    total_non_derivative = 0
    total_derivative = 0
    all_codes = Counter()
    filing_details = []

    for f in filings:
        acc = f["accession_number"]
        xml = f["xml_content"]
        root = ET.fromstring(xml)

        doc_type = root.findtext("documentType") or ""
        period = root.findtext("periodOfReport") or ""
        owner_name = root.findtext(".//rptOwnerName") or ""
        owner_cik = root.findtext(".//rptOwnerCik") or ""

        non_deriv = root.findall(".//nonDerivativeTransaction")
        deriv = root.findall(".//derivativeTransaction")

        nd_codes = [t.findtext(".//transactionCode") for t in non_deriv]
        d_codes = [t.findtext(".//transactionCode") for t in deriv]

        total_non_derivative += len(non_deriv)
        total_derivative += len(deriv)

        for c in nd_codes:
            all_codes[f"Non-Derivative {c}"] += 1
        for c in d_codes:
            all_codes[f"Derivative {c}"] += 1

        details = {
            "accession": acc,
            "doc_type": doc_type,
            "filing_date": f["filing_date"],
            "period": period,
            "owner_name": owner_name,
            "owner_cik": owner_cik,
            "non_deriv_count": len(non_deriv),
            "deriv_count": len(deriv),
            "total_count": len(non_deriv) + len(deriv),
            "nd_codes": nd_codes,
            "d_codes": d_codes,
        }
        filing_details.append(details)

        print(f"Acc: {acc} ({doc_type:4s}) | Date: {f['filing_date']} | Owner: {owner_name:<25s} | NonDeriv: {len(non_deriv):2d} {nd_codes} | Deriv: {len(deriv):2d} {d_codes} | Total: {len(non_deriv)+len(deriv):2d}")

    total_parsed_from_xml = total_non_derivative + total_derivative
    print(f"\nTotal Transactions Parsed from XML: {total_parsed_from_xml}")
    print(f"  - Non-Derivative Transactions: {total_non_derivative}")
    print(f"  - Derivative Transactions: {total_derivative}")
    print(f"Transaction Code Distribution across baseline: {dict(all_codes)}")

    # Stored parsed transactions in database
    db_parsed = export_data["parsed_transactions"]
    print(f"\nStored Parsed Transactions in Database: {len(db_parsed)}")
    db_codes = Counter(t["transaction_code"] for t in db_parsed)
    print(f"Database Transaction Code Breakdown: {dict(db_codes)}")
    db_is_derivative = Counter(t["is_derivative"] for t in db_parsed)
    print(f"Database is_derivative Breakdown: {dict(db_is_derivative)}")

    # Zero-transaction filings
    zero_tx_filings = [d for d in filing_details if d["total_count"] == 0]
    print(f"\nFilings with 0 transactions: {len(zero_tx_filings)}")
    for zf in zero_tx_filings:
        print(f"  Accession: {zf['accession']}")
        print(f"  Document Type: {zf['doc_type']}")
        print(f"  Filing Date: {zf['filing_date']}")
        print(f"  Period of Report: {zf['period']}")
        print(f"  Reporting Owner: {zf['owner_name']} (CIK {zf['owner_cik']})")

        # Let's inspect raw XML for holding tables or remarks
        matching_f = [f for f in filings if f["accession_number"] == zf["accession"]][0]
        root = ET.fromstring(matching_f["xml_content"])
        holding_nd = root.findall(".//nonDerivativeHolding")
        holding_d = root.findall(".//derivativeHolding")
        remarks = root.findtext("remarks") or ""
        print(f"  Non-Derivative Holdings reported: {len(holding_nd)}")
        print(f"  Derivative Holdings reported: {len(holding_d)}")
        print(f"  Remarks: {remarks}")
        for fn in root.findall(".//footnotes/footnote"):
            print(f"  Footnote {fn.attrib.get('id')}: {fn.text}")

    # 3. AMENDMENT ANALYSIS
    print("\n--- SECTION 3: AMENDMENT RECONCILIATION ---")
    amends = export_data["amendments"]
    print(f"Amendments in baseline database: {len(amends)}")
    for a in amends:
        print(f"  Amendment: {a['amendment_accession']} -> Base: {a['base_accession']} | Number: {a['amendment_number']} | Status: {a['resolution_status']}")

    # Let's inspect base vs amendment filings
    base_acc = "0002152188-26-000005"
    amend_acc = "0002152188-26-000007"
    base_f = [f for f in filings if f["accession_number"] == base_acc][0]
    amend_f = [f for f in filings if f["accession_number"] == amend_acc][0]

    base_root = ET.fromstring(base_f["xml_content"])
    amend_root = ET.fromstring(amend_f["xml_content"])

    print(f"\nBase Filing ({base_acc}):")
    print(f"  Doc Type: {base_root.findtext('documentType')}")
    print(f"  Filing Date: {base_f['filing_date']} | Accepted: {base_f['accepted_datetime']}")
    print(f"  Period: {base_root.findtext('periodOfReport')}")
    print(f"  Owner: {base_root.findtext('.//rptOwnerName')} ({base_root.findtext('.//rptOwnerCik')})")
    for tx in base_root.findall(".//nonDerivativeTransaction"):
        print(f"  Transaction: Date={tx.findtext('.//transactionDate/value')} Code={tx.findtext('.//transactionCode')} Shares={tx.findtext('.//transactionShares/value')} Price={tx.findtext('.//transactionPricePerShare/value')} AcqDisp={tx.findtext('.//transactionAcquiredDisposedCode/value')}")

    print(f"\nAmendment Filing ({amend_acc}):")
    print(f"  Doc Type: {amend_root.findtext('documentType')}")
    print(f"  Filing Date: {amend_f['filing_date']} | Accepted: {amend_f['accepted_datetime']}")
    print(f"  Period: {amend_root.findtext('periodOfReport')}")
    print(f"  Date of Original Submission: {amend_root.findtext('dateOfOriginalSubmission')}")
    print(f"  Remarks: {amend_root.findtext('remarks')}")
    for fn in amend_root.findall(".//footnotes/footnote"):
        print(f"  Footnote {fn.attrib.get('id')}: {fn.text}")
    for tx in amend_root.findall(".//nonDerivativeTransaction"):
        print(f"  Transaction: Date={tx.findtext('.//transactionDate/value')} Code={tx.findtext('.//transactionCode')} Shares={tx.findtext('.//transactionShares/value')} Price={tx.findtext('.//transactionPricePerShare/value')} AcqDisp={tx.findtext('.//transactionAcquiredDisposedCode/value')}")

    # Check amendment classification
    from analyst_dashboard.smart_money.form4_parser import Form4XmlParser
    from analyst_dashboard.smart_money.amendment_classifier import classify_amendment_effect
    
    base_parsed = Form4XmlParser.parse_filing(base_f["xml_content"], base_f["accession_number"])
    amend_parsed = Form4XmlParser.parse_filing(amend_f["xml_content"], amend_f["accession_number"])
    base_txs = [t.__dict__ for t in base_parsed.transactions]

    res = classify_amendment_effect(
        amendment_filing=amend_parsed,
        base_filing={"accession_number": base_f["accession_number"], "filing_date": base_f["filing_date"]},
        base_transactions=base_txs,
        raw_xml=amend_f["xml_content"]
    )
    print(f"\nClassifier Result for {amend_acc}:")
    print(f"  Classified Effect: {res.status}")
    print(f"  Classification Reason: {res.reason}")
    print(f"  Mappings: {res.mappings}")

    # 4. SUBMISSIONS INDEX COMPARISON
    print("\n--- SECTION 4: RETAINED SUBMISSIONS INDEX COMPARISON ---")
    print(f"Submissions index file: {SUBMISSIONS_PATH}")
    with open(SUBMISSIONS_PATH, "r", encoding="utf-8") as f:
        sub_data = json.load(f)

    sub_recent = sub_data.get("filings", {}).get("recent", {})
    sub_accs = sub_recent.get("accessionNumber", [])
    sub_forms = sub_recent.get("form", [])
    sub_fdates = sub_recent.get("filingDate", [])
    sub_adts = sub_recent.get("acceptanceDateTime", [])

    print(f"Total filings in submissions.json recent catalog: {len(sub_accs)}")

    # Window definitions
    # Current window: August 26, 2026 to November 23, 2026 inclusive
    WINDOW_START = "2026-08-26"
    WINDOW_END = "2026-11-23"

    form4_in_sub = []
    for acc, form, fdate, adt in zip(sub_accs, sub_forms, sub_fdates, sub_adts):
        if form in ("4", "4/A"):
            form4_in_sub.append({
                "accession": acc,
                "form": form,
                "filing_date": fdate,
                "accepted_datetime": adt,
                "in_window": (WINDOW_START <= fdate <= WINDOW_END),
            })

    print(f"Total Form 4 & 4/A filings in recent catalog: {len(form4_in_sub)}")

    in_window_sub = [f for f in form4_in_sub if f["in_window"]]
    out_window_sub = [f for f in form4_in_sub if not f["in_window"]]

    print(f"\nIn-Window Form 4 & 4/A Filings ({WINDOW_START} to {WINDOW_END}): {len(in_window_sub)}")
    for f in in_window_sub:
        in_base = f["accession"] in accession_inventory
        print(f"  {f['accession']} | Form: {f['form']:3s} | Date: {f['filing_date']} | Accepted: {f['accepted_datetime']} | In Baseline: {in_base}")

    print(f"\nOut-of-Window Form 4 & 4/A Filings (earlier than {WINDOW_START}):")
    for f in out_window_sub[:10]:
        in_base = f["accession"] in accession_inventory
        print(f"  {f['accession']} | Form: {f['form']:3s} | Date: {f['filing_date']} | Accepted: {f['accepted_datetime']} | In Baseline: {in_base}")

    # Check baseline accessions vs submissions index
    print("\n--- BASELINE VS SUBMISSIONS INDEX CROSS-CHECK ---")
    for f in filings:
        acc = f["accession_number"]
        fdate = f["filing_date"]
        in_window = (WINDOW_START <= fdate <= WINDOW_END)
        in_sub = any(s["accession"] == acc for s in form4_in_sub)
        print(f"Baseline {acc} | Date: {fdate} | In-Window: {in_window:<5} | In Submissions Index: {in_sub}")

if __name__ == "__main__":
    main()
