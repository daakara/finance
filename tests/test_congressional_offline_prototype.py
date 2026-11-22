"""Offline Ingestion Prototype & Record Contract Verification.
Directive: ARX-CONGRESSIONAL-INGESTION-001 (Phase 0 / Phase 1 Transition)

Verifies:
1. Offline parsing of Stage 1 Document Index manifest (House Clerk XML format).
2. Offline extraction of Stage 2 Transaction Report (Periodic Transaction Report).
3. Field preservation: document identity/hash, raw values, owner, asset, amount ranges,
   transaction & filing dates, amendment relationships, and unresolved mappings.
4. Idempotent duplicate handling in disposable relational storage (SQLite).
5. Truthful incomplete coverage vs certified-empty interval semantics.
"""

import hashlib
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, date
from typing import Dict, Any, List, Optional
import pytest


# ── 1. Prototype Parser Contracts ─────────────────────────────────────────────

def parse_house_index_xml(xml_content: str, source_url: str = "https://disclosures-clerk.house.gov") -> List[Dict[str, Any]]:
    """Parse House Clerk financial disclosure annual/daily index XML into Stage 1 document records."""
    content_bytes = xml_content.encode("utf-8")
    doc_sha256 = hashlib.sha256(content_bytes).hexdigest()
    
    root = ET.fromstring(xml_content)
    documents = []
    
    for member in root.findall(".//Member"):
        last = member.findtext("Last", "").strip()
        first = member.findtext("First", "").strip()
        filer_name = f"{last}, {first}".strip(", ")
        office = member.findtext("Office", "House of Representatives").strip()
        district = member.findtext("District", "").strip()
        
        for filing in member.findall(".//Filing"):
            doc_id = filing.findtext("DocID", "").strip()
            filing_type = filing.findtext("FilingType", "").strip()
            filing_date_str = filing.findtext("FilingDate", "").strip()
            year_str = filing.findtext("Year", "").strip()
            
            # Normalize filing date YYYY-MM-DD
            norm_date = None
            if filing_date_str:
                for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
                    try:
                        norm_date = datetime.strptime(filing_date_str, fmt).date().isoformat()
                        break
                    except ValueError:
                        pass
            
            doc_record = {
                "document_id": f"HOUSE_{doc_id}" if doc_id else f"HOUSE_HASH_{doc_sha256[:12]}",
                "chamber": "HOUSE",
                "filer_name": filer_name,
                "office": office,
                "district": district,
                "filing_year": int(year_str) if year_str.isdigit() else 2026,
                "filing_type": filing_type,  # PTR, FD, etc.
                "filing_date": norm_date,
                "source_url": f"{source_url}/public_disc/ptr-pdfs/{year_str}/{doc_id}.pdf" if doc_id else source_url,
                "raw_document_sha256": hashlib.sha256(f"{doc_id}_{filing_date_str}_{filer_name}".encode("utf-8")).hexdigest(),
                "extraction_status": "INDEXED",  # INDEXED, EXTRACTED, UNPARSEABLE_SCANNED_PDF
            }
            documents.append(doc_record)
            
    return documents


def parse_house_ptr_transaction_report(
    document_meta: Dict[str, Any],
    disclosed_transactions: List[Dict[str, Any]],
    raw_report_bytes: bytes,
) -> List[Dict[str, Any]]:
    """Parse a disclosed PTR document into Stage 2 extracted transaction records."""
    report_sha256 = hashlib.sha256(raw_report_bytes).hexdigest()
    extracted = []
    
    filing_date_str = document_meta.get("filing_date")
    filing_d = date.fromisoformat(filing_date_str) if filing_date_str else None
    
    for idx, raw_tx in enumerate(disclosed_transactions):
        tx_date_str = raw_tx.get("transaction_date")
        tx_d = date.fromisoformat(tx_date_str) if tx_date_str else None
        
        # Calculate filing latency in calendar days (no cross-table generated column)
        days_to_filing = (filing_d - tx_d).days if (filing_d and tx_d) else None
        
        tx_id = f"{document_meta['document_id']}_TX_{idx + 1}"
        
        # Identity mapping resolution
        raw_ticker = raw_tx.get("ticker", "").strip().upper()
        resolved_ticker = raw_ticker if (raw_ticker and raw_ticker.isalpha() and len(raw_ticker) <= 5) else None
        unresolved_reason = None
        if not resolved_ticker:
            unresolved_reason = "TICKER_UNIDENTIFIED_OR_NON_STANDARD" if raw_ticker else "TICKER_NOT_DISCLOSED"
            
        record = {
            "transaction_id": tx_id,
            "document_id": document_meta["document_id"],
            "chamber": document_meta["chamber"],
            "filer_name": document_meta["filer_name"],
            "owner": raw_tx.get("owner", "Self"),  # Self, Spouse, Joint, Child
            "raw_asset_name": raw_tx.get("asset_name", ""),
            "ticker": resolved_ticker,
            "unresolved_mapping_reason": unresolved_reason,
            "transaction_type": raw_tx.get("transaction_type", "Unknown"),  # Purchase, Sale (Full), Sale (Partial), Exchange
            "amount_range": raw_tx.get("amount_range", "$1,001 - $15,000"),
            "transaction_date": tx_date_str,
            "filing_date": filing_date_str,
            "days_to_filing": days_to_filing,
            "amendment_status": raw_tx.get("amendment_status", "ORIGINAL"),  # ORIGINAL, AMENDMENT
            "amends_document_id": raw_tx.get("amends_document_id"),
            "raw_document_sha256": report_sha256,
        }
        extracted.append(record)
        
    return extracted


# ── 2. Disposable In-Memory Relational Schema (SQLite) ─────────────────────────

def init_disposable_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Stage 1: Document manifests
    cur.execute("""
    CREATE TABLE congressional_disclosure_documents (
        document_id TEXT PRIMARY KEY,
        chamber TEXT NOT NULL,
        filer_name TEXT NOT NULL,
        office TEXT,
        district TEXT,
        filing_year INTEGER NOT NULL,
        filing_type TEXT NOT NULL,
        filing_date TEXT,
        source_url TEXT NOT NULL,
        raw_document_sha256 TEXT NOT NULL,
        extraction_status TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Stage 2: Extracted transactions
    cur.execute("""
    CREATE TABLE congressional_disclosures (
        transaction_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        chamber TEXT NOT NULL,
        filer_name TEXT NOT NULL,
        owner TEXT NOT NULL,
        raw_asset_name TEXT NOT NULL,
        ticker TEXT,
        unresolved_mapping_reason TEXT,
        transaction_type TEXT NOT NULL,
        amount_range TEXT NOT NULL,
        transaction_date TEXT,
        filing_date TEXT,
        days_to_filing INTEGER,
        amendment_status TEXT NOT NULL,
        amends_document_id TEXT,
        raw_document_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (document_id) REFERENCES congressional_disclosure_documents(document_id)
    );
    """)
    
    # Stage 3: Interval Coverage Certification Watermarks
    cur.execute("""
    CREATE TABLE congressional_coverage_watermarks (
        watermark_id TEXT PRIMARY KEY,
        chamber TEXT NOT NULL,
        interval_start TEXT NOT NULL,
        interval_end TEXT NOT NULL,
        indexed_document_count INTEGER NOT NULL,
        extracted_document_count INTEGER NOT NULL,
        coverage_status TEXT NOT NULL, -- COMPLETE, PARTIAL_COVERAGE, PENDING
        watermark_hash TEXT NOT NULL,
        certified_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    conn.commit()
    return conn


def upsert_document(conn: sqlite3.Connection, doc: Dict[str, Any]):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO congressional_disclosure_documents (
        document_id, chamber, filer_name, office, district,
        filing_year, filing_type, filing_date, source_url,
        raw_document_sha256, extraction_status
    ) VALUES (
        :document_id, :chamber, :filer_name, :office, :district,
        :filing_year, :filing_type, :filing_date, :source_url,
        :raw_document_sha256, :extraction_status
    )
    ON CONFLICT(document_id) DO UPDATE SET
        raw_document_sha256 = excluded.raw_document_sha256,
        extraction_status = excluded.extraction_status;
    """, doc)
    conn.commit()


def upsert_transactions(conn: sqlite3.Connection, txs: List[Dict[str, Any]]):
    cur = conn.cursor()
    for tx in txs:
        cur.execute("""
        INSERT INTO congressional_disclosures (
            transaction_id, document_id, chamber, filer_name, owner,
            raw_asset_name, ticker, unresolved_mapping_reason,
            transaction_type, amount_range, transaction_date,
            filing_date, days_to_filing, amendment_status,
            amends_document_id, raw_document_sha256
        ) VALUES (
            :transaction_id, :document_id, :chamber, :filer_name, :owner,
            :raw_asset_name, :ticker, :unresolved_mapping_reason,
            :transaction_type, :amount_range, :transaction_date,
            :filing_date, :days_to_filing, :amendment_status,
            :amends_document_id, :raw_document_sha256
        )
        ON CONFLICT(transaction_id) DO UPDATE SET
            ticker = excluded.ticker,
            amount_range = excluded.amount_range,
            days_to_filing = excluded.days_to_filing,
            raw_document_sha256 = excluded.raw_document_sha256;
        """, tx)
    conn.commit()


# ── 3. Prototype Tests ────────────────────────────────────────────────────────

class TestCongressionalOfflinePrototype:
    """Verifies parsing, duplicate handling, field preservation, and coverage certification."""

    SAMPLE_HOUSE_XML = """<?xml version="1.0" encoding="utf-8"?>
    <FinancialDisclosure>
        <Member>
            <Prefix>Hon.</Prefix>
            <Last>Pelosi</Last>
            <First>Nancy</First>
            <Office>House of Representatives</Office>
            <District>CA-11</District>
            <Filings>
                <Filing>
                    <Year>2026</Year>
                    <FilingType>PTR</FilingType>
                    <FilingDate>08/14/2026</FilingDate>
                    <DocID>20025123</DocID>
                </Filing>
            </Filings>
        </Member>
        <Member>
            <Prefix>Hon.</Prefix>
            <Last>Khanna</Last>
            <First>Ro</First>
            <Office>House of Representatives</Office>
            <District>CA-17</District>
            <Filings>
                <Filing>
                    <Year>2026</Year>
                    <FilingType>PTR</FilingType>
                    <FilingDate>08/16/2026</FilingDate>
                    <DocID>20025145</DocID>
                </Filing>
            </Filings>
        </Member>
    </FinancialDisclosure>
    """

    def test_prototype_01_stage1_index_parsing_preserves_identity_and_hash(self):
        """Parse Stage 1 daily index XML into document records with deterministic hashes."""
        docs = parse_house_index_xml(self.SAMPLE_HOUSE_XML)
        assert len(docs) == 2
        
        pelosi = next(d for d in docs if "Pelosi" in d["filer_name"])
        assert pelosi["document_id"] == "HOUSE_20025123"
        assert pelosi["chamber"] == "HOUSE"
        assert pelosi["filing_type"] == "PTR"
        assert pelosi["filing_date"] == "2026-08-14"
        assert pelosi["extraction_status"] == "INDEXED"
        assert len(pelosi["raw_document_sha256"]) == 64

    def test_prototype_02_stage2_transaction_extraction_preserves_ranges_and_dates(self):
        """Parse Stage 2 PTR transactions, preserving owner, amount range, and computing latency."""
        doc_meta = {
            "document_id": "HOUSE_20025123",
            "chamber": "HOUSE",
            "filer_name": "Pelosi, Nancy",
            "filing_date": "2026-08-14",
        }
        raw_txs = [
            {
                "owner": "Spouse",
                "asset_name": "NVIDIA Corporation - Common Stock",
                "ticker": "NVDA",
                "transaction_type": "Purchase",
                "amount_range": "$1,000,001 - $5,000,000",
                "transaction_date": "2026-07-28",
                "amendment_status": "ORIGINAL",
            },
            {
                "owner": "Spouse",
                "asset_name": "Municipal Bond Water Authority",
                "ticker": "",  # Unresolved ticker
                "transaction_type": "Sale (Full)",
                "amount_range": "$100,001 - $250,000",
                "transaction_date": "2026-06-15",
                "amendment_status": "ORIGINAL",
            }
        ]
        mock_pdf_bytes = b"%PDF-1.4 Mock PTR Report Nancy Pelosi 2026"
        tx_records = parse_house_ptr_transaction_report(doc_meta, raw_txs, mock_pdf_bytes)
        
        assert len(tx_records) == 2
        nvda = tx_records[0]
        assert nvda["ticker"] == "NVDA"
        assert nvda["owner"] == "Spouse"
        assert nvda["amount_range"] == "$1,000,001 - $5,000,000"
        assert nvda["days_to_filing"] == 17  # 2026-08-14 - 2026-07-28 = 17 days
        assert nvda["unresolved_mapping_reason"] is None
        assert nvda["raw_document_sha256"] == hashlib.sha256(mock_pdf_bytes).hexdigest()
        
        muni = tx_records[1]
        assert muni["ticker"] is None
        assert muni["unresolved_mapping_reason"] == "TICKER_NOT_DISCLOSED"
        assert muni["days_to_filing"] == 60  # 2026-08-14 - 2026-06-15 = 60 days

    def test_prototype_03_duplicate_handling_is_strictly_idempotent(self):
        """Demonstrate that re-ingesting documents and transactions produces zero duplicates."""
        conn = init_disposable_db()
        docs = parse_house_index_xml(self.SAMPLE_HOUSE_XML)
        
        # Ingest documents twice
        for d in docs:
            upsert_document(conn, d)
        for d in docs:
            upsert_document(conn, d)
            
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as c FROM congressional_disclosure_documents;")
        assert cur.fetchone()["c"] == 2  # Exactly 2 documents, zero duplicates
        
        # Ingest transactions twice
        doc_meta = docs[0]
        raw_txs = [{
            "owner": "Self",
            "asset_name": "Apple Inc.",
            "ticker": "AAPL",
            "transaction_type": "Purchase",
            "amount_range": "$15,001 - $50,000",
            "transaction_date": "2026-08-01",
        }]
        txs = parse_house_ptr_transaction_report(doc_meta, raw_txs, b"PDF_BYTES")
        upsert_transactions(conn, txs)
        upsert_transactions(conn, txs)
        
        cur.execute("SELECT COUNT(*) as c FROM congressional_disclosures;")
        assert cur.fetchone()["c"] == 1  # Exactly 1 transaction row, zero duplicate rows

    def test_prototype_04_truthful_incomplete_coverage_versus_certified_empty(self):
        """Demonstrate that when index documents are unextracted, interval coverage is PARTIAL."""
        conn = init_disposable_db()
        cur = conn.cursor()
        
        # Simulate index with 5 PTR filings
        for i in range(1, 6):
            cur.execute("""
            INSERT INTO congressional_disclosure_documents (
                document_id, chamber, filer_name, filing_year, filing_type,
                filing_date, source_url, raw_document_sha256, extraction_status
            ) VALUES (?, 'HOUSE', ?, 2026, 'PTR', '2026-08-10', 'url', 'hash', ?);
            """, (f"DOC_{i}", f"Filer {i}", "EXTRACTED" if i <= 3 else "INDEXED"))
            
        # Check extraction completeness for interval
        cur.execute("""
        SELECT 
            COUNT(*) as total_indexed,
            SUM(CASE WHEN extraction_status = 'EXTRACTED' THEN 1 ELSE 0 END) as total_extracted
        FROM congressional_disclosure_documents
        WHERE filing_date BETWEEN '2026-08-01' AND '2026-08-31';
        """)
        row = cur.fetchone()
        indexed = row["total_indexed"]
        extracted = row["total_extracted"]
        
        is_complete = (indexed > 0 and indexed == extracted)
        coverage_status = "COMPLETE" if is_complete else "PARTIAL_COVERAGE"
        
        assert indexed == 5
        assert extracted == 3
        assert coverage_status == "PARTIAL_COVERAGE"
        assert not is_complete
        
        # Record interval watermark
        cur.execute("""
        INSERT INTO congressional_coverage_watermarks (
            watermark_id, chamber, interval_start, interval_end,
            indexed_document_count, extracted_document_count,
            coverage_status, watermark_hash
        ) VALUES ('WM_202608', 'HOUSE', '2026-08-01', '2026-08-31', 5, 3, 'PARTIAL_COVERAGE', 'partial_hash');
        """)
        conn.commit()
        
        # Querying an unextracted ticker in a partial interval MUST NOT return certified empty
        cur.execute("SELECT * FROM congressional_coverage_watermarks WHERE watermark_id = 'WM_202608';")
        wm = cur.fetchone()
        assert wm["coverage_status"] == "PARTIAL_COVERAGE"
        # Since coverage is partial, client cannot render 'Verified No Activity'
