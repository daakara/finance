# ARX Terminal — Source Population Acquisition & Accounting Report

**Classification**: Engineering Artifact — Listing Source Acquisition & Offline Accounting  
**Generated At (UTC)**: `2026-10-16T19:40:58.226402+00:00`  
**Acquisition Version**: `1.0.0`  
**Overall Outcome**: `PARTIAL_ACQUISITION`  
**Production Population Completeness**: `NOT_ESTABLISHED`  

> [!IMPORTANT]
> **Core Product Acceptance Criterion**:
> *“Every stock in the agreed US-listed population is accounted for, and every stock supported by the strategy and available data receives an assessment. Radar explicitly discloses exclusions, unsupported assessments, unavailable data, failures, and coverage limitations.”*
> 
> **A successful download, directory join, parser, or diagnostic subset alone does NOT establish complete market coverage.**

---

## 1. Network Acquisition Evidence

| Source ID | Outcome | HTTP Status | Payload Size | SHA-256 Digest | Publisher TS | Local Raw Artifact |
|:---|:---:|:---:|:---:|:---|:---:|:---|
| `sec_company_tickers_exchange` | `SUCCESS` | `200` | `523,925 bytes` | `d04923d674c6d008...` | `Thu, 15 Oct 2026 13:39:20 GMT` | `sec_company_tickers_exchange.json` |
| `nasdaq_listed_directory` | `SUCCESS` | `200` | `349,723 bytes` | `67f42dd0594aff8e...` | `1016202614:01` | `nasdaq_listed_directory.txt` |
| `other_listed_directory` | `SUCCESS` | `200` | `543,140 bytes` | `a0cb851a772098d6...` | `1016202614:01` | `other_listed_directory.txt` |

---

## 2. Offline Parser Replay & Source Accounting

| Source ID | Feed Type | Replay Usable | Physical Lines | Metadata Rows | Data Observations | Malformed Rows | Unique Records |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `sec_company_tickers_exchange` | `SEC_COMPANY_TICKERS_EXCHANGE` | `NO` | `1` | `1` | `10,276` | `168` | `10,276` |
| `nasdaq_listed_directory` | `NASDAQ_LISTED_DIRECTORY` | `YES` | `5,636` | `2` | `5,634` | `0` | `5,634` |
| `other_listed_directory` | `OTHER_LISTED_DIRECTORY` | `YES` | `7,676` | `2` | `7,674` | `0` | `7,674` |

---

## 3. Diagnostic Cross-Source Matching (Non-Binding)

| Metric | Count | Operational / Boundary Meaning |
|:---|:---:|:---|
| **Total Candidates** | `16,448` | Total distinct join candidates produced across feeds. |
| **Fully Matched Across Feeds** | `7,136` | Observations matched across both SEC and exchange directory feeds. |
| **SEC Only (Unmatched)** | `3,140` | Observations present in SEC master but absent from exchange directories. |
| **Directory Only (Unmatched)** | `6,172` | Observations present in directories but absent from SEC master. |
| **Ambiguous Candidates** | `0` | Candidates with multiple or ambiguous join matches. |
| **Conflict Candidates** | `471` | Candidates with venue or attribute disagreements across feeds. |
| **Malformed Row Observations** | `168` | Malformed raw rows observed during feed replay. |

---

## 4. Disclosed Coverage Limitations

- Directory presence does not establish global instrument primary listing or trading suspension status.
- SEC company_tickers_exchange is issuer-level (CIK-based); multiple share classes share one CIK.
- OTC and unlisted securities are excluded from exchange directory feeds.
- Foreign issuers listed on US exchanges (ADRs, ordinary shares) have distinct regulatory reporting requirements.
- Market coverage denominator across full US-listed population is NOT_ESTABLISHED by directory join alone.

---

## 5. Unresolved Governance & Strategy Decisions

- ? Authoritative venue allowlist definition (NYSE, NASDAQ, AMEX, ARCA, BATS vs OTC/regional).
- ? Handling of share classes, dual listings, and warrants/units in scanner eligibility.
- ? Authoritative source for primary listing verification and active trading status.
- ? OpenFIGI integration model (batch licensing vs free public tier bounds).
