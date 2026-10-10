# SMART MONEY WAVE 2A — MARKET-WIDE UNIVERSE AUTHORITY

## 1. Executive Summary

This document establishes the governing universe authority architecture for the Smart Money scanner in ARX Terminal. It defines the authoritative source population, canonical security identification hierarchy, identifier reconciliation, and separation from existing regression fixtures.

---

## 2. Current Universe Architecture & Separation

```text
CURRENT_UNIVERSE_ARCHITECTURE =
  Source Population (SEC Official Exchange Index / Server Security Master)
  → Canonical Security Identity (SEC_US_{TICKER}_{EXCHANGE} + CIK)
  → Smart Money Eligibility Policy (Section 16 Applicable Common Equities)
  → Point-in-Time Universe Build (SmartMoneyUniverseBuilder)
  → Immutable Build Manifest & Scannable Membership
```

### Decoupling from VCP Reference Fixture
```text
CANONICAL_VCP_UNIVERSE =
  INTERNAL_REFERENCE / REGRESSION SET (35 curated symbols)

SMART_MONEY_MARKET_WIDE_UNIVERSE =
  INDEPENDENT PRODUCT AUTHORITY (Market-wide US equity population)

VCP_35_USED_AS_SMART_MONEY_PRODUCTION_UNIVERSE =
  NO

VCP_35_ALLOWED_FOR_BOUNDED_ENGINEERING_TESTS =
  YES
```
Under no circumstances is the 35-asset VCP regression set used as the market-wide production universe for Smart Money.

---

## 3. Upstream Source Population Evaluation

| Dimension | SEC EDGAR Exchange Index | Alpaca Asset Directory | OpenFIGI Symbology |
|:---|:---|:---|:---|
| **SOURCE_NAME** | `SEC_EDGAR_COMPANY_TICKERS_EXCHANGE` | `ALPACA_ASSET_DIRECTORY` | `OPENFIGI_V3` |
| **SOURCE_AUTHORITY** | U.S. Securities & Exchange Commission | Alpaca Securities LLC | Bloomberg / OpenFIGI Consortium |
| **SOURCE_TYPE** | Regulatory Authority Primary Record | Broker-Dealer Reference Directory | Symbology Provider |
| **UPDATE_CADENCE** | Continuous / Nightly SEC batch | Real-time / Daily | Continuous API |
| **IDENTIFIERS** | CIK (10-digit), Ticker, Issuer Name, Exchange | Alpaca UUID, Symbol, Exchange, Class | FIGI, Composite FIGI, Share Class FIGI |
| **ACTIVE_STATUS** | Exchange listed + reporting | `active` / `inactive` | Active instrument coverage |
| **EXCHANGE_AVAILABLE** | YES (`Nasdaq`, `NYSE`, `CBOE`, `OTC`) | YES (`NASDAQ`, `NYSE`, `ARCA`, `BATS`) | YES (`US`) |
| **SEC_CIK_AVAILABLE** | **YES (Authoritative Primary CIK)** | NO (Missing from Alpaca payload) | YES (Available via FIGI metadata) |
| **SHARE_CLASS** | PARTIAL (via ticker suffix) | PARTIAL | **YES (Explicit share class FIGI)** |
| **DELIVERY_REPRODUCIBLE** | **YES (Deterministic JSON payload)** | YES (Snapshot API) | YES |
| **LICENSE_CONSTRAINT** | Public Domain (US Federal Government) | API Key Required (Commercial Terms) | Free tier rate-limited / API key |
| **SUITABLE_FOR_SMART_MONEY** | **PRIMARY AUTHORITATIVE SOURCE** | SECONDARY ENRICHMENT | SECONDARY ENRICHMENT |

**Binding Source Determination**:
The official SEC EDGAR Exchange Index (`https://www.sec.gov/files/company_tickers_exchange.json`) constitutes the primary upstream source population for Smart Money because SEC Form 4 insider reporting is legally anchored to SEC CIK registration and Section 12 exchange listing.

---

## 4. Canonical Security Identity Hierarchy

To ensure resilience against mutable ticker symbols, corporate renamings, and share class splits, ARX Terminal establishes a five-tier identity hierarchy:

1. **`CANONICAL_ISSUER_ID`**:
   - Format: `SEC_CIK_{10_DIGIT_CIK}` (e.g. `SEC_CIK_0000320193` for Apple Inc.).
   - Role: Immutable legal entity identity under federal securities law. Survives ticker changes and corporate name changes.
2. **`CANONICAL_SECURITY_ID`**:
   - Format: `US_EQ_{EXCHANGE}_{TICKER}_{CIK}` (e.g. `US_EQ_NASDAQ_AAPL_0000320193`).
   - Role: Stable primary instrument identity distinguishing multiple share classes of the same issuer.
3. **`TICKER_ROLE`**:
   - Ephemeral market display identifier (e.g. `GOOGL` vs `GOOG`). Subject to corporate rebranding and exchange reallocation.
4. **`CIK_ROLE`**:
   - Statutory repository key for all Section 16 Form 4 ownership documents on EDGAR.
5. **`EXCHANGE_ROLE`**:
   - Venue authority determining Section 12(b) major exchange status vs OTC.
6. **`SHARE_CLASS_ROLE`**:
   - Distinguishes voting/economic classes (e.g. Class A Common Stock vs Class B Common Stock).

---

## 5. Drift and Relisting Governance

- **Ticker Changes**: When a company changes ticker (e.g. `FB` -> `META`), the `CANONICAL_ISSUER_ID` (`0001326801`) remains constant. The mapping directory records the transition with `previous_tickers: ["FB"]` while preserving historical Form 4 transaction lineage.
- **Dual Share Classes**: Issuers with multiple listed equity share classes (e.g. Berkshire Hathaway `BRK.A` and `BRK.B`, Alphabet `GOOGL` and `GOOG`) share a single `issuer_cik`. Each share class receives a distinct `CANONICAL_SECURITY_ID`. Form 4 transactions report transactions on specific security titles (e.g. `<securityTitle>Class A Common Stock</securityTitle>`), allowing precise share-class allocation.
- **Delisted Securities**: When an issuer delists from NASDAQ/NYSE, its listing status transitions to `INACTIVE` or `OTC`. It is immediately flagged as ineligible in the next point-in-time universe build.
