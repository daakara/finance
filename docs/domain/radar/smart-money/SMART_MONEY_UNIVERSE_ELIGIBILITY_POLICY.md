# SMART MONEY WAVE 2A — UNIVERSE ELIGIBILITY POLICY

## 1. Executive Summary

This document governs the eligibility rules for the Smart Money scanner in ARX Terminal. Every inclusion and exclusion is grounded in federal securities law governing Section 16 reporting requirements under the Securities Exchange Act of 1934.

```text
ELIGIBILITY_POLICY_VERSION = "sm-eligibility-v1.0"
POLICY_IS_DETERMINISTIC = YES
FAIL_CLOSED_ON_AMBIGUITY = YES
```

---

## 2. Canonical Eligibility Rules Matrix

| Rule ID | Parameter | Rule Expression | Legal & Structural Rationale |
|:---|:---|:---|:---|
| **SM-E01** | Primary Exchange | `exchange IN ["NASDAQ", "NYSE", "CBOE", "BATS", "ARCA"]` | Section 12(b) registered national securities exchanges. OTC and Pink Sheets are excluded due to irregular Section 16 enforcement and liquidity risks. |
| **SM-E02** | Asset Class | `asset_class == "EQUITY"` | Debt, currencies, commodities, and derivatives are non-equity. |
| **SM-E03** | Security Subtype | `security_type == "COMMON_STOCK"` | Only domestic common stock represents residual corporate ownership subject to Section 16 insider purchasing signals. |
| **SM-E04** | Listing Status | `listing_status == "ACTIVE"` | Inactive, halted, or delisted securities cannot support actionable trade ideas. |
| **SM-E05** | Currency | `currency == "USD"` | Mandatory US domestic trading currency. |
| **SM-E06** | Primary Listing | `primary_listing == True` | Excludes secondary, tertiary, or unlisted cross-market quotes. |
| **SM-E07** | Verified SEC CIK | `cik IS_NOT_NULL AND LEN(cik) == 10` | Form 4 submissions are filed exclusively under issuer CIK. Absence of verified CIK prevents data retrieval. |

---

## 3. Explicit Instrument Classification & Disposition

### 1. Domestic Common Stock
- **Disposition**: **ELIGIBLE**
- **Rationale**: Officers, directors, and 10% owners of domestic public corporations are required by Section 16(a) to file Form 4 within 2 business days of transactions.

### 2. Foreign Private Issuers (ADRs and Foreign Ordinary Shares)
- **Disposition**: **INELIGIBLE / EXCLUDED**
- **Legal Authority**: **SEC Rule 3a12-3(b)** under the Securities Exchange Act of 1934 explicitly exempts securities of Foreign Private Issuers (FPIs) from Section 16!
- **Consequence**: Insiders of FPIs (e.g. TSM, ASML, NVO, BABA, SAP) are NOT required to file Form 4. Attempting to scan Form 4 for FPIs yields a legal zero that would falsely appear as "no insider buying". They file Form 6-K or annual 20-F instead.
- **Fail-Closed Exclusion Code**: `EXCLUDED_FOREIGN_PRIVATE_ISSUER`

### 3. Exchange-Traded Funds (ETFs) & Mutual Funds
- **Disposition**: **INELIGIBLE / EXCLUDED**
- **Legal Authority**: Regulated under the Investment Company Act of 1940. ETFs have no operating corporate officers or directors trading on personal conviction.
- **Fail-Closed Exclusion Code**: `EXCLUDED_INVESTMENT_COMPANY_ETF`

### 4. Preferred Stock, Warrants, Rights, and Units
- **Disposition**: **INELIGIBLE / EXCLUDED**
- **Structural Rationale**: Hybrid fixed-income or derivative claims that do not represent common equity growth conviction.
- **Fail-Closed Exclusion Code**: `EXCLUDED_DERIVATIVE_HYBRID_SECURITY`

### 5. Over-the-Counter (OTC / Pink Sheets) Securities
- **Disposition**: **INELIGIBLE / EXCLUDED**
- **Structural Rationale**: Unregistered or Section 12(g) issuers with limited filing enforcement, severe bid-ask spreads, and elevated fraud risk.
- **Fail-Closed Exclusion Code**: `EXCLUDED_OTC_EXCHANGE`

### 6. Special Purpose Acquisition Companies (SPACs) Pre-Merger
- **Disposition**: **INELIGIBLE / EXCLUDED**
- **Structural Rationale**: Cash trust shells with nominal yield and no operating earnings.
- **Fail-Closed Exclusion Code**: `EXCLUDED_PRE_MERGER_SPAC`

---

## 4. Deterministic Exclusion Reason Codes

When `SmartMoneyUniverseBuilder` evaluates each security in the source population, it assigns exactly one deterministic reason code:

- `PASSED`: Instrument satisfies all 7 eligibility rules.
- `INELIGIBLE_EXCHANGE`: Primary exchange not in major US list (e.g. OTC, foreign venue).
- `INELIGIBLE_SECURITY_TYPE`: Non-common stock (ETF, ADR, Preferred, Warrant, REIT).
- `INELIGIBLE_LISTING_STATUS`: Inactive, suspended, or delisted.
- `INELIGIBLE_CURRENCY`: Non-USD denomination.
- `INELIGIBLE_SECONDARY_LISTING`: Non-primary cross listing.
- `UNRESOLVED_MISSING_CIK`: Eligible instrument missing authoritative SEC CIK.
- `UNRESOLVED_AMBIGUOUS_CIK`: Ticker collides with multiple candidate CIKs.
