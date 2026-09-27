# ETF CONFIRMATORY SUBTYPE CLASSIFICATION POLICY V1.1
## NORMATIVE DECISION TABLE & INTERPRETATION AUTHORITY

**Policy Name**: `ETF_CONFIRMATORY_SUBTYPE_CLASSIFICATION_POLICY`  
**Policy Version**: `1.1.0`  
**Effective Spec Version**: `1.0.3`  
**Spec Commit**: `a5efc1777d130c23631986422731b7efd75e4624`  
**Policy SHA256**: `864133d98750f7765409153c4305d02ff9d422aa56556b6a0506738299642f52`  
**Governance Status**: `FROZEN_PRE_EXECUTION`  

---

### 1. POSITIVE CERTIFICATION RESTRICTIONS (NORMATIVE)

The following negative constraints are absolute and override any heuristic or inference:

1. `heuristic_name_match_can_certify_confirmatory_subtype`: **`FALSE`**
   - Fund names, marketing labels, or ticker strings may NEVER affirmatively certify an instrument into a confirmatory research subtype.
2. `name_regex_restricted_to_negative_safety_net_only`: **`TRUE`**
   - Text regex on fund names is permitted ONLY as a defensive negative filter (e.g. detecting "Leveraged", "Inverse", "2x", "Buffer", "Cryptocurrency" to disqualify).
3. `legacy_curated_allowlist_as_sole_authority_prohibited`: **`TRUE`**
   - Manual or legacy allowlists cannot serve as the sole classification authority without verifiable primary statutory filings.
4. `unevaluated_etf_defaults_to_other_etf`: **`FALSE`**
   - Unevaluated instruments may NEVER default to `OTHER_ETF`. They must fail closed into quarantine (`UNRESOLVED`).

---

### 2. SOURCE AUTHORITY HIERARCHY

| Tier | Tier Name | Regulatory Authority | Operational Scope |
| :---: | :--- | :--- | :--- |
| **1** | `VERIFIED_REGISTRY_SUBTYPE_OVERRIDE` | SEC Forms S-6, S-1, N-1A | Verified physical precious metal grantor trusts and verified statutory UIT index allowlists with immutable prospectus provenance. |
| **2** | `STRUCTURED_REGULATORY_PORTFOLIO_AND_MANDATES` | SEC EDGAR Forms N-PORT & N-CEN | Quarterly N-PORT holdings value weights (Treasury, agency, corporate debt, common equity, constituent counts) and Form N-CEN Item C.3.b Index Fund flags. |
| **3** | `STATUTORY_SERIES_INDEX_MANDATE_REGISTRATION` | SEC Form 485BPOS / 497K | Series-level principal investment strategy and designated benchmark index mandate filed under the 1940 Act. |
| **4** | `FAIL_CLOSED_UNRESOLVED_SUBTYPE_QUARANTINE` | `INSUFFICIENT_SOURCE_EVIDENCE` | Instruments lacking authoritative portfolio or mandate evidence remain `UNRESOLVED / PENDING_SYSTEMATIC_CLASSIFICATION`. |
| **5** | `EVALUATED_EXPLORATORY_ASSIGNMENT` | `AFFIRMATIVE_NON_CONFIRMATORY_EVIDENCE` | Instruments affirmatively proven to follow non-confirmatory mandates (multi-asset, active equity, covered call, thematic, crypto, leveraged/inverse, mixed aggregate bonds) receive `OTHER_ETF / EXPLORATORY_ONLY`. |

---

### 3. NORMATIVE SUBTYPE DECISION TABLE

```
PRIORITY PRECEDENCE:
1. COMMODITY_PHYSICAL
2. FIXED_INCOME_GOVERNMENT
3. FIXED_INCOME_CREDIT
4. EQUITY_SECTOR
5. EQUITY_INDEX
6. OTHER_ETF (Affirmative Non-Confirmatory)
7. UNRESOLVED (Fail-Closed Quarantine)
```

| Subtype | Required Legal Structure | Required Positive Evidence | Exclusion Conditions | Quantitative Dependencies | Regulatory Data Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`COMMODITY_PHYSICAL`** | `PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST` | • `legal_structure == PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST`<br>• `physical_precious_metal_pct >= 0.90` | • `COMMODITY_FUTURES_POOL`<br>• `EXCHANGE_TRADED_NOTE`<br>• `derivative_commodity_exposure > 0.05` | $\ge 90\%$ physical bullion weight | SEC Form S-1 / S-3 prospectus; Grantor Trust indenture |
| **`FIXED_INCOME_GOVERNMENT`** | `1940_ACT_OPEN_END_ETF` | • `total_govt_pct >= 0.80`<br>• `corporate_debt_pct < 0.10`<br>• `mortgage_backed_pct < 0.10`<br>• `total_equity_pct < 0.05`<br>• Qualifying issuers: US Treasury Bills/Notes/Bonds, TIPS, US Agencies | • `total_govt_pct < 0.80`<br>• `corporate_debt_pct >= 0.10`<br>• `mortgage_backed_pct >= 0.10` | $\ge 80\%$ Govt debt;<br>$< 10\%$ Corporate;<br>$< 10\%$ MBS;<br>$< 5\%$ Equity | SEC Form N-PORT Item Part C;<br>SEC Form 485BPOS Item 4 |
| **`FIXED_INCOME_CREDIT`** | `1940_ACT_OPEN_END_ETF` | • `corporate_debt_pct >= 0.50`<br>• `total_govt_pct < 0.50`<br>• `total_equity_pct < 0.05` | • `corporate_debt_pct < 0.50`<br>• `total_govt_pct >= 0.50`<br>• `total_equity_pct >= 0.05`<br>• Mixed aggregate bond funds ($\ge 50\%$ govt, $< 50\%$ credit) | $\ge 50\%$ Corporate debt;<br>$< 50\%$ Govt debt;<br>$< 5\%$ Equity | SEC Form N-PORT Item Part C;<br>SEC Form 485BPOS Item 4 |
| **`EQUITY_SECTOR`** | `1940_ACT_OPEN_END_ETF` | • `total_equity_pct >= 0.80`<br>• `is_sector_specific_mandate == true`<br>• `designated_sector_in_standard_taxonomy == true`<br>• Sector strictly in approved 11 GICS list | • `total_equity_pct < 0.80`<br>• `is_sector_specific_mandate == false`<br>• Non-standard/thematic sectors | $\ge 80\%$ Equity;<br>Single approved sector mandate | SEC Form N-PORT;<br>SEC Form 485BPOS Item 4 |
| **`EQUITY_INDEX`** | `1940_ACT_OPEN_END_ETF`,<br>`1940_ACT_UNIT_INVESTMENT_TRUST_ETF` | • `total_equity_pct >= 0.80`<br>• `is_index_fund == true`<br>• `distinct_holdings_count >= 30`<br>• `max_security_concentration < 0.15`<br>• `is_broad_or_multi_sector_mandate == true` | • `total_govt_pct >= 0.20`<br>• `total_credit_pct >= 0.20`<br>• `is_sector_specific_mandate == true`<br>• `is_index_fund == false` | $\ge 80\%$ Equity;<br>Holdings $\ge 30$;<br>Max position $< 15\%$ | SEC Form N-PORT;<br>SEC Form N-CEN Item C.3.b;<br>SEC Form 485BPOS Item 4 |
| **`NON_CONFIRMATORY`** | Any allowed vehicle structure | Affirmative evidence of active equity, quantitative tactical, covered call, fund-of-funds, crypto, leveraged, inverse, or mixed aggregate debt. | Valid confirmatory subtype evidence | N/A | Affirmative statutory strategy disclosure |
| **`AMBIGUOUS_MANDATE`** | Any | Insufficient statutory evidence, unresolvable multi-fund series boundary, or conflicting portfolio/mandate metrics. | Complete verifiable data satisfying tiers 1–3 | N/A | Fail-closed quarantine |

---

### 4. APPROVED EQUITY SECTORS (CLOSED TAXONOMY)

Under Policy V1.1 line 99–111, the approved sector list is closed and exhaustive:
1. `TECHNOLOGY`
2. `FINANCIALS`
3. `ENERGY`
4. `HEALTHCARE`
5. `INDUSTRIALS`
6. `MATERIALS`
7. `CONSUMER_STAPLES`
8. `CONSUMER_DISCRETIONARY`
9. `UTILITIES`
10. `REAL_ESTATE`
11. `COMMUNICATION_SERVICES`

*Preferred securities, clean energy, cyber security, genomics, robotics, and water are NOT approved sectors under Policy V1.1.*

---

### 5. SCIENTIFIC POLICY GAPS & RECONCILIATION

1. **Geography Authority**:
   - Policy V1.1 contains **NO** explicit clause excluding ex-US or international equity funds from `EQUITY_INDEX` or `EQUITY_SECTOR`.
   - However, the v1.0.3 specification benchmarks `EQUITY_INDEX` against `SPY` (US S&P 500) and `EQUITY_SECTOR` against `SPY`.
   - Applying `SPY` relative-strength and trend hypotheses to ex-US single-country funds (e.g. `ASHR` China, `EWJ` Japan, `VGK` Europe) violates the asset pricing assumptions of the pre-registered model. A formal policy amendment is required to explicitly codify whether `EQUITY_INDEX` is restricted to US domestic equity.
2. **Quantitative Portfolio Holdings Dependencies**:
   - Policy V1.1 mandates N-PORT quantitative thresholds (`distinct_holdings_count >= 30`, `max_security_concentration < 0.15`, `corporate_debt_pct >= 0.50`, etc.).
   - A prospectus-only extraction pipeline cannot evaluate N-PORT holdings. A formal policy amendment or an operational pipeline extension ingesting N-PORT is required to bridge this contract.
