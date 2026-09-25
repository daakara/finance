import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

from scripts.research.analyze_target_nport import (
    CACHE_DIR,
    fetch_nasdaq_traded_directory,
    load_sec_mf_directory,
    ClassificationAuthorityEngine
)

def run():
    # 1. Load Discovery and MF directory
    df_disc, _ = fetch_nasdaq_traded_directory(CACHE_DIR)
    test_issue_col = 'Test Issue' if 'Test Issue' in df_disc.columns else 'TestIssue'
    df_clean = df_disc[df_disc[test_issue_col] == 'N'] if test_issue_col in df_disc.columns else df_disc
    etf_col = 'ETF' if 'ETF' in df_clean.columns else 'etf'
    df_etfs = df_clean[df_clean[etf_col].astype(str).str.strip().str.upper() == 'Y'].copy()
    sec_mf_map = load_sec_mf_directory(CACHE_DIR)

    # 2. Load Portfolio Metrics from derived N-PORT cache
    df_metrics = pd.read_parquet('data/research/cache/nport_derived/portfolio_metrics.parquet')
    metrics_by_series = {}
    for _, r in df_metrics.iterrows():
        s_id = r['SERIES_ID']
        if pd.notna(s_id):
            metrics_by_series[s_id] = r

    # 3. Load or build Mandate Evidence database
    # Sector symbols from approved standard taxonomy
    sector_mandates = {
        'XLE': 'ENERGY', 'XOP': 'ENERGY',
        'XLF': 'FINANCIALS', 'KRE': 'FINANCIALS', 'KBE': 'FINANCIALS',
        'XLK': 'TECHNOLOGY', 'SMH': 'TECHNOLOGY',
        'XLV': 'HEALTHCARE', 'XBI': 'HEALTHCARE', 'IBB': 'HEALTHCARE',
        'XLI': 'INDUSTRIALS',
        'XLP': 'CONSUMER_STAPLES',
        'XLU': 'UTILITIES',
        'XLY': 'CONSUMER_DISCRETIONARY', 'ITB': 'CONSUMER_DISCRETIONARY', 'XHB': 'CONSUMER_DISCRETIONARY',
        'XLB': 'MATERIALS',
        'VNQ': 'REAL_ESTATE', 'IYR': 'REAL_ESTATE',
    }

    broad_index_symbols = {
        'SPY', 'DIA', 'QQQ', 'IWM', 'VOO', 'IVV', 'VTI', 'SCHX', 'RSP', 'IJH', 'IJR', 'VB', 'VO',
        'SCHD', 'VUG', 'VTV', 'IEFA', 'IEMG'
    }

    govt_symbols = {
        'TLT', 'IEF', 'SHY', 'IEI', 'GOVT', 'VGSH', 'VGIT', 'VGLT', 'SCHO', 'SCHR', 'SPTL',
        'BIL', 'TIP', 'SHV'
    }

    credit_symbols = {
        'HYG', 'LQD', 'JNK', 'VCIT', 'VCSH', 'USIG', 'FLOT', 'SJNK', 'HYLB', 'VUSB'
    }

    physical_bullion_symbols = {
        'AAAU', 'GLD', 'IAU', 'OUNZ', 'PALL', 'PPLT', 'SGOL', 'SIVR', 'SLV'
    }

    # Evaluate all discovered ETFs
    classified_records = []
    
    for _, row in df_etfs.iterrows():
        sym = row.get('Symbol', '')
        name = row.get('Security Name', '')
        exch = row.get('Listing Exchange', '')
        sec_info = sec_mf_map.get(str(sym).strip().upper())
        
        # Stage 1: Legal structure
        rec = ClassificationAuthorityEngine.classify_security(
            symbol=sym,
            security_name=name,
            listing_exchange=exch,
            nasdaq_etf_flag=True,
            sec_mf_info=sec_info
        )
        
        # Systematic Stage 2 execution under Policy v1.1
        if rec['structure_verified']:
            structure = rec['vehicle_structure']
            s_id = sec_info.get('series_id') if sec_info else None
            
            # Tier 1 Overrides (S-1 Grantor Trusts & S-6 UITs)
            if structure == 'PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST':
                rec['research_subtype'] = 'COMMODITY_PHYSICAL'
                rec['research_subtype_state'] = 'CONFIRMATORY_SUPPORTED'
                rec['subtype_authorized'] = True
                rec['classification_source'] = 'TIER_1_VERIFIED_REGISTRY_SUBTYPE_OVERRIDE'
                rec['classification_evidence'] = 'SEC_FORM_S1_S3_PHYSICAL_BULLION_GRANTOR_TRUST_PROSPECTUS'
                rec['exclusion_reason'] = None
            elif structure == '1940_ACT_UNIT_INVESTMENT_TRUST_ETF':
                rec['research_subtype'] = 'EQUITY_INDEX'
                rec['research_subtype_state'] = 'CONFIRMATORY_SUPPORTED'
                rec['subtype_authorized'] = True
                rec['classification_source'] = 'TIER_1_VERIFIED_REGISTRY_SUBTYPE_OVERRIDE'
                rec['classification_evidence'] = 'SEC_FORM_S6_UNIT_INVESTMENT_TRUST_INDEX_PROSPECTUS'
                rec['exclusion_reason'] = None
            else:
                # 1940 Act Open-End ETF
                p_metric = metrics_by_series.get(s_id)
                if p_metric is None:
                    # Missing N-PORT -> Quarantined UNRESOLVED
                    rec['research_subtype'] = 'UNRESOLVED'
                    rec['research_subtype_state'] = 'PENDING_SYSTEMATIC_CLASSIFICATION'
                    rec['subtype_authorized'] = False
                    rec['classification_source'] = 'TIER_4_FAIL_CLOSED_UNRESOLVED_SUBTYPE_QUARANTINE'
                    rec['classification_evidence'] = 'INSUFFICIENT_NPORT_REGULATORY_EVIDENCE'
                    rec['exclusion_reason'] = 'UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION'
                else:
                    # Check portfolio reconciliation tolerance (within 15% residual)
                    rec_ratio = p_metric['reconciliation_ratio']
                    if pd.isna(rec_ratio) or rec_ratio < 0.85 or rec_ratio > 1.15:
                        # Failed reconciliation
                        rec['research_subtype'] = 'UNRESOLVED'
                        rec['research_subtype_state'] = 'PENDING_SYSTEMATIC_CLASSIFICATION'
                        rec['subtype_authorized'] = False
                        rec['classification_source'] = 'TIER_4_FAIL_CLOSED_UNRESOLVED_SUBTYPE_QUARANTINE'
                        rec['classification_evidence'] = f'PORTFOLIO_RECONCILIATION_FAIL_RATIO_{rec_ratio:.2f}'
                        rec['exclusion_reason'] = 'UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION'
                    else:
                        # Evaluate quantitative criteria
                        eq_pct = p_metric['total_equity_pct']
                        gov_pct = p_metric['total_govt_pct']
                        corp_pct = p_metric['corporate_debt_pct']
                        mbs_pct = p_metric['mortgage_backed_pct']
                        distinct_eq = p_metric['distinct_equity_count']
                        max_conc = p_metric['max_concentration']
                        is_index_ncen = bool(p_metric['is_index_ncen'])

                        # Check confirmatory rules
                        passes_gov = (gov_pct >= 0.80 and corp_pct < 0.10 and mbs_pct < 0.10 and eq_pct < 0.05)
                        passes_credit = (corp_pct >= 0.50 and gov_pct < 0.50 and eq_pct < 0.05)
                        
                        is_sector_fund = (sym in sector_mandates)
                        is_broad_index = (sym in broad_index_symbols)
                        
                        passes_sector = (eq_pct >= 0.80 and is_sector_fund)
                        passes_equity_index = (
                            eq_pct >= 0.80 and
                            is_index_ncen and
                            distinct_eq >= 30 and
                            max_conc < 0.15 and
                            gov_pct < 0.20 and
                            corp_pct < 0.20 and
                            not is_sector_fund and
                            is_broad_index
                        )

                        # Ambiguity Precedence Hierarchy:
                        # COMMODITY_PHYSICAL > FIXED_INCOME_GOVERNMENT > FIXED_INCOME_CREDIT > EQUITY_SECTOR > EQUITY_INDEX > OTHER_ETF
                        if passes_gov and (sym in govt_symbols):
                            rec['research_subtype'] = 'FIXED_INCOME_GOVERNMENT'
                            rec['research_subtype_state'] = 'CONFIRMATORY_SUPPORTED'
                            rec['subtype_authorized'] = True
                            rec['classification_source'] = 'TIER_2_STRUCTURED_REGULATORY_PORTFOLIO_AND_MANDATES'
                            rec['classification_evidence'] = f'SEC_FORM_NPORT_GOVT_{gov_pct:.1%}_CORP_{corp_pct:.1%}_MBS_{mbs_pct:.1%}'
                            rec['exclusion_reason'] = None
                        elif passes_credit and (sym in credit_symbols):
                            rec['research_subtype'] = 'FIXED_INCOME_CREDIT'
                            rec['research_subtype_state'] = 'CONFIRMATORY_SUPPORTED'
                            rec['subtype_authorized'] = True
                            rec['classification_source'] = 'TIER_2_STRUCTURED_REGULATORY_PORTFOLIO_AND_MANDATES'
                            rec['classification_evidence'] = f'SEC_FORM_NPORT_CREDIT_{corp_pct:.1%}_GOVT_{gov_pct:.1%}'
                            rec['exclusion_reason'] = None
                        elif passes_sector:
                            sec_name = sector_mandates[sym]
                            rec['research_subtype'] = 'EQUITY_SECTOR'
                            rec['research_subtype_state'] = 'CONFIRMATORY_SUPPORTED'
                            rec['subtype_authorized'] = True
                            rec['classification_source'] = 'TIER_2_STRUCTURED_REGULATORY_PORTFOLIO_AND_MANDATES'
                            rec['classification_evidence'] = f'SEC_FORM_NPORT_EQUITY_{eq_pct:.1%}_SECTOR_{sec_name}'
                            rec['exclusion_reason'] = None
                        elif passes_equity_index:
                            rec['research_subtype'] = 'EQUITY_INDEX'
                            rec['research_subtype_state'] = 'CONFIRMATORY_SUPPORTED'
                            rec['subtype_authorized'] = True
                            rec['classification_source'] = 'TIER_2_STRUCTURED_REGULATORY_PORTFOLIO_AND_MANDATES'
                            rec['classification_evidence'] = f'SEC_FORM_NPORT_EQUITY_{eq_pct:.1%}_HOLDINGS_{distinct_eq}_INDEX_NCEN'
                            rec['exclusion_reason'] = None
                        else:
                            # Affirmatively evaluated exploratory assignment
                            rec['research_subtype'] = 'OTHER_ETF'
                            rec['research_subtype_state'] = 'EXPLORATORY_ONLY'
                            rec['subtype_authorized'] = False
                            rec['classification_source'] = 'TIER_5_EVALUATED_EXPLORATORY_ASSIGNMENT'
                            rec['classification_evidence'] = f'AFFIRMATIVE_NON_CONFIRMATORY_PORTFOLIO: EQ_{eq_pct:.1%}_GOV_{gov_pct:.1%}_CORP_{corp_pct:.1%}'
                            rec['exclusion_reason'] = 'UNAUTHORIZED_RESEARCH_SUBTYPE'

        classified_records.append(rec)

    df_classified = pd.DataFrame(classified_records)
    print(f'Total Discovered ETFs: {len(df_classified)}')
    print('\nVehicle Structure State:')
    print(df_classified['vehicle_structure_state'].value_counts())
    
    struct_elig = df_classified[df_classified['structure_verified']]
    print(f'\nStructure Eligible: {len(struct_elig)}')
    print('\nResearch Subtype Census for Structure-Eligible Population:')
    print(struct_elig['research_subtype'].value_counts(dropna=False))
    
    print('\nResearch Subtype State Census:')
    print(struct_elig['research_subtype_state'].value_counts(dropna=False))
    
    conf_cands = struct_elig[struct_elig['subtype_authorized']]
    print(f'\nFinal Confirmatory Candidate Count: {len(conf_cands)}')
    print('Breakdown of Candidates by Subtype:')
    print(conf_cands['research_subtype'].value_counts())

if __name__ == '__main__':
    run()
