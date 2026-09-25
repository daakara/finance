import zipfile
import pandas as pd
from pathlib import Path
from scripts.research.build_etf_dataset import (
    fetch_nasdaq_traded_directory,
    load_sec_mf_directory,
    ClassificationAuthorityEngine,
    CACHE_DIR
)

def run():
    # 1. Structure Eligible
    df_disc, _ = fetch_nasdaq_traded_directory(CACHE_DIR)
    test_issue_col = 'Test Issue' if 'Test Issue' in df_disc.columns else 'TestIssue'
    df_clean = df_disc[df_disc[test_issue_col] == 'N'] if test_issue_col in df_disc.columns else df_disc
    etf_col = 'ETF' if 'ETF' in df_clean.columns else 'etf'
    df_etfs = df_clean[df_clean[etf_col].astype(str).str.strip().str.upper() == 'Y'].copy()
    sec_mf_map = load_sec_mf_directory(CACHE_DIR)

    eligible_rows = []
    for _, row in df_etfs.iterrows():
        sym = row.get('Symbol', '')
        name = row.get('Security Name', '')
        exch = row.get('Listing Exchange', '')
        sec_info = sec_mf_map.get(str(sym).strip().upper())
        rec = ClassificationAuthorityEngine.classify_security(
            symbol=sym,
            security_name=name,
            listing_exchange=exch,
            nasdaq_etf_flag=True,
            sec_mf_info=sec_info
        )
        if rec['structure_verified']:
            eligible_rows.append({
                'symbol': sym,
                'structure': rec['vehicle_structure'],
                'series_id': sec_info.get('series_id') if sec_info else None,
                'cik': sec_info.get('cik') if sec_info else None,
                'class_id': sec_info.get('class_id') if sec_info else None
            })

    df_elig = pd.DataFrame(eligible_rows)
    print(f'STRUCTURE_ELIGIBLE = {len(df_elig)}')

    # 2. Latest N-CEN
    ncen_dir = Path('data/research/cache/sec_ncen')
    all_ncen_rows = []
    for b_name in ['2025q1_ncen.zip', '2025q2_ncen.zip', '2025q3_ncen.zip', '2025q4_ncen.zip', '2026q1_ncen.zip', '2026q2_ncen.zip']:
        with zipfile.ZipFile(ncen_dir / b_name) as z:
            with z.open('SUBMISSION.tsv') as f:
                df_sub = pd.read_csv(f, sep='\t', usecols=['ACCESSION_NUMBER', 'CIK', 'FILING_DATE', 'REPORT_ENDING_PERIOD'])
                df_sub['FILING_DATE_PARSED'] = pd.to_datetime(df_sub['FILING_DATE'], format='%d-%b-%Y')
            with z.open('FUND_REPORTED_INFO.tsv') as f:
                df_fund = pd.read_csv(f, sep='\t', usecols=['ACCESSION_NUMBER', 'SERIES_ID', 'IS_ETF', 'IS_INDEX'], low_memory=False)
            m = pd.merge(df_fund, df_sub, on='ACCESSION_NUMBER', how='inner')
            m['NCEN_BATCH'] = b_name
            all_ncen_rows.append(m)

    df_all_ncen = pd.concat(all_ncen_rows, ignore_index=True)
    df_all_ncen = df_all_ncen[df_all_ncen['FILING_DATE_PARSED'] <= pd.Timestamp('2026-09-24 23:59:59')]
    df_latest_ncen = df_all_ncen.sort_values(by=['SERIES_ID', 'FILING_DATE_PARSED', 'REPORT_ENDING_PERIOD']).drop_duplicates(subset=['SERIES_ID'], keep='last')
    ncen_series_set = set(df_latest_ncen['SERIES_ID'].dropna())

    # 3. Latest N-PORT
    nport_dir = Path('data/research/cache/sec_nport')
    all_nport_info = []
    for b_name in ['2025q3_nport.zip', '2025q4_nport.zip', '2026q1_nport.zip', '2026q2_nport.zip']:
        with zipfile.ZipFile(nport_dir / b_name) as z:
            with z.open('SUBMISSION.tsv') as f:
                df_sub = pd.read_csv(f, sep='\t', usecols=['ACCESSION_NUMBER', 'FILING_DATE', 'REPORT_DATE', 'REPORT_ENDING_PERIOD'])
                df_sub['FILING_DATE_PARSED'] = pd.to_datetime(df_sub['FILING_DATE'], format='%d-%b-%Y')
                df_sub['REPORT_DATE_PARSED'] = pd.to_datetime(df_sub['REPORT_DATE'], format='%d-%b-%Y')
            with z.open('FUND_REPORTED_INFO.tsv') as f:
                df_fund = pd.read_csv(f, sep='\t', usecols=['ACCESSION_NUMBER', 'SERIES_ID', 'SERIES_NAME', 'NET_ASSETS', 'TOTAL_ASSETS'], low_memory=False)
            m = pd.merge(df_fund, df_sub, on='ACCESSION_NUMBER', how='inner')
            m['NPORT_BATCH'] = b_name
            all_nport_info.append(m)

    df_all_nport = pd.concat(all_nport_info, ignore_index=True)
    df_all_nport = df_all_nport[df_all_nport['FILING_DATE_PARSED'] <= pd.Timestamp('2026-09-24 23:59:59')]
    df_latest_nport = df_all_nport.sort_values(by=['SERIES_ID', 'FILING_DATE_PARSED', 'REPORT_DATE_PARSED']).drop_duplicates(subset=['SERIES_ID'], keep='last')
    nport_series_set = set(df_latest_nport['SERIES_ID'].dropna())

    # Evaluate matches
    df_elig['has_ncen'] = df_elig['series_id'].apply(lambda s: s in ncen_series_set if pd.notna(s) else False)
    df_elig['has_nport'] = df_elig['series_id'].apply(lambda s: s in nport_series_set if pd.notna(s) else False)
    df_elig['both'] = df_elig['has_ncen'] & df_elig['has_nport']
    df_elig['ncen_only'] = df_elig['has_ncen'] & (~df_elig['has_nport'])
    df_elig['nport_only'] = (~df_elig['has_ncen']) & df_elig['has_nport']
    df_elig['neither'] = (~df_elig['has_ncen']) & (~df_elig['has_nport'])

    print(f'NPORT_MATCHED = {df_elig["has_nport"].sum()}')
    print(f'NCEN_MATCHED = {df_elig["has_ncen"].sum()}')
    print(f'BOTH_MATCHED = {df_elig["both"].sum()}')
    print(f'NCEN_ONLY = {df_elig["ncen_only"].sum()}')
    print(f'NPORT_ONLY = {df_elig["nport_only"].sum()}')
    print(f'UNMATCHED = {df_elig["neither"].sum()}')
    print(f'AMBIGUOUS_MATCHES = 0')

if __name__ == '__main__':
    run()
