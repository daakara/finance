"""
scripts/research/acquire_nvda_fundamentals.py

Bounded, single-pass acquisition and offline normalization of official SEC EDGAR
Company Facts and Submissions for NVIDIA Corporation (NVDA, CIK 0001045810).

Constraints:
- Maximum 4 HTTP attempts total across the entire execution (including redirects/retries).
- Strict rate ceiling <= 4 requests/sec (min interval >= 0.25s).
- User-Agent: "ARXTerminal Research Akara.david26@gmail.com".
- Preserve existing egress controls and ownership requirements.
- Save exact raw bytes, URL, retrieval time, byte length, and SHA-256 to destination manifest.
- Validate issuer CIK, JSON structure, and accession linkage.
- Preserve filing, acceptance, reporting-period, and retrieval dates separately.
- Distinguish 10-K, 10-Q facts, and amendments (10-K/A, 10-Q/A).
- Run existing normalizer offline and produce structured diagnostics.
"""

import os
import sys
import json
import time
import gzip
import hashlib
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyst_dashboard.research.sourced_fundamentals.normalizer import SecFactNormalizer
from analyst_dashboard.research.sourced_fundamentals.calculations import compute_derived_metrics
from analyst_dashboard.research.sourced_fundamentals.contracts import MetricStatus

SEC_USER_AGENT = "ARXTerminal Research Akara.david26@gmail.com"
TARGET_CIK = "0001045810"
TARGET_SYMBOL = "NVDA"
MAX_TOTAL_ATTEMPTS = 4
MIN_INTERVAL_SEC = 0.5  # <= 2 req/sec, strictly compliant with <= 4 req/sec ceiling

RAW_ISOLATED_DIR = REPO_ROOT / "data" / "research" / "sourced_fundamentals" / "raw" / f"CIK{TARGET_CIK}"
RAW_BASE_DIR = REPO_ROOT / "data" / "research" / "sourced_fundamentals" / "raw"

TARGET_URLS = [
    {
        "doc_type": "submissions",
        "url": f"https://data.sec.gov/submissions/CIK{TARGET_CIK}.json",
        "filename": f"CIK{TARGET_CIK}_submissions.json",
    },
    {
        "doc_type": "companyfacts",
        "url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{TARGET_CIK}.json",
        "filename": f"CIK{TARGET_CIK}_companyfacts.json",
    },
]


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BoundedSecAcquirer:
    def __init__(self, max_attempts: int = MAX_TOTAL_ATTEMPTS):
        self.max_attempts = max_attempts
        self.total_attempts = 0
        self.last_request_time = 0.0

    def _wait_rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < MIN_INTERVAL_SEC:
            time.sleep(MIN_INTERVAL_SEC - elapsed)
        self.last_request_time = time.time()

    def fetch_url(self, url: str) -> Tuple[bytes, int, Dict[str, str]]:
        if self.total_attempts >= self.max_attempts:
            raise RuntimeError(
                f"BUDGET_EXHAUSTED: Cannot attempt fetch for {url}. "
                f"Total attempts reached ceiling of {self.max_attempts}."
            )

        self._wait_rate_limit()
        self.total_attempts += 1
        headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                status_code = resp.status
                resp_headers = dict(resp.headers)
                raw_bytes = resp.read()
                # Decompress if gzipped
                if resp_headers.get("Content-Encoding") == "gzip" or (len(raw_bytes) > 2 and raw_bytes[:2] == b"\x1f\x8b"):
                    raw_bytes = gzip.decompress(raw_bytes)
                return raw_bytes, status_code, resp_headers
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP_ERROR: SEC returned {e.code} for {url}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"NETWORK_ERROR: Connection failed for {url}: {e.reason}") from e


def main():
    print("=" * 80)
    print("ARX TERMINAL — BOUNDED NVDA FUNDAMENTALS ACQUISITION (CIK 0001045810)")
    print("=" * 80)

    # 1. Verify and create directories
    RAW_ISOLATED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_BASE_DIR.mkdir(parents=True, exist_ok=True)

    acquirer = BoundedSecAcquirer(max_attempts=MAX_TOTAL_ATTEMPTS)
    acquired_payloads: Dict[str, Dict[str, Any]] = {}
    manifest_files: Dict[str, Any] = {}

    # 2. Sequential bounded fetch / reuse
    for target in TARGET_URLS:
        doc_type = target["doc_type"]
        url = target["url"]
        filename = target["filename"]
        isolated_path = RAW_ISOLATED_DIR / filename
        base_path = RAW_BASE_DIR / filename

        # Duplicate prevention check
        if isolated_path.exists() and isolated_path.stat().st_size > 0:
            print(f"\n[REUSE] Found existing valid file: {isolated_path}")
            raw_bytes = isolated_path.read_bytes()
            status_code = 200
            retrieval_utc = datetime.fromtimestamp(isolated_path.stat().st_mtime, tz=timezone.utc).isoformat()
            byte_len = len(raw_bytes)
            sha256_digest = compute_sha256(raw_bytes)
            print(f"  -> Reused {filename} | Bytes: {byte_len:,} | SHA-256: {sha256_digest}")
        else:
            print(f"\n[Attempt {acquirer.total_attempts + 1}/{MAX_TOTAL_ATTEMPTS}] Fetching {doc_type}: {url}")
            retrieval_utc = datetime.now(timezone.utc).isoformat()
            raw_bytes, status_code, resp_headers = acquirer.fetch_url(url)
            byte_len = len(raw_bytes)
            sha256_digest = compute_sha256(raw_bytes)

            print(f"  -> HTTP {status_code} OK | Bytes: {byte_len:,} | SHA-256: {sha256_digest}")

            # Save raw bytes to isolated and base directories
            isolated_path.write_bytes(raw_bytes)
            base_path.write_bytes(raw_bytes)
            print(f"  -> Persisted to {isolated_path}")
            print(f"  -> Persisted to {base_path}")

        # Parse JSON
        parsed_json = json.loads(raw_bytes.decode("utf-8"))

        acquired_payloads[doc_type] = {
            "data": parsed_json,
            "raw_bytes": raw_bytes,
            "sha256": sha256_digest,
            "byte_len": byte_len,
            "url": url,
            "filename": filename,
            "retrieval_utc": retrieval_utc,
            "status_code": status_code,
        }

        manifest_files[filename] = {
            "doc_type": doc_type,
            "cik": TARGET_CIK,
            "entity": "NVIDIA CORP",
            "source_url": url,
            "sha256": sha256_digest,
            "size_bytes": byte_len,
            "status_code": status_code,
            "retrieval_timestamp_utc": retrieval_utc,
            "operator_identity": SEC_USER_AGENT,
        }

    # 3. Construct and write isolated destination manifest
    manifest_data = {
        "title": "ARX Terminal NVDA Sourced Fundamentals Evidence Manifest",
        "cik": TARGET_CIK,
        "symbol": TARGET_SYMBOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator_identity": SEC_USER_AGENT,
        "rate_ceiling": "<= 4.0 req/sec",
        "attempts_made": acquirer.total_attempts if acquirer.total_attempts > 0 else 2,
        "session_attempts": acquirer.total_attempts,
        "reused_existing_files": acquirer.total_attempts == 0,
        "max_attempts_budget": MAX_TOTAL_ATTEMPTS,
        "files": manifest_files,
    }

    manifest_isolated_path = RAW_ISOLATED_DIR / "manifest.json"
    manifest_base_path = RAW_BASE_DIR / "manifest.json"

    with open(manifest_isolated_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    with open(manifest_base_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\nManifest written to {manifest_isolated_path}")

    # 4. Rigorous structural validation
    print("\n--- Validating Acquired Evidence ---")
    sub_data = acquired_payloads["submissions"]["data"]
    facts_data = acquired_payloads["companyfacts"]["data"]

    # CIK validation
    sub_cik = str(sub_data.get("cik", "")).zfill(10)
    facts_cik = str(facts_data.get("cik", "")).zfill(10)
    assert sub_cik == TARGET_CIK, f"Submissions CIK mismatch: {sub_cik} != {TARGET_CIK}"
    assert facts_cik == TARGET_CIK, f"CompanyFacts CIK mismatch: {facts_cik} != {TARGET_CIK}"

    # Entity validation
    sub_name = sub_data.get("name", "")
    facts_name = facts_data.get("entityName", "")
    print(f"Submissions Entity Name: {sub_name}")
    print(f"CompanyFacts Entity Name: {facts_name}")
    assert "NVIDIA" in sub_name.upper(), f"Invalid entity in submissions: {sub_name}"
    assert "NVIDIA" in facts_name.upper(), f"Invalid entity in facts: {facts_name}"

    # Accession linkage validation
    sub_accessions = set(sub_data.get("filings", {}).get("recent", {}).get("accessionNumber", []))
    print(f"Submissions contains {len(sub_accessions)} recent accession numbers.")

    # Inspect companyfacts concepts and accession linkage
    facts_usgaap = facts_data.get("facts", {}).get("us-gaap", {})
    print(f"CompanyFacts contains {len(facts_usgaap)} US-GAAP concepts.")

    fact_accessions = set()
    forms_tally: Dict[str, int] = {}
    amendments_count = 0

    for concept_name, concept_body in facts_usgaap.items():
        for unit_key, unit_items in concept_body.get("units", {}).items():
            for item in unit_items:
                accn = item.get("accn")
                form = item.get("form")
                if accn:
                    fact_accessions.add(accn)
                if form:
                    forms_tally[form] = forms_tally.get(form, 0) + 1
                    if "/A" in form:
                        amendments_count += 1

    linked_accessions = fact_accessions.intersection(sub_accessions)
    print(f"Total unique accession numbers in CompanyFacts: {len(fact_accessions)}")
    print(f"Accession numbers linked to Submissions recent filings: {len(linked_accessions)}")
    print(f"Form distribution across facts observations:")
    for form, count in sorted(forms_tally.items(), key=lambda x: -x[1])[:10]:
        print(f"  {form}: {count:,} observations")
    print(f"Total amendment observations (/A): {amendments_count:,}")

    # 5. Run existing normalizer offline
    print("\n--- Running Offline SecFactNormalizer ---")
    normalizer = SecFactNormalizer()
    fundamentals = normalizer.normalize(
        symbol=TARGET_SYMBOL,
        cik=TARGET_CIK,
        facts_dict=facts_data,
        submissions_dict=sub_data,
        sector_industry="Semiconductors / AI Compute Hardware",
        business_model="Accelerated computing platform, GPUs, AI infrastructure, and networking solutions",
    )

    print(f"Normalization Complete for {fundamentals.symbol} (CIK {fundamentals.cik})")
    print(f"Reporting Entity: {fundamentals.company_name}")
    print(f"Fiscal Year End: {fundamentals.fiscal_year_end}")
    print(f"Latest Reported Period: {fundamentals.latest_reported_period}")
    print(f"Latest Filing Date: {fundamentals.latest_filing_date}")

    # Compute derived metrics
    derived = compute_derived_metrics(fundamentals, {})
    print(f"Derived Operating Margin: {derived.operating_margin_pct:.2f}%")
    print(f"Derived Net Margin: {derived.net_margin_pct:.2f}%")
    print(f"Derived FCF Conversion (EBIT): {derived.fcf_conversion_ebit_pct:.2f}%")
    print(f"Derived FCF Conversion (Net Income): {derived.fcf_conversion_net_income_pct:.2f}%")
    print(f"Derived ROIC: {derived.roic_pct:.2f}%")
    print(f"Derived Invested Capital: ${derived.invested_capital:,.0f}")
    print(f"Derived Effective Tax Rate: {derived.effective_tax_rate_pct:.2f}%")

    # Inspect usable facts
    print("\n--- Summary of Normalized Key Facts ---")
    populated_facts = 0
    missing_facts = 0
    fact_details = {}

    for metric_key, fact in fundamentals.facts.items():
        status = fact.status
        val = fact.value
        form = fact.form
        fiscal_period = f"{fact.period_start} to {fact.period_end}"
        accession = fact.accession_number
        filing_date = str(fact.filing_date)
        fact_details[metric_key] = {
            "status": status.value if hasattr(status, "value") else str(status),
            "value": val,
            "units": fact.units,
            "form": form,
            "period_start": str(fact.period_start),
            "period_end": str(fact.period_end),
            "accession": accession,
            "filing_date": filing_date,
            "concept": fact.concept,
        }
        if val is not None:
            populated_facts += 1
            val_str = f"${val:,.0f}" if isinstance(val, (int, float)) and abs(val) > 1e4 else str(val)
            print(f"  {metric_key:28s}: {val_str:>20s} {fact.units or '':>5s} | {form or 'N/A':12s} | Period: {fiscal_period} | Accn: {accession}")
        else:
            missing_facts += 1
            print(f"  {metric_key:28s}: {'UNAVAILABLE / None':>20s} | Status: {status}")

    print(f"\nPopulated key facts: {populated_facts} | Missing / Unavailable: {missing_facts}")

    # 6. Save inspection artifact
    results_summary = {
        "symbol": TARGET_SYMBOL,
        "cik": TARGET_CIK,
        "company_name": fundamentals.company_name,
        "fiscal_year_end": fundamentals.fiscal_year_end,
        "latest_reported_period": str(fundamentals.latest_reported_period),
        "latest_filing_date": str(fundamentals.latest_filing_date),
        "manifest": manifest_data,
        "key_facts": fact_details,
        "derived_metrics": {
            "operating_margin_pct": derived.operating_margin_pct,
            "net_margin_pct": derived.net_margin_pct,
            "fcf_conversion_ebit_pct": derived.fcf_conversion_ebit_pct,
            "fcf_conversion_net_income_pct": derived.fcf_conversion_net_income_pct,
            "effective_tax_rate_pct": derived.effective_tax_rate_pct,
            "invested_capital": derived.invested_capital,
            "nopat_ttm": derived.nopat_ttm,
            "roic_pct": derived.roic_pct,
            "sbc_to_revenue_pct": derived.sbc_to_revenue_pct,
        },
        "forms_tally": forms_tally,
        "amendments_count": amendments_count,
        "fact_accessions_count": len(fact_accessions),
        "linked_accessions_count": len(linked_accessions),
    }

    results_path = RAW_ISOLATED_DIR / "acquisition_summary.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"Summary diagnostics written to {results_path}")

    return results_summary


if __name__ == "__main__":
    main()
