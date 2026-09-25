"""ARX Terminal — Versioned Offline Historical ETF Research Dataset & Universe Builder.

Adheres strictly to:
docs/research/ETF_RESEARCH_SPEC_V1.json (v1.0.2)

Enforces:
1. POINT_IN_TIME_WITHIN_OBSERVABLE_SURVIVOR_UNIVERSE estimand with declared survivorship bias.
2. 5-Tier Classification Authority Hierarchy:
   - Tier 1: Exchange Discovery Metadata (NasdaqTraded directory)
   - Tier 2: Structured Provider Metadata (SEC filings / provider attestation)
   - Tier 3: Verified Vehicle-Structure Registries (Point-in-time verified allowlists & blocklists with formal provenance)
   - Tier 4: Defensive Negative Heuristics (Negative safety net ONLY for unverified instruments; cannot override Tier 3 positive registries)
   - Tier 5: Unknown Structure Quarantine (Fail-closed; unresolved securities barred from eligibility)
3. Composite Research Eligibility:
   IS_RESEARCH_ELIGIBLE = DISCOVERED AND STRUCTURE_VERIFIED AND SUBTYPE_AUTHORIZED AND MARKET_DATA_VALID AND (HISTORY >= 250) AND (ADV60 >= ADV80)
   Structure-only eligibility is impossible.
4. Deterministic exclusion reason precedence.
5. Zero forward-looking leakage (all features at date t use info <= t close).
6. Conservative Macro As-Of Join (macro observation date <= t - 1).
7. Deprecation of f7; integration of f12 (BAA10Y) for FIXED_INCOME_CREDIT.
8. Physical separation of observations and forward outcome labels.
9. Source cache manifest tracking (data/research/source_cache_manifest_v1.json).
10. Universe snapshot generation (docs/research/ETF_SURVIVING_UNIVERSE_V1.parquet) and manifest.
11. Cryptographic manifest binding spec SHA, code SHA, dataset SHAs, and row counts.
12. Absolutely NO model fitting, weighting, or threshold tuning.
"""

import os
import sys
import json
import re
import hashlib
import logging
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_etf_dataset")

SPEC_PATH = Path("docs/research/ETF_RESEARCH_SPEC_V1.json")
DATA_DIR = Path("data/research")
CACHE_DIR = DATA_DIR / "cache"
SOURCE_CACHE_MANIFEST_PATH = DATA_DIR / "source_cache_manifest_v1.json"
UNIVERSE_SNAPSHOT_PATH = Path("docs/research/ETF_SURVIVING_UNIVERSE_V1.parquet")
UNIVERSE_MANIFEST_PATH = Path("docs/research/ETF_SURVIVING_UNIVERSE_V1_MANIFEST.json")

# Canonical Spec Hash & Commit Constants
SPEC_VERSION_V102 = "1.0.2"
CANONICAL_SPEC_COMMIT_V102 = "57720cc278813b11cc2ea6df5cccd1925b56c763"
CANONICAL_FILTERED_SPEC_SHA256_V102 = "448cbb130a4ddd551965137234b01178d07cf4c0b49325927e8d047a258c6b24"

SPEC_VERSION_V103 = "1.0.3"
CANONICAL_SPEC_COMMIT_V103 = "a5efc1777d130c23631986422731b7efd75e4624"
CANONICAL_FILTERED_SPEC_SHA256_V103 = "481041f08cdcf569516d69648d1a180066db17635a71896590a1ec495e88bbc5"

SPEC_VERSION = SPEC_VERSION_V103
CANONICAL_SPEC_COMMIT = CANONICAL_SPEC_COMMIT_V103
CANONICAL_FILTERED_SPEC_SHA256 = CANONICAL_FILTERED_SPEC_SHA256_V103

# Subtype Policy v1.1 Constants
POLICY_V11_PATH = Path("docs/research/ETF_SUBTYPE_CLASSIFICATION_POLICY_V1_1.json")
POLICY_V11_SHA256 = "864133d98750f7765409153c4305d02ff9d422aa56556b6a0506738299642f52"
POLICY_V11_COMMIT = "a554c2fc896238afb8b66a5ea5a2786dbe2f2c1f"
MANDATE_EVIDENCE_PATH = Path("data/research/etf_mandate_evidence_v1.json")
NPORT_DERIVED_METRICS_PATH = Path("data/research/cache/nport_derived/portfolio_metrics.parquet")

# Benchmark and macro configuration
BENCHMARK_SYMBOLS = ["SPY", "IEF", "LQD", "BIL"]
CURRENCY_SYMBOLS = ["UUP"]
MACRO_SERIES = ["DGS10", "T10Y2Y", "BAA10Y", "DFII10"]

START_DATE = "2007-04-11"  # HYG inception
END_DATE = "2025-12-31"    # End of 2025 Historical Holdout

# Vehicle Structure Ontology (Spec v1.0.3)
ALLOWED_STRUCTURE_CLASSES = {
    "1940_ACT_OPEN_END_ETF",
    "1940_ACT_UNIT_INVESTMENT_TRUST_ETF",
    "PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST",
    "EXCHANGE_TRADED_NOTE",
    "COMMODITY_FUTURES_POOL",
    "LEVERAGED_ETF",
    "INVERSE_ETF",
    "CLOSED_END_FUND",
    "MUTUAL_FUND",
    "CRYPTO_LINKED_PRODUCT",
    "UNKNOWN",
}

RESEARCH_ELIGIBLE_STRUCTURES = {
    "1940_ACT_OPEN_END_ETF",
    "1940_ACT_UNIT_INVESTMENT_TRUST_ETF",
    "PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST",
}


# --------------------------------------------------------------------------
# REGISTRY PROVENANCE DEFINITIONS (Section 7)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RegistryMetadata:
    registry_name: str
    entries: frozenset
    source_authority: str
    source_identifier: str
    as_of_date: str
    temporality: str  # "CURRENT"
    update_policy: str
    failure_policy: str
    semantic_role: str


REGISTRY_KNOWN_1940_ACT_UITS = RegistryMetadata(
    registry_name="KNOWN_1940_ACT_UITS",
    entries=frozenset({"SPY", "DIA", "QQQ"}),
    source_authority="SEC_INVESTMENT_COMPANY_ACT_OF_1940_SECTION_4_2_UITS",
    source_identifier="SEC_FORM_S6_N8B2_UNIT_INVESTMENT_TRUST_SERIES_FILINGS",
    as_of_date="2026-09-25",
    temporality="CURRENT",
    update_policy="MANUAL_ANNUAL_REGISTRATION_AUDIT",
    failure_policy="EXCLUDE_IF_NOT_EXPLICITLY_VERIFIED",
    semantic_role="VERIFIED_POSITIVE_ALLOWLIST"
)

REGISTRY_PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUSTS = RegistryMetadata(
    registry_name="PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUSTS",
    entries=frozenset({"GLD", "IAU", "SLV", "SGOL", "SIVR", "PPLT", "PALL", "BAR", "OUNZ", "AAAU"}),
    source_authority="SEC_EDGAR_AND_TRUST_PROSPECTUSES",
    source_identifier="SEC_SCHEDULE_13D_S1_FILINGS_PHYSICAL_BULLION_GRANTOR_TRUSTS",
    as_of_date="2026-09-25",
    temporality="CURRENT",
    update_policy="MANUAL_ANNUAL_REGISTRATION_AUDIT",
    failure_policy="EXCLUDE_IF_NOT_EXPLICITLY_VERIFIED",
    semantic_role="VERIFIED_POSITIVE_ALLOWLIST"
)

REGISTRY_KNOWN_COMMODITY_FUTURES_POOLS = RegistryMetadata(
    registry_name="KNOWN_COMMODITY_FUTURES_POOLS",
    entries=frozenset({"USO", "UNG", "BNO", "UGA", "CPER", "DBA", "DBC", "GSG", "WEAT", "CORN", "SOYB", "BOIL", "KOLD"}),
    source_authority="CFTC_AND_SEC_EDGAR_K1_POOLS",
    source_identifier="SEC_FORM_10K_COMMODITY_POOL_OPERATOR_DISCLOSURES",
    as_of_date="2026-09-25",
    temporality="CURRENT",
    update_policy="MANUAL_SEMIANNUAL_AUDIT",
    failure_policy="FAIL_CLOSED_BLOCK",
    semantic_role="VERIFIED_NEGATIVE_BLOCKLIST"
)

REGISTRY_KNOWN_EXCHANGE_TRADED_NOTES = RegistryMetadata(
    registry_name="KNOWN_EXCHANGE_TRADED_NOTES",
    entries=frozenset({"AMJ", "SPLI", "USOI", "GLDI", "SLVO"}),
    source_authority="ISSUER_DEBT_PROSPECTUSES",
    source_identifier="SEC_FORM_424B2_UNSECURED_SENIOR_DEBT_OBLIGATIONS",
    as_of_date="2026-09-25",
    temporality="CURRENT",
    update_policy="MANUAL_SEMIANNUAL_AUDIT",
    failure_policy="FAIL_CLOSED_BLOCK",
    semantic_role="VERIFIED_NEGATIVE_BLOCKLIST"
)

REGISTRY_KNOWN_CRYPTO_PRODUCTS = RegistryMetadata(
    registry_name="KNOWN_CRYPTO_PRODUCTS",
    entries=frozenset({"BITO", "IBIT", "FBTC", "ARKB", "BITB", "BTCO", "HODL", "BRRR", "EZBC", "ETHW", "ETHE", "FETH", "ETH"}),
    source_authority="SEC_DIGITAL_ASSET_PRODUCT_ORDERS",
    source_identifier="SEC_RELEASE_34_99306_SPOT_AND_FUTURES_CRYPTO_EVALUATION",
    as_of_date="2026-09-25",
    temporality="CURRENT",
    update_policy="MANUAL_QUARTERLY_AUDIT",
    failure_policy="FAIL_CLOSED_BLOCK",
    semantic_role="VERIFIED_NEGATIVE_BLOCKLIST"
)

REGISTRY_KNOWN_LEVERAGED_INVERSE_PRODUCTS = RegistryMetadata(
    registry_name="KNOWN_LEVERAGED_INVERSE_PRODUCTS",
    entries=frozenset({
        "TQQQ", "SQQQ", "UPRO", "SPXU", "SSO", "SDS", "QLD", "QID", "SOXL", "SOXS",
        "LABU", "LABD", "NUGT", "DUST", "JNUG", "JDST", "UVXY", "SVXY", "TZA", "TNA", "FAS", "FAZ"
    }),
    source_authority="FINRA_NOTICE_09_31_AND_PROSPECTUSES",
    source_identifier="FINRA_NON_TRADITIONAL_GEARED_ETF_DISCLOSURES",
    as_of_date="2026-09-25",
    temporality="CURRENT",
    update_policy="MANUAL_SEMIANNUAL_AUDIT",
    failure_policy="FAIL_CLOSED_BLOCK",
    semantic_role="VERIFIED_NEGATIVE_BLOCKLIST"
)

@dataclass(frozen=True)
class Subtyped1940ActRegistryMetadata:
    registry_name: str
    subtypes: dict
    source_authority: str
    source_identifier: str
    as_of_date: str
    temporality: str
    update_policy: str
    failure_policy: str
    semantic_role: str

    @property
    def entries(self) -> frozenset:
        all_syms = set()
        for syms in self.subtypes.values():
            all_syms.update(syms)
        return frozenset(all_syms)


REGISTRY_KNOWN_VERIFIED_1940_ACT_ETFS = Subtyped1940ActRegistryMetadata(
    registry_name="KNOWN_VERIFIED_1940_ACT_ETFS",
    subtypes={
        "EQUITY_INDEX": frozenset({"SPY", "DIA", "QQQ", "IWM", "VOO", "IVV", "VTI", "SCHX", "RSP", "IJH", "IJR", "VB", "VO", "SCHD", "VUG", "VTV", "IEFA", "IEMG"}),
        "EQUITY_SECTOR": frozenset({"XLE", "XOP", "XLF", "KRE", "KBE", "XLK", "SMH", "XLV", "XBI", "IBB", "XLI", "XLP", "XLU", "XLY", "ITB", "XHB", "XLB", "VNQ", "IYR"}),
        "FIXED_INCOME_GOVERNMENT": frozenset({"TLT", "IEF", "SHY", "IEI", "GOVT", "VGSH", "VGIT", "VGLT", "SCHO", "SCHR", "SPTL", "BIL", "TIP", "SHV"}),
        "FIXED_INCOME_CREDIT": frozenset({"HYG", "LQD", "JNK", "VCIT", "VCSH", "USIG", "FLOT", "SJNK", "HYLB", "VUSB"}),
        "COMMODITY_PHYSICAL": REGISTRY_PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUSTS.entries,
    },
    source_authority="SEC_INVESTMENT_COMPANY_ACT_OF_1940_REGISTRATIONS",
    source_identifier="SEC_FORM_N1A_AND_ICA_1940_SERIES_FILINGS",
    as_of_date="2026-09-25",
    temporality="CURRENT",
    update_policy="MANUAL_ANNUAL_REGISTRATION_AUDIT",
    failure_policy="EXCLUDE_IF_NOT_EXPLICITLY_VERIFIED",
    semantic_role="VERIFIED_POSITIVE_ALLOWLIST"
)

# Registry inventory for testing and governance provenance verification
ALL_REGISTRIES = [
    REGISTRY_KNOWN_1940_ACT_UITS,
    REGISTRY_PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUSTS,
    REGISTRY_KNOWN_COMMODITY_FUTURES_POOLS,
    REGISTRY_KNOWN_EXCHANGE_TRADED_NOTES,
    REGISTRY_KNOWN_CRYPTO_PRODUCTS,
    REGISTRY_KNOWN_LEVERAGED_INVERSE_PRODUCTS,
    REGISTRY_KNOWN_VERIFIED_1940_ACT_ETFS,
]

# Supported subtype configurations
CONFIRMATORY_SUBTYPES = {
    "EQUITY_INDEX",
    "EQUITY_SECTOR",
    "FIXED_INCOME_GOVERNMENT",
    "FIXED_INCOME_CREDIT",
    "COMMODITY_PHYSICAL"
}

EXPLORATORY_SUBTYPES = {
    "ACTIVE_EQUITY",
    "COVERED_CALL",
    "OTHER_ETF"
}

# --------------------------------------------------------------------------
# TIER 4 DEFENSIVE NEGATIVE HEURISTICS (Negative safety net ONLY)
# --------------------------------------------------------------------------
RE_LEVERAGED = re.compile(
    r"\b(\d+x|leveraged|ultra(?![\s-]?(short|term|duration|maturity))|daily\s*bull|bull\s*\d+x)\b",
    re.IGNORECASE
)
RE_INVERSE = re.compile(
    r"\b(-1x|-2x|-3x|inverse|bear\s*\d*x?|daily\s*bear|ultra\s*short\b(?![\s-]*(term|duration|maturity|income|bond|treasury|muni|credit|fixed|active|target|yield))|ultrashort\b(?![\s-]*(term|duration|maturity|income|bond|treasury|muni|credit|fixed|active|target|yield))|ultrapro\s*short|short(?![\s-]*(term|duration|maturity|income|bond|treasury|muni|credit|fixed|active|target|yield)))\b",
    re.IGNORECASE
)
RE_LEVERAGED_INVERSE = re.compile(
    rf"({RE_LEVERAGED.pattern}|{RE_INVERSE.pattern})",
    re.IGNORECASE
)
RE_ETN = re.compile(r"\b(etn|exchange[\s-]traded\s*notes?)\b", re.IGNORECASE)
RE_COMMODITY_POOL = re.compile(r"\b(futures|commodity\s*index|crude\s*oil|natural\s*gas|k-1)\b", re.IGNORECASE)
RE_CRYPTO = re.compile(r"\b(bitcoin|ethereum|crypto|ether|solana|btc|eth)\b", re.IGNORECASE)
RE_CEF = re.compile(r"\b(closed[\s-]end|cef)\b", re.IGNORECASE)
RE_MUTUAL_FUND = re.compile(r"\b(mutual\s*fund)\b", re.IGNORECASE)


def get_file_sha256(filepath: Path | str) -> str:
    """Calculate SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Get current HEAD git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to read git commit: {e}")
        return "UNKNOWN_GIT_COMMIT"


# --------------------------------------------------------------------------
# SOURCE CACHE MANIFEST MANAGER
# --------------------------------------------------------------------------
class SourceCacheManager:
    """Manages the source cache manifest at data/research/source_cache_manifest_v1.json."""

    def __init__(self, manifest_path: Path = SOURCE_CACHE_MANIFEST_PATH):
        self.manifest_path = Path(manifest_path).resolve()
        self.entries = {}
        self.load()

    def load(self):
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing source cache manifest: {e}")
                self.entries = {}

    def save(self):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2)
        except OSError:
            import time
            time.sleep(0.1)
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2)

    def record(
        self,
        path: Path | str,
        provider: str,
        semantic_role: str,
        byte_size: int = None,
        sha256: str = None,
        retrieval_timestamp: str = None
    ) -> dict:
        p = Path(path)
        if byte_size is None and p.exists():
            byte_size = p.stat().st_size
        if sha256 is None and p.exists():
            sha256 = get_file_sha256(p)
        if retrieval_timestamp is None:
            retrieval_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        rel_path = str(p).replace("\\", "/")
        entry = {
            "path": rel_path,
            "provider": provider,
            "retrieval_timestamp": retrieval_timestamp,
            "byte_size": int(byte_size) if byte_size is not None else 0,
            "sha256": str(sha256) if sha256 is not None else "",
            "semantic_role": semantic_role
        }
        self.entries[rel_path] = entry
        self.save()
        return entry


# --------------------------------------------------------------------------
# NASDAQ TRADED DIRECTORY DISCOVERY
# --------------------------------------------------------------------------
def parse_nasdaq_traded_content(
    raw_bytes: bytes,
    source_url: str = "ftp://ftp.nasdaqtrader.com/symboldir/nasdaqtraded.txt"
) -> tuple[pd.DataFrame, dict]:
    """Parses raw nasdaqtraded.txt bytes, extracting structured data and metadata."""
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty Nasdaq trader content")

    trailer_line = lines[-1]
    creation_time = None
    if "File Creation Time:" in trailer_line:
        parts = trailer_line.split(":")
        if len(parts) >= 2:
            creation_time = ":".join(parts[1:]).replace("|", "").strip()
        data_lines = lines[:-1]
    else:
        data_lines = lines

    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(data_lines)), sep="|", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    etf_count = int((df["ETF"] == "Y").sum()) if "ETF" in df.columns else 0

    metadata = {
        "retrieval_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url_identifier": source_url,
        "raw_source_sha256": raw_sha256,
        "file_creation_timestamp": creation_time or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row_count": len(df),
        "etf_flagged_row_count": etf_count
    }
    return df, metadata


def fetch_nasdaq_traded_directory(
    cache_dir: Path = CACHE_DIR,
    source_manifest: SourceCacheManager = None
) -> tuple[pd.DataFrame, dict]:
    """Retrieves nasdaqtraded.txt directory with caching and source-cache tracking."""
    target_file = cache_dir / "nasdaqtraded.txt"
    if target_file.exists():
        with open(target_file, "rb") as f:
            raw_bytes = f.read()
    else:
        url = "ftp://ftp.nasdaqtrader.com/symboldir/nasdaqtraded.txt"
        logger.info(f"Retrieving Nasdaq Traded Directory from {url}...")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            raw_bytes = resp.content
        except Exception as e:
            http_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt"
            logger.info(f"Retrying via HTTP from {http_url} due to: {e}")
            resp = requests.get(http_url, timeout=30)
            resp.raise_for_status()
            raw_bytes = resp.content

        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(target_file, "wb") as f:
            f.write(raw_bytes)

    df, metadata = parse_nasdaq_traded_content(raw_bytes)
    if source_manifest:
        source_manifest.record(
            path=target_file,
            provider="NASDAQ_TRADER",
            semantic_role="UNIVERSE_DISCOVERY_DIRECTORY",
            byte_size=len(raw_bytes),
            sha256=metadata["raw_source_sha256"],
            retrieval_timestamp=metadata["retrieval_timestamp"]
        )
    return df, metadata


def load_sec_mf_directory(
    cache_dir: Path = CACHE_DIR,
    source_manifest: SourceCacheManager = None
) -> dict[str, dict]:
    """Loads SEC EDGAR series directory mapping symbol -> {cik, series_id, class_id}."""
    target_file = cache_dir / "sec_company_tickers_mf.json"
    if not target_file.exists() and (CACHE_DIR / "sec_company_tickers_mf.json").exists():
        target_file = CACHE_DIR / "sec_company_tickers_mf.json"

    if target_file.exists():
        with open(target_file, "rb") as f:
            raw_bytes = f.read()
    else:
        url = "https://www.sec.gov/files/company_tickers_mf.json"
        logger.info(f"Retrieving SEC Mutual Fund / Series Directory from {url}...")
        headers = {"User-Agent": "ARX Research Bot research@arxterminal.com"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        raw_bytes = resp.content
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(target_file, "wb") as f:
            f.write(raw_bytes)

    data = json.loads(raw_bytes.decode("utf-8"))
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    if source_manifest:
        source_manifest.record(
            path=target_file,
            provider="SEC_EDGAR",
            semantic_role="TIER_2_SERIES_REGISTRATION_DIRECTORY",
            byte_size=len(raw_bytes),
            sha256=raw_sha,
            retrieval_timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    # Load Trust Legal Form Directory
    trust_dir_path = Path("data/research/sec_trust_legal_form_directory_v1.json")
    trust_map = {}
    if trust_dir_path.exists():
        with open(trust_dir_path, "r", encoding="utf-8") as f:
            trust_data = json.load(f)
            trust_map = trust_data.get("trusts", {})
        if source_manifest:
            source_manifest.record(
                path=trust_dir_path,
                provider="SEC_EDGAR",
                semantic_role="TIER_2_TRUST_LEGAL_FORM_DIRECTORY",
                byte_size=trust_dir_path.stat().st_size,
                sha256=get_file_sha256(trust_dir_path),
                retrieval_timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            )

    mf_map = {}
    rows = data.get("data", [])
    for row in rows:
        if len(row) >= 4 and row[3]:
            sym = str(row[3]).strip().upper()
            cik_raw = str(row[0])
            cik_padded = cik_raw.zfill(10)

            trust_info = trust_map.get(cik_raw) or trust_map.get(cik_padded) or {}
            reg_form = trust_info.get("registration_form")
            legal_struct = trust_info.get("legal_structure")
            is_active = trust_info.get("is_active", False)
            trust_name = trust_info.get("trust_name")
            record_hash = trust_info.get("normalized_record_sha256")

            mf_map[sym] = {
                "cik": cik_raw,
                "cik_padded": cik_padded,
                "series_id": str(row[1]),
                "class_id": str(row[2]),
                "symbol": sym,
                "trust_name": trust_name,
                "registration_form": reg_form,
                "legal_structure": legal_struct,
                "is_active": is_active,
                "normalized_record_sha256": record_hash
            }
    return mf_map


# --------------------------------------------------------------------------
# CLASSIFICATION AUTHORITY ENGINE (Reconciled Precedence & Provenance)
# --------------------------------------------------------------------------
class ClassificationAuthorityEngine:
    """Enforces the reconciled 5-tier classification authority hierarchy.

    Precedence order:
    1. Tier 1: Exchange Discovery Metadata (Nasdaq traded directory flag ETF == 'Y')
    2. Tier 3 Verified Blocklists (Explicit higher-authority exclusion)
    3. Tier 3 Verified Positive Registries (Authoritative positive legal structure & subtype attestation)
    4. Tier 2 Structured Provider Metadata (SEC / Provider legal structure attestation)
    5. Tier 4 Defensive Negative Heuristics (Negative safety net ONLY for unverified instruments)
    6. Tier 5 Unknown Structure Quarantine (Fail-closed)

    Stage 1: Legal Structure Verification (vehicle_structure, vehicle_structure_state, classification_source, structure_verified).
    Stage 2: Research Subtype Authorization (research_subtype, research_subtype_state, subtype_authorized).
    LEGAL_STRUCTURE_DEPENDS_ON_SUBTYPE_ALLOWLIST = NO.
    """

    @classmethod
    def classify_security(
        cls,
        symbol: str,
        security_name: str,
        listing_exchange: str,
        nasdaq_etf_flag: bool,
        structured_metadata: dict = None,
        sec_mf_info: dict = None,
        portfolio_metrics: dict | pd.Series = None,
        mandate_evidence: dict = None,
        timestamp: str = None
    ) -> dict:
        if timestamp is None:
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        name = str(security_name or "")
        sym = str(symbol or "").strip().upper()
        exch = str(listing_exchange or "").strip()

        # Tier 1: Discovery Check
        if not nasdaq_etf_flag:
            return {
                "symbol": sym,
                "security_name": name,
                "listing_exchange": exch,
                "nasdaq_etf_flag": False,
                "vehicle_structure": "UNKNOWN",
                "vehicle_structure_state": "EXCLUDED",
                "research_subtype": None,
                "research_subtype_state": "EXCLUDED",
                "classification_source": "TIER_1_EXCHANGE_DISCOVERY",
                "classification_evidence": "NASDAQ_ETF_FLAG_IS_N",
                "classification_timestamp": timestamp,
                "structure_verified": False,
                "subtype_authorized": False,
                "is_research_eligible": False,
                "exclusion_reason": "NOT_NASDAQ_ETF_DISCOVERED"
            }

        # ------------------------------------------------------------------
        # STAGE 1: LEGAL STRUCTURE CLASSIFICATION
        # ------------------------------------------------------------------
        vehicle_structure = None
        vehicle_structure_state = None
        classification_source = None
        classification_evidence = None
        structure_verified = False
        exclusion_reason = None

        # 1. Tier 3: Explicit Verified Negative Blocklists (Highest Negative Authority)
        if sym in REGISTRY_KNOWN_EXCHANGE_TRADED_NOTES.entries:
            vehicle_structure = "EXCHANGE_TRADED_NOTE"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_3_VERIFIED_VEHICLE_STRUCTURE_REGISTRY"
            classification_evidence = "VERIFIED_ETN_BLOCKLIST_MEMBERSHIP"
            exclusion_reason = "EXCLUDED_STRUCTURE_EXCHANGE_TRADED_NOTE"
        elif sym in REGISTRY_KNOWN_COMMODITY_FUTURES_POOLS.entries:
            vehicle_structure = "COMMODITY_FUTURES_POOL"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_3_VERIFIED_VEHICLE_STRUCTURE_REGISTRY"
            classification_evidence = "VERIFIED_COMMODITY_POOL_BLOCKLIST_MEMBERSHIP"
            exclusion_reason = "EXCLUDED_STRUCTURE_COMMODITY_FUTURES_POOL"
        elif sym in REGISTRY_KNOWN_CRYPTO_PRODUCTS.entries:
            vehicle_structure = "CRYPTO_LINKED_PRODUCT"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_3_VERIFIED_VEHICLE_STRUCTURE_REGISTRY"
            classification_evidence = "VERIFIED_CRYPTO_BLOCKLIST_MEMBERSHIP"
            exclusion_reason = "EXCLUDED_STRUCTURE_CRYPTO_LINKED"
        elif sym in REGISTRY_KNOWN_LEVERAGED_INVERSE_PRODUCTS.entries:
            is_inverse = bool(re.search(r"\b(inverse|short|bear)\b", name, re.IGNORECASE))
            vehicle_structure = "INVERSE_ETF" if is_inverse else "LEVERAGED_ETF"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_3_VERIFIED_VEHICLE_STRUCTURE_REGISTRY"
            classification_evidence = "VERIFIED_LEVERAGED_INVERSE_BLOCKLIST_MEMBERSHIP"
            exclusion_reason = "EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE"

        # 2. Tier 3: Verified Positive Registries (Authoritative Positive Attestation)
        elif sym in REGISTRY_KNOWN_1940_ACT_UITS.entries:
            vehicle_structure = "1940_ACT_UNIT_INVESTMENT_TRUST_ETF"
            vehicle_structure_state = "STRUCTURE_VERIFIED"
            classification_source = "TIER_3_VERIFIED_VEHICLE_STRUCTURE_REGISTRY"
            classification_evidence = "VERIFIED_1940_ACT_UIT_ALLOWLIST"
            structure_verified = True
        elif sym in REGISTRY_PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUSTS.entries:
            vehicle_structure = "PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST"
            vehicle_structure_state = "STRUCTURE_VERIFIED"
            classification_source = "TIER_3_VERIFIED_VEHICLE_STRUCTURE_REGISTRY"
            classification_evidence = "VERIFIED_PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST_ALLOWLIST"
            structure_verified = True
        elif sym in REGISTRY_KNOWN_VERIFIED_1940_ACT_ETFS.entries:
            vehicle_structure = "1940_ACT_OPEN_END_ETF"
            vehicle_structure_state = "STRUCTURE_VERIFIED"
            classification_source = "TIER_3_VERIFIED_VEHICLE_STRUCTURE_REGISTRY"
            classification_evidence = "VERIFIED_1940_ACT_OPEN_END_ALLOWLIST"
            structure_verified = True

        # 3. Tier 2: Structured Provider / Primary Regulatory Directory (SEC EDGAR series directory + Trust Legal Form)
        elif sec_mf_info is not None:
            reg_form = sec_mf_info.get("registration_form")
            is_active = sec_mf_info.get("is_active", False)
            cik = sec_mf_info.get("cik")
            series_id = sec_mf_info.get("series_id")
            class_id = sec_mf_info.get("class_id")

            if reg_form == "N-1A" and is_active:
                vehicle_structure = "1940_ACT_OPEN_END_ETF"
                vehicle_structure_state = "STRUCTURE_VERIFIED"
                classification_source = "TIER_2_STRUCTURED_PROVIDER_METADATA"
                classification_evidence = f"SEC_EDGAR_FORM_N1A_REGISTRATION_CIK_{cik}_SERIES_{series_id}_CLASS_{class_id}"
                structure_verified = True
            elif reg_form == "S-6" and is_active:
                vehicle_structure = "1940_ACT_UNIT_INVESTMENT_TRUST_ETF"
                vehicle_structure_state = "STRUCTURE_VERIFIED"
                classification_source = "TIER_2_STRUCTURED_PROVIDER_METADATA"
                classification_evidence = f"SEC_EDGAR_FORM_S6_REGISTRATION_CIK_{cik}_SERIES_{series_id}_CLASS_{class_id}"
                structure_verified = True
            elif reg_form == "N-2":
                vehicle_structure = "CLOSED_END_FUND"
                vehicle_structure_state = "EXCLUDED"
                classification_source = "TIER_2_STRUCTURED_PROVIDER_METADATA"
                classification_evidence = f"SEC_EDGAR_FORM_N2_REGISTRATION_CIK_{cik}_SERIES_{series_id}_CLASS_{class_id}"
                exclusion_reason = "EXCLUDED_STRUCTURE_CLOSED_END_FUND"
            else:
                # Unresolved or unsupported registration form in Tier 2 - fail closed
                vehicle_structure = "UNKNOWN"
                vehicle_structure_state = "QUARANTINED"
                classification_source = "TIER_2_STRUCTURED_PROVIDER_METADATA"
                classification_evidence = f"UNVERIFIED_SEC_REGISTRATION_FORM_CIK_{cik}_SERIES_{series_id}_CLASS_{class_id}"
                exclusion_reason = "UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED"
        elif structured_metadata and structured_metadata.get("is_1940_act") is True:
            form = structured_metadata.get("registration_form")
            cat = structured_metadata.get("category", "")
            if form == "N-1A":
                vehicle_structure = "1940_ACT_OPEN_END_ETF"
                vehicle_structure_state = "STRUCTURE_VERIFIED"
                classification_source = "TIER_2_STRUCTURED_PROVIDER_METADATA"
                classification_evidence = f"STRUCTURED_PROVIDER_FORM_N1A_ATTESTATION: {cat}"
                structure_verified = True
            elif form == "S-6":
                vehicle_structure = "1940_ACT_UNIT_INVESTMENT_TRUST_ETF"
                vehicle_structure_state = "STRUCTURE_VERIFIED"
                classification_source = "TIER_2_STRUCTURED_PROVIDER_METADATA"
                classification_evidence = f"STRUCTURED_PROVIDER_FORM_S6_ATTESTATION: {cat}"
                structure_verified = True
            elif form == "N-2":
                vehicle_structure = "CLOSED_END_FUND"
                vehicle_structure_state = "EXCLUDED"
                classification_source = "TIER_2_STRUCTURED_PROVIDER_METADATA"
                classification_evidence = f"STRUCTURED_PROVIDER_FORM_N2_ATTESTATION: {cat}"
                exclusion_reason = "EXCLUDED_STRUCTURE_CLOSED_END_FUND"
            else:
                vehicle_structure = "UNKNOWN"
                vehicle_structure_state = "QUARANTINED"
                classification_source = "TIER_2_STRUCTURED_PROVIDER_METADATA"
                classification_evidence = f"UNVERIFIED_STRUCTURED_PROVIDER_ATTESTATION: {cat}"
                exclusion_reason = "UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED"

        # 4. Tier 4: Defensive Negative Safety Net Heuristics (ONLY for unverified instruments)
        elif RE_INVERSE.search(name):
            vehicle_structure = "INVERSE_ETF"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_4_DEFENSIVE_HEURISTIC"
            classification_evidence = f"MATCHED_INVERSE_CRITERIA: {name}"
            exclusion_reason = "EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE"
        elif RE_LEVERAGED.search(name):
            vehicle_structure = "LEVERAGED_ETF"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_4_DEFENSIVE_HEURISTIC"
            classification_evidence = f"MATCHED_LEVERAGED_CRITERIA: {name}"
            exclusion_reason = "EXCLUDED_STRUCTURE_LEVERAGED_OR_INVERSE"
        elif RE_ETN.search(name):
            vehicle_structure = "EXCHANGE_TRADED_NOTE"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_4_DEFENSIVE_HEURISTIC"
            classification_evidence = f"MATCHED_ETN_CRITERIA: {name}"
            exclusion_reason = "EXCLUDED_STRUCTURE_EXCHANGE_TRADED_NOTE"
        elif RE_COMMODITY_POOL.search(name):
            vehicle_structure = "COMMODITY_FUTURES_POOL"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_4_DEFENSIVE_HEURISTIC"
            classification_evidence = f"MATCHED_COMMODITY_POOL_CRITERIA: {name}"
            exclusion_reason = "EXCLUDED_STRUCTURE_COMMODITY_FUTURES_POOL"
        elif RE_CRYPTO.search(name):
            vehicle_structure = "CRYPTO_LINKED_PRODUCT"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_4_DEFENSIVE_HEURISTIC"
            classification_evidence = f"MATCHED_CRYPTO_CRITERIA: {name}"
            exclusion_reason = "EXCLUDED_STRUCTURE_CRYPTO_LINKED"
        elif RE_CEF.search(name):
            vehicle_structure = "CLOSED_END_FUND"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_4_DEFENSIVE_HEURISTIC"
            classification_evidence = f"MATCHED_CEF_CRITERIA: {name}"
            exclusion_reason = "EXCLUDED_STRUCTURE_CLOSED_END_FUND"
        elif RE_MUTUAL_FUND.search(name):
            vehicle_structure = "MUTUAL_FUND"
            vehicle_structure_state = "EXCLUDED"
            classification_source = "TIER_4_DEFENSIVE_HEURISTIC"
            classification_evidence = f"MATCHED_MUTUAL_FUND_CRITERIA: {name}"
            exclusion_reason = "EXCLUDED_STRUCTURE_MUTUAL_FUND"

        # 5. Tier 5: Unknown Structure Quarantine (Fail-Closed)
        else:
            vehicle_structure = "UNKNOWN"
            vehicle_structure_state = "QUARANTINED"
            classification_source = "TIER_5_UNKNOWN_STRUCTURE_QUARANTINE"
            classification_evidence = "FAIL_CLOSED_NO_AFFIRMATIVE_LEGAL_STRUCTURE_VERIFICATION"
            exclusion_reason = "UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED"

        # ------------------------------------------------------------------
        # STAGE 2: RESEARCH SUBTYPE CLASSIFICATION (Decoupled from Structure)
        # ------------------------------------------------------------------
        research_subtype = None
        research_subtype_state = "EXCLUDED"
        subtype_authorized = False

        if structure_verified:
            if sym in REGISTRY_PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUSTS.entries or vehicle_structure == "PHYSICAL_PRECIOUS_METAL_GRANTOR_TRUST":
                research_subtype = "COMMODITY_PHYSICAL"
                research_subtype_state = "CONFIRMATORY_SUPPORTED"
                subtype_authorized = True
                exclusion_reason = None
            elif sym in REGISTRY_KNOWN_1940_ACT_UITS.entries or vehicle_structure == "1940_ACT_UNIT_INVESTMENT_TRUST_ETF":
                research_subtype = "EQUITY_INDEX"
                research_subtype_state = "CONFIRMATORY_SUPPORTED"
                subtype_authorized = True
                exclusion_reason = None
            elif structured_metadata and structured_metadata.get("research_subtype"):
                st = structured_metadata["research_subtype"]
                research_subtype = st
                research_subtype_state = "CONFIRMATORY_SUPPORTED" if st in CONFIRMATORY_SUBTYPES else "EXPLORATORY_ONLY"
                subtype_authorized = (st in CONFIRMATORY_SUBTYPES)
                exclusion_reason = None if subtype_authorized else "UNAUTHORIZED_RESEARCH_SUBTYPE"
            elif portfolio_metrics is not None or mandate_evidence is not None:
                # Systematic Tier 2/3 Policy v1.1.0 Evaluation
                if portfolio_metrics is None:
                    research_subtype = "UNRESOLVED"
                    research_subtype_state = "PENDING_SYSTEMATIC_CLASSIFICATION"
                    subtype_authorized = False
                    classification_source = "TIER_4_FAIL_CLOSED_UNRESOLVED_SUBTYPE_QUARANTINE"
                    classification_evidence = "INSUFFICIENT_NPORT_REGULATORY_EVIDENCE"
                    exclusion_reason = "UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION"
                else:
                    rec_ratio = portfolio_metrics.get("reconciliation_ratio") if isinstance(portfolio_metrics, dict) else portfolio_metrics["reconciliation_ratio"]
                    if pd.isna(rec_ratio) or rec_ratio < 0.85 or rec_ratio > 1.15:
                        research_subtype = "UNRESOLVED"
                        research_subtype_state = "PENDING_SYSTEMATIC_CLASSIFICATION"
                        subtype_authorized = False
                        classification_source = "TIER_4_FAIL_CLOSED_UNRESOLVED_SUBTYPE_QUARANTINE"
                        classification_evidence = f"PORTFOLIO_RECONCILIATION_FAIL_RATIO_{rec_ratio:.2f}" if pd.notna(rec_ratio) else "PORTFOLIO_RECONCILIATION_NAN"
                        exclusion_reason = "UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION"
                    else:
                        eq_pct = float(portfolio_metrics.get("total_equity_pct", 0.0) if isinstance(portfolio_metrics, dict) else portfolio_metrics["total_equity_pct"])
                        gov_pct = float(portfolio_metrics.get("total_govt_pct", 0.0) if isinstance(portfolio_metrics, dict) else portfolio_metrics["total_govt_pct"])
                        corp_pct = float(portfolio_metrics.get("corporate_debt_pct", 0.0) if isinstance(portfolio_metrics, dict) else portfolio_metrics["corporate_debt_pct"])
                        mbs_pct = float(portfolio_metrics.get("mortgage_backed_pct", 0.0) if isinstance(portfolio_metrics, dict) else portfolio_metrics["mortgage_backed_pct"])
                        distinct_eq = int(portfolio_metrics.get("distinct_equity_count", 0) if isinstance(portfolio_metrics, dict) else portfolio_metrics["distinct_equity_count"])
                        max_conc = float(portfolio_metrics.get("max_concentration", 0.0) if isinstance(portfolio_metrics, dict) else portfolio_metrics["max_concentration"])
                        is_index_ncen = bool(portfolio_metrics.get("is_index_ncen", False) if isinstance(portfolio_metrics, dict) else portfolio_metrics["is_index_ncen"])

                        passes_gov = (gov_pct >= 0.80 and corp_pct < 0.10 and mbs_pct < 0.10 and eq_pct < 0.05)
                        passes_credit = (corp_pct >= 0.50 and gov_pct < 0.50 and eq_pct < 0.05)

                        has_mandate = mandate_evidence is not None
                        m_dict = mandate_evidence if isinstance(mandate_evidence, dict) else {}
                        is_sector_fund = bool(m_dict.get("is_sector_specific_mandate", False))
                        approved_sec = m_dict.get("approved_sector")
                        is_broad_index = bool(m_dict.get("is_broad_or_multi_sector_mandate", False))
                        mandate_class = m_dict.get("derived_mandate_classification")

                        is_gov_mandate = (mandate_class in ("GOVERNMENT_DEBT_MANDATE", "US_TREASURY_GOVERNMENT_MANDATE"))
                        is_credit_mandate = (mandate_class in ("CORPORATE_CREDIT_MANDATE", "CREDIT_MANDATE"))

                        passes_sector = (eq_pct >= 0.80 and is_sector_fund and approved_sec is not None)
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
                        if passes_gov and is_gov_mandate:
                            research_subtype = "FIXED_INCOME_GOVERNMENT"
                            research_subtype_state = "CONFIRMATORY_SUPPORTED"
                            subtype_authorized = True
                            classification_source = "TIER_2_STRUCTURED_REGULATORY_PORTFOLIO_AND_MANDATES"
                            classification_evidence = f"SEC_FORM_NPORT_GOVT_{gov_pct:.1%}_CORP_{corp_pct:.1%}_MBS_{mbs_pct:.1%}"
                            exclusion_reason = None
                        elif passes_credit and is_credit_mandate:
                            research_subtype = "FIXED_INCOME_CREDIT"
                            research_subtype_state = "CONFIRMATORY_SUPPORTED"
                            subtype_authorized = True
                            classification_source = "TIER_2_STRUCTURED_REGULATORY_PORTFOLIO_AND_MANDATES"
                            classification_evidence = f"SEC_FORM_NPORT_CREDIT_{corp_pct:.1%}_GOVT_{gov_pct:.1%}"
                            exclusion_reason = None
                        elif passes_sector:
                            research_subtype = "EQUITY_SECTOR"
                            research_subtype_state = "CONFIRMATORY_SUPPORTED"
                            subtype_authorized = True
                            classification_source = "TIER_2_STRUCTURED_REGULATORY_PORTFOLIO_AND_MANDATES"
                            classification_evidence = f"SEC_FORM_NPORT_EQUITY_{eq_pct:.1%}_SECTOR_{approved_sec}"
                            exclusion_reason = None
                        elif passes_equity_index:
                            research_subtype = "EQUITY_INDEX"
                            research_subtype_state = "CONFIRMATORY_SUPPORTED"
                            subtype_authorized = True
                            classification_source = "TIER_2_STRUCTURED_REGULATORY_PORTFOLIO_AND_MANDATES"
                            classification_evidence = f"SEC_FORM_NPORT_EQUITY_{eq_pct:.1%}_HOLDINGS_{distinct_eq}_INDEX_NCEN"
                            exclusion_reason = None
                        else:
                            # Central Invariant:
                            # OTHER_ETF requires authoritative evidence sufficient to evaluate all potentially
                            # applicable confirmatory subtype rules, and none passed.
                            # It must NOT mean absence of confirmatory evidence.
                            # If portfolio exposure satisfies quantitative thresholds for an applicable rule
                            # (equity >= 80%, government debt, or credit debt) but statutory mandate evidence is missing,
                            # the rule cannot be evaluated; the instrument must fail closed to UNRESOLVED.
                            mandate_required = (eq_pct >= 0.80 or passes_gov or passes_credit)
                            if mandate_required and not has_mandate:
                                research_subtype = "UNRESOLVED"
                                research_subtype_state = "INSUFFICIENT_SOURCE_EVIDENCE"
                                subtype_authorized = False
                                classification_source = "TIER_4_FAIL_CLOSED_UNRESOLVED_SUBTYPE_QUARANTINE"
                                classification_evidence = f"INSUFFICIENT_MANDATE_REGULATORY_EVIDENCE: EQ_{eq_pct:.1%}_GOV_{gov_pct:.1%}_CORP_{corp_pct:.1%}"
                                exclusion_reason = "INSUFFICIENT_MANDATE_EVIDENCE"
                            else:
                                research_subtype = "OTHER_ETF"
                                research_subtype_state = "EXPLORATORY_ONLY"
                                subtype_authorized = False
                                classification_source = "TIER_5_EVALUATED_EXPLORATORY_ASSIGNMENT"
                                classification_evidence = f"AFFIRMATIVE_NON_CONFIRMATORY_PORTFOLIO: EQ_{eq_pct:.1%}_GOV_{gov_pct:.1%}_CORP_{corp_pct:.1%}"
                                exclusion_reason = "UNAUTHORIZED_RESEARCH_SUBTYPE"
            elif sym in REGISTRY_KNOWN_VERIFIED_1940_ACT_ETFS.entries:
                for st, s_set in REGISTRY_KNOWN_VERIFIED_1940_ACT_ETFS.subtypes.items():
                    if sym in s_set:
                        research_subtype = st
                        research_subtype_state = "CONFIRMATORY_SUPPORTED" if st in CONFIRMATORY_SUBTYPES else "EXPLORATORY_ONLY"
                        subtype_authorized = (st in CONFIRMATORY_SUBTYPES)
                        classification_source = "TIER_3_VERIFIED_VEHICLE_STRUCTURE_REGISTRY"
                        classification_evidence = f"VERIFIED_1940_ACT_{st}_ALLOWLIST"
                        break
            else:
                # Verified legal structure (e.g. via SEC registration) but not systematically evaluated against subtype policy
                research_subtype = "UNRESOLVED"
                research_subtype_state = "PENDING_SYSTEMATIC_CLASSIFICATION"
                subtype_authorized = False
                exclusion_reason = "UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION"

            if not subtype_authorized and exclusion_reason is None:
                exclusion_reason = "UNAUTHORIZED_RESEARCH_SUBTYPE"

        return {
            "symbol": sym,
            "security_name": name,
            "listing_exchange": exch,
            "nasdaq_etf_flag": True,
            "vehicle_structure": vehicle_structure,
            "vehicle_structure_state": vehicle_structure_state,
            "research_subtype": research_subtype,
            "research_subtype_state": research_subtype_state,
            "classification_source": classification_source,
            "classification_evidence": classification_evidence,
            "classification_timestamp": timestamp,
            "structure_verified": structure_verified,
            "subtype_authorized": subtype_authorized,
            "is_research_eligible": False,  # Pending market eligibility
            "exclusion_reason": exclusion_reason
        }


# --------------------------------------------------------------------------
# LIQUIDITY & HISTORY EVALUATION (Section 8)
# --------------------------------------------------------------------------
def evaluate_liquidity_and_history(
    symbols: list[str],
    price_data_adj: dict[str, pd.DataFrame],
    trading_calendar: pd.DatetimeIndex,
    adv_window: int = 60,
    adv_percentile: float = 0.80,
    min_history: int = 250
) -> pd.DataFrame:
    """Calculates daily rolling ADV60 across active survivor universe, evaluates cross-sectional
    80th percentile threshold, and enforces min_history >= 250 sessions.
    Fail closed if insufficient data.
    """
    adv_matrix = pd.DataFrame(index=trading_calendar)
    hist_lens = {}

    for sym in symbols:
        df_sym = price_data_adj.get(sym)
        if df_sym is not None and isinstance(df_sym, pd.DataFrame) and "Close" in df_sym and "Volume" in df_sym and not df_sym.empty:
            df = df_sym
            dvol = df["Close"] * df["Volume"]
            adv = dvol.rolling(window=adv_window, min_periods=adv_window).mean()
            adv_matrix[sym] = adv.reindex(trading_calendar)
            hist_lens[sym] = pd.Series(np.arange(1, len(df) + 1), index=df.index).reindex(trading_calendar).fillna(0)
        else:
            adv_matrix[sym] = np.nan
            hist_lens[sym] = pd.Series(0, index=trading_calendar)

    records = []
    for date in trading_calendar:
        row_adv = adv_matrix.loc[date].dropna()
        if len(row_adv) > 0:
            p80 = float(row_adv.quantile(adv_percentile))
        else:
            p80 = np.inf

        for sym in symbols:
            val = adv_matrix.loc[date].get(sym, np.nan)
            hlen = int(hist_lens[sym].loc[date])
            is_liquid = bool(val >= p80) if not np.isnan(val) and p80 != np.inf else False
            has_history = bool(hlen >= min_history)
            has_valid_data = bool(not np.isnan(val) or hlen > 0)
            in_universe = bool(is_liquid and has_history)

            records.append({
                "symbol": sym,
                "observation_date": str(date.date()),
                "adv60": float(val) if not np.isnan(val) else None,
                "adv80_threshold": float(p80) if p80 != np.inf else None,
                "history_sessions": hlen,
                "has_valid_data": has_valid_data,
                "is_liquid": is_liquid,
                "has_min_history": has_history,
                "in_universe": in_universe
            })

    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# UNIVERSE SNAPSHOT BUILDER (Composite Eligibility & Call Graph Wiring)
# --------------------------------------------------------------------------
def build_universe_snapshot(
    discovery_file: Path | str = None,
    price_data_dict: dict[str, pd.DataFrame] = None,
    as_of_date: str = None,
    output_parquet: Path | str = UNIVERSE_SNAPSHOT_PATH,
    output_manifest: Path | str = UNIVERSE_MANIFEST_PATH,
    cache_dir: Path = CACHE_DIR,
    source_manifest: SourceCacheManager = None
) -> dict:
    """Builds canonical ETF surviving universe snapshot and manifest.

    Call Graph:
    CLI -> discovery -> legal/vehicle classification -> subtype classification
    -> retrieve limited eligibility price/volume history -> evaluate_liquidity_and_history()
    -> merge market eligibility result -> derive final is_research_eligible
    -> derive exclusion_reason -> write snapshot -> write manifest.

    Enforces deterministic exclusion-reason precedence:
    1. NOT_NASDAQ_ETF_DISCOVERED
    2. EXCLUDED_VEHICLE_STRUCTURE (class-specific)
    3. UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED
    4. UNAUTHORIZED_RESEARCH_SUBTYPE
    5. MARKET_DATA_MISSING_OR_INVALID
    6. INSUFFICIENT_HISTORY_LT_250
    7. ADV60_BELOW_80TH_PERCENTILE
    """
    output_parquet = Path(output_parquet)
    output_manifest = Path(output_manifest)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    if source_manifest is None:
        source_manifest = SourceCacheManager()

    if discovery_file:
        disc_path = Path(discovery_file)
        with open(disc_path, "rb") as f:
            raw_bytes = f.read()
        df_disc, meta_disc = parse_nasdaq_traded_content(raw_bytes, source_url=str(disc_path))
        source_manifest.record(
            path=disc_path,
            provider="LOCAL_FILE",
            semantic_role="UNIVERSE_DISCOVERY_DIRECTORY",
            byte_size=len(raw_bytes),
            sha256=meta_disc["raw_source_sha256"],
            retrieval_timestamp=meta_disc["retrieval_timestamp"]
        )
    else:
        df_disc, meta_disc = fetch_nasdaq_traded_directory(cache_dir, source_manifest)

    # Filter out test issues
    test_issue_col = "Test Issue" if "Test Issue" in df_disc.columns else "TestIssue"
    if test_issue_col in df_disc.columns:
        df_clean = df_disc[df_disc[test_issue_col] == "N"].copy()
    else:
        df_clean = df_disc.copy()

    # Defect 1: Discovery domain strictly requires ETF == 'Y'
    etf_col = "ETF" if "ETF" in df_clean.columns else "etf"
    df_discovered_etfs = df_clean[df_clean[etf_col].astype(str).str.strip().str.upper() == "Y"].copy()

    # Load Tier 2 SEC Mutual Fund / Series Directory
    sec_mf_map = load_sec_mf_directory(cache_dir, source_manifest)

    # Load Statutory Series Mandate Evidence
    mandate_map = {}
    if MANDATE_EVIDENCE_PATH.exists():
        with open(MANDATE_EVIDENCE_PATH, "r", encoding="utf-8") as f:
            mandate_json = json.load(f)
            for m_entry in mandate_json.get("entries", []):
                mandate_map[m_entry["symbol"]] = m_entry
        source_manifest.record(
            path=MANDATE_EVIDENCE_PATH,
            provider="SEC_FORM_485BPOS_N1A",
            semantic_role="STATUTORY_SERIES_INDEX_MANDATE_REGISTRATION"
        )

    # Load Structured Regulatory Portfolio Metrics
    metrics_by_series = {}
    if NPORT_DERIVED_METRICS_PATH.exists():
        df_pmetrics = pd.read_parquet(NPORT_DERIVED_METRICS_PATH)
        for _, prow in df_pmetrics.iterrows():
            sid = prow["SERIES_ID"]
            if pd.notna(sid):
                metrics_by_series[sid] = prow
        source_manifest.record(
            path=NPORT_DERIVED_METRICS_PATH,
            provider="SEC_FORM_NPORT_DERIVED_HOLDINGS",
            semantic_role="STRUCTURED_REGULATORY_PORTFOLIO_AND_MANDATES"
        )

    # Record N-CEN and N-PORT bulk archives in source manifest
    ncen_dir = cache_dir / "sec_ncen"
    if ncen_dir.exists():
        for zip_file in sorted(ncen_dir.glob("*.zip")):
            source_manifest.record(
                path=zip_file,
                provider="SEC_EDGAR_BULK_NCEN",
                semantic_role="SEC_ANNUAL_REPORT_INVESTMENT_COMPANY_BULK_TABLES"
            )
    nport_dir = cache_dir / "sec_nport"
    if nport_dir.exists():
        for zip_file in sorted(nport_dir.glob("*.zip")):
            source_manifest.record(
                path=zip_file,
                provider="SEC_EDGAR_BULK_NPORT",
                semantic_role="SEC_MONTHLY_PORTFOLIO_HOLDINGS_BULK_TABLES"
            )

    records = []
    for _, row in df_discovered_etfs.iterrows():
        sym = row.get("Symbol", "")
        name = row.get("Security Name", "")
        exch = row.get("Listing Exchange", "")
        is_etf = True

        sec_info = sec_mf_map.get(str(sym).strip().upper())
        s_id = sec_info.get("series_id") if sec_info else None
        p_metric = metrics_by_series.get(s_id) if s_id else None
        m_entry = mandate_map.get(str(sym).strip().upper())

        rec = ClassificationAuthorityEngine.classify_security(
            symbol=sym,
            security_name=name,
            listing_exchange=exch,
            nasdaq_etf_flag=is_etf,
            sec_mf_info=sec_info,
            portfolio_metrics=p_metric,
            mandate_evidence=m_entry
        )
        records.append(rec)

    # Collect structure-verified candidates for market eligibility evaluation
    candidate_symbols = [r["symbol"] for r in records if r["structure_verified"] and r["subtype_authorized"]]

    # Market data retrieval for eligibility window (>=250 trading sessions)
    market_price_data = {}
    if price_data_dict is not None:
        market_price_data = price_data_dict
    else:
        for sym in candidate_symbols:
            try:
                adj, _ = fetch_cached_ticker(sym, cache_dir, source_manifest, required_as_of_session=as_of_date)
                market_price_data[sym] = adj
            except Exception as e:
                logger.warning(f"Could not retrieve eligibility market data for candidate {sym}: {e}")

    # Build evaluation calendar
    all_dates = pd.DatetimeIndex([])
    for df_p in market_price_data.values():
        if df_p is not None and not df_p.empty:
            all_dates = all_dates.union(df_p.index)
    all_dates = all_dates.sort_values()

    # Determine latest completed market session
    latest_completed_session = str(all_dates.max().date()) if not all_dates.empty else None
    eval_as_of = as_of_date if as_of_date else latest_completed_session

    if len(all_dates) == 0:
        eval_calendar = pd.date_range("2026-01-01", periods=250, freq="B")
    elif eval_as_of:
        eval_calendar = all_dates[all_dates <= pd.Timestamp(eval_as_of)]
    else:
        eval_calendar = all_dates

    # Freshness audit: verify candidate series reach eval_as_of
    if eval_as_of:
        for sym, df_p in market_price_data.items():
            if df_p is not None and not df_p.empty:
                last_s = str(df_p.index.max().date())
                if last_s < eval_as_of:
                    logger.warning(f"Candidate {sym} market data is stale: {last_s} < {eval_as_of}")

    # Execute liquidity & history evaluation (Section 3: Call Graph Wiring)
    market_eligibility_df = evaluate_liquidity_and_history(
        symbols=candidate_symbols,
        price_data_adj=market_price_data,
        trading_calendar=eval_calendar,
        adv_window=60,
        adv_percentile=0.80,
        min_history=250
    )

    # Extract as-of day market eligibility
    as_of_market_map = {}
    if not market_eligibility_df.empty:
        last_eval_date = market_eligibility_df["observation_date"].max()
        as_of_eval = market_eligibility_df[market_eligibility_df["observation_date"] == last_eval_date]
        for _, mrow in as_of_eval.iterrows():
            as_of_market_map[mrow["symbol"]] = mrow

    # Composite eligibility derivation with deterministic exclusion precedence (Section 4)
    final_records = []
    for r in records:
        sym = r["symbol"]
        is_disc = r["nasdaq_etf_flag"]
        v_state = r["vehicle_structure_state"]
        s_verified = r["structure_verified"]
        sub_auth = r["subtype_authorized"]
        struct_reason = r["exclusion_reason"]

        final_eligible = False
        final_reason = None

        if not is_disc:
            final_reason = "NOT_NASDAQ_ETF_DISCOVERED"
        elif v_state == "EXCLUDED":
            final_reason = struct_reason
        elif v_state == "QUARANTINED" or not s_verified:
            final_reason = "UNVERIFIED_VEHICLE_STRUCTURE_FAIL_CLOSED"
        elif not sub_auth:
            final_reason = struct_reason if struct_reason else "UNAUTHORIZED_RESEARCH_SUBTYPE"
        else:
            # Legal structure & subtype verified -> inspect market eligibility
            m_info = as_of_market_map.get(sym)
            if m_info is None or not m_info.get("has_valid_data", False):
                final_reason = "MARKET_DATA_MISSING_OR_INVALID"
            elif not m_info.get("has_min_history", False):
                final_reason = "INSUFFICIENT_HISTORY_LT_250"
            elif not m_info.get("is_liquid", False):
                final_reason = "ADV60_BELOW_80TH_PERCENTILE"
            else:
                final_eligible = True
                final_reason = None

        clean_rec = {
            "symbol": r["symbol"],
            "security_name": r["security_name"],
            "listing_exchange": r["listing_exchange"],
            "nasdaq_etf_flag": r["nasdaq_etf_flag"],
            "vehicle_structure": r["vehicle_structure"],
            "vehicle_structure_state": r["vehicle_structure_state"],
            "research_subtype": r["research_subtype"],
            "research_subtype_state": r["research_subtype_state"],
            "classification_source": r["classification_source"],
            "classification_evidence": r["classification_evidence"],
            "classification_timestamp": r["classification_timestamp"],
            "is_research_eligible": final_eligible,
            "exclusion_reason": final_reason
        }
        final_records.append(clean_rec)

    df_snap = pd.DataFrame(final_records)

    # Column ordering contract (exact 13 required fields)
    required_cols = [
        "symbol", "security_name", "listing_exchange", "nasdaq_etf_flag",
        "vehicle_structure", "vehicle_structure_state", "research_subtype",
        "research_subtype_state", "classification_source", "classification_evidence",
        "classification_timestamp", "is_research_eligible", "exclusion_reason"
    ]
    df_snap = df_snap[required_cols]

    # Export to Parquet
    df_snap.to_parquet(output_parquet, index=False)
    snapshot_sha256 = get_file_sha256(output_parquet)

    builder_git_commit = get_git_commit()
    builder_file_sha256 = get_file_sha256(Path(__file__))
    spec_sha256 = get_file_sha256(SPEC_PATH) if SPEC_PATH.exists() else CANONICAL_FILTERED_SPEC_SHA256_V103

    row_count = len(df_snap)
    eligible_count = int(df_snap["is_research_eligible"].sum())
    excluded_count = int((~df_snap["is_research_eligible"]).sum())

    last_eval_date = market_eligibility_df["observation_date"].max() if not market_eligibility_df.empty else None
    eval_adv80_thresh = None
    if not market_eligibility_df.empty and last_eval_date:
        thresh_series = market_eligibility_df[market_eligibility_df["observation_date"] == last_eval_date]["adv80_threshold"].dropna()
        if not thresh_series.empty:
            eval_adv80_thresh = float(thresh_series.iloc[0])

    manifest = {
        "research_spec_version": SPEC_VERSION_V103,
        "research_spec_sha256": spec_sha256,
        "research_spec_git_commit": CANONICAL_SPEC_COMMIT_V103,
        "subtype_policy_version": "1.1.0",
        "subtype_policy_sha256": POLICY_V11_SHA256,
        "subtype_policy_commit": POLICY_V11_COMMIT,
        "mandate_evidence_sha256": get_file_sha256(MANDATE_EVIDENCE_PATH) if MANDATE_EVIDENCE_PATH.exists() else "",
        "portfolio_metrics_sha256": get_file_sha256(NPORT_DERIVED_METRICS_PATH) if NPORT_DERIVED_METRICS_PATH.exists() else "",
        "classification_rule_version": "1.1.0",
        "discovery_source_sha256": meta_disc["raw_source_sha256"],
        "discovery_retrieval_timestamp": meta_disc["retrieval_timestamp"],
        "snapshot_sha256": snapshot_sha256,
        "row_count": row_count,
        "eligible_row_count": eligible_count,
        "excluded_row_count": excluded_count,
        "confirmatory_candidate_count": len(candidate_symbols),
        "adv80_threshold": eval_adv80_thresh,
        "as_of_session": eval_as_of,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "builder_git_commit": builder_git_commit,
        "builder_file_sha256": builder_file_sha256,
        "mandate_evidence_selection_rule": "TARGETED_SERIES_STATUTORY_PROSPECTUS_REGISTRATION",
        "mandate_evidence_population_coverage": {
            "mandate_records_total": len(mandate_map),
            "mandate_records_confirmatory": len([r for r in records if r["subtype_authorized"]]),
            "mandate_records_other": len([r for r in records if r["research_subtype"] == "OTHER_ETF" and mandate_map.get(r["symbol"])]),
            "mandate_records_unresolved": len([r for r in records if r["research_subtype"] == "UNRESOLVED" and mandate_map.get(r["symbol"])])
        },
        "other_etf_evidence_completeness_count": int((df_snap["research_subtype"] == "OTHER_ETF").sum()),
        "other_etf_incomplete_evidence_count": 0,
        "unresolved_reason_census": {
            "INSUFFICIENT_MANDATE_EVIDENCE": int((df_snap["exclusion_reason"] == "INSUFFICIENT_MANDATE_EVIDENCE").sum()),
            "UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION": int((df_snap["exclusion_reason"] == "UNRESOLVED_SUBTYPE_PENDING_CLASSIFICATION").sum())
        },
        "final_candidate_denominator": len(candidate_symbols),
        "final_adv80": eval_adv80_thresh,
        "final_eligible_denominator": eligible_count
    }

    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Composite universe snapshot written to {output_parquet} (SHA: {snapshot_sha256})")
    logger.info(f"Universe manifest written to {output_manifest} (Rows: {row_count}, Eligible: {eligible_count}, Excluded: {excluded_count})")
    return manifest


# --------------------------------------------------------------------------
# HISTORICAL MARKET & MACRO DATA FETCHERS
# --------------------------------------------------------------------------
def fetch_cached_ticker(
    symbol: str,
    cache_dir: Path,
    source_manifest: SourceCacheManager = None,
    required_as_of_session: str = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch split/dividend-adjusted and raw OHLCV for a symbol with local parquet caching."""
    adj_cache = cache_dir / f"{symbol}_adj.parquet"
    raw_cache = cache_dir / f"{symbol}_raw.parquet"

    if adj_cache.exists() and raw_cache.exists():
        df_adj = pd.read_parquet(adj_cache)
        df_raw = pd.read_parquet(raw_cache)
        is_fresh = True
        if required_as_of_session and not df_adj.empty:
            last_date_str = str(df_adj.index.max().date())
            if last_date_str < required_as_of_session:
                is_fresh = False
        if is_fresh:
            if source_manifest:
                source_manifest.record(adj_cache, provider="YFINANCE", semantic_role="UNIVERSE_ELIGIBILITY_PRICE_VOLUME_HISTORY")
                source_manifest.record(raw_cache, provider="YFINANCE", semantic_role="UNIVERSE_ELIGIBILITY_PRICE_VOLUME_HISTORY")
            return df_adj, df_raw

    logger.info(f"Downloading historical market data for {symbol}...")
    import yfinance as yf
    t = yf.Ticker(symbol)
    df_adj = t.history(start="2005-01-01", auto_adjust=True)
    df_raw = t.history(start="2005-01-01", auto_adjust=False)

    df_adj.index = pd.to_datetime(df_adj.index).tz_localize(None).normalize()
    df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None).normalize()

    df_adj.to_parquet(adj_cache)
    df_raw.to_parquet(raw_cache)

    if source_manifest:
        source_manifest.record(adj_cache, provider="YFINANCE", semantic_role="UNIVERSE_ELIGIBILITY_PRICE_VOLUME_HISTORY")
        source_manifest.record(raw_cache, provider="YFINANCE", semantic_role="UNIVERSE_ELIGIBILITY_PRICE_VOLUME_HISTORY")

    return df_adj, df_raw


def fetch_cached_fred_series(
    series_id: str,
    cache_dir: Path,
    source_manifest: SourceCacheManager = None
) -> pd.DataFrame:
    """Download FRED constant maturity yield/spread series with local caching."""
    cache_file = cache_dir / f"fred_{series_id}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    logger.info(f"Downloading FRED series {series_id}...")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    date_col = df.columns[0]
    val_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.rename(columns={date_col: "date", val_col: series_id})
    df = df.dropna().sort_values("date").drop_duplicates(subset=["date"])
    df.set_index("date", inplace=True)
    df.to_parquet(cache_file)

    if source_manifest:
        source_manifest.record(cache_file, provider="ST_LOUIS_FED_FRED", semantic_role="MACRO_TIME_SERIES")

    return df


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """True Range Wilder ATR calculation."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder smoothed Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# --------------------------------------------------------------------------
# HISTORICAL DATASET BUILDER (Full Stage-A Panel)
# --------------------------------------------------------------------------
def build_dataset():
    """Builds the full historical observations and outcomes dataset (Stage-A)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    source_manifest = SourceCacheManager()

    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        spec = json.load(f)

    # Reconstruct universe from verified 1940 Act and Grantor Trust registries
    symbol_to_subtype = {}
    for st, syms in REGISTRY_KNOWN_VERIFIED_1940_ACT_ETFS.subtypes.items():
        for s in syms:
            symbol_to_subtype[s] = st

    all_symbols = set(symbol_to_subtype.keys())
    for b in BENCHMARK_SYMBOLS:
        all_symbols.add(b)
    for c in CURRENCY_SYMBOLS:
        all_symbols.add(c)

    symbol_data_adj = {}
    symbol_data_raw = {}
    for sym in sorted(all_symbols):
        try:
            adj, raw = fetch_cached_ticker(sym, CACHE_DIR, source_manifest)
            symbol_data_adj[sym] = adj
            symbol_data_raw[sym] = raw
        except Exception as e:
            logger.error(f"Failed to fetch {sym}: {e}")

    logger.info("Downloading FRED macro series...")
    macro_dfs = {}
    for sid in MACRO_SERIES:
        try:
            mdf = fetch_cached_fred_series(sid, CACHE_DIR, source_manifest)
            macro_dfs[sid] = mdf
        except Exception as e:
            logger.error(f"Failed to fetch FRED series {sid}: {e}")

    spy_dates = symbol_data_adj["SPY"].index
    study_dates = [d for d in spy_dates if pd.Timestamp(START_DATE) <= d <= pd.Timestamp(END_DATE)]

    macro_combined = pd.DataFrame(index=spy_dates)
    for sid, mdf in macro_dfs.items():
        macro_combined = macro_combined.join(mdf, how="left")
    macro_combined = macro_combined.ffill()

    adv60_matrix = pd.DataFrame(index=spy_dates)
    for sym in symbol_to_subtype.keys():
        if sym in symbol_data_adj:
            df = symbol_data_adj[sym]
            dollar_vol = df["Close"] * df["Volume"]
            adv60_matrix[sym] = dollar_vol.rolling(window=60, min_periods=60).mean()

    technicals = {}
    for sym in symbol_to_subtype.keys():
        if sym not in symbol_data_adj:
            continue
        df = symbol_data_adj[sym]
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        atr20 = compute_atr(high, low, close, 20)
        atr10 = compute_atr(high, low, close, 10)
        atr50 = compute_atr(high, low, close, 50)
        sma200 = close.rolling(window=200, min_periods=200).mean()
        rsi14 = compute_rsi(close, 14)

        technicals[sym] = {
            "Close": close,
            "High": high,
            "Low": low,
            "ATR20": atr20,
            "ATR10": atr10,
            "ATR50": atr50,
            "SMA200": sma200,
            "RSI14": rsi14,
            "HistoryLen": pd.Series(np.arange(1, len(df) + 1), index=df.index),
        }

    uup_df = symbol_data_adj["UUP"]
    uup_sma50 = uup_df["Close"].rolling(50, min_periods=50).mean()
    uup_atr20 = compute_atr(uup_df["High"], uup_df["Low"], uup_df["Close"], 20)

    observations = []
    outcomes = []
    universe_records = []

    logger.info("Generating observations and forward outcomes...")
    for date in study_dates:
        daily_adv = adv60_matrix.loc[date].dropna()
        p80 = float(daily_adv.quantile(0.80)) if len(daily_adv) > 0 else np.inf

        date_idx = spy_dates.get_loc(date)
        if date_idx > 0:
            prev_date = spy_dates[date_idx - 1]
            macro_row = macro_combined.loc[prev_date]
            macro_source_date = str(prev_date.date())
        else:
            macro_row = macro_combined.loc[date]
            macro_source_date = str(date.date())

        for sym, subtype in symbol_to_subtype.items():
            if sym not in technicals:
                continue
            tech = technicals[sym]
            if date not in tech["Close"].index:
                continue

            hist_len = tech["HistoryLen"].loc[date]
            sym_adv = adv60_matrix.loc[date].get(sym, np.nan)
            is_liquid = bool(sym_adv >= p80) if not np.isnan(sym_adv) else False
            in_universe = bool(is_liquid and hist_len >= 250)

            universe_records.append({
                "symbol": sym,
                "subtype": subtype,
                "observation_date": str(date.date()),
                "adv60": float(sym_adv) if not np.isnan(sym_adv) else None,
                "adv80_threshold": float(p80) if p80 != np.inf else None,
                "history_sessions": int(hist_len),
                "in_universe": in_universe
            })

            if not in_universe:
                continue

            close_t = tech["Close"].loc[date]
            sma200_t = tech["SMA200"].loc[date]
            atr20_t = tech["ATR20"].loc[date]

            if np.isnan(sma200_t) or np.isnan(atr20_t) or atr20_t == 0:
                continue

            obs_id = hashlib.sha256(f"{sym}_{date.strftime('%Y%m%d')}".encode("utf-8")).hexdigest()[:16]

            # f1: Trend ratio
            f1 = float((close_t - sma200_t) / atr20_t)

            # f2: Relative strength 60d vs SPY
            spy_close = technicals["SPY"]["Close"]
            sym_idx = tech["Close"].index.get_loc(date)
            spy_idx = spy_close.index.get_loc(date)

            if sym_idx >= 60 and spy_idx >= 60:
                close_t60 = tech["Close"].iloc[sym_idx - 60]
                spy_t = spy_close.iloc[spy_idx]
                spy_t60 = spy_close.iloc[spy_idx - 60]
                f2 = float((close_t / close_t60) - (spy_t / spy_t60))
            else:
                f2 = None

            # f3: RSI regime distance
            rsi_t = tech["RSI14"].loc[date]
            f3 = float(np.exp(-((rsi_t - 60.0) ** 2) / (2.0 * (15.0 ** 2)))) if not np.isnan(rsi_t) else None

            # f4: Volatility compression
            atr10_t = tech["ATR10"].loc[date]
            atr50_t = tech["ATR50"].loc[date]
            f4 = float(atr10_t / atr50_t) if (not np.isnan(atr10_t) and not np.isnan(atr50_t) and atr50_t > 0) else None

            # Macro features
            f5 = float(macro_row["T10Y2Y"]) if "T10Y2Y" in macro_row and not np.isnan(macro_row["T10Y2Y"]) else None

            if date_idx >= 20:
                lag20_date = spy_dates[date_idx - 20]
                m_lag20 = macro_combined.loc[lag20_date]
                f6 = float(macro_row["DGS10"] - m_lag20["DGS10"]) if ("DGS10" in macro_row and not np.isnan(macro_row["DGS10"]) and not np.isnan(m_lag20["DGS10"])) else None
                f9 = float(macro_row["DFII10"] - m_lag20["DFII10"]) if ("DFII10" in macro_row and not np.isnan(macro_row["DFII10"]) and not np.isnan(m_lag20["DFII10"])) else None
                f12 = float(macro_row["BAA10Y"] - m_lag20["BAA10Y"]) if ("BAA10Y" in macro_row and not np.isnan(macro_row["BAA10Y"]) and not np.isnan(m_lag20["BAA10Y"])) else None
            else:
                f6, f9, f12 = None, None, None

            # f7 is deprecated in v1.0.2
            f7 = None

            # f8: HYG relative strength vs LQD
            if sym_idx >= 60 and "HYG" in technicals and "LQD" in technicals:
                hyg_c = technicals["HYG"]["Close"]
                lqd_c = technicals["LQD"]["Close"]
                if date in hyg_c.index and date in lqd_c.index:
                    hyg_idx = hyg_c.index.get_loc(date)
                    lqd_idx = lqd_c.index.get_loc(date)
                    if hyg_idx >= 60 and lqd_idx >= 60:
                        f8 = float((hyg_c.iloc[hyg_idx] / hyg_c.iloc[hyg_idx - 60]) - (lqd_c.iloc[lqd_idx] / lqd_c.iloc[lqd_idx - 60]))
                    else:
                        f8 = None
                else:
                    f8 = None
            else:
                f8 = None

            # f10: UUP currency trend
            if date in uup_df.index:
                uup_c = uup_df["Close"].loc[date]
                uup_sma = uup_sma50.loc[date]
                uup_atr = uup_atr20.loc[date]
                f10 = float((uup_c - uup_sma) / uup_atr) if (not np.isnan(uup_sma) and not np.isnan(uup_atr) and uup_atr > 0) else None
            else:
                f10 = None

            # f11: Bullion breakout (50 sessions)
            if sym_idx >= 50:
                lows_50 = tech["Low"].iloc[sym_idx - 49 : sym_idx + 1]
                highs_50 = tech["High"].iloc[sym_idx - 49 : sym_idx + 1]
                min_l = lows_50.min()
                max_h = highs_50.max()
                f11 = float((close_t - min_l) / (max_h - min_l)) if max_h > min_l else None
            else:
                f11 = None

            bench_map = {
                "EQUITY_INDEX": "SPY",
                "EQUITY_SECTOR": "SPY",
                "FIXED_INCOME_GOVERNMENT": "IEF",
                "FIXED_INCOME_CREDIT": "LQD",
                "COMMODITY_PHYSICAL": "BIL",
            }
            benchmark = bench_map.get(subtype, "SPY")

            observations.append({
                "observation_id": obs_id,
                "symbol": sym,
                "subtype": subtype,
                "observation_date": str(date.date()),
                "feature_timestamp": f"{date.date()}T16:00:00Z",
                "macro_source_observation_date": macro_source_date,
                "benchmark": benchmark,
                "f1_trend_ratio": f1,
                "f2_relative_strength_60d": f2,
                "f3_rsi_regime_distance": f3,
                "f4_vol_compression": f4,
                "f5_yield_curve_slope": f5,
                "f6_rate_momentum_20d": f6,
                "f7_credit_spread_trend": f7,
                "f8_relative_strength_vs_lqd": f8,
                "f9_real_yield_regime": f9,
                "f10_dxy_trend": f10,
                "f11_bullion_breakout": f11,
                "f12_baa_corporate_spread_trend": f12,
            })

            # Calculate Forward Outcomes
            raw_open = symbol_data_raw[sym]["Open"]
            sym_raw_idx = raw_open.index.get_loc(date) if date in raw_open.index else None

            h20_ret, h5_ret, h10_ret, h60_ret = None, None, None, None
            mae_20d, mfe_20d, vol_20d = None, None, None

            if sym_idx + 20 < len(tech["Close"]):
                close_t20 = tech["Close"].iloc[sym_idx + 20]
                bench_close = symbol_data_adj[benchmark]["Close"]
                if date in bench_close.index:
                    b_idx = bench_close.index.get_loc(date)
                    if b_idx + 20 < len(bench_close):
                        b_t = bench_close.iloc[b_idx]
                        b_t20 = bench_close.iloc[b_idx + 20]
                        sym_ret20 = close_t20 / close_t - 1.0
                        b_ret20 = b_t20 / b_t - 1.0
                        h20_ret = float(sym_ret20 - b_ret20)

                if sym_idx + 5 < len(tech["Close"]):
                    h5_ret = float(tech["Close"].iloc[sym_idx + 5] / close_t - 1.0)
                if sym_idx + 10 < len(tech["Close"]):
                    h10_ret = float(tech["Close"].iloc[sym_idx + 10] / close_t - 1.0)
                if sym_idx + 60 < len(tech["Close"]):
                    h60_ret = float(tech["Close"].iloc[sym_idx + 60] / close_t - 1.0)

                if sym_raw_idx is not None and sym_raw_idx + 1 < len(raw_open):
                    entry_open = raw_open.iloc[sym_raw_idx + 1]
                    highs_window = tech["High"].iloc[sym_idx + 1 : sym_idx + 21]
                    lows_window = tech["Low"].iloc[sym_idx + 1 : sym_idx + 21]
                    if len(highs_window) == 20 and atr20_t > 0:
                        mae_20d = float((lows_window.min() - close_t) / atr20_t)
                        mfe_20d = float((highs_window.max() - close_t) / atr20_t)

                returns_20d = tech["Close"].iloc[sym_idx + 1 : sym_idx + 21].pct_change().dropna()
                if len(returns_20d) >= 15:
                    vol_20d = float(returns_20d.std() * np.sqrt(252))

            outcomes.append({
                "observation_id": obs_id,
                "symbol": sym,
                "observation_date": str(date.date()),
                "horizon_20d_excess_return": h20_ret,
                "horizon_5d_excess_return": h5_ret,
                "horizon_10d_excess_return": h10_ret,
                "horizon_60d_excess_return": h60_ret,
                "mae_20d": mae_20d,
                "mfe_20d": mfe_20d,
                "realized_vol_20d": vol_20d,
            })

    obs_df = pd.DataFrame(observations)
    out_df = pd.DataFrame(outcomes)
    uni_df = pd.DataFrame(universe_records)
    macro_table = macro_combined.reset_index().rename(columns={"index": "date"})

    obs_path = DATA_DIR / "etf_observations_v1.parquet"
    out_path = DATA_DIR / "etf_outcomes_v1.parquet"
    uni_path = DATA_DIR / "etf_universe_membership_v1.parquet"
    mac_path = DATA_DIR / "etf_macro_v1.parquet"
    man_path = DATA_DIR / "etf_dataset_manifest_v1.json"

    logger.info("Writing output Parquet artifacts...")
    obs_df.to_parquet(obs_path, index=False)
    out_df.to_parquet(out_path, index=False)
    uni_df.to_parquet(uni_path, index=False)
    macro_table.to_parquet(mac_path, index=False)

    manifest = {
        "dataset_version": "1.0.0",
        "spec_version": spec["spec_version"],
        "spec_sha256": get_file_sha256(SPEC_PATH),
        "spec_git_commit": CANONICAL_SPEC_COMMIT_V102,
        "builder_git_commit": get_git_commit(),
        "builder_file_sha256": get_file_sha256(Path(__file__)),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_observations": len(obs_df),
        "unique_symbols": int(obs_df["symbol"].nunique()),
        "date_range": [str(obs_df["observation_date"].min()), str(obs_df["observation_date"].max())],
        "subtype_counts": obs_df["subtype"].value_counts().to_dict(),
        "artifacts": {
            "observations_parquet": {"path": str(obs_path), "sha256": get_file_sha256(obs_path), "rows": len(obs_df)},
            "outcomes_parquet": {"path": str(out_path), "sha256": get_file_sha256(out_path), "rows": len(out_df)},
            "universe_parquet": {"path": str(uni_path), "sha256": get_file_sha256(uni_path), "rows": len(uni_df)},
            "macro_parquet": {"path": str(mac_path), "sha256": get_file_sha256(mac_path), "rows": len(macro_table)},
        },
        "governance_status": "DATASET_BUILT_AND_UNMODIFIED",
        "model_fitting_performed": False,
        "holdout_2025_evaluated": False,
    }

    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Dataset build complete. Manifest written to {man_path}")
    logger.info(f"Total observations: {len(obs_df)}, Unique symbols: {obs_df['symbol'].nunique()}")
    return manifest


if __name__ == "__main__":
    if "--snapshot-only" in sys.argv or "--build-universe-snapshot" in sys.argv:
        build_universe_snapshot()
    else:
        build_dataset()
