import json
import pytest
from pathlib import Path
from scripts.research.etf_v2.filing_universe import FilingUniverse
from scripts.research.etf_v2.identity_authority import IdentityAuthority
from scripts.research.etf_v2.prospectus_authority import ProspectusAuthorityResolver
from scripts.research.etf_v2.pipeline import ETFPipelineV2

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_lane_b_candidate_discovery_and_qualification():
    sub_dir = REPO_ROOT / "data/research/cache/sec_submissions"
    prosp_dir = REPO_ROOT / "data/research/cache/sec_prospectus"
    fu = FilingUniverse(sub_dir, prosp_dir)
    pipe = ETFPipelineV2(REPO_ROOT)

    bounded_path = REPO_ROOT / "docs/research/ETF_V2_860_BOUNDED_UNIVERSE.json"
    with open(bounded_path, "r", encoding="utf-8") as f:
        bounded_doc = json.load(f)
    bounded_targets = {t["symbol"]: t for t in bounded_doc["targets"]}

    lane_b_expected = {
        "AFOS": "0001592900-25-001694",
        "BUFH": "0001445546-25-004326",
        "HOYY": "0001493152-25-013249",
        "IOYY": "0001493152-25-013250",
        "RTYY": "0001493152-25-013261",
        "SMYY": "0001493152-25-013263",
    }

    for sym, exp_acc in lane_b_expected.items():
        b = bounded_targets[sym]
        identity = IdentityAuthority.create_identity(
            symbol=sym,
            cik=b["cik"],
            series_id=b["series_id"],
            class_id=b["class_id"],
            legal_name=b["legal_name"],
        )
        candidates = fu.get_candidate_prospectuses(identity)
        cand_accs = [c.accession for c in candidates]
        assert exp_acc in cand_accs, f"{sym} candidate discovery failed to include {exp_acc}"

        prosp_auth = ProspectusAuthorityResolver.resolve_authority(candidates, identity)
        assert prosp_auth is not None, f"{sym} failed prospectus authority resolution"
        assert prosp_auth.filing.accession == exp_acc, f"{sym} selected {prosp_auth.filing.accession} != {exp_acc}"
        assert prosp_auth.document_role == "SUMMARY_PROSPECTUS"

        rec = pipe.process_target(identity)
        assert rec is not None
        assert rec.final_classification == "NON_CONFIRMATORY"
