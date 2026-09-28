import json
import pytest
from pathlib import Path
from scripts.research.etf_v2.identity_authority import IdentityAuthority
from scripts.research.etf_v2.models import EntityIdentity

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_normalization_idempotence():
    """Verify bounded fixed-point decoding and normalization idempotence: N(N(x)) == N(x)."""
    adversarial_fixtures = [
        # Double / recursive entity encoding
        ("&amp;amp;", "&"),
        ("&amp;#38;", "&"),
        ("&amp;#8211;", "-"),
        ("&amp;nbsp;", ""),  # Whitespace collapsed and trimmed
        ("Hello&nbsp;&amp;amp;&nbsp;World", "Hello & World"),
        ("S&amp;amp;P 500&#8211;Index&#174;", "S&P 500-Index"),
        ("State&nbsp;Street&reg; SPDR&trade;", "State Street SPDR"),
        ("Barron&#8217;s &#8220;Top&#8221; 100", "Barron's \"Top\" 100"),
        ("Small&#47;Mid&#58;Cap", "Small/Mid:Cap"),
        # Triple encoding
        ("&amp;amp;amp;", "&"),
        # Dashes
        ("FT Vest – April — August − September ‐ October", "FT Vest - April - August - September - October"),
        # Windows-1252 / CP1252 byte-like entities
        ("Alpha\x96Beta\x97Gamma", "Alpha-Beta-Gamma"),
        # Zero-width / BOM
        ("Zero\u200bWidth\ufeffSpace", "ZeroWidthSpace"),
        # Trademark & service marks
        ("Fidelity(R) Fund(TM) SPDR(r)", "Fidelity Fund SPDR"),
    ]

    for raw, expected in adversarial_fixtures:
        n1 = IdentityAuthority.normalize_for_matching(raw)
        n2 = IdentityAuthority.normalize_for_matching(n1)
        assert n1 == expected, f"Failed raw -> expected: {raw!r} -> {n1!r} != {expected!r}"
        assert n2 == n1, f"Failed idempotence N(N(x)) == N(x): {n1!r} != {n2!r}"


def test_manifest_population_normalized_name_collisions():
    """Empirical Collision Oracle: verify zero false name collisions across the entire 2,884 population."""
    manifest_path = REPO_ROOT / "docs/research/ETF_MANDATE_INPUT_MANIFEST_V1.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    targets = manifest if isinstance(manifest, list) else manifest.get("targets", manifest.get("records", []))
    assert len(targets) == 2884, f"Expected 2884 targets, got {len(targets)}"

    seen_names = {}
    collisions = []
    for t in targets:
        sym = t["symbol"]
        norm = IdentityAuthority.normalize_for_matching(t["legal_name"]).lower()
        if norm in seen_names:
            collisions.append((sym, seen_names[norm], norm))
        else:
            seen_names[norm] = sym

    assert len(collisions) == 0, f"Observed false identity matches in population: {collisions}"


def test_negative_collision_cross_fund_isolation():
    """Verify that similar funds differing only by month or category do not cross-match."""
    # APRP (April) vs AUGP (August)
    apr_identity = EntityIdentity(
        symbol="APRP",
        cik="0001992104",
        series_id="S000083281",
        class_id="C000246808",
        legal_name="PGIM S&P 500 Buffer 12 ETF - April",
    )
    aug_text = "PGIM S&amp;P 500 Buffer 12 ETF &#8211; August"
    res = IdentityAuthority.match_identity(aug_text, apr_identity)
    assert not res["has_name"], "APRP (April) falsely matched August text"
    assert not res["is_qualified"], "APRP falsely qualified on August text"

    # APXM (Max Buffer April) vs AUGM (Max Buffer August)
    apxm_identity = EntityIdentity(
        symbol="APXM",
        cik="0001667919",
        series_id="S000084534",
        class_id="C000248882",
        legal_name="FT Vest U.S. Equity Max Buffer ETF - April",
    )
    augm_text = "FT Vest U.S. Equity Max Buffer ETF &#8211; August"
    res_apxm = IdentityAuthority.match_identity(augm_text, apxm_identity)
    assert not res_apxm["has_name"], "APXM (April) falsely matched August text"
    assert not res_apxm["is_qualified"], "APXM falsely qualified on August text"
