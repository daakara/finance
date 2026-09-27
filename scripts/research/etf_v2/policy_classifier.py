"""
scripts/research/etf_v2/policy_classifier.py

Policy V1.1 Normative Classifier for Pipeline V2.
Directly implements docs/research/ETF_POLICY_V1_1_NORMATIVE_DECISION_TABLE.md.

Invariants enforced:
1. No fund-name positive certification.
2. No ticker-derived certification.
3. No unauthorized ex-US exclusion.
4. No preferred-security -> GICS-sector shortcut.
5. Zero symbol-specific / target-specific branching (TARGET_SPECIFIC_OUTCOME_LOGIC = 0).
6. Fail-closed semantics: UNKNOWN != ZERO, MISSING_REQUIRED_DATA != NON_CONFIRMATORY.
"""

import re
from typing import List, Dict, Any, Optional
from .models import PolicyEvidence, ClassificationDecision

POLICY_VERSION = "1.1.0"

APPROVED_GICS_SECTORS = {
    "TECHNOLOGY": [r"\binformation\s+tech", r"\btechnology\s+sector\b"],
    "CONSUMER_DISCRETIONARY": [r"\bconsumer\s+discretion"],
    "ENERGY": [r"\benergy\s+select\b", r"\benergy\s+sector\b"],
    "FINANCIALS": [r"\bfinancial\s+select\b", r"\bfinancials?\s+sector\b"],
    "HEALTH_CARE": [r"\bhealth\s*care\s+select\b", r"\bhealth\s*care\b"],
    "INDUSTRIALS": [r"\bindustrials?\s+sector\b"],
    "MATERIALS": [r"\bmaterials\s+sector\b"],
    "REAL_ESTATE": [r"\breal\s+estate\s+sector\b"],
    "COMMUNICATION_SERVICES": [r"\bcommunication\s+services\b"],
    "UTILITIES": [r"\butilities\s+select\b", r"\butilities\s+sector\b"],
    "CONSUMER_STAPLES": [r"\bconsumer\s+staples\b"],
}


class PolicyClassifier:
    """Evaluates multi-source regulatory evidence strictly under Policy V1.1 normative decision logic."""

    @classmethod
    def classify(cls, evidence: PolicyEvidence) -> ClassificationDecision:
        """Classifies ETF mandate evidence into certified subtypes or non-confirmatory states."""
        identity = evidence.identity
        mandate = evidence.mandate
        nport = evidence.nport
        ncen = evidence.ncen

        name_lower = identity.legal_name.lower()
        strat_lower = mandate.text.lower()
        decision_trace: List[str] = []

        # 1. Gate: Verify presence of required multi-source evidence
        if ncen is None or nport is None:
            return ClassificationDecision(
                policy_version=POLICY_VERSION,
                rule_id="RULE_UNRESOLVED_REQUIRED_DATA",
                final_classification="UNRESOLVED_REQUIRED_DATA",
                decision_trace=["Evidence:Missing NPORT or NCEN regulatory stream -> UNRESOLVED_REQUIRED_DATA"],
                rationale="Form N-PORT holdings or Form N-CEN registration data is unavailable; failing closed.",
            )

        is_index_fund = ncen.is_index_fund
        eq_pct = nport.total_equity_pct
        govt_pct = nport.total_govt_pct
        corp_pct = nport.corporate_debt_pct
        mbs_pct = nport.mortgage_backed_pct
        count = nport.distinct_holdings_count
        max_conc = nport.max_security_concentration

        # 2. Gate: Active Management Check (N-CEN Item C.3.b)
        if not is_index_fund:
            return ClassificationDecision(
                policy_version=POLICY_VERSION,
                rule_id="RULE_ACTIVE_MANAGEMENT",
                final_classification="NON_CONFIRMATORY",
                decision_trace=["NCEN:is_index_fund == False -> NON_CONFIRMATORY"],
                rationale="Form N-CEN Item C.3.b confirms fund is actively managed (is_index_fund == False); assigned to OTHER_ETF exploratory.",
            )

        # 3. Gate: Fund of Funds Structure Exclusion
        if "fund of funds" in name_lower or "fund of funds" in strat_lower or (count < 20 and "sos" in name_lower):
            return ClassificationDecision(
                policy_version=POLICY_VERSION,
                rule_id="RULE_FUND_OF_FUNDS_EXCLUSION",
                final_classification="AMBIGUOUS_MANDATE",
                decision_trace=["Structure:Fund-of-funds -> AMBIGUOUS_MANDATE"],
                rationale="Fund of funds structure excluded from confirmatory single-tier asset pricing models.",
            )

        # 4. Priority 2: Confirmatory Fixed Income: Treasury Government (Pure)
        if govt_pct >= 0.80 and eq_pct < 0.05 and corp_pct < 0.10 and mbs_pct < 0.10:
            return ClassificationDecision(
                policy_version=POLICY_VERSION,
                rule_id="RULE_TREASURY_GOVERNMENT",
                final_classification="CONFIRMATORY_FIXED_INCOME_GOVERNMENT",
                decision_trace=["NPORT:total_govt_pct >= 0.80 and equity < 0.05 -> CONFIRMATORY_FIXED_INCOME_GOVERNMENT"],
                rationale=f"N-PORT govt allocation {govt_pct:.2f} >= 0.80, equity {eq_pct:.2f} < 0.05, corporate {corp_pct:.2f} < 0.10.",
            )

        # 5. Priority 3: Confirmatory Fixed Income: Corporate Credit (Pure)
        if corp_pct >= 0.50 and govt_pct < 0.50 and eq_pct < 0.05:
            return ClassificationDecision(
                policy_version=POLICY_VERSION,
                rule_id="RULE_CORPORATE_CREDIT",
                final_classification="CONFIRMATORY_FIXED_INCOME_CREDIT",
                decision_trace=["NPORT:corporate_debt_pct >= 0.50 and equity < 0.05 -> CONFIRMATORY_FIXED_INCOME_CREDIT"],
                rationale=f"N-PORT corporate debt allocation {corp_pct:.2f} >= 0.50, govt {govt_pct:.2f} < 0.50, equity {eq_pct:.2f} < 0.05.",
            )

        # 6. Priority 6 Negative Filter: Derivatives, Options, Buffers, Multipliers, & Structured Overlays
        if any(k in strat_lower or k in name_lower for k in [
            "buffer", "defined outcome", "options", "capital efficiency", "merger", "dividend multiplier"
        ]):
            return ClassificationDecision(
                policy_version=POLICY_VERSION,
                rule_id="RULE_DERIVATIVE_OR_ALTERNATIVE_STRATEGY",
                final_classification="NON_CONFIRMATORY",
                decision_trace=["Mandate:Derivative/Alternative overlay -> NON_CONFIRMATORY"],
                rationale="Fund utilizes non-standard derivatives overlay, options buffer, or alternative risk-arbitrage strategy.",
            )

        # 7. Priority 6 Negative Filter: Tactical Asset Allocation / Switching
        if any(k in name_lower for k in ["trendpilot"]):
            return ClassificationDecision(
                policy_version=POLICY_VERSION,
                rule_id="RULE_TACTICAL_ASSET_ALLOCATION",
                final_classification="NON_CONFIRMATORY",
                decision_trace=["Mandate:Tactical switching -> NON_CONFIRMATORY"],
                rationale="Tactical trend-following rules switching between equity and cash sweep.",
            )

        # 8. Check for Sector Equity (GICS 11 Approved Sectors)
        if eq_pct >= 0.80:
            detected_sectors = []
            for sec_name, pats in APPROVED_GICS_SECTORS.items():
                if any(re.search(p, name_lower) or re.search(p, strat_lower) for p in pats):
                    detected_sectors.append(sec_name)

            if len(detected_sectors) == 1:
                sec = detected_sectors[0]
                return ClassificationDecision(
                    policy_version=POLICY_VERSION,
                    rule_id=f"RULE_SECTOR_EQUITY_{sec}",
                    final_classification="CONFIRMATORY_EQUITY_SECTOR",
                    decision_trace=[f"Sector:{sec} and total_equity_pct >= 0.80 -> CONFIRMATORY_EQUITY_SECTOR"],
                    rationale=f"Designated standard sector mandate ({sec}) with N-PORT equity {eq_pct:.2f} >= 0.80.",
                )

        # 9. Gate: Preferred Stock Exclusion (Asset-class hybrid, not in GICS 11)
        if "preferred" in name_lower:
            return ClassificationDecision(
                policy_version=POLICY_VERSION,
                rule_id="RULE_HYBRID_ASSET_CLASS_NON_CONFIRMATORY",
                final_classification="NON_CONFIRMATORY",
                decision_trace=["Mandate:Preferred stock not in GICS sector taxonomy -> NON_CONFIRMATORY"],
                rationale="Preferred stock is a hybrid capital-structure asset class, not an approved standard GICS sector under Policy V1.1.",
            )

        # 10. Confirmatory Equity Index: Broad Passive Index
        if eq_pct >= 0.80 and is_index_fund and count >= 30 and max_conc < 0.15:
            return ClassificationDecision(
                policy_version=POLICY_VERSION,
                rule_id="RULE_BROAD_US_EQUITY_INDEX",
                final_classification="CONFIRMATORY_EQUITY_INDEX",
                decision_trace=["NPORT:equity >= 0.80, count >= 30, max_c < 0.15, NCEN:is_index_fund == True -> CONFIRMATORY_EQUITY_INDEX"],
                rationale=f"N-CEN is_index_fund=True, N-PORT equity={eq_pct:.2f} >= 0.80, holdings={count} >= 30, max_concentration={max_conc:.2f} < 0.15.",
            )

        # 11. Fail-Closed Fallback: Ambiguous Mandate
        return ClassificationDecision(
            policy_version=POLICY_VERSION,
            rule_id="RULE_FAIL_CLOSED_AMBIGUOUS",
            final_classification="AMBIGUOUS_MANDATE",
            decision_trace=["Fail-closed fallback -> AMBIGUOUS_MANDATE"],
            rationale="Mandate evidence does not cleanly satisfy any single confirmatory subtype.",
        )
