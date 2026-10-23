#!/usr/bin/env python3
"""
ARX Terminal — Phase 2 Observer Calibration Scoring Utility (Version 2.0.0)
Authority: ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence
Mandate:
  - Measurement A: Observer vs Authoritative Gold Key Calibration Scoring
  - Measurement B: Observer 1 vs Observer 2 Pre-Adjudication Inter-Rater Reliability
  - Measurement C: Critical Decision-Safety Error Recognition & Coverage
Features:
  - Multi-label classification across CM-01..CM-09 and SM-01..SM-06
  - Supplemental dimensions: interpretation_state and action_state
  - Correct statistical handling of undefined Cohen's Kappa when Pe >= 1.0 (never fabricated as 1.0)
  - Strict input validation: duplicate IDs, missing trials, invalid taxonomy codes
  - Comprehensive dispute logging and raw score preservation
"""

import sys
import os
import json
import argparse
import math
from typing import Dict, List, Set, Any, Tuple, Optional

CANONICAL_CM_CODES = [
    "CM-01", "CM-02", "CM-03", "CM-04", "CM-05", "CM-06", "CM-07", "CM-08", "CM-09"
]

SUPPLEMENTAL_SM_CODES = [
    "SM-01", "SM-02", "SM-03", "SM-04", "SM-05", "SM-06"
]

ALL_CODES = CANONICAL_CM_CODES + SUPPLEMENTAL_SM_CODES

VALID_INTERPRETATION_STATES = {
    "INTERPRETATION_CORRECT", "INTERPRETATION_INCORRECT", "EVIDENCE_INSUFFICIENT"
}

VALID_ACTION_STATES = {
    "ACTION_APPROPRIATE", "ACTION_INAPPROPRIATE", "EVIDENCE_INSUFFICIENT"
}

def validate_ratings_payload(data: Dict[str, Any], context_label: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Validate structure, IDs, and taxonomy codes of an observer ratings file."""
    if not isinstance(data, dict):
        raise ValueError(f"[{context_label}] Root payload must be a JSON object.")
        
    observer_id = data.get("observer_id")
    if not observer_id or not isinstance(observer_id, str):
        raise ValueError(f"[{context_label}] Missing or invalid 'observer_id'.")
        
    ratings_list = data.get("ratings")
    if not isinstance(ratings_list, list) or len(ratings_list) == 0:
        raise ValueError(f"[{context_label}] 'ratings' must be a non-empty list.")
        
    seen_trials = set()
    ratings_by_trial = {}
    
    for idx, item in enumerate(ratings_list):
        if not isinstance(item, dict):
            raise ValueError(f"[{context_label}] Rating at index {idx} must be a dictionary.")
            
        tid = item.get("trial_id")
        if not tid or not isinstance(tid, str):
            raise ValueError(f"[{context_label}] Rating at index {idx} missing valid 'trial_id'.")
            
        if tid in seen_trials:
            raise ValueError(f"[{context_label}] Duplicate trial_id detected: '{tid}'.")
        seen_trials.add(tid)
        
        # Validate codes
        cm_codes = item.get("assigned_cm_codes", [])
        if not isinstance(cm_codes, list):
            raise ValueError(f"[{context_label}] Trial '{tid}' assigned_cm_codes must be a list.")
        for code in cm_codes:
            if code not in ALL_CODES:
                raise ValueError(f"[{context_label}] Trial '{tid}' contains invalid taxonomy code '{code}'.")
                
        sm_codes = item.get("assigned_sm_codes", [])
        if not isinstance(sm_codes, list):
            raise ValueError(f"[{context_label}] Trial '{tid}' assigned_sm_codes must be a list.")
        for code in sm_codes:
            if code not in ALL_CODES:
                raise ValueError(f"[{context_label}] Trial '{tid}' contains invalid taxonomy code '{code}'.")
                
        # Optional validation of supplemental dimensions
        interp = item.get("interpretation_state")
        if interp and interp not in VALID_INTERPRETATION_STATES:
            raise ValueError(f"[{context_label}] Trial '{tid}' invalid interpretation_state '{interp}'.")
            
        act = item.get("action_state")
        if act and act not in VALID_ACTION_STATES:
            raise ValueError(f"[{context_label}] Trial '{tid}' invalid action_state '{act}'.")
            
        ratings_by_trial[tid] = item
        
    return observer_id, ratings_by_trial

def validate_gold_key_payload(data: Dict[str, Any]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Validate structure, IDs, and taxonomy codes of an authoritative gold key file."""
    if not isinstance(data, dict):
        raise ValueError("[Gold Key] Root payload must be a JSON object.")
        
    gold_key_id = data.get("gold_key_id")
    if not gold_key_id or not isinstance(gold_key_id, str):
        raise ValueError("[Gold Key] Missing or invalid 'gold_key_id'.")
        
    items = data.get("items")
    if not isinstance(items, dict) or len(items) == 0:
        raise ValueError("[Gold Key] 'items' must be a non-empty dictionary keyed by trial_id.")
        
    for tid, item in items.items():
        if not isinstance(item, dict):
            raise ValueError(f"[Gold Key] Item '{tid}' must be a dictionary.")
            
        cm_codes = item.get("authoritative_cm_codes", [])
        for c in cm_codes:
            if c not in ALL_CODES:
                raise ValueError(f"[Gold Key] Trial '{tid}' has invalid canonical code '{c}'.")
                
        sm_codes = item.get("authoritative_sm_codes", [])
        for c in sm_codes:
            if c not in ALL_CODES:
                raise ValueError(f"[Gold Key] Trial '{tid}' has invalid supplemental code '{c}'.")
                
        interp = item.get("interpretation_state")
        if interp and interp not in VALID_INTERPRETATION_STATES:
            raise ValueError(f"[Gold Key] Trial '{tid}' has invalid interpretation_state '{interp}'.")
            
        act = item.get("action_state")
        if act and act not in VALID_ACTION_STATES:
            raise ValueError(f"[Gold Key] Trial '{tid}' has invalid action_state '{act}'.")
            
    return gold_key_id, items

def compute_contingency(r1_labels: List[Set[str]], r2_labels: List[Set[str]], code: str) -> Tuple[int, int, int, int]:
    """
    Returns (a, b, c, d) for a specific binary category:
    a: True Positive / Agreement Positive (both present)
    b: Rater 1 Positive, Rater 2 Negative
    c: Rater 1 Negative, Rater 2 Positive
    d: True Negative / Agreement Negative (both absent)
    """
    a = b = c = d = 0
    for s1, s2 in zip(r1_labels, r2_labels):
        has1 = code in s1
        has2 = code in s2
        if has1 and has2:
            a += 1
        elif has1 and (not has2):
            b += 1
        elif (not has1) and has2:
            c += 1
        else:
            d += 1
    return a, b, c, d

def calculate_cohens_kappa(a: int, b: int, c: int, d: int) -> Tuple[float, float, Optional[float], bool]:
    """
    Calculate observed agreement (Po), expected agreement (Pe), and chance-corrected Cohen's Kappa.
    Statistical Rule: If Pe >= 1.0 (e.g. 100% negative marginals), chance correction is mathematically
    undefined (0/0). In this case, kappa is None (reported as UNDEFINED), and is_defined = False.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0, 1.0, None, False
        
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    
    # Check floating point proximity to 1.0
    if pe >= 0.9999999:
        # Mathematically undefined
        return round(po, 4), round(pe, 4), None, False
        
    kappa = (po - pe) / (1.0 - pe)
    return round(po, 4), round(pe, 4), round(kappa, 4), True

def evaluate_observer_vs_gold(gold_path: str, observer_path: str, criterion: float = 0.85) -> Dict[str, Any]:
    """Measurement A: Evaluate individual observer accuracy against authoritative gold key."""
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_json = json.load(f)
    with open(observer_path, "r", encoding="utf-8") as f:
        obs_json = json.load(f)
        
    gold_key_id, gold_items = validate_gold_key_payload(gold_json)
    observer_id, obs_ratings = validate_ratings_payload(obs_json, context_label="Observer Ratings")
    
    # Check trial ID alignment
    gold_trial_ids = sorted(list(gold_items.keys()))
    obs_trial_ids = sorted(list(obs_ratings.keys()))
    
    missing_in_obs = set(gold_trial_ids) - set(obs_trial_ids)
    if missing_in_obs:
        raise ValueError(f"Observer ratings missing required trials from gold key: {sorted(list(missing_in_obs))}")
        
    extra_in_obs = set(obs_trial_ids) - set(gold_trial_ids)
    if extra_in_obs:
        raise ValueError(f"Observer ratings contains trials not present in gold key: {sorted(list(extra_in_obs))}")
        
    n_trials = len(gold_trial_ids)
    gold_sets: List[Set[str]] = []
    obs_sets: List[Set[str]] = []
    
    exact_matches = 0
    total_jaccard = 0.0
    disputes: List[Dict[str, Any]] = []
    raw_comparisons: List[Dict[str, Any]] = []
    
    interp_matches = 0
    action_matches = 0
    critical_false_negatives: List[Dict[str, Any]] = []
    
    for tid in gold_trial_ids:
        g = gold_items[tid]
        o = obs_ratings[tid]
        
        g_codes = set(g.get("authoritative_cm_codes", []) + g.get("authoritative_sm_codes", []))
        o_codes = set(o.get("assigned_cm_codes", []) + o.get("assigned_sm_codes", []))
        
        gold_sets.append(g_codes)
        obs_sets.append(o_codes)
        
        union = g_codes | o_codes
        inter = g_codes & o_codes
        jaccard = len(inter) / len(union) if union else 1.0
        total_jaccard += jaccard
        
        is_exact = (g_codes == o_codes)
        if is_exact:
            exact_matches += 1
            
        # Interpretation & Action state agreement
        g_interp = g.get("interpretation_state")
        o_interp = o.get("interpretation_state")
        interp_agree = (g_interp == o_interp) if (g_interp and o_interp) else None
        if interp_agree:
            interp_matches += 1
            
        g_act = g.get("action_state")
        o_act = o.get("action_state")
        act_agree = (g_act == o_act) if (g_act and o_act) else None
        if act_agree:
            action_matches += 1
            
        # Critical decision safety error check
        is_critical = g.get("critical_decision_error", False)
        fn_codes = g_codes - o_codes
        fp_codes = o_codes - g_codes
        
        if is_critical and (fn_codes or (g_act == "ACTION_INAPPROPRIATE" and o_act == "ACTION_APPROPRIATE")):
            critical_false_negatives.append({
                "trial_id": tid,
                "missed_codes": sorted(list(fn_codes)),
                "gold_action": g_act,
                "observer_action": o_act,
                "rationale": g.get("rationale", "")
            })
            
        if not is_exact or (interp_agree is False) or (act_agree is False):
            disputes.append({
                "trial_id": tid,
                "gold_codes": sorted(list(g_codes)),
                "observer_codes": sorted(list(o_codes)),
                "false_positives": sorted(list(fp_codes)),
                "false_negatives": sorted(list(fn_codes)),
                "gold_interpretation": g_interp,
                "observer_interpretation": o_interp,
                "gold_action": g_act,
                "observer_action": o_act,
                "gold_rationale": g.get("rationale", ""),
                "observer_notes": o.get("notes", "")
            })
            
        raw_comparisons.append({
            "trial_id": tid,
            "gold_codes": sorted(list(g_codes)),
            "observer_codes": sorted(list(o_codes)),
            "match": is_exact,
            "jaccard": round(jaccard, 4),
            "gold_interp": g_interp,
            "obs_interp": o_interp,
            "gold_act": g_act,
            "obs_act": o_act
        })
        
    exact_match_rate = exact_matches / n_trials
    mean_jaccard = total_jaccard / n_trials
    interp_accuracy = (interp_matches / n_trials) if any(g.get("interpretation_state") for g in gold_items.values()) else None
    action_accuracy = (action_matches / n_trials) if any(g.get("action_state") for g in gold_items.values()) else None
    
    # Per-category metrics
    code_metrics: Dict[str, Any] = {}
    total_a = total_b = total_c = total_d = 0
    active_defined_kappas: List[float] = []
    
    categories_with_positive_ground_truth: List[str] = []
    categories_untested_in_key: List[str] = []
    
    for code in ALL_CODES:
        a, b, c, d = compute_contingency(obs_sets, gold_sets, code)
        po, pe, kappa, is_def = calculate_cohens_kappa(a, b, c, d)
        
        total_a += a
        total_b += b
        total_c += c
        total_d += d
        
        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = round(a / (a + c), 4) if (a + c) > 0 else None
        # Specificity = TN / (TN + FP)
        specificity = round(d / (d + b), 4) if (d + b) > 0 else None
        
        gold_prevalence = a + c
        if gold_prevalence > 0:
            categories_with_positive_ground_truth.append(code)
        else:
            categories_untested_in_key.append(code)
            
        if is_def and (a + b + c) > 0:
            active_defined_kappas.append(kappa)
            
        code_metrics[code] = {
            "tp": a, "fp": b, "fn": c, "tn": d,
            "gold_prevalence": gold_prevalence,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "po": po,
            "pe": pe,
            "kappa": kappa if is_def else "UNDEFINED",
            "is_defined": is_def
        }
        
    # Micro-pooled Cohen's Kappa
    micro_po, micro_pe, micro_kappa, micro_def = calculate_cohens_kappa(total_a, total_b, total_c, total_d)
    
    # Macro-Active Kappa
    macro_active_kappa = round(sum(active_defined_kappas) / len(active_defined_kappas), 4) if active_defined_kappas else None
    
    # Calibration Gate Evaluation
    passes_micro = (micro_kappa is not None) and (micro_kappa >= criterion)
    passes_macro = (macro_active_kappa is not None) and (macro_active_kappa >= criterion)
    passes_safety = (len(critical_false_negatives) == 0)
    
    if passes_micro and passes_macro and passes_safety:
        verdict = "CALIBRATION_PASSED"
    else:
        verdict = "CALIBRATION_FAILED"
        
    return {
        "mode": "OBSERVER_VS_GOLD",
        "evaluation_summary": {
            "observer_id": observer_id,
            "gold_key_id": gold_key_id,
            "transcript_id": gold_json.get("transcript_id", "UNKNOWN"),
            "total_trials": n_trials,
            "exact_match_rate": round(exact_match_rate, 4),
            "mean_jaccard": round(mean_jaccard, 4),
            "interpretation_accuracy": round(interp_accuracy, 4) if interp_accuracy is not None else "N/A",
            "action_accuracy": round(action_accuracy, 4) if action_accuracy is not None else "N/A",
            "micro_pooled_po": micro_po,
            "micro_pooled_pe": micro_pe,
            "micro_pooled_kappa": micro_kappa if micro_def else "UNDEFINED",
            "macro_active_kappa": macro_active_kappa if macro_active_kappa is not None else "UNDEFINED",
            "active_categories_evaluated": len(active_defined_kappas),
            "critical_decision_false_negatives": len(critical_false_negatives),
            "calibration_criterion": criterion,
            "calibration_verdict": verdict,
            "categories_tested_positive": categories_with_positive_ground_truth,
            "categories_untested_in_corpus": categories_untested_in_key
        },
        "per_code_metrics": code_metrics,
        "critical_safety_violations": critical_false_negatives,
        "disputes": disputes,
        "raw_comparisons": raw_comparisons
    }

def evaluate_pairwise_observers(rater1_path: str, rater2_path: str, threshold: float = 0.85) -> Dict[str, Any]:
    """Measurement B: Evaluate pre-adjudication inter-rater reliability between two independent observers."""
    with open(rater1_path, "r", encoding="utf-8") as f:
        r1_json = json.load(f)
    with open(rater2_path, "r", encoding="utf-8") as f:
        r2_json = json.load(f)
        
    obs1_id, r1_ratings = validate_ratings_payload(r1_json, context_label="Rater 1")
    obs2_id, r2_ratings = validate_ratings_payload(r2_json, context_label="Rater 2")
    
    trial_ids = sorted(list(r1_ratings.keys()))
    if trial_ids != sorted(list(r2_ratings.keys())):
        raise ValueError("Rater 1 and Rater 2 scored different trial sets.")
        
    n_trials = len(trial_ids)
    r1_sets: List[Set[str]] = []
    r2_sets: List[Set[str]] = []
    
    exact_matches = 0
    total_jaccard = 0.0
    disagreements: List[Dict[str, Any]] = []
    
    interp_agree_count = 0
    action_agree_count = 0
    
    for tid in trial_ids:
        o1 = r1_ratings[tid]
        o2 = r2_ratings[tid]
        
        s1 = set(o1.get("assigned_cm_codes", []) + o1.get("assigned_sm_codes", []))
        s2 = set(o2.get("assigned_cm_codes", []) + o2.get("assigned_sm_codes", []))
        
        r1_sets.append(s1)
        r2_sets.append(s2)
        
        union = s1 | s2
        inter = s1 & s2
        j = len(inter) / len(union) if union else 1.0
        total_jaccard += j
        
        is_exact = (s1 == s2)
        if is_exact:
            exact_matches += 1
            
        interp1 = o1.get("interpretation_state")
        interp2 = o2.get("interpretation_state")
        interp_agree = (interp1 == interp2) if (interp1 and interp2) else None
        if interp_agree:
            interp_agree_count += 1
            
        act1 = o1.get("action_state")
        act2 = o2.get("action_state")
        act_agree = (act1 == act2) if (act1 and act2) else None
        if act_agree:
            action_agree_count += 1
            
        if not is_exact or (interp_agree is False) or (act_agree is False):
            disagreements.append({
                "trial_id": tid,
                "rater1_codes": sorted(list(s1)),
                "rater2_codes": sorted(list(s2)),
                "code_differences": sorted(list(s1 ^ s2)),
                "rater1_interpretation": interp1,
                "rater2_interpretation": interp2,
                "rater1_action": act1,
                "rater2_action": act2,
                "rater1_notes": o1.get("notes", ""),
                "rater2_notes": o2.get("notes", "")
            })
            
    total_a = total_b = total_c = total_d = 0
    active_defined_kappas: List[float] = []
    pairwise_per_code: Dict[str, Any] = {}
    
    for code in ALL_CODES:
        a, b, c, d = compute_contingency(r1_sets, r2_sets, code)
        po, pe, kappa, is_def = calculate_cohens_kappa(a, b, c, d)
        
        total_a += a
        total_b += b
        total_c += c
        total_d += d
        
        if is_def and (a + b + c) > 0:
            active_defined_kappas.append(kappa)
            
        pairwise_per_code[code] = {
            "both_present": a,
            "rater1_only": b,
            "rater2_only": c,
            "both_absent": d,
            "po": po,
            "pe": pe,
            "kappa": kappa if is_def else "UNDEFINED",
            "is_defined": is_def
        }
        
    micro_po, micro_pe, micro_kappa, micro_def = calculate_cohens_kappa(total_a, total_b, total_c, total_d)
    macro_active_kappa = round(sum(active_defined_kappas) / len(active_defined_kappas), 4) if active_defined_kappas else None
    
    passes_pairwise = (micro_kappa is not None) and (micro_kappa >= threshold)
    verdict = "PAIRWISE_RELIABILITY_SUFFICIENT" if passes_pairwise else "PAIRWISE_RELIABILITY_INSUFFICIENT"
    
    return {
        "mode": "OBSERVER_PAIRWISE",
        "pairwise_summary": {
            "rater1_id": obs1_id,
            "rater2_id": obs2_id,
            "total_trials": n_trials,
            "exact_match_rate": round(exact_matches / n_trials, 4),
            "mean_jaccard": round(total_jaccard / n_trials, 4),
            "interpretation_agreement_rate": round(interp_agree_count / n_trials, 4),
            "action_agreement_rate": round(action_agree_count / n_trials, 4),
            "micro_pooled_po": micro_po,
            "micro_pooled_pe": micro_pe,
            "micro_pooled_kappa": micro_kappa if micro_def else "UNDEFINED",
            "macro_active_kappa": macro_active_kappa if macro_active_kappa is not None else "UNDEFINED",
            "pairwise_threshold": threshold,
            "pairwise_verdict": verdict,
            "total_disagreements_for_arbitration": len(disagreements)
        },
        "pairwise_per_code": pairwise_per_code,
        "disagreements_requiring_arbitration": disagreements
    }

def print_text_report(results: Dict[str, Any]) -> None:
    mode = results.get("mode")
    if mode == "OBSERVER_VS_GOLD":
        s = results["evaluation_summary"]
        print("\n" + "=" * 78)
        print("ARX OBSERVER CALIBRATION AUDIT REPORT (MEASUREMENT A: OBSERVER VS GOLD)")
        print("=" * 78)
        print(f"Observer ID:         {s['observer_id']}")
        print(f"Gold Standard Key:   {s['gold_key_id']}")
        print(f"Transcript ID:       {s['transcript_id']}")
        print(f"Total Trials Scored: {s['total_trials']}")
        print("-" * 78)
        print(f"Exact Trial Match Rate:       {s['exact_match_rate'] * 100:.1f}%")
        print(f"Mean Jaccard Index:           {s['mean_jaccard']:.4f}")
        print(f"Interpretation State Acc:     {s['interpretation_accuracy']}")
        print(f"Action State Acc:             {s['action_accuracy']}")
        print(f"Observed Agreement (Po):      {s['micro_pooled_po']:.4f}")
        print(f"Expected Agreement (Pe):      {s['micro_pooled_pe']:.4f}")
        print(f"Micro-Pooled Cohen's Kappa:   {s['micro_pooled_kappa']} (Criterion: >= {s['calibration_criterion']:.2f})")
        print(f"Macro-Active Cohen's Kappa:   {s['macro_active_kappa']} (Criterion: >= {s['calibration_criterion']:.2f})")
        print(f"Critical Safety False Negatives: {s['critical_decision_false_negatives']}")
        print("-" * 78)
        print(f"CALIBRATION VERDICT:          {s['calibration_verdict']}")
        print(f"Total Discrepancies:          {len(results['disputes'])}")
        print(f"Tested Positive Categories:   {', '.join(s['categories_tested_positive']) if s['categories_tested_positive'] else 'NONE'}")
        print(f"Untested Categories in Key:   {', '.join(s['categories_untested_in_corpus'])}")
        print("=" * 78 + "\n")
    else:
        s = results["pairwise_summary"]
        print("\n" + "=" * 78)
        print("ARX INTER-OBSERVER RELIABILITY REPORT (MEASUREMENT B: PAIRWISE OBSERVERS)")
        print("=" * 78)
        print(f"Rater 1 ID:          {s['rater1_id']}")
        print(f"Rater 2 ID:          {s['rater2_id']}")
        print(f"Total Trials Scored: {s['total_trials']}")
        print("-" * 78)
        print(f"Exact Trial Match Rate:       {s['exact_match_rate'] * 100:.1f}%")
        print(f"Mean Jaccard Index:           {s['mean_jaccard']:.4f}")
        print(f"Interpretation Agreement:     {s['interpretation_agreement_rate'] * 100:.1f}%")
        print(f"Action Agreement:             {s['action_agreement_rate'] * 100:.1f}%")
        print(f"Micro-Pooled Cohen's Kappa:   {s['micro_pooled_kappa']} (Threshold: >= {s['pairwise_threshold']:.2f})")
        print(f"Macro-Active Cohen's Kappa:   {s['macro_active_kappa']}")
        print("-" * 78)
        print(f"PAIRWISE VERDICT:             {s['pairwise_verdict']}")
        print(f"Disputes for Arbitration:     {s['total_disagreements_for_arbitration']}")
        print("=" * 78 + "\n")

def export_markdown_report(results: Dict[str, Any], filepath: str) -> None:
    mode = results.get("mode")
    lines: List[str] = []
    
    if mode == "OBSERVER_VS_GOLD":
        s = results["evaluation_summary"]
        lines.extend([
            f"# ARX Observer Calibration Audit: {s['observer_id']}",
            f"",
            f"**Authority:** ARX Closure Model v2 — Reconciled Interpretation/Action Separation  ",
            f"**Gold Key ID:** `{s['gold_key_id']}` | **Transcript:** `{s['transcript_id']}`  ",
            f"**Calibration Verdict:** `{s['calibration_verdict']}`  ",
            f"",
            f"## Statistical Reliability Summary",
            f"",
            f"| Metric | Value | Threshold / Standard | Result |",
            f"| :--- | :---: | :---: | :---: |",
            f"| **Micro-Pooled Kappa** | `{s['micro_pooled_kappa']}` | $\\ge {s['calibration_criterion']}$ | {'PASS' if s['micro_pooled_kappa'] != 'UNDEFINED' and s['micro_pooled_kappa'] >= s['calibration_criterion'] else 'FAIL'} |",
            f"| **Macro-Active Kappa** | `{s['macro_active_kappa']}` | $\\ge {s['calibration_criterion']}$ | {'PASS' if s['macro_active_kappa'] != 'UNDEFINED' and s['macro_active_kappa'] >= s['calibration_criterion'] else 'FAIL'} |",
            f"| **Exact Trial Match Rate** | `{s['exact_match_rate'] * 100:.1f}%` | N/A | Monitored |",
            f"| **Mean Jaccard Index** | `{s['mean_jaccard']}` | N/A | Monitored |",
            f"| **Interpretation State Acc** | `{s['interpretation_accuracy']}` | $\\ge 0.85$ | Monitored |",
            f"| **Action State Acc** | `{s['action_accuracy']}` | $\\ge 0.85$ | Monitored |",
            f"| **Critical Safety False Negatives** | `{s['critical_decision_false_negatives']}` | `= 0` | {'PASS' if s['critical_decision_false_negatives'] == 0 else 'FAIL'} |",
            f"",
            f"## Taxonomy Coverage & Corpus Distribution",
            f"",
            f"- **Tested Positive Categories ({len(s['categories_tested_positive'])}):** `{', '.join(s['categories_tested_positive']) if s['categories_tested_positive'] else 'None'}`",
            f"- **Untested Categories in Key ({len(s['categories_untested_in_corpus'])}):** `{', '.join(s['categories_untested_in_corpus'])}`",
            f"> [!WARNING]",
            f"> Observer calibration on this corpus provides empirical validation ONLY for tested positive categories. Competence on untested categories remains unproven.",
            f"",
            f"## Per-Category Performance Breakdown",
            f"",
            f"| Code | TP | FP | FN | TN | Prevalence | Sensitivity | Specificity | Po | Pe | Kappa |",
            f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])
        for c, m in results["per_code_metrics"].items():
            sens = f"{m['sensitivity']:.2f}" if m['sensitivity'] is not None else "N/A"
            spec = f"{m['specificity']:.2f}" if m['specificity'] is not None else "N/A"
            lines.append(f"| `{c}` | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {m['gold_prevalence']} | {sens} | {spec} | {m['po']} | {m['pe']} | `{m['kappa']}` |")
            
        lines.extend([
            f"",
            f"## Dispute & Discrepancy Log ({len(results['disputes'])})",
            f""
        ])
        if not results["disputes"]:
            lines.append("_Zero discrepancies recorded. Perfect alignment with authoritative gold standard._")
        else:
            for d in results["disputes"]:
                lines.append(f"### Trial {d['trial_id']}")
                lines.append(f"- **Gold Codes:** `{d['gold_codes']}` | **Observer Codes:** `{d['observer_codes']}`")
                lines.append(f"- **Gold State:** `{d['gold_interpretation']}` / `{d['gold_action']}`")
                lines.append(f"- **Observer State:** `{d['observer_interpretation']}` / `{d['observer_action']}`")
                if d["false_positives"]:
                    lines.append(f"- **False Positives:** `{d['false_positives']}`")
                if d["false_negatives"]:
                    lines.append(f"- **False Negatives:** `{d['false_negatives']}`")
                lines.append(f"- **Gold Rationale:** {d['gold_rationale']}")
                lines.append(f"- **Observer Notes:** {d['observer_notes']}")
                lines.append("")
    else:
        s = results["pairwise_summary"]
        lines.extend([
            f"# ARX Pre-Adjudication Inter-Rater Reliability: {s['rater1_id']} vs {s['rater2_id']}",
            f"",
            f"**Authority:** ARX Closure Model v2 — Decision-First, Minimum Sufficient Evidence  ",
            f"**Pairwise Verdict:** `{s['pairwise_verdict']}`  ",
            f"",
            f"## Pairwise Agreement Metrics",
            f"",
            f"| Metric | Value | Threshold | Result |",
            f"| :--- | :---: | :---: | :---: |",
            f"| **Micro-Pooled Kappa** | `{s['micro_pooled_kappa']}` | $\\ge {s['pairwise_threshold']}$ | {'PASS' if s['micro_pooled_kappa'] != 'UNDEFINED' and s['micro_pooled_kappa'] >= s['pairwise_threshold'] else 'FAIL'} |",
            f"| **Macro-Active Kappa** | `{s['macro_active_kappa']}` | $\\ge {s['pairwise_threshold']}$ | Monitored |",
            f"| **Exact Trial Match Rate** | `{s['exact_match_rate'] * 100:.1f}%` | N/A | Monitored |",
            f"| **Interpretation Agreement** | `{s['interpretation_agreement_rate'] * 100:.1f}%` | $\\ge 0.85$ | Monitored |",
            f"| **Action Agreement** | `{s['action_agreement_rate'] * 100:.1f}%` | $\\ge 0.85$ | Monitored |",
            f"| **Disputes Requiring Arbitration** | `{s['total_disagreements_for_arbitration']}` | N/A | Escalated |",
            f"",
            f"## Disagreements Scheduled for Written Arbitration",
            f""
        ])
        for d in results["disagreements_requiring_arbitration"]:
            lines.append(f"### Trial {d['trial_id']}")
            lines.append(f"- **Rater 1 ({s['rater1_id']}):** Codes `{d['rater1_codes']}` | State `{d['rater1_interpretation']}` / `{d['rater1_action']}`")
            lines.append(f"- **Rater 2 ({s['rater2_id']}):** Codes `{d['rater2_codes']}` | State `{d['rater2_interpretation']}` / `{d['rater2_action']}`")
            lines.append(f"- **Code Differences:** `{d['code_differences']}`")
            lines.append(f"- **Rater 1 Notes:** {d['rater1_notes']}")
            lines.append(f"- **Rater 2 Notes:** {d['rater2_notes']}")
            lines.append("")
            
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown audit log saved to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="ARX Observer Calibration Scoring Utility v2.0")
    parser.add_argument("--key", help="Path to gold standard answer key JSON (Measurement A)")
    parser.add_argument("--ratings", help="Path to observer ratings JSON (Measurement A)")
    parser.add_argument("--rater1", help="Path to Rater 1 JSON (Measurement B pairwise)")
    parser.add_argument("--rater2", help="Path to Rater 2 JSON (Measurement B pairwise)")
    parser.add_argument("--criterion", type=float, default=0.85, help="Kappa threshold (default: 0.85)")
    parser.add_argument("--report", default=None, help="Optional output path for JSON report")
    parser.add_argument("--markdown", default=None, help="Optional output path for Markdown report")
    
    args = parser.parse_args()
    
    if args.key and args.ratings:
        results = evaluate_observer_vs_gold(args.key, args.ratings, args.criterion)
        print_text_report(results)
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Detailed JSON report saved to: {args.report}")
        if args.markdown:
            export_markdown_report(results, args.markdown)
        return 0 if results["evaluation_summary"]["calibration_verdict"] == "CALIBRATION_PASSED" else 1
        
    elif args.rater1 and args.rater2:
        results = evaluate_pairwise_observers(args.rater1, args.rater2, args.criterion)
        print_text_report(results)
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Detailed JSON report saved to: {args.report}")
        if args.markdown:
            export_markdown_report(results, args.markdown)
        return 0 if results["pairwise_summary"]["pairwise_verdict"] == "PAIRWISE_RELIABILITY_SUFFICIENT" else 1
        
    else:
        parser.print_help()
        print("\n[ERROR]: Specify either (--key and --ratings) for Measurement A, or (--rater1 and --rater2) for Measurement B.")
        return 2

if __name__ == "__main__":
    sys.exit(main())
