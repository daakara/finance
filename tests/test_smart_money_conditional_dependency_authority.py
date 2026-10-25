"""
Unit and Integration Test Suite: Conditional Dependency Predicate Authority Subsystem V1
ARX Terminal — Smart Money

Covers the complete 48-test matrix from Sections 43-49:
- Section 43: Logic (Tests 1-8)
- Section 44: Provenance (Tests 9-15)
- Section 45: Anti-Bypass (Tests 16-22)
- Section 46: Locality (Tests 23-27)
- Section 47: Change Control (Tests 28-34)
- Section 48: Snapshot / TOCTOU (Tests 35-40)
- Section 49: Compilation (Tests 41-48)
"""

import pytest
import copy

from analyst_dashboard.smart_money.governance_control_plane import (
    DSL_VERSION,
    PredicateDefinition,
    PredicateDSLSyntaxError,
    UndeclaredFactError,
    ArbitraryCodeExecutionForbiddenError,
    UnboundRuntimeReadForbiddenError,
    validate_predicate_ast,
    normalize_predicate_ast,
    compute_predicate_hash,
    PredicateFact,
    FactContext,
    PredicateEvaluator,
    PREDICATE_RESULT_TRUE,
    PREDICATE_RESULT_FALSE,
    PREDICATE_RESULT_UNKNOWN,
    PREDICATE_RESULT_CONFLICT,
    DependencyNode,
    GuardedObligation,
    DependencyGraph,
    PredicateDependencyCycleError,
    TargetReferenceCycleError,
    SelfDisablingGuardError,
    CompiledDependencyPolicy,
    ResolvedDependency,
    DependencyResolutionResult,
    compile_dependency_policy,
    resolve_dependencies,
    classify_predicate_change,
    check_predicate_mutation_governance,
    verify_authorization_dependency_binding,
    PolicyCompilationError,
    UnauthorizedPredicateWeakeningException,
    TOCTOUException,
)


def _make_fact(
    fact_id: str,
    value: any,
    provenance_class: str = "CANONICAL_LEDGER",
    authority_class: str = "AUTHORITATIVE",
    validity: str = "CURRENT",
    establishment: str = "ESTABLISHED",
) -> PredicateFact:
    return PredicateFact(
        fact_id=fact_id,
        value=value,
        value_type="string",
        provenance={
            "evidence_id": f"ev-{fact_id}",
            "evidence_hash": "a" * 64,
            "provenance_class": provenance_class,
            "authority_class": authority_class,
        },
        validity=validity,
        establishment=establishment,
        observed_at="2026-10-12T07:30:00Z",
    )


class TestConditionalDependencyAuthority:

    # =========================================================================
    # Section 43: Test Matrix — Logic (Tests 1 - 8)
    # =========================================================================

    def test_01_true_or_unknown_equals_true(self):
        """1. TRUE OR UNKNOWN = TRUE"""
        evaluator = PredicateEvaluator()
        # any([fact_a == 'yes', fact_missing == 'yes'])
        expr = {
            "any": [
                {"eq": {"left": {"fact": "fact_a"}, "right": {"literal": "yes"}}},
                {"eq": {"left": {"fact": "fact_missing"}, "right": {"literal": "yes"}}},
            ]
        }
        pred = PredicateDefinition("p1", "1.0.0", expr, ["fact_a", "fact_missing"])
        ctx = FactContext([_make_fact("fact_a", "yes")])  # fact_missing is not in ctx
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_TRUE
        assert "fact_a" in [d["fact_id"] for d in res.decisive_inputs]

    def test_02_false_or_unknown_equals_unknown(self):
        """2. FALSE OR UNKNOWN = UNKNOWN"""
        evaluator = PredicateEvaluator()
        expr = {
            "any": [
                {"eq": {"left": {"fact": "fact_a"}, "right": {"literal": "no"}}},  # False
                {"eq": {"left": {"fact": "fact_missing"}, "right": {"literal": "yes"}}},  # Unknown
            ]
        }
        pred = PredicateDefinition("p2", "1.0.0", expr, ["fact_a", "fact_missing"])
        ctx = FactContext([_make_fact("fact_a", "yes")])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_UNKNOWN

    def test_03_false_and_unknown_equals_false(self):
        """3. FALSE AND UNKNOWN = FALSE"""
        evaluator = PredicateEvaluator()
        expr = {
            "all": [
                {"eq": {"left": {"fact": "fact_a"}, "right": {"literal": "no"}}},  # False
                {"eq": {"left": {"fact": "fact_missing"}, "right": {"literal": "yes"}}},  # Unknown
            ]
        }
        pred = PredicateDefinition("p3", "1.0.0", expr, ["fact_a", "fact_missing"])
        ctx = FactContext([_make_fact("fact_a", "yes")])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_FALSE
        assert "fact_a" in [d["fact_id"] for d in res.decisive_inputs]

    def test_04_true_and_unknown_equals_unknown(self):
        """4. TRUE AND UNKNOWN = UNKNOWN"""
        evaluator = PredicateEvaluator()
        expr = {
            "all": [
                {"eq": {"left": {"fact": "fact_a"}, "right": {"literal": "yes"}}},  # True
                {"eq": {"left": {"fact": "fact_missing"}, "right": {"literal": "yes"}}},  # Unknown
            ]
        }
        pred = PredicateDefinition("p4", "1.0.0", expr, ["fact_a", "fact_missing"])
        ctx = FactContext([_make_fact("fact_a", "yes")])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_UNKNOWN

    def test_05_nested_expressions_deterministic(self):
        """5. nested expressions deterministic"""
        evaluator = PredicateEvaluator()
        # (A and (B or C))
        expr = {
            "all": [
                {"eq": {"left": {"fact": "f_a"}, "right": {"literal": "1"}}},
                {
                    "any": [
                        {"eq": {"left": {"fact": "f_b"}, "right": {"literal": "2"}}},
                        {"eq": {"left": {"fact": "f_c"}, "right": {"literal": "3"}}},
                    ]
                },
            ]
        }
        pred = PredicateDefinition("p5", "1.0.0", expr, ["f_a", "f_b", "f_c"])
        ctx = FactContext([_make_fact("f_a", "1"), _make_fact("f_c", "3")])
        res1 = evaluator.evaluate(pred, ctx)
        res2 = evaluator.evaluate(pred, ctx)
        assert res1.result == PREDICATE_RESULT_TRUE
        assert res1.evaluation_hash == res2.evaluation_hash

    def test_06_not_unknown_equals_unknown(self):
        """6. NOT UNKNOWN = UNKNOWN"""
        evaluator = PredicateEvaluator()
        expr = {
            "not": {
                "eq": {"left": {"fact": "fact_missing"}, "right": {"literal": "yes"}}
            }
        }
        pred = PredicateDefinition("p6", "1.0.0", expr, ["fact_missing"])
        ctx = FactContext([])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_UNKNOWN

    def test_07_explicit_conflict_propagation(self):
        """7. explicit conflict propagation"""
        evaluator = PredicateEvaluator()
        expr = {"eq": {"left": {"fact": "f_conf"}, "right": {"literal": "v1"}}}
        pred = PredicateDefinition("p7", "1.0.0", expr, ["f_conf"])
        # Two facts with equal precedence and differing values
        f1 = _make_fact("f_conf", "v1", provenance_class="CANONICAL_LEDGER")
        f2 = _make_fact("f_conf", "v2", provenance_class="CANONICAL_LEDGER")
        ctx = FactContext([f1, f2])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_CONFLICT
        assert res.unknown_reason == "AUTHORITATIVE_FACT_CONFLICT"

    def test_08_canonical_short_circuit_semantics_stable(self):
        """8. canonical short-circuit semantics stable"""
        evaluator = PredicateEvaluator()
        # In any: [True, Conflict] -> True short-circuits
        expr_or = {
            "any": [
                {"eq": {"left": {"fact": "f_true"}, "right": {"literal": "ok"}}},
                {"eq": {"left": {"fact": "f_conf"}, "right": {"literal": "val"}}},
            ]
        }
        pred_or = PredicateDefinition("p8a", "1.0.0", expr_or, ["f_true", "f_conf"])
        ctx = FactContext([
            _make_fact("f_true", "ok"),
            _make_fact("f_conf", "v1", provenance_class="CANONICAL_LEDGER"),
            _make_fact("f_conf", "v2", provenance_class="CANONICAL_LEDGER"),
        ])
        res_or = evaluator.evaluate(pred_or, ctx)
        assert res_or.result == PREDICATE_RESULT_TRUE

        # In all: [False, Conflict] -> False short-circuits
        expr_and = {
            "all": [
                {"eq": {"left": {"fact": "f_false"}, "right": {"literal": "not_ok"}}},
                {"eq": {"left": {"fact": "f_conf"}, "right": {"literal": "val"}}},
            ]
        }
        pred_and = PredicateDefinition("p8b", "1.0.0", expr_and, ["f_false", "f_conf"])
        ctx_and = FactContext([
            _make_fact("f_false", "actual_val"),
            _make_fact("f_conf", "v1", provenance_class="CANONICAL_LEDGER"),
            _make_fact("f_conf", "v2", provenance_class="CANONICAL_LEDGER"),
        ])
        res_and = evaluator.evaluate(pred_and, ctx_and)
        assert res_and.result == PREDICATE_RESULT_FALSE

    # =========================================================================
    # Section 44: Test Matrix — Provenance (Tests 9 - 15)
    # =========================================================================

    def test_09_authoritative_decisive_fact_yields_authoritative_predicate(self):
        """9. authoritative decisive fact yields authoritative predicate"""
        evaluator = PredicateEvaluator()
        expr = {"eq": {"left": {"fact": "f_auth"}, "right": {"literal": "ready"}}}
        pred = PredicateDefinition("p9", "1.0.0", expr, ["f_auth"])
        ctx = FactContext([_make_fact("f_auth", "ready", provenance_class="CANONICAL_LEDGER", authority_class="AUTHORITATIVE")])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_TRUE
        assert res.predicate_authority == "AUTHORITATIVE"

    def test_10_synthetic_decisive_fact_yields_non_authoritative_predicate(self):
        """10. synthetic decisive fact yields non-authoritative predicate"""
        evaluator = PredicateEvaluator()
        expr = {"eq": {"left": {"fact": "f_synth"}, "right": {"literal": "ready"}}}
        pred = PredicateDefinition("p10", "1.0.0", expr, ["f_synth"])
        ctx = FactContext([_make_fact("f_synth", "ready", provenance_class="SYNTHETIC_FIXTURE")])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_TRUE
        assert res.predicate_authority == "NON_AUTHORITATIVE"

    def test_11_synthetic_non_participating_fact_does_not_taint_result(self):
        """11. synthetic non-participating fact does not taint result"""
        evaluator = PredicateEvaluator()
        expr = {"eq": {"left": {"fact": "f_auth"}, "right": {"literal": "ready"}}}
        pred = PredicateDefinition("p11", "1.0.0", expr, ["f_auth"])
        ctx = FactContext([
            _make_fact("f_auth", "ready", provenance_class="CANONICAL_LEDGER", authority_class="AUTHORITATIVE"),
            _make_fact("f_unused_synth", "mock", provenance_class="SYNTHETIC_FIXTURE"),
        ])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_TRUE
        assert res.predicate_authority == "AUTHORITATIVE"
        assert "f_unused_synth" in res.fact_participation["observed"]
        assert "f_unused_synth" not in res.fact_participation["decisive"]

    def test_12_test_double_decisive_fact_cannot_authorize(self):
        """12. test-double decisive fact cannot authorize"""
        evaluator = PredicateEvaluator()
        expr = {"eq": {"left": {"fact": "f_td"}, "right": {"literal": "ready"}}}
        pred = PredicateDefinition("p12", "1.0.0", expr, ["f_td"])
        ctx = FactContext([_make_fact("f_td", "ready", provenance_class="TEST_DOUBLE")])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_TRUE
        assert res.predicate_authority == "NON_AUTHORITATIVE"

    def test_13_stale_decisive_fact_yields_unresolved_result(self):
        """13. stale decisive fact yields unresolved result"""
        evaluator = PredicateEvaluator()
        expr = {"eq": {"left": {"fact": "f_stale"}, "right": {"literal": "ok"}}}
        pred = PredicateDefinition("p13", "1.0.0", expr, ["f_stale"])
        ctx = FactContext([_make_fact("f_stale", "ok", validity="STALE")])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_UNKNOWN
        assert res.unknown_reason == "FACT_STALE"

    def test_14_revoked_decisive_fact_yields_unresolved_result(self):
        """14. revoked decisive fact yields unresolved result"""
        evaluator = PredicateEvaluator()
        expr = {"eq": {"left": {"fact": "f_revoked"}, "right": {"literal": "ok"}}}
        pred = PredicateDefinition("p14", "1.0.0", expr, ["f_revoked"])
        ctx = FactContext([_make_fact("f_revoked", "ok", validity="REVOKED")])
        res = evaluator.evaluate(pred, ctx)
        assert res.result == PREDICATE_RESULT_UNKNOWN
        assert res.unknown_reason == "FACT_REVOKED"

    def test_15_not_established_decisive_fact_yields_unknown_unless_explicitly_tested(self):
        """15. NOT_ESTABLISHED decisive fact yields UNKNOWN unless explicitly tested"""
        evaluator = PredicateEvaluator()
        # Value check: yields UNKNOWN
        expr_val = {"eq": {"left": {"fact": "f_not_est"}, "right": {"literal": "something"}}}
        pred_val = PredicateDefinition("p15a", "1.0.0", expr_val, ["f_not_est"])
        ctx = FactContext([_make_fact("f_not_est", "val", establishment="NOT_ESTABLISHED")])
        res_val = evaluator.evaluate(pred_val, ctx)
        assert res_val.result == PREDICATE_RESULT_UNKNOWN
        assert res_val.unknown_reason == "FACT_NOT_ESTABLISHED"

        # Explicit state operator: is_established(f_not_est) -> FALSE
        expr_est = {"is_established": {"fact": "f_not_est"}}
        pred_est = PredicateDefinition("p15b", "1.0.0", expr_est, ["f_not_est"])
        res_est = evaluator.evaluate(pred_est, ctx)
        assert res_est.result == PREDICATE_RESULT_FALSE

    # =========================================================================
    # Section 45: Test Matrix — Anti-Bypass (Tests 16 - 22)
    # =========================================================================

    def test_16_guarded_claim_cannot_self_disable_requirement(self):
        """16. guarded claim cannot self-disable requirement"""
        graph = DependencyGraph()
        graph.add_node(DependencyNode("claim_x", "CLAIM"))
        graph.add_node(DependencyNode("output_y", "OUTPUT"))
        # Predicate references claim_x directly
        expr = {"eq": {"left": {"fact": "claim_x"}, "right": {"literal": "present"}}}
        pred = PredicateDefinition("pred_self", "1.0.0", expr, ["claim_x"])
        edge = GuardedObligation(
            dependency_id="dep-1",
            source_claim_id="claim_x",
            target_output_id="output_y",
            declared_requirement="CONDITIONAL",
            predicate_ref={"predicate_id": "pred_self", "predicate_version": "1.0.0", "predicate_hash": pred.predicate_hash},
        )
        graph.add_edge(edge)
        with pytest.raises(SelfDisablingGuardError):
            graph.validate_graph({"pred_self": pred})

    def test_17_target_output_cycle_rejected(self):
        """17. target-output cycle rejected"""
        graph = DependencyGraph()
        graph.add_node(DependencyNode("claim_a", "CLAIM"))
        graph.add_node(DependencyNode("output_b", "OUTPUT"))
        # Guard references its own target output_b
        expr = {"eq": {"left": {"fact": "output_b"}, "right": {"literal": "ready"}}}
        pred = PredicateDefinition("pred_target", "1.0.0", expr, ["output_b"])
        edge = GuardedObligation(
            dependency_id="dep-target",
            source_claim_id="claim_a",
            target_output_id="output_b",
            declared_requirement="CONDITIONAL",
            predicate_ref={"predicate_id": "pred_target", "predicate_version": "1.0.0", "predicate_hash": pred.predicate_hash},
        )
        graph.add_edge(edge)
        with pytest.raises(TargetReferenceCycleError):
            graph.validate_graph({"pred_target": pred})

    def test_18_descendant_output_cycle_rejected(self):
        """18. descendant-output cycle rejected"""
        graph = DependencyGraph()
        graph.add_node(DependencyNode("claim_a", "CLAIM"))
        graph.add_node(DependencyNode("target_1", "OUTPUT"))
        graph.add_node(DependencyNode("target_2", "OUTPUT"))
        # target_1 -> target_2 edge
        edge_1 = GuardedObligation("e1", "claim_a", "target_1", "MANDATORY")
        edge_2 = GuardedObligation("e2", "target_1", "target_2", "MANDATORY")
        graph.add_edge(edge_1)
        graph.add_edge(edge_2)

        # Now edge on target_1 references target_2 (its descendant)
        expr = {"eq": {"left": {"fact": "target_2"}, "right": {"literal": "ready"}}}
        pred = PredicateDefinition("pred_desc", "1.0.0", expr, ["target_2"])
        edge_cond = GuardedObligation(
            dependency_id="e3",
            source_claim_id="claim_a",
            target_output_id="target_1",
            declared_requirement="CONDITIONAL",
            predicate_ref={"predicate_id": "pred_desc", "predicate_version": "1.0.0", "predicate_hash": pred.predicate_hash},
        )
        graph.add_edge(edge_cond)
        with pytest.raises(TargetReferenceCycleError):
            graph.validate_graph({"pred_desc": pred})

    def test_19_readiness_derived_guard_cycle_rejected(self):
        """19. readiness-derived guard cycle rejected"""
        graph = DependencyGraph()
        graph.add_node(DependencyNode("n_a", "CLAIM"))
        graph.add_node(DependencyNode("n_b", "CLAIM"))
        # n_a -> n_b and n_b -> n_a structural cycle
        graph.add_edge(GuardedObligation("e1", "n_a", "n_b", "MANDATORY"))
        graph.add_edge(GuardedObligation("e2", "n_b", "n_a", "MANDATORY"))
        with pytest.raises(PredicateDependencyCycleError):
            graph.validate_graph()

    def test_20_missing_guarded_claim_cannot_cause_predicate_false_through_self_reference(self):
        """20. missing guarded claim cannot cause predicate FALSE through self-reference"""
        # When compiling a policy where predicate attempts self-disabling check, it is caught at compile time
        source_policy = {
            "policy_id": "pol-anti-bypass",
            "declared_facts": ["witness_claim"],
            "nodes": [
                {"node_id": "witness_claim", "node_type": "CLAIM"},
                {"node_id": "gate3_output", "node_type": "OUTPUT"},
            ],
            "predicates": [
                {
                    "predicate_id": "pred_witness",
                    "expression": {"is_established": {"fact": "witness_claim"}},
                }
            ],
            "edges": [
                {
                    "dependency_id": "dep_witness",
                    "source_claim_id": "witness_claim",
                    "target_output_id": "gate3_output",
                    "declared_requirement": "CONDITIONAL",
                    "predicate_ref": {"predicate_id": "pred_witness"},
                }
            ],
        }
        with pytest.raises(SelfDisablingGuardError):
            compile_dependency_policy(source_policy)

    def test_21_conditional_to_optional_weakening_requires_approval(self):
        """21. conditional-to-optional weakening requires approval"""
        with pytest.raises(UnauthorizedPredicateWeakeningException):
            check_predicate_mutation_governance("WEAKER", governance_approval=False)

    def test_22_predicate_weakening_requires_approval(self):
        """22. predicate weakening requires approval"""
        with pytest.raises(UnauthorizedPredicateWeakeningException):
            check_predicate_mutation_governance("INCOMPARABLE", governance_approval=False)

    # =========================================================================
    # Section 46: Test Matrix — Locality (Tests 23 - 27)
    # =========================================================================

    def _build_two_branch_policy(self):
        return {
            "policy_id": "pol-locality",
            "declared_facts": ["branch_a_flag", "branch_b_flag"],
            "nodes": [
                {"node_id": "claim_a", "node_type": "CLAIM"},
                {"node_id": "claim_b", "node_type": "CLAIM"},
                {"node_id": "target_a1", "node_type": "OUTPUT"},
                {"node_id": "target_a2", "node_type": "OUTPUT"},
                {"node_id": "target_b1", "node_type": "OUTPUT"},
            ],
            "predicates": [
                {
                    "predicate_id": "pred_guard_a",
                    "expression": {"eq": {"left": {"fact": "branch_a_flag"}, "right": {"literal": "enable"}}},
                },
                {
                    "predicate_id": "pred_guard_b",
                    "expression": {"eq": {"left": {"fact": "branch_b_flag"}, "right": {"literal": "enable"}}},
                },
            ],
            "edges": [
                {
                    "dependency_id": "dep_a1",
                    "source_claim_id": "claim_a",
                    "target_output_id": "target_a1",
                    "declared_requirement": "CONDITIONAL",
                    "predicate_ref": {"predicate_id": "pred_guard_a"},
                    "target_safety_class": "AUTHORIZATION_CRITICAL",
                },
                {
                    "dependency_id": "dep_a2",
                    "source_claim_id": "target_a1",
                    "target_output_id": "target_a2",
                    "declared_requirement": "MANDATORY",
                    "target_safety_class": "AUTHORIZATION_CRITICAL",
                },
                {
                    "dependency_id": "dep_b1",
                    "source_claim_id": "claim_b",
                    "target_output_id": "target_b1",
                    "declared_requirement": "CONDITIONAL",
                    "predicate_ref": {"predicate_id": "pred_guard_b"},
                    "target_safety_class": "AUTHORIZATION_CRITICAL",
                },
            ],
        }

    def test_23_unresolved_guard_blocks_direct_target(self):
        """23. unresolved guard blocks direct target"""
        source = self._build_two_branch_policy()
        compiled = compile_dependency_policy(source)
        # branch_a_flag is missing (UNKNOWN)
        ctx = FactContext([_make_fact("branch_b_flag", "enable")])
        res = resolve_dependencies(compiled, ctx)
        assert res.target_statuses["target_a1"] == "BLOCKED"

    def test_24_unresolved_guard_blocks_target_descendants(self):
        """24. unresolved guard blocks target descendants"""
        source = self._build_two_branch_policy()
        compiled = compile_dependency_policy(source)
        ctx = FactContext([_make_fact("branch_b_flag", "enable")])
        res = resolve_dependencies(compiled, ctx)
        assert res.target_statuses["target_a2"] == "BLOCKED"

    def test_25_unrelated_graph_branch_remains_evaluable(self):
        """25. unrelated graph branch remains evaluable"""
        source = self._build_two_branch_policy()
        compiled = compile_dependency_policy(source)
        # branch_a is UNKNOWN, but branch_b is satisfied
        ctx = FactContext([_make_fact("branch_b_flag", "enable")])
        res = resolve_dependencies(compiled, ctx)
        assert res.target_statuses["target_a1"] == "BLOCKED"
        assert res.target_statuses["target_a2"] == "BLOCKED"
        assert res.target_statuses["target_b1"] == "READY"
        assert "target_b1" in res.unaffected_target_ids

    def test_26_system_wide_incident_propagation_occurs_only_when_explicitly_configured(self):
        """26. system-wide incident propagation occurs only when explicitly configured"""
        source = self._build_two_branch_policy()
        compiled = compile_dependency_policy(source)
        ctx = FactContext([_make_fact("branch_b_flag", "enable")])
        res = resolve_dependencies(compiled, ctx)
        # Without explicit incident policy, unrelated branch target_b1 is NOT blocked
        assert res.target_statuses["target_b1"] == "READY"

    def test_27_informational_unresolved_guard_does_not_block_authorization_unrelated_outputs(self):
        """27. informational unresolved guard does not block authorization-unrelated outputs"""
        source = {
            "policy_id": "pol-info",
            "declared_facts": ["info_flag"],
            "nodes": [
                {"node_id": "claim_info", "node_type": "CLAIM"},
                {"node_id": "target_info", "node_type": "OUTPUT"},
                {"node_id": "target_auth", "node_type": "OUTPUT"},
            ],
            "predicates": [
                {
                    "predicate_id": "pred_info",
                    "expression": {"eq": {"left": {"fact": "info_flag"}, "right": {"literal": "yes"}}},
                }
            ],
            "edges": [
                {
                    "dependency_id": "dep_info",
                    "source_claim_id": "claim_info",
                    "target_output_id": "target_info",
                    "declared_requirement": "CONDITIONAL",
                    "predicate_ref": {"predicate_id": "pred_info"},
                    "target_safety_class": "INFORMATIONAL",
                    "unknown_behavior": "MARK_UNKNOWN",
                },
                {
                    "dependency_id": "dep_auth",
                    "source_claim_id": "claim_info",
                    "target_output_id": "target_auth",
                    "declared_requirement": "OPTIONAL",
                    "target_safety_class": "AUTHORIZATION_CRITICAL",
                },
            ],
        }
        compiled = compile_dependency_policy(source)
        ctx = FactContext([])  # info_flag missing
        res = resolve_dependencies(compiled, ctx)
        assert res.target_statuses["target_auth"] == "READY"

    # =========================================================================
    # Section 47: Test Matrix — Change Control (Tests 28 - 34)
    # =========================================================================

    def test_28_equivalent_predicate_change_accepted(self):
        """28. equivalent predicate change accepted"""
        # (A and B) vs (B and A)
        old_expr = {
            "all": [
                {"eq": {"left": {"fact": "flag1"}, "right": {"literal": "yes"}}},
                {"eq": {"left": {"fact": "flag2"}, "right": {"literal": "yes"}}},
            ]
        }
        new_expr = {
            "all": [
                {"eq": {"left": {"fact": "flag2"}, "right": {"literal": "yes"}}},
                {"eq": {"left": {"fact": "flag1"}, "right": {"literal": "yes"}}},
            ]
        }
        domain = {"flag1": ["yes", "no"], "flag2": ["yes", "no"]}
        relation = classify_predicate_change(old_expr, new_expr, domain)
        assert relation == "EQUIVALENT"

    def test_29_stricter_predicate_change_classified_correctly(self):
        """29. stricter predicate change classified correctly"""
        # Old: flag1 == 'yes' AND flag2 == 'yes' (few contexts)
        # New: flag1 == 'yes' OR flag2 == 'yes' (more contexts where condition triggers obligation)
        old_expr = {
            "all": [
                {"eq": {"left": {"fact": "flag1"}, "right": {"literal": "yes"}}},
                {"eq": {"left": {"fact": "flag2"}, "right": {"literal": "yes"}}},
            ]
        }
        new_expr = {
            "any": [
                {"eq": {"left": {"fact": "flag1"}, "right": {"literal": "yes"}}},
                {"eq": {"left": {"fact": "flag2"}, "right": {"literal": "yes"}}},
            ]
        }
        domain = {"flag1": ["yes", "no"], "flag2": ["yes", "no"]}
        relation = classify_predicate_change(old_expr, new_expr, domain)
        assert relation == "STRICTER"

    def test_30_weaker_predicate_change_classified_correctly(self):
        """30. weaker predicate change classified correctly"""
        # Old: flag1 == 'yes' OR flag2 == 'yes' (more contexts)
        # New: flag1 == 'yes' AND flag2 == 'yes' (fewer contexts)
        old_expr = {
            "any": [
                {"eq": {"left": {"fact": "flag1"}, "right": {"literal": "yes"}}},
                {"eq": {"left": {"fact": "flag2"}, "right": {"literal": "yes"}}},
            ]
        }
        new_expr = {
            "all": [
                {"eq": {"left": {"fact": "flag1"}, "right": {"literal": "yes"}}},
                {"eq": {"left": {"fact": "flag2"}, "right": {"literal": "yes"}}},
            ]
        }
        domain = {"flag1": ["yes", "no"], "flag2": ["yes", "no"]}
        relation = classify_predicate_change(old_expr, new_expr, domain)
        assert relation == "WEAKER"

    def test_31_incomparable_change_classified_correctly(self):
        """31. incomparable change classified correctly"""
        old_expr = {"eq": {"left": {"fact": "flag1"}, "right": {"literal": "yes"}}}
        new_expr = {"eq": {"left": {"fact": "flag2"}, "right": {"literal": "yes"}}}
        domain = {"flag1": ["yes", "no"], "flag2": ["yes", "no"]}
        relation = classify_predicate_change(old_expr, new_expr, domain)
        assert relation == "INCOMPARABLE"

    def test_32_unprovable_relation_becomes_unknown(self):
        """32. unprovable relation becomes UNKNOWN"""
        old_expr = {"eq": {"left": {"fact": "flag1"}, "right": {"literal": "yes"}}}
        new_expr = {"eq": {"left": {"fact": "flag2"}, "right": {"literal": "yes"}}}
        relation = classify_predicate_change(old_expr, new_expr, domain_schema=None)
        assert relation == "UNKNOWN"

    def test_33_unknown_relation_requires_governance_review(self):
        """33. UNKNOWN relation requires governance review"""
        with pytest.raises(UnauthorizedPredicateWeakeningException):
            check_predicate_mutation_governance("UNKNOWN", governance_approval=False)

    def test_34_edge_version_change_invalidates_dependent_decision(self):
        """34. edge-version change invalidates dependent decision"""
        source = {
            "policy_id": "pol-ver",
            "declared_facts": ["f1"],
            "nodes": [
                {"node_id": "c1", "node_type": "CLAIM"},
                {"node_id": "t1", "node_type": "OUTPUT"},
            ],
            "predicates": [
                {"predicate_id": "p1", "expression": {"eq": {"left": {"fact": "f1"}, "right": {"literal": "1"}}}}
            ],
            "edges": [
                {
                    "dependency_id": "e1",
                    "dependency_version": "1.0.0",
                    "source_claim_id": "c1",
                    "target_output_id": "t1",
                    "declared_requirement": "CONDITIONAL",
                    "predicate_ref": {"predicate_id": "p1"},
                }
            ],
        }
        compiled_v1 = compile_dependency_policy(source)
        ctx = FactContext([_make_fact("f1", "1")])
        res_v1 = resolve_dependencies(compiled_v1, ctx)

        source_v2 = copy.deepcopy(source)
        source_v2["edges"][0]["dependency_version"] = "1.1.0"
        compiled_v2 = compile_dependency_policy(source_v2)
        res_v2 = resolve_dependencies(compiled_v2, ctx)

        assert res_v1.dependency_resolution_hash != res_v2.dependency_resolution_hash

    # =========================================================================
    # Section 48: Test Matrix — Snapshot / TOCTOU (Tests 35 - 40)
    # =========================================================================

    def test_35_dependency_resolution_hash_deterministic(self):
        """35. dependency-resolution hash deterministic"""
        source = self._build_two_branch_policy()
        compiled = compile_dependency_policy(source)
        ctx = FactContext([_make_fact("branch_a_flag", "enable"), _make_fact("branch_b_flag", "enable")])
        res1 = resolve_dependencies(compiled, ctx)
        res2 = resolve_dependencies(compiled, ctx)
        assert res1.dependency_resolution_hash == res2.dependency_resolution_hash

    def test_36_changed_guard_result_changes_dependency_resolution_hash(self):
        """36. changed guard result changes dependency-resolution hash"""
        source = self._build_two_branch_policy()
        compiled = compile_dependency_policy(source)
        ctx1 = FactContext([_make_fact("branch_a_flag", "enable"), _make_fact("branch_b_flag", "enable")])
        ctx2 = FactContext([_make_fact("branch_a_flag", "disable"), _make_fact("branch_b_flag", "enable")])
        res1 = resolve_dependencies(compiled, ctx1)
        res2 = resolve_dependencies(compiled, ctx2)
        assert res1.dependency_resolution_hash != res2.dependency_resolution_hash

    def test_37_changed_predicate_version_changes_dependency_resolution_hash(self):
        """37. changed predicate version changes dependency-resolution hash"""
        source = self._build_two_branch_policy()
        compiled1 = compile_dependency_policy(source)
        source2 = copy.deepcopy(source)
        source2["predicates"][0]["predicate_version"] = "2.0.0"
        compiled2 = compile_dependency_policy(source2)

        ctx = FactContext([_make_fact("branch_a_flag", "enable"), _make_fact("branch_b_flag", "enable")])
        res1 = resolve_dependencies(compiled1, ctx)
        res2 = resolve_dependencies(compiled2, ctx)
        assert res1.dependency_resolution_hash != res2.dependency_resolution_hash

    def test_38_changed_decisive_fact_hash_changes_dependency_resolution_hash(self):
        """38. changed decisive fact hash changes dependency-resolution hash"""
        source = self._build_two_branch_policy()
        compiled = compile_dependency_policy(source)
        fact1 = _make_fact("branch_a_flag", "enable")
        # Fact with different timestamp or evidence hash
        fact2 = copy.deepcopy(fact1)
        fact2.provenance["evidence_hash"] = "b" * 64
        fact2.canonical_hash = None
        fact2.__post_init__()

        ctx1 = FactContext([fact1, _make_fact("branch_b_flag", "enable")])
        ctx2 = FactContext([fact2, _make_fact("branch_b_flag", "enable")])
        res1 = resolve_dependencies(compiled, ctx1)
        res2 = resolve_dependencies(compiled, ctx2)
        assert res1.dependency_resolution_hash != res2.dependency_resolution_hash

    def test_39_stale_authorization_rejected_after_dependency_hash_changes(self):
        """39. stale authorization rejected after dependency hash changes"""
        auth_hash = "11" * 32
        curr_hash = "22" * 32
        with pytest.raises(TOCTOUException):
            verify_authorization_dependency_binding(auth_hash, curr_hash)

    def test_40_historical_replay_exact(self):
        """40. historical replay exact"""
        source = self._build_two_branch_policy()
        compiled = compile_dependency_policy(source)
        ctx = FactContext([_make_fact("branch_a_flag", "enable"), _make_fact("branch_b_flag", "enable")])
        res1 = resolve_dependencies(compiled, ctx)
        # Replay with identical facts
        res2 = resolve_dependencies(compiled, ctx)
        assert res1.to_dict() == res2.to_dict()

    # =========================================================================
    # Section 49: Test Matrix — Compilation (Tests 41 - 48)
    # =========================================================================

    def test_41_syntax_invalid_predicate_rejected(self):
        """41. syntax-invalid predicate rejected"""
        # Empty dict or multiple keys
        with pytest.raises(PredicateDSLSyntaxError):
            validate_predicate_ast({})
        with pytest.raises(PredicateDSLSyntaxError):
            validate_predicate_ast({"all": [], "not": {}})

    def test_42_undeclared_fact_rejected(self):
        """42. undeclared fact rejected"""
        expr = {"eq": {"left": {"fact": "undeclared_var"}, "right": {"literal": "1"}}}
        with pytest.raises(UndeclaredFactError):
            validate_predicate_ast(expr, declared_facts={"declared_var"})

    def test_43_arbitrary_code_execution_rejected(self):
        """43. arbitrary code execution rejected"""
        expr = {"eq": {"left": {"fact": "eval"}, "right": {"literal": "1"}}}
        with pytest.raises(ArbitraryCodeExecutionForbiddenError):
            validate_predicate_ast(expr)

    def test_44_network_dependent_predicate_rejected(self):
        """44. network-dependent predicate rejected"""
        expr = {"eq": {"left": {"fact": "http_endpoint"}, "right": {"literal": "1"}}}
        with pytest.raises(PredicateDSLSyntaxError):
            validate_predicate_ast(expr)

    def test_45_wall_clock_direct_read_rejected(self):
        """45. wall-clock direct read rejected"""
        expr = {"eq": {"left": {"fact": "datetime_now"}, "right": {"literal": "1"}}}
        with pytest.raises(UnboundRuntimeReadForbiddenError):
            validate_predicate_ast(expr)

    def test_46_cycle_detection_works(self):
        """46. cycle detection works"""
        graph = DependencyGraph()
        graph.add_node(DependencyNode("node_1", "CLAIM"))
        graph.add_node(DependencyNode("node_2", "CLAIM"))
        graph.add_node(DependencyNode("node_3", "CLAIM"))
        graph.add_edge(GuardedObligation("e1", "node_1", "node_2", "MANDATORY"))
        graph.add_edge(GuardedObligation("e2", "node_2", "node_3", "MANDATORY"))
        graph.add_edge(GuardedObligation("e3", "node_3", "node_1", "MANDATORY"))
        with pytest.raises(PredicateDependencyCycleError):
            graph.validate_graph()

    def test_47_self_disabling_guard_detection_works(self):
        """47. self-disabling guard detection works"""
        graph = DependencyGraph()
        graph.add_node(DependencyNode("my_claim", "CLAIM"))
        graph.add_node(DependencyNode("my_output", "OUTPUT"))
        expr = {"is_established": {"fact": "my_claim"}}
        pred = PredicateDefinition("p_sd", "1.0.0", expr, ["my_claim"])
        edge = GuardedObligation(
            dependency_id="e_sd",
            source_claim_id="my_claim",
            target_output_id="my_output",
            declared_requirement="CONDITIONAL",
            predicate_ref={"predicate_id": "p_sd", "predicate_version": "1.0.0", "predicate_hash": pred.predicate_hash},
        )
        graph.add_edge(edge)
        with pytest.raises(SelfDisablingGuardError):
            graph.validate_graph({"p_sd": pred})

    def test_48_deterministic_ast_normalization_works(self):
        """48. deterministic AST normalization works"""
        expr_a = {
            "all": [
                {"eq": {"left": {"fact": "b_var"}, "right": {"literal": "2"}}},
                {"eq": {"left": {"fact": "a_var"}, "right": {"literal": "1"}}},
            ]
        }
        expr_b = {
            "all": [
                {"eq": {"left": {"fact": "a_var"}, "right": {"literal": "1"}}},
                {"eq": {"left": {"fact": "b_var"}, "right": {"literal": "2"}}},
            ]
        }
        norm_a = normalize_predicate_ast(expr_a)
        norm_b = normalize_predicate_ast(expr_b)
        assert norm_a == norm_b
        hash_a = compute_predicate_hash({"predicate_id": "test", "predicate_version": "1.0.0", "expression": expr_a, "fact_schema_refs": []})
        hash_b = compute_predicate_hash({"predicate_id": "test", "predicate_version": "1.0.0", "expression": expr_b, "fact_schema_refs": []})
        assert hash_a == hash_b
