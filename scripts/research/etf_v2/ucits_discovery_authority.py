"""
scripts/research/etf_v2/ucits_discovery_authority.py

Orchestrator and evidence storage engine for the ETF V2 UCITS Discovery Authority.
Coordinates statutory registry harvesting, raw evidence content addressing,
fail-closed completeness validation, deterministic conflict adjudication,
strict accounting conservation, and immutable manifest generation.
Delegates ISIN normalization strictly to global_identifier_authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .global_identifier_authority import normalize_isin, validate_isin
from .ucits_discovery_adapters import (
    AMFFranceAdapter,
    BaFinGermanyAdapter,
    BaseDiscoveryAdapter,
    BaseDiscoveryTransport,
    CSSFLuxembourgAdapter,
    CentralBankOfIrelandAdapter,
    MockDiscoveryTransport,
    StatutoryIssuerAdapter,
)
from .ucits_discovery_models import (
    AuthorityFunction,
    CandidateSpec,
    CandidateStatus,
    ConservationViolationError,
    CorruptedDiscoveryCacheError,
    DiscoveryAccounting,
    DiscoveryCompletenessError,
    DiscoveryConfiguration,
    DiscoveryError,
    DiscoveryInterruptionError,
    DiscoveryJurisdiction,
    DiscoveryRunIdentity,
    DiscoveryRunStatus,
    FixtureContaminationError,
    InvalidDiscoveryConfigurationError,
    ObservationProvenance,
    ParentJoinStatus,
    QuarantineReason,
    RawDiscoveryObservation,
    RawRegisterPayload,
    ResumeIdentityMismatchError,
    SchemaDriftError,
    ShareClassExpansionCompleteness,
    SourceAdapterError,
    SourceAuthorityId,
    SourceAuthorityTier,
    SourceEnumerationState,
    Tier1ParentAccounting,
    Tier1ParentObservation,
    Tier2ShareClassExpansion,
    normalize_fund_name,
)


FORBIDDEN_FIXTURE_PATH_SUBSTRING = "ucits_wave4_fixtures.json"


class UCITSDiscoveryAuthority:
    """
    Holistic orchestrator for UCITS candidate universe discovery.
    Strictly isolated from canonical population execution and document acquisition.
    """

    def __init__(
        self,
        cache_dir: Path,
        software_sha: str = "promoted_discovery_v2",
        transport: Optional[BaseDiscoveryTransport] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.software_sha = software_sha
        self.transport = transport or MockDiscoveryTransport()

        # Enforce fixture firewall on cache_dir path
        if FORBIDDEN_FIXTURE_PATH_SUBSTRING in str(self.cache_dir):
            raise FixtureContaminationError(
                f"Discovery cache directory cannot be located inside {FORBIDDEN_FIXTURE_PATH_SUBSTRING}"
            )

    def _get_adapters_for_jurisdiction(self, jurisdiction: str) -> List[BaseDiscoveryAdapter]:
        """Resolves registered Tier 1 adapters for a given jurisdiction."""
        if jurisdiction == DiscoveryJurisdiction.IE.value:
            return [CentralBankOfIrelandAdapter()]
        elif jurisdiction == DiscoveryJurisdiction.LU.value:
            return [CSSFLuxembourgAdapter()]
        elif jurisdiction == DiscoveryJurisdiction.DE.value:
            return [BaFinGermanyAdapter()]
        elif jurisdiction == DiscoveryJurisdiction.FR.value:
            return [AMFFranceAdapter()]
        else:
            raise InvalidDiscoveryConfigurationError(f"Unsupported jurisdiction: {jurisdiction}")

    def execute_discovery(
        self,
        config: DiscoveryConfiguration,
        run_id: Optional[str] = None,
        custom_adapters: Optional[List[BaseDiscoveryAdapter]] = None,
        interruption_stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes a complete, content-addressed discovery run across configured jurisdictions.
        Returns the completed discovery manifest dictionary.
        """
        # 1. Setup run directory and identity
        run_timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        config_sha = config.compute_sha256()
        actual_run_id = run_id or f"ucits_disc_run_{run_timestamp}_{config_sha[:8]}"
        run_dir = self.cache_dir / actual_run_id
        raw_responses_dir = run_dir / "raw_responses"
        raw_responses_dir.mkdir(parents=True, exist_ok=True)

        if interruption_stage == "C01":
            raise DiscoveryInterruptionError("Interrupted at C01 (Before run identity freeze)")

        identity = DiscoveryRunIdentity(
            discovery_run_id=actual_run_id,
            discovery_software_sha=self.software_sha,
            configuration_sha256=config_sha,
            as_of_boundary=config.as_of_boundary,
            jurisdiction_set=config.jurisdictions,
            created_at=run_timestamp,
        )

        identity_file = run_dir / "discovery_run_identity.json"
        checkpoint_file = run_dir / "discovery_checkpoint.jsonl"

        # Check resume validity if identity file already exists
        if identity_file.exists():
            existing_identity_doc = json.loads(identity_file.read_text(encoding="utf-8"))
            if (
                existing_identity_doc.get("configuration_sha256") != config_sha
                or existing_identity_doc.get("as_of_boundary") != config.as_of_boundary
            ):
                raise ResumeIdentityMismatchError("Existing run directory has mismatched identity/configuration")
        else:
            identity_file.write_text(json.dumps(identity.to_dict(), indent=2), encoding="utf-8")

        if interruption_stage == "C02":
            raise DiscoveryInterruptionError("Interrupted at C02 (After run identity freeze)")

        # 2. Check for completed marker (idempotency check)
        completed_marker = run_dir / ".discovery_completed"
        manifest_file = run_dir / "discovery_manifest.json"
        if completed_marker.exists() and manifest_file.exists():
            return json.loads(manifest_file.read_text(encoding="utf-8"))

        if interruption_stage == "C03":
            raise DiscoveryInterruptionError("Interrupted at C03 (Before first source request)")

        # 3. Resolve active adapters
        adapters: List[BaseDiscoveryAdapter] = []
        if custom_adapters is not None:
            adapters.extend(custom_adapters)
        else:
            for jur in config.jurisdictions:
                adapters.extend(self._get_adapters_for_jurisdiction(jur))

        # Checkpoint helper
        def append_checkpoint(event_type: str, data: Dict[str, Any]) -> None:
            entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event_type, "data": data}
            with open(checkpoint_file, "a", encoding="utf-8") as cp:
                cp.write(json.dumps(entry) + "\n")
                cp.flush()
                os.fsync(cp.fileno())

        # 4. Harvest raw registers and verify completeness per adapter
        all_raw_payloads: List[RawRegisterPayload] = []
        all_observations: List[RawDiscoveryObservation] = []
        all_tier1_parents: List[Tier1ParentObservation] = []
        all_tier2_expansions: List[Tier2ShareClassExpansion] = []
        source_completeness_map: Dict[str, str] = {}
        all_sources_complete = True
        completeness_failure_reason: Optional[str] = None

        run_context = type("RunContext", (), {
            "current_timestamp": run_timestamp,
            "as_of_boundary": config.as_of_boundary,
        })()

        for adapter_idx, adapter in enumerate(adapters):
            if interruption_stage == "C04" and adapter_idx == 0:
                raise DiscoveryInterruptionError("Interrupted at C04 (During source request)")

            # Fetch raw payloads with retry
            retries = 0
            payloads: List[RawRegisterPayload] = []
            while retries <= config.max_retries:
                try:
                    payloads = adapter.fetch_raw_register(run_context, self.transport)
                    break
                except SourceAdapterError as e:
                    retries += 1
                    if retries > config.max_retries:
                        all_sources_complete = False
                        completeness_failure_reason = f"Adapter {adapter.source_authority.value} failed: {e}"
                        source_completeness_map[adapter.source_authority.value] = SourceEnumerationState.FAILED.value
                        break
                    # Bounded exponential backoff (no real sleep needed in mock tests, or minimal)
                    time.sleep(min(config.backoff_max, config.backoff_factor * (2 ** (retries - 1))))

            if interruption_stage == "C05" and adapter_idx == 0:
                raise DiscoveryInterruptionError("Interrupted at C05 (After response, before evidence save)")

            if not payloads:
                all_sources_complete = False
                source_completeness_map[adapter.source_authority.value] = SourceEnumerationState.FAILED.value
                continue

            # Persist each payload into content-addressed evidence store
            for p in payloads:
                # Check fixture contamination in payload bytes or request URI
                if FORBIDDEN_FIXTURE_PATH_SUBSTRING in p.request_uri or FORBIDDEN_FIXTURE_PATH_SUBSTRING.encode("utf-8") in p.raw_bytes:
                    raise FixtureContaminationError(f"Fixture contamination detected in payload from {p.request_uri}")

                bin_path = raw_responses_dir / f"{adapter.source_authority.value}_{p.raw_sha256}.bin"
                meta_path = raw_responses_dir / f"{adapter.source_authority.value}_{p.raw_sha256}.meta.json"

                # Atomic write
                tmp_bin = bin_path.with_suffix(".tmp")
                tmp_bin.write_bytes(p.raw_bytes)
                tmp_bin.replace(bin_path)

                meta_doc = {
                    "source_authority": p.source_authority,
                    "jurisdiction": p.jurisdiction,
                    "request_uri": p.request_uri,
                    "response_status": p.response_status,
                    "content_type": p.content_type,
                    "raw_sha256": p.raw_sha256,
                    "retrieved_at": p.retrieved_at,
                    "byte_length": len(p.raw_bytes),
                }
                tmp_meta = meta_path.with_suffix(".tmp")
                tmp_meta.write_text(json.dumps(meta_doc, indent=2), encoding="utf-8")
                tmp_meta.replace(meta_path)

                all_raw_payloads.append(p)

            if interruption_stage == "C06" and adapter_idx == 0:
                raise DiscoveryInterruptionError("Interrupted at C06 (After evidence save, before parsing)")

            # Parse observations
            obs_list = adapter.parse_observations(payloads)

            if interruption_stage == "C07" and adapter_idx == 0:
                raise DiscoveryInterruptionError("Interrupted at C07 (During parsing)")

            # Verify source completeness
            comp_state, comp_err = adapter.verify_completeness(payloads, obs_list)
            source_completeness_map[adapter.source_authority.value] = comp_state.value
            if comp_state != SourceEnumerationState.COMPLETE:
                all_sources_complete = False
                completeness_failure_reason = comp_err or f"Source {adapter.source_authority.value} not complete"

            all_observations.extend(obs_list)

            # If adapter provides Tier-1 parent observations (Ireland / CBI)
            if hasattr(adapter, "parse_parent_observations") and adapter.jurisdiction.value == "IE":
                try:
                    parents = adapter.parse_parent_observations(payloads)
                    all_tier1_parents.extend(parents)
                except Exception:
                    pass

            # If adapter provides Tier-2 share-class expansions (Statutory Issuer)
            if hasattr(adapter, "parse_expansions") and adapter.source_tier == SourceAuthorityTier.TIER_2_STATUTORY_ISSUER:
                try:
                    expansions = adapter.parse_expansions(payloads)
                    all_tier2_expansions.extend(expansions)
                except Exception:
                    pass

            if interruption_stage == "C08" and adapter_idx == 0:
                raise DiscoveryInterruptionError("Interrupted at C08 (After parsed, before checkpoint)")

            append_checkpoint("ADAPTER_HARVESTED", {
                "source_authority": adapter.source_authority.value,
                "jurisdiction": adapter.jurisdiction.value,
                "observations_count": len(obs_list),
                "completeness": comp_state.value,
            })

            if interruption_stage == "C09" and adapter_idx == 0:
                raise DiscoveryInterruptionError("Interrupted at C09 (After checkpoint written)")

            if interruption_stage == "C11" and adapter_idx == 0:
                raise DiscoveryInterruptionError("Interrupted at C11 (Source complete, before next jurisdiction)")

        if interruption_stage == "C12":
            raise DiscoveryInterruptionError("Interrupted at C12 (After all jurisdictions complete)")

        # 5. Ireland Bounded Authority Decomposition Parent Join & Adjudication
        observations_by_isin: Dict[str, List[RawDiscoveryObservation]] = {}
        quarantined_observations: List[Dict[str, Any]] = []

        total_parents = len(all_tier1_parents)
        resolved_to_tier_2_count = 0
        unresolved_tier_2_count = 0
        ambiguous_tier_2_count = 0
        conflicted_tier_2_count = 0
        matched_exp_ids: Set[str] = set()

        if total_parents > 0 or len(all_tier2_expansions) > 0:
            expansions_by_full_key: Dict[Tuple[str, str], List[Tier2ShareClassExpansion]] = {}
            expansions_by_subfund: Dict[str, List[Tier2ShareClassExpansion]] = {}
            for exp in all_tier2_expansions:
                expansions_by_full_key.setdefault(exp.normalized_join_key, []).append(exp)
                expansions_by_subfund.setdefault(exp.normalized_subfund_key, []).append(exp)

            for parent in all_tier1_parents:
                parents_with_same_subfund = [p for p in all_tier1_parents if p.normalized_subfund_key == parent.normalized_subfund_key]
                if len(parents_with_same_subfund) > 1:
                    parents_with_same_full_key = [p for p in all_tier1_parents if p.normalized_join_key == parent.normalized_join_key]
                    if len(parents_with_same_full_key) > 1:
                        ambiguous_tier_2_count += 1
                        quarantined_observations.append({
                            "parent_id": parent.parent_id,
                            "subfund_name": parent.subfund_name,
                            "reason": QuarantineReason.COLLIDING_PARENT_SUBFUND.value,
                            "details": f"Multiple CBI Tier 1 parents have colliding normalized key: '{parent.subfund_name}'",
                        })
                        continue

                matching_exps = expansions_by_full_key.get(parent.normalized_join_key)
                if not matching_exps:
                    matching_exps = expansions_by_subfund.get(parent.normalized_subfund_key, [])

                if not matching_exps:
                    # ONE_TO_ZERO: Parent has no Tier 2 expansion
                    unresolved_tier_2_count += 1
                    quarantined_observations.append({
                        "parent_id": parent.parent_id,
                        "subfund_name": parent.subfund_name,
                        "reason": QuarantineReason.UNRESOLVED_TIER_2_PARENT.value,
                        "details": f"Tier 1 parent '{parent.subfund_name}' has no authoritative Tier 2 share-class expansion",
                    })
                    continue

                has_conflict = False
                for exp in matching_exps:
                    if parent.status == "TERMINATED" and exp.status == "ACTIVE":
                        has_conflict = True
                        break
                    if not exp.is_ucits and parent.is_ucits:
                        has_conflict = True
                        break

                if has_conflict:
                    conflicted_tier_2_count += 1
                    quarantined_observations.append({
                        "parent_id": parent.parent_id,
                        "subfund_name": parent.subfund_name,
                        "reason": QuarantineReason.CONFLICTED_TIER_2_RECORD.value,
                        "details": "Contradiction between Tier 1 parent and Tier 2 expansion",
                    })
                    continue

                incomplete_exps = [e for e in matching_exps if e.completeness != ShareClassExpansionCompleteness.COMPLETE.value]
                if incomplete_exps:
                    unresolved_tier_2_count += 1
                    quarantined_observations.append({
                        "parent_id": parent.parent_id,
                        "subfund_name": parent.subfund_name,
                        "reason": QuarantineReason.PARTIAL_SHARE_CLASS_EXPANSION.value,
                        "details": "Tier 2 share-class schedule is PARTIAL or NOT_ESTABLISHED; population member requires COMPLETE",
                    })
                    continue

                # Clean match
                resolved_to_tier_2_count += 1
                for exp in matching_exps:
                    matched_exp_ids.add(exp.expansion_id)
                    joined_obs = RawDiscoveryObservation(
                        observation_id=f"joined_{parent.parent_id}_{exp.expansion_id}",
                        source_authority=SourceAuthorityId.CENTRAL_BANK_OF_IRELAND.value,
                        source_authority_tier=SourceAuthorityTier.TIER_1_NCA.value,
                        retrieved_at=exp.retrieved_at,
                        source_as_of=exp.source_as_of,
                        raw_identifier=exp.share_class_isin,
                        normalized_isin=exp.share_class_isin.upper(),
                        fund_name_raw=parent.subfund_name,
                        share_class_name_raw=exp.share_class_name,
                        domicile_raw="IE",
                        is_ucits_raw=parent.is_ucits and exp.is_ucits,
                        is_etf_raw=parent.is_etf or exp.is_etf,
                        listing_status_raw=parent.status if parent.status == "TERMINATED" else exp.status,
                        source_record_uri=exp.source_record_uri,
                        source_payload_sha256=exp.source_payload_sha256,
                        raw_attributes={
                            "parent_id": parent.parent_id,
                            "umbrella_name": parent.umbrella_name,
                            "subfund_name": parent.subfund_name,
                            "parent_source_authority": parent.source_authority,
                            "parent_source_payload_sha256": parent.source_payload_sha256,
                            "parent_source_record_uri": parent.source_record_uri,
                            "expansion_source_authority": exp.source_authority,
                            "expansion_source_payload_sha256": exp.source_payload_sha256,
                            "authorization_date": parent.authorization_date,
                            "termination_date": parent.termination_date,
                        },
                    )
                    all_observations.append(joined_obs)

            extra_exps = [e for e in all_tier2_expansions if e.expansion_id not in matched_exp_ids]
            for exp in extra_exps:
                quarantined_observations.append({
                    "expansion_id": exp.expansion_id,
                    "share_class_isin": exp.share_class_isin,
                    "subfund_name": exp.subfund_name,
                    "reason": QuarantineReason.EXTRA_TIER_2_WITHOUT_TIER_1_PARENT.value,
                    "details": f"Tier 2 expansion '{exp.subfund_name}' lacks authoritative Tier 1 parent in CBI register",
                })

        tier_1_parent_accounting = Tier1ParentAccounting.calculate(
            total=total_parents,
            resolved=resolved_to_tier_2_count,
            unresolved=unresolved_tier_2_count,
            ambiguous=ambiguous_tier_2_count,
            conflicted=conflicted_tier_2_count,
        )
        if not tier_1_parent_accounting.is_conserved:
            raise ConservationViolationError(
                f"Tier 1 parent accounting conservation violation: {tier_1_parent_accounting.to_dict()}"
            )

        # 6. Adjudicate, Filter, and Normalize Observations
        raw_discovered_count = len(all_observations)
        parsed_count = 0
        unparseable_count = 0
        candidate_obs_count = 0
        out_of_scope_count = 0
        invalid_identifier_count = 0
        quarantined_count = 0

        for obs in all_observations:
            # Check for unparseable raw identifier
            if not obs.raw_identifier:
                parsed_count += 1
                invalid_identifier_count += 1
                quarantined_observations.append({
                    "observation_id": obs.observation_id,
                    "reason": QuarantineReason.MISSING_IDENTIFIER.value,
                    "details": "Record has empty identifier",
                })
                continue

            parsed_count += 1

            # Validate ISIN strictly via global_identifier_authority
            try:
                norm_isin = normalize_isin(obs.raw_identifier)
                is_valid = validate_isin(norm_isin, strict=False)
            except Exception:
                norm_isin = obs.raw_identifier.strip().upper()
                is_valid = False

            if not is_valid:
                invalid_identifier_count += 1
                quarantined_observations.append({
                    "observation_id": obs.observation_id,
                    "raw_identifier": obs.raw_identifier,
                    "reason": QuarantineReason.INVALID_CHECKSUM.value,
                    "details": "Failed ISO 6166 check-digit or format validation",
                })
                continue

            # Scope Filtering: Domicile, UCITS, ETF
            valid_jurisdiction_values = {j for j in config.jurisdictions}
            if obs.domicile_raw not in valid_jurisdiction_values:
                out_of_scope_count += 1
                continue

            if not obs.is_ucits_raw or not obs.is_etf_raw:
                out_of_scope_count += 1
                continue

            # Candidate observation accepted
            candidate_obs_count += 1
            observations_by_isin.setdefault(norm_isin, []).append(obs)

        # 6. Multi-Source Conflict Adjudication & Deduplication
        unique_candidates: List[CandidateSpec] = []
        duplicate_observations_count = 0

        # Sort ISIN keys ascending for deterministic ordering
        sorted_isins = sorted(observations_by_isin.keys())

        for isin in sorted_isins:
            obs_group = observations_by_isin[isin]

            # Product status check: if all observations are TERMINATED, classify as out-of-scope
            statuses = {o.listing_status_raw for o in obs_group}
            if statuses == {"TERMINATED"}:
                candidate_obs_count -= len(obs_group)
                out_of_scope_count += len(obs_group)
                continue

            # Status conflict check (e.g. Active in Tier 1 vs Terminated in another)
            if "TERMINATED" in statuses and "ACTIVE" in statuses:
                candidate_obs_count -= len(obs_group)
                quarantined_count += len(obs_group)
                quarantined_observations.append({
                    "share_class_isin": isin,
                    "reason": QuarantineReason.STATUS_CONTRADICTION.value,
                    "details": f"Conflicting listing statuses: {statuses}",
                })
                continue

            # Check for Tier 1 vs Tier 2 precedence
            tier_1_obs = [o for o in obs_group if o.source_authority_tier == SourceAuthorityTier.TIER_1_NCA.value]
            tier_2_obs = [o for o in obs_group if o.source_authority_tier == SourceAuthorityTier.TIER_2_STATUTORY_ISSUER.value]

            # Solitary Tier 2 check: if no Tier 1 observation exists and allow_tier_2_expansion is False
            if not tier_1_obs and tier_2_obs and not config.allow_tier_2_expansion:
                candidate_obs_count -= len(obs_group)
                quarantined_count += len(obs_group)
                quarantined_observations.append({
                    "share_class_isin": isin,
                    "reason": QuarantineReason.TIER_2_UNCONFIRMED.value,
                    "details": "Present in Tier 2 statutory issuer but uncorroborated by Tier 1 NCA",
                })
                continue

            # Domicile conflict check
            domiciles = {o.domicile_raw for o in obs_group}
            if len(domiciles) > 1:
                candidate_obs_count -= len(obs_group)
                quarantined_count += len(obs_group)
                quarantined_observations.append({
                    "share_class_isin": isin,
                    "reason": QuarantineReason.DOMICILE_CONTRADICTION.value,
                    "details": f"Conflicting domiciles across observations: {domiciles}",
                })
                continue

            if len(obs_group) > 1:
                duplicate_observations_count += (len(obs_group) - 1)

            # Primary observation selection (Tier 1 takes precedence)
            primary_obs = tier_1_obs[0] if tier_1_obs else tier_2_obs[0]

            # Build provenance chain
            prov_chain: List[ObservationProvenance] = []
            listing_venues: Set[str] = set()

            parent_subfund = primary_obs.raw_attributes.get("subfund_name")
            parent_umbrella = primary_obs.raw_attributes.get("umbrella_name")
            if primary_obs.raw_attributes.get("parent_source_authority"):
                # Bounded Authority Decomposition dual provenance
                prov_chain.append(
                    ObservationProvenance(
                        source_authority=primary_obs.raw_attributes["parent_source_authority"],
                        source_tier=SourceAuthorityTier.TIER_1_NCA.value,
                        source_record_id=primary_obs.raw_attributes["parent_id"],
                        source_record_uri=primary_obs.raw_attributes["parent_source_record_uri"],
                        source_payload_sha256=primary_obs.raw_attributes["parent_source_payload_sha256"],
                        retrieved_at=primary_obs.retrieved_at,
                        authority_function=AuthorityFunction.POPULATION_MEMBERSHIP.value,
                        parent_subfund_id=primary_obs.raw_attributes["parent_id"],
                        raw_metadata={"umbrella_name": parent_umbrella, "subfund_name": parent_subfund},
                    )
                )
                prov_chain.append(
                    ObservationProvenance(
                        source_authority=primary_obs.raw_attributes.get("expansion_source_authority", primary_obs.source_authority),
                        source_tier=SourceAuthorityTier.TIER_2_STATUTORY_ISSUER.value,
                        source_record_id=primary_obs.observation_id,
                        source_record_uri=primary_obs.source_record_uri,
                        source_payload_sha256=primary_obs.source_payload_sha256,
                        retrieved_at=primary_obs.retrieved_at,
                        authority_function=AuthorityFunction.SHARE_CLASS_EXPANSION.value,
                        parent_subfund_id=primary_obs.raw_attributes["parent_id"],
                        raw_metadata={"listing_status": primary_obs.listing_status_raw},
                    )
                )
                for o in obs_group:
                    if o.observation_id == primary_obs.observation_id:
                        continue
                    auth_func = (
                        AuthorityFunction.SHARE_CLASS_EXPANSION.value
                        if o.source_authority_tier == SourceAuthorityTier.TIER_2_STATUTORY_ISSUER.value
                        else AuthorityFunction.POPULATION_MEMBERSHIP.value
                    )
                    prov_chain.append(
                        ObservationProvenance(
                            source_authority=o.source_authority,
                            source_tier=o.source_authority_tier,
                            source_record_id=o.observation_id,
                            source_record_uri=o.source_record_uri,
                            source_payload_sha256=o.source_payload_sha256,
                            retrieved_at=o.retrieved_at,
                            authority_function=auth_func,
                            raw_metadata={"listing_status": o.listing_status_raw},
                        )
                    )
                    venue = o.raw_attributes.get("venue", o.raw_attributes.get("mic"))
                    if venue:
                        listing_venues.add(str(venue))
            else:
                for o in obs_group:
                    auth_func = (
                        AuthorityFunction.SHARE_CLASS_EXPANSION.value
                        if o.source_authority_tier == SourceAuthorityTier.TIER_2_STATUTORY_ISSUER.value
                        else AuthorityFunction.POPULATION_MEMBERSHIP.value
                    )
                    prov_chain.append(
                        ObservationProvenance(
                            source_authority=o.source_authority,
                            source_tier=o.source_authority_tier,
                            source_record_id=o.observation_id,
                            source_record_uri=o.source_record_uri,
                            source_payload_sha256=o.source_payload_sha256,
                            retrieved_at=o.retrieved_at,
                            authority_function=auth_func,
                            raw_metadata={"listing_status": o.listing_status_raw},
                        )
                    )
                    venue = o.raw_attributes.get("venue", o.raw_attributes.get("mic"))
                    if venue:
                        listing_venues.add(str(venue))

            candidate = CandidateSpec(
                share_class_isin=isin,
                domicile=primary_obs.domicile_raw,
                fund_name=primary_obs.fund_name_raw,
                share_class_name=primary_obs.share_class_name_raw,
                is_ucits=True,
                is_etf=True,
                status=CandidateStatus.ACTIVE.value,
                provenance_chain=tuple(prov_chain),
                listing_venues=tuple(sorted(list(listing_venues))),
                authorization_date=primary_obs.raw_attributes.get("authorization_date"),
                termination_date=primary_obs.raw_attributes.get("termination_date"),
                parent_subfund_name=parent_subfund,
                parent_umbrella_name=parent_umbrella,
            )
            unique_candidates.append(candidate)

        # 7. Accounting & Conservation Verification
        accounting = DiscoveryAccounting.calculate(
            raw_discovered=raw_discovered_count,
            parsed=parsed_count,
            unparseable=unparseable_count,
            candidate_obs=candidate_obs_count,
            out_of_scope=out_of_scope_count,
            invalid_id=invalid_identifier_count,
            quarantined=quarantined_count,
            unique_candidates=len(unique_candidates),
            duplicate_obs=duplicate_observations_count,
        )

        if not accounting.is_conserved:
            raise ConservationViolationError(
                f"Discovery accounting conservation violation: {accounting.to_dict()}"
            )

        if interruption_stage == "C13":
            raise DiscoveryInterruptionError("Interrupted at C13 (Before discovery manifest write)")

        # 8. Compute Aggregate Evidence Hash
        # SHA-256 over sorted observation hashes
        obs_shas = sorted([o.source_payload_sha256 for o in all_observations])
        aggregate_evidence_sha256 = hashlib.sha256("\n".join(obs_shas).encode("utf-8")).hexdigest()

        # Overall completeness determination
        final_completeness = (
            SourceEnumerationState.COMPLETE.value
            if all_sources_complete
            else SourceEnumerationState.FAILED.value
        )

        # 9. Build and atomically write discovery_manifest.json
        manifest_data = {
            "schema_version": "1.0.0",
            "discovery_run_id": actual_run_id,
            "software_sha": self.software_sha,
            "configuration_sha256": config_sha,
            "as_of_boundary": config.as_of_boundary,
            "jurisdiction_scope": list(config.jurisdictions),
            "completeness_state": final_completeness,
            "completeness_failure_reason": completeness_failure_reason,
            "source_completeness_map": source_completeness_map,
            "accounting": accounting.to_dict(),
            "tier_1_parent_accounting": tier_1_parent_accounting.to_dict(),
            "aggregate_evidence_sha256": aggregate_evidence_sha256,
            "quarantined_observations": quarantined_observations,
            "candidate_count": len(unique_candidates),
            "candidates": [c.to_dict() for c in unique_candidates],
        }

        if interruption_stage == "C14":
            tmp_m = manifest_file.with_suffix(".tmp")
            tmp_m.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
            raise DiscoveryInterruptionError("Interrupted at C14 (During manifest write)")

        tmp_manifest = manifest_file.with_suffix(".tmp")
        tmp_manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        tmp_manifest.replace(manifest_file)

        if interruption_stage == "C15":
            raise DiscoveryInterruptionError("Interrupted at C15 (After manifest rename, before marker)")

        # Write completion marker
        completed_marker.write_text(
            json.dumps({"run_id": actual_run_id, "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest()}),
            encoding="utf-8",
        )

        append_checkpoint("DISCOVERY_COMPLETED", {"candidate_count": len(unique_candidates), "completeness": final_completeness})

        return manifest_data

    def replay_discovery(
        self,
        run_dir: Path,
        config: DiscoveryConfiguration,
        custom_adapters: Optional[List[BaseSourceAdapter]] = None,
    ) -> Dict[str, Any]:
        """
        Replays discovery from preserved raw response files without network access.
        Verifies bit-for-bit determinism of candidates, counts, and aggregate evidence hash.
        """
        raw_responses_dir = run_dir / "raw_responses"
        if not raw_responses_dir.exists():
            raise CorruptedDiscoveryCacheError(f"Missing raw_responses directory: {raw_responses_dir}")

        manifest_file = run_dir / "discovery_manifest.json"
        if not manifest_file.exists():
            raise CorruptedDiscoveryCacheError(f"Missing discovery_manifest.json: {manifest_file}")

        original_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

        # Re-read and verify all raw files
        bin_files = list(raw_responses_dir.glob("*.bin"))
        if not bin_files:
            raise CorruptedDiscoveryCacheError("No raw .bin evidence files found for replay")

        payloads: List[RawRegisterPayload] = []
        for bfile in bin_files:
            content = bfile.read_bytes()
            meta_file = bfile.with_suffix(".meta.json")
            if not meta_file.exists():
                raise CorruptedDiscoveryCacheError(f"Missing metadata sidecar for {bfile}")
            meta = json.loads(meta_file.read_text(encoding="utf-8"))

            computed_sha = hashlib.sha256(content).hexdigest()
            if computed_sha != meta["raw_sha256"]:
                raise CorruptedDiscoveryCacheError(
                    f"Evidence tampering detected: {bfile.name} computed={computed_sha} != meta={meta['raw_sha256']}"
                )

            payloads.append(
                RawRegisterPayload(
                    source_authority=meta["source_authority"],
                    jurisdiction=meta["jurisdiction"],
                    request_uri=meta["request_uri"],
                    response_status=meta["response_status"],
                    content_type=meta["content_type"],
                    raw_bytes=content,
                    raw_sha256=computed_sha,
                    retrieved_at=meta["retrieved_at"],
                )
            )

        # Build mock transport preloaded with preserved files
        replay_transport = MockDiscoveryTransport()
        for p in payloads:
            replay_transport.register_response(p.request_uri, p.response_status, p.raw_bytes, {"content-type": p.content_type})

        # Re-execute discovery into a fresh temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_authority = UCITSDiscoveryAuthority(
                cache_dir=Path(tmpdir),
                software_sha=self.software_sha,
                transport=replay_transport,
            )
            replayed_manifest = replay_authority.execute_discovery(
                config,
                run_id="replay_run",
                custom_adapters=custom_adapters,
            )

        # Verify bit-for-bit identity of accounting and candidates
        assert replayed_manifest["candidate_count"] == original_manifest["candidate_count"], "Replay count mismatch"
        assert replayed_manifest["aggregate_evidence_sha256"] == original_manifest["aggregate_evidence_sha256"], "Replay aggregate evidence hash mismatch"
        assert replayed_manifest["accounting"] == original_manifest["accounting"], "Replay accounting mismatch"
        assert replayed_manifest["candidates"] == original_manifest["candidates"], "Replay candidates mismatch"
        assert replayed_manifest.get("tier_1_parent_accounting") == original_manifest.get("tier_1_parent_accounting"), "Replay parent accounting mismatch"

        return replayed_manifest

    def handoff_to_denominator_snapshot(self, manifest_path: Path, output_snapshot_path: Path) -> Dict[str, Any]:
        """
        Validates completeness and produces canonical candidate inputs for UCITSDenominatorSnapshot.
        Fails closed if the discovery run was incomplete or partial.
        """
        if not manifest_path.exists():
            raise DiscoveryCompletenessError(f"Discovery manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completeness = manifest.get("completeness_state")
        if completeness != SourceEnumerationState.COMPLETE.value:
            raise DiscoveryCompletenessError(
                f"Cannot derive denominator: discovery run is {completeness} (reason: {manifest.get('completeness_failure_reason')})"
            )

        candidates = manifest.get("candidates", [])
        # Strict normalized ISIN ascending sort
        sorted_candidates = sorted(candidates, key=lambda c: normalize_isin(c["share_class_isin"]))

        # Structure denominator seed snapshot
        snapshot_doc = {
            "snapshot_version": "1.0.0",
            "as_of_boundary": manifest["as_of_boundary"],
            "discovery_run_id": manifest["discovery_run_id"],
            "discovery_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "candidate_count": len(sorted_candidates),
            "candidates": sorted_candidates,
        }

        output_path = Path(output_snapshot_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(snapshot_doc, indent=2, sort_keys=True)
        tmp_out = output_path.with_suffix(".tmp")
        tmp_out.write_text(serialized, encoding="utf-8")
        tmp_out.replace(output_path)

        return snapshot_doc
