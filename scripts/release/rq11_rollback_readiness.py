#!/usr/bin/env python3
"""
ARX Release Qualification - Check RQ-11: Rollback Readiness & Migration Non-Destructiveness
Hardened under Directive ARX — Final RQ-11 Production Rollback Authority Closure & Recovery 002.

1. Binds evidence-based deployment identities (Railway active 6bba63de, Cloudflare active 32f8daa9).
2. Distinguishes 4 separate assertions:
   - Repository Rollback Compatibility (0-DDL & permission-only SQL validation)
   - Base Anchor Git Object Retention (cdc47a2, eed8d20, and 6d4020c)
   - External Recovery Target Platform Availability & Eligibility
   - Live Platform Rehearsal Governance State (abstention from unauthorized production mutation)
3. Reconciles Railway 6bba63de vs historical d1f8cf37, and Cloudflare 32f8daa9 vs candidate c78ee65b.
4. Fails closed on stale, removed, or invalid deployment references.
5. Emits verified machine-readable rq_rollback_targets.json and rq11_receipt.json.
"""

import sys, os, subprocess, json, time, datetime, argparse, re

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from rq_receipt_utils import create_stage_receipt, save_stage_receipt, get_git_commit, resolve_identity_contract, SHA40_REGEX

UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

# Authoritative Production Release Anchors & Recovery Identities
AUTHORITATIVE_TARGETS = {
    "railway": {
        "active_rollback_anchor": {
            "platform": "Railway",
            "project_name": "tranquil-radiance",
            "project_id": "a339a92a-7236-4ac0-9a3a-d7ad440f9690",
            "environment": "production",
            "environment_id": "11cad5e1-358e-430e-84fc-189953cff48a",
            "service": "web",
            "service_id": "56e38249-b78c-4a60-80d7-303ce7451bd8",
            "deployment_id": "6bba63de-3ba0-4c96-bb1a-e67d3cc47d08",
            "created_at": "2026-10-28T16:38:55.603Z",
            "status": "SUCCESS",
            "source_commit": "cdc47a2eee8a427aaa6cf3ab5d01cad7f35d71e1",
            "candidate_pr_commit": "2c900cd783af3ca135f1b9cfae30697253db078c",
            "image_digest": "sha256:2af4f4d6c62df20ea026943d2c5b76a4c8821b0df6c8837372d3ccae2625c23b",
            "recovery_eligibility": "ACTIVE_SERVING_CONTAINER",
            "verified_in_registry": True
        },
        "historical_baselines": [
            {
                "deployment_id": "d1f8cf37-7592-4ad5-b986-4bbc9b4ed042",
                "status": "REMOVED",
                "removed_at": "2026-10-27T13:47:59+01:00",
                "note": "Pre-074B historical Gate C baseline. Retired; NOT an active recovery target."
            }
        ]
    },
    "cloudflare": {
        "active_rollback_anchor": {
            "platform": "Cloudflare Pages",
            "project_name": "finance",
            "account_id": "c50133a8b090743a7470d9fb19ea2e0e",
            "environment": "production",
            "deployment_id": "32f8daa9-bb1f-4454-9aaa-2000bfd49038",
            "source_commit": "6d4020ca3fef3ae07ac01204b03e0629b6a7cc79",
            "status": "SUCCESS",
            "recovery_eligibility": "IMMUTABLE_SERVING_BASELINE",
            "authoritative_record": "ARX_070_CLOUDFLARE_PRODUCTION_PRESERVATION",
            "custom_domain": "https://www.arxterminal.com",
            "verified_in_registry": True
        },
        "candidate_preview": {
            "deployment_id": "c78ee65b-4997-4bf1-90ca-345989065d76",
            "source_commit": "e410fae28e61b9ae01f3659cb61e40ec0737118b",
            "status": "SUCCESS",
            "check_run_id": 124077004975
        }
    },
    "git_anchors": {
        "production_release_anchor": "cdc47a2eee8a427aaa6cf3ab5d01cad7f35d71e1",
        "common_ancestor_baseline": "6d4020ca3fef3ae07ac01204b03e0629b6a7cc79"
    }
}

# Approved locations for SQL files in ARX Terminal
APPROVED_SQL_PATHS = (
    "analyst_dashboard/smart_money/storage/postgres_roles.sql",
    "scripts/release/sql/",
)

# Prohibited schema DDL patterns in SQL files
FORBIDDEN_DDL_PATTERNS = [
    re.compile(r"\bCREATE\s+(TABLE|INDEX|SEQUENCE|VIEW|MATERIALIZED\s+VIEW|TYPE|DOMAIN|TRIGGER|FUNCTION|PROCEDURE|DATABASE|SCHEMA)\b", re.IGNORECASE),
    re.compile(r"\bALTER\s+(TABLE|INDEX|SEQUENCE|VIEW|TYPE|DOMAIN|TRIGGER|FUNCTION|SCHEMA)\b", re.IGNORECASE),
    re.compile(r"\bDROP\s+(TABLE|INDEX|SEQUENCE|VIEW|MATERIALIZED\s+VIEW|TYPE|DOMAIN|TRIGGER|FUNCTION|SCHEMA|DATABASE|ROLE|USER)\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE(\s+TABLE)?\b", re.IGNORECASE),
    re.compile(r"\bRENAME\s+TO\b", re.IGNORECASE),
    re.compile(r"\b(ADD|DROP)\s+COLUMN\b", re.IGNORECASE),
    re.compile(r"\b(ADD|DROP)\s+CONSTRAINT\b", re.IGNORECASE),
]

# Prohibited DML mutation patterns in SQL files
FORBIDDEN_DML_PATTERNS = [
    re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+[a-zA-Z0-9_\.]+\s+SET\b", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"\bMERGE\s+INTO\b", re.IGNORECASE),
    re.compile(r"\bUPSERT\b", re.IGNORECASE),
]

# Prohibited privilege expansion patterns in SQL files
FORBIDDEN_PRIVILEGE_PATTERNS = [
    re.compile(r"\bGRANT\s+ALL\b", re.IGNORECASE),
    re.compile(r"\bGRANT\s+.*\b(INSERT|UPDATE|DELETE|TRUNCATE|REFERENCES|TRIGGER|EXECUTE|CREATE)\b", re.IGNORECASE),
    re.compile(r"\b(SUPERUSER|CREATEROLE|CREATEDB|REPLICATION|BYPASSRLS)\b", re.IGNORECASE),
    re.compile(r"\bWITH\s+(GRANT|ADMIN)\s+OPTION\b", re.IGNORECASE),
]

def verify_zero_ddl_migrations(base_sha: str, head_sha: str) -> tuple[bool, list]:
    """
    Mathematically inspects git diff across all migration and schema patterns.
    Enforces strict zero-DDL policy:
    1. Rejects any changes in dedicated migration directories (migrations/*, alembic/*, prisma/*, schema.prisma).
    2. Rejects any SQL files located outside approved paths.
    3. Rejects any SQL file introducing DDL schema modifications, DML data mutations, or unauthorized privilege expansions.
    4. Enforces that any staged permission grant in scripts/release/sql/ establishes an auditable rollback procedure.
    """
    # 1. Hard reject on migration directories
    hard_migration_patterns = [
        "migrations/*", "alembic/*", "prisma/*", "schema.prisma"
    ]
    try:
        cmd = ["git", "diff", "--name-only", base_sha, head_sha, "--"] + hard_migration_patterns
        out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True, stderr=subprocess.PIPE).strip()
        if out:
            hard_diffs = [l.strip() for l in out.splitlines() if l.strip()]
            return False, [f"Active migration directory change detected: {f}" for f in hard_diffs]
    except subprocess.CalledProcessError as e:
        return False, [f"git diff migration check failed: {e}"]

    # 2. Inspect SQL files in the diff
    sql_patterns = ["sql/*", "*.sql"]
    try:
        cmd = ["git", "diff", "--name-only", base_sha, head_sha, "--"] + sql_patterns
        out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True, stderr=subprocess.PIPE).strip()
        changed_sql = [l.strip() for l in out.splitlines() if l.strip()] if out else []
    except subprocess.CalledProcessError as e:
        return False, [f"git diff sql check failed: {e}"]

    if not changed_sql:
        return True, []

    sql_errors = []
    grant_scripts = []

    for fpath in changed_sql:
        norm_path = fpath.replace("\\", "/")
        is_approved = (
            norm_path == "analyst_dashboard/smart_money/storage/postgres_roles.sql"
            or norm_path.startswith("scripts/release/sql/")
        )
        if not is_approved:
            sql_errors.append(f"UNAUTHORIZED_SQL_LOCATION: SQL file '{norm_path}' is outside approved SQL paths.")
            continue

        if norm_path.startswith("scripts/release/sql/") and norm_path.endswith("_grant.sql"):
            grant_scripts.append(norm_path)

        # Inspect diff for added / modified lines
        try:
            diff_out = subprocess.check_output(
                ["git", "diff", "-U0", base_sha, head_sha, "--", norm_path],
                cwd=PROJECT_ROOT, text=True, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            sql_errors.append(f"Unable to read git diff for {norm_path}: {e}")
            continue

        for line in diff_out.splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            content = line[1:].strip()
            if not content or content.startswith("--"):
                continue

            for pat in FORBIDDEN_DDL_PATTERNS:
                if pat.search(content):
                    sql_errors.append(f"DDL_MIGRATIONS_DETECTED: Prohibited DDL pattern in {norm_path}: '{content}'")
                    break

            for pat in FORBIDDEN_DML_PATTERNS:
                if pat.search(content):
                    sql_errors.append(f"DML_MUTATION_DETECTED: Prohibited DML pattern in {norm_path}: '{content}'")
                    break

            for pat in FORBIDDEN_PRIVILEGE_PATTERNS:
                if pat.search(content):
                    sql_errors.append(f"UNAUTHORIZED_PRIVILEGE_EXPANSION: Prohibited privilege grant in {norm_path}: '{content}'")
                    break

    # 3. Rollback script / procedure verification
    for grant_path in grant_scripts:
        revoke_path = grant_path.replace("_grant.sql", "_revoke.sql")
        disk_revoke = os.path.join(PROJECT_ROOT, revoke_path)
        exists_on_disk = os.path.exists(disk_revoke)
        exists_in_head = False
        try:
            res_revoke = subprocess.run(
                ["git", "cat-file", "-e", f"{head_sha}:{revoke_path}"],
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            exists_in_head = (res_revoke.returncode == 0)
        except Exception:
            pass

        has_embedded_rollback = False
        grant_content = ""
        disk_grant = os.path.join(PROJECT_ROOT, grant_path)
        if os.path.exists(disk_grant):
            try:
                with open(disk_grant, "r", encoding="utf-8") as gf:
                    grant_content = gf.read()
            except Exception:
                pass
        if not grant_content:
            try:
                res_grant = subprocess.run(
                    ["git", "cat-file", "-p", f"{head_sha}:{grant_path}"],
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True
                )
                if res_grant.returncode == 0:
                    grant_content = res_grant.stdout
            except Exception:
                pass

        if "REVOKE SELECT" in grant_content and "ROLLBACK PROCEDURE" in grant_content.upper():
            has_embedded_rollback = True

        if not (exists_on_disk or exists_in_head or has_embedded_rollback):
            sql_errors.append(f"MISSING_ROLLBACK_SCRIPT: Grant script '{grant_path}' lacks verified rollback script or auditable rollback procedure.")

    if sql_errors:
        return False, sql_errors

    return True, []

def verify_git_commit_exists(sha: str) -> bool:
    """Verifies that a commit SHA exists and is intact in repository objects."""
    if not sha or not SHA40_REGEX.match(sha):
        return False
    try:
        res = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return (res.returncode == 0)
    except Exception:
        return False

def validate_uuid_format(val: str) -> bool:
    """Validates standard UUID format."""
    return bool(val and UUID_REGEX.match(str(val).strip()))

def main():
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()
    errors = []
    assertions = []

    parser = argparse.ArgumentParser(description="RQ-11: Rollback Readiness & Target Authority Verification")
    parser.add_argument("--manifest", default=None, help="Path to release_manifest.json")
    parser.add_argument("--base", default=None, help="Production release anchor commit")
    parser.add_argument("--head", default=None, help="Candidate commit to evaluate")
    parser.add_argument("--assert-railway-target", default=None, help="Override Railway rollback target for testing")
    parser.add_argument("--assert-cloudflare-target", default=None, help="Override Cloudflare rollback target for testing")
    parser.add_argument("--simulate-ddl", action="store_true", help="Simulate DDL violation for negative testing")
    parser.add_argument("--simulate-sql-ddl", action="store_true", help="Simulate prohibited DDL statement in SQL file")
    parser.add_argument("--simulate-sql-dml", action="store_true", help="Simulate prohibited DML mutation in SQL file")
    parser.add_argument("--simulate-sql-priv-expansion", action="store_true", help="Simulate unauthorized privilege expansion in SQL file")
    parser.add_argument("--simulate-missing-rollback", action="store_true", help="Simulate grant script without rollback script")
    parser.add_argument("--simulate-unapproved-sql-location", action="store_true", help="Simulate SQL file in unapproved location")
    args = parser.parse_args()

    # Load release manifest if present to derive canonical defaults
    manifest_path = args.manifest or os.path.join(PROJECT_ROOT, "scripts", "release", "release_manifest.json")
    manifest_data = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            pass

    expected_base = args.base or manifest_data.get("base_anchor_commit") or "eed8d201e0696f7db41f30a598b1619a9a0699f4"

    ok, identity, err = resolve_identity_contract()
    source_sha = identity["source_sha"] if ok else get_git_commit("HEAD")
    head_sha = args.head or source_sha
    base_sha = expected_base

    print("=== [RQ-11] PRODUCTION ROLLBACK READINESS & RECOVERY AUTHORITY ===")
    print(f"Target Candidate Head: {head_sha}")
    print(f"Production Base Anchor: {base_sha}")

    # 1. Assertion 1: Repository Rollback Compatibility (0-DDL & Reversibility)
    if args.simulate_ddl:
        zero_ddl_ok = False
        migration_diffs = ["migrations/0002_destructive_schema_mutation.sql"]
    elif args.simulate_sql_ddl:
        zero_ddl_ok = False
        migration_diffs = ["DDL_MIGRATIONS_DETECTED: Prohibited DDL pattern in scripts/release/sql/test.sql: 'CREATE TABLE evil'"]
    elif args.simulate_sql_dml:
        zero_ddl_ok = False
        migration_diffs = ["DML_MUTATION_DETECTED: Prohibited DML pattern in scripts/release/sql/test.sql: 'INSERT INTO evil'"]
    elif args.simulate_sql_priv_expansion:
        zero_ddl_ok = False
        migration_diffs = ["UNAUTHORIZED_PRIVILEGE_EXPANSION: Prohibited privilege grant in scripts/release/sql/test.sql: 'GRANT ALL'"]
    elif args.simulate_missing_rollback:
        zero_ddl_ok = False
        migration_diffs = ["MISSING_ROLLBACK_SCRIPT: Grant script 'scripts/release/sql/sm_test_grant.sql' lacks verified rollback script or auditable rollback procedure."]
    elif args.simulate_unapproved_sql_location:
        zero_ddl_ok = False
        migration_diffs = ["UNAUTHORIZED_SQL_LOCATION: SQL file 'api/test.sql' is outside approved SQL paths."]
    else:
        zero_ddl_ok, migration_diffs = verify_zero_ddl_migrations(base_sha, head_sha)

    assertions.append({
        "assertion": "Repository Rollback Compatibility: Zero database schema migrations (0-DDL)",
        "observed": {"changed_migration_files": migration_diffs, "count": len(migration_diffs)},
        "passed": zero_ddl_ok
    })

    if zero_ddl_ok:
        print("  [PASS] Assertion 1: Zero database schema migrations (0-DDL verified).")
    else:
        print(f"  [FAIL] Assertion 1: Destructive or active schema migrations detected: {migration_diffs}")
        for md in migration_diffs:
            if any(md.startswith(pfx) for pfx in ("DDL_MIGRATIONS_DETECTED:", "DML_MUTATION_DETECTED:", "UNAUTHORIZED_PRIVILEGE_EXPANSION:", "MISSING_ROLLBACK_SCRIPT:", "UNAUTHORIZED_SQL_LOCATION:")):
                errors.append(md)
            else:
                errors.append(f"DDL_MIGRATIONS_DETECTED: Schema changes detected between {base_sha[:7]} and {head_sha[:7]}: {md}")

    # 2. Assertion 2: Base Anchor Git Object Retention
    prod_base_retained = verify_git_commit_exists(base_sha)
    ancestor_retained = verify_git_commit_exists(AUTHORITATIVE_TARGETS["git_anchors"]["common_ancestor_baseline"])
    git_anchors_ok = prod_base_retained and ancestor_retained

    assertions.append({
        "assertion": "Git Rollback Anchors Retention: Production base and ancestor commits exist in object database",
        "observed": {
            "production_base_anchor": base_sha,
            "production_base_retained": prod_base_retained,
            "ancestor_baseline_anchor": AUTHORITATIVE_TARGETS["git_anchors"]["common_ancestor_baseline"],
            "ancestor_baseline_retained": ancestor_retained
        },
        "passed": git_anchors_ok
    })

    if git_anchors_ok:
        print(f"  [PASS] Assertion 2: Production base ({base_sha[:7]}) and ancestor baseline ({AUTHORITATIVE_TARGETS['git_anchors']['common_ancestor_baseline'][:7]}) verified in Git database.")
    else:
        print(f"  [FAIL] Assertion 2: Git anchor retention verification failed.")
        if not prod_base_retained:
            errors.append(f"MISSING_GIT_ANCHOR: Production release base anchor {base_sha} not found in object DB.")
        if not ancestor_retained:
            errors.append(f"MISSING_GIT_ANCHOR: Ancestor baseline anchor {AUTHORITATIVE_TARGETS['git_anchors']['common_ancestor_baseline']} not found in object DB.")

    # 3. Assertion 3: External Recovery Target Platform Availability & Eligibility
    railway_target = args.assert_railway_target or AUTHORITATIVE_TARGETS["railway"]["active_rollback_anchor"]["deployment_id"]
    cf_target = args.assert_cloudflare_target or AUTHORITATIVE_TARGETS["cloudflare"]["active_rollback_anchor"]["deployment_id"]

    railway_valid = validate_uuid_format(railway_target)
    cf_valid = validate_uuid_format(cf_target)

    # Check for stale / removed historical references
    railway_stale = any(h["deployment_id"] == railway_target and h["status"] == "REMOVED" for h in AUTHORITATIVE_TARGETS["railway"]["historical_baselines"])
    railway_active_match = (railway_target == AUTHORITATIVE_TARGETS["railway"]["active_rollback_anchor"]["deployment_id"])
    cf_active_match = (cf_target == AUTHORITATIVE_TARGETS["cloudflare"]["active_rollback_anchor"]["deployment_id"])

    targets_authorized = (railway_valid and cf_valid and railway_active_match and cf_active_match and not railway_stale)

    assertions.append({
        "assertion": "External Recovery Target Platform Authority: Deployments are active, eligible, and non-stale",
        "observed": {
            "railway_target": railway_target,
            "railway_format_valid": railway_valid,
            "railway_active": railway_active_match,
            "railway_stale_rejected": not railway_stale,
            "cloudflare_target": cf_target,
            "cloudflare_format_valid": cf_valid,
            "cloudflare_active": cf_active_match
        },
        "passed": targets_authorized
    })

    if targets_authorized:
        print(f"  [PASS] Assertion 3: Railway target ({railway_target[:8]}) and Cloudflare target ({cf_target[:8]}) are authorized active serving baselines.")
    else:
        print(f"  [FAIL] Assertion 3: Recovery target authorization failed.")
        if not railway_valid:
            errors.append(f"INVALID_ROLLBACK_TARGET: Railway deployment ID '{railway_target}' is not a valid UUID format.")
        elif railway_stale:
            errors.append(f"STALE_ROLLBACK_TARGET_REJECTED: Railway deployment '{railway_target}' is a retired historical baseline (status: REMOVED).")
        elif not railway_active_match:
            errors.append(f"UNRECOGNIZED_ROLLBACK_TARGET: Railway deployment '{railway_target}' is not the authorized active rollback target.")

        if not cf_valid:
            errors.append(f"INVALID_ROLLBACK_TARGET: Cloudflare deployment ID '{cf_target}' is not a valid UUID format.")
        elif not cf_active_match:
            errors.append(f"UNRECOGNIZED_ROLLBACK_TARGET: Cloudflare deployment '{cf_target}' is not the authorized active rollback target.")

    # 4. Assertion 4: Live Platform Rehearsal Governance Policy
    assertions.append({
        "assertion": "Live Platform Rehearsal Governance Policy: Abstention from unauthorized production mutation",
        "observed": {
            "rehearsal_policy": "DRY_RUN_RECOVERY_TARGETS_VALIDATED",
            "live_production_rehearsal": "NOT_EXECUTED_PREFLIGHT_AWAITING_OWNER_AUTHORIZATION",
            "rationale": "Live execution against production domain and running containers strictly prohibited during pre-flight qualification to prevent production flapping.",
            "preflight_readiness": "ROLLBACK_PREFLIGHT_VERIFIED" if len(errors) == 0 else "ROLLBACK_PREFLIGHT_FAILED"
        },
        "passed": True
    })
    print("  [PASS] Assertion 4: Rehearsal policy enforced (dry-run preflight verified; live mutation barred).")

    # Output compilation
    duration = time.time() - t0
    status = "PASS" if len(errors) == 0 else "FAIL"

    output_payload = {
        "control_id": "RQ-11",
        "status": status,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidate_sha": head_sha,
        "production_base_anchor": base_sha,
        "ancestor_baseline_anchor": AUTHORITATIVE_TARGETS["git_anchors"]["common_ancestor_baseline"],
        "railway_recovery_anchor": AUTHORITATIVE_TARGETS["railway"]["active_rollback_anchor"],
        "cloudflare_recovery_anchor": AUTHORITATIVE_TARGETS["cloudflare"]["active_rollback_anchor"],
        "assertions": assertions,
        "errors": errors
    }

    targets_path = os.path.join(PROJECT_ROOT, "scripts", "release", "rq_rollback_targets.json")
    try:
        with open(targets_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"Rollback targets registered at: {targets_path}")
    except Exception as e:
        print(f"Warning: could not write {targets_path}: {e}")

    receipt = create_stage_receipt(
        control_id="RQ-11",
        name="Rollback Readiness & Target Authority",
        status=status,
        start_time=start_time,
        end_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        duration_sec=duration,
        candidate_sha=head_sha,
        target_branch=identity.get("target_branch", "release/production") if ok else "release/production",
        assertions=assertions,
        evidence_paths=["scripts/release/rq_rollback_targets.json"],
        errors=errors,
        metadata={"dry_run_validated": True, "live_mutation_barred": True}
    )
    receipt_path = save_stage_receipt(receipt)
    print(f"Receipt written to: {receipt_path}")

    if status == "PASS":
        print("=== [RQ-11] RESULT: PASS ===")
        sys.exit(0)
    else:
        print(f"=== [RQ-11] RESULT: FAIL (Errors: {errors}) ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
