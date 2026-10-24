# ARX TERMINAL — OPPORTUNITY DOSSIER SCALING PILOT: OPERATING MANUAL

**Command Reference & Operational Runbook**  
**Tool:** `scripts/research/prepare_opportunity_dossier.py`  
**Governing Contract:** CONTRACT-ARX-OPP-PILOT-SCALING-V1  

---

## 1. Quick Start Commands

The Opportunity Dossier Scaling Engine provides a unified command line interface for preparing, validating, and staging research dossiers.

### Command 1: Prepare and Inspect a Single Company Draft
To prepare or inspect an individual company draft (e.g., `ULTA`):
```bash
python scripts/research/prepare_opportunity_dossier.py --symbol ULTA
```
*Output: Displays company identity, filing accessions, solved 5-year revenue CAGR, roundtrip per-share verification delta, and presentation summary.*

### Command 2: Prepare the Full 20-Company Pilot Staging Corpus
To prepare all 20 companies (5 baseline + 15 additions), run deterministic DCF solvers, and stage output files:
```bash
python scripts/research/prepare_opportunity_dossier.py --all-pilot
```
*Artifacts Produced:*
- `docs/research/opportunity_discovery/pilot_scaling_ledger.json` (authoritative research draft ledger)
- `docs/research/opportunity_discovery/pilot_scaling_manifest.json` (provenance manifest with SHA-256 digests)
- `frontend/lib/opportunityDossierPilotDrafts.json` (frontend staging snapshot for preview)

### Command 3: Validate Artifact Invariants & Integrity
To run automated integrity and consistency checks across all generated files without modifying anything:
```bash
python scripts/research/prepare_opportunity_dossier.py --validate
```
*In CI pipelines, use `--check` which exits with code 1 upon any invariant mismatch or stale file.*

---

## 2. Standard Operating Procedure for Adding a New Company

When scaling from 20 to subsequent cohorts, follow this 4-step procedure:

1. **Verify SEC EDGAR Filings:**
   - Confirm company CIK and accessibility of Forms 10-K and 10-Q.
   - Run SEC harvester to pull company facts and submissions:
     ```bash
     python -c "from analyst_dashboard.research.sourced_fundamentals.sec_harvester import SecEdgarHarvester; SecEdgarHarvester().get_company_facts('<CIK>')"
     ```

2. **Verify Capital Structure & Lease Conventions:**
   - For retail/restaurant companies: verify whether store leases are under ASC 842. Deduct rent in operating expenses and exclude lease liabilities from net debt bridge.
   - For zero-debt companies: verify in 10-Q Notes to Consolidated Financial Statements that credit facilities are undrawn and no senior notes exist.

3. **Define Specification & Run Solver:**
   - Add company specification to `PILOT_SPECS` in `scripts/research/prepare_opportunity_dossier.py`.
   - Run `--symbol <TICKER>` to verify bisection solving and roundtrip verification within 5 cents.

4. **Peer Research Review Gate:**
   - Prior to promoting any draft from `pilot_scaling_ledger.json` into production `odp05_discovery_ledger.json`, a qualified equity analyst must independently audit and sign off on qualitative synthesis claims.
