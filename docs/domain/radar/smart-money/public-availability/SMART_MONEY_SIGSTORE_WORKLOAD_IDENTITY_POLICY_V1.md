# Smart Money Sigstore Workload Identity Policy V1

## Purpose & Scope
This policy governs the unattended short-lived OIDC workload identity profile for Smart Money external Sigstore rooting. It eliminates any reliance on interactive browser/device-code authentication, personal access tokens, static private keys, or persisted bearer tokens.

## Policy Dimensions
1. **Issuer Constraint:** Strictly `https://token.actions.githubusercontent.com`.
2. **Repository Scope:** Strictly `owner/finance`.
3. **Workflow Scope:** Strictly authorized scheduler workflow `.github/workflows/smart_money_rooting.yml`.
4. **Branch / Ref Constraint:** Strictly `refs/heads/main`.
5. **Audience:** Strictly `sigstore`.
6. **Token Lifetime:** Ephemeral short-lived only, maximum 600 seconds.
7. **Secret Prohibition:** Zero static keys, zero PATs, zero disk persistence of tokens.
8. **Fail-Closed Runtime:** If ambient OIDC is absent, runtime must evaluate to `NOT_ESTABLISHED` and halt.
