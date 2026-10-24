import { fileURLToPath } from "node:url";

/**
 * Resolves the canonical frontend release identity using strict precedence:
 * CF_PAGES_COMMIT_SHA > NEXT_PUBLIC_ARX_RELEASE > undefined (fail-safe)
 *
 * Enforces single-authority build boundary:
 * Cloudflare Pages injects CF_PAGES_COMMIT_SHA, which is mapped here to
 * NEXT_PUBLIC_ARX_RELEASE so that statically compiled client bundles embed
 * the immutable deployment commit identity without manual intervention.
 *
 * @param {Record<string, string | undefined>} [env]
 * @returns {string | undefined}
 */
export function resolveFrontendRelease(env = process.env) {
  const raw = (env?.CF_PAGES_COMMIT_SHA || env?.NEXT_PUBLIC_ARX_RELEASE || "").trim();
  return raw || undefined;
}

const effectiveRelease = resolveFrontendRelease();

/** @type {import("next").NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  env: {
    ...(effectiveRelease ? { NEXT_PUBLIC_ARX_RELEASE: effectiveRelease } : {}),
  },
  webpack: (config) => {
    config.resolve.alias = config.resolve.alias || {};

    // In production builds (where pilot drafts are NOT explicitly enabled),
    // alias the pilot drafts JSON to an empty stub so Webpack excludes draft research
    // from client bundles and static export chunks entirely.
    if (process.env.NEXT_PUBLIC_ENABLE_RESEARCH_PILOT_DRAFTS !== "true") {
      const stubPath = fileURLToPath(
        new URL("./lib/opportunityDossierPilotDrafts.stub.json", import.meta.url)
      );
      config.resolve.alias["./opportunityDossierPilotDrafts.json"] = stubPath;
      config.resolve.alias["./opportunityDossierPilotDrafts.json$"] = stubPath;
      config.resolve.alias["@/lib/opportunityDossierPilotDrafts.json"] = stubPath;
      config.resolve.alias[
        fileURLToPath(new URL("./lib/opportunityDossierPilotDrafts.json", import.meta.url))
      ] = stubPath;
    }

    // In production builds, alias alconDraftDossier.json to an empty stub so Webpack
    // excludes it from production client bundles and static export chunks entirely.
    if (
      process.env.NODE_ENV === "production" ||
      process.env.NEXT_PUBLIC_ENABLE_ALCON_DEV_PREVIEW !== "true"
    ) {
      const alconStubPath = fileURLToPath(
        new URL("./lib/alconDraftDossier.stub.json", import.meta.url)
      );
      config.resolve.alias["./alconDraftDossier.json"] = alconStubPath;
      config.resolve.alias["./alconDraftDossier.json$"] = alconStubPath;
      config.resolve.alias["@/lib/alconDraftDossier.json"] = alconStubPath;
      config.resolve.alias[
        fileURLToPath(new URL("./lib/alconDraftDossier.json", import.meta.url))
      ] = alconStubPath;
    }

    return config;
  },
};

export default nextConfig;
