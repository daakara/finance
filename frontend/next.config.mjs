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
};

export default nextConfig;