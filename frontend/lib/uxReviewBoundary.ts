/** Review containment only; this never grants financial or execution authority. */
export function isUxReviewOnly(): boolean {
  const flag = process.env.NEXT_PUBLIC_UX_REVIEW_CANDIDATE;
  if (flag && flag !== "false") return true;
  if (process.env.NEXT_PUBLIC_UX_PREVIEW_DEPLOYMENT === "true") return true;
  // A missing build flag must not enable recording on an alternate review host.
  if (typeof window !== "undefined") {
    return !["arxterminal.com", "www.arxterminal.com"].includes(window.location.hostname);
  }
  return false;
}

export const UX_REVIEW_LOCK_MESSAGE = "Owner review — paper recording locked";
