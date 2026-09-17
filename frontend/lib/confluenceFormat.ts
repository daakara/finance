/**
 * Authoritative Confluence Score Formatter.
 * Validates that score is a finite number between 0 and 100.
 * Missing, undefined, null, or out-of-bounds values strictly format as "Unavailable"
 * to prevent leaking raw "/100" or invalid values.
 */
export function formatConfluenceScore(val: number | null | undefined): string {
  if (typeof val === 'number' && Number.isFinite(val) && val >= 0 && val <= 100) {
    return `${val}/100`;
  }
  return "Unavailable";
}
