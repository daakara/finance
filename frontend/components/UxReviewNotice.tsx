"use client";
import { useUxReviewOnly } from "../hooks/useUxReviewOnly";

export default function UxReviewNotice() {
  const reviewOnly = useUxReviewOnly();
  if (!reviewOnly) return null;
  return <aside role="note" data-testid="review-authority-notice" className="border-b border-[var(--border-subtle)] bg-[var(--surface-2)] px-4 py-2 text-xs text-[var(--text-2)]">
    Owner review preview · research presentation only. Paper recording is locked. Financial authority and production readiness remain unqualified.
  </aside>;
}
