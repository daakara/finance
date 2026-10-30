"use client";
import { useEffect, useState } from "react";
import { isUxReviewOnly } from "../lib/uxReviewBoundary";

export function useUxReviewOnly(): boolean {
  // Stable prerender/hydration value; handlers independently check the runtime host.
  const [reviewOnly, setReviewOnly] = useState(
    Boolean(process.env.NEXT_PUBLIC_UX_REVIEW_CANDIDATE && process.env.NEXT_PUBLIC_UX_REVIEW_CANDIDATE !== "false") ||
    process.env.NEXT_PUBLIC_UX_PREVIEW_DEPLOYMENT === "true"
  );
  useEffect(() => setReviewOnly(isUxReviewOnly()), []);
  return reviewOnly;
}
